"""Offline tests for the checker audit: the corpora, and the replay that feeds it real turns.

Run:  python -m unittest test_audit -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from audit import (Case, answer_corpus, audit_answers, audit_commands, audit_pattern, audit_self_check,
                   check_command_corpora, command_corpus, decline_corpora, replay_attempts, replay_session,
                   session_corpora)
from kyklos import CHECK_COMMAND_RE, CHECK_DECLINED_RE, CHECKED, FAILING, UNCHECKED, check_state_by_order


def assistant(content="", bash=()):
    """An assistant message. `bash` is the commands it calls, as (id, command) pairs."""
    msg = {"role": "assistant", "content": content}
    if bash:
        msg["tool_calls"] = [{"id": i, "type": "function",
                              "function": {"name": "bash", "arguments": json.dumps({"command": c})}}
                             for i, c in bash]
    return msg


def result(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def write_session(lines):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    tmp.write("\n".join(json.dumps(line) for line in lines) + "\n")
    tmp.close()
    return Path(tmp.name)


class ReplayTests(unittest.TestCase):
    """Rebuilding (answer, commands run) from a saved session.

    This is the load-bearing half: a replayed flag only means what a live flag means if the checker
    sees exactly what the live loop handed it — the same commands, paired with the same results.
    """

    def session(self, lines):
        path = write_session(lines)
        self.addCleanup(path.unlink)
        return replay_session(path)

    def test_a_turn_carries_the_commands_that_ran_in_it(self):
        turns = self.session([
            {"role": "user", "content": "fix it"},
            assistant(bash=[("1", "pytest -q")]),
            result("1", "exit code 0\n5 passed"),
            assistant("Done, the tests pass."),
        ])
        self.assertEqual(turns, [("Done, the tests pass.", [("pytest -q", "exit code 0\n5 passed")])])

    def test_commands_do_not_leak_between_turns(self):
        """The second turn ran nothing, so its claim has nothing behind it — the first turn's
        pytest must not be credited to it."""
        turns = self.session([
            {"role": "user", "content": "one"},
            assistant(bash=[("1", "pytest -q")]),
            result("1", "exit code 0"),
            assistant("Tests pass."),
            {"role": "user", "content": "two"},
            assistant("Tests still pass."),
        ])
        self.assertEqual([runs for _, runs in turns], [[("pytest -q", "exit code 0")], []])

    def test_results_are_paired_by_call_id_not_by_order(self):
        turns = self.session([
            {"role": "user", "content": "go"},
            assistant(bash=[("a", "git commit -m x"), ("b", "git push")]),
            result("b", "exit code 1\nrejected"),
            result("a", "exit code 0"),
            assistant("Committed and pushed."),
        ])
        self.assertEqual(dict(turns[0][1]),
                         {"git commit -m x": "exit code 0", "git push": "exit code 1\nrejected"})

    def test_non_bash_calls_and_state_lines_are_ignored(self):
        turns = self.session([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "1", "type": "function",
                 "function": {"name": "read_file", "arguments": json.dumps({"path": "a.py"})}}]},
            result("1", "contents"),
            assistant("Had a look."),
            {"_meta": "state", "turn_log": [], "usage_mark": None},
        ])
        self.assertEqual(turns, [("Had a look.", [])])

    def test_a_malformed_line_does_not_sink_the_file(self):
        path = write_session([{"role": "user", "content": "go"}])
        self.addCleanup(path.unlink)
        path.write_text(path.read_text(encoding="utf-8") + "{not json\n" + json.dumps(
            assistant("Fine.")) + "\n", encoding="utf-8")
        self.assertEqual(replay_session(path), [("Fine.", [])])

    def test_session_corpora_pairs_answers_with_their_own_commands(self):
        path = write_session([
            {"role": "user", "content": "go"},
            assistant(bash=[("1", "pytest -q")]),
            result("1", "exit code 0"),
            assistant("I ran the tests and they pass."),
        ])
        self.addCleanup(path.unlink)
        commands, answers = session_corpora([path])
        self.assertEqual([c.text for c in commands], ["pytest -q"])
        self.assertEqual(audit_answers(answers).fired, [])  # the turn backs the claim up


class CorpusTests(unittest.TestCase):
    """The generated corpora: products of parts, not lists someone enumerated."""

    def test_the_corpora_are_products_not_handfuls(self):
        """A rate needs enough cases that no single phrasing dominates it."""
        self.assertGreater(len(command_corpus()), 100)
        self.assertGreater(len(answer_corpus()), 200)

    def test_every_case_is_grouped(self):
        for case in command_corpus() + answer_corpus():
            with self.subTest(text=case.text):
                self.assertTrue(case.group)

    def test_no_generated_command_writes_installs_or_reaches_the_network(self):
        """The corpus is only a control if every command in it is genuinely benign. If this fails,
        a case was added that deserves its prompt and the rate below it is meaningless."""
        banned = ("rm ", " > ", ">>", "curl", "wget", "ssh", "chmod", "sudo",
                  "install", "git push", "git commit", "git reset", "git clean")
        for case in command_corpus():
            with self.subTest(command=case.text):
                for word in banned:
                    self.assertNotIn(word, case.text)
                # `mv ` used to be banned outright, as a stand-in for destructive, back when mv had
                # no rule of its own. Renaming a file in the workspace is ordinary work and belongs
                # in a control corpus; what `mv ` was really standing in for is carrying something
                # out of the workspace, so that is what this now says -- and it covers cp too.
                self.assertNotRegex(case.text, r"(?:^|\s)[/~]")


class AuditTests(unittest.TestCase):
    """What the two measurements report."""

    def test_benign_commands_are_never_called_dangerous(self):
        """The rate that matters: the explainer calling ordinary work risky would train you to
        click through its prompts."""
        self.assertEqual(audit_commands(command_corpus()).of_kind("danger", "warn"), [])

    def test_an_unrecognised_program_counts_as_coverage_not_miscalibration(self):
        rate = audit_commands([Case("made up", "frobnicate --widgets")])
        self.assertEqual([f.kind for f in rate.of_kind("opaque")], ["opaque"])
        self.assertEqual(rate.of_kind("danger", "warn"), [])

    def test_a_dangerous_command_reports_as_miscalibration(self):
        """Not part of the real corpus — this pins that the split reports the loud kind as loud."""
        rate = audit_commands([Case("deliberate", "rm -rf build/")])
        self.assertEqual([f.kind for f in rate.of_kind("danger", "warn")], ["danger"])

    def test_advice_and_questions_never_read_as_claims(self):
        """The groups the checker already handles. A regression here is a cried wolf in the UI."""
        quiet = [(c, []) for c in answer_corpus() if c.group in ("advice", "asking", "later")]
        self.assertEqual(audit_answers(quiet).fired, [])

    def test_the_groups_breakdown_counts_fired_over_total(self):
        rate = audit_answers([(Case("g", "I installed left-pad."), []),
                              (Case("g", "You can run the tests."), [])])
        self.assertEqual(rate.groups(), {"g": (1, 2)})

    def test_one_reason_per_case_so_the_rate_counts_cases(self):
        """A command with three findings still costs exactly one prompt."""
        rate = audit_commands([Case("many", "rm -rf build/ && chmod -R 777 . && git push --force")])
        self.assertEqual(len(rate.fired), 1)


class ReplayAgreesWithLiveTests(unittest.TestCase):
    """The contract the whole replay rests on: a session written by `save_session` and read back
    must produce the verdict the live loop produced. If these ever disagree, every rate measured
    off saved sessions is measuring the parser instead of the checker."""

    def test_a_replayed_turn_reproduces_the_live_verdict(self):
        from kyklos import Agent
        from test_kyklos import FakeClient, allow_all, reply, tool_call

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ws = Path(tmp.name)
        agent = Agent(FakeClient([reply(tool_calls=[tool_call("1", "bash", command="pytest -q")]),
                                  reply("Done. I ran the tests and they pass.")]),
                      "fake", ws, permission=allow_all, on_event=lambda *a, **k: None)
        agent.send("fix it")
        live = agent.last_claims

        path = ws / "session.jsonl"
        agent.save_session(path)
        turns = replay_session(path)
        replayed = [(f.why.split(": ", 1)[0], f.why.split(": ", 1)[1])
                    for f in audit_answers([(Case("replayed", a), runs) for a, runs in turns]).fired]

        self.assertEqual(live, replayed)
        self.assertTrue(live)  # the fixture is a real mismatch, so this compares something


def file_call(call_id, name, path):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps({"path": path, "content": "x = 1\n"})}}]}


class SelfCheckReplayTests(unittest.TestCase):
    """Rebuilding, from a saved session, what self-checking sees when an answer would end a turn."""

    def attempts(self, lines):
        path = write_session(lines)
        self.addCleanup(path.unlink)
        return replay_attempts(path)

    def test_edits_and_checks_are_ordered_by_call(self):
        attempts = self.attempts([
            {"role": "user", "content": "fix it"},
            file_call("w1", "write_file", "app.py"), result("w1", "Wrote 6 chars to app.py"),
            assistant(bash=[("b", "pytest -q")]), result("b", "exit code 0\n3 passed"),
            assistant("Done."),
            {"role": "user", "content": "one more thing"},
            assistant(bash=[("b2", "pytest -q")]), result("b2", "exit code 1\n1 failed"),
            file_call("w2", "edit_file", "app.py"), result("w2", "Edited app.py."),
            assistant("Done again."),
        ])
        states = [check_state_by_order(a.code_edits, a.last_check) for a in attempts]
        self.assertEqual(states, [CHECKED, UNCHECKED])  # the second turn edited after its check
        self.assertEqual([a.turn for a in attempts], [1, 2])

    def test_only_successful_code_edits_outside_scratch_count(self):
        [attempt] = self.attempts([
            {"role": "user", "content": "go"},
            file_call("1", "write_file", "README.md"), result("1", "Wrote 6 chars to README.md"),
            file_call("2", "write_file", ".kyklos/scratch/probe.py"), result("2", "Wrote 6 chars to .kyklos/scratch/probe.py"),
            file_call("3", "write_file", "app.py"), result("3", "The user denied this write_file call."),
            assistant("Done."),
        ])
        self.assertEqual(attempt.code_edits, {})

    def test_a_failing_check_after_the_edit(self):
        [attempt] = self.attempts([
            {"role": "user", "content": "go"},
            file_call("1", "write_file", "app.py"), result("1", "Wrote 6 chars to app.py"),
            assistant(bash=[("2", "npx tsc --noEmit")]), result("2", "exit code 2\nerror TS2322"),
            assistant("Done."),
        ])
        self.assertEqual(check_state_by_order(attempt.code_edits, attempt.last_check), FAILING)

    def test_kyklos_notes_continue_the_turn_and_a_prefixed_message_starts_one(self):
        attempts = self.attempts([
            {"role": "user", "content": "go"},
            assistant(bash=[("1", "pytest -q")]), result("1", "exit code 0"),
            assistant("Done. I ran the tests."),
            {"role": "user", "content": "[Kyklos: you changed code this turn (app.py) and no check has passed.]"},
            assistant("Done."),
            {"role": "user", "content": "[Kyklos: files have changed since the workspace sketch: a.py]\n\nnext task"},
            assistant("Nothing ran here."),
        ])
        self.assertEqual([a.turn for a in attempts], [1, 1, 2])
        self.assertEqual(attempts[1].commands, [("pytest -q", "exit code 0")])  # carried past the note
        self.assertEqual(attempts[2].commands, [])

    def test_each_reason_is_counted_once_per_turn(self):
        path = write_session([
            {"role": "user", "content": "go"},
            file_call("1", "write_file", "app.py"), result("1", "Wrote 6 chars to app.py"),
            assistant("Done."),
            {"role": "user", "content": "[Kyklos: you changed code this turn (app.py).]"},
            assistant("Done."),
        ])
        self.addCleanup(path.unlink)
        rate, states = audit_self_check([path])
        self.assertEqual([f.kind for f in rate.fired], ["unchecked"])
        self.assertEqual(states[UNCHECKED], 2)


class SelfCheckPatternTests(unittest.TestCase):
    """The two patterns self-checking leans on, held to the audit's corpora."""

    def test_every_check_is_recognised_and_nothing_else(self):
        checks, others = check_command_corpora()
        self.assertGreater(len(checks), 40)
        self.assertEqual(audit_pattern("", checks, CHECK_COMMAND_RE, True).fired, [])
        self.assertEqual(audit_pattern("", others, CHECK_COMMAND_RE, False).fired, [])

    def test_reasons_are_recognised_and_plain_answers_are_not(self):
        declines, plain = decline_corpora()
        self.assertEqual(audit_pattern("", declines, CHECK_DECLINED_RE, True).fired, [])
        self.assertEqual(audit_pattern("", plain, CHECK_DECLINED_RE, False).fired, [])

    def test_the_cases_the_first_audit_found(self):
        """`can't` slipped past a pattern written for `couldn't`, and a bare `not` read descriptions
        of the code as reasons for not checking it."""
        self.assertTrue(CHECK_DECLINED_RE.search("I can't run the tests without the database."))
        self.assertFalse(CHECK_DECLINED_RE.search("The migration does not run twice anymore."))


class SelfCheckReplayAgreesWithLiveTests(unittest.TestCase):
    def test_a_replayed_session_reproduces_the_live_self_check(self):
        from kyklos import Agent
        from test_kyklos import FakeClient, allow_all, reply, tool_call

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ws = Path(tmp.name)
        live = []
        agent = Agent(FakeClient([reply(tool_calls=[tool_call("1", "write_file", path="app.py", content="x = 1\n")]),
                                  reply("Done."),
                                  reply("Done. I couldn't run a check here.")]),
                      "fake", ws, permission=allow_all, workspace_snapshots=False, self_check=True,
                      on_event=lambda event, depth=0, **d: live.append(d.get("reason"))
                      if event == "signal" and d.get("kind") == "self_check" else None)
        agent.send("fix it")
        path = ws / "session.jsonl"
        agent.save_session(path)
        rate, states = audit_self_check([path])
        self.assertEqual(live, ["unchecked"])
        self.assertEqual([f.kind for f in rate.fired], live)
        self.assertEqual(states[UNCHECKED], 2)
        self.assertEqual(agent.last_check, UNCHECKED)


if __name__ == "__main__":
    unittest.main()
