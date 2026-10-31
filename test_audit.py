"""Offline tests for the checker audit: the corpora, and the replay that feeds it real turns.

Run:  python -m unittest test_audit -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from audit import (Case, answer_corpus, audit_answers, audit_commands, command_corpus,
                   replay_session, session_corpora)


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


if __name__ == "__main__":
    unittest.main()
