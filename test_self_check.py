"""Offline tests for self-checking: the model asked, once per turn, to back up its answer's claims
and to check code it changed before the turn ends.

A scripted fake model drives the real loop. Check commands really run (`python3 -m py_compile`), so
"passed" and "failed" come from an exit code, as they do in a session.

Run:  python -m unittest test_self_check -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from test_kyklos import FakeClient, allow_all, reply, tool_call
from test_ui import make_ui, output

from kyklos import (CHECK_COMMAND_RE, CHECKED, FAILING, UNCHECKED, Agent, Trace, choose_self_check,
                    is_code_path)

# From PATH, not sys.executable: the venv sits under ~/Desktop here, which the sandbox hides.
PY = shutil.which("python3") or "python3"
COMPILE = f"{PY} -m py_compile app.py"


def write(call_id, path="app.py", content="x = 1\n"):
    return tool_call(call_id, "write_file", path=path, content=content)


def bash(call_id, command):
    return tool_call(call_id, "bash", command=command)


class Events:
    def __init__(self):
        self.seen = []

    def __call__(self, event, depth=0, **data):
        self.seen.append((event, data))

    def signals(self):
        return [data.get("reason") for event, data in self.seen if event == "signal" and data.get("kind") == "self_check"]

    def notices(self):
        return [data["text"] for event, data in self.seen if event == "notice"]


class SelfCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_turn(self, script, **kw):
        """One turn over `script`; returns (answer, agent, client, events). The client fails the
        test if the loop asks for more replies than the script has."""
        events = Events()
        client = FakeClient(script)
        kw.setdefault("workspace_snapshots", False)
        kw.setdefault("self_check", True)
        agent = Agent(client, "fake", self.ws, permission=allow_all, on_event=events, **kw)
        answer = agent.send("do the thing")
        self.assertEqual(client.replies, [], "the script had replies the loop never asked for")
        return answer, agent, client, events

    def kyklos_messages(self, agent):
        return [m["content"] for m in agent.messages
                if m["role"] == "user" and str(m["content"]).startswith("[Kyklos: ")]

    # ---- layer 1: claims ------------------------------------------------------------------------

    def test_an_unbacked_claim_is_put_to_the_model_once(self):
        answer, agent, _, events = self.run_turn([
            reply(content="Fixed it. I ran the tests and they all pass."),
            reply(content="Fixed it. I haven't run the tests this turn."),
        ])
        self.assertEqual(answer, "Fixed it. I haven't run the tests this turn.")
        [note] = self.kyklos_messages(agent)
        self.assertIn("Your answer says it ran the tests, but no test command ran this turn", note)
        self.assertEqual(events.signals(), ["claims"])
        self.assertTrue(any(n.startswith("Self-check: the answer claimed work") for n in events.notices()))
        self.assertEqual(agent.last_claims, [])

    def test_a_claim_repeated_after_the_note_stands_and_is_reported_to_the_user(self):
        _, agent, _, events = self.run_turn([
            reply(content="I committed the fix."),
            reply(content="I committed the fix in the previous turn."),
        ])
        self.assertEqual(events.signals(), ["claims"])  # asked once, not again
        self.assertEqual(agent.last_claims, [("committed", "no `git commit` ran this turn")])

    # ---- layer 2: unchecked code ----------------------------------------------------------------

    def test_edited_code_with_no_check_is_put_to_the_model_then_checked(self):
        answer, agent, _, events = self.run_turn([
            reply(tool_calls=[write("w")]),
            reply(content="Done."),
            reply(tool_calls=[bash("b", COMPILE)]),
            reply(content="Done, and it compiles."),
        ])
        self.assertEqual(answer, "Done, and it compiles.")
        [note] = self.kyklos_messages(agent)
        self.assertIn("you changed code this turn (app.py)", note.lower())
        self.assertIn("If none fits", note)
        self.assertEqual(events.signals(), ["unchecked"])
        self.assertEqual(agent.last_check, CHECKED)

    def test_an_edit_after_the_check_needs_another(self):
        _, agent, _, events = self.run_turn([
            reply(tool_calls=[write("w1")]),
            reply(tool_calls=[bash("b", COMPILE)]),
            reply(tool_calls=[write("w2", content="x = 2\n")]),
            reply(content="Done."),
            reply(content="Done. I couldn't re-run the check after the last edit."),
        ])
        self.assertEqual(events.signals(), ["unchecked"])
        self.assertEqual(agent.last_check, UNCHECKED)

    def test_a_failing_check_is_put_to_the_model_unless_the_answer_reports_it(self):
        broken = "def f(:\n"
        _, agent, _, events = self.run_turn([
            reply(tool_calls=[write("w", content=broken)]),
            reply(tool_calls=[bash("b", COMPILE)]),
            reply(content="Done."),
            reply(content="Done."),  # asked once: the second answer stands
        ])
        [note] = self.kyklos_messages(agent)
        self.assertIn("The last check after your changes failed", note)
        self.assertIn("py_compile app.py", note)
        self.assertEqual(events.signals(), ["failing"])
        self.assertEqual(agent.last_check, FAILING)

        _, agent, _, events = self.run_turn([
            reply(tool_calls=[write("w", content=broken)]),
            reply(tool_calls=[bash("b", COMPILE)]),
            reply(content="The compile check fails on line 1; I need your call on the signature."),
        ])
        self.assertEqual(events.signals(), [])
        self.assertEqual(agent.last_check, FAILING)

    def test_claims_and_unchecked_code_share_one_note(self):
        _, agent, _, events = self.run_turn([
            reply(tool_calls=[write("w")]),
            reply(content="Done. All tests pass."),
            reply(tool_calls=[bash("b", COMPILE)]),
            reply(content="Done; it compiles. There are no tests in this folder."),
        ])
        [note] = self.kyklos_messages(agent)
        self.assertIn("ran the tests", note)
        self.assertIn("you changed code this turn", note)
        self.assertEqual(events.signals(), ["claims_unchecked"])

    def test_quiet_when_there_is_nothing_to_check_or_the_answer_already_says_why(self):
        cases = {
            "docs only": [reply(tool_calls=[write("w", path="README.md", content="# Hi\n")]),
                          reply(content="Done.")],
            "scratch only": [reply(tool_calls=[write("w", path=".kyklos/scratch/probe.py")]),
                             reply(content="Done.")],
            "said it can't": [reply(tool_calls=[write("w")]),
                              reply(content="Done. I couldn't run any checks: nothing is set up here.")],
            "a question": [reply(tool_calls=[write("w")]),
                           reply(content="Done. Should I add tests for it too?")],
            "read only": [reply(tool_calls=[tool_call("r", "glob", pattern="*")]), reply(content="Nothing here.")],
        }
        for name, script in cases.items():
            with self.subTest(name):
                _, agent, _, events = self.run_turn(script)
                self.assertEqual(events.signals(), [])
                self.assertEqual(self.kyklos_messages(agent), [])

    def test_no_note_without_room_to_act_on_it(self):
        _, agent, _, events = self.run_turn([reply(tool_calls=[write("w")]), reply(content="Done.")],
                                            max_steps=3)
        self.assertEqual(events.signals(), [])
        self.assertEqual(agent.last_check, UNCHECKED)

    def test_off_switch_and_subagents(self):
        _, agent, _, events = self.run_turn([reply(tool_calls=[write("w")]), reply(content="Done.")],
                                            self_check=False)
        self.assertEqual(events.signals(), [])
        self.assertIsNone(agent.last_check)
        # A general subagent that writes code answers its parent without being asked to check.
        _, agent, _, events = self.run_turn([
            reply(tool_calls=[tool_call("t", "task", description="d", prompt="p", mode="general")]),
            reply(tool_calls=[write("s")]),
            reply(content="Sub done."),
            reply(content="Parent done. I didn't run checks."),
        ])
        self.assertEqual(events.signals(), [])

    def test_the_code_graph_names_linked_tests(self):
        (self.ws / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        (self.ws / "test_calc.py").write_text("import calc\n\n\ndef test_add():\n    assert calc.add(1, 2) == 3\n")
        _, agent, _, _ = self.run_turn([
            reply(tool_calls=[write("w", path="calc.py", content="def add(a, b):\n    return b + a\n")]),
            reply(content="Done."),
            reply(content="Done. I haven't run the tests."),
        ])
        [note] = self.kyklos_messages(agent)
        self.assertIn("the code graph links these tests: test_calc.py", note)

    @unittest.skipUnless(shutil.which("git"), "snapshots need git")
    def test_with_snapshots_a_command_that_changes_code_counts_as_a_change(self):
        _, agent, _, events = self.run_turn([
            reply(tool_calls=[bash("b1", "printf 'x = 1\\n' > made.py")]),
            reply(content="Done."),
            reply(tool_calls=[bash("b2", f"{PY} -m py_compile made.py")]),
            reply(content="Done; it compiles."),
        ], workspace_snapshots=True)
        if agent.snapshots_off:
            self.skipTest(f"snapshots unavailable: {agent.snapshots_off}")
        [note] = self.kyklos_messages(agent)
        self.assertIn("(made.py)", note)
        self.assertEqual(agent.last_check, CHECKED)
        # Untouched code before and after the check: a command that changes nothing isn't an edit.
        _, agent, _, events = self.run_turn([
            reply(tool_calls=[bash("b1", f"{PY} -m py_compile made.py")]),
            reply(tool_calls=[bash("b2", "ls")]),
            reply(content="It compiles."),
        ], workspace_snapshots=True)
        self.assertEqual(events.signals(), [])
        self.assertIsNone(agent.last_check)

    # ---- what the user and the trace see ------------------------------------------------------

    def test_the_answer_gets_a_check_note_in_the_terminal(self):
        for script, expected in [
            ([reply(tool_calls=[write("w")]), reply(tool_calls=[bash("b", COMPILE)]), reply(content="Done.")],
             "✓ checked after the last code change"),
            ([reply(tool_calls=[write("w")]), reply(content="Done. I couldn't run a check here.")],
             "! code changed this turn, and no check passed after the last change"),
        ]:
            with self.subTest(expected):
                ui = make_ui(self.ws)
                agent = Agent(FakeClient(script), "fake", self.ws, permission=allow_all, on_event=ui.observer,
                              workspace_snapshots=False, self_check=True)
                ui.agent = agent
                ui.turn(agent, "go", self.ws / "session.json")
                self.assertIn(expected, output(ui))

    def test_turn_end_and_the_trace_carry_the_state_as_a_label(self):
        _, _, _, events = self.run_turn([reply(tool_calls=[write("w")]), reply(tool_calls=[bash("b", COMPILE)]),
                                         reply(content="Done.")])
        [end] = [data for event, data in events.seen if event == "turn_end"]
        self.assertEqual(end["checked"], CHECKED)
        path = self.ws / "trace.jsonl"
        trace = Trace(path)
        trace.record("turn_end", 0, steps=3, outcome="answer", answer_chars=5, claims=0, checked=CHECKED)
        trace.close()
        self.assertEqual(json.loads(path.read_text().splitlines()[0])["checked"], "checked")


class PatternTests(unittest.TestCase):
    def test_check_commands(self):
        for command in ["pytest -q", "python -m unittest test_app", "npx tsc --noEmit", "npm run build",
                        "pnpm lint", "cargo clippy", "go vet ./...", "ruff check .", "mypy src",
                        "python3 -m py_compile app.py", "make check", "./gradlew test", "yarn test:unit",
                        "uv run pytest tests/test_x.py", "npm test"]:
            with self.subTest(command):
                self.assertTrue(CHECK_COMMAND_RE.search(command))
        for command in ["ls", "git status", "cat test_app.py", "echo build", "npm install", "go run .",
                        "python app.py", "grep -r test ."]:
            with self.subTest(command):
                self.assertFalse(CHECK_COMMAND_RE.search(command))

    def test_code_paths(self):
        for path in ["app.py", "src/main.ts", "lib/x.rs", "scripts/run.sh", "a/b/C.java"]:
            self.assertTrue(is_code_path(path), path)
        for path in ["README.md", "package.json", "pyproject.toml", "logo.png", ".kyklos/scratch/x.py", "Makefile"]:
            self.assertFalse(is_code_path(path), path)

    def test_the_setting(self):
        self.assertTrue(choose_self_check({}))
        self.assertTrue(choose_self_check({"KYKLOS_SELF_CHECK": "on"}))
        self.assertFalse(choose_self_check({"KYKLOS_SELF_CHECK": "0"}))
        with self.assertRaises(ValueError):
            choose_self_check({"KYKLOS_SELF_CHECK": "maybe"})


if __name__ == "__main__":
    unittest.main()
