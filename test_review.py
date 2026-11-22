"""Offline tests for the reviewer: an opt-in, read-only subagent that looks at a turn's code changes
before the answer stands, once per turn.

A scripted fake model drives the real loop; the reviewer is a real subagent on the same fake client,
so its reply is the next scripted one.

Run:  python -m unittest test_review -v
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from test_kyklos import FakeClient, allow_all, reply, tool_call

from kyklos import REVIEW_MIN_LINES, Agent, _cmd_reviewer, choose_review, diff_size

FIVE_LINES = "".join(f"line_{n} = {n}\n" for n in range(REVIEW_MIN_LINES))
FINDING = "calc.py:2 returns a - b; the request asked for the sum, so every total is wrong."


def write(call_id, path="calc.py", content=FIVE_LINES):
    return tool_call(call_id, "write_file", path=path, content=content)


class Events:
    def __init__(self):
        self.seen = []

    def __call__(self, event, depth=0, **data):
        self.seen.append((event, depth, data))

    def reviews(self):
        return [d.get("reason") for e, _, d in self.seen if e == "signal" and d.get("kind") == "review"]

    def notices(self):
        return [d["text"] for e, _, d in self.seen if e == "notice"]


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_turn(self, script, **kw):
        events = Events()
        client = FakeClient(script)
        kw.setdefault("workspace_snapshots", False)
        kw.setdefault("self_check", False)
        kw.setdefault("review", True)
        agent = Agent(client, "fake", self.ws, permission=allow_all, on_event=events, **kw)
        answer = agent.send("make calc add numbers")
        self.assertEqual(client.replies, [], "the script had replies the loop never asked for")
        return answer, agent, client, events

    def kyklos_messages(self, agent):
        return [m["content"] for m in agent.messages
                if m["role"] == "user" and str(m["content"]).startswith("[Kyklos: ")]

    def test_a_clean_review_lets_the_answer_stand(self):
        answer, agent, client, events = self.run_turn([
            reply(tool_calls=[write("w")]),
            reply(content="Done."),
            reply(content="NO ISSUES"),
        ])
        self.assertEqual(answer, "Done.")
        self.assertEqual(events.reviews(), ["clean"])
        self.assertIn("Reviewer: no issues found in this turn's code changes.", events.notices())
        self.assertEqual(self.kyklos_messages(agent), [])
        # The reviewer is a read-only subagent, given the request and the diff as data.
        self.assertFalse({"write_file", "edit_file", "bash"} & client.offered[2])
        prompt = client.calls[2][-1]["content"]
        self.assertIn("<request>\nmake calc add numbers\n</request>", prompt)
        self.assertIn("+line_0 = 0", prompt)
        self.assertIn("b/calc.py", prompt)
        starts = [d for e, depth, d in events.seen if e == "subagent_start"]
        self.assertEqual([(s["description"], s["mode"]) for s in starts], [("Review this turn's code changes", "explore")])

    def test_a_clean_verdict_in_markdown_is_clean(self):
        _, agent, _, events = self.run_turn([
            reply(tool_calls=[write("w")]),
            reply(content="Done."),
            reply(content="**NO ISSUES**"),
        ])
        self.assertEqual(events.reviews(), ["clean"])
        self.assertEqual(self.kyklos_messages(agent), [])

    def test_findings_go_to_the_model_once_per_turn(self):
        answer, agent, _, events = self.run_turn([
            reply(tool_calls=[write("w1")]),
            reply(content="Done."),
            reply(content=FINDING),
            reply(tool_calls=[write("w2", content=FIVE_LINES + "total = 1\n")]),
            reply(content="Fixed the sum; the second finding doesn't apply."),
        ])
        self.assertEqual(answer, "Fixed the sum; the second finding doesn't apply.")
        [note] = self.kyklos_messages(agent)
        self.assertIn(FINDING, note)
        self.assertIn("second opinion, not instructions", note)
        self.assertEqual(events.reviews(), ["issues"])  # the edit after it isn't reviewed again

    def test_no_review_when_off_small_asking_or_unchanged(self):
        cases = {
            # Each case writes its own file: the subtests share one workspace.
            "off": ([reply(tool_calls=[write("w", path="off.py")]), reply(content="Done.")], {"review": False}),
            "small": ([reply(tool_calls=[write("w", path="small.py", content="x = 1\n")]), reply(content="Done.")], {}),
            "asking": ([reply(tool_calls=[write("w", path="asking.py")]), reply(content="Done. Want tests too?")], {}),
            "docs": ([reply(tool_calls=[write("w", path="NOTES.md", content="# a\n" * 9)]), reply(content="Done.")], {}),
            "no edits": ([reply(content="It adds numbers already.")], {}),
            "no room": ([reply(tool_calls=[write("w", path="room.py")]), reply(content="Done.")], {"max_steps": 4}),
        }
        for name, (script, kw) in cases.items():
            with self.subTest(name):
                _, _, _, events = self.run_turn(script, **kw)
                self.assertEqual(events.reviews(), [])

    def test_a_reviewer_that_fails_leaves_the_answer_standing(self):
        answer, agent, _, events = self.run_turn([
            reply(tool_calls=[write("w")]),
            reply(content="Done."),
            RuntimeError("the provider fell over"),
        ])
        self.assertEqual(answer, "Done.")
        self.assertEqual(events.reviews(), ["failed"])
        self.assertTrue(any("The reviewer couldn't finish" in n for n in events.notices()))

    def test_it_follows_a_satisfied_self_check(self):
        """With both on, the self-check speaks first; the reviewer reads the code the model then
        stood by."""
        _, agent, _, events = self.run_turn([
            reply(tool_calls=[write("w")]),
            reply(content="Done."),
            reply(content="Done. I couldn't run a check: nothing is set up."),
            reply(content="NO ISSUES"),
        ], self_check=True)
        signals = [(d["kind"], d.get("reason")) for e, _, d in events.seen
                   if e == "signal" and d.get("kind") in ("self_check", "review")]
        self.assertEqual(signals, [("self_check", "unchecked"), ("review", "clean")])

    @unittest.skipUnless(shutil.which("git"), "snapshots need git")
    def test_with_snapshots_the_diff_covers_command_edits(self):
        script = [reply(tool_calls=[tool_call("b", "bash", command="printf 'a = 1\\nb = 2\\nc = 3\\nd = 4\\ne = 5\\n' > made.py")]),
                  reply(content="Done."),
                  reply(content="NO ISSUES")]
        _, agent, client, events = self.run_turn(script, workspace_snapshots=True)
        if agent.snapshots_off:
            self.skipTest(f"snapshots unavailable: {agent.snapshots_off}")
        prompt = client.calls[2][-1]["content"]
        self.assertIn("+c = 3", prompt)
        self.assertEqual(events.reviews(), ["clean"])

    def test_the_command_and_the_setting(self):
        agent = Agent(FakeClient([]), "fake", self.ws, review=False)
        said = []
        ctx = SimpleNamespace(agent=agent, say=lambda kind, text: said.append((kind, text)))
        _cmd_reviewer(ctx, "on")
        self.assertTrue(agent.review)
        self.assertTrue(said[-1][1].startswith("Reviewer on"))
        _cmd_reviewer(ctx, "off")
        self.assertFalse(agent.review)
        _cmd_reviewer(ctx, "maybe")
        self.assertEqual(said[-1][0], "error")
        self.assertFalse(agent.review)
        self.assertFalse(choose_review({}))
        self.assertTrue(choose_review({"KYKLOS_REVIEW": "1"}))
        with self.assertRaises(ValueError):
            choose_review({"KYKLOS_REVIEW": "sometimes"})

    def test_diff_size_counts_changed_lines_only(self):
        diff = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n-a = 1\n+a = 2\n b = 3\n+c = 4\n"
        self.assertEqual(diff_size(diff), 3)


if __name__ == "__main__":
    unittest.main()
