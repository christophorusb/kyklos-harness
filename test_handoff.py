"""Offline tests for handoff: the model handing work Kyklos can't do to the user.

A scripted fake model drives the real loop; events are collected from on_event, and the terminal card
is checked through a recording Aero UI.

Run:  python -m unittest test_handoff -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from test_kyklos import FakeClient, deny_all, reply, tool_call
from test_ui import make_ui, output

from kyklos import (HANDOFF_NOTE, HANDOFF_REASONS, MAX_HANDOFF_PROMPT_CHARS, MAX_HANDOFFS_PER_TURN, Agent,
                    Trace, tool_overview)
from ui import describe_call, summarize_result

PROMPT = ("Open http://127.0.0.1:5173/login after `npm run dev` in /work/site and check the form "
          "renders with two fields. Report what you see.")


def handoff(call_id="h1", **args):
    args = {"title": "Check the login page", "reason": "browser", "prompt": PROMPT, **args}
    return tool_call(call_id, "handoff", **args)


class Events:
    def __init__(self):
        self.seen = []

    def __call__(self, event, depth=0, **data):
        self.seen.append((event, depth, data))

    def named(self, name):
        return [data for event, _, data in self.seen if event == name]


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_turn(self, calls, **kw):
        """One turn: the model makes `calls` (one message each), then answers. Returns tool results."""
        events = Events()
        client = FakeClient([reply(tool_calls=[c]) for c in calls] + [reply(content="Done.")])
        agent = Agent(client, "fake", self.ws, permission=deny_all, on_event=events, **kw)
        agent.send("do the thing")
        return [m["content"] for m in agent.messages if m["role"] == "tool"], events, agent, client

    def test_shown_to_the_user_without_a_permission_prompt(self):
        results, events, _, _ = self.run_turn([handoff(title="  Check the login page \n")])
        self.assertEqual(results, [HANDOFF_NOTE])  # deny_all was never asked
        self.assertEqual(events.named("handoff"),
                         [{"title": "Check the login page", "reason": "browser", "prompt": PROMPT}])
        order = [event for event, _, _ in events.seen if event in ("tool_call", "handoff", "tool_result")]
        self.assertEqual(order, ["tool_call", "handoff", "tool_result"])  # the card comes before its result

    def test_bad_arguments_are_refused_and_nothing_is_shown(self):
        cases = [
            ({"reason": "teleport"}, "reason must be one of"),
            ({"prompt": "   "}, "write the prompt"),
            ({"title": " "}, "short title"),
            ({"prompt": "x" * (MAX_HANDOFF_PROMPT_CHARS + 1)}, "keep it under"),
        ]
        for args, expected in cases:
            with self.subTest(args=list(args)):
                results, events, _, _ = self.run_turn([handoff(**args)])
                self.assertTrue(results[0].startswith("Error:"), results[0])
                self.assertIn(expected, results[0])
                self.assertEqual(events.named("handoff"), [])
        extra = tool_call("h1", "handoff", title="t", reason="web", prompt="p", target="codex")
        results, events, _, _ = self.run_turn([extra])
        self.assertIn("unexpected target", results[0])
        self.assertEqual(events.named("handoff"), [])

    def test_every_reason_is_accepted(self):
        for reason in HANDOFF_REASONS:
            with self.subTest(reason=reason):
                results, _, _, _ = self.run_turn([handoff(reason=reason)])
                self.assertEqual(results, [HANDOFF_NOTE])

    def test_limited_per_turn_and_the_count_resets(self):
        calls = [handoff(f"h{i}") for i in range(MAX_HANDOFFS_PER_TURN + 1)]
        script = [reply(tool_calls=[c]) for c in calls] + [reply(content="Done.")]
        script += [reply(tool_calls=[handoff("next")]), reply(content="Done again.")]
        events = Events()
        agent = Agent(FakeClient(script), "fake", self.ws, on_event=events)
        agent.send("one")
        results = [m["content"] for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(results[:MAX_HANDOFFS_PER_TURN], [HANDOFF_NOTE] * MAX_HANDOFFS_PER_TURN)
        self.assertIn("the most one turn may", results[-1])
        agent.send("two")
        self.assertEqual(agent.messages[-2]["content"], HANDOFF_NOTE)
        self.assertEqual(len(events.named("handoff")), MAX_HANDOFFS_PER_TURN + 1)

    def test_secrets_are_scrubbed_before_anyone_sees_them(self):
        token = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
        results, events, _, _ = self.run_turn([handoff(prompt=f"Push with token {token} to origin.")])
        shown = events.named("handoff")[0]["prompt"]
        self.assertNotIn(token, shown)
        self.assertIn("secret value(s) were removed", results[0])

    def test_offered_with_guidance_in_plan_mode_too(self):
        for plan_mode in (False, True):
            with self.subTest(plan_mode=plan_mode):
                _, _, agent, client = self.run_turn([], plan_mode=plan_mode)
                self.assertIn("handoff", client.offered[0])
                self.assertIn("call handoff with a self-contained prompt", agent.messages[0]["content"])

    def test_subagents_cant_hand_off_and_are_told_what_to_do_instead(self):
        script = [reply(tool_calls=[tool_call("t", "task", description="d", prompt="p", mode="general")]),
                  reply(tool_calls=[handoff("s")]),
                  reply(content="Sub done."), reply(content="Parent done.")]
        events = Events()
        client = FakeClient(script)
        Agent(client, "fake", self.ws, on_event=events).send("delegate")
        self.assertNotIn("handoff", client.offered[1])
        sub_result = [m["content"] for m in client.calls[2] if m["role"] == "tool"][0]
        self.assertIn("Subagents can't hand off to the user", sub_result)
        self.assertEqual(events.named("handoff"), [])

    def test_tools_view_lists_it_as_needing_no_approval(self):
        agent = Agent(FakeClient([]), "fake", self.ws)
        self.assertIn("handoff", tool_overview(agent)["safe"])

    def test_the_trace_keeps_the_reason_and_never_the_words(self):
        path = self.ws / "trace.jsonl"
        trace = Trace(path)
        trace.record("handoff", 0, title="Check the login page", reason="browser", prompt=PROMPT)
        trace.close()
        row = json.loads(path.read_text().splitlines()[0])
        self.assertEqual(row["event"], "handoff")
        self.assertEqual(row["reason"], "browser")
        self.assertNotIn("login", path.read_text())


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_call_line_and_result_summary(self):
        self.assertEqual(describe_call("handoff", {"title": "Check login\nmore"}), "Check login")
        self.assertEqual(summarize_result("handoff", {}, HANDOFF_NOTE)[:2], ("ok", "handed off"))
        self.assertEqual(summarize_result("handoff", {}, "Error: reason must be one of x")[0], "fail")

    def test_the_card_shows_the_prompt_whole_and_unbordered(self):
        ui = make_ui(self.ws)
        long_line = "step " * 40
        ui.observer("handoff", 0, title="Check the login page", reason="long_run",
                    prompt=f"Line one\n{long_line}\nLine three")
        text = output(ui)
        self.assertIn("handoff · long run · Check the login page", text)
        self.assertIn("Line one\n", text)
        self.assertIn(long_line.strip(), text)  # soft-wrapped: no newline put into the prompt
        self.assertIn("nothing was sent", text)
        self.assertNotIn("│ Line one", text)

    def test_escape_codes_in_the_prompt_are_stripped(self):
        ui = make_ui(self.ws)
        ui.observer("handoff", 0, title="T\x1b[2J", reason="web", prompt="read\x1b]52;c;ZXZpbA==\x07 this")
        text = output(ui)
        self.assertNotIn("\x1b]52", text)
        self.assertIn("read", text)


if __name__ == "__main__":
    unittest.main()
