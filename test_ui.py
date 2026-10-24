"""Offline tests for the Aero UI: a recording Rich console + the scripted FakeClient drive the real Agent.

Run:  python -m unittest test_ui -v
"""

import io
import re
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

from kyklos import EXIT, Agent, CommandContext, run_command, wants_aero_ui
from test_kyklos import FakeClient, TransientAPIError, reply, tool_call
from ui import AeroUI, describe_call, diff_counts, diff_ops, summarize_result, trim_context


def make_ui(ws, keys=(), lines=(), **kw):
    """A UI that records its output; `keys` are the single keypresses, `lines` the typed lines."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor", width=100, record=True)
    keys, lines = list(keys), list(lines)
    ui = AeroUI(ws, console=console, animate=False, ascii_mode=kw.pop("ascii_mode", False),
                read_key=lambda: keys.pop(0) if keys else None,
                edit_line=lambda message, default="": lines.pop(0) if lines else default, **kw)
    return ui


def output(ui) -> str:
    return ui.console.export_text()


class DiffTests(unittest.TestCase):
    OLD = "def f():\n    log('x')\n    k = int(w * 0.2)\n    return k\n"
    NEW = "def f():\n    k = int(w * KEEP)\n    m = None\n    return k\n"

    def test_pairs_similar_lines_as_modifications(self):
        kinds = [(kind, text.strip()) for kind, _, text, _ in diff_ops(self.OLD, self.NEW)]
        self.assertEqual(kinds, [
            ("ctx", "def f():"),
            ("del", "log('x')"),
            ("del", "k = int(w * 0.2)"), ("mod", "k = int(w * KEEP)"),
            ("add", "m = None"),
            ("ctx", "return k"),
        ])
        self.assertEqual(diff_counts(self.OLD, self.NEW), (1, 1, 1))

    def test_modification_spans_cover_only_changed_characters(self):
        rows = {kind: (text, spans) for kind, _, text, spans in
                diff_ops("keep = int(window * 0.2)\n", "keep = int(window * KEEP)\n")}
        new_text, spans = rows["mod"]
        self.assertEqual("".join(new_text[s:e] for s, e in spans), "KEEP")
        old_text, old_spans = rows["del"]
        self.assertEqual("".join(old_text[s:e] for s, e in old_spans), "0.2")

    def test_trim_context_collapses_far_unchanged_lines(self):
        old = "\n".join(f"line {i}" for i in range(20))
        new = old.replace("line 10", "line ten")
        kinds = [r[0] for r in trim_context(diff_ops(old, new))]
        self.assertEqual(kinds[0], "gap")
        self.assertEqual(kinds.count("ctx"), 4)  # 2 before, 2 after the change
        self.assertEqual(kinds[-1], "gap")


class SummaryTests(unittest.TestCase):
    def test_tool_summaries(self):
        self.assertEqual(summarize_result("read_file", {}, "     4\ta\n     5\tb")[1::2], ("2 lines", "4–5"))
        self.assertEqual(summarize_result("grep", {}, "a.py:1: x\na.py:3: y\nb.py:2: z")[1], "3 matches in 2 files")
        self.assertEqual(summarize_result("grep", {}, "No matches.")[1], "no matches")
        self.assertEqual(summarize_result("glob", {}, "a.py\nb.py")[1], "2 files")
        status, summary, tail, _ = summarize_result("bash", {}, "exit code 1\nl1\nl2\nl3\nl4")
        self.assertEqual((status, summary, tail), ("exitfail", "exit 1", ["l1", "l2", "l3", "l4"]))  # short: all
        self.assertEqual(summarize_result("read_file", {}, "Error: File not found: x")[:2], ("fail", "File not found: x"))
        # The reason was already shown on the panel's "denied:" line, so the tool line doesn't repeat it.
        self.assertEqual(summarize_result("bash", {}, "The user denied this bash call. Reason: no")[:2],
                         ("denied", "denied by you"))
        self.assertEqual(summarize_result("bash", {}, "The bash call was denied: nobody is available to approve it.")[1],
                         "denied: no one to approve")

    def test_short_bash_output_is_shown_in_full_and_long_output_keeps_the_end(self):
        eight = "exit code 0\n" + "\n".join(f"line {i}" for i in range(8))
        self.assertEqual(len(summarize_result("bash", {}, eight)[2]), 8)
        twenty = "exit code 0\n" + "\n".join(f"line {i}" for i in range(20))
        self.assertEqual(summarize_result("bash", {}, twenty)[2], ["… 17 earlier lines", "line 17", "line 18", "line 19"])

    def test_call_descriptions_are_human_readable(self):
        self.assertEqual(describe_call("grep", {"pattern": "def", "path": "src", "glob": "*.py"}), "/def/ in src  *.py")
        self.assertEqual(describe_call("bash", {"command": "py -m unittest\nsecond line"}), "$ py -m unittest")
        self.assertEqual(describe_call("task", {"description": "find main"}), "explore · find main")


class AeroTurnTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        (self.ws / "app.py").write_text("def main():\n    return 42\n")
        self.session = self.ws / ".sessions" / "last.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, ui, replies, **kw):
        a = Agent(FakeClient(replies), "fake-model", self.ws, permission=ui.permission, on_event=ui.observer, **kw)
        ui.agent = a
        return a

    def test_turn_renders_tool_lines_answer_and_footer(self):
        ui = make_ui(self.ws)
        agent = self.agent(ui, [
            reply(content="Looking first.", tool_calls=[tool_call("1", "read_file", path="app.py"),
                                                       tool_call("2", "grep", pattern="return")]),
            reply(content="# Done\n\nIt returns `42`."),
        ])
        ui.turn(agent, "what does main return?", self.session)
        out = output(ui)
        lines = out.splitlines()
        # The progress note and the answer are each a box titled with the wordmark.
        self.assertEqual(sum(1 for l in lines if re.search(r"^  .─ +○ kyklos ─", l)), 2)
        self.assertRegex(out, r"│ Looking first\. +│")
        self.assertRegex(out, r"● read_file  app\.py  1–2 +2 lines")
        self.assertRegex(out, r"● grep  /return/ +1 match in 1 file")
        self.assertRegex(out, r"│ Done +│")  # h1 is left-aligned inside the box, not centered
        self.assertRegex(out, r"│ It returns 42\. +│")
        # The turn's stats sit in the answer box's bottom border.
        self.assertRegex(out, r"(?m)─ ctx \d+% · 2 tools · \d+s ─.\s*$")
        self.assertTrue(self.session.is_file())  # the session is saved after each turn

    def test_everything_uses_the_full_terminal_width(self):
        console = Console(file=io.StringIO(), force_terminal=True, width=160, record=True)
        ui = AeroUI(self.ws, console=console, animate=False, ascii_mode=False)
        agent = self.agent(ui, [reply(tool_calls=[tool_call("1", "read_file", path="app.py")]),
                                reply(content="Done.")])
        ui.turn(agent, "go", self.session)
        lines = [l.rstrip() for l in console.export_text().splitlines() if l.strip()]
        tool_line = next(l for l in lines if "read_file" in l)
        self.assertEqual(len(tool_line), 160)  # the summary sits at the right edge, not at column 100
        box_top = next(l for l in lines if "kyklos" in l)
        self.assertEqual(len(box_top), 160)

    def test_edit_permission_panel_shows_diff_once_and_applies(self):
        ui = make_ui(self.ws, keys=["y"])
        agent = self.agent(ui, [
            reply(tool_calls=[tool_call("1", "edit_file", path="app.py", old_string="return 42", new_string="return 43")]),
            reply(content="Changed it."),
        ])
        ui.turn(agent, "make it 43", self.session)
        out = output(ui)
        self.assertIn("Allow edit_file?  app.py  ~1", out)
        self.assertIn("y yes  ·  n no  ·  a always edit_file", out)
        self.assertEqual(out.count("return 43"), 1)  # shown in the panel, not repeated after applying
        self.assertRegex(out, r"● edit_file  app\.py  ~1 +✓ applied")
        self.assertIn("return 43", (self.ws / "app.py").read_text())

    def test_auto_approved_edit_still_shows_its_diff(self):
        ui = make_ui(self.ws)
        agent = self.agent(ui, [
            reply(tool_calls=[tool_call("1", "write_file", path="new.txt", content="hello\n")]),
            reply(content="Wrote it."),
        ], always_allowed={"write_file"})
        ui.turn(agent, "write a file", self.session)
        out = output(ui)
        self.assertNotIn("Allow write_file?", out)
        self.assertRegex(out, r"● write_file  new\.txt  \+1 +✓ written")
        self.assertRegex(out, r"1 \+ hello")

    def test_deny_with_reason_reaches_the_model(self):
        ui = make_ui(self.ws, keys=["n"], lines=["use grep instead"])
        client_replies = [
            reply(tool_calls=[tool_call("1", "bash", command="dir")]),
            reply(content="OK, I'll use grep."),
        ]
        agent = self.agent(ui, client_replies)
        ui.turn(agent, "list files", self.session)
        out = output(ui)
        self.assertIn("⊘ denied: use grep instead", out)
        self.assertRegex(out, r"⊘ bash  \$ dir +denied by you")
        self.assertEqual(out.count("use grep instead"), 1)  # the reason is shown once, not twice
        tool_msg = [m for m in agent.messages if m["role"] == "tool"][-1]["content"]
        self.assertIn("Reason: use grep instead", tool_msg)

    def test_prefix_choice_saves_an_editable_rule(self):
        ui = make_ui(self.ws, keys=["p"], lines=["py -m unittest"])
        agent = self.agent(ui, [
            reply(tool_calls=[tool_call("1", "bash", command="py -m unittest test_x")]),
            reply(content="Ran."),
        ])
        ui.turn(agent, "run tests", self.session)
        self.assertIn("bash:py -m unittest", agent.always_allowed)
        self.assertIn("allowed commands starting with 'py -m unittest'", output(ui))

    def test_no_keyboard_denies(self):
        ui = make_ui(self.ws)  # read_key returns None
        agent = self.agent(ui, [
            reply(tool_calls=[tool_call("1", "bash", command="dir")]),
            reply(content="Denied."),
        ])
        ui.turn(agent, "go", self.session)
        self.assertIn("nobody is available", [m for m in agent.messages if m["role"] == "tool"][-1]["content"])

    def test_subagent_collapses_to_one_line(self):
        ui = make_ui(self.ws)
        agent = self.agent(ui, [
            reply(tool_calls=[tool_call("t1", "task", description="find main", prompt="Find main.")]),
            reply(tool_calls=[tool_call("s1", "grep", pattern="def main")]),
            reply(content="main is in app.py:1."),
            reply(content="It's in app.py."),
        ])
        ui.turn(agent, "where is main?", self.session)
        out = output(ui)
        self.assertRegex(out, r"● task  explore · find main +2 steps · \d+s")
        self.assertNotIn("def main", out)  # the subagent's steps were transient
        self.assertRegex(out, r"ctx \d+% · 1 tool · \d+s")

    def test_long_summary_is_shortened_before_the_tool_label(self):
        from rich.text import Text
        ui = make_ui(self.ws)
        row = ui._line(Text("bash  $ git log --oneline"), Text("denied: " + "x" * 120)).plain
        self.assertIn("bash  $ git log --oneline", row)  # the label stays readable
        self.assertIn("…", row)  # the summary is cut instead
        self.assertLessEqual(len(row.rstrip()), ui.width)

    def test_approved_command_shows_as_running_with_a_timer(self):
        import time
        ui = make_ui(self.ws, keys=["y"])
        ui.animate = True  # this test needs the live zone
        ui._running[0] = ("bash", "$ npm test", time.monotonic() - 60)  # the model asked a minute ago
        try:
            allowed, _ = ui.permission("bash", {"command": "npm test"}, set())
            self.assertTrue(allowed)
            self.assertIsNotNone(ui._live)  # the live zone is back after the panel paused it
            self.assertLess(time.monotonic() - ui._running[0][2], 5)  # the timer restarted on approval
        finally:
            ui.reset_transient()
        ui._running[0] = ("bash", "$ npm test", time.monotonic() - 14)
        left, right = ui._running_row(ui._running[0])
        self.assertIn("bash  $ npm test", left.plain)
        self.assertEqual(right.plain, "14s")

    def test_api_error_is_reported_not_raised(self):
        ui = make_ui(self.ws)
        agent = self.agent(ui, [TransientAPIError(401, "invalid api key")])
        ui.turn(agent, "hello", self.session)
        self.assertIn("✗ API error", output(ui))

    def test_ascii_mode_and_plain_theme(self):
        ui = make_ui(self.ws, keys=["y"], ascii_mode=True, theme="plain")
        agent = self.agent(ui, [
            reply(tool_calls=[tool_call("1", "edit_file", path="app.py", old_string="return 42", new_string="return 7")]),
            reply(content="Done."),
        ])
        ui.turn(agent, "edit", self.session)
        out = output(ui)
        self.assertIn("* edit_file", out)
        self.assertNotIn("●", out)
        self.assertNotIn("╭", out)  # ASCII panel borders

    def test_banner_and_status_bar(self):
        ui = make_ui(self.ws)
        agent = self.agent(ui, [], plan_mode=True)
        ui.banner(agent)
        self.assertIn("○ kyklos  ·  fake-model  ·  128k context  ·", output(ui))
        agent.context_source = "default"
        ui.banner(agent)
        self.assertIn("128k context (default)", output(ui))
        bar = "".join(text for _, text in ui.toolbar())
        self.assertIn("fake-model", bar)
        self.assertIn(" PLAN ", bar)
        self.assertRegex(bar, r"ctx \d+% ")
        self.assertIn("(plan) ", "".join(text for _, text in ui.prompt_message()))


class TrustPanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_panel_explains_and_takes_single_keys(self):
        for key, expected in (("y", "yes"), ("o", "once"), ("n", "no"), ("\x1b", "no")):
            ui = make_ui(self.ws, keys=[key])
            self.assertEqual(ui.ask_trust(self.ws, broad=False), expected)
        out = output(ui)
        self.assertIn("Trust this folder?", out)
        self.assertIn("AGENTS.md", out)
        self.assertIn("y yes, trust it  ·  o once, this session  ·  n no, quit", out)

    def test_broad_folder_offers_no_remember_option(self):
        ui = make_ui(self.ws, keys=["y", "o"])  # "y" isn't offered, so it's ignored; "o" answers
        self.assertEqual(ui.ask_trust(self.ws, broad=True), "once")
        out = output(ui)
        self.assertIn("home folder or a drive root", out)
        self.assertNotIn("y yes", out)

    def test_no_keyboard_means_no(self):
        self.assertEqual(make_ui(self.ws).ask_trust(self.ws, broad=False), "no")


class CommandRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        self.said = []
        self.agent = Agent(FakeClient([]), "fake", self.ws, on_event=None)
        self.ctx = CommandContext(self.agent, lambda kind, text: self.said.append((kind, text)),
                                  self.ws / ".sessions" / "last.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_help_lists_every_command(self):
        run_command(self.ctx, "/help")
        for name in ("help", "plan", "undo", "save", "resume", "compact", "reset", "exit"):
            self.assertIn(f"/{name}", self.said[-1][1])

    def test_plan_toggles_and_plan_with_request_returns_it(self):
        self.assertIsNone(run_command(self.ctx, "/plan"))
        self.assertTrue(self.agent.plan_mode)
        run_command(self.ctx, "/plan")
        self.assertFalse(self.agent.plan_mode)
        self.assertEqual(run_command(self.ctx, "/plan design the cache"), "design the cache")
        self.assertTrue(self.agent.plan_mode)

    def test_unknown_command_and_quit(self):
        run_command(self.ctx, "/nope")
        self.assertEqual(self.said[-1][0], "error")
        self.assertIs(run_command(self.ctx, "/quit"), EXIT)
        self.assertIs(run_command(self.ctx, "/exit"), EXIT)

    def test_undo_with_nothing_warns(self):
        run_command(self.ctx, "/undo")
        self.assertEqual(self.said[-1], ("warn", "Nothing to undo."))

    def test_ui_choice(self):
        self.assertTrue(wants_aero_ui(True, True, {}))
        self.assertFalse(wants_aero_ui(True, False, {}))  # piped output
        self.assertFalse(wants_aero_ui(False, True, {}))  # piped input
        self.assertFalse(wants_aero_ui(True, True, {"KYKLOS_UI": "plain"}))


if __name__ == "__main__":
    unittest.main()
