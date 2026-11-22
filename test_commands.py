"""Offline tests for custom slash commands and their terminal and serve entry points."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from kyklos import (COMMANDS, EXIT, MAX_INSTRUCTIONS_CHARS, Agent, CommandContext, permission_options,
                    run_command)
from test_kyklos import FakeClient, png_bytes, reply, tool_call
from test_serve import ServeTestCase
from test_ui import make_ui


def put_command(folder, filename, text):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_text(text, encoding="utf-8")
    return path


class CommandTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        self.ws, self.home = root / "workspace", root / "home"
        self.ws.mkdir()
        self.user, self.project = self.home / "commands", self.ws / ".agents" / "commands"
        self.notices, self.said = [], []

    def agent(self, replies=(), **kw):
        def observer(event, **data):
            if event == "notice":
                self.notices.append(data["text"])

        client = FakeClient(replies)
        agent = Agent(client, "fake", self.ws, home=self.home, on_event=observer, **kw)
        ctx = CommandContext(agent, lambda kind, text: self.said.append((kind, text)), self.ws / "last.jsonl")
        return agent, ctx, client

    def test_frontmatter_and_arguments_go_through_a_whole_turn(self):
        path = put_command(self.project, "review.md",
                           '---\ndescription: "Review: a file"\nargument-hint: <file>\n---\n'
                           'Review $ARGUMENTS.\n\nTarget: $ARGUMENTS.\n')
        (self.ws / "file with spaces.py").write_text("x = 1\n")
        agent, ctx, client = self.agent([
            reply(tool_calls=[tool_call("read", "read_file", path="file with spaces.py")]),
            reply("Looks good."),
        ])
        prompt = run_command(ctx, '/review "file with spaces.py"')
        self.assertEqual(prompt, 'Review "file with spaces.py".\n\nTarget: "file with spaces.py".')
        self.assertEqual(self.said, [("info", f"Using project command /review from {path}.")])
        self.assertEqual(agent.send(prompt), "Looks good.")
        self.assertEqual(client.calls[0][-1], {"role": "user", "content": prompt})
        self.assertIn("x = 1", client.calls[1][-1]["content"])
        run_command(ctx, "/help")
        self.assertRegex(self.said[-1][1], r"/review <file> +Review: a file \[project\]")

    def test_no_frontmatter_or_argument(self):
        put_command(self.user, "check-2.md", "Check [$ARGUMENTS].\n")
        agent, ctx, _ = self.agent()
        self.assertEqual(run_command(ctx, "/check-2"), "Check [].")
        self.assertEqual(run_command(ctx, "/check-2 $ARGUMENTS"), "Check [$ARGUMENTS].")
        self.assertEqual(agent.custom_commands["check-2"].usage, "/check-2")
        run_command(ctx, "/help")
        self.assertRegex(self.said[-1][1], r"/check-2 +Custom command \[user\]")

    def test_arguments_are_appended_without_a_placeholder(self):
        put_command(self.project, "fix.md", "Fix the failing test.\n")
        _, ctx, _ = self.agent()
        self.assertEqual(run_command(ctx, "/fix"), "Fix the failing test.")
        self.assertEqual(run_command(ctx, "/fix only the parser test"),
                         "Fix the failing test.\nonly the parser test")

    def test_malformed_headers_and_bad_names_are_reported(self):
        for name, body in {
            "broken.md": "---\ndescription missing colon\n---\nText",
            "unclosed.md": "---\ndescription: Missing close\n",
            "Uppercase.md": "Text",
            "x" * 41 + ".md": "Text",
            "wrong.txt": "Text",
        }.items():
            put_command(self.project, name, body)
        put_command(self.project, "x" * 40 + ".md", "Text")
        agent, _, _ = self.agent()
        self.assertEqual(set(agent.custom_commands), {"x" * 40})
        self.assertEqual(len(self.notices), 5)
        for filename in ("broken.md", "unclosed.md", "Uppercase.md", "x" * 41 + ".md", "wrong.txt"):
            self.assertEqual(sum(filename in text for text in self.notices), 1)

    def test_precedence_and_session_scoping(self):
        builtins = dict(COMMANDS)
        for folder, text in ((self.user, "User prompt"), (self.project, "Project prompt")):
            for name in ("review.md", "undo.md", "quit.md"):
                put_command(folder, name, text)
        put_command(self.project, "project-only.md", "Project prompt")
        agent, ctx, _ = self.agent()
        self.assertEqual(set(agent.custom_commands), {"review", "project-only"})
        self.assertEqual(run_command(ctx, "/review"), "User prompt")
        self.assertIs(run_command(ctx, "/quit"), EXIT)
        self.assertEqual(len(self.notices), 5)
        self.assertEqual(sum("built-in /undo takes precedence" in t for t in self.notices), 2)
        self.assertEqual(sum("built-in /quit takes precedence" in t for t in self.notices), 2)
        self.assertEqual(sum("user command /review takes precedence" in t for t in self.notices), 1)
        put_command(self.user, "review.md", "Changed prompt")
        run_command(ctx, "/reset")
        self.assertEqual(run_command(ctx, "/review"), "User prompt")
        self.assertEqual(len(self.notices), 5)
        _, new_ctx, _ = self.agent()
        self.assertEqual(run_command(new_ctx, "/review"), "Changed prompt")
        self.assertEqual(COMMANDS, builtins)

    def test_body_is_capped_and_metadata_and_body_are_scrubbed(self):
        secret = "local-commands-test-secret"
        put_command(self.ws, ".env", f"TEST_TOKEN={secret}\n")
        put_command(self.project, "long.md", f"---\ndescription: {secret}\n---\n"
                    + secret + "x" * MAX_INSTRUCTIONS_CHARS)
        agent, ctx, _ = self.agent()
        prompt = run_command(ctx, "/long")
        self.assertLessEqual(len(prompt), MAX_INSTRUCTIONS_CHARS)
        self.assertTrue(prompt.startswith("[REDACTED]"))
        self.assertEqual(agent.custom_commands["long"].summary, "[REDACTED]")

    def test_symlinked_commands_are_skipped(self):
        path = put_command(self.user, "review.md", "User prompt")
        self.project.mkdir(parents=True)
        try:
            (self.project / "link.md").symlink_to(path)
        except OSError:
            self.skipTest("symlinks are unavailable")
        agent, _, _ = self.agent()
        self.assertEqual(set(agent.custom_commands), {"review"})
        self.assertTrue(any("link.md" in t and "symlinked" in t for t in self.notices))

    def test_only_arguments_can_attach_files(self):
        put_command(self.project, "check.md", "Check @template.png and $ARGUMENTS.")
        (self.ws / "template.png").write_bytes(png_bytes())
        (self.ws / "shot.png").write_bytes(png_bytes())
        agent, ctx, client = self.agent([reply("Done.")], vision=True)
        prompt = run_command(ctx, "/check @shot.png")
        self.assertEqual(prompt, r"Check \@template.png and @shot.png.")
        agent.send(prompt)
        parts = client.calls[0][-1]["content"]
        self.assertEqual([part["type"] for part in parts], ["text", "image_url"])
        self.assertIn("Attached shot.png", "\n".join(self.notices))
        self.assertNotIn("Attached template.png", "\n".join(self.notices))

    def test_project_commands_require_per_call_approval_for_writes(self):
        path = put_command(self.project, "review.md", "Original prompt")
        rel = path.relative_to(self.ws).as_posix()
        asked = []

        def deny(name, args, always):
            touches_secret, prefix, findings = permission_options(name, args)
            self.assertFalse(touches_secret)
            self.assertFalse(prefix)
            self.assertTrue(any(level == "danger" and ".agents/commands" in text for level, text in findings))
            asked.append(name)
            return False, "Denied."

        agent, _, _ = self.agent(permission=deny, always_allowed={"write_file", "edit_file", "bash", "bash:cp"})
        agent.auto_bash = True
        calls = [
            ("write_file", {"path": rel, "content": "Changed"}),
            ("edit_file", {"path": rel, "old_string": "Original", "new_string": "Changed"}),
            ("bash", {"command": "printf 'New prompt' > .agents/commands/new.md"}),
            ("bash", {"command": f"cp copy.md {rel}"}),
            ("bash", {"command": f"mv {rel} review-backup.md"}),
        ]
        for name, args in calls:
            with self.subTest(name=name, args=args):
                self.assertEqual(agent.run_tool(name, json.dumps(args)), "Denied.")
        self.assertEqual(asked, [name for name, _ in calls])
        self.assertEqual(path.read_text(), "Original prompt")
        self.assertFalse((self.project / "new.md").exists())

    def test_symlink_alias_to_commands_also_requires_approval(self):
        put_command(self.project, "review.md", "Original prompt")
        try:
            (self.ws / "alias").symlink_to(self.project, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks are unavailable")
        permission = mock.Mock(return_value=(False, "Denied."))
        agent, _, _ = self.agent(permission=permission, always_allowed={"write_file", "bash"})
        self.assertEqual(agent.run_tool("write_file", '{"path": "alias/new.md", "content": "New prompt"}'), "Denied.")
        self.assertEqual(agent.run_tool("bash", '{"command": "printf New > alias/new.md"}'), "Denied.")
        self.assertEqual(permission.call_count, 2)
        self.assertFalse((self.project / "new.md").exists())

    def test_aero_completion_shows_usage_and_description(self):
        put_command(self.project, "review.md", "---\ndescription: Review a file\nargument-hint: <file>\n---\nCheck.")
        agent, _, _ = self.agent()
        ui = make_ui(self.ws)
        ui.agent = agent
        with mock.patch("prompt_toolkit.PromptSession") as session:
            ui._session(self.home / "history")
        completer = session.call_args.kwargs["completer"]
        completions = list(completer.get_completions(Document("/rev"), CompleteEvent()))
        self.assertEqual(sorted(c.text for c in completions), ["/review", "/reviewer"])  # built-in /reviewer too
        completion = next(c for c in completions if c.text == "/review")
        self.assertEqual(completion.text, "/review")
        self.assertEqual(completion.display_text, "/review <file>")
        self.assertEqual(completion.display_meta_text, "Review a file")


class ServeCommandTests(ServeTestCase):
    def test_trusted_start_lists_commands_and_run_sends_source_before_expansion(self):
        self.addCleanup(self.ui.inp.close)
        self.addCleanup(self.client.from_server.close)
        path = put_command(self.ws / ".agents" / "commands", "review.md",
                           "---\ndescription: Review a file\nargument-hint: <file>\n---\nReview $ARGUMENTS.")

        def script(client):
            self.assertNotIn("review", [c["name"] for c in self.init["commands"]])
            commands = self.started["result"]["commands"]
            self.assertIn({"name": "review", "usage": "/review <file>", "summary": "Review a file",
                           "source": "project"}, commands)
            request = client.call("command/run", text="/review app.py")
            notice = client.expect(lambda message: message.get("method") == "say")
            self.assertIn(path.name, notice["params"]["text"])
            prompt = client.result(request)["result"]["then_send"]
            self.assertEqual(prompt, "Review app.py.")
            self.assertEqual(client.result(client.call("session/prompt", text=prompt))["result"]["answer"],
                             "Reviewed.")
            self.shutdown(client)

        _, fake = self.serve([reply("Reviewed.")], script)
        self.assertEqual(fake.calls[0][-1]["content"], "Review app.py.")


if __name__ == "__main__":
    unittest.main()
