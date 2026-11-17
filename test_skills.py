"""Offline tests for installed skills, their catalog, and confined reference reads."""

import json
import tempfile
import unittest
from pathlib import Path

from extensions import MAX_SKILL_CATALOG_CHARS, MAX_SKILL_DESCRIPTION_CHARS
from kyklos import Agent, CommandContext, Trace, permission_options, run_command, tool_overview
from test_kyklos import FakeClient, reply, tool_call
from test_serve import ServeTestCase


def put_skill(folder, name, description="A useful skill.", body="Follow these instructions."):
    path = folder / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n", encoding="utf-8")
    return path


class SkillTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        self.ws, self.home = root / "workspace", root / "home"
        self.user, self.project = self.home / "skills", self.ws / ".agents" / "skills"
        self.ws.mkdir()
        self.events, self.said = [], []

    def agent(self, replies=(), **kwargs):
        def no_permission(*args):
            self.fail(f"A read-only skill call asked for permission: {args[0]}")

        kwargs.setdefault("permission", no_permission)
        kwargs.setdefault("on_event", lambda event, **data: self.events.append((event, data)))
        return Agent(FakeClient(replies), "fake", self.ws, home=self.home, **kwargs)

    @staticmethod
    def load(agent, name, **args):
        return agent.run_tool("load_skill", json.dumps({"name": name, **args}))

    def test_catalog_sorted_capped_and_user_first(self):
        put_skill(self.project, "z-last", "Project skill")
        put_skill(self.project, "a-first", "Project duplicate")
        put_skill(self.user, "a-first", "User skill")
        for i in range(30):
            put_skill(self.project, f"bulk-{i:02d}", "x" * (MAX_SKILL_DESCRIPTION_CHARS + 50))
        agent = self.agent()
        catalog = agent.skill_catalog.prompt
        names = [line.split(":", 1)[0][2:] for line in catalog.splitlines() if line.startswith("- ")]
        self.assertEqual(names, sorted(names))
        self.assertEqual(names[0], "a-first")
        self.assertIn("- a-first: User skill", catalog)
        self.assertNotIn("Project duplicate", catalog)
        self.assertNotIn("Follow these instructions", catalog)
        self.assertLessEqual(len(catalog), MAX_SKILL_CATALOG_CHARS)
        for line in catalog.splitlines():
            if line.startswith("- "):
                self.assertLessEqual(len(line.split(": ", 1)[1]), MAX_SKILL_DESCRIPTION_CHARS)
        self.assertIn(f"{32 - len(names)} additional skills were omitted", catalog)
        agent.reset()
        notices = [data["text"] for event, data in self.events if event == "notice"]
        self.assertEqual(len(notices), 1)
        self.assertIn("a-first", notices[0])

    def test_skills_command_names_invalid_files_and_shows_valid_sizes(self):
        missing = put_skill(self.project, "broken")
        missing.write_text("---\nname: broken\n---\nBody\n", encoding="utf-8")
        mismatch = put_skill(self.project, "mismatch")
        mismatch.write_text("---\nname: different\ndescription: Present\n---\nBody\n", encoding="utf-8")
        put_skill(self.project, "review", "Review a change.", "Project body")
        user_path = put_skill(self.user, "notes", "Use a notes template.", "User body")
        agent = self.agent()
        ctx = CommandContext(agent, lambda kind, text: self.said.append((kind, text)), self.ws / "session.jsonl")
        run_command(ctx, "/skills")
        kind, text = self.said[-1]
        self.assertEqual(kind, "info")
        self.assertRegex(text, r"notes \[user\] \(~[0-9,]+ tokens\)")
        self.assertRegex(text, r"review \[project\] \(~[0-9,]+ tokens\)")
        self.assertIn(str(user_path), text)
        self.assertIn("Review a change.", text)
        self.assertIn(str(missing), text)
        self.assertIn("requires a name and description", text)
        self.assertIn(str(mismatch), text)
        self.assertIn("must match the folder", text)
        self.assertNotIn("User body", text)
        self.assertEqual(set(agent.skill_catalog.skills), {"notes", "review"})

    def test_folded_description_and_nested_optional_metadata(self):
        path = put_skill(self.user, "folded")
        path.write_text("---\nname: 'folded'\ndescription: >-\n  Run a review\n  on a change.\n"
                        "metadata:\n  author: someone\nallowed-tools: Bash\n---\n\nSkill body.\n",
                        encoding="utf-8")
        agent = self.agent(plan_mode=True)
        self.assertIn("- folded: Run a review on a change.", agent.messages[0]["content"])
        self.assertIn("Skill body.", self.load(agent, "folded"))
        self.assertNotIn("bash", agent.tool_names)
        self.assertEqual(agent.always_allowed, set())

    def test_unchanged_in_turns_and_refreshed_on_reset(self):
        path = put_skill(self.project, "review", "First description", "First instructions.")
        agent = self.agent([reply("One."), reply("Two.")])
        beginning = agent.messages[0]["content"]
        agent.send("One")
        put_skill(self.project, "review", "New description", "New instructions.")
        put_skill(self.user, "new-skill", "Another skill.")
        agent.send("Two")
        self.assertEqual([call[0]["content"] for call in agent.client.calls], [beginning, beginning])
        self.assertIn("First instructions.", self.load(agent, "review"))
        self.assertNotIn("new-skill", agent.skill_catalog.skills)
        self.assertEqual(agent.skill_catalog.skills["review"].path, path)
        agent.set_plan_mode(True)
        self.assertIn("First description", agent.messages[0]["content"])
        agent.reset()
        self.assertIn("New description", agent.messages[0]["content"])
        self.assertIn("New instructions.", self.load(agent, "review"))
        self.assertIn("new-skill", agent.skill_catalog.skills)

    def test_tool_and_catalog_appear_and_disappear_at_reset(self):
        agent = self.agent()
        self.assertNotIn("load_skill", agent.tool_names)
        position = put_skill(self.project, "review")
        agent.reset()
        self.assertIn("load_skill", agent.tool_names)
        position.unlink()
        agent.reset()
        self.assertNotIn("load_skill", agent.tool_names)
        self.assertNotIn("Available skills.", agent.messages[0]["content"])

    def test_load_body_and_user_reference_through_an_actual_turn(self):
        path = put_skill(self.user, "review", body="Compare changes to the rubric.")
        references = path.parent / "references"
        references.mkdir()
        reference = references / "rubric  file.txt"
        reference.write_text("Use DB_TOKEN skill-test-value for the example.", encoding="utf-8")
        (self.ws / ".env").write_text("DB_TOKEN=skill-test-value\n", encoding="utf-8")
        agent = self.agent([
            reply(tool_calls=[tool_call("body", "load_skill", name="review")]),
            reply(tool_calls=[tool_call("file", "load_skill", name="review", file="references/rubric  file.txt")]),
            reply("Reviewed."),
        ], plan_mode=True)
        self.assertEqual(agent.send("Review this"), "Reviewed.")
        self.assertIn("load_skill", agent.client.offered[0])
        self.assertIn("load_skill", tool_overview(agent)["safe"])
        body, data = [m["content"] for m in agent.messages if m["role"] == "tool"]
        self.assertIn("Compare changes to the rubric.", body)
        self.assertIn("references/rubric  file.txt", body)
        self.assertNotIn("Use DB_TOKEN", body)
        self.assertIn("(data)", data)
        self.assertIn("DB_TOKEN [REDACTED]", data)
        self.assertNotIn("skill-test-value", "".join(str(event) for event in self.events))

    def test_installation_cannot_follow_a_symlinked_definition(self):
        path = put_skill(self.user, "review")
        target = self.project / "review" / "SKILL.md"
        target.parent.mkdir(parents=True)
        try:
            target.symlink_to(path)
        except OSError:
            self.skipTest("symlinks are unavailable")
        agent = self.agent()
        self.assertEqual(len(agent.skill_catalog.skills), 1)
        self.assertEqual(agent.skill_catalog.skills["review"].source, "user")
        self.assertIn("symlinked", agent.skill_catalog.skipped[0])

    def test_absolute_parent_and_secret_references_are_refused(self):
        path = put_skill(self.user, "review")
        (path.parent / ".env").write_text("Private value", encoding="utf-8")
        agent = self.agent()
        for file, problem in (("../other.txt", "relative"), ("child/../SKILL.md", "relative"),
                              (str(path), "relative"), ("C:\\user\\other.txt", "relative"),
                              ("..\\other.txt", "relative"), (".env", "secret"),
                              ("missing.txt", "missing"), ("", "non-empty")):
            with self.subTest(file=file):
                result = self.load(agent, "review", file=file)
                self.assertTrue(result.startswith("Error:"))
                self.assertIn(problem, result)
        self.assertNotIn(".env", self.load(agent, "review"))
        self.assertTrue(self.load(agent, "unknown").startswith("Error:"))
        self.assertTrue(agent.run_tool("load_skill", "[]").startswith("Error:"))

    def test_reference_symlinks_stay_inside_skill_and_cannot_alias_secrets(self):
        root = put_skill(self.user, "review").parent
        outside = self.ws / "outside.txt"
        outside.write_text("Outside", encoding="utf-8")
        (root / ".env").write_text("Private", encoding="utf-8")
        (root / "reference.txt").write_text("Reference", encoding="utf-8")
        try:
            (root / "outside.txt").symlink_to(outside)
            (root / "hidden.txt").symlink_to(root / ".env")
            (root / "inside.txt").symlink_to(root / "reference.txt")
        except OSError:
            self.skipTest("symlinks are unavailable")
        agent = self.agent()
        self.assertIn("outside the skill folder", self.load(agent, "review", file="outside.txt"))
        self.assertIn("secret file", self.load(agent, "review", file="hidden.txt"))
        self.assertIn("Reference", self.load(agent, "review", file="inside.txt"))
        listing = self.load(agent, "review")
        self.assertNotIn("outside.txt", listing)
        self.assertNotIn("hidden.txt", listing)

    def test_both_subagent_modes_inherit_the_parent_catalog(self):
        put_skill(self.user, "review", "Review a change.", "Instructions for review.")
        agent = self.agent()
        put_skill(self.project, "new-skill", "Added during the turn.")
        for mode in ("explore", "general"):
            with self.subTest(mode=mode):
                sub, _ = agent._spawn(mode)
                self.assertIn("load_skill", sub.tool_names)
                self.assertIs(sub.skill_catalog, agent.skill_catalog)
                self.assertIn("- review: Review a change.", sub.messages[0]["content"])
                self.assertIn("Instructions for review.", self.load(sub, "review"))
                self.assertNotIn("new-skill", sub.skill_catalog.skills)
        agent.set_plan_mode(True)
        self.assertIn("load_skill", agent._spawn("explore")[0].tool_names)
        self.assertNotIn("load_skill", self.agent(depth=1).tool_names)

    def test_project_skill_writes_require_per_call_approval(self):
        path = put_skill(self.project, "review")
        rel = path.relative_to(self.ws).as_posix()
        asked = []

        def deny(name, args, always):
            _, prefix, findings = permission_options(name, args)
            self.assertFalse(prefix)
            self.assertTrue(any(level == "danger" and ".agents/skills" in text for level, text in findings))
            asked.append(name)
            return False, "Denied."

        agent = self.agent(permission=deny, always_allowed={"write_file", "edit_file", "bash", "bash:cp"})
        agent.auto_bash = True
        calls = [
            ("write_file", {"path": rel, "content": "Changed."}),
            ("edit_file", {"path": rel, "old_string": "Follow these instructions.", "new_string": "Changed."}),
            ("bash", {"command": "printf New > .agents/skills/review/new.md"}),
            ("bash", {"command": f"cp other.md {rel}"}),
        ]
        for name, args in calls:
            self.assertEqual(agent.run_tool(name, json.dumps(args)), "Denied.")
        self.assertEqual(asked, [name for name, _ in calls])
        self.assertIn("Follow these instructions.", path.read_text(encoding="utf-8"))
        self.assertFalse((path.parent / "new.md").exists())

    def test_trace_has_only_fixed_tool_names(self):
        put_skill(self.project, "review-private", body="private-skill-body")
        trace = Trace(self.ws / "trace.jsonl")
        try:
            agent = self.agent([reply(tool_calls=[tool_call("one", "load_skill", name="review-private")]),
                                reply("Done.")], on_event=trace.observe(None))
            agent.send("Review")
        finally:
            trace.close()
        trace_text = trace.path.read_text(encoding="utf-8")
        self.assertIn('"tool": "load_skill"', trace_text)
        self.assertNotIn("review-private", trace_text)
        self.assertNotIn("private-skill-body", trace_text)


class ServeSkillsTests(ServeTestCase):
    def test_skills_list_and_load_use_existing_notifications(self):
        self.addCleanup(self.ui.inp.close)
        self.addCleanup(self.client.from_server.close)
        put_skill(self.ws / ".agents" / "skills", "review", "Review a file.", "Check its types.")

        def script(client):
            self.assertIn("skills", [entry["name"] for entry in self.started["result"]["commands"]])
            list_request = client.call("command/run", text="/skills")
            shown = client.expect(lambda message: message.get("method") == "say")
            self.assertIn("review [project]", shown["params"]["text"])
            client.result(list_request)
            request = client.call("session/prompt", text="Review this.")
            self.assertEqual(client.result(request)["result"]["answer"], "Done.")
            [result] = client.updates("tool_result")
            self.assertEqual(result["data"]["name"], "load_skill")
            self.assertIn("Check its types.", result["data"]["result"])
            self.assertFalse(any(message.get("method") == "permission/request" for message in client.log))
            self.shutdown(client)

        self.serve([reply(tool_calls=[tool_call("skill", "load_skill", name="review")]), reply("Done.")], script)


if __name__ == "__main__":
    unittest.main()
