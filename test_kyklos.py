"""Offline tests: a scripted fake model drives the real agent loop and tools.

Run:  python -m unittest test_kyklos -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from kyklos import PRUNED_MARK, Agent, preview
from tools import probe_commands


def tool_call(call_id, name, **args):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def reply(content=None, tool_calls=None, prompt_tokens=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens) if prompt_tokens else None
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class FakeClient:
    """Mimics client.chat.completions.create, returning scripted replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.used_tools = []  # whether each call offered tools (the summary call doesn't)
        self.offered = []  # tool names offered on each call
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs["messages"])))  # snapshot
        self.used_tools.append("tools" in kwargs)
        self.offered.append({t["function"]["name"] for t in kwargs.get("tools", [])})
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def assert_tool_results_paired(test, messages):
    """Every tool result must follow an assistant message that made that call."""
    open_ids = set()
    for m in messages:
        if m["role"] == "assistant":
            open_ids = {tc["id"] for tc in m.get("tool_calls", [])}
        elif m["role"] == "tool":
            test.assertIn(m["tool_call_id"], open_ids, "tool result separated from its call")


def allow_all(name, args, always):
    return True, ""


def deny_all(name, args, always):
    return False, f"The user denied this {name} call."


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def last_tool_result(self, client):
        return [m for m in client.calls[-1] if m["role"] == "tool"][-1]["content"]

    def test_write_edit_read_bash_then_answer(self):
        client = FakeClient([
            reply(tool_calls=[tool_call("1", "write_file", path="hello.py", content="print('hi')\n")]),
            reply(tool_calls=[tool_call("2", "edit_file", path="hello.py", old_string="hi", new_string="hello")]),
            reply(tool_calls=[tool_call("3", "read_file", path="hello.py")]),
            reply(tool_calls=[tool_call("4", "bash", command="echo ok")]),
            reply(content="Done."),
        ])
        agent = Agent(client, "fake", self.ws, permission=allow_all)

        self.assertEqual(agent.send("make hello.py"), "Done.")
        self.assertEqual((self.ws / "hello.py").read_text(), "print('hello')\n")
        read_result = [m for m in client.calls[3] if m["role"] == "tool"][-1]["content"]
        self.assertIn("1\tprint('hello')", read_result)
        self.assertIn("exit code 0", self.last_tool_result(client))
        self.assertIn("ok", self.last_tool_result(client))

    def test_denied_call_is_reported_and_file_untouched(self):
        client = FakeClient([
            reply(tool_calls=[tool_call("1", "write_file", path="x.txt", content="nope")]),
            reply(content="OK, I won't."),
        ])
        agent = Agent(client, "fake", self.ws, permission=deny_all)

        agent.send("write x.txt")
        self.assertFalse((self.ws / "x.txt").exists())
        self.assertIn("denied", self.last_tool_result(client))

    def test_read_file_skips_permission(self):
        (self.ws / "a.txt").write_text("data\n")
        client = FakeClient([
            reply(tool_calls=[tool_call("1", "read_file", path="a.txt")]),
            reply(content="Read it."),
        ])
        Agent(client, "fake", self.ws, permission=lambda *a: self.fail("read_file should not prompt")).send("read")
        self.assertIn("data", self.last_tool_result(client))

    def test_errors_go_back_to_model(self):
        (self.ws / "dup.txt").write_text("x x\n")
        client = FakeClient([
            reply(tool_calls=[tool_call("1", "edit_file", path="dup.txt", old_string="x", new_string="y")]),
            reply(tool_calls=[tool_call("2", "read_file", path="../outside.txt")]),
            reply(tool_calls=[tool_call("3", "no_such_tool")]),
            reply(content="Gave up."),
        ])
        Agent(client, "fake", self.ws, permission=allow_all).send("go")

        results = [m["content"] for m in client.calls[-1] if m["role"] == "tool"]
        self.assertIn("matches 2 places", results[0])
        self.assertIn("outside the workspace", results[1])
        self.assertIn("isn't available here", results[2])
        self.assertEqual((self.ws / "dup.txt").read_text(), "x x\n")


class SearchToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        (self.ws / "src").mkdir()
        (self.ws / "src" / "app.py").write_text("def main():\n    return 42\n")
        (self.ws / "src" / "util.js").write_text("function main() {}\n")
        (self.ws / "node_modules").mkdir()
        (self.ws / "node_modules" / "dep.py").write_text("def main(): pass\n")
        (self.ws / "image.bin").write_bytes(b"\0\0def main")

    def tearDown(self):
        self.tmp.cleanup()

    def test_grep_finds_lines_and_skips_noise(self):
        out = Agent(FakeClient([]), "fake", self.ws).tools.grep(r"def main")
        self.assertIn("src/app.py:1: def main():", out)
        self.assertNotIn("node_modules", out)
        self.assertNotIn("image.bin", out)

    def test_grep_glob_filter_and_case(self):
        tools = Agent(FakeClient([]), "fake", self.ws).tools
        self.assertIn("util.js", tools.grep("FUNCTION", glob="*.js", ignore_case=True))
        self.assertEqual(tools.grep("FUNCTION"), "No matches.")

    def test_glob_lists_files_and_blocks_escape(self):
        tools = Agent(FakeClient([]), "fake", self.ws).tools
        self.assertEqual(tools.glob("**/*.py"), "src/app.py")
        result = Agent(FakeClient([]), "fake", self.ws, permission=allow_all).run_tool("glob", '{"pattern": "../*"}')
        self.assertIn("without '..'", result)

    def test_brace_patterns_expand_like_a_shell(self):
        from tools import expand_braces
        self.assertEqual(expand_braces("**/*.{py,js}"), ["**/*.py", "**/*.js"])
        self.assertEqual(expand_braces("{src,lib}/*.{py,md}"), ["src/*.py", "src/*.md", "lib/*.py", "lib/*.md"])
        self.assertEqual(expand_braces("*.{py,{js,ts}}"), ["*.py", "*.js", "*.ts"])  # nested
        self.assertEqual(expand_braces("a{x}b"), ["a{x}b"])  # no comma: literal
        self.assertEqual(expand_braces("plain/*.py"), ["plain/*.py"])
        self.assertEqual(len(expand_braces("{a,b,c,d}{a,b,c,d}{a,b,c,d}{a,b,c,d}")), 64)  # capped

    def test_glob_and_grep_accept_brace_patterns(self):
        tools = Agent(FakeClient([]), "fake", self.ws).tools
        self.assertEqual(tools.glob("**/*.{py,js}").splitlines(), ["src/app.py", "src/util.js"])
        out = tools.grep("main", glob="*.{js,md}")
        self.assertIn("src/util.js", out)
        self.assertNotIn("app.py", out)
        self.assertIn("without '..'", Agent(FakeClient([]), "fake", self.ws).run_tool("glob", '{"pattern": "{src,..}/*"}'))

    def test_search_tools_run_without_permission(self):
        client = FakeClient([
            reply(tool_calls=[tool_call("1", "grep", pattern="return"), tool_call("2", "glob", pattern="src/*")]),
            reply(content="Found it."),
        ])
        Agent(client, "fake", self.ws, permission=lambda *a: self.fail("search should not prompt")).send("find")
        results = [m["content"] for m in client.calls[-1] if m["role"] == "tool"]
        self.assertIn("src/app.py:2:", results[0])
        self.assertIn("src/util.js", results[1])


class SecretFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        (self.ws / ".env").write_text("API_KEY=sk-super-secret\n")
        (self.ws / ".env.local").write_text("API_KEY=sk-also-secret\n")
        (self.ws / ".env.example").write_text("API_KEY=your-key-here\n")
        (self.ws / "server.pem").write_text("-----BEGIN PRIVATE KEY-----\n")
        self.tools = Agent(FakeClient([]), "fake", self.ws).tools

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_file_refuses_secret_files(self):
        from tools import ToolError
        for name in (".env", ".env.local", "server.pem"):
            with self.assertRaises(ToolError):
                self.tools.read_file(name)
        self.assertIn("your-key-here", self.tools.read_file(".env.example"))

    def test_grep_never_shows_secret_contents(self):
        out = self.tools.grep("API_KEY|PRIVATE")
        self.assertNotIn("secret", out)
        self.assertIn(".env.example", out)

    def test_edit_file_refuses_secret_files(self):
        from tools import ToolError
        with self.assertRaises(ToolError):
            self.tools.edit_file(".env", "sk-super-secret", "x")
        self.assertIn("sk-super-secret", (self.ws / ".env").read_text())

    def test_write_file_refuses_secret_files(self):
        from tools import ToolError
        for name in (".env", ".env.production", "id_rsa"):
            with self.assertRaises(ToolError):
                self.tools.write_file(name, "overwritten")
        self.assertIn("sk-super-secret", (self.ws / ".env").read_text())

    def test_bash_secret_guard_forces_permission_even_when_bash_always_allowed(self):
        asked = []

        def record_and_deny(name, args, always):
            asked.append((name, args["command"]))
            return False, "Denied secret access."

        client = FakeClient([
            reply(tool_calls=[
                tool_call("1", "bash", command="echo safe"),
                tool_call("2", "bash", command="cat .env.example"),
                tool_call("3", "bash", command="cat .env"),
                tool_call("4", "bash", command="Get-Content config/.env.local"),
            ]),
            reply(content="Done."),
        ])
        agent = Agent(client, "fake", self.ws, permission=record_and_deny, always_allowed={"bash"})
        agent.send("run commands")
        self.assertEqual(asked, [("bash", "cat .env"), ("bash", "Get-Content config/.env.local")])


class AgentsMdTests(unittest.TestCase):
    def test_agents_md_goes_into_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "AGENTS.md").write_text("Always use tabs for indentation.")
            prompt = Agent(FakeClient([]), "fake", Path(tmp)).messages[0]["content"]
        self.assertIn("Project instructions from AGENTS.md", prompt)
        self.assertIn("Always use tabs for indentation.", prompt)

    def test_no_agents_md_no_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Agent(FakeClient([]), "fake", Path(tmp)).messages[0]["content"]
        self.assertNotIn("AGENTS.md", prompt)


class CompactionTests(unittest.TestCase):
    """The fake API reports a large prompt_tokens to push the agent past the threshold."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        (self.ws / "big.txt").write_text("x" * 12_000 + "\n")  # ~3,000 tokens, bigger than the keep budget
        (self.ws / "small.txt").write_text("hello\n")

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, client):
        return Agent(client, "fake", self.ws, permission=allow_all, context_window=12_000)

    # Assistant text is never cleared in stage 1, so a long one forces stage 2 (the summary).
    BLOAT = "y" * 26_000  # ~6,500 tokens

    def test_stage1_clears_old_tool_results_without_a_summary_call(self):
        client = FakeClient([
            reply(tool_calls=[tool_call("1", "read_file", path="big.txt")], prompt_tokens=100),
            reply(tool_calls=[tool_call("2", "read_file", path="small.txt")], prompt_tokens=10_000),
            reply(content="Done."),
        ])
        self.assertEqual(self.agent(client).send("inspect the files"), "Done.")

        self.assertEqual(client.used_tools, [True, True, True])  # no summary call
        final_call = client.calls[2]
        self.assertEqual([m["role"] for m in final_call],
                         ["system", "user", "assistant", "tool", "assistant", "tool"])
        self.assertEqual(final_call[1]["content"], "inspect the files")
        self.assertTrue(final_call[3]["content"].startswith(PRUNED_MARK))
        self.assertNotIn("x" * 1000, json.dumps(final_call))
        self.assertEqual(final_call[5]["content"].split("\t")[1], "hello")  # recent result kept verbatim
        assert_tool_results_paired(self, final_call)

    def test_earlier_answers_survive_compaction(self):
        client = FakeClient([
            reply(tool_calls=[tool_call("1", "read_file", path="big.txt")], prompt_tokens=100),
            reply(content="The answer to the first question is 42."),
            reply(tool_calls=[tool_call("2", "read_file", path="big.txt")], prompt_tokens=10_000),
            reply(content="Done."),
        ])
        agent = self.agent(client)
        agent.send("first question")
        agent.send("second question")
        final_call = json.dumps(client.calls[-1])
        self.assertIn("The answer to the first question is 42.", final_call)
        self.assertIn(PRUNED_MARK, final_call)

    def test_stage2_summarizes_when_clearing_is_not_enough(self):
        client = FakeClient([
            reply(content=self.BLOAT, tool_calls=[tool_call("1", "read_file", path="small.txt")], prompt_tokens=100),
            reply(tool_calls=[tool_call("2", "read_file", path="small.txt")], prompt_tokens=10_000),
            reply(content="SUMMARY: user asked to inspect files."),  # the summary call
            reply(content="Done."),
        ])
        self.assertEqual(self.agent(client).send("inspect the files"), "Done.")

        self.assertEqual(client.used_tools, [True, True, False, True])
        self.assertIn("inspect the files", client.calls[2][1]["content"])  # summarizer saw the old turns
        final_call = client.calls[3]
        self.assertEqual([m["role"] for m in final_call], ["system", "user", "assistant", "tool"])
        self.assertIn("SUMMARY: user asked", final_call[1]["content"])
        self.assertIn("current request, verbatim:\ninspect the files", final_call[1]["content"])
        self.assertNotIn(self.BLOAT, json.dumps(final_call))
        assert_tool_results_paired(self, final_call)

    def test_stage2_keeps_earlier_answers_verbatim(self):
        client = FakeClient([
            reply(content="The first answer is 42.", prompt_tokens=100),
            reply(content=self.BLOAT, tool_calls=[tool_call("1", "read_file", path="small.txt")], prompt_tokens=100),
            reply(tool_calls=[tool_call("2", "read_file", path="small.txt")], prompt_tokens=10_000),
            reply(content="SUMMARY that forgot everything"),
            reply(content="Done."),
        ])
        agent = self.agent(client)
        agent.send("first question")
        agent.send("second question")
        self.assertEqual(client.used_tools, [True, True, True, False, True])
        note = client.calls[-1][1]["content"]
        self.assertIn("- User: first question\n  You answered: The first answer is 42.", note)

    def test_never_splits_a_call_from_its_results(self):
        client = FakeClient([
            reply(content=self.BLOAT, tool_calls=[tool_call("1", "read_file", path="small.txt"),
                                                  tool_call("2", "read_file", path="small.txt")], prompt_tokens=100),
            reply(tool_calls=[tool_call("3", "read_file", path="small.txt")], prompt_tokens=10_000),
            reply(content="SUMMARY"),
            reply(content="Done."),
        ])
        self.agent(client).send("go")
        final_call = client.calls[3]
        self.assertNotEqual(final_call[2]["role"], "tool")
        assert_tool_results_paired(self, final_call)

    def test_latest_step_kept_even_when_over_budget(self):
        client = FakeClient([
            reply(content=self.BLOAT, tool_calls=[tool_call("1", "read_file", path="small.txt")], prompt_tokens=100),
            reply(tool_calls=[tool_call("2", "read_file", path="big.txt")], prompt_tokens=10_000),
            reply(content="SUMMARY"),
            reply(content="Done."),
        ])
        self.agent(client).send("go")
        final_call = client.calls[3]
        self.assertEqual([m["role"] for m in final_call], ["system", "user", "assistant", "tool"])
        self.assertIn("x" * 1000, final_call[3]["content"])  # the big result the model just asked for survives
        assert_tool_results_paired(self, final_call)

    def test_stops_instead_of_compacting_forever(self):
        client = FakeClient(
            [reply(tool_calls=[tool_call("0", "read_file", path="big.txt")], prompt_tokens=100)]
            + [reply(tool_calls=[tool_call(str(i), "read_file", path="big.txt")], prompt_tokens=10_000)
               for i in range(1, 5)]
        )
        result = self.agent(client).send("read big.txt over and over")
        self.assertTrue(result.startswith("(Stopped: compacted 3 times"))
        self.assertIn("CONTEXT_WINDOW", result)
        self.assertEqual(client.replies, [])  # used every scripted reply, made no extra API calls

    def test_short_conversation_is_not_compacted(self):
        agent = self.agent(FakeClient([]))
        agent.messages.append({"role": "user", "content": "hi"})
        self.assertFalse(agent.compact())


class SubagentTests(unittest.TestCase):
    """Parent and subagent share one FakeClient, so replies are scripted in the order the calls happen."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        (self.ws / "app.py").write_text("def main():\n    return 42\n")

    def tearDown(self):
        self.tmp.cleanup()

    def task(self, call_id, prompt, mode=None):
        args = {"description": "find main", "prompt": prompt}
        if mode:
            args["mode"] = mode
        return tool_call(call_id, "task", **args)

    def test_subagent_starts_fresh_and_only_its_answer_returns(self):
        client = FakeClient([
            reply(content="Delegating.", tool_calls=[self.task("t1", "Find where main is defined.")]),  # parent
            reply(tool_calls=[tool_call("s1", "grep", pattern="def main")]),  # subagent
            reply(content="main is defined in app.py line 1."),  # subagent's final answer
            reply(content="It's in app.py."),  # parent
        ])
        parent = Agent(client, "fake", self.ws, permission=lambda *a: self.fail("explore must not prompt"))
        parent.send("Where is main? Secret parent context: BANANA")

        sub_first_call = client.calls[1]
        self.assertEqual([m["role"] for m in sub_first_call], ["system", "user"])  # fresh context
        self.assertEqual(sub_first_call[1]["content"], "Find where main is defined.")
        self.assertNotIn("BANANA", json.dumps(sub_first_call))
        self.assertIn("You are a subagent", sub_first_call[0]["content"])
        self.assertEqual(client.offered[1], {"read_file", "grep", "glob"})  # explore is read-only

        parent_final = client.calls[3]
        self.assertEqual(parent_final[-1], {"role": "tool", "tool_call_id": "t1",
                                            "content": "main is defined in app.py line 1."})
        self.assertNotIn("return 42", json.dumps(parent_final))  # the grep output stayed in the subagent

    def test_explore_subagent_cannot_write(self):
        client = FakeClient([
            reply(tool_calls=[self.task("t1", "Delete app.py")]),
            reply(tool_calls=[tool_call("s1", "write_file", path="app.py", content="")]),
            reply(content="I can't write files."),
            reply(content="OK."),
        ])
        Agent(client, "fake", self.ws, permission=allow_all).send("go")
        sub_result = [m for m in client.calls[2] if m["role"] == "tool"][-1]["content"]
        self.assertIn("isn't available here", sub_result)
        self.assertIn("return 42", (self.ws / "app.py").read_text())

    def test_general_subagent_asks_permission_through_the_parent(self):
        asked = []

        def record(name, args, always):
            asked.append(name)
            always.add(name)  # the user answers "always"
            return True, ""

        client = FakeClient([
            reply(tool_calls=[self.task("t1", "Create notes.txt", mode="general")]),
            reply(tool_calls=[tool_call("s1", "write_file", path="notes.txt", content="hi")]),
            reply(content="Created notes.txt."),
            reply(tool_calls=[tool_call("p1", "write_file", path="more.txt", content="x")]),
            reply(content="Done."),
        ])
        Agent(client, "fake", self.ws, permission=record).send("go")
        self.assertEqual((self.ws / "notes.txt").read_text(), "hi")
        self.assertEqual(asked, ["write_file"])  # "always" in the subagent carried over to the parent
        self.assertTrue((self.ws / "more.txt").exists())

    def test_subagents_cannot_start_subagents(self):
        client = FakeClient([
            reply(tool_calls=[self.task("t1", "Go deeper", mode="general")]),
            reply(tool_calls=[self.task("s1", "Even deeper")]),
            reply(content="Couldn't delegate."),
            reply(content="OK."),
        ])
        Agent(client, "fake", self.ws, permission=allow_all).send("go")
        self.assertNotIn("task", client.offered[1])
        sub_result = [m for m in client.calls[2] if m["role"] == "tool"][-1]["content"]
        self.assertIn("isn't available here", sub_result)

    def test_bad_mode_is_reported(self):
        client = FakeClient([reply(tool_calls=[self.task("t1", "x", mode="admin")]), reply(content="OK.")])
        Agent(client, "fake", self.ws).send("go")
        self.assertIn("mode must be one of", client.calls[1][-1]["content"])

    def test_repair_after_interrupt_answers_dangling_calls(self):
        agent = Agent(FakeClient([]), "fake", self.ws)
        agent.messages += [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "ok"},
        ]
        agent.repair_after_interrupt()
        results = {m["tool_call_id"]: m["content"] for m in agent.messages if m["role"] == "tool"}
        self.assertEqual(results, {"a": "ok", "b": "Interrupted by the user."})


class EditAndPermissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        self.tools = Agent(FakeClient([]), "fake", self.ws).tools

    def tearDown(self):
        self.tmp.cleanup()

    def test_edit_file_ignores_trailing_whitespace_differences(self):
        (self.ws / "a.py").write_text("def f():   \n    return 1  \n")
        self.tools.edit_file("a.py", "def f():\n    return 1", "def f():\n    return 2")
        self.assertEqual((self.ws / "a.py").read_text(), "def f():\n    return 2\n")

    def test_edit_file_matches_shifted_indentation_and_reindents_replacement(self):
        (self.ws / "cls.py").write_text("class C:\n    def m(self):\n        if True:\n            return 1\n")
        # Model omits class-level 4-space indentation in both old_string and new_string
        self.tools.edit_file(
            "cls.py",
            "def m(self):\n    if True:\n        return 1",
            "def m(self):\n    if True:\n        return 2",
        )
        self.assertEqual(
            (self.ws / "cls.py").read_text(),
            "class C:\n    def m(self):\n        if True:\n            return 2\n",
        )

    def test_edit_file_rejects_empty_old_string(self):
        from tools import ToolError
        (self.ws / "a.py").write_text("x = 1\n")
        with self.assertRaises(ToolError):
            self.tools.edit_file("a.py", "", "x = 2\n")

    def test_invalid_edit_or_secret_fails_without_prompting_permission(self):
        (self.ws / "a.py").write_text("x = 1\n")
        (self.ws / ".env").write_text("SECRET=1\n")
        client = FakeClient([
            reply(tool_calls=[
                tool_call("1", "edit_file", path="a.py", old_string="not_there", new_string="y = 2"),
                tool_call("2", "edit_file", path=".env", old_string="SECRET=1", new_string="SECRET=2"),
                tool_call("3", "write_file", path=".env", content="SECRET=2"),
                tool_call("4", "write_file", path="../escape.txt", content="bad"),
            ]),
            reply(content="Failed."),
        ])
        agent = Agent(client, "fake", self.ws, permission=lambda *a: self.fail("invalid call must not prompt"))
        agent.send("try bad edits")
        results = [m["content"] for m in client.calls[-1] if m["role"] == "tool"]
        self.assertIn("old_string not found", results[0])
        self.assertIn("Refusing to edit .env", results[1])
        self.assertIn("Refusing to write .env", results[2])
        self.assertIn("outside the workspace", results[3])

    def test_bash_prefix_permission_allows_matching_commands_and_chains_only(self):
        from kyklos import is_auto_allowed, suggest_prefix
        self.assertEqual(suggest_prefix("git status -s"), "git status")
        self.assertEqual(suggest_prefix("py -m unittest test_kyklos -v"), "py -m unittest")
        self.assertEqual(suggest_prefix("pytest -q"), "pytest")

        allowed = {"bash:git status", "bash:py -m unittest"}
        self.assertTrue(is_auto_allowed("bash", {"command": "git status"}, allowed))
        self.assertTrue(is_auto_allowed("bash", {"command": "git status -s"}, allowed))
        self.assertTrue(is_auto_allowed("bash", {"command": "git status && py -m unittest -v"}, allowed))
        # Different subcommand, chained unknown command, redirection, pipe, or subshell must prompt:
        self.assertFalse(is_auto_allowed("bash", {"command": "git push"}, allowed))
        self.assertFalse(is_auto_allowed("bash", {"command": "git status && rm -rf ."}, allowed))
        self.assertFalse(is_auto_allowed("bash", {"command": "git status > out.txt"}, allowed))
        self.assertFalse(is_auto_allowed("bash", {"command": "git status | sh"}, allowed))
        self.assertFalse(is_auto_allowed("bash", {"command": "git status $(whoami)"}, allowed))

    def test_no_prefix_suggested_when_a_command_does_not_start_with_a_program(self):
        from kyklos import suggest_prefix
        # Seen in a real session: a PowerShell script starting with a string literal suggested '"---'.
        self.assertEqual(suggest_prefix('"--- electron binary present? ---"; if (Test-Path x) { 1 }'), "")
        for command in ("if (Test-Path .env) { 1 }", "try { Invoke-RestMethod x } catch { 2 }",
                        "$r = Get-Item x", "{ ls }", "(Get-Date)", "@('a') | % { $_ }"):
            self.assertEqual(suggest_prefix(command), "", command)
        # Real programs still get a prefix.
        self.assertEqual(suggest_prefix("git status -s"), "git status")
        self.assertEqual(suggest_prefix("npm audit --omit=dev 2>&1"), "npm audit")
        self.assertEqual(suggest_prefix(r".\scripts\build.ps1 -Release"), r".\scripts\build.ps1")
        self.assertEqual(suggest_prefix("Get-ChildItem -Recurse"), "Get-ChildItem")

    def test_ask_permission_prefix_option_saves_rule(self):
        import builtins
        from unittest import mock
        from kyklos import ask_permission, is_auto_allowed
        always = set()
        with mock.patch.object(builtins, "input", return_value="p"):
            ok, _ = ask_permission("bash", {"command": "git status -s"}, always)
        self.assertTrue(ok)
        self.assertIn("bash:git status", always)
        self.assertTrue(is_auto_allowed("bash", {"command": "git status --porcelain"}, always))

    def test_format_diff_shows_removed_and_added_lines(self):
        from kyklos import format_diff
        out = format_diff("a = 1\nb = 2\n", "a = 1\nb = 3\n")
        self.assertIn("-b = 2", out)
        self.assertIn("+b = 3", out)


class PlanModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        (self.ws / "app.py").write_text("x = 1\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_mode_restricts_tools_and_toggles_cleanly(self):
        client = FakeClient([
            reply(tool_calls=[
                tool_call("1", "read_file", path="app.py"),
                tool_call("2", "write_file", path="app.py", content="x = 2\n"),
                tool_call("3", "task", description="edit", prompt="change x", mode="general"),
            ]),
            reply(content="Here is the plan."),
            reply(tool_calls=[tool_call("4", "edit_file", path="app.py", old_string="x = 1", new_string="x = 2")]),
            reply(content="Applied the plan."),
        ])
        agent = Agent(client, "fake", self.ws, permission=allow_all, plan_mode=True)
        self.assertIn("PLAN MODE", agent.messages[0]["content"])
        self.assertEqual(agent.send("propose a change"), "Here is the plan.")
        self.assertEqual(client.offered[0], {"read_file", "grep", "glob", "task"})
        self.assertEqual((self.ws / "app.py").read_text(), "x = 1\n")

        tool_outputs = [m["content"] for m in client.calls[1] if m["role"] == "tool"]
        self.assertIn("x = 1", tool_outputs[0])
        self.assertIn("isn't available here", tool_outputs[1])
        self.assertIn("mode 'explore'", tool_outputs[2])

        # Switch out of plan mode: conversation stays, write/edit tools return
        agent.set_plan_mode(False)
        self.assertNotIn("PLAN MODE", agent.messages[0]["content"])
        self.assertEqual(agent.send("go ahead"), "Applied the plan.")
        self.assertIn("edit_file", client.offered[2])
        self.assertEqual((self.ws / "app.py").read_text(), "x = 2\n")


class SessionAndUndoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        (self.ws / "existing.py").write_text("v = 1\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_undo_reverts_created_and_edited_files_and_conversation(self):
        client = FakeClient([
            reply(tool_calls=[
                tool_call("1", "edit_file", path="existing.py", old_string="v = 1", new_string="v = 2"),
                tool_call("2", "write_file", path="new.py", content="created = True\n"),
            ]),
            reply(content="Changed files."),
        ])
        agent = Agent(client, "fake", self.ws, permission=allow_all)
        self.assertIsNone(agent.undo())

        agent.send("update files")
        self.assertEqual((self.ws / "existing.py").read_text(), "v = 2\n")
        self.assertTrue((self.ws / "new.py").exists())

        reverted = agent.undo()
        self.assertEqual(reverted, ["existing.py", "new.py"])
        self.assertEqual((self.ws / "existing.py").read_text(), "v = 1\n")
        self.assertFalse((self.ws / "new.py").exists())
        self.assertEqual(len(agent.messages), 1)  # back to just the system prompt
        self.assertEqual(agent.turn_log, [])

    def test_undo_reverts_subagent_writes(self):
        client = FakeClient([
            reply(tool_calls=[tool_call("t1", "task", description="sub write", prompt="make sub.txt", mode="general")]),
            reply(tool_calls=[tool_call("s1", "write_file", path="sub.txt", content="from sub\n")]),
            reply(content="Subagent done."),
            reply(content="Parent done."),
        ])
        agent = Agent(client, "fake", self.ws, permission=allow_all)
        agent.send("delegate write")
        self.assertTrue((self.ws / "sub.txt").exists())

        reverted = agent.undo()
        self.assertEqual(reverted, ["sub.txt"])
        self.assertFalse((self.ws / "sub.txt").exists())

    def test_save_and_load_session_roundtrip(self):
        client = FakeClient([
            reply(tool_calls=[tool_call("1", "read_file", path="existing.py")], prompt_tokens=321),
            reply(content="existing.py has v = 1."),
        ])
        agent = Agent(client, "fake", self.ws, permission=allow_all, plan_mode=True)
        agent.send("what is in existing.py?")
        session_file = self.ws / "chat.jsonl"
        agent.save_session(session_file)

        restored = Agent(FakeClient([]), "fake", self.ws)
        count = restored.load_session(session_file)
        self.assertEqual(count, 4)  # user, assistant(tool_call), tool, assistant
        self.assertTrue(restored.plan_mode)
        self.assertEqual(restored.turn_log, [("what is in existing.py?", "existing.py has v = 1.")])
        self.assertEqual(restored.usage_mark, (321, 2))


class TransientAPIError(Exception):
    def __init__(self, status_code: int, message: str = "rate limited"):
        super().__init__(message)
        self.status_code = status_code


class RetryAndObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        (self.ws / "app.py").write_text("x = 1\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_retries_transient_429_and_503_with_exponential_backoff(self):
        sleeps, events = [], []
        client = FakeClient([
            TransientAPIError(429, "rate limit"),
            TransientAPIError(503, "overloaded"),
            reply(content="Recovered."),
        ])
        agent = Agent(
            client, "fake", self.ws,
            on_event=lambda ev, **kw: events.append((ev, kw)),
            sleep=sleeps.append,
        )
        self.assertEqual(agent.send("hello"), "Recovered.")
        self.assertEqual(sleeps, [1.0, 2.0])
        # Retries happen inside one model wait, so a UI spinner keeps running through them.
        self.assertEqual([e[0] for e in events], ["model_start", "retry", "retry", "model_end"])
        retries = [kw for ev, kw in events if ev == "retry"]
        self.assertEqual([r["attempt"] for r in retries], [1, 2])

    def test_model_end_fires_even_when_the_call_fails(self):
        events = []
        client = FakeClient([TransientAPIError(401, "invalid api key")])
        agent = Agent(client, "fake", self.ws, on_event=lambda ev, **kw: events.append(ev), sleep=lambda s: None)
        with self.assertRaises(TransientAPIError):
            agent.send("hello")
        self.assertEqual(events, ["model_start", "model_end"])  # the spinner always stops

    def test_non_retryable_error_raises_immediately(self):
        sleeps = []
        client = FakeClient([TransientAPIError(401, "invalid api key")])
        agent = Agent(client, "fake", self.ws, on_event=None, sleep=sleeps.append)
        with self.assertRaises(TransientAPIError):
            agent.send("hello")
        self.assertEqual(sleeps, [])

    def test_custom_observer_receives_events_without_printing_to_stdout(self):
        import contextlib
        import io
        events = []
        client = FakeClient([
            reply(content="Checking.", tool_calls=[
                tool_call("t1", "task", description="search app", prompt="read app.py"),
            ]),
            reply(tool_calls=[tool_call("s1", "read_file", path="app.py")]),
            reply(content="app.py has x = 1."),
            reply(content="All done."),
        ])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            agent = Agent(
                client, "fake", self.ws,
                on_event=lambda ev, depth=0, **kw: events.append((ev, depth, kw)),
            )
            answer = agent.send("inspect app.py")

        self.assertEqual(answer, "All done.")
        self.assertEqual(buf.getvalue(), "")  # zero direct print() calls inside Agent
        # Every API call (2 for the parent, 2 for the subagent) is wrapped in model_start / model_end.
        model = [(ev, depth) for ev, depth, _ in events if ev.startswith("model_")]
        self.assertEqual(model, [("model_start", 0), ("model_end", 0), ("model_start", 1), ("model_end", 1),
                                 ("model_start", 1), ("model_end", 1), ("model_start", 0), ("model_end", 0)])
        kinds = [(ev, depth) for ev, depth, _ in events if not ev.startswith("model_")]
        self.assertEqual(kinds, [
            ("thought", 0),
            ("tool_call", 0),
            ("subagent_start", 1),
            ("tool_call", 1),
            ("tool_result", 1),
            ("subagent_end", 1),
            ("tool_result", 0),
        ])


class TrustTests(unittest.TestCase):
    """Folder trust and per-project data dirs, with a temporary ~/.kyklos."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.home, self.project = root / "kyklos-home", root / "work" / "my app"
        self.project.mkdir(parents=True)
        self.asked = []

    def tearDown(self):
        self.tmp.cleanup()

    def ask(self, answer):
        def fn(folder, broad):
            self.asked.append((folder, broad))
            return answer
        return fn

    def test_project_dirs_are_per_folder_slugs(self):
        from kyklos import project_dir, project_slug
        slug = project_slug(self.project)
        self.assertNotRegex(slug, r"[\\/: ]")
        self.assertTrue(slug.endswith("work-my-app"))
        self.assertEqual(project_dir(self.home, self.project), self.home / "projects" / slug)
        self.assertNotEqual(project_slug(self.project), project_slug(self.project.parent))

    def test_yes_is_remembered_and_covers_subfolders_only(self):
        from kyklos import is_trusted, resolve_trust
        self.assertTrue(resolve_trust(self.home, self.project, True, False, self.ask("yes")))
        self.assertTrue(is_trusted(self.home, self.project))
        (self.project / "src").mkdir()
        self.assertTrue(is_trusted(self.home, self.project / "src"))
        self.assertFalse(is_trusted(self.home, self.project.parent))
        sibling = self.project.parent / "my app 2"  # shares a prefix, but isn't inside
        sibling.mkdir()
        self.assertFalse(is_trusted(self.home, sibling))
        # A trusted folder never asks again.
        self.asked.clear()
        self.assertTrue(resolve_trust(self.home, self.project, True, False, self.ask("no")))
        self.assertEqual(self.asked, [])

    def test_once_and_preapproved_are_not_remembered(self):
        from kyklos import is_trusted, resolve_trust
        self.assertTrue(resolve_trust(self.home, self.project, True, False, self.ask("once")))
        self.assertTrue(resolve_trust(self.home, self.project, False, True, self.ask("no")))  # --trust
        self.assertFalse(is_trusted(self.home, self.project))

    def test_no_and_non_interactive_refuse(self):
        from kyklos import resolve_trust
        self.assertFalse(resolve_trust(self.home, self.project, True, False, self.ask("no")))
        self.asked.clear()
        self.assertFalse(resolve_trust(self.home, self.project, False, False, self.ask("yes")))
        self.assertEqual(self.asked, [])  # nobody to ask in a piped run

    def test_broad_folders_are_never_remembered(self):
        from unittest import mock
        import kyklos
        with mock.patch.object(kyklos, "is_broad_folder", return_value=True):
            self.assertTrue(kyklos.resolve_trust(self.home, self.project, True, False, self.ask("yes")))
        self.assertEqual(self.asked[-1][1], True)
        self.assertFalse(kyklos.is_trusted(self.home, self.project))
        self.assertTrue(kyklos.is_broad_folder(Path.home()))
        self.assertTrue(kyklos.is_broad_folder(Path(Path.home().anchor)))
        self.assertFalse(kyklos.is_broad_folder(self.project))

    def test_trust_store_dedupes_and_survives_a_corrupt_file(self):
        from kyklos import trust_folder, trusted_folders
        trust_folder(self.home, self.project)
        trust_folder(self.home, self.project)
        self.assertEqual(len(trusted_folders(self.home)), 1)
        (self.home / "trusted.json").write_text("{not json")
        self.assertEqual(trusted_folders(self.home), [])

    def test_plain_prompt_answers(self):
        import builtins
        from unittest import mock
        from kyklos import ask_trust_plain
        for typed, expected in (("y", "yes"), ("o", "once"), ("n", "no"), ("", "no")):
            with mock.patch.object(builtins, "input", return_value=typed), mock.patch("builtins.print"):
                self.assertEqual(ask_trust_plain(self.project, False), expected)
        with mock.patch.object(builtins, "input", side_effect=EOFError), mock.patch("builtins.print"):
            self.assertEqual(ask_trust_plain(self.project, False), "no")


class HelperTests(unittest.TestCase):
    def test_permission_prompt_denies_when_no_input(self):
        import builtins
        from unittest import mock
        from kyklos import ask_permission
        with mock.patch.object(builtins, "input", side_effect=EOFError):
            allowed, note = ask_permission("bash", {"command": "dir"}, set())
        self.assertFalse(allowed)
        self.assertIn("denied", note)

    def test_preview_truncates_long_output(self):
        text = "\n".join(f"line {i}" for i in range(20))
        shown = preview(text)
        self.assertIn("line 5", shown)
        self.assertNotIn("line 6", shown)
        self.assertIn("14 more lines", shown)

    def test_preview_short_output_untouched(self):
        self.assertNotIn("more lines", preview("exit code 0\nok"))

    def test_system_prompt_lists_working_commands(self):
        working, broken = probe_commands()
        prompt = Agent(FakeClient([]), "fake", Path.cwd()).messages[0]["content"]
        for cmd in working:
            self.assertIn(cmd, prompt)
        for cmd in broken:
            self.assertNotIn(cmd, working)


if __name__ == "__main__":
    unittest.main()
