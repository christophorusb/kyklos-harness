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
        return self.replies.pop(0)


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
