"""Offline tests for codebase dependency graphs, persistent overlay, DAG plans, waves, and /graph.

Run:  python -m unittest test_graph -v
"""

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

os.environ.setdefault("KYKLOS_SNAPSHOTS", "0")

from graph import KnowledgeOverlay, WorkspaceGraph, validate_dag
from kyklos import Agent, CommandContext, project_dir, run_command
from test_kyklos import FakeClient, RoutingClient, allow_all, deny_all, quiet, reply, stream_chunks, tool_call
from tools import ToolError, Tools, repository_map


class WorkspaceGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.ws = self.root / "ws"
        self.ws.mkdir()

    def write(self, rel: str, text: str) -> Path:
        p = self.ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def test_multi_language_imports_pagerank_and_blast_radius(self):
        self.write("pkg/__init__.py", "")
        self.write("pkg/core.py", "class Engine:\n    pass\n\ndef compute_score(xs):\n    return len(xs)\n")
        self.write("pkg/helpers.py", "from .core import compute_score\n\ndef helper():\n    return compute_score([1])\n")
        self.write("api.py", "from pkg.helpers import helper\n\ndef endpoint():\n    return helper()\n")
        self.write("web/view.ts", "export function renderCard() { return 'ok'; }\n")
        self.write("web/app.ts", "import { renderCard } from './view';\nexport function boot() { return renderCard(); }\n")
        self.write("tests/test_core.py", "from pkg.core import Engine\n\ndef test_engine():\n    assert Engine\n")

        g = WorkspaceGraph(self.ws)
        self.assertEqual(g.scores["pkg/core.py"], 1.0)
        br = g.blast_radius("pkg/core.py", depth=2)
        self.assertEqual([d["path"] for d in br["dependents"]], ["pkg/helpers.py", "api.py"])
        self.assertEqual([d["hops"] for d in br["dependents"]], [1, 2])
        self.assertEqual(br["tests"], ["tests/test_core.py"])
        self.assertIn("web/view.ts", g.neighbors("web/app.ts")["imports"])
        self.assertEqual(g.shortest_path("api.py", "pkg/core.py"), ["api.py", "pkg/helpers.py", "pkg/core.py"])

    def test_rust_and_go_resolution(self):
        self.write("src/lib.rs", "mod engine;\npub fn boot() {}\n")
        self.write("src/engine.rs", "pub fn run_engine() {}\n")
        self.write("internal/store/db.go", "package store\nfunc Open() {}\n")
        self.write("cmd/app.go", 'package main\nimport "example.com/proj/internal/store"\nfunc main() { store.Open() }\n')

        g = WorkspaceGraph(self.ws)
        self.assertIn("src/engine.rs", g.neighbors("src/lib.rs")["imports"])
        self.assertIn("internal/store/db.go", g.neighbors("cmd/app.go")["imports"])

    def test_excludes_secrets_and_outside_symlinks(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "leak.py").write_text("def stolen():\n    pass\n", encoding="utf-8")

        self.write("app.py", "def main():\n    return 1\n")
        self.write(".env", "API_KEY=secret\n")
        self.write("id_rsa", "PRIVATE\n")
        self.write("deploy.pem", "CERT\n")
        try:
            (self.ws / "linked.py").symlink_to(outside / "leak.py")
            (self.ws / "linked_dir").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks unavailable")

        g = WorkspaceGraph(self.ws)
        self.assertEqual(g.files, ["app.py"])

    def test_repository_map_promotes_central_modules_and_keeps_unlinked_shallow_first(self):
        # Unlinked: shallow-first holds
        self.write("top.py", "def top_fn(): pass\n")
        self.write("a/b/c/deep.py", "def deep_fn(): pass\n")
        unlinked_rows = repository_map(self.ws).splitlines()
        self.assertTrue(unlinked_rows[0].startswith("top.py"), unlinked_rows)

        # Linked: central imported module rises to the top
        self.write("top.py", "from a.b.c.deep import deep_fn\ndef top_fn(): return deep_fn()\n")
        linked_rows = repository_map(self.ws).splitlines()
        self.assertTrue(linked_rows[0].startswith("a/b/c/deep.py"), linked_rows)


class CodeGraphToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name).resolve()
        (self.ws / "core.py").write_text("def core_fn():\n    return 42\n", encoding="utf-8")
        (self.ws / "caller.py").write_text("from core import core_fn\n\ndef run():\n    return core_fn()\n", encoding="utf-8")
        (self.ws / ".env").write_text("SECRET=abc\n", encoding="utf-8")

    def test_tool_queries_and_secret_and_escape_guards(self):
        tools = Tools(self.ws)
        summary = tools.code_graph("summary")
        self.assertIn("core.py", summary)
        impact = tools.code_graph("impact", target="core.py")
        self.assertIn("caller.py", impact)
        with self.assertRaises(ToolError):
            tools.check("code_graph", {"action": "impact", "target": ".env"})
        with self.assertRaises(ToolError):
            tools.check("code_graph", {"action": "impact", "target": "../outside.py"})

    def test_agent_offers_code_graph_when_enabled_even_in_plan_mode(self):
        client = FakeClient([
            reply(tool_calls=[tool_call("g1", "code_graph", action="impact", target="core.py")]),
            reply(content="core.py impacts caller.py."),
        ])
        agent = Agent(
            client, "fake", self.ws,
            permission=lambda *a: self.fail("code_graph must run without prompting"),
            on_event=quiet, plan_mode=True, code_graph=True,
        )
        self.assertIn("code_graph", agent.tool_names)
        answer = agent.send("what depends on core.py?")
        self.assertEqual(answer, "core.py impacts caller.py.")
        tool_out = [m["content"] for m in client.calls[-1] if m["role"] == "tool"][-1]
        self.assertIn("caller.py", tool_out)


class KnowledgeOverlayTests(unittest.TestCase):
    BLOAT = "y" * 26_000

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        self.ws = root / "ws"
        self.home = root / "home"
        self.ws.mkdir()
        (self.ws / "core.py").write_text("def core_fn():\n    return 1\n", encoding="utf-8")
        (self.ws / "caller.py").write_text("from core import core_fn\n\ndef entry():\n    return core_fn()\n", encoding="utf-8")
        (self.ws / "test_core.py").write_text("from core import core_fn\n\ndef test_ok():\n    assert core_fn() == 1\n", encoding="utf-8")

    def test_overlay_persists_with_private_permissions_and_survives_compaction(self):
        client = FakeClient([
            reply(tool_calls=[tool_call("n1", "code_graph", action="note", target="core.py",
                                        note="Keep core_fn synchronous")]),
            reply(tool_calls=[tool_call("w1", "edit_file", path="core.py",
                                        old_string="return 1", new_string="return 2")]),
            reply(content=self.BLOAT, tool_calls=[tool_call("r1", "read_file", path="caller.py")],
                  prompt_tokens=100),
            reply(tool_calls=[tool_call("r2", "read_file", path="caller.py")], prompt_tokens=10_000),
            reply(content="SUMMARY that forgot blast radius and notes."),
            reply(content="Ready."),
        ])
        agent = Agent(client, "fake", self.ws, permission=allow_all, context_window=12_000,
                      home=self.home, on_event=quiet, code_graph=True)
        self.assertEqual(agent.send("update core"), "Ready.")

        graph_file = project_dir(self.home, self.ws) / "graph.json"
        self.assertTrue(graph_file.is_file())
        if os.name != "nt":
            self.assertEqual(graph_file.stat().st_mode & 0o777, 0o600)

        saved = json.loads(graph_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["files"]["core.py"]["state"], "edited")
        self.assertEqual(saved["files"]["caller.py"]["state"], "read")
        self.assertEqual(saved["notes"][0]["note"], "Keep core_fn synchronous")

        after = json.dumps(client.calls[-1])
        self.assertIn("Code graph blast radius (dependents of edited files): caller.py", after)
        self.assertIn("Linked test files for edited modules: test_core.py", after)
        self.assertIn("Keep core_fn synchronous", after)

        # Fresh session reloads the persistent graph overlay automatically
        fresh = Agent(FakeClient([]), "fake", self.ws, home=self.home, on_event=quiet)
        self.assertEqual(fresh.knowledge.notes[0]["note"], "Keep core_fn synchronous")


class DagPlanAndWaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name).resolve()
        (self.ws / "app.py").write_text("def main():\n    return 42\n", encoding="utf-8")

    def test_validate_dag_waves_and_rejects_cycles_and_missing_deps(self):
        waves = validate_dag([
            {"id": "a", "depends_on": []},
            {"id": "b", "depends_on": []},
            {"id": "c", "depends_on": ["a", "b"]},
        ])
        self.assertEqual(waves, [[0, 1], [2]])

        with self.assertRaises(ValueError):
            validate_dag([{"id": "a", "depends_on": ["b"]}, {"id": "b", "depends_on": ["a"]}])
        with self.assertRaises(ValueError):
            validate_dag([{"id": "a", "depends_on": ["missing"]}])

    def test_plan_tool_accepts_dag_steps_round_trips_and_rejects_cycles(self):
        tools = Tools(self.ws)
        out = tools.plan(steps=[
            {"id": "scan", "step": "Scan callers"},
            {"id": "edit", "step": "Edit core", "depends_on": ["scan"]},
        ])
        self.assertEqual(tools.plan_steps, ["Scan callers", "Edit core"])
        self.assertIn("[scan] Scan callers", out)
        self.assertIn("[edit] Edit core [depends on: scan]", out)

        with self.assertRaises(ToolError):
            tools.plan(steps=[
                {"id": "x", "step": "A", "depends_on": ["y"]},
                {"id": "y", "step": "B", "depends_on": ["x"]},
            ])
        self.assertEqual(tools.plan_steps, ["Scan callers", "Edit core"])

        # Round-trip through save/load session and plan event
        events = []
        client = FakeClient([
            reply(tool_calls=[tool_call("p1", "plan", steps=[
                {"id": "s1", "step": "Read files"},
                {"id": "s2", "step": "Verify", "depends_on": ["s1"]},
            ])]),
            reply(tool_calls=[tool_call("p2", "plan", active=0)]),
            reply(content="Done."),
        ])
        agent = Agent(client, "fake", self.ws, permission=allow_all,
                      on_event=lambda e, **d: events.append((e, d)) if e == "plan" else None)
        agent.send("run dag plan")
        self.assertEqual(events[0][1]["steps"], ["Read files", "Verify"])
        self.assertEqual(events[0][1]["dag"][1]["depends_on"], ["s1"])

        sess = self.ws / "s.jsonl"
        agent.save_session(sess)
        restored = Agent(FakeClient([]), "fake", self.ws, on_event=quiet)
        restored.load_session(sess)
        self.assertEqual(restored.tools.plan_dag[1]["depends_on"], ["s1"])

    def test_explore_tasks_with_depends_on_run_in_waves_and_forward_outputs(self):
        wave0 = threading.Barrier(2, timeout=5)
        seen_prompts = []

        def held_wave0(text):
            def answer():
                wave0.wait()
                return reply(content=text, prompt_tokens=10, completion_tokens=2)
            return answer

        def wave1_answer():
            seen_prompts.append(client.calls[-1][1]["content"])
            return reply(content="Combined synthesis.", prompt_tokens=20, completion_tokens=3)

        client = RoutingClient(
            [
                reply(tool_calls=[
                    tool_call("t1", "task", id="api", description="Check API", prompt="Check API"),
                    tool_call("t2", "task", id="db", description="Check DB", prompt="Check DB"),
                    tool_call("t3", "task", id="merge", depends_on=["api", "db"],
                              description="Synthesize", prompt="Combine findings."),
                ], prompt_tokens=5, completion_tokens=1),
                reply(content="Final answer.", prompt_tokens=15, completion_tokens=2),
            ],
            {
                "Check API": [held_wave0("API uses REST.")],
                "Check DB": [held_wave0("DB uses SQLite.")],
                "Synthesize_unused": [],
            },
        )
        # Match wave 1 prompt dynamically since prerequisite output is prepended
        from types import SimpleNamespace
        orig_create = client._create

        def routed_create(**kwargs):
            msgs = kwargs["messages"]
            if "You are a subagent" in msgs[0]["content"] and "Combine findings." in msgs[1]["content"]:
                with client.lock:
                    client.calls.append(json.loads(json.dumps(msgs)))
                item = wave1_answer()
                return stream_chunks(item) if kwargs.get("stream") else item
            return orig_create(**kwargs)

        client.chat = SimpleNamespace(completions=SimpleNamespace(create=routed_create))

        agent = Agent(client, "fake", self.ws, permission=allow_all, on_event=quiet)
        self.assertEqual(agent.send("investigate"), "Final answer.")
        self.assertEqual(len(seen_prompts), 1)
        self.assertIn("[Output from prerequisite task api]:\nAPI uses REST.", seen_prompts[0])
        self.assertIn("[Output from prerequisite task db]:\nDB uses SQLite.", seen_prompts[0])
        results = {m["tool_call_id"]: m["content"] for m in agent.messages if m["role"] == "tool"}
        self.assertEqual(results, {"t1": "API uses REST.", "t2": "DB uses SQLite.", "t3": "Combined synthesis."})

    def test_cyclic_task_dag_returns_tool_error(self):
        client = FakeClient([
            reply(tool_calls=[
                tool_call("t1", "task", id="a", depends_on=["b"], description="A", prompt="A"),
                tool_call("t2", "task", id="b", depends_on=["a"], description="B", prompt="B"),
            ]),
            reply(content="Handled."),
        ])
        agent = Agent(client, "fake", self.ws, permission=allow_all, on_event=quiet)
        self.assertEqual(agent.send("go"), "Handled.")
        tool_msgs = [m["content"] for m in agent.messages if m["role"] == "tool"]
        self.assertTrue(all("invalid task DAG" in msg and "cycle" in msg for msg in tool_msgs))


class GraphCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name).resolve()
        (self.ws / "core.py").write_text("def core_fn():\n    return 1\n", encoding="utf-8")
        (self.ws / "api.py").write_text("from core import core_fn\n\ndef handle():\n    return core_fn()\n", encoding="utf-8")
        (self.ws / ".env").write_text("TOKEN=123\n", encoding="utf-8")

    def test_graph_command_renders_view_plain_fallback_and_guards_secrets(self):
        said, drawn = [], []
        agent = Agent(FakeClient([]), "fake", self.ws, on_event=quiet)

        plain_ctx = CommandContext(agent, lambda k, t: said.append((k, t)), self.ws / "s.jsonl")
        run_command(plain_ctx, "/graph")
        self.assertEqual(said[-1][0], "info")
        self.assertIn("core.py", said[-1][1])

        run_command(plain_ctx, "/graph .env")
        self.assertEqual(said[-1][0], "error")

        render_ctx = CommandContext(
            agent, lambda k, t: said.append((k, t)), self.ws / "s.jsonl",
            render=lambda view, data, fallback: drawn.append((view, data, fallback)),
        )
        run_command(render_ctx, "/graph core.py")
        self.assertEqual(drawn[-1][0], "graph")
        self.assertEqual(drawn[-1][1]["summary"]["file_count"], 2)
        self.assertIn("api.py", drawn[-1][2])


if __name__ == "__main__":
    unittest.main()
