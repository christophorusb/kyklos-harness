"""Stdio MCP tests with offline child processes and scripted model turns."""

import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from kyklos import (Agent, CommandContext, Trace, apply_permission_answer, ask_permission, context_breakdown,
                    run_command, tool_overview)
from mcp_client import PROTOCOL_VERSIONS, STDERR_LINE_BYTES, STDERR_LINES
from test_kyklos import FakeClient, deny_all, reply, tool_call
from test_serve import Client, ServeTestCase
from test_ui import make_ui, output

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "test_fixtures" / "fake_mcp_server.py"


def server_entry(log, *options, **config):
    return {"command": sys.executable, "args": [str(FIXTURE), "--log", str(log), *map(str, options)], **config}


def write_config(home, servers):
    home.mkdir(parents=True, exist_ok=True)
    (home / "mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def wait_until(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for the fixture")
        time.sleep(0.01)


def process_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def child_holds_lock(marker, pid):
    import fcntl
    path = Path(f"{marker}.{pid}.lock")
    if not path.is_file():
        return False
    with path.open() as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


class MCPTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        self.ws, self.home = root / "workspace", root / "home"
        self.log = root / "requests.jsonl"
        self.ws.mkdir()
        self.events = []

    def requests(self):
        if not self.log.is_file():
            return []
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def configure(self, *options, server="fixture", **config):
        write_config(self.home, {server: server_entry(self.log, *options, **config)})

    def agent(self, replies=(), **kwargs):
        kwargs.setdefault("permission", deny_all)
        kwargs.setdefault("on_event", lambda event, **data: self.events.append((event, data)))
        agent = Agent(FakeClient(replies), "fake", self.ws, home=self.home, **kwargs)
        self.addCleanup(agent.close)
        return agent

    def call(self, agent, tool, **args):
        return agent.run_tool(f"mcp__fixture__{tool}", json.dumps(args))

    def command(self, agent, text):
        said = []
        ctx = CommandContext(agent, lambda kind, message: said.append((kind, message)), self.ws / "session.jsonl")
        run_command(ctx, text)
        return said

    def test_initialize_pagination_aliases_and_schema_validation(self):
        self.configure("--version", "2025-06-18", "--paginated", server="fixture tools", autoApprove=["a b", "a_b"])
        agent = self.agent()
        requests = self.requests()
        self.assertEqual([r["method"] for r in requests],
                         ["initialize", "notifications/initialized", "tools/list", "tools/list"])
        self.assertEqual(requests[0]["params"]["protocolVersion"], PROTOCOL_VERSIONS[0])
        self.assertEqual(requests[-1]["params"], {"cursor": "second"})
        self.assertEqual(agent.mcp.servers["fixture tools"].version, "2025-06-18")
        tools = agent.mcp.tools
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", alias) for alias in tools))
        colliding = {tool.name: alias for alias, tool in tools.items() if tool.name in {"a b", "a_b"}}
        self.assertEqual(len(set(colliding.values())), 2)
        long_names = [alias for alias, tool in tools.items() if tool.name.startswith("x" * 80)]
        self.assertEqual(len(long_names), 2)
        self.assertNotEqual(*long_names)
        for name, alias in colliding.items():
            self.assertEqual(agent.run_tool(alias, '{"value": "round trip"}'), f"{name}\nround trip")
        function = tools[colliding["a b"]].schema["function"]
        self.assertTrue(function["description"].startswith("fixture tools"))
        self.assertEqual(function["parameters"], {"type": "object", "properties": {"value": {"type": "string"}}})
        self.assertFalse(any(tool.name == "bad-schema" for tool in tools.values()))
        self.assertIn("invalid name or schema", str(self.events))
        agent.restart_mcp("fixture tools")
        self.assertEqual(list(agent.mcp.tools), list(tools))

    def test_user_approval_is_per_tool_and_configured_autoapprove_is_explicit(self):
        self.configure(autoApprove=["echo"])
        asked = []

        def permission(name, display, grants):
            asked.append((name, display))
            return apply_permission_answer(name, "a", grants, False, ""), ""

        agent = self.agent(permission=permission)
        self.assertEqual(self.call(agent, "echo", value="yes"), "echo\nyes")
        self.assertEqual(self.call(agent, "annotated", value="one"), "annotated\none")
        self.assertEqual(self.call(agent, "annotated", value="two"), "annotated\ntwo")
        self.assertEqual(self.call(agent, "secret"), "Forwarded \nUnpassed absent")
        self.assertEqual(asked[0], ("mcp__fixture__annotated",
                                   {"server": "fixture", "tool": "annotated", "arguments": {"value": "one"}}))
        self.assertEqual([name for name, _ in asked], ["mcp__fixture__annotated", "mcp__fixture__secret"])
        self.assertEqual(agent.always_allowed, {"mcp__fixture__annotated", "mcp__fixture__secret"})
        self.assertIn("mcp__fixture__echo", tool_overview(agent)["safe"])

    def test_denial_sends_no_request_and_plain_and_aero_show_the_real_names(self):
        self.configure()
        agent = self.agent()
        self.assertIn("denied", self.call(agent, "echo", value="denied"))
        self.assertFalse(any(request["method"] == "tools/call" for request in self.requests()))
        display = {"server": "fixture name", "tool": "echo tool", "arguments": {"value": "visible"}}
        alias = "mcp__fixture_name__echo_tool"
        with mock.patch("builtins.input", return_value="a"), mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            grants = set()
            self.assertTrue(ask_permission(alias, display, grants)[0])
            self.assertEqual(grants, {alias})
            for text in ("fixture name", "echo tool", "visible"):
                self.assertIn(text, stdout.getvalue())
        ui = make_ui(self.ws, keys=["a"])
        grants = set()
        self.assertTrue(ui.permission(alias, display, grants)[0])
        self.assertEqual(grants, {alias})
        panel = output(ui)
        for text in ("fixture name", "echo tool", "visible"):
            self.assertIn(text, panel)

    def test_plan_subagent_and_unattended_tool_boundaries(self):
        self.configure(autoApprove=["echo", "annotated"], readOnly=["echo"])
        agent = self.agent(plan_mode=True)
        self.assertEqual(set(agent.mcp.tools) & agent.tool_names, {"mcp__fixture__echo"})
        self.assertEqual(self.call(agent, "echo", value="read"), "echo\nread")
        self.assertIn("isn't available", self.call(agent, "annotated"))
        agent.set_plan_mode(False)
        for mode in ("explore", "general"):
            sub, _ = agent._spawn(mode)
            self.assertFalse(sub.mcp.servers)
            self.assertFalse(any(name.startswith("mcp__") for name in sub.tool_names))
        self.assertFalse(self.agent(enable_mcp=False).mcp.servers)
        self.assertEqual(sum(r["method"] == "initialize" for r in self.requests()), 1)

    def test_pending_list_change_waits_until_reset(self):
        self.configure(autoApprove=["change"])
        agent = self.agent([reply("First."), reply("Second.")])
        agent.send("One")
        before = list(agent.schemas)
        self.assertEqual(self.call(agent, "change"), "Changed.")
        self.assertIn("Tool list update pending", agent.mcp.describe())
        agent.send("Two")
        self.assertEqual(agent.schemas, before)
        self.assertEqual(agent.client.offered[0], agent.client.offered[1])
        agent.reset()
        self.assertIn("mcp__fixture__new-tool", agent.tool_names)
        self.assertIn("MCP tools updated for fixture", str(self.events))
        self.assertEqual(sum(r["method"] == "initialize" for r in self.requests()), 1)

    def test_alias_reassignment_does_not_transfer_approval(self):
        self.configure("--paginated", autoApprove=["change"])
        asked = []

        def permission(name, display, grants):
            asked.append(display["tool"])
            if asked == ["a b"]:
                return apply_permission_answer(name, "a", grants, False, ""), ""
            return False, "Denied."

        agent = self.agent(permission=permission)
        self.assertEqual(self.call(agent, "a_b", value="initial"), "a b\ninitial")
        self.assertEqual(self.call(agent, "change"), "Changed.")
        agent.reset()
        self.assertNotIn("mcp__fixture__a_b", agent.always_allowed)
        self.assertEqual(agent.mcp.tools["mcp__fixture__a_b"].name, "a\tb")
        self.assertEqual(self.call(agent, "a_b"), "Denied.")
        self.assertEqual(len(asked), 2)

    def test_timeout_and_late_requests_leave_the_process_usable(self):
        self.configure(autoApprove=["wait", "echo"], timeout=0.15)
        agent = self.agent()
        self.assertIn("timed out", self.call(agent, "wait"))
        self.assertEqual(self.call(agent, "echo", value="after"), "echo\nafter")
        requests = self.requests()
        wait = next(r for r in requests if r.get("params", {}).get("name") == "wait")
        cancelled = next(r for r in requests if r["method"] == "notifications/cancelled")
        self.assertEqual(cancelled["params"]["requestId"], wait["id"])
        self.assertIsNone(agent.mcp.servers["fixture"].wire.process.poll())

    @unittest.skipUnless(hasattr(signal, "pthread_kill"), "needs pthread_kill")
    def test_ctrl_c_sends_cancellation_without_killing_server(self):
        self.configure(autoApprove=["wait", "echo"], timeout=4)
        agent = self.agent()
        armed = threading.Event()
        errors = []

        def cancel():
            try:
                wait_until(lambda: self.log.is_file() and '"name": "wait"' in self.log.read_text())
                if not armed.is_set():
                    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=cancel, daemon=True)
        worker.start()
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.call(agent, "wait")
        finally:
            armed.set()
            worker.join(timeout=4)
        self.assertFalse(errors)
        self.assertEqual(self.call(agent, "echo", value="still alive"), "echo\nstill alive")
        self.assertTrue(any(r["method"] == "notifications/cancelled" for r in self.requests()))

    def test_crash_and_malformed_reply_are_errors_and_restart_recovers(self):
        self.configure(autoApprove=["crash", "echo", "malformed"])
        agent = self.agent()
        process = agent.mcp.servers["fixture"].wire.process
        self.assertRegex(self.call(agent, "crash"), r"^Error: .*down")
        self.assertRegex(self.call(agent, "echo"), r"^Error: .*down")
        self.assertIsNotNone(process.wait(timeout=2))
        report = self.command(agent, "/mcp restart fixture")
        new_process = agent.mcp.servers["fixture"].wire.process
        self.assertNotEqual(process.pid, new_process.pid)
        self.assertIn(f"PID {new_process.pid}", report[-1][1])
        self.assertIn("unsandboxed", report[-1][1])
        self.assertTrue(self.call(agent, "echo").startswith("echo"))
        self.assertRegex(self.call(agent, "malformed"), r"^Error: .*invalid JSON-RPC")
        self.assertIn("invalid JSON-RPC", agent.mcp.describe())

    def test_startup_failures_are_reported_and_other_servers_start(self):
        write_config(self.home, {
            "bad-json": server_entry(self.log, "--startup", "bad-json"),
            "slow": server_entry(self.log, "--startup", "slow"),
            "unsupported": server_entry(self.log, "--version", "1900-01-01"),
            "good": server_entry(self.log),
        })
        with mock.patch("mcp_client.STARTUP_TIMEOUT", 0.5):
            agent = self.agent()
        self.assertIn("mcp__good__echo", agent.tool_names)
        self.assertFalse(any(alias.startswith("mcp__slow__") for alias in agent.tool_names))
        self.assertIn("Unsupported MCP protocol version", str(self.events))
        self.assertIn("invalid JSON-RPC", str(self.events))
        report = self.command(agent, "/mcp")[-1][1]
        self.assertIn("slow [down]", report)
        self.assertIn("initialize timed out", report)
        self.assertIn("good [running]", report)

    def test_servers_start_side_by_side_each_within_its_own_startup_timeout(self):
        write_config(self.home, {
            "slow-a": server_entry(self.log, "--startup", "slow", startupTimeout=1),
            "slow-b": server_entry(self.log, "--startup", "slow", startupTimeout=1),
            "slow-c": server_entry(self.log, "--startup", "slow", startupTimeout=1),
            "good": server_entry(self.log),
        })
        started = time.monotonic()
        agent = self.agent()
        took = time.monotonic() - started
        self.assertLess(took, 2.5, "three 1s timeouts one after another would take 3s or more")
        self.assertIn("mcp__good__echo", agent.tool_names)
        notices = [data["text"] for event, data in self.events if event == "notice"]
        failed = [text for text in notices if "failed to start" in text]
        self.assertEqual([text.split()[2] for text in failed], ["slow-a", "slow-b", "slow-c"],
                         "notices come in server order, whichever thread finished first")
        self.assertEqual(threading.current_thread(), threading.main_thread())

    def test_startup_timeout_defaults_and_must_be_a_positive_number(self):
        from mcp_client import STARTUP_TIMEOUT, MCPError, _server_config
        config, _ = _server_config({"command": "x"}, {})
        self.assertEqual(config.startup_timeout, STARTUP_TIMEOUT)
        self.assertEqual(_server_config({"command": "x", "startupTimeout": 90}, {})[0].startup_timeout, 90)
        for bad in (0, -1, "30", True, float("inf")):
            with self.assertRaisesRegex(MCPError, "startupTimeout must be a positive number"):
                _server_config({"command": "x", "startupTimeout": bad}, {})

    def test_secrets_are_redacted_in_schemas_results_stderr_and_spilled_output(self):
        public_name_value = "ordinary-value-forwarded-as-secret"
        never_passed = "secret-that-stays-with-kyklos"
        self.configure("--startup", "stderr", autoApprove=["secret", "huge", "error"],
                       env={"KEY_FOR_SERVER": "${FIXTURE_VALUE}"})
        with mock.patch.dict(os.environ, {"FIXTURE_VALUE": public_name_value, "UNPASSED_SECRET": never_passed}):
            agent = self.agent()
            self.assertNotIn(public_name_value, json.dumps(agent.schemas))
            self.assertIn("[REDACTED]", json.dumps(agent.schemas))
            self.assertEqual(self.call(agent, "secret"), "Forwarded [REDACTED]\nUnpassed absent")
            self.assertIn("Error: The fixture encountered an error.", self.call(agent, "error"))
            result = self.call(agent, "huge")
            self.assertIn("truncated here", result)
            [spill] = (self.ws / ".kyklos" / "scratch" / "output").glob("*.log")
            saved = spill.read_text()
            self.assertIn("[REDACTED]", saved)
            self.assertNotIn(public_name_value, saved)
            stderr = agent.mcp.describe()
            self.assertLess(len(stderr), STDERR_LINES * STDERR_LINE_BYTES)
            self.assertIn("Long stderr line omitted", stderr)
            self.assertIn("Server log [REDACTED]", stderr)
            self.assertNotIn("\x1b", stderr)

    def test_images_use_the_existing_image_message_path(self):
        self.configure(autoApprove=["image"])
        for vision in (True, False):
            with self.subTest(vision=vision):
                agent = self.agent([reply(tool_calls=[tool_call("one", "mcp__fixture__image")]),
                                    reply("Done.")], vision=vision)
                agent.send("Show the result")
                sent = agent.client.calls[1]
                result = next(m["content"] for m in sent if m["role"] == "tool")
                self.assertIn("resource_link content block", result)
                self.assertNotIn("test:resource", result)
                if vision:
                    self.assertEqual(sent[-1]["role"], "user")
                    [image] = [part for part in sent[-1]["content"] if part["type"] == "image_url"]
                    self.assertTrue(image["image_url"]["url"].startswith("data:image/png;"))
                else:
                    self.assertEqual(sent[-1]["role"], "tool")
                    self.assertIn("returned an image", result)

    def test_schema_cost_is_visible_and_trace_uses_fixed_label(self):
        self.configure(server="private server", autoApprove=["echo"])
        trace = Trace(self.ws / "trace.jsonl")
        self.addCleanup(trace.close)
        agent = self.agent([reply(tool_calls=[tool_call("one", "mcp__private_server__echo", value="private value")]),
                            reply("Done.")],
                           on_event=trace.observe(lambda event, **data: self.events.append((event, data))),
                           context_window=1500)
        agent.send("Use the tool")
        cost = tool_overview(agent)["mcp_schema_tokens"]
        self.assertGreater(cost, 0)
        self.assertIn(("MCP schemas", cost), context_breakdown(agent)["parts"])
        self.assertIn("MCP schemas", self.command(agent, "/tools")[-1][1])
        self.assertIn("MCP schemas", self.command(agent, "/tokens")[-1][1])
        self.assertIn("of the context window", str(self.events))
        rows = trace.path.read_text()
        self.assertEqual(rows.count('"tool": "mcp"'), 2)
        self.assertNotIn("private_server", rows)
        self.assertNotIn("private value", rows)

    def test_project_config_is_ignored_even_when_used_as_kyklos_home(self):
        write_config(self.ws, {"fixture": server_entry(self.log)})
        self.assertFalse(self.agent().mcp.servers)
        self.home = self.ws
        self.assertFalse(self.agent().mcp.servers)
        self.assertFalse(self.log.exists())

    def test_workspace_path_entries_are_ignored_when_launching_servers(self):
        filename = "fixture-executable.exe" if os.name == "nt" else "fixture-executable"
        candidate = self.ws / filename
        candidate.write_text("Not a launchable MCP server.", encoding="utf-8")
        candidate.chmod(0o700)
        write_config(self.home, {"fixture": {"command": filename}})
        with mock.patch.dict(os.environ, {"PATH": os.pathsep.join((".", str(self.ws)))}):
            agent = self.agent()
        self.assertIn("not found on a trusted PATH", agent.mcp.describe())
        self.assertFalse(agent.mcp.tools)

    @unittest.skipIf(os.name == "nt", "uses POSIX process groups")
    def test_restart_and_close_kill_the_process_tree(self):
        child_file = self.home / "child"
        self.configure("--child", child_file)
        agent = self.agent()
        old_child = int(child_file.read_text())
        wait_until(lambda: child_holds_lock(child_file, old_child))
        old_server = agent.mcp.servers["fixture"].wire.process
        agent.restart_mcp("fixture")
        wait_until(lambda: not child_holds_lock(child_file, old_child))
        self.assertIsNotNone(old_server.returncode)
        new_child = int(child_file.read_text())
        self.assertNotEqual(old_child, new_child)
        wait_until(lambda: child_holds_lock(child_file, new_child))
        agent.close()
        wait_until(lambda: not child_holds_lock(child_file, new_child))


class ServeMCPTests(ServeTestCase):
    def test_serve_permission_displays_server_tool_and_arguments(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(self.ui.inp.close)
        self.addCleanup(self.client.from_server.close)
        home = Path(tmp.name)
        write_config(home, {"fixture": server_entry(home / "requests")})
        echo = "mcp__fixture__echo"

        def script(client):
            request = client.call("session/prompt", text="Call the fixture")
            permission = client.asked("permission/request")
            self.assertEqual(permission["params"]["tool"], echo)
            self.assertEqual(permission["params"]["args"],
                             {"server": "fixture", "tool": "echo", "arguments": {"value": "shown"}})
            self.assertEqual(permission["params"]["choices"], ["y", "n", "a"])
            client.answer(permission, answer="a")
            self.assertEqual(client.result(request)["result"]["answer"], "Done.")
            self.shutdown(client)

        agent, _ = self.serve([reply(tool_calls=[tool_call("one", echo, value="shown")]), reply("Done.")],
                              script, agent_kwargs={"home": home})
        self.assertEqual(agent.always_allowed, {echo})
        self.assertIsNone(agent.mcp.servers["fixture"].wire)

    def test_serve_shutdown_stops_the_configured_server_process(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            write_config(home, {"fixture": server_entry(home / "requests")})
            env = {k: v for k, v in os.environ.items() if not k.startswith(("KYKLOS", "BASE_URL", "API_KEY",
                                                                           "MODEL", "CONTEXT_WINDOW"))}
            env.update(KYKLOS_HOME=folder, WORKSPACE=str(self.ws), BASE_URL="http://127.0.0.1:9/v1",
                       API_KEY="test", MODEL="fake-model", CONTEXT_WINDOW="32000", KYKLOS_TRACE="0",
                       KYKLOS_PERSONA="", KYKLOS_PROFILE="")
            with subprocess.Popen([sys.executable, str(HERE / "kyklos.py"), "--serve", "--trust"],
                                  cwd=self.ws, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE) as proc:
                client = Client(proc.stdin, proc.stdout)
                try:
                    client.result(client.call("initialize", protocol_version=0))
                    client.result(client.call("session/start", resume=None))
                    request = client.call("command/run", text="/mcp")
                    report = client.expect(lambda m: m.get("method") == "say")["params"]["text"]
                    pid = int(re.search(r"PID (\d+)", report)[1])
                    client.result(request)
                    self.shutdown(client)
                    self.assertEqual(proc.wait(timeout=5), 0)
                    if os.name != "nt":
                        wait_until(lambda: not process_running(pid))
                finally:
                    if proc.poll() is None:
                        proc.kill()


if __name__ == "__main__":
    unittest.main()
