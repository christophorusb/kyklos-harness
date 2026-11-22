"""Offline tests for `kyklos --serve`: a scripted client on real pipes drives ServeUI and the real Agent.

The server runs on the test's main thread, as it does in production, because cancelling a turn is a
SIGINT delivered to that thread. The client runs beside it on a thread of its own.

Run:  python -m unittest test_serve -v
"""

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from kyklos import (DEFAULT_MAX_STEPS, NO_APPROVER_NOTE, Agent, edit_settings, load_settings,
                    runtime_preferences, save_settings, update_runtime_preferences)
from serve import (BUSY, INVALID_PARAMS, METHOD_NOT_FOUND, NOT_READY, PARSE_ERROR, PROTOCOL_VERSION, ServeUI,
                   line_counts)
from test_kyklos import FakeClient, RoutingClient, reply, tool_call

HERE = Path(__file__).parent
WAIT = 15  # seconds a client waits for any one message before the test fails


class Client:
    """The other end of the pipes: sends lines, and collects every line the server writes."""

    def __init__(self, to_server, from_server):
        self.to_server, self.from_server = to_server, from_server
        self.inbox: queue.Queue = queue.Queue()
        self.log: list[dict] = []
        self.raw: list[bytes] = []
        self._next = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in iter(self.from_server.readline, b""):
            self.raw.append(line)
            msg = json.loads(line)
            self.log.append(msg)
            self.inbox.put(msg)
        self.inbox.put(None)

    def write(self, payload) -> None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.to_server.write(data + b"\n")
        self.to_server.flush()

    def call(self, method, **params) -> int:
        self._next += 1
        self.write({"jsonrpc": "2.0", "id": self._next, "method": method, "params": params})
        return self._next

    def notify(self, method, **params) -> None:
        self.write({"jsonrpc": "2.0", "method": method, "params": params})

    def answer(self, request, **result) -> None:
        self.write({"jsonrpc": "2.0", "id": request["id"], "result": result})

    def expect(self, test) -> dict:
        deadline = time.monotonic() + WAIT
        while True:
            msg = self.inbox.get(timeout=max(deadline - time.monotonic(), 0.01))
            if msg is None:
                raise AssertionError("the server closed its output while the client was waiting")
            if test(msg):
                return msg

    def result(self, rid) -> dict:
        return self.expect(lambda m: m.get("id") == rid and "method" not in m)

    def asked(self, method) -> dict:
        return self.expect(lambda m: m.get("method") == method and "id" in m)

    def close(self) -> None:
        self.to_server.close()

    def updates(self, event=None) -> list[dict]:
        return [m["params"] for m in self.log if m.get("method") == "session/update"
                and (event is None or m["params"]["event"] == event)]


class ServeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        c2s_r, c2s_w = os.pipe()
        s2c_r, s2c_w = os.pipe()
        self.ui = ServeUI(self.ws, os.fdopen(c2s_r, "rb"), os.fdopen(s2c_w, "wb", buffering=0))
        self.client = Client(os.fdopen(c2s_w, "wb", buffering=0), os.fdopen(s2c_r, "rb"))
        self.session = self.ws / "session.jsonl"

    def tearDown(self):
        for stream in (self.ui.out, self.client.to_server):
            try:
                stream.close()
            except OSError:
                pass
        self.tmp.cleanup()

    def serve(self, replies, script, start_params=None, agent_kwargs=None, client_factory=FakeClient):
        """Handshake, then run the agent until `script(client)` (on its own thread) is done with it."""
        fake = client_factory(replies)
        failures = []

        def drive():
            try:
                rid = self.client.call("initialize", protocol_version=PROTOCOL_VERSION,
                                       client={"name": "test", "version": "0"})
                self.init = self.client.result(rid)["result"]
                rid = self.client.call("session/start", **(start_params or {"resume": None}))
                self.started = self.client.result(rid)
                script(self.client)
            except BaseException as e:  # reported on the main thread, which owns the test
                failures.append(e)
                self.client.close()

        thread = threading.Thread(target=drive, daemon=True)
        thread.start()
        start = self.ui.handshake({"kyklos_version": "test", "workspace": str(self.ws),
                                   "sandbox": {"kind": "test", "label": "", "warning": None},
                                   "commands": [], "profiles": [], "profile": None})
        self.assertIsNotNone(start)
        agent = Agent(fake, "fake", self.ws, permission=self.ui.permission, ask=self.ui.ask_user,
                      on_event=self.ui.observer, **(agent_kwargs or {}))
        try:
            self.ui.run(agent, self.session)
        finally:
            agent.close()
        thread.join(WAIT)
        if failures:
            raise failures[0]
        self.assertFalse(thread.is_alive(), "the client script never finished")
        return agent, fake

    @staticmethod
    def shutdown(client):
        client.result(client.call("shutdown"))


class HandshakeTests(ServeTestCase):
    def test_initialize_then_start_then_shutdown(self):
        self.serve([], self.shutdown)
        self.assertEqual(self.init["protocol_version"], PROTOCOL_VERSION)
        state = self.started["result"]["state"]
        self.assertEqual(state["model"], "fake")
        self.assertFalse(state["plan_mode"])
        self.assertIsNone(self.started["result"]["note"])

    def test_every_line_is_one_json_message(self):
        self.serve([reply(content="Hi.")],
                   lambda c: (c.result(c.call("session/prompt", text="hello")), self.shutdown(c)))
        self.assertTrue(self.client.raw)
        for line in self.client.raw:
            self.assertTrue(line.endswith(b"\n"))
            self.assertEqual(line.count(b"\n"), 1)
            json.loads(line)

    def test_errors_for_bad_input(self):
        def script(c):
            c.write(b"{not json")
            self.assertEqual(c.expect(lambda m: "error" in m)["error"]["code"], PARSE_ERROR)
            rid = c.call("no/such/method")
            self.assertEqual(c.result(rid)["error"]["code"], METHOD_NOT_FOUND)
            rid = c.call("session/prompt", text="")
            self.assertEqual(c.result(rid)["error"]["code"], INVALID_PARAMS)
            rid = c.call("command/run", text="not a command")
            self.assertEqual(c.result(rid)["error"]["code"], INVALID_PARAMS)
            self.shutdown(c)
        self.serve([], script)

    def test_nothing_runs_before_the_session_starts(self):
        # No handshake helper here: the client talks before session/start has been answered.
        thread_done = threading.Event()

        def script():
            c = self.client
            rid = c.call("session/prompt", text="too soon")
            self.assertEqual(c.result(rid)["error"]["code"], NOT_READY)
            c.result(c.call("initialize", protocol_version=0))
            rid = c.call("session/state")
            self.assertEqual(c.result(rid)["error"]["code"], NOT_READY)
            c.close()
            thread_done.set()

        threading.Thread(target=script, daemon=True).start()
        self.assertIsNone(self.ui.handshake({}))  # EOF before session/start
        self.assertTrue(thread_done.wait(WAIT))


class TurnTests(ServeTestCase):
    def test_a_turn_streams_events_and_returns_the_answer(self):
        def script(c):
            rid = c.call("session/prompt", text="say hi")
            asked = c.asked("permission/request")
            c.answer(asked, answer="y")
            self.result = c.result(rid)["result"]
            self.shutdown(c)

        self.serve([reply(tool_calls=[tool_call("call_1", "bash", command="echo hi")]),
                    reply(content="It said hi.")], script)
        self.assertEqual(self.result["answer"], "It said hi.")
        self.assertEqual(self.result["outcome"], "answer")
        events = [u["event"] for u in self.client.updates()]
        for name in ("turn_start", "tool_call", "tool_result", "turn_end"):
            self.assertIn(name, events)
        self.assertLess(events.index("tool_call"), events.index("tool_result"))
        call = self.client.updates("tool_call")[0]["data"]
        result = self.client.updates("tool_result")[0]["data"]
        self.assertEqual(call["call_id"], "call_1")
        self.assertEqual(result["call_id"], "call_1")
        self.assertIn("exit code 0", result["result"])
        seqs = [u["seq"] for u in self.client.updates()]
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)))
        self.assertTrue(self.session.is_file(), "the session is saved after a turn")

    def test_permission_request_carries_the_terminal_rules(self):
        def script(c):
            rid = c.call("session/prompt", text="go")
            asked = c.asked("permission/request")
            self.asked = asked["params"]
            c.answer(asked, answer="n", reason="not now")
            c.result(rid)
            self.shutdown(c)

        self.serve([reply(tool_calls=[tool_call("c1", "bash", command="rm -rf build")]),
                    reply(content="OK.")], script)
        self.assertEqual(self.asked["call_id"], "c1")
        self.assertEqual(self.asked["display"], {"kind": "command", "command": "rm -rf build"})
        self.assertTrue(any(f["level"] == "danger" for f in self.asked["findings"]))
        self.assertEqual(self.asked["choices"], ["y", "n"], "a danger finding withholds a and the prefix")
        self.assertIsNone(self.asked["prefix"])
        denied = self.client.updates("tool_result")[0]["data"]["result"]
        self.assertIn("denied", denied)
        self.assertIn("not now", denied)

    def test_an_answer_that_was_not_offered_is_asked_again(self):
        def script(c):
            rid = c.call("session/prompt", text="go")
            first = c.asked("permission/request")
            c.answer(first, answer="a")  # `always` isn't offered on a danger finding
            second = c.asked("permission/request")
            self.second = second["params"]
            c.answer(second, answer="n")
            c.result(rid)
            self.shutdown(c)

        self.serve([reply(tool_calls=[tool_call("c1", "bash", command="rm -rf build")]),
                    reply(content="OK.")], script)
        self.assertEqual(self.second["rejected"]["answer"], "a")
        self.assertIn("denied", self.client.updates("tool_result")[0]["data"]["result"])
        self.assertTrue((self.ws).is_dir())

    def test_edit_permission_shows_the_diff(self):
        (self.ws / "app.py").write_text("one\ntwo\nthree\n")

        def script(c):
            rid = c.call("session/prompt", text="edit")
            asked = c.asked("permission/request")
            self.display = asked["params"]["display"]
            self.choices = asked["params"]["choices"]
            c.answer(asked, answer="a")
            c.result(rid)
            self.shutdown(c)

        agent, _ = self.serve([reply(tool_calls=[tool_call("e1", "edit_file", path="app.py",
                                                           old_string="two", new_string="2")]),
                               reply(content="Edited.")], script)
        self.assertEqual(self.display["kind"], "diff")
        self.assertEqual((self.display["old"], self.display["new"]), ("two", "2"))
        self.assertEqual(self.display["first_line"], 2)
        self.assertEqual((self.display["added"], self.display["removed"]), (1, 1))
        self.assertEqual(self.choices, ["y", "n", "a"])
        self.assertEqual((self.ws / "app.py").read_text(), "one\n2\nthree\n")
        self.assertIn("edit_file", agent.always_allowed)

    def test_a_second_turn_while_one_runs_is_refused(self):
        def script(c):
            rid = c.call("session/prompt", text="go")
            asked = c.asked("permission/request")
            busy = c.call("session/prompt", text="and this")
            self.assertEqual(c.result(busy)["error"]["code"], BUSY)
            command = c.call("command/run", text="/tokens")
            self.assertEqual(c.result(command)["error"]["code"], BUSY)
            state = c.call("session/state")
            self.assertEqual(c.result(state)["result"]["model"], "fake")  # still answered mid-turn
            c.answer(asked, answer="y")
            c.result(rid)
            self.shutdown(c)

        self.serve([reply(tool_calls=[tool_call("c1", "bash", command="echo hi")]), reply(content="Done.")],
                   script)

    def test_api_error_is_an_error_reply(self):
        def script(c):
            rid = c.call("session/prompt", text="go")
            self.error = c.result(rid)["error"]
            self.shutdown(c)

        self.serve([ValueError("bad key")], script)
        self.assertIn("bad key", self.error["data"]["detail"])


class CancelTests(ServeTestCase):
    def test_cancel_background_wait_and_resume_over_rpc(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def worker():
            entered.set()
            if not release.wait(5):
                return AssertionError("the server did not cancel the turn while the worker was blocked")
            return reply("Late report")

        def wait_for_report():
            self.assertTrue(entered.wait(5))
            result = fake.calls[-1][-1]["content"]
            job_id = result.split()[-1].rstrip(".")
            return reply(content="Collecting the report.",
                         tool_calls=[tool_call("wait", "task_wait", job_id=job_id)])

        def script(c):
            rid = c.call("session/prompt", text="start and collect")
            c.expect(lambda m: m.get("method") == "session/update"
                     and m["params"]["event"] == "tool_call"
                     and m["params"]["data"]["name"] == "task_wait")
            c.notify("session/cancel")
            self.assertEqual(c.result(rid)["result"]["outcome"], "interrupted")
            release.set()
            self.assertEqual(c.result(c.call("session/prompt", text="next turn"))["result"]["answer"], "Recovered")
            self.shutdown(c)

        start = tool_call("start", "task_start", description="Inspect file", prompt="Inspect")
        fake = RoutingClient([reply(tool_calls=[start]), wait_for_report, reply("Recovered")],
                             {"Inspect": [worker]})
        agent, _ = self.serve([], script, agent_kwargs={"background_subagents": True},
                              client_factory=lambda _: fake)
        self.assertTrue(agent._background_worker.done.wait(5))
        updates = self.client.updates()
        self.assertEqual([u["seq"] for u in updates], list(range(1, len(updates) + 1)))
        self.assertTrue(all(u["depth"] == 0 for u in updates))
        self.assertEqual([u["data"]["status"] for u in self.client.updates("background_task")],
                         ["running", "cancelled"])
        self.assertTrue(any(u["event"] == "answer_delta" for u in updates))
        self.assertNotIn("Late report", self.session.read_text())

    def test_cancel_while_a_permission_prompt_is_open(self):
        def script(c):
            rid = c.call("session/prompt", text="go")
            asked = c.asked("permission/request")
            c.notify("session/cancel")
            withdrawn = c.expect(lambda m: m.get("method") == "request/withdrawn")
            self.assertEqual(withdrawn["params"]["id"], asked["id"])
            self.result = c.result(rid)["result"]
            self.shutdown(c)

        agent, _ = self.serve([reply(tool_calls=[tool_call("c1", "bash", command="echo hi")])], script)
        self.assertEqual(self.result["outcome"], "interrupted")
        self.assertIsNone(self.result["answer"])
        tool_messages = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(tool_messages[-1]["content"], "Interrupted by the user.",
                         "the unanswered call gets a placeholder, so the next request isn't rejected")

    @unittest.skipUnless(hasattr(__import__("signal"), "pthread_kill"), "needs pthread_kill")
    def test_cancel_stops_a_running_command(self):
        def script(c):
            rid = c.call("session/prompt", text="go")
            c.answer(c.asked("permission/request"), answer="y")
            time.sleep(0.5)  # let the command start
            self.cancelled_at = time.monotonic()
            c.notify("session/cancel")
            self.result = c.result(rid)["result"]
            self.stopped_at = time.monotonic()
            self.shutdown(c)

        self.serve([reply(tool_calls=[tool_call("c1", "bash", command="sleep 30")])], script)
        self.assertEqual(self.result["outcome"], "interrupted")
        self.assertLess(self.stopped_at - self.cancelled_at, 5, "cancel must break the blocked command")

    def test_hang_up_mid_turn_denies_and_ends(self):
        def script(c):
            c.call("session/prompt", text="go")
            c.asked("permission/request")
            c.close()

        agent, _ = self.serve([reply(tool_calls=[tool_call("c1", "bash", command="echo hi")]),
                               reply(content="never")], script)
        self.assertEqual(agent.last_outcome, "interrupted")

    def test_a_late_cancel_between_turns_does_nothing(self):
        def script(c):
            c.result(c.call("session/prompt", text="hi"))
            c.notify("session/cancel")
            self.second = c.result(c.call("session/prompt", text="again"))["result"]
            self.shutdown(c)

        self.serve([reply(content="One."), reply(content="Two.")], script)
        self.assertEqual(self.second["answer"], "Two.")


class CommandTests(ServeTestCase):
    def test_plan_with_a_request_hands_it_back_to_send(self):
        def script(c):
            self.result = c.result(c.call("command/run", text="/plan look around"))["result"]
            self.state = [m["params"] for m in c.log if m.get("method") == "session/state"][-1]
            self.shutdown(c)

        self.serve([], script)
        self.assertEqual(self.result, {"then_send": "look around", "exit": False})
        self.assertTrue(self.state["plan_mode"])
        said = [m["params"] for m in self.client.log if m.get("method") == "say"]
        self.assertTrue(any("Plan mode on" in s["text"] for s in said))

    def test_views_carry_data_and_their_plain_fallback(self):
        def script(c):
            c.result(c.call("command/run", text="/tokens"))
            self.view = next(m["params"] for m in c.log if m.get("method") == "view")
            self.shutdown(c)

        self.serve([], script)
        self.assertEqual(self.view["name"], "tokens")
        self.assertTrue(self.view["fallback"])
        self.assertIsNotNone(self.view["data"])

    def test_exit_ends_the_server(self):
        def script(c):
            self.result = c.result(c.call("command/run", text="/exit"))["result"]

        self.serve([], script)
        self.assertTrue(self.result["exit"])


class SettingsTests(ServeTestCase):
    def test_update_reset_and_environment_precedence(self):
        home = self.ui.settings_home = self.ws / "home"
        save_settings(home, {"persona": "reviewer", "profiles": {"local": {"model": "fake"}}})

        def script(c):
            self.assertTrue(self.init["capabilities"]["runtime_settings"])
            current = c.result(c.call("settings/read"))["result"]
            self.assertEqual(current["fields"]["stream"],
                             {"value": False, "active": False, "default": True,
                              "source": "environment", "variable": "STREAM"})
            self.assertFalse(current["restart_required"])

            for changes in ({"max_steps": True}, {"max_steps": 0}, {"max_steps": 1.5},
                            {"background_subagents": "true"}, {"api_key": "sensitive"},
                            {"max_steps": 75, "stream": True}):
                with self.subTest(changes=list(changes)):
                    self.assertEqual(c.result(c.call("settings/update", changes=changes))["error"]["code"],
                                     INVALID_PARAMS)
            self.assertNotIn("runtime", load_settings(home))

            updated = c.result(c.call("settings/update", changes={
                "max_steps": 75, "background_subagents": True}))["result"]
            self.assertTrue(updated["restart_required"])
            self.assertEqual(updated["fields"]["max_steps"]["active"], DEFAULT_MAX_STEPS)
            effective = runtime_preferences(load_settings(home), {})
            self.assertEqual(effective["max_steps"]["value"], 75)
            self.assertTrue(effective["background_subagents"]["value"])
            self.assertEqual(runtime_preferences(load_settings(home), {"MAX_STEPS": "30"})["max_steps"]["value"], 30)

            reset = c.result(c.call("settings/update", changes={
                "max_steps": None, "background_subagents": None}))["result"]
            self.assertFalse(reset["restart_required"])
            self.assertEqual(reset["fields"]["max_steps"]["source"], "default")
            self.assertEqual(load_settings(home), {"persona": "reviewer", "profiles": {"local": {"model": "fake"}}})
            self.shutdown(c)

        with patch.dict(os.environ, {"STREAM": "0"}):
            os.environ.pop("MAX_STEPS", None)
            os.environ.pop("KYKLOS_BACKGROUND_SUBAGENTS", None)
            self.serve([], script, agent_kwargs={"stream": False})

    def test_overlapping_writes_keep_persona_and_runtime_settings(self):
        home = self.ws / "home"
        waiting = threading.Event()
        result = queue.Queue()

        def update():
            waiting.set()
            try:
                result.put(update_runtime_preferences(home, {}, {"max_steps": 70}))
            except Exception as exc:
                result.put(exc)

        with edit_settings(home) as settings:
            settings["persona"] = "reviewer"
            writer = threading.Thread(target=update, daemon=True)
            writer.start()
            self.assertTrue(waiting.wait(5))
        writer.join(5)
        self.assertFalse(writer.is_alive())
        updated = result.get_nowait()
        if isinstance(updated, Exception):
            raise updated
        self.assertEqual(updated["max_steps"]["value"], 70)
        self.assertEqual(load_settings(home), {"persona": "reviewer", "runtime": {"max_steps": 70}})

    def test_invalid_settings_file_is_preserved(self):
        home = self.ui.settings_home = self.ws / "home"
        home.mkdir()
        path = home / "settings.json"
        malformed = '{"persona": "reviewer",'
        path.write_text(malformed)

        def script(c):
            for method, params in (("settings/read", {}), ("settings/update", {"changes": {"max_steps": 75}})):
                self.assertIn("error", c.result(c.call(method, **params)))
            self.assertEqual(path.read_text(), malformed)
            self.shutdown(c)

        self.serve([], script)


class CallbackTests(unittest.TestCase):
    """The prompts that aren't per-turn, with the client's reply stubbed."""

    def ui(self, reply_value):
        ui = ServeUI(Path("."), None, None)
        ui.asked = []
        ui.request = lambda method, params: (ui.asked.append((method, params)), reply_value)[1]
        return ui

    def test_trust_on_a_broad_folder_never_offers_yes(self):
        ui = self.ui({"choice": "yes"})
        self.assertEqual(ui.ask_trust(Path("/"), broad=True), "no", "an answer not offered is a no")
        self.assertEqual(ui.asked[0][1]["choices"], ["once", "no"])

    def test_trust_and_bash_mode_default_to_the_safe_side(self):
        self.assertEqual(self.ui(None).ask_trust(Path("/x"), broad=False), "no")
        self.assertEqual(self.ui({"choice": "once"}).ask_trust(Path("/x"), broad=False), "once")
        self.assertFalse(self.ui(None).ask_bash_mode())
        self.assertFalse(self.ui({"mode": "yes please"}).ask_bash_mode())
        self.assertTrue(self.ui({"mode": "unflagged"}).ask_bash_mode())

    def test_pick_takes_only_an_index_in_range(self):
        rows = [("a",), ("b",)]
        self.assertEqual(self.ui({"index": 1}).pick("t", rows), 1)
        for bad in ({"index": 2}, {"index": -1}, {"index": True}, {"index": "0"}, None):
            self.assertIsNone(self.ui(bad).pick("t", rows))

    def test_no_approver_is_a_denial(self):
        allowed, note = self.ui(None).permission("bash", {"command": "echo hi"}, set())
        self.assertFalse(allowed)
        self.assertEqual(note, NO_APPROVER_NOTE.format(name="bash"))

    def test_line_counts(self):
        self.assertEqual(line_counts("a\nb\n", "a\nc\nd\n"), (2, 1))
        self.assertEqual(line_counts("", "x\n"), (1, 0))


class ProcessTests(unittest.TestCase):
    """`kyklos --serve` as a real process: startup questions, a command, and a clean stdout."""

    def test_serve_process_handshake(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as ws:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("KYKLOS", "BASE_URL", "API_KEY",
                                                                             "MODEL", "CONTEXT_WINDOW"))}
            env.update(KYKLOS_HOME=home, WORKSPACE=ws, BASE_URL="http://127.0.0.1:9/v1", API_KEY="test",
                       MODEL="fake-model", CONTEXT_WINDOW="32000", KYKLOS_TRACE="0", KYKLOS_PERSONA="",
                       KYKLOS_PROFILE="")
            proc = subprocess.Popen([sys.executable, str(HERE / "kyklos.py"), "--serve"], cwd=ws, env=env,
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            client = Client(proc.stdin, proc.stdout)
            try:
                init = client.result(client.call("initialize", protocol_version=PROTOCOL_VERSION))["result"]
                self.assertIn("help", [c["name"] for c in init["commands"]])
                start = client.call("session/start", resume=None)
                trust = client.asked("trust/request")
                self.assertIn("once", trust["params"]["choices"])
                client.answer(trust, choice="once")
                client.answer(client.asked("session/bash_mode"), mode="ask")
                state = client.result(start)["result"]["state"]
                self.assertEqual(state["model"], "fake-model")
                self.assertFalse(state["auto_bash"])
                client.result(client.call("command/run", text="/help"))
                client.result(client.call("shutdown"))
                self.assertEqual(proc.wait(WAIT), 0)
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                proc.stdout.close()
                proc.stderr.close()
            for line in client.raw:
                json.loads(line)  # nothing but protocol on stdout

    def test_declined_trust_is_an_error_and_an_exit(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as ws:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("KYKLOS", "BASE_URL", "API_KEY",
                                                                             "MODEL", "CONTEXT_WINDOW"))}
            env.update(KYKLOS_HOME=home, WORKSPACE=ws, BASE_URL="http://127.0.0.1:9/v1", API_KEY="test",
                       MODEL="fake-model", CONTEXT_WINDOW="32000", KYKLOS_TRACE="0", KYKLOS_PERSONA="",
                       KYKLOS_PROFILE="")
            proc = subprocess.Popen([sys.executable, str(HERE / "kyklos.py"), "--serve"], cwd=ws, env=env,
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            client = Client(proc.stdin, proc.stdout)
            try:
                client.result(client.call("initialize", protocol_version=PROTOCOL_VERSION))
                start = client.call("session/start", resume=None)
                client.answer(client.asked("trust/request"), choice="no")
                self.assertEqual(client.result(start)["error"]["code"], -32004)
                self.assertNotEqual(proc.wait(WAIT), 0)
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                proc.stdout.close()
                proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
