"""Kyklos serve: the agent behind a JSON-RPC pipe, for a desktop app or an editor (PROTOCOL.md is the spec).

Like ui.py, this is a front end and nothing more. The Agent already emits events, asks for
permission, and takes its startup questions and slash-command output as callbacks; ServeUI supplies
those callbacks and writes JSON lines where AeroUI draws. Every judgement a prompt carries (the
explainer's findings, which answers are on offer, the diff) is made here with the terminal's own
functions, so a client can draw a prompt wrongly but can't make it more permissive.

Threads: the agent runs on the main thread, exactly as in the terminal. A reader thread owns the
input pipe: it routes the client's answers to waiting requests, queues work for the main thread,
and turns `session/cancel` into the SIGINT that Ctrl+C would have been.
"""

import json
import os
import queue
import signal
import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

from kyklos import (BASH_MODE_CAVEAT, BASH_MODE_EXPLANATION, BROAD_WARNING, COMMANDS, EXIT, NO_APPROVER_NOTE,
                    TRUST_EXPLANATION, CommandContext, apply_permission_answer, denial_note, edit_preview,
                    error_detail, has_danger, has_opaque, permission_options, run_command, sandbox_warning,
                    state_snapshot)

PROTOCOL_VERSION = 0

# JSON-RPC's own codes, then Kyklos's (PROTOCOL.md §4).
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602
BUSY, API_ERROR, NOT_READY, UNTRUSTED, COMMAND_FAILED = -32001, -32002, -32003, -32004, -32005

ARG_DISPLAY_CHARS = 600  # what the terminal shows of a non-file, non-bash argument value
POLL = 0.1  # seconds between checks while waiting; short enough that a cancel lands at once


def claim_stdio():
    """Take stdin and stdout for the protocol, and leave the process nothing it can corrupt them with.

    The protocol keeps private copies of fds 0 and 1. Then fd 1 is pointed at stderr, so a stray
    print from anywhere (Kyklos, a library, a C extension) lands in the log rather than in the middle
    of a JSON line. fd 0 is pointed at /dev/null, so nothing that inherits it can swallow the
    client's messages. Bash already runs with stdin closed, and this covers everything else.
    """
    sys.stdout.flush()
    proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
    proto_in = os.fdopen(os.dup(0), "rb")
    os.dup2(2, 1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    return proto_in, proto_out


def server_info(workspace: Path, sandbox: tuple, profiles: list, profile: str | None, version: str) -> dict:
    """The `initialize` result, minus the protocol version ServeUI adds."""
    kind, label = sandbox
    return {
        "kyklos_version": version,
        "workspace": str(workspace),
        "sandbox": {"kind": kind, "label": label, "warning": sandbox_warning() or None},
        "commands": [{"name": name, "usage": usage, "summary": summary}
                     for name, (_, usage, summary) in COMMANDS.items()],
        "profiles": profiles,
        "profile": profile,
    }


def line_counts(old: str, new: str) -> tuple[int, int]:
    """(added, removed) lines between two texts, a modified line counting once each way."""
    a, b = old.splitlines(), new.splitlines()
    added = removed = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            removed += i2 - i1
            added += j2 - j1
    return added, removed


def interrupt_main_thread() -> None:
    """Deliver SIGINT to the main thread itself, which is what breaks a blocked communicate().
    A process-wide signal can land on the reader thread and leave the agent asleep until its timeout."""
    if hasattr(signal, "pthread_kill"):
        signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
    else:  # Windows: lands at the agent's next bytecode, not inside a blocked call (PROTOCOL.md §6)
        import _thread
        _thread.interrupt_main()


class ServeUI:
    def __init__(self, workspace: Path, inp, out, interrupt=interrupt_main_thread):
        self.workspace = Path(workspace)
        self.inp, self.out = inp, out  # binary streams: lines in, lines out
        self._interrupt = interrupt
        self._write_lock = threading.Lock()
        self._lock = threading.Lock()  # pending requests, ids, seq
        self._pending: dict[str, dict] = {}  # id -> {"event", "result"}
        self._next_id = 0
        self._seq = 0
        self.work: queue.Queue = queue.Queue()  # messages for the main thread; None means hang-up
        self.info: dict = {}
        self.initialized = False
        self.started = threading.Event()
        self.start_request: tuple | None = None  # (id, params) of session/start, answered by run()
        self.ready = False  # session/start answered; turns and commands are accepted
        self.busy = False  # a turn or command is queued or running
        self.interruptible = False  # a cancel may interrupt the main thread now
        self.closed = False  # the client hung up
        self.state: dict = {}
        self.agent = None
        self.ctx: CommandContext | None = None
        self._open_calls: dict[int, str | None] = {}  # depth -> call_id of the tool call in flight
        self._running: list[int] = []  # depths with a tool call in flight, innermost last

    # ---- transport --------------------------------------------------------------------------

    def send(self, message: dict) -> None:
        # ASCII-only: every message stays one line whatever the text holds, and a lone surrogate
        # from a badly encoded file can't make the encoder raise mid-turn.
        line = (json.dumps(message, ensure_ascii=True, default=str) + "\n").encode("ascii")
        with self._write_lock:
            if self.closed:
                return
            try:
                self.out.write(line)
                self.out.flush()
            except (BrokenPipeError, OSError, ValueError):
                self.closed = True

    def notify(self, method: str, params: dict) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def respond(self, rid, result) -> None:
        if rid is not None:
            self.send({"jsonrpc": "2.0", "id": rid, "result": result})

    def fail(self, rid, code: int, message: str, data=None) -> None:
        if rid is None and code not in (PARSE_ERROR, INVALID_REQUEST):
            return  # a notification gets no reply, even an error
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        self.send({"jsonrpc": "2.0", "id": rid, "error": error})

    def request(self, method: str, params: dict) -> dict | None:
        """Ask the client and wait for its answer, or None if it can't or won't give one.

        The wait polls, so the main thread stays interruptible: a cancel during a permission prompt
        stops the turn exactly as Ctrl+C at the terminal's prompt does.
        """
        if self.closed:
            return None
        with self._lock:
            self._next_id += 1
            rid = f"k{self._next_id}"
            slot = {"event": threading.Event(), "result": None}
            self._pending[rid] = slot
        try:
            self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            while not slot["event"].wait(POLL):
                if self.closed:
                    return None
        finally:
            with self._lock:
                self._pending.pop(rid, None)
        result = slot["result"]
        return result if isinstance(result, dict) else None

    # ---- the reader thread ------------------------------------------------------------------

    def start_reader(self) -> None:
        threading.Thread(target=self._read_loop, name="kyklos-serve-reader", daemon=True).start()

    def _read_loop(self) -> None:
        try:
            for raw in iter(self.inp.readline, b""):
                if raw.strip():
                    self.handle(raw)
        except (OSError, ValueError):
            pass
        self._hang_up()

    def handle(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw)
        except ValueError:
            self.fail(None, PARSE_ERROR, "not valid JSON")
            return
        if not isinstance(msg, dict):
            self.fail(None, INVALID_REQUEST, "a message must be a JSON object")
            return
        if "method" not in msg:  # an answer to one of our requests
            with self._lock:
                slot = self._pending.get(msg.get("id"))
            if slot is not None:
                slot["result"] = msg.get("result") if "error" not in msg else None
                slot["event"].set()
            return
        method, rid, params = msg.get("method"), msg.get("id"), msg.get("params")
        if params is None:
            params = msg["params"] = {}
        if not isinstance(method, str) or not isinstance(params, dict):
            self.fail(rid, INVALID_PARAMS if isinstance(method, str) else INVALID_REQUEST,
                      "params must be an object" if isinstance(method, str) else "method must be a string")
            return
        if method == "session/cancel":
            self.cancel()
        elif method == "initialize":
            if self.initialized:
                self.fail(rid, INVALID_REQUEST, "already initialized")
            else:
                self.initialized = True
                self.respond(rid, self.info)
        elif not self.initialized:
            self.fail(rid, NOT_READY, "send initialize first")
        elif method == "session/start":
            if self.start_request is not None:
                self.fail(rid, INVALID_REQUEST, "the session has already started")
            else:
                self.start_request = (rid, params)
                self.started.set()
        elif not self.ready:
            self.fail(rid, NOT_READY, "send session/start and wait for its result first")
        elif method == "session/state":
            self.respond(rid, self.state)  # the latest snapshot: never reads the agent mid-turn
        elif method == "shutdown":
            self.cancel()
            self.work.put(msg)
        elif method in ("session/prompt", "command/run"):
            if self.busy:
                self.fail(rid, BUSY, "a turn or command is already running; send session/cancel to stop it")
            else:
                self.busy = True
                self.work.put(msg)
        else:
            self.fail(rid, METHOD_NOT_FOUND, f"unknown method {method}")

    def cancel(self) -> None:
        """Stop the turn in flight, the way Ctrl+C does, and take back any question it was asking."""
        if not self.busy:
            return
        with self._lock:
            asked = list(self._pending.items())
        if self.interruptible:
            self._interrupt()
        for rid, slot in asked:
            slot["event"].set()  # result stays None: nobody answered
            self.notify("request/withdrawn", {"id": rid})

    def _hang_up(self) -> None:
        """EOF: whatever was being asked is answered by nobody, and the turn in flight stops."""
        self.cancel()
        self.closed = True
        with self._lock:
            for slot in self._pending.values():
                slot["event"].set()
        self.started.set()
        self.work.put(None)

    def _on_sigint(self, signum, frame) -> None:
        # Installed only while serving. A late cancel, or a Ctrl+C in the terminal that launched the
        # client, does nothing between turns: only the client decides when the process ends.
        if self.interruptible:
            raise KeyboardInterrupt

    # ---- startup ------------------------------------------------------------------------------

    def handshake(self, info: dict) -> dict | None:
        """Answer `initialize` and wait for `session/start`, whose params are returned (None on EOF).
        session/start itself is answered by run(), once trust and the bash mode are settled."""
        self.info = {"protocol_version": PROTOCOL_VERSION, **info}
        self.start_reader()
        while not self.started.wait(POLL):
            pass
        return self.start_request[1] if self.start_request else None

    def fail_start(self, code: int, message: str) -> None:
        if self.start_request:
            self.fail(self.start_request[0], code, message)

    def ask_trust(self, folder: Path, broad: bool) -> str:
        """Returns "yes" (trust and remember), "once" (this session) or "no"."""
        choices = ([] if broad else ["yes"]) + ["once", "no"]
        reply = self.request("trust/request", {
            "folder": str(folder), "broad": broad, "explanation": TRUST_EXPLANATION,
            "warning": BROAD_WARNING if broad else None, "choices": choices})
        choice = reply.get("choice") if reply else None
        return choice if choice in choices else "no"

    def ask_bash_mode(self) -> bool:
        reply = self.request("session/bash_mode", {
            "explanation": BASH_MODE_EXPLANATION, "caveat": BASH_MODE_CAVEAT,
            "sandbox_warning": sandbox_warning() or None, "choices": ["ask", "unflagged"]})
        return bool(reply) and reply.get("mode") == "unflagged"

    # ---- agent callbacks ------------------------------------------------------------------------

    def observer(self, event: str, depth: int = 0, **data) -> None:
        if event == "tool_call":
            self._open_calls[depth] = data.get("call_id")
            self._running.append(depth)
        elif event == "tool_result":
            self._open_calls.pop(depth, None)
            if self._running and self._running[-1] == depth:  # the stream is strictly nested
                self._running.pop()
        with self._lock:
            self._seq += 1
            seq = self._seq
        self.notify("session/update", {"seq": seq, "depth": depth, "event": event, "data": data})
        if event == "model_end" and depth == 0 and threading.current_thread() is threading.main_thread():
            self.push_state()

    def permission(self, name: str, args: dict, always_allowed: set) -> tuple[bool, str]:
        depth = self._running[-1] if self._running else 0
        touches_secret, prefix, findings = permission_options(name, args)
        danger, opaque = has_danger(findings), has_opaque(findings)
        # The same keys the terminal shows for this call, and no others (ui.py AeroUI.permission).
        choices = ["y", "n"] + (["p"] if prefix else []) \
            + ([] if touches_secret or danger or opaque else ["a"])
        params = {"depth": depth, "call_id": self._open_calls.get(depth), "tool": name, "args": args,
                  "display": self._display(name, args),
                  "findings": [{"level": level, "text": text} for level, text in findings],
                  "choices": choices, "prefix": prefix or None, "touches_secret": touches_secret}
        while True:
            reply = self.request("permission/request", params)
            if reply is None:
                return False, NO_APPROVER_NOTE.format(name=name)
            answer = str(reply.get("answer", "")).strip().lower()
            if answer in choices:
                break
            # An answer the prompt didn't offer is never read as anything, least of all a yes: the
            # same question goes out again, saying why the last answer wasn't taken.
            params = {**params, "rejected": {"answer": reply.get("answer"),
                                             "reason": f"not one of the offered choices {choices}"}}
        if answer == "p":
            custom = str(reply.get("prefix") or "").strip()
            answer = f"p {custom}" if custom else "p"
        if apply_permission_answer(name, answer, always_allowed, touches_secret, prefix, danger, opaque):
            return True, ""
        return False, denial_note(name, str(reply.get("reason") or "").strip())

    def _display(self, name: str, args: dict) -> dict:
        if name in ("edit_file", "write_file"):
            preview = edit_preview(self.workspace, name, args)
            added, removed = line_counts(preview["old"], preview["new"])
            return {"kind": "diff", "path": str(args.get("path", "")), "old": preview["old"],
                    "new": preview["new"], "first_line": preview["first"], "whole_file": preview["trim"],
                    "added": added, "removed": removed,
                    "new_file": name == "write_file" and not preview["old"],
                    "overwrite": name == "write_file" and bool(preview["old"])}
        if name == "bash":
            return {"kind": "command", "command": str(args.get("command", ""))}
        shown = {}
        for key, value in args.items():
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            shown[str(key)] = text if len(text) <= ARG_DISPLAY_CHARS else text[:ARG_DISPLAY_CHARS] + "…"
        return {"kind": "args", "args": shown}

    # ---- command callbacks ----------------------------------------------------------------------

    def say(self, kind: str, text: str) -> None:
        self.notify("say", {"kind": kind, "text": str(text)})

    def pick(self, title: str, rows: list, cursor: int = 0, marked: int = None, filterable: bool = False):
        reply = self.request("ui/pick", {"title": title, "rows": [[str(cell) for cell in row] for row in rows],
                                         "cursor": cursor, "marked": marked, "filterable": filterable})
        index = reply.get("index") if reply else None
        if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(rows):
            return index
        return None

    def render_view(self, view: str, data, fallback: str = "") -> None:
        self.notify("view", {"name": view, "data": data, "fallback": fallback})

    # ---- the loop ------------------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {**state_snapshot(self.agent), "profile": self.ctx.profile if self.ctx else None,
                "sandbox": self.info.get("sandbox", {}).get("kind")}

    def push_state(self) -> None:
        self.state = self.snapshot()
        self.notify("session/state", self.state)

    def run(self, agent, default_session: Path, one_shot: str = None, notice: tuple = None, **ctx_extras) -> None:
        self.agent = agent
        self.ctx = CommandContext(agent, self.say, default_session, pick=self.pick,
                                  render=self.render_view, **ctx_extras)
        try:
            previous = signal.signal(signal.SIGINT, self._on_sigint)
        except ValueError:  # not the main thread: nothing to interrupt the agent with
            previous = None
        try:
            self.state = self.snapshot()
            if self.start_request:
                self.respond(self.start_request[0], {
                    "state": self.state, "note": {"kind": notice[0], "text": notice[1]} if notice else None})
            self.ready = True
            while True:
                msg = self._next_work()
                if msg is None:
                    break
                method, rid = msg["method"], msg.get("id")
                if method == "shutdown":
                    self.respond(rid, None)
                    break
                try:
                    if method == "session/prompt":
                        self._prompt(rid, msg["params"], default_session)
                    elif self._command(rid, msg["params"]):
                        break
                finally:
                    self.busy = False
        finally:
            self.ready = False
            if previous is not None:
                signal.signal(signal.SIGINT, previous)

    def _next_work(self):
        while True:
            try:
                return self.work.get(timeout=POLL)
            except queue.Empty:
                continue

    def _prompt(self, rid, params: dict, default_session: Path) -> None:
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            self.fail(rid, INVALID_PARAMS, "text must be a non-empty string")
            return
        agent, started = self.ctx.agent, time.monotonic()
        answer = None
        try:
            try:
                self.interruptible = True
                answer = agent.send(text)
            finally:
                self.interruptible = False
        except KeyboardInterrupt:
            agent.repair_after_interrupt()
        except Exception as e:  # API errors: bad key, rate limit, network...
            self._settle()
            self.fail(rid, API_ERROR, f"API error: {error_detail(e)}", {"detail": error_detail(e)})
            return
        agent.save_session(default_session)
        self._settle()
        self.respond(rid, {
            "answer": answer, "outcome": agent.last_outcome, "steps": agent.last_steps,
            "secs": round(time.monotonic() - started, 3),
            "claims": [{"label": label, "missing": missing} for label, missing in agent.last_claims]})

    def _command(self, rid, params: dict) -> bool:
        """Run a slash command. Returns True if it was /exit."""
        text = params.get("text")
        if not isinstance(text, str) or not text.strip().startswith("/"):
            self.fail(rid, INVALID_PARAMS, "text must be a slash command, such as /tokens")
            return False
        result = None
        try:
            try:
                self.interruptible = True  # /compact calls the model, and Ctrl+C stops it in the terminal
                result = run_command(self.ctx, text.strip())
            finally:
                self.interruptible = False
        except KeyboardInterrupt:
            self.say("warn", "Interrupted.")
        except Exception as e:
            self._settle()
            self.fail(rid, COMMAND_FAILED, str(e))
            return False
        self._settle()
        self.respond(rid, {"then_send": result if isinstance(result, str) else None, "exit": result is EXIT})
        return result is EXIT

    def _settle(self) -> None:
        """After a turn or command: nothing is in flight any more, and the client hears the new state."""
        self._open_calls.clear()
        self._running.clear()
        self.push_state()
