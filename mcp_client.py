"""A synchronous MCP client for user-configured stdio servers."""

import base64
import binascii
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from tools import (IS_WINDOWS, MAX_IMAGE_BYTES, SENDABLE_IMAGES, child_env, describe_image, image_info,
                   is_secret_env, kill_process_tree, known_secret_values, plain_line, sanitize_terminal_text,
                   scrub_secrets, workspace_secret_values)

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
MCP_PREFIX = "mcp__"
# Seconds a server has to finish the handshake and list its tools. Generous by default because an
# `npx -y` or `uvx` server's first start downloads its package; servers start side by side, so
# the wait is the slowest server's, not the sum. `startupTimeout` in mcp.json sets it per server.
STARTUP_TIMEOUT = 30
CALL_TIMEOUT = 120
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_CONFIG_BYTES = 256 * 1024
MAX_TOOL_PAGES = 100
STDERR_LINES = 20
STDERR_LINE_BYTES = 2048
POLL = 0.1
MAX_TOOL_NAME = 64
ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
UNSAFE_TOOL_CHAR = re.compile(r"[^A-Za-z0-9_-]")
UNSANDBOXED_NOTICE = "MCP servers run unsandboxed with your account's filesystem and network access."


class MCPError(Exception):
    pass


def _invalid_number(value: str):
    raise ValueError(f"{value} is not a JSON number")


def schema_tokens(schemas: list[dict]) -> int:
    return (len(json.dumps(schemas, ensure_ascii=False)) + 3) // 4 if schemas else 0


def _clean_text(text: str, secrets: list[str]) -> str:
    clean = scrub_secrets(text, values=secrets)[0]
    display = sanitize_terminal_text(clean).encode("utf-8", errors="replace").decode("utf-8")
    # Removing a terminal escape can rejoin a secret that was split by the escape.
    return scrub_secrets(display, values=secrets)[0] if display != clean else clean


@dataclass(frozen=True)
class ServerConfig:
    command: str
    args: tuple[str, ...]
    env: dict[str, str]
    auto_approve: frozenset[str]
    read_only: frozenset[str]
    timeout: float
    startup_timeout: float


def _server_config(data, environ) -> tuple[ServerConfig, set[str]]:
    if (not isinstance(data, dict) or data.get("type", "stdio") != "stdio"
            or not isinstance(data.get("command"), str) or not data["command"].strip()):
        raise MCPError("Configure a stdio server with a command.")
    for key in ("args", "autoApprove", "readOnly"):
        values = data.get(key, [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise MCPError(f"{key} must be an array of strings.")
    seconds = _seconds(data, "timeout", CALL_TIMEOUT)
    startup_seconds = _seconds(data, "startupTimeout", STARTUP_TIMEOUT)
    entries = data.get("env", {})
    if not isinstance(entries, dict) or not all(isinstance(value, str) for value in entries.values()):
        raise MCPError("env must be an object with string values.")
    env, secrets = child_env(environ), set()

    def expand(match):
        name = match[1]
        if name not in environ:
            raise MCPError(f"Set {name} in Kyklos's environment before starting this server.")
        value = environ[name]
        if value:
            secrets.add(value)
        return value

    for name, raw in entries.items():
        value = ENV_REFERENCE.sub(expand, raw)
        if value and (value != raw or is_secret_env(name, value)):
            secrets.add(value)
        env[name] = value
    return (ServerConfig(data["command"], tuple(data.get("args", [])), env,
                         frozenset(data.get("autoApprove", [])), frozenset(data.get("readOnly", [])),
                         seconds, startup_seconds), secrets)


def _seconds(data: dict, key: str, default: float) -> float:
    value = data.get(key, default)
    try:
        seconds = float(value) if type(value) in (int, float) else 0
    except OverflowError:
        seconds = 0
    if not 0 < seconds < math.inf:
        raise MCPError(f"{key} must be a positive number of seconds.")
    return seconds


def _launch(config: ServerConfig, home: Path, workspace: Path) -> subprocess.Popen:
    env = dict(config.env)
    # An executable found through a relative or workspace PATH entry would let the repository
    # choose what starts. Resolve the command before launching it from the user config directory.
    entries = [Path(entry) for entry in os.get_exec_path(env)]
    path = os.pathsep.join(str(entry) for entry in entries
                           if entry.is_absolute() and not entry.resolve().is_relative_to(workspace))
    env["PATH"] = path
    command = Path(config.command)
    if not command.is_absolute():
        if command.name == config.command:
            found = shutil.which(config.command, path=path) if path else None
            if not found:
                raise MCPError("The configured executable was not found on a trusted PATH.")
            command = Path(found)
        else:
            command = home / command
        command = command.resolve()
        if command.is_relative_to(workspace):
            raise MCPError("Use an executable outside the workspace for a relative MCP command.")
    # Ctrl+C belongs to the client. The server receives a JSON-RPC cancellation instead.
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
    return subprocess.Popen([str(command), *config.args], cwd=home, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=not IS_WINDOWS,
                            creationflags=flags)


class Stdio:
    """One request at a time. Workers keep both pipes drained and bound blocked writes."""

    def __init__(self, config: ServerConfig, home: Path, workspace: Path):
        self.lock = threading.Lock()
        self.pending = None
        self.next_id = 0
        self.generation = 0
        self.failure = None
        self.stderr = deque(maxlen=STDERR_LINES)
        self.outbox = queue.Queue(maxsize=16)
        self.stopped = threading.Event()
        self.process = _launch(config, home, workspace)
        self.workers = []
        try:
            for target in (self._read, self._write, self._read_stderr):
                worker = threading.Thread(target=target, daemon=True, name=f"kyklos-mcp-{target.__name__}")
                worker.start()
                self.workers.append(worker)
        except BaseException:
            self.close()
            raise

    def _fail(self, reason: str):
        with self.lock:
            if self.failure is None:
                self.failure = reason
            if self.pending is not None:
                _, slot = self.pending
                self.pending = None
                slot.put(MCPError(self.failure))

    def status_error(self) -> str | None:
        code = self.process.poll()
        return self.failure or (f"The server is down (exit {code})." if code is not None else None)

    def _send(self, message: dict):
        try:
            frame = json.dumps({"jsonrpc": "2.0", **message}, allow_nan=False).encode("utf-8") + b"\n"
            if len(frame) > MAX_FRAME_BYTES:
                raise MCPError("The MCP request is too large. Send fewer arguments.")
            self.outbox.put_nowait(frame)
        except (ValueError, RecursionError) as error:
            raise MCPError("The MCP request was not valid JSON. Check its arguments.") from error
        except queue.Full as error:
            raise MCPError("The server is not reading requests. Restart it or use another tool.") from error

    def _write(self):
        try:
            while not self.stopped.is_set():
                try:
                    frame = self.outbox.get(timeout=POLL)
                except queue.Empty:
                    continue
                self.process.stdin.write(frame)
                self.process.stdin.flush()
        except (OSError, ValueError):
            self._fail("The server is down (stdin closed).")
        finally:
            try:
                self.process.stdin.close()
            except OSError:
                pass

    def _read(self):
        try:
            while not self.stopped.is_set():
                frame = self.process.stdout.readline(MAX_FRAME_BYTES + 1)
                if not frame:
                    self._fail("The server is down (stdout closed).")
                    return
                if len(frame) > MAX_FRAME_BYTES:
                    self._fail("The server sent an MCP message that exceeded the size limit.")
                    return
                if not frame.endswith(b"\n"):
                    raise ValueError("incomplete frame")
                message = json.loads(frame.decode("utf-8"), parse_constant=_invalid_number)
                if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                    raise ValueError("invalid envelope")
                self._handle(message)
        except (ValueError, UnicodeError, RecursionError):
            self._fail("The server sent invalid JSON-RPC on stdout.")
        except (OSError, MCPError):
            self._fail("The server's MCP connection closed.")
        finally:
            self.process.stdout.close()

    def _handle(self, message: dict):
        method, rid = message.get("method"), message.get("id")
        if isinstance(method, str):
            if type(rid) in (int, str):
                answer = ({"result": {}} if method == "ping" else
                          {"error": {"code": -32601, "message": "Client method is unsupported"}})
                self._send({"id": rid, **answer})
            elif method == "notifications/tools/list_changed":
                with self.lock:
                    self.generation += 1
            return
        with self.lock:
            if self.pending is not None and type(rid) is int and rid == self.pending[0]:
                _, slot = self.pending
                self.pending = None
                slot.put(message)

    def _read_stderr(self):
        with self.process.stderr as stream:
            try:
                while line := stream.readline(STDERR_LINE_BYTES + 1):
                    if len(line) > STDERR_LINE_BYTES:
                        while line and not line.endswith(b"\n"):
                            line = stream.readline(STDERR_LINE_BYTES + 1)
                        line = b"[Long stderr line omitted]\n"
                    with self.lock:
                        self.stderr.append(line)
            except OSError:
                pass

    def stderr_text(self) -> str:
        with self.lock:
            return b"".join(self.stderr).decode("utf-8", errors="replace")

    def notify(self, method: str, params: dict | None = None):
        self._send({"method": method, **({"params": params} if params is not None else {})})

    def request(self, method: str, params: dict, deadline: float) -> dict:
        slot = queue.SimpleQueue()
        self.next_id += 1
        rid = self.next_id
        with self.lock:
            error = self.status_error()
            if error:
                raise MCPError(error)
            self.pending = rid, slot
        sent = False
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError
            self._send({"id": rid, "method": method, "params": params})
            sent = True
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    response = slot.get(timeout=min(remaining, POLL))
                    break
                except queue.Empty:
                    error = self.status_error()
                    if error:
                        raise MCPError(error)
            if isinstance(response, MCPError):
                raise response
            if "error" in response:
                error = response["error"]
                detail = error.get("message") if isinstance(error, dict) else None
                raise MCPError(detail if isinstance(detail, str) and detail else "The server returned an RPC error.")
            result = response.get("result")
            if not isinstance(result, dict):
                raise MCPError("The server returned an invalid MCP result.")
            return result
        except (TimeoutError, KeyboardInterrupt) as error:
            if sent and method != "initialize":
                try:
                    self.notify("notifications/cancelled", {"requestId": rid, "reason": (
                        "The user cancelled the request." if isinstance(error, KeyboardInterrupt)
                        else "The request timed out.")})
                except MCPError:
                    pass
            if isinstance(error, TimeoutError):
                raise MCPError(f"{method} timed out. Check the server or use another tool.") from None
            raise
        finally:
            with self.lock:
                if self.pending is not None and self.pending[0] == rid:
                    self.pending = None

    def close(self):
        if self.stopped.is_set():
            return
        self.stopped.set()
        kill_process_tree(self.process, group=self.process.pid)
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._fail("The MCP server did not exit after termination.")
        for worker in self.workers:
            worker.join(timeout=POLL)


@dataclass(frozen=True)
class MCPTool:
    server: str
    name: str
    schema: dict
    auto_approve: bool
    read_only: bool


@dataclass
class Server:
    config: ServerConfig | None
    wire: Stdio | None = None
    definitions: dict = field(default_factory=dict)
    version: str | None = None
    generation: int = 0
    error: str | None = None
    stderr: str = ""


class MCP:
    def __init__(self, home: Path | None, workspace: Path, version: str, notice):
        self.home = home
        self.workspace = workspace
        self.version = version
        self.notice = notice
        self.servers: dict[str, Server] = {}
        self.tools: dict[str, MCPTool] = {}
        self.forwarded: set[str] = set()
        self.config_error: str | None = None
        self._closing = False
        self._held: dict[int, list[str]] = {}  # startup thread -> its notices, said in server order
        if home is None:
            return
        try:
            self._configure()
            self._start_all()
            self._catalog()
        except BaseException:
            self.close()
            raise

    def _start_all(self):
        """Start every configured server at once, so startup waits for the slowest, not the sum.

        Each thread holds its notices until all are done; they are then said from this thread, in
        server order, so the front end only ever hears from the thread that owns it."""
        pending = [(name, server) for name, server in self.servers.items() if server.config]
        if not pending:
            return
        held = {name: [] for name, _ in pending}

        def start(name, server):
            self._held[threading.get_ident()] = held[name]
            try:
                self._start(name, server)
            finally:
                self._held.pop(threading.get_ident(), None)

        threads = [threading.Thread(target=start, args=entry, daemon=True, name=f"kyklos-mcp-start-{i}")
                   for i, entry in enumerate(pending)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()  # each start is bounded by its server's own startupTimeout
        for name, _ in pending:
            for text in held[name]:
                self.notice(text)

    def clean(self, text: str) -> str:
        return _clean_text(text, self._secrets())

    def _secrets(self) -> list[str]:
        values = self.forwarded | set(known_secret_values()) | set(workspace_secret_values(self.workspace))
        return sorted(values, key=lambda value: (-len(value), value))

    def _say(self, text: str):
        held = self._held.get(threading.get_ident())
        if held is not None:
            held.append(self.clean(text))
        else:
            self.notice(self.clean(text))

    def _configure(self):
        try:
            if self.home.absolute().is_relative_to(self.workspace):
                return
            self.home = self.home.resolve()
            path = self.home / "mcp.json"
            if self.home.is_relative_to(self.workspace):
                return  # a workspace-controlled KYKLOS_HOME cannot authorize starting programs
            if path.is_symlink():
                raise MCPError("mcp.json must be a regular user config file.")
            if not path.exists():
                return
            if not path.is_file():
                raise MCPError("mcp.json must be a regular user config file.")
            with path.open("rb") as stream:
                text = stream.read(MAX_CONFIG_BYTES + 1)
            if len(text) > MAX_CONFIG_BYTES:
                raise MCPError("mcp.json exceeds the configuration size limit.")
            data = json.loads(text, parse_constant=_invalid_number)
            if not isinstance(data, dict) or not isinstance(data.get("mcpServers"), dict):
                raise MCPError("mcp.json needs an mcpServers object.")
        except MCPError as error:
            self.config_error = str(error)
        except (OSError, ValueError, RuntimeError):
            self.config_error = "Could not read the user mcp.json as JSON."
        if self.config_error:
            self._say(f"MCP: {self.config_error}")
            return
        # Expand every entry before starting any server, so diagnostics use the complete redaction set.
        for name, entry in sorted(data["mcpServers"].items()):
            try:
                if not plain_line(name):
                    raise MCPError("Give the server a non-empty name.")
                config, secrets = _server_config(entry, os.environ)
                self.forwarded.update(secrets)
                self.servers[name] = Server(config)
            except MCPError as error:
                self.servers[name] = Server(None, error=str(error))
        for name, server in self.servers.items():
            if server.error:
                self._say(f"MCP server {plain_line(name)}: {server.error}")

    def _start(self, name: str, server: Server):
        deadline = time.monotonic() + server.config.startup_timeout
        server.version, server.error, server.stderr, server.generation = None, None, "", 0
        try:
            server.wire = Stdio(server.config, self.home, self.workspace)
            if self._closing:  # close() ran while this was launching: don't leave it behind
                self._stop(server)
                return
            reply = server.wire.request("initialize", {"protocolVersion": PROTOCOL_VERSIONS[0],
                                        "capabilities": {},
                                        "clientInfo": {"name": "kyklos", "version": self.version}}, deadline)
            version = reply.get("protocolVersion")
            if version not in PROTOCOL_VERSIONS:
                raise MCPError(f"Unsupported MCP protocol version {plain_line(str(version))}. "
                               f"Expected one of {', '.join(PROTOCOL_VERSIONS)}.")
            server.version = version
            server.wire.notify("notifications/initialized")
            capabilities = reply.get("capabilities")
            if isinstance(capabilities, dict) and "tools" in capabilities:
                server.definitions, server.generation = self._list(name, server, deadline)
        except (MCPError, OSError, ValueError, RuntimeError) as error:
            server.error = str(error)
            self._stop(server)
            self._say(f"MCP server {plain_line(name)} failed to start. {server.error}")
        except BaseException:
            self._stop(server)
            raise

    def _list(self, name: str, server: Server, deadline: float) -> tuple[dict, int]:
        generation = server.wire.generation
        params, definitions, cursors = {}, {}, set()
        size = 0
        for _ in range(MAX_TOOL_PAGES):
            reply = server.wire.request("tools/list", params, deadline)
            items = reply.get("tools")
            if not isinstance(items, list):
                raise MCPError("tools/list did not return a tools array.")
            try:
                size += len(json.dumps(items))
            except RecursionError:
                raise MCPError("tools/list returned a deeply nested response.") from None
            if size > MAX_FRAME_BYTES:
                raise MCPError("tools/list exceeded the catalog size limit.")
            for item in items:
                tool = item.get("name") if isinstance(item, dict) else None
                if not isinstance(tool, str) or not plain_line(tool) or not isinstance(item.get("inputSchema"), dict):
                    self._say(f"MCP server {plain_line(name)} skipped tool {plain_line(str(tool))} with an invalid name or schema.")
                    continue
                if tool in definitions:
                    self._say(f"MCP server {plain_line(name)} skipped duplicate tool {plain_line(tool)}.")
                    continue
                definitions[tool] = item
            cursor = reply.get("nextCursor")
            if cursor is None:
                return definitions, generation
            if not isinstance(cursor, str) or not cursor or cursor in cursors:
                raise MCPError("tools/list returned an invalid or repeated cursor.")
            cursors.add(cursor)
            params = {"cursor": cursor}
        raise MCPError("tools/list exceeded the pagination limit.")

    def _catalog(self):
        tools = {}
        values = self._secrets() if self.servers else []

        def clean(value):
            if isinstance(value, str):
                return _clean_text(value, values)
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, dict):
                return {clean(key): clean(item) for key, item in value.items()}
            return value

        for server_name, server in self.servers.items():
            for name, item in sorted(server.definitions.items()):
                alias = UNSAFE_TOOL_CHAR.sub("_", f"{MCP_PREFIX}{clean(server_name)}__{clean(name)}")[:MAX_TOOL_NAME]
                if alias in tools:
                    identity = json.dumps([server_name, name]).encode("utf-8")
                    suffix = hashlib.sha256(identity).hexdigest()[:12]
                    base = alias[:MAX_TOOL_NAME - len(suffix) - 1]
                    alias = f"{base}_{suffix}"
                    while alias in tools:
                        suffix = hashlib.sha256(suffix.encode("ascii")).hexdigest()[:12]
                        alias = f"{base}_{suffix}"
                description = item.get("description")
                summary = clean(description) if isinstance(description, str) else ""
                try:
                    schema = {"type": "function", "function": {"name": alias,
                              "description": f"{plain_line(clean(server_name))}: {summary}".rstrip(),
                              "parameters": clean(item["inputSchema"])}}
                except RecursionError:
                    self._say(f"MCP server {plain_line(server_name)} skipped tool {plain_line(name)} with a deeply nested schema.")
                    continue
                tools[alias] = MCPTool(server_name, name, schema, name in server.config.auto_approve,
                                       name in server.config.read_only)
        self.tools = tools

    def refresh(self) -> bool:
        updates = []
        for name, server in self.servers.items():
            wire = server.wire
            if wire and wire.generation != server.generation and not wire.status_error():
                try:
                    definitions, generation = self._list(name, server,
                                                         time.monotonic() + server.config.startup_timeout)
                    updates.append((name, server, definitions, generation))
                except MCPError as error:
                    server.error = str(error)
                    self._say(f"MCP could not refresh tools for {plain_line(name)}. {error}")
        for _, server, definitions, generation in updates:
            server.definitions, server.generation, server.error = definitions, generation, None
        if updates:
            self._catalog()
            for name, _, _, _ in updates:
                self._say(f"MCP tools updated for {plain_line(name)}.")
        return bool(updates)

    def restart(self, name: str):
        server = self.servers.get(name)
        if server is None or server.config is None:
            raise MCPError("Choose a configured stdio server from /mcp.")
        self._stop(server)
        server.definitions = {}
        try:
            self._start(name, server)
        finally:
            self._catalog()
        if server.error:
            raise MCPError(self.clean(server.error))
        self._say(f"MCP server {plain_line(name)} restarted and its tool list was refreshed.")

    def _stop(self, server: Server):
        if server.wire:
            server.wire.close()
            server.stderr = server.wire.stderr_text()
            server.wire = None

    def close(self):
        self._closing = True
        for server in self.servers.values():
            self._stop(server)

    def describe(self) -> str:
        lines = [UNSANDBOXED_NOTICE]
        if self.config_error:
            lines.append(self.config_error)
        if not self.servers:
            lines.append("No MCP servers are configured.")
        for name, server in self.servers.items():
            wire = server.wire
            down = wire.status_error() if wire else None
            pid = wire.process.pid if wire and not down else None
            schemas = [tool.schema for tool in self.tools.values() if tool.server == name]
            status = "running" if wire and not down else "down"
            lines.append(f"{plain_line(name)} [{status}] · PID {pid or '—'} · {len(schemas)} tools · "
                         f"~{schema_tokens(schemas):,} schema tokens")
            for detail in (down, server.error):
                if detail:
                    lines.append(f"  {plain_line(detail)}")
            if wire and wire.generation != server.generation:
                lines.append("  Tool list update pending until /reset or restart.")
            stderr = wire.stderr_text() if wire else server.stderr
            if stderr:
                lines.append("  Recent stderr")
                lines.extend(f"    {line}" for line in self.clean(stderr).splitlines())
        return self.clean("\n".join(lines))

    def call(self, tool: MCPTool, args: dict, vision: bool) -> tuple[str, list[dict]]:
        server = self.servers[tool.server]
        if not server.wire:
            raise MCPError("The server is down. Ask the user to restart it with /mcp.")
        result = server.wire.request("tools/call", {"name": tool.name, "arguments": args},
                                     time.monotonic() + server.config.timeout)
        return self._content(result, vision)

    def problem(self, tool: MCPTool) -> str | None:
        wire = self.servers[tool.server].wire
        if wire:
            return wire.status_error()
        return "The server is down. Ask the user to restart it with /mcp."

    def _content(self, result: dict, vision: bool) -> tuple[str, list[dict]]:
        blocks = result.get("content")
        if not isinstance(blocks, list):
            raise MCPError("The server did not return MCP content blocks. Use another tool.")
        failed = result.get("isError") is True
        text, images, image_bytes = [], [], 0
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                text.append(block["text"])
            elif kind == "image":
                if failed or not vision:
                    text.append("The MCP server returned an image (not sent as image input).")
                    continue
                image, size = self._image(block, MAX_IMAGE_BYTES - image_bytes)
                if image is None:
                    text.append("The MCP server returned an unsupported, oversized or sensitive image.")
                else:
                    images.append(image)
                    image_bytes += size
                    text.append(f"The MCP server returned an image ({image['desc']}).")
            else:
                label = plain_line(kind)[:80] if isinstance(kind, str) else "unknown"
                text.append(f"The MCP server returned a {label} content block.")
        body = "\n".join(text) or "The MCP server returned no displayable content."
        cleaned = self.clean(body)
        return f"Error: {cleaned}" if failed else cleaned, images

    def _image(self, block: dict, limit: int) -> tuple[dict | None, int]:
        data = block.get("data")
        if not isinstance(data, str) or len(data) > (limit + 2) // 3 * 4:
            return None, 0
        try:
            raw = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error):
            return None, 0
        info = image_info(raw[:64 * 1024])
        if not info or info[0] not in SENDABLE_IMAGES or len(raw) > limit:
            return None, 0
        if any(secret.encode("utf-8", errors="surrogatepass") in raw for secret in self._secrets()):
            return None, 0
        return {"path": "MCP image", "mime": info[0], "data": data,
                "desc": describe_image(*info, len(raw))}, len(raw)
