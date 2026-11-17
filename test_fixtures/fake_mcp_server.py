"""Offline stdio MCP server for the Kyklos client tests."""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=Path, required=True)
parser.add_argument("--version", default="2025-11-25")
parser.add_argument("--paginated", action="store_true")
parser.add_argument("--startup", choices=("slow", "bad-json", "stderr"))
parser.add_argument("--child", type=Path)
args = parser.parse_args()
SCHEMA = {"type": "object", "properties": {"value": {"type": "string"}}}
changed = False

if args.child:
    child_code = ("import fcntl, os, sys, time\n"
                  "with open(f'{sys.argv[1]}.{os.getpid()}.lock', 'w') as lock:\n"
                  "    fcntl.flock(lock, fcntl.LOCK_EX)\n"
                  "    time.sleep(60)\n")
    child = subprocess.Popen([sys.executable, "-c", child_code, str(args.child)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    args.child.write_text(str(child.pid), encoding="utf-8")

if args.startup == "stderr":
    sys.stderr.write("x" * 100_000 + "\n")
    sys.stderr.write("\x1b[31mServer log " + os.environ.get("KEY_FOR_SERVER", "") + "\x1b[0m\n")
    sys.stderr.flush()


def send(**message):
    print(json.dumps({"jsonrpc": "2.0", **message}), flush=True)


def definition(name, **extra):
    return {"name": name, "description": "Offline fixture.", "inputSchema": SCHEMA, **extra}


def call_result(rid, *blocks, error=False):
    send(id=rid, result={"content": blocks, "isError": error})


def text(value):
    return {"type": "text", "text": value}


for raw in sys.stdin:
    request = json.loads(raw)
    with args.log.open("a", encoding="utf-8") as log:
        log.write(json.dumps(request) + "\n")
    method, rid, params = request.get("method"), request.get("id"), request.get("params", {})
    if method == "initialize":
        if args.startup == "slow":
            time.sleep(60)
        if args.startup == "bad-json":
            print("startup banner on stdout", flush=True)
            continue
        send(id=rid, result={"protocolVersion": args.version, "capabilities": {"tools": {"listChanged": True}},
                             "serverInfo": {"name": "fake", "version": "1"}})
    elif method == "tools/list":
        if params.get("cursor") == "second":
            send(id=rid, result={"tools": [definition("a_b"), definition("x" * 80 + "2"),
                                           definition("bad-schema", inputSchema="invalid")]})
        else:
            names = ["echo", "wait", "crash", "error", "secret", "huge", "image", "change", "malformed"]
            tools = [definition(name) for name in names]
            tools.append(definition("annotated", annotations={"readOnlyHint": True}))
            tools.append(definition("schema-secret", description=os.environ.get("KEY_FOR_SERVER", "Safe"),
                                    inputSchema={"type": "object", "description":
                                                 os.environ.get("KEY_FOR_SERVER", "Safe")}))
            if changed:
                tools.extend([definition("new-tool"), definition("a\tb")])
            result = {"tools": tools}
            if args.paginated:
                tools.extend([definition("a b"), definition("x" * 80 + "1")])
                result["nextCursor"] = "second"
            send(id=rid, result=result)
    elif method == "notifications/cancelled":
        call_result(params["requestId"], text("A late response to a cancelled request."))
    elif method == "tools/call":
        name, arguments = params["name"], params["arguments"]
        secret = os.environ.get("KEY_FOR_SERVER", "")
        if name == "wait":
            continue
        if name == "crash":
            os._exit(5)
        if name == "malformed":
            print("invalid json", flush=True)
            continue
        if name == "error":
            call_result(rid, text("The fixture encountered an error."), error=True)
        elif name == "secret":
            call_result(rid, text("Forwarded " + secret), text("Unpassed " + os.getenv("UNPASSED_SECRET", "absent")))
        elif name == "huge":
            call_result(rid, text("head\n" + "a" * 11_000 + "\n" + secret + "\n" + "b" * 11_000 + "\ntail"))
        elif name == "image":
            png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x04\x00\x00\x00\x03" + b"\x00" * 24
            call_result(rid, text("One image follows."), {"type": "image", "mimeType": "image/png",
                                                          "data": base64.b64encode(png).decode("ascii")},
                        {"type": "resource_link", "uri": "test:resource"})
        elif name == "change":
            changed = True
            send(method="notifications/tools/list_changed")
            call_result(rid, text("Changed."))
        else:
            call_result(rid, text(name), text(arguments.get("value", "")))
