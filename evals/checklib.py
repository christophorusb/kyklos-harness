"""Helpers shared by the eval checks.

eval.py copies this beside a task's check.py and runs the check inside the sandbox, with the
finished workspace as the current directory, on whatever python3 the machine has -- so this stays
compatible with Python 3.9 and the standard library.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

WS = Path.cwd()


def answer():
    """What Kyklos printed: its final answer."""
    return Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace") if len(sys.argv) > 1 else ""


def before():
    """{path: sha256} of every file in the workspace when the task began."""
    return json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")) if len(sys.argv) > 2 else {}


def digest(path):
    return hashlib.sha256((WS / path).read_bytes()).hexdigest()


def fail(reason):
    print("FAIL: " + reason)
    sys.exit(1)


def ok(note="passed"):
    print("PASS: " + note)
    sys.exit(0)


def unchanged(*paths):
    """Fail unless each path -- every original file, when none is named -- is as it was."""
    start = before()
    for path in paths or sorted(start):
        if not (WS / path).is_file():
            fail(path + " was deleted")
        if digest(path) != start.get(path):
            fail(path + " was changed, and the task said to leave it alone")


def run(*argv, timeout=120):
    """(exit code, output) of a command run in the workspace."""
    try:
        proc = subprocess.run(list(argv), cwd=str(WS), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "timed out after %ss" % timeout
    return proc.returncode, proc.stdout + proc.stderr


def python(*args, timeout=120):
    return run(sys.executable, "-B", *args, timeout=timeout)
