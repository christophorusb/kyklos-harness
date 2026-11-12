"""Run Kyklos on fixture tasks against a real model, and report what each run did and cost.

    ./.venv/bin/python eval.py                            # every task once, with your usual model
    ./.venv/bin/python eval.py fix-off-by-one long-log    # just these
    ./.venv/bin/python eval.py --repeat 3 --model kimi-k3 --profile cloud
    ./.venv/bin/python eval.py --list

Each run copies a task's repo into a fresh folder under the system temp directory and runs
`kyklos.py --eval` there, with nobody at the keyboard: a fixed approval policy stands in for the
user, and Kyklos refuses it anywhere else or without the OS sandbox (see eval_workspace_problem).
The task's check.py then decides pass or fail. A check runs the model's code, so it runs inside
the same sandbox. How a run went comes from the trace Kyklos writes, never from what the model
says about itself.

This spends real tokens on whichever provider .env points at. Results are appended to
evals/results/<time>.jsonl, one row per run, for comparing models or harness changes later.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from kyklos import EVAL_MARKER, kyklos_home, short_tokens
from tools import child_env, kill_process_tree, probe_sandbox, sandbox_argv

HERE = Path(__file__).resolve().parent
EVALS = HERE / "evals"
TASKS = EVALS / "tasks"
CHECK_TIMEOUT = 300
COPIED_SETTINGS = ("settings.json", "models.json")  # your model picks, and the cached model catalog


def load_tasks(names: list[str]) -> list[dict]:
    known = {p.name: p for p in sorted(TASKS.iterdir()) if (p / "task.json").is_file()}
    unknown = [n for n in names if n not in known]
    if unknown:
        sys.exit(f"No task named {', '.join(unknown)}. Tasks: {', '.join(known)}.")
    tasks = []
    for name in names or known:
        spec = json.loads((known[name] / "task.json").read_text(encoding="utf-8"))
        tasks.append({"name": name, "dir": known[name], **spec})
    return tasks


def manifest(folder: Path) -> dict[str, str]:
    """{path: sha256} of every file under `folder`, for checks that a file was left alone."""
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def summarize_trace(path: Path) -> dict:
    """What a run did, from its trace rows."""
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    m = {"model": None, "steps": 0, "sub_steps": 0, "tools": 0, "tool_failures": 0, "subagents": 0,
         "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cache_seen": 0,
         "api_errors": 0, "compactions": 0, "outcome": None}
    signals: Counter = Counter()
    for r in rows:
        event, depth = r.get("event"), r.get("depth", 0)
        if event == "session":
            m["model"] = r.get("model")
        elif event == "model_end":
            if r.get("error"):
                m["api_errors"] += 1
            if r.get("purpose") == "turn":
                m["steps" if depth == 0 else "sub_steps"] += 1
            prompt = r.get("prompt_tokens") or 0
            m["prompt_tokens"] += prompt
            m["completion_tokens"] += r.get("completion_tokens") or 0
            if r.get("cached_tokens") is not None:
                m["cached_tokens"] += r["cached_tokens"]
                m["cache_seen"] += prompt
        elif event == "tool_result":
            m["tools"] += 1
            if r.get("status") in ("error", "timeout", "denied"):
                m["tool_failures"] += 1
        elif event == "subagent_start":
            m["subagents"] += 1
        elif event == "compact":
            m["compactions"] += 1
        elif event == "signal":
            signals[r.get("kind", "?") + (f":{r['reason']}" if r.get("reason") else "")] += 1
        elif event == "turn_end" and depth == 0:
            m["outcome"] = r.get("outcome")
    m["signals"] = dict(sorted(signals.items()))
    return m


def private_copy(src: Path, dest: Path) -> None:
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(src.read_bytes())


def run_check(task: dict, root: Path, ws: Path) -> tuple[bool, str]:
    """(passed, the check's last line), with the check run inside the sandbox."""
    shutil.copy(EVALS / "checklib.py", root / "checklib.py")
    shutil.copy(task["dir"] / "check.py", root / "check.py")
    python3 = shutil.which("python3") or sys.executable
    argv = sandbox_argv([python3, "-B", str(root / "check.py"), str(root / "answer.txt"),
                         str(root / "before.json")], ws)
    try:
        proc = subprocess.run(argv, cwd=ws, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=child_env(), timeout=CHECK_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, f"FAIL: the check timed out after {CHECK_TIMEOUT}s"
    lines = (proc.stdout + proc.stderr).strip().splitlines()
    verdict = next((l for l in lines if l.startswith(("PASS:", "FAIL:"))), lines[-1] if lines else "")
    return proc.returncode == 0, verdict or f"FAIL: the check exited {proc.returncode} without a verdict"


def run_one(task: dict, forward: list[str], keep: bool) -> dict:
    root = Path(tempfile.mkdtemp(prefix=f"kyklos-eval-{task['name']}-")).resolve()
    (root / EVAL_MARKER).write_text("made by eval.py\n", encoding="utf-8")
    ws, home, trace = root / "ws", root / "home", root / "trace.jsonl"
    shutil.copytree(task["dir"] / "repo", ws)
    home.mkdir(mode=0o700)
    for name in COPIED_SETTINGS:
        if (kyklos_home() / name).is_file():
            private_copy(kyklos_home() / name, home / name)
    before = manifest(ws)
    (root / "before.json").write_text(json.dumps(before), encoding="utf-8")
    env = dict(os.environ, WORKSPACE=str(ws), KYKLOS_HOME=str(home), KYKLOS_EVAL_ROOT=str(root),
               KYKLOS_TRACE=str(trace))
    env.pop("KYKLOS_UI", None)
    cmd = [sys.executable, str(HERE / "kyklos.py"), "--eval", *forward, task["prompt"]]
    started, timed_out = time.monotonic(), False
    proc = subprocess.Popen(cmd, cwd=ws, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                            env=env, start_new_session=(os.name != "nt"))
    try:
        out, err = proc.communicate(timeout=task.get("timeout", 600))
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        out, err = proc.communicate()
        timed_out = True
    secs = time.monotonic() - started
    (root / "answer.txt").write_text(out or "", encoding="utf-8")
    if timed_out:
        passed, verdict = False, f"FAIL: Kyklos was still running after {task.get('timeout', 600)}s"
    elif proc.returncode:
        tail = (err or out or "").strip().splitlines()
        passed, verdict = False, f"FAIL: Kyklos exited {proc.returncode}: {tail[-1] if tail else ''}"
    else:
        passed, verdict = run_check(task, root, ws)
    row = {"task": task["name"], "passed": passed, "verdict": verdict, "secs": round(secs, 1),
           **summarize_trace(trace)}
    if keep:
        row["kept"] = str(root)
    else:
        shutil.rmtree(root, ignore_errors=True)
    return row


def describe(row: dict) -> str:
    seen = row["cache_seen"]
    cache = f"{row['cached_tokens'] / seen:.0%}" if seen else "n/a"
    tokens = short_tokens(row["prompt_tokens"] + row["completion_tokens"])
    subs = f"  sub {row['subagents']} ({row['sub_steps']} steps)" if row["subagents"] else ""
    signals = "  " + " ".join(f"{k}×{v}" for k, v in row["signals"].items()) if row["signals"] else ""
    return (f"{'PASS' if row['passed'] else 'FAIL'}  {row['task']:<16} steps {row['steps']:>2}  "
            f"tools {row['tools']:>2}{subs}  tokens {tokens:>5}  cache {cache:>3}  {row['secs']:>5.0f}s"
            f"{signals}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tasks", nargs="*", help="task names (default: all)")
    parser.add_argument("--list", action="store_true", help="list the tasks and exit")
    parser.add_argument("--repeat", type=int, default=1, help="runs per task (default 1)")
    parser.add_argument("--keep", action="store_true", help="keep each run's folder, for a look inside")
    parser.add_argument("--model")
    parser.add_argument("--profile")
    parser.add_argument("--submodel")
    parser.add_argument("--persona", default="off", help="default off, so a persona can't skew results")
    parser.add_argument("--out", type=Path, help="results file (default evals/results/<time>.jsonl)")
    args = parser.parse_args(argv)
    tasks = load_tasks(args.tasks)
    if args.list:
        for task in tasks:
            print(f"{task['name']:<16} {task.get('summary', '')}")
        return 0
    kind, label = probe_sandbox()
    if kind not in ("seatbelt", "bwrap"):
        sys.exit(f"eval.py needs the OS sandbox, and bash isn't sandboxed here ({label}).")
    if args.repeat < 1:
        sys.exit("--repeat must be at least 1")
    forward = ["--persona", args.persona]
    for flag in ("model", "profile", "submodel"):
        if getattr(args, flag):
            forward += [f"--{flag}", getattr(args, flag)]
    out = args.out or EVALS / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} task{'s' if len(tasks) != 1 else ''} × {args.repeat}, results to "
          f"{out.relative_to(HERE) if HERE in out.resolve().parents else out}\n")
    rows = []
    for task in tasks:
        for n in range(1, args.repeat + 1):
            row = {"run": n, **run_one(task, forward, args.keep)}
            rows.append(row)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(describe(row))
            if not row["passed"]:
                print(f"      {row['verdict'][:300]}")
            if row.get("kept"):
                print(f"      kept: {row['kept']}")
    passed = sum(r["passed"] for r in rows)
    tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows)
    seen = sum(r["cache_seen"] for r in rows)
    signals = Counter()
    for r in rows:
        signals.update(r["signals"])
    print(f"\n{passed}/{len(rows)} passed  ·  {short_tokens(tokens)} tokens  ·  cache "
          f"{sum(r['cached_tokens'] for r in rows) / seen:.0%}" if seen else
          f"\n{passed}/{len(rows)} passed  ·  {short_tokens(tokens)} tokens  ·  cache not reported")
    if signals:
        print("harness signals: " + ", ".join(f"{k} {v}" for k, v in sorted(signals.items())))
    models = sorted({r["model"] for r in rows if r["model"]})
    if models:
        print(f"model: {', '.join(models)}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
