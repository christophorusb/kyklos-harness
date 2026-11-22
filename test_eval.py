"""Offline tests for eval.py and its fixtures: no model is called.

Run:  python -m unittest test_eval -v
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import eval as evals

HERE = Path(__file__).resolve().parent


def run_fixture_check(task: dict, solved: bool) -> tuple[int, str]:
    """Lay a task out as eval.py does, optionally with its reference solution on top, and run its
    check. Unsandboxed: these are the project's own fixtures, and the subject here is the check."""
    root = Path(tempfile.mkdtemp(prefix="kyklos-eval-test-"))
    try:
        ws = root / "ws"
        shutil.copytree(task["dir"] / "repo", ws)
        (root / "before.json").write_text(json.dumps(evals.manifest(ws)), encoding="utf-8")
        answer = ""
        if solved:
            solution = task["dir"] / "solution"
            for path in solution.rglob("*"):
                rel = path.relative_to(solution)
                if path.is_file() and rel.as_posix() != "answer.txt":
                    (ws / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(path, ws / rel)
            if (solution / "answer.txt").is_file():
                answer = (solution / "answer.txt").read_text(encoding="utf-8")
        (root / "answer.txt").write_text(answer, encoding="utf-8")
        shutil.copy(HERE / "evals" / "checklib.py", root / "checklib.py")
        shutil.copy(task["dir"] / "check.py", root / "check.py")
        proc = subprocess.run([sys.executable, "-B", str(root / "check.py"), str(root / "answer.txt"),
                               str(root / "before.json")], cwd=ws, capture_output=True, text=True, timeout=120)
        return proc.returncode, proc.stdout + proc.stderr
    finally:
        shutil.rmtree(root, ignore_errors=True)


class FixtureTests(unittest.TestCase):
    """A check that passes untouched work measures nothing, and one that fails a correct answer
    measures the check. Each task has a reference solution to hold its check to both."""

    def test_every_task_has_what_the_runner_needs(self):
        tasks = evals.load_tasks([])
        self.assertGreaterEqual(len(tasks), 8)
        for task in tasks:
            with self.subTest(task=task["name"]):
                self.assertTrue(task["prompt"].strip())
                self.assertTrue((task["dir"] / "repo").is_dir())
                self.assertTrue((task["dir"] / "check.py").is_file())
                self.assertTrue((task["dir"] / "solution").is_dir())

    def test_each_check_fails_the_untouched_repo_and_passes_the_solution(self):
        for task in evals.load_tasks([]):
            with self.subTest(task=task["name"]):
                code, out = run_fixture_check(task, solved=False)
                self.assertNotEqual(code, 0, f"untouched repo passed: {out}")
                self.assertIn("FAIL:", out)
                code, out = run_fixture_check(task, solved=True)
                self.assertEqual(code, 0, f"reference solution failed: {out}")
                self.assertIn("PASS:", out)

    def test_the_long_log_failure_is_past_the_cut(self):
        """The point of the task: the failing line must land in the part the model doesn't see."""
        from tools import MAX_OUTPUT_CHARS
        task = evals.load_tasks(["long-log"])[0]
        proc = subprocess.run([sys.executable, "-B", "run_checks.py"], cwd=task["dir"] / "repo",
                              capture_output=True, text=True, timeout=60)
        at = proc.stdout.index("FAILED")
        self.assertGreater(len(proc.stdout), MAX_OUTPUT_CHARS * 1.5)
        self.assertGreater(at, MAX_OUTPUT_CHARS // 2)
        self.assertLess(at, len(proc.stdout) - MAX_OUTPUT_CHARS // 2)

    def test_a_replace_all_fails_bump_version(self):
        task = evals.load_tasks(["bump-version"])[0]
        root = Path(tempfile.mkdtemp(prefix="kyklos-eval-test-"))
        self.addCleanup(shutil.rmtree, root, True)
        ws = root / "ws"
        shutil.copytree(task["dir"] / "solution", ws)
        log = ws / "CHANGELOG.md"
        log.write_text(log.read_text().replace("1.4.2", "1.5.0"))
        for name in ("checklib.py",):
            shutil.copy(HERE / "evals" / name, root / name)
        shutil.copy(task["dir"] / "check.py", root / "check.py")
        proc = subprocess.run([sys.executable, "-B", str(root / "check.py")], cwd=ws, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)

    def test_the_obvious_fix_fails_shared_helper(self):
        """The point of the task: formatting inside the shared helper fixes the report and breaks
        order_total, which only running the tests shows."""
        task = evals.load_tasks(["shared-helper"])[0]
        root = Path(tempfile.mkdtemp(prefix="kyklos-eval-test-"))
        self.addCleanup(shutil.rmtree, root, True)
        ws = root / "ws"
        shutil.copytree(task["dir"] / "repo", ws)
        (root / "before.json").write_text(json.dumps(evals.manifest(ws)), encoding="utf-8")
        money = ws / "money.py"
        money.write_text(money.read_text().replace("    return cents / 100", '    return f"{cents / 100:.2f}"')
                         .replace('    return "$" + str(to_dollars(cents))', '    return "$" + to_dollars(cents)'))
        shutil.copy(HERE / "evals" / "checklib.py", root / "checklib.py")
        shutil.copy(task["dir"] / "check.py", root / "check.py")
        proc = subprocess.run([sys.executable, "-B", str(root / "check.py"), str(root / "answer.txt"),
                               str(root / "before.json")], cwd=ws, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("test_order_total_is_a_number", proc.stdout + proc.stderr)
        self.assertNotIn("test_whole_and_part_dollars", proc.stdout + proc.stderr)  # the report itself is fixed

    def test_unknown_task_names_are_refused(self):
        with self.assertRaises(SystemExit):
            evals.load_tasks(["no-such-task"])


class TraceSummaryTests(unittest.TestCase):
    def test_a_trace_becomes_run_metrics(self):
        rows = [
            {"event": "session", "model": "m1"},
            {"event": "turn_start", "depth": 0},
            {"event": "model_end", "depth": 0, "purpose": "turn", "prompt_tokens": 1000, "completion_tokens": 50,
             "cached_tokens": 0},
            {"event": "tool_result", "depth": 0, "tool": "bash", "status": "exit_nonzero", "exit": 1},
            {"event": "subagent_start", "depth": 1},
            {"event": "model_end", "depth": 1, "purpose": "turn", "prompt_tokens": 400, "completion_tokens": 20},
            {"event": "tool_result", "depth": 1, "tool": "grep", "status": "error"},
            {"event": "signal", "depth": 0, "kind": "nudge", "reason": "empty"},
            {"event": "signal", "depth": 0, "kind": "parallel_batch", "size": 2},
            {"event": "model_end", "depth": 0, "purpose": "summary", "prompt_tokens": 300, "completion_tokens": 30,
             "cached_tokens": 200},
            {"event": "compact", "depth": 0, "stage": "summarized"},
            {"event": "model_end", "depth": 0, "error": "APIConnectionError"},
            {"event": "turn_end", "depth": 1, "outcome": "answer"},
            {"event": "turn_end", "depth": 0, "outcome": "step_limit", "checked": "unchecked"},
        ]
        path = Path(tempfile.mkdtemp()) / "trace.jsonl"
        self.addCleanup(shutil.rmtree, path.parent, True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
        m = evals.summarize_trace(path)
        self.assertEqual((m["model"], m["steps"], m["sub_steps"], m["subagents"]), ("m1", 1, 1, 1))
        self.assertEqual((m["prompt_tokens"], m["completion_tokens"]), (1700, 100))
        self.assertEqual((m["cached_tokens"], m["cache_seen"]), (200, 1300))
        self.assertEqual((m["tools"], m["tool_failures"]), (2, 1))  # a failing command is not a failed tool
        self.assertEqual((m["api_errors"], m["compactions"], m["outcome"]), (1, 1, "step_limit"))
        self.assertEqual(m["signals"], {"nudge:empty": 1, "parallel_batch": 1})
        self.assertEqual(m["checked"], "unchecked")

    def test_a_missing_trace_is_empty_not_an_error(self):
        m = evals.summarize_trace(Path(tempfile.gettempdir()) / "no-such-trace.jsonl")
        self.assertEqual((m["steps"], m["outcome"], m["signals"]), (0, None, {}))


class EvalFlagTests(unittest.TestCase):
    def test_kyklos_refuses_eval_outside_an_eval_folder_before_anything_else(self):
        env = {k: v for k, v in os.environ.items() if k != "KYKLOS_EVAL_ROOT"}
        env["KYKLOS_HOME"] = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, env["KYKLOS_HOME"], True)
        proc = subprocess.run([sys.executable, str(HERE / "kyklos.py"), "--eval", "do something"],
                              cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--eval is for eval.py", proc.stderr)
        self.assertEqual(os.listdir(env["KYKLOS_HOME"]), [])  # refused before it wrote anything


if __name__ == "__main__":
    unittest.main()
