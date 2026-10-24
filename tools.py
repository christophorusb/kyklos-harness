"""Tools the model can call: read_file, grep, glob, write_file, edit_file, bash.

Each tool is a JSON schema (sent to the model) plus a Python function
(run by the harness). Paths are resolved against the workspace root and
may not escape it.
"""

import os
import platform
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

MAX_OUTPUT_CHARS = 20_000  # keep tool results from flooding the context window
MAX_SEARCH_RESULTS = 200
MAX_SEARCH_FILE_BYTES = 2_000_000
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}

IS_WINDOWS = platform.system() == "Windows"
SHELL_NAME = "PowerShell" if IS_WINDOWS else "bash"


PYTHON_CANDIDATES = ["py", "python", "python3"]
OTHER_CANDIDATES = ["node", "npm", "git", "uv"]


@lru_cache(maxsize=1)
def probe_commands() -> tuple[dict, list]:
    """Find which common commands actually run here, with their versions.

    Running `--version` (not just checking PATH) catches stubs like the
    Windows Store `python.exe`, which exists but fails.
    Returns ({command: version}, [python commands that don't work]).
    """
    def version_of(cmd):
        path = shutil.which(cmd)
        if not path:
            return None
        try:
            r = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        out = (r.stdout or r.stderr).strip().splitlines()
        return out[0] if r.returncode == 0 and out else None

    candidates = PYTHON_CANDIDATES + OTHER_CANDIDATES
    with ThreadPoolExecutor() as pool:  # probes run in parallel; npm alone takes ~0.8s
        versions = dict(zip(candidates, pool.map(version_of, candidates)))

    working, broken_python = {}, []
    for cmd, version in versions.items():
        if version:
            working[cmd] = version
        elif cmd in PYTHON_CANDIDATES:
            broken_python.append(cmd)
    return working, broken_python


SAFE_ENV_SUFFIXES = (".example", ".sample", ".template")


def is_secret_file(p: Path) -> bool:
    """Files that usually hold API keys or private keys. The agent may not read their contents."""
    name = p.name.lower()
    if name == ".env" or (name.startswith(".env.") and not name.endswith(SAFE_ENV_SUFFIXES)):
        return True
    return p.suffix.lower() in {".pem", ".key", ".p12", ".pfx"} or name.startswith(("id_rsa", "id_ed25519", "id_ecdsa"))


class ToolError(Exception):
    """An error message meant for the model, not a crash."""


class Tools:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    # ---- helpers -----------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        p = (self.workspace / path).resolve()
        if p != self.workspace and self.workspace not in p.parents:
            raise ToolError(f"Path {path!r} is outside the workspace {self.workspace}")
        return p

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        half = MAX_OUTPUT_CHARS // 2
        cut = len(text) - MAX_OUTPUT_CHARS
        return f"{text[:half]}\n\n... [{cut} chars truncated] ...\n\n{text[-half:]}"

    # ---- tools -------------------------------------------------------------

    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"File not found: {path}")
        if is_secret_file(p):
            raise ToolError(f"Refusing to read {path}: it likely contains secrets such as API keys. "
                            "If you need to know what settings exist, read the .env.example file instead.")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(offset, 1) - 1
        chunk = lines[start : start + limit]
        numbered = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(chunk, start + 1))
        if start + limit < len(lines):
            numbered += f"\n... ({len(lines) - start - limit} more lines; use offset to read on)"
        return self._truncate(numbered or "(empty file)")

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"File not found: {path}")
        if is_secret_file(p):  # edit errors echo nothing, but keep secrets files hands-off entirely
            raise ToolError(f"Refusing to edit {path}: it likely contains secrets. Ask the user to change it.")
        text = p.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            raise ToolError("old_string not found. Re-read the file and copy the text exactly.")
        if count > 1:
            raise ToolError(f"old_string matches {count} places. Include more surrounding lines to make it unique.")
        p.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        return f"Edited {path}"

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.workspace).as_posix()

    def _walk_files(self, root: Path):
        """Yield files under root, skipping dependency and cache folders."""
        if root.is_file():
            yield root
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                yield Path(dirpath) / name

    def grep(self, pattern: str, path: str = ".", glob: str = None, ignore_case: bool = False) -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            raise ToolError(f"Invalid regex: {e}")
        root = self._resolve(path)
        if not root.exists():
            raise ToolError(f"Path not found: {path}")
        hits = []
        for f in self._walk_files(root):
            if (glob and not f.match(glob)) or is_secret_file(f):
                continue
            try:
                if f.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                data = f.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:8192]:  # binary file
                continue
            for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{self._rel(f)}:{n}: {line.strip()[:300]}")
                    if len(hits) >= MAX_SEARCH_RESULTS:
                        hits.append(f"... (stopped at {MAX_SEARCH_RESULTS} matches; narrow the pattern or path)")
                        return self._truncate("\n".join(hits))
        return self._truncate("\n".join(hits) or "No matches.")

    def glob(self, pattern: str) -> str:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ToolError("Pattern must be relative to the workspace, without '..'")
        files = []
        for p in sorted(self.workspace.glob(pattern)):
            if not p.is_file() or SKIP_DIRS.intersection(p.relative_to(self.workspace).parts):
                continue
            files.append(self._rel(p))
            if len(files) >= MAX_SEARCH_RESULTS:
                files.append(f"... (stopped at {MAX_SEARCH_RESULTS} files; narrow the pattern)")
                break
        return "\n".join(files) or "No files matched."

    def bash(self, command: str, timeout: int = 120) -> str:
        argv = ["powershell", "-NoProfile", "-Command", command] if IS_WINDOWS else ["bash", "-c", command]
        try:
            r = subprocess.run(
                argv, cwd=self.workspace, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"Command timed out after {timeout}s")
        out = (r.stdout or "") + (r.stderr or "")
        return self._truncate(f"exit code {r.returncode}\n{out}".rstrip())


# Tools that run without asking. Everything else goes through the permission prompt.
SAFE_TOOLS = {"read_file", "grep", "glob"}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file. Returns lines prefixed with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace"},
                    "offset": {"type": "integer", "description": "1-based line to start from"},
                    "limit": {"type": "integer", "description": "Max lines to return"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents with a regular expression (Python re syntax). "
                "Returns matching lines as path:line: text. Skips .git, .venv, node_modules and binary files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression"},
                    "path": {"type": "string", "description": "File or folder to search, default the workspace"},
                    "glob": {"type": "string", "description": "Only search files matching this, e.g. *.py"},
                    "ignore_case": {"type": "boolean"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by name pattern, e.g. **/*.py or src/**/test_*.js. Returns workspace-relative paths.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a file or overwrite it entirely. Prefer edit_file for changes to existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace one exact occurrence of old_string with new_string. "
                "old_string must match the file exactly (without line-number prefixes) and be unique."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": f"Run a {SHELL_NAME} command in the workspace directory. Returns exit code and output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Seconds, default 120"},
                },
                "required": ["command"],
            },
        },
    },
]
