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
import textwrap
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


def command_touches_secret(command: str) -> bool:
    """True if any token in a shell command looks like a secret file path (e.g. .env, *.pem, id_rsa)."""
    for token in re.split(r'[\s;|&><=,"\'()`]+', command):
        cleaned = token.rstrip("*?")
        if cleaned and is_secret_file(Path(cleaned)):
            return True
    return False


def _base_indent(s: str) -> str:
    """Leading whitespace stripped by textwrap.dedent."""
    for raw, dedented in zip(s.splitlines(), textwrap.dedent(s).splitlines()):
        if dedented.strip():
            return raw[: len(raw) - len(dedented)]
    return ""


def _match_edit(text: str, old_string: str, new_string: str) -> tuple[str, str]:
    """Find old_string in text (exact, trailing-space-tolerant, or indent-shifted).
    Returns (matched_slice_in_text, adjusted_new_string) or raises ToolError."""
    if not old_string:
        raise ToolError("old_string must not be empty. Use write_file to create or overwrite a file.")
    if "\r\n" not in text:
        old_string = old_string.replace("\r\n", "\n")
        new_string = new_string.replace("\r\n", "\n")

    count = text.count(old_string)
    if count == 1:
        return old_string, new_string
    if count > 1:
        raise ToolError(f"old_string matches {count} places. Include more surrounding lines to make it unique.")

    file_lines = text.splitlines(keepends=True)
    old_lines = old_string.splitlines()
    n = len(old_lines)
    if n == 0 or n > len(file_lines):
        raise ToolError("old_string not found. Re-read the file and copy the text exactly.")

    def span_at(i: int) -> str:
        span = "".join(file_lines[i : i + n])
        if not old_string.endswith("\n") and span.endswith("\n"):
            span = span[:-2] if span.endswith("\r\n") else span[:-1]
        return span

    # Fallback 1: ignore trailing whitespace on each line.
    target_rstrip = [l.rstrip() for l in old_lines]
    hits = [i for i in range(len(file_lines) - n + 1)
            if [l.rstrip() for l in file_lines[i : i + n]] == target_rstrip]
    if len(hits) > 1:
        raise ToolError(f"old_string matches {len(hits)} places. Include more surrounding lines to make it unique.")
    if len(hits) == 1:
        return span_at(hits[0]), new_string

    # Fallback 2: ignore uniform leading indentation offset (re-indenting new_string to match the file).
    target_dedent = [l.rstrip() for l in textwrap.dedent(old_string).splitlines()]
    if any(target_dedent):
        hits = [i for i in range(len(file_lines) - n + 1)
                if [l.rstrip() for l in textwrap.dedent("".join(file_lines[i : i + n])).splitlines()] == target_dedent]
        if len(hits) > 1:
            raise ToolError(f"old_string matches {len(hits)} places. Include more surrounding lines to make it unique.")
        if len(hits) == 1:
            span = span_at(hits[0])
            old_ind, file_ind = _base_indent(old_string), _base_indent(span)
            reindented = []
            for line in new_string.splitlines(keepends=True):
                if not line.strip():
                    reindented.append(line)
                elif line.startswith(old_ind):
                    reindented.append(file_ind + line[len(old_ind):])
                else:
                    reindented.append(file_ind + line.lstrip(" \t"))
            return span, "".join(reindented)

    raise ToolError("old_string not found. Re-read the file and copy the text exactly.")


MAX_BRACE_EXPANSIONS = 64


def expand_braces(pattern: str) -> list[str]:
    """Shell-style {a,b} alternatives, which Python's glob doesn't support: '*.{py,md}' -> ['*.py', '*.md'].
    Handles several and nested groups; a group without a comma (like '{x}') stays literal."""
    depth, start = 0, None
    for i, ch in enumerate(pattern):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                inner, alternatives, level, piece = pattern[start + 1:i], [], 0, ""
                for c in inner:  # split on commas that aren't inside a nested group
                    level += c == "{"
                    level -= c == "}"
                    if c == "," and level == 0:
                        alternatives.append(piece)
                        piece = ""
                    else:
                        piece += c
                if not alternatives:
                    continue  # no comma: a literal brace group, keep scanning
                alternatives.append(piece)
                prefix, suffix = pattern[:start], pattern[i + 1:]
                out = []
                for alt in alternatives:
                    for expanded in expand_braces(prefix + alt + suffix):
                        if expanded not in out:
                            out.append(expanded)
                        if len(out) >= MAX_BRACE_EXPANSIONS:
                            return out
                return out
    return [pattern]


class ToolError(Exception):
    """An error message meant for the model, not a crash."""


class Tools:
    def __init__(self, workspace: Path, before_write=None):
        self.workspace = workspace.resolve()
        self.before_write = before_write

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
        if is_secret_file(p):
            raise ToolError(f"Refusing to write {path}: it likely contains secrets. Ask the user to change it.")
        if self.before_write:
            self.before_write(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"

    def _prepare_edit(self, path: str, old_string: str, new_string: str) -> tuple[Path, str, str, str]:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"File not found: {path}")
        if is_secret_file(p):  # edit errors echo nothing, but keep secrets files hands-off entirely
            raise ToolError(f"Refusing to edit {path}: it likely contains secrets. Ask the user to change it.")
        text = p.read_text(encoding="utf-8")
        matched_old, adjusted_new = _match_edit(text, old_string, new_string)
        return p, text, matched_old, adjusted_new

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        p, text, matched_old, adjusted_new = self._prepare_edit(path, old_string, new_string)
        if self.before_write:
            self.before_write(p)
        p.write_text(text.replace(matched_old, adjusted_new, 1), encoding="utf-8")
        return f"Edited {path}"

    def check(self, name: str, args: dict) -> None:
        """Validate a write/edit call before asking the user for permission."""
        if name == "write_file":
            p = self._resolve(args["path"])
            if is_secret_file(p):
                raise ToolError(f"Refusing to write {args['path']}: it likely contains secrets. Ask the user to change it.")
        elif name == "edit_file":
            self._prepare_edit(args["path"], args["old_string"], args["new_string"])

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
        globs = expand_braces(glob) if glob else []
        for f in self._walk_files(root):
            if (globs and not any(f.match(g) for g in globs)) or is_secret_file(f):
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
        patterns = expand_braces(pattern)
        if any(Path(p).is_absolute() or ".." in Path(p).parts for p in patterns):
            raise ToolError("Pattern must be relative to the workspace, without '..'")
        files = []
        matches = {m for p in patterns for m in self.workspace.glob(p)}
        for p in sorted(matches):
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
