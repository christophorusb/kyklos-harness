"""Tools the model can call: read_file, grep, glob, write_file, edit_file, bash.

Each tool is a JSON schema (sent to the model) plus a Python function
(run by the harness). Paths are resolved against the workspace root and
may not escape it.
"""

import os
import platform
import ast
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
MAX_PLAN_STEPS = 12  # a plan longer than this is a transcript, not a shape the user can hold
# A step is model-written text that goes straight to the terminal, where an escape sequence is an
# instruction: recolour the screen, set the window title, or leave an unterminated OSC that eats
# whatever prints next. Rich drops a bare BEL but passes the rest through, so strip them here --
# at the one place that decides what a step says, rather than at each place that draws it.
# Stripping the ESC byte alone would leave "[31m" behind as literal text: harmless, but the step
# then reads as rubbish. Match whole sequences -- CSI, OSC (terminated or not), and the two-byte
# forms -- then fold anything still in the control range.
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]"                  # CSI: colours, cursor moves
                            r"|\x1b[\]PX^_][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC / DCS / APC: clipboard, links, title
                            r"|\x1b[@-_]")                                # anything else introduced by ESC
CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
UNSAFE_TERMINAL_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]+")


def plain_line(text: str) -> str:
    """One line fit to print: escape sequences removed, control characters and runs of whitespace
    folded to single spaces. A step has to survive this with something left to say."""
    return " ".join(CONTROL_CHARS_RE.sub(" ", ANSI_ESCAPE_RE.sub("", text)).split())


def sanitize_terminal_text(text: str) -> str:
    """Multiline text safe to draw in a terminal: strips CSI/OSC escapes (including OSC 52 clipboard
    writes and OSC 8 deceptive links) and unsafe control bytes while keeping newlines and tabs.
    Bare CR becomes a newline so `evil\\rbenign` cannot visually overwrite itself on screen."""
    cleaned = ANSI_ESCAPE_RE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    return UNSAFE_TERMINAL_CTRL_RE.sub("", cleaned)


SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
SHELL_NAME = "PowerShell" if IS_WINDOWS else "bash"

# ---- secrets in the environment and in output (SEC-01, SEC-04) -----------------------------
# Commands must not inherit Kyklos's own settings (its API key) or anything named like a secret,
# and secret values must not reach the model or the screen through tool output.
SECRET_ENV_NAME_RE = re.compile(
    r"(^|_)(API_?|ACCESS_?|SECRET_?|PRIVATE_?)?KEY($|_)|TOKEN|SECRET|PASSW(OR)?D|CREDENTIAL", re.IGNORECASE)
HIDDEN_ENV_NAMES: set[str] = set()  # names from Kyklos's .env; filled by hide_env_names() at startup
MIN_SECRET_VALUE_LEN = 8  # shorter values ("true", "1") would redact ordinary text
REDACTED = "[REDACTED]"
SECRET_VALUE_PATTERNS = [
    re.compile(r"\bsk-(?:proj-|ant-|or-|live-|test-)?[A-Za-z0-9_-]{20,}"),     # OpenAI, Anthropic, OpenRouter
    re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{20,}\b"),                        # Stripe live keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),                               # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"),                             # GitHub fine-grained tokens
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),                              # AWS access key ids
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),                             # Slack tokens
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),                                  # Google API keys
    re.compile(r"\bnpm_[A-Za-z0-9]{36,}\b"),                                   # npm access tokens
    re.compile(r"\bpypi-[A-Za-z0-9_-]{32,}\b"),                                # PyPI API tokens
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),                                    # HuggingFace tokens
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),  # JWTs
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"),
]
URI_PASSWORD_RE = re.compile(
    r"\b((?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|amqp|amqps)://[^/\s:@]*:)([^@\s/]{4,})(@)"
)


def kill_process_tree(process: subprocess.Popen) -> None:
    """Stop a command and everything it started (SEC-06). Killing only the shell leaves its children,
    like a dev server or a Start-Process, running on their own."""
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, timeout=15)
        else:
            import signal
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    if process.poll() is None:
        process.kill()


def hide_env_names(names) -> None:
    """Keep these variables (Kyklos's own .env settings) out of every command's environment."""
    HIDDEN_ENV_NAMES.update(n.upper() for n in names)


def is_secret_env_name(name: str) -> bool:
    return bool(SECRET_ENV_NAME_RE.search(name))


def child_env(base=None) -> dict:
    """The environment for commands: everything except Kyklos's .env settings and secret-named variables.
    KYKLOS_PASS_ENV=GITHUB_TOKEN,NPM_TOKEN lets chosen ones through on purpose."""
    base = os.environ if base is None else base
    passthrough = {n.strip().upper() for n in base.get("KYKLOS_PASS_ENV", "").split(",") if n.strip()}
    env = {k: v for k, v in base.items()
           if k.upper() in passthrough or (k.upper() not in HIDDEN_ENV_NAMES and not is_secret_env_name(k))}
    env["GIT_TERMINAL_PROMPT"] = "0"  # git fails instead of asking for credentials
    return env


def known_secret_values(base=None) -> list[str]:
    """Values of secret-named variables (like Kyklos's API_KEY), longest first, so they can be redacted
    exactly, whatever their format."""
    base = os.environ if base is None else base
    values = {v for k, v in base.items() if is_secret_env_name(k) and len(v) >= MIN_SECRET_VALUE_LEN}
    return sorted(values, key=len, reverse=True)


def workspace_secret_values(workspace: Path | None) -> list[str]:
    """Secret values defined in the workspace's own `.env*` files, read into harness memory purely for
    output redaction (never placed in os.environ, child_env, or any prompt). Catching custom passwords
    here prevents a noisy test failure from leaking a project secret that matches no generic regex."""
    if workspace is None or not workspace.is_dir():
        return []
    candidates = [workspace]
    for sub in ("config", "env", ".config"):
        d = workspace / sub
        if d.is_dir():
            candidates.append(d)
    values: set[str] = set()
    for folder in candidates:
        try:
            entries = list(folder.iterdir())
        except OSError:
            continue
        for entry in entries:
            name = entry.name.lower()
            if not (name == ".env" or (name.startswith(".env.") and not name.endswith(SAFE_ENV_SUFFIXES))):
                continue
            try:
                if not entry.is_file() or entry.stat().st_size > 256_000:
                    continue
                lines = entry.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for raw in lines:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if is_secret_env_name(key) and len(val) >= MIN_SECRET_VALUE_LEN:
                    values.add(val)
                for m in URI_PASSWORD_RE.finditer(val):
                    if len(m.group(2)) >= 4 and m.group(2) != REDACTED:
                        values.add(m.group(2))
    return sorted(values, key=len, reverse=True)


def scrub_secrets(text: str, values=None, workspace: Path | None = None) -> tuple[str, int]:
    """(text with secrets replaced by [REDACTED], how many were replaced): known secret values exactly
    (from os.environ and any workspace .env* files), plus common token and credential-URI formats."""
    count = 0
    if values is None:
        combined = set(known_secret_values()) | set(workspace_secret_values(workspace))
        ordered = sorted(combined, key=len, reverse=True)
    else:
        ordered = list(values)
    for value in ordered:
        if value in text:
            count += text.count(value)
            text = text.replace(value, REDACTED)
    for pattern in SECRET_VALUE_PATTERNS:
        text, n = pattern.subn(REDACTED, text)
        count += n

    def _redact_uri(match: re.Match) -> str:
        nonlocal count
        if match.group(2) == REDACTED:
            return match.group(0)
        count += 1
        return f"{match.group(1)}{REDACTED}{match.group(3)}"

    text = URI_PASSWORD_RE.sub(_redact_uri, text)
    return text, count


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
SECRET_EXACT_NAMES = {
    ".env", ".npmrc", ".pypirc", ".netrc", ".git-credentials", ".pgpass", ".my.cnf",
    "credentials", "credentials.json", "kubeconfig", "master.passwd", "shadow", "sudoers",
}
SECRET_PREFIXES = ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", "service-account")
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".kdbx", ".ovpn",
                   ".tfvars", ".tfstate", ".tfstate.backup")


def is_secret_file(p: Path) -> bool:
    """Files that usually hold API keys, tokens, or private keys. The agent may not read their contents."""
    name = p.name.lower()
    if name.endswith(SAFE_ENV_SUFFIXES):
        return False
    if name in SECRET_EXACT_NAMES or name.startswith(".env."):
        return True
    return name.endswith(SECRET_SUFFIXES) or name.startswith(SECRET_PREFIXES)


SECRET_GLOB_TARGETS = (
    ".env", ".env.local", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    "secret.pem", "secret.key", "secret.p12", "secret.pfx", "secret.kdbx",
    "secret.keystore", "secret.jks", "secret.ovpn", "prod.tfvars", "terraform.tfstate",
    ".npmrc", ".pypirc", ".netrc", ".git-credentials", ".pgpass", "kubeconfig",
    "credentials.json", "service-account.json",
)


def command_touches_secret(command: str) -> bool:
    """True if any token in a shell command looks like a secret file path or a glob matching one
    (e.g. .env, *.env, *env, .e?v, *.pem, *id_rsa*)."""
    import fnmatch
    for token in re.split(r'[\s;|&><=,"\'()`]+', command):
        if not token:
            continue
        cleaned = token.rstrip("*?")
        if cleaned and is_secret_file(Path(cleaned)):
            return True
        name = Path(token).name.lower()
        stripped = name.strip("*?")
        if stripped and is_secret_file(Path(stripped)):
            return True
        if any(c in name for c in "*?[") and not name.endswith(SAFE_ENV_SUFFIXES):
            alnum = sum(c.isalnum() for c in re.sub(r"\[[^\]]*\]", lambda m: m.group(0)[1:2], name))
            if (alnum >= 3 or ("." in name and alnum >= 2)) and \
                    any(fnmatch.fnmatchcase(t, name) for t in SECRET_GLOB_TARGETS):
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


# Commands a model reaches for that the machine may not have. Two different gaps, one per family:
# PowerShell has no answer for the unix text tools, and a BSD userland like macOS ships none of the
# GNU coreutils extras that a model trained mostly on Linux assumes are simply there.
UNIX_TOOLS = ["grep", "sed", "awk", "head", "tail"]
GNU_TOOLS = ["timeout", "tac", "nproc", "sha256sum", "md5sum", "stdbuf"]

MISSING_TOOL_ADVICE = (
    "Instead use the grep tool or Select-String, Get-Content -TotalCount / -Tail, and the "
    "-replace operator." if IS_WINDOWS else
    "This is a BSD userland, not GNU. To bound a command that might hang, pass the bash tool's "
    "own timeout argument rather than reaching for a timeout command."
)


@lru_cache(maxsize=1)
def missing_unix_tools() -> list[str]:
    """Commands a model is likely to reach for that aren't on this machine.

    `timeout` is the one that costs real time. Wrapping a command that might hang is exactly the
    right instinct, it is the first thing a model reaches for, and on macOS it isn't there -- so
    the turn is spent on `command not found` with nothing in the prompt to explain why. Checked
    rather than assumed, since a machine with coreutils installed has them all.
    """
    return [t for t in (UNIX_TOOLS if IS_WINDOWS else GNU_TOOLS) if not shutil.which(t)]


class ToolError(Exception):
    """An error message meant for the model, not a crash."""


# A disposable corner of the workspace for one-off diagnostic scripts. It has to be *inside* the
# workspace: a script in /tmp can't resolve `./src/...` or the project's dependencies, and
# _resolve() refuses to write outside the workspace at all, so the model's only way out was
# `cat > /tmp/x.mjs <<'EOF'` through bash — two permission prompts and a heredoc, to run five lines.
SCRATCH_DIR = ".kyklos/scratch"


def scratch_dir(workspace: Path) -> Path:
    # Resolved on both sides: on macOS /var is a symlink to /private/var, so comparing an
    # unresolved workspace against a resolved path would never match.
    return workspace.resolve() / ".kyklos" / "scratch"


def ensure_scratch(workspace: Path) -> Path:
    """Create the scratch directory, ignoring itself rather than editing the project's .gitignore:
    `.kyklos/.gitignore` holding `*` keeps the whole thing out of git and touches nothing of yours."""
    directory = scratch_dir(workspace)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory.parent / ".gitignore"
    if not marker.exists():
        marker.write_text("*\n", encoding="utf-8")
    return directory


def is_scratch_path(workspace: Path, path: str) -> bool:
    """True if `path` lands inside this workspace's scratch directory."""
    try:
        resolved = (workspace / path).resolve()
    except (OSError, ValueError):
        return False
    root = scratch_dir(workspace)
    return resolved == root or root in resolved.parents


# ---- repository map ---------------------------------------------------------------------------
# What each file defines, so turn one doesn't start blind and spend itself on glob and grep.
#
# Names only — never signatures, never line numbers. A regex reading code gets a name right and a
# parameter list wrong, and a map that misstates a signature is worse than no map, because the model
# believes it. Line numbers are the first thing to go stale after an edit. A file path and a symbol
# name are cheap to state correctly and cheap for the model to verify by reading.
MAP_TOKEN_BUDGET = 1200
MAP_CHARS_PER_FILE = 260
MAP_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".go", ".rs"}
MAP_NAME_RES = [
    re.compile(r"^\s*export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)", re.M),
    re.compile(r"^\s*export\s+(?:abstract\s+)?class\s+(\w+)", re.M),
    re.compile(r"^\s*export\s+(?:const|let|var)\s+(\w+)", re.M),
    re.compile(r"^\s*export\s+(?:type|interface|enum)\s+(\w+)", re.M),
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Z]\w*)", re.M),   # Go, exported
    re.compile(r"^\s*type\s+([A-Z]\w*)\s", re.M),                  # Go
    re.compile(r"^\s*pub\s+(?:async\s+)?fn\s+(\w+)", re.M),        # Rust
    re.compile(r"^\s*pub\s+(?:struct|enum|trait)\s+(\w+)", re.M),  # Rust
]


def _python_names(source: str) -> list[str]:
    """Top-level definitions, via the real parser rather than a regex."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and not n.name.startswith("_")]
            names.append(f"{node.name}({', '.join(methods[:10])})" if methods else node.name)
    return names


def _declared_names(source: str) -> list[str]:
    return [m for pattern in MAP_NAME_RES for m in pattern.findall(source)]


def repository_map(workspace: Path, budget_tokens: int = MAP_TOKEN_BUDGET) -> str:
    """A sketch of the workspace: which source files exist and what each one defines."""
    rows = []
    # Shallow paths first: when the budget runs out, top-level modules are likelier to be the
    # entry points than something eight directories down, and alphabetical order would be arbitrary.
    for path in sorted(workspace.rglob("*"), key=lambda p: (len(p.relative_to(workspace).parts), p)):
        if path.suffix not in MAP_SUFFIXES or not path.is_file():
            continue
        parts = path.relative_to(workspace).parts
        if SKIP_DIRS.intersection(parts) or any(p.startswith(".") for p in parts[:-1]):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        names = _python_names(source) if path.suffix == ".py" else _declared_names(source)
        names = list(dict.fromkeys(names))
        if not names:
            continue
        # Cap by rendered length, not by count: one entry can be a class that expands to ten method
        # names, so counting entries lets a file of classes spend the whole budget on its own.
        # Knowing which file to open matters more than knowing everything inside one of them.
        shown, used = [], 0
        for name in names:
            if used + len(name) + 2 > MAP_CHARS_PER_FILE:
                break
            shown.append(name)
            used += len(name) + 2
        extra = f", +{len(names) - len(shown)} more" if len(names) > len(shown) else ""
        rows.append(f"{path.relative_to(workspace).as_posix()}: {', '.join(shown) or '…'}{extra}")
    if not rows:
        return ""
    kept, used = [], 0
    for i, row in enumerate(rows):
        if used + len(row) // 4 > budget_tokens:
            kept.append(f"... and {len(rows) - i} more files")
            break
        kept.append(row)
        used += len(row) // 4
    return "\n".join(kept)


# ---- OS-level kernel sandboxing for bash (macOS Seatbelt / Linux bubblewrap) ----------------
# Regex parsing is an aid to reading, never a security boundary. Where the OS gives us a real
# kernel boundary without root or third-party dependencies, wrap child commands in it:
#   - writes confined to the workspace (excluding .git/hooks), /tmp, and standard caches
#   - reads blocked on credential stores (~/.ssh, ~/.aws, Keychains, browser profiles, ~/.kyklos)
#   - network denied by default unless the user explicitly approved a network/installer command

SANDBOX_PROTECTED_HOME_DIRS = (
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".kyklos",
    "Library/Keychains", "Library/Cookies", "Library/Safari", "Library/Mail", "Library/Messages",
    "Library/LaunchAgents",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox",
    "Library/Application Support/Arc",
    "Library/Application Support/BraveSoftware",
)
SANDBOX_PROTECTED_HOME_FILES = (
    ".netrc", ".npmrc", ".pypirc", ".git-credentials", ".pgpass", ".my.cnf",
    ".zsh_history", ".bash_history",
)


def _sbpl_quote(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _canonical_variants(path: Path) -> list[str]:
    """Both the literal path and its symlink-resolved counterpart (e.g. /tmp and /private/tmp)."""
    out = [str(path)]
    try:
        resolved = str(path.resolve())
        if resolved not in out:
            out.append(resolved)
    except OSError:
        pass
    return out


def _workspace_inside(workspace: Path, target: Path) -> bool:
    try:
        ws = workspace.resolve()
        tg = target.resolve()
        return ws == tg or tg in ws.parents
    except OSError:
        return False


def seatbelt_profile(workspace: Path, allow_network: bool = False, home: Path | None = None) -> str:
    """macOS sandbox-exec (SBPL v1) profile confining writes, credential reads, and network."""
    ws = workspace.resolve()
    home_dir = (home or Path.home()).resolve()
    writable_subpaths: list[str] = []
    for p in (ws, Path("/tmp"), Path("/private/tmp"), Path("/var/tmp"), Path("/private/var/tmp"),
              Path("/var/folders"), Path("/private/var/folders"),
              home_dir / ".cache", home_dir / "Library" / "Caches"):
        for v in _canonical_variants(p):
            if v not in writable_subpaths:
                writable_subpaths.append(v)

    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
        "(allow file-write*",
    ]
    for sub in writable_subpaths:
        lines.append(f"  (subpath {_sbpl_quote(sub)})")
    for dev in ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom", "/dev/tty", "/dev/dtracehelper"):
        lines.append(f"  (literal {_sbpl_quote(dev)})")
    lines.append('  (subpath "/dev/fd"))')

    # Keep git hooks and repo persistence targets kernel-read-only inside bash.
    for rel_ro in (".git/hooks", ".husky", ".vscode"):
        for v in _canonical_variants(ws / rel_ro):
            lines.append(f"(deny file-write* (subpath {_sbpl_quote(v)}))")
    for rel_file in ("AGENTS.md", ".gitmodules", ".pre-commit-config.yaml", ".pre-commit-config.yml"):
        for v in _canonical_variants(ws / rel_file):
            lines.append(f"(deny file-write* (literal {_sbpl_quote(v)}))")

    deny_reads: list[str] = []
    for rel_dir in SANDBOX_PROTECTED_HOME_DIRS:
        target = home_dir / rel_dir
        if not _workspace_inside(ws, target):
            for v in _canonical_variants(target):
                deny_reads.append(f"  (subpath {_sbpl_quote(v)})")
    for rel_file in SANDBOX_PROTECTED_HOME_FILES:
        target = home_dir / rel_file
        for v in _canonical_variants(target):
            deny_reads.append(f"  (literal {_sbpl_quote(v)})")
    for sys_path, kind in (("/etc/sudoers", "literal"), ("/private/etc/sudoers", "literal"),
                           ("/etc/sudoers.d", "subpath"), ("/private/etc/sudoers.d", "subpath")):
        deny_reads.append(f"  ({kind} {_sbpl_quote(sys_path)})")
    if deny_reads:
        lines.append("(deny file-read*")
        lines.extend(deny_reads)
        lines.append(")")

    if not allow_network:
        lines.extend([
            "(deny network*)",
            "(allow network* (local unix-socket))",
            "(allow network-outbound (remote unix-socket))",
        ])
    return "\n".join(lines)


def bwrap_argv(inner_argv: list[str], workspace: Path, allow_network: bool = False,
               home: Path | None = None, bwrap_bin: str = "bwrap") -> list[str]:
    """Linux bubblewrap command line confining writes, credential reads, and network."""
    ws = workspace.resolve()
    home_dir = (home or Path.home()).resolve()
    cmd = [
        bwrap_bin, "--die-with-parent",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
    ]
    for cache_dir in (home_dir / ".cache",):
        if cache_dir.is_dir():
            cmd += ["--bind", str(cache_dir), str(cache_dir)]
    cmd += ["--bind", str(ws), str(ws)]
    for rel_ro in (".git/hooks", ".husky", ".vscode"):
        ro_dir = ws / rel_ro
        if ro_dir.is_dir():
            cmd += ["--ro-bind", str(ro_dir), str(ro_dir)]
    for rel_file in ("AGENTS.md", ".gitmodules", ".pre-commit-config.yaml", ".pre-commit-config.yml"):
        ro_file = ws / rel_file
        if ro_file.is_file():
            cmd += ["--ro-bind", str(ro_file), str(ro_file)]
    for rel_dir in SANDBOX_PROTECTED_HOME_DIRS:
        target = home_dir / rel_dir
        if target.is_dir() and not _workspace_inside(ws, target):
            cmd += ["--tmpfs", str(target)]
    for rel_file in SANDBOX_PROTECTED_HOME_FILES:
        target = home_dir / rel_file
        if target.is_file():
            cmd += ["--ro-bind", "/dev/null", str(target)]
    if not allow_network:
        cmd.append("--unshare-net")
    cmd.append("--")
    cmd.extend(inner_argv)
    return cmd


@lru_cache(maxsize=1)
def _probe_host_sandbox() -> tuple[str, str]:
    """Detect once whether macOS Seatbelt or Linux bubblewrap can initialize here."""
    if IS_MACOS:
        exe = shutil.which("sandbox-exec")
        if not exe:
            return "unavailable", "sandbox-exec not found"
        true_bin = shutil.which("true") or "/usr/bin/true"
        try:
            r = subprocess.run([exe, "-p", "(version 1)(allow default)", true_bin],
                               capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return "unavailable", "sandbox-exec failed"
        if r.returncode == 0:
            return "seatbelt", "macOS seatbelt"
        err = (r.stderr or "") + (r.stdout or "")
        if "Operation not permitted" in err or "sandbox_init" in err:
            return "outer", "outer OS sandbox active"
        return "unavailable", "sandbox-exec unavailable"
    if IS_LINUX:
        exe = shutil.which("bwrap")
        if not exe:
            return "unavailable", "bwrap not installed"
        true_bin = shutil.which("true") or "/bin/true"
        try:
            r = subprocess.run([exe, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", true_bin],
                               capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return "unavailable", "bwrap failed"
        if r.returncode == 0:
            return "bwrap", "Linux bubblewrap"
        return "unavailable", "bwrap unavailable"
    return "unavailable", "no native OS sandbox"


def probe_sandbox(env=None) -> tuple[str, str]:
    """(kind, label) for bash sandboxing: 'seatbelt', 'bwrap', 'outer', 'off', or 'unavailable'."""
    cfg = ((os.environ if env is None else env).get("KYKLOS_SANDBOX") or "auto").strip().lower()
    if cfg in {"0", "off", "false", "none", "no"}:
        return "off", "off (KYKLOS_SANDBOX=off)"
    return _probe_host_sandbox()


def sandbox_argv(inner_argv: list[str], workspace: Path, allow_network: bool = False,
                 env=None) -> list[str]:
    """Wrap `inner_argv` in the active OS kernel sandbox when available."""
    kind, _ = probe_sandbox(env=env)
    if kind == "seatbelt":
        exe = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        return [exe, "-p", seatbelt_profile(workspace, allow_network=allow_network), *inner_argv]
    if kind == "bwrap":
        exe = shutil.which("bwrap") or "bwrap"
        return bwrap_argv(inner_argv, workspace, allow_network=allow_network, bwrap_bin=exe)
    return inner_argv


class Tools:
    def __init__(self, workspace: Path, before_write=None):
        self.workspace = workspace.resolve()
        self.before_write = before_write
        # The plan the model is working through. Steps are linear, so one index is the whole
        # state: everything before `plan_active` is done, everything after is still to do.
        self.plan_steps: list[str] = []
        self.plan_active = 0

    # ---- helpers -----------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        p = (self.workspace / path).resolve()
        if p != self.workspace and self.workspace not in p.parents:
            raise ToolError(f"Path {path!r} is outside the workspace {self.workspace}")
        return p

    def _resolve_writable(self, path: str, verb: str) -> Path:
        """Like _resolve, but also refuses anything inside a .git directory (any level, any case):
        it's git's own database, and a direct write there can corrupt the repository."""
        p = self._resolve(path)
        if p.is_dir():
            raise ToolError(f"Cannot {verb} {path}: it's a folder, not a file. Give a file path inside it.")
        if any(part.lower() == ".git" for part in p.relative_to(self.workspace).parts):
            raise ToolError(f"Refusing to {verb} {path}: files inside .git/ are git's own database, and changing "
                            "them directly can corrupt the repository. Use git commands with bash instead; for a "
                            "commit message, write it to a file outside .git/ and run `git commit -F <file>`.")
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
        p = self._resolve_writable(path, "write")
        if is_secret_file(p):
            raise ToolError(f"Refusing to write {path}: it likely contains secrets. Ask the user to change it.")
        if self.before_write:
            self.before_write(p)
        if is_scratch_path(self.workspace, path):
            ensure_scratch(self.workspace)  # also writes the .gitignore that keeps it out of git
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"

    def _prepare_edit(self, path: str, old_string: str, new_string: str) -> tuple[Path, str, str, str]:
        p = self._resolve_writable(path, "edit")
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
        """Validate tool arguments before asking the user for permission."""
        if not isinstance(args, dict):
            raise TypeError(f"{name}() arguments must be a JSON object")
        if name == "write_file":
            p = self._resolve_writable(args["path"], "write")
            if is_secret_file(p):
                raise ToolError(f"Refusing to write {args['path']}: it likely contains secrets. Ask the user to change it.")
        elif name == "edit_file":
            self._prepare_edit(args["path"], args["old_string"], args["new_string"])
        elif name == "bash":
            unknown = sorted(set(args) - {"command", "timeout"})
            if unknown:
                raise TypeError(f"bash() got an unexpected keyword argument {unknown[0]!r}")
            if "command" not in args or not isinstance(args["command"], str):
                raise TypeError("bash() missing 1 required string argument: 'command'")

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

    def _timed_out(self, timeout: int, stdout: str | None, stderr: str | None) -> str:
        """What a command managed to say before it hung, kept rather than thrown away.

        Discarding it costs more than it looks. A command that prints as it goes has usually done
        most of the work by the time it hangs, and without its output the only way to read the
        numbers it already produced is to run the whole thing again -- which hangs again. Printing
        nothing at all is itself worth saying: that is the shape of a command waiting for input.
        """
        note = (f"Command timed out after {timeout}s. If it waits for input, pass the answer as "
                "a flag instead (e.g. -Force, --yes); if it's just slow, use a larger timeout.")
        out = ((stdout or "") + (stderr or "")).rstrip()
        if not out:
            return f"{note}\nIt printed nothing before it was stopped."
        return self._truncate(f"{note}\nIt printed this before it was stopped, and there may have "
                              f"been more to come:\n{out}")

    def bash(self, command: str, timeout: int = 120) -> str:
        """Public bash tool callable by the model. Network stays off unless `run_bash` is called
        by the harness after explicit user approval of a network/installer command."""
        return self.run_bash(command, timeout=timeout, allow_network=False)

    def run_bash(self, command: str, timeout: int = 120, allow_network: bool = False) -> str:
        # Commands get no keyboard. Their output is captured, so a prompt ("Confirm", Read-Host, npx's
        # "Ok to proceed?", git asking for a password) would be invisible and hang until the timeout.
        # Without stdin, and with PowerShell's -NonInteractive, such prompts fail at once with an error
        # the model can read and react to.
        base_argv = (["powershell", "-NoProfile", "-NonInteractive", "-Command", command] if IS_WINDOWS
                     else ["bash", "-c", command])
        effective_net = allow_network or os.environ.get("KYKLOS_SANDBOX_NET", "").strip() == "1"
        argv = sandbox_argv(base_argv, self.workspace, allow_network=effective_net)
        process = subprocess.Popen(
            argv, cwd=self.workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", env=child_env(),
            start_new_session=not IS_WINDOWS,  # its own process group, so the whole tree can be stopped
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_process_tree(process)
            try:  # the group is dead, so the pipes are closed and this returns what was buffered
                stdout, stderr = process.communicate()
            except (ValueError, OSError):
                stdout, stderr = "", ""
            raise ToolError(self._timed_out(timeout, stdout, stderr))
        except KeyboardInterrupt:  # Ctrl+C: stop everything the command started, then let the turn end
            kill_process_tree(process)
            raise
        out = (stdout or "") + (stderr or "")
        return self._truncate(f"exit code {process.returncode}\n{out}".rstrip())

    # ---- the plan ----------------------------------------------------------
    # The one tool that changes nothing: it exists so the model's intent reaches the user as
    # structure instead of prose. Raw reasoning arrives faster than anyone can read it, and a
    # provider that summarises it does so out of our reach -- but "step 3 of 6, running the
    # tests" is a stage, and a stage can be drawn. The UI renders it; nothing here prints.

    def plan(self, steps: list = None, active: int = None) -> str:
        """Set the steps for this task, or mark which one is in progress."""
        if steps is None and active is None:
            raise ToolError("plan needs steps (to set the plan) or active (to say which step is in progress)")
        new_steps = self.plan_steps
        if steps is not None:
            if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
                raise ToolError("steps must be a list of strings, one per step")
            if not 1 <= len(steps) <= MAX_PLAN_STEPS:
                raise ToolError(f"a plan has 1 to {MAX_PLAN_STEPS} steps, not {len(steps)}")
            new_steps = [plain_line(s) for s in steps]
            if not all(new_steps):  # a step that was only whitespace or escapes says nothing
                raise ToolError("every step needs some text of its own")
        if not new_steps:
            raise ToolError("there is no plan yet: call plan with steps before marking one active")
        # A new plan starts on its first step unless told otherwise; an `active`-only call moves
        # within the plan already set.
        new_active = (1 if steps is not None else self.plan_active) if active is None else active
        if isinstance(new_active, bool) or not isinstance(new_active, int) \
                or not 0 <= new_active <= len(new_steps):
            raise ToolError(f"active is the 1-based step in progress, or 0 once the plan is finished, "
                            f"not {new_active!r} (this plan has {len(new_steps)} steps)")
        self.plan_steps, self.plan_active = new_steps, new_active
        return self.plan_text()

    def plan_text(self) -> str:
        """The plan as the model gets it back. It returns the whole list rather than an
        acknowledgement so that this result, and not the model's recollection, is what a later
        step reads: a step nobody marked done must not drift into looking done."""
        if not self.plan_steps:
            return "No plan set."
        rows = []
        for i, step in enumerate(self.plan_steps, 1):
            done = self.plan_active == 0 or i < self.plan_active
            mark = "x" if done else (">" if i == self.plan_active else " ")
            rows.append(f"  [{mark}] {i}. {step}")
        head = ("every step done" if self.plan_active == 0
                else f"step {self.plan_active} of {len(self.plan_steps)} in progress")
        return f"Plan ({head}):\n" + "\n".join(rows)


# Tools that run without asking. Everything else goes through the permission prompt.
SAFE_TOOLS = {"read_file", "grep", "glob", "plan"}

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
            "name": "plan",
            "description": (
                "Show the user the steps you are working through and which one you are on. Call it "
                "with `steps` before you start any task that will take more than two or three tool "
                "calls, then with `active` alone as each step finishes. Steps are linear: the ones "
                "before `active` are done, the ones after are still to do. Call it with `steps` "
                "again if the plan changes, and with active 0 when everything is done. It writes "
                "nothing and needs no approval -- it is how your progress reaches the user, who "
                "follows these steps instead of your reasoning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array", "items": {"type": "string"},
                        "description": "The whole plan, in order, as short imperative phrases of at "
                                       "most about seven words. Replaces any previous plan.",
                    },
                    "active": {"type": "integer", "description": "The 1-based step now in progress, "
                                                                 "or 0 when every step is done"},
                },
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
                    "timeout": {"type": "integer", "description": "Seconds, default 120. Bound a "
                                "command that might hang with this, not with a shell timeout command."},
                },
                "required": ["command"],
            },
        },
    },
]
