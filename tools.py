"""Tools the model can call: read_file, grep, glob, write_file, edit_file, bash.

Each tool is a JSON schema (sent to the model) plus a Python function
(run by the harness). Paths are resolved against the workspace root and
may not escape it.
"""

import os
import platform
import ast
import difflib
import json
import re
import shutil
import subprocess
import sys
import textwrap
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

MAX_OUTPUT_CHARS = 20_000  # keep tool results from flooding the context window
MAX_SEARCH_RESULTS = 200
MAX_SEARCH_FILE_BYTES = 2_000_000
# Above this, read_file streams to the requested window instead of loading the file: a 1 GB dump
# read whole would hold the harness's memory for a slice of 2000 lines.
MAX_READ_WHOLE_BYTES = 10_000_000
# An image is re-sent, base64-encoded (a third larger), on every step until compaction drops it,
# so one big screenshot costs on every call that follows. 5 MB holds any ordinary screenshot.
MAX_IMAGE_BYTES = 5_000_000
# A model asked for an hour once, and a day is a valid integer. Ten minutes is longer than any
# test or build Kyklos should wait on in the foreground; Ctrl+C still ends one sooner.
MAX_BASH_TIMEOUT = 600
MAX_EDITS = 20  # in one edit_file call; more than this is a rewrite, and write_file says so honestly
MAX_GREP_CONTEXT = 10
EDIT_CONTEXT_LINES = 2  # unchanged lines shown either side of an edit, in its result
EDIT_SHOWN_LINES = 30   # past this, the middle of an edited region is elided
EDIT_SHOWN_REGIONS = 6  # and past this many separate regions, the rest are counted
# Output past MAX_OUTPUT_CHARS is cut in the middle for the model, and the whole of it is kept in
# the scratch directory so the cut part can be searched instead of produced again.
SPILL_DIR = ".kyklos/scratch/output"
SPILL_MAX_BYTES = 5_000_000
SPILL_KEEP = 20
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
    r"(^|_)(API_?|ACCESS_?|SECRET_?|PRIVATE_?)?KEY($|_)|TOKEN|SECRET|PASSW(OR)?D|CREDENTIAL"
    # A header value (HTTP_AUTHORIZATION, NPM_CONFIG__AUTH), a Sentry DSN, a session cookie: secrets
    # named for what they are used for rather than for being secret. Anchored at the end, so
    # FIREBASE_AUTH_DOMAIN and GIT_AUTHOR_NAME stay what they are.
    r"|(^|_)(AUTH|AUTHORIZATION|DSN|COOKIE)$", re.IGNORECASE)
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
# Credentials carried in a URL. Any scheme: a list of database schemes missed `https://user:pass@`,
# `mssql://`, `clickhouse://` and the rest, and a password in a URL is a password whatever follows.
URI_PASSWORD_RE = re.compile(r"\b([a-z][a-z0-9+.-]*://[^/\s:@]*:)([^@\s/]{4,})(@)", re.IGNORECASE)
# A long token standing alone as the user part: a Sentry DSN (https://<key>@o1.ingest.sentry.io/1),
# or `https://<token>@github.com/...` as CI scripts write it.
URI_TOKEN_USER_RE = re.compile(r"\b([a-z][a-z0-9+.-]*://)([A-Za-z0-9_-]{20,})(@)", re.IGNORECASE)
AUTH_HEADER_RE = re.compile(r"\b(authorization:\s*(?:basic|bearer|token)\s+)([^\s\"'`]{8,})()", re.IGNORECASE)
# A digit is required, so prose ("bearer authentication") is left alone: real tokens nearly always have one.
BEARER_RE = re.compile(r"\b(bearer\s+)((?=[A-Za-z0-9._~+/=-]*\d)[A-Za-z0-9._~+/=-]{16,})()", re.IGNORECASE)
# Each is (keep, secret, keep): only the middle is redacted, so the output still says what it was.
CONTEXT_SECRET_PATTERNS = (URI_PASSWORD_RE, URI_TOKEN_USER_RE, AUTH_HEADER_RE, BEARER_RE)


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


def is_secret_env(name: str, value: str = "") -> bool:
    """Secret by its name, or by what it holds: DATABASE_URL is an ordinary name until its value is
    postgres://app:hunter2@db. Judging the value keeps a password-less local URL usable by tests."""
    return is_secret_env_name(name) or bool(URI_PASSWORD_RE.search(value) or URI_TOKEN_USER_RE.search(value))


def child_env(base=None) -> dict:
    """The environment for commands: everything except Kyklos's .env settings and secret-named variables.
    KYKLOS_PASS_ENV=GITHUB_TOKEN,NPM_TOKEN lets chosen ones through on purpose."""
    base = os.environ if base is None else base
    passthrough = {n.strip().upper() for n in base.get("KYKLOS_PASS_ENV", "").split(",") if n.strip()}
    env = {k: v for k, v in base.items()
           if k.upper() in passthrough or (k.upper() not in HIDDEN_ENV_NAMES and not is_secret_env(k, v))}
    env["GIT_TERMINAL_PROMPT"] = "0"  # git fails instead of asking for credentials
    return env


def known_secret_values(base=None) -> list[str]:
    """Values of secret-named variables (like Kyklos's API_KEY), longest first, so they can be redacted
    exactly, whatever their format."""
    base = os.environ if base is None else base
    values = {v for k, v in base.items() if is_secret_env(k, v) and len(v) >= MIN_SECRET_VALUE_LEN}
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
                if is_secret_env(key, val) and len(val) >= MIN_SECRET_VALUE_LEN:
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

    def _redact_middle(match: re.Match) -> str:
        nonlocal count
        if match.group(2) == REDACTED:
            return match.group(0)
        count += 1
        return f"{match.group(1)}{REDACTED}{match.group(3)}"

    for pattern in CONTEXT_SECRET_PATTERNS:
        text = pattern.sub(_redact_middle, text)
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
    ".zsh_history", ".bash_history", ".sh_history", ".history",
    ".python_history", ".node_repl_history", ".mysql_history", ".psql_history",
}
SECRET_PREFIXES = ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", "service-account")
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".kdbx", ".ovpn",
                   ".tfvars", ".tfstate", ".tfstate.backup", ".keychain", ".keychain-db", ".p8")
PROTECTED_CREDENTIAL_DIRS = {".ssh", ".aws", ".gnupg", ".kube", ".docker"}


def is_secret_file(p: Path) -> bool:
    """Files that usually hold API keys, tokens, or private keys. The agent may not read their contents."""
    name = p.name.lower()
    if name.endswith(SAFE_ENV_SUFFIXES):
        return False
    if name in SECRET_EXACT_NAMES or name.startswith(".env."):
        return True
    if name.endswith(SECRET_SUFFIXES) or name.startswith(SECRET_PREFIXES):
        return True
    parts = {part.lower() for part in p.parts}
    return bool(parts & PROTECTED_CREDENTIAL_DIRS)


SECRET_GLOB_TARGETS = (
    ".env", ".env.local", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    "secret.pem", "secret.key", "secret.p12", "secret.pfx", "secret.kdbx",
    "secret.keystore", "secret.jks", "secret.ovpn", "prod.tfvars", "terraform.tfstate",
    ".npmrc", ".pypirc", ".netrc", ".git-credentials", ".pgpass", "kubeconfig",
    "credentials.json", "service-account.json",
    ".zsh_history", ".bash_history", ".sh_history", ".history", "keychain", "keychain-db",
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
        raise ToolError(_not_found_note(file_lines, old_lines))

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

    raise ToolError(_not_found_note(file_lines, old_lines))


# A failed edit used to say only "re-read the file", which costs a whole step to learn what one
# line of diff says. The file is already in hand here, so the nearest text and the first line
# where it parts from old_string go back with the error.
CLOSEST_SHOWN_LINES = 30
CLOSEST_LINE_CHARS = 240


def _closest_start(file_lines: list[str], old_lines: list[str]) -> int | None:
    """The file line where old_string most plausibly belongs, or None when nothing resembles it.

    Each old line that also appears in the file votes for the start that would put it there, and
    the start with the most votes wins. Only when no line matches at all is the first non-blank
    line matched loosely, since that is the one fuzzy comparison worth its cost on a large file.
    """
    stripped = [l.strip() for l in file_lines]
    where: dict[str, list[int]] = {}
    for i, line in enumerate(stripped):
        if line:
            where.setdefault(line, []).append(i)
    votes: dict[int, int] = {}
    for j, line in enumerate(l.strip() for l in old_lines):
        for i in where.get(line, ())[:50]:  # a line as common as `}` shouldn't decide anything
            if i - j >= 0:
                votes[i - j] = votes.get(i - j, 0) + 1
    if votes:
        return max(sorted(votes), key=votes.get)
    first = next(((j, l.strip()) for j, l in enumerate(old_lines) if l.strip()), None)
    if first is None:
        return None
    close = difflib.get_close_matches(first[1], list(where), n=1, cutoff=0.6)
    return max(where[close[0]][0] - first[0], 0) if close else None


def _not_found_note(file_lines: list[str], old_lines: list[str]) -> str:
    base = "old_string not found."
    start = _closest_start(file_lines, old_lines)
    if start is None:
        return f"{base} Nothing in the file resembles it. Re-read the file and copy the text exactly."
    span = file_lines[start : start + len(old_lines)]
    shown = span[:CLOSEST_SHOWN_LINES]
    numbered = "\n".join(f"{n:>6}\t{line.rstrip()[:CLOSEST_LINE_CHARS]}" for n, line in enumerate(shown, start + 1))
    if len(span) > len(shown):
        numbered += f"\n... ({len(span) - len(shown)} more lines)"
    note = f"{base} The closest text in the file is at lines {start + 1}-{start + len(span)}:\n{numbered}"
    for offset, old in enumerate(old_lines):
        actual = span[offset].rstrip("\r\n") if offset < len(span) else None
        if actual is None:
            note += f"\nThe file ends at line {start + len(span)}, before your old_string does."
            break
        if actual.rstrip() != old.rstrip():
            note += (f"\nFirst difference, at line {start + offset + 1}:\n"
                     f"  yours: {old[:CLOSEST_LINE_CHARS]!r}\n  file:  {actual[:CLOSEST_LINE_CHARS]!r}")
            break
    return note + "\nCopy the file's text exactly (without the line-number prefixes)."


def edit_list(args: dict) -> list[dict]:
    """The edits an edit_file call asks for, whichever of its two shapes it used."""
    edits = args.get("edits")
    single = "old_string" in args or "new_string" in args
    if edits is not None and single:
        raise ToolError("give either old_string and new_string, or edits, not both")
    if edits is None:
        if "old_string" not in args or "new_string" not in args:
            raise ToolError("edit_file needs old_string and new_string, or a list of edits")
        edits = [{"old_string": args["old_string"], "new_string": args["new_string"],
                  "replace_all": args.get("replace_all", False)}]
    elif "replace_all" in args:
        raise ToolError("with edits, put replace_all inside the edit it applies to")
    if not isinstance(edits, list) or not edits:
        raise ToolError("edits must be a non-empty list of {old_string, new_string}")
    if len(edits) > MAX_EDITS:
        raise ToolError(f"at most {MAX_EDITS} edits per call, not {len(edits)}: split them, or use write_file "
                        "if the file is being rewritten")
    out = []
    for i, edit in enumerate(edits, 1):
        if (not isinstance(edit, dict) or not isinstance(edit.get("old_string"), str)
                or not isinstance(edit.get("new_string"), str)
                or set(edit) - {"old_string", "new_string", "replace_all"}):
            raise ToolError(f"edit {i} must be {{old_string, new_string}} with string values "
                            "(and optionally replace_all)")
        out.append({"old_string": edit["old_string"], "new_string": edit["new_string"],
                    "replace_all": edit.get("replace_all") is True})
    return out


def apply_edits(text: str, edits: list[dict]) -> tuple[str, dict]:
    """`text` with every edit applied in order, or a ToolError and nothing applied.

    Each edit sees the text the ones before it left, which is what lets the second of two edits
    build on the first. The dict says how the matching went: `tolerant` and `reindented` for a
    match that needed the fallbacks, and `replaced` for how many places replace_all changed.
    """
    info = {"tolerant": False, "reindented": False, "replaced": 0}
    for i, edit in enumerate(edits, 1):
        old, new = edit["old_string"], edit["new_string"]
        try:
            if edit["replace_all"]:
                if not old:
                    raise ToolError("old_string must not be empty. Use write_file to create or overwrite a file.")
                if "\r\n" not in text:
                    old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
                count = text.count(old)
                if not count:
                    raise ToolError(_not_found_note(text.splitlines(keepends=True), old.splitlines()))
                text = text.replace(old, new)
                info["replaced"] += count
            else:
                matched, adjusted = _match_edit(text, old, new)
                info["tolerant"] |= matched not in (old, old.replace("\r\n", "\n"))
                info["reindented"] |= adjusted not in (new, new.replace("\r\n", "\n"))
                text = text.replace(matched, adjusted, 1)
        except ToolError as e:
            if len(edits) == 1:
                raise
            earlier = " Line numbers are as the file would read after the edits before this one." if i > 1 else ""
            raise ToolError(f"edit {i} of {len(edits)}: {e}{earlier} No edit was applied: the file is unchanged.")
    return text, info


def changed_regions(before: str, after: str) -> list[list[tuple]]:
    """The parts of `after` that differ from `before`, each a list of (line number, text) with a
    little unchanged text around it. Only the stretch between the first and last difference is
    diffed, so a small edit to a large file costs what the edit is worth, not what the file is."""
    a, b = before.splitlines(), after.splitlines()
    head = 0
    while head < min(len(a), len(b)) and a[head] == b[head]:
        head += 1
    tail = 0
    while tail < min(len(a), len(b)) - head and a[-1 - tail] == b[-1 - tail]:
        tail += 1
    head, tail = max(head - EDIT_CONTEXT_LINES, 0), max(tail - EDIT_CONTEXT_LINES, 0)
    mid_a, mid_b = a[head:len(a) - tail], b[head:len(b) - tail]
    if len(mid_a) * len(mid_b) > 4_000_000:  # too far apart to be worth diffing line by line
        return [[(head + k + 1, line) for k, line in enumerate(mid_b)]] if mid_b else []
    regions = []
    matcher = difflib.SequenceMatcher(None, mid_a, mid_b, autojunk=False)
    for group in matcher.get_grouped_opcodes(EDIT_CONTEXT_LINES):
        lo, hi = group[0][3], group[-1][4]
        if hi > lo:
            regions.append([(head + k + 1, mid_b[k]) for k in range(lo, hi)])
    return regions


def _region_text(rows: list[tuple]) -> str:
    if len(rows) > EDIT_SHOWN_LINES:  # a long replacement: its edges are what the next edit needs
        head, tail = EDIT_SHOWN_LINES * 2 // 3, EDIT_SHOWN_LINES // 3
        rows = rows[:head] + [(None, f"... ({len(rows) - head - tail} lines not shown)")] + rows[-tail:]
    return "\n".join(line if n is None else f"{n:>6}\t{line}" for n, line in rows)


# After a write, the cheapest check there is: does the file still parse? Done in-process and only
# for formats Python can read itself, so nothing the model wrote is ever executed to find out.
JSONC_NAMES = ("tsconfig", "jsconfig", ".eslintrc", "devcontainer", ".babelrc", "settings", "launch", "tasks")


_PROJECT_PYTHON: dict[str, tuple[int, int] | None] = {}


def project_python() -> tuple[int, int] | None:
    """(major, minor) of the python3 that bash would run, asked once per interpreter path; None
    if there is none or it can't be asked safely.

    Asked inside the sandbox, with -I, and never when it sits in a workspace-relative PATH entry:
    Kyklos runs this itself, and a repository's own `python3` is not something to start unasked.
    """
    exe = shutil.which("python3")
    if not exe or not os.path.isabs(exe):
        return None
    if exe not in _PROJECT_PYTHON:
        version = None
        if probe_sandbox()[0] in ("seatbelt", "bwrap"):
            import tempfile
            argv = sandbox_argv([exe, "-I", "-c", "import sys; print(*sys.version_info[:2])"],
                                Path(tempfile.gettempdir()))
            try:
                out = subprocess.run(argv, capture_output=True, text=True, timeout=10,
                                     env=child_env(), stdin=subprocess.DEVNULL).stdout.split()
                version = (int(out[0]), int(out[1])) if len(out) == 2 else None
            except (OSError, ValueError, subprocess.SubprocessError):
                version = None
        _PROJECT_PYTHON[exe] = version
    return _PROJECT_PYTHON[exe]


def syntax_problem(path: Path, text: str) -> str:
    """Why `text` doesn't parse as the format its name says, or "" when it does or isn't checked."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".py":
            # Parsed by the Python Kyklos runs on. When the project's is newer, an error may only
            # be syntax this one predates (t-strings, say), and "fix it" would have the model
            # rewrite correct code, so the check stays quiet rather than guess.
            newer = project_python()
            if newer and newer > sys.version_info[:2]:
                return ""
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # an invalid escape is a warning, not a broken file
                ast.parse(text, filename=path.name)
        elif suffix == ".json" and not path.name.lower().startswith(JSONC_NAMES):
            json.loads(text)
        elif suffix == ".toml":
            import tomllib  # Python 3.11+; on 3.10 the check is simply skipped
            tomllib.loads(text)
        else:
            return ""
    except SyntaxError as e:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        return f"line {e.lineno}: {e.msg} (parsed as Python {version})"
    except ImportError:
        return ""
    except ValueError as e:  # json.JSONDecodeError and tomllib.TOMLDecodeError are both ValueErrors
        return str(e)
    return ""


def _syntax_note(path: Path, rel: str, text: str) -> str:
    problem = syntax_problem(path, text)
    if not problem:
        return ""
    return (f"\n\n[Kyklos: {rel} was saved, but it does not parse — {problem}. "
            "Fix it before running anything that loads it.]")


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


NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def ensure_scratch(workspace: Path) -> Path:
    """Create the scratch directory, ignoring itself rather than editing the project's .gitignore:
    `.kyklos/.gitignore` holding `*` keeps the whole thing out of git and touches nothing of yours.

    Kyklos itself does this, outside the sandbox, so it refuses to go through a symlink: a
    repository can ship `.kyklos` as a link to anywhere, and mkdir and the marker would follow it.
    """
    directory = scratch_dir(workspace)
    for part in (directory.parent, directory):
        if part.is_symlink():
            raise OSError(f"{part} is a symlink, so Kyklos won't write through it")
        part.mkdir(exist_ok=True)
    marker = directory.parent / ".gitignore"
    if not os.path.lexists(marker):
        try:
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o644)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("*\n")
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
#   - writes confined to the workspace (excluding .git/hooks, .git/config and the files in
#     SANDBOX_PROTECTED_WORKSPACE_FILES), /tmp, and standard caches
#   - reads blocked on credential stores (~/.ssh, ~/.aws, Keychains, browser profiles, ~/.kyklos)
#   - network denied by default unless the user explicitly approved a network/installer command

# Workspace files bash may not write, even though the rest of the workspace is writable. Each one
# makes something run later, outside the sandbox: AGENTS.md and GUARDRAILS.md steer every future
# session, and .git/config is as strong as .git/hooks -- core.fsmonitor, core.pager, core.hooksPath
# and aliases are commands that the user's own `git status` runs, unconfined, the next time they
# type it. Protecting the hooks folder while leaving the file that can relocate it was half a lock.
# Workspace folders bash may not write into, for the same reason: hooks, and editor tasks that run on open.
SANDBOX_PROTECTED_WORKSPACE_DIRS = (".git/hooks", ".husky", ".vscode")
SANDBOX_PROTECTED_WORKSPACE_FILES = ("AGENTS.md", "GUARDRAILS.md", ".gitmodules", ".pre-commit-config.yaml",
                                     ".pre-commit-config.yml", ".git/config")

SANDBOX_PROTECTED_HOME_DIRS = (
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".kyklos",
    "Library/Keychains", "Library/Cookies", "Library/Safari", "Library/Mail", "Library/Messages",
    "Library/LaunchAgents",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox",
    "Library/Application Support/Arc",
    "Library/Application Support/BraveSoftware",
    # Token stores of the CLIs a developer is already signed in to. These are what an install-time
    # payload goes for first -- the Miasma npm worm (June 2026) took GitHub, GCP, Azure and Vault
    # tokens and password-manager stores -- and an open network during installs is what lets the
    # theft leave. Nothing in a build or a test run needs them.
    ".config/gh", ".config/gcloud", ".azure",
    ".config/op", ".op", ".password-store", ".config/gopass", ".local/share/gopass",
)
SANDBOX_PROTECTED_HOME_FILES = (
    ".netrc", ".npmrc", ".pypirc", ".git-credentials", ".pgpass", ".my.cnf",
    ".zsh_history", ".bash_history",
    # Registry and secret-manager tokens: Vault, RubyGems, crates.io, Terraform Cloud, the XDG
    # git credential store, and yarn's rc, which carries `npmAuthToken` the way ~/.npmrc does.
    ".vault-token", ".gem/credentials", ".cargo/credentials", ".cargo/credentials.toml",
    ".terraform.d/credentials.tfrc.json", ".config/git/credentials", ".yarnrc.yml",
)
# The folders holding the user's own documents rather than any project's code. Nothing in a build
# or a test suite reads these, so denying them costs nothing and closes the pair that makes a
# single approved command an exfiltration: a readable home folder and an open socket.
# Both spellings of the video folder: macOS ships Movies, the XDG desktops ship Videos.
SANDBOX_PROTECTED_PERSONAL_DIRS = (
    "Documents", "Desktop", "Downloads", "Pictures", "Movies", "Videos", "Music",
)

# Scratch space a command may write to. It has to be writable: downloads, build temporaries and
# anything using mktemp land here. It must not also be runnable. `curl | sh`, and its unpack-then-run
# cousins, stage a payload under $TMPDIR and execute it from there -- the write is the harmless half
# of that pair, and the exec is the half worth refusing (SEC-11).
SANDBOX_SCRATCH_ROOTS = (
    "/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp", "/var/folders", "/private/var/folders",
)

# What a kernel refusal looks like by the time it reaches stderr. The kernel does not sign its work,
# so a denied write reads exactly like a permissions problem on the host -- which is how a model ends
# up blaming macOS, asking for sudo, or asking the user to loosen something (SEC-12).
SANDBOX_DENIAL_SIGNATURES = ("operation not permitted", "permission denied", "not permitted",
                             "read-only file system",  # how a bubblewrap read-only bind refuses
                             "err_access_denied")  # Chrome's name for the same refusal on a connect
# Words that say a denial was about a port rather than a file, so the hint can name the right wall.
SANDBOX_NETWORK_SIGNATURES = ("listen", "bind", "connect", "socket", "err_access_denied")

# Why a tool that needs the macOS Keychain misbehaves in the sandbox, and what to do about it. The
# profile hides ~/Library/Keychains on purpose, so a process that goes looking for it does not find
# it -- and macOS answers that with a "Keychain Not Found" dialog on the user's screen, whose button
# offers to "Reset To Defaults" a keychain that is not missing. A test run that launched Chrome by
# hand under the sandbox raised one: the system log shows the Chrome process calling the Keychain
# APIs and SecurityAgent, which draws the dialog, starting 0.04s later. The Chrome runs that hung
# beside it are consistent with a dialog nobody answered, which was not confirmed with the flags in
# place. Playwright passes both flags by default (read in its source), so it does not hit this.
SANDBOX_KEYCHAIN_ADVICE = (
    "this sandbox hides ~/Library/Keychains on purpose, so a tool that needs the Keychain cannot find it, "
    "and macOS may put a 'Keychain Not Found' dialog on the user's screen. Do not retry it unchanged: a "
    "repeat can raise another. Have the tool avoid the Keychain (Chrome and Chromium: --use-mock-keychain "
    "--password-store=basic, which Playwright already passes) or tell the user to run it outside Kyklos. "
    "If a dialog appeared, the user should click Cancel, never 'Reset To Defaults': nothing is missing.")
# Flags that mean Chrome was started by hand rather than by a test framework, which sets its own.
CHROME_DIRECT_FLAGS = ("--headless", "--dump-dom", "--screenshot", "--print-to-pdf", "--remote-debugging",
                       "--user-data-dir")


def _chrome_without_mock_keychain(command: str) -> bool:
    """True for a command that starts Chrome or Chromium directly and does not ask it to leave the
    Keychain alone. A heuristic on the command text: it only decides whether to add a sentence to a
    timeout message, so being wrong costs a line, not a decision."""
    low = command.lower()
    return ("chrom" in low and any(flag in low for flag in CHROME_DIRECT_FLAGS)
            and "--use-mock-keychain" not in low)

# Localhost ports a command may bind and use while the rest of the network stays off (macOS). Off by
# default and chosen by the user for the run: KYKLOS_SANDBOX_PORTS=5173,4173. A dev server, or a
# browser test run against one, needs it -- the profile otherwise denies all networking, loopback
# included, so even `bind` on 127.0.0.1 fails with EPERM.
#
# A list of single ports rather than "all of loopback", because loopback is shared with everything
# else on the machine. Tested against the real sandbox: opening it wholesale lets a command reach a
# listener outside the sandbox -- a database, a debug port, a local proxy that relays to the
# internet -- while a listed port reaches nothing but itself. Not a range, because SBPL rejects one
# ("invalid port in network address"). Not below 1024, where the machine's own services live: an
# `ssh localhost` with a forwarded agent is not something to opt into with a typo.
#
# The second setting, KYKLOS_SANDBOX_CONNECT=5173, is the same list with the listening taken out: a
# command may connect to a localhost port, and nothing in the sandbox may bind or accept on any. For
# a server the user runs in their own terminal, and a browser test or `curl` that talks to it, that
# is all that is needed, and it exposes nothing -- there is no listener in the sandbox to reach.
# One rule per port, `network-outbound` to localhost:PORT. Tested on the real sandbox: it carries a
# whole request and reply (a second rule for the reply direction added nothing), covers a server
# bound to ::1 only as well as 127.0.0.1 (Vite's `localhost` is often the former), runs Chrome
# against it, and leaves every bind refused.
SANDBOX_PORTS_ENV = "KYKLOS_SANDBOX_PORTS"
SANDBOX_CONNECT_ENV = "KYKLOS_SANDBOX_CONNECT"
SANDBOX_MIN_PORT, SANDBOX_MAX_PORT = 1024, 65535
SANDBOX_MAX_LOOPBACK_PORTS = 16  # narrow by construction: a project has a handful of servers, not a hundred


def parse_sandbox_ports(raw: str | None) -> tuple[list[int], list[str]]:
    """(ports, problems) from a KYKLOS_SANDBOX_PORTS or KYKLOS_SANDBOX_CONNECT value such as "5173, 4173".

    Anything that is not one plain port from 1024 up is left out and named in `problems`, never
    guessed at. These settings open a hole in the sandbox, so a typo has to close it rather than
    widen it: "5173-5180" opens nothing, and the problem says why.
    """
    ports: list[int] = []
    problems: list[str] = []
    for token in re.split(r"[,\s]+", (raw or "").strip()):
        if not token:
            continue
        if not re.fullmatch(r"\d{1,5}", token):
            problems.append(f"{token!r} is not a single port number (ranges are not supported)")
            continue
        port = int(token)
        if port < SANDBOX_MIN_PORT:
            problems.append(f"{port} is below {SANDBOX_MIN_PORT}, where the machine's own services listen")
        elif port > SANDBOX_MAX_PORT:
            problems.append(f"{port} is not a port number")
        elif port in ports:
            continue
        elif len(ports) >= SANDBOX_MAX_LOOPBACK_PORTS:
            problems.append(f"{port} is past the limit of {SANDBOX_MAX_LOOPBACK_PORTS} ports")
        else:
            ports.append(port)
    return sorted(ports), problems


def sandbox_port_settings(env=None) -> tuple[list[int], list[int], list[str]]:
    """(listen, connect_only, problems) from the two settings.

    `listen` ports may run a server and be connected to; `connect_only` ports may only be connected
    to, and leave out any port already in `listen`, which has that and more. `problems` name the
    entries that were dropped, each prefixed with the variable it came from.
    """
    source = os.environ if env is None else env
    listen, bad_listen = parse_sandbox_ports(source.get(SANDBOX_PORTS_ENV))
    connect, bad_connect = parse_sandbox_ports(source.get(SANDBOX_CONNECT_ENV))
    problems = ([f"{SANDBOX_PORTS_ENV}: {p}" for p in bad_listen]
                + [f"{SANDBOX_CONNECT_ENV}: {p}" for p in bad_connect])
    return listen, [p for p in connect if p not in listen], problems


def sandbox_loopback_ports(env=None) -> list[int]:
    """The localhost ports the user opened for a server to run on. Only the macOS seatbelt backend uses them."""
    return sandbox_port_settings(env)[0]


def sandbox_connect_ports(env=None) -> list[int]:
    """The localhost ports the user opened for connecting only, to a server they run outside Kyklos."""
    return sandbox_port_settings(env)[1]


def _is_loopback_port(port) -> bool:
    return isinstance(port, int) and not isinstance(port, bool) and SANDBOX_MIN_PORT <= port <= SANDBOX_MAX_PORT


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


def _personal_ancestors(workspace: Path, ws: Path, home_dir: Path) -> list[str]:
    """Folders above the workspace that sit inside a protected personal folder, in both spellings.

    These are the directories the read-deny covers but the workspace allow-back does not: for
    ~/Desktop/agentic-projects/app they are ~/Desktop and ~/Desktop/agentic-projects. ~ and /Users
    are not denied, so they are not listed. Checked against the path as given and as resolved, since
    a process may walk either one.
    """
    personal = [home_dir / rel for rel in SANDBOX_PROTECTED_PERSONAL_DIRS]
    out: list[str] = []
    for base in (workspace.absolute(), ws):
        for anc in base.parents:
            if any(_workspace_inside(anc, p) for p in personal):
                for v in _canonical_variants(anc):
                    if v not in out:
                        out.append(v)
    return out


def seatbelt_profile(workspace: Path, allow_network: bool = False, home: Path | None = None,
                     allow_scratch_exec: bool = False, loopback_ports=(), connect_ports=()) -> str:
    """macOS sandbox-exec (SBPL v1) profile confining writes, execution, credential reads, and network.

    `loopback_ports` are localhost ports opened for a server to run on, `connect_ports` ones opened
    for connecting to a server that runs outside (see SANDBOX_PORTS_ENV and SANDBOX_CONNECT_ENV), in
    an otherwise network-less profile. Anything that is not an int in 1024-65535 is dropped here as
    well, so a caller that skipped the parsing still cannot put a stray string into the profile."""
    ws = workspace.resolve()
    home_dir = (home or Path.home()).resolve()
    writable_subpaths: list[str] = []
    for p in (ws, *(Path(r) for r in SANDBOX_SCRATCH_ROOTS),
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
    for rel_ro in SANDBOX_PROTECTED_WORKSPACE_DIRS:
        for v in _canonical_variants(ws / rel_ro):
            lines.append(f"(deny file-write* (subpath {_sbpl_quote(v)}))")
    # The rules above and below name paths, and a path is only as fixed as its parents. Tested:
    # `mv .git .g` succeeded under both, after which .g/config and .g/hooks matched no rule, and
    # `mv .g .git` put a planted hook back where git runs it. The .git entry itself (directory, or
    # the `gitdir:` file of a worktree) may not be renamed, removed or created; what is inside it
    # stays writable, which is what commits, branches and fetches need.
    for v in _canonical_variants(ws / ".git"):
        lines.append(f"(deny file-write* (literal {_sbpl_quote(v)}))")
    for rel_file in SANDBOX_PROTECTED_WORKSPACE_FILES:
        for v in _canonical_variants(ws / rel_file):
            lines.append(f"(deny file-write* (literal {_sbpl_quote(v)}))")

    # Writable scratch stays un-runnable, so a download cannot become a process. The workspace is
    # deliberately not on this list: a build has to be able to run what it just compiled, and the
    # project's own code is what the agent was pointed at in the first place. Toolchains that
    # compile to $TMPDIR and run it from there -- `go test` is the common one -- need
    # KYKLOS_SANDBOX_EXEC_TMP=1, which trades this back for them.
    if not allow_scratch_exec:
        noexec: list[str] = []
        for root in SANDBOX_SCRATCH_ROOTS:
            for v in _canonical_variants(Path(root)):
                if v not in noexec:
                    noexec.append(v)
        for v in noexec:
            lines.append(f"(deny process-exec* (subpath {_sbpl_quote(v)}))")

    # Reads are emitted in three passes because SBPL takes the last matching rule, which is what
    # the blanket `(deny file-write*)` above already relies on. Personal folders are denied first
    # and the workspace is allowed back afterwards, so a repository living under ~/Desktop or
    # ~/Documents -- the ordinary layout, not the exception -- still reads its own files while its
    # siblings stay unreadable. Credentials are denied last so that nothing re-opens them, not even
    # a workspace that happens to contain them (the home folder, trusted for one session).
    personal_denies: list[str] = []
    for rel_dir in SANDBOX_PROTECTED_PERSONAL_DIRS:
        for v in _canonical_variants(home_dir / rel_dir):
            personal_denies.append(f"  (subpath {_sbpl_quote(v)})")
    if personal_denies:
        lines.append("(deny file-read*")
        lines.extend(personal_denies)
        lines.append(")")
        for v in _canonical_variants(ws):
            lines.append(f"(allow file-read* (subpath {_sbpl_quote(v)}))")
        # Allowing the workspace back is not enough on its own. realpath() lstat()s every ancestor
        # of a path, so with ~/Desktop denied, `lstat ~/Desktop` fails with EPERM and Node's
        # `fs.realpathSync` -- which runs on every entry script -- dies before reading a byte of
        # its own project. Metadata only, and literal only: that says the workspace's parent
        # folders exist, which the workspace path already says, and neither lists nor reads them.
        for v in _personal_ancestors(workspace, ws, home_dir):
            lines.append(f"(allow file-read-metadata (literal {_sbpl_quote(v)}))")

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
        # One trio per listed port: listen on it, accept on it, connect to it. What this keeps out is
        # every other port, and every other host as a destination. What it does NOT do is keep a
        # server on loopback. The grammar takes only `*` or `localhost` as a host (no IP literals),
        # and for a listening socket `localhost` matches every local address, so a server that binds
        # 0.0.0.0 or :: -- the Express and `python -m http.server` default -- is reachable from the
        # network on that port while it runs. Tested, and no rule changes it: denying inbound from
        # every peer left connections served, because accept() never consults the peer. Only the
        # server's own bind address decides, so the model is told to use 127.0.0.1.
        listening = sorted({p for p in loopback_ports if _is_loopback_port(p)})[:SANDBOX_MAX_LOOPBACK_PORTS]
        for port in listening:
            lines.extend([
                f'(allow network-bind (local ip "localhost:{port}"))',
                f'(allow network-inbound (local ip "localhost:{port}"))',
                f'(allow network-outbound (remote ip "localhost:{port}"))',
            ])
        # Connect only: the one rule, and nothing that can listen. A server the user runs outside
        # is reachable and nothing in here can open a port of its own, so no network-bind and no
        # network-inbound -- which is also why this cannot expose anything to the network.
        connecting = sorted({p for p in connect_ports if _is_loopback_port(p)} - set(listening))
        for port in connecting[:SANDBOX_MAX_LOOPBACK_PORTS]:
            lines.append(f'(allow network-outbound (remote ip "localhost:{port}"))')
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
    # Personal folders are masked *before* the workspace is bound, because bwrap applies mounts in
    # order: a repository under ~/Desktop is the ordinary layout, and masking the parent afterwards
    # would shadow the workspace itself. Binding it second puts it back on top of the empty tmpfs
    # (bwrap creates the missing mountpoint), so the project reads while its siblings do not.
    for rel_dir in SANDBOX_PROTECTED_PERSONAL_DIRS:
        target = home_dir / rel_dir
        if target.is_dir():
            cmd += ["--tmpfs", str(target)]
    cmd += ["--bind", str(ws), str(ws)]
    # A mount point cannot be renamed or removed (EBUSY). Binding .git onto itself, writable, is
    # what stops `mv .git .old && mv .copy .git` from swapping in a directory whose config and hooks
    # carry none of the read-only binds below. Seatbelt gets the same from a literal deny on .git.
    if (ws / ".git").is_dir():
        cmd += ["--bind", str(ws / ".git"), str(ws / ".git")]
    for rel_ro in SANDBOX_PROTECTED_WORKSPACE_DIRS:
        ro_dir = ws / rel_ro
        if ro_dir.is_dir():
            cmd += ["--ro-bind", str(ro_dir), str(ro_dir)]
    for rel_file in SANDBOX_PROTECTED_WORKSPACE_FILES:
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


def sandbox_status(env=None) -> str:
    """The sandbox backend's label for `/tools`, plus the localhost ports opened in it.

    A setting that opens a hole in the sandbox should be visible while it is doing something, and
    when it was ignored: a typo that silently opened nothing would read as the feature not working.
    """
    kind, label = probe_sandbox(env=env)
    if kind != "seatbelt":
        return label
    listen, connect, problems = sandbox_port_settings(env)
    notes = [f"localhost ports {', '.join(map(str, listen))}"] if listen else []
    if connect:
        notes.append(f"connect-only ports {', '.join(map(str, connect))}")
    notes += [f"{problem}, ignored" for problem in problems]
    return f"{label} ({'; '.join(notes)})" if notes else label


def sandbox_argv(inner_argv: list[str], workspace: Path, allow_network: bool = False,
                 env=None, allow_scratch_exec: bool = False, loopback_ports=(), connect_ports=()) -> list[str]:
    """Wrap `inner_argv` in the active OS kernel sandbox when available."""
    kind, _ = probe_sandbox(env=env)
    if kind == "seatbelt":
        exe = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        return [exe, "-p", seatbelt_profile(workspace, allow_network=allow_network,
                                            allow_scratch_exec=allow_scratch_exec,
                                            loopback_ports=loopback_ports,
                                            connect_ports=connect_ports), *inner_argv]
    if kind == "bwrap":
        # No scratch-exec equivalent here: bubblewrap binds mounts but cannot remount one noexec,
        # so SEC-11 is a seatbelt-only guarantee today. Saying so beats implying parity.
        exe = shutil.which("bwrap") or "bwrap"
        return bwrap_argv(inner_argv, workspace, allow_network=allow_network, bwrap_bin=exe)
    return inner_argv


IMAGE_LABELS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/gif": "GIF", "image/webp": "WebP"}
# What generate_image may save, by the type the bytes turn out to be: the file's extension is made
# to match its content, so a JPEG never sits in a .png.
GENERATED_IMAGE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
GENERATED_IMAGE_PATH_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IMAGE_PROMPT_CHARS = 4000
# What is actually sent. GIF is recognised, so it can be named, but not sent: Ollama cloud answers
# 400 "invalid image input" to GIF87a and GIF89a alike (checked on minimax-m3.1-flash and kimi-k3),
# and a refused request costs every image in the conversation (see Agent._drop_images).
SENDABLE_IMAGES = {"image/png", "image/jpeg", "image/webp"}


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:  # markers with no length
            i += 2
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # a start-of-frame
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return None


def image_info(data: bytes) -> tuple[str, int | None, int | None] | None:
    """(mime type, width, height) of a PNG, JPEG, GIF or WebP, from its bytes; None for anything else.

    Read from the magic number rather than the extension, so a renamed file is still what it is.
    The size is best effort (None where the header doesn't say) and is only ever shown, never relied on.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", *((int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big"))
                              if len(data) >= 24 else (None, None))
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", *(_jpeg_size(data) or (None, None))
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif", *((int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little"))
                              if len(data) >= 10 else (None, None))
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk, size = data[12:16], (None, None)
        if chunk == b"VP8X" and len(data) >= 30:
            size = 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
        elif chunk == b"VP8 " and len(data) >= 30:
            size = int.from_bytes(data[26:28], "little") & 0x3FFF, int.from_bytes(data[28:30], "little") & 0x3FFF
        elif chunk == b"VP8L" and len(data) >= 25:
            bits = int.from_bytes(data[21:25], "little")
            size = (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        return "image/webp", *size
    return None


def describe_image(mime: str, width: int | None, height: int | None, nbytes: int) -> str:
    """'PNG, 1280×800, 214 KB' -- what a person (or a model) needs to know about an image at a glance."""
    size = f"{nbytes / 1_000_000:.1f} MB" if nbytes >= 1_000_000 else f"{max(1, round(nbytes / 1000))} KB"
    dims = f", {width}×{height}" if width and height else ""
    return f"{IMAGE_LABELS.get(mime, mime)}{dims}, {size}"


def load_image(p: Path, shown_as: str) -> dict:
    """An image file read for the model: {"path", "mime", "data" (base64), "desc"}.

    Raises ToolError for anything that isn't a supported image or is over MAX_IMAGE_BYTES, with the
    reason in words, so the caller can hand the message on unchanged.
    """
    import base64
    try:
        size = p.stat().st_size
        if size > MAX_IMAGE_BYTES:
            with p.open("rb") as f:
                info = image_info(f.read(64 * 1024))
            what = describe_image(info[0], info[1], info[2], size) if info else f"{size:,} bytes"
            raise ToolError(f"{shown_as} is too large to send as an image ({what}; the limit is "
                            f"{MAX_IMAGE_BYTES // 1_000_000} MB). Make a smaller copy first, e.g. "
                            f"`sips -Z 1600 in.png --out {SCRATCH_DIR}/small.png` on macOS or "
                            f"`magick in.png -resize 1600x1600 {SCRATCH_DIR}/small.png`.")
        data = p.read_bytes()
    except OSError as e:
        raise ToolError(f"Couldn't read {shown_as}: {e}")
    info = image_info(data)
    if not info:
        raise ToolError(f"{shown_as} isn't a PNG, JPEG or WebP image.")
    if info[0] not in SENDABLE_IMAGES:
        raise ToolError(f"{shown_as} is a {IMAGE_LABELS[info[0]]}, which Ollama refuses as image input, so Kyklos "
                        f"doesn't send it. Convert it to PNG first, e.g. `sips -s format png in.gif --out "
                        f"{SCRATCH_DIR}/in.png` on macOS or `ffmpeg -i in.gif -frames:v 1 {SCRATCH_DIR}/in.png`.")
    return {"path": shown_as, "mime": info[0], "data": base64.b64encode(data).decode("ascii"),
            "desc": describe_image(info[0], info[1], info[2], len(data))}


class Tools:
    def __init__(self, workspace: Path, before_write=None, sees_images=None):
        self.workspace = workspace.resolve()
        self.before_write = before_write
        # Whether the model on the other end accepts images; asked at each read, because /model
        # can change the answer mid-session. read_file leaves the images it reads in
        # pending_images, and the agent loop sends them on after the call's text result.
        self.sees_images = sees_images or (lambda: False)
        self.pending_images: list[dict] = []
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

    def _spill(self, text: str) -> str:
        """Save output too long to return whole, and give back its workspace-relative path ("" if
        it could not be saved).

        The middle of a long build or test log is exactly where the first error sits, and the cut
        used to throw it away: the only way back was to run the command again and hope to filter
        it. Kept on disk it is one grep away. Secrets are scrubbed before it is written, since the
        loop's own scrubbing only sees what comes back through a tool result.
        """
        try:
            # Kyklos writes and prunes here itself, unsandboxed, so the folder must really be the
            # one inside this workspace: ensure_scratch refuses a symlinked .kyklos or scratch, and
            # the output folder is held to the same.
            directory = ensure_scratch(self.workspace) / Path(SPILL_DIR).name
            if directory.is_symlink():
                return ""
            directory.mkdir(exist_ok=True)
            if Path(os.path.realpath(directory)) != directory:
                return ""
            old = sorted(directory.glob("bash-*.log"), key=lambda f: f.lstat().st_mtime)
            for stale in old[:max(len(old) - SPILL_KEEP + 1, 0)]:
                stale.unlink(missing_ok=True)
            body = scrub_secrets(text, workspace=self.workspace)[0].encode("utf-8", errors="replace")
            if len(body) > SPILL_MAX_BYTES:
                body = body[:SPILL_MAX_BYTES] + f"\n... [stopped at {SPILL_MAX_BYTES:,} bytes]\n".encode()
            # Nanoseconds, not a counter: a subagent has its own Tools and would count from one again.
            target = directory / f"bash-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}.log"
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(body)
            return self._rel(target)
        except OSError:
            return ""

    def _truncate_spilling(self, text: str) -> str:
        """_truncate for command output: the part cut from the middle stays reachable on disk."""
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        saved = self._spill(text)
        if not saved:
            return self._truncate(text)
        half = MAX_OUTPUT_CHARS // 2
        lines = text.count("\n") + 1
        return (f"{text[:half]}\n\n... [{len(text) - MAX_OUTPUT_CHARS} chars truncated here. The full output "
                f"({lines:,} lines) is saved at {saved}: grep it, or read_file it with offset and limit, "
                f"instead of running the command again] ...\n\n{text[-half:]}")

    # ---- tools -------------------------------------------------------------

    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"File not found: {path}")
        if is_secret_file(p):
            raise ToolError(f"Refusing to read {path}: it likely contains secrets such as API keys. "
                            "If you need to know what settings exist, read the .env.example file instead.")
        try:
            size = p.stat().st_size
            with p.open("rb") as f:
                head = f.read(8192)
        except OSError as e:
            raise ToolError(f"Couldn't read {path}: {e}")
        if image_info(head):
            if not self.sees_images():
                info = image_info(head)
                raise ToolError(f"{path} is an image ({describe_image(info[0], info[1], info[2], size)}), and the "
                                "current model doesn't take images, so Kyklos can't show it to you. Say so if "
                                "you need to see it; `file` with bash gives its type and size.")
            image = load_image(p, path)
            self.pending_images.append(image)
            return f"{path}: {image['desc']}. The image follows, in the next message."
        if b"\0" in head:  # the same test grep uses; UTF-16 text has NULs too, and is rare in a repo
            raise ToolError(f"{path} looks binary ({size:,} bytes): as text it would be replacement "
                            "characters. Inspect it with bash instead (`file`, `xxd | head`, `sqlite3`, `unzip -l`).")
        start = max(offset, 1) - 1
        if size <= MAX_READ_WHOLE_BYTES:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            total, chunk = len(lines), lines[start : start + limit]
        else:  # stream to the window; memory stays at one window, whatever the file's size
            chunk, total = [], 0
            with p.open(encoding="utf-8", errors="replace") as f:
                for total, line in enumerate(f, 1):
                    if start < total <= start + limit:
                        chunk.append(line.rstrip("\r\n"))
        numbered = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(chunk, start + 1))
        if start + limit < total:
            numbered += f"\n... ({total - start - limit} more lines; use offset to read on)"
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
        return f"Wrote {len(content)} chars to {path}" + _syntax_note(p, path, content)

    def save_image(self, path: str, data: bytes) -> Path:
        """Write a generated image. The same path rules as write_file, and the same snapshot, so
        /undo removes it or puts back what it replaced."""
        p = self._resolve_writable(path, "write")
        if is_secret_file(p):
            raise ToolError(f"Refusing to write {path}: it likely contains secrets.")
        if self.before_write:
            self.before_write(p)
        if is_scratch_path(self.workspace, path):
            ensure_scratch(self.workspace)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def _prepare_edits(self, args: dict) -> tuple[Path, str, str, dict]:
        """(file, its text, the text with every edit applied, how the matching went). Raises a
        ToolError, with nothing written, if any edit can't be applied."""
        p = self._resolve_writable(args["path"], "edit")
        if not p.is_file():
            raise ToolError(f"File not found: {args['path']}")
        if is_secret_file(p):  # edit errors echo nothing, but keep secrets files hands-off entirely
            raise ToolError(f"Refusing to edit {args['path']}: it likely contains secrets. Ask the user to change it.")
        text = p.read_text(encoding="utf-8")
        updated, info = apply_edits(text, edit_list(args))
        return p, text, updated, info

    def edit_file(self, path: str, old_string: str = None, new_string: str = None,
                  replace_all: bool = None, edits: list = None) -> str:
        args = {k: v for k, v in (("path", path), ("old_string", old_string), ("new_string", new_string),
                                  ("replace_all", replace_all), ("edits", edits)) if v is not None}
        p, text, updated, info = self._prepare_edits(args)
        if self.before_write:
            self.before_write(p)
        p.write_text(updated, encoding="utf-8")
        # "Edited path" left the model working from its own picture of the file, which is wrong
        # whenever the match was a tolerant one: the text on disk then differs from both strings it
        # sent. The regions as they now stand go back instead, numbered like read_file's output.
        notes = []
        count = len(edits) if edits else 1
        if count > 1:
            notes.append(f"{count} edits applied")
        if info["replaced"]:
            notes.append(f"replace_all changed {info['replaced']} place{'' if info['replaced'] == 1 else 's'}")
        if info["tolerant"]:
            notes.append("old_string matched after ignoring whitespace differences"
                         + (", and new_string was re-indented to fit" if info["reindented"] else ""))
        how = f" ({'; '.join(notes)})" if notes else ""
        regions = changed_regions(text, updated)
        if not regions:
            result = f"Edited {path}{how}."
        elif len(regions) == 1:
            rows = regions[0]
            result = f"Edited {path}{how}. Lines {rows[0][0]}-{rows[-1][0]} now read:\n{_region_text(rows)}"
        else:
            shown = regions[:EDIT_SHOWN_REGIONS]
            result = (f"Edited {path}{how}. The changed regions now read:\n"
                      + "\n   ...\n".join(map(_region_text, shown)))
            if len(regions) > len(shown):
                result += f"\n... ({len(regions) - len(shown)} more changed regions not shown)"
        return self._truncate(result) + _syntax_note(p, path, updated)

    def check(self, name: str, args: dict) -> None:
        """Validate tool arguments before asking the user for permission."""
        if not isinstance(args, dict):
            raise TypeError(f"{name}() arguments must be a JSON object")
        if name == "write_file":
            p = self._resolve_writable(args["path"], "write")
            if is_secret_file(p):
                raise ToolError(f"Refusing to write {args['path']}: it likely contains secrets. Ask the user to change it.")
        elif name == "edit_file":
            unknown = sorted(set(args) - {"path", "old_string", "new_string", "replace_all", "edits"})
            if unknown:
                raise TypeError(f"edit_file() got an unexpected keyword argument {unknown[0]!r}")
            self._prepare_edits(args)
        elif name == "generate_image":
            unknown = sorted(set(args) - {"prompt", "path", "size"})
            if unknown:
                raise TypeError(f"generate_image() got an unexpected keyword argument {unknown[0]!r}")
            prompt, path = args.get("prompt"), args.get("path")
            if not isinstance(prompt, str) or not prompt.strip():
                raise TypeError("generate_image() needs a non-empty string 'prompt'")
            if len(prompt) > MAX_IMAGE_PROMPT_CHARS:
                raise ToolError(f"The prompt is {len(prompt):,} characters; image models take up to "
                                f"{MAX_IMAGE_PROMPT_CHARS:,}. Describe the image more briefly.")
            if not isinstance(path, str) or not path.strip():
                raise TypeError("generate_image() needs a string 'path' to save the image to, e.g. assets/hero.png")
            if Path(path).suffix.lower() not in GENERATED_IMAGE_PATH_SUFFIXES:
                raise ToolError(f"Save the image as .png, .jpg or .webp, not {path!r}.")
            p = self._resolve_writable(path, "write")
            if is_secret_file(p):
                raise ToolError(f"Refusing to write {path}: it likely contains secrets.")
            size = args.get("size")
            if size is not None and not (isinstance(size, str) and re.match(r"^\d{2,4}x\d{2,4}$", size)):
                raise ToolError(f"size must look like 1024x1024, not {size!r}.")
        elif name == "bash":
            unknown = sorted(set(args) - {"command", "timeout"})
            if unknown:
                raise TypeError(f"bash() got an unexpected keyword argument {unknown[0]!r}")
            if "command" not in args or not isinstance(args["command"], str):
                raise TypeError("bash() missing 1 required string argument: 'command'")
            if "timeout" in args:
                timeout = args["timeout"]
                if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                    raise TypeError(f"bash() timeout must be a number of seconds, not {timeout!r}")
                if not 1 <= timeout <= MAX_BASH_TIMEOUT:
                    # Refused rather than quietly capped, so the model knows the wait it asked for
                    # is not the one it would get.
                    longer = (" For longer work, split it into steps or give the user the command to "
                              "run in their own terminal." if timeout > MAX_BASH_TIMEOUT else "")
                    raise ToolError(f"bash timeout must be between 1 and {MAX_BASH_TIMEOUT} seconds "
                                    f"(got {timeout}).{longer}")

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.workspace).as_posix()

    def _walk_files(self, root: Path):
        """Yield files under root, skipping dependency and cache folders."""
        if root.is_file():
            yield root
            return
        # Saved command output is searched only when asked for by path: a workspace-wide grep that
        # also matched every log Kyklos kept would bury the code under its own build output.
        spill = self.workspace / SPILL_DIR
        skip_spill = root != spill and spill not in root.parents
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS
                                 and not (skip_spill and Path(dirpath) / d == spill))
            for name in sorted(filenames):
                yield Path(dirpath) / name

    def grep(self, pattern: str, path: str = ".", glob: str = None, ignore_case: bool = False,
             context: int = 0, output: str = "lines") -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            raise ToolError(f"Invalid regex: {e}")
        if output not in ("lines", "files", "count"):
            raise ToolError(f"output must be 'lines', 'files' or 'count', not {output!r}")
        if isinstance(context, bool) or not isinstance(context, int) or not 0 <= context <= MAX_GREP_CONTEXT:
            raise ToolError(f"context is a number of lines from 0 to {MAX_GREP_CONTEXT}, not {context!r}")
        root = self._resolve(path)
        if not root.exists():
            raise ToolError(f"Path not found: {path}")
        hits, matched, files = [], 0, 0
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
            lines = data.decode("utf-8", errors="replace").splitlines()
            found = [i for i, line in enumerate(lines) if regex.search(line)]
            if not found:
                continue
            rel = self._rel(f)
            if output != "lines":
                # Which files, or how many times: the answer to "where is this used" without paying
                # for every matching line of it.
                hits.append(rel if output == "files" else f"{rel}: {len(found)}")
                files += 1
                if files >= MAX_SEARCH_RESULTS:
                    hits.append(f"... (stopped at {MAX_SEARCH_RESULTS} files; narrow the pattern or path)")
                    break
                continue
            room = MAX_SEARCH_RESULTS - matched
            stopped = len(found) > room
            found = found[:room]
            matched += len(found)
            if not context:
                hits += [f"{rel}:{i + 1}: {lines[i].strip()[:300]}" for i in found]
            else:
                # As `grep -n -C` writes it: `:` marks a matching line and `-` one shown for context,
                # with `--` between stretches that don't touch. Indentation is kept here, because
                # the point of context is to read the code around the match.
                hit, last = set(found), None
                for i in found:
                    for k in range(max(i - context, 0), min(i + context, len(lines) - 1) + 1):
                        if last is not None and k <= last:
                            continue
                        if (last is not None and k > last + 1) or (last is None and hits):
                            hits.append("--")
                        mark = ":" if k in hit else "-"
                        hits.append(f"{rel}{mark}{k + 1}{mark} {lines[k].rstrip()[:300]}")
                        last = k
            if stopped or matched >= MAX_SEARCH_RESULTS:
                hits.append(f"... (stopped at {MAX_SEARCH_RESULTS} matches; narrow the pattern or path)")
                break
        return self._truncate("\n".join(hits) or "No matches.")

    def glob(self, pattern: str) -> str:
        patterns = expand_braces(pattern)
        if any(Path(p).is_absolute() or ".." in Path(p).parts for p in patterns):
            raise ToolError("Pattern must be relative to the workspace, without '..'")
        files, hidden = [], 0
        matches = {m for p in patterns for m in self.workspace.glob(p)}
        for p in sorted(matches):
            if not p.is_file() or SKIP_DIRS.intersection(p.relative_to(self.workspace).parts):
                continue
            if is_secret_file(p):  # as read_file and grep already refuse: not even where they are
                hidden += 1
                continue
            files.append(self._rel(p))
            if len(files) >= MAX_SEARCH_RESULTS:
                files.append(f"... (stopped at {MAX_SEARCH_RESULTS} files; narrow the pattern)")
                break
        if hidden:  # a count, not names: otherwise "No files matched" would be a lie the model acts on
            files.append(f"({hidden} matching file{'' if hidden == 1 else 's'} not listed: "
                         f"{'it' if hidden == 1 else 'they'} likely hold{'s' if hidden == 1 else ''} secrets)")
        return "\n".join(files) or "No files matched."

    def _sandbox_note(self, out: str) -> str:
        """Name the sandbox when it is the likely author of a failure.

        A seatbelt denial surfaces as `Operation not permitted` and nothing else -- the same text a
        genuine host permissions problem produces. A model reading that reasonably concludes the OS
        is in the way and reaches for the fix that works on a real OS problem: sudo, or asking the
        user to grant something. Both are worse than the original failure. Saying which wall was hit
        costs one line and removes the reason to go looking for a way around it.

        Two walls that an ordinary job walks into get the way through named as well, because the
        generic line alone cost a model a turn each time: `git init` copies sample hooks into
        .git/hooks, which the profile keeps read-only on purpose, and a dev server cannot listen
        because loopback is off unless the user opened the port.

        A third needs no denial at all to show up: a tool that goes looking for the Keychain fails
        with its own wording ("the specified keychain could not be found"), so on macOS any output
        that mentions it gets the explanation, and the warning about the dialog's wrong button.
        """
        kind, label = probe_sandbox()
        if kind not in {"seatbelt", "bwrap"} or not out:
            return ""
        low = out.lower()
        denied = any(sig in low for sig in SANDBOX_DENIAL_SIGNATURES)
        keychain = kind == "seatbelt" and "keychain" in low
        if not denied and not keychain:
            return ""
        note = f"\n\n[harness] bash runs under {label}."
        if denied:
            note += (" If that was a write outside the workspace "
                     "or a scratch dir, a change to a path the sandbox keeps read-only (.git/hooks, "
                     ".git/config, .husky, .vscode, AGENTS.md), or running a file from "
                     f"{SANDBOX_SCRATCH_ROOTS[0]} and friends, this sandbox refused it -- not the host OS, "
                     "and not a missing privilege.")
            if ".git/config" in low or "could not write config file" in low:
                # Settings are what make git run programs (core.fsmonitor, core.hooksPath, aliases),
                # so they are the user's to change. Commits, branches, merges and tags still work.
                note += (" Git settings in .git/config are read-only here on purpose: `git config`, "
                         "`git remote add`, `git push -u` and `git branch -u` cannot change them from inside "
                         "the sandbox. Commits, branches, merges and tags still work. Give the user the exact "
                         "command to run in their own terminal, for one-off values use `git -c name=value`, "
                         "and push with an explicit `git push origin <branch>` instead of setting upstream.")
            elif ".git" in low and "operation not permitted" in low:
                note += (" The workspace's .git entry and .git/hooks are read-only here on purpose, so "
                         "`git init` and `git clone` into the workspace itself fail: ask the user to run "
                         "`git init` in their own terminal. Cloning into a new subfolder still works.")
            if any(sig in low for sig in SANDBOX_NETWORK_SIGNATURES):
                note += self._network_note(kind)
        if keychain:
            note += f" If it used the macOS Keychain: {SANDBOX_KEYCHAIN_ADVICE}"
        return note + (" Escalating will not change it: work inside the workspace, or tell the user what "
                       "has to happen outside it and let them do it.")

    @staticmethod
    def _network_note(kind: str) -> str:
        """What to say when a denial reads like a port rather than a file."""
        if kind != "seatbelt":
            return (" If it was a network connection: the network is cut off in this sandbox, so tell the "
                    "user rather than looking for another route.")
        listen, connect, _ = sandbox_port_settings()
        opened = []
        if listen:
            opened.append(f"Localhost ports open for this run: {', '.join(map(str, listen))}; use one of those, "
                          "listening on 127.0.0.1 (a server on every interface is reachable from the network).")
        if connect:
            opened.append(f"Connect-only localhost ports: {', '.join(map(str, connect))}; the user runs a server "
                          "there outside Kyklos, and nothing in here can listen.")
        return (" If it was listening on or connecting to a port: the network is off here, loopback included, "
                "so a dev server (or a browser test against one) cannot start. "
                f"{' '.join(opened) or 'No ports are open.'} The user can open specific ones by restarting "
                f"Kyklos with {SANDBOX_PORTS_ENV}=<port> (the sandbox may run the server) or "
                f"{SANDBOX_CONNECT_ENV}=<port> (it may only connect to a server the user runs outside "
                "Kyklos), or run that command themselves outside it.")

    def _timed_out(self, timeout: int, stdout: str | None, stderr: str | None, command: str = "") -> str:
        """What a command managed to say before it hung, kept rather than thrown away.

        Discarding it costs more than it looks. A command that prints as it goes has usually done
        most of the work by the time it hangs, and without its output the only way to read the
        numbers it already produced is to run the whole thing again -- which hangs again. Printing
        nothing at all is itself worth saying: that is the shape of a command waiting for input.

        One hang has a cause nobody at the keyboard would guess, so it is named when the command
        gives it away: Chrome started by hand under the macOS sandbox goes to the Keychain, and the
        dialog that answers waits for a person. It prints nothing, so no failure hint could catch it.
        """
        note = (f"Command timed out after {timeout}s. If it waits for input, pass the answer as "
                "a flag instead (e.g. -Force, --yes); if it's just slow, use a larger timeout.")
        if command and probe_sandbox()[0] == "seatbelt" and _chrome_without_mock_keychain(command):
            note += f" This command starts Chrome without --use-mock-keychain, and {SANDBOX_KEYCHAIN_ADVICE}"
        out = ((stdout or "") + (stderr or "")).rstrip()
        if not out:
            return f"{note}\nIt printed nothing before it was stopped."
        return self._truncate_spilling(f"{note}\nIt printed this before it was stopped, and there may have "
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
        scratch_exec = os.environ.get("KYKLOS_SANDBOX_EXEC_TMP", "").strip() == "1"
        argv = sandbox_argv(base_argv, self.workspace, allow_network=effective_net,
                            allow_scratch_exec=scratch_exec, loopback_ports=sandbox_loopback_ports(),
                            connect_ports=sandbox_connect_ports())
        watch = self._absent_protected_paths() if probe_sandbox()[0] == "bwrap" else []
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
            raise ToolError(self._timed_out(timeout, stdout, stderr, command) + self._quarantine(watch))
        except KeyboardInterrupt:  # Ctrl+C: stop everything the command started, then let the turn end
            kill_process_tree(process)
            self._quarantine(watch)
            raise
        out = (stdout or "") + (stderr or "")
        if process.returncode != 0:
            out += self._sandbox_note(out)
        out += self._quarantine(watch)
        return self._truncate_spilling(f"exit code {process.returncode}\n{out}".rstrip())

    # ---- protected paths that did not exist yet (bubblewrap) ---------------------------------
    # Seatbelt refuses by path, so a deny on .git/config or AGENTS.md holds whether or not the file
    # exists. Bubblewrap can only hold an existing file read-only, by binding it onto itself; and a
    # placeholder bound where a missing one would go is no answer, because bwrap creates the mount
    # point on the real disk and leaves it there -- every command would litter the repo with empty
    # AGENTS.md, .gitmodules and .pre-commit-config.yaml files, the last of which breaks pre-commit.
    # So the protected paths absent before a command are checked after it, and any that appeared
    # are moved out of the workspace's way. The outcome matches seatbelt's (the path stays absent);
    # the difference is that it existed while the command ran, which is inside the sandbox anyway.

    def _absent_protected_paths(self) -> list[str]:
        rels = (".git", *SANDBOX_PROTECTED_WORKSPACE_DIRS, *SANDBOX_PROTECTED_WORKSPACE_FILES)
        return [rel for rel in rels if not os.path.lexists(self.workspace / rel)]

    def _quarantine(self, absent: list[str]) -> str:
        """Move the paths in `absent` that a command created into .kyklos/quarantine/, and say so.
        `.git` comes first, so a repository made by `git init` moves whole, hooks and config with it."""
        appeared = [rel for rel in absent if os.path.lexists(self.workspace / rel)]
        if not appeared:
            return ""
        import tempfile
        import time
        root = self.workspace / ".kyklos" / "quarantine"
        try:
            ensure_scratch(self.workspace)  # .kyklos/.gitignore, so none of this reaches git
            root.mkdir(parents=True, exist_ok=True)
            batch = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=root))
        except OSError as e:
            return (f"\n\n[harness] This command created {', '.join(appeared)}, which bash may not create, "
                    f"and Kyklos could not move it aside ({e}). Tell the user.")
        moved, failed = [], []
        for rel in appeared:
            src = self.workspace / rel
            if not os.path.lexists(src):
                continue  # went with .git
            dest = batch / rel
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.rename(src, dest)  # same filesystem: a rename, which moves symlinks as links
                moved.append(rel)
            except OSError:
                failed.append(rel)
        shown = batch.relative_to(self.workspace).as_posix()
        note = ""
        if moved:
            note += (f"\n\n[harness] This command created {', '.join(moved)}, which bash may not create here: "
                     f"they control git hooks and settings, persistent instructions, or editor tasks. Kyklos "
                     f"moved {'it' if len(moved) == 1 else 'them'} to {shown}/. If one is really needed, give "
                     "the user what to create and let them do it.")
        if failed:
            note += (f"\n\n[harness] This command also created {', '.join(failed)}, which Kyklos could not move "
                     "aside. Tell the user.")
        return note

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
            "description": ("Read a file. Text comes back as lines prefixed with line numbers. An image "
                            "(PNG, JPEG, WebP) is shown to you when the model takes images; "
                            "offset and limit don't apply to it."),
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
                    "context": {"type": "integer", "description": f"Lines to show before and after each match, "
                                f"0 to {MAX_GREP_CONTEXT} (default 0). Use it instead of a read_file after the search."},
                    "output": {"type": "string", "enum": ["lines", "files", "count"],
                               "description": "lines (default): the matching lines. files: only the paths "
                                              "that match. count: each path with its number of matches."},
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
                "old_string must match the file exactly (without line-number prefixes) and be unique. "
                "For several changes to one file, pass edits instead of old_string and new_string: "
                "they are applied in order, and if any one fails none is applied. "
                "The result shows the changed lines as they now read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence of old_string "
                                    "instead of requiring it to be unique, e.g. to rename a variable"},
                    "edits": {
                        "type": "array",
                        "description": f"Up to {MAX_EDITS} edits to this file, instead of old_string and new_string",
                        "items": {
                            "type": "object",
                            "properties": {"old_string": {"type": "string"}, "new_string": {"type": "string"},
                                           "replace_all": {"type": "boolean"}},
                            "required": ["old_string", "new_string"],
                        },
                    },
                },
                "required": ["path"],
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
            "description": (f"Run a {SHELL_NAME} command in the workspace directory. Returns exit code and output. "
                            "Every call starts fresh in the workspace root: a cd or an exported variable does "
                            "not carry over to the next call, so chain the commands in one call when it matters."),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": f"Seconds, default 120, at most {MAX_BASH_TIMEOUT}. "
                                "Bound a command that might hang with this, not with a shell timeout command."},
                },
                "required": ["command"],
            },
        },
    },
]
