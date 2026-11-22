"""Workspace snapshots, so /undo can revert what a turn's commands changed and not only its file tools.

write_file and edit_file record a file's bytes before they touch it, but a turn also changes files
through bash: `sed -i`, `rm`, a code generator, a formatter. Those happened behind the file tools'
backs, so undoing the turn used to leave them in place.

A snapshot here is a git tree of the workspace, written into a private git directory of Kyklos's own
(a fresh temporary folder, removed when Kyklos exits). The workspace's own `.git` is never read or
written. Git is used only as a content-addressed store: `add -A` hashes what changed since the last
snapshot, `write-tree` names the result. Nothing is committed, and nothing from the user's git setup
applies: system and global config are blanked, so no hook, filter, or fsmonitor can run, and every
path is stored byte for byte, whatever the workspace's `.gitattributes` say about line endings.

A turn records the tree when it starts and when it ends. Undo restores each path that changed
between the two to its start-of-turn bytes, unless the file has changed again since the turn ended,
in which case it was you who changed it, and it is left alone and named.

Not covered, by design: folders Kyklos skips everywhere (`.venv`, `node_modules`, ...), anything the
workspace's `.gitignore` ignores, `.kyklos/`, and secret files, whose contents must never be copied
anywhere. Files outside the workspace aren't covered either.
"""

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tools import (PROTECTED_CREDENTIAL_DIRS, SAFE_ENV_SUFFIXES, SECRET_EXACT_NAMES, SECRET_PREFIXES,
                   SECRET_SUFFIXES, SKIP_DIRS)

GIT_TIMEOUT = 30  # seconds per git call; a workspace too large to hash in that is not snapshotted
EMPTY_MODES = {"000000"}
LINK_MODE, GITLINK_MODE, EXEC_MODE = "120000", "160000", "100755"

# The same files Kyklos refuses to read or write, as gitignore patterns, plus the folders it skips.
# They go in the store's info/exclude, so their bytes never reach the store at all.
EXCLUDES = (
    [".kyklos/"]
    + [f"{d}/" for d in sorted(SKIP_DIRS)]
    + [f"{d}/" for d in sorted(PROTECTED_CREDENTIAL_DIRS)]
    + sorted(SECRET_EXACT_NAMES) + [".env.*"]
    + [f"{p}*" for p in SECRET_PREFIXES]
    + [f"*{s}" for s in SECRET_SUFFIXES]
    + [f"!*{s}" for s in SAFE_ENV_SUFFIXES]  # .env.example and friends are ordinary files
)
# Highest-precedence attributes: no end-of-line conversion, no filter, no encoding, so a restored
# file has exactly the bytes it had, and no driver named by the workspace could ever run.
ATTRIBUTES = "* -text !eol -filter -ident !working-tree-encoding -diff -merge\n"


class SnapshotError(Exception):
    pass


class WorkspaceSnapshots:
    """A private git store holding trees of one workspace. Create with WorkspaceSnapshots.open()."""

    def __init__(self, workspace: Path, store: Path, git: str):
        self.workspace = Path(workspace).resolve()
        self.store = store
        self.git = git
        self.disabled: str | None = None  # why snapshots stopped, once they have

    @classmethod
    def open(cls, workspace: Path) -> "WorkspaceSnapshots":
        """A store for this workspace, or a SnapshotError saying why there can't be one."""
        git = shutil.which("git")
        if not git:
            raise SnapshotError("git isn't installed")
        store = Path(tempfile.mkdtemp(prefix="kyklos-snap-")).resolve()  # mode 0700
        atexit.register(shutil.rmtree, store, True)
        snaps = cls(workspace, store, git)
        # The few files a git directory needs, written directly: `git init` costs 40ms or more,
        # paid once per conversation, and it would copy in the user's own init templates.
        for sub in ("objects", "refs/heads", "refs/tags", "info"):
            (store / sub).mkdir(parents=True)
        (store / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        # Ignore rules match case-insensitively where the file system is, as git's own init decides.
        ignorecase = "true" if sys.platform in ("darwin", "win32") else "false"
        (store / "config").write_text("[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
                                      f"\tignorecase = {ignorecase}\n", encoding="utf-8")
        info = store / "info"
        excludes = list(EXCLUDES)
        try:  # the store is outside the workspace unless the workspace holds the temp folder
            excludes.insert(0, "/" + store.relative_to(snaps.workspace).as_posix() + "/")
        except ValueError:
            pass
        (info / "exclude").write_text("\n".join(excludes) + "\n", encoding="utf-8")
        (info / "attributes").write_text(ATTRIBUTES, encoding="utf-8")
        return snaps

    def close(self) -> None:
        shutil.rmtree(self.store, ignore_errors=True)

    # --- git plumbing ---------------------------------------------------------------------------

    def _env(self) -> dict:
        # Built from nothing, so no inherited GIT_DIR, GIT_INDEX_FILE or GIT_OBJECT_DIRECTORY can
        # point git somewhere else, and HOME is the store, so no ~/.gitconfig is read.
        env = {"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "HOME": str(self.store),
               "XDG_CONFIG_HOME": str(self.store), "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0"}
        for name in ("SYSTEMROOT", "TEMP", "TMP"):  # Windows needs these to start a process at all
            if name in os.environ:
                env[name] = os.environ[name]
        return env

    def _git(self, *args, work_tree: bool = True, stdin: bytes | None = None, ok=(0,)) -> bytes:
        cmd = [self.git, "-c", "core.hooksPath=" + os.devnull, "-c", "core.fsmonitor=false",
               "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-c", "core.symlinks=true",
               "-c", "gc.auto=0", "-c", "core.quotepath=false", "-c", "advice.addEmbeddedRepo=false"]
        if work_tree:
            cmd += ["--git-dir", str(self.store), "--work-tree", str(self.workspace)]
        try:
            done = subprocess.run(cmd + list(args), cwd=self.workspace, env=self._env(), input=stdin,
                                  capture_output=True, timeout=GIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise SnapshotError(f"git took over {GIT_TIMEOUT}s; the workspace may be too large")
        except OSError as e:
            raise SnapshotError(f"git didn't start: {e}")
        if done.returncode not in ok:
            message = done.stderr.decode("utf-8", "replace").strip().splitlines()
            raise SnapshotError(f"git {args[0]} failed: {message[-1] if message else done.returncode}")
        return done.stdout

    # --- snapshots ------------------------------------------------------------------------------

    def snapshot(self) -> str | None:
        """The workspace as a tree id, or None once snapshots are off (self.disabled says why)."""
        if self.disabled:
            return None
        try:
            self._git("add", "--all", "--ignore-errors", "--", ".")
            return self._git("write-tree").decode().strip()
        except SnapshotError as e:
            self.disabled = str(e)
            return None

    def changes(self, old: str, new: str) -> dict[str, tuple[str, str]]:
        """path -> (mode, blob) as it was in `old`, for each path that differs in `new`.
        A path absent from `old` has mode 000000."""
        out = self._git("diff-tree", "-r", "-z", "--no-renames", "--raw", old, new)
        fields = out.split(b"\0")
        result = {}
        for i in range(0, len(fields) - 1, 2):
            meta, path = fields[i].decode(), fields[i + 1].decode("utf-8", "surrogateescape")
            old_mode, _, old_blob, _, _ = meta.lstrip(":").split(" ")
            result[path] = (old_mode, old_blob)
        return result

    def diff(self, old: str, new: str, paths: list[str]) -> str:
        """A unified diff of these paths between two trees. External diff drivers and textconv are
        off, so nothing the workspace configures runs. `--text` because the store marks every file
        `-text` to keep bytes exact, which git's diff would otherwise read as binary: callers pass
        source files only."""
        if not paths:
            return ""
        out = self._git("diff-tree", "-p", "-r", "--text", "--no-renames", "--no-ext-diff", "--no-textconv",
                        "--no-color", old, new, "--", *paths)
        return out.decode("utf-8", "replace")

    def ignored(self, paths: list[str]) -> set[str]:
        """Which of these workspace-relative paths snapshots can't see."""
        if not paths:
            return set()
        out = self._git("check-ignore", "--no-index", "-z", "--stdin", ok=(0, 1),  # 1: none ignored
                        stdin="\0".join(paths).encode("utf-8", "surrogateescape") + b"\0")
        return {p.decode("utf-8", "surrogateescape") for p in out.split(b"\0") if p}

    def restore(self, start: str, end: str) -> tuple[list[str], list[str]]:
        """Put back what changed between the `start` and `end` trees.

        Returns (reverted, left_alone): left alone are paths that changed again after `end`, which
        is to say by someone else, and paths that now lead outside the workspace.
        """
        changed = self.changes(start, end)
        if not changed:
            return [], []
        now = self.snapshot()
        if now is None:
            raise SnapshotError(self.disabled or "the workspace couldn't be read")
        drifted = set(self.changes(end, now)) & set(changed)
        reverted, left_alone = [], sorted(drifted)
        todo = {p: v for p, v in changed.items() if p not in drifted}
        # Removals first, deepest first, so a file can come back where a folder was made, and so on.
        for path in sorted((p for p, (mode, _) in todo.items() if mode in EMPTY_MODES),
                           key=lambda p: p.count("/"), reverse=True):
            target = self._target(path)
            try:
                if target is None:
                    raise OSError
                if target.is_symlink() or target.is_file():
                    target.unlink()
                reverted.append(path)
            except OSError:
                left_alone.append(path)
        for path, (mode, blob) in sorted(todo.items()):
            if mode in EMPTY_MODES:
                continue
            target = self._target(path)
            try:
                if target is None or mode == GITLINK_MODE:
                    raise OSError
                self._put(target, mode, self._git("cat-file", "blob", blob))
                reverted.append(path)
            except OSError:
                left_alone.append(path)
        return sorted(reverted), sorted(left_alone)

    @staticmethod
    def _put(target: Path, mode: str, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or (mode == LINK_MODE and target.exists()):
            target.unlink()  # never write through a link, wherever it points
        elif target.is_dir() and not any(target.iterdir()):
            target.rmdir()  # a folder the turn made where this file was
        if mode == LINK_MODE:
            os.symlink(os.fsdecode(data), target)
            return
        target.write_bytes(data)  # in place, so the file keeps its permissions
        bits = target.stat().st_mode & 0o777
        want = bits | ((bits & 0o444) >> 2) if mode == EXEC_MODE else bits & ~0o111
        if want != bits:
            target.chmod(want)

    def _target(self, rel: str) -> Path | None:
        """The path inside the workspace, or None if a folder on the way now leads outside it."""
        target = self.workspace / rel
        parent = target.parent
        while not parent.exists() and parent != self.workspace:
            parent = parent.parent
        try:
            parent.resolve().relative_to(self.workspace)
        except ValueError:
            return None
        return target
