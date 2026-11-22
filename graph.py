"""Workspace code dependency graph, PageRank centrality, persistent overlay, and Task DAGs."""

import ast
import json
import os
import re
import tempfile
import threading
from collections import deque
from pathlib import Path

MAX_GRAPH_FILES = 500
MAX_SOURCE_BYTES = 512_000
MAX_SYMBOLS_PER_FILE = 24
MAX_REF_EDGES_PER_FILE = 12
MAX_PINNED_NOTES = 50

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", "target", "out", ".next", ".nuxt", ".turbo",
    "coverage", "vendor", ".idea", ".vscode", ".kyklos",
}

SOURCE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".go", ".rs"}

SAFE_ENV_SUFFIXES = (".example", ".sample", ".template")
SECRET_EXACT_NAMES = {
    ".env", ".npmrc", ".pypirc", ".netrc", ".git-credentials", ".pgpass", ".my.cnf",
    "credentials", "credentials.json", "kubeconfig", "master.passwd", "shadow", "sudoers",
    ".zsh_history", ".bash_history", ".sh_history", ".history",
    ".python_history", ".node_repl_history", ".mysql_history", ".psql_history",
}
SECRET_PREFIXES = ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", "service-account")
SECRET_SUFFIXES = (
    ".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".kdbx", ".ovpn",
    ".tfvars", ".tfstate", ".tfstate.backup", ".keychain", ".keychain-db", ".p8", ".mobileprovision",
)
PROTECTED_CREDENTIAL_DIRS = {".ssh", ".aws", ".gnupg", ".kube", ".docker"}

ANSI_ESCAPE_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[\]PX^_][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-_]"
)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
QUOTED_SPEC_RE = re.compile(r"""["']([^"']+)["']""")
JS_IMPORT_RE = re.compile(
    r"""(?:^\s*(?:import|export)\b.*?from\s+|^\s*import\s+|require\s*\(\s*|import\s*\(\s*)["']([^"']+)["']""",
    re.M,
)
RS_MOD_RE = re.compile(r"^\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", re.M)
RS_USE_CRATE_RE = re.compile(r"^\s*(?:pub\s+)?use\s+crate::([A-Za-z_][A-Za-z0-9_]*)", re.M)
RS_USE_SUPER_RE = re.compile(r"^\s*(?:pub\s+)?use\s+super::([A-Za-z_][A-Za-z0-9_]*)", re.M)

DECL_PATTERNS = {
    "js": [
        re.compile(r"^\s*export\s+(?:default\s+)?(?:abstract\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)", re.M),
        re.compile(r"^\s*export\s+(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_]\w*)", re.M),
        re.compile(r"^\s*export\s+(?:const|let|var|type|interface|enum)\s+([A-Za-z_]\w*)", re.M),
    ],
    "js_fallback": [
        re.compile(r"^(?:async\s+)?function\s+([A-Za-z_]\w*)", re.M),
        re.compile(r"^(?:abstract\s+)?class\s+([A-Za-z_]\w*)", re.M),
        re.compile(r"^(?:const|let)\s+([A-Za-z_]\w*)", re.M),
    ],
    "go": [
        re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", re.M),
        re.compile(r"^\s*type\s+([A-Za-z_]\w*)\b", re.M),
    ],
    "rs": [
        re.compile(r"^\s*pub(?:\([^)]*\))?\s+(?:async\s+|unsafe\s+)?fn\s+([A-Za-z_]\w*)", re.M),
        re.compile(r"^\s*pub(?:\([^)]*\))?\s+(?:struct|enum|trait|type|mod)\s+([A-Za-z_]\w*)", re.M),
    ],
    "rs_fallback": [
        re.compile(r"^(?:async\s+|unsafe\s+)?fn\s+([A-Za-z_]\w*)", re.M),
        re.compile(r"^(?:struct|enum|trait|type|mod)\s+([A-Za-z_]\w*)", re.M),
    ],
}

TEST_CMD_RE = re.compile(
    r"\b(?:pytest|py\.test|unittest|cargo\s+test|go\s+test|npm\s+test|pnpm\s+test|vitest|jest|bun\s+test)\b",
    re.I,
)


def _clean_line(text: str) -> str:
    return " ".join(CONTROL_CHARS_RE.sub(" ", ANSI_ESCAPE_RE.sub("", str(text))).split())


def _is_secret_path(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(SAFE_ENV_SUFFIXES):
        return False
    if name in SECRET_EXACT_NAMES or name.startswith(".env."):
        return True
    if name.endswith(SECRET_SUFFIXES) or name.startswith(SECRET_PREFIXES):
        return True
    parts = {part.lower() for part in path.parts}
    return bool(parts & PROTECTED_CREDENTIAL_DIRS)


def _is_test_rel(rel: str) -> bool:
    file_name = rel.rsplit("/", 1)[-1].lower()
    return (
        file_name.startswith("test_")
        or file_name.endswith(("_test.py", "_test.go", "_test.rs"))
        or ".test." in file_name
        or ".spec." in file_name
    )


def _file_stem_key(rel: str) -> str:
    file_name = rel.rsplit("/", 1)[-1]
    stem = file_name.split(".", 1)[0].lower()
    if stem.startswith("test_"):
        stem = stem[5:]
    if stem.endswith("_test"):
        stem = stem[:-5]
    return stem


def _normalize_rel(base_dir: str, rel_spec: str) -> str | None:
    parts = [p for p in base_dir.split("/") if p] if base_dir else []
    for seg in rel_spec.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts) if parts else None


def _resolve_ts_candidate(candidate: str, known: set[str]) -> str | None:
    if candidate in known:
        return candidate
    trimmed = candidate
    for ext in (".js", ".mjs", ".ts", ".tsx", ".jsx"):
        if trimmed.endswith(ext):
            trimmed = trimmed[: -len(ext)]
            break
    for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", "/index.ts", "/index.tsx", "/index.js", "/index.mjs"):
        full = f"{trimmed}{ext}"
        if full in known:
            return full
    return None


def _python_symbols_and_imports(rel: str, text: str, known: set[str]) -> tuple[list[str], list[str]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return [], []
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            symbols.append(node.name)
        elif isinstance(node, ast.ClassDef):
            symbols.append(node.name)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                    if len(symbols) < MAX_SYMBOLS_PER_FILE:
                        symbols.append(child.name)
        if len(symbols) >= MAX_SYMBOLS_PER_FILE:
            break

    imports: list[str] = []
    cur_dir = rel.rsplit("/", 1)[0] if "/" in rel else ""

    def add_mod(mod_path: str) -> None:
        if not mod_path:
            return
        for suf in (".py", "/__init__.py"):
            cand = f"{mod_path}{suf}"
            if cand != rel and cand in known and cand not in imports:
                imports.append(cand)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_mod(alias.name.replace(".", "/"))
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            mod_name = (node.module or "").replace(".", "/")
            if level > 0:
                parts = [p for p in cur_dir.split("/") if p]
                for _ in range(level - 1):
                    if parts:
                        parts.pop()
                base = "/".join(parts + ([mod_name] if mod_name else []))
            else:
                base = mod_name
            add_mod(base)
            prefix = f"{base}/" if base else ""
            for alias in node.names:
                add_mod(f"{prefix}{alias.name}")

    return list(dict.fromkeys(symbols))[:MAX_SYMBOLS_PER_FILE], imports


def _extract_symbols_and_imports(rel: str, suffix: str, text: str, known: set[str]) -> tuple[list[str], list[str]]:
    if suffix == ".py":
        return _python_symbols_and_imports(rel, text, known)

    cur_dir = rel.rsplit("/", 1)[0] if "/" in rel else ""
    imports: list[str] = []

    def push_imp(target: str | None) -> None:
        if target and target != rel and target in known and target not in imports:
            imports.append(target)

    if suffix in {".ts", ".tsx", ".js", ".jsx", ".mjs"}:
        syms = [m for pat in DECL_PATTERNS["js"] for m in pat.findall(text)]
        if not syms:
            syms = [m for pat in DECL_PATTERNS["js_fallback"] for m in pat.findall(text)]
        for spec in JS_IMPORT_RE.findall(text):
            if spec.startswith("."):
                norm = _normalize_rel(cur_dir, spec)
                if norm:
                    push_imp(_resolve_ts_candidate(norm, known))
            elif spec.startswith("@/"):
                rest = spec[2:]
                for prefix in ("src/", ""):
                    push_imp(_resolve_ts_candidate(f"{prefix}{rest}", known))
        return list(dict.fromkeys(syms))[:MAX_SYMBOLS_PER_FILE], imports

    if suffix == ".rs":
        syms = [m for pat in DECL_PATTERNS["rs"] for m in pat.findall(text)]
        if not syms:
            syms = [m for pat in DECL_PATTERNS["rs_fallback"] for m in pat.findall(text)]
        prefix = f"{cur_dir}/" if cur_dir else ""
        for name in RS_MOD_RE.findall(text):
            for base in (prefix, "src/"):
                push_imp(f"{base}{name}.rs")
                push_imp(f"{base}{name}/mod.rs")
        for seg in RS_USE_CRATE_RE.findall(text):
            for base in ("src/", ""):
                push_imp(f"{base}{seg}.rs")
                push_imp(f"{base}{seg}/mod.rs")
        if cur_dir:
            parent_dir = cur_dir.rsplit("/", 1)[0] if "/" in cur_dir else ""
            parent_prefix = f"{parent_dir}/" if parent_dir else ""
            for seg in RS_USE_SUPER_RE.findall(text):
                push_imp(f"{parent_prefix}{seg}.rs")
                push_imp(f"{parent_prefix}{seg}/mod.rs")
        return list(dict.fromkeys(syms))[:MAX_SYMBOLS_PER_FILE], imports

    if suffix == ".go":
        syms = [m for pat in DECL_PATTERNS["go"] for m in pat.findall(text)]
        in_block = False
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("//"):
                continue
            if line.startswith("import ("):
                in_block = True
                continue
            if in_block and line.startswith(")"):
                in_block = False
                continue
            if in_block or line.startswith("import "):
                for spec in QUOTED_SPEC_RE.findall(line):
                    tail = spec.rstrip("/")
                    last_seg = tail.rsplit("/", 1)[-1]
                    for cand in known:
                        if cand.endswith(".go") and "/" in cand:
                            cand_dir = cand.rsplit("/", 1)[0]
                            if cand_dir == tail or cand_dir.endswith(f"/{tail}") or cand_dir == last_seg:
                                push_imp(cand)
        return list(dict.fromkeys(syms))[:MAX_SYMBOLS_PER_FILE], imports

    return [], []


class WorkspaceGraph:
    """Directed multi-language codebase graph with PageRank centrality and blast radius."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.files: list[str] = []
        self.symbols: dict[str, list[str]] = {}
        self.edges: list[dict[str, str]] = []
        self.outgoing: dict[str, list[str]] = {}
        self.incoming: dict[str, list[str]] = {}
        self.scores: dict[str, float] = {}
        self._build()

    def _collect_files(self) -> list[tuple[str, Path]]:
        collected: list[tuple[str, Path]] = []
        visited_dirs: set[Path] = {self.workspace}
        for root, dirnames, filenames in os.walk(self.workspace, followlinks=False):
            root_path = Path(root)
            kept_dirs = []
            for d in sorted(dirnames):
                if d.startswith(".") or d in SKIP_DIRS:
                    continue
                cand = root_path / d
                try:
                    if cand.is_symlink():
                        continue
                    real = cand.resolve()
                except OSError:
                    continue
                if (real == self.workspace or self.workspace in real.parents) and real not in visited_dirs:
                    visited_dirs.add(real)
                    kept_dirs.append(d)
            dirnames[:] = kept_dirs

            for fname in sorted(filenames):
                cand = root_path / fname
                try:
                    if cand.is_symlink():
                        continue
                    real = cand.resolve()
                    if self.workspace not in real.parents:
                        continue
                    if cand.suffix.lower() not in SOURCE_SUFFIXES or _is_secret_path(cand):
                        continue
                    if real.stat().st_size > MAX_SOURCE_BYTES:
                        continue
                except OSError:
                    continue
                rel = real.relative_to(self.workspace).as_posix()
                if not _is_secret_path(Path(rel)):
                    collected.append((rel, real))
                    if len(collected) >= MAX_GRAPH_FILES:
                        return sorted(collected, key=lambda item: item[0])
        return sorted(collected, key=lambda item: item[0])

    def _build(self) -> None:
        collected = self._collect_files()
        self.files = [rel for rel, _ in collected]
        known = set(self.files)
        contents: dict[str, str] = {}
        symbol_owners: dict[str, list[str]] = {}
        linked: set[tuple[str, str]] = set()

        def add_edge(src: str, dst: str, kind: str) -> None:
            if src == dst or (src, dst) in linked:
                return
            linked.add((src, dst))
            self.edges.append({"source": src, "target": dst, "kind": kind})
            self.outgoing.setdefault(src, []).append(dst)
            self.incoming.setdefault(dst, []).append(src)

        for rel, full_path in collected:
            try:
                text = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            contents[rel] = text
            syms, imps = _extract_symbols_and_imports(rel, full_path.suffix.lower(), text, known)
            self.symbols[rel] = syms
            for sym in syms:
                if len(sym) >= 4:
                    symbol_owners.setdefault(sym, []).append(rel)
            is_test = _is_test_rel(rel)
            for target in imps:
                kind = "tests" if is_test and not _is_test_rel(target) else "imports"
                add_edge(rel, target, kind)

        # Stem-matched test files
        prod_by_stem: dict[str, list[str]] = {}
        for rel in self.files:
            if not _is_test_rel(rel):
                prod_by_stem.setdefault(_file_stem_key(rel), []).append(rel)
        for rel in self.files:
            if _is_test_rel(rel):
                for target in prod_by_stem.get(_file_stem_key(rel), ()):
                    add_edge(rel, target, "tests")

        # Unique cross-file symbol reference edges
        unique_owners = {sym: owners[0] for sym, owners in symbol_owners.items() if len(owners) == 1}
        for rel in self.files:
            text = contents.get(rel, "")
            is_test = _is_test_rel(rel)
            targets = sorted({
                unique_owners[tok]
                for tok in IDENT_RE.findall(text)
                if tok in unique_owners and unique_owners[tok] != rel and (rel, unique_owners[tok]) not in linked
            })
            for target in targets[:MAX_REF_EDGES_PER_FILE]:
                kind = "tests" if is_test and not _is_test_rel(target) else "references"
                add_edge(rel, target, kind)

        self.scores = self._compute_pagerank()

    def _compute_pagerank(self, damping: float = 0.85, iterations: int = 15) -> dict[str, float]:
        n = len(self.files)
        if n == 0:
            return {}
        scores = {rel: 1.0 / n for rel in self.files}
        base = (1.0 - damping) / n
        for _ in range(iterations):
            next_scores = {rel: base for rel in self.files}
            dangling = 0.0
            for src in self.files:
                targets = self.outgoing.get(src)
                if not targets:
                    dangling += scores[src]
                else:
                    share = damping * scores[src] / len(targets)
                    for dst in targets:
                        if dst in next_scores:
                            next_scores[dst] += share
            dangling_share = damping * dangling / n
            for rel in next_scores:
                next_scores[rel] += dangling_share
            scores = next_scores
        peak = max(scores.values(), default=0.0)
        if peak > 0.0:
            scores = {rel: round(val / peak, 3) for rel, val in scores.items()}
        return scores

    def _edge_kind(self, src: str, dst: str) -> str:
        for e in self.edges:
            if e["source"] == src and e["target"] == dst:
                return e["kind"]
        return "references"

    def blast_radius(self, targets: list[str] | str, depth: int = 2) -> dict:
        if isinstance(targets, str):
            targets = [targets]
        roots = [t for t in targets if t in set(self.files)]
        dependents: list[dict] = []
        tests: list[str] = []
        seen = set(roots)
        queue = deque((r, 0) for r in roots)
        while queue:
            cur, hop = queue.popleft()
            if hop >= depth:
                continue
            for dep in sorted(self.incoming.get(cur, [])):
                if dep in seen:
                    continue
                seen.add(dep)
                kind = self._edge_kind(dep, cur)
                if kind == "tests" or _is_test_rel(dep):
                    tests.append(dep)
                else:
                    dependents.append({"path": dep, "hops": hop + 1, "kind": kind})
                queue.append((dep, hop + 1))
        return {
            "targets": roots,
            "dependents": sorted(dependents, key=lambda d: (d["hops"], d["path"])),
            "tests": sorted(set(tests)),
        }

    def neighbors(self, target: str) -> dict:
        if target not in set(self.files):
            return {"found": False, "path": target}
        outgoing = [
            {"path": dst, "kind": self._edge_kind(target, dst)}
            for dst in sorted(self.outgoing.get(target, []))
        ]
        incoming = [
            {"path": src, "kind": self._edge_kind(src, target)}
            for src in sorted(self.incoming.get(target, []))
        ]
        return {
            "found": True,
            "path": target,
            "score": self.scores.get(target, 0.0),
            "symbols": self.symbols.get(target, []),
            "imports": [e["path"] for e in outgoing if e["kind"] != "tests"],
            "dependents": [e["path"] for e in incoming if e["kind"] != "tests"],
            "tests": sorted({
                e["path"] for e in (*outgoing, *incoming) if e["kind"] == "tests" or _is_test_rel(e["path"])
            }),
        }

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        known = set(self.files)
        if source not in known or target not in known:
            return None
        if source == target:
            return [source]
        # Undirected + directed BFS for dependency connection between two files
        for directed in (True, False):
            queue = deque([[source]])
            visited = {source}
            while queue:
                chain = queue.popleft()
                cur = chain[-1]
                nbrs = list(self.outgoing.get(cur, []))
                if not directed:
                    nbrs.extend(self.incoming.get(cur, []))
                for nxt in sorted(set(nbrs)):
                    if nxt == target:
                        return chain + [nxt]
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(chain + [nxt])
        return None

    def ranked_files(self) -> list[str]:
        """Sort files promoting modules imported/depended on by others using PageRank,
        while preserving shallow-first alphabetical order among unlinked files."""
        return sorted(
            self.files,
            key=lambda rel: (
                -1 if self.incoming.get(rel) else 0,
                -self.scores.get(rel, 0.0) if self.incoming.get(rel) else 0.0,
                rel.count("/"),
                rel,
            ),
        )

    def query(self, action: str, target: str = "", to: str = "", depth: int = 2,
              overlay: "KnowledgeOverlay | None" = None, note: str = "") -> str:
        if action == "summary":
            if not self.files:
                return "No source files found in this workspace."
            ranked = sorted(self.files, key=lambda r: (-self.scores.get(r, 0.0), r.count("/"), r))
            lines = [f"Workspace graph: {len(self.files)} files, {len(self.edges)} edges. Top central files:"]
            for rel in ranked[:15]:
                br = len(self.blast_radius(rel, depth=2)["dependents"])
                syms = ", ".join(self.symbols.get(rel, [])[:6])
                lines.append(f"  {rel} (rank {self.scores.get(rel, 0.0):.2f}, {br} dependents)"
                             + (f": {syms}" if syms else ""))
            return "\n".join(lines)

        if action == "note":
            if overlay is None:
                return "Persistent overlay is unavailable for this session."
            if not target.strip() or not note.strip():
                return "Action 'note' requires both target path and note text."
            overlay.add_note(target, note)
            return f"Pinned note on {target}: {_clean_line(note)}"

        if not target or target not in set(self.files):
            matches = [f for f in self.files if target and target.lower() in f.lower()]
            hint = f" Did you mean: {', '.join(matches[:5])}?" if matches else ""
            return f"File not in code graph: {target!r}.{hint}"

        if action == "impact":
            info = self.blast_radius([target], depth=depth)
            deps = [f"{d['path']} (hop {d['hops']}, {d['kind']})" for d in info["dependents"]]
            lines = [f"Blast radius for {target} (rank {self.scores.get(target, 0.0):.2f}, depth {depth}):"]
            lines.append("  Dependents: " + (", ".join(deps[:25]) if deps else "none"))
            if len(deps) > 25:
                lines.append(f"  ... ({len(deps) - 25} more dependents)")
            lines.append("  Tests: " + (", ".join(info["tests"]) if info["tests"] else "none linked"))
            return "\n".join(lines)

        if action == "neighbors":
            info = self.neighbors(target)
            lines = [f"Neighbors of {target} (rank {info['score']:.2f}):"]
            if info["symbols"]:
                lines.append("  Symbols: " + ", ".join(info["symbols"][:12]))
            lines.append("  Imports: " + (", ".join(info["imports"]) or "none"))
            lines.append("  Imported/referenced by: " + (", ".join(info["dependents"]) or "none"))
            lines.append("  Tests: " + (", ".join(info["tests"]) or "none"))
            return "\n".join(lines)

        if action == "tests":
            info = self.blast_radius([target], depth=depth)
            if not info["tests"]:
                return f"No linked tests found for {target}."
            return f"Linked tests for {target}: {', '.join(info['tests'])}"

        if action == "path":
            if not to or to not in set(self.files):
                return f"Target 'to' file not in code graph: {to!r}."
            chain = self.shortest_path(target, to)
            if not chain:
                return f"No dependency connection found between {target} and {to}."
            return " -> ".join(chain)

        return f"Unknown code_graph action: {action!r}"

    def to_dict(self, overlay: "KnowledgeOverlay | None" = None) -> dict:
        ov_files = overlay.files if overlay else {}
        notes_by_path: dict[str, list[str]] = {}
        if overlay:
            for item in overlay.notes:
                if item.get("path") and item.get("note"):
                    notes_by_path.setdefault(item["path"], []).append(item["note"])

        edited = [p for p, meta in ov_files.items() if meta.get("state") == "edited"]
        impacted = {d["path"] for d in self.blast_radius(edited, depth=2)["dependents"]} - set(edited)

        nodes = []
        for rel in self.files:
            meta = ov_files.get(rel, {})
            br = len(self.blast_radius([rel], depth=2)["dependents"])
            nodes.append({
                "path": rel,
                "symbols": self.symbols.get(rel, []),
                "score": self.scores.get(rel, 0.0),
                "blast_radius": br,
                "state": meta.get("state", "untouched"),
                "test_status": meta.get("test_status"),
                "notes": notes_by_path.get(rel, []),
                "is_test": _is_test_rel(rel),
            })
        nodes.sort(key=lambda n: (-n["score"], n["path"]))
        return {
            "nodes": nodes,
            "edges": list(self.edges),
            "summary": {
                "file_count": len(nodes),
                "edge_count": len(self.edges),
                "edited_count": len([n for n in nodes if n["state"] == "edited"]),
                "impacted_count": len(impacted),
            },
        }


def _ensure_private_dir(path: Path) -> None:
    missing = []
    cur = path
    while not cur.exists() and cur != cur.parent:
        missing.append(cur)
        cur = cur.parent
    path.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        for d in (*reversed(missing), path):
            try:
                d.chmod(0o700)
            except OSError:
                pass


def _write_private_json_atomic(path: Path, payload: dict) -> None:
    if os.path.islink(path.parent) or os.path.islink(path):
        raise OSError(f"Refusing to write through symlink near {path}")
    _ensure_private_dir(path.parent)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if os.name == "nt":
        path.write_text(text, encoding="utf-8")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd, tmp_name = tempfile.mkstemp(prefix=".graph-", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


class KnowledgeOverlay:
    """Session and cross-session architectural memory overlaid onto the code graph."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = threading.Lock()
        self.files: dict[str, dict] = {}
        self.notes: list[dict[str, str]] = []
        if self.path is not None:
            self.load()

    def load(self) -> None:
        if self.path is None or not self.path.is_file() or self.path.is_symlink():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        files = data.get("files")
        if isinstance(files, dict):
            cleaned: dict[str, dict] = {}
            for k, v in files.items():
                if isinstance(k, str) and isinstance(v, dict):
                    entry: dict = {}
                    if v.get("state") in ("read", "edited"):
                        entry["state"] = v["state"]
                    if v.get("test_status") in ("pass", "fail"):
                        entry["test_status"] = v["test_status"]
                    if isinstance(v.get("symbols"), list):
                        entry["symbols"] = [str(s) for s in v["symbols"][:MAX_SYMBOLS_PER_FILE]]
                    if entry:
                        cleaned[k] = entry
            self.files = cleaned
        notes = data.get("notes")
        if isinstance(notes, list):
            self.notes = [
                {"path": str(n["path"]), "note": _clean_line(str(n["note"]))}
                for n in notes
                if isinstance(n, dict) and n.get("path") and n.get("note")
            ][-MAX_PINNED_NOTES:]

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            payload = {"files": self.files, "notes": self.notes}
            try:
                _write_private_json_atomic(self.path, payload)
            except OSError:
                pass

    def record_read(self, rel_path: str) -> None:
        with self._lock:
            entry = self.files.setdefault(rel_path, {})
            if entry.get("state") != "edited":
                entry["state"] = "read"
        self.save()

    def record_edit(self, rel_path: str, symbols: list[str] | tuple[str, ...] = ()) -> None:
        with self._lock:
            entry = self.files.setdefault(rel_path, {})
            entry["state"] = "edited"
            if symbols:
                entry["symbols"] = list(symbols)[:MAX_SYMBOLS_PER_FILE]
        self.save()

    def record_test_run(self, command: str, succeeded: bool, touched_paths: list[str] = ()) -> None:
        if not TEST_CMD_RE.search(command or ""):
            return
        status = "pass" if succeeded else "fail"
        with self._lock:
            targets = list(touched_paths) or [
                p for p, meta in self.files.items() if meta.get("state") == "edited" or _is_test_rel(p)
            ]
            for rel in targets:
                self.files.setdefault(rel, {})["test_status"] = status
        self.save()

    def add_note(self, rel_path: str, note: str) -> None:
        clean = _clean_line(note)
        if not clean or not rel_path.strip():
            return
        entry = {"path": rel_path.strip(), "note": clean[:240]}
        with self._lock:
            if entry not in self.notes:
                self.notes.append(entry)
                if len(self.notes) > MAX_PINNED_NOTES:
                    self.notes = self.notes[-MAX_PINNED_NOTES:]
        self.save()

    def clear_session_states(self) -> None:
        with self._lock:
            self.files.clear()
        self.save()

    def compaction_brief(self, graph: WorkspaceGraph, touched_files: list[str]) -> str:
        parts: list[str] = []
        if touched_files:
            info = graph.blast_radius(touched_files, depth=2)
            deps = [d["path"] for d in info["dependents"][:15]]
            tests = info["tests"][:10]
            if deps:
                parts.append("Code graph blast radius (dependents of edited files): " + ", ".join(deps))
            if tests:
                parts.append("Linked test files for edited modules: " + ", ".join(tests))
        failing = sorted(p for p, m in self.files.items() if m.get("test_status") == "fail")
        if failing:
            parts.append("Files with failing test status in overlay: " + ", ".join(failing[:10]))
        if self.notes:
            note_lines = [f"- {n['path']}: {n['note']}" for n in self.notes[-8:]]
            parts.append("Pinned codebase notes:\n" + "\n".join(note_lines))
        return "\n".join(parts)


def validate_dag(nodes: list[dict]) -> list[list[int]]:
    """Validate DAG node ids and depends_on links, returning topological wave index groups.

    Raises ValueError on duplicate/empty ids, unknown dependency ids, self-loops, or cycles.
    """
    if not nodes:
        raise ValueError("DAG must contain at least one node")
    id_to_idx: dict[str, int] = {}
    deps_by_idx: list[list[int]] = []

    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"node {idx} must be an object")
        nid = node.get("id")
        if not isinstance(nid, str) or not nid.strip():
            raise ValueError(f"node {idx} is missing a non-empty string 'id'")
        nid = _clean_line(nid)
        if nid in id_to_idx:
            raise ValueError(f"duplicate step id {nid!r}")
        id_to_idx[nid] = idx

    for idx, node in enumerate(nodes):
        raw_deps = node.get("depends_on", [])
        if raw_deps is None:
            raw_deps = []
        if not isinstance(raw_deps, list) or not all(isinstance(d, str) and d.strip() for d in raw_deps):
            raise ValueError(f"depends_on for {node['id']!r} must be a list of step id strings")
        dep_indices: list[int] = []
        for dep in raw_deps:
            dep_clean = _clean_line(dep)
            if dep_clean not in id_to_idx:
                raise ValueError(f"unknown dependency {dep_clean!r} in depends_on for {node['id']!r}")
            dep_idx = id_to_idx[dep_clean]
            if dep_idx == idx:
                raise ValueError(f"cycle detected in depends_on: {node['id']!r} depends on itself")
            if dep_idx not in dep_indices:
                dep_indices.append(dep_idx)
        deps_by_idx.append(dep_indices)

    in_degree = [len(deps) for deps in deps_by_idx]
    children: list[list[int]] = [[] for _ in nodes]
    for idx, deps in enumerate(deps_by_idx):
        for dep_idx in deps:
            children[dep_idx].append(idx)

    waves: list[list[int]] = []
    ready = [i for i, deg in enumerate(in_degree) if deg == 0]
    visited = 0
    while ready:
        waves.append(ready)
        visited += len(ready)
        next_ready = []
        for parent in ready:
            for child in children[parent]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_ready.append(child)
        ready = next_ready

    if visited != len(nodes):
        stuck = [nodes[i]["id"] for i, deg in enumerate(in_degree) if deg > 0]
        raise ValueError(f"cycle detected in depends_on involving: {', '.join(stuck)}")
    return waves
