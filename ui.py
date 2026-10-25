"""Kyklos Aero: the inline terminal UI (DESIGN.md is the spec, design/aero_preview.py the reference).

The Agent doesn't print anything; it emits events. AeroUI turns them into output:
  - permanent lines (finished tool calls, notices, answers) printed into scrollback;
  - a transient "live zone" at the bottom (spinner, running tool, subagent block) that
    animates in place and disappears when the work is done.
"""

import difflib
import json
import os
import re
import shutil
import sys
import time
from collections import deque
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Heading, Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from kyklos import (BROAD_WARNING, EXIT, NO_APPROVER_NOTE, COMMANDS, TRUST_EXPLANATION, CommandContext,
                    apply_permission_answer, denial_note, permission_options, run_command, short_tokens)

BASH_FULL_OUTPUT_LINES = 8  # bash output up to this many lines is shown in full; longer shows the last 3

# ---- tokens (DESIGN.md §2) ------------------------------------------------------------------

AERO = {
    "sky": "#3BB8FF", "aqua": "#7FE7FF", "deep": "#0A6FD6", "mint": "#4FF0C1", "glint": "#C2FFEC",
    "sun": "#FFC83D", "coral": "#FF5A64", "orchid": "#C38BFF", "frost": "#8FA9BF",
}
PLAIN = {  # KYKLOS_THEME=plain: the 16 named ANSI colors
    "sky": "cyan", "aqua": "bright_cyan", "deep": "blue", "mint": "green", "glint": "bright_green",
    "sun": "yellow", "coral": "red", "orchid": "magenta", "frost": "bright_black",
}

# Diff blocks (DESIGN.md §4): a glass sheet per changed line.
DIFF_STYLES = {
    "add": {"sheen": ("#11564A", "#0A2A25"), "text": "#E4FFF7", "glow": "#21A383", "rim": "#C2FFEC", "mark": "+"},
    "del": {"sheen": ("#561A27", "#2B0D14"), "text": "#FFE0E3", "glow": "#B0344A", "rim": "#FF5A64", "mark": "−"},
    "mod": {"sheen": ("#12436A", "#0A2139"), "text": "#E9F8FF", "glow": "#2A86D6", "rim": "#7FE7FF", "mark": "~"},
}
PLAIN_DIFF = {"add": "green", "del": "red", "mod": "cyan"}

# ---- glyphs (DESIGN.md §3) ------------------------------------------------------------------

UNICODE = {"run": "○", "done": "●", "fail": "✗", "denied": "⊘", "compact": "◌", "retry": "↻", "prompt": "❯",
           "tee": "├", "elbow": "└", "pipe": "│", "dot": "·", "arrow": "→", "on": "▰", "off": "▱", "rim": "▌",
           "rule": "─", "minus": "−", "ellipsis": "…"}
ASCII = {"run": "o", "done": "*", "fail": "x", "denied": "-", "compact": "~", "retry": "r", "prompt": ">",
         "tee": "|", "elbow": "`", "pipe": "|", "dot": "-", "arrow": "->", "on": "#", "off": ".", "rim": "|",
         "rule": "-", "minus": "-", "ellipsis": "..."}
BUBBLE = [("·", "deep"), ("∘", "sky"), ("○", "sky"), ("◯", "aqua"), ("○", "sky"), ("∘", "sky")]
ASCII_BUBBLE = [(".", "deep"), ("o", "sky"), ("O", "aqua"), ("o", "sky")]

#   KYKLOS in slant Figlet. Each row is padded to the same width, so the block reads as a
#   rectangle here and the gradient's x axis spans it evenly. Keep the one-column-per-row
#   shear (4, 3, 2, 1, 0 leading spaces): it's what makes the letters sit on a diagonal.
LARGE_WORDMARK_LINES = [
    r"    __ ____  ____ __ __    ____  _____",
    r"   / //_/\ \/ / //_// /   / __ \/ ___/",
    r"  / ,<    \  / ,<  / /   / / / /\__ \ ",
    r" / /| |   / / /| |/ /___/ /_/ /___/ / ",
    r"/_/ |_|  /_/_/ |_/_____/\____//____/  ",
]
LARGE_WORDMARK_WIDTH = max(len(line) for line in LARGE_WORDMARK_LINES)


def build_theme(colors: dict) -> Theme:
    styles = dict(colors)
    styles.update({
        "markdown.code": colors["aqua"], "markdown.code_block": "", "markdown.link": colors["sky"],
        "markdown.link_url": colors["frost"], "markdown.item.bullet": colors["sky"],
        "markdown.item.number": colors["sky"], "markdown.block_quote": colors["frost"],
        "markdown.hr": colors["frost"], "markdown.h1": f"bold {colors['sky']}", "markdown.h2": f"bold {colors['sky']}",
        "markdown.h3": "bold", "markdown.h4": "bold", "markdown.h5": "bold", "markdown.h6": "bold",
    })
    return Theme(styles)


class AeroHeading(Heading):
    """Modest headings: left-aligned and bold, with no centered h1 (DESIGN.md §4, Answer)."""
    LEVEL_ALIGN = {f"h{i}": "left" for i in range(1, 7)}


class AeroMarkdown(Markdown):
    elements = {**Markdown.elements, "heading_open": AeroHeading}


# ---- color helpers -------------------------------------------------------------------------

def _rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(a: str, b: str, t: float) -> str:
    """Blend two hex colors; t=0 gives a, t=1 gives b."""
    ra, rb = _rgb(a), _rgb(b)
    return "#" + "".join(f"{round(ra[k] + (rb[k] - ra[k]) * t):02X}" for k in range(3))


def gradient_text(text: str, stops: list, bold: bool = False) -> Text:
    out = Text()
    n = max(len(text) - 1, 1)
    for i, ch in enumerate(text):
        t = i / n * (len(stops) - 1)
        k = min(int(t), len(stops) - 2)
        out.append(ch, style=("bold " if bold else "") + mix(stops[k], stops[k + 1], t - k))
    return out


# ---- diffs (DESIGN.md §4, Diff blocks) -----------------------------------------------------

def diff_ops(old: str, new: str, first_line: int = 1) -> list:
    """(kind, lineno, text, spans) rows; kind is ctx/add/del/mod. A modified line is a del row (old,
    with deleted characters as spans) followed by a mod row (new, with inserted characters as spans)."""
    a, b = old.splitlines(), new.splitlines()
    rows = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op == "equal":
            rows += [("ctx", first_line + j1 + k, a[i1 + k], ()) for k in range(i2 - i1)]
            continue
        olds, news = a[i1:i2], b[j1:j2]
        pairs, next_new = [], 0  # each old line pairs with the next similar-enough new line, in order
        for oi, o in enumerate(olds):
            for nj in range(next_new, len(news)):
                if difflib.SequenceMatcher(None, o, news[nj]).ratio() > 0.6:
                    pairs.append((oi, nj))
                    next_new = nj + 1
                    break
        oi = nj = 0
        for po, pn in pairs + [(len(olds), len(news))]:
            rows += [("del", first_line + i1 + k, olds[k], ()) for k in range(oi, po)]
            rows += [("add", first_line + j1 + k, news[k], ()) for k in range(nj, pn)]
            if po < len(olds):
                ops = difflib.SequenceMatcher(None, olds[po], news[pn]).get_opcodes()
                rows.append(("del", first_line + i1 + po, olds[po],
                             tuple((x1, x2) for t, x1, x2, _, _ in ops if t in ("replace", "delete"))))
                rows.append(("mod", first_line + j1 + pn, news[pn],
                             tuple((y1, y2) for t, _, _, y1, y2 in ops if t in ("replace", "insert"))))
            oi, nj = po + 1, pn + 1
    return rows


def diff_counts(old: str, new: str) -> tuple:
    """(added, deleted, modified) line counts, where a modified pair counts once as modified."""
    rows = diff_ops(old, new)
    mods = sum(1 for r in rows if r[0] == "mod")
    return (sum(1 for r in rows if r[0] == "add"), sum(1 for r in rows if r[0] == "del") - mods, mods)


def trim_context(rows: list, keep: int = 2) -> list:
    """Collapse unchanged runs that are more than `keep` lines away from any change into one gap row."""
    changed = [i for i, r in enumerate(rows) if r[0] != "ctx"]
    if not changed:
        return rows
    near = {j for i in changed for j in range(i - keep, i + keep + 1)}
    out, gap = [], 0
    for i, r in enumerate(rows):
        if r[0] == "ctx" and i not in near:
            gap += 1
            continue
        if gap:
            out.append(("gap", 0, f"{gap} unchanged lines", ()))
            gap = 0
        out.append(r)
    if gap:
        out.append(("gap", 0, f"{gap} unchanged lines", ()))
    return out


# ---- tool call and result summaries (DESIGN.md §4, Tool call line) -------------------------

def _plural(n: int, word: str) -> str:
    if n == 1:
        return f"{n} {word}"
    return f"{n} {word}es" if word.endswith(("ch", "sh", "s", "x")) else f"{n} {word}s"


def describe_call(name: str, args: dict) -> str:
    """Human-readable arguments for a tool call line."""
    if name == "read_file":
        return str(args.get("path", ""))
    if name == "grep":
        where = args.get("path") or "."
        text = f"/{args.get('pattern', '')}/" + (f" in {where}" if where != "." else "")
        return text + (f"  {args['glob']}" if args.get("glob") else "")
    if name == "glob":
        return str(args.get("pattern", ""))
    if name in ("write_file", "edit_file"):
        return str(args.get("path", ""))
    if name == "bash":
        return "$ " + str(args.get("command", "")).strip().splitlines()[0] if args.get("command") else "$"
    if name == "task":
        return f"{args.get('mode') or 'explore'} · {args.get('description', '')}"
    return json.dumps(args, ensure_ascii=False)[:120]


def result_status(result: str) -> str:
    if result.startswith("Error:"):
        return "fail"
    if result == "Interrupted by the user.":
        return "interrupted"
    if result.startswith("The user denied") or "call was denied" in result[:80]:
        return "denied"
    return "ok"


def summarize_result(name: str, args: dict, result: str) -> tuple:
    """(status, summary, preview_lines, range_label) for a finished tool call."""
    status = result_status(result)
    if status == "fail":
        return status, result[len("Error:"):].strip().splitlines()[0], [], ""
    if status == "denied":
        # The reason is already shown on the permission panel's "⊘ denied: …" line; don't repeat it here.
        return status, "denied: no one to approve" if "nobody is available" in result else "denied by you", [], ""
    if status == "interrupted":
        return status, "interrupted", [], ""
    if name == "read_file":
        numbers = re.findall(r"^\s*(\d+)\t", result, re.M)
        span = f"{numbers[0]}–{numbers[-1]}" if numbers else ""
        return status, _plural(len(numbers), "line"), [], span
    if name == "grep":
        if result.startswith("No matches"):
            return status, "no matches", [], ""
        hits = re.findall(r"^(.+?):\d+: ", result, re.M)
        return status, f"{_plural(len(hits), 'match')} in {_plural(len(set(hits)), 'file')}", [], ""
    if name == "glob":
        if result.startswith("No files"):
            return status, "no files", [], ""
        return status, _plural(len([l for l in result.splitlines() if l and not l.startswith("...")]), "file"), [], ""
    if name == "write_file":
        return status, "✓ written", [], ""
    if name == "edit_file":
        return status, "✓ applied", [], ""
    if name == "bash":
        first, _, rest = result.partition("\n")
        code = first.replace("exit code", "").strip()
        lines = [l for l in rest.splitlines() if l.strip()]
        if len(lines) > BASH_FULL_OUTPUT_LINES:  # long output: the end matters most for commands
            lines = [f"… {len(lines) - 3} earlier lines"] + lines[-3:]
        return ("ok" if code == "0" else "exitfail"), f"exit {code}", lines, ""
    return status, "", [], ""


# ---- picker state --------------------------------------------------------------------------

class PickerState:
    """What the picker shows and where the cursor is; kept apart from drawing so it can be tested."""

    def __init__(self, rows: list, cursor: int = 0, marked: int = None, filterable: bool = False):
        self.rows, self.cursor, self.marked, self.filterable = rows, cursor, marked, filterable
        self.filter = ""

    def matches(self) -> list[int]:
        needle = self.filter.lower()
        return [i for i, row in enumerate(self.rows) if needle in " ".join(map(str, row)).lower()]

    def move(self, delta: int) -> None:
        matches = self.matches()
        if matches:
            pos = matches.index(self.cursor) if self.cursor in matches else 0
            self.cursor = matches[(pos + delta) % len(matches)]

    def type(self, text: str) -> None:
        self.filter += text
        self._snap()

    def backspace(self) -> None:
        self.filter = self.filter[:-1]
        self._snap()

    def _snap(self) -> None:
        matches = self.matches()
        if matches and self.cursor not in matches:
            self.cursor = matches[0]

    def selected(self):
        return self.cursor if self.cursor in self.matches() else None


# ---- keyboard ------------------------------------------------------------------------------

def read_key_default():
    """One keypress without Enter. Returns None when there's no interactive keyboard."""
    if not sys.stdin.isatty():
        return None
    if os.name == "nt":
        import msvcrt
        while True:
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # arrow / function keys come as two codes; ignore them
                msvcrt.getwch()
                continue
            return ch
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def edit_line_default(message: str, default: str = "") -> str:
    from prompt_toolkit import prompt
    try:
        return prompt(message, default=default)
    except (EOFError, KeyboardInterrupt):
        return ""


# ---- the UI --------------------------------------------------------------------------------

class AeroUI:
    def __init__(self, workspace, console: Console = None, theme: str = None, ascii_mode: bool = None,
                 animate: bool = True, read_key=None, edit_line=None):
        self.workspace = Path(workspace).resolve()
        self.theme_name = (theme or os.environ.get("KYKLOS_THEME", "aero")).lower()
        self.plain = self.theme_name == "plain"
        self.colors = PLAIN if self.plain else AERO
        if console is None:
            console = Console(theme=build_theme(self.colors), highlight=False)
        else:
            console.push_theme(build_theme(self.colors))
        self.console = console
        encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
        if ascii_mode is None:
            ascii_mode = os.environ.get("KYKLOS_ASCII") == "1" or "utf" not in encoding
        self.ascii = ascii_mode
        self.g = ASCII if ascii_mode else UNICODE
        self.animate = animate
        self.read_key = read_key or read_key_default
        self.edit_line = edit_line or edit_line_default
        self.agent = None
        self._live = None
        self._thinking: dict[int, tuple] = {}   # depth -> (start time, purpose)
        self._running: dict[int, tuple] = {}    # depth -> (tool name, label, start time)
        self._pending: dict[int, dict] = {}     # depth -> info about the call in progress
        self._subs: list[dict] = []             # running subagents, innermost last
        self._last_sub = None
        self.turn_tools = 0

    # -- layout helpers --

    @property
    def width(self) -> int:
        """The full terminal width, read on every use so the layout follows window resizes."""
        return self.console.width

    def _c(self, token: str) -> str:
        return self.colors[token]

    def _line(self, left: Text, right: Text = None, indent: int = 2) -> Text:
        """A row with an optional summary aligned to the right edge of the terminal. When space runs out, the
        summary is shortened first (to at most ~35% of the row), so the tool name and its target stay readable."""
        right = (right or Text()).copy()
        left = left.copy()
        available = self.width - indent
        if left.cell_len + 2 + right.cell_len > available:
            right_max = max(12, int(available * 0.35))
            if right.cell_len > right_max:
                right.truncate(right_max, overflow="ellipsis")
        room = available - right.cell_len - (2 if right.cell_len else 0)
        if left.cell_len > room:
            left.truncate(max(room, 8), overflow="ellipsis")
        gap = self.width - indent - left.cell_len - right.cell_len
        return Text(" " * indent) + left + Text(" " * max(gap, 1)) + right

    def _tool_text(self, glyph: str, glyph_style: str, name: str, label: str, extra: Text = None) -> Text:
        t = Text(glyph, style=glyph_style) + Text(" ") + Text(name, style="bold")
        if label:
            t += Text("  ") + Text(label, style="frost")
        if extra:
            t += Text("  ") + extra
        return t

    def _counts(self, added: int, deleted: int, modified: int) -> Text:
        parts = [(f"+{added}", "mint") if added else None, (f"{self.g['minus']}{deleted}", "coral") if deleted else None,
                 (f"~{modified}", "aqua") if modified else None]
        out = Text()
        for p in [p for p in parts if p]:
            out += Text((" " if out else "") + p[0], style=p[1])
        return out or Text("no changes", style="frost")

    def _bubble(self) -> tuple:
        frames = ASCII_BUBBLE if self.ascii else BUBBLE
        return frames[int(time.monotonic() * 6) % len(frames)]

    def print(self, renderable="", indent: int = 0) -> None:
        if indent:
            renderable = Padding(renderable, (0, 0, 0, indent))
        self.console.print(renderable, width=self.width)

    # -- the live zone --

    def _refresh(self) -> None:
        if not self.animate:
            return
        needed = bool(self._thinking or self._subs or any(r[0] != "task" for r in self._running.values()))
        if needed and self._live is None:
            self._live = Live(get_renderable=self._render_live, console=self.console,
                              refresh_per_second=12, transient=True)
            self._live.start()
        elif not needed and self._live is not None:
            self._live.stop()
            self._live = None

    def _pause_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def reset_transient(self) -> None:
        self._pause_live()
        self._thinking.clear()
        self._running.clear()
        self._pending.clear()
        self._subs.clear()

    def _running_row(self, entry: tuple) -> tuple:
        """(left, right) for a tool that's still running, with a timer once it passes a second,
        so a slow command looks busy instead of frozen."""
        name, label, start = entry
        secs = int(time.monotonic() - start)
        return self._tool_text(self.g["run"], "aqua", name, label), (Text(f"{secs}s", style="frost") if secs else None)

    def _render_live(self):
        rows = []
        running = self._running.get(0)
        if running and running[0] != "task":
            rows.append(self._line(*self._running_row(running)))
        for i, sub in enumerate(self._subs):
            depth = i + 1
            indent = 2 + 4 * i
            glyph, color = self._bubble()
            secs = int(time.monotonic() - sub["start"])
            head = Text(glyph, style=color) + Text(" ") + Text("task", style="bold") + Text("  ") + \
                Text(sub["mode"], style="orchid") + Text(f" {self.g['dot']} {sub['description']}", style="frost")
            rows.append(self._line(head, Text(f"{secs}s", style="frost"), indent))
            items = list(sub["steps"])  # (left, right) pairs of finished steps
            now = self._running.get(depth)
            if now:
                items.append(self._running_row(now))
            elif depth in self._thinking:
                items.append((Text(f"{self.g['dot']} thinking", style="frost"), None))
            for k, (left, right) in enumerate(items):
                branch = self.g["elbow"] if k == len(items) - 1 else self.g["tee"]
                rows.append(self._line(Text(f"{branch} ", style="orchid") + left, right, indent=indent + 2))
        if 0 in self._thinking and not self._subs:
            start, purpose = self._thinking[0]
            glyph, color = self._bubble()
            what = "compacting context" if purpose == "summary" else "thinking"
            rows.append(Text("  ") + Text(glyph, style=color) +
                        Text(f" {what} {self.g['dot']} {int(time.monotonic() - start)}s "
                             f"{self.g['dot']} ctrl+c to interrupt", style="frost"))
        return Group(*rows)

    # -- events from the Agent --

    def observer(self, event: str, depth: int = 0, **data) -> None:
        handler = getattr(self, f"_on_{event}", None)
        if handler:
            handler(depth, **data)
        self._refresh()

    def _on_model_start(self, depth, purpose="turn", **_):
        self._thinking[depth] = (time.monotonic(), purpose)

    def _on_model_end(self, depth, **_):
        self._thinking.pop(depth, None)

    def _on_thought(self, depth, text="", **_):
        """Progress notes the model writes between tool calls: a kyklos box with a deep border and frost text,
        quieter than the final answer's box."""
        if depth == 0 and text.strip():
            self.print(self.kyklos_box(AeroMarkdown(text.strip(), style="frost", code_theme="ansi_dark"),
                                       border="deep"), indent=2)

    def _on_tool_call(self, depth, name="", arguments="", **_):
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        info = {"name": name, "args": args, "panel_shown": False}
        if name in ("edit_file", "write_file"):  # capture "before" now: the tool overwrites it
            info.update(self._diff_source(name, args))
        self._pending[depth] = info
        self._running[depth] = (name, describe_call(name, args), time.monotonic())

    def _on_tool_result(self, depth, name="", result="", **_):
        self._running.pop(depth, None)
        info = self._pending.pop(depth, {"args": {}, "panel_shown": False})
        args = info["args"]
        status, summary, tail, span = summarize_result(name, args, result)
        label = describe_call(name, args) + (f"  {span}" if span else "")
        if name == "task" and self._last_sub and status == "ok":
            summary = f"{_plural(self._last_sub['steps'], 'step')} {self.g['dot']} {self._last_sub['secs']:.0f}s"
        glyph, glyph_style, summary_style = {
            "ok": (self.g["done"], "sky", "mint" if summary.startswith("✓") or summary == "exit 0" else "frost"),
            "exitfail": (self.g["done"], "sky", "coral"),
            "fail": (self.g["fail"], "coral", "coral"),
            "denied": (self.g["denied"], "coral", "coral"),
            "interrupted": (self.g["denied"], "sun", "sun"),
        }[status]
        if self.ascii:
            summary = summary.replace("✓ ", "")
        if name == "task":
            left = self._tool_text(glyph, glyph_style, name, "") + Text("  ") + \
                Text(args.get("mode") or "explore", style="orchid") + \
                Text(f" {self.g['dot']} {args.get('description', '')}", style="frost")
        else:
            extra = None
            if name in ("edit_file", "write_file") and "old" in info:
                extra = self._counts(*diff_counts(info["old"], info["new"]))
            left = self._tool_text(glyph, glyph_style, name, label, extra)
        right = Text(summary, style=summary_style)
        if depth > 0 and self._subs:
            sub = self._subs[-1]
            sub["steps"].append((left, right))  # drawn inside the subagent's tree
            sub["count"] += 1
            return
        self.turn_tools += 1
        self.print(self._line(left, right))
        for row in tail:
            row = row.replace("…", self.g["ellipsis"]) if row.startswith("…") else row
            self.print(Text(f"{self.g['pipe']} {row}", style="frost", overflow="ellipsis", no_wrap=True), indent=4)
        if status == "ok" and "old" in info and not info["panel_shown"]:
            self._print_diff(info, indent=4, max_rows=12)  # auto-approved edits still show what changed, once

    def _on_subagent_start(self, depth, description="", mode="explore", **_):
        self._subs.append({"description": description, "mode": mode, "start": time.monotonic(),
                           "steps": deque(maxlen=3), "count": 0})

    def _on_subagent_end(self, depth, steps=0, **_):
        sub = self._subs.pop() if self._subs else {"start": time.monotonic()}
        self._last_sub = {"steps": steps, "secs": time.monotonic() - sub["start"]}

    def _on_compact(self, depth, stage="", before=0, after=0, pruned=0, **_):
        what = f"cleared {_plural(pruned, 'old tool result')}" if stage == "pruned" else "summarized older turns"
        self.print(Text(f"{self.g['compact']} context compacted: {what}, ~{before:,} {self.g['arrow']} ~{after:,} tokens",
                        style="frost"), indent=2 + 4 * depth)

    def _on_retry(self, depth, error="", delay=0, attempt=0, max_attempts=0, **_):
        short = error.splitlines()[0][:120] if error else "error"
        self.print(Text(f"{self.g['retry']} API error: {short}; retrying in {delay:.0f}s [{attempt}/{max_attempts}]",
                        style="sun"), indent=2 + 4 * depth)

    # -- diffs --

    def _diff_source(self, name: str, args: dict) -> dict:
        """The before/after text and starting line for an edit or write, read before the tool runs."""
        try:
            path = (self.workspace / str(args.get("path", ""))).resolve()
            current = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        except OSError:
            current = ""
        if name == "write_file":
            return {"old": current, "new": str(args.get("content", "")), "first": 1, "trim": True}
        old, new = str(args.get("old_string", "")), str(args.get("new_string", ""))
        at = current.find(old) if old else -1
        return {"old": old, "new": new, "first": current.count("\n", 0, at) + 1 if at >= 0 else 1, "trim": False}

    def diff_rows(self, info: dict, width: int, max_rows: int = 40) -> list:
        ops = diff_ops(info["old"], info["new"], info.get("first", 1))
        if info.get("trim"):
            ops = trim_context(ops)
        rows = [self._diff_row(kind, n, text, spans, width) for kind, n, text, spans in ops[:max_rows]]
        if len(ops) > max_rows:
            rows.append(Text(f"  {self.g['ellipsis']} {len(ops) - max_rows} more lines", style="frost"))
        return rows

    def _diff_row(self, kind: str, lineno: int, text: str, spans, width: int) -> Text:
        text = text.replace("\t", "    ")
        if kind == "gap":
            return Text(f"  {'':>4}   {self.g['ellipsis']} {text}", style="frost")
        if kind == "ctx":
            return Text(f"  {lineno:>4}   {text}", style="frost", overflow="ellipsis", no_wrap=True)
        mark = DIFF_STYLES[kind]["mark"] if not self.ascii else {"add": "+", "del": "-", "mod": "~"}[kind]
        gutter = f" {lineno:>4} {mark} "
        body = (gutter + text)[:width - 1].ljust(width - 1)
        if self.plain:
            return Text(self.g["rim"] + body, style=PLAIN_DIFF[kind])
        s = DIFF_STYLES[kind]
        glowing = {i + len(gutter) for start, end in spans for i in range(start, end)}
        row = Text(self.g["rim"], style=f"{s['rim']} on {s['sheen'][0]}")
        for i, ch in enumerate(body):
            back = s["glow"] if i in glowing else mix(*s["sheen"], i / max(len(body) - 1, 1))
            front = mix(s["text"], s["sheen"][1], 0.45) if i < len(gutter) else s["text"]
            row.append(ch, style=f"{front} on {back}")
        return row

    def _print_diff(self, info: dict, indent: int, max_rows: int) -> None:
        for row in self.diff_rows(info, self.width - indent - 2, max_rows):
            self.print(row, indent=indent)

    # -- permission panel (DESIGN.md §4) --

    def permission(self, name: str, args: dict, always_allowed: set) -> tuple:
        self._pause_live()
        depth = max(self._running) if self._running else 0
        indent = 2 + 4 * depth
        touches_secret, prefix = permission_options(name, args)
        width = self.width - indent
        body = []
        title = Text(f"Allow {name}?", style="bold")
        info = self._pending.get(depth)
        if name in ("edit_file", "write_file") and info and "old" in info:
            counts = self._counts(*diff_counts(info["old"], info["new"]))
            kind = "" if name == "edit_file" else ("  overwrite" if info["old"] else "  new file")
            title += Text(f"  {args.get('path', '')}", style="frost") + Text("  ") + counts + Text(kind, style="frost")
            body += self.diff_rows(info, width - 4)
            info["panel_shown"] = True
        elif name == "bash":
            body.append(Text("$ ", style="frost") + Text(str(args.get("command", "")), style="bold"))
            if touches_secret:
                body.append(Text(f"! This command references a secret file. It can only be allowed once.",
                                 style="coral"))
        else:
            for key, value in args.items():
                shown = str(value)
                body.append(Text(f"{key}: ", style="frost") + Text(shown[:600] + ("…" if len(shown) > 600 else "")))
        keys = [("y", "yes"), ("n", "no")]
        if prefix:
            keys.append(("p", f"prefix '{prefix}'"))
        if not touches_secret:
            keys.append(("a", f"always {name}"))
        choices = Text()
        for i, (k, label) in enumerate(keys):
            if i:
                choices += Text(f"  {self.g['dot']}  ", style="frost")
            choices += Text(k, style="bold sun") + Text(f" {label}")
        panel = Panel(Group(*body, Text(""), choices), title=title, title_align="left",
                      box=box.ASCII if self.ascii else box.ROUNDED,
                      border_style="coral" if touches_secret else "sun", width=width, padding=(0, 1))
        self.print(panel, indent=indent)

        valid = {k for k, _ in keys}
        while True:
            key = self.read_key()
            if key is None:
                self.print(Text("(no keyboard available: denied)", style="frost"), indent=indent + 2)
                return False, NO_APPROVER_NOTE.format(name=name)
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "\x1b":
                key = "n"
            key = key.lower()
            if key in valid:
                break
        answer = key
        if key == "p":
            custom = self.edit_line("  prefix: ", default=prefix).strip()
            answer = f"p {custom}" if custom else "p"
        if apply_permission_answer(name, answer, always_allowed, touches_secret, prefix):
            how = {"y": "allowed once", "a": f"allowed {name} for this session",
                   "p": f"allowed commands starting with '{answer[2:].strip() or prefix}'"}[key]
            self.print(Text(f"{self.g['arrow']} {how}", style="frost"), indent=indent + 2)
            # The panel paused the live zone; bring it back so the approved command shows as running,
            # with its timer starting now rather than when the model asked.
            if depth in self._running:
                self._running[depth] = (*self._running[depth][:2], time.monotonic())
            self._refresh()
            return True, ""
        reason = self.edit_line("  reason for the model (optional): ").strip()
        self.print(Text(f"{self.g['denied']} denied" + (f": {reason}" if reason else ""), style="coral"),
                   indent=indent + 2)
        return False, denial_note(name, reason)

    # -- picker (DESIGN.md §4, Picker): /model and /effort --

    def picker_lines(self, state: "PickerState", title: str, width: int, max_rows: int = 12) -> list:
        """The picker as prompt_toolkit (style, text) fragments: a rounded box with an optional filter
        line, the visible rows (❯ cursor, ● the current choice) and the key hints in the bottom border."""
        sky, frost, aqua = (f"fg:{self._pt_color(t)}" for t in ("sky", "frost", "aqua"))
        h, v = ("-", "|") if self.ascii else ("─", "│")
        tl, tr, bl, br = ("+", "+", "+", "+") if self.ascii else ("╭", "╮", "╰", "╯")
        inner = max(width - 4, 20)
        out = [(sky, f"{tl}{h} "), ("bold", title), (sky, f" {h * max(inner - len(title) - 1, 1)}{tr}\n")]

        def row(fragments):
            used = sum(len(t) for _, t in fragments)
            out.extend([(sky, f"{v} ")] + fragments + [("", " " * max(inner - used, 0)), (sky, f" {v}\n")])

        if state.filterable:
            row([(frost, "filter: "), ("", state.filter), (aqua, "█")])
        matches = state.matches()
        if not matches:
            row([(frost, "no match" if state.rows else "nothing to pick")])
        columns = len(state.rows[0]) if state.rows else 0
        cols = [max((len(str(state.rows[i][c])) for i in matches), default=0) for c in range(columns)]
        start = max(0, min(matches.index(state.cursor) - max_rows // 2 if state.cursor in matches else 0,
                           len(matches) - max_rows))
        for i in matches[start:start + max_rows]:
            first, *rest = [f"{str(cell):<{cols[c]}}" for c, cell in enumerate(state.rows[i])]
            pointer = (f"{sky} bold", f"{self.g['prompt']} ") if i == state.cursor else ("", "  ")
            mark = (aqua, f"{self.g['done']} ") if i == state.marked else ("", "  ")
            text = ("  ".join([first] + rest)).rstrip()[:inner - 4]
            row([pointer, mark, ("bold" if i == state.cursor else "", text[:len(first)]),
                 (frost, text[len(first):])])
        hint = f" ↑↓ move {self.g['dot']} enter pick {self.g['dot']} esc cancel "
        if len(matches) > max_rows:
            hint = f" {len(matches)} {'matches' if state.filter else 'models'}  {self.g['dot']}{hint}"
        out += [(sky, f"{bl}{h * max(inner + 2 - len(hint) - 2, 1)}"), (frost, hint), (sky, f"{h}{h}{br}")]
        return out

    def pick(self, title: str, rows: list, cursor: int = 0, marked: int = None, filterable: bool = False,
             pt_input=None, pt_output=None):
        """An inline arrow-key picker (↑/↓, type to filter, Enter, Esc). Returns the chosen index or None."""
        from prompt_toolkit.application import Application
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import FormattedTextControl, Layout, Window

        if not rows:  # nothing to choose from: don't open an empty box
            return None
        state = PickerState(rows, cursor, marked, filterable)
        self._pause_live()
        bindings = KeyBindings()

        @bindings.add("up")
        @bindings.add("c-p")
        def _(event):
            state.move(-1)

        @bindings.add("down")
        @bindings.add("c-n")
        def _(event):
            state.move(1)

        @bindings.add("enter")
        def _(event):
            event.app.exit(result=state.selected())

        @bindings.add("escape", eager=True)
        @bindings.add("c-c")
        def _(event):
            event.app.exit(result=None)

        @bindings.add("backspace")
        def _(event):
            state.backspace()

        @bindings.add("<any>")
        def _(event):
            if state.filterable and event.data and event.data.isprintable():
                state.type(event.data)

        width = shutil.get_terminal_size((100, 30)).columns - 2
        control = FormattedTextControl(lambda: FormattedText(self.picker_lines(state, title, width)))
        app = Application(layout=Layout(Window(control, wrap_lines=False)), key_bindings=bindings,
                          full_screen=False, erase_when_done=True, input=pt_input, output=pt_output)
        try:
            choice = app.run()
        except (EOFError, KeyboardInterrupt):
            choice = None
        if choice is not None:  # leave a one-line record of the choice in the scrollback
            self.print(Text(f"{self.g['arrow']} {title}: ", style="frost") + Text(str(rows[choice][0]), style="sky"),
                       indent=2)
        return choice

    # -- folder trust (shown once per folder, before anything is read from it) --

    def ask_trust(self, folder: Path, broad: bool) -> str:
        """Returns "yes" (trust and remember), "once" (this session) or "no"."""
        keys = ([] if broad else [("y", "yes, trust it")]) + [("o", "once, this session"), ("n", "no, quit")]
        choices = Text()
        for i, (k, label) in enumerate(keys):
            if i:
                choices += Text(f"  {self.g['dot']}  ", style="frost")
            choices += Text(k, style="bold sun") + Text(f" {label}")
        body = [Text(str(folder), style="bold"), Text(""), Text(TRUST_EXPLANATION, style="frost")]
        if broad:
            body += [Text(""), Text(f"! {BROAD_WARNING}", style="coral")]
        self.print()
        self.print(Panel(Group(*body, Text(""), choices), title=Text("Trust this folder?", style="bold"),
                         title_align="left", box=box.ASCII if self.ascii else box.ROUNDED,
                         border_style="coral" if broad else "sun", width=self.width - 2, padding=(0, 1)), indent=2)
        valid = {k for k, _ in keys}
        while True:
            key = self.read_key()
            if key is None or key in ("\x03", "\x1b"):
                key = "n"
            key = key.lower()
            if key in valid:
                break
        choice = {"y": "yes", "o": "once", "n": "no"}[key]
        note = {"yes": f"{self.g['arrow']} trusted; Kyklos won't ask again for this folder",
                "once": f"{self.g['arrow']} trusted for this session only", "no": f"{self.g['denied']} not trusted"}[choice]
        self.print(Text(note, style="coral" if choice == "no" else "frost"), indent=4)
        self.print()
        return choice

    # -- chrome: banner, notices, answer, footer --

    def wordmark(self) -> Text:
        """The gradient "kyklos" (DESIGN.md §4, Banner): the white first letter is the glint."""
        if self.plain:
            return Text("kyklos", style="bold sky")
        return gradient_text("kyklos", ["#E8FBFF", self._c("aqua"), self._c("sky"), self._c("deep")], bold=True)

    def large_wordmark(self) -> Text:
        """The large stylized wordmark rendered in Frutiger Aero gradient (white glint -> aqua -> sky -> deep)."""
        if self.plain:
            return Text("\n".join(LARGE_WORDMARK_LINES), style="bold sky")
        stops = ["#FFFFFF", "#E8FBFF", self._c("aqua"), self._c("sky"), self._c("deep")]
        max_w = max(len(l) for l in LARGE_WORDMARK_LINES)
        height = len(LARGE_WORDMARK_LINES)
        out = Text()
        num_stops = len(stops)

        for row_idx, line in enumerate(LARGE_WORDMARK_LINES):
            for col_idx, ch in enumerate(line):
                if ch == " ":
                    out.append(" ")
                    continue
                x_t = col_idx / max(max_w - 1, 1)
                y_t = row_idx / max(height - 1, 1)
                t = max(0.0, min(1.0, 0.85 * x_t + 0.15 * y_t))
                scaled = t * (num_stops - 1)
                idx = min(int(scaled), num_stops - 2)
                color = mix(stops[idx], stops[idx + 1], scaled - idx)
                out.append(ch, style=f"bold {color}")
            if row_idx < height - 1:
                out.append("\n")
        return out

    def gradient_rule(self, length: int = LARGE_WORDMARK_WIDTH) -> Text:
        """A sleek gradient rule shelf (aqua -> sky -> deep) to underline the large wordmark."""
        if self.plain:
            return Text(self.g["rule"] * length, style="sky")
        stops = [self._c("aqua"), self._c("sky"), self._c("deep")]
        out = Text()
        num_stops = len(stops)
        for i in range(length):
            t = i / max(length - 1, 1) * (num_stops - 1)
            idx = min(int(t), num_stops - 2)
            color = mix(stops[idx], stops[idx + 1], t - idx)
            out.append(self.g["rule"], style=color)
        return out

    def kyklos_box(self, body, border: str = "sky", footer: Text = None) -> Panel:
        """A rounded box titled with the wordmark, for the model's words (progress notes and answers),
        shaped like the permission panel and as wide as the terminal."""
        title = Text(f"{self.g['run']} ", style="aqua") + self.wordmark()  # Rich pads titles with a space
        return Panel(body, title=title, title_align="left", subtitle=footer, subtitle_align="right",
                     box=box.ASCII if self.ascii else box.ROUNDED, border_style=border,
                     width=self.width - 2, padding=(0, 1))

    def banner(self, agent) -> None:
        if self.width >= 48:
            self.print()
            self.print(self.large_wordmark(), indent=2)
            self.print(self.gradient_rule(), indent=2)  # the shelf is exactly as wide as the wordmark
            self.print()
        word = self.wordmark()
        dot = self.g["dot"]
        context = f"{short_tokens(agent.context_window)} context"
        if getattr(agent, "context_source", "") == "default":
            context += " (default)"  # neither detected nor set: a hint that CONTEXT_WINDOW may be worth setting
        effort = getattr(agent, "effort", "default")
        model = agent.model + (f" ({effort} effort)" if effort != "default" else "")
        meta = f"  {dot}  {model}  {dot}  {context}  {dot}  {self.workspace.name}"
        if (self.workspace / "AGENTS.md").is_file():
            meta += f"  {dot}  AGENTS.md loaded"
        self.print(Text(" ") + Text(self.g["run"], style="aqua") + Text(" ") + word + Text(meta, style="frost"))
        self.print(Text(f"   /help for commands {dot} ctrl+c to interrupt", style="frost"))
        self.print()

    def say(self, kind: str, text: str) -> None:
        style = {"info": "frost", "ok": "mint", "warn": "sun", "error": "coral"}.get(kind, "frost")
        prefix = {"ok": self.g["done"] + " ", "error": self.g["fail"] + " ", "warn": "! "}.get(kind, "")
        self.print(Text(prefix + text, style=style), indent=2)

    def answer(self, text: str, footer: Text = None) -> None:
        """The final answer: rendered Markdown in a sky-bordered kyklos box, with the turn's stats
        (ctx · tools · time) set into the bottom border."""
        self._pause_live()
        self.print()
        self.print(self.kyklos_box(AeroMarkdown(text or "(no answer)", code_theme="ansi_dark"), footer=footer),
                   indent=2)
        self.print()

    def footer_text(self, agent, seconds: float) -> Text:
        pct = agent.context_tokens() / agent.context_window
        parts = [f"ctx {pct:.0%}"]
        if self.turn_tools:
            parts.append(_plural(self.turn_tools, "tool"))
        parts.append(f"{seconds:.0f}s")
        return Text(f" {self.g['dot']} ".join(parts), style="frost")

    # -- input: prompt, status bar, completion --

    def _pt_color(self, token: str) -> str:
        value = self.colors[token]
        return value if value.startswith("#") else f"ansi{value.replace('bright_', 'bright')}"

    def prompt_message(self):
        from prompt_toolkit.formatted_text import FormattedText
        parts = []
        if self.agent and self.agent.plan_mode:
            parts.append((f"fg:{self._pt_color('sky')}", "(plan) "))
        parts.append((f"fg:{self._pt_color('sky')} bold", f"{self.g['prompt']} "))
        return FormattedText(parts)

    def toolbar(self):
        from prompt_toolkit.formatted_text import FormattedText
        cols = max(shutil.get_terminal_size((100, 30)).columns - 2, 10)
        parts = [("", " ")]
        if self.plain:
            parts.append(("fg:ansicyan", self.g["rule"] * cols))
        else:
            stops = [self._c("aqua"), self._c("sky"), self._c("deep")]
            for i in range(cols):
                t = i / max(cols - 1, 1) * 2
                k = min(int(t), 1)
                parts.append((f"fg:{mix(stops[k], stops[k + 1], t - k)}", self.g["rule"]))
        parts.append(("", "\n "))
        agent = self.agent
        frost = f"fg:{self._pt_color('frost')}"
        dot = ("", f" {self.g['dot']} ")
        parts.append((f"fg:{self._pt_color('sky')}", agent.model if agent else ""))
        if agent and getattr(agent, "effort", "default") != "default":
            parts.append((f"fg:{self._pt_color('aqua')}", f" {agent.effort}"))
        if agent:
            pct = agent.context_tokens() / agent.context_window
            filled = min(5, max(1 if pct > 0 else 0, round(pct * 5)))
            parts += [dot, ("", f"ctx {pct:.0%} ")]
            for i in range(5):
                if i < filled:
                    color = self._pt_color("mint") if self.plain else mix(self._c("glint"), self._c("mint"), i / 4)
                    parts.append((f"fg:{color}", self.g["on"]))
                else:
                    parts.append((frost, self.g["off"]))
            if agent.plan_mode:
                parts += [dot, (f"bg:{self._pt_color('deep')} fg:#FFFFFF bold", " PLAN ")]
        parts += [dot, (frost, self.workspace.name), dot, (frost, "/help")]
        return FormattedText(parts)

    def _session(self, history_path: Path):
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style

        class SlashCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                if not text.startswith("/") or " " in text:
                    return
                for name, (_, usage, summary) in COMMANDS.items():
                    if name.startswith(text[1:].lower()):
                        yield Completion(f"/{name}", start_position=-len(text), display=usage, display_meta=summary)

        bindings = KeyBindings()

        @bindings.add("escape", "enter")  # Alt+Enter: new line instead of submitting
        def _(event):
            event.current_buffer.insert_text("\n")

        history_path.parent.mkdir(parents=True, exist_ok=True)
        style = Style.from_dict({
            "bottom-toolbar": "noreverse",
            "completion-menu.completion": f"bg:#0A2139 fg:{self._pt_color('frost')}",
            "completion-menu.completion.current": f"bg:{self._pt_color('deep')} fg:#FFFFFF",
            "completion-menu.meta.completion": f"bg:#0A2139 fg:{self._pt_color('frost')}",
            "completion-menu.meta.completion.current": f"bg:{self._pt_color('deep')} fg:#FFFFFF",
        })
        return PromptSession(history=FileHistory(str(history_path)), completer=SlashCompleter(),
                             complete_while_typing=True, key_bindings=bindings, style=style,
                             bottom_toolbar=self.toolbar, multiline=False)

    # -- the loop --

    def turn(self, agent, text: str, default_session: Path) -> None:
        self.turn_tools = 0
        start = time.monotonic()
        try:
            answer = agent.send(text)
        except KeyboardInterrupt:
            self.reset_transient()
            agent.repair_after_interrupt()
            agent.save_session(default_session)
            self.say("warn", "Interrupted.")
            self.print()
            return
        except Exception as e:  # API errors: bad key, rate limit, network...
            self.reset_transient()
            self.say("error", f"API error: {e}")
            self.print()
            return
        self.reset_transient()
        self.answer(answer, footer=self.footer_text(agent, time.monotonic() - start))
        agent.save_session(default_session)

    def run(self, agent, default_session: Path, one_shot: str = None, notice: tuple = None, **ctx_extras) -> None:
        self.agent = agent
        self.banner(agent)
        if notice:
            self.say(*notice)
        if one_shot:
            self.turn(agent, one_shot, default_session)
            return
        ctx = CommandContext(agent, self.say, default_session, pick=self.pick, **ctx_extras)
        session = self._session(default_session.parent / "history")
        last_interrupt = 0.0
        while True:
            try:
                text = session.prompt(self.prompt_message).strip()
            except KeyboardInterrupt:
                if time.monotonic() - last_interrupt < 2:
                    break
                last_interrupt = time.monotonic()
                self.say("info", "Press Ctrl+C again to exit.")
                continue
            except EOFError:
                break
            if not text:
                continue
            if text.startswith("/"):
                try:
                    result = run_command(ctx, text)
                except KeyboardInterrupt:
                    self.reset_transient()
                    self.say("warn", "Interrupted.")
                    continue
                except Exception as e:
                    self.reset_transient()
                    self.say("error", str(e))
                    continue
                self.print()
                if result is EXIT:
                    break
                if result is None:
                    continue
                text = result
            self.turn(agent, text, default_session)
