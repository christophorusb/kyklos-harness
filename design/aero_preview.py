"""Kyklos Aero: a preview of the proposed CLI design system, tuned for dark terminals.

No dependencies: raw 24-bit ANSI escape codes. Run:
    py design\\aero_preview.py          (animated bubble spinner)
    py design\\aero_preview.py --still  (no animation)
"""

import os
import shutil
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
os.system("")  # enables ANSI escape processing in older Windows consoles

PALETTE = {
    "sky":    ("#3BB8FF", "accent: prompt, borders, done bubbles"),
    "aqua":   ("#7FE7FF", "gloss highlight: wordmark glint, spinner peak, inline code"),
    "deep":   ("#0A6FD6", "gradient end"),
    "mint":   ("#4FF0C1", "ok: success, added lines, context meter (aqua-leaning, glossy)"),
    "glint":  ("#C2FFEC", "mint's gloss highlight: diff edges, meter peak"),
    "sun":    ("#FFC83D", "warn: permission prompts, retries"),
    "coral":  ("#FF5A64", "danger: errors, denials, - diff lines"),
    "orchid": ("#C38BFF", "subagents"),
    "frost":  ("#8FA9BF", "muted: thoughts, previews, metadata"),
}
RESET, BOLD = "\033[0m", "\033[1m"
WIDTH = min(shutil.get_terminal_size((100, 30)).columns, 100)


def rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def fg(name_or_hex):
    hex_color = PALETTE[name_or_hex][0] if name_or_hex in PALETTE else name_or_hex
    r, g, b = rgb(hex_color)
    return f"\033[38;2;{r};{g};{b}m"


def c(name, text):
    return f"{fg(name)}{text}{RESET}"


def gradient(text, stops):
    """Color each character along a multi-stop gradient."""
    cols = [rgb(PALETTE[s][0] if s in PALETTE else s) for s in stops]
    out, n = [], max(len(text) - 1, 1)
    for i, ch in enumerate(text):
        t = i / n * (len(cols) - 1)
        a, b = cols[min(int(t), len(cols) - 2)], cols[min(int(t), len(cols) - 2) + 1]
        f = t - min(int(t), len(cols) - 2)
        r, g, bl = (round(a[k] + (b[k] - a[k]) * f) for k in range(3))
        out.append(f"\033[38;2;{r};{g};{bl}m{ch}")
    return "".join(out) + RESET


def bg(hex_color):
    r, g, b = rgb(hex_color)
    return f"\033[48;2;{r};{g};{b}m"


def mix(a, b, t):
    """Blend two hex colors; t=0 gives a, t=1 gives b."""
    ra, rb = rgb(a), rgb(b)
    return "#" + "".join(f"{round(ra[k] + (rb[k] - ra[k]) * t):02X}" for k in range(3))


# Diff blocks: a glass sheet per line. The background fades left to right (the sheen),
# a bright edge bar sits on the left (the glass rim), and changed characters glow brighter.
DIFF_STYLES = {
    "add": {"sheen": ("#11564A", "#0A2A25"), "text": "#E4FFF7", "glow": "#21A383", "rim": "#C2FFEC", "mark": "+"},
    "del": {"sheen": ("#561A27", "#2B0D14"), "text": "#FFE0E3", "glow": "#B0344A", "rim": "#FF5A64", "mark": "−"},
    "mod": {"sheen": ("#12436A", "#0A2139"), "text": "#E9F8FF", "glow": "#2A86D6", "rim": "#7FE7FF", "mark": "~"},
}


def diff_line(kind, lineno, text, width, spans=()):
    """One full-width diff row. kind is add/del/mod, or None for an unchanged context line."""
    text = text.replace("\t", "    ")
    if kind is None:
        return f"  {c('frost', f'{lineno:>4}')}   {c('frost', text[:width - 9])}"  # aligned with the ▌ rows
    s = DIFF_STYLES[kind]
    gutter = f" {lineno:>4} {s['mark']} "
    body = (gutter + text)[:width - 1].ljust(width - 1)
    offset = len(gutter)
    glowing = set()
    for start, end in spans:
        glowing.update(range(start + offset, end + offset))
    out = [fg(s["rim"]) + bg(s["sheen"][0]) + "▌"]
    for i, ch in enumerate(body):
        back = s["glow"] if i in glowing else mix(*s["sheen"], i / max(len(body) - 1, 1))
        front = mix(s["text"], s["sheen"][1], 0.45) if i < offset else s["text"]
        out.append(bg(back) + fg(front) + ch)
    return "".join(out) + RESET


def render_diff(old, new, width, first_line=1):
    """Unified-style diff with add, delete, and modify blocks (a modify shows the old line, then the new)."""
    import difflib
    a, b = old.splitlines(), new.splitlines()
    rows = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op == "equal":
            rows += [diff_line(None, first_line + j1 + k, a[i1 + k], width) for k in range(i2 - i1)]
            continue
        olds, news = a[i1:i2], b[j1:j2]
        # Pair each old line with the next sufficiently similar new line (in order): those are modifications.
        pairs, next_new = [], 0
        for oi, o in enumerate(olds):
            for nj in range(next_new, len(news)):
                if difflib.SequenceMatcher(None, o, news[nj]).ratio() > 0.6:
                    pairs.append((oi, nj))
                    next_new = nj + 1
                    break
        oi = nj = 0
        for po, pn in pairs + [(len(olds), len(news))]:
            rows += [diff_line("del", first_line + i1 + k, olds[k], width) for k in range(oi, po)]
            rows += [diff_line("add", first_line + j1 + k, news[k], width) for k in range(nj, pn)]
            if po < len(olds):
                o, n = olds[po], news[pn]
                ops = difflib.SequenceMatcher(None, o, n).get_opcodes()
                rows.append(diff_line("del", first_line + i1 + po, o, width,
                                      [(x1, x2) for t, x1, x2, _, _ in ops if t in ("replace", "delete")]))
                rows.append(diff_line("mod", first_line + j1 + pn, n, width,
                                      [(y1, y2) for t, _, _, y1, y2 in ops if t in ("replace", "insert")]))
            oi, nj = po + 1, pn + 1
    return rows


def visible_len(s):
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def row(left, right=""):
    """Left text with a right-aligned muted summary, like a tool call line."""
    gap = max(WIDTH - visible_len(left) - visible_len(right) - 2, 2)
    return f"{left}{' ' * gap}{right}"


def panel(title, body_lines, border="sun"):
    inner = WIDTH - 6
    top = f"  {fg(border)}╭─ {RESET}{BOLD}{title}{RESET}{fg(border)} {'─' * max(inner - visible_len(title) - 2, 1)}╮{RESET}"
    lines = [top]
    for b in body_lines:
        pad = max(inner - visible_len(b), 0)
        lines.append(f"  {fg(border)}│{RESET} {b}{' ' * pad}{fg(border)}│{RESET}")
    lines.append(f"  {fg(border)}╰{'─' * (inner + 1)}╯{RESET}")
    return "\n".join(lines)


def section(title):
    print(f"\n{c('frost', '── ' + title + ' ' + '─' * max(WIDTH - len(title) - 4, 0))}\n")


def swatches():
    section("palette")
    for name, (hex_color, use) in PALETTE.items():
        print(f"  {fg(name)}██████{RESET}  {fg(name)}{BOLD}{name:<7}{RESET} {c('frost', hex_color)}  {use}")


def large_wordmark():
    lines = [  # keep in step with LARGE_WORDMARK_LINES in ui.py
        r"    __ ____  ____ __ __    ____  _____",
        r"   / //_/\ \/ / //_// /   / __ \/ ___/",
        r"  / ,<    \  / ,<  / /   / / / /\__ \ ",
        r" / /| |   / / /| |/ /___/ /_/ /___/ / ",
        r"/_/ |_|  /_/_/ |_/_____/\____//____/  ",
    ]
    stops = ["#FFFFFF", "#E8FBFF", "aqua", "sky", "deep"]
    cols = [rgb(PALETTE[s][0] if s in PALETTE else s) for s in stops]
    max_w = max(len(l) for l in lines)
    h = len(lines)
    out = []
    for r, line in enumerate(lines):
        out.append("  ")
        for c_idx, ch in enumerate(line):
            if ch == " ":
                out.append(" ")
                continue
            x_t = c_idx / max(max_w - 1, 1)
            y_t = r / max(h - 1, 1)
            t = max(0.0, min(1.0, 0.85 * x_t + 0.15 * y_t))
            scaled = t * (len(cols) - 1)
            idx = min(int(scaled), len(cols) - 2)
            frac = scaled - idx
            c1, c2 = cols[idx], cols[idx + 1]
            red, green, blue = (round(c1[k] + (c2[k] - c1[k]) * frac) for k in range(3))
            out.append(f"\033[38;2;{red};{green};{blue}m{BOLD}{ch}")
        out.append(f"{RESET}\n")
    return "".join(out)


def gradient_rule(length=38, label=None):  # as wide as the wordmark above it
    stops = ["aqua", "sky", "deep"]
    cols = [rgb(PALETTE[s][0] if s in PALETTE else s) for s in stops]
    tail = f" {label} " if label else ""
    dashes = max(length - len(tail) - 1, 0) if label else length

    def dash(i):
        t = i / max(length - 1, 1) * (len(cols) - 1)
        idx = min(int(t), len(cols) - 2)
        frac = t - idx
        c1, c2 = cols[idx], cols[idx + 1]
        red, green, blue = (round(c1[k] + (c2[k] - c1[k]) * frac) for k in range(3))
        return f"\033[38;2;{red};{green};{blue}m─"

    out = ["  "] + [dash(i) for i in range(dashes)]
    if label:  # the version rests in a gap near the right end, like the turn footer in a box border
        out.append(f"{RESET}{c('frost', tail)}{dash(length - 1)}")
    out.append(RESET)
    return "".join(out)


def banner():
    section("banner")
    print(large_wordmark(), end="")
    print(gradient_rule(label="v0.1.0"))  # keep in step with __version__ in kyklos.py
    print()
    word = gradient("kyklos", ["#E8FBFF", "aqua", "sky", "deep"])  # first letter is the white glint
    meta = c("frost", "  ·  kimi-k2.7-code  ·  kyklos-harness  ·  AGENTS.md loaded")
    print(f" {c('aqua', '○')} {BOLD}{word}{meta}")
    print(c("frost", "   /help for commands · ctrl+c to interrupt"))


def spinner(still):
    section("thinking spinner (a breathing bubble)")
    frames = [("·", "deep"), ("∘", "sky"), ("○", "sky"), ("◯", "aqua"), ("○", "sky"), ("∘", "sky")]
    if still:
        print("  " + "  ".join(c(col, f) for f, col in frames) + c("frost", "   ← frames"))
        print(f"  {c('aqua', '◯')} {c('frost', 'thinking · 12s · ctrl+c to interrupt')}")
        return
    start = time.time()
    while time.time() - start < 4:
        f, col = frames[int((time.time() - start) * 6) % len(frames)]
        secs = int(time.time() - start)
        sys.stdout.write(f"\r  {c(col, f)} {c('frost', f'thinking · {secs}s · ctrl+c to interrupt')}   ")
        sys.stdout.flush()
        time.sleep(0.08)
    sys.stdout.write(f"\r  {c('aqua', '◯')} {c('frost', 'thinking · 4s · ctrl+c to interrupt')}   \n")


def turn():
    section("a full turn")
    print(f"{c('sky', '❯')} add a --version flag to kyklos\n")
    print(c("frost", "  Looking at how flags are parsed first."))
    print(row(f"  {c('sky', '●')} {BOLD}grep{RESET}  {c('frost', '/--plan|--resume/ in kyklos.py')}", c("frost", "3 matches")))
    print(row(f"  {c('sky', '●')} {BOLD}read_file{RESET}  {c('frost', 'kyklos.py  660–700')}", c("frost", "41 lines")))
    print(row(f"  {c('aqua', '○')} {BOLD}edit_file{RESET}  {c('frost', 'kyklos.py')}  {c('mint', '+3')} {c('coral', '−0')}"))
    old = "    args = sys.argv[1:]\n    plan = \"--plan\" in args"
    new = ("    args = sys.argv[1:]\n    if \"--version\" in args:\n        print(f\"kyklos {__version__}\")\n"
           "        return\n    plan = \"--plan\" in args")
    print(panel(f"Allow edit_file?  {c('frost', 'kyklos.py')}", [
        *render_diff(old, new, WIDTH - 7, first_line=668),
        "",
        f"{c('sun', BOLD + 'y')} yes  {c('frost', '·')}  {c('sun', BOLD + 'n')} no  {c('frost', '·')}  {c('sun', BOLD + 'a')} always edit_file",
    ]))
    print(row(f"  {c('sky', '●')} {BOLD}edit_file{RESET}  {c('frost', 'kyklos.py')}  {c('mint', '+3')} {c('coral', '−0')}", c("mint", "✓ applied")))
    cmd = c("frost", "$ .\\kyklos --version")
    print(row(f"  {c('sky', '●')} {BOLD}bash{RESET}  {cmd}", c("mint", "exit 0")))
    print(c("frost", "    │ kyklos 0.4.0"))
    print()
    print(f"  Added a {c('aqua', '--version')} flag to {c('aqua', 'kyklos.py')}. It prints the version and exits")
    verified = c("aqua", ".\\kyklos --version")
    print(f"  before any API call, so it works without a key. Verified with {verified}.")
    print()
    print(c("frost", "  ctx 9% · 4 tools · 14s"))


def diff_blocks():
    section("diff blocks: added, deleted, modified")
    old = """def compact(self) -> bool:
    logger.debug("compacting now")
    keep_budget = int(self.context_window * 0.2)
    cut = len(body)
    if cut == 0:
        return False
    return True"""
    new = """def compact(self) -> bool:
    keep_budget = int(self.context_window * KEEP_RECENT)
    cut = len(body)
    if cut == 0:
        return False
    self.usage_mark = None
    return True"""
    print(f"  {c('sky', '●')} {BOLD}edit_file{RESET}  {c('frost', 'kyklos.py')}  {c('mint', '+1')} {c('coral', '−1')} {c('aqua', '~1')}")
    for line in render_diff(old, new, WIDTH - 4, first_line=470):
        print("    " + line)
    print()
    print("    " + "   ".join([
        f"{fg(DIFF_STYLES['add']['rim'])}▌{RESET} {c('frost', 'added')}",
        f"{fg(DIFF_STYLES['del']['rim'])}▌{RESET} {c('frost', 'deleted (or the old side of a modify)')}",
        f"{fg(DIFF_STYLES['mod']['rim'])}▌{RESET} {c('frost', 'modified (the new side)')}",
    ]))
    print(c("frost", "    Brighter cells inside a line mark the exact characters that changed."))


def subagent():
    section("subagent: live while running, then one line")
    print(f"  {c('orchid', '◯')} {BOLD}task{RESET}  {c('orchid', 'explore')} {c('frost', '· Find compaction constants')}")
    print(row(f"    {c('orchid', '├')} {c('sky', '●')} {BOLD}grep{RESET}  {c('frost', '/COMPACT|PRUNE/')}", c("frost", "9 matches")))
    print(f"    {c('orchid', '└')} {c('aqua', '○')} {BOLD}read_file{RESET}  {c('frost', 'kyklos.py  20–50')}")
    print(c("frost", "              ↓ collapses to"))
    print(row(f"  {c('sky', '●')} {BOLD}task{RESET}  {c('orchid', 'explore')} {c('frost', '· Find compaction constants')}", c("frost", "12 steps · 41s")))


def notices():
    section("notices")
    print(c("frost", "  ◌ context compacted: cleared 3 old tool results, ~96k → ~41k tokens"))
    print(c("sun", "  ↻ API error 429; retrying in 2s [1/3]"))
    print(row(f"  {c('coral', '✗')} {BOLD}read_file{RESET}  {c('frost', '.env')}", c("coral", "refused: secret file")))
    print(row(f"  {c('coral', '⊘')} {BOLD}bash{RESET}  {c('frost', '$ rm -r build')}", c("coral", "denied by you")))


def status_bar():
    section("status bar (shown while you type)")
    print("  " + gradient("─" * (WIDTH - 4), ["aqua", "sky", "deep"]))
    meter = gradient("▰▰", ["glint", "mint"]) + c("frost", "▱▱▱")
    print(f"  {c('sky', 'kimi-k2.7-code')} {c('frost', '·')} ctx 12% {meter} {c('frost', '·')} "
          f"\033[48;2;10;111;214m{BOLD} PLAN {RESET} {c('frost', '·  kyklos-harness  ·  /help')}")


def plain_comparison():
    section("for comparison: the plain 16-color fallback (terminal 'cyan')")
    print(f" \033[36m○\033[0m {BOLD}kyklos{RESET}\033[2m  ·  kimi-k2.7-code  ·  kyklos-harness\033[0m")
    print(f"  \033[36m●\033[0m {BOLD}grep{RESET}  \033[2m/--plan|--resume/ in kyklos.py\033[0m      \033[2m3 matches\033[0m")
    print(f"  \033[32m+    if \"--version\" in args:\033[0m   \033[31m-    old line\033[0m")


if __name__ == "__main__":
    still = "--still" in sys.argv
    swatches()
    banner()
    spinner(still)
    turn()
    diff_blocks()
    subagent()
    notices()
    status_bar()
    plain_comparison()
    print()
