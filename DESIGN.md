# Kyklos Aero: CLI design system

The look and behavior of Kyklos's terminal UI. It's an **inline** UI: output scrolls in the terminal like a normal program, and a polished input line sits at the bottom. It isn't a full-screen app.

**Style:** Frutiger Aero (the glossy, sky-and-water optimism of roughly 2004–2013), tuned for **dark** terminal backgrounds. A CLI is already a vintage medium; dressing a modern AI agent in Aero is a deliberate contrast.

**Reference implementation:** [`design/aero_preview.py`](design/aero_preview.py) draws every component below with no dependencies. Run it to see the system in your terminal:

```powershell
.\.venv\Scripts\python.exe design\aero_preview.py          # animated
.\.venv\Scripts\python.exe design\aero_preview.py --still  # static
```

## 1. Principles

1. **Aero for brand moments, the terminal's own colors for reading.** Truecolor hues appear in the chrome (prompt, borders, glyphs, status), while answers and code use the terminal's normal text color.
2. **Never use color alone to carry meaning.** Every state also has a symbol or a word (`●`/`✗`/`⊘`, `+`/`−`/`~`, "denied"), so it works for color-blind users, with `NO_COLOR` and in logs.
3. **Dense while working, roomy for answers.** One line per tool call; the final answer gets space and full Markdown.
4. **Show what the model sees, in human terms.** `read_file kyklos.py 660–700`, not raw JSON.
5. **Keep scrollback clean.** Transient things (the spinner, a running subagent's steps) disappear or collapse to one line once they finish.
6. **Gloss in moderation.** Gradients appear in exactly five places: the wordmark, the spinner, the status-bar rule, the context meter and diff blocks.

## 2. Color tokens

Tuned for dark backgrounds. All are 24-bit; see §8 for the fallback.

| Token | Hex | Role |
|---|---|---|
| `sky` | `#3BB8FF` | Accent: the `❯` prompt, panel borders, completed `●` |
| `aqua` | `#7FE7FF` | Gloss highlight: wordmark glint, spinner peak, inline code, running `○` |
| `deep` | `#0A6FD6` | Gradient end, PLAN badge background. Never used for text |
| `mint` | `#4FF0C1` | OK: success, `exit 0`, `+N` counts, context meter |
| `glint` | `#C2FFEC` | Mint's gloss highlight: added-line rim, meter peak |
| `sun` | `#FFC83D` | Warn: permission prompts, retries |
| `coral` | `#FF5A64` | Danger: errors, denials, `−N` counts, deleted-line rim |
| `orchid` | `#C38BFF` | Subagents |
| `frost` | `#8FA9BF` | Muted: thoughts, previews, metadata, unchanged diff lines |

**Typography:**
- **Bold** only for tool names and headings.
- `frost` instead of the terminal's dim style for secondary text (more readable, and on-theme).
- No italics for essential information (Windows conhost renders them badly).
- No backgrounds, with two exceptions: the PLAN badge and diff blocks.

## 3. Glyphs: the circle motif

*Kyklos* means circle, so the glyphs tell the story of the loop: a step is an open bubble that fills when it completes.

| Glyph | Meaning | Color | ASCII fallback |
|---|---|---|---|
| `○` | Step running | aqua | `o` |
| `●` | Step done | sky | `*` |
| `✗` | Failed or refused | coral | `x` |
| `⊘` | Denied by the user | coral | `-` |
| `◌` | Context compacted | frost | `~` |
| `↻` | Retrying | sun | `r` |
| `❯` | Input prompt | sky | `>` |
| `├ └ │` | Subagent tree, output gutter | orchid / frost | `\| \`` |

**Spinner, a "breathing bubble":** `· ∘ ○ ◯ ○ ∘` at about 6 frames per second, pulsing from deep through sky to aqua. In ASCII mode it becomes `. o O o`.

## 4. Components

### Banner
Three lines, with no ASCII-art logo. `kyklos` is drawn as a gradient (`#E8FBFF` → aqua → sky → deep), so the first letter is the white "specular" glint.
```
 ○ kyklos  ·  kimi-k2.7-code  ·  kyklos-harness  ·  AGENTS.md loaded
   /help for commands · ctrl+c to interrupt
```

### Input prompt
`❯` in sky blue, with `(plan) ❯` in plan mode. Your input stays in scrollback as typed, so it isn't echoed again.

### Status bar
Shown only while you type (prompt_toolkit's bottom toolbar). A sky → deep gradient rule sits above it.
```
 kimi-k2.7-code · ctx 12% ▰▰▱▱▱ ·  PLAN  · kyklos-harness · /help
```
The meter's filled cells are a glint → mint gradient. The `PLAN` badge is white bold text on `deep`, and appears only in plan mode.

### Spinner line
`◯ thinking · 12s · ctrl+c to interrupt`, in frost with the bubble animating. It's replaced in place when output arrives, and it **stops before any permission prompt**.

### Kyklos boxes (the model's words)
Everything the model says to you appears in a rounded box shaped like the permission panel, titled with the wordmark (`○ kyklos`, the `○` in aqua and `kyklos` as the gradient), spanning the full terminal width:
- **Progress notes** (text the model writes alongside tool calls) get a `deep`-bordered box with frost Markdown: quieter than the answer.
- **The final answer** gets a `sky`-bordered box (see Answer below).
```
  ╭─ ○ kyklos ───────────────────────────────────────────────────────────╮
  │ I'll check where MAX_STEPS and COMPACT_AT are defined.               │
  ╰──────────────────────────────────────────────────────────────────────╯
```

### Tool call line
`○`/`●` + the **bold** tool name + human-readable arguments (frost), with a short summary aligned to the right edge of the terminal.

| Tool | Arguments shown | Summary |
|---|---|---|
| read_file | `path start–end` | `N lines` |
| grep | `/pattern/ in path` | `N matches in M files` |
| glob | `pattern` | `N files` |
| write_file | `path` + `+N` | `✓ created` / `✓ written` |
| edit_file | `path` + `+N −N ~N` | `✓ applied` |
| bash | `$ command` | `exit N` (mint for 0, coral otherwise), plus the **last** 3 lines of output, frost, under a `│` gutter |
| task | `mode · description` | `N steps · Ns` |
| any failure | — | a coral `✗` line with the reason |

### Permission panel
A rounded box (`╭╮╰╯`) with a sun-colored border. The title reads `Allow <tool>?` plus the target. For edits the body is the diff (§4, "Diff blocks"); for `bash` it's the `$ command`. The last line holds single-key choices, accepted without pressing Enter: `y yes · n no · p prefix '<prefix>' · a always <tool>`. For commands that touch secret files, the border turns coral and the only choices are `y` and `n`.

### Diff blocks
Each changed line is a **glass sheet**:
- **Sheen:** the background fades left to right between the two sheen colors.
- **Rim:** a bright `▌` edge on the left.
- **Glow:** in a modified pair, the exact characters that changed get the glow color.

| Kind | Marker | Rim | Sheen (left → right) | Text | Glow |
|---|---|---|---|---|---|
| Added | `+` | `#C2FFEC` | `#11564A` → `#0A2A25` | `#E4FFF7` | `#21A383` |
| Deleted | `−` | `#FF5A64` | `#561A27` → `#2B0D14` | `#FFE0E3` | `#B0344A` |
| Modified (new side) | `~` | `#7FE7FF` | `#12436A` → `#0A2139` | `#E9F8FF` | `#2A86D6` |

- **Unchanged lines** are frost with no background, aligned with the text in the sheets.
- **Line numbers** sit in each sheet's gutter, dimmed: old numbers for `−` lines, new numbers for `+`/`~` and unchanged lines.
- **Pairing rule:** within a changed region, each removed line is paired, in order, with the next added line that's at least 60% similar (`difflib` ratio). A pair is drawn as `−` (old, with deletions glowing) directly followed by `~` (new, with insertions glowing). Unpaired lines are plain `−` or `+`.
- The header shows counts: `+N` mint, `−N` coral, `~N` aqua.

### Subagents
While a subagent runs, a **live block** updates in place, showing its last 3 steps as an orchid tree:
```
  ◯ task  explore · Find compaction constants
    ├ ● grep  /COMPACT|PRUNE/                 9 matches
    └ ○ read_file  kyklos.py  20–50
```
When it finishes, the block **collapses** to one line: `● task  explore · Find compaction constants   12 steps · 41s`. Its permission prompts still appear, indented under it.

### Answer
Rendered Markdown in a `sky`-bordered kyklos box. Inline code is aqua; code blocks use Rich's terminal-color syntax theme. Headings are modest: bold and left-aligned, with no full-width rules.

### Turn footer
`ctx 9% · 4 tools · 14s` in frost, set into the answer box's bottom border on the right:
```
  ╰──────────────────────────────────────────────── ctx 9% · 4 tools · 14s ─╯
```

### Notices
One line each:
- `◌` compaction, in frost
- `↻` retry, in sun
- `✗` error, in coral

### Picker (`/model`, `/effort`)
An inline rounded box (sky border) drawn below the prompt, erased when done, and replaced by one frost record line (`→ Pick a model: kimi-k3`):
```
  ╭─ Pick a model ───────────────────────────────────────────╮
  │ filter: kim█                                             │
  │ ❯   kimi-k3          1M    off · low · high · max        │
  │     kimi-k2.7-code   262k  off · on                      │
  │   ● minimax-m3.1-flash 1M  low · medium · high · max     │
  ╰────────────── 22 models · ↑↓ move · enter pick · esc cancel ──╯
```
- `❯` (sky, bold) marks the cursor; `●` (aqua) marks the current model or effort.
- The first column is bold on the cursor row; the other columns are frost.
- Typing filters the rows (model picker only), and the filter line shows an aqua `█` caret.
- Up to 12 rows show at once, scrolling with the cursor. The bottom border carries the key hints, plus the count when the list scrolls.
- The effort picker lists `default` first, with a note: "model decides (max)", or "best effort" for generic levels.

## 5. Layout

- A 2-space left margin. Each subagent level adds 4 spaces with an orchid `├`/`└`/`│` guide.
- One blank line between turns and around answers; none between tool calls.
- Everything uses the **full terminal width**, re-read on every line so it follows window resizes. Code, diffs and commands never wrap; long lines are cut off with `…`.
- Right-aligned summaries align to the right edge of the terminal.

## 6. Interaction

- **Input:** history (↑/↓, saved under `.sessions/`), Tab completion for slash commands, Alt+Enter for a new line.
- **Slash commands** come from one registry that also generates `/help`.
- **Ctrl+C** interrupts the running turn (the conversation is repaired automatically); a second Ctrl+C at an empty prompt exits.
- **Single-key permission choices**, with no Enter needed.

## 7. Implementation map

| Piece | Built with |
|---|---|
| Tokens | One `rich.theme.Theme` (names as in §2) |
| Output | A `RichObserver` subscribing to `Agent.on_event` |
| Permission | A Rich panel + a prompt_toolkit single-key prompt, sharing one decision core with the plain prompt |
| Input, status bar | `prompt_toolkit.PromptSession` with `bottom_toolbar` |
| Spinner, subagent live block | `rich.live.Live` (transient) |
| Diffs | The `render_diff` pairing logic from the preview, ported to Rich `Text` |

## 8. Fallbacks and accessibility

| Condition | Behavior |
|---|---|
| Output is not a terminal (piped or scripted) | The plain `console_observer`: no colors, no animation |
| `NO_COLOR` set | No color; glyphs and words still carry every meaning |
| `KYKLOS_THEME=plain` | The 16 named ANSI colors (sky → cyan, mint → green, coral → red, …) |
| No truecolor support | Rich downgrades the colors automatically |
| No UTF-8 output, or `KYKLOS_ASCII=1` | The ASCII glyphs from §3 |

## 9. Decisions log (2026-10-24)

- Inline rather than full-screen (Textual): it reuses `on_event` and the blocking permission prompt, and keeps scrollback.
- Frutiger Aero on a dark background, using a truecolor hybrid rather than the 16 ANSI colors only.
- The original lime green replaced by a glossy aqua-mint (`mint` + `glint`).
- Glass diff blocks with three kinds: added, deleted, modified.
- `frost` for muted text, the breathing-bubble spinner, and a PLAN badge with a solid background: all approved.
- No prefix on answers, collapsing subagent blocks, a turn footer: approved from the preview.
- Later the same day, after using it: the layout uses the full terminal width instead of capping at 100 columns, and the model's progress notes and answers go into `○ kyklos` boxes shaped like the permission panel, with the turn footer set into the answer box's border. `design/aero_preview.py` still shows the earlier, unboxed versions.
