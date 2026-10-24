# Changelog

All notable changes to **Kyklos** since the repository was initialized on 2026-10-24.

---

## `3123b47` — Kyklos Aero UI (Phase 1), Design System & Brace Globs (2026-10-24)

- **Kyklos Aero inline UI (`ui.py`, new):**
  - Inline terminal UI built on Rich + prompt_toolkit, following `DESIGN.md`: Frutiger Aero truecolor tokens (`sky`, `aqua`, `deep`, `mint`, `glint`, `sun`, `coral`, `orchid`, `frost`) with a 16-color `KYKLOS_THEME=plain` variant, and circle-motif glyphs (`○` running → `●` done, `✗`, `⊘`, `◌`, `↻`) with an ASCII fallback (`KYKLOS_ASCII=1`, or automatic on non-UTF-8 output).
  - `AeroUI.observer` renders Agent events: one line per tool call with human-readable arguments and right-aligned summaries (`N lines`, `N matches in M files`, `exit N` + last 3 output lines), frost-colored thoughts, compaction/retry notices, Markdown answers with left-aligned headings, and a `ctx N% · N tools · Ns` turn footer.
  - Transient live zone (`rich.live.Live`): a breathing-bubble "thinking" spinner (`· ∘ ○ ◯`), the running step, and a live subagent tree (last 3 steps) that collapses to one line when the subagent finishes.
  - Glass diff blocks (`diff_ops`, `diff_counts`, `trim_context`): added (mint), deleted (coral), and modified (aqua) lines with a left-to-right sheen, a bright rim, and glowing changed characters. Modified lines pair each removed line with the next ≥60%-similar added line. Auto-approved edits still show their diff once.
  - Permission panel: a rounded sun-bordered box (coral for secret-file commands) with the diff or command, and single-key `y / n / p / a` answers (Esc denies, `p` opens an editable prefix, `n` asks for an optional reason).
  - Input: `PromptSession` with file history (`.sessions/history`), Tab completion for slash commands, Alt+Enter for new lines, and a status bar (gradient rule, model, glint→mint context meter, `PLAN` badge, workspace). Ctrl+C twice at an empty prompt exits.
- **Core refactors for front ends (`kyklos.py`):**
  - Split `ask_permission` into `permission_options` and `apply_permission_answer` (plus `denial_note` / `NO_APPROVER_NOTE`), so the plain prompt and the UI share one tested set of permission rules.
  - `Agent._complete` now emits `model_start` / `model_end` (with `purpose="turn"` or `"summary"`) around every API wait; `model_end` fires even when the call fails, so spinners can't get stuck.
  - Replaced the REPL's `if` chain with a slash-command registry (`COMMANDS`, `@command`, `CommandContext`, `run_command`, `EXIT`) that generates `/help` and accepts `/quit`; both front ends use it.
  - `main()` picks the Aero UI when stdin and stdout are terminals (`wants_aero_ui`), and falls back to `plain_repl` for piped runs, `KYKLOS_UI=plain`, or a missing Rich/prompt_toolkit install.
- **Brace patterns in search (`tools.py`):**
  - Added `expand_braces` so `glob` and `grep`'s `glob` filter accept shell-style `{a,b}` alternatives (`**/*.{py,js}`), including several and nested groups, capped at 64 expansions; `..` inside a group is still rejected. Found in a live run where `**/*.{py,js,…}` returned "no files".
- **Design system (`DESIGN.md`, `design/aero_preview.py`, new):**
  - `DESIGN.md` specifies principles, color tokens, glyphs, every component, layout, interaction, fallbacks, and a decisions log.
  - `design/aero_preview.py` is a dependency-free 24-bit-ANSI reference rendering of the whole system.
- **Dependencies (`requirements.txt`):** added `rich>=13.7` and `prompt_toolkit>=3.0.43`.
- **Tests (`test_kyklos.py`, `test_ui.py`):**
  - Added `test_ui.py` (20 tests): diff pairing and spans, tool summaries, full turns through a recording console, the permission panel (approve, deny with reason, editable prefix, no keyboard), subagent collapse, API errors, ASCII/plain modes, banner and status bar, the command registry, and UI selection.
  - Updated the event-order tests for `model_start` / `model_end`, and added tests for `model_end` on failure and for brace patterns (70 total offline tests).
- **Docs (`README.md`):** new "The interface: Kyklos Aero" section with settings, a commands table, an updated file map, and updated line counts (corrected in the follow-up commit to ~2,100 lines of Python plus ~1,060 lines of tests; the first count skipped blank lines).

---

## `68ebecd` — API Retry Backoff & Decoupled Event Observer (2026-10-24)

- **API retry resilience (`kyklos.py`):**
  - Added `is_retryable_error` and `Agent._complete` to retry transient API failures (`429 RateLimitError`, `5xx InternalServerError`, `APIConnectionError`, `APITimeoutError`) up to 3 times with exponential backoff (`1s`, `2s`) during both regular turns and compaction summarization.
  - Non-retryable errors (e.g., `401` invalid key) raise immediately without sleeping.
  - Added an injectable `sleep` parameter on `Agent` for instant offline testing.
- **UI-ready event observer (`kyklos.py`):**
  - Removed all direct `print()` calls from inside `Agent`.
  - `Agent._emit` now dispatches structured events (`thought`, `tool_call`, `tool_result`, `subagent_start`, `subagent_end`, `compact`, `retry`) with `depth` metadata to an `on_event` callback.
  - Added `console_observer` as the default terminal renderer so CLI behavior stays unchanged while allowing a future UI to subscribe to `on_event` with zero `stdout` side effects.
- **Tests (`test_kyklos.py`):**
  - Added `RetryAndObserverTests` (47 total offline tests).

---

## `2b2dccf` — Plan Mode, JSONL Session Save/Resume & Turn Undo (2026-10-24)

- **Plan mode (`kyklos.py`):**
  - Added `Agent.set_plan_mode` and `Agent._update_mode` to toggle a read-only planning mode (`read_file`, `grep`, `glob`, and `explore` subagents only) without clearing the conversation.
  - Restricted subagents in plan mode so `mode="general"` is rejected.
  - Added `--plan` CLI flag and `/plan` (or `/plan <request>`) REPL command, with a `(plan) > ` prompt indicator.
- **Turn undo with file rollback (`kyklos.py`, `tools.py`):**
  - Added `before_write` hook on `Tools` and `Agent.undo` to snapshot conversation state and pre-write file bytes before each top-level turn (including files modified by `general` subagents).
  - Added `/undo` REPL command to restore the previous conversation state, delete newly created files, and restore edited files on disk.
- **JSONL session persistence (`kyklos.py`, `.gitignore`):**
  - Added `Agent.save_session` and `Agent.load_session` to persist and restore conversations (`messages`, `turn_log`, `usage_mark`, `plan_mode`) in JSONL format and repair any interrupted tool calls on load.
  - Added automatic session saving to `.sessions/last.jsonl` (ignored in `.gitignore`), `--resume [file]` CLI flag, and `/save [file]` & `/resume [file]` REPL commands.
- **Tests (`test_kyklos.py`):**
  - Added `PlanModeTests` and `SessionAndUndoTests` (43 total offline tests).

---

## `f181331` — Tolerant `edit_file` Matching, Diff Previews & `bash` Prefix Permissions (2026-10-24)

- **Tolerant `edit_file` matching (`tools.py`):**
  - Added `_match_edit` with two fallbacks when an exact match isn't found:
    1. Line-by-line matching ignoring trailing whitespace and normalizing `\r\n` vs `\n`.
    2. Uniform indentation-offset matching via `textwrap.dedent`, automatically re-indenting `new_string` to match the target file's indentation.
  - Added explicit `ToolError` for empty `old_string` and extended `is_secret_file` protection to `write_file`.
- **Pre-permission validation & unified diff previews (`kyklos.py`, `tools.py`):**
  - Added `Tools.check` in `Agent.run_tool` so invalid paths, secret files, missing files, or unmatched/ambiguous `old_string` edits fail immediately without prompting the user.
  - Added `format_diff` in `ask_permission` to display colored unified diffs (`@@` cyan, `-` red, `+` green) before approving `edit_file` and `write_file`.
- **Granular `bash` prefix rules & secret guard (`kyklos.py`, `tools.py`):**
  - Added `suggest_prefix` and `command_matches_prefixes` so users can approve reusable command prefixes (`[p]refix 'git status'`, `'py -m unittest'`, or custom `p <prefix>`).
  - Prefix rules auto-approve matching commands and safe `&&`/`||`/`;` chains while still prompting if pipes (`|`), redirects (`>`, `<`), subshells (`$(...)`, `` ` ``), or unknown commands appear.
  - Added `command_touches_secret` so `bash` commands referencing `.env`, `*.pem`, `id_rsa`, etc. always require one-time `[y]es / [n]o` approval even if `bash` or a prefix was allowed for the session.
- **Windows UTF-8 console fix (`kyklos.py`):**
  - Reconfigured `sys.stdout` to UTF-8 at startup to prevent `UnicodeEncodeError` on `cp1252` pipes.
- **Tests (`test_kyklos.py`):**
  - Added `EditAndPermissionTests` and expanded `SecretFileTests` (39 total offline tests).

---

## `02aa34a` & `1d39496` — Initial Commit: Kyklos (2026-10-24)

- **Core harness (`kyklos.py`, `kyklos.cmd`, `requirements.txt`, `.env.example`):**
  - OpenAI-compatible agent loop (`MAX_STEPS = 50`), `.env` loader, `AGENTS.md` project instructions loader, Ctrl+C conversation repair (`repair_after_interrupt`), and interactive REPL (`/reset`, `/compact`, `/exit`).
  - Two-stage context compaction at 75% of `CONTEXT_WINDOW`: stage 1 clears old tool outputs >=500 chars; stage 2 summarizes older turns while keeping the current request and a verbatim log of earlier requests and answers.
  - Subagent delegation via the `task` tool (`explore` read-only mode and `general` mode).
- **Tools & safety (`tools.py`, `.gitignore`):**
  - Implemented `read_file`, `grep`, `glob`, `write_file`, `edit_file`, and `bash`, plus parallel startup command probing (`probe_commands`).
  - Workspace path jail (`_resolve`) and secret file blocking (`is_secret_file`).
- **Tests & naming (`test_kyklos.py`, `NAMING.md`, `README.md`):**
  - 31 offline unit tests driven by `FakeClient`.
