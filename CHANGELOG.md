# Changelog

All notable changes to **Kyklos** since the repository was initialized on 2026-10-24.

---

## Unreleased — Narrow Interpreter Prefixes & a Protected `.git/` (2026-10-24)

Two safety gaps seen in a real session.

- **Interpreter prefixes name what they run (`kyklos.py`):**
  - For `py -3 "C:/Users/me/.wf/wf.py" check`, `suggest_prefix` offered `py`, a rule that would have auto-approved any Python for the rest of the session. Interpreters (`py`, `python`, `node`, `deno`, `bun`, `ruby`, `perl`, `php`, `pwsh`, `powershell`, `bash`, `sh`) now get a prefix through their script or module: `py -3 "C:/Users/me/.wf/wf.py"`, `py -3.11 -m unittest`, `node --no-warnings scripts/dev.mjs`.
  - Inline code (`-c`, `-e`, `-p`, `-r`, `--eval`, `--print`, `-Command`, `-EncodedCommand`) and interpreter-only commands (`py`, `node --version`) get no prefix.
  - Quoted script paths stay whole (`TOKEN_RE`), so the saved rule matches the same script again. `npx` joins the two-word commands (`npx tsc`).
- **No writes inside `.git/` (`tools.py`):**
  - Added `Tools._resolve_writable`: `write_file`, `edit_file` and the pre-permission `check` refuse any path with a `.git` component (any depth or case, including nested repositories). The error points the model to git commands and `git commit -F <file>` with a message file outside `.git/`. In the session, the model had written `.git/COMMIT_EDITMSG_forma.tmp`.
  - Names like `.gitignore`, `.github/` and `my.git/` stay writable.
- **Docs (`README.md`):** both rules are in the Safety section.
- **Tests:** added an interpreter-prefix test (using the session's command, inline-code cases, and rule matching) and `GitDirTests` (3 tests: refused writes and edits incl. nested and upper-case `.GIT`, similar names allowed, refusal before any permission prompt) (98 total offline tests).

---

## `0d14e50` — Full-Width Layout & Kyklos Boxes (2026-10-24)

- **Full terminal width (`ui.py`):** removed the 100-column cap (`MAX_WIDTH`). Tool-line summaries, panels and answers now reach the right edge of the terminal, and the width is re-read on every line, so the layout follows window resizes.
- **Kyklos boxes (`ui.py`):**
  - Added `wordmark()` (the gradient `kyklos`, now shared with the banner) and `kyklos_box()`, a rounded panel titled `○ kyklos`, shaped like the permission panel and spanning the terminal.
  - The model's progress notes (`thought` events) appear in a `deep`-bordered box as frost Markdown; the final answer is Markdown in a `sky`-bordered box.
  - The turn footer (`ctx N% · N tools · Ns`) moved into the answer box's bottom border (`footer_text`, replacing `footer`).
- **Docs (`DESIGN.md`, `README.md`):** updated the components, layout and decisions log; `design/aero_preview.py` still shows the earlier unboxed design.
- **Tests (`test_ui.py`):** the turn test now checks both boxes, their content and the footer in the border; added a full-width test at 160 columns (94 total offline tests).

---

## `53d9105` — No More Frozen Commands (2026-10-24)

Fixes the CLI looking stuck after answering `y` to an "Allow bash?" prompt. There were two causes, which made each other worse.

- **Commands get no keyboard (`tools.py`):**
  - `bash` now runs with `stdin=DEVNULL`, PowerShell gets `-NonInteractive`, and `GIT_TERMINAL_PROMPT=0` is set. Previously a command that asked something ("Confirm", `Read-Host`, `npx`'s "Ok to proceed?", a git password) waited on the user's keyboard while its output, including the question, was captured and invisible, until the 120 s timeout. Now such a prompt fails in about a second with an error the model can read (verified: `Read-Host` and a confirming `Remove-Item` both exit after ~1 s, and nothing is deleted).
  - The timeout error now suggests passing the answer as a flag (`-Force`, `--yes`) or using a larger timeout.
- **A live running line after approval (`ui.py`):**
  - The permission panel paused the live zone, and nothing resumed it until the command finished, so even a normal slow command (`npm install`, a test run) showed a frozen screen. Approving now resumes the live zone, showing `○ bash  $ npm test   14s` with a timer that starts at approval (`_running_row`; running entries now carry a start time). Running steps inside a subagent's tree get the same timer.
- **Missing Unix tools named up front (`tools.py`, `kyklos.py`):**
  - Added `missing_unix_tools`: on Windows, the system prompt now says "Not available in this PowerShell: grep, sed, awk, head, tail" with PowerShell equivalents (the grep tool or `Select-String`, `Get-Content -TotalCount / -Tail`, `-replace`). In a real session, the model had piped into `grep` twice and failed with `CommandNotFoundException`.
- **Tests:** added `NoKeyboardTests` (4 tests: an input prompt fails fast, git's credential prompt is off, the timeout hint, missing tools in the prompt) and a UI test for the resumed live zone and timer (93 total offline tests).

---

## `aaf6ee9` — Automatic Context Window Detection (2026-10-24)

- **Context window detection (`kyklos.py`):**
  - Added `detect_context_window`, which asks Ollama cloud's `/api/show` for the model's `context_length` at startup (5 s timeout; any failure falls back quietly). Only `ollama.com` base URLs are asked, so the API key is never sent anywhere else. Local Ollama is skipped on purpose: it reports the trained maximum, while the server runs with a smaller `num_ctx` unless configured.
  - Added `choose_context_window` with a fixed priority: `CONTEXT_WINDOW` (always wins; accepts `1_000_000`; a non-number exits with a clear message) → the detected value → `DEFAULT_CONTEXT_WINDOW` (128k). Detection is skipped when `CONTEXT_WINDOW` is set.
  - Live: `minimax-m3.1-flash` is detected as 1,000,000 tokens in about 1.5 s.
- **Banners (`kyklos.py`, `ui.py`):** the Aero banner shows the window (`1M context`, or `128k context (default)` when neither detected nor set), and the plain REPL prints `context=1M (detected)`. Added `short_tokens` for the compact form.
- **Docs (`README.md`, `.env.example`):** explained the priority order, the Ollama-cloud-only detection, and how to set `CONTEXT_WINDOW` for local Ollama.
- **Tests:** added `ContextWindowTests` (5 tests with a fake `urlopen`: the request's URL, body and auth header; other providers and local Ollama never contacted; failures, missing and implausibly small values ignored; priority order; `short_tokens`) and banner assertions (88 total offline tests).

---

## `a6513f8` — Run Anywhere, Folder Trust & Audit-Session UI Fixes (2026-10-24)

- **Folder trust (`kyklos.py`, `ui.py`):**
  - Added `resolve_trust`, called in `main()` **before** the `Agent` is created, so an untrusted folder's `AGENTS.md` never reaches the model's instructions.
  - New trust prompts: `AeroUI.ask_trust` (a rounded panel with single keys: `y` trust and remember, `o` this session only, `n` quit; Esc or no keyboard means no) and `ask_trust_plain` for the plain REPL.
  - Trusting a folder covers its subfolders (`is_trusted`, with case-insensitive path matching on Windows; sibling folders that merely share a prefix are not covered).
  - Home folders and drive roots (`is_broad_folder`) get a coral warning, and can only be trusted for one session; the core enforces this whatever the front end returns.
  - Non-interactive runs exit unless the folder is trusted or `--trust` / `KYKLOS_TRUST=1` allows that run (never remembered).
- **Per-user data in `~/.kyklos` (`kyklos.py`):**
  - Added `kyklos_home` (overridable with `KYKLOS_HOME`), `project_slug` and `project_dir`.
  - `trusted.json` lists trusted folders. Each folder's session (`last.jsonl`) and input history now live in `projects/<folder-slug>/` instead of a shared `.sessions/` in the Kyklos folder, so `/resume` in one project can no longer load another project's conversation.
- **UI fixes from a real security-audit session (`ui.py`, `kyklos.py`):**
  - A denial reason is shown once, on the panel's `⊘ denied: …` line; the tool line now just says `denied by you` (or `denied: no one to approve`), instead of repeating the reason and squeezing the command off the line.
  - `suggest_prefix` no longer offers rules for commands that don't start with a program name (a quoted string, a `$variable`, a script block, or keywords like `if` / `try`), which had suggested `p prefix '"---'`.
  - When a tool line runs out of room, the right-aligned summary is shortened first (to at most ~35% of the row), so the tool name and its target stay readable.
  - `bash` output of up to 8 lines is shown in full; longer output shows `… N earlier lines` and the last 3. A short multi-section result (like a `Test-Path` answer) had been cut to its last 3 lines.
- **Run from any folder (`README.md`, `.env.example`):**
  - Documented adding the Kyklos folder to the user PATH, and that leaving `WORKSPACE` unset makes Kyklos work in the launch folder.
- **Tests (`test_kyklos.py`, `test_ui.py`):**
  - Added `TrustTests` (7 tests: slugs, remember vs once vs `--trust`, subfolder and sibling matching, broad folders, non-interactive refusal, corrupt-file tolerance, plain prompt answers) and `TrustPanelTests` (3 tests).
  - Added regression tests for the four UI fixes, using commands from the audit session (83 total offline tests).

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
