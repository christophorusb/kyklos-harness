# Kyklos

*κύκλος: "circle, cycle." At its core, a harness is one loop: ask the model, run its tools, feed back the results, repeat.*

A small coding-agent harness (~4,200 lines of Python, plus ~2,900 lines of tests) for any OpenAI-compatible API — Ollama cloud, DeepSeek, OpenRouter, OpenAI, Gemini, local Ollama, LM Studio — with an inline terminal UI, **Kyklos Aero**, in a Frutiger Aero style ([DESIGN.md](DESIGN.md)).

## Features

**The loop**
- Six tools: `read_file`, `grep`, `glob`, `write_file`, `edit_file`, `bash` — plus `task` for subagents.
- **Subagents** with their own empty context, so a long search costs the main conversation one result instead of every file read. Routable to a cheaper model with `/submodel`.
- **Plan mode** (`/plan`) — read-only, for working out an approach before touching anything.
- **Undo** (`/undo`) reverts a turn's conversation *and* its file edits.
- Sessions save and resume per folder; `continue` picks up a turn that hit the step cap.

**Keeping turns cheap**
- **Context compaction** in two stages: clear old tool results, then summarise — triggered at 75% of the window *or* `COMPACT_ABOVE` tokens, whichever comes first.
- **A workspace sketch** in the system prompt, so turn one doesn't start blind.
- **Token accounting** per session, shown live in the status bar.

**Knowing what's happening**
- **Every `bash` prompt explains the command** in plain language before you answer `y` or `n`.
- **Answer claims are checked** against what the turn actually ran.
- `/tokens`, `/tools`, `/diff` — what fills the window, what can run, what changed on disk.
- **A scratch directory** for throwaway scripts, inside the workspace so imports resolve.

**Safety**
- Folder trust, asked before anything in the folder is read.
- Secret files can't be read, written, or reached through `bash`; secrets are stripped from command environments and redacted from output.
- Paths can't leave the workspace; `.git/` is never written directly.
- Ctrl+C or a timeout kills the whole process tree.

## Setup

Kyklos needs **Python 3.10 or newer** (it annotates with `X | None`).

```bash
python3 -m venv .venv         # macOS ships 3.9, too old: brew install python@3.13
./.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env          # then fill in BASE_URL, API_KEY, MODEL
```

<details>
<summary>Windows (PowerShell)</summary>

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env   # then fill in BASE_URL, API_KEY, MODEL
```
</details>

### Running it from any folder

`kyklos` (macOS / Linux) and `kyklos.cmd` (Windows) run `kyklos.py` with this folder's venv, from wherever you are. Set it up once, then open a new terminal.

```bash
mkdir -p ~/.local/bin
ln -s "$PWD/kyklos" ~/.local/bin/kyklos
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zprofile   # if it isn't already
```

Symlink rather than putting this folder on your PATH, so nothing here can shadow a system command; the launcher resolves symlinks and still finds this folder's venv.

<details>
<summary>Windows PATH</summary>

```powershell
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";C:\path\to\kyklos-harness", "User")
```
</details>

Kyklos works in the folder you launch it from. Set `WORKSPACE` in `.env` to pin it to one folder instead.

### Settings in `.env`

| Variable | Effect |
|---|---|
| `BASE_URL`, `API_KEY`, `MODEL` | The provider. One of each; see `.env.example` for seven worked examples |
| `CONTEXT_WINDOW` | Override the detected window. Lower is faster and cheaper |
| `COMPACT_ABOVE` | Compact once the conversation passes this many tokens (default 100000) |
| `MAX_STEPS` | Model turns one request may take (default 50) |
| `SUBAGENT_MODEL`, `SUBAGENT_EFFORT` | Route subagents to a faster or cheaper model |
| `KYKLOS_PASS_ENV` | Secret-named variables to let through to commands anyway |

## Run

```bash
kyklos                                   # interactive
kyklos --plan                            # start in read-only plan mode
kyklos --resume                          # resume this folder's last session
kyklos "create fizzbuzz.py and run it"   # one task, then exit
kyklos --trust "..."                     # trust this folder for one run (for scripts)
kyklos --model kimi-k3                   # another model for one run (not remembered)
kyklos --submodel minimax-m3.1-flash     # another subagent model for one run (or 'same')
```

Type `/help` for the commands; Tab completes them.

| Command | What it does |
|---|---|
| `/plan [task]` | Toggle read-only plan mode, or plan one task in it |
| `/undo` | Revert the last turn's conversation and file edits |
| `/save [file]`, `/resume [file]` | Save or load a JSONL session |
| `/model [name]` | Pick the model, then its thinking effort. `--refresh` reloads the list |
| `/effort [level]` | Change the thinking effort for the current model |
| `/submodel [name\|same]` | Pick a separate model and effort for subagents |
| `/compact` | Shrink the conversation now |
| `/tokens` | What fills the context window, the session's billed tokens, when compaction hits |
| `/tools` | The tools the model can call now, and the rules that auto-approve them |
| `/diff` | What Kyklos has changed in the workspace this session |
| `/map` | The workspace sketch the model starts each turn with |
| `/reset` | Clear the conversation |
| `/exit` | Quit (or Ctrl+C twice at an empty prompt) |

## The first run in a folder

**Folder trust**, asked before Kyklos reads anything there — including `AGENTS.md`, which goes straight into the model's instructions. `y` trusts and remembers, `o` is this session only, `n` quits. Your home folder and drive roots can only ever be `o`. Piped runs can't answer, so they exit unless the folder is already trusted or you pass `--trust`.

**Bash approval**, asked next, for this run only:

- `a` — **ask every time.** The default, and what happens with no keyboard.
- `u` — **run unflagged commands unprompted**, stopping for the rest.

In `u` mode a command still asks if it carries a red finding, names a secret file, runs a program Kyklos has no rules for, or runs a script whose contents it can't see — "Kyklos can't tell what this does" must never come to mean "go ahead". The line is **red = changes something outside the conversation** (deleting, overwriting, escalating, installing, reaching the network) and **warning = reads somewhere you might not expect**. Warnings don't stop a command in `u` mode. `write_file` and `edit_file` always ask.

The choice is never remembered, and the status bar carries `bash: unflagged runs` while it's on.

**Where Kyklos keeps its data:** `~/.kyklos` (or `KYKLOS_HOME`) — `trusted.json`, `settings.json`, a day-cached `models.json`, and `projects/<folder-slug>/` holding each folder's last session and input history.

## The interface: Kyklos Aero

An inline UI built with [Rich](https://github.com/Textualize/rich) and [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit): output scrolls normally and the input line sits at the bottom.

- **Tool calls** are one line each — `● read_file  kyklos.py  24–31 … 8 lines` — with a short summary; `bash` also shows its exit code and last 3 lines.
- **A breathing-bubble spinner** while the model thinks. Running steps show live and vanish when done, so scrollback stays clean.
- **Subagents** show a live tree that collapses to one line when they finish.
- **Permission prompts** are a rounded panel with **glass diff blocks** — added lines mint, deleted coral, modified aqua, changed characters glowing. Single key: `y` / `n` / `p` / `a`.
- **The model's words** go in rounded `○ kyklos` boxes, the final answer as Markdown with `ctx 9% · 4 tools · 14s` set into the border.
- **A status bar** with the model, a context meter, session tokens, a PLAN badge and the workspace — present while the model works, not just while you type.

| Variable | Effect |
|---|---|
| `KYKLOS_UI=plain` | The plain, dependency-free REPL instead |
| `KYKLOS_THEME=plain` | 16 named ANSI colors instead of the Aero truecolor palette |
| `KYKLOS_ASCII=1` | ASCII glyphs (`o * x`) instead of Unicode |
| `NO_COLOR=1` | No colors; glyphs and words still carry every meaning |

Piped or scripted runs use the plain REPL automatically.

## How it works

| File | Role |
|---|---|
| `kyklos.py` | The agent loop, system prompt, permission rules, command registry, plain REPL |
| `tools.py` | Tool schemas, their implementations, the repository map, the environment probe |
| `ui.py` | Kyklos Aero: renders the Agent's events, the permission panel, the prompt and status bar |
| `kyklos`, `kyklos.cmd` | Launchers for macOS / Linux and Windows |
| `test_kyklos.py`, `test_ui.py` | Offline tests: a scripted fake model drives the real loop and UI |
| `DESIGN.md`, `design/aero_preview.py` | The UI design system, and a dependency-free preview of it |

The `Agent` never prints: it emits events (`model_start`, `tool_call`, `tool_result`, `subagent_start`, …) and a front end renders them. The plain REPL and Kyklos Aero share one set of permission rules and one command registry.

`Agent.send` sends the conversation, runs the tool calls that come back, appends the results, and repeats — until the model answers without calling a tool, or `MAX_STEPS` (50) is reached. Hitting the cap isn't a failure: Kyklos reports what the turn managed and which files changed, and `continue` resumes from there.

**Tools.** `read_file` (with line numbers), `grep`, `glob`, `write_file`, `edit_file` (unique string replacement, tolerant of trailing whitespace and indentation shifts), `bash` (PowerShell on Windows). Search skips `.git`, `.venv`, `node_modules` and binary files.

**Subagents.** `task` hands a self-contained job to a fresh `Agent` with an empty context; only its answer comes back. `explore` mode is read-only and never asks permission; `general` can also edit and run commands. Subagents can't nest and run one at a time. Route them elsewhere with `/submodel`, `SUBAGENT_MODEL`, or `--submodel`; the `task` line names the model when it differs. In a live test a subagent took 12 steps and read five files while the main context stayed at 6%.

**Workspace sketch.** The system prompt carries a generated list of source files and what each defines — names only, no signatures or line numbers, since a regex reading code gets names right and parameter lists wrong. Python names come from the real `ast`. Capped per file and overall (~340 tokens here), regenerated after any turn that wrote files. `/map` shows it.

**Scratch space.** `.kyklos/scratch/` is for throwaway scripts; it's inside the workspace so relative imports and dependencies resolve. Writing there needs no approval; *running* a script still asks. It keeps itself out of git without touching your `.gitignore`.

**Project instructions.** An `AGENTS.md` in the workspace is added to the system prompt, like CLAUDE.md or GEMINI.md elsewhere.

**Models and thinking effort.** `/model` opens an arrow-key picker — ↑/↓, type to filter, Enter, Esc — then a second picker offers *that model's own* effort levels:

| Model kind | Levels offered | Example |
|---|---|---|
| Lists its levels (Ollama `/api/show`) | `default` + its levels | kimi-k3: off · low · high · max |
| Thinking on/off only | `default` · off · on | kimi-k2.7-code, gemma4 |
| Thinks, but lists no levels | `default` + low · medium · high · max | minimax-m3.1-flash |
| Can't think | no effort picker | mistral-large-3 |

`default` sends nothing. Kyklos remembers the picked model and **each model's own effort**. The conversation carries over when you switch; models that can't use tools are left out.

**Context window.** `CONTEXT_WINDOW` wins if set; otherwise Ollama cloud is asked via `/api/show`; otherwise 128k. Local Ollama isn't asked — it reports the model's trained maximum while the server runs with a smaller limit, so set `CONTEXT_WINDOW` to `OLLAMA_CONTEXT_LENGTH`.

**Compaction.** At 75% of the window *or* `COMPACT_ABOVE` tokens (100k), whichever is first, everything older than the most recent ~20% is shrunk: first by clearing old tool results (no API call, and usually enough), then by summarising — with the current request and a verbatim log of earlier requests and answers attached, so facts can't be lost to a weak summary. The current step is always kept whole and a tool call is never split from its result.

The absolute cap matters because every step re-sends the whole conversation: a 178k conversation costs 178k per step whether the ceiling is 128k or 1M. On a real 234-call session, capping at 100k would have cut prompt tokens **39%** (27.7M → 16.8M).

**Environment awareness.** At startup Kyklos runs `--version` on common commands and tells the model which work, so it doesn't waste calls finding out. On Windows it also names the Unix tools PowerShell lacks.

## Safety

- `read_file`, `grep` and `glob` run without asking. Everything else asks first, after validating paths and edits and showing a diff: `y` once, `p` for a `bash` prefix rule, `a` for the session, `n` denies and sends your reason back to the model.
- **Every `bash` prompt says what the command would do** — what it deletes or overwrites, whether it reaches the network, inspects host posture (`lsof`, `crontab`, `systemsetup`, `profiles`, `fdesetup`, `socketfilterfw`, `launchctl`, `ps aux`), leaves the workspace, runs as administrator, or pipes a download into a shell. When connected to a remote `BASE_URL`, host and outside-workspace commands explicitly remind you which provider host will receive the output. A red finding turns the panel coral, removes `p` and `a`, and **always stops for approval even when `bash` was previously allowed for the session**. Kyklos works this out itself and never asks the model, since a model steered by text planted in a file would describe a destructive command as routine.
- **OS-level command sandboxing** (`/tools` shows the active backend). On macOS (`sandbox-exec`) and Linux (`bwrap`, when installed), `bash` commands run inside a kernel sandbox: file writes are restricted to the workspace and temp directories (with `.git/hooks`, `.husky`, `.vscode`, and `AGENTS.md` write-protected against shell tampering), sensitive host stores (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.docker`, `~/.kyklos`, Keychains, Cookies, browser profiles) are unreadable even if approved, and outbound network access is denied by default unless you explicitly approve a command that declares network or package-install activity (`KYKLOS_SANDBOX_NET=1` keeps network open inside the sandbox; `KYKLOS_SANDBOX=off` disables OS sandboxing).
- Prefix rules for interpreters name what they run (`py -3 tools/build.py`, never bare `py`), and inline code (`py -c`, `node -e`, `bash -c`) never gets one.
- Paths can't leave the workspace; `write_file` / `edit_file` refuse anything inside `.git/`, and persistent instruction or hook files (`AGENTS.md`, `.husky/`, `.pre-commit-config.yaml`, `.gitmodules`, `.vscode/{tasks,launch,settings}.json`) always require per-call approval.
- **Secret files** — `.env`, `.env.*` (bar `.example`/`.sample`/`.template`), `.npmrc`, `.pypirc`, `.netrc`, `.git-credentials`, `.pgpass`, `credentials.json`, `kubeconfig`, `*.tfvars`, `*.tfstate*`, `service-account*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.kdbx`, `id_rsa*`, `id_ed25519*`, `id_ecdsa*` — can't be read, written or edited, `grep` skips them, and a `bash` command naming one (including globs like `*.tfvars`) always asks, even with `bash` allowed for the session.
- **Commands can't see secrets**, in the environment or in output: Kyklos's `.env` and anything named like a secret (`*_KEY`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*CREDENTIAL*`) is stripped from every command's environment (`KYKLOS_PASS_ENV` lets specific ones through), and values are redacted from tool output before the model or screen see them — secret-named values from both the host environment and the workspace's own `.env*` files, database/broker URI passwords, JWTs, `npm_…`, `pypi-…`, `hf_…`, `sk_live_…`, `sk-…`, `ghp_…`, `AKIA…`, `xox…`, `AIza…` and private-key blocks.
- **Terminal & state hygiene.** Untrusted command strings and outputs have ANSI/OSC/APC/DCS escapes (such as OSC 52 clipboard writes and OSC 8 link spoofing) and bare carriage-return line overwrites stripped before rendering, and `~/.kyklos/` state files (`last.jsonl`, `history`, `settings.json`, `trusted.json`, `models.json`) are written with owner-only `0700`/`0600` permissions.
- The system prompt tells the model that file contents and command output are data, not instructions — planted directions get reported to you, not followed.
- Ctrl+C or a timeout stops **the whole process tree**. `bash` gets no keyboard, so a command that would stop to ask something fails immediately instead of hanging; mid-turn, unfinished calls are marked "Interrupted" so the conversation stays valid.
- Errors — bad path, ambiguous edit, timeout, invalid JSON — go back to the model as tool results, so it can correct itself.

**Checking the answer's story.** A harness can't make a model truthful, but in a coding agent the untruths are about checkable things. `edit_file` already does the strongest version of this: it needs the exact existing text, so a model that imagines a file can't edit it. The same idea is applied to what an answer claims about its own turn:

```
╭─ ○ kyklos ───────────────────────────────────────────────╮
│ Fixed it. I ran the tests and they all pass.             │
╰────────────────────────────── ctx 0% · 1 tool · 0s ──────╯
! this answer says it ran the tests, but no test command ran this turn
```

Claims to have run the tests, committed, pushed or installed are compared against what ran — and whether it *succeeded*, since a denied or failing command is as much a gap as none. It reports to you, never to the model, and only on past-tense claims, so advice and hedging stay quiet.

## Tests

```bash
./.venv/bin/python -m unittest test_kyklos test_ui
```

```powershell
.\.venv\Scripts\python.exe -m unittest test_kyklos test_ui
```