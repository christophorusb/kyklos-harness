# Kyklos

*κύκλος: "circle, cycle." At its core, a harness is one loop: ask the model, run its tools, feed back the results, repeat.*

A small coding-agent harness (~2,100 lines of Python, plus ~1,060 lines of tests) that works with any OpenAI-compatible API: Ollama cloud, DeepSeek, OpenRouter, OpenAI, Gemini, local Ollama, LM Studio. It has an inline terminal UI, **Kyklos Aero**, in a Frutiger Aero style (see [DESIGN.md](DESIGN.md)).

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env   # then fill in BASE_URL, API_KEY, MODEL
```

## Run

```powershell
kyklos                                   # interactive
kyklos --plan                            # start in read-only plan mode
kyklos --resume                          # resume this folder's last session
kyklos "create fizzbuzz.py and run it"   # one task, then exit
kyklos --trust "..."                     # trust this folder for one run, without asking (for scripts)
```

**Using `kyklos` from any folder.** `kyklos.cmd` runs `kyklos.py` with this folder's venv, from wherever you are. Add this folder to your user PATH once, then open a new terminal:

```powershell
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";C:\path\to\kyklos-harness", "User")
```

Kyklos works in the folder you launch it from. Setting `WORKSPACE` in `.env` pins it to one folder instead.

**Folder trust.** The first time you run Kyklos in a folder, it asks whether you trust it. This happens before it reads anything there, including `AGENTS.md`, which goes straight into the model's instructions.
- `y` trusts the folder and everything inside it, and remembers that.
- `o` trusts it for this session only.
- `n` quits.

Your home folder and drive roots can only be trusted for one session. Piped or scripted runs can't answer the question, so they exit unless the folder is already trusted, or you pass `--trust` (or set `KYKLOS_TRUST=1`) for that run.

**Where Kyklos keeps its data:** `~/.kyklos` (or `KYKLOS_HOME`), so projects stay clean:
- `trusted.json`: the folders you've trusted. Delete an entry to be asked again.
- `projects/<folder-slug>/last.jsonl` and `history`: each folder's own session and input history.

Commands (type `/help` for the list, Tab completes them):

| Command | What it does |
|---|---|
| `/plan [task]` | Toggle read-only plan mode, or plan one task in it |
| `/undo` | Revert the last turn's conversation and file edits |
| `/save [file]`, `/resume [file]` | Save or load a JSONL session. Each folder's latest session is also saved automatically to `~/.kyklos/projects/<folder-slug>/last.jsonl` |
| `/compact` | Shrink the conversation now |
| `/reset` | Clear the conversation |
| `/exit` | Quit (or Ctrl+C twice at an empty prompt) |

## The interface: Kyklos Aero

In a real terminal, Kyklos uses an inline UI built with [Rich](https://github.com/Textualize/rich) and [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit). Output scrolls normally and the input line sits at the bottom:

- **Tool calls** show as one line each, `● read_file  kyklos.py  24–31 … 8 lines`, with human-readable arguments and a short summary. `bash` also shows its exit code and last 3 output lines.
- **A breathing-bubble spinner** shows while the model thinks. Running steps are shown live and disappear once they finish, so the scrollback stays clean.
- **Subagents** show a live tree of their latest steps, which collapses to one line when they finish.
- **Permission prompts** are a rounded panel with **glass diff blocks**: added lines in mint, deleted in coral, and modified lines in aqua, with the changed characters glowing. Answer with a single key: `y` / `n` / `p` / `a`. `p` lets you edit the suggested prefix before saving it, and `n` asks for an optional reason.
- **The model's words** appear in rounded `○ kyklos` boxes (with the gradient wordmark), shaped like the permission panel. Progress notes get a subtle deep-blue border; the final answer is rendered Markdown in a sky-blue box, with `ctx 9% · 4 tools · 14s` set into its bottom border.
- **Full width:** everything spans the whole terminal and follows window resizes.
- **A status bar** while you type shows the model, a context meter, a PLAN badge and the workspace.
- **Input:** history with ↑/↓ (kept per folder in `~/.kyklos/projects/<folder-slug>/history`), and Alt+Enter for a new line.

Settings:

| Variable | Effect |
|---|---|
| `KYKLOS_UI=plain` | Use the plain, dependency-free REPL instead |
| `KYKLOS_THEME=plain` | Use the 16 named ANSI colors instead of the Aero truecolor palette |
| `KYKLOS_ASCII=1` | ASCII glyphs (`o * x`) instead of Unicode (also used automatically when output isn't UTF-8) |
| `NO_COLOR=1` | No colors at all; glyphs and words still carry every meaning |

When input or output isn't a terminal (piped or scripted runs), Kyklos automatically uses the plain REPL.

## How it works

| File | Role |
|---|---|
| `kyklos.py` | The agent loop, the system prompt, the permission rules, the command registry, and the plain REPL |
| `tools.py` | Tool schemas sent to the model, their implementations, and the environment probe |
| `ui.py` | Kyklos Aero: renders the Agent's events, the permission panel, and the prompt with its status bar |
| `kyklos.cmd` | Windows launcher |
| `test_kyklos.py`, `test_ui.py` | Offline tests: a scripted fake model drives the real loop (and the UI, through a recording console) |
| `DESIGN.md`, `design/aero_preview.py` | The UI design system, and a dependency-free preview of it |

The `Agent` never prints anything itself: it emits events (`model_start`, `thought`, `tool_call`, `tool_result`, `subagent_start`, …), and a front end renders them. The plain REPL and Kyklos Aero both use the same permission rules (`permission_options` / `apply_permission_answer`) and the same command registry.

The loop in `Agent.send` sends the conversation to the model, runs any tool calls it returns, appends the results, and repeats. It stops when the model answers without calling a tool, or after `MAX_STEPS`.

**Tools:** `read_file` (with line numbers), `grep` (regex search across files), `glob` (find files by pattern), `write_file`, `edit_file` (unique string replacement with trailing-whitespace and indentation-offset tolerance), and `bash` (PowerShell on Windows, bash elsewhere). Search skips `.git`, `.venv`, `node_modules` and binary files. Each result appears as a short dimmed preview under its call.

**Subagents:** the `task` tool hands a self-contained job to a fresh `Agent` with an empty context. Only its final answer comes back, so a long search costs the main conversation one tool result instead of every file read. Subagent output is indented under the `task` call.
- `explore` mode (default) has only `read_file`, `grep` and `glob`, so it never asks for permission.
- `general` mode can also edit files and run commands. Its permission prompts come to you like the main agent's, and "always" carries over between them.
- Subagents can't start subagents of their own, and they run one at a time.

In a live test, a subagent took 12 steps and read five whole files, while the main agent's context stayed at 6%.

**Project instructions:** if the workspace has an `AGENTS.md`, it's added to the system prompt, like CLAUDE.md or GEMINI.md in other harnesses. Put project conventions there, such as "use tabs" or "run `npm test` after changes".

**Context window:** Kyklos needs to know how big the model's context window is, so it can compact in time. It's chosen at startup and shown in the banner (e.g. `1M context`):
1. `CONTEXT_WINDOW`, if you set it (in `.env` or the environment), always wins. Setting it lower than the model's maximum keeps turns faster and cheaper.
2. Otherwise, on **Ollama cloud**, Kyklos asks `/api/show` for the model's `context_length`. For example, `minimax-m3.1-flash` gives 1M and `kimi-k2.7-code` gives 262k, so switching models adjusts the window automatically. This adds about a second to startup and gives up after 5 s.
3. Otherwise it uses 128k, shown as `128k context (default)`.

Local Ollama isn't asked: it reports the model's trained maximum, but the server runs with a smaller limit unless you configure it (`OLLAMA_CONTEXT_LENGTH`), and silently cuts off what doesn't fit. Set `CONTEXT_WINDOW` to that value instead.

**Context compaction:** Kyklos tracks token use from the API's `usage` field, or estimates it at ~4 characters per token. When the conversation passes 75% of the context window, it shrinks everything older than the most recent ~20%:
1. **Clear old tool results.** File contents and command output make up most of the context and can be fetched again. All user and assistant messages stay word for word, and no API call is needed. Usually this is enough.
2. **Summarize, only if that's not enough.** The same model summarizes the older turns. Kyklos then attaches the current request and a log of earlier requests and answers, both word for word, so facts the user was already told can't be lost in a weak summary.

The step the model is working on is always kept whole, and a tool call is never separated from its result. If one turn needs more than 3 compactions, Kyklos stops with a message instead of looping: the task doesn't fit the window.

**Environment awareness:** at startup Kyklos runs `--version` on common commands (`py`, `python`, `node`, `git`…). It tells the model which ones work and which don't, so the model doesn't waste calls finding out. On Windows it also names the Unix tools PowerShell lacks (`grep`, `sed`, `awk`, `head`, `tail`) and what to use instead.

**Safety:**
- `read_file`, `grep` and `glob` run without asking. Every other tool asks first (after validating paths and edits, and showing a colored unified diff for `edit_file` / `write_file`): `y` allows once, `p` (or `p <prefix>`) allows safe `bash` commands matching that prefix (`git status`, `py -m unittest`), `a` allows that tool for the rest of the session, and `n` denies it and sends your reason back to the model.
- Prefix suggestions for interpreters always name what they run: `py -3 tools/build.py` or `node scripts/dev.mjs`, never just `py` or `node`, which would allow any code. Inline code (`py -c`, `node -e`, `powershell -Command`, `bash -c`) gets no prefix at all.
- Paths can't leave the workspace, and `write_file` / `edit_file` refuse anything inside a `.git/` folder (git's own database). Git changes go through git commands.
- Secret files can't be read, written or edited: `.env` and `.env.*` (except `.env.example`, `.sample`, `.template`), `*.pem`, `*.key`, `*.p12`, `*.pfx` and `id_rsa*`-style SSH keys. `grep` skips them too. `bash` commands that reference a secret file always ask for one-time approval, even if `bash` or a prefix was allowed for the session.
- `bash` commands get no keyboard (PowerShell runs with `-NonInteractive`, and git with `GIT_TERMINAL_PROMPT=0`). A command that would stop to ask something (a "Confirm" prompt, `Read-Host`, `npx`'s "Ok to proceed?", a git password) fails at once with an error the model can read, instead of hanging invisibly until the 120-second timeout.
- Tool output is truncated to 20k characters, and each turn is capped at 50 steps.
- Ctrl+C during a turn gives any unfinished tool call an "Interrupted" result, so the conversation stays valid for the next request.

Errors such as a bad path, an ambiguous edit, a timeout, or invalid JSON go back to the model as tool results, so it can correct itself.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest test_kyklos test_ui
```

## Ideas for next steps

Roughly in order of how much each one helps:

1. **UI phase 2:** stream answers as they're written (`stream=True`, assembling tool calls from pieces), Esc to interrupt, and a token/cost readout.
2. **Skills:** list `skills/*/SKILL.md` names and descriptions in the prompt, and load a full file only when the model asks for it.
3. **Parallel explore subagents** (they're read-only, so they can safely run at the same time), and a cheaper model for subagents.
4. **MCP client, hooks, web fetch.**
5. **A sandbox:** run `bash` inside Docker or a git worktree.
