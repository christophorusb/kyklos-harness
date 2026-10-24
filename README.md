# Kyklos

*κύκλος: "circle, cycle." At its core, a harness is one loop: ask the model, run its tools, feed back the results, repeat.*

A minimal coding-agent harness (~760 lines of Python, plus ~475 lines of tests) that works with any OpenAI-compatible API: Ollama cloud, DeepSeek, OpenRouter, OpenAI, Gemini, local Ollama, LM Studio.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env   # then fill in BASE_URL, API_KEY, MODEL
```

## Run

```powershell
.\kyklos                                   # interactive
.\kyklos "create fizzbuzz.py and run it"   # one task, then exit
```

`kyklos.cmd` runs `kyklos.py` with this folder's venv. If you add this folder to your PATH, you can type `kyklos` from anywhere.

The agent works in the directory you launch from, or in `WORKSPACE` if you set it in `.env`. A throwaway `sandbox` folder is a good place to start.

REPL commands: `/reset` clears the conversation, `/compact` summarizes it now, `/exit` quits, Ctrl+C interrupts a running turn. After each answer, a dimmed line shows how much of the context window is used.

## How it works

| File | Role |
|---|---|
| `kyklos.py` | The agent loop, the system prompt, the permission prompt, and the REPL |
| `tools.py` | Tool schemas sent to the model, their implementations, and the environment probe |
| `kyklos.cmd` | Windows launcher |
| `test_kyklos.py` | Offline tests: a scripted fake model drives the real loop |

The loop in `Agent.send` sends the conversation to the model, runs any tool calls it returns, appends the results, and repeats. It stops when the model answers without calling a tool, or after `MAX_STEPS`.

**Tools:** `read_file` (with line numbers), `grep` (regex search across files), `glob` (find files by pattern), `write_file`, `edit_file` (unique string replacement with trailing-whitespace and indentation-offset tolerance), and `bash` (PowerShell on Windows, bash elsewhere). Search skips `.git`, `.venv`, `node_modules` and binary files. Each result appears as a short dimmed preview under its call.

**Subagents:** the `task` tool hands a self-contained job to a fresh `Agent` with an empty context. Only its final answer comes back, so a long search costs the main conversation one tool result instead of every file read. Subagent output is indented under the `task` call.
- `explore` mode (default) has only `read_file`, `grep` and `glob`, so it never asks for permission.
- `general` mode can also edit files and run commands. Its permission prompts come to you like the main agent's, and "always" carries over between them.
- Subagents can't start subagents of their own, and they run one at a time.

In a live test, a subagent took 12 steps and read five whole files, while the main agent's context stayed at 6%.

**Project instructions:** if the workspace has an `AGENTS.md`, it's added to the system prompt, like CLAUDE.md or GEMINI.md in other harnesses. Put project conventions there, such as "use tabs" or "run `npm test` after changes".

**Context compaction:** Kyklos tracks token use from the API's `usage` field, or estimates it at ~4 characters per token. When the conversation passes 75% of `CONTEXT_WINDOW`, it shrinks everything older than the most recent ~20%:
1. **Clear old tool results.** File contents and command output make up most of the context and can be fetched again. All user and assistant messages stay word for word, and no API call is needed. Usually this is enough.
2. **Summarize, only if that's not enough.** The same model summarizes the older turns. Kyklos then attaches the current request and a log of earlier requests and answers, both word for word, so facts the user was already told can't be lost in a weak summary.

The step the model is working on is always kept whole, and a tool call is never separated from its result. If one turn needs more than 3 compactions, Kyklos stops with a message instead of looping: the task doesn't fit the window.

**Environment awareness:** at startup Kyklos runs `--version` on common commands (`py`, `python`, `node`, `git`…). It tells the model which ones work and which don't, so the model doesn't waste calls finding out.

**Safety:**
- `read_file`, `grep` and `glob` run without asking. Every other tool asks first (after validating paths and edits, and showing a colored unified diff for `edit_file` / `write_file`): `y` allows once, `p` (or `p <prefix>`) allows safe `bash` commands matching that prefix (`git status`, `py -m unittest`), `a` allows that tool for the rest of the session, and `n` denies it and sends your reason back to the model.
- Paths can't leave the workspace.
- Secret files can't be read, written or edited: `.env` and `.env.*` (except `.env.example`, `.sample`, `.template`), `*.pem`, `*.key`, `*.p12`, `*.pfx` and `id_rsa*`-style SSH keys. `grep` skips them too. `bash` commands that reference a secret file always ask for one-time approval, even if `bash` or a prefix was allowed for the session.
- Tool output is truncated to 20k characters, and each turn is capped at 50 steps.
- Ctrl+C during a turn gives any unfinished tool call an "Interrupted" result, so the conversation stays valid for the next request.

Errors such as a bad path, an ambiguous edit, a timeout, or invalid JSON go back to the model as tool results, so it can correct itself.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest test_kyklos -v
```

## Ideas for next steps

Roughly in order of how much each one helps:

1. **Streaming output**, so you can watch the answer being written (`stream=True`).
2. **Plan mode:** a read-only tool set for exploring and proposing before changing anything.
3. **Skills:** list `skills/*/SKILL.md` names and descriptions in the prompt, and load a full file only when the model asks for it.
4. **Parallel explore subagents** (they're read-only, so they can safely run at the same time), and a cheaper model for subagents.
5. **Session save and resume:** write conversations to JSONL.
6. **MCP client, hooks, web fetch.**
7. **A sandbox:** run `bash` inside Docker or a git worktree.
