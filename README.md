# Kyklos

*κύκλος: "circle, cycle." At its core, a harness is one loop: ask the model, run its tools, feed back the results, repeat.*

A small coding-agent harness (~4,200 lines of Python, plus ~2,900 lines of tests) for any OpenAI-compatible API — Ollama cloud, DeepSeek, OpenRouter, OpenAI, Gemini, local Ollama, LM Studio — with an inline terminal UI, **Kyklos Aero**, in a Frutiger Aero style ([DESIGN.md](DESIGN.md)).

## Features

**The loop**
- Six workspace tools (`read_file`, `grep`, `glob`, `write_file`, `edit_file`, `bash`) — plus `code_graph` for codebase dependency and blast-radius queries, and `task` for subagents.
- **Questions mid-turn** (`ask_user`): when the model is blocked on a decision that's yours, it asks and carries on with the answer, instead of guessing or ending the turn.
- **Subagents** with their own empty context, so a long search costs the main conversation one result instead of every file read. `explore` tasks can chain via `id` and `depends_on` in topological waves, or route to a cheaper model with `/submodel`.
- **Plan mode** (`/plan`) — read-only, for working out an approach before touching anything; `plan` accepts linear steps or a validated task DAG (`id` / `depends_on`).
- **Undo** (`/undo`) reverts a turn's conversation *and* its file changes — including the ones its commands made, from a snapshot of the workspace.
- Sessions save and resume per folder; `continue` picks up a turn that hit the step cap.
- **Custom slash commands** turn reusable Markdown prompts into commands you can run from the prompt.
- **Skills** give the model a short catalog of instruction packs it can load when relevant.
- **MCP tools** connect the model to stdio servers you configure.
- **Images** for models that take them: `read_file` shows the model a PNG, JPEG or WebP, and `@path/to/shot.png` in a message attaches one.
- **Image generation** (`generate_image`) when an image provider is set up: the model writes a prompt, you approve it, and the image is saved in the workspace.

**Keeping turns cheap**
- **Context compaction** in two stages: clear old tool results, then summarise — triggered at 75% of the window *or* `COMPACT_ABOVE` tokens, whichever comes first — backed by a persistent `graph.json` knowledge overlay so reads, edits, blast radius and pinned notes survive summary truncation.
- **A workspace sketch** in the system prompt, ranked by PageRank centrality, so turn one doesn't start blind.
- **Token accounting** per session, shown live in the status bar.

**Knowing what's happening**
- **Every `bash` prompt explains the command** in plain language before you answer `y` or `n`.
- **Answer claims are checked** against what the turn actually ran.
- **Self-checking:** before a turn ends, the model is asked once to back up its claims and to check code it changed, and each answer says whether its code was checked.
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
| `KYKLOS_SNAPSHOTS` | `0` limits `/undo` to the file tools' edits, without snapshotting the workspace (see [Undo](#how-it-works)) |
| `KYKLOS_SELF_CHECK` | `0` stops asking the model to back up its claims and check changed code before answering (see [Self-checking](#safety)) |
| `KYKLOS_TRACE` | `0` turns off the session trace; a path writes it there instead (see [Evals](#evals)) |
| `STREAM` | Stream the API call (default on). Transport only — nothing on screen changes. `0` to disable |
| `KYKLOS_BACKGROUND_SUBAGENTS` | `1` lets the main model run one read-only background task per turn (default off) |
| `SUBAGENT_MODEL`, `SUBAGENT_EFFORT` | Route subagents to a faster or cheaper model |
| `KYKLOS_PERSONA` | A persona to start in, from `~/.kyklos/personas/<name>.md` |
| `KYKLOS_PASS_ENV` | Secret-named variables to let through to commands anyway |
| `IMAGE_BASE_URL`, `IMAGE_API_KEY`, `IMAGE_MODEL` | An image provider for `generate_image`, set up on its own (see below) |
| `IMAGE_API`, `IMAGE_SIZE` | `images` (the OpenAI Images API, default) or `chat` (OpenRouter-style); a default size like `1024x1024` |

Kyklos Desktop can save preferences for background subagents, max steps, and streaming in
`~/.kyklos/settings.json` (or under `KYKLOS_HOME`). They apply the next time Kyklos starts,
in the desktop and terminal. An explicit environment or `.env` setting takes precedence.

### Profiles

One `.env` can describe several providers. Suffix a block with `_<NAME>` and pick it per run:

```bash
kyklos --profile local        # BASE_URL_LOCAL, API_KEY_LOCAL, MODEL_LOCAL, CONTEXT_WINDOW_LOCAL
kyklos                        # the unsuffixed block, as before
```

`KYKLOS_PROFILE` does the same without the flag, and `/provider` switches without restarting:

```
/provider              pick from the blocks .env defines
/provider cloud        switch now; the conversation carries over
/provider default      back to the unsuffixed block
```

The choice is remembered, so the next `kyklos` starts where you left off (`--profile` still wins for
one run). A profile reads **only** its own suffixed names — nothing falls back to the default block,
so one provider's key can't reach another's host — and each profile remembers its own `/model` pick
separately.

Running against a local Ollama needs one thing `.env.example` can't express: the server's context
window is 4096 by default and Kyklos cannot detect it, so it has to be set by hand. See
[LOCAL-MODELS.md](LOCAL-MODELS.md).

## Run

```bash
kyklos                                   # interactive
kyklos --plan                            # start in read-only plan mode
kyklos --resume                          # resume this folder's last session
kyklos "create fizzbuzz.py and run it"   # one task, then exit
kyklos --trust "..."                     # trust this folder for one run (for scripts)
kyklos --model kimi-k3                   # another model for one run (not remembered)
kyklos --submodel minimax-m3.1-flash     # another subagent model for one run (or 'same')
kyklos --persona chief                   # adopt a persona prompt layer for one run (or 'off')
kyklos --serve                           # no UI: JSON-RPC on stdin/stdout, for a desktop app or editor
```

`--serve` is the front end other programs build on. It runs the same agent, under the same rules,
and asks the same questions over a pipe instead of the keyboard. [PROTOCOL.md](PROTOCOL.md) is the
spec.

Type `/help` for the commands; Tab completes them.

| Command | What it does |
|---|---|
| `/help` | List built-in and custom commands |
| `/plan [task]` | Toggle read-only plan mode, or plan one task in it |
| `/undo` | Revert the last turn's conversation and the files it changed, by tools or commands |
| `/save [file]`, `/resume [file]` | Save or load a JSONL session |
| `/model [name]` | Pick the model, then its thinking effort. `--refresh` reloads the list |
| `/effort [level]` | Change the thinking effort for the current model |
| `/submodel [name\|same]` | Pick a separate model and effort for subagents |
| `/persona [name\|off]` | Adopt a persona prompt layer from `~/.kyklos/personas`, or switch it off |
| `/compact` | Shrink the conversation now |
| `/tokens` | What fills the context window, the session's billed tokens, the prompt-cache hit rate, when compaction hits |
| `/tools` | The tools the model can call now, and the rules that auto-approve them |
| `/skills` | List installed skills, their sources and estimated sizes, and skipped definitions |
| `/mcp [restart <server>]` | Show MCP server status, PID, tool count, schema cost and recent stderr, or restart one server |
| `/diff` | What Kyklos has changed in the workspace this session |
| `/map` | The workspace sketch the model starts each turn with |
| `/graph [path]` | Show the workspace dependency graph, PageRank centrality, blast radius and pinned notes |
| `/reset` | Clear the conversation |
| `/exit` | Quit (or Ctrl+C twice at an empty prompt) |
| `/<name> [args]` | Run a custom Markdown prompt from your user or project commands folder |

## The first run in a folder

**Folder trust**, asked before Kyklos reads anything there — including `AGENTS.md`, which goes straight into the model's instructions. `y` trusts and remembers, `o` is this session only, `n` quits. Your home folder and drive roots can only ever be `o`. Piped runs can't answer, so they exit unless the folder is already trusted or you pass `--trust`.

**Bash approval**, asked next, for this run only:

- `a` — **ask every time.** The default, and what happens with no keyboard.
- `u` — **run unflagged commands unprompted**, stopping for the rest.

In `u` mode a command still asks if it carries a red finding, names a secret file, runs a program Kyklos has no rules for, or runs a script whose contents it can't see — "Kyklos can't tell what this does" must never come to mean "go ahead". The line is **red = changes something outside the conversation** (deleting, overwriting, escalating, installing, reaching the network) **or reads data that must not leave the machine** (credentials, and your own documents — `~/Documents`, `~/Desktop`, `~/Downloads`, `~/Pictures`, `~/Movies`, `~/Videos`, `~/Music`); **warning = reads somewhere you might not expect** but a project legitimately reaches, like `/tmp` or `/srv`. Warnings don't stop a command in `u` mode; red findings always do. Three things that look like that line and aren't stay out of red: a redirect that creates a file which doesn't exist yet (plain path, inside the workspace, not a hook, CI or instruction file), deleting files inside `.kyklos/scratch/`, and piping into an interpreter that is given its program with `-c`, `-m` or a script file, so the pipe is only its input. All three go back to red in a command that changes directory, since the explainer doesn't follow `cd` and would be judging the wrong path. A read is red here because the output goes to the model and then to whichever provider you're pointed at, so the read *is* the data leaving — there's no later step to catch it at. A workspace that lives inside one of those folders (a repo under `~/Desktop` is the ordinary case) still reads its own files freely; the gate is on leaving the workspace, not on the folder's name. `write_file` and `edit_file` always ask.

The choice is never remembered, and the status bar carries `bash: unflagged runs` while it's on.

**Where Kyklos keeps its data:** `~/.kyklos` (or `KYKLOS_HOME`) — `trusted.json`, `settings.json`, a day-cached `models.json`, and `projects/<folder-slug>/` holding each folder's last session, input history and `graph.json` knowledge overlay.

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
| `graph.py` | Multi-language dependency graph, PageRank centrality, persistent `graph.json` overlay, and Task DAG validation |
| `extensions.py` | Markdown headers, skill discovery, and bundled reference reads |
| `mcp_client.py` | Stdio MCP configuration, process lifecycle, and tool discovery and calls |
| `tools.py` | Tool schemas, their implementations, the repository map, the environment probe |
| `snapshots.py` | Workspace snapshots in a private git store, so `/undo` covers what commands changed |
| `ui.py` | Kyklos Aero: renders the Agent's events, the permission panel, the prompt and status bar |
| `kyklos`, `kyklos.cmd` | Launchers for macOS / Linux and Windows |
| `test_kyklos.py`, `test_ui.py`, `test_graph.py` | Offline tests: a scripted fake model drives the real loop, UI and graph tooling |
| `eval.py`, `evals/`, `test_eval.py` | Fixture tasks run against a real model, and the offline tests that keep their checks honest |
| `DESIGN.md`, `design/aero_preview.py` | The UI design system, and a dependency-free preview of it |

The `Agent` never prints: it emits events (`model_start`, `tool_call`, `tool_result`, `subagent_start`, …) and a front end renders them. The plain REPL and Kyklos Aero share one set of permission rules and one command registry.

`Agent.send` sends the conversation, runs the tool calls that come back, appends the results, and repeats — until the model answers without calling a tool, or `MAX_STEPS` (50) is reached. A fifth of the budget from the end the model is told how many steps are left, so it can wrap up instead of walking into the limit. Hitting the cap isn't a failure: Kyklos reports what the turn managed and which files changed, and `continue` resumes from there.

A reply without a tool call ends the turn, so three replies that aren't answers get one nudge (one per turn in total) instead: an empty reply, a tool call written out as text (which runs nothing), and an answer given while the plan still has steps never started. Each is told from the reply and Kyklos's own state, and each shows as a notice, so you see when it happened.

**Tools.** `read_file` (with line numbers), `grep`, `glob`, `write_file`, `edit_file` (unique string replacement, tolerant of trailing whitespace and indentation shifts), `code_graph` (read-only dependency, blast-radius, shortest-path, linked-test and pinned-note queries across Python, JS/TS, Go and Rust; available in plan mode too, while excluding secret files and outside symlinks), `plan` (accepts linear short steps or validated `{id, step, depends_on?}` DAG step objects), and `bash` (PowerShell on Windows). Search skips `.git`, `.venv`, `node_modules` and binary files.

`grep` takes `context` (lines around each match, written as `grep -n -C` writes them) and `output` (`files` or `count` when the matching lines themselves aren't needed). `edit_file` takes `replace_all`, or a list of `edits` applied in order and all-or-nothing, shown as one diff to approve.

**Images.** Ollama says which of its models take images (the `vision` capability in `/api/show`, marked `images` in the `/model` picker), and for those `read_file` on a PNG, JPEG or WebP shows the model the picture: the tool result names the file, type, size and dimensions, and the image follows as one user message after the step's results, because the API takes only text in a tool result. You can attach one yourself with `@path` in a message: relative to the workspace, absolute, or `~/…`; quote a path with spaces, or drag the file into the terminal after the `@`. A model that isn't known to take images gets a clear refusal from `read_file` and the text of your message, with a notice saying what was left out. No other provider says which models take images, so for them set `VISION=1` in `.env` (or `VISION=0` to turn images off). Images are limited to 5 MB. An image counts as a fixed 1,500 tokens toward compaction, and compaction removes old images before anything else. If a provider refuses a request that carries an image, Kyklos takes every image out of the conversation, says so, and retries once with text only, so a bad guess can't wedge the session. Ollama doesn't refuse: it drops the image from a text-only model's request without saying so, and the model answers as though none was sent. So on Ollama, leave `VISION` unset and let the model's own capability list decide; if `VISION=1` contradicts that list, Kyklos says so the first time an image goes to that model. GIF is recognised but not sent, since Ollama refuses it outright ("invalid image input"); Kyklos says so and gives the command to convert it to PNG. **A text-only model with a vision submodel** can still work with images: a subagent uses the submodel's own capability, so when the main model can't take images but `/submodel` can, Kyklos tells it so. `read_file` on an image, and an `@path` you attach, both come with a note that a `task` subagent on that model can look and describe what it sees. That pairs a strong text model with a cheaper vision model, and only the description enters the main conversation.

**Generating images.** With an image provider set up in `.env` (`IMAGE_BASE_URL`, `IMAGE_API_KEY`, `IMAGE_MODEL`; all three or none), the model gets a `generate_image` tool: a prompt, a path in the workspace, and an optional size. The image provider is separate from the chat provider and never borrows its settings, so any chat model can ask for images and one provider's key can't reach the other's host. `IMAGE_API=images` speaks the OpenAI Images API (OpenAI, xAI, Together and others); `IMAGE_API=chat` asks a chat-served image model for an image reply, as OpenRouter serves Gemini's. Every call asks you first, showing the full prompt, where it is going and that image models usually bill per image; `a` allows it for the rest of the session. Before anything is sent, Kyklos checks the arguments (a `.png`, `.jpg` or `.webp` path inside the workspace, never `.git/` or a secret file) and takes known secret values out of the prompt. What comes back must really be a PNG, JPEG or WebP under 20 MB, and the file's extension is changed to match its content if the provider chose another format. The file is written like any other write, so `/undo` removes it. The tool isn't offered in plan mode, to subagents or in `--eval`, which runs unattended and shouldn't spend money.

An edit answers with the lines as they now read, and a miss with the closest text in the file and the first line where it differs, so neither costs a re-read. After a write, `.py`, `.json` and `.toml` files are parsed in-process and a file that no longer parses is flagged in the result; nothing is executed to find out. `bash` output past 20,000 characters is cut in the middle for the model and saved whole under `.kyklos/scratch/output/` (secrets scrubbed, newest 20 kept), where it can be searched by path instead of run again.

**Undo.** Each turn snapshots the workspace when it starts and when it ends, as git trees in a private store of Kyklos's own: a fresh temporary folder, removed when Kyklos exits. The workspace's own `.git` is never read or written, and none of your git setup applies (no hooks, filters, or line-ending conversion), so files come back byte for byte with their permissions and symlinks. `/undo` puts back every path that changed between the two — by `write_file` and `edit_file`, by a `sed -i`, an `rm`, a generator or a subagent alike — except a file that changed again after the turn ended: that change was yours, so it is left alone and named. Snapshots skip what Kyklos skips everywhere (`.venv`, `node_modules`, …), what the workspace's `.gitignore` ignores, `.kyklos/`, and secret files, whose contents are never copied; the file tools' own record still covers their writes to anything snapshots can't see. Files outside the workspace aren't covered. Without git, in your home folder or a drive root, or with `KYKLOS_SNAPSHOTS=0`, `/undo` reverts only the file tools' edits and says so. A snapshot costs two quick git calls per turn: git rehashes only the files whose size or modification time changed.

**Questions.** `ask_user` puts one question to you without ending the turn: a panel with the question and up to six numbered options, answered on one line. A number picks an option (several, comma-separated, when the model allows more than one), anything else is your own answer, and Enter alone skips; the answer goes back to the model as the tool result. Asking needs no approval, since answering is already your call. The model is told to ask only about decisions that are yours to make — not for permission, not to confirm a plan it can carry out, not for anything its tools can find out — and a turn may ask at most three. It's offered in plan mode, where ambiguities get settled, and never to subagents, which answer to another agent: one that tries is told to make the most reasonable assumption and name it in its report. With nobody to answer (a piped run, `--eval`), the model is told so and to go on and state what it assumed.

**Handoffs.** `handoff` shows you work Kyklos can't do (see [What Kyklos hands off](#what-kyklos-hands-off)): a title, a reason such as `browser`, `remote` or `credentials`, and one self-contained prompt with the goal, the workspace path and files, what was already done, the steps, the expected results and what to report back. The terminal prints the prompt between two rules with no border beside it, so selecting it copies only the prompt; Kyklos Desktop draws it as a card with a Copy button. Nothing is sent anywhere and no other program starts, so it needs no approval. Secret values are removed from it first, and a turn may hand off at most twice. It's offered in plan mode, and never to subagents, which put the steps in their report for the agent that delegated them.

**Subagents.** `task` hands a self-contained job to a fresh `Agent` with an empty context; only its answer comes back. `explore` mode is read-only and never asks permission; `general` can also edit and run commands. Subagents can't nest. Explore tasks sent in one message run at the same time, up to four at once, since they only read and never ask; anything else runs one at a time. Explore tasks can also declare `id` and `depends_on` to run in topological waves (`validate_dag`), forwarding completed prerequisite outputs into downstream tasks and rejecting cycles. The interface still draws one subagent after another: the others' steps are held back and replayed when their turn comes, each with its real duration. A subagent whose API call fails comes back as a failed tool call the parent can act on, rather than ending the turn. One that is stopped — by the step cap, or for repeating itself — is asked for a report of what it found, so its work reaches the parent as a partial answer rather than being lost. Route them elsewhere with `/submodel`, `SUBAGENT_MODEL`, or `--submodel`; the `task` line names the model when it differs. In a live test a subagent took 12 steps and read five files while the main context stayed at 6%.

**Background subagents.** Opt in with `KYKLOS_BACKGROUND_SUBAGENTS=1` in the harness's `.env`,
then restart Kyklos. The main agent can start one read-only job with `task_start`, continue
independent work, and collect the report with `task_wait` using the returned job ID. It has up to
two minutes to finish. The worker reads the live workspace, so use it for files the main agent
can leave alone while it works. Stop cancels the job with the turn. Job IDs expire when the turn ends.

**Workspace sketch.** The system prompt carries a generated list of source files and what each defines — names only, no signatures or line numbers, since a regex reading code gets names right and parameter lists wrong. Linked modules are promoted by PageRank centrality (`graph.py`) so core dependencies surface first while unlinked files stay shallow-first; Python names come from the real `ast`. Capped per file and overall (~340 tokens here). `/map` shows the sketch and `/graph [path]` inspects the full dependency graph, blast radius and pinned notes.

The sketch is scanned once per conversation and then left alone, because providers that cache prompts key the cache on the prefix: rewriting the system prompt after every turn that edited a file made each following request a cache miss from its first token. Files that change are named in a short note in front of your next message instead, and folded into the prompt when it is rewritten anyway (`/reset`, a resumed session, compaction). `/tokens` shows how much of the prompt the provider says it read from cache, when it says.

**Scratch space.** `.kyklos/scratch/` is for throwaway scripts; it's inside the workspace so relative imports and dependencies resolve. Writing there needs no approval; *running* a script still asks. It keeps itself out of git without touching your `.gitignore`.

**Project instructions.** An `AGENTS.md` in the workspace is added to the system prompt, like CLAUDE.md or GEMINI.md elsewhere.

**Custom slash commands.** Save a prompt as `~/.kyklos/commands/<name>.md` (under `KYKLOS_HOME`
if set), or share it in `.agents/commands/<name>.md` in a trusted project. Use up to 40 lowercase
letters, digits and hyphens, starting with a letter or digit. An optional header between `---`
lines accepts `description: ...` for help and Tab completion, and `argument-hint: ...` for usage.
A file at `.agents/commands/review.md` can use this format.

```markdown
---
description: Review a file
argument-hint: <file>
---
Review $ARGUMENTS for correctness and report any problems with line references.
```

Type `/review src/app.py` to send the expanded prompt as your message. Each `$ARGUMENTS` gets
the text after the command name, or an empty string when omitted. Without a placeholder, any
arguments go on a new line after the prompt. An `@path` in the template stays literal, and an
`@path` typed as an argument can attach an image. Kyklos shows the source file before sending the
prompt. Built-ins take precedence, followed by user commands, then project commands. The command
list is loaded at startup, and skipped files are reported there. Restart Kyklos after changing a
command. Prompts are capped at 20,000 characters, and modifying project command files through
Kyklos requires approval each time.

**Skills.** Put `SKILL.md` in `~/.kyklos/skills/<name>/` (or under `KYKLOS_HOME`) or share it in a
trusted project's `.agents/skills/<name>/`. The header between `---` lines needs a `name` matching
the folder and a `description` explaining when to use the skill. Kyklos reads simple scalars and
multiline block descriptions and ignores optional metadata. A user skill takes precedence over
a project skill with the same name. See the [Agent Skills specification](https://agentskills.io/specification).

The model gets a capped list of names and descriptions. `/reset` and session resume rebuild this
list from disk, and subagents share the parent conversation's catalog. The model calls
`load_skill` to get the instructions and a list of bundled files, then can read a bundled text
reference by passing its relative path as `file`. Reference contents are data. Text definitions
and references are limited to 20,000 characters each. Plan mode can load skills, and the tool
requires no approval. Changes inside `.agents/skills/` require approval each time. Running any
script from a skill goes through the ordinary `bash` approval rules. Paths to scripts in user
skills typically sit outside the workspace, so expect them to be flagged.

**MCP tools.** Configure servers in `~/.kyklos/mcp.json` (or under `KYKLOS_HOME`). Kyklos only
loads this file from outside the workspace. Each entry in `mcpServers` names one program with a
`command` and optional `args` and `env`. The child starts from Kyklos's filtered command
environment. An `env` value such as `"${SERVICE_TOKEN}"` forwards that variable to that server,
and Kyklos redacts the value from its output and diagnostics.

```json
{
  "mcpServers": {
    "catalog": {
      "command": "python3",
      "args": ["/absolute/path/to/catalog_server.py"],
      "env": {"SERVICE_TOKEN": "${SERVICE_TOKEN}"},
      "readOnly": ["lookup"],
      "timeout": 120
    }
  }
}
```

MCP servers run **unsandboxed** with your account's filesystem and network access. Kyklos starts
them when it starts, all at once, and stops them when it exits. Each has `startupTimeout` seconds
(default 30) to finish its handshake; the first run of an `npx -y` or `uvx` server downloads its
package, so raise it for a large one. A server that misses it is reported and the session goes on
without it; `/mcp restart <server>` tries again.

Servers run with the configuration directory (`~/.kyklos`) as their working directory, never the
project, so a repository can't decide where they run. A server that acts on "the current folder"
by default, such as a git or filesystem server, needs the project's absolute path in its `args`. Each call needs
approval unless you add its original tool name to the server's `autoApprove` list or choose
`a` for that tool in the current session. The server's `readOnly` list controls which tools
appear in plan mode. Both lists come from your configuration. `timeout` is in seconds and defaults
to 120. Subagents and `--eval` run without MCP tools.

`/reset` applies pending tool list updates. `/mcp restart <server>` starts that server again with
the session's configuration and refreshes its tools. Restart Kyklos to load configuration edits.
`/tools` and `/tokens` show the schema cost. Kyklos warns when the offered MCP schemas use more than
10% of the context window.

This version supports tools over stdio with the `2025-11-25`, `2025-06-18`, `2025-03-26` and
`2024-11-05` initialization handshakes. HTTP, SSE, OAuth, resources, prompts, sampling, and roots
are outside its scope. [MCP initialization lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle).

**Personas.** A persona is a prompt layer the user chooses — an architect, a read-only auditor — kept as `~/.kyklos/personas/<name>.md` and selected with `--persona`, `/persona` or `KYKLOS_PERSONA`. It is a system-prompt section, so switching mid-session keeps the conversation and only costs the next request; it is re-sent on every step, so the picker shows each one's token size. Read from `~/.kyklos` only, never from the workspace: a persona defines who the agent is, so a checked-out repository must not be able to name one. It goes into the prompt before the guardrails and `AGENTS.md`, which keeps the harness's own rules the last word, and a subagent doesn't inherit it — it gets a charter instead.

**Models and thinking effort.** `/model` opens an arrow-key picker — ↑/↓, type to filter, Enter, Esc — then a second picker offers *that model's own* effort levels:

| Model kind | Levels offered | Example |
|---|---|---|
| Lists its levels (Ollama `/api/show`) | `default` + its levels | kimi-k3: off · low · high · max |
| Thinking on/off only | `default` · off · on | kimi-k2.7-code, gemma4 |
| Thinks, but lists no levels | `default` + low · medium · high · max | minimax-m3.1-flash |
| Can't think | no effort picker | mistral-large-3 |

`default` sends nothing. Kyklos remembers the picked model and **each model's own effort**. The conversation carries over when you switch; models that can't use tools are left out.

**Context window.** `CONTEXT_WINDOW` wins if set; otherwise Ollama cloud is asked via `/api/show`; otherwise 128k. Local Ollama isn't asked — it reports the model's trained maximum while the server runs with a smaller limit, so set `CONTEXT_WINDOW` to `OLLAMA_CONTEXT_LENGTH`.

**Compaction.** At 75% of the window *or* `COMPACT_ABOVE` tokens (100k), whichever is first, everything older than the most recent ~20% is shrunk: first by clearing old tool results (no API call, and usually enough), then by summarising — with the current request and a verbatim log of earlier requests and answers attached, so facts can't be lost to a weak summary. The current step is always kept whole and a tool call is never split from its result. A summary is also followed by where the work stands as Kyklos recorded it: the plan (linear steps or DAG), the files written this session, the last command if it failed, and the persistent `graph.json` knowledge overlay (tracking reads, edits, test runs, edited modules' blast-radius dependents, linked tests and pinned notes — reloaded automatically across sessions). The plan is saved with the session too.

The absolute cap matters because every step re-sends the whole conversation: a 178k conversation costs 178k per step whether the ceiling is 128k or 1M. On a real 234-call session, capping at 100k would have cut prompt tokens **39%** (27.7M → 16.8M).

**Environment awareness.** At startup Kyklos runs `--version` on common commands and tells the model which work, so it doesn't waste calls finding out. On Windows it also names the Unix tools PowerShell lacks.

## Safety

- `read_file`, `grep`, `glob`, `plan` and `code_graph` run without asking. Everything else asks first, after validating paths and edits and showing a diff: `y` once, `p` for a `bash` prefix rule, `a` for the session, `n` denies and sends your reason back to the model.
- **Every `bash` prompt says what the command would do** — what it deletes or overwrites, whether it reaches the network, inspects host posture (`lsof`, `crontab`, `systemsetup`, `profiles`, `fdesetup`, `socketfilterfw`, `launchctl`, `ps aux`), leaves the workspace, runs as administrator, or pipes a download into a shell. When connected to a remote `BASE_URL`, host and outside-workspace commands explicitly remind you which provider host will receive the output. A red finding turns the panel coral, removes `p` and `a`, and **always stops for approval even when `bash` was previously allowed for the session**. A yellow "can't read" finding (a program Kyklos has no rules for, inline code, a script whose contents aren't shown) removes `a` and stops for approval the same way, so one `a always bash` pressed on an ordinary prompt never covers unreadable commands; `p prefix` stays, since a prefix names exactly what you trusted. Kyklos works this out itself and never asks the model, since a model steered by text planted in a file would describe a destructive command as routine.
- **OS-level command sandboxing** (`/tools` shows the active backend; where there is none, as on Windows or on Linux without `bwrap`, or with `KYKLOS_SANDBOX=off`, Kyklos says so at startup and in the bash-approval choice, since commands then run directly on your machine). On macOS (`sandbox-exec`) and Linux (`bwrap`, when installed), `bash` commands run inside a kernel sandbox: file writes are restricted to the workspace and temp directories (with `.git/hooks`, `.git/config`, `.husky`, `.vscode`, and `AGENTS.md` write-protected against shell tampering, and `.git` itself impossible to rename or replace; under `bwrap`, which can only hold an existing file read-only, any of these a command creates is moved to `.kyklos/quarantine/` and reported), sensitive host stores (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.docker`, `~/.kyklos`, the token stores of `gh`, `gcloud`, `az`, 1Password, gopass and pass, `~/.vault-token`, the cargo, RubyGems, Terraform and yarn credential files, Keychains, Cookies, browser profiles) **and your own document folders** (`~/Documents`, `~/Desktop`, `~/Downloads`, `~/Pictures`, `~/Movies`, `~/Videos`, `~/Music`) are unreadable even if approved — the workspace is readable even when it sits inside one of them, so a repo under `~/Desktop` works normally while its siblings stay out of reach — and outbound network access is denied by default unless you explicitly approve a command that declares network or package-install activity (`KYKLOS_SANDBOX_NET=1` keeps network open inside the sandbox; `KYKLOS_SANDBOX=off` disables OS sandboxing). On macOS, `KYKLOS_SANDBOX_PORTS=5173,4173` opens just those localhost ports, so a dev server can run inside the sandbox without opening the network. It is off by default, takes single ports from 1024 up (no ranges), and `/tools` shows what is open. **It limits which ports, not which interface:** a server that listens on every interface (`0.0.0.0`, the default for Express and `python -m http.server`) is reachable from your network on that port while it runs, because no sandbox rule can restrict who a listener serves. Bind `127.0.0.1` (Vite's default), and know that whatever else already listens on a listed port is reachable from the sandbox too. Everything else — other ports, other hosts, the same port on another machine — stays refused. For the case where nothing in the sandbox needs to listen, `KYKLOS_SANDBOX_CONNECT=5173` is the safer variant: commands may *connect* to that localhost port and nothing may bind or accept on any, so it suits a dev server you start in your own terminal and `curl` against it, and exposes nothing — there is no listener in the sandbox to reach. Whatever listens on that port is still reachable from the sandbox. Git settings are read-only inside the sandbox, because settings like `core.fsmonitor` and `core.hooksPath` are commands your own `git status` would later run unconfined: commits, branches, merges and tags work as usual, while `git init`, `git config`, `git remote add` and `git push -u` have to be run in your own terminal (Kyklos tells the model so when one fails). A tool that needs the macOS Keychain cannot find it from inside the sandbox (the folder is hidden on purpose), and macOS may answer with a **"Keychain Not Found" dialog** on your screen. If one appears, **click Cancel, never "Reset To Defaults"**: nothing is missing, and it is the sandbox that hides the keychain from the tool. Chrome started by hand raised one in testing. Kyklos does not drive browsers at all (`GUARDRAILS.md` rule 7): the model may write Playwright specs and test plans, but when a page has to be seen or clicked it stops and hands you the start command, URL, pages to check and expected results, to run yourself or give to another tool. If you explicitly ask it to run a browser anyway, Chrome needs `--use-mock-keychain --password-store=basic` (Playwright passes both itself); anything else that reads the Keychain can hit the same wall and is best run outside Kyklos. When a command fails mentioning the Keychain, or a hand-started Chrome times out, Kyklos tells the model all of this as well, so it stops retrying and hands the check to you. On macOS, temp directories are writable but **not executable**: a command can download a payload into `/tmp` or `/var/folders` and cannot then run it, which is the second half of the `curl | sh` pattern. The workspace stays executable so builds can run what they compile; toolchains that compile into `$TMPDIR` and execute from there (`go test`) need `KYKLOS_SANDBOX_EXEC_TMP=1`.
- Prefix rules for interpreters name what they run (`py -3 tools/build.py`, never bare `py`), and inline code (`py -c`, `node -e`, `bash -c`) never gets one.
- Paths can't leave the workspace; `write_file` / `edit_file` refuse anything inside `.git/`, and persistent instruction or hook files (`AGENTS.md`, `.husky/`, `.pre-commit-config.yaml`, `.gitmodules`, `.vscode/{tasks,launch,settings}.json`), CI configuration that runs on push (`.github/workflows/`, `.github/actions/`, `.gitlab-ci.yml`, `.circleci/`, `.buildkite/`, `Jenkinsfile`, `.travis.yml`, `azure-pipelines.yml`, `bitbucket-pipelines.yml`) and `.envrc` always require per-call approval, from the file tools and from bash alike.
- **Secret files** — `.env`, `.env.*` (bar `.example`/`.sample`/`.template`), `.npmrc`, `.pypirc`, `.netrc`, `.git-credentials`, `.pgpass`, `credentials.json`, `kubeconfig`, `*.tfvars`, `*.tfstate*`, `service-account*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.kdbx`, `id_rsa*`, `id_ed25519*`, `id_ecdsa*` — can't be read, written or edited, `grep` skips them, and a `bash` command naming one (including globs like `*.tfvars`) always asks, even with `bash` allowed for the session.
- **Commands can't see secrets**, in the environment or in output: Kyklos's `.env` and anything named like a secret (`*_KEY`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*CREDENTIAL*`, `*_AUTH`, `*AUTHORIZATION`, `*_DSN`, `*_COOKIE`), or holding a URL with credentials in it (`DATABASE_URL=postgres://app:pw@db`, but not a password-less local one), is stripped from every command's environment (`KYKLOS_PASS_ENV` lets specific ones through), and values are redacted from tool output before the model or screen see them — secret-named values from both the host environment and the workspace's own `.env*` files, passwords in any URL, tokens used as a URL's user (Sentry DSNs, `https://<token>@github.com`), `Authorization:` header values and `Bearer` tokens, JWTs, `npm_…`, `pypi-…`, `hf_…`, `sk_live_…`, `sk-…`, `ghp_…`, `AKIA…`, `xox…`, `AIza…` and private-key blocks.
- **Terminal & state hygiene.** Untrusted command strings and outputs have ANSI/OSC/APC/DCS escapes (such as OSC 52 clipboard writes and OSC 8 link spoofing) and bare carriage-return line overwrites stripped before rendering, and `~/.kyklos/` state files (`last.jsonl`, `history`, `graph.json`, `settings.json`, `trusted.json`, `models.json`) are written with owner-only `0700`/`0600` permissions.
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

Claims to have run the tests, committed, pushed or installed are compared against what ran — and whether it *succeeded*, since a denied or failing command is as much a gap as none. It reports to you, and only on past-tense claims, so advice and hedging stay quiet. With self-checking on (the default), the model hears about it too, while it can still act: see below.

**Self-checking.** Before a turn's answer stands, Kyklos looks at two things and, if either is off, sends the model one message and lets it carry on instead of ending the turn:

- **The answer's story.** The claims above ("I ran the tests", "committed", …) that nothing this turn backs up. The model runs what it claimed or corrects the answer; a claim about an earlier turn is fine once it says so.
- **Unchecked code.** Code changed this turn and no check passed after the last change. A check is a test run, a type check, a lint or a build (`pytest`, `unittest`, `npm test`, `tsc`, `mypy`, `ruff`, `eslint`, `cargo check`, `go vet`, `npm run build`, `py_compile` and the like) that exited 0. The message names the changed files and the tests the code graph links to them, and asks for the narrowest check that covers the change, or, when none fits (nothing set up, or it needs a browser, credentials or the network), for the answer to say so instead. A check that ran after the change and failed is put to the model the same way, unless the answer already reports the failure.

**Kyklos never runs a check itself.** The model runs it through `bash`, under the same approvals and sandbox as any other command, so self-checking adds no new way for anything to run, and a repository can't choose a command that runs unasked. Each kind of message goes at most once per turn, both together share one message, and none is sent in plan mode, to subagents, after a cut-off reply, with fewer than two steps left, when the answer asks you a question, or when it already says why it couldn't check. Code means source files by extension: docs, data, config and `.kyklos/scratch/` don't count.

With snapshots on, "changed" is read off the workspace itself (from the turn's start, and from just after the last check), so a `sed -i` or a generator counts as much as `edit_file`; without them, only the file tools' edits are seen. Under the answer you then see one of:

```
✓ checked after the last code change
! code changed this turn, and no check passed after the last change
! the last check after the code changes failed
```

Nothing is shown for a turn that changed no code. The state is `checked` on `turn_end` and in the `session/prompt` result, and the trace keeps it as a label. `KYKLOS_SELF_CHECK=0` turns the messages and the line off.

## What Kyklos hands off

Some work stays outside Kyklos, mostly because of how the sandbox and guardrails are built rather than for lack of time. For these, the model does what it can in the workspace (writes the code, the tests, the plan) and hands the rest to you with the `handoff` tool: one self-contained prompt, shown as a card, to do yourself or give to a larger agent such as Claude Code, Antigravity or Codex. Each of those has its own security model: a handoff moves the work into that model and does not make it safe, so look at what you hand over when it is sensitive.

**Blocked on purpose**

| Limit | Why | Hand off |
|---|---|---|
| **Browsers** | `GUARDRAILS.md` rule 7: a rendered page is untrusted input that can carry instructions | Visual checks, end-to-end and Playwright runs, screenshots of a running site, filling in forms. Kyklos writes the tests and ends its turn with the start command, URL, pages to check and expected results |
| **Web research** | No web fetch or search tool; the sandbox cuts off the network, and any network command is red and needs approval | Reading docs, comparing libraries, checking API changes, looking up error messages online |
| **GitHub and remotes** | `git init`, `git config`, `git remote add` and `git push -u` fail in the sandbox; `~/.ssh` and the `gh` token store are unreadable | Pushing, opening PRs, replying to review comments, watching and fixing CI, creating repos. Local commits and branches work |
| **Your credentials** | Secret files are blocked, secrets are stripped from command environments, and `~/.aws`, `~/.kube`, `~/.docker` and Keychains are hidden | Deploys, cloud CLIs (`aws`, `gcloud`, `kubectl`, `docker push`), migrations on real databases, logged-in API calls, publishing packages |
| **Keychain and signing** | The sandbox hides `~/Library/Keychains`, and rule 1 forbids `security` | Code signing, notarization, Xcode signing, App Store and TestFlight uploads |
| **Privileges and host setup** | Rule 1 forbids `sudo`, `su`, `osascript` prompts and debuggers; installers are red network commands | System packages, `brew install`, toolchain setup, browser binaries, OS settings |
| **GUI and device apps** | Rule 5 prefers headless runs, and there is no screen to drive | iOS and Android simulators, desktop app UI checks, menu-bar apps |
| **Outside the workspace** | Paths can't leave the workspace, and sibling folders under `~/Desktop` and `~/Documents` are unreadable | Changes that span several repositories, or one Kyklos session per repository |

**Limits of what's built today**

| Limit | Detail | Hand off |
|---|---|---|
| **Long runs** | A `bash` call stops after 600 seconds and a turn after `MAX_STEPS` (50); nothing manages a process that keeps running | Builds or test suites over 10 minutes, watchers, multi-hour refactors, benchmarks |
| **Parallel work** | Subagents can't nest; background subagents are read-only, one per turn, and get two minutes | Parallel work across branches or worktrees, large fan-out jobs, separate PRs at once |
| **Scheduled or cloud runs** | Kyklos is a local process; `--eval` runs unattended but only on this machine | Nightly or recurring jobs, work in the cloud |
| **No sandbox on some hosts** | Windows and Linux without `bwrap` have no OS sandbox, and the port allowlist is macOS only | On those hosts, risky or network-heavy work goes to a tool with its own sandbox |
| **Model strength** | Kyklos runs whatever model `.env` names, often a local one; vision depends on the model or a vision submodel | Hard design calls, tricky debugging, security reviews and large reasoning tasks may need a stronger model, in Kyklos or elsewhere |

Some things look like handoffs and aren't: edits to CI configuration, hooks, `AGENTS.md` and `.envrc`, `bash` commands Kyklos can't read, and `generate_image` only need your approval for each call. A network command can also run in Kyklos once you approve it, but that is an exception you grant each time, not the normal path.

## Tests

```bash
./.venv/bin/python -m unittest test_kyklos test_ui test_audit test_eval test_serve test_imagegen test_snapshots test_question test_handoff test_self_check test_commands test_skills test_mcp test_graph
```

```powershell
.\.venv\Scripts\python.exe -m unittest test_kyklos test_ui test_audit test_eval test_serve test_imagegen test_snapshots test_question test_handoff test_self_check test_commands test_skills test_mcp test_graph
```

## Evals

The tests prove the loop does what it says; they can't say whether a change helps a real model. `eval.py` does that: it runs Kyklos on small fixture tasks against the provider in `.env` and reports what each run did and cost. **It spends real tokens.**

```bash
./.venv/bin/python eval.py --list                     # the tasks, one line each
./.venv/bin/python eval.py                            # every task once, with your usual model
./.venv/bin/python eval.py long-log --repeat 3 --model kimi-k3 --keep
```

```
PASS  fix-off-by-one   steps  5  tools  5  tokens   18k  cache 74%      9s
PASS  long-log         steps  8  tools  8  tokens   34k  cache 70%     19s
PASS  parallel-lookup  steps  6  tools 11  tokens   28k  cache 59%     13s
```

Each task in `evals/tasks/` is a small repository, a prompt, a `check.py` that decides pass or fail, and a reference solution that `test_eval.py` holds the check to: every check must fail the untouched repository and pass the solution, or it measures nothing. The tasks are chosen to land on the harness's own mechanisms — a failure buried past the 20,000-character output cut, a version bump where a blind replace-all rewrites history, a rename with a near-miss name, a read-only question with a stale doc as a decoy, three independent lookups that could go to parallel subagents, and a bug report whose obvious fix, in a shared helper, breaks another caller that only the tests show (`shared-helper`, with a prompt that never mentions tests).

Each run gets a fresh copy under the system temp folder and runs `kyklos.py --eval` there with nobody at the keyboard. `--eval` swaps the person for a fixed policy — approve edits and commands, including ones Kyklos can't read, but never a red finding or a secret — and Kyklos refuses it unless the folder is one `eval.py` made (inside the temp folder, with a marker file) and the OS sandbox is on. The check runs the model's code too, so it runs in the same sandbox. Your model picks and model catalog are copied in; nothing else from `~/.kyklos` is. A persona is off unless `--persona` names one. Results are appended to `evals/results/<time>.jsonl`, one row per run, which is kept out of git.

**Measuring self-checking.** Each run's line ends with the state of the code when the turn ended (`code checked`, `unchecked` or `failing`), and its `self_check:…` signals count how often the model was asked. `--no-self-check` runs the same tasks with `KYKLOS_SELF_CHECK=0`, so the two can be compared on the task built for it:

```bash
./.venv/bin/python eval.py shared-helper --repeat 5
./.venv/bin/python eval.py shared-helper --repeat 5 --no-self-check
```

`audit.py` measures the other side without spending tokens: whether self-checking recognises a check command when it sees one (and nothing else), whether it recognises an answer that already says why nothing was checked (without taking a plain description of the code for one), and, given saved sessions, how often it would hold an answer back:

```bash
./.venv/bin/python audit.py ~/.kyklos/projects/*/last.jsonl
```

A replayed session sees only the file tools' edits, since command edits need the workspace, so live counts can be higher. Its first run found two gaps that are now fixed: `can't` never read as a reason, and a bare `not` read descriptions like "the migration does not run twice" as one.

**The trace.** Every session writes one, not only evals: a JSONL row per event to `~/.kyklos/projects/<folder>/traces/` (the newest 50 kept, owner-only permissions). It records numbers and fixed labels only — tokens and cache reads per API call, each tool call's outcome, size and duration, and each time the harness stepped in (a nudge, a budget warning, a repeat, a parallel batch) — and never a path, a command, a prompt or an answer, so it can be read or shared without exposing the work. `KYKLOS_TRACE=0` turns it off.
