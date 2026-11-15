# Kyklos serve protocol

**Status: protocol version 0, implemented in [`serve.py`](serve.py) and tested in [`test_serve.py`](test_serve.py). Version 0 means it may still change in breaking ways.**

`kyklos --serve` runs the same agent the terminal runs, without drawing anything. Instead it speaks
JSON-RPC over stdin and stdout to whatever started it: a desktop app, an editor, a test. The
terminal UIs stay what they are; this is a third front end beside `AeroUI` and the plain REPL, and
the one every later front end is built on.

It is possible without touching the loop because of a choice made long ago: the `Agent` doesn't
print. It emits events through `on_event`, asks through `permission`, and the startup questions
(`ask_trust`, `ask_bash_mode`) and slash commands (`CommandContext`'s `say`, `pick`, `render`) are
all callbacks. `--serve` supplies a set of those callbacks that write JSON instead of drawing.

## 1. Principles

1. **Kyklos decides; the client draws.** Every judgement the terminal makes before you answer a
   prompt is made on the server and sent: the explainer's findings, which answers are on offer, the
   suggested prefix, the diff an edit would make. A client never re-derives any of it, so a client
   bug can make a prompt look wrong but cannot make it permissive. This is the rule
   `permission_options` already enforces between the plain REPL and Aero, extended to a third.
2. **A client can't widen a grant.** An answer is checked against the choices that prompt offered.
   `always` on a prompt that withheld it is refused the same way the terminal refuses a key it
   didn't show. `apply_permission_answer` remains the only code that turns an answer into a rule.
3. **stdio only.** No socket, no port. The only thing that can answer a permission prompt is the
   process that started Kyklos. A listener on `localhost` would let any local process, or a web
   page via DNS rebinding, approve commands (see §8).
4. **Everything the server sends is untrusted text.** Model output, tool results, file contents and
   paths are data. Clients render them as text and never as HTML. Markdown answers go through a
   sanitizer with raw HTML off. In a webview with native bindings, one unescaped answer is code
   execution.
5. **The stream keeps its shape.** Events arrive strictly nested, exactly as the terminal sees
   them, because `EventLanes` already serializes parallel subagents. One turn is in flight at a
   time. A client can be a straightforward state machine.
6. **Parity before features.** v0 carries what the terminal shows, and nothing the terminal
   lacks. A desktop-only capability goes into the agent first, and both front ends get it.

## 2. Transport

- **Framing:** one JSON-RPC 2.0 message per line, UTF-8, terminated by `\n`. No `Content-Length`
  headers, and no batches.
- **stdout is the protocol and nothing else.** On startup `--serve` keeps the real stdout file
  descriptor for itself and points `sys.stdout` at stderr. A stray `print` anywhere, from Kyklos or
  a library, then lands in the log instead of corrupting the stream.
- **stderr is a log** for humans. Clients may show it in a debug pane and must not parse it.
- **Both sides send requests.** The client asks the server to run a turn; the server asks the
  client to approve a call. IDs are per direction: the client uses integers, and the server uses
  strings prefixed `k` (`"k17"`), so the two can't collide in a log.
- **Unknown is ignored.** A client ignores event names and fields it doesn't know. A server
  answers an unknown method with `-32601`. Everything additive can ship without a version bump.

The client starts Kyklos with the workspace as its working directory, exactly like a terminal:

```bash
kyklos --serve                     # also accepts --profile NAME, --plan, --trust
```

`.env`, profiles, personas and saved picks are read exactly as they are in the terminal. A desktop
app adds no configuration of its own.

## 3. Lifecycle

```
client                                   kyklos --serve
  │ initialize ─────────────────────────────▶│
  │◀──────────────── result: versions, sandbox, commands
  │ session/start {resume} ─────────────────▶│
  │                                          │ folder not yet trusted?
  │◀────────────────────── trust/request ────│   (before the Agent exists:
  │ {choice: "once"} ───────────────────────▶│    it reads AGENTS.md)
  │◀────────────────── session/bash_mode ────│
  │ {mode: "ask"} ──────────────────────────▶│
  │◀──────── result: state, resume note ─────│
  │                                          │
  │ session/prompt {text} ──────────────────▶│
  │◀─────── session/update (turn_start) ─────│
  │◀─────── session/update (tool_call) ──────│
  │◀──────────────── permission/request ─────│
  │ {answer: "y"} ──────────────────────────▶│
  │◀─────── session/update (tool_result) ────│
  │◀─────── session/update (turn_end) ───────│
  │◀─────── session/state ───────────────────│
  │◀──────── result: answer, outcome ────────│
  │                                          │
  │ shutdown ───────────────────────────────▶│ saves the session, exits 0
```

The order of `session/start` matches `main()`: trust is decided first, then the Agent is created,
then the bash mode is chosen, then a resume is loaded. `--trust` and `KYKLOS_TRUST=1` skip the
trust request, as they do in the terminal, and also skip the bash-mode question (the session runs
in `ask`).

**EOF on stdin is a hang-up.** Any pending permission request resolves as `NO_APPROVER_NOTE`, the
turn in flight is cancelled (§6) and saved, and the server exits. Every turn is saved when it
ends, as in the terminal, so `shutdown` has nothing left to save.

Startup failures that happen before `initialize` can be answered (a missing `BASE_URL`, an unknown
profile) exit non-zero with the reason on stderr. A client shows stderr when the process ends
before the handshake.

## 4. Client → server

| Method | Params | Result |
|---|---|---|
| `initialize` | `protocol_version`, `client: {name, version}` | `protocol_version`, `kyklos_version`, `workspace`, `sandbox: {kind, label, warning}`, `commands: [{name, usage, summary}]`, `profiles`, `profile` |
| `session/start` | `resume: null \| "last" \| path` | `state` (§7.3), `note: {kind, text} \| null` |
| `session/prompt` | `text` | `answer`, `outcome`, `steps`, `secs`, `claims: [{label, missing}]` |
| `session/cancel` | *(notification)* | none: the in-flight `session/prompt` returns with `outcome: "interrupted"` |
| `command/run` | `text` (the whole line, `/model gpt-5`) | `then_send: string \| null`, `exit: bool` |
| `session/state` | none | `state` |
| `shutdown` | none | `null`, then the process exits |

- **`session/prompt`** is `Agent.send`. `outcome` is the agent's `last_outcome` (`answer`,
  `cut_off`, `step_limit`, `repeat_stop`, `compaction_stop`, `interrupted`, `error`). `claims` is
  `last_claims`: answer claims the turn didn't back up, which the terminal prints as `! this answer
  says it …`. The client shows them, and no client may hide them. The session is saved before the
  result is sent, exactly as `AeroUI.turn` does.
- **`command/run`** goes through `run_command` and the shared `COMMANDS` registry, so every slash
  command, including ones added later, works in every client with no client change. A handler that
  returns text to send (`/plan <request>`) comes back as `then_send`, and the client then sends it
  with `session/prompt` itself. There is one way to start a turn.
- **Busy:** while a turn runs, only `session/cancel`, `session/state` and `shutdown` are accepted.
  Anything else gets error `-32001 busy`. The terminal can't run `/model` mid-turn either.
- **Errors:** `-32001 busy` (above). `-32002 api_error` is an API failure that ends a turn, with
  `data.detail` from `error_detail`. `-32003 not_ready` is anything but `initialize` sent first, or
  a turn or command sent before `session/start` has been answered. `-32004 untrusted` means folder
  trust was declined, after which the server exits. `-32005 command_failed` is a slash command
  that raised. The JSON-RPC codes `-32700`, `-32600`, `-32601` and `-32602` mean what they always
  mean.

## 5. Server → client requests

The server waits on these with no timeout, because a person may be away from the desk. It waits
in a loop that a cancel can interrupt (§6).

### `permission/request`

Sent wherever the terminal would show its yellow panel.

```json
{"jsonrpc": "2.0", "id": "k4", "method": "permission/request", "params": {
  "depth": 0,
  "call_id": "call_8f2",
  "tool": "bash",
  "args": {"command": "rm -rf build && make"},
  "display": {"kind": "command", "command": "rm -rf build && make"},
  "findings": [{"level": "danger", "text": "deletes build/ and everything in it"}],
  "choices": ["y", "n"],
  "prefix": null,
  "touches_secret": false
}}
```

A danger finding withholds both `a` and the prefix, so this call can only be approved once. The
same command without the `rm` would come with `"prefix": "make"` and `"choices": ["y", "n", "p",
"a"]`.

- `display.kind` is `command` (bash: `command`), `diff` or `args`. A `diff` is for edit_file or
  write_file and has `path`, `old`, `new`, `first_line`, `whole_file`, `added`, `removed`,
  `new_file` and `overwrite`. When `whole_file` is set, `old` and `new` are the entire file before
  and after, and worth trimming to what moved. `args` covers anything else, with values cut at 600
  characters as the terminal cuts them. The diff is computed on the server from the file as it is
  now and the edit as it would apply. The client gets no file access and doesn't reimplement
  `apply_edits`.
- `findings` are `explain_command`'s, plus the remote-provider warning `permission_options` adds,
  and are shown before the choices, in order. Levels are `danger`, `warn`, `note` and `opaque`.
  Colour alone never carries the level (DESIGN.md §1.2).
- `choices` is the exact set the terminal offers for this call: `y`, `n`, plus `p` when there is a
  prefix, plus `a` when the call touches no secret and has no danger or opaque finding.

Result: `{"answer": "y" | "n" | "a" | "p", "prefix": string?, "reason": string?}`. `prefix`
overrides the suggested one for `p`; `reason` goes to the model with a denial (`denial_note`).

An answer outside `choices` is never read as anything. A JSON-RPC response can't itself be
refused, so the same request goes out again under a new id, with `rejected: {answer, reason}`
added. A confused client gets asked again rather than turning into an accidental yes. An error
response, or no answer before the client hangs up, is a denial with `NO_APPROVER_NOTE`.

### `trust/request`

`{folder, broad, explanation: TRUST_EXPLANATION, warning: BROAD_WARNING | null, choices}`. When
`broad` is set, `choices` drops `yes`, as the terminal does. Result: `{choice: "yes" | "once" |
"no"}`. `yes` is remembered in `~/.kyklos`, the same file the terminal uses, so trusting a folder in
one front end trusts it in both.

### `session/bash_mode`

`{explanation: BASH_MODE_EXPLANATION, caveat: BASH_MODE_CAVEAT, sandbox_warning: string | null,
choices: ["ask", "unflagged"]}`. Result: `{mode: "ask" | "unflagged"}`, and anything else is
`ask`. It is never remembered, as in the terminal. The client must
show `sandbox_warning` next to the choice, because that is the one place it changes anything.

### `ui/pick`

`CommandContext.pick` for `/model`, `/provider`, `/persona`, `/effort`, `/submodel`: `{title, rows:
[[cell, …]], cursor, marked, filterable}`. Result: `{index: int | null}`, where `null` is cancel.

## 6. Cancellation

The terminal stops a turn with Ctrl+C, and the agent is built around that: `KeyboardInterrupt`
kills a running command's whole process tree (`tools.py` `kill_process_tree`), stops parallel
subagents through `EventLanes.close`, and lets `repair_after_interrupt` give unanswered tool calls
a placeholder result. That path is tested, and `--serve` reuses it rather than adding a second
way to stop.

- The agent runs on the **main thread**. A reader thread owns stdin and puts messages on a queue.
- `session/cancel` makes the reader thread send `SIGINT` to the main thread specifically:
  `signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)`. Aiming at the thread
  matters. A process-wide signal, or `raise_signal` from the reader, can land on the reader thread,
  which sets Python's flag but leaves the main thread asleep in `communicate()` until its timeout.
  Delivered to the main thread, the signal breaks the blocked call, and Python raises
  `KeyboardInterrupt` there exactly as a keypress would.
- An outstanding `permission/request` is withdrawn with a `request/withdrawn {id}` notification,
  and its late answer is ignored.
- `session/prompt` then returns normally with `outcome: "interrupted"`, after
  `repair_after_interrupt` and a save, which is what `AeroUI.turn` does.
- While serving, the SIGINT handler raises only during a turn or command. A cancel that arrives
  just after a turn ended, or a Ctrl+C in the terminal that launched the client, does nothing.
  Only the client decides when the process ends.

Open question: Windows has no `pthread_kill`. The candidates are `GenerateConsoleCtrlEvent` on
Kyklos's own process group, or running the agent on a worker thread and cancelling through
`Agent.cancel` plus `kill_process_tree`. Either needs a test before Windows support is claimed.

## 7. Server → client notifications

### 7.1 `session/update`: agent events

Every `on_event` call becomes one notification:

```json
{"jsonrpc": "2.0", "method": "session/update",
 "params": {"seq": 41, "depth": 1, "event": "tool_result", "data": {"call_id": "call_8f2", "name": "grep", "result": "…", "secs": 0.04}}}
```

`seq` increases by one per notification for the whole session, so a client can tell it missed
something. `depth` is 0 for the main agent and 1 inside a subagent. Data fields are the agent's own
keyword arguments, unchanged:

| Event | Data | Notes |
|---|---|---|
| `turn_start` | — | |
| `turn_end` | `steps`, `outcome`, `answer_chars`, `claims` | Not sent for a cancelled parallel subagent |
| `model_start` | `purpose` | `turn` or `summary` (compaction) |
| `model_end` | `purpose`, `secs`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `finish_reason`, `error` | Usage fields may be missing; see the `no_usage` notice |
| `answer_delta` | `text` | Transient: may be discarded and redrawn by a retry. Never put it in history; the answer is `session/prompt`'s result |
| `thought` | `text` | Content written alongside tool calls |
| `tool_call` | `call_id`, `name`, `arguments` | `arguments` is the raw JSON string the model sent, which may be invalid |
| `tool_result` | `call_id`, `name`, `result`, `secs` | `result` is already secret-scrubbed. Status is read off its first words, as `result_status` does |
| `subagent_start` | `description`, `mode`, `model`, `effort`, `routed` | |
| `subagent_end` | the same, plus `steps`, `secs` | `secs` is wall time; a parallel lane's events may have been replayed late |
| `plan` | `steps`, `active` | The resolved plan, never the call's arguments |
| `compact` | `stage`, `before`, `after`, `pruned` | `stage` is `pruned` or `summarized` |
| `retry` | `attempt`, `max_attempts`, `delay`, `error` | |
| `notice` | `text` | A caveat meant for the person, said once per session |
| `image` | `path`, `rel`, `mime`, `width`, `height`, `bytes` | `generate_image` saved an image: `path` is absolute, `rel` is workspace-relative. Sent before that call's `tool_result` |
| `signal` | `kind`, plus `reason`, `left`, `repeats`, `size`, `report`, `count` | The harness stepping in: `nudge`, `budget_warning`, `repeat_warn`, `repeat_stop`, `redacted`, `parallel_batch`, `delegation_hint`, `truncated_call`, `map_note`, `images_sent`, `stopped_report` |

**Nesting guarantee.** At each depth, a `tool_result` closes the most recent `tool_call` (and
`call_id` makes that checkable). A `task` call's `tool_call` and `tool_result` bracket its
subagent's `subagent_start` … `subagent_end`. Parallel explore tasks arrive one lane after
another, never interleaved.

### 7.2 Command output

- `say {kind, text}`: `CommandContext.say`, where `kind` is `info`, `ok`, `warn` or `error`.
- `view {name, data, fallback}`: `CommandContext.show` for `tokens`, `tools` and `diff`. A client
  that has no drawing for a view shows `fallback`, the plain text the REPL prints. A new view
  therefore works everywhere on the day it ships.

### 7.3 `session/state`

This is pushed after every turn, every command and every `model_end`, and is also returned by
`session/start` and the `session/state` request. It is the status bar's data:

```json
{"model": "…", "effort": "default", "submodel": null, "profile": null, "persona": null,
 "plan_mode": false, "auto_bash": false, "sandbox": "seatbelt",
 "context_tokens": 18234, "context_window": 128000, "compact_at": 96000,
 "session_tokens": 51022, "cache_read_ratio": 0.62, "can_undo": true}
```

A client keeps no copy of this state that could disagree with the agent's. It redraws from the
latest snapshot.

## 8. Security

What the terminal guarantees, and how each guarantee survives a second front end:

| Guarantee | In the terminal | Under `--serve` |
|---|---|---|
| Only the user approves | Only the person at the keyboard can press `y` | Only the parent process holds the pipes. No socket exists to connect to |
| Prompts can't be talked past | The keys shown are the keys accepted | `choices` is enforced server-side; anything else is asked again |
| You see what it does before answering | Findings and diffs printed above the keys | Sent in the request; the client must show them before its buttons |
| Secrets stay out | Scrubbed from tool output | Scrubbed before the event is built, the same code path |
| Terminal escapes can't spoof the screen | `sanitize_terminal_text` | Not needed for JSON, but the client must still treat text as text (§1.4) |
| Folder trust before any read | Asked before the Agent exists | Asked inside `session/start`, before the Agent exists |
| Unflagged bash lasts one run | Never saved | Never saved; the client must not store it either |

Two client obligations follow and belong in any client's own review:

- **No click-through.** A permission prompt needs the same deliberateness as a keypress. No
  auto-focused *Allow* button, and Enter doesn't mean yes. A `danger` finding gets the coral
  treatment and can't be approved with an `always`.
- **No HTML from the model.** Render answers, thoughts and results through a Markdown renderer
  with raw HTML disabled, and open links only on an explicit click, in the system browser.

A later WebSocket transport (for `kyklos --web`) is not part of v0. If it comes, it binds to
`127.0.0.1`, requires a random per-launch token in the first message, and checks `Origin`.

## 9. Where it lives in the code

- **[`serve.py`](serve.py)**: `ServeUI`, with the same surface as `AeroUI` (`observer`,
  `permission`, `ask_trust`, `ask_bash_mode`, `pick`, `say`, `render_view`, `run`), plus the
  transport and the reader thread. It needs nothing beyond the standard library.
- **`main()`** takes `--serve` before anything can print, calls `claim_stdio`, and then goes through
  the same startup as the terminal, with `ServeUI` in place of the UI. The trace wraps its observer
  as it wraps Aero's.
- **Shared with the terminal:** `edit_preview` (the before/after text an edit prompt shows, which
  `AeroUI._diff_source` now calls too), `state_snapshot` (§7.3), and `call_id` on `tool_call` and
  `tool_result`, which the other observers ignore. `CommandContext.show` now passes the plain
  fallback to the renderer.
- **Not yet shared:** Aero's status line still reads the agent directly instead of going through
  `state_snapshot`.

## 10. Relation to the Agent Client Protocol

Zed's Agent Client Protocol (ACP) has the same shape: JSON-RPC over stdio, `initialize`, a prompt
method, streamed session updates, a permission request, and cancel. The method names above follow
it where the meaning matches. ACP has no slot for findings, prefix grants, folder trust or command
views, so Kyklos speaks its own protocol, and an ACP adapter, if one is wanted, translates down to
it and drops what ACP can't carry. Check the current ACP spec before building that; this section is
about shape, not wire compatibility.

## 11. Open questions

- Windows cancellation (§6).
- Should `answer_delta` be coalesced (for example, every 50 ms) to spare a webview thousands of
  tiny messages? This is probably not worth doing before there's a measurement.
- Sessions: v0 is one session per process, and a client wanting two folders starts two processes.
  This matches the per-folder session model and keeps trust per folder.
- `/undo` and `/diff` touch files. Does a client need a `files/changed` notification so an open
  editor tab can reload, or is the `diff` view enough?
