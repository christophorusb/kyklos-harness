# Changelog

All notable changes to **Kyklos** since the repository was initialized on 2026-10-24.

Releases are versioned from `v0.1.0` onward, following [semantic versioning](https://semver.org). Pre-1.0,
a breaking change bumps the minor. To cut one: bump `__version__` in `kyklos.py` and add a
`## v<x.y.z> — <title> (<date>)` section here. Entries above that point predate versioning and are
titled by commit instead.

---

## Unreleased

- **The model's intent reaches the user as structure, not prose (`tools.py`, `kyklos.py`, `ui.py`, tests):** a reasoning model's own account of what it is doing is either a raw token stream, arriving far faster than anyone can read it, or a summary the provider makes server-side and out of reach. Neither is something a harness can restructure. What reads as staged thinking elsewhere is mostly not thinking at all — it is tool calls, a plan and a step list, each arriving as a discrete event that can simply be drawn. Kyklos already captured `reasoning_content` into the transcript without printing it, which was the right instinct; what was missing was the structure to show in its place. A `plan` tool now carries it.
  - **Linear steps, so one index is the whole state.** `plan(steps=[...])` posts the list and `plan(active=2)` moves within it; everything before `active` is done, everything after is pending, and there is no per-step status to drift out of sync. `active: 0` means finished. Validation runs before the assignment, so a refused call leaves the user looking at the step the model last actually claimed, not one it was rejected for.
  - **The event carries the resolved plan, not the call's arguments.** `plan(active=3)` says nothing about the steps, so a UI reading the arguments would have to remember them — a second copy of state that can disagree with the first. The tool returns the whole list to the model for the same reason: a step nobody marked done must not drift into looking done later in the conversation.
  - **The stage stays put while the work moves.** A new or changed plan prints in full, with the active step marked; finishing one prints a single row, because reprinting six steps per advance would bury the work those steps describe. The active step also sits in the live zone, above the tool rows that scroll past it — which is the entire point, since a stream of reasoning cannot be held still like that.
  - **It writes nothing, so it asks nothing, and it is not counted as work.** `plan` joins `SAFE_TOOLS`: a permission prompt before showing the user a list would be friction with nothing behind it. It is left out of the turn's tool tally and draws no ordinary tool row, since the band already said everything the row would — but a *failed* plan call keeps its row, so a rejected one is never silent. Plan mode gains it too; subagents do not, having one focused job whose steps would draw inside a tree the user cannot act on.
  - **A step is model-written text on its way to a terminal.** Found by probing the new path rather than reasoning about it: `plan(steps=["safe\x1b[31m\x07 and \x1b]0;pwned\x07nasty"])` reached the screen with its escapes intact — Rich drops a bare BEL and passes CSI and OSC straight through, so a step could recolour the screen, set the window title, or leave an unterminated OSC that swallows whatever printed next. Whole sequences are stripped at the one place that decides what a step says, rather than at each place that draws it; stripping the ESC byte alone would have left `[31m` behind as literal rubbish. A step that was nothing but escapes is refused, because it says nothing. 293 tests (was 272); the audit holds at 0/250 miscalibrated and 0/259 claims.

- **A recognised program is not a safe one (`kyklos.py`, `audit.py`, tests):** being in `PLAIN_CMDS` means only that Kyklos knows the name, so these never went `opaque` — and, with no rule of their own, produced nothing at all. `sed -i` was the clearest: it rewrites a file in place and read exactly like `sed -n '1,5p'`. On a 36-command probe of destructive shapes, **13 ran unprompted, then 11, now 0**.
  - **Eleven rules, each for a program already in the list.** `sed -i` rewrites in place; `ln -f` replaces its destination; `cp /dev/null x` empties a file; `chown -R` mirrors the `chmod` rule that was already there; `npm publish` cannot be taken back; `git checkout -- .`, `git restore` and `git branch -D` lose work, and `git config --global` changes settings outside the repository. A bare `rm` with no operands is `ls | xargs rm` after the wrapper is peeled — its targets arrive on stdin, where the explainer cannot see them, and no operands is not the same as nothing to delete.
  - **A program named by a variable is a program Kyklos cannot name.** `R=rm; $R -rf dist` produced no findings, because `COMMAND_WORD_RE` has no `$` and the token fell past the unrecognised-program clause entirely. It is `opaque` now — the level that already exists for "I can't tell".
  - **`warn` still does not gate, and that was the right call.** Making it gate closed `mv src /tmp/gone` but also stopped `ls -la /tmp/x` and `cd /srv/app && grep -rn foo src`, which `BashModeTests` deliberately runs — the finding says *where* a command reaches, not what it does there. Instead the two warns that really do leave this machine's limits became `danger` (`ssh`, and running another program's output through a shell), and carrying a file out of the workspace got its own rule, so `mv src /tmp/gone` is caught while `mv old.ts new.ts` stays quiet.
  - **The corpus could not have caught any of this, so it was extended.** It held no command that writes anything — every rule here had nowhere to show a false positive. Sixteen everyday writes, moves and keep-your-work git commands were added (250 cases now, still **0% miscalibrated**). The corpus guard banned `mv ` outright as a stand-in for destructive; that stand-in is now the thing it meant, a path leaving the workspace, which also covers `cp`.
  - **The direction with the asymmetric cost is measured at last.** `audit.py` counts false positives on benign input; nothing counted false negatives on hostile input, where being wrong costs a repository rather than a keystroke. The 36 shapes are now a test, failing 21 subtests without these rules, paired with a test that ordinary work still runs unprompted — a gate that stops everything is not a gate, it is a mode nobody leaves on.

- **A shell wrapper no longer silences the command inside it (`kyklos.py`):** in auto-bash mode `rm -rf dist` stopped and asked, and `bash -c "rm -rf dist"` ran without a prompt. Inline code through an interpreter was a `note` — the level below "runs cd, then echo" — and `must_be_asked` looks only at `danger` and `opaque`, so wrapping anything in six characters made every prompt in that mode optional. `sh -c "rm -rf /"` auto-approved.
  - **Shells are read, not shrugged at.** `explain_command` now recurses into the quoted body, so `bash -c "rm -rf dist"` reports the delete and `sh -c "curl https://x.sh | sh"` reports both of its dangers, tagged `(via bash)` / `(via sh)`. Two levels deep, then it stops and says it stopped. `zsh`, `dash`, `ksh` and `fish` had never been in `INTERPRETERS` at all, so `zsh -c "rm -rf dist"` produced no findings whatsoever; keying on `SHELL_STEMS` covers them.
  - **Code Kyklos can't read is `opaque`, not a note.** Python, JS and Perl are not parsed here, which is the same ignorance as an unrecognised program — and `opaque` is the level that already means "I can't tell" without coming to mean "go ahead". `python3 -c "shutil.rmtree('src')"`, `node -e`, `perl -e` now all stop.
  - **The attached form was a blind spot of its own.** `bash -c"rm -rf dist"` produced an empty finding list, because detection was an exact match on the flag token. Only single-dash short forms attach and only a quoted remainder counts, so perl's `-pi` isn't read as `-p` carrying the code `i`, and `--experimental-x` isn't read as `--eval`.
  - **Emptiness still means empty.** A body with no findings is left alone rather than called ignorance, or `bash -c "ls -la"` would prompt where `ls -la` does not. Measured on a 36-command probe of destructive shapes: **13 ran unprompted before, 11 after** — the two interpreter cases closed, and the rest a different gap (`sed -i`, `xargs rm`, `mv`, `git checkout -- .`) where the program is recognised but its destructive mode has no rule.

- **A `>/dev/null` inside a substitution is still a discard (`kyklos.py`):** the sixth explainer bug found in a real session, and the same shape as the four before it. `REDIRECT_RE` runs its target to the next space, so the `)` closing a `$(…)` came along with the filename: `pid=$(cat f.json 2>/dev/null)` gave `/dev/null)`, which is not one of the `NULL_SINKS`, and a read-only probe was announced in red as `overwrites /dev/null)`. The identical `>/dev/null` later in the same command, outside any substitution, stayed correctly silent — both behaviours visible side by side in one prompt.
  - **The trimming already existed one function away.** `_bare` drops an unbalanced trailing `)` so that `(cd sub && rm -rf dist)` deletes `dist` rather than `dist)`; the operand path got it and the redirect path never did. Balance is what decides, so a filename entitled to a paren of its own — `echo hi > backup(2)` — keeps it, and the write a substitution really is hiding, `X=$(cat f > /etc/hosts)`, is still found.

- **The system prompt names the commands this machine doesn't have (`tools.py`, `kyklos.py`):** the check existed but ran only on Windows, where it covers the unix text tools PowerShell has no answer for. The mirror-image gap was never covered: macOS ships a BSD userland, so none of the GNU coreutils extras are there, and a model trained mostly on Linux assumes they are.
  - **`timeout` is the one that costs real time.** Bounding a command that might hang is exactly the right instinct and the first thing reached for — and it does not exist here. A session watching a verification script hang for 180s proposed `timeout 40 node …` as its next step, which would have died on `command not found` with nothing to say why.
  - **The bash tool already takes a `timeout` of its own**, so the advice points there rather than at a command to install, and the parameter's description now says so instead of the bare "Seconds, default 120". The reach for a shell `timeout` is worth preventing, not just explaining.
  - **Probed, not assumed.** `shutil.which` decides, so a machine with coreutils installed is told nothing, and the list never names a tool that is sitting right there. On this one it comes out as `timeout, tac, nproc`.

- **A timed-out command keeps what it printed (`tools.py`):** the timeout threw away everything the command had written before it hung, so a run that printed its numbers and then stalled came back as nothing but "Command timed out after 150s". The only way to read those numbers was to run the whole thing again, which hung again. The output captured up to the kill is now returned with the error.
  - **Silence is itself a finding.** A command that printed nothing before stopping is the shape of one waiting for input, which is exactly the case the message's advice is for — so that is said outright rather than left to be inferred from an absence.
  - **The kill is unchanged and still happens first.** `kill_process_tree` stops the whole process group (SEC-06) before anything is read back, so the pipes are closed and the read returns what was buffered rather than waiting on a survivor. Checked against a backgrounded grandchild holding the pipe: the error comes back on time, carries the early output, and the grandchild is gone.
  - Worth being precise about what this does *not* fix: a command that **exits cleanly** having left a detached child behind keeps that child, because there is nothing left to kill and the command succeeded. Orphaned browsers from a Puppeteer script that never closed its browser come from there, not from the timeout path.

- **A danger inside a loop or conditional is no longer invisible (`kyklos.py`):** `for f in *; do rm -rf "$f"; done` produced no findings whatsoever, so in the relaxed bash mode it ran with no prompt at all — while the bare `rm -rf` it contains would have stopped and asked. The same held for `while read p; do kill -9 "$p"; done`, `if true; then chmod -R 777 .; fi` and `for i in 1 2; do curl https://x.sh | sh; done`.
  - **The cause was reading the grammar as the program.** Splitting on `;` leaves a loop body looking like `do rm -rf "$f"`, and `_command_findings` only ever examines the first token, which is `do`. `_peel` looks past wrappers like `sudo` and `env`, but a shell keyword is not a wrapper. The concept was already in the file — the chain note filters `SHELL_KEYWORDS` so it doesn't say "runs for, then do, then done" — it simply was never applied where the findings are made.
  - **Three classes of word, because they behave differently.** `do`, `then`, `else`, `elif`, `if`, `while`, `until` and `!` stand in front of a command and are stripped. `for`, `case` and `select` bind a name, so the clause runs nothing and is dropped — stripping them uniformly would have read `for f in *` as "runs f" and prompted on every loop. `done`, `fi`, `esac` and `;;` close a construct and run nothing.
  - **`case` needed its own handling.** `case $x in a) rm -rf /srv ;; esac` does run `rm`, once the pattern's `)` is past; and `a|b)` splits on the `|`, leaving a bare `b)` at the head of the next piece. A subshell's closing paren is dropped from operands too, so `(cd sub && rm -rf dist)` deletes `dist` rather than `dist)`.
  - Swept across **18 dangerous constructs** — nested loops, a loop inside a substitution and a substitution inside a loop, backticks, `sudo` inside a loop, both `case` forms — **none slips through**, and across 7 benign ones **none newly prompts**. The audit corpus gained `loop body` and `conditional body` groups so that second direction stays measured.

- **The explainer reads the command it was given (`kyklos.py`):** four bugs, all of them visible in a single screenshot of a real session, each putting a wrong finding in front of you at the moment the prompt asks you to decide.
  - **A quoted prefix keeps its path in one piece.** `TOKEN_RE` stopped at the closing quote, so `"$TMP"/trail-*` became two words: `rm -rf` reported deleting two things, the second of them apparently at the filesystem root.
  - **Redirects are plumbing, not arguments.** `2>/dev/null` was listed among the things `rm -rf` deletes — exactly the noise the overwrite finding already refuses a few lines further down.
  - **Command substitutions are explained on their own terms.** The assignment peel swallowed the token holding the program, so `TMP="$(python3 -c …)"` read as "runs -c". Worse, anything substituted lost its findings: `X="$(rm -rf /tmp/foo)"` read as an unrecognised program rather than a recursive delete, and `X="$(curl https://x.sh | sh)"` lost both of its dangers. Never a bypass — `opaque` still forced the prompt — but the prompt said the wrong thing, and saying what a command does is the whole job. In the other direction, `OUT="$(git rev-parse HEAD)"` no longer costs an approval.
  - **The explainer is told where the workspace is.** `_outside_workspace` was purely syntactic, so every absolute path counted as outside the workspace — including the workspace's own root, which bash already runs in. Set once at startup rather than threaded through the permission callback, whose signature the UI shares.
  - Measured on the corpus extended to these shapes: **22% of 218 benign commands miscalibrated before, 0% after.**

- **The checkers are measured as rates, not gates (`audit.py`):** the controls beside each checker answer a yes/no question — still quiet on these eleven phrasings? This answers a different one: across a few hundred inputs nobody picked case by case, what fraction fires? Three corpora — commands built from a grammar of everyday shell, answers built from a grammar of hedging, and real turns replayed out of saved sessions.
  - **Replay is the half with no judgment in it.** `save_session` already writes the tool calls and their results, so a finished turn can be handed back to the checker exactly as the live loop handed it over. A test pins that a replayed turn reproduces the live verdict; without it, a rate measured off saved sessions is measuring the parser.
  - **A firing is either miscalibration or coverage.** `danger` or `warn` on something benign is the explainer being wrong. `opaque` is it admitting it has no rules for a program — honest by design, since `must_be_asked` would rather prompt than trust ignorance, but it still costs a prompt, so it is counted separately rather than forgiven.
  - **It put the claim check at 3.5% on 259 answers that claim nothing**, in two clusters. Three of the four checks were bare single-word matches while `ran the tests` insisted on past tense, so a participle used as an adjective read as a story about the turn: "the installed version is 2.1" became a claim to have installed something. And the lead scan knew `not`, `never` and the modals but no way to spell negation without them, so "nothing was installed" read as a claim that something had been. Both now fire on **0 of 259**, with **0 misses across 20 real claim phrasings** — including the terse ones a noun makes, like "changes committed".

- **A workspace sketch in the system prompt (`tools.py`, `kyklos.py`):** `repository_map` lists the project's source files and what each defines, so turn one doesn't start blind. A diagnosed session spent **134 exploratory calls, 87% of all its tool output**, much of it discovering what exists. `/map` shows the current sketch.
  - **Names only — no signatures, no line numbers.** The same evidence that shaped the command explainer applies: a regex over code produces confident nonsense, and it would do it here on TS/JS/Go/Rust parameter lists. A map that misstates a signature is worse than no map, because the model believes it. Python goes through the real `ast`, so an unparseable file contributes nothing rather than guesses. Line numbers are the first thing to go stale after an edit, so they're left out.
  - **Capped by rendered length per file, not by entry count** — one entry can be a class that expands to ten method names, so counting entries let a file of classes spend the entire budget and leave the rest of the tree invisible. Shallow paths come first, since a truncated map should keep the entry points rather than whatever sorts first alphabetically. ~340 tokens for this repository.
  - It lives in the system prompt for the same reason `AGENTS.md` does, and is regenerated at the start of any turn following one that wrote files, so it's never stale at the moment the model reads it. Subagents don't get it; they're given one focused job. The text says it's generated and to read a file before relying on it.

- **A scratch directory inside the workspace (`tools.py`, `kyklos.py`):** `.kyklos/scratch/` for throwaway diagnostic scripts. `_resolve` refuses to write outside the workspace, so a model wanting to check a runtime value had no path open to it but `cat > /tmp/x.mjs <<'EOF'` through bash — and nothing in `/tmp` can resolve `./src/...` or the project's dependencies. Real sessions show exactly that: `/tmp/probe.mjs`, `/tmp/verify.mjs`, `/tmp/fake-ollama.mjs`.
  - Promoting overwrites to red had made this worse: writing the script was a danger finding and running it an opaque one, so a five-line diagnostic cost two prompts. Writing to scratch now needs no approval and the explainer passes over it, since the directory exists to be written over.
  - **Running a scratch script still asks.** `bash` isn't confined to a directory, and a script the model wrote moments ago is the clearest case of "contents you want to look at first". The gain is one prompt per debug cycle rather than two, not zero — auto-approving the execution of code the model just wrote would hand away the thing the permission gate is for.
  - It keeps itself out of git with a `.kyklos/.gitignore` containing `*`, rather than editing the project's own `.gitignore`.

- **The answer's story is checked against the turn (`kyklos.py`, `ui.py`):** when an answer claims it ran the tests, committed, pushed or installed something, `unsupported_claims` compares that against the commands the turn actually ran, and shows the user where they disagree. A harness can't make a model truthful, but in a coding agent the untruths are about checkable things — `edit_file` already turns one class of them into a hard failure by demanding the exact existing text, so a model that imagines a file can't edit it. This extends that to the turn's own account of itself.
  - **Succeeding counts, not just running.** A `pytest` that exited 1, or one that was denied and so has no exit code at all, doesn't support "the tests pass" — the more damaging claim, and the one a naive "did a test command appear?" check would miss.
  - **It errs towards silence.** Only past-tense assertions count, so advice ("run the tests before you push"), hedging ("I haven't run the tests yet") and conditionals ("if you push, CI will run the tests") stay quiet. Measured at 0 false positives across 14 such phrasings and 0 misses across 9 real claims; a check that cries wolf gets ignored.
  - Reported to **you, never to the model**, and worded as a mismatch rather than a verdict, since the claim may be about an earlier turn. Like the command explainer, the harness derives a fact and shows where it differs from what was claimed, instead of asking the model to mark its own work.

- **A bash approval mode is chosen at startup (`kyklos.py`, `ui.py`):** after the trust question, Kyklos asks whether to ask for every bash command (the default) or to run the unflagged ones unprompted and stop only for the rest.
  - **What still asks in the relaxed mode:** red findings, secret files, and anything `opaque` — a program Kyklos has no rules for, or a script whose contents it can't see. Those last two are the explainer admitting it doesn't know, and a mode that ran them unasked would be treating ignorance as evidence of safety. Unknown findings got their own level for this, rendered like a note but never auto-approved.
  - **The levels were given one meaning.** Red means a command changes something outside the conversation; a warning means it reads somewhere you might not expect. The rules had drifted from that: **overwrites (`>`, `>>`, `tee`), network access (`curl`, `wget`, `git push/pull/clone/fetch`), package installs, a plain non-recursive `rm`, `find -delete`/`-exec`, `pkill`/`kill` and `chmod -R`** were all promoted from warnings to fit it. The drift was the real bug — `find . -delete` ran unprompted while `rm old.txt` stopped, so the command that could remove a thousand files was the one that didn't ask. A test now pins the line across nineteen changing and three reading commands, so a future rule can't quietly disagree with it.
  - Stopping on *all* warnings was considered and rejected on evidence: in a real session **94% of bash commands were `warn`**, nearly all from "touches … outside the workspace" firing on any absolute path, so that bar would have auto-approved nothing and made the mode pointless. With the promotion, 89% of that session's commands still run unprompted and 11% stop.
  - **Side effect in the default mode:** a danger finding withholds both `p prefix` and `a always`, so a rule can no longer be saved for a promoted command — no "always allow `npm install`". That is consistent with treating them as red, but it is a change to the ask-every-time mode, not only the new one.
  - The choice is **never remembered**, unlike folder trust: it relaxes rules that protect every run, so it's agreed once per session. `write_file` and `edit_file` are untouched, no keyboard means ask, and the status bar carries `bash: unflagged runs` while it's on.
  - This promotes the explainer from an aid to reading into something approvals depend on, which the docs now say plainly, caveat and all — the startup panel itself says unflagged does not mean proven safe.

- **Subagent delegation is prompted where it matters (`kyklos.py`):** the same diagnosed session made **134 exploratory tool calls — 87% of all its tool output — and used `task` zero times out of 277 calls**. The subagent exists precisely so a long search costs the main conversation one result instead of every file read, so this was the feature working as designed and never being reached for.
  - The static instruction said "for broad searches… delegate", which is too vague to act on. It now gives a threshold and the reason: every step re-sends the whole conversation, so anything taking more than two or three reads should be delegated.
  - A one-off note is appended to a tool result once the conversation passes half the compaction trigger, naming the current size. The model can see the files it read but not the cost of carrying them on every later step; this supplies the missing signal at the moment it applies. It fires once per turn, never when `task` isn't available, and stays silent on short turns — following the same convention as the existing redaction note.

- **Compaction has an absolute ceiling (`kyklos.py`, `ui.py`):** `COMPACT_ABOVE` (100k by default, settable in `.env`) caps how large a conversation may grow before Kyklos prunes it, whatever the context window. `COMPACT_AT` alone is a *fraction of the window*, so a 1M-token model wouldn't compact until 750k — and a real session diagnosed from its saved JSONL compacted **zero times across 30.8M billed tokens**, because its conversation peaked at 178k, only 17% of the window.
  - Every step re-sends the whole conversation, so cost is set by how much context each step carries, not by how much window is spare: that session's 234 calls averaged 118k tokens each, and prompt re-sending alone accounts for 90% of the bill. The perverse consequence was that declaring a *larger* window made the session more expensive, because it pushed the pruning threshold further away.
  - `compaction_trigger()` is `min(window × COMPACT_AT, COMPACT_ABOVE)`, and `compaction_budget()` is derived from the trigger rather than the window — otherwise compaction fires and then keeps everything, since 20% of a 1M window is larger than the conversation that set it off.
  - Replaying that real session: capping at 100k would have cut prompt tokens **39%** (27.7M → 16.8M) with three compactions; 60k saves 51%, 150k saves 17%. Windows of 128k and below behave exactly as before.
  - `/tokens` reports the real trigger and says when it's `COMPACT_ABOVE` rather than the window doing the limiting.

---

## v0.3.0 — Readable Permissions (2026-10-25)

You shouldn't have to parse a shell command in your head to answer `y` or `n`. Kyklos now says what a command would do before you decide, and when a turn runs out of steps it says what it managed and how to carry on.

- **The step cap is configurable and explains itself (`kyklos.py`):** `MAX_STEPS` can be set in `.env` (default 50, validated like `CONTEXT_WINDOW`), carried on the `Agent` and inherited by subagents. Reaching it used to print `(Stopped after 50 steps without a final answer.)` and nothing else, which reads like a failure and gives you nowhere to go. It now reports what the turn managed and how to carry on:
  ```
  (Stopped after 50 steps: 55 tool calls, 2 files changed (a.ts, b.ts). The conversation is
   kept — send "continue" to carry on, /undo to revert this turn, or raise MAX_STEPS in .env.)
  ```
  The cap is a budget, not an error: the conversation is untouched, so `continue` resumes mid-task. Worth knowing that a capped turn ends with no final answer and so adds no `turn_log` entry, though its messages remain.

- **Permission prompts explain the command (`kyklos.py`, `ui.py`):** `explain_command` turns a shell command into plain-language findings at three levels — `danger`, `warn`, `note` — shown in the panel under the `$ command`, worst first, so you aren't choosing `y` or `n` from the raw text alone. It recognises piping a download into a shell, recursive deletes, `sudo`, force-pushes, `reset --hard`, raw-device writes, world-writable `chmod`, redirects, network calls, package installs, `find -delete`, process kills, paths outside the workspace, interpreters running inline code, and programs it doesn't know (which it says plainly).
  - **The harness derives the explanation; the model is never asked.** The permission prompt exists because the model may propose something you didn't intend, so letting it narrate its own command would put the thing being checked in charge of the check — and a model steered by injected text would describe exactly the command you should refuse as routine. It also costs no API call at the moment you're waiting.
  - **Findings gate the options, not just the text.** Any `danger` finding now withholds `p` and `a` and turns the border coral, extending to all destructive commands the rule that previously applied only to secret files. `apply_permission_answer` takes a `danger` flag; `permission_options` returns the findings so both front ends show and enforce the same thing.
  - **Quote-aware parsing**, from a real command that broke all of it: `_split_top_level` splits on `&&`, `||`, `;` and `|` only outside quotes, so `grep -n "hasKey\|serverKeys\|verify" f.ts` is one command with one argument, not a four-stage pipeline running a program called `serverKeys`. `_unquoted` blanks quoted spans before the redirect scan, so `echo "a > b"` writes nothing.
  - **A wrapper can no longer silence the command it runs.** `env X=1 rm -rf build` produced *no findings at all*, and `sudo rm -rf /srv` reported the escalation but never the delete — silence being far worse than noise, since an empty panel reads as "nothing to see". `_peel` now looks past leading `VAR=value` assignments and wrappers (`env`, `sudo`, `time`, `timeout`, `nohup`, `nice`, `xargs`, `exec`, …), and each layer is examined, so `sudo rm -rf /srv` reports both.
  - Running a script says so — `./scripts/deploy.sh --prod` and `node server.js` note that the behaviour lives in a file the panel isn't showing, instead of the misleading "runs deploy, which Kyklos doesn't recognise". `ssh` says the command runs on another machine where none of Kyklos's limits apply; `tee` is reported as a write; shell keywords (`done`, `esac`, `elif`, …) are no longer listed as programs.
  - **Heredoc bodies are data, not commands.** `cat > server.mjs <<'EOF' … EOF` writes a document; reading its contents as shell turned one file of JavaScript into thirteen findings, eight of them invented (`runs import`, `overwrites {`, a path from a URL string), burying the five real ones. `_strip_heredocs` removes the body and notes that it was written, leaving the findings that matter: what it overwrites and where it reaches.
  - **`=>` is no longer read as a redirect.** Inline JavaScript is full of arrows, and `(c) => (body += c)` was reporting "overwrites (body". `->` and `>=` are excluded too; `2>>log.txt` still appends.
  - Redirects understand `2>` and `&>`, and discarding output to `/dev/null` is no longer reported as overwriting a file. Shell builtins and everyday tools (`test`, `printf`, `xargs`, `jq`, `rg`, …) are recognised rather than reported as unknown programs. A chain longer than four names is summarised.
  - Piping *any* command into a shell is flagged, not just `curl`/`wget`: `base64 -d payload | sh` earns a warning for the same reason.
  - Documented in `DESIGN.md` and `README.md` as an aid to reading and **not a security boundary**: a regex over a string is not a shell, so an empty finding list means nothing was recognised, never that a command is safe — and the panel never claims otherwise.

---

## v0.2.0 — Harness Observability (2026-10-25)

Kyklos stops being opaque about itself: what it's holding in the window, what it can run, what it has changed on disk, and what it has spent — all visible without leaving the REPL.

- **Harness observability commands (`kyklos.py`, `ui.py`):** `/tokens` breaks the context window down by system prompt, messages and tool results, next to the session's billed tokens and the compaction threshold; `/tools` lists the tools the model can call right now and the rules auto-approving them, including saved `bash` prefixes; `/diff` shows what Kyklos has changed in the workspace as glass diff blocks.
  - The breakdown is labelled as an estimate throughout, with its own subtotal beside the reported total, because providers report one number for the whole prompt and never a split. Showing the two figures side by side beats quietly presenting a guess as exact.
  - `/diff` reads the undo stack's own file snapshots rather than shelling out to git, so it shows what the agent changed rather than what you edited by hand, works in a folder that isn't a repository, needs no access to `.git/`, and follows `/undo` for free.
  - Commands compute data (`context_breakdown`, `tool_overview`, `workspace_changes`) and pass it to the front end via `CommandContext.show(view, data, fallback)`. Kyklos Aero renders the panels; the plain REPL prints the fallback. `/help` and tab completion pick the commands up from `COMMANDS` with no extra wiring.

- **The status bar stays put (`ui.py`):** it used to vanish the moment you pressed Enter and return only after the answer, because it lives in prompt_toolkit's `bottom_toolbar`, which only exists while `PromptSession.prompt()` is running. It is now also pinned to the foot of the live zone for the duration of a turn. `AeroUI.status_segments` is the one list of segments both renderings draw from, so the toolbar and the live zone can't drift apart. A `_turn_active` flag keeps the live zone up across the gaps between a tool finishing and the next model call, so the line doesn't blink.
- **Session token use in the status bar (`kyklos.py`, `ui.py`):** `Agent.session_tokens` sums prompt + completion tokens over every API call and shows beside the context meter as `48k used`. The tally is taken in `_complete`, the one place every call passes through, so compaction summaries count too; a subagent's tokens roll into its parent when it finishes. It survives `/reset` — that clears the conversation, not what you've already spent — and round-trips through a saved session. Providers that report no `usage` are skipped rather than estimated, so the figure is either real or absent.
- **Tests (`test_kyklos.py`, `test_ui.py`):** the tally accumulates, survives `/reset`, stays at zero without provider usage, and round-trips; the status line is absent from the live zone when idle and present during a turn, and both renderings share one source.

---

## v0.1.0 — First Versioned Release (2026-10-25)

Kyklos starts carrying a version number.

- **`__version__` (`kyklos.py`):** the single source of truth, read by the UI rather than duplicated.
- **Version in the banner (`ui.py`, `design/aero_preview.py`):** `gradient_rule` takes an optional `label`, set into the shelf's right end the way a turn footer sits in an answer box's border. The label rests in a gap, so the rule keeps its full width and gradient:
  ```
    ───────────────────────────── v0.1.0 ─
  ```
  It rides with the large wordmark, so it's hidden along with it on terminals under 48 columns.
- **Tests (`test_ui.py`):** the shelf keeps its width when labelled, and the banner shows the current `__version__`.

---

## Startup UI: Large Rich Text Banner in Frutiger Aero Gradient (2026-10-25)

Inspired by startup displays in Antigravity CLI, Claude Code, and Hermes, the Kyklos Aero startup screen now features a stylized 2D large rich-text wordmark with full Frutiger Aero gradient styling:

- **Large stylized wordmark (`ui.py`, `design/aero_preview.py`):** `AeroUI.large_wordmark` renders `KYKLOS` in slant Figlet typography across a 2D specular Frutiger Aero gradient (`#FFFFFF` → `#E8FBFF` → `aqua` → `sky` → `deep`), capturing the glossy water and sky optimism of Aero with a top-left glint highlight that flows diagonally into deep ocean blue.
- **Gradient rule shelf (`ui.py`):** an `aqua` → `sky` → `deep` rule underlines the wordmark before the metadata line.
- **Narrow terminal auto-collapse:** if terminal width is less than 48 columns (e.g., small split panes or compact terminals), the banner automatically collapses to the compact inline wordmark line to prevent awkward wrapping.
- **Fallbacks:** full support for `plain` ANSI themes (`bold sky` styling) and ASCII mode (`-` rules and ASCII-safe slant characters).
- **Tests & Docs (`test_ui.py`, `DESIGN.md`):** added unit tests for large wordmark rendering, narrow-width collapse, and plain/ASCII fallbacks (39 tests passing in `test_ui.py`). Updated design documentation and reference mockup `design/aero_preview.py`.

---

## `f23f9c5` — Repository Folder Renamed to kyklos-harness (2026-10-24)

The repository folder moved from `Harness Playground` to `kyklos-harness`. The rename itself happened outside git (the user PATH entry, the venv's `activate` scripts and `pyvenv.cfg`, and `~/.kyklos/projects/` were updated by hand); this commit updates the docs that named the old folder.

- **Docs (`README.md`, `.env.example`):** the PATH setup command and the example `WORKSPACE` path use the new folder. The README's size line is current again: ~2,950 lines of Python plus ~1,870 of tests (it said ~2,100 and ~1,060).
- **Mockups (`DESIGN.md`, `design/aero_preview.py`):** the folder shown in the welcome line and status bar is now `kyklos-harness`.

---

## `b9c9a71` — Security Hardening from the Second Audit (2026-10-24)

A second Gemini 3.8 Flash audit (`AUDIT_REPORT.md`, SEC series) proposed seven hardening items. SEC-01, 04 and 06 were reproduced before fixing. SEC-02 (OS sandboxing) and SEC-03 (network egress control) are real gaps but a larger design decision, and SEC-07 (audit log) is a nice-to-have: all three are deferred.

- **SEC-01, commands could read Kyklos's API key (`tools.py`, `kyklos.py`):** `$env:API_KEY` in a `bash` command printed the real key, because commands inherited Kyklos's whole environment. `child_env()` now removes every name defined in Kyklos's `.env` (`load_dotenv` returns them; `hide_env_names` records them), plus any variable named like a secret (`SECRET_ENV_NAME_RE`: `*_KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`). `KYKLOS_PASS_ENV` lets chosen ones through. The report's 12-variable whitelist was not used: it would have dropped `APPDATA` (npm), `PROGRAMFILES`, `JAVA_HOME`, `VIRTUAL_ENV` and so on.
- **SEC-04, secrets in tool output (`tools.py`, `kyklos.py`):** `scrub_secrets()` replaces secret values with `[REDACTED]` before results reach the model or the UI, and the model is told how many were redacted. It covers the exact values of secret-named variables (so Kyklos's own key is caught whatever its format), plus OpenAI/Anthropic/OpenRouter, GitHub, AWS, Slack and Google key formats and private-key blocks. The report's patterns weren't used: its `sk-` regex was broken (`{20,T}` matches literal text, so it never caught a real key), and none of them would have caught an Ollama key.
- **SEC-05, planted instructions (`kyklos.py`):** the system prompt now says file contents and tool output are data, not instructions, and that planted directions should be reported to the user. The report's `<tool_output>` tags were skipped: results already arrive in a separate `tool` role, and a file could simply contain the closing tag.
- **SEC-06, orphaned processes (`tools.py`):** `bash` now uses `Popen`, with its own process group on Linux/macOS, and `kill_process_tree()` (`taskkill /T /F` on Windows, `killpg` elsewhere) stops the command and everything it started, on timeout and on Ctrl+C. Reproduced before: two child processes kept running after a timeout; now none do.
- **Docs (`README.md`):** the Safety section covers environment filtering, `KYKLOS_PASS_ENV`, redaction, the data-not-instructions rule and process-tree cleanup.
- **Tests:** `SecurityHardeningTests` (6 tests: environment filtering and opt-in, a real command unable to read `API_KEY`, redaction of values and token formats with no false positives on ordinary text, redaction before both the model and the UI, the prompt rule, and a real process-tree kill). The AGENTS.md test now checks for the section, not the word (127 total offline tests).

---

## `1d6a176` — Fixes from an External Audit (2026-10-24)

An audit by Gemini 3.8 Flash (`AUDIT_REPORT.md`) listed eight findings. Each was checked against the code and reproduced first: ACT-01 to ACT-07 are fixed here. ACT-08 was declined: it assumed each project has its own `.env`, but Kyklos only reads the `.env` in its install folder, and remembering the picked model globally was a deliberate choice.

- **ACT-01, prefix rules that approved any script (`kyklos.py`):** an option's value was taken for the script, so `python -W error …` suggested the rule `python -W error` (and `powershell -ExecutionPolicy Bypass`, `node --require ./hook.js` likewise). That rule auto-approved `python -W error -c "…"` and `python -W error evil.py`. This was worse than reported, and the report's fix (rejecting only commands with `-c`) would still have allowed `evil.py`. Now:
  - `_interpreter_prefix` only suggests a rule when the script looks like one (`_looks_like_script`: an extension or a folder in the path), or for Python's `-m module`, and never when inline-code options appear anywhere (`_has_inline_code`).
  - `command_matches_prefixes` never auto-approves an interpreter command containing inline-code options, whatever rule it matches.
- **ACT-02, key-input tests hanging in sandboxes (`test_ui.py`):** the picker tests that send real keystrokes start an asyncio event loop, which on Windows needs a socketpair over 127.0.0.1 and can hang where loopback is blocked. `event_loop_can_start()` probes this once, in a thread with a 3 s timeout, and skips those tests if it fails; `KYKLOS_SKIP_KEY_TESTS=1` skips them outright.
- **ACT-03, writing to a folder (`tools.py`):** `write_file` / `edit_file` on a folder now fail before any permission prompt, with "it's a folder, not a file". Previously the prompt came first, then a `PermissionError`.
- **ACT-04, `max` effort on other providers (`kyklos.py`):** non-Ollama catalog entries now offer OpenAI's `low · medium · high` (`STANDARD_EFFORTS`); `max` is Ollama-only and would be rejected with a 400.
- **ACT-05, empty picker (`ui.py`):** `picker_lines` no longer raises `IndexError` with no rows, and `pick` returns at once without opening a box.
- **ACT-06, `/save` help (`kyklos.py`):** now names the real default, `~/.kyklos/projects/<this folder>/last.jsonl`.
- **ACT-07, result order after Ctrl+C (`kyklos.py`):** `repair_after_interrupt` adds placeholders after the turn's real results, so results stay in call order.
- **Docs:** the README's Safety section describes the script-shape rule and inline-code refusal.
- **Tests:** regression tests for ACT-01, 03, 04, 05, 06 and 07 (121 total offline tests; the report's counts of 67 + 30 didn't match the code).

---

## `edc83ba` — Model Picker & Per-Model Thinking Effort (2026-10-24)

- **Model catalog (`kyklos.py`):**
  - `fetch_model_catalog` lists Ollama cloud's models (`/api/tags`) and describes each one in parallel (`/api/show`): context length, tool support and thinking levels (`parse_model_info`). Models without tool support (e.g. `clef:27b`) are left out. Other providers list `/v1/models` ids.
  - `load_model_catalog` caches the list for a day in `~/.kyklos/models.json` (per `BASE_URL`); `/model --refresh` reloads it. Startup now reads the model's context window from this cache when it can (`cached_model_info`), skipping the `/api/show` request.
  - Live: 22 tool-capable models fetched in 4.6 s; cached loads are instant.
- **Thinking effort (`kyklos.py`):**
  - Effort labels per model (`effort_label`, `effort_options`): a model's own levels from `thinking.values` (`False` → `off`, `True` → `on`), the generic `low · medium · high · max` for models that think without listing levels, and none for models that can't think. `default` is always first.
  - `effort_request_value` maps labels to `reasoning_effort`: `default` sends nothing, `off` → `"none"`, `on` → `"high"`, others as is. `Agent` takes `effort=` and sends it on every request (summaries too); subagents inherit it. Verified live: kimi-k3 `low` and `off`, minimax-m3.1-flash `high` (generic) and gemma4 `on` all succeed.
  - Added `Agent.set_model` to switch model, effort and context window mid-conversation.
- **Commands (`kyklos.py`):** `/model [name]` (a picker, then an effort picker; by name it uses that model's saved effort and suggests close names for typos) and `/effort [level]` (validated against the model's own levels). `CommandContext` gained `pick`, `home`, `load_models` and `context_override`. The plain REPL gets a numbered picker (`pick_plain`).
- **Settings (`kyklos.py`):** `~/.kyklos/settings.json` stores the last picked model and each model's effort (`load_settings` / `save_settings`). Model priority: `--model` (one run, not saved) > the last picked model > `MODEL` in `.env` (`choose_model`). `MODEL` is only required when nothing else picks a model.
- **Aero picker (`ui.py`):** `AeroUI.pick`, an inline prompt_toolkit picker in a rounded sky box: ↑/↓ (and Ctrl+P / Ctrl+N), type to filter, Enter, Esc; `❯` cursor, `●` current choice, 12 visible rows with scrolling, key hints in the bottom border, erased afterwards with a one-line record. `PickerState` keeps the logic testable. The banner shows `kimi-k3 (high effort)` and the status bar `kimi-k3 high`.
- **Docs (`README.md`, `DESIGN.md`):** a models-and-effort section, the new commands and flag, the new `~/.kyklos` files, and a Picker component in the design system.
- **Tests:** `ModelCatalogTests` (6), `ModelCommandTests` (8) and `PickerTests` (6, including real prompt_toolkit key input through a pipe) (118 total offline tests).

---

## `0309fd2` — Narrow Interpreter Prefixes & a Protected `.git/` (2026-10-24)

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
