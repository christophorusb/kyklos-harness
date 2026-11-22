#!/usr/bin/env python3
"""Rates for Kyklos's own checkers, measured on input nobody picked case by case.

The checkers here err towards silence on purpose. The explainer (`explain_command`) and the
answer's-story check (`unsupported_claims`) are only worth anything while you still believe them,
and a check that cries wolf gets ignored. The test suite pins that with controls: a few benign
commands that must stay quiet, a few hedged sentences that must not read as claims.

Controls written beside the regexes can only probe the cases their author already thought of, and
they answer a yes/no question — "still quiet on these eleven?" This answers a different one: on a
few hundred inputs built from a grammar of everyday shell and everyday hedging, *what fraction*
fires? A gate tells you nothing regressed. A rate tells you where the edge actually is.

Two kinds of firing are worth keeping apart, because only one is a miscalibration:

  * `danger` / `warn` on something benign — the explainer thinking an ordinary command is risky.
  * `opaque` on something benign — the explainer saying it doesn't recognise the program. That is
    honest by design (`must_be_asked` prompts rather than trust ignorance), but it still costs a
    prompt in auto-bash mode, so the rate is a coverage list worth reading.

Self-checking (`self_check_reasons`) is measured the same way, from both sides: how often it would
hold back an answer from a replayed session, and how well its two patterns read their input --
check commands it must recognise, and answers that already say why nothing was checked, which it
must recognise without mistaking a plain answer for one.

Nothing here is a pass/fail gate, and nothing here is a verdict: it prints rates and names every
case that fired, in groups, so you can decide which groups you consider wrong. Run it:

    python audit.py                        # the generated corpora
    python audit.py .kyklos/*.jsonl        # those, plus real turns replayed out of saved sessions
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

from types import SimpleNamespace

from kyklos import (CHECK_DECLINED_RE, CHECKED, FAILING, UNCHECKED, check_kind, check_passed,
                    check_state_by_order, explain_command, is_code_path, self_check_reasons,
                    unsupported_claims)

# check_kind as a pattern, so audit_pattern can hold it to the corpora like the regexes.
CHECK_COMMANDS = SimpleNamespace(search=check_kind)


# ---- cases and results --------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    """One input, and the group it belongs to. Groups are how the report stays readable: a rate over
    everything hides that one family of phrasing accounts for all of it."""
    group: str
    text: str


@dataclass(frozen=True)
class Fired:
    case: Case
    kind: str   # "danger", "warn", "opaque", "claim", "missed", "read in", or self-check reasons
    why: str


@dataclass
class Rate:
    what: str
    cases: list[Case] = field(default_factory=list)
    fired: list[Fired] = field(default_factory=list)

    def of_kind(self, *kinds: str) -> list[Fired]:
        return [f for f in self.fired if f.kind in kinds]

    def groups(self) -> dict[str, tuple[int, int]]:
        """group -> (fired, total), so a rate can be read per family of phrasing."""
        totals: dict[str, int] = {}
        hits: dict[str, int] = {}
        for case in self.cases:
            totals[case.group] = totals.get(case.group, 0) + 1
        for f in self.fired:
            hits[f.case.group] = hits.get(f.case.group, 0) + 1
        return {g: (hits.get(g, 0), n) for g, n in totals.items()}


def _pct(n: int, total: int) -> str:
    return "—" if not total else f"{100 * n / total:.1f}%"


# ---- a grammar of commands you would approve without reading -------------------------------------
# Every command below reads, lists, searches or tests. None writes, deletes, installs, or reaches the
# network, so none of them should need a prompt. They are products of parts rather than a hand-made
# list, which is the point: the combinations are ones nobody sat down and enumerated.

READERS = ("cat", "head -n 20", "tail -n 50", "wc -l", "stat", "nl", "bat --plain", "file", "less")
FILES = ("kyklos.py", "README.md", "src/main.ts", "tests/test_api.py", "pkg/util/strings.go",
         ".gitignore")
SEARCHES = ("grep -rn TODO", "grep -rni fixme", "grep -c def", "rg --no-heading parse", "rg -l import",
            "fd -e py")
DIRS = (".", "src/", "lib/", "tests/")
GIT_READS = ("git status", "git status --short", "git log --oneline -10", "git log -1 --stat",
             "git diff", "git diff --staged", "git show HEAD", "git branch -a", "git remote -v",
             "git stash list", "git blame kyklos.py", "git describe --tags")
LISTINGS = ("ls", "ls -la", "ls -1 src/", "tree -L 2", "du -sh .", "pwd", "df -h", "which python",
            "find . -name '*.py'", "find src -type f")
TESTS = ("pytest -q", "pytest -x tests/", "pytest --collect-only", "npm test", "cargo test",
         "go test ./...", "make test", "python -m unittest test_kyklos")
SINKS = ("wc -l", "head -n 5", "sort | uniq -c")
# The shapes the first version of this corpus had no entry for, and so measured nothing about: a
# path whose prefix is a quoted variable, a command substitution, and a stderr redirect. All three
# turned up in one screenshot of a real session, each one carrying a wrong finding.
PREFIX_VARS = ('"$DIR"', '"$REPO"', '"$TMP"', '"$HOME"')
PREFIX_TAILS = ("/notes.md", "/src/main.ts", "/README.md")
SUBST_NAMES = ("OUT", "REV", "TMP", "V")
SUBST_INNER = ("git rev-parse HEAD", "ls -1", "date", "git branch --show-current", "pwd")
# Loop and conditional bodies. A danger inside one used to produce no findings whatsoever, so the
# corpus needs the benign shape too: the fix must not start prompting for every `for` loop.
LOOP_BODIES = ('cat "$f"', 'wc -l "$f"', 'echo "$f"', "git log -1 --oneline")
# Everyday writes, moves and git that keep your work. The destructive sibling of each of these has
# a rule now, so the corpus has to hold the benign form or a rule that fires on both reads as clean.
EVERYDAY_WRITES = ("cp README.md README.bak", "cp -r src/ build/", "mv old.ts new.ts",
                   "mv src/a.py src/b.py", "sed -n '1,20p' kyklos.py", "sed 's/a/b/' f.txt",
                   "chown me:staff file.txt", "ln -s shared lib", "touch src/new.ts",
                   "mkdir -p build/out", "git checkout main", "git checkout -b feature/x",
                   "git branch -d merged", "git restore --staged f.txt",
                   "git config user.email a@b.c", "git switch main")
LOOP_HEADS = ("for f in *.py", "for f in src/*", "while read f")


def command_corpus() -> list[Case]:
    """A few hundred commands that should all pass without a prompt."""
    cases: list[Case] = []
    cases += [Case("read a file", f"{r} {f}") for r, f in product(READERS, FILES)]
    cases += [Case("search", f"{s} {d}") for s, d in product(SEARCHES, DIRS)]
    cases += [Case("git, read-only", c) for c in GIT_READS]
    cases += [Case("list and inspect", c) for c in LISTINGS]
    cases += [Case("run tests", c) for c in TESTS]
    cases += [Case("pipeline", f"{s} {d} | {k}")
              for s, d, k in product(SEARCHES[:3], DIRS[:2], SINKS)]
    cases += [Case("chain", f"{g} && {t}") for g, t in product(GIT_READS[:4], TESTS[:3])]
    cases += [Case("quoted path prefix", f"{r} {v}{t}")
              for r, v, t in product(READERS[:4], PREFIX_VARS, PREFIX_TAILS)]
    cases += [Case("command substitution", f'{n}="$({i})"')
              for n, i in product(SUBST_NAMES, SUBST_INNER)]
    cases += [Case("redirect to /dev/null", f"{c} 2>/dev/null")
              for c in GIT_READS[:5] + TESTS[:4] + LISTINGS[:3]]
    cases += [Case("loop body", f"{h}; do {b}; done")
              for h, b in product(LOOP_HEADS, LOOP_BODIES)]
    cases += [Case("conditional body", f"if true; then {t}; fi") for t in TESTS[:4]]
    cases += [Case("everyday write", c) for c in EVERYDAY_WRITES]
    return cases


# ---- a grammar of sentences that claim nothing ---------------------------------------------------
# `unsupported_claims` looks for the turn's own story in past tense: "I ran the tests". Advice, a
# question, a refusal and a plain description are none of them. These are the sentences that must
# stay quiet; the last two groups are the ones worth watching, since they negate or describe using
# words the checker's lead-scan has no entry for.

ACTIONS = ("run the tests", "run the test suite", "run pytest -q", "run npm test", "re-run the tests",
           "commit this", "commit the change", "make a commit", "push the branch", "push to main",
           "install left-pad", "install the dependencies")
ADVICE = ("You can {}.", "You could {} next.", "I'd suggest you {}.", "The next step is to {}.",
          "{} when you're ready.", "You should {} before merging.", "Please {} and tell me what breaks.")
ASKING = ("Want me to {}?", "Shall I {}?", "Should I {} now?", "Do you want me to {}?")
REFUSING = ("I haven't {} yet.", "I did not {}.", "I was unable to {}.", "I couldn't {}.",
            "I can't {} without your approval.")
LATER = ("Next I'll {}.", "I'm about to {}.", "I'll {} once you confirm.", "If you {}, CI will too.")

# Prose about the repository that happens to use the same words. None of it says the turn did
# anything; "the installed version" is a fact about the environment, not a story about this turn.
DESCRIBING = (
    "The test suite covers this path.",
    "There's a test for that in test_kyklos.py.",
    "The failing test is in test_ui.py.",
    "The installed version is 2.1.",
    "Check the installed packages with pip list.",
    "The committed version differs from your working copy.",
    "The pushed branch is behind main.",
    "Tests are a good idea here.",
    "The test command lives in the Makefile.",
    "A commit here would be premature.",
)
# Negations the lead-scan has no word for: it knows `not`, `n't`, `never`, `without` and the modals,
# but not `no`, `nothing`, `none`, `failed to` or `skipped`.
NEGATING = (
    "Nothing was installed.",
    "No packages were installed.",
    "None of the tests ran.",
    "Nothing was committed.",
    "No branch was pushed.",
    "The install step was skipped.",
    "I skipped the tests to keep this quick.",
    "The build failed before the tests ran.",
    "Zero tests pass right now, because the suite won't import.",
)


def answer_corpus() -> list[Case]:
    """Answers that make no claim about what this turn did."""
    cases: list[Case] = []
    for group, templates in (("advice", ADVICE), ("asking", ASKING),
                             ("refusing", REFUSING), ("later", LATER)):
        cases += [Case(group, t.format(a)) for t, a in product(templates, ACTIONS)]
    cases += [Case("describing", s) for s in DESCRIBING]
    cases += [Case("negating, in other words", s) for s in NEGATING]
    return cases


# ---- self-checking: what counts as a check, and what counts as saying why there was none ----------
# Two patterns decide whether an answer waits. CHECK_COMMAND_RE must know a check when it sees one
# (a miss asks the model for a check it already ran), and must not read an ordinary command as one
# (that would let unchecked code through). CHECK_DECLINED_RE must recognise an answer that already
# says why nothing was checked (a miss asks again), and must not read a plain answer as one (that
# would skip the check).

CHECKS = {
    "tests": ("pytest -q", "pytest tests/test_api.py::test_login", "python -m unittest test_kyklos",
              "python3 -m unittest discover -s tests", "npm test", "npm run test", "pnpm test",
              "yarn test:unit", "bun test", "npx vitest run", "npx jest src", "go test ./...",
              "cargo test", "make test", "uv run pytest -x", "tox -e py312", "bundle exec rspec",
              "./gradlew test", "mvn -q test", "dotnet test", "swift test"),
    "type checks": ("npx tsc --noEmit", "tsc -p .", "mypy src", "pyright", "vue-tsc --noEmit",
                    "npm run typecheck", "cargo check", "go vet ./..."),
    "lints": ("ruff check .", "flake8 app.py", "pylint pkg", "npx eslint src", "npm run lint",
              "cargo clippy", "shellcheck run.sh", "biome check src"),
    "builds": ("npm run build", "pnpm build", "cargo build", "go build ./...", "make build",
               "python3 -m py_compile app.py", "python -m compileall src", "swift build"),
    "chained": ("cd web && npm test", "source .venv/bin/activate && pytest -q",
                "npm run build 2>&1 | tail -20", "pytest -q; echo done"),
    "launched": (".venv/bin/pytest -q", "./node_modules/.bin/tsc --noEmit", "poetry run pytest",
                 "python -m pytest -x", "py -3 -m unittest", "CI=1 npm test", "time cargo test",
                 "npx --yes eslint .", "env PYTHONPATH=src pytest"),
}
NOT_CHECKS = ("ls -la", "git status", "git diff", "cat test_api.py", "grep -rn test src",
              "npm install", "pip install -r requirements.txt", "python app.py", "node server.js",
              "go run .", "npm start", "echo build", "mkdir build", "rm -rf build",
              "sed -i 's/a/b/' src/app.py", "git commit -m 'fix the tests'", "find . -name '*_test.go'",
              "head -n 30 Makefile", "open coverage/index.html", "python -m http.server 8000",
              # A checker's name that doesn't run it: the first audit missed these.
              "pip install pytest", "grep -rn pytest .", "which tsc", "cat pyproject.toml | grep mypy",
              "npm install --save-dev eslint", "echo 'run pytest next'")

DECLINES = {
    "no runner": ("I couldn't run the tests: there's no test runner set up here.",
                  "I couldn't run any checks, since nothing is set up in this folder.",
                  "I wasn't able to run the build, because it needs network access.",
                  "I can't run the tests without the database."),
    "not yet": ("I haven't run the tests yet.", "I didn't run the type checker.",
                "The change has not been tested.", "This hasn't been verified against the real API.",
                "Not tested: it needs a browser, so I handed that part off."),
    "one word": ("This is untested.", "The fix is unverified.", "Untested, as there are no tests."),
    "other wording": ("There are no tests in this project, so I skipped that step.",
                      "No checks exist for this module.",
                      "Nothing to run here: the repo has no test suite.",
                      "I left testing to you, since it needs your credentials."),
}
PLAIN_ANSWERS = {
    "done": ("Done.", "Fixed the off-by-one in range_sum.", "Renamed the function and updated callers.",
             "Updated the parser to accept trailing commas.", "Added a check for empty input.",
             "The handler now returns 404 for missing users."),
    "describing behaviour": ("The migration does not run twice anymore.",
                             "Empty lists are not checked twice now; the loop exits early.",
                             "The build step can't be skipped by accident any more.",
                             "It doesn't compile the templates on every request now."),
    "naming tests": ("I added a test for the empty case in test_parser.py.",
                     "The new test covers the 404 path.", "Tests for the CLI live in tests/test_cli.py."),
}


def check_command_corpora() -> tuple[list[Case], list[Case]]:
    """(commands that check code, commands that don't)."""
    checks = [Case(group, c) for group, commands in CHECKS.items() for c in commands]
    return checks, [Case("not a check", c) for c in NOT_CHECKS]


def decline_corpora() -> tuple[list[Case], list[Case]]:
    """(answers that say why nothing was checked, plain answers that don't)."""
    declines = [Case(group, s) for group, sentences in DECLINES.items() for s in sentences]
    plain = [Case(group, s) for group, sentences in PLAIN_ANSWERS.items() for s in sentences]
    return declines, plain


# ---- real turns, replayed out of saved sessions ---------------------------------------------------
# `Agent.save_session` writes the whole conversation to JSONL, tool calls and results included, so a
# saved session is already the corpus: every finished turn can be handed back to the checker exactly
# as the live loop handed it over. This is the half that needs no grammar and no guessing.

def _is_image_result(content) -> bool:
    """Whether a user-role message is Kyklos passing on what read_file read (see IMAGE_RESULT_NOTE)."""
    if isinstance(content, list):
        content = next((p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"), "")
    return isinstance(content, str) and content.startswith("[Kyklos: the image")


def _is_kyklos_note(content) -> bool:
    """Whether a user-role message is Kyklos alone speaking mid-turn ("[Kyklos: …]": a nudge or a
    self-check), so the turn goes on. A note riding in front of the user's own words, such as the
    workspace-sketch note, is followed by them and starts a turn like any other message."""
    if isinstance(content, list):
        content = next((p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"), "")
    text = content.strip() if isinstance(content, str) else ""
    return text.startswith("[Kyklos: ") and text.endswith("]") and "]\n\n" not in text


@dataclass
class Attempt:
    """One reply that would end its turn, with what the turn had done by then: the inputs the live
    loop hands to `unsupported_claims` and `self_check_reasons` at that moment."""
    answer: str
    commands: list[tuple[str, str]]
    code_edits: dict[str, int]  # code path -> the call that last changed it
    last_check: tuple | None  # (call number, passed, command, None), as the loop records it
    turn: int  # which turn of the session: self-check raises each reason once per turn


def _edited_code(arguments: str, result: str) -> str | None:
    """The code path a successful write_file/edit_file changed, outside the scratch directory."""
    if not result.startswith(("Wrote ", "Edited ")):
        return None
    try:
        path = str(json.loads(arguments or "{}").get("path", "")).replace("\\", "/")
    except (ValueError, AttributeError):
        return None
    if not path or "/.kyklos/" in f"/{path}" or not is_code_path(path):
        return None
    return path


def replay_attempts(path: Path) -> list[Attempt]:
    """Every reply in a saved session that would end its turn, with the turn's commands, code edits
    and last check so far.

    A reply ends a turn when it has no tool calls, the same test the live loop uses. A turn starts at
    the user's message and runs through Kyklos's own mid-turn notes, so a reply after a nudge sees
    the whole turn, as it did live. Results are paired with calls by tool-call id, so a denied or
    errored call keeps the text that is read for an exit code. Without the workspace, edits made by
    commands can't be seen: only the file tools' edits count, as in a session without snapshots.
    """
    attempts: list[Attempt] = []
    turn, calls_run = 0, 0
    commands: list[tuple[str, str]] = []
    code_edits: dict[str, int] = {}
    last_check = None
    pending: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if obj.get("_meta") == "state":
            continue
        role = obj.get("role")
        if role == "user" and (_is_image_result(obj.get("content")) or _is_kyklos_note(obj.get("content"))):
            continue  # Kyklos speaking mid-turn, or the images a read_file call read: not the user
        if role == "user":
            turn += 1
            calls_run, commands, code_edits, last_check, pending = 0, [], {}, None, {}
        elif role == "assistant":
            calls = obj.get("tool_calls") or []
            for tc in calls:
                fn = tc.get("function") or {}
                if fn.get("name") in ("bash", "write_file", "edit_file"):
                    pending[tc.get("id", "")] = (fn["name"], fn.get("arguments") or "")
            if not calls:
                attempts.append(Attempt(obj.get("content") or "", list(commands), dict(code_edits),
                                        last_check, turn))
        elif role == "tool":
            calls_run += 1
            name, arguments = pending.pop(obj.get("tool_call_id", ""), ("", ""))
            result = obj.get("content") or ""
            if not isinstance(result, str):
                continue
            if name == "bash":
                try:
                    command = str(json.loads(arguments or "{}").get("command", ""))
                except (ValueError, AttributeError):
                    command = ""
                if not command:
                    continue
                commands.append((command, result))
                ran = result.startswith("exit code ") or "timed out" in result[:60]
                kind = check_kind(command) if ran else None
                if kind:
                    last_check = (calls_run, check_passed(kind, result), command, None)
            elif name:
                edited = _edited_code(arguments, result)
                if edited:
                    code_edits[edited] = calls_run
    return attempts


def session_label(path: Path) -> str:
    """A short name for a saved session in the report. Every project's session is `last.jsonl`, so
    the folder it sits in tells them apart."""
    folder = path.parent.name
    return f"{'…' + folder[-27:] if len(folder) > 28 else folder}/{path.name}"


def replay_session(path: Path) -> list[tuple[str, list[tuple[str, str]]]]:
    """(answer, commands run in its turn so far) for each reply that would end a turn."""
    return [(a.answer, a.commands) for a in replay_attempts(path)]


def session_corpora(paths: list[Path]) -> tuple[list[Case], list[tuple[Case, list[tuple[str, str]]]]]:
    """Commands and answers harvested from saved sessions.

    The commands come back as cases to explain. The answers come back paired with the commands their
    own turn ran, so the claim check sees what the live loop saw — the only way a replayed flag means
    the same thing as a live one.
    """
    commands: list[Case] = []
    answers: list[tuple[Case, list[tuple[str, str]]]] = []
    for path in paths:
        name = session_label(path)
        for answer, runs in replay_session(path):
            if answer.strip():
                answers.append((Case(f"replayed: {name}", answer), runs))
            commands += [Case(f"replayed: {name}", c) for c, _ in runs]
    return commands, answers


# ---- measuring -----------------------------------------------------------------------------------

def audit_commands(cases: list[Case]) -> Rate:
    """How often the explainer would stop a command that needed no stopping."""
    rate = Rate("commands that would still be asked about", list(cases))
    for case in cases:
        findings = explain_command(case.text)
        for level, text in findings:
            if level in ("danger", "warn", "opaque"):
                rate.fired.append(Fired(case, level, text))
                break  # one reason per command: the first is the one that costs the prompt
    return rate


def audit_answers(cases: list[tuple[Case, list[tuple[str, str]]]]) -> Rate:
    """How often the claim check reads a claim into an answer that made none."""
    rate = Rate("answers read as unsupported claims", [c for c, _ in cases])
    for case, runs in cases:
        for label, why in unsupported_claims(case.text, runs):
            rate.fired.append(Fired(case, "claim", f"{label}: {why}"))
            break
    return rate


def audit_pattern(what: str, cases: list[Case], pattern, should_match: bool) -> Rate:
    """How often a pattern gets these cases wrong: misses when they should match, and matches when
    they shouldn't."""
    rate = Rate(what, list(cases))
    for case in cases:
        if bool(pattern.search(case.text)) != should_match:
            rate.fired.append(Fired(case, "missed" if should_match else "read in",
                                    "not recognised" if should_match else "recognised, wrongly"))
    return rate


def audit_self_check(paths: list[Path]) -> tuple[Rate, dict[str, int]]:
    """How often self-checking would hold back a replayed answer, and the check state of every answer
    that came after a code change. Reasons are raised once per turn, as live, so an answer after a
    self-check note counts only for what is new."""
    rate = Rate("replayed answers self-checking would hold back", [])
    states = {CHECKED: 0, UNCHECKED: 0, FAILING: 0}
    for path in paths:
        raised: dict[int, set[str]] = {}
        for attempt in replay_attempts(path):
            if not attempt.answer.strip():
                continue
            case = Case(f"replayed: {session_label(path)}", attempt.answer)
            rate.cases.append(case)
            state = check_state_by_order(attempt.code_edits, attempt.last_check)
            if state:
                states[state] += 1
            claims = unsupported_claims(attempt.answer, attempt.commands)
            seen = raised.setdefault(attempt.turn, set())
            reasons = self_check_reasons(attempt.answer, claims, state, seen)
            seen.update(reasons)
            if reasons:
                detail = f"code {state}" if state else "no code changed"
                rate.fired.append(Fired(case, "+".join(reasons), detail))
    return rate, states


# ---- the report ----------------------------------------------------------------------------------

def report(rate: Rate, examples: int = 6, split_opaque: bool = False) -> None:
    total = len(rate.cases)
    print(f"\n{rate.what}")
    print("-" * len(rate.what))
    if split_opaque:
        loud = rate.of_kind("danger", "warn")
        unknown = rate.of_kind("opaque")
        print(f"  miscalibrated  {len(loud):4d} / {total:<4d}  {_pct(len(loud), total):>6}"
              f"   (benign, called danger or warn)")
        print(f"  coverage gap   {len(unknown):4d} / {total:<4d}  {_pct(len(unknown), total):>6}"
              f"   (benign, not recognised)")
    else:
        print(f"  fired          {len(rate.fired):4d} / {total:<4d}  {_pct(len(rate.fired), total):>6}")

    for group, (hit, n) in sorted(rate.groups().items(), key=lambda kv: -kv[1][0]):
        if hit:
            print(f"    {group:<28} {hit:3d} / {n:<4d} {_pct(hit, n):>6}")

    if rate.fired:
        print("\n  what fired:")
        seen: set[str] = set()
        shown = 0
        for f in rate.fired:
            key = f"{f.kind}:{f.why}"
            if key in seen:
                continue
            seen.add(key)
            shown += 1
            if shown > examples:
                print(f"    … and {len(seen) - examples} more distinct reasons")
                break
            flat = " ".join(f.case.text.split())  # an answer's own line breaks would break the report's
            text = flat if len(flat) <= 68 else flat[:65] + "..."
            print(f"    [{f.kind}] {text}")
            print(f"            → {f.why}")


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv]
    missing = [p for p in paths if not p.is_file()]
    for p in missing:
        print(f"skipping {p}: no such file", file=sys.stderr)
    paths = [p for p in paths if p.is_file()]

    commands = command_corpus()
    answers = [(c, []) for c in answer_corpus()]
    generated = (len(commands), len(answers))

    if paths:
        more_commands, more_answers = session_corpora(paths)
        commands += more_commands
        answers += more_answers

    print(f"Kyklos checker rates — {generated[0]} generated commands, {generated[1]} generated answers"
          + (f", {len(paths)} session(s) replayed" if paths else ", no sessions replayed"))

    report(audit_commands(commands), split_opaque=True)
    report(audit_answers(answers))

    checks, not_checks = check_command_corpora()
    declines, plain = decline_corpora()
    report(audit_pattern("check commands self-checking wouldn't recognise", checks, CHECK_COMMANDS, True))
    report(audit_pattern("other commands self-checking would take for a check", not_checks,
                         CHECK_COMMANDS, False))
    report(audit_pattern("answers saying why nothing was checked, not recognised (asked again)", declines,
                         CHECK_DECLINED_RE, True))
    report(audit_pattern("plain answers taken as saying why nothing was checked (check skipped)", plain,
                         CHECK_DECLINED_RE, False))
    if paths:
        rate, states = audit_self_check(paths)
        report(rate)
        after = sum(states.values())
        print(f"\n  answers after a code change: {after}"
              + (f"  ·  checked {states[CHECKED]}  ·  unchecked {states[UNCHECKED]}  ·  failing {states[FAILING]}"
                 if after else "")
              + "\n  (replayed from the file tools' edits; command edits need the workspace, so live counts can be higher)")
    print("\nNeither number is a verdict. A group you consider correctly flagged is a group to move"
          "\nout of the corpus; one you consider wrong is a gap in the checker.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
