#!/usr/bin/env python3
"""Rates for Kyklos's own checkers, measured on input nobody picked case by case.

Both checkers here err towards silence on purpose. The explainer (`explain_command`) and the
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

from kyklos import explain_command, unsupported_claims


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
    kind: str   # "danger", "warn", "opaque" or "claim"
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


# ---- real turns, replayed out of saved sessions ---------------------------------------------------
# `Agent.save_session` writes the whole conversation to JSONL, tool calls and results included, so a
# saved session is already the corpus: every finished turn can be handed back to the checker exactly
# as the live loop handed it over. This is the half that needs no grammar and no guessing.

def replay_session(path: Path) -> list[tuple[str, list[tuple[str, str]]]]:
    """(final answer, commands run) for each finished turn in a saved session.

    A turn ends at an assistant message with no tool calls, which is the same test the live loop
    uses. Commands are paired with their results by tool-call id, so a denied or errored call keeps
    the result text that `unsupported_claims` reads for an exit code.
    """
    turns: list[tuple[str, list[tuple[str, str]]]] = []
    commands: list[tuple[str, str]] = []
    pending: dict[str, str] = {}
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
        if role == "user":
            commands, pending = [], {}
        elif role == "assistant":
            calls = obj.get("tool_calls") or []
            for tc in calls:
                fn = tc.get("function") or {}
                if fn.get("name") != "bash":
                    continue
                try:
                    command = str(json.loads(fn.get("arguments") or "{}").get("command", ""))
                except ValueError:
                    command = ""
                if command:
                    pending[tc.get("id", "")] = command
            if not calls:
                turns.append((obj.get("content") or "", list(commands)))
                commands, pending = [], {}
        elif role == "tool":
            command = pending.pop(obj.get("tool_call_id", ""), "")
            if command:
                commands.append((command, obj.get("content") or ""))
    return turns


def session_corpora(paths: list[Path]) -> tuple[list[Case], list[tuple[Case, list[tuple[str, str]]]]]:
    """Commands and answers harvested from saved sessions.

    The commands come back as cases to explain. The answers come back paired with the commands their
    own turn ran, so the claim check sees what the live loop saw — the only way a replayed flag means
    the same thing as a live one.
    """
    commands: list[Case] = []
    answers: list[tuple[Case, list[tuple[str, str]]]] = []
    for path in paths:
        name = path.name
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
            text = f.case.text if len(f.case.text) <= 68 else f.case.text[:65] + "..."
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
    print("\nNeither number is a verdict. A group you consider correctly flagged is a group to move"
          "\nout of the corpus; one you consider wrong is a gap in the checker.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
