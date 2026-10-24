"""Kyklos: a minimal coding-agent harness for any OpenAI-compatible API.

The whole idea in one loop:
    send messages -> model replies -> run the tools it asked for
    -> append the results -> repeat until it answers without tools.

Run:  kyklos            (interactive REPL, via kyklos.cmd)
      kyklos "do X"     (one task, then exit)
"""

import json
import os
import platform
import sys
import textwrap
from pathlib import Path

from openai import OpenAI

from tools import SAFE_TOOLS, SCHEMAS, SHELL_NAME, ToolError, Tools, probe_commands

MAX_STEPS = 50  # hard stop so a confused model can't loop forever
MAX_INSTRUCTIONS_CHARS = 20_000

# Context management: when the conversation passes COMPACT_AT of the window, compact the part
# older than the most recent KEEP_RECENT of the window, in two stages:
#   1. clear old tool results (they're bulky and can be re-fetched); keep all user and assistant text;
#   2. only if still above PRUNE_TARGET, replace the old part with a model-written summary, plus a
#      verbatim log of earlier requests and answers (summaries can't be trusted to keep those).
DEFAULT_CONTEXT_WINDOW = 128_000
COMPACT_AT = 0.75
KEEP_RECENT = 0.2
PRUNE_TARGET = 0.6
PRUNE_MIN_CHARS = 500  # tool results shorter than this aren't worth clearing
PRUNED_MARK = "[Old tool result cleared to save context]"
MAX_COMPACTIONS_PER_TURN = 3  # more than this means the window is too small for the task

SUMMARY_PROMPT = """You are compacting a coding agent's conversation so it can continue working with less context.
Write a concise summary that keeps everything needed to continue:
- The user's goals and requests, including exact wording of any constraints
- Answers and conclusions already given to the user, with their specifics (the user may ask about them again)
- Decisions made and why
- Files created or changed, with exact paths and their current state
- Commands run and their key results or errors
- What is done and what remains to do
Keep exact file paths, names, numbers and error messages. Do not add anything that isn't in the conversation.
Long tool outputs are clipped in this transcript; record what was learned from them rather than noting the clipping."""

SYSTEM_PROMPT = """You are Kyklos, a coding agent working in the user's workspace: {workspace}
OS: {os}. Shell for the bash tool: {shell}.
{commands}

How to work:
- Use grep and glob to find code; they're faster than listing folders with bash.
- Read files before editing them. Use edit_file for small changes, write_file for new files.
- After changing code, run it or its tests with the bash tool to verify.
- If a tool call is denied, do not retry it unchanged; ask the user or take another approach.
- When done, reply with a short summary of what you changed and what you verified."""

# Subagents: the `task` tool runs a fresh Agent with a clean context and returns only its final
# answer, so long searches and investigations don't fill the main conversation.
# Subagents can't start subagents of their own.
SUBAGENT_TOOLS = {
    "explore": {"read_file", "grep", "glob"},  # read-only: never asks for permission
    "general": {"read_file", "grep", "glob", "write_file", "edit_file", "bash"},  # asks like the parent
}

TASK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "task",
        "description": (
            "Delegate a self-contained job to a subagent that starts with a fresh, empty context. "
            "It can't see this conversation, so the prompt must contain everything it needs. "
            "Only its final answer comes back to you, which keeps your own context small. "
            "Good for broad codebase searches, investigating a question across many files, or a "
            "well-defined change. mode 'explore' is read-only (read_file, grep, glob); "
            "mode 'general' can also edit files and run commands, with the user's approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "3-6 word label shown to the user"},
                "prompt": {"type": "string", "description": "Complete, self-contained instructions, "
                                                            "including what the answer should contain"},
                "mode": {"type": "string", "enum": ["explore", "general"], "description": "Default explore"},
            },
            "required": ["description", "prompt"],
        },
    },
}

SUBAGENT_PROMPT = """

You are a subagent. Another agent delegated this task to you; your final answer goes back to it, not to the user.
You can't see that agent's conversation, only the task below. Work on your own, without asking questions.
Make your final answer self-contained: specific findings, exact file paths and line numbers, and anything you could not determine."""


def load_dotenv(path: Path) -> None:
    """Tiny .env reader so we don't need python-dotenv. Real env vars win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def describe_commands() -> str:
    """Environment facts for the system prompt, so the model doesn't have to discover them."""
    working, broken_python = probe_commands()
    lines = []
    if working:
        lines.append("Commands that work here: " + ", ".join(f"{c} ({v})" for c, v in working.items()) + ".")
    if broken_python:
        lines.append(f"Not available here, don't use: {', '.join(broken_python)}.")
    return "\n".join(lines)


def estimate_tokens(messages: list) -> int:
    """Rough token count (~4 chars per token) for when the API doesn't report usage."""
    return len(json.dumps(messages, ensure_ascii=False)) // 4


def render_transcript(messages: list, max_chars: int) -> str:
    """Plain-text view of messages for the summarizer, with long tool output clipped."""
    parts = []
    for m in messages:
        content = m.get("content") or ""
        if m["role"] == "tool":
            parts.append(f"TOOL RESULT: {content[:4000]}" + (" ...(clipped)" if len(content) > 4000 else ""))
            continue
        if content:
            parts.append(f"{m['role'].upper()}: {content[:4000]}")
        for tc in m.get("tool_calls", []):
            parts.append(f"ASSISTANT CALLED {tc['function']['name']}({tc['function']['arguments'][:1000]})")
    text = "\n\n".join(parts)
    if len(text) > max_chars:  # keep the start (the original task) and the most recent part
        head = max_chars // 5
        text = text[:head] + "\n\n... (middle omitted) ...\n\n" + text[-(max_chars - head):]
    return text


def render_turn_log(turns: list, max_chars: int) -> str:
    """Earlier requests and answers, newest kept first when space runs out, shown oldest first."""
    entries, used = [], 0
    for request, answer in reversed(turns):
        answer = answer if len(answer) <= 1500 else answer[:1500] + " ...(clipped)"
        entry = f"- User: {request[:500]}\n  You answered: {answer}"
        if used + len(entry) > max_chars:
            break
        entries.append(entry)
        used += len(entry)
    return "\n".join(reversed(entries))


def project_instructions(workspace: Path) -> str:
    """The workspace's AGENTS.md, if any, appended to the system prompt (like CLAUDE.md / GEMINI.md)."""
    path = workspace / "AGENTS.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > MAX_INSTRUCTIONS_CHARS:
        text = text[:MAX_INSTRUCTIONS_CHARS] + "\n... (AGENTS.md truncated)"
    return f"\n\nProject instructions from AGENTS.md (follow these):\n{text}"


def preview(text: str, max_lines: int = 6, max_chars: int = 500) -> str:
    """Short dimmed view of a tool result, so the user sees what the model sees."""
    lines = text[:max_chars].splitlines()
    more = len(text.splitlines()) - len(lines[:max_lines])
    shown = [f"  │ {line}" for line in lines[:max_lines]]
    if more > 0 or len(text) > max_chars:
        shown.append(f"  │ ... ({more} more lines)" if more > 0 else "  │ ...")
    return "\033[2m" + "\n".join(shown or ["  │ (no output)"]) + "\033[0m"


def ask_permission(name: str, args: dict, always_allowed: set) -> tuple[bool, str]:
    """Ask the user before a risky tool runs. Returns (allowed, note_for_model)."""
    print(f"\n\033[33m? {name}\033[0m")
    for key, value in args.items():
        shown = str(value)
        if len(shown) > 600:
            shown = shown[:600] + f"... ({len(shown)} chars)"
        print(f"  {key}: {shown}")
    try:
        answer = input("  Allow? [y]es / [n]o / [a]lways for this tool: ").strip().lower()
    except EOFError:  # no keyboard, e.g. piped or scripted run
        print("  (no input available: denied)")
        return False, f"The {name} call was denied: nobody is available to approve it. Use read-only tools instead."
    if answer in ("a", "always"):
        always_allowed.add(name)
        return True, ""
    if answer in ("y", "yes"):
        return True, ""
    reason = input("  Reason to tell the model (optional): ").strip()
    return False, f"The user denied this {name} call." + (f" Reason: {reason}" if reason else "")


class Agent:
    def __init__(self, client, model: str, workspace: Path, permission=ask_permission,
                 context_window: int = DEFAULT_CONTEXT_WINDOW, tool_names: set = None,
                 depth: int = 0, always_allowed: set = None):
        self.client = client
        self.model = model
        self.tools = Tools(workspace)
        self.permission = permission
        self.context_window = context_window
        self.depth = depth  # 0 for the main agent, 1 for a subagent
        self.indent = "    " * depth  # subagent output is indented under the task call
        all_schemas = SCHEMAS + [TASK_SCHEMA]
        if tool_names is None:
            tool_names = {s["function"]["name"] for s in all_schemas}
        self.schemas = [s for s in all_schemas if s["function"]["name"] in tool_names]
        self.tool_names = {s["function"]["name"] for s in self.schemas}
        # Shared with subagents, so "always allow" carries over in both directions.
        self.always_allowed: set[str] = always_allowed if always_allowed is not None else set()
        self.last_steps = 0
        self.reset()

    def reset(self) -> None:
        prompt = SYSTEM_PROMPT.format(
            workspace=self.tools.workspace, os=platform.system(), shell=SHELL_NAME,
            commands=describe_commands(),
        )
        if "task" in self.tool_names:
            prompt += ("\n- For broad searches or investigations that would take many steps, "
                       "delegate them to a subagent with the task tool.")
        if self.depth > 0:
            prompt += SUBAGENT_PROMPT + f"\nYour tools: {', '.join(sorted(self.tool_names))}."
        prompt += project_instructions(self.tools.workspace)
        self.messages = [{"role": "system", "content": prompt}]
        self.usage_mark = None  # (prompt_tokens the API reported, message count at that call)
        self.current_request = None  # the user message this turn is answering; never summarized away
        self.turn_log: list[tuple[str, str]] = []  # (request, final answer) of finished turns

    def context_tokens(self) -> int:
        """Current conversation size: the API's last count plus an estimate for what was added since."""
        if self.usage_mark:
            tokens, count = self.usage_mark
            return tokens + estimate_tokens(self.messages[count:])
        return estimate_tokens(self.messages)

    def compact(self) -> bool:
        """Shrink the older part of the conversation; keep recent turns verbatim.
        Returns False if there was nothing to compact."""
        body = self.messages[1:]
        keep_budget = int(self.context_window * KEEP_RECENT)
        cut, kept = len(body), 0
        while cut > 0 and kept + estimate_tokens(body[cut - 1:cut]) <= keep_budget:
            cut -= 1
            kept += estimate_tokens(body[cut:cut + 1])
        # Never start the kept part on a tool result: it must stay with the call that produced it.
        while cut < len(body) and body[cut]["role"] == "tool":
            cut += 1
        # Always keep the latest step (the newest user message, or the newest tool call and its
        # results) verbatim, even if it's over budget: it's what the model is working on right now.
        last_step = max(i for i, m in enumerate(body) if m["role"] != "tool") if body else 0
        cut = min(cut, last_step)
        if cut == 0:
            return False
        before = self.context_tokens()

        # Stage 1: clear old tool results, keep every user and assistant message as is.
        pruned_body, pruned = list(body), 0
        for i, m in enumerate(body[:cut]):
            content = m.get("content") or ""
            if m["role"] == "tool" and len(content) >= PRUNE_MIN_CHARS and not content.startswith(PRUNED_MARK):
                first_line = content.strip().splitlines()[0][:120]
                pruned_body[i] = {**m, "content": f"{PRUNED_MARK} It was {len(content):,} chars and began: "
                                                  f"{first_line!r}. Run the tool again if you need it."}
                pruned += 1
        if pruned:
            self.messages = [self.messages[0]] + pruned_body
            self.usage_mark = None
            if self.context_tokens() <= self.context_window * PRUNE_TARGET:
                print(f"{self.indent}\033[2m(Context compacted: cleared {pruned} old tool results, "
                      f"~{before:,} -> ~{self.context_tokens():,} tokens)\033[0m")
                return True

        # Stage 2: summarize the old part (from the original messages, so nothing is lost to stage 1).
        transcript = render_transcript(body[:cut], max_chars=int(self.context_window * 0.5) * 4)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": transcript}],
        )
        summary = (response.choices[0].message.content or "").strip()
        note = f"[Summary of the earlier conversation, compacted to save context]\n{summary}"

        tail = body[cut:]
        request = self.current_request
        if request and not any(m["role"] == "user" and (m.get("content") or "").endswith(request) for m in tail):
            note += f"\n\nThe user's current request, verbatim:\n{request}"
        tail_answers = {m.get("content") for m in tail if m["role"] == "assistant"}
        log = render_turn_log([t for t in self.turn_log if t[1] not in tail_answers],
                              max_chars=int(self.context_window * 0.1) * 4)
        if log:
            note += f"\n\nEarlier requests and your answers, verbatim (long ones clipped):\n{log}"
        note += "\n\nContext is limited: prefer grep and read_file with offset/limit over re-reading whole files."
        if tail and tail[0]["role"] == "user":  # avoid two user messages in a row
            tail[0] = {**tail[0], "content": f"{note}\n\n{tail[0]['content']}"}
        else:
            tail.insert(0, {"role": "user", "content": note})
        self.messages = [self.messages[0]] + tail
        self.usage_mark = None
        print(f"{self.indent}\033[2m(Context compacted: summarized older turns, ~{before:,} -> ~{self.context_tokens():,} tokens)\033[0m")
        return True

    def run_tool(self, name: str, raw_args: str) -> str:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as e:
            return f"Error: arguments were not valid JSON ({e}). Send valid JSON."
        if name not in self.tool_names:
            return f"Error: tool {name!r} isn't available here. Available tools: {', '.join(sorted(self.tool_names))}"
        if name == "task":
            try:
                return self.run_subagent(**args)
            except TypeError as e:
                return f"Error: bad arguments for task: {e}"
        fn = getattr(self.tools, name)
        if name not in SAFE_TOOLS and name not in self.always_allowed:
            allowed, note = self.permission(name, args, self.always_allowed)
            if not allowed:
                return note
        try:
            return fn(**args)
        except ToolError as e:
            return f"Error: {e}"
        except TypeError as e:  # wrong / missing arguments
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def repair_after_interrupt(self) -> None:
        """After Ctrl+C, give any tool call that never got a result a placeholder one.
        Providers reject a conversation where a tool call has no matching result."""
        answered = {m["tool_call_id"] for m in self.messages if m["role"] == "tool"}
        repaired = []
        for m in self.messages:
            repaired.append(m)
            for tc in m.get("tool_calls", []):
                if tc["id"] not in answered:
                    repaired.append({"role": "tool", "tool_call_id": tc["id"], "content": "Interrupted by the user."})
        self.messages = repaired

    def run_subagent(self, description: str, prompt: str, mode: str = "explore") -> str:
        """Run `prompt` in a fresh Agent and return only its final answer."""
        if mode not in SUBAGENT_TOOLS:
            return f"Error: mode must be one of {', '.join(SUBAGENT_TOOLS)}"
        sub = Agent(self.client, self.model, self.tools.workspace, permission=self.permission,
                    context_window=self.context_window, tool_names=SUBAGENT_TOOLS[mode],
                    depth=self.depth + 1, always_allowed=self.always_allowed)
        print(f"{sub.indent}\033[35m[subagent: {description} ({mode})]\033[0m")
        answer = sub.send(prompt)
        print(f"{sub.indent}\033[35m[subagent done in {sub.last_steps} steps]\033[0m")
        return answer or "(The subagent finished without an answer.)"

    def send(self, user_text: str) -> str:
        """Run one user turn through the agent loop and return the final answer."""
        self.messages.append({"role": "user", "content": user_text})
        self.current_request = user_text
        compactions = 0
        for step in range(1, MAX_STEPS + 1):
            self.last_steps = step
            if self.context_tokens() > self.context_window * COMPACT_AT:
                if compactions >= MAX_COMPACTIONS_PER_TURN:
                    return (f"(Stopped: compacted {compactions} times in one turn and the conversation keeps "
                            f"outgrowing the {self.context_window:,}-token context window. Raise CONTEXT_WINDOW "
                            f"if your model supports more, or split the task into smaller steps.)")
                if self.compact():
                    compactions += 1
            response = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=self.schemas
            )
            usage = getattr(response, "usage", None)
            if usage and getattr(usage, "prompt_tokens", None):
                self.usage_mark = (usage.prompt_tokens, len(self.messages))
            msg = response.choices[0].message

            # Store the assistant turn as a plain dict so any provider accepts it back.
            entry = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            reasoning = getattr(msg, "reasoning_content", None)  # DeepSeek thinking mode
            if reasoning:
                entry["reasoning_content"] = reasoning
            self.messages.append(entry)

            if not msg.tool_calls:
                self.turn_log.append((user_text, msg.content or ""))
                return msg.content or ""

            if msg.content:
                print(textwrap.indent(f"\033[2m{msg.content}\033[0m", self.indent))
            for tc in msg.tool_calls:
                print(f"{self.indent}\033[36m> {tc.function.name}\033[0m \033[2m{tc.function.arguments[:200]}\033[0m")
                result = self.run_tool(tc.function.name, tc.function.arguments)
                print(textwrap.indent(preview(result), self.indent))
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return f"(Stopped after {MAX_STEPS} steps without a final answer.)"


def main() -> None:
    here = Path(__file__).parent
    load_dotenv(here / ".env")
    missing = [k for k in ("BASE_URL", "API_KEY", "MODEL") if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing {', '.join(missing)}. Copy .env.example to .env and fill it in.")

    workspace = Path(os.environ.get("WORKSPACE") or Path.cwd())
    client = OpenAI(base_url=os.environ["BASE_URL"], api_key=os.environ["API_KEY"])
    window = int(os.environ.get("CONTEXT_WINDOW") or DEFAULT_CONTEXT_WINDOW)
    agent = Agent(client, os.environ["MODEL"], workspace, context_window=window)

    if len(sys.argv) > 1:
        print(agent.send(" ".join(sys.argv[1:])))
        return

    agents_md = " (AGENTS.md loaded)" if (agent.tools.workspace / "AGENTS.md").is_file() else ""
    print(f"Kyklos ready. model={agent.model}  workspace={agent.tools.workspace}{agents_md}")
    print("Commands: /reset clears the conversation, /compact summarizes it, /exit quits.\n")
    while True:
        try:
            text = input("\033[1m> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/exit":
            break
        if text == "/reset":
            agent.reset()
            print("Conversation cleared.")
            continue
        try:
            if text == "/compact":
                if not agent.compact():
                    print("Nothing to compact yet.")
                continue
            print(f"\n{agent.send(text)}")
            used = agent.context_tokens()
            print(f"\033[2m(context: ~{used:,} / {agent.context_window:,} tokens, {used / agent.context_window:.0%})\033[0m\n")
        except KeyboardInterrupt:
            agent.repair_after_interrupt()
            print("\n(Interrupted.)")
        except Exception as e:  # API errors: bad key, rate limit, network...
            print(f"\n\033[31mAPI error: {e}\033[0m\n")


if __name__ == "__main__":
    main()
