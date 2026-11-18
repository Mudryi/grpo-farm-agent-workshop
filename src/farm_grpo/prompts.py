"""System prompts and chat formatting."""

from __future__ import annotations

from .environment import DAY_PARTS, TODAY, tool_spec_text

_DAY_PARTS_TEXT = ", ".join(f"{k} = {a}-{b}" for k, (a, b) in DAY_PARTS.items())

SYSTEM_PROMPT = f"""You are a farm-operations assistant for a dairy farm.
Today is {TODAY.isoformat()} ({TODAY.strftime('%A')}). Day parts: {_DAY_PARTS_TEXT}.

You can call exactly ONE of these read-only tools:
{tool_spec_text()}

Answer with ONLY a JSON object of the form
{{"tool": "<tool name>", "arguments": {{...}}}}
Use 24h HH:MM times and YYYY-MM-DD dates. Do not add any explanation."""


AGENT_SYSTEM_PROMPT = f"""You are a farm-operations agent for a dairy farm.
Today is {TODAY.isoformat()} ({TODAY.strftime('%A')}). Day parts: {_DAY_PARTS_TEXT}.

Tools (read-only):
{tool_spec_text()}

Work step by step. On each turn output ONLY one JSON object, either
  {{"tool": "<tool name>", "arguments": {{...}}}}   to call a tool, or
  {{"final_answer": "<short answer for the farm manager>"}}   when you are done.
Tool results will be sent back to you as messages starting with TOOL_RESULT."""


def build_messages(question: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """Conversational prompt in the format TRL's GRPOTrainer expects."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
