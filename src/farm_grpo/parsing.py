"""Parse and validate model outputs as tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .environment import TOOL_SPECS, normalize_args

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S | re.I)


def completion_text(completion: Any) -> str:
    """TRL passes completions either as str or as [{'role': 'assistant', 'content': ...}]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "") or ""
    if isinstance(completion, dict):
        return completion.get("content", "") or ""
    return str(completion)


def find_json_objects(text: str) -> list[Any]:
    """All JSON values that start with '{' found anywhere in `text`."""
    dec = json.JSONDecoder()
    objs, i = [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
            objs.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1
    return objs


@dataclass
class ParsedCall:
    raw: str
    is_json: bool = False          # the whole output (minus a code fence) is one JSON object
    has_schema: bool = False       # keys are exactly {"tool", "arguments"}, arguments is a dict
    tool: str | None = None
    arguments: dict | None = None
    norm_args: dict | None = None  # canonical arguments, None if invalid for the tool
    n_calls: int = 0               # number of tool-call-looking JSON objects in the output
    error: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def valid_call(self) -> bool:
        """Parseable, right structure, known tool and valid argument names/values."""
        return self.has_schema and self.tool in TOOL_SPECS and self.norm_args is not None


def parse_tool_call(text: str) -> ParsedCall:
    raw = text
    s = text.strip()
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()

    pc = ParsedCall(raw=raw)
    objs = find_json_objects(raw)
    pc.n_calls = sum(1 for o in objs if isinstance(o, dict) and "tool" in o)

    try:
        action = json.loads(s)
    except (ValueError, TypeError):
        pc.error = "invalid_json" if not objs else "extra_text"
        return pc
    if not isinstance(action, dict):
        pc.error = "not_an_object"
        return pc
    pc.is_json = True

    if set(action) != {"tool", "arguments"} or not isinstance(action.get("arguments"), dict):
        pc.error = "bad_schema"
        return pc
    pc.has_schema = True
    pc.tool = action["tool"] if isinstance(action["tool"], str) else None
    pc.arguments = action["arguments"]

    if pc.tool not in TOOL_SPECS:
        pc.error = "unknown_tool"
        return pc
    pc.norm_args = normalize_args(pc.tool, pc.arguments)
    if pc.norm_args is None:
        pc.error = "invalid_arguments"
    return pc
