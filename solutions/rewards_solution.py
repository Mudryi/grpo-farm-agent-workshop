"""REFERENCE SOLUTION for the reward functions (instructor copy).

Teaching reward (max = 1.0):
    format   valid JSON with {"tool", "arguments"} structure   +0.1
    tool     correct tool selected                            +0.2
    args     correct tool AND all arguments match             +0.7
    invalid  cannot parse / validate                          -0.2

TRL calls every reward function with:
    completions  : list of completions (conversational: [{"role": "assistant", "content": ...}])
    **dataset columns, here expected_tool: list[str], expected_args: list[str (JSON)]
and sums the returned lists (optionally weighted by GRPOConfig.reward_weights).
"""

from __future__ import annotations

import json

from farm_grpo.environment import normalize_args, normalize_value
from farm_grpo.parsing import completion_text, parse_tool_call


def format_reward(completions, **kwargs) -> list[float]:
    """+0.1 for one clean JSON object with keys {tool, arguments}; -0.2 otherwise."""
    rewards = []
    for c in completions:
        pc = parse_tool_call(completion_text(c))
        rewards.append(0.1 if pc.has_schema else -0.2)
    return rewards


def tool_reward(completions, expected_tool, **kwargs) -> list[float]:
    """+0.2 if the selected tool equals the expected tool."""
    rewards = []
    for c, tool in zip(completions, expected_tool):
        pc = parse_tool_call(completion_text(c))
        rewards.append(0.2 if pc.has_schema and pc.tool == tool else 0.0)
    return rewards


def args_reward(completions, expected_tool, expected_args, **kwargs) -> list[float]:
    """+0.7 if the tool is right and every (normalised) argument matches."""
    rewards = []
    for c, tool, args_json in zip(completions, expected_tool, expected_args):
        pc = parse_tool_call(completion_text(c))
        target = normalize_args(tool, json.loads(args_json))
        ok = pc.has_schema and pc.tool == tool and pc.norm_args is not None and pc.norm_args == target
        rewards.append(0.7 if ok else 0.0)
    return rewards


def args_partial_reward(completions, expected_tool, expected_args, **kwargs) -> list[float]:
    """Optional shaped variant: 0.7 * fraction of matching arguments (denser signal)."""
    rewards = []
    for c, tool, args_json in zip(completions, expected_tool, expected_args):
        pc = parse_tool_call(completion_text(c))
        target = normalize_args(tool, json.loads(args_json))
        if not (pc.has_schema and pc.tool == tool and isinstance(pc.arguments, dict)):
            rewards.append(0.0)
            continue
        hits = sum(normalize_value(k, pc.arguments.get(k)) == v for k, v in target.items())
        extra = len(set(pc.arguments) - set(target))
        rewards.append(0.7 * max(hits - extra, 0) / len(target))
    return rewards


REWARD_FUNCS = [format_reward, tool_reward, args_reward]
