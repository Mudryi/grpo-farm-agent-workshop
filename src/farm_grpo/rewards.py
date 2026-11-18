"""STUDENT EXERCISE: reward functions for GRPO.

Implement the three TODO functions below (or the copies in Section 5 of the notebook).

Teaching reward (max = 1.0):
    format   valid JSON with {"tool", "arguments"} structure   +0.1
    tool     correct tool selected                            +0.2
    args     correct tool AND all arguments match             +0.7
    invalid  cannot parse / validate                          -0.2

Useful helpers:
    completion_text(c)       -> str            (TRL completions can be chat dicts)
    parse_tool_call(text)    -> ParsedCall     (.has_schema, .tool, .arguments, .norm_args)
    normalize_args(tool, d)  -> canonical dict or None

Run `check_rewards(format_reward, tool_reward, args_reward)` to test your implementation.
"""

from __future__ import annotations

import json  # noqa: F401  (you will need it for expected_args)

from .environment import normalize_args  # noqa: F401
from .parsing import completion_text, parse_tool_call  # noqa: F401


def format_reward(completions, **kwargs) -> list[float]:
    """+0.1 for a clean {"tool", "arguments"} JSON object, -0.2 otherwise."""
    # TODO(student)
    raise NotImplementedError


def tool_reward(completions, expected_tool, **kwargs) -> list[float]:
    """+0.2 when the selected tool equals expected_tool, else 0."""
    # TODO(student)
    raise NotImplementedError


def args_reward(completions, expected_tool, expected_args, **kwargs) -> list[float]:
    """+0.7 when the tool is right AND all normalised arguments match, else 0.

    expected_args is a list of JSON *strings*. Use json.loads(...).
    """
    # TODO(student)
    raise NotImplementedError


# --------------------------------------------------------------------------------------
# Test harness
# --------------------------------------------------------------------------------------
def _c(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


TEST_CASES = [
    # (name, completion text, expected total reward)
    ("perfect",
     '{"tool": "get_robot_events", "arguments": {"barn_id": "A", "start_time": "07:00", "end_time": "08:00"}}', 1.0),
    ("perfect, non-canonical time",
     '{"tool": "get_robot_events", "arguments": {"barn_id": "A", "start_time": "7:00", "end_time": "8:00"}}', 1.0),
    ("perfect inside code fence",
     '```json\n{"tool": "get_robot_events", "arguments": {"barn_id": "A", "start_time": "07:00", "end_time": "08:00"}}\n```', 1.0),
    ("right tool, wrong barn",
     '{"tool": "get_robot_events", "arguments": {"barn_id": "B", "start_time": "07:00", "end_time": "08:00"}}', 0.3),
    ("wrong tool",
     '{"tool": "get_feeding_schedule", "arguments": {"barn_id": "A", "date": "2026-03-10"}}', 0.1),
    ("extra key in JSON",
     '{"tool": "get_robot_events", "arguments": {}, "reason": "x"}', -0.2),
    ("prose before JSON",
     'Sure! {"tool": "get_robot_events", "arguments": {"barn_id": "A", "start_time": "07:00", "end_time": "08:00"}}', -0.2),
    ("not JSON", "I would call the robot tool.", -0.2),
]
TEST_EXPECTED_TOOL = "get_robot_events"
TEST_EXPECTED_ARGS = json.dumps({"barn_id": "A", "start_time": "07:00", "end_time": "08:00"})


def check_rewards(*funcs, verbose: bool = True) -> bool:
    completions = [_c(t) for _, t, _ in TEST_CASES]
    n = len(completions)
    cols = {"expected_tool": [TEST_EXPECTED_TOOL] * n, "expected_args": [TEST_EXPECTED_ARGS] * n}
    per_func = [f(completions, **cols) for f in funcs]
    ok_all = True
    for i, (name, _, want) in enumerate(TEST_CASES):
        got = round(sum(p[i] for p in per_func), 3)
        ok = abs(got - want) < 1e-6
        ok_all &= ok
        if verbose:
            parts = " + ".join(f"{p[i]:+.1f}" for p in per_func)
            print(f"{'PASS' if ok else 'FAIL'}  {name:<28} {parts:<20} = {got:+.2f} (want {want:+.2f})")
    if verbose:
        print("\nAll tests passed!" if ok_all else "\nSome tests failed.")
    return ok_all


def get_reward_funcs(prefer_student: bool = True):
    """Return the student's functions if implemented, else the reference solution."""
    if prefer_student:
        try:
            check_rewards(format_reward, tool_reward, args_reward, verbose=False)
            return [format_reward, tool_reward, args_reward]
        except NotImplementedError:
            print("[rewards] student functions not implemented yet -> using reference solution")
    from solutions.rewards_solution import REWARD_FUNCS

    return REWARD_FUNCS
