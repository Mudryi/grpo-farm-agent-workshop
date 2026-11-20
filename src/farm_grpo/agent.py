"""Multi-step farm investigation (advanced demo / challenge).

The GRPO exercise trains ONE tool call. Here we test whether that skill transfers to
an agent loop:  model -> tool call -> TOOL_RESULT -> next call ... -> final_answer.
Tasks are graded by the ENVIRONMENT OUTCOME (was the final answer right?), not by
matching a reference JSON. That is the reward a "real" RL environment would use.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .environment import DAY_PARTS, TODAY, FarmWorld
from .parsing import find_json_objects
from .prompts import AGENT_SYSTEM_PROMPT

LATE_THRESHOLD_MIN = 10


def _minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return 60 * h + m


# --------------------------------------------------------------------------------------
# Ground truth computed from the simulated world
# --------------------------------------------------------------------------------------
def robot_lateness_truth(world: FarmWorld, barn: str, period: str = "morning") -> dict:
    start, end = DAY_PARTS[period]
    planned = [t for t in world.get_feeding_schedule(barn, TODAY.isoformat())["planned_feed_push"]
               if start <= t < end]
    events = [e["time"] for e in world.get_robot_events(barn, start, end)["events"]]
    first = planned[0]
    after = [e for e in events if 0 <= _minutes(e) - _minutes(first) <= 60]
    if not after:
        return {"barn": barn, "scheduled": first, "actual": None, "delay_min": None,
                "late": True, "missed": True}
    delay = _minutes(after[0]) - _minutes(first)
    return {"barn": barn, "scheduled": first, "actual": after[0], "delay_min": delay,
            "late": delay >= LATE_THRESHOLD_MIN, "missed": False}


@dataclass
class AgentTask:
    question: str
    required_tools: set[str]
    truth: dict

    def grade(self, final_answer: str | None, tools_called: list[str]) -> dict:
        ans = (final_answer or "").lower()
        negated = re.search(r"\bnot late\b|\bon time\b|не запізн|не затрим|вчасно", ans)
        said_yes = bool(re.search(r"\byes\b|\blate\b|\bdelayed\b|\bmissed\b|\bтак\b|запізн|затрим|пропуст", ans)) and not negated
        said_no = bool(re.search(r"\bno\b|\bon time\b|\bnot late\b|\bні\b|вчасно|не запізн|не затрим", ans)) and not said_yes
        yes_no_ok = said_yes if self.truth["late"] else said_no
        minutes_ok = True
        if self.truth.get("delay_min") and self.truth["late"]:
            nums = [int(n) for n in re.findall(r"\d+", ans)]
            minutes_ok = any(abs(n - self.truth["delay_min"]) <= 2 for n in nums)
        tools_ok = self.required_tools <= set(tools_called)
        return {"answer_correct": yes_no_ok and minutes_ok, "used_required_tools": tools_ok,
                "success": yes_no_ok and minutes_ok and tools_ok}


QUESTION = ("Чи запізнився робот-підштовхувач корму на перше годування в корівнику {barn} сьогодні вранці? "
            "Якщо так, то на скільки хвилин?")


def make_tasks(world: FarmWorld) -> list[AgentTask]:
    tasks = []
    for barn in ["A", "B", "C", "D"]:
        tasks.append(AgentTask(
            question=QUESTION.format(barn=barn),
            required_tools={"get_feeding_schedule", "get_robot_events"},
            truth=robot_lateness_truth(world, barn),
        ))
    return tasks


# --------------------------------------------------------------------------------------
# Scripted oracle: what a perfect investigation looks like (used in Section 2)
# --------------------------------------------------------------------------------------
def oracle_investigation(world: FarmWorld, barn: str = "A") -> list[dict]:
    steps = []
    call1 = {"tool": "get_feeding_schedule", "arguments": {"barn_id": barn, "date": TODAY.isoformat()}}
    steps.append({"call": call1, "result": world.execute(call1["tool"], call1["arguments"])})
    s, e = DAY_PARTS["morning"]
    call2 = {"tool": "get_robot_events", "arguments": {"barn_id": barn, "start_time": s, "end_time": e}}
    steps.append({"call": call2, "result": world.execute(call2["tool"], call2["arguments"])})
    t = robot_lateness_truth(world, barn)
    if t["missed"]:
        ans = f"Yes. The robot missed the {t['scheduled']} feed push in Barn {barn}."
    elif t["late"]:
        ans = f"Yes. Planned {t['scheduled']}, robot arrived {t['actual']}: {t['delay_min']} minutes late."
    else:
        ans = f"No. Planned {t['scheduled']}, robot arrived {t['actual']} (on time)."
    steps.append({"final_answer": ans})
    return steps


# --------------------------------------------------------------------------------------
# LLM agent loop
# --------------------------------------------------------------------------------------
@dataclass
class AgentResult:
    question: str
    trajectory: list[dict] = field(default_factory=list)
    final_answer: str | None = None

    @property
    def tools_called(self) -> list[str]:
        return [s["call"].get("tool") for s in self.trajectory if "call" in s]

    def pretty(self) -> str:
        lines = [f"Q: {self.question}"]
        for i, s in enumerate(self.trajectory, 1):
            if "call" in s:
                lines.append(f"  [{i}] CALL   {json.dumps(s['call'])}")
                lines.append(f"      RESULT {json.dumps(s['result'])[:300]}")
            elif "raw" in s:
                lines.append(f"  [{i}] UNPARSEABLE: {s['raw'][:200]!r}")
        lines.append(f"  FINAL: {self.final_answer}")
        return "\n".join(lines)


def run_agent(model, tok, question: str, world: FarmWorld, max_turns: int = 4,
              max_new_tokens: int = 160) -> AgentResult:
    from .evaluate import generate

    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    res = AgentResult(question)
    for _ in range(max_turns):
        text = generate(model, tok, [messages], max_new_tokens=max_new_tokens)[0][0]
        objs = [o for o in find_json_objects(text) if isinstance(o, dict)]
        if not objs:
            res.trajectory.append({"raw": text})
            res.final_answer = text.strip()  # treat free text as the answer
            break
        action = objs[0]
        if "final_answer" in action:
            res.final_answer = str(action["final_answer"])
            break
        tool, args = action.get("tool"), action.get("arguments", {})
        result = world.execute(tool, args if isinstance(args, dict) else {})
        res.trajectory.append({"call": {"tool": tool, "arguments": args}, "result": result})
        messages.append({"role": "assistant", "content": json.dumps(action)})
        messages.append({"role": "user", "content": "TOOL_RESULT: " + json.dumps(result)})
    return res


def evaluate_agent(model, tok, world: FarmWorld, tasks: list[AgentTask] | None = None):
    import pandas as pd

    tasks = tasks or make_tasks(world)
    rows, results = [], []
    for t in tasks:
        r = run_agent(model, tok, t.question, world)
        g = t.grade(r.final_answer, r.tools_called)
        results.append(r)
        rows.append({"question": t.question, "truth": t.truth, "tools_called": r.tools_called,
                     "final_answer": r.final_answer, **g})
    return pd.DataFrame(rows), results
