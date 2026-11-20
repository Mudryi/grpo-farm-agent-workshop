"""A deliberately BAD reward for the reward-hacking challenge (Section 8.1).

It looks plausible ("more detailed calls are better") but is not task success:
it never looks at the expected tool or arguments.
"""

from __future__ import annotations

from .parsing import completion_text, find_json_objects


def many_args_reward(completions, **kwargs) -> list[float]:
    """0.25 per argument key (capped at 1.0). Hack: stuff the arguments dict with extra keys."""
    out = []
    for c in completions:
        objs = [o for o in find_json_objects(completion_text(c)) if isinstance(o, dict)]
        n = max((len(o.get("arguments", {})) for o in objs if isinstance(o.get("arguments"), dict)), default=0)
        out.append(min(0.25 * n, 1.0))
    return out
