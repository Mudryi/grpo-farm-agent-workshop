import json

from farm_grpo.evaluate import score_completion
from farm_grpo.rewards import REWARD_FUNCS, TEST_CASES, check_rewards
from farm_grpo.rewards_bad import many_args_reward


def test_solution_passes_harness():
    assert check_rewards(*REWARD_FUNCS, verbose=False)


def test_reward_range():
    for _, text, want in TEST_CASES:
        assert -0.2 <= want <= 1.0


def test_score_completion_matches_solution():
    args = json.dumps({"barn_id": "A", "start_time": "07:00", "end_time": "08:00"})
    for _, text, want in TEST_CASES:
        assert abs(score_completion(text, "get_robot_events", args)["reward"] - want) < 1e-6


def test_many_args_reward_is_hackable():
    stuffed = '{"tool": "get_robot_events", "arguments": {"a": 1, "b": 2, "c": 3, "d": 4}}'
    assert many_args_reward([[{"role": "assistant", "content": stuffed}]]) == [1.0]
    assert score_completion(stuffed, "get_robot_events", json.dumps({"barn_id": "A"}))["reward"] < 0.5
