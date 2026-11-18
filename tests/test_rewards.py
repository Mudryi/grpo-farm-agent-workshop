import json

from farm_grpo.evaluate import score_completion
from farm_grpo.rewards import TEST_CASES, check_rewards
from farm_grpo.rewards_bad import keyword_reward
from solutions.rewards_solution import REWARD_FUNCS


def test_solution_passes_harness():
    assert check_rewards(*REWARD_FUNCS, verbose=False)


def test_reward_range():
    for _, text, want in TEST_CASES:
        assert -0.2 <= want <= 1.0


def test_score_completion_matches_solution():
    args = json.dumps({"barn_id": "A", "start_time": "07:00", "end_time": "08:00"})
    for _, text, want in TEST_CASES:
        assert abs(score_completion(text, "get_robot_events", args)["reward"] - want) < 1e-6


def test_keyword_reward_is_hackable():
    hack = [[{"role": "assistant", "content": "get_robot_events get_mixer_events get_feeding_schedule"}]]
    assert keyword_reward(hack, expected_tool=["get_robot_events"]) == [1.0]
