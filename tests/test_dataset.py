import json
import re

from farm_grpo.agent import make_tasks
from farm_grpo.dataset import TRAIN_BARNS, generate_records
from farm_grpo.environment import TOOL_NAMES, build_world, normalize_args


def test_sizes_and_balance():
    r = generate_records()
    assert [len(r[s]) for s in ("train", "val", "test")] == [240, 40, 80]
    for split in r.values():
        tools = {x.expected_tool for x in split}
        assert tools == set(TOOL_NAMES)


def test_no_template_family_leakage():
    r = generate_records()
    fam = {s: {x.family for x in r[s]} for s in r}
    assert not fam["train"] & fam["val"]
    assert not fam["train"] & fam["test"]
    assert not fam["val"] & fam["test"]


def test_no_duplicate_questions_across_splits():
    r = generate_records()
    q = {s: {x.question for x in r[s]} for s in r}
    assert not q["train"] & q["test"]


def test_heldout_barn_only_in_eval():
    r = generate_records()
    train_barns = {json.loads(x.expected_args).get("barn_id") for x in r["train"]} - {None}
    assert train_barns <= set(TRAIN_BARNS)


def test_labels_are_valid_and_executable():
    world = build_world()
    for split in generate_records().values():
        for x in split:
            args = json.loads(x.expected_args)
            assert normalize_args(x.expected_tool, args) is not None
            assert "error" not in world.execute(x.expected_tool, args)


def test_headline_example():
    w = build_world()
    assert w.get_feeding_schedule("A", "2026-03-10")["planned_feed_push"][0] == "07:00"
    assert w.get_robot_events("A", "06:00", "12:00")["events"][0]["time"] == "07:18"


def test_requests_are_ukrainian():
    for split in generate_records().values():
        for x in split:
            assert re.search("[а-яіїєґ]", x.question.lower()), x.question
            assert not re.search(r"\b(barn|today|yesterday|morning|the)\b", x.question.lower()), x.question


def test_agent_grading():
    w = build_world()
    task = make_tasks(w)[0]          # barn A: 18 min late
    assert task.question.startswith("Чи запізнився")
    tools = ["get_feeding_schedule", "get_robot_events"]
    assert task.grade("Так, робот запізнився на 18 хвилин.", tools)["success"]
    assert not task.grade("Ні, робот прибув вчасно.", tools)["success"]


def test_no_double_period():
    assert not any(".." in x.question for s in generate_records().values() for x in s)
