import json
import re

import pytest

from farm_grpo.agent import make_tasks
from farm_grpo.dataset import SPLIT_OF_TEMPLATE_INDEX, TEMPLATES, TEMPLATES_UK, TRAIN_BARNS, generate_records
from farm_grpo.environment import TOOL_NAMES, build_world, normalize_args


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_sizes_and_balance(lang):
    r = generate_records(lang=lang)
    assert [len(r[s]) for s in ("train", "val", "test")] == [240, 40, 80]
    for split in r.values():
        tools = {x.expected_tool for x in split}
        assert tools == set(TOOL_NAMES)


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_no_template_family_leakage(lang):
    r = generate_records(lang=lang)
    fam = {s: {x.family for x in r[s]} for s in r}
    assert not fam["train"] & fam["val"]
    assert not fam["train"] & fam["test"]
    assert not fam["val"] & fam["test"]


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_no_duplicate_questions_across_splits(lang):
    r = generate_records(lang=lang)
    q = {s: {x.question for x in r[s]} for s in r}
    assert not q["train"] & q["test"]


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_heldout_barn_only_in_eval(lang):
    r = generate_records(lang=lang)
    train_barns = {json.loads(x.expected_args).get("barn_id") for x in r["train"]} - {None}
    assert train_barns <= set(TRAIN_BARNS)


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_labels_are_valid_and_executable(lang):
    world = build_world()
    for split in generate_records(lang=lang).values():
        for x in split:
            args = json.loads(x.expected_args)
            assert normalize_args(x.expected_tool, args) is not None
            assert "error" not in world.execute(x.expected_tool, args)


def test_headline_example():
    w = build_world()
    assert w.get_feeding_schedule("A", "2026-03-10")["planned_feed_push"][0] == "07:00"
    assert w.get_robot_events("A", "06:00", "12:00")["events"][0]["time"] == "07:18"


def test_uk_templates_mirror_en():
    for tool in TEMPLATES:
        assert [(f, k) for f, _, k in TEMPLATES[tool]] == [(f, k) for f, _, k in TEMPLATES_UK[tool]]
    assert len(SPLIT_OF_TEMPLATE_INDEX) == 8


def test_uk_requests_are_ukrainian():
    for split in generate_records(lang="uk").values():
        for x in split:
            assert re.search("[а-яіїєґ]", x.question.lower()), x.question
            assert not re.search(r"\b(barn|today|yesterday|morning|the)\b", x.question.lower()), x.question


def test_uk_agent_grading():
    w = build_world()
    task = make_tasks(w, lang="uk")[0]          # barn A: 18 min late
    assert task.question.startswith("Чи запізнився")
    tools = ["get_feeding_schedule", "get_robot_events"]
    assert task.grade("Так, робот запізнився на 18 хвилин.", tools)["success"]
    assert not task.grade("Ні, робот прибув вчасно.", tools)["success"]


def test_uk_no_double_period():
    assert not any(".." in x.question for s in generate_records(lang="uk").values() for x in s)
