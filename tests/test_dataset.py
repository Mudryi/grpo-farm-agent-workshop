import json

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
