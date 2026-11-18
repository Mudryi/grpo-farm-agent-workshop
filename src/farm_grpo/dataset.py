"""Synthetic training / validation / test data for single-step tool selection.

Key teaching point: we split by TEMPLATE FAMILY, not by random shuffling.
Every tool has 8 phrasing families. 5 are used for training, 1 for validation and 2
only for testing, so the test set measures generalisation to *new phrasings*.
Barn D and cameras 7-8 also only appear in validation/test (unseen values).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

from .environment import BARNS, CAMERAS, DAY_PARTS, TODAY
from .prompts import SYSTEM_PROMPT, build_messages

# --------------------------------------------------------------------------------------
# Templates. Placeholders: {barn} {start} {end} {period} {date} {cam} {other_barn}
# Each entry: (family_id, template, kind) where kind selects how values are sampled.
# --------------------------------------------------------------------------------------
TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "get_robot_events": [
        ("robot_show", "Show robot visits in Barn {barn} from {start} to {end}.", "window"),
        ("robot_find", "Find robot events in Barn {barn} between {start} and {end}.", "window"),
        ("robot_when", "When did the feed-pushing robot pass through barn {barn} between {start} and {end}?", "window"),
        ("robot_period", "Did the robot push feed in Barn {barn} this {period}?", "period"),
        ("robot_distract", "Never mind the mixer, I need the robot log for barn {barn} from {start} until {end}.", "window"),
        # --- validation family
        ("robot_terse", "robot activity barn {barn} {start}-{end}", "window"),
        # --- test-only families
        ("robot_count", "How many times did the pusher robot run in Barn {barn} during the {period}?", "period"),
        ("robot_distract2", "The schedule for Barn {other_barn} is fine. Which feed-push runs did the robot make in Barn {barn} between {start} and {end}?", "window"),
    ],
    "get_mixer_events": [
        ("mixer_when", "When did the mixer arrive in Barn {barn} {date}?", "date"),
        ("mixer_show", "Show mixer events for barn {barn} {date}.", "date"),
        ("mixer_deliv", "Give me the feed mixer deliveries to Barn {barn} {date}.", "date"),
        ("mixer_visit", "Did the mixer wagon visit Barn {barn} {date}?", "date"),
        ("mixer_distract", "Ignore the robot for now. What time was the mixer in Barn {barn} {date}?", "date"),
        ("mixer_terse", "mixer log barn {barn} {date}", "date"),
        ("mixer_fresh", "At what time was fresh feed delivered by the mixer to barn {barn} {date}?", "date"),
        ("mixer_distract2", "Camera {cam} looks fine. I want the TMR mixer arrivals in Barn {barn} {date}.", "date"),
    ],
    "get_feeding_schedule": [
        ("sched_what", "What is the scheduled feeding time in Barn {barn} {date}?", "date"),
        ("sched_plan", "Show the feeding plan for Barn {barn} {date}.", "date"),
        ("sched_supposed", "When is feed supposed to be pushed in barn {barn} {date}?", "date"),
        ("sched_says", "What does the feeding schedule say for Barn {barn} {date}?", "date"),
        ("sched_distract", "Not the actual robot events, just the planned feeding times for Barn {barn} {date}.", "date"),
        ("sched_terse", "feeding timetable barn {barn} {date}", "date"),
        ("sched_cows", "According to the plan, when should the cows in Barn {barn} be fed {date}?", "date"),
        ("sched_distract2", "Forget the mixer in Barn {other_barn}. What are the expected feeding times for barn {barn} {date}?", "date"),
    ],
    "get_camera_status": [
        ("cam_working", "Was Camera {cam} working {date}?", "cam"),
        ("cam_online", "Is cam {cam} online {date}?", "cam"),
        ("cam_check", "Check the status of camera {cam} {date}.", "cam"),
        ("cam_outage", "Did camera #{cam} have any outages {date}?", "cam"),
        ("cam_distract", "The robot data looks fine. I only want to know if Camera {cam} was recording {date}.", "cam"),
        ("cam_terse", "camera {cam} health {date}", "cam"),
        ("cam_video", "Do we have video from camera number {cam} {date}?", "cam"),
        ("cam_distract2", "Any downtime for camera {cam} in Barn {barn} {date}?", "cam"),
    ],
}
SPLIT_OF_TEMPLATE_INDEX = ["train"] * 5 + ["val"] + ["test"] * 2

TRAIN_BARNS = ["A", "B", "C"]          # barn D is held out
TRAIN_CAMERAS = [1, 2, 3, 4, 5, 6]     # cameras 7-8 are held out


@dataclass
class Record:
    id: str
    split: str
    family: str
    question: str
    expected_tool: str
    expected_args: str  # JSON string (keeps the HF dataset schema simple)

    def to_row(self) -> dict:
        row = asdict(self)
        row["prompt"] = build_messages(self.question, SYSTEM_PROMPT)
        return row


# --------------------------------------------------------------------------------------
# Value sampling and surface-form variation
# --------------------------------------------------------------------------------------
def _fmt_time(hhmm: str, rng: random.Random) -> str:
    h, m = map(int, hhmm.split(":"))
    style = rng.choice(["24h", "24h", "nopad", "ampm"])
    if style == "24h":
        return hhmm
    if style == "nopad":
        return f"{h}:{m:02d}"
    suffix = "am" if h < 12 else "pm"
    h12 = h % 12 or 12
    return f"{h12}{suffix}" if m == 0 else f"{h12}:{m:02d}{suffix}"


def _sample_window(rng: random.Random) -> tuple[str, str]:
    start_h = rng.randint(5, 19)
    start_m = rng.choice([0, 0, 0, 30])
    dur_h = rng.choice([1, 1, 2, 3, 4])
    end_h = min(start_h + dur_h, 23)
    return f"{start_h:02d}:{start_m:02d}", f"{end_h:02d}:{start_m:02d}"


def _sample_date(rng: random.Random, camera: bool = False) -> tuple[str, str]:
    """Return (phrase used in the question, canonical ISO date)."""
    offset = rng.choice([0, 0, 1, 1, 2, 3, 4, 5, 6])
    d = TODAY - timedelta(days=offset)
    iso = d.isoformat()
    choices = []
    if offset == 0:
        choices += ["today", "today", "this morning"] if not camera else ["today", "today"]
    if offset == 1:
        choices += ["yesterday", "yesterday"] + (["last night"] if camera else [])
    choices += [f"on {iso}", f"on {d.strftime('%B')} {d.day}", f"on {d.day} {d.strftime('%B %Y')}"]
    return rng.choice(choices), iso


def _make_record(tool, family, template, kind, split, rng, idx) -> Record:
    barns = TRAIN_BARNS if split == "train" else BARNS
    cams = TRAIN_CAMERAS if split == "train" else list(CAMERAS)
    barn = rng.choice(barns)
    other_barn = rng.choice([b for b in BARNS if b != barn])
    cam = rng.choice(cams)
    fill = {"barn": barn, "other_barn": other_barn, "cam": cam}

    if kind == "window":
        start, end = _sample_window(rng)
        fill |= {"start": _fmt_time(start, rng), "end": _fmt_time(end, rng)}
        args = {"barn_id": barn, "start_time": start, "end_time": end}
    elif kind == "period":
        period = rng.choice(list(DAY_PARTS))
        start, end = DAY_PARTS[period]
        fill |= {"period": period}
        args = {"barn_id": barn, "start_time": start, "end_time": end}
    elif kind == "date":
        phrase, iso = _sample_date(rng)
        fill |= {"date": phrase}
        args = {"barn_id": barn, "date": iso}
    elif kind == "cam":
        phrase, iso = _sample_date(rng, camera=True)
        fill |= {"date": phrase, "barn": CAMERAS[cam]}
        args = {"camera_id": cam, "date": iso}
    else:
        raise ValueError(kind)

    question = template.format(**fill)
    question = question[0].upper() + question[1:]
    return Record(id=f"{split}-{tool}-{idx:03d}", split=split, family=family,
                  question=question, expected_tool=tool, expected_args=json.dumps(args))


def generate_records(n_train=240, n_val=40, n_test=80, seed=42) -> dict[str, list[Record]]:
    rng = random.Random(seed)
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    out: dict[str, list[Record]] = {s: [] for s in sizes}
    seen_questions: set[str] = set()
    n_tools = len(TEMPLATES)
    for split, n in sizes.items():
        per_tool = n // n_tools
        for tool, templates in TEMPLATES.items():
            fams = [t for t, s in zip(templates, SPLIT_OF_TEMPLATE_INDEX) if s == split]
            i = tries = 0
            while i < per_tool:
                family, template, kind = fams[i % len(fams)]
                rec = _make_record(tool, family, template, kind, split, rng, i)
                tries += 1
                if rec.question in seen_questions and tries < 50 * per_tool:
                    continue  # avoid exact duplicates (across splits too)
                seen_questions.add(rec.question)
                out[split].append(rec)
                i += 1
        rng.shuffle(out[split])
    return out


def save_records(records: dict[str, list[Record]], data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, recs in records.items():
        with open(data_dir / f"{split}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(asdict(r)) + "\n")


def load_records(data_dir: str | Path) -> dict[str, list[Record]]:
    out = {}
    for split in ["train", "val", "test"]:
        with open(Path(data_dir) / f"{split}.jsonl") as f:
            out[split] = [Record(**json.loads(line)) for line in f]
    return out


def to_hf_dataset(records: list[Record]):
    from datasets import Dataset

    return Dataset.from_list([r.to_row() for r in records])
