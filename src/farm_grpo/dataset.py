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
# User requests are Ukrainian; tool names, argument keys and labels stay canonical (English / ISO).
TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "get_robot_events": [
        ("robot_show", "Покажи проходи робота в корівнику {barn} з {start} до {end}.", "window"),
        ("robot_find", "Знайди події робота в корівнику {barn} між {start} та {end}.", "window"),
        ("robot_when", "Коли робот-підштовхувач корму проходив корівником {barn} між {start} і {end}?", "window"),
        ("robot_period", "Чи підштовхував робот корм у корівнику {barn} {period}?", "period"),
        ("robot_distract", "Забудь про міксер, мені потрібен журнал робота для корівника {barn} з {start} до {end}.", "window"),
        # --- validation family
        ("robot_terse", "активність робота корівник {barn} {start}-{end}", "window"),
        # --- test-only families
        ("robot_count", "Скільки разів робот-підштовхувач працював у корівнику {barn} {period}?", "period"),
        ("robot_distract2", "Графік для корівника {other_barn} в нормі. Які проходи з підштовхування корму зробив робот у корівнику {barn} між {start} і {end}?", "window"),
    ],
    "get_mixer_events": [
        ("mixer_when", "Коли міксер приїхав у корівник {barn} {date}?", "date"),
        ("mixer_show", "Покажи події міксера для корівника {barn} {date}.", "date"),
        ("mixer_deliv", "Дай мені доставки кормоміксера в корівник {barn} {date}.", "date"),
        ("mixer_visit", "Чи заїжджав кормозмішувач у корівник {barn} {date}?", "date"),
        ("mixer_distract", "Поки що не зважай на робота. О котрій міксер був у корівнику {barn} {date}?", "date"),
        ("mixer_terse", "журнал міксера корівник {barn} {date}", "date"),
        ("mixer_fresh", "О котрій міксер привіз свіжий корм у корівник {barn} {date}?", "date"),
        ("mixer_distract2", "Камера {cam} працює нормально. Мене цікавлять приїзди TMR-міксера в корівник {barn} {date}.", "date"),
    ],
    "get_feeding_schedule": [
        ("sched_what", "Який запланований час годування в корівнику {barn} {date}?", "date"),
        ("sched_plan", "Покажи план годування для корівника {barn} {date}.", "date"),
        ("sched_supposed", "Коли має підштовхуватися корм у корівнику {barn} {date}?", "date"),
        ("sched_says", "Що каже графік годування для корівника {barn} {date}?", "date"),
        ("sched_distract", "Не фактичні події робота, а лише заплановані часи годування для корівника {barn} {date}.", "date"),
        ("sched_terse", "розклад годування корівник {barn} {date}", "date"),
        ("sched_cows", "Згідно з планом, коли мають годувати корів у корівнику {barn} {date}?", "date"),
        ("sched_distract2", "Забудь про міксер у корівнику {other_barn}. Які очікувані часи годування для корівника {barn} {date}?", "date"),
    ],
    "get_camera_status": [
        ("cam_working", "Чи працювала камера {cam} {date}?", "cam"),
        ("cam_online", "Камера {cam} онлайн {date}?", "cam"),
        ("cam_check", "Перевір стан камери {cam} {date}.", "cam"),
        ("cam_outage", "Чи були збої в камери №{cam} {date}?", "cam"),
        ("cam_distract", "Дані робота в порядку. Я лише хочу знати, чи записувала камера {cam} {date}.", "cam"),
        ("cam_terse", "камера {cam} стан {date}", "cam"),
        ("cam_video", "Чи є в нас відео з камери номер {cam} {date}?", "cam"),
        ("cam_distract2", "Чи був простій камери {cam} у корівнику {barn} {date}?", "cam"),
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
    style = rng.choice(["24h", "24h", "nopad", "dot", "spoken"])
    if style == "24h":
        return hhmm
    if style == "nopad":
        return f"{h}:{m:02d}"
    if style == "dot":
        return f"{h}.{m:02d}"
    part = "ранку" if h < 12 else "дня" if h < 17 else "вечора"  # "7 ранку", "3 дня", "8 вечора"
    h12 = h % 12 or 12
    return f"{h12} {part}" if m == 0 else f"{h12}:{m:02d} {part}"


def _sample_window(rng: random.Random) -> tuple[str, str]:
    start_h = rng.randint(5, 19)
    start_m = rng.choice([0, 0, 0, 30])
    dur_h = rng.choice([1, 1, 2, 3, 4])
    end_h = min(start_h + dur_h, 23)
    return f"{start_h:02d}:{start_m:02d}", f"{end_h:02d}:{start_m:02d}"


UK_MONTHS_GEN = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня",
                 "серпня", "вересня", "жовтня", "листопада", "грудня"]
# Ukrainian day-part phrases -> DAY_PARTS key
UK_PERIODS = {"morning": ["сьогодні вранці", "сьогодні зранку"],
              "afternoon": ["сьогодні вдень", "сьогодні після обіду"],
              "evening": ["сьогодні ввечері"]}


def _sample_date(rng: random.Random, camera: bool = False) -> tuple[str, str]:
    """Return (phrase used in the question, canonical ISO date)."""
    offset = rng.choice([0, 0, 1, 1, 2, 3, 4, 5, 6])
    d = TODAY - timedelta(days=offset)
    iso = d.isoformat()
    choices = []
    if offset == 0:
        choices += ["сьогодні", "сьогодні", "сьогодні зранку"] if not camera else ["сьогодні", "сьогодні"]
    if offset == 1:
        choices += ["вчора", "вчора"] + (["минулої ночі"] if camera else [])
    month = UK_MONTHS_GEN[d.month - 1]
    choices += [iso, f"{d.day} {month}", f"{d.day} {month} {d.year} р.", d.strftime("%d.%m.%Y")]
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
        fill |= {"period": rng.choice(UK_PERIODS[period])}
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

    question = template.format(**fill).replace("р..", "р.")  # "2026 р." at the end of a sentence
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


def to_hf_dataset(records: list[Record]):
    from datasets import Dataset

    return Dataset.from_list([r.to_row() for r in records])
