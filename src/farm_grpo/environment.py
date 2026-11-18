"""A small, fully synthetic dairy-farm world with four read-only tools.

NOTHING here is real farm data. All barns, cameras, schedules and events are
generated deterministically from a seed so that every student sees the same
world and the labels in the dataset are reproducible.

Tools exposed to the agent
--------------------------
get_robot_events(barn_id, start_time, end_time)   feed-pushing robot visits *today*
get_mixer_events(barn_id, date)                   feed-mixer wagon arrivals
get_feeding_schedule(barn_id, date)               planned feed-push times
get_camera_status(camera_id, date)                camera online/offline windows
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Fixed "now" of the simulation (dummy date, so relative dates are deterministic)
# --------------------------------------------------------------------------------------
TODAY = date(2026, 3, 10)  # a Tuesday
N_DAYS = 7                 # world history: TODAY-6 ... TODAY

BARNS = ["A", "B", "C", "D"]
# two cameras per barn
CAMERAS = {1: "A", 2: "A", 3: "B", 4: "B", 5: "C", 6: "C", 7: "D", 8: "D"}

# Day-part conventions (also written into the system prompt)
DAY_PARTS = {
    "morning": ("06:00", "12:00"),
    "afternoon": ("12:00", "18:00"),
    "evening": ("18:00", "22:00"),
}

# Planned feed-push times per barn (same every day in this toy world)
BASE_SCHEDULE = {
    "A": ["07:00", "10:00", "13:00", "16:00", "19:00"],
    "B": ["06:30", "09:30", "12:30", "15:30", "18:30"],
    "C": ["07:15", "11:15", "15:15", "19:15"],
    "D": ["08:00", "12:00", "16:00", "20:00"],
}
MIXER_SCHEDULE = {"A": ["06:15", "14:15"], "B": ["05:45", "13:45"],
                  "C": ["06:45", "14:45"], "D": ["07:30", "15:30"]}


# --------------------------------------------------------------------------------------
# Tool specifications (JSON-schema-like, used by the prompt and the validator)
# --------------------------------------------------------------------------------------
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "get_robot_events": {
        "description": "Feed-pushing robot visits in a barn TODAY within a time window.",
        "arguments": {
            "barn_id": "Barn letter, one of A, B, C, D",
            "start_time": "HH:MM (24h)",
            "end_time": "HH:MM (24h)",
        },
    },
    "get_mixer_events": {
        "description": "Arrival times of the feed-mixer wagon in a barn on a date.",
        "arguments": {
            "barn_id": "Barn letter, one of A, B, C, D",
            "date": "YYYY-MM-DD",
        },
    },
    "get_feeding_schedule": {
        "description": "PLANNED feed-push times for a barn on a date.",
        "arguments": {
            "barn_id": "Barn letter, one of A, B, C, D",
            "date": "YYYY-MM-DD",
        },
    },
    "get_camera_status": {
        "description": "Online/offline status and outage windows of a camera on a date.",
        "arguments": {
            "camera_id": "Camera number, integer 1-8",
            "date": "YYYY-MM-DD",
        },
    },
}
TOOL_NAMES = list(TOOL_SPECS)


# --------------------------------------------------------------------------------------
# Argument normalisation (so "7:00" == "07:00", "today" == "2026-03-10", "barn a" == "A")
# --------------------------------------------------------------------------------------
_TIME_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", re.I)


def normalize_time(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    m = _TIME_RE.match(value)
    if not m:
        return None
    h, mnt, ampm = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
    if ampm == "pm" and h < 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    if not (0 <= h <= 24 and 0 <= mnt < 60):
        return None
    return f"{h:02d}:{mnt:02d}"


def normalize_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v == "today":
        return TODAY.isoformat()
    if v == "yesterday":
        return (TODAY - timedelta(days=1)).isoformat()
    try:
        return datetime.strptime(v, "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def normalize_barn(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().upper().removeprefix("BARN").strip()
    return v if v in BARNS else None


def normalize_camera(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in CAMERAS else None
    if isinstance(value, str):
        v = value.strip().lower().removeprefix("camera").removeprefix("cam").strip(" #")
        if v.isdigit() and int(v) in CAMERAS:
            return int(v)
    return None


_NORMALIZERS = {
    "barn_id": normalize_barn,
    "start_time": normalize_time,
    "end_time": normalize_time,
    "date": normalize_date,
    "camera_id": normalize_camera,
}


def normalize_value(key: str, value: Any):
    """Canonical form of one argument value (None if unknown key / invalid value)."""
    fn = _NORMALIZERS.get(key)
    return fn(value) if fn else None


def normalize_args(tool: str, args: dict) -> dict | None:
    """Return canonical arguments, or None if names/values are invalid for `tool`."""
    if tool not in TOOL_SPECS or not isinstance(args, dict):
        return None
    expected = set(TOOL_SPECS[tool]["arguments"])
    if set(args) != expected:
        return None
    out = {}
    for k, v in args.items():
        nv = _NORMALIZERS[k](v)
        if nv is None:
            return None
        out[k] = nv
    return out


# --------------------------------------------------------------------------------------
# The simulated world
# --------------------------------------------------------------------------------------
def _add_minutes(hhmm: str, minutes: int) -> str:
    t = datetime.strptime(hhmm, "%H:%M") + timedelta(minutes=minutes)
    return t.strftime("%H:%M")


@dataclass
class FarmWorld:
    schedules: dict      # (barn) -> {date: [HH:MM]}
    robot_events: dict   # (barn) -> [{"time", "scheduled", "delay_min"}]  (today only)
    mixer_events: dict   # (barn) -> {date: [HH:MM]}
    camera_status: dict  # (cam)  -> {date: {"status", "outages": [[start, end]]}}

    # ---- tools -------------------------------------------------------------------
    def get_robot_events(self, barn_id: str, start_time: str, end_time: str) -> dict:
        evs = [e for e in self.robot_events[barn_id] if start_time <= e["time"] < end_time]
        return {"barn_id": barn_id, "date": TODAY.isoformat(), "window": [start_time, end_time],
                "events": [{"time": e["time"], "type": "feed_push"} for e in evs]}

    def get_mixer_events(self, barn_id: str, date: str) -> dict:
        return {"barn_id": barn_id, "date": date,
                "arrivals": self.mixer_events[barn_id].get(date, [])}

    def get_feeding_schedule(self, barn_id: str, date: str) -> dict:
        return {"barn_id": barn_id, "date": date,
                "planned_feed_push": self.schedules[barn_id].get(date, [])}

    def get_camera_status(self, camera_id: int, date: str) -> dict:
        st = self.camera_status[str(camera_id)].get(date, {"status": "unknown", "outages": []})
        return {"camera_id": camera_id, "barn_id": CAMERAS[camera_id], "date": date, **st}

    def execute(self, tool: str, args: dict) -> dict:
        """Validate + normalise + run a tool call. Never raises; returns {'error': ...}."""
        if tool not in TOOL_SPECS:
            return {"error": f"unknown tool '{tool}'"}
        norm = normalize_args(tool, args)
        if norm is None:
            return {"error": f"invalid arguments for {tool}: {args}"}
        return getattr(self, tool)(**norm)

    # ---- io -------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {"today": TODAY.isoformat(), "schedules": self.schedules,
                "robot_events": self.robot_events, "mixer_events": self.mixer_events,
                "camera_status": self.camera_status}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))


def build_world(seed: int = 7) -> FarmWorld:
    rng = random.Random(seed)
    dates = [(TODAY - timedelta(days=i)).isoformat() for i in range(N_DAYS)][::-1]

    schedules = {b: {d: list(BASE_SCHEDULE[b]) for d in dates} for b in BARNS}

    # Robot events for TODAY: mostly on time, some late, one missed
    robot_events: dict[str, list] = {}
    for b in BARNS:
        evs = []
        for slot in BASE_SCHEDULE[b]:
            r = rng.random()
            if r < 0.1:
                continue  # missed push
            delay = rng.randint(15, 30) if r < 0.3 else rng.randint(0, 4)
            evs.append({"time": _add_minutes(slot, delay), "scheduled": slot, "delay_min": delay})
        robot_events[b] = evs
    # The workshop's headline example: Barn A, 07:00 planned, robot came 07:18
    robot_events["A"][0] = {"time": "07:18", "scheduled": "07:00", "delay_min": 18}

    mixer_events = {
        b: {d: [_add_minutes(t, rng.randint(-5, 20)) for t in MIXER_SCHEDULE[b]] for d in dates}
        for b in BARNS
    }

    camera_status: dict[str, dict] = {}
    for cam in CAMERAS:
        per_day = {}
        for d in dates:
            if rng.random() < 0.2:
                start = f"{rng.randint(0, 22):02d}:{rng.choice(['00', '10', '25', '40'])}"
                per_day[d] = {"status": "degraded",
                              "outages": [[start, _add_minutes(start, rng.randint(20, 90))]]}
            else:
                per_day[d] = {"status": "online", "outages": []}
        camera_status[str(cam)] = per_day
    # Headline example for the camera tool: camera 3 went down last night
    camera_status["3"][(TODAY - timedelta(days=1)).isoformat()] = {
        "status": "degraded", "outages": [["22:10", "23:40"]]}

    return FarmWorld(schedules, robot_events, mixer_events, camera_status)


def tool_spec_text() -> str:
    """Human/LLM-readable description of the tools (used in the system prompt)."""
    lines = []
    for name, spec in TOOL_SPECS.items():
        args = ", ".join(spec["arguments"])
        lines.append(f"- {name}({args}): {spec['description']}")
        for a, desc in spec["arguments"].items():
            lines.append(f"    * {a}: {desc}")
    return "\n".join(lines)
