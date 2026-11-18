"""Run GRPO trainings as background processes, so the lecture can continue while they train.

    from farm_grpo.background import start_runs, status, wait_for
    start_runs(["live", "hack"])     # Section 0: kick off both runs (returns immediately)
    status()                          # any time: step, held-out success, done / failed
    wait_for(["live"])                # Section 6: block (with progress prints) until finished
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .config import OUTPUT_DIR, PROJECT_ROOT

LOG_DIR = OUTPUT_DIR / "logs"

# name -> (output dir, CLI arguments for `python -m farm_grpo.train`)
RUNS = {
    "live": (OUTPUT_DIR / "live_run",
             ["--steps", "100", "--heldout-every", "25"]),
    "hack": (OUTPUT_DIR / "hack_many_args",
             ["--steps", "60", "--reward", "many_args", "--lr", "5e-5", "--heldout-every", "20"]),
}

_STEP_RE = re.compile(r"(\d+)/(\d+) \[")
_HELD_RE = re.compile(r"held-out @ step (\d+)\] success=([\d.]+)%")


def _log(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _running_pid(name: str) -> int | None:
    pid_file = LOG_DIR / f"{name}.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text())
        if _pid_alive(pid):
            return pid
    return None


def start_runs(names=("live", "hack")) -> None:
    """Launch each run as a separate process (a run that is already going is not restarted)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        if _running_pid(name):
            print(f"[{name}] already running (pid {_running_pid(name)})")
            continue
        out_dir, args = RUNS[name]
        cmd = [sys.executable, "-m", "farm_grpo.train", "--output", str(out_dir), *args]
        with open(_log(name), "w") as log:
            p = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "5"},
                                 start_new_session=True)
        (LOG_DIR / f"{name}.pid").write_text(str(p.pid))
        print(f"[{name}] started pid {p.pid}  ->  {out_dir.relative_to(PROJECT_ROOT)}  (log: {_log(name).relative_to(PROJECT_ROOT)})")


def run_status(name: str) -> dict:
    log = _log(name)
    if not log.exists():
        return {"run": name, "state": "not started"}
    text = log.read_text(errors="replace")
    steps = _STEP_RE.findall(text)
    held = _HELD_RE.findall(text)
    if "timing:" in text:
        state = "done"
    elif "Traceback" in text:
        state = "FAILED"
    elif _running_pid(name):
        state = "running"
    else:
        state = "stopped"
    step, total = (int(steps[-1][0]), int(steps[-1][1])) if steps else (0, None)
    return {"run": name, "state": state, "step": f"{step}/{total or '?'}",
            "held-out success %": [float(s) for _, s in held],
            "minutes since start": round((time.time() - (LOG_DIR / f"{name}.pid").stat().st_mtime) / 60, 1)}


def status(names=tuple(RUNS)):
    import pandas as pd

    return pd.DataFrame([run_status(n) for n in names]).set_index("run")


def wait_for(names=("live",), poll_s: int = 15) -> None:
    """Block until every run is done; print progress. Raises if a run failed."""
    last = None
    while True:
        st = [run_status(n) for n in names]
        line = "  |  ".join(f"{s['run']}: {s['state']} {s.get('step', '')} "
                            f"held-out={s.get('held-out success %', [])}" for s in st)
        if line != last:
            print(time.strftime("%H:%M:%S"), line, flush=True)
            last = line
        if any(s["state"] in ("FAILED", "stopped") for s in st):
            bad = [s["run"] for s in st if s["state"] in ("FAILED", "stopped")]
            raise RuntimeError(f"run(s) {bad} did not finish; see {LOG_DIR}/<run>.log")
        if all(s["state"] == "done" for s in st):
            return
        if any(s["state"] == "not started" for s in st):
            raise RuntimeError("run not started; call start_runs() first or use TRAINING_MODE='pretrained'")
        time.sleep(poll_s)
