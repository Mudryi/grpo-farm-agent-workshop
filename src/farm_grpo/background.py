"""Run the two GRPO trainings as background processes, so the lecture continues while they train.

    from farm_grpo.background import start_runs, status, wait_for
    start_runs(["live", "hack"])     # Section 0: kick off both runs (returns immediately)
    status()                          # any time: step, held-out success, done / failed
    wait_for(["live"])                # Section 6: block (with progress prints) until finished

Each run is a separate process: `python -m farm_grpo.background <name>`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

from .config import OUTPUT_DIR, PROJECT_ROOT

LOG_DIR = OUTPUT_DIR / "logs"

# name -> output sub-directory and training settings (the only difference between the runs is the reward)
RUNS = {
    "live": dict(out="live_run", reward="good", steps=100, heldout_every=25),
    "hack": dict(out="hack_many_args", reward="many_args", steps=60, heldout_every=20, learning_rate=5e-5),
}
DONE = "TRAINING DONE"

_STEP_RE = re.compile(r"(\d+)/(\d+) \[")
_HELD_RE = re.compile(r"held-out @ step (\d+)\] success=([\d.]+)%")


def _running_pid(name: str) -> int | None:
    pid_file = LOG_DIR / f"{name}.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text())
        try:
            os.kill(pid, 0)
            return pid
        except OSError:
            pass
    return None


def start_runs(names=("live", "hack")) -> None:
    """Launch each run as a separate process (a run that is already going is not restarted)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        if pid := _running_pid(name):
            print(f"[{name}] already running (pid {pid})")
            continue
        with open(LOG_DIR / f"{name}.log", "w") as log:
            p = subprocess.Popen([sys.executable, "-m", "farm_grpo.background", name], cwd=PROJECT_ROOT,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "5"})
        (LOG_DIR / f"{name}.pid").write_text(str(p.pid))
        print(f"[{name}] started pid {p.pid} -> outputs/{RUNS[name]['out']}  (log: outputs/logs/{name}.log)")


def run_status(name: str) -> dict:
    log, pid = LOG_DIR / f"{name}.log", LOG_DIR / f"{name}.pid"
    if not log.exists():
        return {"run": name, "state": "not started"}
    text = log.read_text(errors="replace")
    if DONE in text:
        state = "done"
    elif "Traceback" in text:
        state = "FAILED"
    else:
        state = "running" if _running_pid(name) else "stopped"
    steps = _STEP_RE.findall(text)
    return {"run": name, "state": state,
            "step": f"{steps[-1][0]}/{steps[-1][1]}" if steps else "0/?",
            "held-out success %": [float(s) for _, s in _HELD_RE.findall(text)],
            "minutes since start": round((time.time() - pid.stat().st_mtime) / 60, 1) if pid.exists() else None}


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
        if bad := [s["run"] for s in st if s["state"] in ("FAILED", "stopped", "not started")]:
            raise RuntimeError(f"run(s) {bad} did not finish; see {LOG_DIR}/<run>.log (call start_runs() first)")
        if all(s["state"] == "done" for s in st):
            return
        time.sleep(poll_s)


def _train(name: str) -> None:
    from .config import get_config
    from .dataset import generate_records, save_records, to_hf_dataset
    from .rewards import REWARD_FUNCS
    from .rewards_bad import many_args_reward
    from .train import train_grpo

    run = dict(RUNS[name])
    cfg = get_config()
    records = generate_records(cfg.n_train, cfg.n_val, cfg.n_test, cfg.seed)
    if not (cfg.data_dir / "train.jsonl").exists():  # don't rewrite while the notebook reads it
        save_records(records, cfg.data_dir)
    reward_funcs = REWARD_FUNCS if run.pop("reward") == "good" else [many_args_reward]
    train_grpo(cfg, to_hf_dataset(records["train"]), reward_funcs, cfg.output_dir / run.pop("out"),
               max_steps=run.pop("steps"), heldout_records=records["val"], heldout_every=run.pop("heldout_every"),
               **run)
    print(DONE)


if __name__ == "__main__":
    _train(sys.argv[1])
