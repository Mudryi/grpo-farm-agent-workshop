# Demo walkthrough: what to run, and when

Lecture machine: home server, RTX 3090 24 GB.
Project: `~/phd_projects/grpo-farm-agent-workshop`. All data is synthetic dummy data.

---

## A. The day before (≈ 20 min)

```bash
cd ~/phd_projects/grpo-farm-agent-workshop

# 1. Environment (only once). uv-managed Python is required: Triton needs Python.h
uv python install 3.12
uv venv --python-preference only-managed --python 3.12
uv sync

# 2. Sanity checks (≈ 5 s)
uv run pytest -q                        # expect: 10 passed

# 3. Checkpoints. Already in outputs/ (copied from data/checkpoints/). If outputs/ is ever wiped:
for r in live_run prepared_run hack_many_args; do cp -r data/checkpoints/$r outputs/; done

# 4. Download the model into the HF cache (no Wi-Fi surprises in class)
uv run python -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

# 5. Full dry run (≈ 15 min): proves the whole notebook works on this GPU
uv run jupyter nbconvert --to notebook --execute workshop.ipynb --output workshop_executed.ipynb
```

Keep `workshop_executed.ipynb` open in a second tab as a **fallback**. If anything breaks live, you show its outputs.

## B. Start of the lecture

```bash
cd ~/phd_projects/grpo-farm-agent-workshop
uv run jupyter lab --no-browser --port 8888
# from your laptop, if the server is remote:
#   ssh -p 2324 -L 8888:localhost:8888 mudryi@<server>   then open http://localhost:8888
```

Open `workshop.ipynb`. Run **Section 0** (setup) immediately; it prints the GPU and config.
Check the switches in the second cell:

| Switch | Default | Set to `False` when… |
|---|---|---|
| `RUN_LIVE_TRAINING` | `True` | you are short on time. Section 6 then loads `outputs/live_run` |
| `RUN_HACK_TRAINING` | `True` | short on time. Section 8 then loads `outputs/hack_many_args` |
| `HACK_STEPS` | 60 | (60 steps ≈ 4 min) |

⚠️ Running live training **overwrites** `outputs/live_run` / `outputs/hack_many_args`. The originals stay in `data/checkpoints/`.

## C. Section by section

| Time | Section | What you do | What to show / say | Runtime |
|---|---|---|---|---|
| 0–20 | **1. Why RL** | Talk through the 4 questions + vocabulary table | RL ≠ automatically better than SFT | — |
| 20–35 | **1.3 GRPO** | Run the advantage cell | 3 groups: mixed → signal; all-correct / all-wrong → **A = 0, no learning** | 1 s |
| 35–55 | **2. Environment** | Run all cells | Tools, robot table (Barn A 07:00 planned → **07:18** actual), invalid call returns an error instead of crashing, the scripted 2-call "was the robot late?" oracle | 1 s |
| 55–80 | **3. Dataset** | Run all cells | Template-family table: 5 train / 1 val / 2 test per tool; barn D test-only; leakage asserts ✔ | 1 s |
| — | **4. Baseline** | Run all cells | Base model ≈ **74–78 %** task success. Failure table: day-part questions ("during the morning") are the weak spot | ~4 s |
| — | **4.1 Signal check** | Run | ~36 % of groups have zero reward std, so the rest give GRPO a signal. **Ask:** what if it were 100 %? | ~10 s |
| 80–105 | **5. Rewards** | Students fill the TODO cell, then run `check_rewards` | 8 PASS lines. If students aren't done, the next cell falls back to `solutions/` automatically | 1 s |
| 105–120 | **6.1 Benchmark** | Run | Measured s/step → `LIVE_STEPS` chosen automatically (≤ 100) | ~30 s |
| | **6.2 Live run** | Run, then **talk while it trains**: logs show `reward`, `reward_std`, `frac_reward_zero_std`, and held-out success every 25 steps | | **~4–5 min** |
| | **6.3 Prepared run** | Run | Curves of the 500-step run: validation plateaus at ~92 % | 1 s |
| 120–150 | **7. Evaluation** | Run all cells | Before / live / prepared table, error breakdown, per-family table, fixed vs broken prompts | ~15 s |
| | **7 key story** | Point at the per-family table | **500 steps < 100 steps.** `robot_distract2` → 0 %: the long run picks the schedule tool whenever "schedule" appears. Validation didn't catch it, so the test split matters | — |
| 150–165 | **8.1 Reward hacking** | Run (or skip training with `RUN_HACK_TRAINING=False`) | Training reward climbs to **1.0** while task success → **0 %**, and outputs contain extra keys and multiple calls | ~4 min |
| 165–180 | **8.2 Multi-step agent** | Run | Base and trained models both **fail** the 2-tool investigation (they never call the schedule). Single-step training doesn't transfer. Discuss outcome rewards / TRL `environment_factory` | ~30 s |

## D. Expected numbers (test set, 80 prompts)

| Metric | Before | Live (100 steps) | Prepared (500 steps) | Hack (`many_args`) |
|---|---|---|---|---|
| Task success % | 74–78 | **87.5–97.5** | 87.5 | 0 |
| Correct tool % | 91–93 | 94–100 | 87.5 | 0–89 |
| Unnecessary calls % | 0 | 0 | 0 | 7–69 |

**The live run is retrained in class, so its result varies:** 97.5 % on the DGX Spark, 87.5 % on the 3090 dry run.
The story holds either way (clear gain over the base model, validation ~92 %). If you want to show the best
checkpoint (97.5 %), set `RUN_LIVE_TRAINING = False` after copying it back:
`cp -r data/checkpoints/live_run/. outputs/live_run/`.
The hack run always ends at **0 %** task success. How it hacks varies (extra argument keys and/or several calls).

## E. If something goes wrong

| Symptom | Fix |
|---|---|
| `CalledProcessError … gcc … driver.c` | venv is on system Python. Rerun step A.1 (uv-managed Python) |
| CUDA OOM | Restart kernel. Other GPU jobs? `nvidia-smi`. Lower `cfg.per_device_train_batch_size` to 8 (keep it divisible by 4) |
| Training too slow | `LIVE_BUDGET_MIN = 4` in Section 6.1, or `RUN_LIVE_TRAINING = False` |
| Students' rewards fail tests | Section 5 falls back to `solutions/` (instructor copy only) |
| Kernel dies mid-lecture | Show `workshop_executed.ipynb`; set both `RUN_*` switches to `False` and rerun. Everything loads from `outputs/` |

## F. Student handout

```bash
uv run python scripts/make_student_copy.py   # -> dist/student/ (no solutions, no checkpoints, no outputs)
```
Students need the same setup (A.1). In the student copy, Section 5 **requires** their rewards to pass.
