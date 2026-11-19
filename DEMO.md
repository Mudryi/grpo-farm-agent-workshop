# Demo walkthrough: 90–100 minute version

Lecture machine: home server, RTX 3090 24 GB. Project: `~/phd_projects/grpo-farm-agent-workshop`.
All data is synthetic dummy data.

**The trick that makes 95 min work:** the first cell after setup starts **both GRPO trainings in the background**
(live run + reward-hacking run, in parallel). They finish in **~6 min**, while you give the RL / GRPO talk.
When you reach Section 6, the results are already there. No waiting in front of the class.

---

## A. The day before (≈ 15 min)

```bash
cd ~/phd_projects/grpo-farm-agent-workshop

# environment (once). uv-managed Python is required: Triton needs Python.h
uv python install 3.12
uv venv --python-preference only-managed --python 3.12
uv sync

uv run pytest -q                         # expect: 10 passed

# model into the HF cache (no download surprises in class)
uv run python -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

# full dry run exactly as in class (background mode): ~6 min
uv run jupyter nbconvert --to notebook --execute workshop.ipynb --output workshop_executed.ipynb
```

Open `workshop_executed.ipynb` in a second tab as your **fallback slides**.

## B. Right before the lecture

```bash
nvidia-smi                               # GPU free? (nothing else training)
cd ~/phd_projects/grpo-farm-agent-workshop
uv run jupyter lab --no-browser --port 8888
# from your laptop:  ssh -p 2324 -L 8888:localhost:8888 mudryi@146.120.190.241  -> http://localhost:8888
```

Open `workshop.ipynb` (English) or `workshop_uk.ipynb` (Ukrainian, see section G). The only switch is in the **2nd code cell**:

| `TRAINING_MODE` | What happens | Use when |
|---|---|---|
| `"background"` **(default)** | Section 0 starts both runs as separate processes; Section 6/8 cells wait if needed | normal lecture |
| `"inline"` | trains inside the Section 6 and 8 cells (~4 min each, blocks) | you want students' own rewards trained, or other `HACK` rewards |
| `"pretrained"` | no training; loads `data/checkpoints/` (best run: 97.5 %) | GPU problems / very short on time (whole notebook ≈ 1 min) |

## C. Timeline (95 min + 5 buffer)

| Min | Section | Do | Say / show |
|---|---|---|---|
| **0–5** | **0 Setup** | Run the setup cells. **The 3rd cell starts training.** You see `[live] started …` `[hack] started …` | "The GPU is now training two models while we talk. We'll check on them." |
| 5–25 | **1 Why RL / GRPO** | Talk 1.1–1.2. Run the advantage cell (1.3) | Mixed group → signal; all-correct / all-wrong → A = 0, **no learning** |
| | **1.4 Bigger picture** *(new)* | Talk: SFT vs RLHF-PPO vs DPO vs GRPO table, RLVR, DeepSeek-R1, TRL knobs (`beta`, `loss_type`, `scale_rewards`), cost (vLLM), agentic RL. Run `status()` | Training is visible: step 40/100, held-out % rising. Ask the room the 2 questions at the end |
| 25–33 | **2 Environment** | Run all | Robot table: Barn A **07:00 planned → 07:18 actual**; invalid call returns an error; scripted 2-call oracle |
| 33–40 | **3 Dataset** | Run all | Template-family split (5 train / 1 val / 2 test), barn D test-only, leakage asserts ✔. *By now both runs are finished.* |
| 40–47 | **4 Baseline + signal** | Run all | Base ≈ **74 %** task success; weak spot = day-part questions. Signal check: ~40 % zero-std groups → the rest give GRPO a signal |
| 47–62 | **5 Rewards** | Walk through the implemented reward functions line by line, run `check_rewards` → 8 PASS | Ask the room *before* showing each function: what should this case score? Then the sanity-check cell scores real base-model samples |
| 62–70 | **6 Results** | Run all (no waiting) | Training curves + printed summary: reward ↑, `frac_reward_zero_std` ↑ (signal used up), held-out 87.5 → ~92.5 %. The note under the plot says what to read |
| 70–82 | **7 Evaluation** | Run all | Table: base **74 %** → after GRPO **~90 %** (varies 87–97). Gain is in *arguments*, not format. Error/family tables: which phrasings still fail |
| 82–90 | **8.1 Reward hacking** | Run (already trained) | Training reward climbs to **1.0** while task success → **0 %**. Show the completions with stuffed arguments |
| 90–95 | **8.2 Multi-step agent** | Run (~30 s) | Both models **fail** the 2-tool "was the robot late?" question. Single-step training doesn't transfer, which motivates outcome rewards / TRL `environment_factory` |
| 95–100 | **9 Metrics** + wrap-up | Run the summary cell, last markdown cell | All metrics in one place (saved to `outputs/`). 4 take-aways. Buffer for questions |

**If you run late**, cut in this order: 8.2 (say it in one sentence) → 3 (show only the family table) → 7.1 fixed/broken tables.

## D. Measured on the 3090 (2026-11-19)

| | Value |
|---|---|
| Both runs in parallel (background) | live 100 steps + hack 60 steps, **done 5.7 min after start** |
| Whole notebook, background mode | **6.2 min** |
| Whole notebook, pretrained mode | **49 s** |

| Test (80 prompts) | Before | After GRPO (retrained runs) | After GRPO (saved best) | Hack |
|---|---|---|---|---|
| Task success %, English | 73.8 | 87.5–92.5 | 97.5 | 0 |
| Task success %, Ukrainian | 37.5 | 60.0 | 60.0 | 0 |

The live run is retrained in class, so expect **87–97 %** (English). The story holds either way.
The 500-step "prepared" run is no longer part of the notebooks (it was confusing); `python -m farm_grpo.train --steps 500` still works for your own experiments.

## E. If something goes wrong

| Symptom | Fix |
|---|---|
| `status()` shows `FAILED` | Read `outputs/logs/live.log`. Switch to `TRAINING_MODE = "pretrained"`, rerun from the top |
| `CalledProcessError … gcc … driver.c` | venv on system Python. Redo step A (uv-managed Python) |
| CUDA OOM | `nvidia-smi`: kill other jobs. Restart kernel |
| You restart the kernel while runs train | Fine: `start_runs` does not restart a run that is still going; `status()` reconnects via the pid files |
| Kernel dies / anything else | Show `workshop_executed.ipynb`, or `"pretrained"` mode (≈ 1 min end-to-end) |

Restore the best saved checkpoints into `outputs/` any time:
`for r in live_run hack_many_args; do cp -r data/checkpoints/$r/. outputs/$r/; done`

## F. Student handout

```bash
uv run python scripts/make_student_copy.py   # -> dist/student/ (both notebooks, no checkpoints, no outputs)
```
The notebooks are student-ready: no TODO cells and no timings; the reward functions are implemented in Section 5.
Students without a GPU just follow your screen. With a GPU, they use `"inline"` mode.

## G. Ukrainian version (`workshop_uk.ipynb`)

Same sections, same timeline, same **English** text. Only the dataset is Ukrainian. `LANG = "uk"` in the first code cell:
* the dataset **requests** are Ukrainian (same template families and split; tool names, system prompt and labels stay English/ISO);
* the background runs train on Ukrainian requests and write to `outputs/uk/` (logs: `outputs/uk/logs/`);
* all metrics are saved separately: `outputs/uk/results.json`, `results_by_family.csv`, `agent_results.json`. Summary in `RESULTS_UK.md`.

Day before:
```bash
uv run jupyter nbconvert --to notebook --execute workshop_uk.ipynb --output workshop_uk_executed.ipynb   # ~6 min
mkdir -p data/uk/checkpoints && for r in live_run hack_many_args; do cp -r outputs/uk/$r data/uk/checkpoints/; done
```
The last line enables `TRAINING_MODE = "pretrained"` for the Ukrainian notebook.
Do not start the English and Ukrainian background runs at the same time (4 processes on one GPU roughly doubles the time).
