# Demo walkthrough: 90–100 minute version

Lecture machine: home server, RTX 3090 24 GB. Project: `~/phd_projects/grpo-farm-agent-workshop`.
Notebook: `workshop_uk.ipynb` (English text, Ukrainian requests). All data is synthetic dummy data.

**The trick that makes 95 min work:** the third code cell starts **both GRPO trainings in the background**
(`live` + reward-hacking `hack`, in parallel). They finish in **~6 min**, while you give the RL / GRPO talk.
When you reach Section 6, the results are already there.

---

## A. The day before (≈ 15 min)

```bash
cd ~/phd_projects/grpo-farm-agent-workshop

# environment (once). uv-managed Python is required: Triton needs Python.h
uv python install 3.12
uv venv --python-preference only-managed --python 3.12
uv sync

uv run pytest -q                         # expect: all passed

# model into the HF cache (no download surprises in class)
uv run python -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

# full dry run exactly as in class: ~6.5 min. Use a copy, so the committed notebook keeps its outputs
uv run jupyter nbconvert --to notebook --execute workshop_uk.ipynb --output /tmp/workshop_dryrun.ipynb
```

The committed `workshop_uk.ipynb` already contains outputs from a full run: open it in a second tab as your **fallback slides**.

## B. Right before the lecture

```bash
nvidia-smi                               # GPU free? (nothing else training)
cd ~/phd_projects/grpo-farm-agent-workshop
uv run jupyter lab --no-browser --port 8888
# from your laptop:  ssh -p 2324 -L 8888:localhost:8888 mudryi@146.120.190.241  -> http://localhost:8888
```

Open `workshop_uk.ipynb`. **Run the third code cell early.** It starts the training, and you will see
`[live] started …` and `[hack] started …`.

## C. Timeline (95 min + 5 buffer)

| Min | Section | Do | Say / show |
|---|---|---|---|
| **0–5** | **0 Setup** | Run the setup cells. **The 3rd cell starts training.** You see `[live] started …` `[hack] started …` | "The GPU is now training two models while we talk. We'll check on them." |
| 5–25 | **1 Why RL / GRPO** | Talk 1.1–1.2. Run the advantage cell (1.3) | Mixed group → signal; all-correct / all-wrong → A = 0, **no learning** |
| | **1.4 Bigger picture** | Talk: SFT vs RLHF-PPO vs DPO vs GRPO table, RLVR, DeepSeek-R1, TRL knobs (`beta`, `loss_type`, `scale_rewards`), cost (vLLM), agentic RL. Run `status()` | Training is visible: step 40/100, held-out % rising. Ask the room the 2 questions at the end |
| 25–33 | **2 Environment** | Run all | Robot table: Barn A **07:00 planned → 07:18 actual**; invalid call returns an error; scripted 2-call oracle |
| 33–40 | **3 Dataset** | Run all | Template-family split (5 train / 1 val / 2 test), barn D test-only, leakage asserts ✔. *By now both runs are finished.* |
| 40–47 | **4 Baseline + signal** | Run all | Base ≈ **37.5 %** task success; weak spots = Ukrainian dates, times and day parts. Signal check: ~25–35 % zero-std groups → the rest give GRPO a signal |
| 47–62 | **5 Rewards** | Walk through the implemented reward functions line by line, run `check_rewards` → 8 PASS | Ask the room *before* showing each function: what should this case score? Then the sanity-check cell scores real base-model samples |
| 62–70 | **6 Results** | Run all (no waiting) | Training curves + printed summary: reward ↑, `frac_reward_zero_std` ↑ (signal used up), held-out 57.5 → ~90 %. The note under the plot says what to read |
| 70–82 | **7 Evaluation** | Run all | Table: base **37.5 %** → after GRPO **~60 %**. Gain is in *arguments*, not format. Error tables: distractors («Камера 3 працює нормально…») still fail; validation (90 %) overstates the test result |
| 82–90 | **8.1 Reward hacking** | Run (already trained) | Training reward climbs to **~0.9** while task success → **0 %**. Show the completions with stuffed arguments |
| 90–95 | **8.2 Multi-step agent** | Run (~30 s) | Both models **fail** the 2-tool "was the robot late?" question. Single-step training doesn't transfer, which motivates outcome rewards / TRL `environment_factory` |
| 95–100 | **9 Summary** + wrap-up | Run the summary cell, last markdown cell | All metrics in one place (saved to `outputs/results.json`). 4 take-aways. Buffer for questions |

**If you run late**, cut in this order: 8.2 (say it in one sentence) → 3 (show only the family table) → 7.1 fixed/broken tables.

## D. If something goes wrong

| Symptom | Fix |
|---|---|
| `status()` shows `FAILED` | Read `outputs/logs/live.log` (or `hack.log`). Fix, then rerun the start cell |
| `CalledProcessError … gcc … driver.c` | venv on system Python. Redo step A (uv-managed Python) |
| CUDA OOM | `nvidia-smi`: kill other jobs. Restart kernel |
| You restart the kernel while runs train | Fine: `start_runs` does not restart a run that is still going; `status()` reconnects via the pid files |
| Training can't run at all | Use the saved adapters and **skip the start cell** (Section 0, third code cell). In a terminal: `for r in live_run hack_many_args; do cp -r data/checkpoints/$r/. outputs/$r/; done; mkdir -p outputs/logs; for n in live hack; do echo "TRAINING DONE" > outputs/logs/$n.log; done`. The wait cells then pass immediately. Or just show the committed outputs |

The live run is retrained in class, so the exact numbers vary from the committed outputs. The story holds either way.
