# Workshop: Training Qwen with GRPO for a Farm Operations Agent

90–100 min practical lecture · Qwen2.5-0.5B-Instruct · GRPO + LoRA · Hugging Face TRL

Students train a small LLM with reinforcement learning (GRPO) to pick the right **read-only farm tool** and its
arguments for a farm manager's request. The requests are in **Ukrainian**. Tool names, argument keys, the system
prompt and the labels stay English / ISO, as with most real APIs. The notebook text is English.
**All data is synthetic dummy data**, generated from a seed (`src/farm_grpo/environment.py`).

The instructor timeline and the commands for the day are in **[`DEMO.md`](DEMO.md)**.

## Contents

```
workshop_uk.ipynb           the workshop notebook (Sections 0-9), committed with outputs
src/farm_grpo/
  config.py                 hyper-parameters and paths (WorkshopConfig)
  environment.py            dummy farm world, 4 tools, argument normalisation
  prompts.py                system prompts (single-step + agent)
  dataset.py                Ukrainian request templates, template-family split 240/40/80
  parsing.py                JSON tool-call parsing / validation
  rewards.py                reward functions (format / tool / args) + check_rewards test harness
  rewards_bad.py            many_args_reward: a deliberately hackable reward (Section 8.1)
  train.py                  train_grpo: GRPOTrainer + LoRA + held-out evaluation callback
  background.py             runs the two trainings as background processes (start_runs, status, wait_for)
  evaluate.py               generation, metrics, error breakdown, reward-variance check
  agent.py                  multi-step investigation loop + outcome grading (Section 8.2)
  plotting.py               advantage toy plot, training curves, comparison chart, training_summary
tests/                      pytest checks: split leakage, labels, rewards, agent grading
data/                       generated train/val/test jsonl + farm_world.json
outputs/                    live_run/, hack_many_args/ (training logs), results.json
```

## Setup

```bash
uv python install 3.12            # uv-managed Python ships headers (Triton needs Python.h)
uv venv --python-preference only-managed --python 3.12
uv sync
uv run pytest -q
uv run jupyter lab workshop_uk.ipynb
```

> With a system Python that has no `python3.12-dev` headers, TRL's Triton kernel fails with
> `CalledProcessError ... gcc`. Use the uv-managed interpreter (above) or install the dev headers.

Colab / other machines: `pip install -e . && pip install jupyter` (the `pytorch-cu130` index in `pyproject.toml` only applies to uv).

## How training runs

The third code cell calls `start_runs(["live", "hack"])`. That starts two GRPO trainings as separate processes
(`python -m farm_grpo.background <name>`), defined in `RUNS` in `background.py`:

| Run | Reward | Steps | Output |
|---|---|---|---|
| `live` | `rewards.REWARD_FUNCS` (format + tool + arguments) | 100 | `outputs/live_run/` |
| `hack` | `rewards_bad.many_args_reward` | 60 (lr 5e-5) | `outputs/hack_many_args/` |

Both finish ~6 min later, while Section 1 is presented. Sections 6 and 8 call `wait_for(...)` and then read the results.
Logs: `outputs/logs/<run>.log`.

## Results (RTX 3090, 80 test prompts, greedy decoding)

| Metric | Before GRPO | After GRPO (100 steps) | Bad reward `many_args` (60 steps) |
|---|---|---|---|
| Valid tool-call JSON (%) | 95.0 | 100.0 | 100.0 |
| Correct tool selection (%) | 81.2 | 86.2 | 25.0 |
| **Overall task success (%)** | **37.5** | **58.8** | **0.0** |

* **GRPO helps, through the arguments.** Day parts («сьогодні ввечері» → 18:00–22:00) and Ukrainian dates improve the most. Format was already fine before training.
* **Validation overstates it.** Validation reaches ~88–90 %, test only ~60 %, because the test families contain distractors.
  «Камера 3 працює нормально…» still makes the model call the camera tool.
* **Reward hacking:** the `many_args` training reward rises ~0.57 → ~0.95 while task success falls to 0 %.
* **No transfer to multi-step:** the 2-tool "was the robot late?" questions stay at 0 % success.

Timing on the 3090: ~3 s per GRPO step; the whole notebook runs in ~6.5 min. Single-seed runs, so numbers vary between runs.
