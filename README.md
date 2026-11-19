# Workshop: Training Qwen with GRPO for a Farm Operations Agent

90–100 min practical lecture · Qwen2.5-0.5B-Instruct · GRPO + LoRA · Hugging Face TRL

Students train a small LLM with reinforcement learning (GRPO) to choose the right
**read-only farm tool** and its arguments for a farm manager's question.
**All data is synthetic dummy data**, generated from a seed (`src/farm_grpo/environment.py`).

## Contents

```
workshop.ipynb              the workshop notebook, English (Sections 0-9)
workshop_uk.ipynb           the same workshop in Ukrainian: Ukrainian requests, own training runs, metrics in outputs/uk/
src/farm_grpo/
  config.py                 all hyper-parameters / paths (WorkshopConfig)
  environment.py            dummy farm world + 4 tools + argument normalisation
  prompts.py                system prompts (single-step + agent)
  dataset.py                synthetic queries (English + Ukrainian templates), template-family split 240/40/80
  parsing.py                JSON tool-call parsing / validation
  rewards.py                reward functions (format / tool / args) + test harness (check_rewards)
  rewards_bad.py            deliberately hackable rewards (Section 8)
  evaluate.py               generation, metrics, error breakdown, reward-variance check
  train.py                  GRPOTrainer + LoRA + timing / held-out callbacks + CLI
  background.py             start/monitor training runs as background processes (start_runs, status, wait_for)
  agent.py                  multi-step investigation loop + outcome grading
  plotting.py               advantage toy plot, training curves, comparison chart
scripts/build_notebook.py   regenerates workshop.ipynb and workshop_uk.ipynb
tests/                      pytest sanity checks (split leakage, rewards)
data/                       generated jsonl + farm_world.json (Ukrainian: data/uk/)
outputs/                    checkpoints: live_run/, hack_*/ (Ukrainian: outputs/uk/)
```

Format: **90–100 min** (the notebooks themselves carry no timings). GRPO training starts in the background at minute ~3 and finishes during the intro.
The minute-by-minute plan and the exact commands are in **[`DEMO.md`](DEMO.md)**.

## Ukrainian version

`workshop_uk.ipynb` has the same English text and sets `LANG = "uk"`. Only the user **requests** are Ukrainian (same 8 template families per tool,
same split). Tool names, argument keys, the system prompt and the labels stay canonical English / ISO.
All UK data, checkpoints, logs and metrics are kept separate: `data/uk/`, `outputs/uk/`. Results: [`RESULTS_UK.md`](RESULTS_UK.md).

## Setup

### DGX Spark (GB10, aarch64, CUDA 13)

```bash
uv python install 3.12            # uv-managed Python ships headers (Triton needs Python.h)
uv venv --python-preference only-managed --python 3.12
uv sync
uv run pytest -q
uv run jupyter lab workshop.ipynb
```

> The system `/usr/bin/python3.12` has no `python3.12-dev` headers. TRL's GRPO log-prob
> kernel is compiled with Triton at runtime and fails with `CalledProcessError ... gcc`.
> Use the uv-managed interpreter (above) or `sudo apt install python3.12-dev`.

### Colab / x86

```bash
pip install -e . && pip install jupyter
```
(The `pytorch-cu130` index in `pyproject.toml` only applies to uv.)

## Before the lecture (instructor)

```bash
# 1. optional: pre-train the hacking demo, so Section 8 can skip live training
uv run python -m farm_grpo.train --steps 60 --reward many_args --lr 5e-5
# 2. run the whole notebook once to check it (also: workshop_uk.ipynb)
uv run jupyter nbconvert --to notebook --execute workshop.ipynb --output workshop_executed.ipynb
```

Notebook switch (2nd code cell): `TRAINING_MODE = "background" | "inline" | "pretrained"` (see `DEMO.md`).

Student handout: `uv run python scripts/make_student_copy.py` → `dist/student/` (both notebooks without outputs,
no checkpoints, no outputs). The notebooks are complete: reward functions are implemented in Section 5.

## Measured on DGX Spark (GB10)

See `MEASURED.md`: ~2.9 s/step, live 100 steps ≈ 5 min, test task success 77.5% → 97.5%.
