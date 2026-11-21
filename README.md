# Training an LLM Farm Agent with GRPO

**Qwen2.5-0.5B-Instruct · GRPO + LoRA · Hugging Face TRL** · practical workshop

> *Can we improve an LLM's ability to investigate farm operations by rewarding it for choosing the correct tools and using the returned data?*

In this workshop you use reinforcement learning (GRPO) to teach a small language model to answer a farm manager's request
with the right **tool call**: one of four read-only farm tools, with correct arguments. The requests are in **Ukrainian**:

```text
Скільки разів робот-підштовхувач працював у корівнику D сьогодні ввечері?
  -> {"tool": "get_robot_events", "arguments": {"barn_id": "D", "start_time": "18:00", "end_time": "22:00"}}
```

Tool names, argument keys and the system prompt stay in English, as with most real APIs. The notebook text is English.

⚠️ **All data is synthetic dummy data.** The farm, its barns, cameras and events are generated from a random seed.

---

## What you need

* An **NVIDIA GPU with at least 12 GB** of memory. A full run peaked at ~9.7 GB on an RTX 3090.
  A CPU is far too slow for training.
* Linux with an NVIDIA driver **≥ 580** for the default setup (CUDA 13 PyTorch). For an older driver, see [Troubleshooting](#troubleshooting).
* Internet access on the first run, to download the model (~1 GB) from Hugging Face.
* [`uv`](https://docs.astral.sh/uv/getting-started/installation/), the Python package manager.

## Quick start

```bash
git clone https://github.com/Mudryi/grpo-farm-agent-workshop.git
cd grpo-farm-agent-workshop

uv python install 3.12
uv venv --python-preference only-managed --python 3.12
uv sync

uv run pytest -q                       # sanity check: all tests should pass
uv run jupyter lab workshop_uk.ipynb
```

Then run the notebook **from top to bottom** (*Run → Run All Cells*).
The whole notebook takes **~6–7 minutes** on an RTX 3090.

> The notebook in the repo already contains the outputs from our run, so you can read the results before running anything.

## How the notebook works

The **third code cell starts two GRPO trainings in the background**. They are separate processes, so you keep working through
the notebook while they train (~6 min). Call `status()` any time to see their progress.
Sections 6 and 8 wait for them automatically.

| Run | Reward | Steps | Purpose |
|---|---|---|---|
| `live` | the real reward: format + correct tool + correct arguments | 100 | the actual result |
| `hack` | a deliberately bad reward (`many_args`: more argument keys = more reward) | 60 | reward-hacking demo |

The two runs use the same data and the same settings; only the reward differs.

| Section | What you do |
|---|---|
| 0 | Setup, start the training |
| 1 | Why RL? How GRPO works (group-relative advantages) and where it fits next to SFT, PPO and DPO |
| 2 | Explore the farm environment and call the tools by hand |
| 3 | The dataset and the **template-family split** (test = phrasings the model never saw) |
| 4 | Baseline of the untrained model + "does GRPO even have a learning signal?" check |
| 5 | The reward functions, checked against test cases |
| 6 | Training curves of the live run |
| 7 | Evaluation on the test set + error analysis |
| 8 | Reward hacking, and a multi-step agent ("was the robot late?") |
| 9 | Summary of all metrics |

## Results from our run

RTX 3090, 80 test prompts, greedy decoding. Your numbers will differ a little: these are single training runs.

| Metric | Before GRPO | After GRPO (100 steps) | Bad reward `many_args` |
|---|---|---|---|
| Valid tool-call JSON | 95.0 % | 100.0 % | 100.0 % |
| Correct tool selection | 81.2 % | 86.2 % | 25.0 % |
| **Overall task success** | **37.5 %** | **58.8 %** | **0.0 %** |

What to take away:
* **GRPO helps, mostly through the arguments.** The model learns to turn «сьогодні ввечері», «7.30» or «8 березня» into canonical values. Format was already fine before training.
* **Validation can overstate success.** Validation reaches ~88–90 %, the test set only ~60 %: the test phrasings contain distractors such as «Камера 3 працює нормально…», which still make the model pick the camera tool.
* **The reward is the specification.** With `many_args` the training reward rises to ~0.95 while real task success falls to 0 %.
* **Single-step training does not transfer to multi-step investigation.** The 2-tool question stays at 0 % success.

## Where to look in the code

All logic lives in `src/farm_grpo/`; the notebook only calls it.

| File | What's inside |
|---|---|
| `rewards.py` | the three reward functions and the `check_rewards` test cases |
| `rewards_bad.py` | `many_args_reward`, the hackable reward |
| `train.py` | `train_grpo`: TRL `GRPOTrainer` + LoRA + held-out evaluation every N steps |
| `background.py` | the two runs (`RUNS`) and `start_runs` / `status` / `wait_for` |
| `environment.py` | the dummy farm, the 4 tools, argument normalisation (`"7:00"` = `"07:00"`) |
| `dataset.py` | Ukrainian request templates and the train / val / test split |
| `evaluate.py` | metrics, error breakdown, reward-variance check |
| `agent.py` | the multi-step agent loop and its grading |
| `config.py` | all hyper-parameters (`WorkshopConfig`) |

**GRPO itself happens inside TRL.** In `.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py`, search for these methods:
* `_generate_and_score_completions`: samples G completions per prompt, calls the reward functions, computes the advantage `(r − group mean) / group std`.
* `_compute_loss`: the clipped policy-gradient loss.

## Things to try

* Change hyper-parameters in `src/farm_grpo/config.py` (e.g. `num_generations`, `learning_rate`, `live_max_steps`) or the runs in `RUNS` (`background.py`), then rerun the notebook.
* Fix `many_args_reward` minimally so it can no longer be hacked.
* Design an **outcome reward** for the multi-step agent: reward 1 only if the final answer is right *and* both required tools were used.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CalledProcessError ... gcc` during training | Python has no development headers (Triton needs `Python.h`). Use the uv-managed Python from Quick start, or install `python3.12-dev` |
| `uv sync` installs a PyTorch that does not see your GPU (driver older than 580) | In `pyproject.toml`, change `whl/cu130` to `whl/cu128`, then run `uv lock && uv sync` |
| `status()` shows `FAILED` | Read `outputs/logs/live.log` or `outputs/logs/hack.log` |
| CUDA out of memory | Close other GPU programs (`nvidia-smi`), restart the kernel |
| You restarted the kernel during training | Fine: the background runs keep going, and `start_runs` does not start them twice |
| No `uv` (e.g. Colab) | `pip install -e . jupyter`. The default PyPI PyTorch works there |
