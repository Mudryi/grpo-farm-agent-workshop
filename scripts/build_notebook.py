"""Generate workshop.ipynb (single notebook, one section per workshop spec item).

    uv run python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells: list = []


def md(s: str):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s: str):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# ======================================================================================
md(r"""
# From LLM to Farm Operations Agent: Reinforcement Learning with GRPO

**3-hour practical workshop · Qwen2.5-0.5B-Instruct + GRPO + LoRA · Hugging Face TRL**

> *Can we improve an LLM's ability to investigate farm operations by rewarding it for choosing the correct tools and using the returned data?*

⚠️ **All data in this notebook is synthetic dummy data.** Barns, cameras, schedules and events are generated from a random seed.

| Section | Time | Topic |
|---|---|---|
| 0 | — | Setup |
| 1 | 0–35 min | Why RL? · How GRPO works |
| 2 | 35–55 min | The farm environment (tools + simulated data) |
| 3 | 55–80 min | Training dataset |
| 4 | — | Baseline: the pretrained model |
| 5 | 80–105 min | **Student exercise:** reward functions |
| 6 | 105–120 min | Train Qwen with GRPO |
| 7 | 120–150 min | Evaluate the trained policy |
| 8 | 150–180 min | Challenge: reward hacking & multi-step agent |

Code lives in `src/farm_grpo/` (one module per section); the notebook only orchestrates.
""")

# ======================================================================================
md("""
---
# Section 0 · Setup
""")
code("""
%load_ext autoreload
%autoreload 2

import json, time, os
import torch, pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 120)

from farm_grpo.config import get_config
cfg = get_config()

print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
pd.Series(cfg.to_dict(), name="config").to_frame()
""")
code("""
# ---- Instructor switches --------------------------------------------------------------
RUN_LIVE_TRAINING = True              # Section 6: train live in class
LIVE_STEPS = cfg.live_max_steps       # re-set after measuring sec/step (Section 6.1)
RUN_HACK_TRAINING = True              # Section 8: train with a deliberately bad reward
HACK_STEPS = 60
""")

# ======================================================================================
md(r"""
---
# Section 1 · Why reinforcement learning? How GRPO works  *(0–35 min)*

## 1.1 Four questions to start

1. **What does next-token prediction actually optimize?**  The likelihood of the *training text*, not whether the answer works.
2. **What is the difference between predicting the correct action and maximizing success in an environment?**  Imitation vs. outcomes.
3. **Why doesn't supervised fine-tuning always solve that problem?**  SFT needs a perfect demonstration for every task, and it never sees its *own* mistakes.
4. **Can we teach an LLM from outcomes without a perfect example response for every task?**  Yes, if we can *score* a response. That score is the reward.

## 1.2 RL vocabulary, mapped to our farm agent

| RL term | In this workshop |
|---|---|
| **Policy** $\pi_\theta$ | Qwen2.5-0.5B (+ LoRA adapter) |
| **Action** | The generated JSON tool call |
| **Environment** | Simulated farm with 4 read-only tools |
| **Rollout** | Sampling a completion for a prompt |
| **Reward** | Python function scoring the completion |
| **Exploration** | Sampling at temperature 1.0 → varied completions |

> **Honest caveat: RL is not automatically better than SFT.** For one-step tool selection with reliable labels, SFT may be simpler and more effective.
> RL becomes attractive when the correct *sequence* of actions is unknown but **task success can be evaluated**.
> We use the single-step task because it trains in minutes, which makes it a good way to *learn the mechanics*.

## 1.3 GRPO: Group Relative Policy Optimization

For each prompt $q$ we sample a **group** of $G$ completions $o_1,\dots,o_G$, score each with the reward, and normalise **within the group**:

$$A_i = \frac{r_i - \operatorname{mean}(r_1,\ldots,r_G)}{\operatorname{std}(r_1,\ldots,r_G)}$$

* Completions better than their siblings get $A_i>0$ and become more likely. Worse ones get $A_i<0$.
* **No critic / value network** (unlike PPO). The group mean *is* the baseline, which saves memory and compute.
* The full objective adds a clipped policy-ratio update (like PPO) and optionally a KL penalty $\beta\,\mathrm{KL}(\pi_\theta\|\pi_\text{ref})$. We use $\beta=0$.
* **Limitations:** generation is expensive (G samples per prompt), the model can *hack* the reward, and **if every completion in a group gets the same reward then $A_i=0$, so there is no learning signal**.
""")
code("""
from farm_grpo.plotting import group_advantages, plot_advantages

groups = {
    "mixed group (useful signal)":   [1.0, 0.3, 1.0, -0.2],
    "all correct (no signal)":       [1.0, 1.0, 1.0, 1.0],
    "all invalid (no signal)":       [-0.2, -0.2, -0.2, -0.2],
}
for name, r in groups.items():
    print(f"{name:<30} rewards={r}  advantages={group_advantages(r).round(2).tolist()}")
plot_advantages(groups);
""")

# ======================================================================================
md("""
---
# Section 2 · The farm environment  *(35–55 min)*

A simplified, **synthetic** dairy farm: 4 barns (A–D), 8 cameras (two per barn), a feed-pushing robot, a feed mixer wagon and a feeding plan.
"Today" in the simulation is fixed at **2026-03-10**, so labels are reproducible.

## 2.1 The tools (read-only)
""")
code("""
from farm_grpo.environment import TOOL_SPECS, build_world, tool_spec_text, TODAY, CAMERAS
print(tool_spec_text())
""")
md("## 2.2 The simulated world")
code("""
world = build_world()
world.save(cfg.data_dir / "farm_world.json")

print("Feeding schedule (planned, every day):")
display(pd.DataFrame({b: world.schedules[b][TODAY.isoformat()] for b in "ABCD"}.items(),
                     columns=["barn", "planned_feed_push"]))
print("Robot events TODAY:")
display(pd.DataFrame([{"barn": b, **e} for b, evs in world.robot_events.items() for e in evs]))
""")
md("## 2.3 Calling tools by hand")
code("""
print(world.execute("get_feeding_schedule", {"barn_id": "A", "date": "today"}))
print(world.execute("get_robot_events",     {"barn_id": "A", "start_time": "06:00", "end_time": "12:00"}))
print(world.execute("get_mixer_events",     {"barn_id": "B", "date": "yesterday"}))
print(world.execute("get_camera_status",    {"camera_id": 3, "date": "2026-03-09"}))
print(world.execute("get_robot_events",     {"barn_id": "Z", "start_time": "6"}))  # invalid -> error, never raises
""")
md("""
## 2.4 The target workflow: *"Was the feed-pushing robot late in Barn A this morning?"*

A perfect (scripted) investigation needs **two** tool calls plus reasoning over the results:
""")
code("""
from farm_grpo.agent import oracle_investigation
for step in oracle_investigation(world, "A"):
    print(json.dumps(step))
""")
md("""
For the **core exercise** we train only **one correct tool call with valid arguments**.
Single-step tool selection is much easier to train than a multi-turn agent that must execute tools, observe results and decide what to do next.
The full investigation returns in Section 8.
""")

# ======================================================================================
md("""
---
# Section 3 · Training dataset  *(55–80 min)*

Synthetic farm requests for 4 tools, with linguistic variation, **distractors** (mentions of other tools/barns) and unseen values.

**Crucial:** we split by **template family**, not by random shuffling. Each tool has 8 phrasing families:
5 → train, 1 → validation, 2 → test only. Barn **D** and cameras **7–8** never appear in training.
Otherwise near-duplicate sentences leak into the test set and accuracy looks artificially good.
""")
code("""
from farm_grpo.dataset import generate_records, save_records, to_hf_dataset, TEMPLATES, SPLIT_OF_TEMPLATE_INDEX
from farm_grpo.prompts import SYSTEM_PROMPT

records = generate_records(cfg.n_train, cfg.n_val, cfg.n_test, cfg.seed)
save_records(records, cfg.data_dir)
print({k: len(v) for k, v in records.items()})

fam_split = pd.DataFrame([(tool, fam, split, tmpl) for tool, ts in TEMPLATES.items()
                          for (fam, tmpl, _), split in zip(ts, SPLIT_OF_TEMPLATE_INDEX)],
                         columns=["tool", "family", "split", "template"])
fam_split
""")
code("""
# A few examples per split
for split in ["train", "test"]:
    print(f"===== {split} =====")
    for r in records[split][:6]:
        print(f"{r.question:<95} -> {r.expected_tool} {r.expected_args}")
""")
code("""
# Leakage checks
fam = {s: {r.family for r in records[s]} for s in records}
assert not fam["train"] & fam["test"] and not fam["train"] & fam["val"], "template leakage!"
assert not {r.question for r in records["train"]} & {r.question for r in records["test"]}
print("No template-family or question leakage between splits ✔")

train_ds = to_hf_dataset(records["train"])
print(train_ds)
print("\\nSystem prompt the model sees:\\n" + "-" * 60 + "\\n" + SYSTEM_PROMPT)
""")

# ======================================================================================
md("""
---
# Section 4 · Baseline: the pretrained model

Before training: how well does the base Qwen2.5-0.5B-Instruct pick tools (greedy decoding)?
""")
code("""
from farm_grpo.evaluate import load_model, evaluate, error_breakdown, reward_variance_check, compare

base_model, tok = load_model(cfg.model_id)
t0 = time.time()
base_metrics, base_df = evaluate(base_model, tok, records["test"], cfg.eval_batch_size, cfg.eval_max_new_tokens)
print(f"evaluated {len(base_df)} test prompts in {time.time() - t0:.1f}s")
pd.Series(base_metrics, name="Before GRPO").to_frame()
""")
code("""
display(error_breakdown(base_df))
base_df[base_df.error != "ok"][["question", "completion", "error"]].head(10)
""")
md("""
## 4.1 ⚠️ Does GRPO have a signal?  (the most important check before training)

GRPO only learns from groups whose rewards **differ**. Sample G=4 completions per training prompt at temperature 1.0
and count how many groups have zero reward variance.
""")
code("""
var_summary, var_df = reward_variance_check(base_model, tok, records["train"][:80],
                                            num_generations=cfg.num_generations, temperature=cfg.temperature)
display(pd.Series(var_summary, name="value").to_frame())
var_df[var_df["std"] > 0][["question", "rewards"]].head(8)
""")
md("""
**Discuss:** If ~100% of groups were "all correct", the task would be too easy and RL would have nothing to learn.
If ~100% were "all equally wrong", there would be no signal either. You would then fix the prompt, temperature, or reward shaping before spending GPU time.
""")

# ======================================================================================
md("""
---
# Section 5 · Student exercise: reward functions  *(80–105 min)*

| Reward | Condition | Score |
|---|---|---|
| Format | Valid JSON with exactly `{"tool", "arguments"}` (a code fence is OK, prose is not) | **+0.1** |
| Tool selection | Correct tool | **+0.2** |
| Arguments | Correct tool **and** all arguments match (after normalisation) | **+0.7** |
| Invalid output | Cannot parse / wrong structure | **−0.2** |

Maximum = 1.0. TRL calls each reward function with `completions` plus **every extra dataset column** as a keyword argument
(`expected_tool`, `expected_args`), and **sums** the returned lists.

Helpers: `completion_text(c)`, `parse_tool_call(text)` → `.has_schema .tool .arguments .norm_args`, `normalize_args(tool, dict)`.
""")
code("""
from farm_grpo.parsing import completion_text, parse_tool_call
from farm_grpo.environment import normalize_args

pc = parse_tool_call('{"tool": "get_robot_events", "arguments": {"barn_id": "a", "start_time": "7am", "end_time": "8:00"}}')
print(pc.has_schema, pc.tool, pc.arguments, "->", pc.norm_args)
print(parse_tool_call("Sure! I'll call the robot tool.").error)
""")
code("""
# ============================ TODO(student) ============================
def format_reward(completions, **kwargs):
    \"\"\"+0.1 if the completion is a clean {"tool", "arguments"} JSON object, else -0.2.\"\"\"
    rewards = []
    for c in completions:
        text = completion_text(c)
        # TODO: parse `text` and append 0.1 or -0.2
        raise NotImplementedError
    return rewards


def tool_reward(completions, expected_tool, **kwargs):
    \"\"\"+0.2 if the selected tool == expected tool, else 0.\"\"\"
    rewards = []
    for c, tool in zip(completions, expected_tool):
        # TODO
        raise NotImplementedError
    return rewards


def args_reward(completions, expected_tool, expected_args, **kwargs):
    \"\"\"+0.7 if the tool is right AND normalised args match. expected_args are JSON strings.\"\"\"
    rewards = []
    for c, tool, args_json in zip(completions, expected_tool, expected_args):
        # TODO: compare parse_tool_call(...).norm_args with normalize_args(tool, json.loads(args_json))
        raise NotImplementedError
    return rewards
""")
code("""
from farm_grpo.rewards import check_rewards

try:
    STUDENT_OK = check_rewards(format_reward, tool_reward, args_reward)
except NotImplementedError:
    STUDENT_OK = False
    print("Not implemented yet. Fill in the TODO cell above.")
""")
code("""
# Use the student's rewards if they pass the tests, otherwise the reference solution
if STUDENT_OK:
    reward_funcs = [format_reward, tool_reward, args_reward]
    print("Using STUDENT reward functions")
else:
    try:
        from solutions.rewards_solution import REWARD_FUNCS as reward_funcs
        print("Using REFERENCE reward functions (solutions/rewards_solution.py)")
    except ImportError:
        raise RuntimeError("Your reward functions do not pass check_rewards yet. Fix the TODO cell first.")

# Sanity check: score the base model's sampled completions from Section 4.1
row = var_df.iloc[0]
n = len(row.completions)
comps = [[{"role": "assistant", "content": t}] for t in row.completions]
r = records["train"][0]
cols = {"expected_tool": [r.expected_tool] * n, "expected_args": [r.expected_args] * n}
for t, *scores in zip(row.completions, *[f(comps, **cols) for f in reward_funcs]):
    print(f"{sum(scores):+.2f}  {scores}  {t[:110]!r}")
""")
md("""
**Extension idea:** `solutions/rewards_solution.py::args_partial_reward` gives *partial* credit per correct argument.
Does a denser reward help GRPO? What could it break?
""")

# ======================================================================================
md("""
---
# Section 6 · Train Qwen with GRPO  *(105–120 min)*

Configuration: LoRA r=8, G=4 completions per prompt, 4 prompts per step (16 completions), max 96 new tokens, β=0, lr 2e-5.
""")
code("""
from farm_grpo.train import build_grpo_config, train_grpo
del base_model; torch.cuda.empty_cache()

g = build_grpo_config(cfg, cfg.live_run_dir, LIVE_STEPS)
print({k: getattr(g, k) for k in ["num_generations", "per_device_train_batch_size", "max_completion_length",
                                  "learning_rate", "beta", "temperature", "loss_type", "scale_rewards", "bf16"]})
""")
md("""
## 6.1 Measure time per step first

Never assume a step count fits into 10 minutes. Measure seconds/step **including rollout generation** after a warm-up, then choose `LIVE_STEPS`.
""")
code("""
_, bench_timer = train_grpo(cfg, train_ds, reward_funcs, cfg.output_dir / "benchmark", max_steps=6, save=False)
bench = bench_timer.summary(warmup=2)
print(bench)
LIVE_BUDGET_MIN = 8
LIVE_STEPS = min(cfg.live_max_steps, int(LIVE_BUDGET_MIN * 60 / bench["sec_per_step_after_warmup"]))
print(f"-> LIVE_STEPS = {LIVE_STEPS}  (~{LIVE_STEPS * bench['sec_per_step_after_warmup'] / 60:.1f} min)")
""")
md("""
## 6.2 Live training run

While it runs, watch the logs. `reward` should rise, `reward_std` and `frac_reward_zero_std` tell you how much signal is left.
""")
code("""
if RUN_LIVE_TRAINING:
    trainer, timer = train_grpo(cfg, train_ds, reward_funcs, cfg.live_run_dir, max_steps=LIVE_STEPS,
                                heldout_records=records["val"], heldout_every=max(LIVE_STEPS // 4, 1))
    live_log, live_heldout = trainer.state.log_history, trainer.heldout_history
    print(timer.summary())
    del trainer; torch.cuda.empty_cache()
else:
    live_log = json.loads((cfg.live_run_dir / "log_history.json").read_text())
    live_heldout = json.loads((cfg.live_run_dir / "heldout_history.json").read_text())
""")
code("""
from farm_grpo.plotting import plot_training
plot_training(live_log, live_heldout, title=f"Live run ({LIVE_STEPS} steps)");
""")
md("""
## 6.3 Prepared run (trained before class)

`uv run python -m farm_grpo.train --steps 500 --output outputs/prepared_run`
""")
code("""
HAS_PREPARED = (cfg.prepared_run_dir / "adapter_model.safetensors").exists()
if HAS_PREPARED:
    prep_log = json.loads((cfg.prepared_run_dir / "log_history.json").read_text())
    prep_heldout = json.loads((cfg.prepared_run_dir / "heldout_history.json").read_text())
    print(json.loads((cfg.prepared_run_dir / "timing.json").read_text()))
    plot_training(prep_log, prep_heldout, title="Prepared run");
else:
    print("No prepared checkpoint found. Run the command above before class.")
""")
md("**Note:** training reward alone does *not* prove learning. Always look at the held-out curve.")

# ======================================================================================
md("""
---
# Section 7 · Evaluate the trained policy  *(120–150 min)*

Same 80 **test** prompts (unseen template families, unseen barn D / cameras 7–8), greedy decoding.
""")
code("""
results, dfs = {"Before GRPO": base_metrics}, {"Before GRPO": base_df}
for name, path in [("After GRPO (live)", cfg.live_run_dir), ("After GRPO (prepared)", cfg.prepared_run_dir)]:
    if (path / "adapter_model.safetensors").exists():
        m, tok = load_model(cfg.model_id, adapter_path=str(path))
        results[name], dfs[name] = evaluate(m, tok, records["test"], cfg.eval_batch_size, cfg.eval_max_new_tokens)
        del m; torch.cuda.empty_cache()

table = compare(results)
(cfg.output_dir / "results.json").write_text(json.dumps(results, indent=1))
table
""")
code("""
from farm_grpo.plotting import plot_comparison
plot_comparison(table);
""")
md("## 7.1 Error analysis")
code("""
best = list(dfs)[-1]
print("Errors by type:", best)
display(error_breakdown(dfs[best]))

# Per template family: where does the model still fail?
pd.concat({k: d.groupby("family")["task_success"].mean().mul(100).round(0) for k, d in dfs.items()}, axis=1)
""")
code("""
# Prompts fixed by training and prompts broken by training
b, a = dfs["Before GRPO"].set_index("id"), dfs[best].set_index("id")
fixed = a[(~b.task_success) & a.task_success]
broken = a[b.task_success & (~a.task_success)]
print(f"fixed by GRPO: {len(fixed)}   broken by GRPO: {len(broken)}")
pd.concat([b.loc[fixed.index, ["question", "completion"]].rename(columns={"completion": "before"}),
           a.loc[fixed.index, ["completion"]].rename(columns={"completion": "after"})], axis=1).head(8)
""")
code("""
# Remaining failures after training
a[~a.task_success][["question", "expected_args", "completion", "error"]].head(10)
""")
md("""
**Instructor note (from the dry run on DGX Spark, see `MEASURED.md`):** the 100-step *live* run beat the 500-step *prepared* run on test.
The longer run learned a shortcut on the test-only distractor family `robot_distract2` ("The schedule for Barn B is fine. Which feed-push runs…").
It now picks `get_feeding_schedule` because the word *schedule* appears. Validation accuracy never showed this, because validation uses a *different* family.
This is exactly why we split by template family and keep a separate test set: **more RL steps ≠ better policy**.

**Discuss:**
* Did the model *really* improve (correct arguments on unseen phrasings) or did it mostly learn the **format**?
* Which families still fail, and why? (day-part conventions? relative dates? distractors?)
* Would SFT on the 240 labelled examples have done as well?
""")

# ======================================================================================
md("""
---
# Section 8 · Challenge  *(150–180 min)*

## 8.1 Reward hacking

`src/farm_grpo/rewards_bad.py` contains rewards that *look* reasonable but are not task success:

| Bad reward | What it rewards | How it can be hacked |
|---|---|---|
| `format_only_reward` | Any JSON with a `"tool"` key | Correct format, wrong tool / args |
| `keyword_reward` | Correct tool **name** appears anywhere | Mention every tool name |
| `many_args_reward` | 0.25 per argument key ("detailed calls") | Stuff the arguments with extra keys |

Train with one of them, then measure with the **real** metrics. Training reward goes up. Does task success?
""")
code("""
from farm_grpo.rewards_bad import BAD_REWARDS
HACK = "many_args"          # try: "format_only", "keyword", "many_args"
hack_dir = cfg.output_dir / f"hack_{HACK}"

if RUN_HACK_TRAINING:
    trainer, _ = train_grpo(cfg, train_ds, BAD_REWARDS[HACK], hack_dir, max_steps=HACK_STEPS,
                            heldout_records=records["val"], heldout_every=max(HACK_STEPS // 3, 1),
                            learning_rate=5e-5)
    hack_log, hack_heldout = trainer.state.log_history, trainer.heldout_history
    del trainer; torch.cuda.empty_cache()
else:
    hack_log = json.loads((hack_dir / "log_history.json").read_text())
    hack_heldout = json.loads((hack_dir / "heldout_history.json").read_text())

plot_training(hack_log, hack_heldout, title=f"Bad reward: {HACK}");
""")
code("""
m, tok = load_model(cfg.model_id, adapter_path=str(hack_dir))
results[f"Hacked ({HACK})"], dfs[f"Hacked ({HACK})"] = evaluate(m, tok, records["test"])
del m; torch.cuda.empty_cache()
display(compare(results))
dfs[f"Hacked ({HACK})"][["question", "completion", "error"]].head(8)
""")
md("""
**Exercises**
1. Change `HACK` and compare. Which bad reward is hacked fastest, and why?
2. Fix the bad reward *minimally* so it cannot be hacked. What did you need to add?
3. Use `reward_weights` in `GRPOConfig` (e.g. `train_grpo(..., reward_weights=[1.0, 0.0, 0.0])`) to train on format only. What happens to `frac_reward_zero_std`?

## 8.2 Multi-step investigation (transfer test)

Questions that need **both** the feeding schedule **and** the robot events, graded by **environment outcome** (was the final answer right?).
Our training was single-step, so does it transfer?
""")
code("""
from farm_grpo.agent import make_tasks, evaluate_agent
tasks = make_tasks(world)
pd.DataFrame([{"question": t.question, **t.truth} for t in tasks])
""")
code("""
agent_results = {}
candidates = [("Before GRPO", None)] + [(n, p) for n, p in [("After GRPO (prepared)", cfg.prepared_run_dir),
                                                             ("After GRPO (live)", cfg.live_run_dir)]
                                        if (p / "adapter_model.safetensors").exists()][:1]
for name, path in candidates:
    m, tok = load_model(cfg.model_id, adapter_path=str(path) if path else None)
    agent_df, trajectories = evaluate_agent(m, tok, world, tasks)
    agent_results[name] = agent_df[["answer_correct", "used_required_tools", "success"]].mean().mul(100).round(0)
    print(f"\\n################ {name} ################")
    for tr in trajectories[:2]:
        print(tr.pretty(), "\\n")
    del m; torch.cuda.empty_cache()
pd.DataFrame(agent_results)
""")
md("""
**Discuss / advanced extension**
* Did single-step GRPO help, hurt, or not matter for the multi-step agent? (Training only ever rewarded *one* JSON object.)
* Design an **outcome reward** for the multi-step task: reward 1 only if the final answer is right *and* the required tools were used.
* Current TRL supports tool-calling rollouts and environment-managed rewards (`GRPOTrainer(tools=..., environment_factory=...)`), which is the natural next stage.

---
### Wrap-up
* RL lets us optimise **outcomes** we can score, without a perfect demonstration.
* GRPO: group sampling + relative advantages, no critic.
* The reward **is** the specification. Models optimise what you measure, not what you mean.
* Always check: signal (reward variance), held-out success, failure inspection.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (farm-grpo)", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "workshop.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
