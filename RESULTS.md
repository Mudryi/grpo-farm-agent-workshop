# Results: GRPO Farm-Operations Agent (dry run, 2026-11-18)

**Status: the model was trained and the pipeline works end to end** on the DGX Spark (GB10).
All data is synthetic dummy data. Detailed timings: [`MEASURED.md`](MEASURED.md). Full notebook with outputs: `workshop_executed.ipynb`.

## Setup
- Model: Qwen2.5-0.5B-Instruct (bf16) + LoRA r=8, trained with TRL `GRPOTrainer`
- Task: answer a farm manager's question with **one JSON tool call** (4 read-only tools)
- Data: 240 train / 40 val / 80 test, split by **template family** (test uses unseen phrasings and unseen barn D / cameras 7–8)
- Reward: +0.1 valid format, +0.2 correct tool, +0.7 correct arguments, −0.2 invalid output
- GRPO: G=4 completions per prompt, 16 completions per step, β=0, lr 2e-5, ~2.9 s/step

## Test-set results (80 prompts, greedy decoding)

| Metric | Before GRPO | Live run (100 steps, ~5 min) | Prepared run (500 steps, ~28 min) | Bad reward `many_args` (60 steps) |
|---|---|---|---|---|
| Valid tool-call JSON (%) | 97.5 | 100.0 | 100.0 | 0.0 |
| Correct tool selection (%) | 92.5 | 100.0 | 87.5 | 0.0 |
| Correct arguments (%) | 77.5 | 97.5 | 87.5 | 0.0 |
| **Overall task success (%)** | **77.5** | **97.5** | **87.5** | **0.0** |
| Unnecessary tool calls (%) | 0.0 | 0.0 | 0.0 | 65.0 |

## Findings
1. **GRPO works:** 100 steps (~5 min) improved task success from 77.5% to 97.5%. The biggest gains are on day-part questions ("during the morning" → 06:00–12:00): 30% → 100%.
2. **More steps ≠ better:** the 500-step run regressed to 87.5%. It learned a shortcut on the test-only distractor family `robot_distract2` ("The schedule for Barn B is fine. Which feed-push runs did the robot make…") and picks `get_feeding_schedule` (0% on that family). Validation (different family) stayed at 92.5% and did not reveal it. That is why the split is by template family with a separate test set.
3. **Reward hacking reproduced:** with `many_args_reward` the training reward rose from 0.50 to 1.00, while held-out success fell from 87.5% to 10% (0% on test). The model stuffs arguments with extra keys and emits several calls.
4. **GRPO signal check:** before training, 36% of sampled groups (G=4, T=1.0) had zero reward variance, so most groups provided a learning signal.
5. **No transfer to multi-step:** on 4 "was the robot late?" investigations, success was 0% before and after training. The trained model uses the correct morning window but never calls the feeding schedule. Single-step training does not teach multi-step investigation (discussion point for Section 8.2).

## Artifacts (not in git, regenerate locally)
| Path | What |
|---|---|
| `outputs/live_run/` | best checkpoint (LoRA adapter) |
| `outputs/prepared_run/` | 500-step checkpoint |
| `outputs/hack_many_args/` | reward-hacked checkpoint |
| `outputs/results.json` | metric table above |

Regenerate: `uv run python -m farm_grpo.train --steps 500` (prepared) and run `workshop.ipynb` (live + hack).

*Single-seed runs; numbers will vary with seed and hardware.*
