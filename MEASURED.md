# Dry-run measurements: DGX Spark (GB10), 2026-11-18

Stack: torch 2.14.1+cu130, TRL 1.18.1, transformers 5.x, peft, Qwen2.5-0.5B-Instruct (bf16), LoRA r=8,
G=4, 16 completions/step, max 96 new tokens, lr 2e-5, β=0. Rewards: `farm_grpo.rewards.REWARD_FUNCS`.

## Timing
| What | Measured |
|---|---|
| GRPO step (incl. rollout generation) | **2.8–2.9 s** (3.3 s when another job shared the GPU) |
| Live run, 100 steps | **~4.8 min** |
| Prepared run, 500 steps | ~28 min |
| Greedy eval, 80 test prompts | ~4 s |
| Full notebook end-to-end (live + hack training) | ~15 min |

## Test set (80 prompts, unseen template families + barn D / cameras 7–8)
| Metric | Before GRPO | Live (100 steps) | Prepared (500 steps) | Hacked (`many_args`, 60 steps) |
|---|---|---|---|---|
| Valid tool-call JSON (%) | 97.5 | 100.0 | 100.0 | 0.0 |
| Correct tool selection (%) | 92.5 | 100.0 | 87.5 | 0.0 |
| Correct arguments (%) | 77.5 | 97.5 | 87.5 | 0.0 |
| Overall task success (%) | 77.5 | **97.5** | 87.5 | 0.0 |
| Unnecessary tool calls (%) | 0.0 | 0.0 | 0.0 | 65.0 |

* Baseline GRPO signal: 36% of groups (G=4, T=1.0) had zero reward std, so most groups give a learning signal.
* Validation success: 87.5% at step 0, 92.5% from step ~100 onwards (plateau).
* Prepared run regression: family `robot_distract2` dropped to 0% (it picks the schedule tool when "schedule" appears). Validation did not reveal it. This is used as a discussion point in Section 7.
* Hack run: training reward 0.50 → 1.00 while held-out success fell 87.5% → 10%. Completions stuff extra argument keys / emit several calls.
* Multi-step agent (4 tasks): 0% success both before and after GRPO. The trained model now uses the correct morning window (06:00–12:00) but never calls the feeding schedule. Single-step training does not transfer to multi-step investigation (Section 8.2 discussion).

These are single-seed runs; numbers will vary with seed and hardware.

---

# Re-run on home server: RTX 3090 24 GB (x86_64), 2026-11-19

Same code, uv-managed Python 3.12.15, torch 2.14.1+cu130, TRL 1.18.1. `uv run pytest`: 10 passed.

## Saved DGX checkpoints re-evaluated on the 3090 (no retraining)
Identical to the DGX numbers: live 97.5 %, prepared 87.5 %, hack 0 %. Base model: 73.8 % (77.5 % on GB10, greedy-decoding numeric differences).

## Full notebook execution (retrains live + hack): ~15 min, no errors
| What | Measured |
|---|---|
| GRPO step, live run | **2.4 s** (benchmark cell: 3.1 s) |
| Live run, 100 steps | ~4 min |
| Hack run, 60 steps | 4.0 s/step, ~4 min |
| Greedy eval, 80 test prompts | ~4 s |
| Zero-std groups (signal check) | 42.5 % |

| Metric (test, 80) | Before | Live (retrained, 100 steps) | Prepared (DGX, 500 steps) | Hack (retrained) |
|---|---|---|---|---|
| Valid tool-call JSON (%) | 95.0 | 100.0 | 100.0 | 90.0 |
| Correct tool selection (%) | 91.2 | 93.8 | 87.5 | 88.8 |
| Correct arguments (%) | 73.8 | 87.5 | 87.5 | 0.0 |
| Overall task success (%) | 73.8 | **87.5** | 87.5 | **0.0** |
| Unnecessary tool calls (%) | 0.0 | 0.0 | 0.0 | 7.5 |

* The live run varies between runs: 97.5 % (GB10) vs 87.5 % (3090). Validation went 87.5 → 80 → 90 → 92.5 → 92.5 %.
* Hack run: held-out success 87.5 % → 0–12.5 %. This time it hacked mainly through extra argument keys (correct tool, invalid args) rather than multiple calls.
* The original 97.5 % live checkpoint is preserved in `data/checkpoints/live_run/`.
