# Results: Ukrainian requests (RTX 3090 dry run, 2026-11-19)

The same workshop (`workshop_uk.ipynb`) with the dataset requests in **Ukrainian**. All data is synthetic dummy data.
Tool names, argument keys, the system prompt and the labels stay English / ISO; only the user requests and the multi-step questions are Ukrainian.
These metrics are reported separately from the English ones (`RESULTS.md`). Raw files: `outputs/uk/results.json`, `outputs/uk/results_by_family.csv`, `outputs/uk/agent_results.json`.

## Setup
- **Model and training:** same as English.
  - Qwen2.5-0.5B-Instruct + LoRA r=8, TRL `GRPOTrainer`, G=4, 16 completions per step, β=0.
  - lr 2e-5 for the GRPO run; the `many_args` hack run uses 5e-5.
- **Data:** 240 train / 40 val / 80 test, the same 8 template families per tool and the same family split as English (test = unseen phrasings, unseen barn D / cameras 7–8).
- **Surface forms:**
  - Times: `07:00`, `7:00`, `7.30`, «7 ранку», «3 дня», «8 вечора».
  - Dates: «сьогодні», «вчора», «минулої ночі», «8 березня», «8 березня 2026 р.», `08.03.2026`, `2026-03-08`.
  - Day parts: «сьогодні вранці», «вдень», «після обіду», «ввечері».
- **Timing:** both background runs (GRPO 100 steps + `many_args` hack 60 steps) run in parallel; the whole notebook takes **6.5 min** in background mode and **51 s** in pretrained mode.
  - GRPO run: 2.95 s/step.
  - Hack run: 3.4 s/step.

## Test-set results (80 Ukrainian prompts, greedy decoding)

| Metric | Before GRPO | After GRPO (100 steps) | Bad reward `many_args` (60 steps) |
|---|---|---|---|
| Valid tool-call JSON (%) | 95.0 | 100.0 | 100.0 |
| Correct tool selection (%) | 81.2 | 82.5 | 25.0 |
| Correct arguments (%) | 37.5 | 60.0 | 0.0 |
| **Overall task success (%)** | **37.5** | **60.0** | **0.0** |
| Unnecessary tool calls (%) | 0.0 | 0.0 | 0.0 |
| Mean reward | 0.51 | 0.69 | 0.15 |

Held-out (validation, 40 prompts) during training:

| Step | 0 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|
| After GRPO, task success % | 57.5 | 77.5 | 87.5 | 87.5 | 90.0 |

The hack run's validation success fell from 57.5 % at step 0 to 0 % by step 60.

Training reward:

| Run | First 10 steps | Last 10 steps | Zero-std groups (no signal), first 10 → last 10 steps |
|---|---|---|---|
| After GRPO | 0.50 | 0.86 | 30 % → 92 % |
| `many_args` hack | 0.56 | 0.91 | 75 % → 80 % |

In the GRPO run, `args_reward` went 0.27 → 0.57; format and tool were nearly saturated from the start.

## Task success per test template family (%)

| Family | Example | Before | After GRPO |
|---|---|---|---|
| `cam_distract2` | «Чи був простій камери 5 у корівнику C сьогодні?» | 50 | 90 |
| `cam_video` | «Чи є в нас відео з камери номер 6 минулої ночі?» | 30 | 70 |
| `mixer_distract2` | «Камера 3 працює нормально. Мене цікавлять приїзди TMR-міксера в корівник D …» | 0 | 0 |
| `mixer_fresh` | «О котрій міксер привіз свіжий корм у корівник A 8 березня?» | 80 | 90 |
| `robot_count` | «Скільки разів робот-підштовхувач працював у корівнику D сьогодні ввечері?» | 10 | 60 |
| `robot_distract2` | «Графік для корівника A в нормі. Які проходи … зробив робот у корівнику D між 13:00 і 16:00?» | 70 | 50 |
| `sched_cows` | «Згідно з планом, коли мають годувати корів у корівнику D 7 березня?» | 40 | 90 |
| `sched_distract2` | «Забудь про міксер у корівнику B. Які очікувані часи годування для корівника A …?» | 20 | 30 |

## Multi-step agent (4 «Чи запізнився робот…?» questions)

| | Before GRPO | After GRPO |
|---|---|---|
| Answer correct (%) | 0 | 25 |
| Used both required tools (%) | 0 | 0 |
| Success (%) | 0 | 0 |

## Findings
1. **The Ukrainian task is much harder for the base model:** 37.5 % task success vs 73.8 % with English requests. The model picks the right tool about as often (81 % vs 91 %), but it fails to convert Ukrainian dates, times and day parts into canonical arguments.
2. **GRPO still helps:**
   - Test task success rose from 37.5 % to 60 % in 100 steps (~5 min).
   - The gain is almost entirely in **arguments**: day parts («сьогодні ввечері» → 18:00–22:00) went from 10 % to 60 %, and Ukrainian dates (`sched_cows`) from 40 % to 90 %.
3. **Validation overstates the result:** 90 % on validation vs 60 % on test.
   - Validation is a single terse family per tool. The test families contain distractors.
   - `mixer_distract2` stays at 0 %: «Камера 3 працює нормально…» makes the model call `get_camera_status`.
   - `robot_distract2` dropped from 70 % to 50 %: the model reacts to the word «Графік».
   - This is the same distractor weakness as in English, only stronger.
4. **Reward hacking reproduced:** the `many_args` training reward rose 0.56 → 0.91, while task success fell to 0 % on validation and test.
5. **No transfer to multi-step:** success was 0 % before and after; the model never calls both tools.

*Single-seed run on an RTX 3090; numbers vary with seed and hardware. Saved checkpoints for pretrained mode: `data/uk/checkpoints/` (not in git).*
