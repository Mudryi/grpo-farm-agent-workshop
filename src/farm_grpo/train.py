"""GRPO + LoRA training with Hugging Face TRL.

Notebook:   trainer, timer = train_grpo(cfg, train_ds, reward_funcs, output_dir, max_steps=20)
CLI (prepared checkpoint before class):
    uv run python -m farm_grpo.train --steps 300 --output outputs/prepared_run
    uv run python -m farm_grpo.train --steps 60 --reward format_only --output outputs/hack_format_only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

from .config import WorkshopConfig, get_config
from .evaluate import best_dtype


class StepTimer(TrainerCallback):
    """Wall-clock seconds per optimizer step (includes rollout generation)."""

    def __init__(self):
        self.times: list[float] = []
        self._t0 = None

    def on_step_begin(self, args, state, control, **kw):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times.append(time.perf_counter() - self._t0)

    def summary(self, warmup: int = 2) -> dict:
        t = self.times[warmup:] or self.times
        return {"steps": len(self.times), "sec_per_step_after_warmup": round(sum(t) / max(len(t), 1), 2),
                "first_steps": [round(x, 2) for x in self.times[:warmup]]}


class HeldOutEvalCallback(TrainerCallback):
    """Every `every` steps: greedy task-success on a held-out set (training reward is not enough)."""

    def __init__(self, records, tokenizer, every: int = 25, max_new_tokens: int = 96):
        self.records, self.tok, self.every, self.max_new_tokens = records, tokenizer, every, max_new_tokens
        self.history: list[dict] = []

    def _run(self, model, step):
        from .evaluate import evaluate

        was_training = model.training
        model.eval()
        metrics, _ = evaluate(model, self.tok, self.records, max_new_tokens=self.max_new_tokens)
        if was_training:
            model.train()
        self.history.append({"step": step, **metrics})
        print(f"[held-out @ step {step}] success={metrics['Overall task success (%)']}%  "
              f"valid_json={metrics['Valid tool-call JSON (%)']}%")

    def on_train_begin(self, args, state, control, model=None, **kw):
        self._run(model, 0)

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % self.every == 0:
            self._run(model, state.global_step)


def build_grpo_config(cfg: WorkshopConfig, output_dir: str | Path, max_steps: int, **overrides):
    from trl import GRPOConfig

    bf16 = best_dtype() == torch.bfloat16
    kwargs = dict(
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="constant",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        beta=cfg.beta,
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        report_to="none",
        seed=cfg.seed,
        model_init_kwargs={"dtype": best_dtype()},
    )
    kwargs.update(overrides)
    return GRPOConfig(**kwargs)


def build_lora_config(cfg: WorkshopConfig):
    from peft import LoraConfig

    return LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=cfg.lora_target_modules, task_type="CAUSAL_LM")


def train_grpo(cfg: WorkshopConfig, train_dataset, reward_funcs, output_dir: str | Path,
               max_steps: int, heldout_records=None, heldout_every: int = 25, save: bool = True,
               **grpo_overrides):
    from transformers import AutoTokenizer
    from trl import GRPOTrainer

    output_dir = Path(output_dir)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    timer = StepTimer()
    callbacks = [timer]
    heldout = None
    if heldout_records:
        heldout = HeldOutEvalCallback(heldout_records, tok, every=heldout_every)
        callbacks.append(heldout)

    trainer = GRPOTrainer(
        model=cfg.model_id,
        reward_funcs=reward_funcs,
        args=build_grpo_config(cfg, output_dir, max_steps, **grpo_overrides),
        train_dataset=train_dataset,
        processing_class=tok,
        peft_config=build_lora_config(cfg),
        callbacks=callbacks,
    )
    trainer.train()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_dir))
        (output_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
        (output_dir / "timing.json").write_text(json.dumps(timer.summary(), indent=1))
        if heldout:
            (output_dir / "heldout_history.json").write_text(json.dumps(heldout.history, indent=1))
    trainer.heldout_history = heldout.history if heldout else []
    return trainer, timer


def main():
    from .dataset import generate_records, save_records, to_hf_dataset
    from .rewards_bad import BAD_REWARDS

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--reward", default="good", choices=["good", "partial", *BAD_REWARDS])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--heldout-every", type=int, default=25)
    a = p.parse_args()

    cfg = get_config()
    if a.lr:
        cfg.learning_rate = a.lr
    records = generate_records(cfg.n_train, cfg.n_val, cfg.n_test, cfg.seed)
    if not (cfg.data_dir / "train.jsonl").exists():  # don't rewrite while other runs/notebook read it
        save_records(records, cfg.data_dir)

    if a.reward == "good":
        from solutions.rewards_solution import REWARD_FUNCS as funcs
    elif a.reward == "partial":
        from solutions.rewards_solution import args_partial_reward, format_reward, tool_reward
        funcs = [format_reward, tool_reward, args_partial_reward]
    else:
        funcs = BAD_REWARDS[a.reward]

    out = a.output or str(cfg.prepared_run_dir if a.reward == "good" else cfg.output_dir / f"hack_{a.reward}")
    _, timer = train_grpo(cfg, to_hf_dataset(records["train"]), funcs, out, a.steps or cfg.prepared_max_steps,
                          heldout_records=records["val"], heldout_every=a.heldout_every)
    print("timing:", timer.summary())


if __name__ == "__main__":
    main()
