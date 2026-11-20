"""GRPO + LoRA training with Hugging Face TRL.

    trainer = train_grpo(cfg, train_ds, reward_funcs, output_dir, max_steps=100, heldout_records=records["val"])

In class the runs start as background processes, see background.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import TrainerCallback

from .config import WorkshopConfig
from .evaluate import best_dtype


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
               max_steps: int, heldout_records, heldout_every: int = 25, **grpo_overrides):
    """Train, then save the LoRA adapter, log_history.json and heldout_history.json to output_dir."""
    from transformers import AutoTokenizer
    from trl import GRPOTrainer

    output_dir = Path(output_dir)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    heldout = HeldOutEvalCallback(heldout_records, tok, every=heldout_every)
    trainer = GRPOTrainer(
        model=cfg.model_id,
        reward_funcs=reward_funcs,
        args=build_grpo_config(cfg, output_dir, max_steps, **grpo_overrides),
        train_dataset=train_dataset,
        processing_class=tok,
        peft_config=build_lora_config(cfg),
        callbacks=[heldout],
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    (output_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (output_dir / "heldout_history.json").write_text(json.dumps(heldout.history, indent=1))
    return trainer
