"""Model loading, batched generation and evaluation metrics."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import torch

from .dataset import Record
from .environment import normalize_args
from .parsing import parse_tool_call
from .prompts import SYSTEM_PROMPT, build_messages


# --------------------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------------------
def best_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_model(model_id: str, adapter_path: str | None = None, device: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=best_dtype()).to(device)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, messages_list: list[list[dict]], max_new_tokens: int = 128,
             batch_size: int = 32, do_sample: bool = False, temperature: float = 1.0,
             num_return_sequences: int = 1) -> list[list[str]]:
    """Return, for every prompt, a list of `num_return_sequences` completions."""
    tok.padding_side = "left"
    outs: list[list[str]] = []
    for i in range(0, len(messages_list), batch_size):
        batch = messages_list[i:i + batch_size]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                          num_return_sequences=num_return_sequences)
        if do_sample:
            gen_kwargs |= dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        else:
            gen_kwargs |= dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        ids = model.generate(**enc, **gen_kwargs)
        dec = tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j in range(len(batch)):
            outs.append(dec[j * num_return_sequences:(j + 1) * num_return_sequences])
    return outs


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def score_completion(text: str, expected_tool: str, expected_args: str | dict) -> dict:
    """Per-completion metrics + error category + the teaching reward."""
    if isinstance(expected_args, str):
        expected_args = json.loads(expected_args)
    target = normalize_args(expected_tool, expected_args)
    pc = parse_tool_call(text)

    correct_tool = pc.has_schema and pc.tool == expected_tool
    correct_args = correct_tool and pc.norm_args is not None and pc.norm_args == target
    multiple = pc.n_calls > 1

    if not pc.is_json:
        error = "multiple_calls" if multiple else pc.error  # invalid_json / extra_text / not_an_object
    elif not pc.has_schema:
        error = "bad_schema"
    elif not correct_tool:
        error = "wrong_tool"
    elif pc.norm_args is None:
        error = "invalid_arguments"
    elif not correct_args:
        error = "wrong_args"
    else:
        error = "ok"

    reward = (0.1 if pc.has_schema else -0.2) + (0.2 if correct_tool else 0) + (0.7 if correct_args else 0)
    return {
        "valid_json": pc.has_schema,
        "valid_call": pc.valid_call,
        "correct_tool": correct_tool,
        "correct_args": correct_args,
        "task_success": correct_args and not multiple,
        "multiple_calls": multiple,
        "error": error,
        "reward": round(reward, 3),
        "predicted_tool": pc.tool,
        "predicted_args": json.dumps(pc.arguments) if pc.arguments is not None else None,
    }


METRIC_NAMES = {
    "valid_json": "Valid tool-call JSON (%)",
    "correct_tool": "Correct tool selection (%)",
    "correct_args": "Correct arguments (%)",
    "task_success": "Overall task success (%)",
    "multiple_calls": "Unnecessary tool calls (%)",
}


def evaluate(model, tok, records: list[Record], batch_size: int = 32,
             max_new_tokens: int = 128) -> tuple[dict, pd.DataFrame]:
    msgs = [build_messages(r.question, SYSTEM_PROMPT) for r in records]
    comps = [c[0] for c in generate(model, tok, msgs, max_new_tokens, batch_size)]
    rows = []
    for r, text in zip(records, comps):
        s = score_completion(text, r.expected_tool, r.expected_args)
        rows.append({"id": r.id, "family": r.family, "question": r.question,
                     "expected_tool": r.expected_tool, "expected_args": r.expected_args,
                     "completion": text, **s})
    df = pd.DataFrame(rows)
    return summarize(df), df


def summarize(df: pd.DataFrame) -> dict:
    m = {label: round(float(100 * df[col].mean()), 1) for col, label in METRIC_NAMES.items()}
    m["Mean reward"] = round(float(df["reward"].mean()), 3)
    return m


def compare(results: dict[str, dict]) -> pd.DataFrame:
    """results = {"Before GRPO": metrics, "After GRPO": metrics, ...} -> table."""
    return pd.DataFrame(results)


def error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["expected_tool", "error"]).size().unstack(fill_value=0)
            .assign(total=lambda x: x.sum(axis=1)))


# --------------------------------------------------------------------------------------
# GRPO signal check: does the base model produce *varied* rewards inside a group?
# --------------------------------------------------------------------------------------
def reward_variance_check(model, tok, records: list[Record], num_generations: int = 4,
                          temperature: float = 1.0, max_new_tokens: int = 96,
                          batch_size: int = 16) -> tuple[dict, pd.DataFrame]:
    msgs = [build_messages(r.question, SYSTEM_PROMPT) for r in records]
    groups = generate(model, tok, msgs, max_new_tokens, batch_size, do_sample=True,
                      temperature=temperature, num_return_sequences=num_generations)
    rows = []
    for r, comps in zip(records, groups):
        rewards = [score_completion(c, r.expected_tool, r.expected_args)["reward"] for c in comps]
        rows.append({"id": r.id, "question": r.question, "rewards": rewards,
                     "mean": float(np.mean(rewards)), "std": float(np.std(rewards)),
                     "completions": comps})
    df = pd.DataFrame(rows)
    zero = df["std"] < 1e-6
    summary = {
        "groups": len(df),
        "groups with zero reward std (%)": round(float(100 * zero.mean()), 1),
        "  ...all correct (%)": round(float(100 * (zero & (df["mean"] >= 0.999)).mean()), 1),
        "  ...all equally wrong (%)": round(float(100 * (zero & (df["mean"] < 0.999)).mean()), 1),
        "mean reward": round(float(df["mean"].mean()), 3),
        "reward distribution": dict(Counter(x for rs in df["rewards"] for x in rs).most_common()),
    }
    return summary, df
