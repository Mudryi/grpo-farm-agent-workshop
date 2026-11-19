"""Plots for the workshop notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, RED, GREY = "#2a6fdb", "#e8833a", "#2e9e5b", "#d64545", "#8a8f98"


def group_advantages(rewards, eps: float = 1e-4) -> np.ndarray:
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


def plot_advantages(groups: dict[str, list[float]]):
    """Bar chart of rewards and the resulting group-normalised advantages, one panel per group."""
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.2), squeeze=False)
    for ax, (title, rewards) in zip(axes[0], groups.items()):
        adv = group_advantages(rewards)
        x = np.arange(len(rewards))
        ax.bar(x - 0.2, rewards, 0.4, label="reward $r_i$", color=BLUE)
        ax.bar(x + 0.2, adv, 0.4, label="advantage $A_i$",
               color=[GREEN if a > 0 else RED if a < 0 else GREY for a in adv])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [f"o{i + 1}" for i in x])
        ax.set_title(title, fontsize=10)
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def plot_training(log_history: list[dict], heldout: list[dict] | None = None, title: str = ""):
    logs = pd.DataFrame([h for h in log_history if "reward" in h])
    n = 3 if heldout else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.4))
    ax = axes[0]
    ax.plot(logs["step"], logs["reward"], color=GREY, alpha=0.4, label="mean reward (per step)")
    ax.plot(logs["step"], logs["reward"].rolling(10, min_periods=1).mean(), color=BLUE, lw=2,
            label="mean reward (rolling 10)")
    ax.fill_between(logs["step"], logs["reward"] - logs["reward_std"], logs["reward"] + logs["reward_std"],
                    color=BLUE, alpha=0.08, label="± reward std")
    ax.set_ylim(-0.3, 1.1)
    ax.set_xlabel("step")
    ax.set_title("Training reward")
    ax.legend(fontsize=8)

    ax = axes[1]
    for col, c in [("rewards/format_reward/mean", GREY), ("rewards/tool_reward/mean", ORANGE),
                   ("rewards/args_reward/mean", GREEN)]:
        if col in logs:
            ax.plot(logs["step"], logs[col].rolling(10, min_periods=1).mean(), color=c,
                    label=col.split("/")[1])
    if "frac_reward_zero_std" in logs:
        ax.plot(logs["step"], logs["frac_reward_zero_std"].rolling(10, min_periods=1).mean(),
                color=RED, ls="--", label="frac groups with zero std")
    ax.set_xlabel("step")
    ax.set_title("Reward components / no-signal groups")
    ax.legend(fontsize=8)

    if heldout:
        h = pd.DataFrame(heldout)
        ax = axes[2]
        ax.plot(h["step"], h["Overall task success (%)"], "o-", color=GREEN, label="task success")
        ax.plot(h["step"], h["Valid tool-call JSON (%)"], "s--", color=GREY, label="valid JSON")
        ax.set_ylim(0, 105)
        ax.set_xlabel("step")
        ax.set_title("Held-out (validation) %")
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_comparison(table: pd.DataFrame):
    """table: rows = metrics, columns = model variants (output of evaluate.compare)."""
    t = table.drop(index=[i for i in table.index if "reward" in i.lower()], errors="ignore")
    ax = t.plot.barh(figsize=(8, 0.6 * len(t) + 1.5), color=[GREY, BLUE, GREEN, ORANGE, RED][:t.shape[1]])
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.set_xlabel("%")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    return ax.figure


def training_summary(log_history: list[dict], heldout: list[dict] | None = None, window: int = 10) -> str:
    """1-3 sentence summary of a run: reward / no-signal groups at start vs end, held-out success."""
    logs = pd.DataFrame([h for h in log_history if "reward" in h])
    first, last = logs.head(window), logs.tail(window)
    lines = [f"Mean training reward rose from {first['reward'].mean():.2f} (first {window} steps) "
             f"to {last['reward'].mean():.2f} (last {window} steps)."]
    if "frac_reward_zero_std" in logs:
        lines.append(f"Groups with zero reward std (no learning signal): {first['frac_reward_zero_std'].mean():.0%} "
                     f"-> {last['frac_reward_zero_std'].mean():.0%}.")
    if heldout:
        h = [x["Overall task success (%)"] for x in heldout]
        lines.append(f"Held-out task success: {h[0]:.1f}% at step 0 -> {h[-1]:.1f}% at the end (best {max(h):.1f}%).")
    return "\n".join(lines)
