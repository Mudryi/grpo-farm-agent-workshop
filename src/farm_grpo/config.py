"""Central configuration for the workshop.

Every number here is a *starting value to validate*, not a tuned optimum.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class WorkshopConfig:
    # ---- model -------------------------------------------------------------
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    seed: int = 42

    # ---- dataset sizes (teaching values) ----------------------------------
    n_train: int = 240
    n_val: int = 40
    n_test: int = 80

    # ---- LoRA ---------------------------------------------------------------
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ]
    )

    # ---- GRPO -----------------------------------------------------------------
    num_generations: int = 4           # G: completions per prompt
    max_completion_length: int = 96    # tokens
    per_device_train_batch_size: int = 16  # completions per step (= 4 prompts x 4 gens)
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    beta: float = 0.0                  # KL coefficient (0 = no reference model -> faster)
    temperature: float = 1.0           # sampling temperature during rollouts
    live_max_steps: int = 100          # live run in class (~5 min on DGX Spark @ ~3 s/step)
    prepared_max_steps: int = 500     # "prepared" checkpoint trained before class (~25 min)
    logging_steps: int = 1

    # ---- evaluation -------------------------------------------------------------
    eval_batch_size: int = 32
    eval_max_new_tokens: int = 128

    # ---- paths --------------------------------------------------------------------
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR

    @property
    def live_run_dir(self) -> Path:
        return self.output_dir / "live_run"

    @property
    def prepared_run_dir(self) -> Path:
        return self.output_dir / "prepared_run"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["output_dir"] = str(self.output_dir)
        return d


def get_config(**overrides) -> WorkshopConfig:
    cfg = WorkshopConfig(**overrides)
    if os.environ.get("FARM_GRPO_MODEL"):
        cfg.model_id = os.environ["FARM_GRPO_MODEL"]
    return cfg
