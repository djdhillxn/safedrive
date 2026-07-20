"""General utilities."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch may not be imported/installed while inspecting utilities.
        pass


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(output_dir: str | Path, experiment_name: str, algo: str, seed: int) -> Path:
    run_dir = Path(output_dir) / f"{timestamp()}_{experiment_name}_{algo}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for subdir in ["models", "checkpoints", "logs", "eval", "videos", "plots"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out
