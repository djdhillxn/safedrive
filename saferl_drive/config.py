"""Configuration helpers for YAML experiments."""

import copy
from pathlib import Path

import yaml


def load_yaml(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping, got {type(data)}")
    return data


def save_yaml(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def deep_update(base, updates):
    """Recursively update a copy of base with updates."""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_scalar(value):
    """Parse a CLI override scalar using YAML rules."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_dotlist_overrides(cfg, overrides):
    """Apply overrides like ['train.total_timesteps=10000', 'metadrive.traffic_density=0.2']."""
    if not overrides:
        return cfg
    result = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        parts = key.split(".")
        cursor = result
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = _parse_scalar(raw_value)
    return result


def get_evaluation_config(cfg, split="test"):
    """Return settings for validation or the held-out test split.

    Older saved runs used an ``eval`` section. Keeping that fallback lets us
    audit those models without editing their immutable resolved configs.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(
            f"Unknown evaluation split {split!r}; use 'train', 'validation', or 'test'."
        )
    if split == "train":
        evaluation = copy.deepcopy(cfg.get("metadrive", {}))
        evaluation["random_traffic"] = False
        evaluation["random_spawn_lane_index"] = False
        evaluation.setdefault("episodes", int(evaluation.get("num_scenarios", 1)))
        evaluation.setdefault("deterministic", True)
        evaluation.setdefault("progress", True)
        evaluation.setdefault("vec_env", "subproc")
        return evaluation
    evaluation = copy.deepcopy(cfg.get("eval", {}))
    if split in cfg:
        evaluation.update(copy.deepcopy(cfg[split]))
    return evaluation


def make_eval_metadrive_config(cfg, split="test"):
    """Merge MetaDrive settings with one deterministic evaluation split."""
    env_cfg = copy.deepcopy(cfg.get("metadrive", {}))
    eval_cfg = get_evaluation_config(cfg, split)
    override_keys = [
        "start_seed",
        "num_scenarios",
        "traffic_density",
        "random_traffic",
        "random_spawn_lane_index",
        "horizon",
        "map",
    ]
    for key in override_keys:
        if key in eval_cfg:
            env_cfg[key] = eval_cfg[key]
    env_cfg["use_render"] = False
    env_cfg.setdefault("log_level", 50)
    if "random_traffic" not in eval_cfg:
        env_cfg["random_traffic"] = False
    if "random_spawn_lane_index" not in eval_cfg:
        env_cfg["random_spawn_lane_index"] = False
    return env_cfg
