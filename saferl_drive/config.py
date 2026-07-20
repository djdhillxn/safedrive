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


def make_eval_metadrive_config(cfg):
    """Merge the main MetaDrive config with eval-specific overrides."""
    env_cfg = copy.deepcopy(cfg.get("metadrive", {}))
    eval_cfg = cfg.get("eval", {})
    for key in ["start_seed", "num_scenarios", "traffic_density", "horizon"]:
        if key in eval_cfg:
            env_cfg[key] = eval_cfg[key]
    env_cfg["use_render"] = False
    env_cfg.setdefault("log_level", 50)
    return env_cfg
