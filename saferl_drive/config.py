"""Configuration helpers for YAML experiments."""

import copy
import hashlib
import json
from pathlib import Path

import yaml


FINGERPRINT_SCHEMA_VERSION = 2

FINGERPRINT_ENVIRONMENT_KEYS = [
    "map",
    "map_config",
    "traffic_density",
    "traffic_mode",
    "random_traffic",
    "random_spawn_lane_index",
    "random_lane_num",
    "random_lane_width",
    "random_agent_model",
    "sequential_seed",
    "horizon",
    "truncate_as_terminate",
    "crash_vehicle_done",
    "crash_object_done",
    "out_of_road_done",
    "accident_prob",
]

FINGERPRINT_ACTION_KEYS = [
    "discrete_action",
    "use_multi_discrete",
    "discrete_steering_dim",
    "discrete_throttle_dim",
    "steering_limit",
]

FINGERPRINT_REWARD_KEYS = [
    "success_reward",
    "out_of_road_penalty",
    "crash_vehicle_penalty",
    "crash_object_penalty",
    "crash_sidewalk_penalty",
    "driving_reward",
    "speed_reward",
    "use_lateral_reward",
    "reward_shaping",
]


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


def _fingerprint_values(source, keys):
    return {key: copy.deepcopy(source.get(key)) for key in keys}


def _fingerprint_hash(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_experiment_fingerprint(cfg, split="test", episodes=None, controller=None):
    """Describe the exact evaluation task and policy-facing action interface."""
    evaluation = get_evaluation_config(cfg, split)
    environment = make_eval_metadrive_config(cfg, split)
    episode_count = int(episodes or evaluation.get("episodes", environment.get("num_scenarios", 1)))

    action = _fingerprint_values(environment, FINGERPRINT_ACTION_KEYS)
    action["kind"] = (
        "internal_policy"
        if controller
        else ("discrete" if bool(environment.get("discrete_action", False)) else "continuous")
    )
    if controller:
        action["controller"] = str(controller)

    task = {
        "environment": _fingerprint_values(environment, FINGERPRINT_ENVIRONMENT_KEYS),
        "reward": _fingerprint_values(environment, FINGERPRINT_REWARD_KEYS),
        "evaluation": {
            "split": split,
            "start_seed": int(evaluation.get("start_seed", environment.get("start_seed", 0))),
            "num_scenarios": int(
                evaluation.get("num_scenarios", environment.get("num_scenarios", 1))
            ),
            "episodes": episode_count,
            "deterministic": bool(evaluation.get("deterministic", True)),
        },
    }
    algorithm = copy.deepcopy(cfg.get("algorithm", {}))
    curriculum_stages = cfg.get("curriculum", {}).get("stages", [])
    final_gate = curriculum_stages[-1].get("gate", {}) if curriculum_stages else {}
    training = {
        "algorithm": algorithm.get("name"),
        "policy": algorithm.get("policy"),
        "kwargs": algorithm.get("kwargs", {}),
        "maximum_timesteps": int(
            cfg.get("curriculum", {}).get(
                "total_timesteps",
                cfg.get("train", {}).get("total_timesteps", 0),
            )
        ),
        "normalize_obs": bool(cfg.get("train", {}).get("normalize_obs", False)),
        "normalize_reward": bool(cfg.get("train", {}).get("normalize_reward", False)),
        "target_success_rate": cfg.get("train", {}).get(
            "stop_success_rate", final_gate.get("success_rate")
        ),
        "target_route_completion": cfg.get("train", {}).get(
            "stop_route_completion", final_gate.get("route_completion")
        ),
        "target_max_collision_rate": cfg.get("train", {}).get(
            "stop_max_collision_rate", final_gate.get("max_collision_rate")
        ),
        "target_max_out_of_road_rate": cfg.get("train", {}).get(
            "stop_max_out_of_road_rate", final_gate.get("max_out_of_road_rate")
        ),
        "target_max_timeout_rate": cfg.get("train", {}).get(
            "stop_max_timeout_rate", final_gate.get("max_timeout_rate")
        ),
    }
    strict = {**copy.deepcopy(task), "action": action, "training": training}
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "task_id": _fingerprint_hash(task),
        "strict_id": _fingerprint_hash(strict),
        "task": task,
        "action": action,
        "training": training,
    }


def fingerprint_differences(first, second, include_action=True):
    """Return readable field differences between two experiment fingerprints."""
    first_values = copy.deepcopy(first.get("task", {}))
    second_values = copy.deepcopy(second.get("task", {}))
    if include_action:
        first_values["action"] = copy.deepcopy(first.get("action", {}))
        second_values["action"] = copy.deepcopy(second.get("action", {}))
        first_values["training"] = copy.deepcopy(first.get("training", {}))
        second_values["training"] = copy.deepcopy(second.get("training", {}))

    differences = []

    def visit(left, right, prefix=""):
        keys = sorted(set(left) | set(right))
        for key in keys:
            name = f"{prefix}.{key}" if prefix else str(key)
            left_value = left.get(key)
            right_value = right.get(key)
            if isinstance(left_value, dict) and isinstance(right_value, dict):
                visit(left_value, right_value, name)
            elif left_value != right_value:
                differences.append(
                    {
                        "field": name,
                        "first": left_value,
                        "second": right_value,
                    }
                )

    visit(first_values, second_values)
    return differences
