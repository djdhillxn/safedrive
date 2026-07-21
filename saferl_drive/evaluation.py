"""Closed-loop evaluation metrics for MetaDrive policies."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

from saferl_drive.utils import json_text, set_global_seeds, write_json


SUCCESS_KEYS = ("success", "arrive_dest", "arrive_destination", "arrived_dest")
CRASH_KEYS = (
    "crash",
    "crash_vehicle",
    "crash_object",
    "crash_building",
    "crash_sidewalk",
    "crash_human",
)
OUT_OF_ROAD_KEYS = ("out_of_road", "out_of_route")
MAX_STEP_KEYS = ("max_step", "timeout", "TimeLimit.truncated")
ENV_SEED_KEYS = ("env_seed", "scenario_index", "seed")


def _safe_bool(info, keys):
    for key in keys:
        if key in info and bool(info[key]):
            return True
    return False


def _safe_float(info, keys, default=np.nan):
    for key in keys:
        if key in info:
            try:
                return float(info[key])
            except (TypeError, ValueError):
                pass
    return default


def _safe_int(info, keys, default=-1):
    for key in keys:
        if key in info:
            try:
                return int(info[key])
            except (TypeError, ValueError):
                pass
    return default


def _speed_km_h(info):
    # MetaDrive reports ``velocity`` in km/h. Generic ``speed`` fields are
    # treated as m/s so adapters for other Gymnasium environments still work.
    direct = _safe_float(info, ("mean_speed_km_h", "speed_km_h", "velocity"))
    if not np.isnan(direct):
        return direct
    speed_m_s = _safe_float(info, ("speed_m_s", "speed"))
    if np.isnan(speed_m_s):
        return np.nan
    return speed_m_s * 3.6


def _episode_seed(start_seed, num_scenarios, episode):
    if start_seed is None:
        return None
    scenario_count = max(int(num_scenarios or 1), 1)
    return int(start_seed) + episode % scenario_count


def _recorded_action(info, requested_action):
    """Prefer the action MetaDrive actually applied over the requested action."""
    candidate = info.get("action", info.get("raw_action", requested_action))
    try:
        values = np.asarray(candidate, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < 2:
        return None
    return values[:2]


def _episode_row(
    episode,
    requested_seed,
    return_sum,
    base_return_sum,
    shaping_penalty_sum,
    length,
    cost_sum,
    speeds,
    actions,
    route_completion,
    final_info,
):
    row = {
        "episode": episode,
        "env_seed": _safe_int(
            final_info,
            ENV_SEED_KEYS,
            default=requested_seed if requested_seed is not None else -1,
        ),
        "return_sum": return_sum,
        "base_return_sum": base_return_sum,
        "shaping_penalty_sum": shaping_penalty_sum,
        "length": length,
        "success": _safe_bool(final_info, SUCCESS_KEYS),
        "crash": _safe_bool(final_info, CRASH_KEYS),
        "out_of_road": _safe_bool(final_info, OUT_OF_ROAD_KEYS),
        "max_step": _safe_bool(final_info, MAX_STEP_KEYS),
        "cost_sum": cost_sum,
        "route_completion": _safe_float(
            final_info,
            ("route_completion",),
            default=route_completion,
        ),
        "mean_speed_km_h": float(np.mean(speeds)) if speeds else np.nan,
        "final_info": json_text(final_info),
    }
    if actions:
        action_array = np.asarray(actions, dtype=float)
        steering = action_array[:, 0]
        throttle_brake = action_array[:, 1]
        row.update(
            {
                "mean_steering": float(np.mean(steering)),
                "mean_abs_steering": float(np.mean(np.abs(steering))),
                "steering_saturation_rate": float(np.mean(np.abs(steering) >= 0.95)),
                "mean_throttle_brake": float(np.mean(throttle_brake)),
                "throttle_rate": float(np.mean(throttle_brake > 0.05)),
                "brake_rate": float(np.mean(throttle_brake < -0.05)),
            }
        )
        if len(action_array) > 1:
            row["mean_action_change"] = float(np.mean(np.abs(np.diff(action_array, axis=0))))
        else:
            row["mean_action_change"] = 0.0
    return row


def evaluate_policy_closed_loop(
    model,
    env,
    episodes,
    deterministic=True,
    progress=True,
    start_seed=None,
    num_scenarios=None,
):
    """Evaluate one policy in a single non-vector MetaDrive environment."""
    rows = []
    iterator = trange(episodes, desc="Evaluating", disable=not progress)
    for episode in iterator:
        requested_seed = _episode_seed(start_seed, num_scenarios, episode)
        if requested_seed is None:
            observation, info = env.reset()
        else:
            set_global_seeds(requested_seed)
            observation, info = env.reset(seed=requested_seed)
        done = False
        return_sum = 0.0
        base_return_sum = 0.0
        shaping_penalty_sum = 0.0
        length = 0
        cost_sum = 0.0
        speeds = []
        actions = []
        route_completion = 0.0
        final_info = dict(info)

        while not done:
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            return_sum += float(reward)
            base_return_sum += _safe_float(info, ("base_reward",), default=float(reward))
            shaping_penalty_sum += _safe_float(
                info,
                ("shaping_penalty",),
                default=0.0,
            )
            length += 1
            action_values = _recorded_action(info, action)
            if action_values is not None:
                actions.append(action_values)
            cost_sum += _safe_float(info, ("cost",), default=0.0)
            speed = _speed_km_h(info)
            if not np.isnan(speed):
                speeds.append(speed)
            route_completion = _safe_float(
                info,
                ("route_completion",),
                default=route_completion,
            )
            final_info = dict(info)

        rows.append(
            _episode_row(
                episode,
                requested_seed,
                return_sum,
                base_return_sum,
                shaping_penalty_sum,
                length,
                cost_sum,
                speeds,
                actions,
                route_completion,
                final_info,
            )
        )
    return pd.DataFrame(rows)


def evaluate_policy_vecenv(
    model,
    venv,
    episodes,
    deterministic=True,
    progress=True,
    start_seed=None,
    num_scenarios=None,
):
    """Evaluate a policy in a single-environment SB3 VecEnv."""
    if getattr(venv, "num_envs", 1) != 1:
        raise ValueError("evaluate_policy_vecenv expects a VecEnv with exactly one environment.")

    rows = []
    iterator = trange(episodes, desc="Evaluating", disable=not progress)
    for episode in iterator:
        requested_seed = _episode_seed(start_seed, num_scenarios, episode)
        if requested_seed is not None:
            set_global_seeds(requested_seed)
            venv.seed(requested_seed)
        observation = venv.reset()
        done = False
        return_sum = 0.0
        base_return_sum = 0.0
        shaping_penalty_sum = 0.0
        length = 0
        cost_sum = 0.0
        speeds = []
        actions = []
        route_completion = 0.0
        final_info = {}

        while not done:
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, rewards, dones, infos = venv.step(action)
            info = dict(infos[0])
            done = bool(dones[0])
            reward = float(rewards[0])
            return_sum += reward
            base_return_sum += _safe_float(info, ("base_reward",), default=reward)
            shaping_penalty_sum += _safe_float(
                info,
                ("shaping_penalty",),
                default=0.0,
            )
            length += 1
            action_values = _recorded_action(info, action)
            if action_values is not None:
                actions.append(action_values)
            cost_sum += _safe_float(info, ("cost",), default=0.0)
            speed = _speed_km_h(info)
            if not np.isnan(speed):
                speeds.append(speed)
            route_completion = _safe_float(
                info,
                ("route_completion",),
                default=route_completion,
            )
            final_info = info

        rows.append(
            _episode_row(
                episode,
                requested_seed,
                return_sum,
                base_return_sum,
                shaping_penalty_sum,
                length,
                cost_sum,
                speeds,
                actions,
                route_completion,
                final_info,
            )
        )
    return pd.DataFrame(rows)


def _wilson_interval(successes, episodes):
    """Return a 95% Wilson interval for a binomial success rate."""
    if episodes <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    rate = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (rate + z * z / (2.0 * episodes)) / denominator
    margin = (
        z
        * np.sqrt(rate * (1.0 - rate) / episodes + z * z / (4.0 * episodes * episodes))
        / denominator
    )
    return float(center - margin), float(center + margin)


def summarize_metrics(frame):
    """Aggregate per-episode metrics into report-ready numbers."""
    if frame.empty:
        return {}
    episodes = int(len(frame))
    successes = int(frame["success"].sum())
    success_low, success_high = _wilson_interval(successes, episodes)
    summary = {
        "episodes": episodes,
        "mean_return": float(frame["return_sum"].mean()),
        "std_return": float(frame["return_sum"].std(ddof=0)),
        "mean_length": float(frame["length"].mean()),
        "success_rate": float(frame["success"].mean()),
        "success_rate_95ci_low": success_low,
        "success_rate_95ci_high": success_high,
        "collision_rate": float(frame["crash"].mean()),
        "out_of_road_rate": float(frame["out_of_road"].mean()),
        "timeout_or_max_step_rate": float(frame["max_step"].mean()),
        "mean_cost": float(frame["cost_sum"].mean()),
        "mean_route_completion": float(frame["route_completion"].mean()),
    }
    optional_metrics = {
        "base_return_sum": "mean_base_return",
        "shaping_penalty_sum": "mean_shaping_penalty",
        "mean_steering": "mean_steering",
        "mean_abs_steering": "mean_abs_steering",
        "steering_saturation_rate": "steering_saturation_rate",
        "mean_throttle_brake": "mean_throttle_brake",
        "throttle_rate": "throttle_rate",
        "brake_rate": "brake_rate",
        "mean_action_change": "mean_action_change",
    }
    for column, summary_name in optional_metrics.items():
        if column in frame and frame[column].notna().any():
            summary[summary_name] = float(frame[column].mean())
    if "mean_speed_km_h" in frame and frame["mean_speed_km_h"].notna().any():
        summary["mean_speed_km_h"] = float(frame["mean_speed_km_h"].mean())
    return summary


def checkpoint_selection_score(summary):
    """Rank validation checkpoints by task success, then useful safe progress."""
    return (
        float(summary.get("success_rate", 0.0)),
        float(summary.get("mean_route_completion", 0.0)),
        -float(summary.get("out_of_road_rate", 1.0)),
        -float(summary.get("collision_rate", 1.0)),
        -float(summary.get("timeout_or_max_step_rate", 1.0)),
        float(summary.get("mean_return", float("-inf"))),
    )


def comparison_summary_row(name, summary, run_dir=None, summary_path=None):
    """Return one stable row for Phase-1 CSV, JSON, and plotting."""
    metric_names = [
        "episodes",
        "mean_return",
        "std_return",
        "mean_length",
        "success_rate",
        "success_rate_95ci_low",
        "success_rate_95ci_high",
        "collision_rate",
        "out_of_road_rate",
        "timeout_or_max_step_rate",
        "mean_cost",
        "mean_route_completion",
        "mean_speed_km_h",
        "mean_base_return",
        "mean_shaping_penalty",
        "mean_steering",
        "mean_abs_steering",
        "steering_saturation_rate",
        "mean_throttle_brake",
        "throttle_rate",
        "brake_rate",
        "mean_action_change",
    ]
    row = {"name": name}
    for metric in metric_names:
        row[metric] = summary.get(metric)
    if run_dir is not None:
        row["run_dir"] = str(run_dir)
    if summary_path is not None:
        row["summary_path"] = str(summary_path)
    return row


def save_eval_outputs(frame, out_dir, prefix="eval"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}_episodes.csv"
    json_path = out_dir / f"{prefix}_summary.json"
    frame.to_csv(csv_path, index=False)
    write_json(summarize_metrics(frame), json_path)
    return {"episodes_csv": csv_path, "summary_json": json_path}
