"""Closed-loop evaluation metrics for MetaDrive policies."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

from saferl_drive.utils import json_text, write_json


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
    direct = _safe_float(info, ("mean_speed_km_h", "speed_km_h"))
    if not np.isnan(direct):
        return direct
    speed_m_s = _safe_float(info, ("speed", "velocity"))
    if np.isnan(speed_m_s):
        return np.nan
    return speed_m_s * 3.6


def _episode_seed(start_seed, num_scenarios, episode):
    if start_seed is None:
        return None
    scenario_count = max(int(num_scenarios or 1), 1)
    return int(start_seed) + episode % scenario_count


def _episode_row(
    episode,
    requested_seed,
    return_sum,
    length,
    cost_sum,
    speeds,
    route_completion,
    final_info,
):
    return {
        "episode": episode,
        "env_seed": _safe_int(
            final_info,
            ENV_SEED_KEYS,
            default=requested_seed if requested_seed is not None else -1,
        ),
        "return_sum": return_sum,
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
            observation, info = env.reset(seed=requested_seed)
        done = False
        return_sum = 0.0
        length = 0
        cost_sum = 0.0
        speeds = []
        route_completion = 0.0
        final_info = dict(info)

        while not done:
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            return_sum += float(reward)
            length += 1
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
                length,
                cost_sum,
                speeds,
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
            venv.seed(requested_seed)
        observation = venv.reset()
        done = False
        return_sum = 0.0
        length = 0
        cost_sum = 0.0
        speeds = []
        route_completion = 0.0
        final_info = {}

        while not done:
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, rewards, dones, infos = venv.step(action)
            info = dict(infos[0])
            done = bool(dones[0])
            return_sum += float(rewards[0])
            length += 1
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
                length,
                cost_sum,
                speeds,
                route_completion,
                final_info,
            )
        )
    return pd.DataFrame(rows)


def summarize_metrics(frame):
    """Aggregate per-episode metrics into report-ready numbers."""
    if frame.empty:
        return {}
    summary = {
        "episodes": int(len(frame)),
        "mean_return": float(frame["return_sum"].mean()),
        "std_return": float(frame["return_sum"].std(ddof=0)),
        "mean_length": float(frame["length"].mean()),
        "success_rate": float(frame["success"].mean()),
        "collision_rate": float(frame["crash"].mean()),
        "out_of_road_rate": float(frame["out_of_road"].mean()),
        "timeout_or_max_step_rate": float(frame["max_step"].mean()),
        "mean_cost": float(frame["cost_sum"].mean()),
        "mean_route_completion": float(frame["route_completion"].mean()),
    }
    if "mean_speed_km_h" in frame and frame["mean_speed_km_h"].notna().any():
        summary["mean_speed_km_h"] = float(frame["mean_speed_km_h"].mean())
    return summary


def comparison_summary_row(name, summary, run_dir=None, summary_path=None):
    """Return one stable row for Phase-1 CSV, JSON, and plotting."""
    metric_names = [
        "episodes",
        "mean_return",
        "std_return",
        "mean_length",
        "success_rate",
        "collision_rate",
        "out_of_road_rate",
        "timeout_or_max_step_rate",
        "mean_cost",
        "mean_route_completion",
        "mean_speed_km_h",
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
