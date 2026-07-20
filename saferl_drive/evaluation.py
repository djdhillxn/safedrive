"""Closed-loop evaluation metrics for MetaDrive policies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import trange


SUCCESS_KEYS = ("success", "arrive_dest", "arrive_destination", "arrived_dest")
CRASH_KEYS = ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_sidewalk", "crash_human")
OUT_OF_ROAD_KEYS = ("out_of_road", "out_of_route")
MAX_STEP_KEYS = ("max_step", "timeout", "TimeLimit.truncated")
ENV_SEED_KEYS = ("env_seed", "scenario_index", "seed")


def _safe_bool(info: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        if key in info and bool(info[key]):
            return True
    return False


def _safe_float(info: dict[str, Any], keys: tuple[str, ...], default: float = np.nan) -> float:
    for key in keys:
        if key in info:
            try:
                return float(info[key])
            except (TypeError, ValueError):
                pass
    return default


def _safe_int(info: dict[str, Any], keys: tuple[str, ...], default: int = -1) -> int:
    for key in keys:
        if key in info:
            try:
                return int(info[key])
            except (TypeError, ValueError):
                pass
    return default


@dataclass
class EpisodeMetrics:
    episode: int
    env_seed: int
    return_sum: float
    length: int
    success: bool
    crash: bool
    out_of_road: bool
    max_step: bool
    cost_sum: float
    route_completion: float
    mean_speed_km_h: float
    final_info: dict[str, Any]


def evaluate_policy_closed_loop(
    model,
    env,
    episodes: int,
    deterministic: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate one SB3 policy in a single non-vector MetaDrive environment.

    This avoids relying on SB3's reward-only evaluator and collects AV-specific metrics.
    """
    rows: list[EpisodeMetrics] = []
    iterator = trange(episodes, desc="Evaluating", disable=not progress)
    for ep in iterator:
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_cost = 0.0
        speeds: list[float] = []
        route_completion = 0.0
        final_info: dict[str, Any] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_len += 1
            ep_cost += float(info.get("cost", 0.0) or 0.0)
            if "velocity" in info:
                try:
                    speeds.append(float(info["velocity"]))
                except Exception:
                    pass
            if "speed" in info:
                try:
                    speeds.append(float(info["speed"]))
                except Exception:
                    pass
            if "route_completion" in info:
                try:
                    route_completion = float(info["route_completion"])
                except Exception:
                    pass
            final_info = dict(info)

        row = EpisodeMetrics(
            episode=ep,
            env_seed=_safe_int(final_info, ENV_SEED_KEYS, default=-1),
            return_sum=ep_return,
            length=ep_len,
            success=_safe_bool(final_info, SUCCESS_KEYS),
            crash=_safe_bool(final_info, CRASH_KEYS),
            out_of_road=_safe_bool(final_info, OUT_OF_ROAD_KEYS),
            max_step=_safe_bool(final_info, MAX_STEP_KEYS),
            cost_sum=ep_cost,
            route_completion=_safe_float(final_info, ("route_completion",), default=route_completion),
            mean_speed_km_h=float(np.mean(speeds)) if speeds else np.nan,
            final_info=final_info,
        )
        rows.append(row)
    return pd.DataFrame([asdict(r) for r in rows])


def summarize_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate per-episode metrics into resume/report-ready numbers."""
    if df.empty:
        return {}
    summary = {
        "episodes": int(len(df)),
        "mean_return": float(df["return_sum"].mean()),
        "std_return": float(df["return_sum"].std(ddof=0)),
        "mean_length": float(df["length"].mean()),
        "success_rate": float(df["success"].mean()),
        "collision_rate": float(df["crash"].mean()),
        "out_of_road_rate": float(df["out_of_road"].mean()),
        "timeout_or_max_step_rate": float(df["max_step"].mean()),
        "mean_cost": float(df["cost_sum"].mean()),
        "mean_route_completion": float(df["route_completion"].mean()),
    }
    if "mean_speed_km_h" in df and df["mean_speed_km_h"].notna().any():
        summary["mean_speed_km_h"] = float(df["mean_speed_km_h"].mean())
    return summary


def save_eval_outputs(df: pd.DataFrame, out_dir: str | Path, prefix: str = "eval") -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}_episodes.csv"
    json_path = out / f"{prefix}_summary.json"
    df.to_csv(csv_path, index=False)
    summary = summarize_metrics(df)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return {"episodes_csv": csv_path, "summary_json": json_path}


def evaluate_policy_vecenv(
    model,
    venv,
    episodes: int,
    deterministic: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate a policy in a single-environment SB3 VecEnv.

    Use this when observations were normalized during training, because the VecNormalize
    wrapper must remain in the inference path.
    """
    if getattr(venv, "num_envs", 1) != 1:
        raise ValueError("evaluate_policy_vecenv expects a VecEnv with exactly one environment.")

    rows: list[EpisodeMetrics] = []
    obs = venv.reset()
    iterator = trange(episodes, desc="Evaluating", disable=not progress)
    for ep in iterator:
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_cost = 0.0
        speeds: list[float] = []
        route_completion = 0.0
        final_info: dict[str, Any] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = venv.step(action)
            reward = float(rewards[0])
            info = dict(infos[0])
            done = bool(dones[0])
            ep_return += reward
            ep_len += 1
            ep_cost += float(info.get("cost", 0.0) or 0.0)
            for speed_key in ("speed", "velocity"):
                if speed_key in info:
                    try:
                        speeds.append(float(info[speed_key]))
                    except Exception:
                        pass
            if "route_completion" in info:
                try:
                    route_completion = float(info["route_completion"])
                except Exception:
                    pass
            final_info = info

        # VecEnv auto-resets after done, so obs is already the next initial observation.
        row = EpisodeMetrics(
            episode=ep,
            env_seed=_safe_int(final_info, ENV_SEED_KEYS, default=-1),
            return_sum=ep_return,
            length=ep_len,
            success=_safe_bool(final_info, SUCCESS_KEYS),
            crash=_safe_bool(final_info, CRASH_KEYS),
            out_of_road=_safe_bool(final_info, OUT_OF_ROAD_KEYS),
            max_step=_safe_bool(final_info, MAX_STEP_KEYS),
            cost_sum=ep_cost,
            route_completion=_safe_float(final_info, ("route_completion",), default=route_completion),
            mean_speed_km_h=float(np.mean(speeds)) if speeds else np.nan,
            final_info=final_info,
        )
        rows.append(row)
    return pd.DataFrame([asdict(r) for r in rows])
