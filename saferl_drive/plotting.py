"""Plotting utilities for training and evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def load_monitor_csvs(run_dir: str | Path) -> pd.DataFrame:
    """Load SB3 Monitor CSV files under a run directory."""
    run_dir = Path(run_dir)
    frames = []
    for path in run_dir.rglob("*.monitor.csv"):
        try:
            df = pd.read_csv(path, comment="#")
            df["source"] = str(path.relative_to(run_dir))
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "t" in df.columns:
        df = df.sort_values("t").reset_index(drop=True)
    return df


def plot_training_returns(run_dir: str | Path, smoothing: int = 20) -> Path:
    """Plot episode return from Monitor files."""
    run_dir = Path(run_dir)
    out_path = run_dir / "plots" / "training_returns.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_monitor_csvs(run_dir)
    if df.empty or "r" not in df.columns:
        raise FileNotFoundError(f"No monitor return data found under {run_dir}")
    y = df["r"].astype(float)
    y_smooth = y.rolling(window=smoothing, min_periods=1).mean()

    plt.figure(figsize=(9, 5))
    plt.plot(range(len(y)), y, alpha=0.35, label="episode return")
    plt.plot(range(len(y_smooth)), y_smooth, label=f"rolling mean ({smoothing})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training episode return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_eval_summary(eval_csv: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    """Create simple evaluation plots from per-episode CSV."""
    eval_csv = Path(eval_csv)
    df = pd.read_csv(eval_csv)
    if out_dir is None:
        out_dir = eval_csv.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Route completion distribution.
    if "route_completion" in df.columns:
        path = out_dir / "eval_route_completion.png"
        plt.figure(figsize=(7, 4))
        plt.hist(df["route_completion"].dropna(), bins=15)
        plt.xlabel("Route completion")
        plt.ylabel("Episodes")
        plt.title("Unseen-scenario route completion")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    # Outcome rates.
    rate_cols = [c for c in ["success", "crash", "out_of_road", "max_step"] if c in df.columns]
    if rate_cols:
        values = [float(df[c].mean()) for c in rate_cols]
        path = out_dir / "eval_outcome_rates.png"
        plt.figure(figsize=(7, 4))
        plt.bar(rate_cols, values)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Closed-loop outcome rates")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    return paths


def compare_eval_summaries(summary_paths: Iterable[str | Path], out_path: str | Path) -> Path:
    """Plot multiple eval summary JSON files side by side."""
    rows = []
    for p in summary_paths:
        p = Path(p)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["run"] = p.parents[1].name if len(p.parents) > 1 else p.stem
        rows.append(data)
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = [
        "success_rate",
        "collision_rate",
        "out_of_road_rate",
        "mean_route_completion",
    ]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise ValueError("No comparable metrics found.")

    x = range(len(df))
    width = 0.8 / len(metrics)
    plt.figure(figsize=(10, 5))
    for j, metric in enumerate(metrics):
        plt.bar([i + j * width for i in x], df[metric], width=width, label=metric)
    plt.xticks([i + width * (len(metrics) - 1) / 2 for i in x], df["run"], rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Value")
    plt.title("Evaluation summary comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path
