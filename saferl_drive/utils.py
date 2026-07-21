"""General, logging, artifact, and plotting utilities."""

import importlib.metadata
import json
import logging
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DISTRIBUTIONS = {
    "MetaDrive": "metadrive-simulator",
    "Stable-Baselines3": "stable-baselines3",
    "Gymnasium": "gymnasium",
    "PyTorch": "torch",
    "NumPy": "numpy",
    "pandas": "pandas",
}


class _ConciseConsoleFormatter(logging.Formatter):
    """Keep exception tracebacks in files without flooding notebook output."""

    def format(self, record):
        exception_info = record.exc_info
        exception_text = record.exc_text
        record.exc_info = None
        record.exc_text = None
        try:
            return super().format(record)
        finally:
            record.exc_info = exception_info
            record.exc_text = exception_text


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch may not be imported or installed while inspecting utilities.
        pass


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def make_run_dir(output_dir, experiment_name, algo, seed):
    name_parts = [timestamp(), experiment_name]
    if algo:
        name_parts.append(algo)
    name_parts.append(f"seed{seed}")
    run_dir = Path(output_dir) / "_".join(name_parts)
    run_dir.mkdir(parents=True, exist_ok=False)
    for subdir in ["models", "checkpoints", "logs", "eval", "videos", "plots"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def flatten_dict(d, prefix=""):
    out = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, new_key))
        else:
            out[new_key] = value
    return out


def _logging_level(value):
    if isinstance(value, int):
        return value
    level = getattr(logging, str(value).upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown logging level: {value}")
    return level


def setup_logging(log_path, console_level="INFO", file_level="DEBUG"):
    """Create one concise console logger and one detailed file logger."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = f"saferl_drive.{log_path.resolve()}"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(_logging_level(console_level))
    console.setFormatter(_ConciseConsoleFormatter("%(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(_logging_level(file_level))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.debug("Logging initialized: %s", log_path.resolve())
    return logger


def _package_versions():
    versions = {}
    for label, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "not installed"
    return versions


def _git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "unavailable"


def log_system_info(logger, run_dir=None):
    """Log reproducibility details and return them for run metadata."""
    cuda_available = False
    gpu_name = "none"
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "PyTorch unavailable"

    info = {
        "timestamp_utc": utc_timestamp(),
        "working_directory": str(Path.cwd().resolve()),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "packages": _package_versions(),
    }
    if run_dir is not None:
        info["run_directory"] = str(Path(run_dir).resolve())

    logger.info(
        "Runtime: Python %s | CUDA %s | GPU %s",
        sys.version.split()[0],
        "available" if cuda_available else "unavailable",
        gpu_name,
    )
    for key, value in info.items():
        logger.debug("System %s: %s", key, value)
    return info


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_text(data):
    return json.dumps(
        _json_ready(data),
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    )


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            _json_ready(data),
            file,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        file.write("\n")
    return path


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _portable_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def update_latest_run_file(output_dir, algo_or_name, run_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pointer = output_dir / f"latest_{algo_or_name}.txt"
    pointer.write_text(f"{_portable_path(run_dir)}\n", encoding="utf-8")
    return pointer


def read_latest_run(output_dir, algo_or_name):
    pointer = Path(output_dir) / f"latest_{algo_or_name}.txt"
    if not pointer.exists():
        raise FileNotFoundError(f"Latest-run pointer not found: {pointer}")
    raw_path = pointer.read_text(encoding="utf-8").strip()
    if not raw_path:
        raise ValueError(f"Latest-run pointer is empty: {pointer}")
    run_dir = Path(raw_path)
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory from {pointer} does not exist: {run_dir}")
    return run_dir.resolve()


def append_phase1_manifest(output_dir, record):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "phase1_manifest.jsonl"
    entry = dict(record)
    entry.setdefault("timestamp_utc", utc_timestamp())
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json_text(entry))
        file.write("\n")
    return manifest_path


def load_monitor_csvs(run_dir):
    """Load SB3 Monitor CSV files under a run directory."""
    run_dir = Path(run_dir)
    train_monitor_dir = run_dir / "logs" / "train_monitor"
    search_dir = train_monitor_dir if train_monitor_dir.exists() else run_dir
    monitor_paths = list(search_dir.rglob("*.monitor.csv"))
    vector_monitor_paths = [path for path in monitor_paths if "vec_monitor" in path.name]
    if vector_monitor_paths:
        monitor_paths = vector_monitor_paths
    frames = []
    for path in monitor_paths:
        try:
            frame = pd.read_csv(path, comment="#")
            frame["source"] = str(path.relative_to(run_dir))
            frames.append(frame)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    if "t" in frame.columns:
        frame = frame.sort_values("t").reset_index(drop=True)
    return frame


def _plotter():
    """Load Matplotlib with a file-only backend that works in Colab and terminals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_returns(run_dir, smoothing=20):
    """Plot episode return from Monitor files."""
    plt = _plotter()

    run_dir = Path(run_dir)
    out_path = run_dir / "plots" / "training_returns.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = load_monitor_csvs(run_dir)
    if frame.empty or "r" not in frame.columns:
        raise FileNotFoundError(f"No monitor return data found under {run_dir}")
    returns = frame["r"].astype(float)
    smoothed = returns.rolling(window=smoothing, min_periods=1).mean()

    plt.figure(figsize=(9, 5))
    plt.plot(range(len(returns)), returns, alpha=0.3, label="episode return")
    plt.plot(range(len(smoothed)), smoothed, label=f"rolling mean ({smoothing})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training episode return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_eval_summary(eval_csv, out_dir=None):
    """Create simple evaluation plots from per-episode CSV."""
    plt = _plotter()

    eval_csv = Path(eval_csv)
    frame = pd.read_csv(eval_csv)
    if out_dir is None:
        out_dir = eval_csv.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    if "route_completion" in frame.columns:
        path = out_dir / "eval_route_completion.png"
        plt.figure(figsize=(7, 4))
        plt.hist(frame["route_completion"].dropna(), bins=15)
        plt.xlabel("Route completion")
        plt.ylabel("Episodes")
        plt.title("Evaluation route completion")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    rate_columns = [
        column
        for column in ["success", "crash", "out_of_road", "max_step"]
        if column in frame.columns
    ]
    if rate_columns:
        values = [float(frame[column].mean()) for column in rate_columns]
        label_map = {
            "success": "success",
            "crash": "collision",
            "out_of_road": "off-road",
            "max_step": "timeout",
        }
        labels = [label_map[column] for column in rate_columns]
        path = out_dir / "eval_outcome_rates.png"
        plt.figure(figsize=(7, 4))
        plt.bar(labels, values)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Closed-loop outcome rates")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    return paths


def plot_comparison_rows(rows, out_path):
    """Plot the four headline Phase-1 metrics for multiple agents."""
    plt = _plotter()

    frame = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        "success_rate",
        "collision_rate",
        "out_of_road_rate",
        "mean_route_completion",
    ]
    metrics = [metric for metric in metrics if metric in frame.columns]
    if not metrics:
        raise ValueError("No comparable metrics found.")

    labels = {
        "success_rate": "Success",
        "collision_rate": "Collision",
        "out_of_road_rate": "Off-road",
        "mean_route_completion": "Route completion",
    }
    names = frame["name"].tolist()
    positions = np.arange(len(metrics))
    width = 0.8 / max(len(names), 1)

    plt.figure(figsize=(9, 5))
    for index, name in enumerate(names):
        offset = (index - (len(names) - 1) / 2) * width
        values = []
        for metric in metrics:
            value = frame.iloc[index].get(metric, np.nan)
            values.append(float(value) if value is not None else np.nan)
        plt.bar(positions + offset, values, width=width, label=name)
    plt.xticks(positions, [labels[metric] for metric in metrics])
    plt.ylim(0, 1)
    plt.ylabel("Rate / fraction")
    plt.title("Phase-1 held-out test comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def compare_eval_summaries(summary_paths, out_path):
    """Plot multiple eval summary JSON files side by side."""
    rows = []
    for path in summary_paths:
        path = Path(path)
        row = read_json(path)
        row["name"] = path.parents[1].name if len(path.parents) > 1 else path.stem
        rows.append(row)
    return plot_comparison_rows(rows, out_path)


def plot_phase1_training_returns(run_dirs, out_path, smoothing=20):
    """Create one optional training-return plot for PPO and SAC."""
    plt = _plotter()

    plotted = False
    plt.figure(figsize=(9, 5))
    for name, run_dir in run_dirs.items():
        frame = load_monitor_csvs(run_dir)
        if frame.empty or "r" not in frame.columns:
            continue
        returns = frame["r"].astype(float)
        smoothed = returns.rolling(window=smoothing, min_periods=1).mean()
        plt.plot(range(len(smoothed)), smoothed, label=f"{name} rolling mean")
        plotted = True
    if not plotted:
        plt.close()
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO and SAC training returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path
