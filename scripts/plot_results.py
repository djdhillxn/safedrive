"""Generate plots from a training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from saferl_drive.plotting import plot_eval_summary, plot_training_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training/evaluation outputs.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--eval-csv", type=str, default=None)
    parser.add_argument("--smoothing", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    paths = []
    try:
        paths.append(plot_training_returns(run_dir, smoothing=args.smoothing))
    except Exception as exc:
        print(f"Training-return plot skipped: {exc}")

    eval_csv = Path(args.eval_csv) if args.eval_csv else run_dir / "eval" / "final_unseen_episodes.csv"
    if eval_csv.exists():
        paths.extend(plot_eval_summary(eval_csv, out_dir=run_dir / "plots"))
    else:
        print(f"Eval CSV not found: {eval_csv}")

    for p in paths:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
