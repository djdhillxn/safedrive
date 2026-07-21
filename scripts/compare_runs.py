"""Compare manual summaries or the latest Phase-1 IDM, PPO, and SAC runs."""

import argparse
from pathlib import Path

import pandas as pd

from saferl_drive.evaluation import comparison_summary_row
from saferl_drive.utils import (
    compare_eval_summaries,
    log_system_info,
    plot_comparison_rows,
    plot_phase1_training_returns,
    read_json,
    read_latest_run,
    setup_logging,
    utc_timestamp,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Phase-1 evaluation summaries.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--summaries", nargs="+", help="Paths to summary JSON files.")
    mode.add_argument("--phase1", action="store_true", help="Compare latest IDM, PPO, and SAC.")
    parser.add_argument("--output", default=None, help="Manual comparison PNG output path.")
    parser.add_argument("--runs-dir", default="runs", help="Phase-1 runs directory.")
    return parser.parse_args()


def _preferred_summary(run_dir, name):
    if name == "idm":
        candidates = [
            run_dir / "eval" / "idm_test_summary.json",
            run_dir / "eval" / "idm_unseen_summary.json",
        ]
    else:
        candidates = [
            run_dir / "eval" / "best_test_summary.json",
            run_dir / "eval" / "final_test_summary.json",
            run_dir / "eval" / "best_unseen_summary.json",
            run_dir / "eval" / "final_unseen_summary.json",
        ]

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No held-out test summary found under {run_dir / 'eval'}")


def _run_phase1(args, logger):
    runs_dir = Path(args.runs_dir)
    run_dirs = {
        "IDM": read_latest_run(runs_dir, "idm"),
        "PPO": read_latest_run(runs_dir, "ppo"),
        "SAC": read_latest_run(runs_dir, "sac"),
    }
    rows = []
    for display_name, run_dir in run_dirs.items():
        summary_path = _preferred_summary(run_dir, display_name.lower())
        summary = read_json(summary_path)
        rows.append(
            comparison_summary_row(
                display_name,
                summary,
                run_dir=run_dir,
                summary_path=summary_path,
            )
        )
        logger.debug("Selected %s summary: %s", display_name, summary_path)

    csv_path = runs_dir / "phase1_comparison.csv"
    json_path = runs_dir / "phase1_comparison.json"
    plot_path = runs_dir / "phase1_comparison.png"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_json(
        {
            "generated_at_utc": utc_timestamp(),
            "experiments": rows,
        },
        json_path,
    )
    plot_comparison_rows(rows, plot_path)
    training_plot = plot_phase1_training_returns(
        {"PPO": run_dirs["PPO"], "SAC": run_dirs["SAC"]},
        runs_dir / "phase1_training_returns.png",
    )
    logger.info("Phase-1 comparison written: %s", csv_path)
    logger.info("Comparison plot written: %s", plot_path)
    if training_plot is not None:
        logger.info("Training comparison written: %s", training_plot)
    else:
        logger.debug("Combined training plot skipped because monitor returns were unavailable.")


def main():
    args = parse_args()
    if args.phase1:
        log_path = Path(args.runs_dir) / "phase1_compare.log"
    else:
        output = Path(args.output or "runs/comparison_eval_summary.png")
        log_path = output.with_suffix(".log")
    logger = setup_logging(log_path)

    try:
        log_system_info(logger)
        logger.debug("Arguments: %s", vars(args))
        if args.phase1:
            _run_phase1(args, logger)
        else:
            output = Path(args.output or "runs/comparison_eval_summary.png")
            summary_paths = [Path(path) for path in args.summaries]
            compare_eval_summaries(summary_paths, output)
            logger.debug("Manual summaries: %s", summary_paths)
            logger.info("Comparison plot written: %s", output)
    except Exception:
        logger.exception("Run comparison failed.")
        raise


if __name__ == "__main__":
    main()
