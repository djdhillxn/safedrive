"""Generate plots from a training or baseline run."""

import argparse
from pathlib import Path

from saferl_drive.config import load_yaml
from saferl_drive.utils import (
    log_system_info,
    plot_eval_summary,
    plot_training_returns,
    setup_logging,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training/evaluation outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--smoothing", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = run_dir / "resolved_config.yaml"
    config = load_yaml(config_path) if config_path.exists() else {}
    logging_config = config.get("logging", {})
    logger = setup_logging(
        run_dir / "logs" / "plot_results.log",
        console_level=logging_config.get("console_level", "INFO"),
        file_level=logging_config.get("file_level", "DEBUG"),
    )

    try:
        log_system_info(logger, run_dir=run_dir)
        logger.debug("Arguments: %s", vars(args))
        paths = []
        try:
            paths.append(plot_training_returns(run_dir, smoothing=args.smoothing))
        except Exception as error:
            logger.warning("Training-return plot skipped: %s", error)
            logger.debug("Training-return plot exception", exc_info=True)

        eval_csv = (
            Path(args.eval_csv) if args.eval_csv else run_dir / "eval" / "final_unseen_episodes.csv"
        )
        if eval_csv.exists():
            paths.extend(plot_eval_summary(eval_csv, out_dir=run_dir / "plots"))
        else:
            logger.warning("Evaluation CSV not found: %s", eval_csv)

        logger.debug("Plot outputs: %s", paths)
        logger.info("Generated %s plot(s) under %s.", len(paths), run_dir / "plots")
    except Exception:
        logger.exception("Plot generation failed.")
        raise


if __name__ == "__main__":
    main()
