"""Compare PPO/SAC run summaries."""

import argparse
from pathlib import Path

from saferl_drive.utils import compare_eval_summaries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare eval summary JSON files from multiple runs."
    )
    parser.add_argument(
        "--summaries", nargs="+", required=True, help="Paths to *_summary.json files."
    )
    parser.add_argument("--output", type=str, default="runs/comparison_eval_summary.png")
    return parser.parse_args()


def main():
    args = parse_args()
    out = compare_eval_summaries([Path(p) for p in args.summaries], args.output)
    print(f"Wrote comparison plot: {out}")


if __name__ == "__main__":
    main()
