"""Evaluate MetaDrive's rule-based IDMPolicy on a configured scenario split."""

import argparse
import sys

from saferl_drive.config import (
    apply_dotlist_overrides,
    get_evaluation_config,
    load_yaml,
    make_eval_metadrive_config,
    save_yaml,
)
from saferl_drive.envs import make_vec_env
from saferl_drive.evaluation import evaluate_policy_vecenv, save_eval_outputs, summarize_metrics
from saferl_drive.utils import (
    append_phase1_manifest,
    log_system_info,
    make_run_dir,
    plot_eval_summary,
    set_global_seeds,
    setup_logging,
    update_latest_run_file,
    utc_timestamp,
    write_json,
)


class _DummyActionPolicy:
    """Supply a valid action while MetaDrive's IDMPolicy controls the ego vehicle."""

    def predict(self, observation, deterministic=True):
        return [[0.0, 0.0]], None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the MetaDrive IDM baseline.")
    parser.add_argument("--config", required=True, help="PPO or SAC config defining evaluation.")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--verify-repeat",
        action="store_true",
        help="Repeat seeded evaluation and verify outcomes plus aggregate metrics.",
    )
    parser.add_argument("overrides", nargs="*", help="Dotlist config overrides.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    config = apply_dotlist_overrides(config, args.overrides)

    experiment = config.get("experiment", {})
    evaluation = get_evaluation_config(config, args.split)
    prefix = args.prefix or f"idm_{args.split}"
    episode_count = int(args.episodes or evaluation.get("episodes", 50))
    logging_config = config.get("logging", {})
    output_dir = experiment.get("output_dir", "runs")
    seed = int(experiment.get("seed", 0))
    run_dir = make_run_dir(output_dir, "idm_baseline", None, seed)
    logger = setup_logging(
        run_dir / "logs" / "idm_baseline.log",
        console_level=logging_config.get("console_level", "INFO"),
        file_level=logging_config.get("file_level", "DEBUG"),
    )
    environment = None
    metadata_path = run_dir / "run_metadata.json"
    metadata = {
        "status": "running",
        "started_at_utc": utc_timestamp(),
        "command": [sys.executable, "-m", "scripts.evaluate_baseline", *sys.argv[1:]],
        "arguments": vars(args),
        "config_path": str(args.config),
        "run_dir": str(run_dir),
        "algorithm": "IDMPolicy",
        "seed": seed,
        "evaluation": {
            "episodes": episode_count,
            "start_seed": int(evaluation.get("start_seed", 1000)),
            "num_scenarios": int(evaluation.get("num_scenarios", 50)),
            "split": args.split,
            "prefix": prefix,
        },
        "outputs": {},
    }

    try:
        save_yaml(config, run_dir / "resolved_config.yaml")
        metadata["system"] = log_system_info(logger, run_dir=run_dir)
        write_json(metadata, metadata_path)
        logger.info("Run directory: %s", run_dir)
        logger.debug("Arguments: %s", vars(args))
        logger.debug("Resolved source config: %s", config)

        environment_config = make_eval_metadrive_config(config, args.split)
        # The worker converts this serializable marker into IDMPolicy. Importing
        # MetaDrive in the parent can leave native Panda3D state alive at exit.
        environment_config["_safedrive_agent_policy"] = "IDMPolicy"
        environment_config["manual_control"] = False
        start_seed = int(evaluation.get("start_seed", 1000))
        set_global_seeds(start_seed)
        logger.info(
            "Evaluating IDMPolicy for %s episodes on the %s split beginning at seed %s.",
            episode_count,
            args.split,
            start_seed,
        )
        logger.debug(
            "MetaDrive evaluation config: %s",
            environment_config,
        )

        environment = make_vec_env(
            env_config=environment_config,
            n_envs=1,
            seed=start_seed,
            monitor_dir=run_dir / "logs" / "idm_monitor",
            vec_env_type=evaluation.get("vec_env", "subproc"),
            normalize_obs=False,
            normalize_reward=False,
            training=False,
        )
        frame = evaluate_policy_vecenv(
            _DummyActionPolicy(),
            environment,
            episodes=episode_count,
            deterministic=True,
            progress=bool(evaluation.get("progress", True)),
            start_seed=start_seed,
            num_scenarios=int(evaluation.get("num_scenarios", episode_count)),
        )
        paths = save_eval_outputs(frame, run_dir / "eval", prefix=prefix)
        environment.close()
        environment = None
        repeat_paths = None
        if args.verify_repeat:
            logger.info(
                "Repeating the same seeded episodes in a fresh environment for the "
                "reproducibility gate."
            )
            set_global_seeds(start_seed)
            environment = make_vec_env(
                env_config=environment_config,
                n_envs=1,
                seed=start_seed,
                monitor_dir=run_dir / "logs" / "idm_monitor_repeat",
                vec_env_type=evaluation.get("vec_env", "subproc"),
                normalize_obs=False,
                normalize_reward=False,
                training=False,
            )
            repeat_frame = evaluate_policy_vecenv(
                _DummyActionPolicy(),
                environment,
                episodes=episode_count,
                deterministic=True,
                progress=bool(evaluation.get("progress", True)),
                start_seed=start_seed,
                num_scenarios=int(evaluation.get("num_scenarios", episode_count)),
            )
            repeat_paths = save_eval_outputs(
                repeat_frame,
                run_dir / "eval",
                prefix=f"{prefix}_repeat",
            )
            outcome_columns = [
                "env_seed",
                "success",
                "crash",
                "out_of_road",
                "max_step",
            ]
            try:
                import pandas as pd

                pd.testing.assert_frame_equal(
                    frame[outcome_columns].reset_index(drop=True),
                    repeat_frame[outcome_columns].reset_index(drop=True),
                    check_exact=True,
                )
            except AssertionError as error:
                raise RuntimeError(
                    "Reproducibility gate failed: repeated IDM outcomes changed."
                ) from error

            first_summary = summarize_metrics(frame)
            repeat_summary = summarize_metrics(repeat_frame)
            tolerances = {
                "mean_return": {"relative": 0.02, "absolute": 0.1},
                "mean_length": {"relative": 0.02, "absolute": 1.0},
                "mean_cost": {"relative": 0.0, "absolute": 0.01},
                "mean_route_completion": {"relative": 0.0, "absolute": 0.01},
                "mean_speed_km_h": {"relative": 0.02, "absolute": 0.1},
            }
            differences = {}
            for name, tolerance in tolerances.items():
                first_value = float(first_summary[name])
                repeat_value = float(repeat_summary[name])
                difference = abs(first_value - repeat_value)
                allowed = tolerance["absolute"] + tolerance["relative"] * max(
                    abs(first_value),
                    abs(repeat_value),
                )
                differences[name] = {
                    "first": first_value,
                    "repeat": repeat_value,
                    "absolute_difference": difference,
                    "allowed_difference": allowed,
                }
                if difference > allowed:
                    raise RuntimeError(
                        f"Reproducibility gate failed: {name} changed by {difference:.6g}, "
                        f"above the allowed {allowed:.6g}."
                    )
            reproducibility_path = run_dir / "eval" / f"{prefix}_check.json"
            write_json(
                {
                    "passed": True,
                    "requirement": (
                        "Exact per-scenario categorical outcomes and near-identical "
                        "aggregate continuous metrics."
                    ),
                    "outcome_columns": outcome_columns,
                    "differences": differences,
                },
                reproducibility_path,
            )
            logger.info(
                "Reproducibility gate passed: categorical outcomes match and aggregate "
                "metrics remain within tolerance."
            )
            environment.close()
            environment = None
        plot_paths = plot_eval_summary(paths["episodes_csv"], run_dir / "plots")
        metadata["outputs"] = {
            "episodes_csv": str(paths["episodes_csv"]),
            "summary_json": str(paths["summary_json"]),
            "plots": [str(path) for path in plot_paths],
        }
        if repeat_paths is not None:
            metadata["outputs"]["repeat_episodes_csv"] = str(repeat_paths["episodes_csv"])
            metadata["outputs"]["repeat_summary_json"] = str(repeat_paths["summary_json"])
            metadata["outputs"]["reproducibility_check"] = str(reproducibility_path)
        metadata["status"] = "complete"
        metadata["completed_at_utc"] = utc_timestamp()
        if args.split == "test":
            pointer = update_latest_run_file(output_dir, "idm", run_dir)
            metadata["outputs"]["latest_pointer"] = str(pointer)
        write_json(metadata, metadata_path)
        if args.split == "test":
            manifest = append_phase1_manifest(
                output_dir,
                {
                    "kind": "idm",
                    "algorithm": "IDMPolicy",
                    "status": "complete",
                    "run_dir": str(run_dir),
                    "summary": str(paths["summary_json"]),
                },
            )
            logger.debug("Updated latest pointer %s and manifest %s", pointer, manifest)
        logger.info("IDM baseline complete: %s", paths["summary_json"])
    except Exception as error:
        metadata["status"] = "failed"
        metadata["failed_at_utc"] = utc_timestamp()
        metadata["error"] = str(error)
        write_json(metadata, metadata_path)
        if args.split == "test":
            append_phase1_manifest(
                output_dir,
                {
                    "kind": "idm",
                    "algorithm": "IDMPolicy",
                    "status": "failed",
                    "run_dir": str(run_dir),
                    "error": str(error),
                },
            )
        logger.exception("IDM baseline evaluation failed.")
        raise
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    main()
