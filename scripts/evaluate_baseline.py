"""Evaluate MetaDrive's rule-based IDMPolicy on unseen scenarios."""

import argparse
import sys

from saferl_drive.config import (
    apply_dotlist_overrides,
    load_yaml,
    make_eval_metadrive_config,
    save_yaml,
)
from saferl_drive.evaluation import evaluate_policy_closed_loop, save_eval_outputs
from saferl_drive.utils import (
    append_phase1_manifest,
    log_system_info,
    make_run_dir,
    plot_eval_summary,
    setup_logging,
    update_latest_run_file,
    utc_timestamp,
    write_json,
)


class _DummyActionPolicy:
    """Supply a valid action while MetaDrive's IDMPolicy controls the ego vehicle."""

    def predict(self, observation, deterministic=True):
        return [0.0, 0.0], None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the MetaDrive IDM baseline.")
    parser.add_argument("--config", required=True, help="PPO or SAC config defining evaluation.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--prefix", default="idm_unseen")
    parser.add_argument("overrides", nargs="*", help="Dotlist config overrides.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    config = apply_dotlist_overrides(config, args.overrides)
    if args.episodes is not None:
        config.setdefault("eval", {})["episodes"] = args.episodes

    experiment = config.get("experiment", {})
    evaluation = config.get("eval", {})
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
            "episodes": int(evaluation.get("episodes", 50)),
            "start_seed": int(evaluation.get("start_seed", 1000)),
            "num_scenarios": int(evaluation.get("num_scenarios", 50)),
            "prefix": args.prefix,
        },
        "outputs": {},
    }

    try:
        # Keep these imports here so a missing MetaDrive/IDM installation produces a clear run log.
        from metadrive.envs import MetaDriveEnv
        from metadrive.policy.idm_policy import IDMPolicy

        save_yaml(config, run_dir / "resolved_config.yaml")
        metadata["system"] = log_system_info(logger, run_dir=run_dir)
        write_json(metadata, metadata_path)
        logger.info("Run directory: %s", run_dir)
        logger.debug("Arguments: %s", vars(args))
        logger.debug("Resolved source config: %s", config)

        environment_config = make_eval_metadrive_config(config)
        environment_config["agent_policy"] = IDMPolicy
        environment_config["manual_control"] = False
        start_seed = int(evaluation.get("start_seed", 1000))
        episode_count = int(evaluation.get("episodes", 50))
        logger.info(
            "Evaluating IDMPolicy for %s episodes on unseen seeds beginning at %s.",
            episode_count,
            start_seed,
        )
        logger.debug(
            "MetaDrive evaluation config: %s",
            {**environment_config, "agent_policy": "IDMPolicy"},
        )

        environment = MetaDriveEnv(environment_config)
        frame = evaluate_policy_closed_loop(
            _DummyActionPolicy(),
            environment,
            episodes=episode_count,
            deterministic=True,
            progress=bool(evaluation.get("progress", True)),
            start_seed=start_seed,
            num_scenarios=int(evaluation.get("num_scenarios", episode_count)),
        )
        paths = save_eval_outputs(frame, run_dir / "eval", prefix=args.prefix)
        plot_paths = plot_eval_summary(paths["episodes_csv"], run_dir / "plots")
        metadata["outputs"] = {
            "episodes_csv": str(paths["episodes_csv"]),
            "summary_json": str(paths["summary_json"]),
            "plots": [str(path) for path in plot_paths],
        }
        pointer = update_latest_run_file(output_dir, "idm", run_dir)
        metadata["outputs"]["latest_pointer"] = str(pointer)
        metadata["status"] = "complete"
        metadata["completed_at_utc"] = utc_timestamp()
        write_json(metadata, metadata_path)
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
