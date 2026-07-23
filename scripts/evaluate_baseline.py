"""Evaluate MetaDrive's IDM or expert controller on fixed traffic conditions."""

import argparse
import sys

import pandas as pd

from saferl_drive.config import (
    apply_dotlist_overrides,
    deep_update,
    get_evaluation_config,
    load_yaml,
    make_experiment_fingerprint,
    make_eval_metadrive_config,
    save_yaml,
)
from saferl_drive.envs import make_vec_env
from saferl_drive.evaluation import evaluate_policy_vecenv, save_eval_outputs, summarize_metrics
from saferl_drive.utils import (
    append_run_manifest,
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
    """Supply an action while MetaDrive's internal policy controls the ego vehicle."""

    def predict(self, observation, deterministic=True):
        return [[0.0, 0.0]], None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a built-in MetaDrive policy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", default="idm", choices=["idm", "expert"])
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--densities", nargs="+", type=float, default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--verify-repeat", action="store_true")
    parser.add_argument(
        "--training-smoke",
        action="store_true",
        help="Validate the vector, action, single-agent, and traffic interfaces before evaluation.",
    )
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", dest="progress", action="store_true")
    progress.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _density_label(density):
    return f"d{int(round(float(density) * 100)):03d}"


def _validate_training_smoke(environment, run_dir, prefix, logger):
    records = environment.env_method("safedrive_diagnostics")
    if len(records) != 1:
        raise RuntimeError(f"Training smoke expected one environment, received {len(records)}.")
    diagnostics = records[0]
    if len(diagnostics["observation_shape"]) != 1 or diagnostics["observation_shape"][0] <= 0:
        raise RuntimeError(
            "Training smoke requires a one-dimensional vector observation; "
            f"received {diagnostics['observation_shape']}."
        )
    if diagnostics["action_shape"] != [2]:
        raise RuntimeError(
            "Training smoke requires MetaDrive's two-value continuous action; "
            f"received {diagnostics['action_shape']}."
        )
    if diagnostics["num_agents"] != 1 or diagnostics["is_multi_agent"]:
        raise RuntimeError(f"Training smoke is not single-agent: {diagnostics}.")
    if diagnostics["image_observation"] or diagnostics["render_mode"] != "none":
        raise RuntimeError(f"Training smoke unexpectedly enabled rendering: {diagnostics}.")
    if (
        diagnostics["traffic_density"] > 0.0
        and diagnostics["traffic_vehicle_count"] < 1
    ):
        raise RuntimeError(f"Training smoke requested traffic but spawned none: {diagnostics}.")
    diagnostics["status"] = "passed"
    diagnostics_path = run_dir / "eval" / f"{prefix}_training_smoke.json"
    write_json(diagnostics, diagnostics_path)
    logger.info("Training-critical environment smoke passed: %s", diagnostics)
    return diagnostics


def _evaluate_once(config, args, run_dir, prefix, policy_class_name, progress, logger):
    evaluation = get_evaluation_config(config, args.split)
    episode_count = int(args.episodes or evaluation.get("episodes", 50))
    start_seed = int(evaluation.get("start_seed", 1000))
    environment_config = make_eval_metadrive_config(config, args.split)
    environment_config["_safedrive_agent_policy"] = policy_class_name
    environment_config["manual_control"] = False
    set_global_seeds(start_seed)
    environment = make_vec_env(
        env_config=environment_config,
        n_envs=1,
        seed=start_seed,
        monitor_dir=run_dir / "logs" / f"{prefix}_monitor",
        vec_env_type=evaluation.get("vec_env", "subproc"),
        normalize_obs=False,
        normalize_reward=False,
        training=False,
    )
    try:
        if args.training_smoke:
            _validate_training_smoke(environment, run_dir, prefix, logger)
        logger.info(
            "Evaluating %s for %s episodes at traffic density %.2f.",
            policy_class_name,
            episode_count,
            float(environment_config.get("traffic_density", 0.0)),
        )
        frame = evaluate_policy_vecenv(
            _DummyActionPolicy(),
            environment,
            episodes=episode_count,
            deterministic=True,
            progress=progress,
            start_seed=start_seed,
            num_scenarios=int(evaluation.get("num_scenarios", episode_count)),
        )
    finally:
        environment.close()
    fingerprint = make_experiment_fingerprint(
        config,
        split=args.split,
        episodes=episode_count,
        controller=policy_class_name,
    )
    frame["training_seed"] = None
    frame["traffic_density"] = float(environment_config.get("traffic_density", 0.0))
    frame["source_checkpoint"] = policy_class_name
    frame["adaptation_condition"] = policy_class_name
    frame["experiment_fingerprint"] = fingerprint["task_id"]
    frame["experiment_task_fingerprint"] = fingerprint["task_id"]
    frame["experiment_strict_fingerprint"] = fingerprint["strict_id"]
    paths = save_eval_outputs(
        frame,
        run_dir / "eval",
        prefix=prefix,
        summary_metadata={
            "evaluation_split": args.split,
            "traffic_density": float(environment_config.get("traffic_density", 0.0)),
            "controller": policy_class_name,
            "experiment_fingerprint": fingerprint,
        },
    )
    plot_eval_summary(paths["episodes_csv"], run_dir / "plots" / prefix)
    return frame, summarize_metrics(frame), paths


def _verify_repeat(config, args, run_dir, prefix, policy_class_name, first_frame, logger):
    evaluation = get_evaluation_config(config, args.split)
    progress = bool(evaluation.get("progress", True)) if args.progress is None else args.progress
    repeated, repeated_summary, paths = _evaluate_once(
        config,
        args,
        run_dir,
        f"{prefix}_repeat",
        policy_class_name,
        progress,
        logger,
    )
    outcome_columns = [
        "env_seed",
        "success",
        "crash",
        "crash_vehicle",
        "out_of_road",
        "max_step",
        "terminal_outcome",
    ]
    try:
        pd.testing.assert_frame_equal(
            first_frame[outcome_columns].reset_index(drop=True),
            repeated[outcome_columns].reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError("Reproducibility gate failed: seeded outcomes changed.") from error
    write_json(
        {
            "passed": True,
            "outcome_columns": outcome_columns,
            "repeat_summary": repeated_summary,
            "repeat_episodes_csv": str(paths["episodes_csv"]),
        },
        run_dir / "eval" / f"{prefix}_check.json",
    )


def main():
    args = parse_args()
    base_config = apply_dotlist_overrides(load_yaml(args.config), args.overrides)
    experiment = base_config.get("experiment", {})
    evaluation = get_evaluation_config(base_config, args.split)
    policy_class_name = "IDMPolicy" if args.policy == "idm" else "ExpertPolicy"
    run_name = f"{args.policy}_traffic_baseline"
    base_prefix = args.prefix or f"{args.policy}_{args.split}"
    output_dir = experiment.get("output_dir", "runs")
    phase = str(experiment.get("phase", "phase1"))
    baseline_prefix = experiment.get("baseline_latest_prefix")
    pointer_name = f"{baseline_prefix}_{args.policy}" if baseline_prefix else args.policy
    publish_run = not args.training_smoke and (args.split == "test" or bool(baseline_prefix))
    run_dir = make_run_dir(output_dir, run_name, None, int(experiment.get("seed", 0)))
    logging_config = base_config.get("logging", {})
    logger = setup_logging(
        run_dir / "logs" / f"{args.policy}_baseline.log",
        console_level=logging_config.get("console_level", "INFO"),
        file_level=logging_config.get("file_level", "DEBUG"),
    )
    metadata_path = run_dir / "run_metadata.json"
    metadata = {
        "status": "running",
        "started_at_utc": utc_timestamp(),
        "command": [sys.executable, "-m", "scripts.evaluate_baseline", *sys.argv[1:]],
        "arguments": vars(args),
        "config_path": str(args.config),
        "run_dir": str(run_dir),
        "algorithm": policy_class_name,
        "latest_name": pointer_name,
        "seed": int(experiment.get("seed", 0)),
        "evaluation": {"split": args.split, "conditions": []},
        "outputs": {},
    }

    try:
        save_yaml(base_config, run_dir / "resolved_config.yaml")
        metadata["system"] = log_system_info(logger, run_dir=run_dir)
        write_json(metadata, metadata_path)
        densities = args.densities if args.densities is not None else [None]
        progress = bool(evaluation.get("progress", True)) if args.progress is None else args.progress
        frames = []
        for density in densities:
            config = base_config
            prefix = base_prefix
            if density is not None:
                config = deep_update(
                    base_config,
                    {
                        args.split: {
                            "traffic_density": float(density),
                            "traffic_mode": "respawn",
                            "random_traffic": False,
                        }
                    },
                )
                prefix = f"{base_prefix}_{_density_label(density)}"
            frame, summary, paths = _evaluate_once(
                config,
                args,
                run_dir,
                prefix,
                policy_class_name,
                progress,
                logger,
            )
            frames.append(frame)
            metadata["evaluation"]["conditions"].append(
                {
                    "traffic_density": float(frame["traffic_density"].iloc[0]),
                    "summary": summary,
                    "episodes_csv": str(paths["episodes_csv"]),
                    "summary_json": str(paths["summary_json"]),
                }
            )
            if args.verify_repeat:
                _verify_repeat(
                    config,
                    args,
                    run_dir,
                    prefix,
                    policy_class_name,
                    frame,
                    logger,
                )

        if args.densities is not None:
            combined = pd.concat(frames, ignore_index=True)
            matrix_csv = run_dir / "eval" / f"{base_prefix}_matrix_episodes.csv"
            matrix_json = run_dir / "eval" / f"{base_prefix}_matrix_summary.json"
            combined.to_csv(matrix_csv, index=False)
            write_json(
                {
                    "generated_at_utc": utc_timestamp(),
                    "controller": policy_class_name,
                    "split": args.split,
                    "conditions": metadata["evaluation"]["conditions"],
                },
                matrix_json,
            )
            metadata["outputs"]["matrix_episodes_csv"] = str(matrix_csv)
            metadata["outputs"]["matrix_summary_json"] = str(matrix_json)

        metadata["status"] = "complete"
        metadata["completed_at_utc"] = utc_timestamp()
        if publish_run:
            pointer = update_latest_run_file(output_dir, pointer_name, run_dir)
            metadata["outputs"]["latest_pointer"] = str(pointer)
        write_json(metadata, metadata_path)
        if publish_run:
            append_run_manifest(
                output_dir,
                phase,
                {
                    "kind": pointer_name,
                    "algorithm": policy_class_name,
                    "status": "complete",
                    "run_dir": str(run_dir),
                    "conditions": metadata["evaluation"]["conditions"],
                },
            )
        logger.info("%s baseline matrix complete: %s", policy_class_name, run_dir)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["failed_at_utc"] = utc_timestamp()
        metadata["error"] = str(error)
        write_json(metadata, metadata_path)
        logger.exception("Baseline evaluation failed.")
        raise


if __name__ == "__main__":
    main()
