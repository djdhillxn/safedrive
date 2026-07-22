"""Evaluate a trained PPO/SAC model on a configured MetaDrive split."""

import argparse
from pathlib import Path

from stable_baselines3.common.vec_env import VecNormalize

from saferl_drive.algorithms import get_algorithm_class
from saferl_drive.config import (
    apply_dotlist_overrides,
    get_evaluation_config,
    load_yaml,
    make_experiment_fingerprint,
    make_eval_metadrive_config,
)
from saferl_drive.envs import find_vecnormalize_path, make_vec_env
from saferl_drive.evaluation import evaluate_policy_vecenv, save_eval_outputs
from saferl_drive.utils import (
    append_run_manifest,
    log_system_info,
    plot_eval_summary,
    read_json,
    set_global_seeds,
    setup_logging,
    utc_timestamp,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MetaDrive policy.")
    parser.add_argument("--run-dir", required=True, help="Run directory produced by scripts.train.")
    parser.add_argument("--model", default="final", choices=["final", "best"])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--device", default=None, help="SB3 load device override, such as cpu or auto."
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing evaluation with the same prefix.",
    )
    parser.add_argument("overrides", nargs="*", help="Dotlist config overrides.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    prefix = args.prefix or f"{args.model}_{args.split}"
    config = load_yaml(run_dir / "resolved_config.yaml")
    config = apply_dotlist_overrides(config, args.overrides)

    logging_config = config.get("logging", {})
    logger = setup_logging(
        run_dir / "logs" / f"{prefix}.log",
        console_level=logging_config.get("console_level", "INFO"),
        file_level=logging_config.get("file_level", "DEBUG"),
    )
    environment = None

    try:
        log_system_info(logger, run_dir=run_dir)
        logger.debug("Arguments: %s", vars(args))
        logger.debug("Resolved evaluation config: %s", config)
        summary_path = run_dir / "eval" / f"{prefix}_summary.json"
        if summary_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Evaluation already exists: {summary_path}. Use a new --prefix or "
                "pass --overwrite intentionally."
            )

        algorithm_name = config.get("algorithm", {}).get("name", "ppo").lower()
        algorithm_class = get_algorithm_class(algorithm_name)
        filename = "best_model.zip" if args.model == "best" else "final_model.zip"
        model_path = run_dir / "models" / filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"Requested {args.model} model not found: {model_path}. "
                "Use --model final if validation did not create a best model."
            )

        evaluation = get_evaluation_config(config, args.split)
        default_seeds = {"train": 0, "validation": 1000, "test": 4000}
        default_seed = default_seeds[args.split]
        start_seed = int(evaluation.get("start_seed", default_seed))
        set_global_seeds(start_seed)
        base_env = make_vec_env(
            env_config=make_eval_metadrive_config(config, args.split),
            n_envs=1,
            seed=start_seed,
            monitor_dir=run_dir / "logs" / f"{prefix}_monitor",
            vec_env_type=evaluation.get("vec_env", "subproc"),
            normalize_obs=False,
            normalize_reward=False,
            training=False,
        )
        environment = base_env
        vecnormalize_path = find_vecnormalize_path(run_dir, args.model)
        if vecnormalize_path is not None:
            environment = VecNormalize.load(vecnormalize_path, base_env)
            environment.training = False
            environment.norm_reward = False
            logger.debug("Loaded VecNormalize statistics: %s", vecnormalize_path)
            if args.model == "best" and vecnormalize_path.name != "best_vecnormalize.pkl":
                logger.warning(
                    "This older run has no best-model normalization snapshot; using the "
                    "final normalization statistics."
                )
        else:
            logger.debug("No VecNormalize statistics found; using raw observations.")

        configured_device = config.get("algorithm", {}).get("kwargs", {}).get("device")
        load_device = (
            args.device or configured_device or ("cpu" if algorithm_name == "ppo" else "auto")
        )
        logger.info("Loading %s model on %s: %s", args.model, load_device, model_path)
        model = algorithm_class.load(model_path, env=environment, device=load_device)
        episode_count = int(args.episodes or evaluation.get("episodes", 50))
        fingerprint = make_experiment_fingerprint(
            config,
            split=args.split,
            episodes=episode_count,
        )
        logger.info(
            "Evaluating %s episodes on the %s split beginning at seed %s.",
            episode_count,
            args.split,
            start_seed,
        )
        frame = evaluate_policy_vecenv(
            model,
            environment,
            episodes=episode_count,
            deterministic=bool(evaluation.get("deterministic", True)),
            progress=bool(evaluation.get("progress", True)),
            start_seed=start_seed,
            num_scenarios=int(evaluation.get("num_scenarios", episode_count)),
        )
        paths = save_eval_outputs(
            frame,
            run_dir / "eval",
            prefix=prefix,
            summary_metadata={
                "evaluation_split": args.split,
                "experiment_fingerprint": fingerprint,
            },
        )
        plot_paths = plot_eval_summary(paths["episodes_csv"], run_dir / "plots")
        if args.split == "test":
            metadata_path = run_dir / "run_metadata.json"
            if metadata_path.exists():
                metadata = read_json(metadata_path)
                metadata.setdefault("test", {}).update(
                    {
                        "status": "complete",
                        "completed_at_utc": utc_timestamp(),
                        "model": args.model,
                        "episodes": episode_count,
                        "start_seed": start_seed,
                        "num_scenarios": int(evaluation.get("num_scenarios", episode_count)),
                        "prefix": prefix,
                        "experiment_fingerprint": fingerprint,
                    }
                )
                metadata.setdefault("outputs", {})["test_eval_csv"] = str(paths["episodes_csv"])
                metadata["outputs"]["test_eval_summary"] = str(paths["summary_json"])
                write_json(metadata, metadata_path)
                experiment = config.get("experiment", {})
                phase = str(experiment.get("phase", "phase1"))
                latest_name = str(experiment.get("latest_name", algorithm_name)).format(
                    seed=experiment.get("seed", 0)
                )
                append_run_manifest(
                    experiment.get("output_dir", "runs"),
                    phase,
                    {
                        "kind": latest_name,
                        "algorithm": algorithm_name,
                        "status": "test_complete",
                        "run_dir": str(run_dir),
                        "summary": str(paths["summary_json"]),
                    },
                )
        logger.debug("Evaluation CSV: %s", paths["episodes_csv"])
        logger.debug("Evaluation plots: %s", plot_paths)
        logger.info("Evaluation complete: %s", paths["summary_json"])
    except Exception:
        logger.exception("Evaluation failed.")
        raise
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    main()
