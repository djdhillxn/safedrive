"""Evaluate a trained PPO/SAC model on one split or a traffic-density matrix."""

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3.common.vec_env import VecNormalize

from saferl_drive.algorithms import get_algorithm_class
from saferl_drive.config import (
    apply_dotlist_overrides,
    deep_update,
    get_evaluation_config,
    load_yaml,
    make_experiment_fingerprint,
    make_eval_metadrive_config,
)
from saferl_drive.envs import find_vecnormalize_path, make_vec_env
from saferl_drive.evaluation import evaluate_policy_vecenv, save_eval_outputs
from saferl_drive.utils import (
    append_run_manifest,
    json_text,
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
    parser.add_argument("--run-dir", required=True, help="Run directory produced by training.")
    parser.add_argument("--model", default="final", choices=["final", "best"])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--densities", nargs="+", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--overwrite", action="store_true")
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", dest="progress", action="store_true")
    progress.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("overrides", nargs="*", help="Dotlist config overrides.")
    return parser.parse_args()


def _density_label(density):
    return f"d{int(round(float(density) * 100)):03d}"


def _condition_config(config, split, density):
    if density is None:
        return config
    return deep_update(
        config,
        {
            split: {
                "traffic_density": float(density),
                "traffic_mode": "respawn",
                "random_traffic": False,
            }
        },
    )


def _load_model(algorithm_class, model_path, environment, load_device):
    try:
        return algorithm_class.load(model_path, env=environment, device=load_device)
    except ModuleNotFoundError as error:
        missing_module = error.name or ""
        if missing_module.startswith("numpy."):
            raise RuntimeError(
                "Checkpoint deserialization failed before observation/action-space "
                f"compatibility could be checked: missing module {missing_module!r}. "
                "Use the project-pinned NumPy runtime that created the checkpoint, then "
                "restart the Python process if NumPy was replaced in a live session."
            ) from error
        raise


def _evaluate_condition(
    args,
    config,
    run_dir,
    prefix,
    model_path,
    algorithm_class,
    algorithm_name,
    load_device,
    density,
    logger,
):
    evaluation = get_evaluation_config(config, args.split)
    default_seeds = {"train": 0, "validation": 1000, "test": 4000}
    start_seed = int(evaluation.get("start_seed", default_seeds[args.split]))
    episode_count = int(args.episodes or evaluation.get("episodes", 50))
    progress = bool(evaluation.get("progress", True)) if args.progress is None else args.progress
    summary_path = run_dir / "eval" / f"{prefix}_summary.json"
    episodes_path = run_dir / "eval" / f"{prefix}_episodes.csv"
    fingerprint = make_experiment_fingerprint(
        config,
        split=args.split,
        episodes=episode_count,
    )
    if summary_path.exists() and not args.overwrite:
        saved = read_json(summary_path)
        saved_fingerprint = saved.get("experiment_fingerprint", {})
        if (
            saved_fingerprint.get("strict_id") == fingerprint["strict_id"]
            and episodes_path.exists()
        ):
            logger.info("Reusing exact fingerprint-matched evaluation: %s", summary_path)
            return (
                pd.read_csv(episodes_path),
                saved,
                {"episodes_csv": episodes_path, "summary_json": summary_path},
            )
        raise FileExistsError(
            f"Evaluation prefix exists with a different fingerprint: {summary_path}. "
            "Use a new prefix; historical evaluations are not overwritten automatically."
        )

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
    try:
        vecnormalize_path = find_vecnormalize_path(run_dir, args.model)
        if vecnormalize_path is not None:
            environment = VecNormalize.load(vecnormalize_path, base_env)
            environment.training = False
            environment.norm_reward = False
            logger.debug("Loaded VecNormalize statistics: %s", vecnormalize_path)

        model = _load_model(
            algorithm_class,
            model_path,
            environment,
            load_device,
        )
        logger.info(
            "Checkpoint interface passed SB3 compatibility checks: observation=%s; action=%s.",
            model.observation_space,
            model.action_space,
        )
        logger.info(
            "Evaluating %s episodes at traffic density %.2f from scenario seed %s.",
            episode_count,
            float(density if density is not None else evaluation.get("traffic_density", 0.0)),
            start_seed,
        )
        frame = evaluate_policy_vecenv(
            model,
            environment,
            episodes=episode_count,
            deterministic=bool(evaluation.get("deterministic", True)),
            progress=progress,
            start_seed=start_seed,
            num_scenarios=int(evaluation.get("num_scenarios", episode_count)),
        )
        experiment = config.get("experiment", {})
        resolved_density = float(
            density if density is not None else evaluation.get("traffic_density", 0.0)
        )
        frame["training_seed"] = int(experiment.get("seed", 0))
        frame["traffic_density"] = resolved_density
        frame["source_checkpoint"] = str(model_path)
        frame["adaptation_condition"] = config.get(
            "selected_variant",
            experiment.get("name", algorithm_name),
        )
        frame["experiment_fingerprint"] = fingerprint["strict_id"]
        frame["experiment_task_fingerprint"] = fingerprint["task_id"]
        frame["experiment_strict_fingerprint"] = fingerprint["strict_id"]
        paths = save_eval_outputs(
            frame,
            run_dir / "eval",
            prefix=prefix,
            summary_metadata={
                "evaluation_split": args.split,
                "training_seed": int(experiment.get("seed", 0)),
                "traffic_density": resolved_density,
                "source_checkpoint": str(model_path),
                "adaptation_condition": frame["adaptation_condition"].iloc[0],
                "experiment_fingerprint": fingerprint,
            },
        )
        plot_eval_summary(paths["episodes_csv"], run_dir / "plots" / prefix)
        return frame, read_json(paths["summary_json"]), paths
    finally:
        environment.close()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    base_prefix = args.prefix or f"{args.model}_{args.split}"
    config = apply_dotlist_overrides(
        load_yaml(run_dir / "resolved_config.yaml"),
        args.overrides,
    )
    logging_config = config.get("logging", {})
    logger = setup_logging(
        run_dir / "logs" / f"{base_prefix}.log",
        console_level=logging_config.get("console_level", "INFO"),
        file_level=logging_config.get("file_level", "DEBUG"),
    )

    try:
        log_system_info(logger, run_dir=run_dir)
        logger.debug("Arguments: %s", vars(args))
        logger.debug("Resolved evaluation config: %s", config)
        algorithm_name = config.get("algorithm", {}).get("name", "ppo").lower()
        algorithm_class = get_algorithm_class(algorithm_name)
        model_path = run_dir / "models" / (
            "best_model.zip" if args.model == "best" else "final_model.zip"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Requested {args.model} model not found: {model_path}")
        configured_device = config.get("algorithm", {}).get("kwargs", {}).get("device")
        load_device = (
            args.device or configured_device or ("cpu" if algorithm_name == "ppo" else "auto")
        )
        densities = args.densities if args.densities is not None else [None]
        frames = []
        conditions = []
        for density in densities:
            condition_prefix = (
                f"{base_prefix}_{_density_label(density)}"
                if args.densities is not None
                else base_prefix
            )
            condition_config = _condition_config(config, args.split, density)
            frame, summary, paths = _evaluate_condition(
                args,
                condition_config,
                run_dir,
                condition_prefix,
                model_path,
                algorithm_class,
                algorithm_name,
                load_device,
                density,
                logger,
            )
            frames.append(frame)
            conditions.append(
                {
                    "traffic_density": float(frame["traffic_density"].iloc[0]),
                    "prefix": condition_prefix,
                    "episodes_csv": str(paths["episodes_csv"]),
                    "summary_json": str(paths["summary_json"]),
                    "summary": summary,
                }
            )

        output_paths = {}
        if args.densities is not None:
            combined = pd.concat(frames, ignore_index=True)
            combined_csv = run_dir / "eval" / f"{base_prefix}_matrix_episodes.csv"
            combined_json = run_dir / "eval" / f"{base_prefix}_matrix_summary.json"
            combined.to_csv(combined_csv, index=False)
            write_json(
                {
                    "generated_at_utc": utc_timestamp(),
                    "model": args.model,
                    "split": args.split,
                    "densities": [float(value) for value in args.densities],
                    "conditions": conditions,
                },
                combined_json,
            )
            output_paths = {
                "matrix_episodes_csv": str(combined_csv),
                "matrix_summary_json": str(combined_json),
            }

        if args.split == "test":
            metadata_path = run_dir / "run_metadata.json"
            if metadata_path.exists():
                metadata = read_json(metadata_path)
                metadata.setdefault("test", {}).update(
                    {
                        "status": "complete",
                        "completed_at_utc": utc_timestamp(),
                        "model": args.model,
                        "prefix": base_prefix,
                        "conditions": conditions,
                    }
                )
                metadata.setdefault("outputs", {}).update(output_paths)
                write_json(metadata, metadata_path)
                experiment = config.get("experiment", {})
                append_run_manifest(
                    experiment.get("output_dir", "runs"),
                    str(experiment.get("phase", "phase1")),
                    {
                        "kind": str(
                            experiment.get("latest_name", algorithm_name)
                        ).format(
                            seed=experiment.get("seed", 0),
                            variant=config.get("selected_variant", "reference"),
                        ),
                        "algorithm": algorithm_name,
                        "status": "test_complete",
                        "run_dir": str(run_dir),
                        "evaluation_conditions": [
                            {
                                "density": item["traffic_density"],
                                "summary": item["summary_json"],
                            }
                            for item in conditions
                        ],
                    },
                )
        logger.info(
            "Evaluation complete for densities: %s",
            ", ".join(f"{item['traffic_density']:.2f}" for item in conditions),
        )
        logger.debug("Evaluation conditions: %s", json_text(conditions))
    except Exception:
        logger.exception("Evaluation failed.")
        raise


if __name__ == "__main__":
    main()
