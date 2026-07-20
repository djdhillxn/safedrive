"""Evaluate a trained PPO/SAC model on unseen MetaDrive scenarios."""

import argparse
from pathlib import Path

from stable_baselines3.common.vec_env import VecNormalize

from saferl_drive.algorithms import get_algorithm_class
from saferl_drive.config import apply_dotlist_overrides, load_yaml, make_eval_metadrive_config
from saferl_drive.envs import make_vec_env
from saferl_drive.evaluation import evaluate_policy_vecenv, save_eval_outputs
from saferl_drive.utils import log_system_info, plot_eval_summary, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MetaDrive policy.")
    parser.add_argument("--run-dir", required=True, help="Run directory produced by scripts.train.")
    parser.add_argument("--model", default="final", choices=["final", "best"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--prefix", default="eval_unseen", help="Output filename prefix.")
    parser.add_argument("overrides", nargs="*", help="Dotlist config overrides.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "resolved_config.yaml")
    config = apply_dotlist_overrides(config, args.overrides)
    if args.episodes is not None:
        config.setdefault("eval", {})["episodes"] = args.episodes

    logging_config = config.get("logging", {})
    logger = setup_logging(
        run_dir / "logs" / f"{args.prefix}.log",
        console_level=logging_config.get("console_level", "INFO"),
        file_level=logging_config.get("file_level", "DEBUG"),
    )
    environment = None

    try:
        log_system_info(logger, run_dir=run_dir)
        logger.debug("Arguments: %s", vars(args))
        logger.debug("Resolved evaluation config: %s", config)

        algorithm_name = config.get("algorithm", {}).get("name", "ppo").lower()
        algorithm_class = get_algorithm_class(algorithm_name)
        filename = "best_model.zip" if args.model == "best" else "final_model.zip"
        model_path = run_dir / "models" / filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"Requested {args.model} model not found: {model_path}. "
                "Use --model final if the EvalCallback did not create a best model."
            )

        evaluation = config.get("eval", {})
        start_seed = int(evaluation.get("start_seed", 1000))
        base_env = make_vec_env(
            env_config=make_eval_metadrive_config(config),
            n_envs=1,
            seed=start_seed,
            monitor_dir=run_dir / "logs" / f"{args.prefix}_monitor",
            vec_env_type="dummy",
            normalize_obs=False,
            normalize_reward=False,
            training=False,
        )
        environment = base_env
        vecnormalize_path = run_dir / "models" / "vecnormalize.pkl"
        if vecnormalize_path.exists():
            environment = VecNormalize.load(vecnormalize_path, base_env)
            environment.training = False
            environment.norm_reward = False
            logger.debug("Loaded VecNormalize statistics: %s", vecnormalize_path)
        else:
            logger.debug("No VecNormalize statistics found; using raw observations.")

        logger.info("Loading %s model: %s", args.model, model_path)
        model = algorithm_class.load(model_path, env=environment)
        episode_count = int(evaluation.get("episodes", 50))
        logger.info(
            "Evaluating %s episodes on unseen seeds beginning at %s.",
            episode_count,
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
        paths = save_eval_outputs(frame, run_dir / "eval", prefix=args.prefix)
        plot_paths = plot_eval_summary(paths["episodes_csv"], run_dir / "plots")
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
