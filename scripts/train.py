"""Train PPO or SAC on MetaDrive.

Examples:
    python -m scripts.train --config configs/ppo_mvp.yaml
    python -m scripts.train --config configs/sac_mvp.yaml train.total_timesteps=1000000
"""

import argparse
import sys

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecNormalize

from saferl_drive.algorithms import build_model
from saferl_drive.config import (
    apply_dotlist_overrides,
    load_yaml,
    make_eval_metadrive_config,
    save_yaml,
)
from saferl_drive.envs import make_vec_env
from saferl_drive.evaluation import evaluate_policy_vecenv, save_eval_outputs
from saferl_drive.utils import (
    append_phase1_manifest,
    log_system_info,
    make_run_dir,
    plot_eval_summary,
    plot_training_returns,
    set_global_seeds,
    setup_logging,
    update_latest_run_file,
    utc_timestamp,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO/SAC on MetaDrive.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", default=None, help="Custom run name override.")
    parser.add_argument("--algo", choices=["ppo", "sac"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default=None)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotlist overrides, such as train.n_envs=2 metadrive.traffic_density=0.2",
    )
    return parser.parse_args()


def _close_env(environment, logger):
    if environment is None:
        return
    try:
        environment.close()
    except Exception:
        if logger is not None:
            logger.debug("Environment close failed.", exc_info=True)


class _SaveBestVecNormalizeCallback(BaseCallback):
    """Save observation statistics at the same moment as a new best model."""

    def __init__(self, save_path):
        super().__init__(verbose=0)
        self.save_path = save_path

    def _on_step(self):
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is not None:
            vecnormalize.save(self.save_path)
        return True


def main():
    args = parse_args()
    config = load_yaml(args.config)
    config = apply_dotlist_overrides(config, args.overrides)

    if args.run_name is not None:
        config.setdefault("experiment", {})["name"] = args.run_name
    if args.algo is not None:
        config.setdefault("algorithm", {})["name"] = args.algo
    if args.seed is not None:
        config.setdefault("experiment", {})["seed"] = args.seed
    if args.total_timesteps is not None:
        config.setdefault("train", {})["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        config.setdefault("train", {})["n_envs"] = args.n_envs
    if args.vec_env is not None:
        config.setdefault("train", {})["vec_env"] = args.vec_env

    experiment = config.get("experiment", {})
    training = config.get("train", {})
    algorithm = config.get("algorithm", {})
    evaluation = config.get("eval", {})
    logging_config = config.get("logging", {})
    seed = int(experiment.get("seed", 0))
    algorithm_name = algorithm.get("name", "ppo").lower()
    output_dir = experiment.get("output_dir", "runs")
    run_dir = make_run_dir(
        output_dir=output_dir,
        experiment_name=experiment.get("name", "metadrive_mvp"),
        algo=algorithm_name,
        seed=seed,
    )
    logger = setup_logging(
        run_dir / "logs" / "train.log",
        console_level=logging_config.get("console_level", "INFO"),
        file_level=logging_config.get("file_level", "DEBUG"),
    )
    train_env = None
    eval_env = None
    metadata_path = run_dir / "run_metadata.json"
    metadata = {
        "status": "running",
        "started_at_utc": utc_timestamp(),
        "command": [sys.executable, "-m", "scripts.train", *sys.argv[1:]],
        "arguments": vars(args),
        "config_path": str(args.config),
        "run_dir": str(run_dir),
        "algorithm": algorithm_name,
        "seed": seed,
        "training": {
            "total_timesteps": int(training.get("total_timesteps", 500_000)),
            "n_envs": int(training.get("n_envs", 1)),
            "vec_env": training.get("vec_env", "dummy"),
        },
        "evaluation": {
            "episodes": int(evaluation.get("episodes", 50)),
            "start_seed": int(evaluation.get("start_seed", 1000)),
            "num_scenarios": int(evaluation.get("num_scenarios", 50)),
            "vec_env": evaluation.get("vec_env", "subproc"),
        },
        "outputs": {},
    }

    try:
        set_global_seeds(seed)
        save_yaml(config, run_dir / "resolved_config.yaml")
        metadata["system"] = log_system_info(logger, run_dir=run_dir)
        write_json(metadata, metadata_path)
        logger.info("Run directory: %s", run_dir)
        logger.debug("Arguments: %s", vars(args))
        logger.debug("Config path: %s", args.config)
        logger.debug("Resolved config: %s", config)

        train_env = make_vec_env(
            env_config=config.get("metadrive", {}),
            n_envs=int(training.get("n_envs", 1)),
            seed=seed,
            monitor_dir=run_dir / "logs" / "train_monitor",
            vec_env_type=training.get("vec_env", "dummy"),
            normalize_obs=bool(training.get("normalize_obs", True)),
            normalize_reward=bool(training.get("normalize_reward", False)),
            training=True,
        )

        eval_config = make_eval_metadrive_config(config)
        eval_start_seed = int(evaluation.get("start_seed", 1000))
        eval_vec_env_type = evaluation.get("vec_env", "subproc")
        if training.get("vec_env", "dummy") == "dummy" and eval_vec_env_type == "dummy":
            logger.warning(
                "Training already uses the in-process MetaDrive engine; forcing callback "
                "evaluation into a subprocess."
            )
            eval_vec_env_type = "subproc"
        metadata["evaluation"]["vec_env"] = eval_vec_env_type
        logger.debug("Callback evaluation VecEnv: %s", eval_vec_env_type)
        eval_env = make_vec_env(
            env_config=eval_config,
            n_envs=1,
            seed=eval_start_seed,
            monitor_dir=run_dir / "logs" / "eval_monitor",
            vec_env_type=eval_vec_env_type,
            normalize_obs=bool(training.get("normalize_obs", True)),
            normalize_reward=bool(training.get("normalize_reward", False)),
            training=False,
        )

        tensorboard_log = None
        if experiment.get("tensorboard", True):
            tensorboard_log = str(run_dir / "logs" / "tensorboard")
        model = build_model(algorithm, env=train_env, seed=seed, tensorboard_log=tensorboard_log)
        logger.debug("Model: %s", model)

        callbacks = []
        checkpoint_frequency = int(training.get("checkpoint_freq", 100_000))
        environment_count = max(int(training.get("n_envs", 1)), 1)
        if checkpoint_frequency > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(checkpoint_frequency // environment_count, 1),
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix=algorithm_name,
                    save_replay_buffer=(algorithm_name == "sac"),
                    save_vecnormalize=True,
                )
            )

        evaluation_frequency = int(training.get("eval_freq", 50_000))
        if evaluation_frequency > 0:
            best_vecnormalize_callback = _SaveBestVecNormalizeCallback(
                run_dir / "models" / "best_vecnormalize.pkl"
            )
            callbacks.append(
                EvalCallback(
                    eval_env,
                    callback_on_new_best=best_vecnormalize_callback,
                    best_model_save_path=str(run_dir / "models"),
                    log_path=str(run_dir / "eval" / "sb3_eval"),
                    eval_freq=max(evaluation_frequency // environment_count, 1),
                    n_eval_episodes=int(training.get("eval_episodes", 10)),
                    deterministic=True,
                    render=False,
                    verbose=0,
                )
            )

        callback = CallbackList(callbacks) if callbacks else None
        total_timesteps = int(training.get("total_timesteps", 500_000))
        logger.info("Training %s for %s timesteps.", algorithm_name.upper(), total_timesteps)
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=bool(training.get("progress_bar", True)),
        )

        final_model_path = run_dir / "models" / "final_model"
        model.save(final_model_path)
        metadata["outputs"]["final_model"] = f"{final_model_path}.zip"
        logger.info("Saved final model: %s.zip", final_model_path)

        if isinstance(train_env, VecNormalize):
            vecnormalize_path = run_dir / "models" / "vecnormalize.pkl"
            train_env.save(vecnormalize_path)
            metadata["outputs"]["vecnormalize"] = str(vecnormalize_path)
            logger.debug("Saved VecNormalize statistics: %s", vecnormalize_path)

        best_vecnormalize_path = run_dir / "models" / "best_vecnormalize.pkl"
        if best_vecnormalize_path.exists():
            metadata["outputs"]["best_vecnormalize"] = str(best_vecnormalize_path)
            logger.debug("Saved best-model VecNormalize statistics: %s", best_vecnormalize_path)

        if algorithm_name == "sac" and hasattr(model, "save_replay_buffer"):
            replay_path = run_dir / "models" / "replay_buffer.pkl"
            model.save_replay_buffer(replay_path)
            metadata["outputs"]["replay_buffer"] = str(replay_path)
            logger.debug("Saved SAC replay buffer: %s", replay_path)

        if isinstance(eval_env, VecNormalize) and isinstance(train_env, VecNormalize):
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms
            eval_env.training = False
            eval_env.norm_reward = False

        episode_count = int(evaluation.get("episodes", 50))
        logger.info(
            "Evaluating final model on %s unseen scenarios from seed %s.",
            episode_count,
            eval_start_seed,
        )
        eval_frame = evaluate_policy_vecenv(
            model,
            eval_env,
            episodes=episode_count,
            deterministic=bool(evaluation.get("deterministic", True)),
            progress=bool(evaluation.get("progress", True)),
            start_seed=eval_start_seed,
            num_scenarios=int(evaluation.get("num_scenarios", episode_count)),
        )
        eval_paths = save_eval_outputs(eval_frame, run_dir / "eval", prefix="final_unseen")
        metadata["outputs"].update(
            {
                "final_eval_csv": str(eval_paths["episodes_csv"]),
                "final_eval_summary": str(eval_paths["summary_json"]),
            }
        )
        logger.info("Saved unseen evaluation: %s", eval_paths["summary_json"])

        try:
            training_plot = plot_training_returns(run_dir)
            eval_plots = plot_eval_summary(eval_paths["episodes_csv"], run_dir / "plots")
            metadata["outputs"]["training_plot"] = str(training_plot)
            metadata["outputs"]["evaluation_plots"] = [str(path) for path in eval_plots]
            logger.debug("Saved plots: %s, %s", training_plot, eval_plots)
        except Exception as error:
            logger.warning("Plot generation skipped: %s", error)
            logger.debug("Plot generation exception", exc_info=True)

        pointer_name = experiment.get("latest_name", algorithm_name)
        latest_pointer = update_latest_run_file(output_dir, pointer_name, run_dir)
        metadata["outputs"]["latest_pointer"] = str(latest_pointer)
        metadata["status"] = "complete"
        metadata["completed_at_utc"] = utc_timestamp()
        write_json(metadata, metadata_path)
        manifest_path = append_phase1_manifest(
            output_dir,
            {
                "kind": pointer_name,
                "algorithm": algorithm_name,
                "status": "complete",
                "run_dir": str(run_dir),
                "summary": str(eval_paths["summary_json"]),
            },
        )
        logger.debug("Updated latest pointer %s and manifest %s", latest_pointer, manifest_path)
        logger.info("Training run complete.")
    except Exception as error:
        metadata["status"] = "failed"
        metadata["failed_at_utc"] = utc_timestamp()
        metadata["error"] = str(error)
        write_json(metadata, metadata_path)
        append_phase1_manifest(
            output_dir,
            {
                "kind": experiment.get("latest_name", algorithm_name),
                "algorithm": algorithm_name,
                "status": "failed",
                "run_dir": str(run_dir),
                "error": str(error),
            },
        )
        logger.exception("Training run failed.")
        raise
    finally:
        _close_env(train_env, logger)
        _close_env(eval_env, logger)


if __name__ == "__main__":
    main()
