"""Train PPO or SAC on MetaDrive.

Examples:
    python -m scripts.train --config configs/ppo_mvp.yaml
    python -m scripts.train --config configs/sac_mvp.yaml train.total_timesteps=1000000
"""

import argparse
import csv
import subprocess
import sys
import time
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from saferl_drive.algorithms import build_model, validate_algorithm_config
from saferl_drive.config import (
    apply_dotlist_overrides,
    get_evaluation_config,
    load_yaml,
    make_experiment_fingerprint,
    make_eval_metadrive_config,
    save_yaml,
)
from saferl_drive.envs import make_vec_env
from saferl_drive.evaluation import (
    checkpoint_selection_score,
    evaluate_policy_vecenv,
    save_eval_outputs,
    summarize_metrics,
)
from saferl_drive.utils import (
    append_run_manifest,
    log_system_info,
    make_run_dir,
    model_space_diagnostics,
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
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", dest="progress", action="store_true")
    progress.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
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


class _TrainingDiagnosticsCallback(BaseCallback):
    """Persist rolling outcomes and optimizer health in a readable JSON file."""

    def __init__(self, output_path, save_freq, window=100):
        super().__init__(verbose=0)
        self.output_path = output_path
        self.save_freq = max(int(save_freq), 1)
        self.outcomes = deque(maxlen=window)
        self.actions = deque(maxlen=10_000)
        self.history = []

    def _record_finished_episodes(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for index, done in enumerate(dones):
            if not bool(done) or index >= len(infos):
                continue
            info = infos[index]
            self.outcomes.append(
                {
                    "success": bool(info.get("arrive_dest", False)),
                    "crash": bool(info.get("crash", False)),
                    "out_of_road": bool(info.get("out_of_road", False)),
                    "max_step": bool(info.get("max_step", False)),
                    "route_completion": float(info.get("route_completion", 0.0)),
                }
            )

    def _rolling_mean(self, key):
        if not self.outcomes:
            return None
        return sum(float(item[key]) for item in self.outcomes) / len(self.outcomes)

    def _record_actions(self):
        recorded = False
        for info in self.locals.get("infos", []):
            candidate = info.get("action", info.get("raw_action"))
            if candidate is None:
                continue
            values = np.asarray(candidate, dtype=float).reshape(-1)
            if values.size >= 2:
                self.actions.append(values[:2].copy())
                recorded = True
        if recorded:
            return

        # Generic Gymnasium environments may not report the applied action.
        actions = np.asarray(self.locals.get("actions", []), dtype=float)
        if actions.size == 0:
            return
        for action in actions.reshape(-1, actions.shape[-1]):
            if action.size >= 2:
                self.actions.append(action[:2].copy())

    def _on_step(self):
        self._record_finished_episodes()
        self._record_actions()
        if self.n_calls % self.save_freq != 0:
            return True

        row = {"timesteps": int(self.num_timesteps)}
        logger_values = getattr(self.model.logger, "name_to_value", {})
        diagnostic_names = [
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/actor_loss",
            "train/critic_loss",
            "train/ent_coef",
            "train/ent_coef_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/explained_variance",
            "train/policy_gradient_loss",
            "train/value_loss",
        ]
        for name in diagnostic_names:
            if name in logger_values:
                row[name] = logger_values[name]

        if self.outcomes:
            rolling = {
                "episodes": len(self.outcomes),
                "success_rate": self._rolling_mean("success"),
                "collision_rate": self._rolling_mean("crash"),
                "out_of_road_rate": self._rolling_mean("out_of_road"),
                "timeout_rate": self._rolling_mean("max_step"),
                "mean_route_completion": self._rolling_mean("route_completion"),
            }
            row["rolling_outcomes"] = rolling
            for name, value in rolling.items():
                if name != "episodes":
                    self.logger.record(f"train_outcomes/{name}", value)

        if self.actions:
            action_array = np.asarray(self.actions, dtype=float)
            steering = action_array[:, 0]
            throttle_brake = action_array[:, 1]
            action_behavior = {
                "steps": len(action_array),
                "mean_steering": float(np.mean(steering)),
                "mean_abs_steering": float(np.mean(np.abs(steering))),
                "steering_saturation_rate": float(np.mean(np.abs(steering) >= 0.95)),
                "mean_throttle_brake": float(np.mean(throttle_brake)),
                "throttle_rate": float(np.mean(throttle_brake > 0.05)),
                "brake_rate": float(np.mean(throttle_brake < -0.05)),
            }
            row["action_behavior"] = action_behavior
            for name, value in action_behavior.items():
                if name != "steps":
                    self.logger.record(f"train_actions/{name}", value)

        self.history.append(row)
        write_json({"history": self.history}, self.output_path)
        return True


class _ResourceUsageCallback(BaseCallback):
    """Sample resource use periodically without adding notebook console noise."""

    def __init__(self, output_path, interval_seconds=60):
        super().__init__(verbose=0)
        self.output_path = output_path
        self.interval_seconds = max(float(interval_seconds), 1.0)
        self.last_time = None
        self.last_timesteps = 0

    def _gpu_usage(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None, None
        if result.returncode != 0 or not result.stdout.strip():
            return None, None
        values = result.stdout.strip().splitlines()[0].split(",")
        try:
            return float(values[0].strip()), float(values[1].strip())
        except (IndexError, ValueError):
            return None, None

    def _write_sample(self, now):
        import psutil

        elapsed = max(now - self.last_time, 1e-9)
        timesteps = int(self.num_timesteps)
        gpu_utilization, gpu_memory_mb = self._gpu_usage()
        memory = psutil.virtual_memory()
        row = {
            "timestamp_utc": utc_timestamp(),
            "timesteps": timesteps,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_bytes": int(memory.used),
            "ram_percent": float(memory.percent),
            "gpu_utilization_percent": gpu_utilization,
            "gpu_memory_used_mb": gpu_memory_mb,
            "environment_steps_per_second": (timesteps - self.last_timesteps) / elapsed,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.output_path.exists()
        with self.output_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        self.last_time = now
        self.last_timesteps = timesteps

    def _on_training_start(self):
        self.last_time = time.monotonic()
        self.last_timesteps = int(self.num_timesteps)

    def _on_step(self):
        now = time.monotonic()
        if now - self.last_time >= self.interval_seconds:
            self._write_sample(now)
        return True

    def _on_training_end(self):
        now = time.monotonic()
        if self.last_time is not None and now > self.last_time:
            self._write_sample(now)


class _SuccessFirstValidationCallback(BaseCallback):
    """Evaluate fixed validation seeds and save the best task-success checkpoint."""

    def __init__(
        self,
        eval_env,
        run_dir,
        eval_freq,
        episodes,
        start_seed,
        num_scenarios,
        deterministic,
        run_logger,
        stop_success_rate=None,
        stop_route_completion=None,
        stop_max_collision_rate=None,
        stop_max_out_of_road_rate=None,
        stop_max_timeout_rate=None,
        eval_prefix="validation",
        model_filename="best_model",
        summary_metadata=None,
    ):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.run_dir = run_dir
        self.eval_freq = max(int(eval_freq), 1)
        self.episodes = int(episodes)
        self.start_seed = int(start_seed)
        self.num_scenarios = int(num_scenarios)
        self.deterministic = bool(deterministic)
        self.run_logger = run_logger
        self.stop_success_rate = stop_success_rate
        self.stop_route_completion = stop_route_completion
        self.stop_max_collision_rate = stop_max_collision_rate
        self.stop_max_out_of_road_rate = stop_max_out_of_road_rate
        self.stop_max_timeout_rate = stop_max_timeout_rate
        self.eval_prefix = str(eval_prefix)
        self.model_filename = str(model_filename)
        self.summary_metadata = dict(summary_metadata or {})
        self.best_score = None
        self.target_reached = False
        self.history = []

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

        sync_envs_normalization(self.training_env, self.eval_env)
        frame = evaluate_policy_vecenv(
            self.model,
            self.eval_env,
            episodes=self.episodes,
            deterministic=self.deterministic,
            progress=False,
            start_seed=self.start_seed,
            num_scenarios=self.num_scenarios,
        )
        summary = summarize_metrics(frame)
        step_prefix = f"{self.eval_prefix}_{self.num_timesteps:09d}"
        save_eval_outputs(
            frame,
            self.run_dir / "eval",
            prefix=step_prefix,
            summary_metadata=self.summary_metadata,
        )

        row = {"timesteps": int(self.num_timesteps), **summary}
        self.history.append(row)
        write_json(
            {
                "selection_order": [
                    "highest success rate",
                    "highest route completion",
                    "lowest off-road rate",
                    "lowest collision rate",
                    "lowest timeout rate",
                    "highest mean return",
                ],
                "evaluations": self.history,
            },
            self.run_dir / "eval" / f"{self.eval_prefix}_history.json",
        )

        for name in [
            "success_rate",
            "collision_rate",
            "out_of_road_rate",
            "timeout_or_max_step_rate",
            "mean_route_completion",
            "mean_return",
        ]:
            self.logger.record(f"validation/{name}", summary[name])

        score = checkpoint_selection_score(summary)
        is_new_best = self.best_score is None or score > self.best_score
        target_reached = False
        if self.stop_success_rate is not None and self.stop_route_completion is not None:
            target_reached = summary["success_rate"] >= float(self.stop_success_rate) and summary[
                "mean_route_completion"
            ] >= float(self.stop_route_completion)
            if self.stop_max_collision_rate is not None:
                target_reached = target_reached and summary["collision_rate"] <= float(
                    self.stop_max_collision_rate
                )
            if self.stop_max_out_of_road_rate is not None:
                target_reached = target_reached and summary["out_of_road_rate"] <= float(
                    self.stop_max_out_of_road_rate
                )
            if self.stop_max_timeout_rate is not None:
                target_reached = target_reached and summary["timeout_or_max_step_rate"] <= float(
                    self.stop_max_timeout_rate
                )
        self.run_logger.info(
            "Validation at %s steps: success %.1f%% | route %.1f%% | "
            "collision %.1f%% | off-road %.1f%% | timeout %.1f%%%s",
            self.num_timesteps,
            100.0 * summary["success_rate"],
            100.0 * summary["mean_route_completion"],
            100.0 * summary["collision_rate"],
            100.0 * summary["out_of_road_rate"],
            100.0 * summary["timeout_or_max_step_rate"],
            " | new best" if is_new_best else "",
        )
        if not is_new_best and not target_reached:
            return True

        if is_new_best:
            self.best_score = score
        self.model.save(self.run_dir / "models" / self.model_filename)
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is not None:
            vecnormalize_name = (
                "best_vecnormalize.pkl"
                if self.model_filename == "best_model"
                else f"{self.model_filename}_vecnormalize.pkl"
            )
            vecnormalize.save(self.run_dir / "models" / vecnormalize_name)
        best_prefix = f"best_{self.eval_prefix}"
        save_eval_outputs(
            frame,
            self.run_dir / "eval",
            prefix=best_prefix,
            summary_metadata=self.summary_metadata,
        )
        selection_name = (
            "best_checkpoint_selection.json"
            if self.eval_prefix == "validation" and self.model_filename == "best_model"
            else f"best_{self.eval_prefix}_selection.json"
        )
        write_json(
            {
                "timesteps": int(self.num_timesteps),
                "score": list(score),
                "qualified": target_reached,
                "summary": summary,
            },
            self.run_dir / "eval" / selection_name,
        )
        if target_reached:
            self.target_reached = True
            self.run_logger.info(
                "Learning target reached at %s steps; stopping after saving the checkpoint.",
                self.num_timesteps,
            )
            return False
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
    if args.progress is not None:
        config.setdefault("train", {})["progress_bar"] = args.progress

    experiment = config.get("experiment", {})
    training = config.get("train", {})
    algorithm = config.get("algorithm", {})
    validate_algorithm_config(algorithm)
    validation = get_evaluation_config(config, "validation")
    testing = get_evaluation_config(config, "test")
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
        "experiment_name": experiment.get("name", "metadrive_mvp"),
        "latest_name": str(experiment.get("latest_name", algorithm_name)).format(seed=seed),
        "seed": seed,
        "training": {
            "total_timesteps": int(training.get("total_timesteps", 500_000)),
            "n_envs": int(training.get("n_envs", 1)),
            "vec_env": training.get("vec_env", "dummy"),
        },
        "validation": {
            "episodes": int(validation.get("episodes", 20)),
            "start_seed": int(validation.get("start_seed", 1000)),
            "num_scenarios": int(validation.get("num_scenarios", 50)),
            "vec_env": validation.get("vec_env", "subproc"),
        },
        "test": {
            "episodes": int(testing.get("episodes", 100)),
            "start_seed": int(testing.get("start_seed", 4000)),
            "num_scenarios": int(testing.get("num_scenarios", 100)),
            "vec_env": testing.get("vec_env", "subproc"),
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

        eval_config = make_eval_metadrive_config(config, "validation")
        eval_start_seed = int(validation.get("start_seed", 1000))
        validation_fingerprint = make_experiment_fingerprint(
            config,
            split="validation",
            episodes=int(validation.get("episodes", training.get("eval_episodes", 20))),
        )
        metadata["validation"]["experiment_fingerprint"] = validation_fingerprint
        eval_vec_env_type = validation.get("vec_env", "subproc")
        if training.get("vec_env", "dummy") == "dummy" and eval_vec_env_type == "dummy":
            logger.warning(
                "Training already uses the in-process MetaDrive engine; forcing callback "
                "evaluation into a subprocess."
            )
            eval_vec_env_type = "subproc"
        metadata["validation"]["vec_env"] = eval_vec_env_type
        logger.debug("Validation VecEnv: %s", eval_vec_env_type)
        eval_env = make_vec_env(
            env_config=eval_config,
            n_envs=1,
            seed=eval_start_seed,
            monitor_dir=run_dir / "logs" / "eval_monitor",
            vec_env_type=eval_vec_env_type,
            normalize_obs=bool(training.get("normalize_obs", True)),
            normalize_reward=False,
            training=False,
        )

        tensorboard_log = None
        if experiment.get("tensorboard", True):
            tensorboard_log = str(run_dir / "logs" / "tensorboard")
        model = build_model(algorithm, env=train_env, seed=seed, tensorboard_log=tensorboard_log)
        model_device = str(model.device)
        metadata["training"]["model_device"] = model_device
        diagnostics = model_space_diagnostics(model, train_env, config.get("metadrive", {}))
        metadata["training"]["interface"] = diagnostics
        logger.info("%s policy device: %s", algorithm_name.upper(), model_device)
        logger.info(
            "Policy interface: observation %s | action %s | lidar beams %s | "
            "policy/actor/critic parameters %s/%s/%s",
            diagnostics["observation_shape"],
            diagnostics["action_shape"],
            diagnostics["lidar_beams"],
            diagnostics["policy_parameters"],
            diagnostics["actor_parameters"],
            diagnostics["critic_parameters"],
        )
        environment_count = max(int(training.get("n_envs", 1)), 1)
        if algorithm_name == "ppo":
            ppo_kwargs = algorithm.get("kwargs", {})
            rollout_size = int(ppo_kwargs.get("n_steps", 2048)) * environment_count
            batch_size = int(ppo_kwargs.get("batch_size", 64))
            minibatches = (rollout_size + batch_size - 1) // batch_size
            metadata["training"].update(
                {
                    "rollout_size": rollout_size,
                    "batch_size": batch_size,
                    "minibatches_per_epoch": minibatches,
                }
            )
            logger.info(
                "PPO workload: %s environments x %s steps = %s samples; "
                "%s minibatches of up to %s samples per epoch.",
                environment_count,
                int(ppo_kwargs.get("n_steps", 2048)),
                rollout_size,
                minibatches,
                batch_size,
            )
        logger.debug("Model: %s", model)

        callbacks = []
        checkpoint_frequency = int(training.get("checkpoint_freq", 100_000))
        if checkpoint_frequency > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(checkpoint_frequency // environment_count, 1),
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix=algorithm_name,
                    # The final SAC replay buffer is saved once below. Copying it at
                    # every checkpoint is very expensive on mounted Google Drive.
                    save_replay_buffer=False,
                    save_vecnormalize=True,
                )
            )

        diagnostic_frequency = int(training.get("diagnostic_freq", 10_000))
        callbacks.append(
            _TrainingDiagnosticsCallback(
                run_dir / "logs" / "training_diagnostics.json",
                save_freq=max(diagnostic_frequency // environment_count, 1),
            )
        )
        callbacks.append(
            _ResourceUsageCallback(
                run_dir / "logs" / "resource_usage.csv",
                interval_seconds=training.get("resource_sample_seconds", 60),
            )
        )

        evaluation_frequency = int(training.get("eval_freq", 50_000))
        if evaluation_frequency > 0:
            callbacks.append(
                _SuccessFirstValidationCallback(
                    eval_env,
                    run_dir=run_dir,
                    eval_freq=max(evaluation_frequency // environment_count, 1),
                    episodes=int(validation.get("episodes", training.get("eval_episodes", 20))),
                    start_seed=eval_start_seed,
                    num_scenarios=int(validation.get("num_scenarios", 50)),
                    deterministic=bool(validation.get("deterministic", True)),
                    run_logger=logger,
                    stop_success_rate=training.get("stop_success_rate"),
                    stop_route_completion=training.get("stop_route_completion"),
                    stop_max_collision_rate=training.get("stop_max_collision_rate"),
                    stop_max_out_of_road_rate=training.get("stop_max_out_of_road_rate"),
                    stop_max_timeout_rate=training.get("stop_max_timeout_rate"),
                    summary_metadata={
                        "evaluation_split": "validation",
                        "experiment_fingerprint": validation_fingerprint,
                    },
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
        completed_timesteps = int(model.num_timesteps)
        metadata["training"]["completed_timesteps"] = completed_timesteps
        metadata["training"]["stopped_early"] = completed_timesteps < total_timesteps

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

        if (
            algorithm_name == "sac"
            and bool(training.get("save_replay_buffer", False))
            and hasattr(model, "save_replay_buffer")
        ):
            replay_path = run_dir / "models" / "replay_buffer.pkl"
            model.save_replay_buffer(replay_path)
            metadata["outputs"]["replay_buffer"] = str(replay_path)
            logger.debug("Saved SAC replay buffer: %s", replay_path)

        best_model_path = run_dir / "models" / "best_model.zip"
        selected_model = "best" if best_model_path.exists() else "final"
        metadata["outputs"]["selected_model"] = selected_model
        metadata["test"]["status"] = "pending_explicit_evaluation"
        validation_csv = run_dir / "eval" / "best_validation_episodes.csv"
        validation_summary = run_dir / "eval" / "best_validation_summary.json"
        if validation_summary.exists():
            metadata["outputs"]["best_validation_summary"] = str(validation_summary)
        logger.info(
            "Training is complete with the %s checkpoint frozen. Run scripts.evaluate "
            "with --split test only after all experiment configurations are frozen.",
            selected_model,
        )

        try:
            training_plot = plot_training_returns(run_dir)
            metadata["outputs"]["training_plot"] = str(training_plot)
            logger.debug("Saved training plot: %s", training_plot)
            if validation_csv.exists():
                validation_plots = plot_eval_summary(validation_csv, run_dir / "plots")
                metadata["outputs"]["validation_plots"] = [str(path) for path in validation_plots]
                logger.debug("Saved validation plots: %s", validation_plots)
        except Exception as error:
            logger.warning("Plot generation skipped: %s", error)
            logger.debug("Plot generation exception", exc_info=True)

        pointer_name = str(experiment.get("latest_name", algorithm_name)).format(seed=seed)
        latest_pointer = update_latest_run_file(output_dir, pointer_name, run_dir)
        metadata["outputs"]["latest_pointer"] = str(latest_pointer)
        metadata["status"] = "complete"
        metadata["completed_at_utc"] = utc_timestamp()
        write_json(metadata, metadata_path)
        manifest_path = append_run_manifest(
            output_dir,
            str(experiment.get("phase", "phase1")),
            {
                "kind": pointer_name,
                "algorithm": algorithm_name,
                "status": "complete",
                "run_dir": str(run_dir),
                "summary": str(validation_summary) if validation_summary.exists() else None,
            },
        )
        logger.debug("Updated latest pointer %s and manifest %s", latest_pointer, manifest_path)
        logger.info("Training run complete.")
    except Exception as error:
        metadata["status"] = "failed"
        metadata["failed_at_utc"] = utc_timestamp()
        metadata["error"] = str(error)
        write_json(metadata, metadata_path)
        append_run_manifest(
            output_dir,
            str(experiment.get("phase", "phase1")),
            {
                "kind": str(experiment.get("latest_name", algorithm_name)).format(seed=seed),
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
