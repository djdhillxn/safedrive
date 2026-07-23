"""Train SAC through a resumable sequence of MetaDrive road geometries."""

import argparse
import shutil
import sys
import time
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from saferl_drive.algorithms import (
    build_model,
    get_algorithm_class,
    validate_algorithm_config,
)
from saferl_drive.config import (
    apply_dotlist_overrides,
    deep_update,
    get_evaluation_config,
    load_yaml,
    make_eval_metadrive_config,
    make_experiment_fingerprint,
    resolve_reward_variant,
    save_yaml,
)
from saferl_drive.envs import make_vec_env
from saferl_drive.utils import (
    append_run_manifest,
    log_system_info,
    make_run_dir,
    model_space_diagnostics,
    plot_training_returns,
    read_json,
    read_latest_run,
    set_global_seeds,
    sha256_file,
    setup_logging,
    update_latest_run_file,
    utc_timestamp,
    write_json,
)
from scripts.train import (
    _SuccessFirstValidationCallback,
    _TrainingDiagnosticsCallback,
    _ResourceUsageCallback,
    _close_env,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the staged Phase-2 SAC curriculum.")
    parser.add_argument("--config", required=True, help="Curriculum YAML configuration.")
    parser.add_argument("--run-dir", default=None, help="Existing curriculum run to resume.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default=None)
    parser.add_argument("--variant", default=None, help="Declared reward variant.")
    parser.add_argument("--source-run-dir", default=None)
    parser.add_argument("--source-model", choices=["best", "final"], default=None)
    parser.add_argument(
        "--load-source-replay-buffer",
        action="store_true",
        help="Explicitly import the source replay buffer. Never used by canonical traffic runs.",
    )
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", dest="progress", action="store_true")
    progress.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument(
        "--stop-after-stage",
        default=None,
        help="Pause after this stage so its resume artifacts can be copied to Drive.",
    )
    parser.add_argument("overrides", nargs="*", help="Dotlist configuration overrides.")
    return parser.parse_args()


def stage_config(config, stage):
    result = deep_update(config, {"metadrive": stage.get("metadrive", {})})
    result = deep_update(result, {"validation": stage.get("validation", {})})
    return result


def validate_curriculum_config(config):
    validate_algorithm_config(config.get("algorithm", {}))
    if config.get("algorithm", {}).get("name", "").lower() != "sac":
        raise ValueError("The Phase-2 curriculum currently supports SAC only.")
    if config.get("metadrive", {}).get("discrete_action", False):
        raise ValueError("The Phase-2 SAC curriculum requires continuous actions.")
    if "steering_limit" in config.get("metadrive", {}):
        raise ValueError(
            "Phase 2 requires the full MetaDrive steering range; remove steering_limit."
        )
    if not config.get("train", {}).get("save_replay_buffer", False):
        raise ValueError(
            "Curriculum SAC must save its replay buffer so stage resumes preserve "
            "the complete off-policy training state."
        )

    curriculum = config.get("curriculum", {})
    stages = curriculum.get("stages", [])
    if not stages:
        raise ValueError("curriculum.stages must contain at least one stage.")
    names = [str(stage.get("name", "")).strip() for stage in stages]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Every curriculum stage needs a unique non-empty name.")
    total = int(
        curriculum.get("total_timesteps", config.get("train", {}).get("total_timesteps", 0))
    )
    remaining_stages = [stage for stage in stages if stage.get("use_remaining_timesteps")]
    if remaining_stages and (
        len(remaining_stages) != 1 or stages[-1] is not remaining_stages[0]
    ):
        raise ValueError("Only the final curriculum stage may use remaining timesteps.")
    fixed = sum(
        int(stage.get("max_timesteps", 0))
        for stage in stages
        if not stage.get("use_remaining_timesteps")
    )
    if total <= 0:
        raise ValueError("The curriculum total must be positive.")
    if remaining_stages and fixed >= total:
        raise ValueError("The curriculum total must leave a positive budget for the final stage.")
    if not remaining_stages and fixed != total:
        raise ValueError(
            "Fixed curriculum stage budgets must add up exactly to curriculum.total_timesteps."
        )

    source = config.get("source")
    if source:
        metadrive = config.get("metadrive", {})
        if int(metadrive.get("num_agents", 1)) != 1:
            raise ValueError("Traffic adaptation requires metadrive.num_agents=1.")
        if bool(metadrive.get("is_multi_agent", False)):
            raise ValueError("Traffic adaptation requires metadrive.is_multi_agent=false.")
        if bool(metadrive.get("image_observation", False)):
            raise ValueError("Traffic adaptation must retain the vector LidarState observation.")
        if not bool(source.get("fresh_replay_buffer", True)):
            raise ValueError("Canonical traffic adaptation requires a fresh source replay buffer.")
        for stage in stages:
            if stage.get("metadrive", {}).get("traffic_mode", "respawn") != "respawn":
                raise ValueError("Every traffic-adaptation stage must use traffic_mode=respawn.")
    return stages


def curriculum_stage_budget(config, stage, completed_timesteps):
    total = int(
        config.get("curriculum", {}).get(
            "total_timesteps",
            config.get("train", {}).get("total_timesteps", 0),
        )
    )
    remaining = max(total - int(completed_timesteps), 0)
    if stage.get("use_remaining_timesteps"):
        return remaining
    return min(int(stage.get("max_timesteps", 0)), remaining)


def _gate_settings(stage):
    gate = stage.get("gate", {})
    return {
        "stop_success_rate": gate.get("success_rate"),
        "stop_route_completion": gate.get("route_completion"),
        "stop_max_collision_rate": gate.get("max_collision_rate"),
        "stop_max_out_of_road_rate": gate.get("max_out_of_road_rate"),
        "stop_max_timeout_rate": gate.get("max_timeout_rate"),
    }


def _new_state(stages, total_timesteps):
    return {
        "status": "running",
        "created_at_utc": utc_timestamp(),
        "updated_at_utc": utc_timestamp(),
        "total_budget": int(total_timesteps),
        "completed_timesteps": 0,
        "next_stage_index": 0,
        "stage_order": [stage["name"] for stage in stages],
        "completed_stages": [],
    }


def _save_resume_artifacts(model, run_dir, save_replay_buffer, logger):
    model_path = run_dir / "models" / "curriculum_resume_model"
    model.save(model_path)
    outputs = {"model": f"{model_path}.zip"}
    if save_replay_buffer and hasattr(model, "save_replay_buffer"):
        replay_path = run_dir / "models" / "curriculum_resume_replay_buffer.pkl"
        model.save_replay_buffer(replay_path)
        outputs["replay_buffer"] = str(replay_path)
        logger.info("Saved resumable SAC replay buffer at the stage boundary.")
    return outputs


def _promote_failed_gate_model(model, run_dir, stage_name):
    """Expose the success-first Stage-A checkpoint for honest failure analysis."""
    stage_best = Path(run_dir) / "models" / f"{stage_name}_best_model.zip"
    canonical_best = Path(run_dir) / "models" / "best_model.zip"
    if stage_best.exists():
        shutil.copy2(stage_best, canonical_best)
    else:
        model.save(canonical_best.with_suffix(""))
    return canonical_best


def _load_resume_model(config, run_dir, environment, logger):
    model_path = run_dir / "models" / "curriculum_resume_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Curriculum resume model not found: {model_path}")
    algorithm = config.get("algorithm", {})
    model = get_algorithm_class("sac").load(
        model_path,
        env=environment,
        device=algorithm.get("kwargs", {}).get("device", "auto"),
    )
    replay_path = run_dir / "models" / "curriculum_resume_replay_buffer.pkl"
    if replay_path.exists():
        model.load_replay_buffer(replay_path)
        logger.info("Loaded curriculum replay buffer: %s", replay_path)
    elif config.get("train", {}).get("save_replay_buffer", False):
        raise FileNotFoundError(
            "Curriculum replay buffer is missing: "
            f"{replay_path}. Restore the run with training artifacts before resuming."
        )
    else:
        logger.warning(
            "No replay buffer was found. Resume will continue from the saved actor and "
            "critics with an empty replay buffer."
        )
    return model


def _space_signature(space):
    return {
        "class": space.__class__.__name__,
        "shape": list(getattr(space, "shape", ()) or ()),
        "dtype": str(getattr(space, "dtype", "unknown")),
        "low": getattr(space, "low", None),
        "high": getattr(space, "high", None),
    }


def _spaces_match(first, second):
    first_signature = _space_signature(first)
    second_signature = _space_signature(second)
    for key in ["class", "shape", "dtype"]:
        if first_signature[key] != second_signature[key]:
            return False
    try:
        import numpy as np

        return np.array_equal(first_signature["low"], second_signature["low"]) and np.array_equal(
            first_signature["high"], second_signature["high"]
        )
    except Exception:
        return True


def _source_summary_fingerprint(source_run, source_model):
    candidates = [
        source_run / "eval" / f"{source_model}_test_summary.json",
        source_run / "eval" / "best_test_summary.json",
        source_run / "eval" / "best_validation_summary.json",
    ]
    for path in candidates:
        if path.exists():
            summary = read_json(path)
            fingerprint = summary.get("experiment_fingerprint")
            if fingerprint:
                return fingerprint, path
    source_config = load_yaml(source_run / "resolved_config.yaml")
    return make_experiment_fingerprint(source_config, split="test"), None


def _resolve_source_lineage(config, args, output_dir):
    source = config.get("source")
    if not source:
        return None
    seed = int(config.get("experiment", {}).get("seed", 0))
    source_run = (
        Path(args.source_run_dir)
        if args.source_run_dir
        else read_latest_run(
            output_dir,
            str(source.get("latest_name", "sac_phase2_curriculum_seed{seed}")).format(seed=seed),
        )
    )
    source_model = args.source_model or source.get("model", "best")
    checkpoint = source_run / "models" / f"{source_model}_model.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Warm-start {source_model} checkpoint is missing: {checkpoint}. "
            "Restore training artifacts from Drive before adaptation."
        )
    fingerprint, summary_path = _source_summary_fingerprint(source_run, source_model)
    return {
        "source_run_dir": str(source_run.resolve()),
        "source_model": source_model,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_run_fingerprint": fingerprint,
        "source_summary": str(summary_path) if summary_path else None,
        "fresh_replay_buffer": not args.load_source_replay_buffer,
    }


def _load_warm_start_model(config, lineage, environment, seed, tensorboard_log, args, logger):
    algorithm = config.get("algorithm", {})
    source_model = get_algorithm_class("sac").load(
        lineage["source_checkpoint"],
        device=algorithm.get("kwargs", {}).get("device", "auto"),
    )
    if not _spaces_match(source_model.observation_space, environment.observation_space):
        raise ValueError(
            "Warm-start observation space does not match the traffic environment: "
            f"checkpoint={_space_signature(source_model.observation_space)}, "
            f"environment={_space_signature(environment.observation_space)}."
        )
    if not _spaces_match(source_model.action_space, environment.action_space):
        raise ValueError(
            "Warm-start action space does not match the traffic environment: "
            f"checkpoint={_space_signature(source_model.action_space)}, "
            f"environment={_space_signature(environment.action_space)}."
        )
    model = build_model(
        algorithm,
        env=environment,
        seed=seed,
        tensorboard_log=tensorboard_log,
    )
    model.policy.load_state_dict(source_model.policy.state_dict(), strict=True)
    if args.load_source_replay_buffer:
        source_run = Path(lineage["source_run_dir"])
        candidates = [
            source_run / "models" / "curriculum_resume_replay_buffer.pkl",
            source_run / "models" / "replay_buffer.pkl",
        ]
        replay_path = next((path for path in candidates if path.exists()), None)
        if replay_path is None:
            raise FileNotFoundError(
                "Source replay import was explicitly requested, but no source replay buffer exists."
            )
        model.load_replay_buffer(replay_path)
        lineage["source_replay_buffer"] = str(replay_path.resolve())
        logger.warning("Explicitly loaded source replay buffer: %s", replay_path)
    else:
        logger.info("Warm-started actor and critics with a fresh traffic replay buffer.")
    del source_model
    return model


def _latest_name(config, seed):
    experiment = config.get("experiment", {})
    return str(experiment.get("latest_name", "sac_phase2_curriculum_seed{seed}")).format(
        seed=seed,
        variant=config.get("selected_variant", "reference"),
    )


def main():
    args = parse_args()
    requested_config = apply_dotlist_overrides(load_yaml(args.config), args.overrides)
    requested_config = resolve_reward_variant(requested_config, args.variant)
    if args.seed is not None:
        requested_config.setdefault("experiment", {})["seed"] = args.seed
    if args.total_timesteps is not None:
        requested_config.setdefault("train", {})["total_timesteps"] = args.total_timesteps
        requested_config.setdefault("curriculum", {})["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        requested_config.setdefault("train", {})["n_envs"] = args.n_envs
    if args.vec_env is not None:
        requested_config.setdefault("train", {})["vec_env"] = args.vec_env
    if args.progress is not None:
        requested_config.setdefault("train", {})["progress_bar"] = args.progress
    if requested_config.get("source"):
        n_envs = int(requested_config.get("train", {}).get("n_envs", 1))
        requested_config.setdefault("algorithm", {}).setdefault("kwargs", {})[
            "gradient_steps"
        ] = n_envs
    stages = validate_curriculum_config(requested_config)

    experiment = requested_config.get("experiment", {})
    training = requested_config.get("train", {})
    curriculum = requested_config.get("curriculum", {})
    logging_config = requested_config.get("logging", {})
    seed = int(experiment.get("seed", 0))
    total_timesteps = int(curriculum.get("total_timesteps", training.get("total_timesteps", 0)))
    output_dir = experiment.get("output_dir", "runs")
    lineage = None

    if args.stop_after_stage is not None and args.stop_after_stage not in {
        stage["name"] for stage in stages
    }:
        raise ValueError(f"Unknown stop stage: {args.stop_after_stage}")

    if args.run_dir:
        run_dir = Path(args.run_dir)
        saved_config = load_yaml(run_dir / "resolved_config.yaml")
        if saved_config != requested_config:
            raise ValueError(
                "Resume configuration differs from the run's immutable resolved_config.yaml."
            )
        state = read_json(run_dir / "curriculum_state.json")
    else:
        lineage = _resolve_source_lineage(requested_config, args, output_dir)
        run_dir = make_run_dir(
            output_dir,
            experiment.get("name", "sac_phase2_curriculum"),
            "sac",
            seed,
        )
        save_yaml(requested_config, run_dir / "resolved_config.yaml")
        state = _new_state(stages, total_timesteps)
        write_json(state, run_dir / "curriculum_state.json")
        if lineage is not None:
            write_json(lineage, run_dir / "source_lineage.json")

    if args.run_dir and requested_config.get("source"):
        lineage_path = run_dir / "source_lineage.json"
        if not lineage_path.exists():
            raise FileNotFoundError(f"Source lineage metadata is missing: {lineage_path}")
        lineage = read_json(lineage_path)
        if bool(lineage.get("fresh_replay_buffer", True)) == bool(
            args.load_source_replay_buffer
        ):
            raise ValueError(
                "Resume replay-source mode differs from the immutable source lineage."
            )

    logger = setup_logging(
        run_dir / "logs" / "curriculum_train.log",
        console_level=logging_config.get("console_level", "INFO"),
        file_level=logging_config.get("file_level", "DEBUG"),
    )
    metadata_path = run_dir / "run_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    metadata.update(
        {
            "status": "running",
            "started_at_utc": metadata.get("started_at_utc", utc_timestamp()),
            "resumed_at_utc": utc_timestamp() if args.run_dir else None,
            "command": [sys.executable, "-m", "scripts.train_curriculum", *sys.argv[1:]],
            "arguments": vars(args),
            "config_path": str(args.config),
            "run_dir": str(run_dir),
            "algorithm": "sac",
            "experiment_name": experiment.get("name", "sac_phase2_curriculum"),
            "latest_name": _latest_name(requested_config, seed),
            "seed": seed,
            "phase": str(experiment.get("phase", "phase2")),
            "variant": requested_config.get("selected_variant"),
            "source_lineage": lineage,
            "curriculum": {
                "total_timesteps": total_timesteps,
                "stage_order": [stage["name"] for stage in stages],
            },
            "outputs": metadata.get("outputs", {}),
        }
    )

    train_env = None
    eval_env = None
    model = None
    try:
        set_global_seeds(seed)
        metadata["system"] = log_system_info(logger, run_dir=run_dir)
        write_json(metadata, metadata_path)
        logger.info("Curriculum run directory: %s", run_dir)
        logger.info(
            "Curriculum progress: %s/%s timesteps; next stage index %s.",
            state.get("completed_timesteps", 0),
            total_timesteps,
            state.get("next_stage_index", 0),
        )

        next_stage_index = int(state.get("next_stage_index", 0))
        for stage_index in range(next_stage_index, len(stages)):
            stage = stages[stage_index]
            name = stage["name"]
            budget = curriculum_stage_budget(
                requested_config,
                stage,
                state.get("completed_timesteps", 0),
            )
            if budget <= 0:
                raise RuntimeError(f"No remaining timestep budget for curriculum stage {name}.")

            current_config = stage_config(requested_config, stage)
            current_training = current_config.get("train", {})
            validation = get_evaluation_config(current_config, "validation")
            validation_episodes = int(validation.get("episodes", 25))
            fingerprint = make_experiment_fingerprint(
                current_config,
                split="validation",
                episodes=validation_episodes,
            )
            logger.info(
                "Starting curriculum stage %s (%s/%s), map=%r, traffic density=%s, "
                "maximum %s new timesteps.",
                name,
                stage_index + 1,
                len(stages),
                current_config.get("metadrive", {}).get("map"),
                current_config.get("metadrive", {}).get("traffic_density"),
                budget,
            )

            train_env = make_vec_env(
                env_config=current_config.get("metadrive", {}),
                n_envs=int(current_training.get("n_envs", 1)),
                seed=seed,
                monitor_dir=run_dir / "logs" / "train_monitor" / name,
                vec_env_type=current_training.get("vec_env", "dummy"),
                normalize_obs=bool(current_training.get("normalize_obs", False)),
                normalize_reward=bool(current_training.get("normalize_reward", False)),
                training=True,
            )
            eval_start_seed = int(validation.get("start_seed", 20000))
            eval_env = make_vec_env(
                env_config=make_eval_metadrive_config(current_config, "validation"),
                n_envs=1,
                seed=eval_start_seed,
                monitor_dir=run_dir / "logs" / "eval_monitor" / name,
                vec_env_type=validation.get("vec_env", "subproc"),
                normalize_obs=bool(current_training.get("normalize_obs", False)),
                normalize_reward=False,
                training=False,
            )

            if model is None and args.run_dir:
                model = _load_resume_model(requested_config, run_dir, train_env, logger)
            elif model is None:
                tensorboard_log = None
                if experiment.get("tensorboard", True):
                    tensorboard_log = str(run_dir / "logs" / "tensorboard")
                if lineage is not None:
                    model = _load_warm_start_model(
                        requested_config,
                        lineage,
                        train_env,
                        seed,
                        tensorboard_log,
                        args,
                        logger,
                    )
                    write_json(lineage, run_dir / "source_lineage.json")
                else:
                    model = build_model(
                        requested_config.get("algorithm", {}),
                        env=train_env,
                        seed=seed,
                        tensorboard_log=tensorboard_log,
                    )
            else:
                model.set_env(train_env)

            environment_count = max(int(current_training.get("n_envs", 1)), 1)
            diagnostics = model_space_diagnostics(
                model,
                train_env,
                current_config.get("metadrive", {}),
            )
            metadata.setdefault("training", {}).update(
                {
                    "model_device": str(model.device),
                    "n_envs": environment_count,
                    "gradient_steps": int(
                        current_config.get("algorithm", {})
                        .get("kwargs", {})
                        .get("gradient_steps", 1)
                    ),
                    "interface": diagnostics,
                }
            )
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
            callbacks = []
            checkpoint_frequency = int(current_training.get("checkpoint_freq", 50000))
            if checkpoint_frequency > 0:
                callbacks.append(
                    CheckpointCallback(
                        save_freq=max(checkpoint_frequency // environment_count, 1),
                        save_path=str(run_dir / "checkpoints" / name),
                        name_prefix=f"sac_{name}",
                        save_replay_buffer=False,
                        save_vecnormalize=True,
                    )
                )
            callbacks.append(
                _TrainingDiagnosticsCallback(
                    run_dir / "logs" / f"training_diagnostics_{name}.json",
                    save_freq=max(
                        int(current_training.get("diagnostic_freq", 10000)) // environment_count,
                        1,
                    ),
                )
            )
            callbacks.append(
                _ResourceUsageCallback(
                    run_dir / "logs" / "resource_usage.csv",
                    interval_seconds=current_training.get("resource_sample_seconds", 60),
                )
            )
            final_stage = stage_index == len(stages) - 1
            model_filename = "best_model" if final_stage else f"{name}_best_model"
            eval_prefix = "validation" if final_stage else f"{name}_validation"
            validation_callback = _SuccessFirstValidationCallback(
                eval_env,
                run_dir=run_dir,
                eval_freq=max(
                    int(current_training.get("eval_freq", 25000)) // environment_count,
                    1,
                ),
                episodes=validation_episodes,
                start_seed=eval_start_seed,
                num_scenarios=int(validation.get("num_scenarios", validation_episodes)),
                deterministic=bool(validation.get("deterministic", True)),
                run_logger=logger,
                eval_prefix=eval_prefix,
                model_filename=model_filename,
                summary_metadata={
                    "evaluation_split": "validation",
                    "curriculum_stage": name,
                    "experiment_fingerprint": fingerprint,
                },
                **_gate_settings(stage),
            )
            callbacks.append(validation_callback)

            before = int(model.num_timesteps)
            stage_started = time.monotonic()
            model.learn(
                total_timesteps=budget,
                callback=CallbackList(callbacks),
                log_interval=10,
                progress_bar=bool(current_training.get("progress_bar", True)),
                reset_num_timesteps=False,
            )
            completed = int(model.num_timesteps) - before
            elapsed_seconds = time.monotonic() - stage_started
            state["completed_timesteps"] = int(state.get("completed_timesteps", 0)) + completed
            stage_record = {
                "name": name,
                "map": current_config.get("metadrive", {}).get("map"),
                "started_with_total_timesteps": before,
                "completed_timesteps": completed,
                "ended_with_total_timesteps": int(model.num_timesteps),
                "gate_reached": bool(validation_callback.target_reached),
                "required_gate": bool(stage.get("require_gate", False)),
                "completed_at_utc": utc_timestamp(),
                "elapsed_seconds": elapsed_seconds,
                "environment_steps_per_second": completed / max(elapsed_seconds, 1e-9),
                "traffic_density": current_config.get("metadrive", {}).get("traffic_density"),
                "traffic_mode": current_config.get("metadrive", {}).get("traffic_mode"),
                "scenario_start_seed": current_config.get("metadrive", {}).get("start_seed"),
                "scenario_count": current_config.get("metadrive", {}).get("num_scenarios"),
                "validation_fingerprint": fingerprint,
            }
            state.setdefault("completed_stages", []).append(stage_record)
            state["next_stage_index"] = stage_index + 1
            state["updated_at_utc"] = utc_timestamp()
            state["resume_artifacts"] = _save_resume_artifacts(
                model,
                run_dir,
                bool(current_training.get("save_replay_buffer", False)),
                logger,
            )
            metadata["curriculum"].update(
                {
                    "completed_timesteps": state["completed_timesteps"],
                    "completed_stages": state["completed_stages"],
                    "next_stage_index": state["next_stage_index"],
                }
            )
            metadata["outputs"]["resume_artifacts"] = state["resume_artifacts"]

            _close_env(train_env, logger)
            _close_env(eval_env, logger)
            train_env = None
            eval_env = None

            if stage.get("require_gate", False) and not validation_callback.target_reached:
                canonical_best = _promote_failed_gate_model(model, run_dir, name)
                metadata["outputs"]["selected_model"] = "best"
                metadata["outputs"]["failed_gate_best_model"] = str(canonical_best)
                state["status"] = "failed_gate"
                metadata["status"] = "failed_gate"
                metadata["completed_at_utc"] = utc_timestamp()
                write_json(state, run_dir / "curriculum_state.json")
                write_json(metadata, metadata_path)
                update_latest_run_file(output_dir, _latest_name(requested_config, seed), run_dir)
                logger.error(
                    "Required curriculum gate was not reached in stage %s. The run was "
                    "saved for diagnosis and will not advance automatically.",
                    name,
                )
                return

            write_json(state, run_dir / "curriculum_state.json")
            metadata["status"] = "paused"
            write_json(metadata, metadata_path)
            update_latest_run_file(output_dir, _latest_name(requested_config, seed), run_dir)
            logger.info(
                "Curriculum stage %s complete: %s timesteps, gate reached=%s.",
                name,
                completed,
                validation_callback.target_reached,
            )
            if args.stop_after_stage == name:
                state["status"] = "paused"
                state["updated_at_utc"] = utc_timestamp()
                write_json(state, run_dir / "curriculum_state.json")
                logger.info(
                    "Paused after stage %s. Copy this run to Drive, then resume with "
                    "--run-dir %s.",
                    name,
                    run_dir,
                )
                return

        final_model = run_dir / "models" / "final_model"
        model.save(final_model)
        state["status"] = "complete"
        state["updated_at_utc"] = utc_timestamp()
        metadata["status"] = "complete"
        metadata["completed_at_utc"] = utc_timestamp()
        metadata["outputs"]["final_model"] = f"{final_model}.zip"
        metadata["outputs"]["selected_model"] = (
            "best" if (run_dir / "models" / "best_model.zip").exists() else "final"
        )
        write_json(state, run_dir / "curriculum_state.json")
        try:
            metadata["outputs"]["training_plot"] = str(plot_training_returns(run_dir))
        except Exception as error:
            logger.warning("Training plot generation skipped: %s", error)
        pointer = update_latest_run_file(
            output_dir,
            _latest_name(requested_config, seed),
            run_dir,
        )
        metadata["outputs"]["latest_pointer"] = str(pointer)
        write_json(metadata, metadata_path)
        append_run_manifest(
            output_dir,
            str(experiment.get("phase", "phase2")),
            {
                "kind": _latest_name(requested_config, seed),
                "algorithm": "sac",
                "status": "complete",
                "run_dir": str(run_dir),
                "curriculum": True,
            },
        )
        logger.info("Curriculum training complete. Run scripts.evaluate on the frozen best model.")
    except Exception as error:
        state["status"] = "failed"
        state["updated_at_utc"] = utc_timestamp()
        state["error"] = str(error)
        metadata["status"] = "failed"
        metadata["failed_at_utc"] = utc_timestamp()
        metadata["error"] = str(error)
        write_json(state, run_dir / "curriculum_state.json")
        write_json(metadata, metadata_path)
        append_run_manifest(
            output_dir,
            str(experiment.get("phase", "phase2")),
            {
                "kind": _latest_name(requested_config, seed),
                "algorithm": "sac",
                "status": "failed",
                "run_dir": str(run_dir),
                "error": str(error),
            },
        )
        logger.exception("Curriculum training failed.")
        raise
    finally:
        _close_env(train_env, logger)
        _close_env(eval_env, logger)


if __name__ == "__main__":
    main()
