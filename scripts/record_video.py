"""Record one top-down rollout of a trained MetaDrive policy."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from saferl_drive.algorithms import get_algorithm_class
from saferl_drive.config import (
    apply_dotlist_overrides,
    get_evaluation_config,
    load_yaml,
    make_eval_metadrive_config,
)
from saferl_drive.envs import find_vecnormalize_path, make_vec_env
from saferl_drive.utils import log_system_info, set_global_seeds, setup_logging, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Record a policy rollout video in MetaDrive.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="final", choices=["final", "best"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None, help="Scenario seed to record.")
    parser.add_argument("--device", default=None, help="SB3 load device override.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", default=None)
    parser.add_argument("--screen-size", type=int, default=800)
    parser.add_argument("overrides", nargs="*", help="Dotlist config overrides.")
    return parser.parse_args()


def _to_uint8_frame(frame):
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "resolved_config.yaml")
    config = apply_dotlist_overrides(config, args.overrides)
    logging_config = config.get("logging", {})
    logger = setup_logging(
        run_dir / "logs" / f"video_{args.model}.log",
        console_level=logging_config.get("console_level", "INFO"),
        file_level=logging_config.get("file_level", "DEBUG"),
    )
    environment = None

    try:
        log_system_info(logger, run_dir=run_dir)
        logger.debug("Arguments: %s", vars(args))
        logger.debug("Resolved video config: %s", config)

        algorithm_name = config.get("algorithm", {}).get("name", "ppo").lower()
        algorithm_class = get_algorithm_class(algorithm_name)
        selected_model = args.model
        model_path = run_dir / "models" / f"{selected_model}_model.zip"
        if not model_path.exists() and selected_model == "best":
            final_path = run_dir / "models" / "final_model.zip"
            if final_path.exists():
                logger.warning("Best model is missing; recording the final model instead.")
                selected_model = "final"
                model_path = final_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        testing = get_evaluation_config(config, "test")
        video_seed = args.seed
        if video_seed is None:
            video_seed = int(testing.get("start_seed", 4000))
        set_global_seeds(video_seed)

        environment_config = make_eval_metadrive_config(config, "test")
        environment_config["start_seed"] = video_seed
        environment_config["num_scenarios"] = 1
        environment_config["random_traffic"] = False
        environment_config["random_spawn_lane_index"] = False
        environment_config["use_render"] = False
        base_env = make_vec_env(
            env_config=environment_config,
            n_envs=1,
            seed=video_seed,
            monitor_dir=run_dir / "logs" / f"video_{selected_model}_monitor",
            vec_env_type="dummy",
            normalize_obs=False,
            normalize_reward=False,
            training=False,
        )
        environment = base_env
        vecnormalize_path = find_vecnormalize_path(run_dir, selected_model)
        if vecnormalize_path is not None:
            environment = VecNormalize.load(vecnormalize_path, base_env)
            environment.training = False
            environment.norm_reward = False
            logger.debug("Loaded VecNormalize statistics: %s", vecnormalize_path)
            if selected_model == "best" and vecnormalize_path.name != "best_vecnormalize.pkl":
                logger.warning(
                    "This older run has no best-model normalization snapshot; using the "
                    "final normalization statistics."
                )

        configured_device = config.get("algorithm", {}).get("kwargs", {}).get("device")
        load_device = (
            args.device or configured_device or ("cpu" if algorithm_name == "ppo" else "auto")
        )
        logger.info(
            "Loading %s model on %s for deterministic scenario seed %s: %s",
            selected_model,
            load_device,
            video_seed,
            model_path,
        )
        model = algorithm_class.load(model_path, env=environment, device=load_device)
        environment.seed(video_seed)
        observation = environment.reset()
        frames = []
        final_info = {}
        for _ in range(args.steps):
            rendered = environment.env_method(
                "render",
                mode="topdown",
                window=False,
                screen_size=(args.screen_size, args.screen_size),
            )[0]
            if rendered is not None:
                frames.append(_to_uint8_frame(rendered))
            action, _ = model.predict(observation, deterministic=True)
            observation, rewards, dones, infos = environment.step(action)
            final_info = dict(infos[0])
            if bool(dones[0]):
                # VecEnv resets automatically at termination, so another render here
                # would append the first frame of a new episode to this video.
                break

        if not frames:
            raise RuntimeError(
                "No frames were rendered. Check the MetaDrive installation and render backend."
            )

        output = (
            Path(args.output)
            if args.output
            else run_dir
            / "videos"
            / f"{algorithm_name}_{selected_model}_seed{video_seed}_topdown.mp4"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output, frames, fps=args.fps)
        write_json(
            {
                "algorithm": algorithm_name,
                "model": selected_model,
                "scenario_seed": video_seed,
                "deterministic_policy": True,
                "random_traffic": False,
                "frames": len(frames),
                "fps": args.fps,
                "video": str(output),
                "final_outcome": {
                    "success": bool(final_info.get("arrive_dest", False)),
                    "crash": bool(final_info.get("crash", False)),
                    "out_of_road": bool(final_info.get("out_of_road", False)),
                    "max_step": bool(final_info.get("max_step", False)),
                    "route_completion": float(final_info.get("route_completion", 0.0)),
                },
            },
            output.with_suffix(".json"),
        )
        logger.debug("Recorded %s frames at %s FPS.", len(frames), args.fps)
        logger.info("Saved video: %s", output)
    except Exception:
        logger.exception("Video recording failed.")
        raise
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    main()
