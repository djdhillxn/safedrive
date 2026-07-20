"""Record one top-down rollout of a trained MetaDrive policy."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from saferl_drive.algorithms import get_algorithm_class
from saferl_drive.config import apply_dotlist_overrides, load_yaml, make_eval_metadrive_config
from saferl_drive.envs import make_vec_env
from saferl_drive.utils import log_system_info, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Record a policy rollout video in MetaDrive.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="final", choices=["final", "best"])
    parser.add_argument("--steps", type=int, default=1000)
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

        environment_config = make_eval_metadrive_config(config)
        environment_config["num_scenarios"] = 1
        environment_config["use_render"] = False
        start_seed = int(config.get("eval", {}).get("start_seed", 1000))
        base_env = make_vec_env(
            env_config=environment_config,
            n_envs=1,
            seed=start_seed,
            monitor_dir=run_dir / "logs" / f"video_{selected_model}_monitor",
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

        logger.info("Loading %s model: %s", selected_model, model_path)
        model = algorithm_class.load(model_path, env=environment)
        observation = environment.reset()
        frames = []
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
            if bool(dones[0]):
                rendered = environment.env_method(
                    "render",
                    mode="topdown",
                    window=False,
                    screen_size=(args.screen_size, args.screen_size),
                )[0]
                if rendered is not None:
                    frames.append(_to_uint8_frame(rendered))
                break

        if not frames:
            raise RuntimeError(
                "No frames were rendered. Check the MetaDrive installation and render backend."
            )

        output = (
            Path(args.output)
            if args.output
            else run_dir / "videos" / f"{algorithm_name}_{selected_model}_topdown.mp4"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output, frames, fps=args.fps)
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
