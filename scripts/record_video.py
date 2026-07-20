"""Record a top-down video of a trained MetaDrive policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from saferl_drive.algorithms import get_algorithm_class
from saferl_drive.config import apply_dotlist_overrides, load_yaml, make_eval_metadrive_config
from saferl_drive.envs import make_vec_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a policy rollout video in MetaDrive.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="final", choices=["final", "best"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--screen-size", type=int, default=800)
    parser.add_argument("overrides", nargs="*", help="Dotlist config overrides.")
    return parser.parse_args()


def _to_uint8_frame(frame) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    cfg = apply_dotlist_overrides(cfg, args.overrides)

    algo_name = cfg.get("algorithm", {}).get("name", "ppo").lower()
    algo_cls = get_algorithm_class(algo_name)
    model_path = run_dir / "models" / ("best_model.zip" if args.model == "best" else "final_model.zip")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Use eval config but force one scenario for a clean video.
    env_cfg = make_eval_metadrive_config(cfg)
    env_cfg["num_scenarios"] = 1
    env_cfg["use_render"] = False

    base_env = make_vec_env(
        env_config=env_cfg,
        n_envs=1,
        seed=int(cfg.get("experiment", {}).get("seed", 0)) + 30_000,
        monitor_dir=run_dir / "logs" / "video_monitor",
        vec_env_type="dummy",
        normalize_obs=False,
        normalize_reward=False,
        training=False,
    )
    vecnorm_path = run_dir / "models" / "vecnormalize.pkl"
    if vecnorm_path.exists():
        env = VecNormalize.load(vecnorm_path, base_env)
        env.training = False
        env.norm_reward = False
    else:
        env = base_env

    model = algo_cls.load(model_path, env=env)
    obs = env.reset()
    frames = []
    for _ in range(args.steps):
        # Render before stepping so the starting scene is included.
        rendered = env.env_method(
            "render",
            mode="topdown",
            window=False,
            screen_size=(args.screen_size, args.screen_size),
        )[0]
        if rendered is not None:
            frames.append(_to_uint8_frame(rendered))
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if bool(dones[0]):
            rendered = env.env_method(
                "render",
                mode="topdown",
                window=False,
                screen_size=(args.screen_size, args.screen_size),
            )[0]
            if rendered is not None:
                frames.append(_to_uint8_frame(rendered))
            break

    if not frames:
        raise RuntimeError("No frames were rendered. Try checking your MetaDrive installation/render backend.")

    output = Path(args.output) if args.output else run_dir / "videos" / f"{algo_name}_{args.model}_topdown.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=args.fps)
    print(f"Saved video: {output}")
    env.close()


if __name__ == "__main__":
    main()
