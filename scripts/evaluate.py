"""Evaluate a trained PPO/SAC model on unseen MetaDrive scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.vec_env import VecNormalize

from saferl_drive.algorithms import get_algorithm_class
from saferl_drive.config import apply_dotlist_overrides, load_yaml, make_eval_metadrive_config
from saferl_drive.envs import make_vec_env
from saferl_drive.evaluation import evaluate_policy_vecenv, save_eval_outputs
from saferl_drive.plotting import plot_eval_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MetaDrive policy.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory produced by scripts.train.")
    parser.add_argument("--model", type=str, default="final", choices=["final", "best"], help="Which model to load.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes override.")
    parser.add_argument("--prefix", type=str, default="eval_unseen", help="Output filename prefix.")
    parser.add_argument("overrides", nargs="*", help="Dotlist config overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    cfg = apply_dotlist_overrides(cfg, args.overrides)
    if args.episodes is not None:
        cfg.setdefault("eval", {})["episodes"] = args.episodes

    algo_name = cfg.get("algorithm", {}).get("name", "ppo").lower()
    algo_cls = get_algorithm_class(algo_name)
    model_path = run_dir / "models" / ("best_model.zip" if args.model == "best" else "final_model.zip")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Build unnormalized base VecEnv first; load VecNormalize stats if available.
    base_env = make_vec_env(
        env_config=make_eval_metadrive_config(cfg),
        n_envs=1,
        seed=int(cfg.get("experiment", {}).get("seed", 0)) + 20_000,
        monitor_dir=run_dir / "logs" / f"{args.prefix}_monitor",
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
    df = evaluate_policy_vecenv(
        model,
        env,
        episodes=int(cfg.get("eval", {}).get("episodes", 50)),
        deterministic=bool(cfg.get("eval", {}).get("deterministic", True)),
    )
    paths = save_eval_outputs(df, run_dir / "eval", prefix=args.prefix)
    plot_eval_summary(paths["episodes_csv"], out_dir=run_dir / "plots")

    print(f"Evaluation complete: {paths['summary_json']}")
    env.close()


if __name__ == "__main__":
    main()
