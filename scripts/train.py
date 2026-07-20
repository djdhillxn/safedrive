"""Train PPO or SAC on MetaDrive.

Example:
    python -m scripts.train --config configs/ppo_mvp.yaml
    python -m scripts.train --config configs/sac_mvp.yaml train.total_timesteps=1000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from saferl_drive.algorithms import build_model
from saferl_drive.config import apply_dotlist_overrides, load_yaml, make_eval_metadrive_config, save_yaml
from saferl_drive.envs import make_vec_env
from saferl_drive.evaluation import evaluate_policy_vecenv, save_eval_outputs
from saferl_drive.plotting import plot_eval_summary, plot_training_returns
from saferl_drive.utils import make_run_dir, set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/SAC on MetaDrive.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name override.")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], default=None, help="Algorithm override.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Training steps override.")
    parser.add_argument("--n-envs", type=int, default=None, help="Number of envs override.")
    parser.add_argument("--vec-env", type=str, choices=["dummy", "subproc"], default=None)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra dotlist overrides, e.g. metadrive.traffic_density=0.2 train.n_envs=2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_dotlist_overrides(cfg, args.overrides)

    if args.run_name is not None:
        cfg.setdefault("experiment", {})["name"] = args.run_name
    if args.algo is not None:
        cfg.setdefault("algorithm", {})["name"] = args.algo
    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = args.seed
    if args.total_timesteps is not None:
        cfg.setdefault("train", {})["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        cfg.setdefault("train", {})["n_envs"] = args.n_envs
    if args.vec_env is not None:
        cfg.setdefault("train", {})["vec_env"] = args.vec_env

    exp_cfg = cfg.get("experiment", {})
    train_cfg = cfg.get("train", {})
    algo_cfg = cfg.get("algorithm", {})
    seed = int(exp_cfg.get("seed", 0))
    algo_name = algo_cfg.get("name", "ppo").lower()
    set_global_seeds(seed)

    run_dir = make_run_dir(
        output_dir=exp_cfg.get("output_dir", "runs"),
        experiment_name=exp_cfg.get("name", "metadrive_mvp"),
        algo=algo_name,
        seed=seed,
    )
    save_yaml(cfg, run_dir / "resolved_config.yaml")
    print(f"[SafeRL-Drive] Run directory: {run_dir}")

    env = make_vec_env(
        env_config=cfg.get("metadrive", {}),
        n_envs=int(train_cfg.get("n_envs", 1)),
        seed=seed,
        monitor_dir=run_dir / "logs" / "train_monitor",
        vec_env_type=train_cfg.get("vec_env", "dummy"),
        normalize_obs=bool(train_cfg.get("normalize_obs", True)),
        normalize_reward=bool(train_cfg.get("normalize_reward", False)),
        training=True,
    )

    eval_env = make_vec_env(
        env_config=make_eval_metadrive_config(cfg),
        n_envs=1,
        seed=seed + 10_000,
        monitor_dir=run_dir / "logs" / "eval_monitor",
        vec_env_type="dummy",
        normalize_obs=bool(train_cfg.get("normalize_obs", True)),
        normalize_reward=bool(train_cfg.get("normalize_reward", False)),
        training=False,
    )

    tensorboard_log = str(run_dir / "logs" / "tensorboard") if exp_cfg.get("tensorboard", True) else None
    model = build_model(algo_cfg, env=env, seed=seed, tensorboard_log=tensorboard_log)

    callbacks = []
    checkpoint_freq = int(train_cfg.get("checkpoint_freq", 100_000))
    if checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(checkpoint_freq // max(int(train_cfg.get("n_envs", 1)), 1), 1),
                save_path=str(run_dir / "checkpoints"),
                name_prefix=algo_name,
                save_replay_buffer=(algo_name == "sac"),
                save_vecnormalize=True,
            )
        )

    eval_freq = int(train_cfg.get("eval_freq", 50_000))
    if eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "models"),
                log_path=str(run_dir / "eval" / "sb3_eval"),
                eval_freq=max(eval_freq // max(int(train_cfg.get("n_envs", 1)), 1), 1),
                n_eval_episodes=int(train_cfg.get("eval_episodes", 10)),
                deterministic=True,
                render=False,
            )
        )

    callback = CallbackList(callbacks) if callbacks else None
    model.learn(
        total_timesteps=int(train_cfg.get("total_timesteps", 500_000)),
        callback=callback,
        log_interval=10,
        progress_bar=True,
    )

    final_model_path = run_dir / "models" / "final_model"
    model.save(final_model_path)
    if isinstance(env, VecNormalize):
        env.save(run_dir / "models" / "vecnormalize.pkl")
    print(f"[SafeRL-Drive] Saved final model to {final_model_path}.zip")

    # Post-training closed-loop evaluation with AV-specific metrics.
    if isinstance(eval_env, VecNormalize) and isinstance(env, VecNormalize):
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms
        eval_env.training = False
        eval_env.norm_reward = False

    eval_df = evaluate_policy_vecenv(
        model,
        eval_env,
        episodes=int(cfg.get("eval", {}).get("episodes", 50)),
        deterministic=bool(cfg.get("eval", {}).get("deterministic", True)),
    )
    paths = save_eval_outputs(eval_df, run_dir / "eval", prefix="final_unseen")
    print(f"[SafeRL-Drive] Wrote eval CSV: {paths['episodes_csv']}")
    print(f"[SafeRL-Drive] Wrote eval summary: {paths['summary_json']}")

    try:
        plot_training_returns(run_dir)
        plot_eval_summary(paths["episodes_csv"], out_dir=run_dir / "plots")
    except Exception as exc:
        print(f"[SafeRL-Drive] Plotting skipped: {exc}")

    env.close()
    eval_env.close()
    print("[SafeRL-Drive] Done.")


if __name__ == "__main__":
    main()
