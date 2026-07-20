"""MetaDrive environment factories for Stable-Baselines3."""

import copy
from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


def sanitize_metadrive_config(env_config):
    """Add phase-1 defaults while preserving user-provided config values."""
    cfg = copy.deepcopy(env_config)
    cfg.setdefault("use_render", False)
    cfg.setdefault("log_level", 50)
    cfg.setdefault("discrete_action", False)
    cfg.setdefault("horizon", 1000)
    cfg.setdefault("traffic_density", 0.1)
    cfg.setdefault("num_scenarios", 1)
    cfg.setdefault("start_seed", 0)
    return cfg


def make_metadrive_env(env_config, seed=None, monitor_file=None):
    """Create a single MetaDriveEnv instance wrapped with SB3 Monitor."""
    from metadrive.envs import MetaDriveEnv

    cfg = sanitize_metadrive_config(env_config)
    env = MetaDriveEnv(cfg)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    return Monitor(
        env, filename=str(monitor_file) if monitor_file else None, allow_early_resets=True
    )


def make_env_fn(env_config, rank, seed, monitor_dir=None):
    """Return a thunk for DummyVecEnv/SubprocVecEnv."""

    def _init():
        monitor_file = None
        if monitor_dir is not None:
            Path(monitor_dir).mkdir(parents=True, exist_ok=True)
            monitor_file = Path(monitor_dir) / f"env_{rank}.monitor.csv"
        return make_metadrive_env(env_config, seed=seed + rank, monitor_file=monitor_file)

    return _init


def make_vec_env(
    env_config,
    n_envs,
    seed,
    monitor_dir,
    vec_env_type="subproc",
    normalize_obs=True,
    normalize_reward=False,
    training=True,
):
    """Build an SB3 VecEnv for MetaDrive."""
    env_fns = [
        make_env_fn(env_config, rank=i, seed=seed, monitor_dir=monitor_dir) for i in range(n_envs)
    ]
    if vec_env_type == "subproc" and n_envs > 1:
        # spawn is safer on macOS and with graphics-related libraries.
        venv = SubprocVecEnv(env_fns, start_method="spawn")
    elif vec_env_type == "dummy" or n_envs == 1:
        venv = DummyVecEnv(env_fns)
    else:
        raise ValueError(f"Unknown vec_env_type={vec_env_type!r}; expected 'dummy' or 'subproc'.")

    if normalize_obs or normalize_reward:
        venv = VecNormalize(
            venv,
            training=training,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            clip_obs=10.0,
        )
    return venv
