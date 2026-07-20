"""Stable-Baselines3 algorithm construction."""

from stable_baselines3 import PPO, SAC


def get_algorithm_class(name):
    name = name.lower()
    if name == "ppo":
        return PPO
    if name == "sac":
        return SAC
    raise ValueError(f"Unsupported algorithm {name!r}. Phase 1 supports: ppo, sac.")


def build_model(algo_cfg, env, seed, tensorboard_log=None):
    """Instantiate a PPO/SAC model from YAML config."""
    algo_name = algo_cfg.get("name", "ppo").lower()
    policy = algo_cfg.get("policy", "MlpPolicy")
    kwargs = dict(algo_cfg.get("kwargs", {}))
    if tensorboard_log is not None:
        kwargs["tensorboard_log"] = tensorboard_log
    algo_cls = get_algorithm_class(algo_name)
    return algo_cls(policy, env, seed=seed, **kwargs)
