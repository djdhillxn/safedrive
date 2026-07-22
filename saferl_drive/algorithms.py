"""Stable-Baselines3 algorithm construction."""

from stable_baselines3 import PPO, SAC


def get_algorithm_class(name):
    name = name.lower()
    if name == "ppo":
        return PPO
    if name == "sac":
        return SAC
    raise ValueError(f"Unsupported algorithm {name!r}. Phase 1 supports: ppo, sac.")


def validate_algorithm_config(algo_cfg):
    """Reject algorithm settings that cannot produce a valid SB3 model."""
    algo_name = algo_cfg.get("name", "ppo").lower()
    get_algorithm_class(algo_name)
    kwargs = algo_cfg.get("kwargs", {})

    if algo_name == "sac":
        replay_kwargs = kwargs.get("replay_buffer_kwargs", {}) or {}
        optimize_memory = bool(kwargs.get("optimize_memory_usage", False))
        handle_timeouts = bool(replay_kwargs.get("handle_timeout_termination", True))
        if optimize_memory and handle_timeouts:
            raise ValueError(
                "SAC cannot combine optimize_memory_usage=true with "
                "replay_buffer_kwargs.handle_timeout_termination=true. Keep timeout "
                "handling enabled and set optimize_memory_usage=false so horizon "
                "truncations are not learned as terminal failures."
            )


def build_model(algo_cfg, env, seed, tensorboard_log=None):
    """Instantiate a PPO/SAC model from YAML config."""
    validate_algorithm_config(algo_cfg)
    algo_name = algo_cfg.get("name", "ppo").lower()
    policy = algo_cfg.get("policy", "MlpPolicy")
    kwargs = dict(algo_cfg.get("kwargs", {}))
    if tensorboard_log is not None:
        kwargs["tensorboard_log"] = tensorboard_log
    algo_cls = get_algorithm_class(algo_name)
    return algo_cls(policy, env, seed=seed, **kwargs)
