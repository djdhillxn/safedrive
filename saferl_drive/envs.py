"""MetaDrive environment factories for Stable-Baselines3."""

import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from saferl_drive.utils import set_global_seeds


MONITOR_INFO_KEYS = (
    "arrive_dest",
    "crash",
    "out_of_road",
    "max_step",
    "route_completion",
    "cost",
)


class _SafeDriveRewardWrapper(gym.Wrapper):
    """Make stalling and unstable control visibly worse than useful progress."""

    def __init__(self, env, settings):
        super().__init__(env)
        self.settings = copy.deepcopy(settings)
        self.previous_action = None

    def reset(self, **kwargs):
        self.previous_action = None
        return self.env.reset(**kwargs)

    def _lane_penalty(self):
        weight = float(self.settings.get("lateral_penalty", 0.0))
        if weight <= 0.0:
            return 0.0, 0.0
        try:
            vehicle = self.env.unwrapped.agent
            reference_lanes = vehicle.navigation.current_ref_lanes
            lane = vehicle.lane if vehicle.lane in reference_lanes else reference_lanes[0]
            _, lateral = lane.local_coordinates(vehicle.position)
            half_width = vehicle.navigation.get_current_lane_width() / 2.0
            deviation = min(abs(float(lateral)) / max(float(half_width), 1e-6), 1.0)
        except (AttributeError, AssertionError, IndexError, TypeError, ValueError):
            return 0.0, 0.0
        return weight * deviation**2, deviation

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        recorded_action = info.get("action", info.get("raw_action", action))
        if recorded_action is None:
            recorded_action = action
        action_values = np.asarray(recorded_action, dtype=float).reshape(-1)

        minimum_speed = float(self.settings.get("minimum_speed_km_h", 0.0))
        low_speed_weight = float(self.settings.get("low_speed_penalty", 0.0))
        speed = float(info.get("velocity", 0.0))
        low_speed_penalty = 0.0
        if minimum_speed > 0.0 and not info.get("arrive_dest", False):
            shortfall = max(minimum_speed - speed, 0.0) / minimum_speed
            low_speed_penalty = low_speed_weight * shortfall

        lateral_penalty, lateral_deviation = self._lane_penalty()

        steering_penalty = 0.0
        if action_values.size:
            steering_weight = float(self.settings.get("steering_penalty", 0.0))
            steering_penalty = steering_weight * float(action_values[0] ** 2)

        smoothness_penalty = 0.0
        if self.previous_action is not None and action_values.size:
            smoothness_weight = float(self.settings.get("steering_smoothness_penalty", 0.0))
            steering_change = float(action_values[0] - self.previous_action[0])
            smoothness_penalty = smoothness_weight * steering_change**2
        self.previous_action = action_values.copy()

        timeout_penalty = 0.0
        if info.get("max_step", False) and not info.get("arrive_dest", False):
            timeout_penalty = float(self.settings.get("timeout_penalty", 0.0))

        shaping_penalty = (
            low_speed_penalty
            + lateral_penalty
            + steering_penalty
            + smoothness_penalty
            + timeout_penalty
        )
        reward = float(base_reward) - shaping_penalty
        info["base_reward"] = float(base_reward)
        info["low_speed_penalty"] = low_speed_penalty
        info["lateral_penalty"] = lateral_penalty
        info["lateral_deviation_ratio"] = lateral_deviation
        info["steering_penalty"] = steering_penalty
        info["steering_smoothness_penalty"] = smoothness_penalty
        info["timeout_penalty"] = timeout_penalty
        info["shaping_penalty"] = shaping_penalty
        info["step_reward"] = reward
        return observation, reward, terminated, truncated, info


class _MetaDriveScenarioSeedWrapper(gym.Wrapper):
    """Keep reset seeds valid and optionally cycle through configured scenarios."""

    def __init__(self, env, start_seed, num_scenarios, sequential_seed=False):
        super().__init__(env)
        self.start_seed = int(start_seed)
        self.num_scenarios = max(int(num_scenarios), 1)
        self.sequential_seed = bool(sequential_seed)
        self.next_sequential_seed = self.start_seed

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            kwargs["seed"] = _valid_scenario_seed(seed, self.start_seed, self.num_scenarios)
        elif self.sequential_seed:
            kwargs["seed"] = self.next_sequential_seed
            offset = self.next_sequential_seed - self.start_seed + 1
            self.next_sequential_seed = self.start_seed + offset % self.num_scenarios
        return self.env.reset(**kwargs)


class _SteeringLimitWrapper(gym.ActionWrapper):
    """Expose a smaller steering range for the straight-road learning control."""

    def __init__(self, env, steering_limit):
        super().__init__(env)
        limit = float(steering_limit)
        if not 0.0 < limit <= 1.0:
            raise ValueError("steering_limit must be greater than 0 and at most 1.")
        if not isinstance(env.action_space, gym.spaces.Box) or env.action_space.shape[0] < 2:
            raise ValueError("steering_limit requires MetaDrive's continuous two-action space.")
        low = env.action_space.low.copy()
        high = env.action_space.high.copy()
        low[0] = max(float(low[0]), -limit)
        high[0] = min(float(high[0]), limit)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=env.action_space.dtype)

    def reset(self, **kwargs):
        # MetaDrive 0.4.3 predates Gymnasium's ``options`` reset argument.
        return self.env.reset(**kwargs)

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


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


def make_metadrive_env(env_config, seed=None, monitor_file=None, scenario_seed=None):
    """Create a single MetaDriveEnv instance wrapped with SB3 Monitor."""
    from metadrive.envs import MetaDriveEnv

    cfg = sanitize_metadrive_config(env_config)
    reward_shaping = cfg.pop("reward_shaping", None)
    steering_limit = cfg.pop("steering_limit", None)
    sequential_seed = bool(cfg.pop("sequential_seed", False))
    policy_name = cfg.pop("_safedrive_agent_policy", None)
    main_camera_capture = bool(cfg.pop("_safedrive_main_camera", False))
    if main_camera_capture:
        from metadrive.engine.core.main_camera import MainCamera
        from metadrive.obs.state_obs import LidarStateObservation

        cfg.setdefault("sensors", {})["main_camera"] = (MainCamera,)
        # MetaDrive 0.4.3 only retains offscreen cameras when this rendering
        # service switch is enabled. agent_observation keeps the policy-facing
        # observation as LidarState instead of changing it to RGB.
        cfg["image_observation"] = True
        cfg["agent_observation"] = LidarStateObservation
    if policy_name == "IDMPolicy":
        # Import the native MetaDrive policy only in the process that owns the
        # engine. This keeps the parent evaluator free of Panda3D state.
        from metadrive.policy.idm_policy import IDMPolicy

        cfg["agent_policy"] = IDMPolicy
    elif policy_name == "ExpertPolicy":
        from metadrive.policy.expert_policy import ExpertPolicy

        cfg["agent_policy"] = ExpertPolicy
    elif policy_name is not None:
        raise ValueError(f"Unknown SafeDrive agent policy: {policy_name!r}")
    if seed is not None:
        set_global_seeds(seed)
    env = MetaDriveEnv(cfg)
    if steering_limit is not None:
        env = _SteeringLimitWrapper(env, steering_limit)
    env = _MetaDriveScenarioSeedWrapper(
        env,
        start_seed=cfg.get("start_seed", 0),
        num_scenarios=cfg.get("num_scenarios", 1),
        sequential_seed=sequential_seed,
    )
    if reward_shaping:
        env = _SafeDriveRewardWrapper(env, reward_shaping)
    if scenario_seed is None:
        scenario_seed = seed
    if scenario_seed is not None:
        try:
            env.reset(seed=scenario_seed)
        except TypeError:
            env.unwrapped.seed(scenario_seed)
    if seed is not None:
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    return Monitor(
        env,
        filename=str(monitor_file) if monitor_file else None,
        allow_early_resets=True,
        info_keywords=MONITOR_INFO_KEYS,
    )


def make_env_fn(env_config, rank, seed, monitor_dir=None):
    """Return a thunk for DummyVecEnv/SubprocVecEnv."""

    scenario_seed = _worker_scenario_seed(env_config, rank)

    def _init():
        monitor_file = None
        if monitor_dir is not None:
            Path(monitor_dir).mkdir(parents=True, exist_ok=True)
            monitor_file = Path(monitor_dir) / f"env_{rank}.monitor.csv"
        return make_metadrive_env(
            env_config,
            seed=seed + rank,
            monitor_file=monitor_file,
            scenario_seed=scenario_seed,
        )

    return _init


def _worker_scenario_seed(env_config, rank):
    """Map a worker onto a valid MetaDrive scenario independently of its RNG seed."""
    start_seed = int(env_config.get("start_seed", 0))
    num_scenarios = max(int(env_config.get("num_scenarios", 1)), 1)
    return start_seed + rank % num_scenarios


def _valid_scenario_seed(seed, start_seed, num_scenarios):
    """Preserve explicit scenario IDs and map generic RNG seeds into the valid range."""
    seed = int(seed)
    start_seed = int(start_seed)
    num_scenarios = max(int(num_scenarios), 1)
    if start_seed <= seed < start_seed + num_scenarios:
        return seed
    return start_seed + seed % num_scenarios


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
    if vec_env_type == "subproc":
        # MetaDrive has one global engine per process. A subprocess is therefore
        # required even for one environment when another environment is active.
        venv = SubprocVecEnv(env_fns, start_method="spawn")
    elif vec_env_type == "dummy":
        if n_envs > 1:
            raise ValueError(
                "MetaDrive DummyVecEnv supports only one environment because its engine "
                "is process-global. Use vec_env_type='subproc' or set n_envs=1."
            )
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


def find_vecnormalize_path(run_dir, model_name):
    """Find normalization statistics matching the selected model when available."""
    run_dir = Path(run_dir)
    candidates = []
    if model_name == "best":
        candidates.append(run_dir / "models" / "best_vecnormalize.pkl")
    candidates.append(run_dir / "models" / "vecnormalize.pkl")
    for path in candidates:
        if path.exists():
            return path
    return None
