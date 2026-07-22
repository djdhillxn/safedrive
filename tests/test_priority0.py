import ast
import math
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from saferl_drive.config import get_evaluation_config, load_yaml, make_eval_metadrive_config
from saferl_drive.evaluation import (
    _episode_row,
    _speed_km_h,
    checkpoint_selection_score,
    evaluate_policy_vecenv,
    summarize_metrics,
)
from saferl_drive.envs import (
    _MetaDriveScenarioSeedWrapper,
    _SafeDriveRewardWrapper,
    _SteeringLimitWrapper,
    _valid_scenario_seed,
    _worker_scenario_seed,
)


def test_worker_scenario_seed_is_independent_of_training_rng_seed():
    config = {"start_seed": 5, "num_scenarios": 1}

    assert [_worker_scenario_seed(config, rank) for rank in range(4)] == [5, 5, 5, 5]


def test_workers_cycle_through_only_valid_scenario_seeds():
    config = {"start_seed": 1000, "num_scenarios": 2}

    assert [_worker_scenario_seed(config, rank) for rank in range(5)] == [
        1000,
        1001,
        1000,
        1001,
        1000,
    ]


def test_generic_rng_seeds_are_mapped_but_explicit_scenario_seeds_are_preserved():
    assert _valid_scenario_seed(0, start_seed=5, num_scenarios=1) == 5
    assert _valid_scenario_seed(42, start_seed=1000, num_scenarios=2) == 1000
    assert _valid_scenario_seed(1001, start_seed=1000, num_scenarios=2) == 1001


def test_scenario_seed_wrapper_maps_each_reset():
    class _SeedCaptureEnvironment(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,))
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
            self.last_seed = None

        def reset(self, *, seed=None, options=None):
            self.last_seed = seed
            return np.zeros(1), {}

    base = _SeedCaptureEnvironment()
    environment = _MetaDriveScenarioSeedWrapper(base, start_seed=5, num_scenarios=1)

    environment.reset(seed=0)

    assert base.last_seed == 5


def test_steering_limit_changes_the_action_space_and_clips_actions():
    class _ActionCaptureEnvironment(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,))
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
            self.last_action = None

        def step(self, action):
            self.last_action = action
            return np.zeros(1), 0.0, False, False, {}

    base = _ActionCaptureEnvironment()
    environment = _SteeringLimitWrapper(base, steering_limit=0.1)

    environment.step(np.array([0.8, -0.5]))

    assert np.allclose(environment.action_space.low, [-0.1, -1.0])
    assert np.allclose(environment.action_space.high, [0.1, 1.0])
    assert np.allclose(base.last_action, [0.1, -0.5])


def test_vector_evaluation_does_not_reseed_the_training_process():
    class _OneStepEnvironment(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,))
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1), {}

        def step(self, action):
            info = {"arrive_dest": True, "route_completion": 1.0}
            return np.zeros(1), 1.0, True, False, info

    class _ZeroModel:
        def predict(self, observation, deterministic=True):
            return np.zeros((1, 2)), None

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected = (random.random(), np.random.random(), float(torch.rand(1)))

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    environment = DummyVecEnv([_OneStepEnvironment])
    try:
        evaluate_policy_vecenv(
            _ZeroModel(),
            environment,
            episodes=2,
            progress=False,
            start_seed=1000,
            num_scenarios=2,
        )
    finally:
        environment.close()
    actual = (random.random(), np.random.random(), float(torch.rand(1)))

    assert np.allclose(actual, expected)


def test_metadrive_velocity_is_already_km_h():
    assert _speed_km_h({"velocity": 18.5}) == 18.5


def test_generic_speed_is_converted_from_metres_per_second():
    assert _speed_km_h({"speed": 10.0}) == 36.0


def test_explicit_speed_km_h_takes_priority():
    info = {"speed_km_h": 21.0, "velocity": 18.5, "speed": 10.0}
    assert _speed_km_h(info) == 21.0


def test_old_eval_config_becomes_deterministic():
    config = {
        "metadrive": {
            "start_seed": 0,
            "num_scenarios": 50,
            "random_traffic": True,
            "random_spawn_lane_index": True,
        },
        "eval": {"start_seed": 1000, "num_scenarios": 50},
    }

    evaluation = get_evaluation_config(config, "test")
    metadrive = make_eval_metadrive_config(config, "test")

    assert evaluation["start_seed"] == 1000
    assert metadrive["start_seed"] == 1000
    assert metadrive["random_traffic"] is False
    assert metadrive["random_spawn_lane_index"] is False


def test_validation_and_test_settings_are_separate():
    config = {
        "metadrive": {"start_seed": 0, "num_scenarios": 50},
        "validation": {"start_seed": 1000, "num_scenarios": 50},
        "test": {"start_seed": 3000, "num_scenarios": 100},
    }

    assert get_evaluation_config(config, "validation")["start_seed"] == 1000
    assert get_evaluation_config(config, "test")["start_seed"] == 3000
    assert make_eval_metadrive_config(config, "test")["num_scenarios"] == 100


def test_phase_one_configs_define_the_scoped_learning_tasks():
    for path in [Path("configs/ppo_mvp.yaml"), Path("configs/sac_mvp.yaml")]:
        config = load_yaml(path)

        assert config["metadrive"]["random_traffic"] is False
        assert config["train"]["normalize_obs"] is False
        assert config["metadrive"]["map"] == "S"
        assert config["metadrive"]["num_scenarios"] == 1
        assert config["metadrive"]["traffic_density"] == 0.0
        assert config["metadrive"]["random_spawn_lane_index"] is False
        assert config["metadrive"]["truncate_as_terminate"] is False
        assert config["metadrive"]["success_reward"] == 10.0
        assert config["metadrive"]["driving_reward"] == 1.0
        assert config["metadrive"]["reward_shaping"] == {}
        assert config["train"]["stop_success_rate"] == 0.80
        assert config["train"]["stop_route_completion"] == 0.90
        assert config["train"]["stop_max_collision_rate"] == 0.10
        assert config["train"]["stop_max_out_of_road_rate"] == 0.10
        assert config["train"]["stop_max_timeout_rate"] == 0.10
        assert config["validation"]["start_seed"] == 1000
        assert config["validation"]["num_scenarios"] == 10
        assert config["validation"]["episodes"] == 10
        assert config["validation"]["map"] == "S"
        assert config["validation"]["random_traffic"] is False
        assert config["validation"]["random_spawn_lane_index"] is False
        assert config["test"]["start_seed"] == 4000
        assert config["test"]["num_scenarios"] == 20
        assert config["test"]["episodes"] == 20
        assert config["test"]["map"] == "S"
        assert config["test"]["random_traffic"] is False
        assert config["test"]["random_spawn_lane_index"] is False
        assert config["train"]["checkpoint_freq"] == 25_000

    ppo = load_yaml("configs/ppo_mvp.yaml")
    sac = load_yaml("configs/sac_mvp.yaml")
    assert ppo["metadrive"]["discrete_action"] is True
    assert ppo["metadrive"]["use_multi_discrete"] is True
    assert ppo["metadrive"]["discrete_steering_dim"] == 3
    assert ppo["metadrive"]["discrete_throttle_dim"] == 3
    assert "steering_limit" not in ppo["metadrive"]
    assert sac["metadrive"]["discrete_action"] is False
    assert sac["metadrive"]["steering_limit"] == 0.1
    assert sac["train"]["save_replay_buffer"] is False
    assert sac["algorithm"]["kwargs"]["ent_coef"] == 0.05


class _FakeRewardEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(2,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

    def reset(self, **kwargs):
        return np.zeros(2), {}

    def step(self, action):
        info = {"velocity": 0.0, "max_step": True, "arrive_dest": False}
        return np.zeros(2), 1.0, True, False, info


def test_reward_wrapper_penalizes_stall_timeout_and_control():
    environment = _SafeDriveRewardWrapper(
        _FakeRewardEnvironment(),
        {
            "minimum_speed_km_h": 5.0,
            "low_speed_penalty": 0.05,
            "timeout_penalty": 50.0,
            "steering_penalty": 0.005,
            "steering_smoothness_penalty": 0.01,
        },
    )

    environment.reset()
    _, reward, terminated, truncated, info = environment.step(np.array([1.0, 0.0]))

    assert terminated is True
    assert truncated is False
    assert math.isclose(reward, -49.055)
    assert info["base_reward"] == 1.0
    assert info["low_speed_penalty"] == 0.05
    assert info["timeout_penalty"] == 50.0
    assert info["steering_penalty"] == 0.005
    assert math.isclose(info["shaping_penalty"], 50.055)


def test_reward_wrapper_resets_smoothness_history():
    environment = _SafeDriveRewardWrapper(
        _FakeRewardEnvironment(),
        {"steering_smoothness_penalty": 1.0},
    )

    environment.reset()
    environment.step(np.array([1.0, 0.0]))
    environment.reset()
    _, _, _, _, info = environment.step(np.array([-1.0, 0.0]))

    assert info["steering_smoothness_penalty"] == 0.0


def test_reward_wrapper_uses_the_action_metadrive_actually_applied():
    class _InternalPolicyEnvironment(_FakeRewardEnvironment):
        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            info["action"] = [1.0, 0.0]
            return observation, reward, terminated, truncated, info

    environment = _SafeDriveRewardWrapper(
        _InternalPolicyEnvironment(),
        {"steering_penalty": 0.005},
    )

    environment.reset()
    _, reward, _, _, info = environment.step(np.array([0.0, 0.0]))

    assert reward == 0.995
    assert info["steering_penalty"] == 0.005


def test_episode_metrics_capture_full_action_behavior():
    row = _episode_row(
        episode=0,
        requested_seed=3000,
        return_sum=10.0,
        base_return_sum=12.0,
        shaping_penalty_sum=2.0,
        length=3,
        cost_sum=0.0,
        speeds=[5.0, 10.0, 15.0],
        actions=[[1.0, 1.0], [0.0, -1.0], [-1.0, 0.0]],
        route_completion=0.5,
        final_info={"arrive_dest": False, "max_step": True},
    )

    assert row["mean_abs_steering"] == 2.0 / 3.0
    assert row["steering_saturation_rate"] == 2.0 / 3.0
    assert row["throttle_rate"] == 1.0 / 3.0
    assert row["brake_rate"] == 1.0 / 3.0
    assert row["mean_action_change"] == 1.25


def test_checkpoint_selection_prioritizes_success_then_route_completion():
    safer_but_stationary = {
        "success_rate": 0.0,
        "mean_route_completion": 0.01,
        "out_of_road_rate": 0.0,
        "collision_rate": 0.0,
        "timeout_or_max_step_rate": 1.0,
        "mean_return": 0.0,
    }
    useful_progress = {
        "success_rate": 0.0,
        "mean_route_completion": 0.4,
        "out_of_road_rate": 0.1,
        "collision_rate": 0.1,
        "timeout_or_max_step_rate": 0.8,
        "mean_return": 20.0,
    }
    successful = {**safer_but_stationary, "success_rate": 0.05}

    assert checkpoint_selection_score(useful_progress) > checkpoint_selection_score(
        safer_but_stationary
    )
    assert checkpoint_selection_score(successful) > checkpoint_selection_score(useful_progress)


def test_summary_includes_a_finite_success_confidence_interval():
    frame = pd.DataFrame(
        {
            "return_sum": [1.0, 2.0, 3.0, 4.0],
            "length": [10, 10, 10, 10],
            "success": [True, False, False, True],
            "crash": [False, True, False, False],
            "out_of_road": [False, False, True, False],
            "max_step": [False, False, False, False],
            "cost_sum": [0.0, 1.0, 1.0, 0.0],
            "route_completion": [1.0, 0.5, 0.4, 1.0],
            "mean_speed_km_h": [10.0, 11.0, 12.0, 13.0],
        }
    )

    summary = summarize_metrics(frame)

    assert summary["success_rate"] == 0.5
    assert math.isfinite(summary["success_rate_95ci_low"])
    assert math.isfinite(summary["success_rate_95ci_high"])
    assert summary["success_rate_95ci_low"] < 0.5 < summary["success_rate_95ci_high"]


def test_project_python_stays_free_of_type_hints_and_future_annotations():
    roots = [Path("saferl_drive"), Path("scripts"), Path("tests")]
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    assert node.returns is None, f"Return type hint found in {path}"
                    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                    if node.args.vararg is not None:
                        arguments.append(node.args.vararg)
                    if node.args.kwarg is not None:
                        arguments.append(node.args.kwarg)
                    assert all(argument.annotation is None for argument in arguments), (
                        f"Parameter type hint found in {path}"
                    )
                assert not isinstance(node, ast.AnnAssign), f"Variable type hint found in {path}"
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    imported = {name.name for name in node.names}
                    assert "annotations" not in imported, (
                        f"Future annotations import found in {path}"
                    )
