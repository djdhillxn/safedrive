import ast
import math
import random
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from saferl_drive.algorithms import validate_algorithm_config
from saferl_drive.config import (
    fingerprint_differences,
    get_evaluation_config,
    load_yaml,
    make_eval_metadrive_config,
    make_experiment_fingerprint,
    resolve_reward_variant,
)
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
    enable_main_camera_capture,
)
from scripts.compare_runs import (
    _compatibility,
    _traffic_aggregates,
    _traffic_effects,
    select_traffic_pilot,
)
from scripts.evaluate import _condition_config, _load_model
from scripts.record_video import (
    _camera_config,
    _raw_render_environment,
    _rendering_sidecar_fields,
    _validate_frame,
    make_rendering_status,
    select_video_scenario,
)
from scripts.train_curriculum import (
    _promote_failed_gate_model,
    _spaces_match,
    curriculum_stage_budget,
    stage_config,
    validate_curriculum_config,
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


def test_scenario_seed_wrapper_cycles_without_passing_project_option_to_metadrive():
    class _SeedCaptureEnvironment(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,))
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
            self.seeds = []

        def reset(self, *, seed=None, options=None):
            self.seeds.append(seed)
            return np.zeros(1), {}

    base = _SeedCaptureEnvironment()
    environment = _MetaDriveScenarioSeedWrapper(
        base,
        start_seed=10000,
        num_scenarios=3,
        sequential_seed=True,
    )

    for _ in range(5):
        environment.reset()

    assert base.seeds == [10000, 10001, 10002, 10000, 10001]


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


def test_fingerprint_separates_task_and_action_compatibility():
    config = {
        "metadrive": {
            "map": 3,
            "traffic_density": 0.0,
            "horizon": 1000,
            "discrete_action": False,
            "success_reward": 10.0,
        },
        "test": {
            "start_seed": 30000,
            "num_scenarios": 100,
            "episodes": 100,
            "map": 3,
            "traffic_density": 0.0,
        },
    }
    changed_action = {
        **config,
        "metadrive": {**config["metadrive"], "steering_limit": 0.1},
    }
    first = make_experiment_fingerprint(config)
    second = make_experiment_fingerprint(changed_action)

    assert first["task_id"] == second["task_id"]
    assert first["strict_id"] != second["strict_id"]
    assert fingerprint_differences(first, second, include_action=False) == []
    assert fingerprint_differences(first, second, include_action=True)[0]["field"] == (
        "action.steering_limit"
    )


def test_strict_compatibility_refuses_a_changed_test_split():
    config = {
        "metadrive": {"map": 3, "traffic_density": 0.0, "discrete_action": False},
        "test": {"start_seed": 30000, "num_scenarios": 100, "episodes": 100},
    }
    changed = {
        **config,
        "test": {**config["test"], "start_seed": 31000},
    }
    records = [
        {"label": "first", "fingerprint": make_experiment_fingerprint(config)},
        {"label": "second", "fingerprint": make_experiment_fingerprint(changed)},
    ]

    try:
        _compatibility(records, strict=True)
    except ValueError as error:
        assert "evaluation.start_seed" in str(error)
    else:
        raise AssertionError("An incompatible split should have been refused.")


def test_strict_fingerprint_refuses_a_changed_training_budget():
    config = {
        "algorithm": {"name": "sac", "policy": "MlpPolicy", "kwargs": {}},
        "train": {"total_timesteps": 500000},
        "metadrive": {"map": 3, "traffic_density": 0.0, "discrete_action": False},
        "test": {"start_seed": 30000, "num_scenarios": 100, "episodes": 100},
    }
    shorter = {
        **config,
        "train": {"total_timesteps": 100000},
    }

    first = make_experiment_fingerprint(config)
    second = make_experiment_fingerprint(shorter)

    assert first["task_id"] == second["task_id"]
    assert first["strict_id"] != second["strict_id"]
    differences = fingerprint_differences(first, second, include_action=True)
    assert differences[0]["field"] == "training.maximum_timesteps"


def test_video_renderer_reaches_the_unwrapped_environment():
    class _RenderEnvironment(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,))
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def reset(self, *, seed=None, options=None):
            return np.zeros(1), {}

        def step(self, action):
            return np.zeros(1), 0.0, False, False, {}

    base = _RenderEnvironment()
    vector = DummyVecEnv([lambda: gym.Wrapper(base)])
    try:
        assert _raw_render_environment(vector) is base
    finally:
        vector.close()


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


def test_phase_two_configs_share_the_exact_final_task_and_action_interface():
    direct = load_yaml("configs/sac_phase2_direct.yaml")
    curriculum = load_yaml("configs/sac_phase2_curriculum.yaml")
    validate_curriculum_config(curriculum)

    direct_fingerprint = make_experiment_fingerprint(direct, split="test")
    curriculum_fingerprint = make_experiment_fingerprint(curriculum, split="test")

    assert direct_fingerprint["strict_id"] == curriculum_fingerprint["strict_id"]
    assert direct["metadrive"]["map"] == 3
    assert direct["metadrive"]["num_scenarios"] == 100
    assert direct["metadrive"]["sequential_seed"] is True
    assert direct["validation"]["start_seed"] == 20000
    assert direct["test"]["start_seed"] == 30000
    assert direct["test"]["episodes"] == 100
    assert direct["train"]["total_timesteps"] == 500000
    assert curriculum["curriculum"]["total_timesteps"] == 500000
    assert "steering_limit" not in direct["metadrive"]
    assert "steering_limit" not in curriculum["metadrive"]
    for config in [direct, curriculum]:
        kwargs = config["algorithm"]["kwargs"]
        assert kwargs["optimize_memory_usage"] is False
        assert kwargs["replay_buffer_kwargs"]["handle_timeout_termination"] is True
        validate_algorithm_config(config["algorithm"])


def test_traffic_config_is_single_agent_and_matches_the_fixed_plan():
    config = load_yaml("configs/sac_traffic_curriculum.yaml")
    stages = validate_curriculum_config(config)

    assert config["metadrive"]["num_agents"] == 1
    assert config["metadrive"]["is_multi_agent"] is False
    assert config["metadrive"]["image_observation"] is False
    assert config["metadrive"]["map"] == 3
    assert config["metadrive"]["num_scenarios"] == 200
    assert config["metadrive"]["start_seed"] == 40000
    assert config["metadrive"]["traffic_mode"] == "respawn"
    assert config["metadrive"]["random_traffic"] is True
    assert config["source"]["model"] == "best"
    assert config["source"]["fresh_replay_buffer"] is True
    assert config["curriculum"]["total_timesteps"] == 300000
    assert [stage["max_timesteps"] for stage in stages] == [100000, 200000]
    assert [stage["metadrive"]["traffic_density"] for stage in stages] == [0.02, 0.05]
    assert curriculum_stage_budget(config, stages[0], 0) == 100000
    assert curriculum_stage_budget(config, stages[1], 100000) == 200000
    assert config["validation"]["start_seed"] == 50000
    assert config["validation"]["episodes"] == 25
    assert config["test"]["start_seed"] == 60000
    assert config["test"]["episodes"] == 100
    assert config["test"]["densities"] == [0.0, 0.05, 0.10]


def test_traffic_reward_variants_change_only_the_controlled_penalties():
    config = load_yaml("configs/sac_traffic_curriculum.yaml")
    reference = resolve_reward_variant(config, "reference")
    safety = resolve_reward_variant(config, "safety")
    changed = {
        key
        for key in reference["metadrive"]
        if reference["metadrive"].get(key) != safety["metadrive"].get(key)
    }

    assert changed == {"crash_vehicle_penalty", "crash_object_penalty"}
    assert reference["metadrive"]["crash_vehicle_penalty"] == 5.0
    assert safety["metadrive"]["crash_vehicle_penalty"] == 10.0


def test_traffic_config_rejects_multi_agent_or_image_training():
    for key, value, message in [
        ("num_agents", 2, "num_agents=1"),
        ("is_multi_agent", True, "is_multi_agent=false"),
        ("image_observation", True, "vector LidarState"),
    ]:
        changed = load_yaml("configs/sac_traffic_curriculum.yaml")
        changed["metadrive"][key] = value
        try:
            validate_curriculum_config(changed)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"Invalid traffic setting {key} was accepted.")


def test_density_matrix_changes_only_the_requested_evaluation_condition():
    config = load_yaml("configs/sac_traffic_curriculum.yaml")
    changed = _condition_config(config, "test", 0.10)

    assert changed["test"]["traffic_density"] == 0.10
    assert changed["test"]["traffic_mode"] == "respawn"
    assert changed["test"]["random_traffic"] is False
    assert config["test"]["traffic_density"] == 0.05
    assert make_experiment_fingerprint(changed)["task_id"] != make_experiment_fingerprint(
        config
    )["task_id"]


def test_warm_start_space_validation_requires_exact_shapes_and_bounds():
    source = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    same = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    changed_shape = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    changed_bounds = gym.spaces.Box(-0.5, 1.0, shape=(2,), dtype=np.float32)

    assert _spaces_match(source, same)
    assert not _spaces_match(source, changed_shape)
    assert not _spaces_match(source, changed_bounds)


def test_sac_config_rejects_the_incompatible_replay_buffer_options():
    config = {
        "name": "sac",
        "kwargs": {
            "buffer_size": 1000,
            "learning_starts": 100,
            "batch_size": 64,
            "optimize_memory_usage": True,
            "replay_buffer_kwargs": {"handle_timeout_termination": True},
        },
    }

    try:
        validate_algorithm_config(config)
    except ValueError as error:
        assert "cannot combine" in str(error)
    else:
        raise AssertionError("The incompatible SAC replay settings were accepted.")


def test_curriculum_uses_early_stage_savings_on_the_final_stage():
    config = load_yaml("configs/sac_phase2_curriculum.yaml")
    stages = validate_curriculum_config(config)

    assert config["train"]["save_replay_buffer"] is True
    assert curriculum_stage_budget(config, stages[0], 0) == 100000
    assert curriculum_stage_budget(config, stages[1], 75000) == 150000
    assert curriculum_stage_budget(config, stages[2], 200000) == 300000
    assert stage_config(config, stages[0])["metadrive"]["map"] == "C"
    assert stage_config(config, stages[1])["metadrive"]["map"] == "SC"


def test_curriculum_requires_replay_state_for_faithful_resume():
    config = load_yaml("configs/sac_phase2_curriculum.yaml")
    config["train"]["save_replay_buffer"] = False

    try:
        validate_curriculum_config(config)
    except ValueError as error:
        assert "must save its replay buffer" in str(error)
    else:
        raise AssertionError("Curriculum accepted a resume path without replay state.")


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
    assert row["steering_std"] > 0.0
    assert row["steering_saturation_rate"] == 2.0 / 3.0
    assert row["throttle_rate"] == 1.0 / 3.0
    assert row["brake_rate"] == 1.0 / 3.0
    assert row["mean_action_change"] == 1.25


def test_episode_outcome_is_mutually_exclusive_but_raw_flags_are_retained():
    row = _episode_row(
        episode=0,
        requested_seed=60000,
        return_sum=0.0,
        base_return_sum=0.0,
        shaping_penalty_sum=0.0,
        length=1,
        cost_sum=1.0,
        speeds=[],
        actions=[],
        route_completion=0.4,
        final_info={
            "crash": True,
            "crash_vehicle": True,
            "out_of_road": True,
            "max_step": True,
        },
    )

    assert row["terminal_outcome"] == "collision"
    assert row["crash"] is True
    assert row["crash_vehicle"] is True
    assert row["out_of_road"] is True
    assert row["max_step"] is True
    assert row["collision_free"] is False


def test_predeclared_traffic_selection_prefers_the_only_qualifying_pilot():
    source_success = 0.90
    reference = {
        "0.00": {"success_rate": 0.85},
        "0.05": {
            "success_rate": 0.82,
            "collision_rate": 0.08,
            "out_of_road_rate": 0.08,
            "mean_route_completion": 0.92,
        },
        "0.10": {"success_rate": 0.60},
    }
    safety = {
        "0.00": {"success_rate": 0.88},
        "0.05": {
            "success_rate": 0.78,
            "collision_rate": 0.06,
            "out_of_road_rate": 0.08,
            "mean_route_completion": 0.91,
        },
        "0.10": {"success_rate": 0.62},
    }

    decision = select_traffic_pilot(reference, safety, source_success)

    assert decision["selected_variant"] == "reference"
    assert decision["qualification"] == {"reference": True, "safety": False}


def test_predeclared_traffic_selection_uses_collision_for_near_tied_failures():
    reference = {
        "0.00": {"success_rate": 0.90},
        "0.05": {
            "success_rate": 0.70,
            "collision_rate": 0.20,
            "out_of_road_rate": 0.10,
            "mean_route_completion": 0.85,
        },
        "0.10": {"success_rate": 0.50},
    }
    safety = {
        "0.00": {"success_rate": 0.90},
        "0.05": {
            "success_rate": 0.67,
            "collision_rate": 0.12,
            "out_of_road_rate": 0.10,
            "mean_route_completion": 0.84,
        },
        "0.10": {"success_rate": 0.50},
    }

    decision = select_traffic_pilot(reference, safety, 0.90)

    assert decision["selected_variant"] == "safety"


def test_failed_confirmation_is_not_mixed_into_completed_adaptation_aggregates():
    rows = []
    for seed, condition, success in [
        (0, "Frozen curriculum SAC", 0.60),
        (0, "Adapted SAC", 0.85),
        (1, "Frozen curriculum SAC", 0.55),
        (1, "Adapted SAC (failed gate)", 0.40),
    ]:
        for density in [0.0, 0.05, 0.10]:
            rows.append(
                {
                    "condition": condition,
                    "training_seed": seed,
                    "traffic_density": density,
                    "success_rate": success,
                    "collision_rate": 0.10,
                    "mean_route_completion": 0.80,
                }
            )

    aggregates = _traffic_aggregates(rows)
    adapted = [
        row
        for row in aggregates
        if row["condition"] == "Adapted SAC" and row["traffic_density"] == 0.05
    ]
    effects = _traffic_effects(rows)

    assert adapted[0]["training_seeds"] == 1
    assert [row["training_seed"] for row in effects] == [0]
    assert any(row["condition"] == "Adapted SAC (failed gate)" for row in aggregates)


def test_failed_gate_best_checkpoint_is_promoted_for_diagnosis(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    stage_best = models / "traffic_002_best_model.zip"
    stage_best.write_bytes(b"checkpoint")

    canonical = _promote_failed_gate_model(None, tmp_path, "traffic_002")

    assert canonical == models / "best_model.zip"
    assert canonical.read_bytes() == b"checkpoint"


def test_video_scenario_selection_is_systematic(tmp_path):
    path = tmp_path / "episodes.csv"
    pd.DataFrame(
        {
            "episode": [2, 0, 1],
            "env_seed": [60002, 60000, 60001],
            "success": [True, False, True],
        }
    ).to_csv(path, index=False)

    assert select_video_scenario(path, "first") == 60000
    assert select_video_scenario(path, "first_success") == 60001
    assert select_video_scenario(path, "first_failure") == 60000


def test_chase_config_uses_native_offscreen_main_camera_and_vector_policy_input():
    config = load_yaml("configs/sac_traffic_curriculum.yaml")
    args = SimpleNamespace(view="chase", width=320, height=192, density=0.05)

    camera_config = _camera_config(config, args, video_seed=50000)

    assert camera_config["use_render"] is False
    assert camera_config["image_observation"] is True
    assert camera_config["image_on_cuda"] is False
    assert camera_config["sensors"]["main_camera"] == ()
    assert camera_config["vehicle_config"]["image_source"] == "main_camera"
    assert camera_config["agent_observation"].__name__ == "LidarStateObservation"
    assert camera_config["show_mouse"] is True
    assert camera_config["window_size"] == (320, 192)
    assert camera_config["num_agents"] == 1
    assert camera_config["is_multi_agent"] is False
    assert camera_config["traffic_density"] == 0.05


def test_main_camera_helper_does_not_mutate_training_config():
    training_config = {"image_observation": False, "vehicle_config": {}}

    camera_config = enable_main_camera_capture(training_config)

    assert training_config == {"image_observation": False, "vehicle_config": {}}
    assert camera_config["sensors"]["main_camera"] == ()
    assert camera_config["image_observation"] is True
    assert camera_config["image_on_cuda"] is False
    assert camera_config["vehicle_config"]["image_source"] == "main_camera"
    assert camera_config["agent_observation"].__name__ == "LidarStateObservation"


def test_topdown_diagnostic_does_not_enable_the_main_camera_image_service():
    config = load_yaml("configs/sac_traffic_curriculum.yaml")
    args = SimpleNamespace(view="topdown", width=320, height=192, density=0.05)

    diagnostic_config = _camera_config(config, args, video_seed=50000)

    assert diagnostic_config["image_observation"] is False
    assert "main_camera" not in diagnostic_config.get("sensors", {})


def test_frame_validator_accepts_uint8_color_and_rejects_blank_or_malformed_frames():
    valid = np.zeros((4, 6, 3), dtype=np.uint8)
    valid[1, 2] = [10, 20, 30]

    accepted = _validate_frame(valid, (4, 6, 3))

    assert accepted.dtype == np.uint8
    assert accepted.shape == (4, 6, 3)
    invalid_frames = [
        np.zeros((4, 6, 3), dtype=np.uint8),
        np.zeros((4, 6), dtype=np.uint8),
        np.zeros((4, 6, 2), dtype=np.uint8),
        np.zeros((4, 6, 4), dtype=np.uint8),
        np.full((4, 6, 3), np.nan),
    ]
    for invalid in invalid_frames:
        try:
            _validate_frame(invalid, (4, 6, 3))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid renderer frame was accepted: {invalid.shape}")


def test_video_sidecar_contains_renderer_and_policy_interface_metadata():
    environment_config = {
        "window_size": (320, 192),
        "camera_dist": 8.5,
        "camera_height": 2.8,
        "camera_smooth": True,
        "use_chase_camera_follow_lane": True,
        "camera_fov": 65,
    }
    renderer = {
        "metadrive_version": "0.4.3",
        "metadrive_pinned_commit": "commit",
        "panda3d_version": "1.10.16",
        "render_mode": "offscreen",
        "graphics_pipe": "Pipe",
        "graphics_output": "Buffer",
        "image_source": "main_camera",
        "sensor_name": "main_camera",
        "frame": {"shape": [192, 320, 3], "dtype": "uint8"},
    }
    frame = np.zeros((192, 320, 3), dtype=np.uint8)

    fields = _rendering_sidecar_fields(environment_config, renderer, (259,), frame)

    assert fields["renderer"] == renderer
    assert fields["camera"]["image_source"] == "main_camera"
    assert fields["camera"]["sensor_name"] == "main_camera"
    assert fields["camera"]["policy_observation"] == "LidarStateObservation"
    assert fields["camera"]["offscreen"] is True
    assert fields["vector_observation_shape"] == [259]
    assert fields["frame_shape"] == [192, 320, 3]
    assert fields["frame_dtype"] == "uint8"


def test_rendering_failure_keeps_training_ready_and_names_the_failed_boundary():
    status = make_rendering_status(
        "failed",
        "official_metadrive_verifier",
        exit_code=139,
        diagnostics={"log": "runs/metadrive_headless_verifier.log"},
    )

    assert status["training_status"] == "ready"
    assert status["rendering_status"] == "failed"
    assert status["first_failing_boundary"] == "official_metadrive_verifier"
    assert status["completed_boundary"] is None
    assert status["exit_code"] == 139


def test_compiled_dependency_pins_agree_for_clean_colab_installation():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    project = Path("pyproject.toml").read_text(encoding="utf-8")

    for requirement in [
        "numpy==2.0.2",
        "opencv-python==4.11.0.86",
        "panda3d==1.10.16",
    ]:
        assert requirement in requirements
        assert f'"{requirement}"' in project


def test_checkpoint_numpy_deserialization_error_is_not_mislabeled_as_space_mismatch():
    class BrokenAlgorithm:
        @classmethod
        def load(cls, model_path, env, device):
            raise ModuleNotFoundError(
                "No module named 'numpy._core.numeric'",
                name="numpy._core.numeric",
            )

    try:
        _load_model(BrokenAlgorithm, "model.zip", object(), "cpu")
    except RuntimeError as error:
        message = str(error)
        assert "deserialization failed before observation/action-space" in message
        assert "numpy._core.numeric" in message
    else:
        raise AssertionError("A NumPy checkpoint deserialization error should fail clearly.")


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
                    assert all(
                        argument.annotation is None for argument in arguments
                    ), f"Parameter type hint found in {path}"
                assert not isinstance(node, ast.AnnAssign), f"Variable type hint found in {path}"
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    imported = {name.name for name in node.names}
                    assert (
                        "annotations" not in imported
                    ), f"Future annotations import found in {path}"
