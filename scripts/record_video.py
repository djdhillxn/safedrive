"""Record a chase-camera or diagnostic top-down MetaDrive rollout."""

import argparse
import importlib.metadata
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import VecNormalize

from saferl_drive.algorithms import get_algorithm_class
from saferl_drive.config import (
    apply_dotlist_overrides,
    deep_update,
    get_evaluation_config,
    load_yaml,
    make_experiment_fingerprint,
    make_eval_metadrive_config,
)
from saferl_drive.envs import find_vecnormalize_path, make_vec_env
from saferl_drive.utils import log_system_info, set_global_seeds, setup_logging, write_json


PINNED_METADRIVE_COMMIT = "85e5dadc6c7436d324348f6e3d8f8e680c06b4db"


class _DummyActionPolicy:
    def predict(self, observation, deterministic=True):
        return [[0.0, 0.0]], None


def parse_args():
    parser = argparse.ArgumentParser(description="Record a MetaDrive rollout video.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir")
    source.add_argument("--config")
    parser.add_argument("--policy", choices=["idm", "expert"], default=None)
    parser.add_argument("--model", default="best", choices=["final", "best"])
    parser.add_argument("--view", default="chase", choices=["chase", "topdown"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes-csv", default=None)
    parser.add_argument(
        "--scenario-rule",
        default="first",
        choices=["first", "first_success", "first_failure"],
    )
    parser.add_argument("--density", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _to_uint8_frame(frame):
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if array.size and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected a three-dimensional video frame, received {array.shape}.")
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array


def _raw_render_environment(vector_environment):
    current = vector_environment
    visited = set()
    while hasattr(current, "venv") and id(current) not in visited:
        visited.add(id(current))
        current = current.venv
    environments = getattr(current, "envs", None)
    if not environments or len(environments) != 1:
        raise RuntimeError("Video capture requires one in-process MetaDrive environment.")
    return environments[0].unwrapped


def select_video_scenario(episodes_csv, rule):
    """Select the lowest scenario seed satisfying a declared outcome rule."""
    frame = pd.read_csv(episodes_csv)
    if "env_seed" not in frame:
        raise ValueError(f"Evaluation CSV has no env_seed column: {episodes_csv}")
    if rule == "first_success":
        selected = frame[frame["success"].astype(bool)]
    elif rule == "first_failure":
        selected = frame[~frame["success"].astype(bool)]
    else:
        selected = frame
    if selected.empty:
        raise ValueError(f"No scenario satisfies video selection rule {rule!r}.")
    return int(selected.sort_values(["env_seed", "episode"]).iloc[0]["env_seed"])


def _camera_config(config, args, video_seed):
    environment_config = make_eval_metadrive_config(config, "test")
    environment_config.update(
        {
            "start_seed": video_seed,
            "num_scenarios": 1,
            "random_traffic": False,
            "random_spawn_lane_index": False,
            "use_render": False,
            "image_observation": False,
            "window_size": (args.width, args.height),
            "camera_dist": 8.5,
            "camera_height": 2.8,
            "camera_smooth": True,
            "use_chase_camera_follow_lane": True,
            "camera_fov": 65,
            "show_logo": False,
            "show_fps": False,
            "show_interface": False,
            "show_interface_navi_mark": False,
            "show_policy_mark": False,
            "show_coordinates": False,
            # MetaDrive 0.4.3 calls a window-only requestProperties() method
            # when show_mouse is false. An offscreen GraphicsBuffer has no such
            # method, so leave the unused cursor path enabled for chase capture.
            "show_mouse": True,
            "interface_panel": [],
            "_safedrive_main_camera": args.view == "chase",
            "num_agents": 1,
            "is_multi_agent": False,
        }
    )
    if args.density is not None:
        environment_config["traffic_density"] = float(args.density)
        environment_config["traffic_mode"] = "respawn"
    return environment_config


def _chase_frame(render_environment, expected_shape):
    camera = getattr(render_environment, "main_camera", None)
    engine = getattr(render_environment, "engine", None)
    sensor_names = sorted(getattr(engine, "sensors", {}).keys()) if engine else []
    render_mode = render_environment.config.get("_render_mode")
    try:
        version = importlib.metadata.version("metadrive-simulator")
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    if camera is None:
        raise RuntimeError(
            "MetaDrive main camera was not created. "
            f"offscreen_mode={render_mode!r}, sensors={sensor_names}, version={version}, "
            f"pinned_commit={PINNED_METADRIVE_COMMIT}."
        )
    try:
        frame = _to_uint8_frame(camera.perceive(to_float=False))
    except Exception as error:
        raise RuntimeError(
            "MetaDrive main-camera frame capture failed. "
            f"offscreen_mode={render_mode!r}, sensors={sensor_names}, version={version}, "
            f"pinned_commit={PINNED_METADRIVE_COMMIT}."
        ) from error
    if tuple(frame.shape) != tuple(expected_shape):
        raise RuntimeError(
            "MetaDrive main camera returned an unexpected frame shape: "
            f"expected={expected_shape}, received={frame.shape}, offscreen_mode={render_mode!r}, "
            f"sensors={sensor_names}, version={version}, pinned_commit={PINNED_METADRIVE_COMMIT}."
        )
    return frame


def _terminal_outcome(info):
    if bool(info.get("arrive_dest", False)):
        return "success"
    if any(
        bool(info.get(key, False))
        for key in ["crash", "crash_vehicle", "crash_object", "crash_sidewalk"]
    ):
        return "collision"
    if bool(info.get("out_of_road", False)):
        return "out_of_road"
    if bool(info.get("max_step", False)):
        return "timeout"
    return "other"


def main():
    args = parse_args()
    if args.config and args.policy is None:
        raise ValueError("--config recording requires --policy idm or --policy expert.")
    run_dir = Path(args.run_dir) if args.run_dir else None
    config_path = run_dir / "resolved_config.yaml" if run_dir else Path(args.config)
    config = apply_dotlist_overrides(load_yaml(config_path), args.overrides)
    video_root = run_dir if run_dir else Path(config.get("experiment", {}).get("output_dir", "runs"))
    log_dir = run_dir / "logs" if run_dir else video_root
    logger = setup_logging(log_dir / f"video_{args.view}.log")
    environment = None

    try:
        log_system_info(logger, run_dir=run_dir)
        testing = get_evaluation_config(config, "test")
        if args.episodes_csv:
            video_seed = select_video_scenario(args.episodes_csv, args.scenario_rule)
        else:
            video_seed = int(args.seed if args.seed is not None else testing.get("start_seed", 4000))
        set_global_seeds(video_seed)
        environment_config = _camera_config(config, args, video_seed)
        policy_class_name = None
        if args.policy:
            policy_class_name = "IDMPolicy" if args.policy == "idm" else "ExpertPolicy"
            environment_config["_safedrive_agent_policy"] = policy_class_name
            environment_config["manual_control"] = False

        base_env = make_vec_env(
            env_config=environment_config,
            n_envs=1,
            seed=video_seed,
            monitor_dir=log_dir / f"video_{args.view}_monitor",
            vec_env_type="dummy",
            normalize_obs=False,
            normalize_reward=False,
            training=False,
        )
        environment = base_env
        selected_model = args.model
        model_path = None
        if run_dir:
            model_path = run_dir / "models" / f"{selected_model}_model.zip"
            if selected_model == "best" and not model_path.exists():
                selected_model = "final"
                model_path = run_dir / "models" / "final_model.zip"
                logger.warning("Best model is missing; recording the final model.")
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            vecnormalize_path = find_vecnormalize_path(run_dir, selected_model)
            if vecnormalize_path is not None:
                environment = VecNormalize.load(vecnormalize_path, base_env)
                environment.training = False
                environment.norm_reward = False
            algorithm_name = config.get("algorithm", {}).get("name", "sac").lower()
            load_device = (
                args.device
                or config.get("algorithm", {}).get("kwargs", {}).get("device")
                or ("cpu" if algorithm_name == "ppo" else "auto")
            )
            model = get_algorithm_class(algorithm_name).load(
                model_path,
                env=environment,
                device=load_device,
            )
            if tuple(model.observation_space.shape) != tuple(environment.observation_space.shape):
                raise ValueError(
                    "Recording environment changed the trained vector observation: "
                    f"model={model.observation_space.shape}, "
                    f"recording={environment.observation_space.shape}."
                )
            model_label = str(model_path)
        else:
            model = _DummyActionPolicy()
            algorithm_name = args.policy
            model_label = policy_class_name

        environment.seed(video_seed)
        observation = environment.reset()
        render_environment = _raw_render_environment(environment)
        if np.asarray(observation).ndim != 2:
            raise RuntimeError(
                "Recording did not preserve the vector LidarState observation: "
                f"received observation shape {np.asarray(observation).shape}."
            )
        frames = []
        final_info = {}
        return_sum = 0.0
        expected_shape = (args.height, args.width, 3)
        for _ in range(args.steps):
            if args.view == "chase":
                frame = _chase_frame(render_environment, expected_shape)
            else:
                frame = render_environment.render(
                    mode="topdown",
                    window=False,
                    screen_size=(args.width, args.height),
                )
                frame = _to_uint8_frame(frame)
            frames.append(frame)
            action, _ = model.predict(observation, deterministic=True)
            observation, rewards, dones, infos = environment.step(action)
            return_sum += float(rewards[0])
            final_info = dict(infos[0])
            if bool(dones[0]):
                break

        if not frames:
            raise RuntimeError("No video frames were produced.")
        density = float(environment_config.get("traffic_density", 0.0))
        output = (
            Path(args.output)
            if args.output
            else video_root
            / "videos"
            / f"{algorithm_name}_{selected_model}_{args.view}_d{int(round(density * 100)):03d}"
            f"_seed{video_seed}.mp4"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output, frames, fps=args.fps)
        controller = policy_class_name if policy_class_name else None
        fingerprint_config = deep_update(
            config,
            {
                "test": {
                    "start_seed": video_seed,
                    "num_scenarios": 1,
                    "episodes": 1,
                    "traffic_density": density,
                    "traffic_mode": environment_config.get("traffic_mode"),
                    "random_traffic": False,
                }
            },
        )
        fingerprint = make_experiment_fingerprint(
            fingerprint_config,
            split="test",
            episodes=1,
            controller=controller,
        )
        sidecar = {
            "model_run": str(run_dir) if run_dir else model_label,
            "model": model_label,
            "scenario_seed": video_seed,
            "scenario_selection_rule": args.scenario_rule if args.episodes_csv else "explicit_or_split_start",
            "scenario_selection_csv": args.episodes_csv,
            "traffic_density": density,
            "traffic_mode": environment_config.get("traffic_mode"),
            "outcome": _terminal_outcome(final_info),
            "return": return_sum,
            "route_completion": float(final_info.get("route_completion", 0.0)),
            "collision": bool(final_info.get("crash", False)),
            "crash_vehicle": bool(final_info.get("crash_vehicle", False)),
            "frames": len(frames),
            "fps": args.fps,
            "view": args.view,
            "camera": {
                "resolution": [args.width, args.height],
                "camera_dist": environment_config["camera_dist"],
                "camera_height": environment_config["camera_height"],
                "camera_smooth": environment_config["camera_smooth"],
                "use_chase_camera_follow_lane": environment_config[
                    "use_chase_camera_follow_lane"
                ],
                "camera_fov": environment_config["camera_fov"],
                "offscreen": True,
                "metadrive_image_service_switch": args.view == "chase",
                "policy_observation": "LidarStateObservation",
            },
            "vector_observation_shape": list(environment.observation_space.shape),
            "experiment_fingerprint": fingerprint,
            "metadrive_pinned_commit": PINNED_METADRIVE_COMMIT,
            "video": str(output),
        }
        write_json(sidecar, output.with_suffix(".json"))
        logger.info(
            "Saved %s video with %s frames for scenario %s: %s",
            args.view,
            len(frames),
            video_seed,
            output,
        )
    except Exception:
        logger.exception("Video recording failed.")
        raise
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    main()
