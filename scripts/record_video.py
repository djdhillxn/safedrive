"""Record a chase-camera or diagnostic top-down MetaDrive rollout."""

import argparse
import importlib.metadata
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from saferl_drive.config import (
    apply_dotlist_overrides,
    deep_update,
    get_evaluation_config,
    load_yaml,
    make_experiment_fingerprint,
    make_eval_metadrive_config,
)
from saferl_drive.envs import enable_main_camera_capture, find_vecnormalize_path, make_vec_env
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
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--smoke-renderer",
        action="store_true",
        help="Capture 3-5 direct MetaDrive main-camera PNGs without SB3 or VecEnv.",
    )
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
    if not array.size:
        raise ValueError("The renderer returned an empty frame.")
    try:
        finite = bool(np.isfinite(array).all())
    except TypeError as error:
        raise ValueError(
            f"The renderer returned a non-numeric frame with dtype {array.dtype}."
        ) from error
    if not finite:
        raise ValueError("The renderer returned a frame containing non-finite values.")
    if array.ndim != 3:
        raise ValueError(f"Expected an HxWx3 video frame, received {array.shape}.")
    if array.shape[-1] != 3:
        raise ValueError(f"Expected three color channels, received {array.shape}.")
    if array.dtype != np.uint8:
        if array.size and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _validate_frame(frame, expected_shape):
    array = _to_uint8_frame(frame)
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(
            "The main camera returned an unexpected frame shape: "
            f"expected={tuple(expected_shape)}, received={array.shape}."
        )
    minimum = int(array.min())
    maximum = int(array.max())
    if maximum == 0 or maximum == minimum:
        raise ValueError(
            "The main camera returned a blank or constant frame: "
            f"min={minimum}, max={maximum}, shape={array.shape}."
        )
    return array


def _frame_statistics(frame):
    return {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "minimum": int(frame.min()),
        "maximum": int(frame.max()),
        "range": int(frame.max()) - int(frame.min()),
        "variance": float(np.var(frame)),
    }


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
    import pandas as pd

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
            "image_on_cuda": False,
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
            "num_agents": 1,
            "is_multi_agent": False,
        }
    )
    if args.view == "chase":
        environment_config = enable_main_camera_capture(environment_config)
    if args.density is not None:
        environment_config["traffic_density"] = float(args.density)
        environment_config["traffic_mode"] = "respawn"
    return environment_config


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _renderer_metadata(render_environment, frame=None):
    engine = getattr(render_environment, "engine", None)
    sensor_names = sorted(getattr(engine, "sensors", {}).keys()) if engine else []
    pipe = getattr(engine, "pipe", None) if engine else None
    window = getattr(engine, "win", None) if engine else None
    config = getattr(render_environment, "config", {})
    metadata = {
        "metadrive_version": _package_version("metadrive-simulator"),
        "metadrive_pinned_commit": PINNED_METADRIVE_COMMIT,
        "panda3d_version": _package_version("panda3d"),
        "render_mode": config.get("_render_mode"),
        "graphics_pipe": pipe.__class__.__name__ if pipe is not None else None,
        "graphics_output": window.__class__.__name__ if window is not None else None,
        "sensor_names": sensor_names,
        "sensor_name": "main_camera",
        "image_source": config.get("vehicle_config", {}).get("image_source"),
    }
    if frame is not None:
        metadata["frame"] = _frame_statistics(frame)
    return metadata


def _capture_main_camera_frame(render_environment, expected_shape, logger=None):
    engine = getattr(render_environment, "engine", None)
    camera = None
    if engine is not None:
        try:
            camera = engine.get_sensor("main_camera")
        except (KeyError, RuntimeError, ValueError):
            camera = None
    if camera is None:
        camera = getattr(render_environment, "main_camera", None)
    metadata = _renderer_metadata(render_environment)
    if camera is None:
        raise RuntimeError(
            "MetaDrive main camera was not created. "
            f"renderer={metadata}."
        )
    try:
        frame = _validate_frame(camera.perceive(to_float=False), expected_shape)
    except Exception as error:
        raise RuntimeError(
            "MetaDrive main-camera frame capture failed. "
            f"renderer={metadata}."
        ) from error
    metadata = _renderer_metadata(render_environment, frame)
    if logger is not None:
        logger.info(
            "Main-camera frame: mode=%s sensor=%s pipe=%s output=%s shape=%s "
            "dtype=%s min=%s max=%s variance=%.3f",
            metadata["render_mode"],
            metadata["sensor_name"],
            metadata["graphics_pipe"],
            metadata["graphics_output"],
            metadata["frame"]["shape"],
            metadata["frame"]["dtype"],
            metadata["frame"]["minimum"],
            metadata["frame"]["maximum"],
            metadata["frame"]["variance"],
        )
    return frame, metadata


def _rendering_sidecar_fields(environment_config, renderer_metadata, observation_shape, frame):
    image_source = renderer_metadata.get("image_source")
    sensor_name = renderer_metadata.get("sensor_name")
    return {
        "camera": {
            "resolution": [
                int(environment_config["window_size"][0]),
                int(environment_config["window_size"][1]),
            ],
            "camera_dist": environment_config["camera_dist"],
            "camera_height": environment_config["camera_height"],
            "camera_smooth": environment_config["camera_smooth"],
            "use_chase_camera_follow_lane": environment_config[
                "use_chase_camera_follow_lane"
            ],
            "camera_fov": environment_config["camera_fov"],
            "offscreen": renderer_metadata.get("render_mode") == "offscreen",
            "image_source": image_source,
            "sensor_name": sensor_name,
            "policy_observation": "LidarStateObservation",
        },
        "renderer": renderer_metadata,
        "vector_observation_shape": list(observation_shape),
        "frame_shape": list(frame.shape),
        "frame_dtype": str(frame.dtype),
    }


def make_rendering_status(rendering_status, boundary, exit_code=None, diagnostics=None):
    if rendering_status not in {"passed", "failed"}:
        raise ValueError("rendering_status must be 'passed' or 'failed'.")
    return {
        "training_status": "ready",
        "rendering_status": rendering_status,
        "first_failing_boundary": boundary if rendering_status == "failed" else None,
        "completed_boundary": boundary if rendering_status == "passed" else None,
        "exit_code": exit_code,
        "diagnostics": diagnostics or {},
    }


def _run_bare_renderer_smoke(config, args, video_seed, logger):
    from metadrive.envs import MetaDriveEnv
    from metadrive.policy.expert_policy import ExpertPolicy

    if args.view != "chase":
        raise ValueError("--smoke-renderer validates the chase main camera only.")
    smoke_steps = int(args.steps or 5)
    if not 3 <= smoke_steps <= 5:
        raise ValueError("--smoke-renderer requires --steps between 3 and 5.")
    output_directory = Path(args.output or "/tmp/safedrive_renderer_smoke")
    output_directory.mkdir(parents=True, exist_ok=True)
    environment_config = _camera_config(config, args, video_seed)
    # These keys are implemented by SafeDrive wrappers in make_metadrive_env().
    # A direct MetaDriveEnv smoke must not pass project-only settings into
    # MetaDrive's strict configuration validator.
    for project_key in ["reward_shaping", "steering_limit", "sequential_seed"]:
        environment_config.pop(project_key, None)
    environment_config["agent_policy"] = ExpertPolicy
    environment_config["manual_control"] = False
    environment = None
    frames = []
    renderer_metadata = None
    return_sum = 0.0
    final_info = {}
    try:
        environment = MetaDriveEnv(environment_config)
        observation, final_info = environment.reset(seed=video_seed)
        observation_shape = tuple(np.asarray(observation).shape)
        vector_space_shape = tuple(environment.observation_space.shape)
        if len(observation_shape) != 1 or observation_shape != vector_space_shape:
            raise RuntimeError(
                "Bare renderer smoke changed the policy-facing vector observation: "
                f"space={vector_space_shape}, received={observation_shape}."
            )
        if tuple(environment.action_space.shape) != (2,):
            raise RuntimeError(
                "Bare renderer smoke changed the continuous action interface: "
                f"received={environment.action_space.shape}."
            )
        traffic_count = len(environment.engine.traffic_manager.traffic_vehicles)
        if float(environment_config.get("traffic_density", 0.0)) > 0.0 and traffic_count < 1:
            raise RuntimeError("Bare renderer smoke requested traffic but spawned none.")
        expected_shape = (args.height, args.width, 3)
        for frame_index in range(smoke_steps):
            observation, reward, terminated, truncated, final_info = environment.step(
                [0.0, 0.0]
            )
            return_sum += float(reward)
            frame, renderer_metadata = _capture_main_camera_frame(
                environment,
                expected_shape,
                logger=logger,
            )
            frame_path = output_directory / f"main_camera_{frame_index:02d}.png"
            imageio.imwrite(frame_path, frame)
            frames.append(frame_path)
            if terminated or truncated:
                break
        if len(frames) < 3:
            raise RuntimeError(f"Bare renderer smoke produced only {len(frames)} valid frames.")
        summary = {
            "status": "passed",
            "scenario_seed": video_seed,
            "traffic_density": float(environment_config.get("traffic_density", 0.0)),
            "traffic_vehicle_count": traffic_count,
            "frames": [str(path) for path in frames],
            "frame_count": len(frames),
            "vector_observation_shape": list(observation_shape),
            "policy_observation": environment_config["agent_observation"].__name__,
            "action_shape": list(environment.action_space.shape),
            "return": return_sum,
            "outcome": _terminal_outcome(final_info),
            "renderer": renderer_metadata,
        }
        write_json(summary, output_directory / "renderer_smoke.json")
        logger.info("Bare native-offscreen renderer smoke passed: %s", output_directory)
        return summary
    finally:
        if environment is not None:
            environment.close()


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
    if args.smoke_renderer and not args.config:
        raise ValueError("--smoke-renderer requires --config.")
    if args.smoke_renderer and args.policy not in {None, "expert"}:
        raise ValueError("--smoke-renderer uses MetaDrive ExpertPolicy.")
    if args.config and args.policy is None and not args.smoke_renderer:
        raise ValueError("--config recording requires --policy idm or --policy expert.")
    run_dir = Path(args.run_dir) if args.run_dir else None
    config_path = run_dir / "resolved_config.yaml" if run_dir else Path(args.config)
    config = apply_dotlist_overrides(load_yaml(config_path), args.overrides)
    video_root = run_dir if run_dir else Path(config.get("experiment", {}).get("output_dir", "runs"))
    log_dir = run_dir / "logs" if run_dir else video_root
    log_name = "renderer_smoke.log" if args.smoke_renderer else f"video_{args.view}.log"
    logger = setup_logging(log_dir / log_name)
    environment = None

    try:
        log_system_info(logger, run_dir=run_dir)
        testing = get_evaluation_config(config, "test")
        if args.episodes_csv:
            video_seed = select_video_scenario(args.episodes_csv, args.scenario_rule)
        else:
            video_seed = int(args.seed if args.seed is not None else testing.get("start_seed", 4000))
        set_global_seeds(video_seed)
        if args.smoke_renderer:
            _run_bare_renderer_smoke(config, args, video_seed, logger)
            return
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
            from stable_baselines3.common.vec_env import VecNormalize

            from saferl_drive.algorithms import get_algorithm_class

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
        observation_array = np.asarray(observation)
        expected_observation_shape = tuple(environment.observation_space.shape)
        if (
            observation_array.ndim != 2
            or tuple(observation_array.shape[1:]) != expected_observation_shape
        ):
            raise RuntimeError(
                "Recording did not preserve the vector LidarState observation: "
                f"space={expected_observation_shape}, received={observation_array.shape}."
            )
        frames = []
        renderer_metadata = None
        final_info = {}
        return_sum = 0.0
        expected_shape = (args.height, args.width, 3)
        recording_steps = int(args.steps or 1000)
        for _ in range(recording_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, rewards, dones, infos = environment.step(action)
            return_sum += float(rewards[0])
            final_info = dict(infos[0])
            if args.view == "chase":
                frame, renderer_metadata = _capture_main_camera_frame(
                    render_environment,
                    expected_shape,
                    logger=logger if not frames else None,
                )
            else:
                frame = render_environment.render(
                    mode="topdown",
                    window=False,
                    screen_size=(args.width, args.height),
                )
                frame = _validate_frame(frame, expected_shape)
                renderer_metadata = _renderer_metadata(render_environment, frame)
                renderer_metadata["sensor_name"] = "topdown"
                renderer_metadata["image_source"] = "topdown"
            frames.append(frame)
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
        logger.info(
            "SafeDrive frame-collection boundary passed with %s valid frames; "
            "starting imageio MP4 encoding.",
            len(frames),
        )
        try:
            imageio.mimsave(output, frames, fps=args.fps)
        except Exception as error:
            raise RuntimeError(
                f"Imageio MP4 encoding failed after collecting {len(frames)} valid frames."
            ) from error
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("Imageio returned without producing a nonempty MP4.")
        logger.info("Imageio MP4 encoding boundary passed: %s bytes.", output.stat().st_size)
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
            "scenario_selection_rule": (
                args.scenario_rule if args.episodes_csv else "explicit_or_split_start"
            ),
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
            "experiment_fingerprint": fingerprint,
            "metadrive_pinned_commit": PINNED_METADRIVE_COMMIT,
            "video": str(output),
        }
        sidecar.update(
            _rendering_sidecar_fields(
                environment_config,
                renderer_metadata,
                environment.observation_space.shape,
                frames[-1],
            )
        )
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
