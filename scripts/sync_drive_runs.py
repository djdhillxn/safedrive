"""Copy SafeDrive run artifacts from Google Drive into this repository.

The default Mac workflow is an analysis-only, one-way merge. It keeps logs,
metrics, plots, and videos while leaving model and replay artifacts in Drive.
"""

import argparse
import filecmp
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from saferl_drive.config import load_yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IGNORED_NAMES = {".DS_Store"}
TRAINING_ARTIFACT_DIRECTORIES = {"checkpoints", "models"}
TRAINING_ARTIFACT_SUFFIXES = {".ckpt", ".pickle", ".pkl", ".pt", ".pth", ".zip"}
SYNC_STATE_NAME = ".safedrive_runs_sync_state.json"
SYNC_WORKERS = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge Google Drive SafeDrive runs into the repository runs folder."
    )
    parser.add_argument(
        "--drive-project",
        help="SafeDrive folder on Google Drive. Auto-detected on macOS and Colab by default.",
    )
    parser.add_argument(
        "--local-runs",
        default=str(REPOSITORY_ROOT / "runs"),
        help="Destination runs folder. Defaults to this repository's runs folder.",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Also copy runs whose metadata status is 'running'.",
    )
    parser.add_argument(
        "--include-training-artifacts",
        action="store_true",
        help="Copy model, checkpoint, replay-buffer, and archive files. Used by Colab.",
    )
    parser.add_argument(
        "--prune-local-training-artifacts",
        action="store_true",
        help="Remove already-downloaded training artifacts locally; never changes Drive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without changing files.",
    )
    parser.add_argument(
        "--to-drive",
        action="store_true",
        help="Persist one local run to Drive instead of restoring Drive runs locally.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Local run directory required with --to-drive.",
    )
    parser.add_argument(
        "--project-artifacts-to-drive",
        action="store_true",
        help="Persist comparison tables, plots, videos, report sources, and PDFs.",
    )
    return parser.parse_args()


def discover_drive_project(explicit_path=None):
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    environment_path = os.environ.get("SAFEDRIVE_DRIVE_PROJECT")
    if environment_path:
        return Path(environment_path).expanduser().resolve()

    colab_path = Path("/content/drive/MyDrive/SafeDrive")
    if colab_path.exists():
        return colab_path

    cloud_storage = Path.home() / "Library" / "CloudStorage"
    matches = sorted(cloud_storage.glob("GoogleDrive-*/My Drive/SafeDrive"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        choices = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"Multiple Google Drive SafeDrive folders were found. Pass --drive-project:\n{choices}"
        )
    raise FileNotFoundError(
        "Could not find Google Drive's SafeDrive folder. Pass its location with "
        "--drive-project or set SAFEDRIVE_DRIVE_PROJECT."
    )


def read_metadata(run_dir):
    metadata_path = run_dir / "run_metadata.json"
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def run_signature(run_dir):
    metadata_path = run_dir / "run_metadata.json"
    try:
        metadata_stat = metadata_path.stat()
    except OSError:
        return None
    videos = []
    videos_path = run_dir / "videos"
    if videos_path.is_dir():
        for path in sorted(videos_path.rglob("*")):
            if not path.is_file():
                continue
            try:
                file_stat = path.stat()
            except OSError:
                continue
            videos.append(
                {
                    "path": str(path.relative_to(videos_path)),
                    "size": file_stat.st_size,
                    "mtime_ns": file_stat.st_mtime_ns,
                }
            )
    return {
        "metadata_size": metadata_stat.st_size,
        "metadata_mtime_ns": metadata_stat.st_mtime_ns,
        "videos": videos,
    }


def load_sync_state(local_runs):
    state_path = local_runs.parent / SYNC_STATE_NAME
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    state.setdefault("analysis", {})
    state.setdefault("full", {})
    return state_path, state


def save_sync_state(state_path, state):
    temporary = state_path.with_name(f".{state_path.name}.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(state_path)


def file_is_current(source, destination, verify_contents=False):
    if not destination.is_file():
        return False
    source_stat = source.stat()
    destination_stat = destination.stat()
    # Completed run artifacts are immutable. A size-only check avoids forcing
    # Google Drive for desktop to download identical multi-gigabyte buffers.
    if source_stat.st_size != destination_stat.st_size:
        return False
    if verify_contents:
        return filecmp.cmp(source, destination, shallow=False)
    return True


def copy_file(source, destination, dry_run=False, verify_contents=False):
    if file_is_current(source, destination, verify_contents=verify_contents):
        return False
    if dry_run:
        print(f"Would copy: {source} -> {destination}")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.safedrive-sync.tmp")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return True


def iter_run_files(source, include_training_artifacts=False, include_checkpoints=True):
    for directory, directory_names, file_names in os.walk(source):
        directory_names[:] = [name for name in directory_names if name != "tensorboard"]
        if not include_checkpoints:
            directory_names[:] = [name for name in directory_names if name != "checkpoints"]
        if not include_training_artifacts:
            directory_names[:] = [
                name for name in directory_names if name not in TRAINING_ARTIFACT_DIRECTORIES
            ]
        directory_path = Path(directory)
        for file_name in file_names:
            path = directory_path / file_name
            if path.name in IGNORED_NAMES:
                continue
            if not include_training_artifacts and path.suffix.lower() in TRAINING_ARTIFACT_SUFFIXES:
                continue
            yield path


def merge_directory(
    source,
    destination,
    dry_run=False,
    include_training_artifacts=False,
    include_checkpoints=True,
):
    paths = list(
        iter_run_files(
            source,
            include_training_artifacts=include_training_artifacts,
            include_checkpoints=include_checkpoints,
        )
    )

    def copy_one(path):
        relative_path = path.relative_to(source)
        return copy_file(path, destination / relative_path, dry_run=dry_run)

    if dry_run or len(paths) < 2:
        return sum(bool(copy_one(path)) for path in paths)
    with ThreadPoolExecutor(max_workers=min(SYNC_WORKERS, len(paths))) as executor:
        return sum(bool(changed) for changed in executor.map(copy_one, paths))


def prune_local_training_artifacts(local_runs, dry_run=False):
    removed_directories = 0
    removed_files = 0
    reclaimed_bytes = 0
    if not local_runs.exists():
        return removed_directories, removed_files, reclaimed_bytes

    for run_dir in local_runs.iterdir():
        if not run_dir.is_dir():
            continue
        for directory_name in TRAINING_ARTIFACT_DIRECTORIES:
            artifact_dir = run_dir / directory_name
            if not artifact_dir.is_dir():
                continue
            file_sizes = [path.stat().st_size for path in artifact_dir.rglob("*") if path.is_file()]
            removed_files += len(file_sizes)
            reclaimed_bytes += sum(file_sizes)
            removed_directories += 1
            if dry_run:
                print(f"Would remove local training artifacts: {artifact_dir}")
            else:
                shutil.rmtree(artifact_dir)

        for path in run_dir.rglob("*"):
            relative_parts = path.relative_to(run_dir).parts
            if any(name in TRAINING_ARTIFACT_DIRECTORIES for name in relative_parts[:-1]):
                continue
            if not path.is_file() or path.suffix.lower() not in TRAINING_ARTIFACT_SUFFIXES:
                continue
            removed_files += 1
            reclaimed_bytes += path.stat().st_size
            if dry_run:
                print(f"Would remove local training artifact: {path}")
            else:
                path.unlink()

    return removed_directories, removed_files, reclaimed_bytes


def pointer_name(metadata):
    latest_name = metadata.get("latest_name")
    if latest_name:
        return str(latest_name)

    algorithm = str(metadata.get("algorithm", "")).lower()
    config_name = Path(str(metadata.get("config_path", ""))).name
    split = metadata.get("evaluation", {}).get("split")
    if algorithm == "idmpolicy" and split != "validation":
        return "idm"
    if algorithm == "expertpolicy" and split != "validation":
        return "expert"
    if config_name == "smoke_test.yaml":
        return "smoke"
    if algorithm in {"ppo", "sac"}:
        return algorithm
    return None


def refresh_latest_pointers(local_runs, dry_run=False):
    candidates = {}
    if not local_runs.exists():
        return {}
    for run_dir in local_runs.iterdir():
        if not run_dir.is_dir():
            continue
        metadata = read_metadata(run_dir)
        if not metadata or metadata.get("status") not in {"complete", "paused", "failed_gate"}:
            continue
        name = pointer_name(metadata)
        if name:
            candidates.setdefault(name, []).append(run_dir)

    pointers = {}
    for name, run_dirs in candidates.items():
        latest = max(run_dirs, key=lambda path: path.name)
        pointer = local_runs / f"latest_{name}.txt"
        contents = f"runs/{latest.name}\n"
        current_contents = None
        if pointer.exists():
            current_contents = pointer.read_text(encoding="utf-8")
        if dry_run and current_contents != contents:
            print(f"Would write pointer: {pointer} -> runs/{latest.name}")
        elif not dry_run and current_contents != contents:
            pointer.write_text(contents, encoding="utf-8")
        pointers[name] = latest
    return pointers


def sync_runs(
    drive_project,
    local_runs,
    include_running=False,
    include_training_artifacts=False,
    dry_run=False,
):
    drive_runs = drive_project / "runs"
    if not drive_runs.is_dir():
        raise FileNotFoundError(f"Google Drive runs folder was not found: {drive_runs}")
    if not dry_run:
        local_runs.mkdir(parents=True, exist_ok=True)
    state_path, state = load_sync_state(local_runs)
    state_section = "full" if include_training_artifacts else "analysis"
    run_states = state[state_section]

    copied_runs = []
    skipped_running = []
    skipped_invalid = []
    copied_files = 0

    for source in sorted(drive_runs.iterdir(), key=lambda path: path.name):
        if source.name in IGNORED_NAMES:
            continue
        if source.is_file():
            if source.name.startswith("latest_") and source.suffix == ".txt":
                # Rebuild pointers from local metadata below. Drive pointers can
                # lag when a run directory is copied before the final bulk sync.
                continue
            if copy_file(
                source,
                local_runs / source.name,
                dry_run=dry_run,
                verify_contents=True,
            ):
                copied_files += 1
            continue
        if not source.is_dir():
            continue

        signature = run_signature(source)
        if signature is None:
            skipped_invalid.append(source.name)
            continue
        destination = local_runs / source.name
        if destination.is_dir() and run_states.get(source.name) == signature:
            continue
        if (
            not include_training_artifacts
            and source.name not in run_states
            and (destination / "run_metadata.json").is_file()
            and (destination / "run_metadata.json").stat().st_size == signature["metadata_size"]
            and run_signature(destination).get("videos") == signature.get("videos")
        ):
            # Bootstrap the cache for artifacts brought over by the old full
            # sync or the one-time downloaded-folder migration.
            run_states[source.name] = signature
            continue

        metadata = read_metadata(source)
        if metadata is None:
            skipped_invalid.append(source.name)
            continue
        if metadata.get("status") == "running" and not include_running:
            skipped_running.append(source.name)
            continue

        changed = merge_directory(
            source,
            destination,
            dry_run=dry_run,
            include_training_artifacts=include_training_artifacts,
        )
        copied_files += changed
        if changed:
            copied_runs.append(source.name)
        run_states[source.name] = signature

    pointers = refresh_latest_pointers(local_runs, dry_run=dry_run)
    if not dry_run:
        save_sync_state(state_path, state)
    return {
        "copied_files": copied_files,
        "changed_runs": copied_runs,
        "skipped_running": skipped_running,
        "skipped_invalid": skipped_invalid,
        "pointers": pointers,
    }


def critical_run_files(run_dir):
    """Return the restart-critical files expected for one persisted run."""
    run_dir = Path(run_dir)
    required = [
        run_dir / "resolved_config.yaml",
        run_dir / "run_metadata.json",
    ]
    if (run_dir / "curriculum_state.json").exists():
        required.append(run_dir / "curriculum_state.json")
    config_path = run_dir / "resolved_config.yaml"
    try:
        config = load_yaml(config_path)
    except (FileNotFoundError, OSError, ValueError):
        config = {}
    is_traffic_adaptation = bool(config.get("source"))
    if is_traffic_adaptation or (run_dir / "source_lineage.json").exists():
        required.append(run_dir / "source_lineage.json")
    if is_traffic_adaptation:
        required.extend(
            [
                run_dir / "logs" / "curriculum_train.log",
                run_dir / "logs" / "resource_usage.csv",
            ]
        )
        evaluation_summaries = sorted((run_dir / "eval").glob("best_*_summary.json"))
        evaluation_episodes = sorted((run_dir / "eval").glob("best_*_episodes.csv"))
        required.append(
            evaluation_summaries[0]
            if evaluation_summaries
            else run_dir / "eval" / "best_validation_summary.json"
        )
        required.append(
            evaluation_episodes[0]
            if evaluation_episodes
            else run_dir / "eval" / "best_validation_episodes.csv"
        )

    metadata = read_metadata(run_dir) or {}
    status = metadata.get("status")
    if status in {"complete", "paused", "failed_gate"}:
        model_candidates = [
            run_dir / "models" / "best_model.zip",
            run_dir / "models" / "final_model.zip",
            run_dir / "models" / "curriculum_resume_model.zip",
        ]
        if not any(path.exists() for path in model_candidates):
            required.append(model_candidates[0])
        if (run_dir / "curriculum_state.json").exists():
            required.extend(
                [
                    run_dir / "models" / "curriculum_resume_model.zip",
                    run_dir / "models" / "curriculum_resume_replay_buffer.pkl",
                ]
            )
    return required


def verify_critical_files(run_dir):
    """Fail closed when a copied training run cannot be resumed or evaluated."""
    run_dir = Path(run_dir)
    missing = [path for path in critical_run_files(run_dir) if not path.exists()]
    if missing:
        relative = [
            str(path.relative_to(run_dir)) if path.is_relative_to(run_dir) else str(path)
            for path in missing
        ]
        raise FileNotFoundError(
            "Critical persisted artifacts are missing from "
            f"{run_dir}: {', '.join(relative)}"
        )
    return [str(path.relative_to(run_dir)) for path in critical_run_files(run_dir)]


def sync_run_to_drive(drive_project, run_dir, dry_run=False):
    """Atomically merge one local run and its latest pointer into Drive."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Local run directory was not found: {run_dir}")
    verify_critical_files(run_dir)
    drive_runs = Path(drive_project) / "runs"
    destination = drive_runs / run_dir.name
    if not dry_run:
        drive_runs.mkdir(parents=True, exist_ok=True)
    copied_files = merge_directory(
        run_dir,
        destination,
        dry_run=dry_run,
        include_training_artifacts=True,
        include_checkpoints=False,
    )
    metadata = read_metadata(run_dir) or {}
    latest_name = pointer_name(metadata)
    pointer = None
    if latest_name:
        pointer = drive_runs / f"latest_{latest_name}.txt"
        if dry_run:
            print(f"Would write pointer: {pointer} -> runs/{run_dir.name}")
        else:
            temporary = pointer.with_name(f".{pointer.name}.safedrive-sync.tmp")
            temporary.write_text(f"runs/{run_dir.name}\n", encoding="utf-8")
            temporary.replace(pointer)
    if not dry_run:
        verified = verify_critical_files(destination)
        if pointer is not None and not pointer.exists():
            raise FileNotFoundError(f"Drive latest pointer was not persisted: {pointer}")
    else:
        verified = []
    return {
        "copied_files": copied_files,
        "drive_run": destination,
        "pointer": pointer,
        "verified_critical_files": verified,
    }


def sync_project_artifacts_to_drive(drive_project, repository_root=None, dry_run=False):
    """Persist compact final project outputs without copying generated build clutter."""
    repository_root = Path(repository_root or REPOSITORY_ROOT)
    drive_project = Path(drive_project)
    copied = []
    allowed_run_suffixes = {".csv", ".json", ".png", ".log", ".txt", ".mp4"}
    local_runs = repository_root / "runs"
    if local_runs.exists():
        for path in local_runs.iterdir():
            if path.is_file() and path.suffix.lower() in allowed_run_suffixes:
                destination = drive_project / "runs" / path.name
                if copy_file(path, destination, dry_run=dry_run, verify_contents=True):
                    copied.append(str(destination))
        videos = local_runs / "videos"
        if videos.is_dir():
            for path in iter_run_files(videos, include_training_artifacts=False):
                destination = drive_project / "runs" / "videos" / path.relative_to(videos)
                if copy_file(path, destination, dry_run=dry_run):
                    copied.append(str(destination))

    allowed_report_suffixes = {".tex", ".bib", ".pdf"}
    reports = repository_root / "reports"
    if reports.exists():
        for path in reports.iterdir():
            if path.is_file() and path.suffix.lower() in allowed_report_suffixes:
                destination = drive_project / "reports" / path.name
                if copy_file(path, destination, dry_run=dry_run, verify_contents=True):
                    copied.append(str(destination))
    return copied


def main():
    args = parse_args()
    if args.include_training_artifacts and args.prune_local_training_artifacts:
        raise ValueError(
            "Choose either --include-training-artifacts or "
            "--prune-local-training-artifacts, not both."
        )
    drive_project = discover_drive_project(args.drive_project)
    if args.project_artifacts_to_drive:
        copied = sync_project_artifacts_to_drive(
            drive_project,
            repository_root=REPOSITORY_ROOT,
            dry_run=args.dry_run,
        )
        print(f"Project artifact persistence complete: {len(copied)} files updated.")
        return
    if args.to_drive:
        if not args.run_dir:
            raise ValueError("--to-drive requires --run-dir.")
        result = sync_run_to_drive(
            drive_project,
            args.run_dir,
            dry_run=args.dry_run,
        )
        print(
            f"Drive persistence complete: {result['copied_files']} files updated in "
            f"{result['drive_run']}."
        )
        if result["verified_critical_files"]:
            print(
                "Verified critical files: "
                + ", ".join(result["verified_critical_files"])
            )
        return
    local_runs = Path(args.local_runs).expanduser().resolve()
    print(f"Drive source: {drive_project / 'runs'}")
    print(f"Local destination: {local_runs}")
    result = sync_runs(
        drive_project,
        local_runs,
        include_running=args.include_running,
        include_training_artifacts=args.include_training_artifacts,
        dry_run=args.dry_run,
    )
    print(
        f"Sync complete: {result['copied_files']} files updated across "
        f"{len(result['changed_runs'])} run directories."
    )
    if args.include_training_artifacts:
        print("Training artifacts: included (full Colab restore).")
    else:
        print("Training artifacts: excluded (analysis-only local sync).")
    if args.prune_local_training_artifacts:
        directories, files, byte_count = prune_local_training_artifacts(
            local_runs,
            dry_run=args.dry_run,
        )
        action = "Would reclaim" if args.dry_run else "Reclaimed"
        print(
            f"{action} {byte_count / (1024**3):.2f} GiB from {files} files "
            f"in {directories} local training-artifact directories."
        )
    if result["skipped_running"]:
        print(f"Skipped running experiments: {', '.join(result['skipped_running'])}")
    if result["skipped_invalid"]:
        print(f"Skipped folders without readable metadata: {', '.join(result['skipped_invalid'])}")
    for name, run_dir in sorted(result["pointers"].items()):
        print(f"Latest {name.upper()}: {run_dir}")


if __name__ == "__main__":
    main()
