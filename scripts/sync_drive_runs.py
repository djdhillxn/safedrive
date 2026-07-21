"""Copy SafeDrive run artifacts from Google Drive into this repository.

This command is intentionally one-way and non-destructive: Drive artifacts are
merged into the local ``runs`` folder, while local-only files are never deleted.
"""

import argparse
import filecmp
import json
import os
import shutil
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IGNORED_NAMES = {".DS_Store"}


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
        "--dry-run",
        action="store_true",
        help="Show what would be copied without changing files.",
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
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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


def merge_directory(source, destination, dry_run=False):
    copied_files = 0
    for path in source.rglob("*"):
        if path.name in IGNORED_NAMES or not path.is_file():
            continue
        relative_path = path.relative_to(source)
        if copy_file(path, destination / relative_path, dry_run=dry_run):
            copied_files += 1
    return copied_files


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
        if not metadata or metadata.get("status") != "complete":
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


def sync_runs(drive_project, local_runs, include_running=False, dry_run=False):
    drive_runs = drive_project / "runs"
    if not drive_runs.is_dir():
        raise FileNotFoundError(f"Google Drive runs folder was not found: {drive_runs}")
    if not dry_run:
        local_runs.mkdir(parents=True, exist_ok=True)

    copied_runs = []
    skipped_running = []
    skipped_invalid = []
    copied_files = 0

    for source in sorted(drive_runs.iterdir(), key=lambda path: path.name):
        if source.name in IGNORED_NAMES:
            continue
        if source.is_file():
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

        metadata = read_metadata(source)
        if metadata is None:
            skipped_invalid.append(source.name)
            continue
        if metadata.get("status") == "running" and not include_running:
            skipped_running.append(source.name)
            continue

        changed = merge_directory(source, local_runs / source.name, dry_run=dry_run)
        copied_files += changed
        if changed:
            copied_runs.append(source.name)

    pointers = refresh_latest_pointers(local_runs, dry_run=dry_run)
    return {
        "copied_files": copied_files,
        "changed_runs": copied_runs,
        "skipped_running": skipped_running,
        "skipped_invalid": skipped_invalid,
        "pointers": pointers,
    }


def main():
    args = parse_args()
    drive_project = discover_drive_project(args.drive_project)
    local_runs = Path(args.local_runs).expanduser().resolve()
    print(f"Drive source: {drive_project / 'runs'}")
    print(f"Local destination: {local_runs}")
    result = sync_runs(
        drive_project,
        local_runs,
        include_running=args.include_running,
        dry_run=args.dry_run,
    )
    print(
        f"Sync complete: {result['copied_files']} files updated across "
        f"{len(result['changed_runs'])} run directories."
    )
    if result["skipped_running"]:
        print(f"Skipped running experiments: {', '.join(result['skipped_running'])}")
    if result["skipped_invalid"]:
        print(f"Skipped folders without readable metadata: {', '.join(result['skipped_invalid'])}")
    for name, run_dir in sorted(result["pointers"].items()):
        print(f"Latest {name.upper()}: {run_dir}")


if __name__ == "__main__":
    main()
