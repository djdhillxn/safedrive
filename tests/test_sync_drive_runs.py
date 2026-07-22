import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.sync_drive_runs import prune_local_training_artifacts, sync_runs


def make_run(runs_dir, name, status, algorithm, latest_name=None, split=None):
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    metadata = {
        "status": status,
        "algorithm": algorithm,
        "config_path": "configs/ppo_mvp.yaml",
    }
    if latest_name is not None:
        metadata["latest_name"] = latest_name
    if split is not None:
        metadata["evaluation"] = {"split": split}
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "train.log").write_text(name, encoding="utf-8")
    return run_dir


def test_sync_merges_complete_and_failed_runs_but_skips_running(tmp_path):
    drive_project = tmp_path / "drive" / "SafeDrive"
    drive_runs = drive_project / "runs"
    local_runs = tmp_path / "repository" / "runs"
    drive_runs.mkdir(parents=True)
    make_run(drive_runs, "20260101_000000_ppo", "complete", "ppo")
    make_run(drive_runs, "20260101_000001_sac", "failed", "sac")
    make_run(drive_runs, "20260101_000002_ppo", "running", "ppo")
    (drive_runs / "phase1_comparison.json").write_text("{}", encoding="utf-8")

    result = sync_runs(drive_project, local_runs)

    assert (local_runs / "20260101_000000_ppo" / "logs" / "train.log").exists()
    assert (local_runs / "20260101_000001_sac" / "logs" / "train.log").exists()
    assert not (local_runs / "20260101_000002_ppo").exists()
    assert (local_runs / "phase1_comparison.json").exists()
    assert result["skipped_running"] == ["20260101_000002_ppo"]


def test_sync_rebuilds_full_and_pilot_pointers(tmp_path):
    drive_project = tmp_path / "SafeDrive"
    drive_runs = drive_project / "runs"
    local_runs = tmp_path / "runs"
    drive_runs.mkdir(parents=True)
    make_run(drive_runs, "20260101_000000_ppo", "complete", "ppo")
    make_run(drive_runs, "20260102_000000_ppo", "complete", "ppo")
    make_run(
        drive_runs,
        "20260103_000000_ppo_pilot",
        "complete",
        "ppo",
        latest_name="ppo_pilot",
    )
    make_run(
        drive_runs,
        "20260104_000000_idm_validation",
        "complete",
        "IDMPolicy",
        split="validation",
    )
    make_run(
        drive_runs,
        "20260105_000000_idm_test",
        "complete",
        "IDMPolicy",
        split="test",
    )
    (drive_runs / "latest_ppo.txt").write_text(
        "runs/20260101_000000_ppo\n",
        encoding="utf-8",
    )

    sync_runs(drive_project, local_runs)

    assert (local_runs / "latest_ppo.txt").read_text(encoding="utf-8") == (
        "runs/20260102_000000_ppo\n"
    )
    assert (local_runs / "latest_ppo_pilot.txt").read_text(encoding="utf-8") == (
        "runs/20260103_000000_ppo_pilot\n"
    )
    assert (local_runs / "latest_idm.txt").read_text(encoding="utf-8") == (
        "runs/20260105_000000_idm_test\n"
    )


def test_sync_rebuilds_pointer_for_a_paused_curriculum(tmp_path):
    drive_project = tmp_path / "SafeDrive"
    drive_runs = drive_project / "runs"
    local_runs = tmp_path / "runs"
    drive_runs.mkdir(parents=True)
    make_run(
        drive_runs,
        "20260106_000000_curriculum",
        "paused",
        "sac",
        latest_name="sac_phase2_curriculum_seed0",
    )

    sync_runs(drive_project, local_runs)

    assert (local_runs / "latest_sac_phase2_curriculum_seed0.txt").read_text(
        encoding="utf-8"
    ) == "runs/20260106_000000_curriculum\n"


def test_sync_does_not_delete_local_only_files(tmp_path):
    drive_project = tmp_path / "SafeDrive"
    drive_runs = drive_project / "runs"
    local_runs = tmp_path / "runs"
    drive_runs.mkdir(parents=True)
    local_runs.mkdir(parents=True)
    local_file = local_runs / "local_analysis.txt"
    local_file.write_text("keep me", encoding="utf-8")
    make_run(drive_runs, "20260101_000000_ppo", "complete", "ppo")

    sync_runs(drive_project, local_runs)

    assert local_file.read_text(encoding="utf-8") == "keep me"


def test_sync_replaces_same_size_top_level_file_when_contents_changed(tmp_path):
    drive_project = tmp_path / "SafeDrive"
    drive_runs = drive_project / "runs"
    local_runs = tmp_path / "runs"
    drive_runs.mkdir(parents=True)
    local_runs.mkdir(parents=True)
    source = drive_runs / "phase1_comparison.json"
    destination = local_runs / source.name
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")

    sync_runs(drive_project, local_runs)

    assert destination.read_text(encoding="utf-8") == "new"


def test_analysis_sync_skips_models_and_checkpoints_but_keeps_context(tmp_path):
    drive_project = tmp_path / "SafeDrive"
    drive_runs = drive_project / "runs"
    local_runs = tmp_path / "runs"
    run_dir = make_run(drive_runs, "20260101_000000_sac", "complete", "sac")
    (run_dir / "models").mkdir()
    (run_dir / "models" / "replay_buffer.pkl").write_bytes(b"large buffer")
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "model.zip").write_bytes(b"model")
    (run_dir / "plots").mkdir()
    (run_dir / "plots" / "outcomes.png").write_bytes(b"plot")
    (run_dir / "videos").mkdir()
    (run_dir / "videos" / "rollout.mp4").write_bytes(b"video")

    sync_runs(drive_project, local_runs)

    local_run = local_runs / run_dir.name
    assert not (local_run / "models").exists()
    assert not (local_run / "checkpoints").exists()
    assert (local_run / "logs" / "train.log").exists()
    assert (local_run / "plots" / "outcomes.png").exists()
    assert (local_run / "videos" / "rollout.mp4").exists()


def test_full_sync_includes_training_artifacts(tmp_path):
    drive_project = tmp_path / "SafeDrive"
    drive_runs = drive_project / "runs"
    local_runs = tmp_path / "runs"
    run_dir = make_run(drive_runs, "20260101_000000_sac", "complete", "sac")
    (run_dir / "models").mkdir()
    (run_dir / "models" / "replay_buffer.pkl").write_bytes(b"buffer")

    sync_runs(drive_project, local_runs, include_training_artifacts=True)

    assert (local_runs / run_dir.name / "models" / "replay_buffer.pkl").exists()


def test_sync_detects_a_video_added_after_run_metadata_stops_changing(tmp_path):
    drive_project = tmp_path / "SafeDrive"
    drive_runs = drive_project / "runs"
    local_runs = tmp_path / "runs"
    drive_runs.mkdir(parents=True)
    run_dir = make_run(drive_runs, "20260101_000000_sac", "complete", "sac")

    sync_runs(drive_project, local_runs)
    (run_dir / "videos").mkdir()
    (run_dir / "videos" / "rollout.mp4").write_bytes(b"late video")
    sync_runs(drive_project, local_runs)

    assert (local_runs / run_dir.name / "videos" / "rollout.mp4").read_bytes() == b"late video"


def test_local_prune_removes_training_artifacts_only(tmp_path):
    local_runs = tmp_path / "runs"
    run_dir = local_runs / "20260101_000000_sac"
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "models" / "replay_buffer.pkl").write_bytes(b"buffer")
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "model.zip").write_bytes(b"model")
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "train.log").write_text("keep", encoding="utf-8")
    (run_dir / "videos").mkdir()
    (run_dir / "videos" / "rollout.mp4").write_bytes(b"video")

    directories, files, byte_count = prune_local_training_artifacts(local_runs)

    assert directories == 2
    assert files == 2
    assert byte_count == 11
    assert not (run_dir / "models").exists()
    assert not (run_dir / "checkpoints").exists()
    assert (run_dir / "logs" / "train.log").exists()
    assert (run_dir / "videos" / "rollout.mp4").exists()


def test_phase2_notebook_restores_saved_variables_and_complete_pairs(tmp_path):
    repository = tmp_path / "repository"
    runs = repository / "runs"
    runs.mkdir(parents=True)

    def add_learned_run(condition, seed, status, with_summary):
        name = f"run_{condition}_{seed}"
        run_dir = runs / name
        (run_dir / "eval").mkdir(parents=True)
        pointer = runs / f"latest_sac_phase2_{condition}_seed{seed}.txt"
        pointer.write_text(f"runs/{name}\n", encoding="utf-8")
        if condition == "curriculum":
            state = {
                "status": status,
                "next_stage_index": 3 if status == "complete" else 1,
                "completed_stages": [{"name": "curve"}],
            }
            (run_dir / "curriculum_state.json").write_text(
                json.dumps(state),
                encoding="utf-8",
            )
        else:
            (run_dir / "run_metadata.json").write_text(
                json.dumps({"status": status}),
                encoding="utf-8",
            )
        if with_summary:
            summary = {
                "episodes": 100,
                "success_rate": 0.9 if condition == "curriculum" else 0.2,
                "mean_route_completion": 0.95 if condition == "curriculum" else 0.6,
            }
            (run_dir / "eval" / "best_test_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

    for seed in [0, 1]:
        add_learned_run("direct", seed, "complete", True)
        add_learned_run("curriculum", seed, "complete", True)
    add_learned_run("direct", 2, "complete", True)
    add_learned_run("curriculum", 2, "failed_gate", False)

    notebook = json.loads(Path("notebooks/phase2_colab_driver.ipynb").read_text())
    helper_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def restore_phase2_session_state" in "".join(cell["source"])
    )
    namespace = {
        "Path": Path,
        "json": json,
        "os": os,
        "shutil": shutil,
        "subprocess": subprocess,
        "sys": sys,
        "REPO_DIR": repository,
        "DRIVE_PROJECT": tmp_path / "drive" / "SafeDrive",
        "PHASE2_TIMESTEPS": 500_000,
        "PHASE2_TEST_EPISODES": 100,
        "VIDEO_STEPS": 1_000,
    }

    exec(helper_source, namespace)

    assert namespace["PILOTS_PROMISING"] is True
    assert namespace["COMPLETED_PHASE2_SEEDS"] == [0, 1]
    assert namespace["DIRECT_SEED2_SUMMARY"]["episodes"] == 100
    assert namespace["learned_run_status"](namespace["CURRICULUM_SEED2_RUN"]) == (
        "failed_gate"
    )
