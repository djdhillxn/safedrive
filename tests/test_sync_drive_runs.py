import json
from pathlib import Path

from scripts.sync_drive_runs import (
    prune_local_training_artifacts,
    sync_run_to_drive,
    sync_runs,
    verify_critical_files,
)


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


def test_critical_file_check_requires_resumable_curriculum_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text("algorithm: {name: sac}\n", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"status": "paused", "algorithm": "sac"}),
        encoding="utf-8",
    )
    (run_dir / "curriculum_state.json").write_text(
        json.dumps({"status": "paused"}),
        encoding="utf-8",
    )
    (run_dir / "models" / "curriculum_resume_model.zip").write_bytes(b"model")

    try:
        verify_critical_files(run_dir)
    except FileNotFoundError as error:
        assert "curriculum_resume_replay_buffer.pkl" in str(error)
    else:
        raise AssertionError("A paused curriculum without replay state was accepted.")

    (run_dir / "models" / "curriculum_resume_replay_buffer.pkl").write_bytes(b"replay")
    verified = verify_critical_files(run_dir)
    assert "curriculum_state.json" in verified


def test_traffic_run_requires_lineage_logs_resource_samples_and_validation(tmp_path):
    run_dir = tmp_path / "traffic_run"
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "eval").mkdir()
    (run_dir / "resolved_config.yaml").write_text(
        "source:\n  latest_name: sac_phase2_curriculum_seed{seed}\n",
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"status": "paused", "algorithm": "sac"}),
        encoding="utf-8",
    )
    (run_dir / "curriculum_state.json").write_text(
        json.dumps({"status": "paused"}),
        encoding="utf-8",
    )
    (run_dir / "models" / "curriculum_resume_model.zip").write_bytes(b"model")
    (run_dir / "models" / "curriculum_resume_replay_buffer.pkl").write_bytes(b"replay")

    try:
        verify_critical_files(run_dir)
    except FileNotFoundError as error:
        message = str(error)
        assert "source_lineage.json" in message
        assert "resource_usage.csv" in message
        assert "best_validation_summary.json" in message
    else:
        raise AssertionError("An incomplete traffic-adaptation artifact set was accepted.")

    (run_dir / "source_lineage.json").write_text("{}", encoding="utf-8")
    (run_dir / "logs" / "curriculum_train.log").write_text("log", encoding="utf-8")
    (run_dir / "logs" / "resource_usage.csv").write_text("timesteps\n1\n", encoding="utf-8")
    (run_dir / "eval" / "best_traffic_002_validation_summary.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (run_dir / "eval" / "best_traffic_002_validation_episodes.csv").write_text(
        "episode\n0\n",
        encoding="utf-8",
    )

    verified = verify_critical_files(run_dir)
    assert "source_lineage.json" in verified
    assert "logs/resource_usage.csv" in verified


def test_drive_push_copies_and_verifies_one_complete_run(tmp_path):
    drive_project = tmp_path / "drive" / "SafeDrive"
    run_dir = tmp_path / "repository" / "runs" / "traffic_run"
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text("algorithm: {name: sac}\n", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "algorithm": "sac",
                "latest_name": "sac_traffic_reference_seed0",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "models" / "final_model.zip").write_bytes(b"model")
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "intermediate.zip").write_bytes(b"checkpoint")

    result = sync_run_to_drive(drive_project, run_dir)

    drive_run = drive_project / "runs" / "traffic_run"
    assert (drive_run / "models" / "final_model.zip").exists()
    assert not (drive_run / "checkpoints").exists()
    assert result["verified_critical_files"]
    assert (
        drive_project / "runs" / "latest_sac_traffic_reference_seed0.txt"
    ).read_text(encoding="utf-8") == "runs/traffic_run\n"


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


def test_canonical_notebook_is_valid_concise_and_has_all_twenty_sections():
    notebook = json.loads(Path("notebooks/phase2_colab_driver.ipynb").read_text())
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    headings = [
        "".join(cell["source"]).splitlines()[0]
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
        and "".join(cell["source"]).splitlines()[0].startswith(("# ", "## "))
        and not "".join(cell["source"]).splitlines()[0].startswith("### ")
    ]

    assert notebook["nbformat"] == 4
    assert all(cell.get("source") for cell in notebook["cells"])
    assert len(headings) == 20
    assert [int(heading.split(".")[0].lstrip("# ")) for heading in headings] == list(
        range(20)
    )
    assert "print(\"hello\")" not in text
    assert "subprocess.run" not in text
    assert "pilot_status not in {\"complete\", \"failed_gate\"}" in text
    assert "confirmation_status not in {\"complete\", \"failed_gate\"}" in text
    assert "excluded from completed-lineage aggregates" in text
    assert "no success video will be fabricated" in text
    assert "### 7A. Training-critical gate (blocking)" in text
    assert "### 7B. Rendering-capability gate (diagnostic and non-blocking)" in text
    assert "--training-smoke" in text
    assert "TRAINING STATUS: READY" in text
    assert "RENDERING STATUS:" in text
    assert "rendering_status.json" in text
    assert "metadrive.examples.verify_headless_installation --camera main" in text
    assert "--smoke-renderer" in text
    assert "TRAINING REMAINS AUTHORIZED" in text
    source_text = text.lower()
    for forbidden in [
        "xvfb",
        "llvmpipe",
        "libgl_always_software",
        "mesa_loader_driver_override",
        "gallium_driver",
        "force-reinstall",
        "apt-get install",
    ]:
        assert forbidden not in source_text
    for command in [
        "!python -m scripts.train_curriculum",
        "!python -m scripts.evaluate ",
        "!python -m scripts.record_video",
        "!python -m scripts.compare_runs",
        "!python -m scripts.sync_drive_runs",
    ]:
        assert command in text
    assert not Path("notebooks/colab_smoke_test.ipynb").exists()
    assert not Path("notebooks/phase1_colab_driver.ipynb").exists()
