"""Compare compatible SafeDrive evaluation runs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from saferl_drive.config import (
    fingerprint_differences,
    load_yaml,
    make_experiment_fingerprint,
)
from saferl_drive.evaluation import comparison_summary_row, summarize_metrics
from saferl_drive.utils import (
    _plotter,
    log_system_info,
    plot_comparison_rows,
    plot_phase1_training_returns,
    read_json,
    read_latest_run,
    setup_logging,
    utc_timestamp,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare compatible SafeDrive evaluations.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--summaries", nargs="+", help="Paths to summary JSON files.")
    mode.add_argument("--phase1", action="store_true", help="Build the Phase-1 table.")
    mode.add_argument(
        "--phase2",
        action="store_true",
        help="Compare Phase-2 direct and curriculum SAC runs.",
    )
    mode.add_argument(
        "--traffic-extension",
        action="store_true",
        help="Select traffic pilots or build the final traffic-extension comparison.",
    )
    parser.add_argument("--output", default=None, help="Manual comparison PNG output path.")
    parser.add_argument("--runs-dir", default="runs", help="Runs directory.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Phase-2 algorithm seeds to include.",
    )
    parser.add_argument(
        "--select-pilots",
        action="store_true",
        help="Write the predeclared seed-0 traffic-pilot decision without requiring seed 1.",
    )
    return parser.parse_args()


def _preferred_summary(run_dir, name):
    name = name.lower()
    if "idm" in name:
        candidates = [
            run_dir / "eval" / "phase2_idm_test_summary.json",
            run_dir / "eval" / "idm_test_summary.json",
            run_dir / "eval" / "idm_unseen_summary.json",
        ]
    elif "expert" in name:
        candidates = [
            run_dir / "eval" / "phase2_expert_test_summary.json",
            run_dir / "eval" / "expert_test_summary.json",
        ]
    else:
        candidates = [
            run_dir / "eval" / "best_test_summary.json",
            run_dir / "eval" / "final_test_summary.json",
            run_dir / "eval" / "best_unseen_summary.json",
            run_dir / "eval" / "final_unseen_summary.json",
        ]

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No held-out test summary found under {run_dir / 'eval'}")


def _controller_name(label):
    lowered = label.lower()
    if "idm" in lowered:
        return "IDMPolicy"
    if "expert" in lowered:
        return "ExpertPolicy"
    return None


def _summary_record(label, run_dir, summary_path=None):
    run_dir = Path(run_dir)
    summary_path = Path(summary_path or _preferred_summary(run_dir, label))
    summary = read_json(summary_path)
    fingerprint = summary.get("experiment_fingerprint")
    if fingerprint is None:
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.exists():
            raise ValueError(
                f"{label} has no experiment fingerprint and no resolved config: {summary_path}"
            )
        config = load_yaml(config_path)
        split = summary.get("evaluation_split", "test")
        fingerprint = make_experiment_fingerprint(
            config,
            split=split,
            episodes=int(summary.get("episodes", 0)),
            controller=_controller_name(label),
        )
    return {
        "label": label,
        "run_dir": run_dir,
        "summary_path": summary_path,
        "summary": summary,
        "fingerprint": fingerprint,
    }


def _compatibility(records, strict):
    if not records:
        raise ValueError("No evaluation records were provided.")
    identifier = "strict_id" if strict else "task_id"
    reference = records[0]
    mismatches = []
    for record in records[1:]:
        if record["fingerprint"].get(identifier) == reference["fingerprint"].get(identifier):
            continue
        mismatches.append(
            {
                "reference": reference["label"],
                "other": record["label"],
                "differences": fingerprint_differences(
                    reference["fingerprint"],
                    record["fingerprint"],
                    include_action=strict,
                ),
            }
        )
    if mismatches:
        lines = [
            "Evaluation fingerprints are incompatible; comparison was refused.",
            f"Compatibility mode: {'strict task and action' if strict else 'task only'}.",
        ]
        for mismatch in mismatches:
            lines.append(f"{mismatch['reference']} versus {mismatch['other']}:")
            for difference in mismatch["differences"]:
                lines.append(
                    f"  {difference['field']}: {difference['first']!r} != "
                    f"{difference['second']!r}"
                )
        raise ValueError("\n".join(lines))
    return {
        "mode": "strict" if strict else "task_only",
        "identifier": reference["fingerprint"].get(identifier),
        "compatible": True,
    }


def _rows(records):
    return [
        comparison_summary_row(
            record["label"],
            record["summary"],
            run_dir=record["run_dir"],
            summary_path=record["summary_path"],
        )
        for record in records
    ]


def _tex_percent(value):
    if value is None:
        return "--"
    return f"{100.0 * float(value):.1f}\\%"


def _write_phase1_report_data(rows):
    by_name = {row["name"]: row for row in rows}
    names = {
        "PpoSuccess": ("PPO", "success_rate"),
        "SacSuccess": ("SAC", "success_rate"),
        "IdmSuccess": ("IDM", "success_rate"),
        "PpoCollision": ("PPO", "collision_rate"),
        "SacCollision": ("SAC", "collision_rate"),
        "IdmCollision": ("IDM", "collision_rate"),
        "PpoOffRoad": ("PPO", "out_of_road_rate"),
        "SacOffRoad": ("SAC", "out_of_road_rate"),
        "IdmOffRoad": ("IDM", "out_of_road_rate"),
        "PpoCompletion": ("PPO", "mean_route_completion"),
        "SacCompletion": ("SAC", "mean_route_completion"),
        "IdmCompletion": ("IDM", "mean_route_completion"),
    }
    lines = ["% Generated by python -m scripts.compare_runs --phase1"]
    for macro, (row_name, metric) in names.items():
        lines.append(f"\\renewcommand{{\\{macro}}}{{{_tex_percent(by_name[row_name][metric])}}}")
    path = Path("reports/generated_phase1_results.tex")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_phase2_report_data(aggregates):
    by_name = {row["name"]: row for row in aggregates}
    if "Direct SAC" not in by_name or "Curriculum SAC" not in by_name:
        return None
    names = {
        "DirectSuccess": ("Direct SAC", "success_rate"),
        "CurriculumSuccess": ("Curriculum SAC", "success_rate"),
        "DirectCollision": ("Direct SAC", "collision_rate"),
        "CurriculumCollision": ("Curriculum SAC", "collision_rate"),
        "DirectOffRoad": ("Direct SAC", "out_of_road_rate"),
        "CurriculumOffRoad": ("Curriculum SAC", "out_of_road_rate"),
        "DirectCompletion": ("Direct SAC", "mean_route_completion"),
        "CurriculumCompletion": ("Curriculum SAC", "mean_route_completion"),
    }
    lines = ["% Generated by python -m scripts.compare_runs --phase2"]
    for macro, (row_name, metric) in names.items():
        lines.append(
            f"\\renewcommand{{\\{macro}}}{{{_tex_percent(by_name[row_name].get(metric))}}}"
        )
    path = Path("reports/generated_phase2_results.tex")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_phase1(args, logger):
    runs_dir = Path(args.runs_dir)
    records = []
    for label, pointer in [("IDM", "idm"), ("PPO", "ppo"), ("SAC", "sac")]:
        record = _summary_record(label, read_latest_run(runs_dir, pointer))
        records.append(record)
        logger.debug("Selected %s summary: %s", label, record["summary_path"])

    task_compatibility = _compatibility(records, strict=False)
    action_ids = {record["label"]: record["fingerprint"]["strict_id"] for record in records}
    rows = _rows(records)
    csv_path = runs_dir / "phase1_comparison.csv"
    json_path = runs_dir / "phase1_comparison.json"
    plot_path = runs_dir / "phase1_comparison.png"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_json(
        {
            "generated_at_utc": utc_timestamp(),
            "comparison_scope": "descriptive task outcomes",
            "fair_algorithm_ranking": False,
            "note": (
                "The environment, reward, and test split match. Controller action "
                "interfaces differ, so this table does not establish algorithm superiority."
            ),
            "task_compatibility": task_compatibility,
            "strict_action_fingerprints": action_ids,
            "experiments": rows,
        },
        json_path,
    )
    plot_comparison_rows(rows, plot_path, title="Phase-1 held-out task outcomes")
    report_data = _write_phase1_report_data(rows)
    training_plot = plot_phase1_training_returns(
        {"PPO": records[1]["run_dir"], "SAC": records[2]["run_dir"]},
        runs_dir / "phase1_training_returns.png",
    )
    logger.warning(
        "Phase-1 action interfaces differ. The output is descriptive and cannot rank "
        "PPO, SAC, and IDM as a fair algorithm comparison."
    )
    logger.info("Phase-1 descriptive table written: %s", csv_path)
    logger.info("Phase-1 LaTeX result macros written: %s", report_data)
    logger.info("Comparison plot written: %s", plot_path)
    if training_plot is not None:
        logger.info("Training comparison written: %s", training_plot)


def _aggregate_phase2(rows):
    frame = pd.DataFrame(rows)
    metric_names = [
        "success_rate",
        "collision_rate",
        "out_of_road_rate",
        "timeout_or_max_step_rate",
        "mean_route_completion",
        "mean_return",
        "mean_length",
        "mean_cost",
        "mean_speed_km_h",
        "mean_abs_steering",
        "mean_action_change",
    ]
    aggregates = []
    for condition, group in frame.groupby("condition", sort=False):
        row = {"name": condition, "training_seeds": int(len(group))}
        for metric in metric_names:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[metric] = float(values.mean())
            row[f"{metric}_across_seed_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
        aggregates.append(row)
    return aggregates


def select_traffic_pilot(reference, safety, source_success, settings=None):
    """Apply the predeclared seed-0 traffic-pilot selection rules."""
    settings = settings or {}
    minimum_success = float(settings.get("minimum_success_rate", 0.80))
    maximum_collision = float(settings.get("maximum_collision_rate", 0.10))
    maximum_offroad = float(settings.get("maximum_out_of_road_rate", 0.10))
    minimum_route = float(settings.get("minimum_route_completion", 0.90))
    maximum_drop = float(settings.get("maximum_traffic_free_success_drop", 0.10))
    collision_tie = float(settings.get("collision_tie", 0.02))
    success_tie = float(settings.get("success_tie", 0.05))

    def qualifies(conditions):
        primary = conditions["0.05"]
        traffic_free = conditions["0.00"]
        return (
            float(primary["success_rate"]) >= minimum_success
            and float(primary["collision_rate"]) <= maximum_collision
            and float(primary["out_of_road_rate"]) <= maximum_offroad
            and float(primary["mean_route_completion"]) >= minimum_route
            and float(source_success) - float(traffic_free["success_rate"]) <= maximum_drop
        )

    candidates = {"reference": reference, "safety": safety}
    qualification = {name: qualifies(value) for name, value in candidates.items()}
    qualified = [name for name, passed in qualification.items() if passed]
    reference_primary = reference["0.05"]
    safety_primary = safety["0.05"]
    success_difference = abs(
        float(reference_primary["success_rate"]) - float(safety_primary["success_rate"])
    )
    collision_difference = abs(
        float(reference_primary["collision_rate"]) - float(safety_primary["collision_rate"])
    )

    if len(qualified) == 1:
        selected = qualified[0]
        reason = "Exactly one pilot met every predeclared qualification threshold."
    elif len(qualified) == 2:
        if collision_difference > collision_tie:
            selected = min(
                qualified,
                key=lambda name: float(candidates[name]["0.05"]["collision_rate"]),
            )
            reason = "Both qualified; selected the lower-collision pilot."
        elif success_difference > 1e-12:
            selected = max(
                qualified,
                key=lambda name: float(candidates[name]["0.05"]["success_rate"]),
            )
            reason = "Both qualified and collision differed by at most two points; used success."
        else:
            selected = max(
                qualified,
                key=lambda name: float(candidates[name]["0.05"]["mean_route_completion"]),
            )
            reason = "Both qualified with tied collision and success; used route completion."
    elif success_difference > success_tie:
        selected = max(
            candidates,
            key=lambda name: float(candidates[name]["0.05"]["success_rate"]),
        )
        reason = "Neither qualified; selected the pilot with higher primary success."
    else:
        selected = min(
            candidates,
            key=lambda name: float(candidates[name]["0.05"]["collision_rate"]),
        )
        reason = "Neither qualified and success differed by at most five points; used collision."

    return {
        "selected_variant": selected,
        "reason": reason,
        "qualification": qualification,
        "source_traffic_free_success_rate": float(source_success),
        "settings": {
            "minimum_success_rate": minimum_success,
            "maximum_collision_rate": maximum_collision,
            "maximum_out_of_road_rate": maximum_offroad,
            "minimum_route_completion": minimum_route,
            "maximum_traffic_free_success_drop": maximum_drop,
            "collision_tie": collision_tie,
            "success_tie": success_tie,
        },
        "compared_metrics": candidates,
    }


def _density_key(value):
    return f"{float(value):.2f}"


def _matrix_summaries(run_dir, prefix):
    run_dir = Path(run_dir)
    summaries = {}
    for density in [0.0, 0.05, 0.10]:
        label = f"d{int(round(density * 100)):03d}"
        path = run_dir / "eval" / f"{prefix}_{label}_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing traffic evaluation summary: {path}")
        summaries[_density_key(density)] = read_json(path)
    return summaries


def _write_traffic_selection(runs_dir):
    runs_dir = Path(runs_dir)
    source_run = read_latest_run(runs_dir, "sac_phase2_curriculum_seed0")
    reference_run = read_latest_run(runs_dir, "sac_traffic_reference_seed0")
    safety_run = read_latest_run(runs_dir, "sac_traffic_safety_seed0")
    source = _matrix_summaries(source_run, "traffic_source_validation")
    reference = _matrix_summaries(reference_run, "traffic_pilot")
    safety = _matrix_summaries(safety_run, "traffic_pilot")
    traffic_config = load_yaml("configs/sac_traffic_curriculum.yaml")
    decision = select_traffic_pilot(
        reference,
        safety,
        source["0.00"]["success_rate"],
        traffic_config.get("pilot_selection", {}),
    )
    decision.update(
        {
            "generated_at_utc": utc_timestamp(),
            "source_run": str(source_run),
            "reference_run": str(reference_run),
            "safety_run": str(safety_run),
            "reference_training_status": read_json(
                reference_run / "run_metadata.json"
            ).get("status", "unknown"),
            "safety_training_status": read_json(
                safety_run / "run_metadata.json"
            ).get("status", "unknown"),
        }
    )
    path = runs_dir / "traffic_extension_selection.json"
    write_json(decision, path)
    return decision, path


def _traffic_matrix_frame(
    run_dir,
    prefix,
    condition,
    training_seed,
    training_status="complete",
):
    path = Path(run_dir) / "eval" / f"{prefix}_matrix_episodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing final traffic matrix: {path}")
    frame = pd.read_csv(path)
    required = {
        "traffic_density",
        "terminal_outcome",
        "success",
        "crash",
        "crash_vehicle",
        "collision_free",
        "out_of_road",
        "max_step",
        "route_completion",
        "experiment_task_fingerprint",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Traffic evaluation {path} is missing columns: {missing}")
    frame["condition"] = condition
    frame["training_seed"] = training_seed
    frame["training_status"] = training_status
    frame["run_dir"] = str(run_dir)
    return frame


def _validate_traffic_fingerprints(records):
    """Require one scenario protocol, allowing only the declared collision ablation."""
    allowed_reward_differences = {
        "crash_vehicle_penalty",
        "crash_object_penalty",
    }
    checks = []
    for density in [0.0, 0.05, 0.10]:
        label = f"d{int(round(density * 100)):03d}"
        condition_records = []
        for condition, run_dir, prefix, learned in records:
            summary_path = Path(run_dir) / "eval" / f"{prefix}_{label}_summary.json"
            summary = read_json(summary_path)
            fingerprint = summary.get("experiment_fingerprint")
            if not fingerprint:
                raise ValueError(f"Traffic summary has no experiment fingerprint: {summary_path}")
            condition_records.append((condition, fingerprint, learned, summary_path))
        reference_name, reference, _, _ = condition_records[0]
        reference_task = reference["task"]
        for condition, fingerprint, learned, summary_path in condition_records[1:]:
            task = fingerprint["task"]
            if task["environment"] != reference_task["environment"]:
                raise ValueError(
                    f"Traffic environment mismatch at density {density}: "
                    f"{reference_name} versus {condition} ({summary_path})."
                )
            if task["evaluation"] != reference_task["evaluation"]:
                raise ValueError(
                    f"Traffic scenario split mismatch at density {density}: "
                    f"{reference_name} versus {condition} ({summary_path})."
                )
            reward_differences = {
                key
                for key in set(task["reward"]) | set(reference_task["reward"])
                if task["reward"].get(key) != reference_task["reward"].get(key)
            }
            if not reward_differences <= allowed_reward_differences:
                raise ValueError(
                    f"Undeclared reward mismatch at density {density}: "
                    f"{sorted(reward_differences)}."
                )
            if learned and fingerprint["action"] != reference["action"]:
                raise ValueError(
                    f"Learned-policy action mismatch at density {density}: "
                    f"{reference_name} versus {condition}."
                )
        checks.append(
            {
                "traffic_density": density,
                "environment": reference_task["environment"],
                "evaluation": reference_task["evaluation"],
                "allowed_reward_differences": sorted(allowed_reward_differences),
            }
        )
    return checks


def _traffic_seed_rows(frame):
    rows = []
    for condition, condition_frame in frame.groupby("condition", sort=False):
        if condition_frame["training_seed"].notna().any():
            grouped = condition_frame.groupby(
                ["training_seed", "traffic_density"],
                sort=False,
            )
        else:
            grouped = condition_frame.groupby(["traffic_density"], sort=False)
        for key, group in grouped:
            if isinstance(key, tuple) and len(key) == 2:
                seed, density = key
            else:
                density = key[0] if isinstance(key, tuple) else key
                seed = None
            summary = summarize_metrics(group)
            rows.append(
                {
                    "condition": condition,
                    "training_seed": int(seed) if seed is not None else None,
                    "traffic_density": float(density),
                    **summary,
                    "source_checkpoint": group["source_checkpoint"].iloc[0],
                    "experiment_fingerprint": group["experiment_fingerprint"].iloc[0],
                    "training_status": group["training_status"].iloc[0],
                    "episodes": int(len(group)),
                }
            )
    return rows


def _traffic_aggregates(seed_rows):
    frame = pd.DataFrame(seed_rows)
    metrics = [
        "success_rate",
        "collision_rate",
        "crash_vehicle_rate",
        "collision_free_rate",
        "out_of_road_rate",
        "timeout_or_max_step_rate",
        "mean_route_completion",
        "mean_return",
        "mean_length",
        "mean_cost",
        "mean_speed_km_h",
        "mean_abs_steering",
        "mean_action_change",
    ]
    rows = []
    for (condition, density), group in frame.groupby(
        ["condition", "traffic_density"],
        sort=False,
    ):
        row = {
            "condition": condition,
            "traffic_density": float(density),
            "training_seeds": int(len(group)),
        }
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[metric] = float(values.mean())
            row[f"{metric}_across_seed_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
        rows.append(row)
    return rows


def _traffic_effects(seed_rows):
    frame = pd.DataFrame(seed_rows)
    effects = []
    for seed in sorted(frame["training_seed"].unique()):
        source = frame[
            (frame["training_seed"] == seed) & (frame["condition"] == "Frozen curriculum SAC")
        ].set_index("traffic_density")
        adapted = frame[
            (frame["training_seed"] == seed) & (frame["condition"] == "Adapted SAC")
        ].set_index("traffic_density")
        if source.empty or adapted.empty:
            continue
        effects.append(
            {
                "training_seed": int(seed),
                "density_005_success_gain": float(
                    adapted.loc[0.05, "success_rate"] - source.loc[0.05, "success_rate"]
                ),
                "density_005_collision_change": float(
                    adapted.loc[0.05, "collision_rate"] - source.loc[0.05, "collision_rate"]
                ),
                "traffic_free_success_change": float(
                    adapted.loc[0.0, "success_rate"] - source.loc[0.0, "success_rate"]
                ),
                "traffic_free_route_completion_change": float(
                    adapted.loc[0.0, "mean_route_completion"]
                    - source.loc[0.0, "mean_route_completion"]
                ),
                "adapted_success_degradation_005_to_010": float(
                    adapted.loc[0.10, "success_rate"] - adapted.loc[0.05, "success_rate"]
                ),
            }
        )
    return effects


def _plot_traffic_results(all_episodes, seed_rows, runs_dir, training_dirs):
    plt = _plotter()
    aggregate = pd.DataFrame(_traffic_aggregates(seed_rows))
    learned = aggregate[
        aggregate["condition"].isin(["Frozen curriculum SAC", "Adapted SAC"])
    ]

    path = runs_dir / "traffic_extension_success_collision.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for condition, group in learned.groupby("condition", sort=False):
        group = group.sort_values("traffic_density")
        axes[0].plot(group["traffic_density"], group["success_rate"], marker="o", label=condition)
        axes[1].plot(
            group["traffic_density"],
            group["collision_rate"],
            marker="o",
            label=condition,
        )
    axes[0].set_title("Success")
    axes[1].set_title("Collision")
    for axis in axes:
        axis.set_xlabel("Traffic density")
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Rate")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)

    path = runs_dir / "traffic_extension_outcomes.png"
    primary = all_episodes[
        (all_episodes["traffic_density"] == 0.05)
        & all_episodes["condition"].isin(["Frozen curriculum SAC", "Adapted SAC"])
    ]
    composition = (
        primary.groupby(["condition", "terminal_outcome"]).size()
        / primary.groupby("condition").size()
    ).unstack(fill_value=0)
    composition = composition.reindex(
        columns=["success", "collision", "out_of_road", "timeout", "other"],
        fill_value=0,
    )
    axis = composition.plot(kind="bar", stacked=True, figsize=(8, 4))
    axis.set_ylabel("Episode fraction")
    axis.set_title("Mutually exclusive outcomes at density 0.05")
    axis.set_ylim(0, 1)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=180)
    plt.close(axis.figure)

    path = runs_dir / "traffic_extension_route_completion.png"
    figure, axis = plt.subplots(figsize=(7, 4))
    for condition, group in learned.groupby("condition", sort=False):
        group = group.sort_values("traffic_density")
        axis.plot(
            group["traffic_density"],
            group["mean_route_completion"],
            marker="o",
            label=condition,
        )
    axis.set_xlabel("Traffic density")
    axis.set_ylabel("Mean route completion")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)

    path = runs_dir / "traffic_extension_retention.png"
    retention = learned[learned["traffic_density"] == 0.0]
    positions = np.arange(len(retention))
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(positions - 0.18, retention["success_rate"], 0.36, label="Success")
    axis.bar(
        positions + 0.18,
        retention["mean_route_completion"],
        0.36,
        label="Route completion",
    )
    axis.set_xticks(positions, retention["condition"])
    axis.set_ylim(0, 1)
    axis.set_title("Traffic-free retention")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)

    plot_phase1_training_returns(
        training_dirs,
        runs_dir / "traffic_extension_training_returns.png",
        title="Traffic-adaptation learning curves",
    )


def _write_traffic_report_data(aggregates, effects):
    by_condition_density = {
        (row["condition"], float(row["traffic_density"])): row
        for row in aggregates
    }
    adapted = {
        float(row["traffic_density"]): row
        for row in aggregates
        if row["condition"] == "Adapted SAC"
    }
    if set(adapted) != {0.0, 0.05, 0.1}:
        return None
    mean_effects = pd.DataFrame(effects).mean(numeric_only=True).to_dict()
    lines = ["% Generated by python -m scripts.compare_runs --traffic-extension"]
    values = {
        "TrafficAdaptedSeedCount": int(adapted[0.0].get("training_seeds", 0)),
        "TrafficSuccessZero": adapted[0.0]["success_rate"],
        "TrafficSuccessPrimary": adapted[0.05]["success_rate"],
        "TrafficSuccessStress": adapted[0.1]["success_rate"],
        "TrafficCollisionZero": adapted[0.0]["collision_rate"],
        "TrafficCollisionPrimary": adapted[0.05]["collision_rate"],
        "TrafficCollisionStress": adapted[0.1]["collision_rate"],
        "TrafficRouteZero": adapted[0.0]["mean_route_completion"],
        "TrafficRoutePrimary": adapted[0.05]["mean_route_completion"],
        "TrafficRouteStress": adapted[0.1]["mean_route_completion"],
        "TrafficPrimaryGain": mean_effects.get("density_005_success_gain"),
        "TrafficPrimaryCollisionChange": mean_effects.get(
            "density_005_collision_change"
        ),
        "TrafficFreeSuccessChange": mean_effects.get(
            "traffic_free_success_change"
        ),
        "TrafficFreeRouteChange": mean_effects.get(
            "traffic_free_route_completion_change"
        ),
        "TrafficStressSuccessChange": mean_effects.get(
            "adapted_success_degradation_005_to_010"
        ),
    }
    for macro_prefix, condition in [
        ("TrafficSource", "Frozen curriculum SAC"),
        ("TrafficIdm", "IDMPolicy"),
        ("TrafficExpert", "ExpertPolicy"),
    ]:
        row = by_condition_density.get((condition, 0.05), {})
        values[f"{macro_prefix}SuccessPrimary"] = row.get("success_rate")
        values[f"{macro_prefix}CollisionPrimary"] = row.get("collision_rate")
        values[f"{macro_prefix}RoutePrimary"] = row.get("mean_route_completion")
    seed_count = values.pop("TrafficAdaptedSeedCount")
    lines.append(f"\\renewcommand{{\\TrafficAdaptedSeedCount}}{{{seed_count}}}")
    for name, value in values.items():
        lines.append(f"\\renewcommand{{\\{name}}}{{{_tex_percent(value)}}}")
    path = Path("reports/generated_traffic_results.tex")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_traffic_extension(args, logger):
    runs_dir = Path(args.runs_dir)
    decision_path = runs_dir / "traffic_extension_selection.json"
    if args.select_pilots or not decision_path.exists():
        decision, decision_path = _write_traffic_selection(runs_dir)
        logger.info(
            "Selected traffic variant %s: %s",
            decision["selected_variant"],
            decision["reason"],
        )
        if args.select_pilots:
            return
    else:
        decision = read_json(decision_path)

    selected = decision["selected_variant"]
    frames = []
    training_dirs = {}
    fingerprint_records = []
    for seed in [0, 1]:
        source_run = read_latest_run(runs_dir, f"sac_phase2_curriculum_seed{seed}")
        adapted_run = read_latest_run(runs_dir, f"sac_traffic_{selected}_seed{seed}")
        frames.append(
            _traffic_matrix_frame(
                source_run,
                "traffic_before",
                "Frozen curriculum SAC",
                seed,
                "complete",
            )
        )
        fingerprint_records.append(
            ("Frozen curriculum SAC", source_run, "traffic_before", True)
        )
        adapted_metadata = read_json(adapted_run / "run_metadata.json")
        adapted_status = adapted_metadata.get("status", "unknown")
        adapted_condition = (
            "Adapted SAC"
            if adapted_status == "complete"
            else f"Adapted SAC ({adapted_status.replace('_', ' ')})"
        )
        frames.append(
            _traffic_matrix_frame(
                adapted_run,
                "traffic_final",
                adapted_condition,
                seed,
                adapted_status,
            )
        )
        fingerprint_records.append((adapted_condition, adapted_run, "traffic_final", True))
        training_dirs[f"{adapted_condition} seed {seed}"] = adapted_run

    for condition, pointer, prefix in [
        ("IDMPolicy", "traffic_extension_idm", "traffic_final_idm"),
        ("ExpertPolicy", "traffic_extension_expert", "traffic_final_expert"),
    ]:
        run_dir = read_latest_run(runs_dir, pointer)
        frames.append(
            _traffic_matrix_frame(
                run_dir,
                prefix,
                condition,
                np.nan,
                "native_controller",
            )
        )
        fingerprint_records.append((condition, run_dir, prefix, False))

    compatibility = _validate_traffic_fingerprints(fingerprint_records)
    all_episodes = pd.concat(frames, ignore_index=True)
    seed_rows = _traffic_seed_rows(all_episodes)
    aggregates = _traffic_aggregates(seed_rows)
    effects = _traffic_effects(seed_rows)
    failed_lineages = [
        row
        for row in seed_rows
        if row["condition"].startswith("Adapted SAC (")
    ]
    all_episodes.to_csv(runs_dir / "traffic_extension_episodes.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(
        runs_dir / "traffic_extension_seed_results.csv",
        index=False,
    )
    pd.DataFrame(aggregates).to_csv(
        runs_dir / "traffic_extension_comparison.csv",
        index=False,
    )
    write_json(
        {
            "generated_at_utc": utc_timestamp(),
            "research_question": (
                "Can geometry-competent SAC adapt to interactive procedural traffic while "
                "reducing collisions and retaining traffic-free road generalization?"
            ),
            "single_agent": True,
            "background_traffic_control": "MetaDrive native traffic manager",
            "selected_variant": selected,
            "compatibility": compatibility,
            "selection": decision,
            "seed_results": seed_rows,
            "aggregates": aggregates,
            "adaptation_effects": effects,
            "failed_adaptation_lineages": failed_lineages,
            "limitations": (
                "Two training seeds in simulation do not establish real-world "
                "autonomous-driving safety."
            ),
        },
        runs_dir / "traffic_extension_comparison.json",
    )
    _plot_traffic_results(all_episodes, seed_rows, runs_dir, training_dirs)
    report_data = _write_traffic_report_data(aggregates, effects)
    logger.info("Traffic-extension comparison written under %s.", runs_dir)
    if report_data is not None:
        logger.info("Traffic report macros written: %s", report_data)


def _run_phase2(args, logger):
    runs_dir = Path(args.runs_dir)
    learned_records = []
    training_dirs = {}
    rows = []
    for seed in args.seeds:
        for condition, pointer_stem in [
            ("Direct SAC", "sac_phase2_direct_seed"),
            ("Curriculum SAC", "sac_phase2_curriculum_seed"),
        ]:
            pointer = f"{pointer_stem}{seed}"
            record = _summary_record(
                f"{condition} seed {seed}",
                read_latest_run(runs_dir, pointer),
            )
            learned_records.append(record)
            training_dirs[f"{condition} seed {seed}"] = record["run_dir"]
            row = comparison_summary_row(
                record["label"],
                record["summary"],
                run_dir=record["run_dir"],
                summary_path=record["summary_path"],
            )
            row["condition"] = condition
            row["training_seed"] = seed
            rows.append(row)
            logger.debug("Selected %s summary: %s", record["label"], record["summary_path"])

    strict_compatibility = _compatibility(learned_records, strict=True)
    baseline_records = []
    for label, pointer in [("Phase-2 IDM", "phase2_idm"), ("Phase-2 Expert", "phase2_expert")]:
        try:
            baseline_records.append(_summary_record(label, read_latest_run(runs_dir, pointer)))
        except FileNotFoundError:
            logger.warning("Optional baseline pointer is missing: latest_%s.txt", pointer)
    baseline_compatibility = None
    if baseline_records:
        baseline_compatibility = _compatibility(
            [learned_records[0], *baseline_records],
            strict=False,
        )

    light_traffic_records = []
    light_traffic_rows = []
    for record, row in zip(learned_records, rows):
        summary_path = record["run_dir"] / "eval" / "best_light_traffic_summary.json"
        if not summary_path.exists():
            continue
        light_record = _summary_record(
            record["label"],
            record["run_dir"],
            summary_path=summary_path,
        )
        light_traffic_records.append(light_record)
        light_row = comparison_summary_row(
            light_record["label"],
            light_record["summary"],
            run_dir=light_record["run_dir"],
            summary_path=light_record["summary_path"],
        )
        light_row["condition"] = row["condition"]
        light_row["training_seed"] = row["training_seed"]
        light_traffic_rows.append(light_row)
    light_traffic_compatibility = None
    light_traffic_aggregates = []
    if light_traffic_records:
        if len(light_traffic_records) != len(learned_records):
            raise ValueError(
                "Light-traffic results exist for only some Phase-2 runs. Evaluate every "
                "selected direct and curriculum run before comparing the stress test."
            )
        light_traffic_compatibility = _compatibility(light_traffic_records, strict=True)
        light_traffic_aggregates = _aggregate_phase2(light_traffic_rows)
        pd.DataFrame(light_traffic_rows).to_csv(
            runs_dir / "phase2_light_traffic_seed_results.csv",
            index=False,
        )
        pd.DataFrame(light_traffic_aggregates).to_csv(
            runs_dir / "phase2_light_traffic_comparison.csv",
            index=False,
        )
        plot_comparison_rows(
            light_traffic_aggregates,
            runs_dir / "phase2_light_traffic_comparison.png",
            title="Phase-2 zero-shot light-traffic stress test",
        )

    aggregates = _aggregate_phase2(rows)
    seed_csv = runs_dir / "phase2_seed_results.csv"
    aggregate_csv = runs_dir / "phase2_comparison.csv"
    json_path = runs_dir / "phase2_comparison.json"
    plot_path = runs_dir / "phase2_comparison.png"
    pd.DataFrame(rows).to_csv(seed_csv, index=False)
    pd.DataFrame(aggregates).to_csv(aggregate_csv, index=False)
    plot_comparison_rows(
        aggregates,
        plot_path,
        title="Phase-2 direct versus curriculum SAC",
    )
    training_plot = plot_phase1_training_returns(
        training_dirs,
        runs_dir / "phase2_training_returns.png",
        title="Phase-2 SAC training returns",
    )
    report_data = _write_phase2_report_data(aggregates)
    write_json(
        {
            "generated_at_utc": utc_timestamp(),
            "research_question": (
                "Does staged road-geometry curriculum improve unseen procedural-road "
                "generalization over equal-budget direct SAC training?"
            ),
            "strict_compatibility": strict_compatibility,
            "baseline_task_compatibility": baseline_compatibility,
            "light_traffic_strict_compatibility": light_traffic_compatibility,
            "training_seeds": args.seeds,
            "learned_runs": rows,
            "aggregates": aggregates,
            "baselines": _rows(baseline_records),
            "light_traffic_runs": light_traffic_rows,
            "light_traffic_aggregates": light_traffic_aggregates,
        },
        json_path,
    )
    logger.info("Phase-2 seed results written: %s", seed_csv)
    logger.info("Phase-2 aggregate comparison written: %s", aggregate_csv)
    logger.info("Phase-2 comparison plot written: %s", plot_path)
    if report_data is not None:
        logger.info("Phase-2 LaTeX result macros written: %s", report_data)
    if training_plot is not None:
        logger.info("Phase-2 training plot written: %s", training_plot)


def _run_manual(args, logger):
    records = []
    for summary in args.summaries:
        summary_path = Path(summary)
        run_dir = summary_path.parents[1]
        records.append(_summary_record(run_dir.name, run_dir, summary_path=summary_path))
    _compatibility(records, strict=True)
    rows = _rows(records)
    output = Path(args.output or "runs/comparison_eval_summary.png")
    plot_comparison_rows(rows, output)
    logger.info("Strictly compatible comparison plot written: %s", output)


def main():
    args = parse_args()
    if args.phase1:
        log_path = Path(args.runs_dir) / "phase1_compare.log"
    elif args.phase2:
        log_path = Path(args.runs_dir) / "phase2_compare.log"
    elif args.traffic_extension:
        log_path = Path(args.runs_dir) / "traffic_extension_compare.log"
    else:
        output = Path(args.output or "runs/comparison_eval_summary.png")
        log_path = output.with_suffix(".log")
    logger = setup_logging(log_path)

    try:
        log_system_info(logger)
        logger.debug("Arguments: %s", vars(args))
        if args.phase1:
            _run_phase1(args, logger)
        elif args.phase2:
            _run_phase2(args, logger)
        elif args.traffic_extension:
            _run_traffic_extension(args, logger)
        else:
            _run_manual(args, logger)
    except Exception:
        logger.exception("Run comparison failed.")
        raise


if __name__ == "__main__":
    main()
