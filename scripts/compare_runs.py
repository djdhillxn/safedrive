"""Compare compatible SafeDrive evaluation runs."""

import argparse
from pathlib import Path

import pandas as pd

from saferl_drive.config import (
    fingerprint_differences,
    load_yaml,
    make_experiment_fingerprint,
)
from saferl_drive.evaluation import comparison_summary_row
from saferl_drive.utils import (
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
    parser.add_argument("--output", default=None, help="Manual comparison PNG output path.")
    parser.add_argument("--runs-dir", default="runs", help="Runs directory.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Phase-2 algorithm seeds to include.",
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
        else:
            _run_manual(args, logger)
    except Exception:
        logger.exception("Run comparison failed.")
        raise


if __name__ == "__main__":
    main()
