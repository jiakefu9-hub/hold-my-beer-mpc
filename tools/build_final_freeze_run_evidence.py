#!/usr/bin/env python3
"""Build and verify the two-run final-freeze evidence package.

The two source directories are frozen constants.  This tool never scans for a
"latest" run, never deletes source data, and never copies trajectory, metrics,
raw samples, previews, or video.  It reuses the strict environment, timing,
handoff, safety, and quality readers from ``build_final_freeze_evidence.py``.

Generated manifests use repository-relative source paths and package-relative
output paths so the committed package remains relocatable after a fresh clone.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

# Support both ``python tools/...py`` and module/test imports.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.build_final_freeze_evidence import (
    EXPECTED_TEMPLATE_MANIFEST_SHA256,
    EXPECTED_TEMPLATE_SHA256,
    RUN_EVIDENCE_FILES,
    TEMPLATE_ROOT,
    EvidenceError,
    aggregate_perf_columns,
    aggregate_predictor_timing,
    aggregate_quality,
    load_json,
    predictor_timing_from_summary,
    read_control_quality,
    read_perf_intervals,
    require,
    run_safety,
    sha256_file,
    validate_formal_environment,
    validate_handoff,
    write_json,
)


FINAL_RUN_SOURCE_ROOT = Path("evaluation/t2_full_task_closed_loop")
DEFAULT_OUTPUT = Path(
    "evaluation_summary/full_task_template_v2_final_freeze/final_runs"
)

# These are the only accepted inputs.  Do not replace them with discovery or a
# "latest" selector: the evidence contract is tied to these exact freeze runs.
FINAL_RUNS = (
    ("20260815_231454_final_freeze", "nominal"),
    (
        "20260815_231555_final_freeze_heldout_pair_02_minus",
        "heldout_pair_02_minus",
    ),
)

# The same thirteen-item lightweight whitelist used for the earlier controlled
# six-run pack is retained for each final run.
FINAL_RUN_EVIDENCE_FILES = RUN_EVIDENCE_FILES

# Each run gets its own representative plots.  Large numerical arrays and
# previews remain derivation-only sources and are never copied.
FINAL_RUN_PLOTS = (
    "metrics.png",
    "startup_pd_handoff_transition.png",
    "base_disturbance_interval_template_prediction_vs_actual.png",
    "mpc_end_effector_task_prediction_vs_actual.png",
    "ddq_tracking.png",
    "full_task_xy_trajectory.png",
    "full_task_planned_runtime_commands.png",
    "full_task_event_grid.png",
    "full_task_tail_coverage.png",
    "full_task_torso_rpy.png",
    "heading_control.png",
)

DERIVATION_ONLY_FILES = (
    "metrics.npz",
    "trajectory.npz",
)

AGGREGATE_JSON = "final_freeze_runs.json"
AGGREGATE_CSV = "final_freeze_runs.csv"
FILE_MANIFEST = "final_freeze_file_manifest.json"
SCHEMA_VERSION = "full_task_template_v2_final_freeze_two_run_evidence_v1"
FILE_MANIFEST_SCHEMA_VERSION = (
    "full_task_template_v2_final_freeze_two_run_files_v1"
)


def repository_relative(repository: Path, path: Path) -> str:
    """Return a relocatable path and reject anything outside the repository."""

    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"path escapes repository: {path}") from exc


def copy_relative(
    *,
    source: Path,
    destination: Path,
    repository: Path,
    output: Path,
    category: str,
    run_id: str,
) -> dict[str, Any]:
    require(source.is_file(), f"missing evidence source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha256_file(source)
    require(digest == sha256_file(destination), f"copy checksum mismatch: {source}")
    return {
        "category": category,
        "run_id": run_id,
        "source_repository_path": repository_relative(repository, source),
        "output_package_path": destination.resolve().relative_to(output.resolve()).as_posix(),
        "bytes": int(destination.stat().st_size),
        "sha256": digest,
    }


def validate_template(repository: Path) -> dict[str, Any]:
    template = repository / TEMPLATE_ROOT / "full_task_template.npz"
    manifest_path = repository / TEMPLATE_ROOT / "full_task_template_manifest.json"
    require(template.is_file(), f"missing frozen template: {template}")
    require(manifest_path.is_file(), f"missing frozen template manifest: {manifest_path}")
    template_sha = sha256_file(template)
    manifest_sha = sha256_file(manifest_path)
    require(template_sha == EXPECTED_TEMPLATE_SHA256, "frozen template checksum mismatch")
    require(
        manifest_sha == EXPECTED_TEMPLATE_MANIFEST_SHA256,
        "frozen template manifest checksum mismatch",
    )
    manifest = load_json(manifest_path)
    require(
        manifest.get("template_schema_version") == "full_task_template_v2",
        "frozen template schema mismatch",
    )
    require(
        manifest["collection"]["protocol"]["version"] == "full_task_direct_step_v1",
        "frozen template protocol mismatch",
    )
    validation = manifest["template_validation"]
    require(validation["anchor_count"] == 1334, "frozen template anchor count")
    require(validation["horizon"] == 9, "frozen template horizon")
    require(
        validation["heading_frame_version"] == "full_task_continuous_heading_v2",
        "frozen template heading frame",
    )
    return {
        "template_repository_path": repository_relative(repository, template),
        "template_sha256": template_sha,
        "manifest_repository_path": repository_relative(repository, manifest_path),
        "manifest_sha256": manifest_sha,
        "schema_version": manifest["template_schema_version"],
        "protocol_version": manifest["collection"]["protocol"]["version"],
        "heading_frame_version": validation["heading_frame_version"],
        "anchor_count": int(validation["anchor_count"]),
        "horizon": int(validation["horizon"]),
    }


def validate_protocol_and_template_use(
    *,
    manifest: dict[str, Any],
    smoke: dict[str, Any],
    template: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    predictor = manifest["predictor"]
    protocol = manifest["protocol"]
    require(predictor["predictor_type"] == "full_task_template", f"{run_id}: predictor")
    require(predictor["sha256"] == template["template_sha256"], f"{run_id}: template hash")
    require(
        predictor["manifest_sha256"] == template["manifest_sha256"],
        f"{run_id}: template manifest hash",
    )
    require(
        predictor["template_schema_version"] == template["schema_version"],
        f"{run_id}: template schema",
    )
    require(
        predictor["protocol_version"] == template["protocol_version"],
        f"{run_id}: predictor protocol",
    )
    require(
        predictor["heading_definition"] == template["heading_frame_version"],
        f"{run_id}: continuous-H definition",
    )
    require(
        predictor["anchor_mode"] == "exact_absolute_task_time_6ms_no_interpolation",
        f"{run_id}: exact anchor lookup",
    )
    require(predictor["slow_bias_enabled"] is False, f"{run_id}: slow bias")
    require(predictor["template_smoothing"] == "none", f"{run_id}: smoothing")
    require(protocol["version"] == "full_task_direct_step_v1", f"{run_id}: protocol")
    require(abs(float(protocol["task_epoch_origin_simulation_time"])) <= 1e-12, f"{run_id}: epoch")
    require(abs(float(protocol["physics_dt"]) - 0.002) <= 1e-12, f"{run_id}: physics dt")
    require(abs(float(protocol["mpc_dt"]) - 0.006) <= 1e-12, f"{run_id}: MPC dt")
    require(abs(float(protocol["policy_dt"]) - 0.020) <= 1e-12, f"{run_id}: policy dt")
    require(abs(float(protocol["stop_time"]) - 6.4) <= 1e-12, f"{run_id}: stop")
    require(protocol["headline_interval"] == "[0.0,8.0)", f"{run_id}: headline")
    require(float(protocol["record_end"]) >= 8.06, f"{run_id}: record tail")
    require(int(protocol["horizon"]) == 9, f"{run_id}: horizon")
    require(abs(float(protocol["horizon_duration"]) - 0.054) <= 1e-12, f"{run_id}: horizon duration")

    require(smoke["strict_pre_step"] is True, f"{run_id}: strict pre-step")
    require(smoke["heading_enabled"] is True, f"{run_id}: heading disabled")
    require(smoke["direct_step_effective"] is True, f"{run_id}: direct step")
    require(smoke["tail_complete"] is True, f"{run_id}: incomplete tail")
    require(int(smoke["headline_anchor_count"]) == 1334, f"{run_id}: headline anchors")
    require(int(smoke["raw_sample_count"]) == 4030, f"{run_id}: raw sample count")
    require(float(smoke["last_raw_time"]) >= 8.058 - 2e-12, f"{run_id}: last raw time")
    require(abs(float(smoke["last_horizon_node"]) - 8.052) <= 2e-12, f"{run_id}: last horizon node")
    return {
        "direct_step_effective": True,
        "heading_enabled": True,
        "strict_pre_step": True,
        "tail_complete": True,
        "raw_sample_count": int(smoke["raw_sample_count"]),
        "headline_anchor_count": int(smoke["headline_anchor_count"]),
        "last_raw_time_s": float(smoke["last_raw_time"]),
        "last_horizon_node_s": float(smoke["last_horizon_node"]),
    }


def validate_final_safety(
    *,
    quality: dict[str, Any],
    right_arm: dict[str, Any],
    smoke: dict[str, Any],
    handoff_summary: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    safety = run_safety(quality=quality, right_arm=right_arm, smoke=smoke)
    branches = right_arm["right_arm_execution_branches"]
    whole_trace = handoff_summary["whole_trace"]
    require(int(smoke["qp_failure_count"]) == 0, f"{run_id}: QP failure")
    require(int(smoke["qp_fallback_count"]) == 0, f"{run_id}: QP fallback")
    require(
        quality["qp"]["success_count"] == quality["qp"]["update_count"],
        f"{run_id}: not every headline QP update succeeded",
    )
    require(quality["qp"]["fallback_count"] == 0, f"{run_id}: trajectory QP fallback")
    require(
        quality["predictor_fallback"]["headline_count"] == 0,
        f"{run_id}: predictor fallback",
    )
    require(int(whole_trace["predictor_fallback_count"]) == 0, f"{run_id}: trace predictor fallback")
    require(int(branches["final_output_uncertified_count"]) == 0, f"{run_id}: uncertified output")
    require(int(branches["no_safe_torque_count"]) == 0, f"{run_id}: NO_SAFE_TORQUE")
    require(int(branches["final_unsafe_count"]) == 0, f"{run_id}: final unsafe")
    require(int(whole_trace["final_unsafe_count"]) == 0, f"{run_id}: trace final unsafe")
    require(int(whole_trace["no_safe_torque_count"]) == 0, f"{run_id}: trace no-safe torque")
    require(smoke["fallen"] is False, f"{run_id}: fall")
    require(int(smoke["nan_inf_count"]) == 0, f"{run_id}: NaN/Inf")
    require(
        int(smoke["runtime_executor_nonzero_flag_count"]) == 0,
        f"{run_id}: runtime executor flag",
    )
    require(safety["certified_control_gate_pass"] is True, f"{run_id}: certified control gate")

    certified_fallbacks = {
        "legacy_runtime_mapping_safety_fallback_count": int(
            smoke["runtime_mapping_safety_fallback_count"]
        ),
        "second_pass_triggered_count": int(branches["second_pass_triggered_count"]),
        "second_pass_accepted_count": int(branches["second_pass_accepted_count"]),
        "rescue_used_count": int(branches["rescue_used_count"]),
        "one_rescue_pass_count": int(branches["one_rescue_pass_count"]),
        "two_rescue_pass_count": int(branches["two_rescue_pass_count"]),
        "rescue_succeeded_before_hold_count": int(
            branches["rescue_succeeded_before_hold_count"]
        ),
        "hold_last_succeeded_count": int(branches["hold_last_succeeded_count"]),
        "safe_hold_used_count": int(branches["safe_hold_used_count"]),
        "safety_line_search_used_count": int(
            branches["safety_line_search_used_count"]
        ),
        "final_output_uncertified_count": int(
            branches["final_output_uncertified_count"]
        ),
    }
    require(
        certified_fallbacks["rescue_used_count"]
        == certified_fallbacks["rescue_succeeded_before_hold_count"]
        + certified_fallbacks["hold_last_succeeded_count"],
        f"{run_id}: rescue outcomes are not fully accounted for",
    )
    safety["qp_failure_count"] = int(smoke["qp_failure_count"])
    safety["certified_fallbacks"] = certified_fallbacks
    return safety


def csv_row(run: dict[str, Any]) -> dict[str, Any]:
    timing = run["timing"]
    complete = timing["complete_6ms_ms"]
    components = timing["components_ms"]
    quality = run["control_quality"]
    safety = run["safety"]
    fallback = safety["certified_fallbacks"]
    return {
        "run_id": run["run_id"],
        "scenario": run["scenario"],
        "complete_6ms_count": complete["count"],
        "complete_6ms_mean_ms": complete["mean"],
        "complete_6ms_p95_ms": complete["p95"],
        "complete_6ms_p99_ms": complete["p99"],
        "complete_6ms_max_ms": complete["max"],
        "complete_6ms_overrun_count": complete["overrun_count"],
        "mpc_policy_mean_ms": components["mpc_policy_update"]["mean"],
        "ddq_total_mean_ms": components["all_ddq_to_torque_calls"]["mean"],
        "ddq_call_1_mean_ms": components["ddq_call_1"]["mean"],
        "ddq_call_2_mean_ms": components["ddq_call_2"]["mean"],
        "predictor_mean_ms": timing["predictor_time_ms"]["mean"],
        "other_path_mean_ms": components["other_right_arm_path"]["mean"],
        "tilt_rms_rad": quality["tilt_angle_rad"]["rms"],
        "tilt_p95_rad": quality["tilt_angle_rad"]["p95"],
        "tilt_max_rad": quality["tilt_angle_rad"]["max"],
        "position_rms_m": quality["position_error_norm_m"]["rms"],
        "position_p95_m": quality["position_error_norm_m"]["p95"],
        "position_max_m": quality["position_error_norm_m"]["max"],
        "ee_acc_rms_m_s2": quality["right_ee_linear_acceleration_norm_m_s2"]["rms"],
        "ee_alpha_rms_rad_s2": quality["right_ee_angular_acceleration_norm_rad_s2"]["rms"],
        "xy_displacement_m": run["xy"]["displacement_m"],
        "xy_arc_length_m": run["xy"]["arc_length_m"],
        "handoff_tau_jump_l2_nm": run["handoff"]["tau_jump_l2_nm"],
        "legacy_smoke_passed": run["source_smoke_status"]["smoke_passed"],
        "legacy_mapping_safety_fallback_count": fallback[
            "legacy_runtime_mapping_safety_fallback_count"
        ],
        "rescue_used_count": fallback["rescue_used_count"],
        "rescue_succeeded_before_hold_count": fallback[
            "rescue_succeeded_before_hold_count"
        ],
        "hold_last_succeeded_count": fallback["hold_last_succeeded_count"],
        "safe_hold_used_count": fallback["safe_hold_used_count"],
        "final_output_uncertified_count": safety[
            "mapper_final_output_uncertified_count"
        ],
        "no_safe_torque_count": safety["mapper_no_safe_torque_count"],
        "final_unsafe_count": safety["mapper_final_unsafe_count"],
        "qp_failure_count": safety["qp_failure_count"],
        "qp_fallback_count": safety["qp_fallback_count"],
        "predictor_fallback_count": safety["predictor_fallback_count"],
        "fallen": safety["fallen"],
        "nan_inf_count": safety["nan_inf_count"],
        "formal_environment_pass": run["formal_environment"]["passed"],
        "certified_control_gate_pass": safety["certified_control_gate_pass"],
    }


def build(repository: Path, output: Path) -> dict[str, Any]:
    repository = repository.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    template = validate_template(repository)
    copied_files: list[dict[str, Any]] = []
    derived_sources: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    perf_arrays: list[dict[str, np.ndarray]] = []
    quality_arrays: list[dict[str, np.ndarray]] = []

    for run_id, scenario in FINAL_RUNS:
        source = repository / FINAL_RUN_SOURCE_ROOT / run_id
        require(source.is_dir(), f"missing fixed final run source: {source}")
        destination = output / "runs" / run_id
        for filename in FINAL_RUN_EVIDENCE_FILES:
            copied_files.append(
                copy_relative(
                    source=source / filename,
                    destination=destination / filename,
                    repository=repository,
                    output=output,
                    category="final_run_13_file_whitelist",
                    run_id=run_id,
                )
            )
        for filename in FINAL_RUN_PLOTS:
            copied_files.append(
                copy_relative(
                    source=source / filename,
                    destination=destination / "plots" / filename,
                    repository=repository,
                    output=output,
                    category="final_run_representative_plot",
                    run_id=run_id,
                )
            )

        for filename in DERIVATION_ONLY_FILES:
            path = source / filename
            require(path.is_file(), f"missing derivation source: {path}")
            derived_sources.append(
                {
                    "run_id": run_id,
                    "purpose": "recompute full-headline control quality",
                    "source_repository_path": repository_relative(repository, path),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                    "copied_to_evidence_package": False,
                }
            )

        metadata = load_json(source / "run_metadata.json")
        preflight = load_json(source / "formal_full_task_runtime_preflight.json")
        perf_summary = load_json(source / "perf_summary.json")
        handoff_summary = load_json(source / "startup_pd_handoff_summary.json")
        right_arm = load_json(source / "right_arm_diagnostics.json")
        smoke = load_json(source / "full_task_smoke_summary.json")
        manifest = load_json(source / "full_task_manifest.json")

        formal = validate_formal_environment(preflight, metadata, run_id)
        timing, perf_columns = read_perf_intervals(source / "perf_intervals.csv")
        timing["predictor_time_ms"] = predictor_timing_from_summary(perf_summary)
        require(
            timing["complete_6ms_ms"]["count"] == 1329,
            f"{run_id}: complete 6 ms interval count",
        )
        require(
            timing["complete_6ms_ms"]["overrun_count"] == 0,
            f"{run_id}: complete 6 ms overrun",
        )
        quality, arrays = read_control_quality(
            source / "metrics.npz", source / "trajectory.npz"
        )
        protocol = validate_protocol_and_template_use(
            manifest=manifest,
            smoke=smoke,
            template=template,
            run_id=run_id,
        )
        handoff = validate_handoff(handoff_summary, run_id)
        safety = validate_final_safety(
            quality=quality,
            right_arm=right_arm,
            smoke=smoke,
            handoff_summary=handoff_summary,
            run_id=run_id,
        )

        reports.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "source_run_repository_path": repository_relative(repository, source),
                "headline": "[0.0,8.0)",
                "formal_environment": formal,
                "protocol_and_raw_contract": protocol,
                "handoff": handoff,
                "timing": timing,
                "control_quality": quality,
                "safety": safety,
                "xy": {
                    "displacement_m": float(smoke["xy_displacement_m"]),
                    "arc_length_m": float(smoke["xy_arc_length_m"]),
                },
                "source_smoke_status": {
                    "status": smoke["status"],
                    "smoke_passed": bool(smoke["smoke_passed"]),
                    "is_final_freeze_acceptance_gate": False,
                    "legacy_runtime_mapping_safety_fallback_count": int(
                        smoke["runtime_mapping_safety_fallback_count"]
                    ),
                    "note": (
                        "Preserved verbatim. The legacy smoke gate treats a mapping "
                        "safety fallback as failure; final-freeze acceptance instead "
                        "requires every selected output to be certified and requires "
                        "zero final_unsafe, NO_SAFE_TORQUE, QP/predictor fallback, "
                        "fall, NaN/Inf, and complete-interval overrun."
                    ),
                },
            }
        )
        perf_arrays.append(perf_columns)
        quality_arrays.append(arrays)

    timing_aggregate = aggregate_perf_columns(perf_arrays)
    timing_aggregate["predictor_time_ms"] = aggregate_predictor_timing(
        [report["timing"]["predictor_time_ms"] for report in reports]
    )
    safety_total_names = (
        "mapper_execution_call_count",
        "mapper_rescue_used_count",
        "mapper_hold_last_succeeded_count",
        "mapper_safe_hold_used_count",
        "mapper_safety_line_search_used_count",
        "mapper_final_output_uncertified_count",
        "mapper_no_safe_torque_count",
        "mapper_final_unsafe_count",
        "qp_failure_count",
        "qp_fallback_count",
        "predictor_fallback_count",
        "nan_inf_count",
    )
    safety_totals = {
        name: sum(int(report["safety"][name]) for report in reports)
        for name in safety_total_names
    }
    require(timing_aggregate["complete_6ms_ms"]["count"] == 2658, "two-run interval count")
    require(timing_aggregate["complete_6ms_ms"]["overrun_count"] == 0, "two-run overrun")
    require(
        all(value == 0 for name, value in safety_totals.items() if name not in {
            "mapper_execution_call_count",
            "mapper_rescue_used_count",
            "mapper_hold_last_succeeded_count",
            "mapper_safe_hold_used_count",
            "mapper_safety_line_search_used_count",
        }),
        "two-run safety total gate",
    )

    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "scope": {
            "control_candidate": "MPC + FullTaskTemplatePredictor v2",
            "source_selection": "two explicit run directories; no latest-run scan",
            "headline": "[0.0,8.0)",
            "startup_pd_prefix": "[0.0,0.024) is included in the headline",
            "mpc_handoff": "simulation/task time 0.024 s; absolute template anchor 4",
            "hardware_claim": "controlled MuJoCo simulation only; not hardware real-time evidence",
        },
        "frozen_template": template,
        "runs": reports,
        "two_run_aggregate": {
            "run_count": 2,
            "complete_interval_count": 2658,
            "timing": timing_aggregate,
            "control_quality": aggregate_quality(quality_arrays),
            "safety_totals": safety_totals,
            "all_environment_preflights_pass": True,
            "all_certified_control_gates_pass": True,
            "all_legacy_smoke_status_fields_preserved": True,
            "legacy_smoke_status_is_acceptance_gate": False,
        },
        "provenance": {
            "quality": "recomputed from metrics.npz and trajectory.npz; arrays are hashed but not copied",
            "timing": "recomputed from all 1329 perf_intervals.csv rows per run",
            "copied_file_contract": "13-item JSON/CSV whitelist and per-run representative plots",
        },
    }
    aggregate_path = output / AGGREGATE_JSON
    write_json(aggregate_path, aggregate)

    csv_path = output / AGGREGATE_CSV
    rows = [csv_row(report) for report in reports]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    generated_files = [
        {
            "category": category,
            "output_package_path": path.relative_to(output).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path, category in (
            (aggregate_path, "generated_two_run_aggregate_json"),
            (csv_path, "generated_two_run_metrics_csv"),
        )
    ]
    builder = Path(__file__).resolve()
    file_manifest = {
        "schema_version": FILE_MANIFEST_SCHEMA_VERSION,
        "status": "PASS",
        "builder": {
            "repository_path": repository_relative(repository, builder),
            "sha256": sha256_file(builder),
        },
        "source_selection": {
            "root_repository_path": FINAL_RUN_SOURCE_ROOT.as_posix(),
            "run_ids": [run_id for run_id, _ in FINAL_RUNS],
            "selection_method": "fixed constants; no directory scan",
        },
        "output_repository_path": repository_relative(repository, output),
        "selection": {
            "whitelist_file_count_per_run": len(FINAL_RUN_EVIDENCE_FILES),
            "whitelist_files": list(FINAL_RUN_EVIDENCE_FILES),
            "representative_plot_count_per_run": len(FINAL_RUN_PLOTS),
            "representative_plots": list(FINAL_RUN_PLOTS),
            "excluded_large_artifacts": [
                "trajectory.npz",
                "metrics.npz",
                "full_task_nominal_raw.npz",
                "control_preview.csv",
                "metrics_preview.csv",
                "mpc_diagnostics_preview.csv",
                "mpc_tracking_preview.csv",
                "startup_pd_handoff_trace.npz",
                "video",
            ],
        },
        "frozen_template": template,
        "copied_files": copied_files,
        "derived_only_source_files": derived_sources,
        "generated_files": generated_files,
        "validation": {
            "all_copy_checksums_match": True,
            "all_two_runtime_preflights_pass": True,
            "all_two_certified_control_gates_pass": True,
            "all_2658_complete_intervals_within_6ms": True,
            "all_source_smoke_status_fields_preserved_but_not_gating": True,
            "large_arrays_and_video_not_copied": True,
        },
    }
    manifest_path = output / FILE_MANIFEST
    write_json(manifest_path, file_manifest)
    verify(repository, output, require_sources=True)
    return {
        "status": "PASS",
        "output_repository_path": repository_relative(repository, output),
        "run_count": 2,
        "copied_file_count": len(copied_files),
        "complete_interval_count": 2658,
        "complete_interval_overrun_count": 0,
        "aggregate": repository_relative(repository, aggregate_path),
        "metrics_csv": repository_relative(repository, csv_path),
        "file_manifest": repository_relative(repository, manifest_path),
    }


def verify(
    repository: Path, output: Path, *, require_sources: bool = False
) -> dict[str, Any]:
    repository = repository.resolve()
    output = output.resolve()
    manifest = load_json(output / FILE_MANIFEST)
    aggregate = load_json(output / AGGREGATE_JSON)
    require(manifest["schema_version"] == FILE_MANIFEST_SCHEMA_VERSION, "file manifest schema")
    require(manifest["status"] == "PASS", "file manifest status")
    require(aggregate["schema_version"] == SCHEMA_VERSION, "aggregate schema")
    require(aggregate["status"] == "PASS", "aggregate status")
    require(
        manifest["source_selection"]["run_ids"]
        == [run_id for run_id, _ in FINAL_RUNS],
        "source run selection drift",
    )
    require(
        manifest["source_selection"]["selection_method"]
        == "fixed constants; no directory scan",
        "source selection method",
    )
    builder = repository / manifest["builder"]["repository_path"]
    require(builder.is_file(), f"missing evidence builder: {builder}")
    require(sha256_file(builder) == manifest["builder"]["sha256"], "builder hash drift")
    live_template = validate_template(repository)
    require(live_template == manifest["frozen_template"], "file manifest template drift")
    require(live_template == aggregate["frozen_template"], "aggregate template drift")

    copied = manifest["copied_files"]
    whitelist = [item for item in copied if item["category"] == "final_run_13_file_whitelist"]
    plots = [item for item in copied if item["category"] == "final_run_representative_plot"]
    require(len(whitelist) == 2 * len(FINAL_RUN_EVIDENCE_FILES), "13-file whitelist count")
    require(len(plots) == 2 * len(FINAL_RUN_PLOTS), "representative plot count")
    missing_sources = 0
    for item in copied:
        require("source_absolute_path" not in item, "absolute source path in package manifest")
        packaged = output / item["output_package_path"]
        source = repository / item["source_repository_path"]
        require(packaged.is_file(), f"missing packaged evidence: {packaged}")
        require(int(packaged.stat().st_size) == int(item["bytes"]), f"size drift: {packaged}")
        require(sha256_file(packaged) == item["sha256"], f"hash drift: {packaged}")
        if source.is_file():
            require(sha256_file(source) == item["sha256"], f"source drift: {source}")
        else:
            missing_sources += 1
            require(not require_sources, f"missing required source: {source}")

    for item in manifest["derived_only_source_files"]:
        source = repository / item["source_repository_path"]
        if source.is_file():
            require(int(source.stat().st_size) == int(item["bytes"]), f"derived size drift: {source}")
            require(sha256_file(source) == item["sha256"], f"derived hash drift: {source}")
        else:
            missing_sources += 1
            require(not require_sources, f"missing derivation source: {source}")

    for item in manifest["generated_files"]:
        generated = output / item["output_package_path"]
        require(generated.is_file(), f"missing generated evidence: {generated}")
        require(int(generated.stat().st_size) == int(item["bytes"]), f"generated size drift: {generated}")
        require(sha256_file(generated) == item["sha256"], f"generated hash drift: {generated}")

    for run_id, _ in FINAL_RUNS:
        packaged = output / "runs" / run_id
        metadata = load_json(packaged / "run_metadata.json")
        preflight = load_json(packaged / "formal_full_task_runtime_preflight.json")
        validate_formal_environment(preflight, metadata, run_id)
        timing, _ = read_perf_intervals(packaged / "perf_intervals.csv")
        require(timing["complete_6ms_ms"]["count"] == 1329, f"{run_id}: packaged interval count")
        require(timing["complete_6ms_ms"]["overrun_count"] == 0, f"{run_id}: packaged overrun")
        smoke = load_json(packaged / "full_task_smoke_summary.json")
        full_manifest = load_json(packaged / "full_task_manifest.json")
        validate_protocol_and_template_use(
            manifest=full_manifest,
            smoke=smoke,
            template=live_template,
            run_id=run_id,
        )
        validate_handoff(load_json(packaged / "startup_pd_handoff_summary.json"), run_id)
        require(smoke["fallen"] is False, f"{run_id}: packaged fall")
        require(int(smoke["nan_inf_count"]) == 0, f"{run_id}: packaged NaN/Inf")
        require(int(smoke["qp_failure_count"]) == 0, f"{run_id}: packaged QP failure")
        require(int(smoke["qp_fallback_count"]) == 0, f"{run_id}: packaged QP fallback")
        branches = load_json(packaged / "right_arm_diagnostics.json")[
            "right_arm_execution_branches"
        ]
        for name in (
            "final_output_uncertified_count",
            "no_safe_torque_count",
            "final_unsafe_count",
        ):
            require(int(branches[name]) == 0, f"{run_id}: packaged {name}")

    two_run = aggregate["two_run_aggregate"]
    require(two_run["run_count"] == 2, "aggregate run count")
    require(two_run["complete_interval_count"] == 2658, "aggregate interval count")
    require(
        two_run["timing"]["complete_6ms_ms"]["overrun_count"] == 0,
        "aggregate overrun",
    )
    require(two_run["all_environment_preflights_pass"] is True, "aggregate preflight")
    require(two_run["all_certified_control_gates_pass"] is True, "aggregate control gate")
    require(two_run["legacy_smoke_status_is_acceptance_gate"] is False, "legacy gate")
    zero_gate_names = (
        "mapper_final_output_uncertified_count",
        "mapper_no_safe_torque_count",
        "mapper_final_unsafe_count",
        "qp_failure_count",
        "qp_fallback_count",
        "predictor_fallback_count",
        "nan_inf_count",
    )
    require(
        all(int(two_run["safety_totals"][name]) == 0 for name in zero_gate_names),
        "aggregate safety totals",
    )
    validation = manifest["validation"]
    require(all(bool(value) for value in validation.values()), "file manifest validation")
    return {
        "status": "PASS",
        "output_repository_path": repository_relative(repository, output),
        "copied_file_count": len(copied),
        "generated_file_count": len(manifest["generated_files"]),
        "missing_source_count": missing_sources,
        "source_presence_required": require_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=str(REPOSITORY_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the existing package without copying or regenerating files",
    )
    parser.add_argument(
        "--require-sources",
        action="store_true",
        help="also require both original run directories and derivation arrays",
    )
    args = parser.parse_args()
    repository = Path(args.repository).expanduser().resolve()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else repository / DEFAULT_OUTPUT
    )
    result = (
        verify(repository, output, require_sources=args.require_sources)
        if args.verify_only
        else build(repository, output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
