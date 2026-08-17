#!/usr/bin/env python3
"""Build and verify the compact full-task-template-v2 freeze evidence pack.

The inputs are intentionally explicit.  This script never scans for a "latest"
run, never deletes a source artifact, and never copies the large trajectory/raw
arrays into the evidence pack.  Control-quality statistics are recomputed from
the source arrays before those arrays are archived outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_RUN_ROOT = Path("evaluation/fixed_startup_pd_cpu7_controlled")
TEMPLATE_ROOT = Path(
    "disturbance_template/data/full_task_template_v2/20260815_162850"
)
PARITY_SOURCE = Path(
    "evaluation/t2_full_task_template_online/"
    "20260815_163435_offline_online_parity/offline_online_parity.json"
)
DEFAULT_OUTPUT = Path(
    "evaluation_summary/full_task_template_v2_final_freeze"
)

EXPECTED_TEMPLATE_SHA256 = (
    "d4a0109adcff696936ef96160976161833ff9a7a7531e2e5d7ad9e50c10e17d4"
)
EXPECTED_TEMPLATE_MANIFEST_SHA256 = (
    "6b48ee196d1f7d923dde057d3c0fb0e182f08512a65402c4c39c5e070a3243c6"
)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

RUNS = (
    ("20260815_210609_nominal_rep01", "nominal", 1),
    ("20260815_210704_nominal_rep02", "nominal", 2),
    ("20260815_210733_nominal_rep03", "nominal", 3),
    ("20260815_210806_heldout_pair_02_minus_rep01", "heldout_pair_02_minus", 1),
    ("20260815_210837_heldout_pair_02_minus_rep02", "heldout_pair_02_minus", 2),
    ("20260815_210907_heldout_pair_02_minus_rep03", "heldout_pair_02_minus", 3),
)

# Exactly thirteen lightweight files are retained from each controlled run.
RUN_EVIDENCE_FILES = (
    "run_metadata.json",
    "formal_full_task_runtime_preflight.json",
    "perf_summary.json",
    "perf_intervals.csv",
    "startup_pd_handoff_summary.json",
    "right_arm_diagnostics.json",
    "mpc_diagnostics.json",
    "mpc_tracking_diagnostics.json",
    "full_task_smoke_summary.json",
    "full_task_manifest.json",
    "heading_control_diagnostics.json",
    "mpc_command_delay_summary.json",
    "summary.json",
)

REPRESENTATIVE_PLOTS = (
    "metrics.png",
    "startup_pd_handoff_transition.png",
    "base_disturbance_interval_template_prediction_vs_actual.png",
    "mpc_end_effector_task_prediction_vs_actual.png",
    "ddq_tracking.png",
    "full_task_xy_trajectory.png",
    "full_task_planned_runtime_commands.png",
)
REPRESENTATIVE_RUNS = {
    "nominal": RUNS[0][0],
    "heldout_pair_02_minus": RUNS[3][0],
}

TEMPLATE_EVIDENCE_FILES = (
    "full_task_template_manifest.json",
    "batch_result.json",
    "heldout_metrics.json",
    "v1_v2_comparison.json",
    "plots/build_xy_trajectories.png",
    "plots/causal_h_metrics.png",
    "plots/heldout_truth_vs_template.png",
    "plots/initial_perturbations_and_diversity.png",
    "plots/startup_0_2p4s_build_band.png",
    "plots/stop_6p2_8p0s_build_band.png",
    "plots/v1_v2_heldout_error_comparison.png",
    "plots/v1_v2_max_h_yaw_jump.png",
    "plots/v2_adjacent_h_yaw_jump.png",
    "plots/v2_all_h_yaw_trajectories.png",
)

AGGREGATE_JSON = "controlled_runs_aggregate.json"
AGGREGATE_CSV = "controlled_runs_metrics.csv"
FILE_MANIFEST = "evidence_file_manifest.json"


class EvidenceError(RuntimeError):
    """Raised when a frozen evidence contract is not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def as_float_array(values: Iterable[Any]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    require(result.ndim == 1 and result.size > 0, "expected a non-empty vector")
    require(bool(np.all(np.isfinite(result))), "non-finite evidence value")
    return result


def scalar_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    require(finite.size > 0, "cannot summarize an empty or non-finite metric")
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(finite**2))),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
    }


def vector_norm_values(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    require(values.ndim == 2, "expected a matrix of vector samples")
    valid = np.all(np.isfinite(values), axis=1)
    result = np.linalg.norm(values[valid], axis=1)
    require(result.size > 0, "no finite vector samples")
    return result


def relative_to_repository(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError:
        return str(path.resolve())


def copy_with_record(
    *,
    source: Path,
    destination: Path,
    repository: Path,
    output: Path,
    category: str,
) -> dict[str, Any]:
    require(source.is_file(), f"missing evidence source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    output_hash = sha256_file(destination)
    require(source_hash == output_hash, f"copy checksum mismatch: {source}")
    return {
        "category": category,
        "source_absolute_path": str(source.resolve()),
        "source_repository_path": relative_to_repository(source, repository),
        "output_repository_path": relative_to_repository(destination, repository),
        "output_package_path": str(destination.resolve().relative_to(output.resolve())),
        "bytes": int(destination.stat().st_size),
        "sha256": output_hash,
    }


def validate_formal_environment(
    preflight: dict[str, Any], metadata: dict[str, Any], run_id: str
) -> dict[str, Any]:
    prefix = f"{run_id}: formal runtime preflight"
    require(preflight.get("passed") is True, f"{prefix} did not pass")
    require(preflight.get("launcher") == "disturbance_lab_run_sh", f"{prefix} launcher")
    require(str(preflight.get("requested_control_cpu")) == "7", f"{prefix} CPU")
    require(preflight.get("parent_cpu_affinity") == [7], f"{prefix} parent affinity")
    require(preflight.get("worker_cpu_affinity") == [7], f"{prefix} worker affinity")
    thread_environment = preflight.get("thread_environment", {})
    require(
        all(str(thread_environment.get(name)) == "1" for name in THREAD_VARIABLES),
        f"{prefix} numerical thread environment",
    )
    require(preflight.get("torch_num_threads") == 1, f"{prefix} torch intra-op")
    require(preflight.get("torch_num_interop_threads") == 1, f"{prefix} torch inter-op")
    require(preflight.get("gc_disabled_during_control") is True, f"{prefix} GC")
    require(preflight.get("dynamic_arming_enabled") is False, f"{prefix} dynamic arming")
    require(
        abs(float(preflight.get("startup_pd_duration_s")) - 0.024) <= 1e-12,
        f"{prefix} startup-PD duration",
    )
    require(preflight.get("mpc_handoff_anchor_index") == 4, f"{prefix} handoff anchor")

    runtime = metadata["runtime_timing_environment"]
    require(runtime["cpu_affinity"] == [7], f"{run_id}: metadata affinity")
    require(
        runtime["scheduler"]["right_arm_worker"]["cpu_affinity"] == [7],
        f"{run_id}: metadata worker affinity",
    )
    require(
        all(str(runtime["thread_environment"].get(name)) == "1" for name in THREAD_VARIABLES),
        f"{run_id}: metadata numerical thread environment",
    )
    require(runtime["torch_num_threads"] == 1, f"{run_id}: metadata torch intra-op")
    require(
        runtime["torch_num_interop_threads"] == 1,
        f"{run_id}: metadata torch inter-op",
    )
    require(runtime["gc_disabled_during_control_loop"] is True, f"{run_id}: metadata GC")
    handoff = metadata["fixed_startup_pd_handoff"]
    require(handoff["dynamic_arming_enabled"] is False, f"{run_id}: metadata arming")
    require(abs(float(handoff["duration_s"]) - 0.024) <= 1e-12, f"{run_id}: metadata duration")
    require(handoff["handoff_anchor_index"] == 4, f"{run_id}: metadata anchor")
    require(handoff["handoff_resets_any_clock"] is False, f"{run_id}: clock reset")
    require(
        metadata["robot_model_backends"]["right_arm_execution_runtime"] == "process",
        f"{run_id}: process runtime",
    )

    return {
        "passed": True,
        "launcher": preflight["launcher"],
        "requested_control_cpu": 7,
        "parent_cpu_affinity": preflight["parent_cpu_affinity"],
        "worker_cpu_affinity": preflight["worker_cpu_affinity"],
        "thread_environment": {name: 1 for name in THREAD_VARIABLES},
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "gc_disabled_during_control": True,
        "dynamic_arming_enabled": False,
        "startup_pd_duration_s": 0.024,
        "mpc_handoff_anchor_index": 4,
        "right_arm_runtime_mode": "process",
    }


def read_perf_intervals(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 1329, f"{path}: expected 1329 complete intervals")
    required = {
        "simulation_start_time_s",
        "simulation_end_time_s",
        "complete_interval_ms",
        "budget_ms",
        "overrun",
        "mpc_policy_update_ms",
        "ddq_call_count",
        "ddq_total_ms",
        "cpp_executor_bridge_ms",
        "other_right_arm_path_ms",
        "ddq_call_1_ms",
        "ddq_call_2_ms",
        "ddq_call_3_ms",
    }
    require(required.issubset(rows[0]), f"{path}: unexpected perf schema")

    columns = {
        name: as_float_array(float(row[name]) for row in rows)
        for name in required - {"overrun"}
    }
    starts = columns["simulation_start_time_s"]
    ends = columns["simulation_end_time_s"]
    require(abs(float(starts[0]) - 0.024) <= 1e-12, f"{path}: first MPC anchor")
    require(abs(float(ends[-1]) - 7.998) <= 2e-12, f"{path}: last complete interval")
    require(bool(np.all(np.diff(starts) > 0.0)), f"{path}: non-monotonic starts")
    require(bool(np.allclose(columns["budget_ms"], 6.0)), f"{path}: timing budget")
    require(bool(np.all(columns["ddq_call_count"] == 2.0)), f"{path}: DDQ call count")
    require(bool(np.all(columns["ddq_call_3_ms"] == 0.0)), f"{path}: third DDQ call")

    recomposed = (
        columns["mpc_policy_update_ms"]
        + columns["ddq_total_ms"]
        + columns["cpp_executor_bridge_ms"]
        + columns["other_right_arm_path_ms"]
    )
    require(
        bool(np.allclose(recomposed, columns["complete_interval_ms"], atol=2e-9, rtol=0.0)),
        f"{path}: complete interval composition",
    )
    ddq_sum = (
        columns["ddq_call_1_ms"]
        + columns["ddq_call_2_ms"]
        + columns["ddq_call_3_ms"]
    )
    require(
        bool(np.allclose(ddq_sum, columns["ddq_total_ms"], atol=2e-9, rtol=0.0)),
        f"{path}: DDQ timing composition",
    )
    derived_overrun = columns["complete_interval_ms"] > columns["budget_ms"]
    recorded_overrun = np.asarray(
        [str(row["overrun"]).strip().lower() in {"true", "1", "1.0"} for row in rows],
        dtype=bool,
    )
    require(bool(np.array_equal(derived_overrun, recorded_overrun)), f"{path}: overrun flags")

    summary = {
        "definition": "complete 6 ms right-arm intervals in [0,8)",
        "budget_ms": 6.0,
        "first_interval_start_simulation_time_s": float(starts[0]),
        "last_interval_end_simulation_time_s": float(ends[-1]),
        "complete_6ms_ms": {
            **scalar_stats(columns["complete_interval_ms"]),
            "overrun_count": int(np.count_nonzero(derived_overrun)),
            "overrun_fraction": float(np.mean(derived_overrun)),
        },
        "components_ms": {
            "mpc_policy_update": scalar_stats(columns["mpc_policy_update_ms"]),
            "all_ddq_to_torque_calls": scalar_stats(columns["ddq_total_ms"]),
            "ddq_call_1": scalar_stats(columns["ddq_call_1_ms"]),
            "ddq_call_2": scalar_stats(columns["ddq_call_2_ms"]),
            "cpp_executor_bridge": scalar_stats(columns["cpp_executor_bridge_ms"]),
            "other_right_arm_path": scalar_stats(columns["other_right_arm_path_ms"]),
        },
        "max_abs_composition_error_ms": float(
            np.max(np.abs(recomposed - columns["complete_interval_ms"]))
        ),
        "max_abs_ddq_sum_error_ms": float(
            np.max(np.abs(ddq_sum - columns["ddq_total_ms"]))
        ),
    }
    return summary, columns


def predictor_timing_from_summary(perf: dict[str, Any]) -> dict[str, Any]:
    source = perf["total"]["real_hardware_control"]["mpc_breakdown"][
        "disturbance_prediction_time"
    ]
    return {
        "source": "perf_summary.json: total.real_hardware_control.mpc_breakdown.disturbance_prediction_time",
        "count": int(source["count"]),
        "mean": float(source["mean"]),
        "p50": float(source["p50"]),
        "p95": float(source["p95"]),
        "p99": float(source["p99"]),
        "max": float(source["max"]),
    }


def read_control_quality(
    metrics_path: Path, trajectory_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(metrics_path, allow_pickle=False) as metrics:
        metrics_time = np.asarray(metrics["time"], dtype=np.float64)
        metrics_headline = (metrics_time >= 0.0) & (metrics_time < 8.0 - 1e-12)
        upright = np.asarray(metrics["right_ee_upright_alignment"], dtype=np.float64)
        tilt = np.arccos(np.clip(upright[metrics_headline], -1.0, 1.0))
        ee_acc = vector_norm_values(
            np.asarray(metrics["right_ee_lin_acc_world"])[metrics_headline]
        )
        ee_alpha = vector_norm_values(
            np.asarray(metrics["right_ee_ang_acc_world"])[metrics_headline]
        )

    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        time = np.asarray(trajectory["time"], dtype=np.float64)
        headline = (time >= 0.0) & (time < 8.0 - 1e-12)
        position = vector_norm_values(
            np.asarray(trajectory["right_ee_position_error_torso"])[headline]
        )
        torso_acc = vector_norm_values(
            np.asarray(trajectory["torso_acc_world_used"])[headline]
        )
        torso_alpha = vector_norm_values(
            np.asarray(trajectory["torso_alpha_world_used"])[headline]
        )

        interval_valid = (
            headline
            & np.isfinite(trajectory["right_mpc_tracking_interval_dt"])
            & np.all(np.isfinite(trajectory["right_mpc_interval_acc_error"]), axis=1)
        )
        pred_acc = vector_norm_values(
            np.asarray(trajectory["right_mpc_interval_acc_error"])[interval_valid]
        )
        pred_alpha = vector_norm_values(
            np.asarray(trajectory["right_mpc_interval_alpha_error"])[interval_valid]
        )
        pred_omega = vector_norm_values(
            np.asarray(trajectory["right_mpc_interval_omega_error"])[interval_valid]
        )
        orientation_valid = (
            headline
            & np.asarray(
                trajectory["right_mpc_template_one_step_prediction_valid"], dtype=bool
            )
            & np.isfinite(trajectory["right_mpc_template_one_step_rotation_error_angle"])
        )
        pred_orientation = np.asarray(
            trajectory["right_mpc_template_one_step_rotation_error_angle"],
            dtype=np.float64,
        )[orientation_valid]
        ddq_valid = headline & np.asarray(
            trajectory["right_arm_ddq_tracking_valid"], dtype=bool
        )
        ddq_error = vector_norm_values(
            np.asarray(trajectory["right_arm_ddq_tracking_error"])[ddq_valid]
        )

        arm_update = headline & np.asarray(trajectory["arm_policy_updated"], dtype=bool)
        qp_success = np.asarray(trajectory["right_mpc_solver_success"], dtype=bool)
        qp_fallback = np.asarray(trajectory["right_mpc_fallback_used"], dtype=bool)
        qp_feasible = np.asarray(trajectory["right_mpc_fallback_feasible"], dtype=bool)
        predictor_fallback = np.asarray(
            trajectory["right_mpc_predictor_fallback_used"], dtype=bool
        )

    arrays = {
        "tilt_angle_rad": tilt,
        "position_error_norm_m": position,
        "right_ee_linear_acceleration_norm_m_s2": ee_acc,
        "right_ee_angular_acceleration_norm_rad_s2": ee_alpha,
        "torso_acceleration_norm_m_s2": torso_acc,
        "torso_angular_acceleration_norm_rad_s2": torso_alpha,
        "predictor_interval_acc_error_norm_m_s2": pred_acc,
        "predictor_interval_alpha_error_norm_rad_s2": pred_alpha,
        "predictor_interval_omega_error_norm_rad_s": pred_omega,
        "predictor_orientation_geodesic_error_rad": pred_orientation,
        "ddq_tracking_error_norm_rad_s2": ddq_error,
    }
    quality = {name: scalar_stats(values) for name, values in arrays.items()}
    quality["qp"] = {
        "update_count": int(np.count_nonzero(arm_update)),
        "success_count": int(np.count_nonzero(arm_update & qp_success)),
        "fallback_count": int(np.count_nonzero(arm_update & qp_fallback)),
        "feasible_fallback_count": int(
            np.count_nonzero(arm_update & qp_fallback & qp_feasible)
        ),
    }
    quality["predictor_fallback"] = {
        "headline_count": int(np.count_nonzero(arm_update & predictor_fallback)),
        "headline_update_count": int(np.count_nonzero(arm_update)),
    }
    quality["sample_counts"] = {
        "metrics_headline": int(np.count_nonzero(metrics_headline)),
        "trajectory_headline": int(np.count_nonzero(headline)),
        "interval_prediction": int(np.count_nonzero(interval_valid)),
        "orientation_prediction": int(np.count_nonzero(orientation_valid)),
        "ddq_tracking": int(np.count_nonzero(ddq_valid)),
    }
    return quality, arrays


def validate_handoff(handoff_summary: dict[str, Any], run_id: str) -> dict[str, Any]:
    handoff = handoff_summary["handoff"]
    require(
        abs(float(handoff_summary["first_lower_policy_update_simulation_time_s"]) - 0.020)
        <= 1e-12,
        f"{run_id}: first lower-policy update",
    )
    for field in ("simulation_time_s", "task_time_s", "template_absolute_task_time_s"):
        require(abs(float(handoff[field]) - 0.024) <= 1e-12, f"{run_id}: handoff {field}")
    require(handoff["template_anchor_index"] == 4, f"{run_id}: template anchor")
    require(handoff["previous_tau_available"] is True, f"{run_id}: previous torque")
    require(
        np.allclose(
            np.asarray(handoff["last_fixed_pd_tau_nm"], dtype=np.float64),
            np.asarray(handoff["previous_executed_tau_input_nm"], dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        ),
        f"{run_id}: previous torque continuity",
    )
    require(handoff_summary["prefix"]["included_in_headline"] is True, f"{run_id}: prefix")
    return {
        "first_lower_policy_update_simulation_time_s": float(
            handoff_summary["first_lower_policy_update_simulation_time_s"]
        ),
        "first_lower_policy_update_task_time_s": float(
            handoff_summary["first_lower_policy_update_task_time_s"]
        ),
        "mpc_handoff_simulation_time_s": float(handoff["simulation_time_s"]),
        "mpc_handoff_task_time_s": float(handoff["task_time_s"]),
        "template_absolute_task_time_s": float(
            handoff["template_absolute_task_time_s"]
        ),
        "template_anchor_index": int(handoff["template_anchor_index"]),
        "previous_tau_available": bool(handoff["previous_tau_available"]),
        "previous_tau_matches_last_executed_pd": True,
        "tau_jump_l2_nm": float(handoff["tau_jump_l2_nm"]),
        "tau_jump_max_abs_nm": float(handoff["tau_jump_max_abs_nm"]),
        "startup_pd_prefix_included_in_headline": True,
    }


def run_safety(
    *,
    quality: dict[str, Any],
    right_arm: dict[str, Any],
    smoke: dict[str, Any],
) -> dict[str, Any]:
    branches = right_arm["right_arm_execution_branches"]
    result = {
        "mapper_execution_call_count": int(branches["execution_call_count"]),
        "mapper_rescue_used_count": int(branches["rescue_used_count"]),
        "mapper_hold_last_succeeded_count": int(branches["hold_last_succeeded_count"]),
        "mapper_safe_hold_used_count": int(branches["safe_hold_used_count"]),
        "mapper_safety_line_search_used_count": int(
            branches["safety_line_search_used_count"]
        ),
        "mapper_final_output_uncertified_count": int(
            branches["final_output_uncertified_count"]
        ),
        "mapper_no_safe_torque_count": int(branches["no_safe_torque_count"]),
        "mapper_final_unsafe_count": int(branches["final_unsafe_count"]),
        "qp_update_count": int(quality["qp"]["update_count"]),
        "qp_fallback_count": int(quality["qp"]["fallback_count"]),
        "predictor_fallback_count": int(
            quality["predictor_fallback"]["headline_count"]
        ),
        "fallen": bool(smoke["fallen"]),
        "nan_inf_count": int(smoke["nan_inf_count"]),
        "runtime_executor_nonzero_flag_count": int(
            smoke["runtime_executor_nonzero_flag_count"]
        ),
    }
    result["certified_control_gate_pass"] = bool(
        result["mapper_final_output_uncertified_count"] == 0
        and result["mapper_no_safe_torque_count"] == 0
        and result["mapper_final_unsafe_count"] == 0
        and result["qp_fallback_count"] == 0
        and result["predictor_fallback_count"] == 0
        and result["runtime_executor_nonzero_flag_count"] == 0
        and not result["fallen"]
        and result["nan_inf_count"] == 0
    )
    return result


def aggregate_perf_columns(
    columns_per_run: list[dict[str, np.ndarray]]
) -> dict[str, Any]:
    def joined(name: str) -> np.ndarray:
        return np.concatenate([columns[name] for columns in columns_per_run])

    elapsed = joined("complete_interval_ms")
    budget = joined("budget_ms")
    return {
        "complete_6ms_ms": {
            **scalar_stats(elapsed),
            "budget_ms": 6.0,
            "overrun_count": int(np.count_nonzero(elapsed > budget)),
            "overrun_fraction": float(np.mean(elapsed > budget)),
        },
        "components_ms": {
            "mpc_policy_update": scalar_stats(joined("mpc_policy_update_ms")),
            "all_ddq_to_torque_calls": scalar_stats(joined("ddq_total_ms")),
            "ddq_call_1": scalar_stats(joined("ddq_call_1_ms")),
            "ddq_call_2": scalar_stats(joined("ddq_call_2_ms")),
            "cpp_executor_bridge": scalar_stats(joined("cpp_executor_bridge_ms")),
            "other_right_arm_path": scalar_stats(joined("other_right_arm_path_ms")),
        },
    }


def aggregate_quality(arrays_per_run: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    names = tuple(arrays_per_run[0])
    return {
        name: scalar_stats(np.concatenate([arrays[name] for arrays in arrays_per_run]))
        for name in names
    }


def aggregate_predictor_timing(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.asarray([item["count"] for item in items], dtype=np.int64)
    means = np.asarray([item["mean"] for item in items], dtype=np.float64)
    p95 = np.asarray([item["p95"] for item in items], dtype=np.float64)
    p99 = np.asarray([item["p99"] for item in items], dtype=np.float64)
    maxima = np.asarray([item["max"] for item in items], dtype=np.float64)
    return {
        "source": "weighted per-run perf_summary values; pooled raw predictor samples were not persisted",
        "count": int(np.sum(counts)),
        "weighted_mean": float(np.average(means, weights=counts)),
        "per_run_p95_min": float(np.min(p95)),
        "per_run_p95_max": float(np.max(p95)),
        "per_run_p99_min": float(np.min(p99)),
        "per_run_p99_max": float(np.max(p99)),
        "max": float(np.max(maxima)),
    }


def validate_parity(parity: dict[str, Any]) -> dict[str, Any]:
    require(parity.get("status") == "PASS", "offline-online parity did not pass")
    require(parity["template"]["sha256"] == EXPECTED_TEMPLATE_SHA256, "parity template hash")
    require(
        parity["template"]["manifest_sha256"] == EXPECTED_TEMPLATE_MANIFEST_SHA256,
        "parity manifest hash",
    )
    require(len(parity["episodes"]) == 4, "parity must contain four held-out episodes")
    for episode in parity["episodes"]:
        require(episode["status"] == "PASS", f"parity failed: {episode['episode_id']}")
        require(episode["predictor_fallback_count"] == 0, "parity predictor fallback")
        require(episode["terminal_hold_count_in_headline"] == 0, "parity terminal hold")
        require(episode["invalid_rotation_batch_count"] == 0, "parity rotation")
        require(episode["query_indices_exact_0_to_1333"] is True, "parity anchor indices")
        require(episode["implicit_interpolation_used"] is False, "parity interpolation")
        for name, error in episode["max_abs_errors"].items():
            tolerance = float(episode["tolerances"][name])
            require(float(error) <= tolerance, f"parity tolerance: {episode['episode_id']} {name}")
    return {
        "status": "PASS",
        "episode_count": 4,
        "predictor_fallback_count": 0,
        "implicit_interpolation_used": False,
        "global_max_abs_errors": parity["global_max_abs_errors"],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def csv_row(run: dict[str, Any]) -> dict[str, Any]:
    timing = run["timing"]["complete_6ms_ms"]
    quality = run["control_quality"]
    safety = run["safety"]
    return {
        "run_id": run["run_id"],
        "scenario": run["scenario"],
        "repetition": run["repetition"],
        "complete_6ms_count": timing["count"],
        "complete_6ms_mean_ms": timing["mean"],
        "complete_6ms_p95_ms": timing["p95"],
        "complete_6ms_p99_ms": timing["p99"],
        "complete_6ms_max_ms": timing["max"],
        "complete_6ms_overrun_count": timing["overrun_count"],
        "mpc_policy_mean_ms": run["timing"]["components_ms"]["mpc_policy_update"]["mean"],
        "ddq_total_mean_ms": run["timing"]["components_ms"]["all_ddq_to_torque_calls"]["mean"],
        "ddq_call_1_mean_ms": run["timing"]["components_ms"]["ddq_call_1"]["mean"],
        "ddq_call_2_mean_ms": run["timing"]["components_ms"]["ddq_call_2"]["mean"],
        "predictor_mean_ms": run["timing"]["predictor_time_ms"]["mean"],
        "other_path_mean_ms": run["timing"]["components_ms"]["other_right_arm_path"]["mean"],
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
        "mapper_execution_call_count": safety["mapper_execution_call_count"],
        "mapper_rescue_used_count": safety["mapper_rescue_used_count"],
        "mapper_hold_last_succeeded_count": safety["mapper_hold_last_succeeded_count"],
        "mapper_final_output_uncertified_count": safety[
            "mapper_final_output_uncertified_count"
        ],
        "mapper_no_safe_torque_count": safety["mapper_no_safe_torque_count"],
        "mapper_final_unsafe_count": safety["mapper_final_unsafe_count"],
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
    copied_files: list[dict[str, Any]] = []
    derived_sources: list[dict[str, Any]] = []
    run_reports: list[dict[str, Any]] = []
    perf_arrays: dict[str, dict[str, np.ndarray]] = {}
    quality_arrays: dict[str, dict[str, np.ndarray]] = {}

    template_path = repository / TEMPLATE_ROOT / "full_task_template.npz"
    template_manifest_path = repository / TEMPLATE_ROOT / "full_task_template_manifest.json"
    template_sha = sha256_file(template_path)
    template_manifest_sha = sha256_file(template_manifest_path)
    require(template_sha == EXPECTED_TEMPLATE_SHA256, "frozen template checksum mismatch")
    require(
        template_manifest_sha == EXPECTED_TEMPLATE_MANIFEST_SHA256,
        "frozen template manifest checksum mismatch",
    )
    template_manifest = load_json(template_manifest_path)
    require(template_manifest["template_schema_version"] == "full_task_template_v2", "template schema")
    require(
        template_manifest["collection"]["protocol"]["version"]
        == "full_task_direct_step_v1",
        "template protocol",
    )
    require(template_manifest["template_validation"]["anchor_count"] == 1334, "template anchors")
    require(template_manifest["template_validation"]["horizon"] == 9, "template horizon")
    require(
        template_manifest["template_validation"]["heading_frame_version"]
        == "full_task_continuous_heading_v2",
        "template heading frame",
    )

    for run_id, scenario, repetition in RUNS:
        run_dir = repository / CONTROLLED_RUN_ROOT / run_id
        for filename in RUN_EVIDENCE_FILES:
            copied_files.append(
                copy_with_record(
                    source=run_dir / filename,
                    destination=output / "controlled_runs" / run_id / filename,
                    repository=repository,
                    output=output,
                    category="controlled_run_whitelist",
                )
            )

        metadata = load_json(run_dir / "run_metadata.json")
        preflight = load_json(run_dir / "formal_full_task_runtime_preflight.json")
        perf_summary = load_json(run_dir / "perf_summary.json")
        handoff_summary = load_json(run_dir / "startup_pd_handoff_summary.json")
        right_arm = load_json(run_dir / "right_arm_diagnostics.json")
        smoke = load_json(run_dir / "full_task_smoke_summary.json")
        manifest = load_json(run_dir / "full_task_manifest.json")
        metrics_path = run_dir / "metrics.npz"
        trajectory_path = run_dir / "trajectory.npz"
        for source, purpose in (
            (metrics_path, "control quality derivation"),
            (trajectory_path, "control quality and exact event-count derivation"),
        ):
            require(source.is_file(), f"missing derivation source: {source}")
            derived_sources.append(
                {
                    "run_id": run_id,
                    "purpose": purpose,
                    "source_absolute_path": str(source.resolve()),
                    "source_repository_path": relative_to_repository(source, repository),
                    "bytes": int(source.stat().st_size),
                    "sha256": sha256_file(source),
                    "copied_to_evidence_pack": False,
                }
            )

        formal = validate_formal_environment(preflight, metadata, run_id)
        timing, perf_columns = read_perf_intervals(run_dir / "perf_intervals.csv")
        timing["predictor_time_ms"] = predictor_timing_from_summary(perf_summary)
        quality, arrays = read_control_quality(metrics_path, trajectory_path)
        handoff = validate_handoff(handoff_summary, run_id)
        safety = run_safety(quality=quality, right_arm=right_arm, smoke=smoke)

        predictor_metadata = manifest["predictor"]
        require(predictor_metadata["predictor_type"] == "full_task_template", f"{run_id}: predictor")
        require(predictor_metadata["sha256"] == template_sha, f"{run_id}: template hash")
        require(
            predictor_metadata["manifest_sha256"] == template_manifest_sha,
            f"{run_id}: template manifest hash",
        )
        require(smoke["strict_pre_step"] is True, f"{run_id}: strict pre-step")
        require(smoke["heading_enabled"] is True, f"{run_id}: heading disabled")
        require(smoke["direct_step_effective"] is True, f"{run_id}: direct step")
        require(smoke["tail_complete"] is True, f"{run_id}: label tail")
        require(smoke["headline_anchor_count"] == 1334, f"{run_id}: headline anchors")
        require(safety["certified_control_gate_pass"], f"{run_id}: certified control gate")
        require(timing["complete_6ms_ms"]["overrun_count"] == 0, f"{run_id}: timing overrun")

        run_reports.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "repetition": repetition,
                "source_run_absolute_path": str(run_dir.resolve()),
                "headline": "[0.0,8.0)",
                "formal_environment": formal,
                "template": {
                    "predictor_type": predictor_metadata["predictor_type"],
                    "template_schema_version": predictor_metadata[
                        "template_schema_version"
                    ],
                    "protocol_version": predictor_metadata["protocol_version"],
                    "heading_definition": predictor_metadata["heading_definition"],
                    "template_sha256": template_sha,
                    "manifest_sha256": template_manifest_sha,
                },
                "timing": timing,
                "handoff": handoff,
                "control_quality": quality,
                "safety": safety,
                "xy": {
                    "displacement_m": float(smoke["xy_displacement_m"]),
                    "arc_length_m": float(smoke["xy_arc_length_m"]),
                },
                "raw_contract": {
                    "strict_pre_step": bool(smoke["strict_pre_step"]),
                    "raw_sample_count": int(smoke["raw_sample_count"]),
                    "headline_anchor_count": int(smoke["headline_anchor_count"]),
                    "last_raw_time_s": float(smoke["last_raw_time"]),
                    "last_horizon_node_s": float(smoke["last_horizon_node"]),
                    "tail_complete": bool(smoke["tail_complete"]),
                },
                "source_smoke_status": {
                    "status": smoke["status"],
                    "smoke_passed": bool(smoke["smoke_passed"]),
                    "note": (
                        "Preserved verbatim for provenance. Final freeze acceptance is based "
                        "on certified output, NO_SAFE_TORQUE/final_unsafe, QP/predictor "
                        "fallback, stability, and complete-interval gates."
                    ),
                },
            }
        )
        perf_arrays[run_id] = perf_columns
        quality_arrays[run_id] = arrays

    for scenario, run_id in REPRESENTATIVE_RUNS.items():
        source_dir = repository / CONTROLLED_RUN_ROOT / run_id
        for filename in REPRESENTATIVE_PLOTS:
            copied_files.append(
                copy_with_record(
                    source=source_dir / filename,
                    destination=output / "representative_plots" / scenario / filename,
                    repository=repository,
                    output=output,
                    category="representative_control_plot",
                )
            )

    parity_source = repository / PARITY_SOURCE
    parity = load_json(parity_source)
    parity_validation = validate_parity(parity)
    copied_files.append(
        copy_with_record(
            source=parity_source,
            destination=output / "offline_online_parity" / parity_source.name,
            repository=repository,
            output=output,
            category="offline_online_parity",
        )
    )

    for relative in TEMPLATE_EVIDENCE_FILES:
        source = repository / TEMPLATE_ROOT / relative
        copied_files.append(
            copy_with_record(
                source=source,
                destination=output / "template_evidence" / relative,
                repository=repository,
                output=output,
                category="template_build_and_heldout_evidence",
            )
        )

    scenario_aggregates: dict[str, Any] = {}
    for scenario in ("nominal", "heldout_pair_02_minus"):
        selected = [run for run in run_reports if run["scenario"] == scenario]
        ids = [run["run_id"] for run in selected]
        timing_aggregate = aggregate_perf_columns([perf_arrays[run_id] for run_id in ids])
        timing_aggregate["predictor_time_ms"] = aggregate_predictor_timing(
            [run["timing"]["predictor_time_ms"] for run in selected]
        )
        scenario_aggregates[scenario] = {
            "run_count": len(selected),
            "run_ids": ids,
            "timing": timing_aggregate,
            "control_quality": aggregate_quality(
                [quality_arrays[run_id] for run_id in ids]
            ),
            "safety_totals": {
                key: sum(int(run["safety"][key]) for run in selected)
                for key in (
                    "mapper_execution_call_count",
                    "mapper_rescue_used_count",
                    "mapper_hold_last_succeeded_count",
                    "mapper_safe_hold_used_count",
                    "mapper_safety_line_search_used_count",
                    "mapper_final_output_uncertified_count",
                    "mapper_no_safe_torque_count",
                    "mapper_final_unsafe_count",
                    "qp_fallback_count",
                    "predictor_fallback_count",
                    "nan_inf_count",
                )
            },
            "all_environment_preflights_pass": all(
                run["formal_environment"]["passed"] for run in selected
            ),
            "all_certified_control_gates_pass": all(
                run["safety"]["certified_control_gate_pass"] for run in selected
            ),
        }

    all_timing = aggregate_perf_columns([perf_arrays[run_id] for run_id, _, _ in RUNS])
    all_timing["predictor_time_ms"] = aggregate_predictor_timing(
        [run["timing"]["predictor_time_ms"] for run in run_reports]
    )
    safety_total_keys = (
        "mapper_execution_call_count",
        "mapper_rescue_used_count",
        "mapper_hold_last_succeeded_count",
        "mapper_safe_hold_used_count",
        "mapper_safety_line_search_used_count",
        "mapper_final_output_uncertified_count",
        "mapper_no_safe_torque_count",
        "mapper_final_unsafe_count",
        "qp_fallback_count",
        "predictor_fallback_count",
        "nan_inf_count",
    )
    all_safety_totals = {
        key: sum(int(run["safety"][key]) for run in run_reports)
        for key in safety_total_keys
    }
    overall_pass = bool(
        len(run_reports) == 6
        and all(run["formal_environment"]["passed"] for run in run_reports)
        and all(run["safety"]["certified_control_gate_pass"] for run in run_reports)
        and all_timing["complete_6ms_ms"]["count"] == 7974
        and all_timing["complete_6ms_ms"]["overrun_count"] == 0
        and all_safety_totals["mapper_execution_call_count"] == 16074
        and all_safety_totals["mapper_final_output_uncertified_count"] == 0
        and parity_validation["status"] == "PASS"
    )
    require(overall_pass, "controlled-run aggregate gate failed")

    aggregate = {
        "schema_version": "full_task_template_v2_final_freeze_evidence_v1",
        "status": "PASS",
        "scope": {
            "control_candidate": "MPC + FullTaskTemplatePredictor v2",
            "template_semantics": (
                "fixed absolute-task-time baseline that knows the 6.4 s direct stop"
            ),
            "headline": "[0.0,8.0)",
            "startup_pd_prefix": "[0.0,0.024) is included in the headline",
            "mpc_handoff": "task/simulation time 0.024 s, absolute template anchor 4",
            "hardware_claim": "controlled MuJoCo simulation only; not hardware real-time evidence",
        },
        "frozen_template": {
            "repository_path": str(TEMPLATE_ROOT / "full_task_template.npz"),
            "sha256": template_sha,
            "manifest_repository_path": str(
                TEMPLATE_ROOT / "full_task_template_manifest.json"
            ),
            "manifest_sha256": template_manifest_sha,
            "schema_version": template_manifest["template_schema_version"],
            "protocol_version": template_manifest["collection"]["protocol"]["version"],
            "heading_frame_version": template_manifest["template_validation"][
                "heading_frame_version"
            ],
        },
        "offline_online_parity": parity_validation,
        "controlled_runs": run_reports,
        "scenario_aggregates": scenario_aggregates,
        "all_six_runs": {
            "timing": all_timing,
            "safety_totals": all_safety_totals,
            "all_environment_preflights_pass": True,
            "all_certified_control_gates_pass": True,
            "all_fall_counts_zero": all(not run["safety"]["fallen"] for run in run_reports),
            "complete_6ms_interval_count": int(
                all_timing["complete_6ms_ms"]["count"]
            ),
            "mapper_execution_call_count": int(
                all_safety_totals["mapper_execution_call_count"]
            ),
        },
        "provenance": {
            "run_selection": "six explicit directory names; no latest-run scan",
            "run_whitelist_file_count_each": len(RUN_EVIDENCE_FILES),
            "control_quality_derivation": (
                "recomputed from metrics.npz and trajectory.npz using the existing "
                "headline definitions; large arrays are hashed but not copied"
            ),
            "timing_derivation": "recomputed from every row of perf_intervals.csv",
        },
    }
    aggregate_path = output / AGGREGATE_JSON
    write_json(aggregate_path, aggregate)

    csv_path = output / AGGREGATE_CSV
    rows = [csv_row(run) for run in run_reports]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    generated_files = []
    for path, category in (
        (aggregate_path, "generated_aggregate_json"),
        (csv_path, "generated_aggregate_csv"),
    ):
        generated_files.append(
            {
                "category": category,
                "output_repository_path": relative_to_repository(path, repository),
                "output_package_path": str(path.resolve().relative_to(output)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )

    file_manifest = {
        "schema_version": "full_task_template_v2_compact_evidence_files_v1",
        "status": "PASS",
        "repository_root": str(repository),
        "output_absolute_path": str(output),
        "builder": {
            "repository_path": relative_to_repository(Path(__file__), repository),
            "sha256": sha256_file(Path(__file__)),
        },
        "selection": {
            "controlled_run_ids": [run_id for run_id, _, _ in RUNS],
            "whitelist_file_count_per_run": len(RUN_EVIDENCE_FILES),
            "whitelist_files": list(RUN_EVIDENCE_FILES),
            "representative_runs": REPRESENTATIVE_RUNS,
            "representative_plot_files": list(REPRESENTATIVE_PLOTS),
            "template_evidence_files": list(TEMPLATE_EVIDENCE_FILES),
        },
        "frozen_assets": {
            "template": {
                "absolute_path": str(template_path.resolve()),
                "repository_path": relative_to_repository(template_path, repository),
                "bytes": int(template_path.stat().st_size),
                "sha256": template_sha,
            },
            "template_manifest": {
                "absolute_path": str(template_manifest_path.resolve()),
                "repository_path": relative_to_repository(template_manifest_path, repository),
                "bytes": int(template_manifest_path.stat().st_size),
                "sha256": template_manifest_sha,
            },
        },
        "copied_files": copied_files,
        "derived_only_source_files": derived_sources,
        "generated_files": generated_files,
        "validation": {
            "copied_file_count": len(copied_files),
            "controlled_run_whitelist_file_count": 6 * len(RUN_EVIDENCE_FILES),
            "all_copy_checksums_match": True,
            "all_six_runtime_preflights_pass": True,
            "all_six_certified_control_gates_pass": True,
            "all_7974_complete_intervals_within_6ms": True,
            "all_16074_mapper_outputs_certified": True,
            "offline_online_parity_pass": True,
        },
    }
    manifest_path = output / FILE_MANIFEST
    write_json(manifest_path, file_manifest)
    verify(repository, output, require_sources=True)
    return {
        "status": "PASS",
        "output": str(output),
        "aggregate": str(aggregate_path),
        "metrics_csv": str(csv_path),
        "file_manifest": str(manifest_path),
        "copied_file_count": len(copied_files),
        "complete_interval_count": int(all_timing["complete_6ms_ms"]["count"]),
        "complete_interval_overrun_count": int(
            all_timing["complete_6ms_ms"]["overrun_count"]
        ),
        "mapper_execution_call_count": int(
            all_safety_totals["mapper_execution_call_count"]
        ),
        "mapper_uncertified_output_count": int(
            all_safety_totals["mapper_final_output_uncertified_count"]
        ),
    }


def verify(
    repository: Path, output: Path, *, require_sources: bool = False
) -> dict[str, Any]:
    repository = repository.resolve()
    output = output.resolve()
    manifest = load_json(output / FILE_MANIFEST)
    aggregate = load_json(output / AGGREGATE_JSON)
    require(manifest["status"] == "PASS", "evidence file manifest status")
    require(aggregate["status"] == "PASS", "aggregate status")
    builder_path = repository / manifest["builder"]["repository_path"]
    require(builder_path.is_file(), f"missing evidence builder: {builder_path}")
    require(
        sha256_file(builder_path) == manifest["builder"]["sha256"],
        "evidence builder changed after the package was generated",
    )
    require(
        aggregate["frozen_template"]["sha256"] == EXPECTED_TEMPLATE_SHA256,
        "aggregate template checksum",
    )
    require(
        aggregate["frozen_template"]["manifest_sha256"]
        == EXPECTED_TEMPLATE_MANIFEST_SHA256,
        "aggregate manifest checksum",
    )
    copied = manifest["copied_files"]
    require(
        len([item for item in copied if item["category"] == "controlled_run_whitelist"])
        == 6 * len(RUN_EVIDENCE_FILES),
        "controlled-run whitelist file count",
    )
    missing_archived_sources = 0
    for item in copied:
        output_path = output / item["output_package_path"]
        source_path = Path(item["source_absolute_path"])
        require(output_path.is_file(), f"missing packaged evidence: {output_path}")
        require(int(output_path.stat().st_size) == item["bytes"], f"size drift: {output_path}")
        require(sha256_file(output_path) == item["sha256"], f"hash drift: {output_path}")
        if source_path.is_file():
            require(
                sha256_file(source_path) == item["sha256"],
                f"source hash drift: {source_path}",
            )
        else:
            missing_archived_sources += 1
            require(
                not require_sources,
                f"missing source evidence before archival: {source_path}",
            )
    for item in manifest["derived_only_source_files"]:
        source_path = Path(item["source_absolute_path"])
        if source_path.is_file():
            require(
                sha256_file(source_path) == item["sha256"],
                f"derivation source drift: {source_path}",
            )
        else:
            missing_archived_sources += 1
            require(
                not require_sources,
                f"missing derivation source before archival: {source_path}",
            )
    for item in manifest["generated_files"]:
        output_path = output / item["output_package_path"]
        require(output_path.is_file(), f"missing generated evidence: {output_path}")
        require(sha256_file(output_path) == item["sha256"], f"generated hash drift: {output_path}")
    require(
        sha256_file(repository / TEMPLATE_ROOT / "full_task_template.npz")
        == EXPECTED_TEMPLATE_SHA256,
        "live template checksum drift",
    )
    require(
        sha256_file(repository / TEMPLATE_ROOT / "full_task_template_manifest.json")
        == EXPECTED_TEMPLATE_MANIFEST_SHA256,
        "live template manifest checksum drift",
    )
    validation = manifest["validation"]
    require(validation["all_six_runtime_preflights_pass"] is True, "preflight gate")
    require(validation["all_six_certified_control_gates_pass"] is True, "control gate")
    require(validation["all_7974_complete_intervals_within_6ms"] is True, "timing gate")
    require(validation["all_16074_mapper_outputs_certified"] is True, "mapper gate")
    return {
        "status": "PASS",
        "output": str(output),
        "copied_file_count": len(copied),
        "generated_file_count": len(manifest["generated_files"]),
        "missing_archived_source_count": missing_archived_sources,
        "source_presence_required": require_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=str(REPOSITORY_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the existing compact pack without copying or regenerating files",
    )
    parser.add_argument(
        "--require-sources",
        action="store_true",
        help="also require every source run/derivation file to still be present",
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
