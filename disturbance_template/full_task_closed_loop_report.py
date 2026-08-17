"""Create a T2 nominal safety-gate report from two explicit run directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"rms": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "rms": float(np.sqrt(np.mean(finite**2))),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _norm_stats(vectors: np.ndarray) -> dict[str, float]:
    values = np.asarray(vectors, dtype=np.float64)
    valid = np.all(np.isfinite(values), axis=1)
    return _stats(np.linalg.norm(values[valid], axis=1))


def _load_run(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    with np.load(run_dir / "metrics.npz", allow_pickle=False) as source:
        metrics = {name: source[name].copy() for name in source.files}
    with np.load(run_dir / "trajectory.npz", allow_pickle=True) as source:
        trajectory = {name: source[name].copy() for name in source.files}
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    smoke = json.loads(
        (run_dir / "full_task_smoke_summary.json").read_text(encoding="utf-8")
    )
    tracking = json.loads(
        (run_dir / "mpc_tracking_diagnostics.json").read_text(encoding="utf-8")
    )
    right_arm = json.loads(
        (run_dir / "right_arm_diagnostics.json").read_text(encoding="utf-8")
    )
    perf = json.loads((run_dir / "perf_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "full_task_manifest.json").read_text(encoding="utf-8")
    )
    time = np.asarray(trajectory["time"], dtype=np.float64)
    headline = (time >= 0.0) & (time < 8.0 - 1e-12)
    metrics_time = np.asarray(metrics["time"], dtype=np.float64)
    metrics_headline = (metrics_time >= 0.0) & (metrics_time < 8.0 - 1e-12)
    tilt_angle = np.arccos(
        np.clip(
            np.asarray(metrics["right_ee_upright_alignment"], dtype=np.float64),
            -1.0,
            1.0,
        )
    )
    position_norm = np.linalg.norm(
        trajectory["right_ee_position_error_torso"], axis=1
    )
    interval_valid = (
        headline
        & np.isfinite(trajectory["right_mpc_tracking_interval_dt"])
        & np.all(np.isfinite(trajectory["right_mpc_interval_acc_error"]), axis=1)
    )
    orientation_valid = (
        headline
        & trajectory["right_mpc_template_one_step_prediction_valid"]
        & np.isfinite(
            trajectory["right_mpc_template_one_step_rotation_error_angle"]
        )
    )
    ddq_valid = headline & trajectory["right_arm_ddq_tracking_valid"]
    arm_update = headline & trajectory["arm_policy_updated"]
    fallback_count = int(
        np.count_nonzero(
            arm_update & trajectory["right_mpc_predictor_fallback_used"]
        )
    )
    with (run_dir / "perf_intervals.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        perf_rows = list(csv.DictReader(stream))
    perf_time = np.asarray(
        [float(row["simulation_start_time_s"]) for row in perf_rows]
    )
    perf_ms = np.asarray([float(row["complete_interval_ms"]) for row in perf_rows])
    perf_headline = perf_time < 8.0 - 1e-12
    timing = _stats(perf_ms[perf_headline])
    timing.update(
        {
            "mean": float(np.mean(perf_ms[perf_headline])),
            "count": int(np.count_nonzero(perf_headline)),
            "budget_ms": 6.0,
            "overrun_count": int(np.count_nonzero(perf_ms[perf_headline] > 6.0)),
            "overrun_fraction": float(np.mean(perf_ms[perf_headline] > 6.0)),
            "first_interval_ms": float(perf_ms[0]),
        }
    )
    mapping_events = []
    with (run_dir / "control_preview.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            row_time = float(row["time"])
            if (
                row_time < 8.0 - 1e-12
                and row["right_arm_forward_dynamics_safety_fallback_used"]
                == "1.0"
            ):
                mapping_events.append(
                    {
                        "time": row_time,
                        "satisfied": row[
                            "right_arm_forward_dynamics_safety_fallback_satisfied"
                        ]
                        == "1.0",
                        "hold_last_used": row[
                            "right_arm_forward_dynamics_hold_last_safe_used"
                        ]
                        == "1.0",
                        "hold_last_satisfied": row[
                            "right_arm_forward_dynamics_hold_last_safe_satisfied"
                        ]
                        == "1.0",
                    }
                )
    qp_count = int(np.count_nonzero(arm_update))
    qp_success_count = int(
        np.count_nonzero(arm_update & trajectory["right_mpc_solver_success"])
    )
    qp_fallback_count = int(
        np.count_nonzero(arm_update & trajectory["right_mpc_fallback_used"])
    )
    qp_feasible_fallback_count = int(
        np.count_nonzero(
            arm_update
            & trajectory["right_mpc_fallback_used"]
            & trajectory["right_mpc_fallback_feasible"]
        )
    )
    result = {
        "mode": mode,
        "run_dir": str(run_dir),
        "headline": "[0.0,8.0)",
        "predictor_metadata": manifest["predictor"],
        "safety_gate_pass": (
            int(right_arm["right_arm_execution_branches"]["final_unsafe_count"])
            == 0
        ),
        "tilt_angle_rad": _stats(tilt_angle[metrics_headline]),
        "position_error_norm_m": _stats(position_norm[headline]),
        "right_ee_linear_acceleration_norm_m_s2": _norm_stats(
            metrics["right_ee_lin_acc_world"][metrics_headline]
        ),
        "right_ee_angular_acceleration_norm_rad_s2": _norm_stats(
            metrics["right_ee_ang_acc_world"][metrics_headline]
        ),
        "torso_acceleration_norm_m_s2": _norm_stats(
            trajectory["torso_acc_world_used"][headline]
        ),
        "torso_angular_acceleration_norm_rad_s2": _norm_stats(
            trajectory["torso_alpha_world_used"][headline]
        ),
        "closed_loop_predictor_error": {
            "interval_acc_norm_m_s2": _norm_stats(
                trajectory["right_mpc_interval_acc_error"][interval_valid]
            ),
            "interval_alpha_norm_rad_s2": _norm_stats(
                trajectory["right_mpc_interval_alpha_error"][interval_valid]
            ),
            "interval_omega_norm_rad_s": _norm_stats(
                trajectory["right_mpc_interval_omega_error"][interval_valid]
            ),
            "one_step_orientation_geodesic_rad": _stats(
                trajectory[
                    "right_mpc_template_one_step_rotation_error_angle"
                ][orientation_valid]
            ),
            "interval_sample_count": int(np.count_nonzero(interval_valid)),
            "orientation_sample_count": int(np.count_nonzero(orientation_valid)),
        },
        "ddq_tracking": {
            "vector_error_norm_rad_s2": _norm_stats(
                trajectory["right_arm_ddq_tracking_error"][ddq_valid]
            ),
            "per_joint_rmse_rad_s2": tracking["ddq_tracking"]["rmse"],
            "per_joint_correlation": tracking["ddq_tracking"]["correlation"],
            "per_joint_gain": tracking["ddq_tracking"]["gain"],
            "sample_count": int(np.count_nonzero(ddq_valid)),
        },
        "qp": {
            "update_count": qp_count,
            "success_count": qp_success_count,
            "success_fraction": qp_success_count / max(qp_count, 1),
            "fallback_count": qp_fallback_count,
            "feasible_fallback_count": qp_feasible_fallback_count,
        },
        "mapping": {
            **right_arm["right_arm_execution_branches"],
            "headline_2ms_fallback_rows": mapping_events,
        },
        "predictor_fallback": {
            "headline_count": fallback_count,
            "headline_fraction_of_updates": fallback_count / max(qp_count, 1),
        },
        "complete_6ms_timing_ms": timing,
        "xy": {
            "displacement_m": smoke["xy_displacement_m"],
            "arc_length_m": smoke["xy_arc_length_m"],
        },
        "stability": {
            "fallen": smoke["fallen"],
            "nan_inf_count": smoke["nan_inf_count"],
            "max_abs_heading_error_rad": smoke[
                "max_abs_filtered_heading_error_rad"
            ],
        },
        "raw_contract": {
            "samples": smoke["raw_sample_count"],
            "headline_anchors": smoke["headline_anchor_count"],
            "last_raw_time": smoke["last_raw_time"],
            "last_horizon_node": smoke["last_horizon_node"],
            "strict_pre_step": smoke["strict_pre_step"],
        },
        "_plot": {
            "time": time,
            "metrics_time": metrics_time,
            "tilt": tilt_angle,
            "position": position_norm,
            "interval_valid": interval_valid,
            "interval_acc_prediction": trajectory["right_mpc_interval_acc_k0"],
            "interval_acc_actual": trajectory["right_mpc_interval_acc_actual"],
            "interval_alpha_prediction": trajectory["right_mpc_interval_alpha_k0"],
            "interval_alpha_actual": trajectory["right_mpc_interval_alpha_actual"],
            "interval_omega_prediction": trajectory["right_mpc_interval_omega_k0"],
            "interval_omega_actual": trajectory["right_mpc_interval_omega_actual"],
            "ddq_valid": ddq_valid,
            "ddq_des": trajectory["right_arm_ddq_des"],
            "ddq_real": trajectory["right_arm_ddq_real"],
            "perf_time": perf_time,
            "perf_ms": perf_ms,
            "mapping_events": mapping_events,
        },
    }
    return result


def _plot_metric_window(
    runs: list[dict[str, Any]], output: Path, start: float, end: float, title: str
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for run in runs:
        data = run["_plot"]
        metric_mask = (data["metrics_time"] >= start) & (data["metrics_time"] < end)
        trajectory_mask = (data["time"] >= start) & (data["time"] < end)
        axes[0].plot(
            data["metrics_time"][metric_mask],
            data["tilt"][metric_mask],
            label=run["mode"],
            lw=1.0,
        )
        axes[1].plot(
            data["time"][trajectory_mask],
            data["position"][trajectory_mask],
            label=run["mode"],
            lw=1.0,
        )
    axes[0].set_ylabel("cup tilt angle [rad]")
    axes[1].set_ylabel("EE position error norm [m]")
    axes[1].set_xlabel("task time [s]")
    axes[0].set_title(title)
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _save_plots(runs: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    paths = []
    windows = (
        (0.0, 8.0, "nominal_full_task_tilt_position.png", "Full headline [0,8)"),
        (0.0, 2.4, "nominal_startup_0_2p4s.png", "Startup [0,2.4)"),
        (6.2, 8.0, "nominal_stop_6p2_8p0s.png", "Direct stop [6.2,8.0)"),
    )
    for start, end, name, title in windows:
        path = output_dir / name
        _plot_metric_window(runs, path, start, end, title)
        paths.append(path)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    quantities = (
        ("interval_acc_prediction", "interval_acc_actual", "acc norm [m/s²]"),
        ("interval_alpha_prediction", "interval_alpha_actual", "alpha norm [rad/s²]"),
        ("interval_omega_prediction", "interval_omega_actual", "omega norm [rad/s]"),
    )
    for column, run in enumerate(runs):
        data = run["_plot"]
        valid = data["interval_valid"]
        for row, (prediction_key, actual_key, label) in enumerate(quantities):
            axis = axes[row, column]
            axis.plot(
                data["time"][valid],
                np.linalg.norm(data[actual_key][valid], axis=1),
                label="truth",
                lw=0.9,
            )
            axis.plot(
                data["time"][valid],
                np.linalg.norm(data[prediction_key][valid], axis=1),
                label="prediction",
                lw=0.9,
            )
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
            if row == 0:
                axis.set_title(run["mode"])
            if row == 2:
                axis.set_xlabel("task time [s]")
            axis.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "nominal_predictor_truth_vs_prediction.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, run in zip(axes, runs):
        data = run["_plot"]
        valid = data["ddq_valid"]
        axis.plot(
            data["time"][valid],
            np.linalg.norm(data["ddq_des"][valid], axis=1),
            label="||ddq_des||",
            lw=0.9,
        )
        axis.plot(
            data["time"][valid],
            np.linalg.norm(data["ddq_real"][valid], axis=1),
            label="||6ms interval-average ddq_real||",
            lw=0.9,
        )
        axis.set_title(run["mode"])
        axis.set_ylabel("rad/s²")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("task time [s]")
    fig.tight_layout()
    path = output_dir / "nominal_ddq_des_vs_interval_real.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, run in zip(axes, runs):
        data = run["_plot"]
        mask = data["perf_time"] < 8.0
        axis.plot(data["perf_time"][mask], data["perf_ms"][mask], lw=0.8)
        axis.axhline(6.0, color="red", ls="--", label="6 ms budget")
        for event in data["mapping_events"]:
            axis.axvline(
                event["time"],
                color=("tab:orange" if event["satisfied"] else "red"),
                alpha=0.25,
            )
        axis.set_title(
            f"{run['mode']} | predictor fallback="
            f"{run['predictor_fallback']['headline_count']} | final unsafe="
            f"{run['mapping']['final_unsafe_count']}"
        )
        axis.set_ylabel("complete interval [ms]")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("task time [s]")
    fig.tight_layout()
    path = output_dir / "nominal_timing_and_mapping_fallback.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-run", required=True)
    parser.add_argument("--full-task-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    runs = [
        _load_run(Path(args.phase_run), "phase_template"),
        _load_run(Path(args.full_task_run), "full_task_template"),
    ]
    plot_paths = _save_plots(runs, output_dir)
    serializable_runs = []
    for run in runs:
        serializable_runs.append(
            {name: value for name, value in run.items() if name != "_plot"}
        )
    phase, full = serializable_runs
    comparison_keys = (
        ("tilt_rms_rad", phase["tilt_angle_rad"]["rms"], full["tilt_angle_rad"]["rms"]),
        ("position_rms_m", phase["position_error_norm_m"]["rms"], full["position_error_norm_m"]["rms"]),
        ("ee_acc_rms_m_s2", phase["right_ee_linear_acceleration_norm_m_s2"]["rms"], full["right_ee_linear_acceleration_norm_m_s2"]["rms"]),
        ("ee_alpha_rms_rad_s2", phase["right_ee_angular_acceleration_norm_rad_s2"]["rms"], full["right_ee_angular_acceleration_norm_rad_s2"]["rms"]),
        ("predictor_acc_error_rms_m_s2", phase["closed_loop_predictor_error"]["interval_acc_norm_m_s2"]["rms"], full["closed_loop_predictor_error"]["interval_acc_norm_m_s2"]["rms"]),
        ("predictor_alpha_error_rms_rad_s2", phase["closed_loop_predictor_error"]["interval_alpha_norm_rad_s2"]["rms"], full["closed_loop_predictor_error"]["interval_alpha_norm_rad_s2"]["rms"]),
        ("predictor_omega_error_rms_rad_s", phase["closed_loop_predictor_error"]["interval_omega_norm_rad_s"]["rms"], full["closed_loop_predictor_error"]["interval_omega_norm_rad_s"]["rms"]),
        ("timing_mean_ms", phase["complete_6ms_timing_ms"]["mean"], full["complete_6ms_timing_ms"]["mean"]),
    )
    nominal_differences = {
        name: {
            "phase": phase_value,
            "full_task": full_value,
            "full_minus_phase": full_value - phase_value,
            "relative_change_percent": (
                100.0 * (full_value - phase_value) / phase_value
                if phase_value != 0.0
                else np.nan
            ),
        }
        for name, phase_value, full_value in comparison_keys
    }
    batch_allowed = all(run["safety_gate_pass"] for run in serializable_runs)
    report = {
        "stage": "T2 nominal closed-loop safety gate",
        "headline": "[0.0,8.0)",
        "status": "PASS" if batch_allowed else "FAIL",
        "formal_heldout_pair_batch_run": False,
        "formal_heldout_pair_batch_block_reason": (
            None
            if batch_allowed
            else "at least one nominal mode has final_unsafe_count > 0"
        ),
        "runs": serializable_runs,
        "nominal_full_minus_phase": nominal_differences,
        "plots": [str(path) for path in plot_paths],
    }
    report_path = output_dir / "t2_nominal_safety_gate_report.json"
    report_path.write_text(
        json.dumps(_json_value(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "t2_nominal_headline_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "phase_template", "full_task_template", "full_minus_phase"))
        for name, values in nominal_differences.items():
            writer.writerow((name, values["phase"], values["full_task"], values["full_minus_phase"]))
    print(json.dumps(_json_value(report), indent=2, ensure_ascii=False))
    print(f"report={report_path}")
    print(f"table={csv_path}")
    return 0 if batch_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
