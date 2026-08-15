#!/usr/bin/env python3
"""Summarize five explicit v2 closed-loop runs without changing control code."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from disturbance_learning.full_task_closed_loop_report import _json_value, _load_run


ROOT = Path(__file__).resolve().parents[1]


def _unsafe_events(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with (run_dir / "control_preview.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if (
                row["ddq_execution_updated"] == "1.0"
                and row["right_arm_forward_dynamics_safety_fallback_used"] == "1.0"
                and row["right_arm_forward_dynamics_safety_fallback_satisfied"] != "1.0"
                and row["right_arm_forward_dynamics_hold_last_safe_satisfied"] != "1.0"
            ):
                def vector(prefix: str) -> list[float]:
                    suffixes = (
                        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"
                    )
                    return [float(row[f"{prefix}_{suffix}"]) for suffix in suffixes]

                events.append(
                    {
                        "task_time_s": float(row["time"]),
                        "ddq_des_rad_s2": vector("right_arm_ddq_des"),
                        "first_validated_qacc_rad_s2": vector("right_arm_first_pass_qacc_validated"),
                        "second_validated_qacc_rad_s2": vector("right_arm_second_pass_qacc_validated"),
                        "hold_last_available": row["right_arm_forward_dynamics_hold_last_safe_available"] == "1.0",
                        "safe_candidate_count": int(float(row["right_arm_forward_dynamics_safe_candidate_count"])),
                        "qacc_limit_rejections": int(float(row["right_arm_forward_dynamics_qacc_limit_rejections"])),
                        "max_abs_qacc_limit_rad_s2": 10.0,
                    }
                )
    return events


def _plot_window(runs: list[dict[str, Any]], output: Path, start: float, end: float) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for run in runs:
        data = run["_plot"]
        metric_mask = (data["metrics_time"] >= start) & (data["metrics_time"] < end)
        trajectory_mask = (data["time"] >= start) & (data["time"] < end)
        axes[0].plot(data["metrics_time"][metric_mask], data["tilt"][metric_mask], lw=0.8, label=run["mode"])
        axes[1].plot(data["time"][trajectory_mask], data["position"][trajectory_mask], lw=0.8, label=run["mode"])
    axes[0].set_ylabel("cup tilt [rad]")
    axes[1].set_ylabel("EE position error [m]")
    axes[1].set_xlabel("task time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run(run_paths: list[tuple[str, Path]], output_dir: Path) -> dict[str, Any]:
    runs = [_load_run(path, label) for label, path in run_paths]
    for item in runs:
        perf = item["_plot"]
        headline_perf = perf["perf_ms"][perf["perf_time"] < 8.0 - 1e-12]
        item["complete_6ms_timing_ms"]["p99"] = float(
            np.percentile(headline_perf, 99.0)
        )
    nominal = runs[0]
    heldout = runs[1:]
    nominal_pass = bool(
        nominal["safety_gate_pass"]
        and not nominal["stability"]["fallen"]
        and nominal["stability"]["nan_inf_count"] == 0
        and nominal["predictor_fallback"]["headline_count"] == 0
    )
    heldout_unsafe = sum(run["mapping"]["final_unsafe_count"] for run in heldout)
    status = "FAIL" if not nominal_pass else "PARTIAL" if heldout_unsafe else "PASS"
    output_dir.mkdir(parents=True, exist_ok=False)

    plot_paths = [
        output_dir / "v2_closed_loop_full_0_8s_tilt_position.png",
        output_dir / "v2_closed_loop_startup_0_2p4s_tilt_position.png",
        output_dir / "v2_closed_loop_stop_6p2_8p0s_tilt_position.png",
    ]
    for path, window in zip(plot_paths, ((0.0, 8.0), (0.0, 2.4), (6.2, 8.0))):
        _plot_window(runs, path, *window)

    error_path = output_dir / "v2_closed_loop_predictor_errors.png"
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fields = (
        ("interval_acc_prediction", "interval_acc_actual", "acc [m/s²]"),
        ("interval_alpha_prediction", "interval_alpha_actual", "alpha [rad/s²]"),
        ("interval_omega_prediction", "interval_omega_actual", "omega [rad/s]"),
    )
    for run in runs:
        data = run["_plot"]
        valid = data["interval_valid"]
        time = data["time"][valid]
        for axis, (prediction, actual, label) in zip(axes[:3], fields):
            error = np.linalg.norm(data[prediction][valid] - data[actual][valid], axis=1)
            axis.plot(time, error, lw=0.65, label=run["mode"])
            axis.set_ylabel(label + " error")
        ddq_valid = data["ddq_valid"]
        axes[3].plot(
            data["time"][ddq_valid],
            np.linalg.norm(data["ddq_des"][ddq_valid] - data["ddq_real"][ddq_valid], axis=1),
            lw=0.65,
            label=run["mode"],
        )
    axes[3].set_ylabel("ddq error [rad/s²]")
    axes[3].set_xlabel("task time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(error_path, dpi=170)
    plt.close(fig)
    plot_paths.append(error_path)

    timing_path = output_dir / "v2_closed_loop_timing_mapping_fallback.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    labels = [run["mode"] for run in runs]
    x = np.arange(len(runs))
    axes[0].bar(x - 0.2, [run["complete_6ms_timing_ms"]["mean"] for run in runs], 0.4, label="mean")
    axes[0].bar(x + 0.2, [run["complete_6ms_timing_ms"]["p95"] for run in runs], 0.4, label="p95")
    axes[0].scatter(x, [run["complete_6ms_timing_ms"]["max"] for run in runs], color="red", label="max")
    axes[0].axhline(6.0, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel("complete interval [ms]")
    axes[0].legend()
    axes[1].bar(x - 0.25, [run["mapping"]["rescue_used_count"] for run in runs], 0.25, label="rescue")
    axes[1].bar(x, [run["mapping"]["hold_last_succeeded_count"] for run in runs], 0.25, label="hold-last success")
    axes[1].bar(x + 0.25, [run["mapping"]["final_unsafe_count"] for run in runs], 0.25, label="final unsafe")
    axes[1].set_ylabel("event count")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(timing_path, dpi=170)
    plt.close(fig)
    plot_paths.append(timing_path)

    serializable_runs = []
    for run in runs:
        item = {key: value for key, value in run.items() if key != "_plot"}
        item["final_unsafe_events"] = _unsafe_events(Path(run["run_dir"]))
        serializable_runs.append(item)
    report = {
        "stage": "continuous-H full-task template v2 simulation acceptance",
        "status": status,
        "headline": "[0.0,8.0)",
        "nominal_gate_pass": nominal_pass,
        "heldout_final_unsafe_count": heldout_unsafe,
        "runs": serializable_runs,
        "plots": [str(path.resolve()) for path in plot_paths],
        "scope_disclosures": {
            "baseline": "fixed absolute-task-time template that knows the 6.4 s direct stop in advance",
            "not_general": "not validated for arbitrary speed, direction, or unknown stop time",
            "neural_route": "frozen exploratory work; no longer in the development plan",
            "hardware_blocker": "DDQ-to-torque final_unsafe is not fail-closed and remains an independent blocker before active hardware closed loop",
            "thresholds_changed": False,
        },
    }
    report_path = output_dir / "full_task_template_v2_stage_report.json"
    report_path.write_text(json.dumps(_json_value(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = output_dir / "full_task_template_v2_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run", "tilt_rms_rad", "tilt_p95_rad", "tilt_max_rad", "position_rms_m", "position_p95_m", "position_max_m", "ee_acc_rms_m_s2", "ee_alpha_rms_rad_s2", "pred_acc_rms", "pred_alpha_rms", "pred_omega_rms", "pred_orientation_rms_rad", "qp_fallback", "mapping_rescue", "mapping_hold_last_success", "final_unsafe", "predictor_fallback", "timing_mean_ms", "timing_p99_ms", "timing_max_ms", "timing_overrun", "xy_displacement_m"))
        for run in runs:
            writer.writerow((run["mode"], run["tilt_angle_rad"]["rms"], run["tilt_angle_rad"]["p95"], run["tilt_angle_rad"]["max"], run["position_error_norm_m"]["rms"], run["position_error_norm_m"]["p95"], run["position_error_norm_m"]["max"], run["right_ee_linear_acceleration_norm_m_s2"]["rms"], run["right_ee_angular_acceleration_norm_rad_s2"]["rms"], run["closed_loop_predictor_error"]["interval_acc_norm_m_s2"]["rms"], run["closed_loop_predictor_error"]["interval_alpha_norm_rad_s2"]["rms"], run["closed_loop_predictor_error"]["interval_omega_norm_rad_s"]["rms"], run["closed_loop_predictor_error"]["one_step_orientation_geodesic_rad"]["rms"], run["qp"]["fallback_count"], run["mapping"]["rescue_used_count"], run["mapping"]["hold_last_succeeded_count"], run["mapping"]["final_unsafe_count"], run["predictor_fallback"]["headline_count"], run["complete_6ms_timing_ms"]["mean"], run["complete_6ms_timing_ms"]["p99"], run["complete_6ms_timing_ms"]["max"], run["complete_6ms_timing_ms"]["overrun_count"], run["xy"]["displacement_m"]))
    return {"report": str(report_path), "metrics": str(csv_path), "status": status}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal", required=True)
    parser.add_argument("--heldout-01-plus", required=True)
    parser.add_argument("--heldout-01-minus", required=True)
    parser.add_argument("--heldout-02-plus", required=True)
    parser.add_argument("--heldout-02-minus", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT / "evaluation/t2_full_task_closed_loop" / f"{datetime.now():%Y%m%d_%H%M%S}_v2_stage_report"
    result = run(
        [
            ("nominal", Path(args.nominal).resolve()),
            ("heldout_pair_01_plus", Path(args.heldout_01_plus).resolve()),
            ("heldout_pair_01_minus", Path(args.heldout_01_minus).resolve()),
            ("heldout_pair_02_plus", Path(args.heldout_02_plus).resolve()),
            ("heldout_pair_02_minus", Path(args.heldout_02_minus).resolve()),
        ],
        output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
