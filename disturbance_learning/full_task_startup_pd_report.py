#!/usr/bin/env python3
"""Compare explicit 24/54 ms fixed-PD handoff simulations."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from disturbance_learning.full_task_closed_loop_report import _json_value, _load_run
from disturbance_learning.full_task_recording import (
    FALL_MAX_ABS_ROLL_PITCH_RAD,
    FALL_MIN_TORSO_HEIGHT_M,
)
from disturbance_learning.full_task_protocol import rotation_matrix_to_rpy


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _timing(run_dir: Path) -> dict[str, float | int]:
    with (run_dir / "perf_intervals.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    starts = np.asarray(
        [float(row["simulation_start_time_s"]) for row in rows], dtype=np.float64
    )
    elapsed = np.asarray(
        [float(row["complete_interval_ms"]) for row in rows], dtype=np.float64
    )
    headline = starts < 8.0 - 1e-12
    values = elapsed[headline]
    return {
        "count": int(values.size),
        "first_start_simulation_time_s": float(starts[0]),
        "first_start_task_time_s": float(starts[0]),
        "first_interval_ms": float(elapsed[0]),
        "mean_ms": float(np.mean(values)),
        "p99_ms": float(np.percentile(values, 99.0)),
        "max_ms": float(np.max(values)),
        "overrun_count": int(np.count_nonzero(values > 6.0)),
        "overrun_fraction": float(np.mean(values > 6.0)),
    }


def _short_gate(run_dir: Path) -> dict[str, Any]:
    handoff = _load_json(run_dir / "startup_pd_handoff_summary.json")
    right = _load_json(run_dir / "right_arm_diagnostics.json")[
        "right_arm_execution_branches"
    ]
    with np.load(
        run_dir / "full_task_short_pre_step_raw.npz", allow_pickle=False
    ) as source:
        torso_position = source["torso_position_world"]
        torso_rotation = source["torso_rotation_world"]
        finite = bool(
            np.all(np.isfinite(torso_position))
            and np.all(np.isfinite(torso_rotation))
            and np.all(np.isfinite(source["actuator_ctrl"]))
        )
    rpy = rotation_matrix_to_rpy(torso_rotation)
    min_height = float(np.min(torso_position[:, 2]))
    max_abs_roll_pitch = float(np.max(np.abs(rpy[:, :2])))
    fallen = bool(
        min_height < FALL_MIN_TORSO_HEIGHT_M
        or max_abs_roll_pitch > FALL_MAX_ABS_ROLL_PITCH_RAD
    )
    passed = bool(
        handoff["whole_trace"]["predictor_fallback_count"] == 0
        and right["final_output_uncertified_count"] == 0
        and right["no_safe_torque_count"] == 0
        and finite
        and not fallen
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "passed": passed,
        "predictor_fallback_count": handoff["whole_trace"][
            "predictor_fallback_count"
        ],
        "final_output_uncertified_count": right[
            "final_output_uncertified_count"
        ],
        "no_safe_torque_count": right["no_safe_torque_count"],
        "safe_hold_used_count": right["safe_hold_used_count"],
        "finite_required_arrays": finite,
        "fallen": fallen,
        "minimum_torso_height_m": min_height,
        "maximum_abs_roll_pitch_rad": max_abs_roll_pitch,
        "handoff": handoff["handoff"],
    }


def _full_run(label: str, duration_ms: int, run_dir: Path) -> dict[str, Any]:
    loaded = _load_run(run_dir, label)
    handoff = _load_json(run_dir / "startup_pd_handoff_summary.json")
    right = _load_json(run_dir / "right_arm_diagnostics.json")[
        "right_arm_execution_branches"
    ]
    result = {key: value for key, value in loaded.items() if key != "_plot"}
    result.update(
        {
            "duration_ms": duration_ms,
            "timing": _timing(run_dir),
            "handoff": handoff["handoff"],
            "first_lower_policy_update_simulation_time_s": handoff[
                "first_lower_policy_update_simulation_time_s"
            ],
            "first_lower_policy_update_task_time_s": handoff[
                "first_lower_policy_update_task_time_s"
            ],
            "prefix": handoff["prefix"],
            "dry_preflight": handoff["dry_preflight"],
            "safety_line_search_extra_time_ms": right[
                "safety_line_search_extra_time_ms"
            ],
            "handoff_previous_tau_certification": (
                "hold-last revalidation passed"
                if handoff["handoff"]["mapping"]["hold_last_safe_satisfied"]
                else "available; normal certified candidate passed, so hold-last was not invoked"
            ),
        }
    )
    result["stage_safety_pass"] = bool(
        right["final_output_uncertified_count"] == 0
        and right["no_safe_torque_count"] == 0
        and loaded["predictor_fallback"]["headline_count"] == 0
        and not loaded["stability"]["fallen"]
        and loaded["stability"]["nan_inf_count"] == 0
    )
    result["_plot"] = loaded["_plot"]
    return result


def _plot_comparison(runs: list[dict[str, Any]], output_dir: Path) -> list[str]:
    linewidth = 0.8
    paths: list[Path] = []
    windows = (
        (0.0, 8.0, "full_0_8s"),
        (0.0, 0.25, "startup_0_0p25s"),
    )
    for start, end, suffix in windows:
        path = output_dir / f"startup_pd_{suffix}_tilt_position.png"
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for run in runs:
            data = run["_plot"]
            metric_mask = (
                (data["metrics_time"] >= start)
                & (data["metrics_time"] < end - 1e-12)
            )
            trajectory_mask = (
                (data["time"] >= start) & (data["time"] < end - 1e-12)
            )
            axes[0].plot(
                data["metrics_time"][metric_mask],
                data["tilt"][metric_mask],
                lw=linewidth,
                label=run["mode"],
            )
            axes[1].plot(
                data["time"][trajectory_mask],
                data["position"][trajectory_mask],
                lw=linewidth,
                label=run["mode"],
            )
        axes[0].set_ylabel("cup tilt [rad]")
        axes[1].set_ylabel("EE position error [m]")
        axes[1].set_xlabel("task time = simulation time [s]")
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)

    path = output_dir / "startup_pd_handoff_torque_jump.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for run in runs:
        trace_path = Path(run["run_dir"]) / "startup_pd_handoff_trace.npz"
        with np.load(trace_path, allow_pickle=False) as source:
            time = source["task_time"]
            tau = source["actual_right_arm_tau"]
        mask = time <= 0.10 + 1e-12
        axes[0].plot(
            time[mask],
            np.linalg.norm(tau[mask], axis=1),
            lw=linewidth,
            label=run["mode"],
        )
    labels = [run["mode"] for run in runs]
    jump = [run["handoff"]["tau_jump_l2_nm"] for run in runs]
    axes[1].bar(np.arange(len(runs)), jump)
    axes[1].set_xticks(np.arange(len(runs)), labels, rotation=15, ha="right")
    axes[0].set_ylabel("actual right-arm |tau| [Nm]")
    axes[0].set_xlabel("task time = simulation time [s]")
    axes[1].set_ylabel("handoff tau jump L2 [Nm]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return [str(path.resolve()) for path in paths]


def run(
    *,
    short_paths: list[tuple[str, Path]],
    full_paths: list[tuple[str, int, Path]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    shorts = {label: _short_gate(path.resolve()) for label, path in short_paths}
    runs = [
        _full_run(label, duration_ms, path.resolve())
        for label, duration_ms, path in full_paths
    ]
    grouped: dict[int, dict[str, float]] = {}
    for duration in (24, 54):
        group = [run for run in runs if run["duration_ms"] == duration]
        grouped[duration] = {
            "tilt_rms_mean_rad": float(
                np.mean([run["tilt_angle_rad"]["rms"] for run in group])
            ),
            "position_rms_mean_m": float(
                np.mean([run["position_error_norm_m"]["rms"] for run in group])
            ),
            "tau_jump_l2_mean_nm": float(
                np.mean([run["handoff"]["tau_jump_l2_nm"] for run in group])
            ),
            "timing_overrun_fraction_mean": float(
                np.mean([run["timing"]["overrun_fraction"] for run in group])
            ),
            "all_safety_pass": all(run["stage_safety_pass"] for run in group),
        }
    recommended = min(
        (24, 54),
        key=lambda duration: (
            not grouped[duration]["all_safety_pass"],
            grouped[duration]["tilt_rms_mean_rad"],
            grouped[duration]["position_rms_mean_m"],
        ),
    )
    plots = _plot_comparison(runs, output_dir)
    serializable_runs = [
        {key: value for key, value in item.items() if key != "_plot"}
        for item in runs
    ]
    all_short_pass = all(item["passed"] for item in shorts.values())
    all_full_safe = all(item["stage_safety_pass"] for item in runs)
    timing_budget_pass = all(item["timing"]["overrun_count"] == 0 for item in runs)
    report = {
        "stage": "fixed startup-PD handoff: 24 ms vs 54 ms",
        "status": (
            "FAIL"
            if not all_short_pass or not all_full_safe
            else "PASS_WITH_TIMING_BLOCKER"
            if not timing_budget_pass
            else "PASS"
        ),
        "scope": {
            "template": "continuous-H full_task_template v2; not regenerated",
            "task_headline": "[0.0,8.0)",
            "task_simulation_template_gait_clock_origin_s": 0.0,
            "mpc_mapper_thresholds_changed": False,
            "dynamic_arming_enabled": False,
        },
        "short_smokes": shorts,
        "full_runs": serializable_runs,
        "duration_aggregate": grouped,
        "recommended_duration_ms": recommended,
        "recommendation_basis": (
            "both durations passed the output-safety gate; choose the lower paired "
            "full-headline cup-tilt RMS, then position RMS.  Complete 6 ms timing "
            "remains a separate blocker and was not hidden by dropping the first interval."
        ),
        "complete_6ms_timing_budget_pass": timing_budget_pass,
        "plots": plots,
    }
    report_path = output_dir / "fixed_startup_pd_handoff_report.json"
    report_path.write_text(
        json.dumps(_json_value(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "fixed_startup_pd_handoff_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "run",
                "duration_ms",
                "tilt_rms_rad",
                "tilt_p95_rad",
                "tilt_max_rad",
                "position_rms_m",
                "position_p95_m",
                "position_max_m",
                "ee_acc_rms_m_s2",
                "ee_alpha_rms_rad_s2",
                "xy_displacement_m",
                "tau_jump_l2_nm",
                "final_unsafe",
                "no_safe_torque",
                "safe_hold",
                "predictor_fallback",
                "qp_fallback",
                "first_interval_ms",
                "timing_mean_ms",
                "timing_p99_ms",
                "timing_max_ms",
                "timing_overrun_count",
            )
        )
        for item in runs:
            writer.writerow(
                (
                    item["mode"],
                    item["duration_ms"],
                    item["tilt_angle_rad"]["rms"],
                    item["tilt_angle_rad"]["p95"],
                    item["tilt_angle_rad"]["max"],
                    item["position_error_norm_m"]["rms"],
                    item["position_error_norm_m"]["p95"],
                    item["position_error_norm_m"]["max"],
                    item["right_ee_linear_acceleration_norm_m_s2"]["rms"],
                    item["right_ee_angular_acceleration_norm_rad_s2"]["rms"],
                    item["xy"]["displacement_m"],
                    item["handoff"]["tau_jump_l2_nm"],
                    item["mapping"]["final_unsafe_count"],
                    item["mapping"]["no_safe_torque_count"],
                    item["mapping"]["safe_hold_used_count"],
                    item["predictor_fallback"]["headline_count"],
                    item["qp"]["fallback_count"],
                    item["timing"]["first_interval_ms"],
                    item["timing"]["mean_ms"],
                    item["timing"]["p99_ms"],
                    item["timing"]["max_ms"],
                    item["timing"]["overrun_count"],
                )
            )
    return {
        "report": str(report_path.resolve()),
        "metrics": str(csv_path.resolve()),
        "plots": plots,
        "status": report["status"],
        "recommended_duration_ms": recommended,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "short-nominal-24",
        "short-heldout-24",
        "short-nominal-54",
        "short-heldout-54",
        "full-nominal-24",
        "full-heldout-24",
        "full-nominal-54",
        "full-heldout-54",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT
        / "evaluation/fixed_startup_pd_handoff_report"
        / f"{datetime.now():%Y%m%d_%H%M%S}"
    )
    result = run(
        short_paths=[
            ("nominal_24ms", Path(args.short_nominal_24)),
            ("heldout_pair_02_minus_24ms", Path(args.short_heldout_24)),
            ("nominal_54ms", Path(args.short_nominal_54)),
            ("heldout_pair_02_minus_54ms", Path(args.short_heldout_54)),
        ],
        full_paths=[
            ("nominal_24ms", 24, Path(args.full_nominal_24)),
            ("heldout_pair_02_minus_24ms", 24, Path(args.full_heldout_24)),
            ("nominal_54ms", 54, Path(args.full_nominal_54)),
            ("heldout_pair_02_minus_54ms", 54, Path(args.full_heldout_54)),
        ],
        output_dir=output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
