#!/usr/bin/env python3
"""Run and summarize the first template/neural/hybrid closed-loop ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
MODES = ("template", "neural", "hybrid_residual")
STAGES = {
    "start": (0.8, 1.4),
    "steady": (1.4, 2.2),
    "velocity_change": (2.2, 3.0),
    "stop": (3.0, 3.6),
    "stopped": (3.6, 4.8),
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _nested(data: dict, *names: str) -> dict:
    value = data
    for name in names:
        value = value[name]
    return value


def _distribution(values: np.ndarray, scale: float = 1.0) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)] * scale
    if not finite.size:
        return {
            "count": 0,
            "mean": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _vector_rms(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[mask]
    return float(np.sqrt(np.mean(np.sum(selected * selected, axis=1))))


def _stage_metrics(
    trajectory: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    start: float,
    end: float,
) -> dict:
    trajectory_time = np.asarray(trajectory["time"], dtype=np.float64)
    metric_time = np.asarray(metrics["time"], dtype=np.float64)
    trajectory_mask = (trajectory_time >= start) & (trajectory_time < end)
    metric_mask = (metric_time >= start) & (metric_time < end)
    arm_updates = np.asarray(trajectory["arm_policy_updated"], dtype=bool)
    mpc_mask = trajectory_mask & arm_updates
    solver_success = np.asarray(
        trajectory["right_mpc_solver_success"], dtype=bool
    )
    fallback = np.asarray(
        trajectory["right_mpc_fallback_used"], dtype=bool
    )
    ddq_saturation = np.asarray(
        trajectory["right_arm_ddq_saturation_mask"], dtype=bool
    )
    return {
        "duration_s": float(end - start),
        "physics_sample_count": int(np.count_nonzero(trajectory_mask)),
        "mpc_update_count": int(np.count_nonzero(mpc_mask)),
        "right_ee_acc_norm_rms": _vector_rms(
            metrics["right_ee_lin_acc_world"], metric_mask
        ),
        "right_ee_alpha_norm_rms": _vector_rms(
            metrics["right_ee_ang_acc_world"], metric_mask
        ),
        "right_ee_tilt_xy_norm_rms": _vector_rms(
            np.asarray(metrics["right_ee_tilt_error"])[:, :2], metric_mask
        ),
        "torso_acc_norm_rms": _vector_rms(
            trajectory["torso_acc_world_used"], trajectory_mask
        ),
        "torso_alpha_norm_rms": _vector_rms(
            trajectory["torso_alpha_world_used"], trajectory_mask
        ),
        "qp_success_fraction": float(np.mean(solver_success[mpc_mask])),
        "qp_fallback_fraction": float(np.mean(fallback[mpc_mask])),
        "ddq_saturation_any_fraction": float(
            np.mean(np.any(ddq_saturation[trajectory_mask], axis=1))
        ),
    }


def _critical_nonfinite_count(
    trajectory: dict[str, np.ndarray], metrics: dict[str, np.ndarray]
) -> int:
    dense_names = (
        "right_arm_ddq_des",
        "right_arm_ctrl",
        "torso_acc_world_used",
        "torso_alpha_world_used",
    )
    count = sum(
        int(np.count_nonzero(~np.isfinite(trajectory[name])))
        for name in dense_names
    )
    arm_updates = np.asarray(trajectory["arm_policy_updated"], dtype=bool)
    for name in (
        "right_mpc_interval_acc_k0",
        "right_mpc_interval_alpha_k0",
    ):
        count += int(
            np.count_nonzero(~np.isfinite(trajectory[name][arm_updates]))
        )
    for name in (
        "right_ee_lin_acc_world",
        "right_ee_ang_acc_world",
        "right_ee_tilt_error",
    ):
        count += int(np.count_nonzero(~np.isfinite(metrics[name])))
    return count


def summarize_run(run_dir: Path, mode: str) -> dict:
    # trajectory.npz 含有本项目自行写入的字段名 object 数组；
    # 此处只读取刚在本地生成的可信评估文件。
    with np.load(run_dir / "trajectory.npz", allow_pickle=True) as source:
        trajectory = {name: source[name] for name in source.files}
    with np.load(run_dir / "metrics.npz", allow_pickle=False) as source:
        metrics = {name: source[name] for name in source.files}
    perf = _load_json(run_dir / "perf_summary.json")
    overall = _load_json(run_dir / "summary.json")
    mpc = _load_json(run_dir / "mpc_diagnostics.json")
    hardware = _nested(perf, "total", "real_hardware_control")
    predictor_timing = _nested(
        hardware, "mpc_breakdown", "disturbance_prediction_time"
    )
    interval_timing = hardware["right_arm_interval"]

    time_values = np.asarray(trajectory["time"], dtype=np.float64)
    arm_updates = np.asarray(trajectory["arm_policy_updated"], dtype=bool)
    evaluation = (time_values >= 0.8) & (time_values < 4.8) & arm_updates
    full_updates = arm_updates
    predictor_fallback = np.asarray(
        trajectory["right_mpc_predictor_fallback_used"], dtype=bool
    )
    neural_inference_valid = np.asarray(
        trajectory["right_mpc_predictor_neural_inference_valid"], dtype=bool
    )
    neural_inference_time = np.asarray(
        trajectory["right_mpc_predictor_neural_inference_time"],
        dtype=np.float64,
    )
    neural_timing_mask = evaluation & neural_inference_valid
    evaluation_dense = (time_values >= 0.8) & (time_values < 4.8)
    ddq_saturation = np.asarray(
        trajectory["right_arm_ddq_saturation_mask"], dtype=bool
    )

    result = {
        "mode": mode,
        "run_dir_local_ignored": str(run_dir.relative_to(REPO_DIR)),
        "overall_evaluation_0p8_to_4p8_s": {
            "right_ee_acc_norm_rms": float(overall["right_acc_rms"]),
            "right_ee_alpha_norm_rms": float(overall["right_alpha_rms"]),
            "right_ee_tilt_xy_norm_rms": float(overall["right_tilt_rms"]),
            "torso_acc_norm_rms": _vector_rms(
                trajectory["torso_acc_world_used"],
                evaluation_dense,
            ),
            "torso_alpha_norm_rms": _vector_rms(
                trajectory["torso_alpha_world_used"],
                evaluation_dense,
            ),
            "qp_success_fraction": float(mpc["solver"]["success_fraction"]),
            "qp_fallback_fraction": float(
                mpc["solver"]["fallback_fraction"]
            ),
            "ddq_saturation_any_fraction": float(
                np.mean(np.any(ddq_saturation[evaluation_dense], axis=1))
            ),
        },
        "by_stage": {
            name: _stage_metrics(trajectory, metrics, start, end)
            for name, (start, end) in STAGES.items()
        },
        "predictor_timing_ms": predictor_timing,
        "neural_core_inference_timing_ms": _distribution(
            neural_inference_time[neural_timing_mask], scale=1000.0
        ),
        "complete_6ms_right_arm_interval_ms": interval_timing,
        "fallback": {
            "full_run_count": int(
                np.count_nonzero(predictor_fallback[full_updates])
            ),
            "evaluation_count": int(
                np.count_nonzero(predictor_fallback[evaluation])
            ),
            "evaluation_fraction": float(
                np.mean(predictor_fallback[evaluation])
            ),
        },
        "safety": {
            "critical_nonfinite_count": _critical_nonfinite_count(
                trajectory, metrics
            ),
            "complete_interval_overrun_count": int(
                interval_timing["overrun_count"]
            ),
            "complete_interval_overrun_fraction": float(
                interval_timing["overrun_fraction"]
            ),
        },
    }
    return result


def _latest_run(group_dir: Path, label: str) -> Path:
    candidates = sorted(group_dir.glob(f"*_{label}"))
    if not candidates:
        raise FileNotFoundError(f"找不到 {label} 的闭环输出。")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run neural predictor ablation")
    parser.add_argument(
        "--group",
        default=f"neural_closed_loop_{datetime.now():%Y%m%d_%H%M%S}",
    )
    parser.add_argument(
        "--summary-dir",
        default="evaluation_summary/neural_closed_loop_ablation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用同 group 中已有 summary.json 的 mode run。",
    )
    args = parser.parse_args()
    group_dir = REPO_DIR / "evaluation" / args.group
    group_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for mode in MODES:
        existing = list(group_dir.glob(f"*_{mode}"))
        if args.resume and any(
            (path / "summary.json").is_file() for path in existing
        ):
            results.append(summarize_run(_latest_run(group_dir, mode), mode))
            continue
        log_path = group_dir / f"{mode}.log"
        command = [
            str(REPO_DIR / "run.sh"),
            "--headless",
            "--no-video",
            "--predictor-ablation",
            "--disturbance-predictor",
            mode,
            "--evaluation-group",
            args.group,
            "--run-label",
            mode,
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=REPO_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{mode} 闭环运行失败，returncode={completed.returncode}，"
                f"见 {log_path}"
            )
        results.append(summarize_run(_latest_run(group_dir, mode), mode))

    summary = {
        "stage": "first_neural_predictor_closed_loop_ablation",
        "modes": list(MODES),
        "command_schedule": {
            "heading_warmup_s": [0.0, 0.8],
            "start_s": list(STAGES["start"]),
            "steady_s": list(STAGES["steady"]),
            "velocity_change_s": list(STAGES["velocity_change"]),
            "stop_s": list(STAGES["stop"]),
            "stopped_s": list(STAGES["stopped"]),
            "start_command_vx_vy_wz": [0.5, 0.0, 0.0127],
            "changed_command_vx_vy_wz": [0.275, 0.10, -0.04],
            "heading_hold_disabled_for_training_match": True,
        },
        "repeats_per_mode": 1,
        "results": results,
    }
    by_mode = {result["mode"]: result for result in results}
    template_overall = by_mode["template"][
        "overall_evaluation_0p8_to_4p8_s"
    ]
    hybrid_overall = by_mode["hybrid_residual"][
        "overall_evaluation_0p8_to_4p8_s"
    ]
    quality_names = (
        "right_ee_acc_norm_rms",
        "right_ee_alpha_norm_rms",
        "right_ee_tilt_xy_norm_rms",
    )
    summary["hybrid_vs_template"] = {
        "relative_change_percent": {
            name: 100.0
            * (hybrid_overall[name] / template_overall[name] - 1.0)
            for name in quality_names
        },
        "qp_success_change_percentage_points": 100.0
        * (
            hybrid_overall["qp_success_fraction"]
            - template_overall["qp_success_fraction"]
        ),
        "ddq_saturation_change_percentage_points": 100.0
        * (
            hybrid_overall["ddq_saturation_any_fraction"]
            - template_overall["ddq_saturation_any_fraction"]
        ),
        "all_three_quality_metrics_better_in_every_stage": all(
            by_mode["hybrid_residual"]["by_stage"][stage][name]
            < by_mode["template"]["by_stage"][stage][name]
            for stage in STAGES
            for name in quality_names
        ),
        "current_best_mode": "hybrid_residual",
        "evidence_scope": (
            "one deterministic 4.8 s MuJoCo closed-loop run per mode; "
            "not hardware evidence"
        ),
    }
    summary_dir = Path(args.summary_dir)
    if not summary_dir.is_absolute():
        summary_dir = REPO_DIR / summary_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
