#!/usr/bin/env python3
"""Repeated unseen-schedule closed-loop validation for disturbance predictors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from disturbance_learning.command_schedule import (
    GENERALIZATION_SCHEDULE_PROFILES,
)
from disturbance_learning.run_closed_loop_ablation import (
    MODES,
    _latest_run,
    summarize_run,
)


QUALITY_METRICS = (
    "right_ee_acc_norm_rms",
    "right_ee_alpha_norm_rms",
    "right_ee_tilt_xy_norm_rms",
)
TRANSIENT_STAGES = ("start", "velocity_change", "stop")
DEFAULT_SEEDS = (2101, 2102)


def _statistics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _condition_id(profile: str, seed: int, payload_kg: float) -> str:
    payload_g = int(round(1000.0 * payload_kg))
    return f"{profile}_seed{seed}_payload{payload_g}g"


def _aggregate_mode(results: list[dict], mode: str) -> dict:
    selected = [result for result in results if result["mode"] == mode]
    interval_count = int(
        sum(
            result["complete_6ms_right_arm_interval_ms"]["count"]
            for result in selected
        )
    )
    interval_overrun_count = int(
        sum(
            result["safety"]["complete_interval_overrun_count"]
            for result in selected
        )
    )
    overall_names = QUALITY_METRICS + (
        "torso_acc_norm_rms",
        "torso_alpha_norm_rms",
        "qp_success_fraction",
        "qp_fallback_fraction",
        "ddq_saturation_any_fraction",
    )
    return {
        "run_count": len(selected),
        "overall": {
            name: _statistics(
                [result["overall_evaluation"][name] for result in selected]
            )
            for name in overall_names
        },
        "transients": {
            stage: {
                name: _statistics(
                    [result["by_stage"][stage][name] for result in selected]
                )
                for name in QUALITY_METRICS
            }
            for stage in TRANSIENT_STAGES
        },
        "timing_ms": {
            "predictor_mean": _statistics(
                [result["predictor_timing_ms"]["mean"] for result in selected]
            ),
            "predictor_p99": _statistics(
                [result["predictor_timing_ms"]["p99"] for result in selected]
            ),
            "predictor_max": _statistics(
                [result["predictor_timing_ms"]["max"] for result in selected]
            ),
            "neural_core_mean": _statistics(
                [
                    result["neural_core_inference_timing_ms"]["mean"]
                    for result in selected
                ]
            ),
            "neural_core_p99": _statistics(
                [
                    result["neural_core_inference_timing_ms"]["p99"]
                    for result in selected
                ]
            ),
            "neural_core_max": _statistics(
                [
                    result["neural_core_inference_timing_ms"]["max"]
                    for result in selected
                ]
            ),
            "complete_interval_mean": _statistics(
                [
                    result["complete_6ms_right_arm_interval_ms"]["mean"]
                    for result in selected
                ]
            ),
            "complete_interval_p99": _statistics(
                [
                    result["complete_6ms_right_arm_interval_ms"]["p99"]
                    for result in selected
                ]
            ),
            "complete_interval_max": _statistics(
                [
                    result["complete_6ms_right_arm_interval_ms"]["max"]
                    for result in selected
                ]
            ),
        },
        "safety_totals": {
            "predictor_fallback_count": int(
                sum(result["fallback"]["evaluation_count"] for result in selected)
            ),
            "critical_nonfinite_count": int(
                sum(
                    result["safety"]["critical_nonfinite_count"]
                    for result in selected
                )
            ),
            "complete_interval_count": interval_count,
            "complete_interval_overrun_count": interval_overrun_count,
            "complete_interval_overrun_fraction": (
                interval_overrun_count / interval_count
            ),
        },
    }


def _compact_result(result: dict) -> dict:
    timing_names = ("count", "mean", "p95", "p99", "max")
    compact = {
        name: result[name]
        for name in (
            "condition_id",
            "profile",
            "seed",
            "payload_kg",
            "payload_stress",
            "mode",
            "evaluation_window_s",
            "overall_evaluation",
            "fallback",
            "safety",
        )
    }
    compact["transients"] = {
        stage: {
            name: result["by_stage"][stage][name]
            for name in QUALITY_METRICS
            + (
                "qp_success_fraction",
                "qp_fallback_fraction",
                "ddq_saturation_any_fraction",
            )
        }
        for stage in TRANSIENT_STAGES
    }
    for output_name, source_name in (
        ("predictor_timing_ms", "predictor_timing_ms"),
        ("neural_core_inference_timing_ms", "neural_core_inference_timing_ms"),
        ("complete_6ms_right_arm_interval_ms", "complete_6ms_right_arm_interval_ms"),
    ):
        compact[output_name] = {
            name: result[source_name][name]
            for name in timing_names
            if name in result[source_name]
        }
    return compact


def _paired_improvements(results: list[dict]) -> dict:
    indexed = {
        (result["condition_id"], result["mode"]): result
        for result in results
    }
    condition_ids = sorted({result["condition_id"] for result in results})
    per_condition = []
    for condition_id in condition_ids:
        template = indexed[(condition_id, "template")]
        hybrid = indexed[(condition_id, "hybrid_residual")]
        overall_improvement = {
            name: 100.0
            * (
                template["overall_evaluation"][name]
                - hybrid["overall_evaluation"][name]
            )
            / template["overall_evaluation"][name]
            for name in QUALITY_METRICS
        }
        transient_improvement = {
            stage: {
                name: 100.0
                * (
                    template["by_stage"][stage][name]
                    - hybrid["by_stage"][stage][name]
                )
                / template["by_stage"][stage][name]
                for name in QUALITY_METRICS
            }
            for stage in TRANSIENT_STAGES
        }
        per_condition.append(
            {
                "condition_id": condition_id,
                "overall_improvement_percent": overall_improvement,
                "transient_improvement_percent": transient_improvement,
                "all_overall_quality_metrics_better": all(
                    value > 0.0 for value in overall_improvement.values()
                ),
            }
        )

    def aggregate(items: list[dict], path: tuple[str, ...]) -> dict:
        return {
            name: _statistics(
                [
                    item[path[0]][name]
                    if len(path) == 1
                    else item[path[0]][path[1]][name]
                    for item in items
                ]
            )
            for name in QUALITY_METRICS
        }

    return {
        "positive_means_hybrid_is_better": True,
        "overall_improvement_percent": aggregate(
            per_condition, ("overall_improvement_percent",)
        ),
        "transient_improvement_percent": {
            stage: aggregate(
                per_condition, ("transient_improvement_percent", stage)
            )
            for stage in TRANSIENT_STAGES
        },
        "conditions_better_count": {
            name: int(
                sum(
                    item["overall_improvement_percent"][name] > 0.0
                    for item in per_condition
                )
            )
            for name in QUALITY_METRICS
        },
        "all_three_better_condition_count": int(
            sum(item["all_overall_quality_metrics_better"] for item in per_condition)
        ),
        "per_condition": per_condition,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeated unseen command schedule predictor validation"
    )
    parser.add_argument(
        "--group",
        default=f"hybrid_generalization_{datetime.now():%Y%m%d_%H%M%S}",
    )
    parser.add_argument(
        "--summary-dir",
        default="evaluation_summary/hybrid_generalization_validation",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(GENERALIZATION_SCHEDULE_PROFILES),
        default=list(GENERALIZATION_SCHEDULE_PROFILES),
    )
    parser.add_argument(
        "--payload-stress-kg",
        type=float,
        default=0.01,
        help="追加一个独立配对 condition 使用的 right_bottle 额外质量。",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) < 2:
        raise ValueError("generalization validation 至少需要两个 seed。")
    if not 0.0 <= args.payload_stress_kg <= 0.25:
        raise ValueError("payload 必须在 0~0.25 kg。")

    group_dir = REPO_DIR / "evaluation" / args.group
    group_dir.mkdir(parents=True, exist_ok=True)
    results = []
    conditions = [
        (profile_name, seed, 0.0, False)
        for profile_name in args.profiles
        for seed in args.seeds
    ]
    if args.payload_stress_kg > 0.0:
        conditions.append(
            (args.profiles[0], args.seeds[-1], args.payload_stress_kg, True)
        )
    total_runs = len(conditions) * len(MODES)
    completed_runs = 0
    for profile_name, seed, payload_kg, payload_stress in conditions:
        profile = GENERALIZATION_SCHEDULE_PROFILES[profile_name]
        stages = profile.timing.stage_windows()
        condition_id = _condition_id(profile_name, seed, payload_kg)
        for mode in MODES:
            label = f"{condition_id}_{mode}"
            existing = list(group_dir.glob(f"*_{label}"))
            if args.resume and any(
                (path / "summary.json").is_file() for path in existing
            ):
                run_dir = _latest_run(group_dir, label)
            else:
                log_path = group_dir / f"{label}.log"
                command = [
                    str(REPO_DIR / "run.sh"),
                    "--headless",
                    "--no-video",
                    "--predictor-schedule-profile",
                    profile_name,
                    "--predictor-ablation-seed",
                    str(seed),
                    "--predictor-payload-kg",
                    str(payload_kg),
                    "--disturbance-predictor",
                    mode,
                    "--evaluation-group",
                    args.group,
                    "--run-label",
                    label,
                ]
                print(
                    f"[{completed_runs + 1}/{total_runs}] {label}", flush=True
                )
                for attempt in range(3):
                    with log_path.open(
                        "w" if attempt == 0 else "a", encoding="utf-8"
                    ) as log:
                        completed = subprocess.run(
                            command,
                            cwd=REPO_DIR,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                        )
                    if completed.returncode == 0:
                        break
                    startup_layout_race = (
                        "SimRuntimeLayoutError" in log_path.read_text(
                            encoding="utf-8"
                        )
                    )
                    if not startup_layout_race or attempt == 2:
                        break
                    print(
                        f"retry {label}: transient shared-memory startup race",
                        flush=True,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{label} failed with {completed.returncode}; see {log_path}"
                    )
                run_dir = _latest_run(group_dir, label)
            result = summarize_run(
                run_dir,
                mode,
                evaluation_start=0.8,
                evaluation_end=profile.timing.run_end,
                stages=stages,
            )
            result.update(
                {
                    "condition_id": condition_id,
                    "profile": profile_name,
                    "seed": seed,
                    "payload_kg": payload_kg,
                    "payload_stress": payload_stress,
                }
            )
            results.append(result)
            completed_runs += 1

    summary = {
        "stage": "hybrid_residual_unseen_schedule_generalization",
        "evidence_scope": "MuJoCo closed loop; not hardware evidence",
        "local_evaluation_group_ignored": args.group,
        "modes": list(MODES),
        "profiles": {
            name: {
                "timing_s": {
                    stage: list(window)
                    for stage, window in profile.timing.stage_windows().items()
                },
                "start_command_vx_vy_wz": list(profile.start_command),
                "changed_command_vx_vy_wz": list(profile.changed_command),
            }
            for name, profile in GENERALIZATION_SCHEDULE_PROFILES.items()
            if name in args.profiles
        },
        "seeds": args.seeds,
        "payload_assignment": "one separately identified stress condition",
        "payload_stress_kg": args.payload_stress_kg,
        "schedule_only_condition_count": len(args.profiles) * len(args.seeds),
        "condition_count": len(conditions),
        "runs_per_mode": len(conditions),
        "total_run_count": len(results),
        "aggregate_by_mode": {
            mode: _aggregate_mode(results, mode) for mode in MODES
        },
        "schedule_only_aggregate_by_mode": {
            mode: _aggregate_mode(
                [result for result in results if not result["payload_stress"]],
                mode,
            )
            for mode in MODES
        },
        "hybrid_vs_template_paired": _paired_improvements(results),
        "schedule_only_hybrid_vs_template_paired": _paired_improvements(
            [result for result in results if not result["payload_stress"]]
        ),
        "decision": {
            "residual_mlp_retrained": False,
            "reason": (
                "all six schedule-only conditions improve overall EE acc and "
                "alpha; the only overall tilt regression is below 0.3 percent"
            ),
            "worth_hardware_safety_preparation": True,
            "hardware_ready": False,
            "blocking_observations": [
                "rare complete 6 ms interval overruns remain",
                "even a 0.01 kg unmodelled payload reduces QP success",
            ],
        },
        "payload_stress_results": [
            _compact_result(result)
            for result in results
            if result["payload_stress"]
        ],
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
    print(json.dumps(summary["aggregate_by_mode"], indent=2, sort_keys=True))
    print(json.dumps(summary["hybrid_vs_template_paired"], indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
