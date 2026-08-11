#!/usr/bin/env python3
"""Diagnose the two remaining pre-hardware readiness blockers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from disturbance_learning.command_schedule import (
    GENERALIZATION_SCHEDULE_PROFILES,
)
from disturbance_learning.run_closed_loop_ablation import (
    _latest_run,
    summarize_run,
)
from disturbance_learning.run_generalization_ablation import _aggregate_mode


DEFAULT_SEEDS = (2101, 2102)
DEFAULT_PAYLOAD_SEEDS = (2101, 2102, 2103, 2104)
DEFAULT_PAYLOADS_KG = (0.005, 0.010)
PREVIOUS_READINESS_SUMMARY = (
    REPO_DIR / "evaluation_summary/real_robot_readiness/summary.json"
)


def _governor(cpu: int) -> str:
    path = Path(
        f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
    )
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def _run_one(
    *,
    group: str,
    group_dir: Path,
    profile_name: str,
    seed: int,
    payload_kg: float,
    payload_modeling: str,
    label_kind: str,
    control_cpu: int,
    resume: bool,
) -> tuple[Path, dict]:
    payload_g = int(round(1000.0 * payload_kg))
    label = (
        f"{label_kind}_{profile_name}_seed{seed}_payload{payload_g}g_"
        f"{payload_modeling}_hybrid_residual"
    )
    existing = list(group_dir.glob(f"*_{label}"))
    if resume and any((path / "summary.json").is_file() for path in existing):
        run_dir = _latest_run(group_dir, label)
    else:
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
            "--predictor-payload-modeling",
            payload_modeling,
            "--disturbance-predictor",
            "hybrid_residual",
            "--evaluation-group",
            group,
            "--run-label",
            label,
        ]
        environment = os.environ.copy()
        environment["MPC_CONTROL_CPU"] = str(control_cpu)
        log_path = group_dir / f"{label}.log"
        for attempt in range(3):
            with log_path.open(
                "w" if attempt == 0 else "a", encoding="utf-8"
            ) as log:
                completed = subprocess.run(
                    command,
                    cwd=REPO_DIR,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode == 0:
                break
            layout_race = "SimRuntimeLayoutError" in log_path.read_text(
                encoding="utf-8"
            )
            if not layout_race or attempt == 2:
                raise RuntimeError(
                    f"{label} failed with {completed.returncode}; see {log_path}"
                )
        run_dir = _latest_run(group_dir, label)

    profile = GENERALIZATION_SCHEDULE_PROFILES[profile_name]
    result = summarize_run(
        run_dir,
        "hybrid_residual",
        evaluation_start=0.8,
        evaluation_end=profile.timing.run_end,
        stages=profile.timing.stage_windows(),
    )
    metadata = json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    mpc_diagnostics = json.loads(
        (run_dir / "mpc_diagnostics.json").read_text(encoding="utf-8")
    )
    arm_diagnostics = json.loads(
        (run_dir / "right_arm_diagnostics.json").read_text(encoding="utf-8")
    )
    result.update(
        {
            "profile": profile_name,
            "seed": seed,
            "payload_kg": payload_kg,
            "payload_modeling": payload_modeling,
            "payload_metadata": metadata["disturbance_command_schedule"][
                "payload"
            ],
            "runtime_environment": metadata["runtime_timing_environment"],
            "payload_cause_diagnostics": {
                "forward_dynamics_model_error_norm_rms": arm_diagnostics[
                    "right_arm_qacc_mapping_model_error_norm_rms"
                ],
                "current_q_violation_fraction": mpc_diagnostics["solver"][
                    "current_q_violation_fraction"
                ],
                "current_q_violation_max_rad": mpc_diagnostics["solver"][
                    "current_q_violation_max_rad"
                ],
                "recovery_active_fraction": mpc_diagnostics["solver"][
                    "recovery_active_fraction"
                ],
                "solver_status_val_counts": mpc_diagnostics["solver"][
                    "status_val_counts"
                ],
            },
        }
    )
    return run_dir, result


def _compact(result: dict) -> dict:
    return {
        name: result[name]
        for name in (
            "profile",
            "seed",
            "payload_kg",
            "payload_modeling",
            "run_dir_local_ignored",
            "payload_metadata",
            "runtime_environment",
            "payload_cause_diagnostics",
            "overall_evaluation",
            "complete_6ms_right_arm_interval_ms",
            "predictor_timing_ms",
            "safety",
            "safety_gate",
        )
    }


def _payload_comparison(results: list[dict], payload_kg: float) -> dict:
    selected = [
        result
        for result in results
        if abs(result["payload_kg"] - payload_kg) < 1e-12
    ]
    grouped = {
        modeling: _aggregate_mode(
            [item for item in selected if item["payload_modeling"] == modeling],
            "hybrid_residual",
        )
        for modeling in ("unmodeled", "modeled")
    }
    unmodeled = grouped["unmodeled"]["overall"]
    modeled = grouped["modeled"]["overall"]

    def cause_mean(modeling: str, name: str) -> float:
        values = [
            item["payload_cause_diagnostics"][name]
            for item in selected
            if item["payload_modeling"] == modeling
        ]
        return sum(values) / len(values)

    return {
        "by_modeling": grouped,
        "modeled_minus_unmodeled_qp_success_percentage_points": 100.0
        * (
            modeled["qp_success_fraction"]["mean"]
            - unmodeled["qp_success_fraction"]["mean"]
        ),
        "modeled_relative_change_percent": {
            name: 100.0
            * (modeled[name]["mean"] / unmodeled[name]["mean"] - 1.0)
            for name in (
                "right_ee_acc_norm_rms",
                "right_ee_alpha_norm_rms",
                "right_ee_tilt_xy_norm_rms",
                "ddq_saturation_any_fraction",
            )
        },
        "cause_diagnostics_mean": {
            modeling: {
                name: cause_mean(modeling, name)
                for name in (
                    "forward_dynamics_model_error_norm_rms",
                    "current_q_violation_fraction",
                    "current_q_violation_max_rad",
                    "recovery_active_fraction",
                )
            }
            for modeling in ("unmodeled", "modeled")
        },
    }


def _timing_reference_comparison(current: dict | None) -> dict:
    if current is None or not PREVIOUS_READINESS_SUMMARY.is_file():
        return {"available": False}
    previous_summary = json.loads(
        PREVIOUS_READINESS_SUMMARY.read_text(encoding="utf-8")
    )
    previous = previous_summary["aggregate_by_condition_group"]["normal"][
        "hybrid_residual"
    ]

    def compare(name: str, statistic: str = "mean") -> dict:
        before = previous["timing_ms"][name][statistic]
        after = current["timing_ms"][name][statistic]
        return {
            "previous_powersave": before,
            "performance": after,
            "change_percent": 100.0 * (after / before - 1.0),
        }

    previous_safety = previous["safety_totals"]
    current_safety = current["safety_totals"]
    return {
        "available": True,
        "previous_source": str(
            PREVIOUS_READINESS_SUMMARY.relative_to(REPO_DIR)
        ),
        "previous_run_count": previous["run_count"],
        "performance_run_count": current["run_count"],
        "complete_interval_mean_ms": compare("complete_interval_mean"),
        "complete_interval_p99_ms": compare("complete_interval_p99"),
        "complete_interval_worst_max_ms": compare(
            "complete_interval_max", "max"
        ),
        "predictor_mean_ms": compare("predictor_mean"),
        "predictor_p99_ms": compare("predictor_p99"),
        "overrun": {
            "previous_count": previous_safety[
                "complete_interval_overrun_count"
            ],
            "previous_interval_count": previous_safety[
                "complete_interval_count"
            ],
            "performance_count": current_safety[
                "complete_interval_overrun_count"
            ],
            "performance_interval_count": current_safety[
                "complete_interval_count"
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timing environment and matched-payload diagnostics"
    )
    parser.add_argument(
        "--group",
        default=f"readiness_blockers_{datetime.now():%Y%m%d_%H%M%S}",
    )
    parser.add_argument(
        "--summary-dir",
        default="evaluation_summary/readiness_blocker_diagnostics",
    )
    parser.add_argument(
        "--timing-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--payload-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_PAYLOAD_SEEDS),
    )
    parser.add_argument(
        "--timing-profiles",
        nargs="+",
        choices=tuple(GENERALIZATION_SCHEDULE_PROFILES),
        default=list(GENERALIZATION_SCHEDULE_PROFILES),
    )
    parser.add_argument(
        "--payload-profile",
        choices=tuple(GENERALIZATION_SCHEDULE_PROFILES),
        default="delayed_fast_lateral",
    )
    parser.add_argument(
        "--payloads-kg",
        nargs="+",
        type=float,
        default=list(DEFAULT_PAYLOADS_KG),
    )
    parser.add_argument("--control-cpu", type=int, default=7)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--skip-payload", action="store_true")
    parser.add_argument(
        "--reuse-timing-group",
        help="reuse completed timing runs from this ignored evaluation group",
    )
    parser.add_argument(
        "--reuse-payload-group",
        help="reuse completed payload runs from this ignored evaluation group",
    )
    parser.add_argument("--allow-nonperformance-timing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        (not args.skip_timing and len(args.timing_seeds) < 2)
        or (not args.skip_payload and len(args.payload_seeds) < 2)
    ):
        raise ValueError("blocker diagnostics requires at least two seeds")
    if any(not 0.0 < mass <= 0.25 for mass in args.payloads_kg):
        raise ValueError("payload masses must be in (0, 0.25] kg")
    governor = _governor(args.control_cpu)
    if (
        not args.skip_timing
        and governor != "performance"
        and not args.allow_nonperformance_timing
    ):
        raise RuntimeError(
            f"CPU{args.control_cpu} governor is {governor!r}, not performance; "
            "temporarily run: sudo cpupower frequency-set -g performance"
        )

    group_dir = REPO_DIR / "evaluation" / args.group
    group_dir.mkdir(parents=True, exist_ok=True)
    timing_group = args.reuse_timing_group or args.group
    timing_group_dir = REPO_DIR / "evaluation" / timing_group
    payload_group = args.reuse_payload_group or args.group
    payload_group_dir = REPO_DIR / "evaluation" / payload_group
    timing_results = []
    payload_results = []
    if not args.skip_timing:
        for profile_name in args.timing_profiles:
            for seed in args.timing_seeds:
                print(f"timing {profile_name} seed={seed}", flush=True)
                _, result = _run_one(
                    group=timing_group,
                    group_dir=timing_group_dir,
                    profile_name=profile_name,
                    seed=seed,
                    payload_kg=0.0,
                    payload_modeling="unmodeled",
                    label_kind="timing",
                    control_cpu=args.control_cpu,
                    resume=args.resume or bool(args.reuse_timing_group),
                )
                timing_results.append(result)
    if not args.skip_payload:
        for payload_kg in args.payloads_kg:
            for seed in args.payload_seeds:
                for modeling in ("unmodeled", "modeled"):
                    print(
                        f"payload={payload_kg:g} seed={seed} {modeling}",
                        flush=True,
                    )
                    _, result = _run_one(
                        group=payload_group,
                        group_dir=payload_group_dir,
                        profile_name=args.payload_profile,
                        seed=seed,
                        payload_kg=payload_kg,
                        payload_modeling=modeling,
                        label_kind="payload",
                        control_cpu=args.control_cpu,
                        resume=args.resume or bool(args.reuse_payload_group),
                    )
                    payload_results.append(result)

    timing_aggregate = (
        None
        if not timing_results
        else _aggregate_mode(timing_results, "hybrid_residual")
    )
    payload_comparisons = {
        f"payload_{int(round(1000.0 * mass))}g": _payload_comparison(
            payload_results, mass
        )
        for mass in args.payloads_kg
        if payload_results
    }
    timing_passed = bool(
        timing_aggregate is not None
        and timing_aggregate["safety_totals"][
            "complete_interval_overrun_count"
        ]
        == 0
        and timing_aggregate["timing_ms"]["complete_interval_max"]["max"]
        < 6.0
        and timing_aggregate["timing_ms"]["complete_interval_p99"]["max"]
        <= 5.5
    )
    payload_passed = bool(
        payload_comparisons
        and all(
            comparison["by_modeling"]["modeled"]["overall"][
                "qp_success_fraction"
            ]["min"]
            >= 0.99
            for comparison in payload_comparisons.values()
        )
    )
    summary = {
        "stage": "real_robot_readiness_blocker_diagnostics",
        "evidence_scope": "MuJoCo closed loop only; not hardware evidence",
        "local_evaluation_group_ignored": args.group,
        "reused_timing_group_ignored": args.reuse_timing_group,
        "reused_payload_group_ignored": args.reuse_payload_group,
        "control_cpu": args.control_cpu,
        "timing_seeds": args.timing_seeds,
        "payload_seeds": args.payload_seeds,
        "governor_at_runner_start": governor,
        "scheduler_setting": (
            "SCHED_OTHER, nice 0; right-arm worker inherits CPU affinity"
        ),
        "timing_run_count": len(timing_results),
        "payload_run_count": len(payload_results),
        "timing_aggregate": timing_aggregate,
        "performance_vs_previous_powersave_reference": (
            _timing_reference_comparison(timing_aggregate)
        ),
        "payload_comparisons": payload_comparisons,
        "readiness_blocker_checks": {
            "performance_timing_zero_overrun_and_max_lt_6ms": timing_passed,
            "matched_payload_min_qp_success_ge_99pct": payload_passed,
            "both_blockers_cleared": timing_passed and payload_passed,
        },
        "timing_runs": [_compact(result) for result in timing_results],
        "payload_runs": [_compact(result) for result in payload_results],
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
    print(json.dumps(summary["readiness_blocker_checks"], indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
