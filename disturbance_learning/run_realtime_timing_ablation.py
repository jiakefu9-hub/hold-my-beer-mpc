#!/usr/bin/env python3
"""A/B the optional SCHED_RR launch against the SCHED_OTHER baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from disturbance_learning.command_schedule import (
    GENERALIZATION_SCHEDULE_PROFILES,
)
from disturbance_learning.run_blocker_diagnostics import _compact, _run_one
from disturbance_learning.run_generalization_ablation import _aggregate_mode
from realtime_runtime import (
    collect_realtime_snapshot,
    validate_realtime_launcher_prerequisites,
)
from realtime_environment import (
    collect_target_environment,
    validate_target_environment,
)


DEFAULT_SEEDS = (2101, 2102, 2103, 2104)
MIN_TIMING_RUNS = 12
MIN_COMPLETE_INTERVALS = 9588
MAX_WORST_RUN_P99_MS = 5.5
CONTROL_PERIOD_MS = 6.0
MIN_NORMAL_QP_SUCCESS = 0.99
BASELINE_SUMMARY = (
    REPO_DIR
    / "evaluation_summary/readiness_blocker_diagnostics/performance/summary.json"
)


def _require_run_environment(
    result: dict, *, policy: str, priority: int, cpu: int
) -> None:
    environment = result["runtime_environment"]
    scheduler = environment["scheduler"]
    errors = []
    if scheduler["policy_name"] != policy:
        errors.append(f"policy={scheduler['policy_name']}")
    if int(scheduler["priority"]) != priority:
        errors.append(f"priority={scheduler['priority']}")
    if environment["cpu_affinity"] != [cpu]:
        errors.append(f"affinity={environment['cpu_affinity']}")
    cpu_environment = environment["cpu_frequency_at_start"].get(str(cpu), {})
    if cpu_environment.get("scaling_governor") != "performance":
        errors.append(
            f"governor={cpu_environment.get('scaling_governor')}"
        )
    worker = scheduler.get("right_arm_worker", {})
    if worker.get("policy_name") != policy:
        errors.append(f"worker_policy={worker.get('policy_name')}")
    if int(worker.get("priority", -1)) != priority:
        errors.append(f"worker_priority={worker.get('priority')}")
    if worker.get("cpu_affinity") != [cpu]:
        errors.append(f"worker_affinity={worker.get('cpu_affinity')}")
    if errors:
        raise RuntimeError("timing run environment mismatch: " + ", ".join(errors))


def _overrun_profile(results: list[dict]) -> dict:
    rows = []
    for result in results:
        path = REPO_DIR / result["run_dir_local_ignored"] / "perf_intervals.csv"
        with path.open("r", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["overrun"] == "True":
                    rows.append(
                        {
                            "profile": result["profile"],
                            "seed": result["seed"],
                            **row,
                        }
                    )
    components = (
        "mpc_policy_update_ms",
        "ddq_total_ms",
        "cpp_executor_bridge_ms",
        "other_right_arm_path_ms",
    )
    return {
        "overrun_count": len(rows),
        "without_ddq_second_pass_count": sum(
            int(row["ddq_second_pass_count"]) == 0 for row in rows
        ),
        "without_ddq_rescue_count": sum(
            int(row["ddq_rescue_count"]) == 0 for row in rows
        ),
        "component_mean_ms": {
            name: (
                sum(float(row[name]) for row in rows) / len(rows)
                if rows
                else 0.0
            )
            for name in components
        },
        "largest_intervals": [
            {
                name: row[name]
                for name in (
                    "profile",
                    "seed",
                    "complete_interval_ms",
                    "mpc_policy_update_ms",
                    "ddq_total_ms",
                    "ddq_max_ms",
                    "cpp_executor_bridge_ms",
                    "other_right_arm_path_ms",
                    "ddq_second_pass_count",
                    "ddq_rescue_count",
                )
            }
            for row in sorted(
                rows,
                key=lambda value: float(value["complete_interval_ms"]),
                reverse=True,
            )[:10]
        ],
    }


def _tail_source_profile(results: list[dict]) -> dict:
    """Summarize compute time separately from process wake-up latency."""

    samples = {
        "mpc_policy_update": [],
        "ddq_to_torque_call": [],
        "worker_roundtrip": [],
        "worker_compute": [],
        "worker_queue": [],
        "worker_ipc_and_copy": [],
    }
    for result in results:
        path = REPO_DIR / result["run_dir_local_ignored"] / "perf_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        hardware = summary["total"]["real_hardware_control"]
        process = hardware["independent_cpp_process"]
        sources = {
            "mpc_policy_update": hardware["mpc_policy_update"],
            "ddq_to_torque_call": hardware["ddq_to_torque_call"],
            "worker_roundtrip": process["roundtrip_time"],
            "worker_compute": process["worker_time"],
            "worker_queue": process["queue_time"],
            "worker_ipc_and_copy": process["ipc_and_copy_time"],
        }
        for name, statistics in sources.items():
            samples[name].append(statistics)

    return {
        name: {
            "run_count": len(statistics),
            "mean_of_run_means_ms": sum(
                item["mean"] for item in statistics
            )
            / len(statistics),
            "mean_of_run_p99_ms": sum(
                item["p99"] for item in statistics
            )
            / len(statistics),
            "worst_run_p99_ms": max(item["p99"] for item in statistics),
            "worst_sample_ms": max(item["max"] for item in statistics),
        }
        for name, statistics in samples.items()
    }


def _comparison(baseline: dict, realtime: dict) -> dict:
    def compare(name: str, statistic: str) -> dict:
        left = baseline["timing_ms"][name][statistic]
        right = realtime["timing_ms"][name][statistic]
        return {
            "sched_other": left,
            "sched_rr": right,
            "rr_change_percent": 100.0 * (right / left - 1.0),
        }

    return {
        "complete_interval_mean_ms": compare(
            "complete_interval_mean", "mean"
        ),
        "complete_interval_p99_ms": compare(
            "complete_interval_p99", "mean"
        ),
        "complete_interval_worst_max_ms": compare(
            "complete_interval_max", "max"
        ),
        "predictor_mean_ms": compare("predictor_mean", "mean"),
        "predictor_p99_ms": compare("predictor_p99", "mean"),
        "overrun": {
            "sched_other_count": baseline["safety_totals"][
                "complete_interval_overrun_count"
            ],
            "sched_other_interval_count": baseline["safety_totals"][
                "complete_interval_count"
            ],
            "sched_rr_count": realtime["safety_totals"][
                "complete_interval_overrun_count"
            ],
            "sched_rr_interval_count": realtime["safety_totals"][
                "complete_interval_count"
            ],
        },
    }


def _target_irq_checks(results: list[dict]) -> dict:
    activity = [
        result["runtime_environment"].get("evaluation_irq_activity", {})
        for result in results
    ]
    return {
        "evaluation_irq_activity_captured_for_all_runs": bool(activity)
        and all(item.get("captured") is True for item in activity),
        "zero_evaluation_irq_on_physical_core": bool(activity)
        and all(
            item.get("total_delta_on_physical_core") == 0
            for item in activity
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeated SCHED_OTHER versus SCHED_RR timing validation"
    )
    parser.add_argument(
        "--group", default=f"realtime_rr10_{datetime.now():%Y%m%d_%H%M%S}"
    )
    parser.add_argument(
        "--summary-dir",
        default="evaluation_summary/realtime_timing_ablation",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(GENERALIZATION_SCHEDULE_PROFILES),
        default=list(GENERALIZATION_SCHEDULE_PROFILES),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS)
    )
    parser.add_argument("--control-cpu", type=int, default=7)
    parser.add_argument("--expected-policy", default="SCHED_RR")
    parser.add_argument("--expected-priority", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-target-realtime", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) < 2:
        raise ValueError("real-time timing ablation requires at least two seeds")
    if not BASELINE_SUMMARY.is_file():
        raise FileNotFoundError(BASELINE_SUMMARY)

    launch_snapshot = collect_realtime_snapshot(args.control_cpu)
    launch_errors = validate_realtime_launcher_prerequisites(
        launch_snapshot, required_priority=args.expected_priority
    )
    if launch_errors:
        raise RuntimeError("invalid real-time launch: " + "; ".join(launch_errors))
    target_environment = None
    target_gate = None
    if args.require_target_realtime:
        target_environment = collect_target_environment(args.control_cpu)
        target_gate = validate_target_environment(target_environment)
        if not target_gate["passed"]:
            raise RuntimeError(
                "target real-time environment failed: "
                + ", ".join(target_gate["failed_checks"])
            )

    group_dir = REPO_DIR / "evaluation" / args.group
    group_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for profile_name in args.profiles:
        for seed in args.seeds:
            print(f"SCHED_RR {profile_name} seed={seed}", flush=True)
            _, result = _run_one(
                group=args.group,
                group_dir=group_dir,
                profile_name=profile_name,
                seed=seed,
                payload_kg=0.0,
                payload_modeling="unmodeled",
                label_kind="timing_rr10",
                control_cpu=args.control_cpu,
                resume=args.resume,
            )
            _require_run_environment(
                result,
                policy=args.expected_policy,
                priority=args.expected_priority,
                cpu=args.control_cpu,
            )
            results.append(result)

    realtime = _aggregate_mode(results, "hybrid_residual")
    baseline_source = json.loads(
        BASELINE_SUMMARY.read_text(encoding="utf-8")
    )
    baseline = baseline_source["timing_aggregate"]
    checks = {
        "zero_critical_nonfinite": (
            realtime["safety_totals"]["critical_nonfinite_count"] == 0
        ),
        "zero_6ms_overrun": (
            realtime["safety_totals"]["complete_interval_overrun_count"] == 0
        ),
        "worst_run_complete_p99_le_5p5ms": (
            realtime["timing_ms"]["complete_interval_p99"]["max"]
            <= MAX_WORST_RUN_P99_MS
        ),
        "worst_run_complete_max_lt_6ms": (
            realtime["timing_ms"]["complete_interval_max"]["max"]
            < CONTROL_PERIOD_MS
        ),
        "normal_qp_success_min_ge_99pct": (
            realtime["overall"]["qp_success_fraction"]["min"]
            >= MIN_NORMAL_QP_SUCCESS
        ),
        "at_least_12_repeated_runs": len(results) >= MIN_TIMING_RUNS,
        "at_least_9588_complete_intervals": (
            realtime["safety_totals"]["complete_interval_count"]
            >= MIN_COMPLETE_INTERVALS
        ),
    }
    if args.require_target_realtime:
        checks.update(_target_irq_checks(results))
    gate_passed = all(checks.values())
    sched_other_tail = _tail_source_profile(
        baseline_source["timing_runs"]
    )
    sched_rr_tail = _tail_source_profile(results)
    summary = {
        "stage": "sched_other_vs_safe_sched_rr_timing_ablation",
        "evidence_scope": "paced MuJoCo closed loop; not hardware evidence",
        "baseline_summary": str(BASELINE_SUMMARY.relative_to(REPO_DIR)),
        "local_evaluation_group_ignored": args.group,
        "profiles": args.profiles,
        "seeds": args.seeds,
        "launch_snapshot": launch_snapshot,
        "target_environment": target_environment,
        "target_environment_gate": target_gate,
        "safety_design": {
            "policy": args.expected_policy,
            "priority": args.expected_priority,
            "cpu": args.control_cpu,
            "main_and_worker_same_policy_priority_and_cpu": True,
            "kernel_rt_throttling_required": True,
            "transient_systemd_unit_runtime_limit_minutes": 15,
            "fifo_rejected_reason": (
                "no expected latency benefit for blocking handoff, with a "
                "larger runaway starvation risk"
            ),
        },
        "sched_other_baseline": baseline,
        "sched_rr_result": realtime,
        "comparison": _comparison(baseline, realtime),
        "sched_other_overrun_profile": _overrun_profile(
            baseline_source["timing_runs"]
        ),
        "sched_rr_overrun_profile": _overrun_profile(results),
        "tail_source_comparison": {
            "sched_other": sched_other_tail,
            "sched_rr": sched_rr_tail,
        },
        "timing_gate": {
            "requirements": {
                "minimum_run_count": MIN_TIMING_RUNS,
                "minimum_complete_interval_count": MIN_COMPLETE_INTERVALS,
                "control_period_ms": CONTROL_PERIOD_MS,
                "maximum_worst_run_p99_ms": MAX_WORST_RUN_P99_MS,
                "minimum_normal_qp_success_fraction": (
                    MIN_NORMAL_QP_SUCCESS
                ),
                "maximum_complete_interval_overrun_count": 0,
                "maximum_critical_nonfinite_count": 0,
                "maximum_evaluation_irq_count_on_physical_core": (
                    0 if args.require_target_realtime else None
                ),
            },
            "checks": checks,
            "passed": gate_passed,
        },
        "decision": {
            "generic_kernel_reliably_meets_6ms": (
                gate_passed if not args.require_target_realtime else False
            ),
            "target_realtime_reliably_meets_6ms": (
                gate_passed if args.require_target_realtime else None
            ),
            "hard_realtime_claim": False,
            "worker_queue_worst_sample_reduction_percent": 100.0
            * (
                1.0
                - sched_rr_tail["worker_queue"]["worst_sample_ms"]
                / sched_other_tail["worker_queue"]["worst_sample_ms"]
            ),
            "best_effort_runtime": (
                "performance governor; one performance CPU; single-thread "
                "numeric libraries; main and blocking C++ worker at "
                "SCHED_RR priority 10; kernel RT throttling retained"
            ),
            "next_gate_if_failed": (
                None
                if gate_passed
                else "repeat on the target PREEMPT_RT or equivalent "
                "real-time environment with IRQ/CPU isolation"
            ),
        },
        "runs": [_compact(result) for result in results],
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
    print(json.dumps(summary["timing_gate"], indent=2, sort_keys=True))
    print(f"summary: {summary_path}")
    if args.fail_on_gate and not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
