"""Fail-closed checks for the optional real-time launch environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource

from realtime_environment import (
    collect_target_environment,
    validate_target_environment,
)


POLICY_NAMES = {
    getattr(os, name): name
    for name in (
        "SCHED_OTHER",
        "SCHED_BATCH",
        "SCHED_IDLE",
        "SCHED_FIFO",
        "SCHED_RR",
    )
    if hasattr(os, name)
}


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def collect_realtime_snapshot(control_cpu: int) -> dict:
    """Collect only scheduling state relevant to the control process tree."""

    policy = int(os.sched_getscheduler(0))
    rtprio_soft, rtprio_hard = resource.getrlimit(resource.RLIMIT_RTPRIO)
    return {
        "policy": policy,
        "policy_name": POLICY_NAMES.get(policy, str(policy)),
        "priority": int(os.sched_getparam(0).sched_priority),
        "affinity": sorted(int(cpu) for cpu in os.sched_getaffinity(0)),
        "control_cpu": int(control_cpu),
        "governor": _read_text(
            f"/sys/devices/system/cpu/cpu{control_cpu}/cpufreq/"
            "scaling_governor"
        ),
        "rt_runtime_us": _read_text(
            "/proc/sys/kernel/sched_rt_runtime_us"
        ),
        "rt_period_us": _read_text(
            "/proc/sys/kernel/sched_rt_period_us"
        ),
        "rr_timeslice_ms": _read_text(
            "/proc/sys/kernel/sched_rr_timeslice_ms"
        ),
        "rtprio_limit_soft": int(rtprio_soft),
        "rtprio_limit_hard": int(rtprio_hard),
    }


def validate_realtime_snapshot(
    snapshot: dict,
    *,
    expected_policy: str,
    expected_priority: int,
    expected_cpu: int,
) -> list[str]:
    """Return configuration errors; an empty list means safe to proceed."""

    errors = []
    if snapshot.get("policy_name") != expected_policy:
        errors.append(
            f"policy={snapshot.get('policy_name')} expected={expected_policy}"
        )
    if int(snapshot.get("priority", -1)) != int(expected_priority):
        errors.append(
            f"priority={snapshot.get('priority')} expected={expected_priority}"
        )
    affinity = list(snapshot.get("affinity", []))
    if affinity != [int(expected_cpu)]:
        errors.append(f"affinity={affinity} expected={[int(expected_cpu)]}")
    if snapshot.get("governor") != "performance":
        errors.append(
            f"governor={snapshot.get('governor')} expected=performance"
        )
    try:
        rt_runtime_us = int(snapshot.get("rt_runtime_us"))
        rt_period_us = int(snapshot.get("rt_period_us"))
    except (TypeError, ValueError):
        errors.append("kernel RT throttling values are unavailable")
    else:
        if not 0 < rt_runtime_us < rt_period_us:
            errors.append(
                "kernel RT throttling must remain enabled with "
                "0 < runtime < period"
            )
    return errors


def validate_realtime_launcher_prerequisites(
    snapshot: dict, *, required_priority: int
) -> list[str]:
    """Validate the transient service before it launches RT child runs."""

    errors = []
    if snapshot.get("governor") != "performance":
        errors.append(
            f"governor={snapshot.get('governor')} expected=performance"
        )
    try:
        rt_runtime_us = int(snapshot.get("rt_runtime_us"))
        rt_period_us = int(snapshot.get("rt_period_us"))
    except (TypeError, ValueError):
        errors.append("kernel RT throttling values are unavailable")
    else:
        if not 0 < rt_runtime_us < rt_period_us:
            errors.append(
                "kernel RT throttling must remain enabled with "
                "0 < runtime < period"
            )
    if int(snapshot.get("rtprio_limit_soft", 0)) < int(required_priority):
        errors.append(
            f"RLIMIT_RTPRIO={snapshot.get('rtprio_limit_soft')} "
            f"is below required priority {required_priority}"
        )
    return errors


def require_recorded_run_environment(
    result: dict, *, policy: str, priority: int, cpu: int
) -> None:
    """Fail if a recorded control run did not inherit the target runtime."""

    environment = result["runtime_environment"]
    scheduler = environment["scheduler"]
    errors = []
    if scheduler["policy_name"] != policy:
        errors.append(f"policy={scheduler['policy_name']}")
    if int(scheduler["priority"]) != priority:
        errors.append(f"priority={scheduler['priority']}")
    if environment["cpu_affinity"] != [cpu]:
        errors.append(f"affinity={environment['cpu_affinity']}")
    cpu_environment = environment["cpu_frequency_at_start"].get(
        str(cpu), {}
    )
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
        raise RuntimeError(
            "timing run environment mismatch: " + ", ".join(errors)
        )


def target_irq_checks(results: list[dict]) -> dict:
    """Summarize whether every recorded evaluation window was IRQ-quiet."""

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
        description="Validate inherited scheduling before a control run"
    )
    parser.add_argument(
        "--expected-policy", choices=("SCHED_RR",), default="SCHED_RR"
    )
    parser.add_argument("--expected-priority", type=int, default=10)
    parser.add_argument("--expected-cpu", type=int, default=7)
    parser.add_argument("--require-target-environment", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    snapshot = collect_realtime_snapshot(args.expected_cpu)
    errors = validate_realtime_snapshot(
        snapshot,
        expected_policy=args.expected_policy,
        expected_priority=args.expected_priority,
        expected_cpu=args.expected_cpu,
    )
    target_environment = None
    if args.require_target_environment:
        target_environment = collect_target_environment(args.expected_cpu)
        target_gate = validate_target_environment(target_environment)
        errors.extend(
            f"target environment: {name}"
            for name in target_gate["failed_checks"]
        )
    print(json.dumps(snapshot, sort_keys=True))
    if target_environment is not None:
        print(
            json.dumps(
                {
                    "target_environment": target_environment,
                    "target_gate": target_gate,
                },
                sort_keys=True,
            )
        )
    if errors:
        raise SystemExit(
            "unsafe or incomplete real-time environment: " + "; ".join(errors)
        )
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if command:
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
