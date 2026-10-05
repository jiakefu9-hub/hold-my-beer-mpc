#!/usr/bin/env python3
"""Read-only host evidence and complete-loop timing for G1 field MPC.

This module never initializes DDS or contacts a robot. ``--check`` is read-only
and reports the calling process. ControlThreadScope is an explicit opt-in
thread affinity/scheduler scope; it never changes system governors or boot
settings and restores the caller's original thread settings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics


class ControlThreadScope:
    """Separate worker affinity from control; opt-in FIFO on control only.

    Does not change governors, IRQs or boot configuration. No root/privilege
    escalation: an unavailable requested RT priority fails before output.
    """
    def __init__(self, cpu, rt_priority=0):
        self.cpu = select_cpu(cpu)
        self.priority = int(rt_priority)
        if not 0 <= self.priority <= 40:
            raise ValueError('RT priority must be 0 (ordinary) or 1..40')
        self.affinity = set(os.sched_getaffinity(0))
        self.policy = os.sched_getscheduler(0)
        self.param = os.sched_getparam(0)
        self.prepared = False

    def prepare_workers(self):
        if self.priority:
            # Probe and immediately restore on this thread BEFORE workers or
            # DDS exist, so they never accidentally inherit FIFO priority.
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(self.priority))
            except PermissionError as exc:
                raise RuntimeError('FIFO permission unavailable; in the launching terminal run '
                                   '`sudo prlimit --pid $$ --rtprio=40:40` first') from exc
            finally:
                os.sched_setscheduler(0, self.policy, self.param)
        from realtime_environment import parse_cpu_list
        siblings = parse_cpu_list(_read(f'/sys/devices/system/cpu/cpu{self.cpu}/topology/thread_siblings_list'))
        housekeeping = self.affinity - siblings - {self.cpu}
        if not housekeeping:
            raise ValueError('need housekeeping CPUs too; do not taskset the whole process to one CPU')
        os.sched_setaffinity(0, housekeeping)
        self.prepared = True
        return sorted(housekeeping)

    def activate(self):
        if not self.prepared:
            raise RuntimeError('prepare worker affinity before starting control')
        os.sched_setaffinity(0, {self.cpu})
        if self.priority:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(self.priority))
        return host_evidence(self.cpu)

    def restore(self):
        os.sched_setscheduler(0, self.policy, self.param)
        os.sched_setaffinity(0, self.affinity)


def _read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def select_cpu(cpu=None):
    """Validate against the current cpuset instead of assuming CPU 7 exists."""
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("empty process CPU affinity")
    selected = (2 if 2 in allowed else allowed[0]) if cpu is None else int(cpu)
    if selected not in allowed:
        raise ValueError(f"CPU {selected} not in current allowed affinity {allowed}")
    return selected


def host_evidence(cpu=None):
    selected = select_cpu(cpu)
    policy = os.sched_getscheduler(0)
    policies = {getattr(os, key): key for key in (
        "SCHED_OTHER", "SCHED_FIFO", "SCHED_RR", "SCHED_BATCH", "SCHED_IDLE"
    ) if hasattr(os, key)}
    base = Path(f"/sys/devices/system/cpu/cpu{selected}")
    status = _read("/proc/self/status") or ""
    capabilities = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in status.splitlines() if line.startswith(("CapEff:", "CapPrm:", "CapBnd:"))
    }
    rt_value = _read("/sys/kernel/realtime")
    return {
        "schema": "g1_mpc_host_evidence_v1",
        "kernel_release": platform.release(),
        "kernel_version": platform.version(),
        "kernel_realtime_raw": rt_value,
        "preempt_rt_active": rt_value == "1" if rt_value is not None else None,
        "process_scheduler": policies.get(policy, str(policy)),
        "process_scheduler_priority": os.sched_getparam(0).sched_priority,
        "allowed_cpus": sorted(os.sched_getaffinity(0)),
        "selected_control_cpu": selected,
        "selected_cpu_smt_siblings": _read(base / "topology/thread_siblings_list"),
        "scaling_driver": _read(base / "cpufreq/scaling_driver"),
        "scaling_governor": _read(base / "cpufreq/scaling_governor"),
        "energy_performance_preference": _read(base / "cpufreq/energy_performance_preference"),
        "frequency_khz_snapshot": _read(base / "cpufreq/scaling_cur_freq"),
        "platform_profile": _read("/sys/firmware/acpi/platform_profile"),
        "isolated_cpus": _read("/sys/devices/system/cpu/isolated"),
        "nohz_full_cpus": _read("/sys/devices/system/cpu/nohz_full"),
        "rt_priority_limits": list(resource.getrlimit(resource.RLIMIT_RTPRIO)),
        "memlock_bytes_limits": list(resource.getrlimit(resource.RLIMIT_MEMLOCK)),
        "capabilities": capabilities,
        "numerical_thread_environment": {key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "system_settings_changed": False,
        "scope": "current kernel and calling process; cpuset and PID namespace may be sandbox-restricted",
        "limitations": [
            "PREEMPT_RT kernel does not imply SCHED_FIFO/RR for the control thread",
            "pinning a CPU is not exclusive-core isolation; SMT sibling may remain busy",
            "governor label alone does not specify current CPU frequency or performance",
            "host compute and local DDS Write durations do not measure motor execution latency",
        ],
    }


def percentiles(values):
    ordered = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not ordered:
        return None
    def at(p):
        point = (len(ordered) - 1) * p
        index = int(point)
        fraction = point - index
        return ordered[index] * (1 - fraction) + ordered[min(index + 1, len(ordered) - 1)] * fraction
    return {"mean": statistics.fmean(ordered), "p50": at(.5), "p95": at(.95),
            "p99": at(.99), "max": ordered[-1]}


def summarize_timing(rows):
    if not rows:
        return {"available": False, "samples": 0}
    columns = sorted({key for row in rows for key in row if key.endswith("_ms")})
    result = {"available": True, "samples": len(rows)}
    for key in columns:
        summary = percentiles(row.get(key) for row in rows)
        if summary is not None:
            result[key] = summary
    result.update(
        deadline_misses=sum(bool(row.get("deadline_missed")) for row in rows),
        skipped_slots=sum(int(row.get("skipped_slots", 0)) for row in rows),
    )
    for key in ("reused_lowstate", "reused_imu"):
        result[key] = sum(bool(row[key]) for row in rows) if all(key in row for row in rows) else None
    result["deadline_miss_fraction"] = result["deadline_misses"] / len(rows)
    result["scope"] = (
        "host loop metrics; full_work includes command logging enqueue and timing-row bookkeeping; "
        "final timestamp assignment/clock-advance overhead is bounded separately by actual_period; "
        "not network roundtrip or physical motor latency"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="print read-only host evidence (default)")
    parser.add_argument("--cpu", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        result = host_evidence(args.cpu)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
