"""Local monotonic timing only; no SDK, DDS or robot output."""

import math
import os
import time

import numpy as np


class PeriodicClock:
    """Absolute monotonic slots. Skip expired slots without catch-up bursts."""

    def __init__(self, epoch_ns, period_s):
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError("period must be finite and positive")
        self.period_ns = int(round(period_s * 1e9))
        self.scheduled_ns = int(epoch_ns)
        self.skipped_slots = 0

    def advance(self, finished_ns):
        next_ns = self.scheduled_ns + self.period_ns
        skipped = max(0, (int(finished_ns) - next_ns) // self.period_ns + 1)
        self.scheduled_ns = next_ns + skipped * self.period_ns
        self.skipped_slots += skipped
        return skipped

    def wait(self):
        remaining = (self.scheduled_ns - time.monotonic_ns()) * 1e-9
        if remaining > 0:
            time.sleep(remaining)


def pin_control_thread(cpu):
    """Pin the calling Linux thread only; existing I/O workers stay unpinned."""
    allowed = sorted(os.sched_getaffinity(0))
    selected = (7 if 7 in allowed else allowed[0]) if cpu is None else cpu
    if selected not in allowed:
        raise ValueError(f"CPU {selected} not in allowed affinity {allowed}")
    os.sched_setaffinity(0, {selected})
    return {
        "control_cpu": selected,
        "control_thread_affinity": sorted(os.sched_getaffinity(0)),
        "original_affinity": allowed,
        "scheduler_policy": os.sched_getscheduler(0),
        "scheduler_priority": os.sched_getparam(0).sched_priority,
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def timing_summary(rows):
    """Keep complete-loop timing separate from local DDS write latency."""
    if not rows:
        return {"available": False}
    result = {"available": True, "samples": len(rows)}
    for name in (
        "actual_period_ms", "wake_lateness_ms", "work_ms", "state_age_at_write_ms",
        "imu_age_at_write_ms", "state_imu_skew_ms", "write_ms",
    ):
        values = np.asarray([row[name] for row in rows if row.get(name) is not None], dtype=float)
        if len(values):
            result[name] = dict(zip(("p50", "p95", "p99", "max"),
                                   map(float, np.percentile(values, [50, 95, 99, 100]))))
    result.update(
        deadline_misses=sum(bool(row.get("deadline_missed")) for row in rows),
        skipped_slots=sum(int(row.get("skipped_slots", 0)) for row in rows),
        reused_lowstate=sum(bool(row.get("reused_lowstate")) for row in rows),
        reused_imu=sum(bool(row.get("reused_imu")) for row in rows),
    )
    result["deadline_miss_fraction"] = result["deadline_misses"] / len(rows)
    result["scope"] = "host loop including command enqueue; excludes timing-row enqueue and physical motor/network latency"
    return result
