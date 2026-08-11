#!/usr/bin/env python3
"""Read-only target PREEMPT_RT environment audit.

The checker deliberately does not change governors, IRQ affinity, scheduling,
boot parameters, or services.  It describes the exact host on which the
timing gate will run and fails closed when target real-time prerequisites are
missing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


ISOLCPUS_FLAGS = {"domain", "managed_irq", "nohz"}


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def parse_cpu_list(value: str | None) -> set[int]:
    """Parse Linux CPU-list syntax such as ``0-3,8,10-11``."""

    if value is None:
        return set()
    value = value.strip()
    if not value or value == "(null)":
        return set()
    cpus: set[int] = set()
    for part in value.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return cpus


def format_cpu_list(cpus: set[int] | list[int]) -> str:
    """Format a CPU set using compact Linux CPU-list syntax."""

    ordered = sorted(set(int(cpu) for cpu in cpus))
    if not ordered:
        return ""
    ranges = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append((start, previous))
        start = previous = cpu
    ranges.append((start, previous))
    return ",".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges
    )


def _kernel_parameters(command_line: str) -> dict[str, str | None]:
    parameters: dict[str, str | None] = {}
    for token in shlex.split(command_line):
        name, separator, value = token.partition("=")
        parameters[name] = value if separator else None
    return parameters


def _parse_isolcpus(value: str | None) -> tuple[set[str], set[int]]:
    if not value:
        return set(), set()
    flags: set[str] = set()
    cpu_parts = []
    for part in value.split(","):
        if part in ISOLCPUS_FLAGS and not cpu_parts:
            flags.add(part)
        else:
            cpu_parts.append(part)
    return flags, parse_cpu_list(",".join(cpu_parts))


def _preempt_rt_evidence(release: str, version: str) -> dict:
    sysfs_realtime = _read_text("/sys/kernel/realtime")
    config_text = _read_text(f"/boot/config-{release}") or ""
    config_preempt_rt = bool(
        re.search(r"^CONFIG_PREEMPT_RT=y$", config_text, re.MULTILINE)
    )
    version_preempt_rt = bool(re.search(r"\bPREEMPT_RT\b", version))
    return {
        "detected": bool(
            sysfs_realtime == "1" or config_preempt_rt or version_preempt_rt
        ),
        "sysfs_realtime": sysfs_realtime,
        "config_preempt_rt": config_preempt_rt,
        "version_preempt_rt": version_preempt_rt,
    }


def _irqbalance_state() -> str:
    if shutil.which("systemctl") is None:
        return "unavailable"
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "irqbalance"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    state = result.stdout.strip()
    if state:
        return state
    if "Failed to connect to bus" in result.stderr:
        return "unavailable"
    return "not-found"


def _interrupt_descriptions() -> tuple[list[int], dict[int, dict]]:
    text = _read_text("/proc/interrupts") or ""
    lines = text.splitlines()
    if not lines:
        return [], {}
    cpu_labels = re.findall(r"CPU(\d+)", lines[0])
    cpus = [int(cpu) for cpu in cpu_labels]
    interrupts = {}
    for line in lines[1:]:
        prefix, separator, remainder = line.partition(":")
        if not separator or not prefix.strip().isdigit():
            continue
        irq = int(prefix.strip())
        fields = remainder.split()
        if len(fields) < len(cpus):
            continue
        try:
            counts = [int(value) for value in fields[: len(cpus)]]
        except ValueError:
            continue
        interrupts[irq] = {
            "counts": dict(zip(cpus, counts)),
            "description": " ".join(fields[len(cpus) :]),
            "total_count": sum(counts),
        }
    return cpus, interrupts


def parse_interrupt_counts(text: str, cpus: set[int]) -> dict:
    """Parse numeric IRQ counters for selected CPUs from ``/proc/interrupts``."""

    lines = text.splitlines()
    if not lines:
        return {
            "available": False,
            "cpus": sorted(cpus),
            "error": "/proc/interrupts is empty",
            "interrupts": {},
        }
    header = lines[0].split()
    cpu_columns = {}
    for column, name in enumerate(header):
        match = re.fullmatch(r"CPU(\d+)", name)
        if match:
            cpu = int(match.group(1))
            if cpu in cpus:
                cpu_columns[cpu] = column
    missing = sorted(cpus - set(cpu_columns))
    if missing:
        return {
            "available": False,
            "cpus": sorted(cpus),
            "error": f"CPU columns missing: {format_cpu_list(missing)}",
            "interrupts": {},
        }

    interrupts = {}
    for line in lines[1:]:
        match = re.match(r"^\s*(\d+):\s+(.*)$", line)
        if not match:
            continue
        fields = match.group(2).split()
        if len(fields) < len(header):
            continue
        try:
            all_counts = [int(value) for value in fields[: len(header)]]
        except ValueError:
            continue
        irq = int(match.group(1))
        per_cpu = {
            str(cpu): all_counts[column]
            for cpu, column in sorted(cpu_columns.items())
        }
        interrupts[str(irq)] = {
            "per_cpu": per_cpu,
            "total_on_cpus": sum(per_cpu.values()),
            "description": " ".join(fields[len(header) :]),
        }
    available = bool(interrupts)
    return {
        "available": available,
        "cpus": sorted(cpus),
        "error": None if available else "no numeric IRQ counters found",
        "interrupts": interrupts,
    }


def read_interrupt_counts(cpus: set[int]) -> dict:
    """Read one lightweight IRQ-count snapshot for selected CPUs."""

    text = _read_text("/proc/interrupts")
    return parse_interrupt_counts(text or "", set(cpus))


def summarize_interrupt_activity(before: dict, after: dict) -> dict:
    """Return positive IRQ deltas between two evaluation-boundary snapshots."""

    cpus = sorted(set(before.get("cpus", ())) | set(after.get("cpus", ())))
    captured = bool(before.get("available") and after.get("available"))
    if not captured:
        errors = [
            message
            for message in (before.get("error"), after.get("error"))
            if message
        ]
        return {
            "captured": False,
            "physical_core_cpus": cpus,
            "error": "; ".join(errors) or "IRQ count snapshot unavailable",
            "start_total_on_physical_core": None,
            "end_total_on_physical_core": None,
            "total_delta_on_physical_core": None,
            "active_irqs": [],
        }

    before_irqs = before.get("interrupts", {})
    after_irqs = after.get("interrupts", {})
    active = []
    for irq in sorted(
        set(before_irqs) | set(after_irqs), key=lambda value: int(value)
    ):
        left = before_irqs.get(irq, {})
        right = after_irqs.get(irq, {})
        per_cpu_delta = {}
        for cpu in cpus:
            delta = int(right.get("per_cpu", {}).get(str(cpu), 0)) - int(
                left.get("per_cpu", {}).get(str(cpu), 0)
            )
            if delta > 0:
                per_cpu_delta[str(cpu)] = delta
        delta_total = sum(per_cpu_delta.values())
        if delta_total:
            active.append(
                {
                    "irq": int(irq),
                    "count_delta": delta_total,
                    "per_cpu_delta": per_cpu_delta,
                    "description": right.get("description")
                    or left.get("description", ""),
                }
            )

    def total(snapshot: dict) -> int:
        return sum(
            int(item.get("total_on_cpus", 0))
            for item in snapshot.get("interrupts", {}).values()
        )

    start_total = total(before)
    end_total = total(after)
    return {
        "captured": True,
        "physical_core_cpus": cpus,
        "error": None,
        "start_total_on_physical_core": start_total,
        "end_total_on_physical_core": end_total,
        "total_delta_on_physical_core": sum(
            item["count_delta"] for item in active
        ),
        "active_irqs": active,
    }


def _irq_affinity_conflicts(isolated_cpus: set[int]) -> list[dict]:
    _, interrupts = _interrupt_descriptions()
    conflicts = []
    irq_root = Path("/proc/irq")
    if not irq_root.is_dir():
        return conflicts
    for directory in irq_root.iterdir():
        if not directory.name.isdigit():
            continue
        irq = int(directory.name)
        details = interrupts.get(irq)
        if not details or details["total_count"] <= 0:
            continue
        affinity = parse_cpu_list(
            _read_text(directory / "effective_affinity_list")
        )
        overlap = affinity & isolated_cpus
        if not overlap:
            continue
        conflicts.append(
            {
                "irq": irq,
                "description": details["description"],
                "effective_affinity": format_cpu_list(affinity),
                "isolated_overlap": format_cpu_list(overlap),
                "total_count": details["total_count"],
                "count_on_isolated": sum(
                    details["counts"].get(cpu, 0) for cpu in isolated_cpus
                ),
            }
        )
    return sorted(conflicts, key=lambda item: item["irq"])


def _sample_active_irq_conflicts(
    isolated_cpus: set[int], observation_s: float = 0.1
) -> tuple[list[dict], list[dict]]:
    """Separate configured affinity from IRQs active during observation.

    Managed MSI-X vectors can retain a single-CPU effective affinity even
    when ``isolcpus=managed_irq`` removes that CPU from the block-mq queue.
    Historical setup interrupts are not runtime activity, so gate on positive
    count deltas while preserving all configured conflicts as diagnostics.
    """

    before = {
        item["irq"]: item["total_count"]
        for item in _irq_affinity_conflicts(isolated_cpus)
    }
    time.sleep(max(0.0, float(observation_s)))
    configured = _irq_affinity_conflicts(isolated_cpus)
    active = []
    for item in configured:
        delta = item["total_count"] - before.get(item["irq"], 0)
        if delta > 0:
            active.append({**item, "count_delta": delta})
    return configured, active


def collect_target_environment(control_cpu: int) -> dict:
    """Collect PREEMPT_RT, CPU isolation, IRQ, and frequency evidence."""

    control_cpu = int(control_cpu)
    release = platform.release()
    version = platform.version()
    online_cpus = parse_cpu_list(
        _read_text("/sys/devices/system/cpu/online")
    )
    sibling_cpus = parse_cpu_list(
        _read_text(
            f"/sys/devices/system/cpu/cpu{control_cpu}/topology/"
            "thread_siblings_list"
        )
    )
    if not sibling_cpus and control_cpu in online_cpus:
        sibling_cpus = {control_cpu}
    housekeeping_cpus = online_cpus - sibling_cpus
    command_line = _read_text("/proc/cmdline") or ""
    parameters = _kernel_parameters(command_line)
    isolcpus_flags, isolcpus_cpus = _parse_isolcpus(
        parameters.get("isolcpus")
    )
    nohz_full_cmdline = parse_cpu_list(parameters.get("nohz_full"))
    nohz_full_active = parse_cpu_list(
        _read_text("/sys/devices/system/cpu/nohz_full")
    )
    rcu_nocbs = parse_cpu_list(parameters.get("rcu_nocbs"))
    irqaffinity = parse_cpu_list(parameters.get("irqaffinity"))
    governors = {
        str(cpu): _read_text(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
        )
        for cpu in sorted(sibling_cpus)
    }
    rt_runtime_us = _read_text("/proc/sys/kernel/sched_rt_runtime_us")
    rt_period_us = _read_text("/proc/sys/kernel/sched_rt_period_us")
    required_isolation = format_cpu_list(sibling_cpus)
    housekeeping = format_cpu_list(housekeeping_cpus)
    configured_irq_conflicts, active_irq_conflicts = (
        _sample_active_irq_conflicts(sibling_cpus)
    )
    recommended_boot_parameters = (
        " ".join(
            (
                f"isolcpus=domain,managed_irq,{required_isolation}",
                f"nohz_full={required_isolation}",
                f"rcu_nocbs={required_isolation}",
                f"irqaffinity={housekeeping}",
            )
        )
        if required_isolation and housekeeping
        else None
    )
    return {
        "control_cpu": control_cpu,
        "online_cpus": sorted(online_cpus),
        "physical_core_cpus": sorted(sibling_cpus),
        "housekeeping_cpus": sorted(housekeeping_cpus),
        "kernel": {
            "release": release,
            "version": version,
            "command_line": command_line,
            "preempt_rt": _preempt_rt_evidence(release, version),
        },
        "isolation": {
            "isolcpus_flags": sorted(isolcpus_flags),
            "isolcpus_cpus": sorted(isolcpus_cpus),
            "sysfs_isolated_cpus": sorted(
                parse_cpu_list(
                    _read_text("/sys/devices/system/cpu/isolated")
                )
            ),
            "nohz_full_cmdline_cpus": sorted(nohz_full_cmdline),
            "nohz_full_active_cpus": sorted(nohz_full_active),
            "rcu_nocbs_cpus": sorted(rcu_nocbs),
            "irqaffinity_cpus": sorted(irqaffinity),
            "irqbalance_state": _irqbalance_state(),
            "irq_observation_ms": 100.0,
            "irq_affinity_evidence_available": bool(
                _read_text("/proc/interrupts")
                and any(Path("/proc/irq").glob("*/effective_affinity_list"))
            ),
            "configured_irq_conflicts": configured_irq_conflicts,
            "active_irq_conflicts": active_irq_conflicts,
        },
        "governors": governors,
        "rt_throttling": {
            "runtime_us": rt_runtime_us,
            "period_us": rt_period_us,
        },
        "tools": {
            name: shutil.which(name)
            for name in ("chrt", "taskset", "systemd-run", "sudo")
        },
        "recommended_boot_parameters": recommended_boot_parameters,
    }


def validate_target_environment(snapshot: dict) -> dict:
    """Return a strict target-runtime gate without changing the machine."""

    control_cpu = int(snapshot["control_cpu"])
    online = set(snapshot["online_cpus"])
    core = set(snapshot["physical_core_cpus"])
    isolation = snapshot["isolation"]
    isolcpus = set(isolation["isolcpus_cpus"])
    sysfs_isolated = set(isolation["sysfs_isolated_cpus"])
    nohz_cmdline = set(isolation["nohz_full_cmdline_cpus"])
    nohz_active = set(isolation["nohz_full_active_cpus"])
    rcu_nocbs = set(isolation["rcu_nocbs_cpus"])
    irqaffinity = set(isolation["irqaffinity_cpus"])
    flags = set(isolation["isolcpus_flags"])
    try:
        rt_runtime_us = int(snapshot["rt_throttling"]["runtime_us"])
        rt_period_us = int(snapshot["rt_throttling"]["period_us"])
        throttled = 0 < rt_runtime_us < rt_period_us
    except (TypeError, ValueError):
        throttled = False
    checks = {
        "preempt_rt_kernel": bool(
            snapshot["kernel"]["preempt_rt"]["detected"]
        ),
        "control_cpu_online": control_cpu in online,
        "physical_core_known": bool(core) and control_cpu in core,
        "isolcpus_whole_physical_core": bool(core)
        and core <= isolcpus
        and core <= sysfs_isolated,
        "isolcpus_domain_and_managed_irq": {
            "domain",
            "managed_irq",
        }
        <= flags,
        "nohz_full_whole_physical_core": bool(core)
        and core <= nohz_cmdline
        and core <= nohz_active,
        "rcu_nocbs_whole_physical_core": bool(core) and core <= rcu_nocbs,
        "boot_irqaffinity_excludes_physical_core": bool(irqaffinity)
        and not bool(core & irqaffinity),
        "irq_affinity_evidence_available": bool(
            isolation["irq_affinity_evidence_available"]
        ),
        "no_active_irq_on_physical_core": bool(
            isolation["irq_affinity_evidence_available"]
        )
        and not bool(isolation["active_irq_conflicts"]),
        "irqbalance_inactive": isolation["irqbalance_state"]
        in {"inactive", "failed", "not-found"},
        "performance_governor_on_physical_core": bool(core)
        and all(
            snapshot["governors"].get(str(cpu)) == "performance"
            for cpu in core
        ),
        "bounded_kernel_rt_throttling": throttled,
        "required_runtime_tools_available": all(
            snapshot["tools"].get(name)
            for name in ("chrt", "taskset", "systemd-run", "sudo")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [
            name for name, passed in checks.items() if not passed
        ],
    }


def _print_human(snapshot: dict, result: dict) -> None:
    print("TARGET_REALTIME_ENVIRONMENT:", "PASS" if result["passed"] else "FAIL")
    for name, passed in result["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    core = format_cpu_list(snapshot["physical_core_cpus"])
    print(f"control CPU/core: {snapshot['control_cpu']} / {core or 'unknown'}")
    print(
        "recommended one-boot parameters:",
        snapshot["recommended_boot_parameters"] or "unavailable",
    )
    conflicts = snapshot["isolation"]["active_irq_conflicts"]
    configured_conflicts = snapshot["isolation"].get(
        "configured_irq_conflicts", []
    )
    if conflicts:
        print("active IRQ conflicts:")
        for conflict in conflicts:
            print(
                f"  IRQ {conflict['irq']} affinity="
                f"{conflict['effective_affinity']} "
                f"{conflict['description']}"
            )
    elif configured_conflicts:
        print(
            "configured but idle IRQ affinities during "
            f"{snapshot['isolation']['irq_observation_ms']:.0f} ms:"
        )
        for conflict in configured_conflicts:
            print(
                f"  IRQ {conflict['irq']} affinity="
                f"{conflict['effective_affinity']} "
                f"{conflict['description']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only PREEMPT_RT target environment checker"
    )
    parser.add_argument(
        "--control-cpu",
        type=int,
        default=int(os.environ.get("MPC_CONTROL_CPU", "7")),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    snapshot = collect_target_environment(args.control_cpu)
    result = validate_target_environment(snapshot)
    if args.json:
        print(
            json.dumps(
                {"environment": snapshot, "gate": result},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_human(snapshot, result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
