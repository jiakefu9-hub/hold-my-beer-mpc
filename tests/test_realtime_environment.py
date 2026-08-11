"""Tests for the read-only PREEMPT_RT target environment gate."""

import unittest

from realtime_environment import (
    format_cpu_list,
    parse_cpu_list,
    validate_target_environment,
)


def _passing_snapshot():
    return {
        "control_cpu": 7,
        "online_cpus": list(range(18)),
        "physical_core_cpus": [6, 7],
        "kernel": {"preempt_rt": {"detected": True}},
        "isolation": {
            "isolcpus_flags": ["domain", "managed_irq"],
            "isolcpus_cpus": [6, 7],
            "sysfs_isolated_cpus": [6, 7],
            "nohz_full_cmdline_cpus": [6, 7],
            "nohz_full_active_cpus": [6, 7],
            "rcu_nocbs_cpus": [6, 7],
            "irqaffinity_cpus": [0, 1, 2, 3, 4, 5, 8, 9],
            "irqbalance_state": "inactive",
            "irq_affinity_evidence_available": True,
            "active_irq_conflicts": [],
        },
        "governors": {"6": "performance", "7": "performance"},
        "rt_throttling": {"runtime_us": "950000", "period_us": "1000000"},
        "tools": {
            "chrt": "/usr/bin/chrt",
            "taskset": "/usr/bin/taskset",
            "systemd-run": "/usr/bin/systemd-run",
            "sudo": "/usr/bin/sudo",
        },
    }


class CpuListTest(unittest.TestCase):
    def test_parse_and_format_linux_cpu_lists(self):
        self.assertEqual(
            parse_cpu_list("0-3,6,8-9"), {0, 1, 2, 3, 6, 8, 9}
        )
        self.assertEqual(format_cpu_list({9, 2, 3, 4, 8}), "2-4,8-9")
        self.assertEqual(parse_cpu_list("(null)"), set())


class TargetRealtimeGateTest(unittest.TestCase):
    def test_complete_target_environment_passes(self):
        result = validate_target_environment(_passing_snapshot())
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_checks"], [])

    def test_generic_unisolated_environment_fails_closed(self):
        snapshot = _passing_snapshot()
        snapshot["kernel"]["preempt_rt"]["detected"] = False
        snapshot["isolation"]["isolcpus_cpus"] = []
        snapshot["isolation"]["active_irq_conflicts"] = [{"irq": 161}]
        snapshot["governors"]["7"] = "powersave"
        result = validate_target_environment(snapshot)
        self.assertFalse(result["passed"])
        self.assertIn("preempt_rt_kernel", result["failed_checks"])
        self.assertIn("isolcpus_whole_physical_core", result["failed_checks"])
        self.assertIn("no_active_irq_on_physical_core", result["failed_checks"])
        self.assertIn(
            "performance_governor_on_physical_core",
            result["failed_checks"],
        )

    def test_missing_irq_evidence_fails_closed(self):
        snapshot = _passing_snapshot()
        snapshot["isolation"]["irq_affinity_evidence_available"] = False
        snapshot["isolation"]["irqbalance_state"] = "unavailable"
        result = validate_target_environment(snapshot)
        self.assertFalse(result["passed"])
        self.assertIn(
            "irq_affinity_evidence_available", result["failed_checks"]
        )
        self.assertIn("irqbalance_inactive", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
