"""Tests for the fail-closed optional real-time environment guard."""

import unittest
from unittest import mock

from realtime_runtime import (
    validate_realtime_launcher_prerequisites,
    validate_realtime_snapshot,
)


class RealtimeRuntimeGuardTest(unittest.TestCase):
    def test_safe_rr_environment_passes(self):
        snapshot = {
            "policy_name": "SCHED_RR",
            "priority": 10,
            "affinity": [7],
            "governor": "performance",
            "rt_runtime_us": "950000",
            "rt_period_us": "1000000",
        }
        self.assertEqual(
            validate_realtime_snapshot(
                snapshot,
                expected_policy="SCHED_RR",
                expected_priority=10,
                expected_cpu=7,
            ),
            [],
        )

    def test_unthrottled_or_misconfigured_environment_is_rejected(self):
        snapshot = {
            "policy_name": "SCHED_OTHER",
            "priority": 0,
            "affinity": [6, 7],
            "governor": "powersave",
            "rt_runtime_us": "-1",
            "rt_period_us": "1000000",
        }
        errors = validate_realtime_snapshot(
            snapshot,
            expected_policy="SCHED_RR",
            expected_priority=10,
            expected_cpu=7,
        )
        self.assertEqual(len(errors), 5)
        self.assertTrue(any("policy=" in error for error in errors))
        self.assertTrue(any("throttling" in error for error in errors))

    def test_transient_launcher_requires_bounded_rtprio(self):
        snapshot = {
            "governor": "performance",
            "rt_runtime_us": "950000",
            "rt_period_us": "1000000",
            "rtprio_limit_soft": 20,
        }
        self.assertEqual(
            validate_realtime_launcher_prerequisites(
                snapshot, required_priority=10
            ),
            [],
        )
        snapshot["rtprio_limit_soft"] = 0
        self.assertTrue(
            any(
                "RLIMIT_RTPRIO" in error
                for error in validate_realtime_launcher_prerequisites(
                    snapshot, required_priority=10
                )
            )
        )


class RealtimeResultEnvironmentTest(unittest.TestCase):
    def test_main_and_worker_must_both_match(self):
        from disturbance_learning.run_realtime_timing_ablation import (
            _require_run_environment,
        )

        scheduler = {
            "policy_name": "SCHED_RR",
            "priority": 10,
            "right_arm_worker": {
                "policy_name": "SCHED_RR",
                "priority": 10,
                "cpu_affinity": [7],
            },
        }
        result = {
            "runtime_environment": {
                "scheduler": scheduler,
                "cpu_affinity": [7],
                "cpu_frequency_at_start": {
                    "7": {"scaling_governor": "performance"}
                },
            }
        }
        _require_run_environment(
            result, policy="SCHED_RR", priority=10, cpu=7
        )
        scheduler["right_arm_worker"]["priority"] = 9
        with self.assertRaisesRegex(RuntimeError, "worker_priority"):
            _require_run_environment(
                result, policy="SCHED_RR", priority=10, cpu=7
            )


class TargetTimingIrqGateTest(unittest.TestCase):
    def test_target_gate_requires_captured_quiet_evaluation_window(self):
        from disturbance_learning.run_realtime_timing_ablation import (
            _target_irq_checks,
        )

        result = {
            "runtime_environment": {
                "evaluation_irq_activity": {
                    "captured": True,
                    "total_delta_on_physical_core": 0,
                }
            }
        }
        self.assertEqual(
            _target_irq_checks([result]),
            {
                "evaluation_irq_activity_captured_for_all_runs": True,
                "zero_evaluation_irq_on_physical_core": True,
            },
        )
        result["runtime_environment"]["evaluation_irq_activity"][
            "total_delta_on_physical_core"
        ] = 1
        self.assertFalse(
            _target_irq_checks([result])[
                "zero_evaluation_irq_on_physical_core"
            ]
        )

    @mock.patch("sim_support.read_interrupt_counts")
    def test_monitor_captures_irqs_outside_measurement_timer(self, read):
        from sim_support import PerformanceMonitor

        read.side_effect = [
            {
                "available": True,
                "cpus": [6, 7],
                "error": None,
                "interrupts": {
                    "169": {
                        "per_cpu": {"6": 0, "7": 4},
                        "total_on_cpus": 4,
                        "description": "nvme0q8",
                    }
                },
            },
            {
                "available": True,
                "cpus": [6, 7],
                "error": None,
                "interrupts": {
                    "169": {
                        "per_cpu": {"6": 0, "7": 4},
                        "total_on_cpus": 4,
                        "description": "nvme0q8",
                    }
                },
            },
        ]
        environment = {
            "evaluation_irq_monitoring_enabled": True,
            "physical_core_cpus": [6, 7],
        }
        monitor = PerformanceMonitor(
            step_budget=0.002,
            arm_budget=0.006,
            measurement_start_time=0.8,
            measurement_end_time=1.0,
            runtime_environment=environment,
        )
        monitor.start_step(0.8)
        monitor.start_step(1.0)
        activity = environment["evaluation_irq_activity"]
        self.assertEqual(read.call_count, 2)
        self.assertTrue(activity["captured"])
        self.assertEqual(activity["start_simulation_time_s"], 0.8)
        self.assertEqual(activity["end_simulation_time_s"], 1.0)
        self.assertEqual(activity["total_delta_on_physical_core"], 0)


if __name__ == "__main__":
    unittest.main()
