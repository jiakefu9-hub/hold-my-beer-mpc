"""Tests for the fail-closed optional real-time environment guard."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
