"""Strict first-session hardware state inspection tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
import unittest

import numpy as np
import yaml

from right_arm_runtime.hardware_shadow import HardwareStateError
from right_arm_runtime.unitree_shm import RobotStateSnapshot
from run_hardware_shadow import _inspect_only


REPO_ROOT = Path(__file__).resolve().parents[2]


def _configs() -> tuple[dict, dict]:
    with (REPO_ROOT / "configs/g1.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        controller = yaml.safe_load(stream)
    with (REPO_ROOT / "configs/g1_hardware_shadow.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        hardware = yaml.safe_load(stream)
    return hardware, controller


def _state(sample_id: int, timestamp_ns: int) -> RobotStateSnapshot:
    return RobotStateSnapshot(
        monotonic_timestamp_ns=timestamp_ns,
        sample_id=sample_id,
        robot_tick=1000 + sample_id,
        mode_pr=0,
        mode_machine=4,
        q=tuple(0.001 * index for index in range(35)),
        dq=(0.0,) * 35,
        ddq=(0.0,) * 35,
        tau_est=(0.0,) * 35,
        motor_temperature_c=((20, 25),) * 35,
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        imu_gyroscope=(0.0, 0.0, 0.0),
        imu_accelerometer=(0.0, 0.0, 9.81),
        imu_rpy=(0.0, 0.0, 0.0),
    )


class _FreshClient:
    def __init__(self, *, timestamp_mode: str = "fresh", nan_q: bool = False):
        self.sample_id = 0
        self.timestamp_mode = timestamp_mode
        self.nan_q = nan_q
        self.fixed_timestamp = None

    def read_state(self):
        self.sample_id += 1
        if self.timestamp_mode == "fixed":
            if self.fixed_timestamp is None:
                self.fixed_timestamp = time.monotonic_ns() - 100_000
            timestamp = self.fixed_timestamp
        elif self.timestamp_mode == "stale":
            timestamp = time.monotonic_ns() - 100_000_000
        else:
            timestamp = time.monotonic_ns() - 100_000
        state = _state(self.sample_id, timestamp)
        if self.nan_q:
            q = list(state.q)
            q[22] = float("nan")
            state = replace(state, q=tuple(q))
        return state


class _EmptyClient:
    def read_state(self):
        return _state(0, 0)


class HardwareStateInspectionTest(unittest.TestCase):
    def test_requires_complete_fresh_monotonic_trace(self):
        hardware, controller = _configs()
        summary, records = _inspect_only(
            client=_FreshClient(),
            hardware_config=hardware,
            controller_config=controller,
            sample_count=3,
            timeout_s=1.0,
        )
        self.assertEqual(summary["sample_count"], 3)
        self.assertTrue(summary["complete_requested_sample_count"])
        self.assertEqual(summary["observed_mode_machine"], [4])
        self.assertTrue(summary["mode_machine_matches_reference"])
        self.assertEqual(summary["command_publish_count"], 0)
        self.assertFalse(summary["controller_executed"])
        self.assertFalse(summary["predictor_executed"])
        self.assertEqual(len(records), 3)
        self.assertEqual(len(records[0]["q_rad"]), 35)
        self.assertEqual(records[0]["mapped_right_arm"], records[0]["q_rad"][22:27])

    def test_incomplete_or_only_stale_input_fails_closed(self):
        hardware, controller = _configs()
        for client in (_EmptyClient(), _FreshClient(timestamp_mode="stale")):
            with self.subTest(client=type(client).__name__):
                with self.assertRaisesRegex(
                    HardwareStateError, "inspection incomplete"
                ):
                    _inspect_only(
                        client=client,
                        hardware_config=hardware,
                        controller_config=controller,
                        sample_count=2,
                        timeout_s=0.01,
                    )

    def test_timestamp_regression_and_nonfinite_fail_closed(self):
        hardware, controller = _configs()
        with self.assertRaisesRegex(HardwareStateError, "timestamp"):
            _inspect_only(
                client=_FreshClient(timestamp_mode="fixed"),
                hardware_config=hardware,
                controller_config=controller,
                sample_count=2,
                timeout_s=1.0,
            )
        with self.assertRaisesRegex(HardwareStateError, "NaN/Inf"):
            _inspect_only(
                client=_FreshClient(nan_q=True),
                hardware_config=hardware,
                controller_config=controller,
                sample_count=1,
                timeout_s=1.0,
            )


if __name__ == "__main__":
    unittest.main()
