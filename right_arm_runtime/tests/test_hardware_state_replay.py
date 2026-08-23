"""Offline H2-preparation audit tests for persisted state traces."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from right_arm_runtime.hardware_state_replay import (
    HardwareStateTraceAuditError,
    audit_state_trace,
    audit_state_trace_files,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _record(sample_id: int) -> dict:
    timestamp = 1_000_000_000 + sample_id * 2_000_000
    q = [0.001 * index for index in range(35)]
    return {
        "read_monotonic_ns": timestamp + 100_000,
        "source_monotonic_timestamp_ns": timestamp,
        "state_age_ms": 0.1,
        "sample_id": sample_id,
        "robot_tick": 100 + sample_id,
        "mode_pr": 0,
        "mode_machine": 4,
        "q_rad": q,
        "dq_rad_s": [0.0] * 35,
        "ddq_rad_s2": [0.0] * 35,
        "tau_est_nm": [0.0] * 35,
        "motor_temperature_c": [[20, 25] for _ in range(35)],
        "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "imu_gyroscope_rad_s": [0.0, 0.0, 0.0],
        "imu_accelerometer_raw_m_s2": [0.0, 0.0, 9.81],
        "imu_rpy_rad": [0.0, 0.0, 0.0],
        "quaternion_norm": 1.0,
        "mapped_right_arm": q[22:27],
    }


def _bridge_summary(count: int) -> dict:
    return {
        "output_capability": "absent",
        "lowstate_crc_valid_count": count,
        "lowstate_crc_rejected_count": 0,
        "paired_state_count": count,
    }


class HardwareStateReplayTest(unittest.TestCase):
    def test_structural_pass_never_becomes_hardware_verification(self):
        report = audit_state_trace(
            [_record(1), _record(2), _record(3)],
            source_kind="synthetic_test_fixture",
            bridge_summary=_bridge_summary(3),
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["offline_trace_contract_passed"])
        self.assertFalse(report["hardware_session_verified"])
        self.assertFalse(report["verification_flags_modified"])
        self.assertIsNone(report["command_output_observed"])
        self.assertTrue(
            report["output_capability_absent_from_bridge_summary"]
        )
        self.assertEqual(report["observed_mode_machine_candidates"], [4])
        self.assertGreater(len(report["site_gates_remaining"]), 0)

    def test_file_round_trip_is_explicitly_labeled_synthetic(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "raw_state_trace.jsonl"
            trace.write_text(
                "".join(json.dumps(_record(i)) + "\n" for i in (1, 2)),
                encoding="utf-8",
            )
            bridge = root / "state_bridge_summary.json"
            bridge.write_text(json.dumps(_bridge_summary(2)), encoding="utf-8")
            report = audit_state_trace_files(
                trace,
                source_kind="synthetic_test_fixture",
                bridge_summary_path=bridge,
            )
        self.assertEqual(report["source_kind"], "synthetic_test_fixture")
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(len(report["trace_sha256"]), 64)
        self.assertEqual(len(report["bridge_summary_sha256"]), 64)

    def test_regression_nonfinite_mapping_and_bridge_mismatch_fail(self):
        duplicate = _record(1)
        with self.assertRaisesRegex(HardwareStateTraceAuditError, "sample_id"):
            audit_state_trace(
                [_record(1), duplicate], source_kind="synthetic_test_fixture"
            )
        nonfinite = _record(1)
        nonfinite["q_rad"][22] = float("nan")
        with self.assertRaisesRegex(HardwareStateTraceAuditError, "finite"):
            audit_state_trace(
                [nonfinite], source_kind="synthetic_test_fixture"
            )
        mismatch = _record(1)
        mismatch["mapped_right_arm"][0] += 1.0
        with self.assertRaisesRegex(HardwareStateTraceAuditError, "22..26"):
            audit_state_trace(
                [mismatch], source_kind="synthetic_test_fixture"
            )
        with self.assertRaisesRegex(HardwareStateTraceAuditError, "counters"):
            audit_state_trace(
                [_record(1), _record(2)],
                source_kind="synthetic_test_fixture",
                bridge_summary=_bridge_summary(1),
            )

    def test_repository_verification_flags_remain_false(self):
        path = REPO_ROOT / "configs/g1_hardware_shadow.yaml"
        before = path.read_bytes()
        payload = yaml.safe_load(before)
        self.assertFalse(payload["hardware_shadow"]["joint_mapping_verified"])
        self.assertFalse(
            payload["hardware_shadow"]["robot_tick_monotonic_verified"]
        )
        self.assertFalse(
            payload["hardware_shadow"]["imu"]["contract_verified"]
        )
        self.assertEqual(payload["hardware_shadow"]["allowed_mode_pr"], [])
        self.assertEqual(
            payload["hardware_shadow"]["allowed_mode_machine"], []
        )
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
