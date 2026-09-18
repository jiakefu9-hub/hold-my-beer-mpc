"""Synthetic offline test for full-window H0 PID metrics."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_hardware_pid import analyze, write_outputs


class HardwarePidAnalysisTest(unittest.TestCase):
    def test_primary_window_is_walk_start_through_stop_settle(self):
        epoch = 10_000_000_000
        target = np.deg2rad([
            -4.0, -1.0, 0.0, -8.1, 0.0,
            -4.0, 1.0, 0.0, -7.8, 0.0,
            0.0, 0.0, 0.0,
        ])
        motor_indices = (15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 12)
        rows = [
            {"schema": "g1_pid_session_v1", "event": "session_start",
             "primary_metric_window_s": [5.0, 18.0]},
            {"schema": "g1_pid_event_v1", "event": "task_epoch",
             "task_epoch_monotonic_ns": epoch},
            {"schema": "g1_pid_event_v1", "event": "heading_reference_frozen",
             "yaw0_rad": 0.3},
        ]
        for index in range(2101):
            task_s = index * 0.01
            received = epoch + int(task_s * 1e9)
            motors = [{"index": motor, "q_rad": 0.0} for motor in range(30)]
            for slot, motor in enumerate(motor_indices):
                motors[motor]["q_rad"] = float(target[slot])
            rows.extend([
                {"schema": "g1_lowstate_raw_v1", "received_monotonic_ns": received,
                 "crc_valid": True, "motors": motors},
                {"schema": "g1_torso_imu_raw_v1", "received_monotonic_ns": received,
                 "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                 "gyroscope_rad_s": [0.0, 0.0, 0.0],
                 "accelerometer_raw_m_s2": [0.0, 0.0, 9.81]},
            ])
            if index % 2 == 0:
                rows.append({
                    "schema": "g1_hardware_pid_command_v1", "event": "dds_write",
                    "task_elapsed_s": task_s, "q_command_rad": target.tolist(),
                    "q_measured_rad": target.tolist(), "pid_active": 3.0 <= task_s < 18.0,
                    "gravity_error_before_m_s2": [0.0, 0.0],
                    "q_reference_clipped": [False] * 5,
                    "controller_compute_us": 500.0, "write_duration_us": 50.0,
                })
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            with raw.open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            data, summary = analyze(raw, sample_hz=100.0, filter_window_s=0.11)
            output = Path(directory) / "analysis"
            write_outputs(data, summary, output)
            for name in ("metrics.npz", "metrics.csv", "summary.json", "endpoint_metrics_h0.png"):
                self.assertTrue((output / name).is_file())
        primary = summary["windows"]["primary_walk_start_through_stop_settle_end"]
        self.assertEqual(primary["interval_s"], [5.0, 18.0])
        self.assertEqual(primary["role"], "headline")
        self.assertEqual(primary["metrics"]["samples"], 1300)
        for side in ("left", "right"):
            self.assertLess(
                primary["metrics"][side]["endpoint_linear_acceleration_h0_m_s2_norm"]["rms"],
                1e-7,
            )
            self.assertLess(
                primary["metrics"][side]["endpoint_angular_acceleration_h0_rad_s2_norm"]["rms"],
                1e-7,
            )
        self.assertEqual(data["task_elapsed_s"].shape[0], 2100)


if __name__ == "__main__":
    unittest.main()
