"""Synthetic offline capture acceptance checks; no DDS or robot access."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_hardware_pid import ANALYSIS_MOTOR_INDICES, analyze


def capture_rows(task="stationary"):
    """A static, fully recorded 21-second run with successful software release."""
    epoch = 10_000_000_000
    target = np.deg2rad([
        -4., -1., 0., -8.1, 0., -4., 1., 0., -7.8, 0., 0., 0., 0.,
    ]).tolist()
    motors = [{"index": index, "q_rad": 0.} for index in range(30)]
    for slot, index in enumerate(ANALYSIS_MOTOR_INDICES):
        motors[index]["q_rad"] = target[slot]
    rows = [
        {"schema": "g1_mpc_session_v1", "event": "session_start", "task": task,
         "primary_metric_window_s": [5., 18.], "predictor_mode": "hold_current"},
        {"schema": "g1_mpc_event_v1", "event": "task_epoch",
         "task_epoch_monotonic_ns": epoch},
        {"schema": "g1_mpc_event_v1", "event": "heading_reference_frozen",
         "yaw0_rad": .3},
    ]
    for index in range(1051):
        task_s = index / 50
        received = epoch + index * 20_000_000
        rows.extend([
            {"schema": "g1_lowstate_raw_v1", "received_monotonic_ns": received,
             "crc_valid": True, "motors": motors},
            {"schema": "g1_torso_imu_raw_v1", "received_monotonic_ns": received,
             "quaternion_wxyz": [1., 0., 0., 0.], "gyroscope_rad_s": [0., 0., 0.],
             "accelerometer_raw_m_s2": [0., 0., 9.81]},
            {"schema": "g1_hardware_mpc_command_v1", "event": "dds_write",
             "task_elapsed_s": task_s, "q_command_rad": target, "q_measured_rad": target,
             "mpc_active": 3. <= task_s < 18., "gravity_error_before_m_s2": [0., 0.],
             "mpc": {"solved": True, "fallback_used": False},
             "controller_compute_us": 500., "write_duration_us": 50.},
            {"schema": "g1_mpc_cycle_complete_v1", "task_elapsed_s": task_s,
             "complete_work_ms": .8, "complete_deadline_missed": False},
        ])
    rows.extend([
        {"schema": "g1_mpc_event_v1", "event": "session_end",
         "outcome": "normal_release_completed", "final_weight": 0.,
         "physical_stop_verified": False},
        {"schema": "g1_mpc_event_v1", "event": "sdk_shutdown", "endpoints": {
            "low_subscriber": "listener_detached_then_closed",
            "imu_subscriber": "listener_detached_then_closed",
            "arm_publisher": "listener_detached_then_closed"}},
        {"schema": "g1_mpc_event_v1", "event": "capture_drained", "queue_dropped": 0},
    ])
    return rows


class HardwareMpcAnalysisTest(unittest.TestCase):
    def analyze_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            with raw.open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            return analyze(raw, sample_hz=50., filter_window_s=.11)

    def test_stationary_and_walk_use_distinct_labels_with_identical_physics(self):
        summaries = {}
        for task in ("stationary", "walk"):
            _, summary = self.analyze_rows(capture_rows(task))
            summaries[task] = summary
            self.assertEqual(summary["schema"], "g1_hardware_mpc_analysis_v1")
            self.assertEqual(summary["task"], task)
            self.assertEqual(summary["task_source"], "session_header")
            self.assertEqual(summary["session_status"]["status"], "normal_release_recorded")
            self.assertFalse(summary["session_status"]["physical_stop_verified"])
            self.assertEqual(summary["capture_quality"]["status"], "recorded_checks_passed")
            self.assertEqual(summary["capture_quality"]["warnings"], [])
            primary = summary["windows"][summary["primary_window_key"]]
            self.assertEqual(primary["interval_s"], [5., 18.])
            self.assertEqual(primary["metrics"]["samples"], 650)
            for side in ("left", "right"):
                self.assertLess(primary["metrics"][side][
                    "endpoint_linear_acceleration_h0_m_s2_norm"]["rms"], 1e-7)
        stationary, walk = summaries["stationary"], summaries["walk"]
        self.assertEqual(stationary["primary_window_key"], "primary_stationary_control")
        self.assertTrue(all("walk" not in key and "stop_settle" not in key
                            for key in stationary["windows"]))
        self.assertEqual(walk["primary_window_key"], "primary_walk_start_through_stop_settle_end")
        self.assertEqual(stationary["windows"][stationary["primary_window_key"]],
                         walk["windows"][walk["primary_window_key"]])

    def test_early_operator_stop_cannot_use_release_rows_to_complete_primary_window(self):
        for active_during_release in (False, True):
            with self.subTest(active_during_release=active_during_release):
                rows = capture_rows()
                for row in rows:
                    if row.get("event") == "dds_write" and row["task_elapsed_s"] >= 17.:
                        row.update(mpc_active=active_during_release, operator_stop=True)
                    elif row.get("event") == "session_end":
                        row["outcome"] = "operator_stop_release_completed"
                rows.append({"schema": "g1_mpc_event_v1", "event": "operator_stop_requested",
                             "task_elapsed_s": 17.})
                with self.assertRaisesRegex(ValueError, "inactive control or early-stop/release"):
                    self.analyze_rows(rows)

    def test_fault_release_without_task_time_cannot_fill_missing_commands(self):
        rows = capture_rows()
        for row in rows:
            if row.get("event") == "dds_write" and row["task_elapsed_s"] >= 17.:
                row.update(task_elapsed_s=None, fault_release=True, mpc_active=False)
        with self.assertRaisesRegex(ValueError, "MPC commands has incomplete"):
            self.analyze_rows(rows)

    def test_late_release_fault_preserves_metrics_and_reports_fault(self):
        rows = capture_rows()
        rows = [row for row in rows if row.get("event") != "session_end"]
        rows.append({"schema": "g1_mpc_event_v1", "event": "session_fault",
                     "reason": "injected release write failure", "task_elapsed_s": 19.,
                     "fault_release": {"completed": False, "final_weight": .4}})
        _, summary = self.analyze_rows(rows)
        primary = summary["windows"][summary["primary_window_key"]]
        self.assertEqual(primary["metrics"]["samples"], 650)
        status = summary["session_status"]
        self.assertEqual(status["status"], "fault_recorded")
        self.assertFalse(status["release_completed"])
        self.assertEqual(status["final_weight"], .4)
        self.assertEqual(status["faults"][0]["reason"], "injected release write failure")
        self.assertEqual(summary["capture_quality"]["status"], "review_required")
        self.assertTrue(any("fault_recorded" in warning
                            for warning in summary["capture_quality"]["warnings"]))

    def test_missing_completion_evidence_never_passes_recorded_checks(self):
        for missing in ("session_end", "capture_drained", "sdk_shutdown"):
            with self.subTest(missing=missing):
                rows = [row for row in capture_rows() if row.get("event") != missing]
                _, summary = self.analyze_rows(rows)
                self.assertEqual(summary["capture_quality"]["status"], "review_required")
                self.assertTrue(summary["capture_quality"]["warnings"])
                if missing == "session_end":
                    self.assertEqual(summary["session_status"]["status"], "completion_unknown")
                    self.assertIsNone(summary["session_status"]["release_completed"])

    def test_recording_and_shutdown_failures_require_review(self):
        rows = capture_rows()
        for row in rows:
            if row.get("event") == "capture_drained":
                row["queue_dropped"] = 2
            elif row.get("event") == "sdk_shutdown":
                row["endpoints"]["imu_subscriber"] = "close_failed: injected failure"
        rows = [row for row in rows if not (row.get("schema") == "g1_mpc_cycle_complete_v1"
                                            and row["task_elapsed_s"] == 6.)]
        _, summary = self.analyze_rows(rows)
        warnings = summary["capture_quality"]["warnings"]
        self.assertTrue(any("dropped records" in warning for warning in warnings))
        self.assertTrue(any("shutdown failure" in warning for warning in warnings))
        self.assertTrue(any("timing record count differs" in warning for warning in warnings))
        self.assertEqual(summary["capture_quality"]["status"], "review_required")

    def test_legacy_pid_without_task_or_lifecycle_evidence_remains_analyzable(self):
        rows = []
        for original in capture_rows("walk"):
            if original.get("schema") == "g1_mpc_cycle_complete_v1" or original.get("event") in {
                "session_end", "sdk_shutdown", "capture_drained",
            }:
                continue
            row = copy.deepcopy(original)
            row["schema"] = row["schema"].replace("mpc", "pid")
            row.pop("task", None)
            if "mpc_active" in row:
                row["pid_active"] = row.pop("mpc_active")
                row.pop("mpc")
            rows.append(row)
        _, summary = self.analyze_rows(rows)
        self.assertEqual(summary["schema"], "g1_hardware_pid_analysis_v1")
        self.assertEqual(summary["task"], "walk")
        self.assertEqual(summary["task_source"], "legacy_walk_default")
        self.assertEqual(summary["primary_window_key"], "primary_walk_start_through_stop_settle_end")
        self.assertEqual(summary["windows"][summary["primary_window_key"]]["metrics"]["samples"], 650)
        self.assertEqual(summary["session_status"]["status"], "completion_unknown")
        self.assertEqual(summary["capture_quality"]["status"], "review_required")
        self.assertNotIn("mpc", summary["control"])


if __name__ == "__main__":
    unittest.main()
