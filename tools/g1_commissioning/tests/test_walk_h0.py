"""Offline checks for the fixed-H0 conversion; no hardware or recorded data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "walk_h0", Path(__file__).parents[1] / "derive_walk_h0.py")
walk_h0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(walk_h0)


class WalkH0Test(unittest.TestCase):
    def imu(self, yaw_deg, received, sequence):
        half = np.deg2rad(yaw_deg) / 2
        return dict(received_monotonic_ns=received, host_callback_sequence=sequence,
                    quaternion_wxyz=[np.cos(half), 0, 0, np.sin(half)],
                    rpy_rad=[0, 0, np.deg2rad(yaw_deg)],
                    gyroscope_rad_s=[1, 0, 0],
                    accelerometer_raw_m_s2=[0, 0, 9.81], temperature_raw=25)

    def test_every_sample_transforms_to_fixed_h0(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            event = lambda name, **kw: dict(schema="g1_capture_event_v1", event=name, **kw)
            rows = [event("task_epoch", task_epoch_monotonic_ns=1_000_000_000),
                    self.imu(90, 900_000_000, 1),
                    dict(schema="g1_lowstate_raw_v1", received_monotonic_ns=1_100_000_000,
                         host_callback_sequence=1, pelvis_imu={k: v for k, v in self.imu(180, 0, 0).items()
                                                              if k not in ("received_monotonic_ns", "host_callback_sequence")}),
                    event("heading_reference_frozen", yaw0_rad=np.pi/2,
                          h0_definition="fixed_run_frame_x_along_pre_walk_mean_yaw_z_vertical",
                          reference_sample_count=40, reference_observed_span_s=1.95,
                          reference_first_sample_task_s=3.01, reference_last_sample_task_s=4.96,
                          requested_reference_start_s=3, requested_reference_end_s=5),
                    self.imu(90, 2_000_000_000, 2)]
            for row in rows:
                if "quaternion_wxyz" in row:
                    row["schema"] = "g1_torso_imu_raw_v1"
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
            arrays, manifest = walk_h0.derive(raw)
            self.assertEqual(manifest["torso_samples"], 2)
            self.assertEqual(manifest["pelvis_samples"], 1)
            np.testing.assert_allclose(arrays["torso_rpy_h0_rad"], 0, atol=1e-12)
            np.testing.assert_allclose(arrays["torso_angular_velocity_h0_rad_s"],
                                       [[1, 0, 0], [1, 0, 0]], atol=1e-12)
            np.testing.assert_allclose(arrays["torso_linear_acceleration_h0_m_s2"], 0, atol=1e-12)
            np.testing.assert_allclose(arrays["pelvis_rpy_h0_rad"][0, 2], np.pi/2, atol=1e-12)
            self.assertAlmostEqual(arrays["torso_task_elapsed_s"][0], -0.1)

    def test_reference_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            raw.write_text(json.dumps(dict(schema="g1_capture_event_v1", event="task_epoch",
                                           task_epoch_monotonic_ns=1)) + "\n")
            with self.assertRaisesRegex(ValueError, "heading_reference_frozen"):
                walk_h0.derive(raw)


if __name__ == "__main__":
    unittest.main()
