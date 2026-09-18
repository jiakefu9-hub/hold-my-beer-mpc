"""Offline tests for the real-G1 PID control core; no SDK or DDS."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from endpoint_pose import EndpointModel
from hardware_pid_control import (
    END_S,
    FixedH0Heading,
    HardwarePidPlan,
    PidParameters,
    RELEASE_START_S,
    RightArmHardwarePid,
    WALK_START_S,
    WALK_STOP_S,
)
from g1_walk_pid import REQUIRED_CONFIRMATIONS, load_pid_parameters, load_profile, main as device_main


PARAMETERS = {
    "pid_kp_pose": [1.2, 1.2],
    "pid_kd_pose": [1.2, 1.2],
    "pid_ki_pose": [0.0, 0.0],
    "pid_posture_gain": [1.15, 1.15, 2.10, 1.15, 0.95],
    "pid_finite_diff_eps": 1e-4,
    "pid_damping": 0.15,
    "pid_integral_limit": 0.20,
    "pid_max_dq": 0.48,
    "pid_de_g_alpha": 0.07,
    "pid_q_offset_limit_deg": [5, 5, 5, 5, 5],
}


class HardwarePidControlTest(unittest.TestCase):
    def test_field_template_is_fail_closed_and_reviewed_copy_loads(self):
        template = Path(__file__).resolve().parents[1] / "profiles/pid_walk_capture.template"
        with self.assertRaisesRegex(ValueError, "FIELD_REVIEWED"):
            load_profile(template)
        text = template.read_text()
        text = text.replace("robot_id=UNSET", "robot_id=field-g1")
        text = text.replace("model_name=UNSET", "model_name=G1-EDU-23DoF")
        text = text.replace("confirmed_by=UNSET", "confirmed_by=field-review")
        text = text.replace("profile_status=DRAFT", "profile_status=FIELD_REVIEWED")
        for name in REQUIRED_CONFIRMATIONS:
            text = text.replace(f"{name}=false", f"{name}=true")
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.conf"
            path.write_text(text)
            profile = load_profile(path)
            parameters, mapping = load_pid_parameters(
                Path(__file__).resolve().parents[3] / "configs/g1.yaml", profile
            )
        self.assertEqual(profile["required_fsm"], "500")
        self.assertEqual(mapping["pid_q_offset_limit_deg"], [5.0] * 5)
        self.assertAlmostEqual(parameters.max_dq, 0.48)

    def test_draft_profile_refuses_before_creating_output_directory(self):
        template = Path(__file__).resolve().parents[1] / "profiles/pid_walk_capture.template"
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_exist"
            result = device_main([
                "fake0", "--profile", str(template), "--output-dir", str(output),
                "--permit-real-output", "PID_WALK_H0_CAPTURE",
            ])
            self.assertEqual(result, 1)
            self.assertFalse(output.exists())

    def test_heading_uses_fixed_pre_walk_circular_mean(self):
        heading = FixedH0Heading()
        epoch = 1_000_000_000
        for index in range(121):
            task_s = 2.8 + index * 0.02
            yaw = math.pi - 0.02 if index % 2 else -math.pi + 0.02
            heading.observe(epoch + index * 20_000_000, task_s, yaw, 0.0)
        reference = heading.freeze()
        self.assertLess(abs(abs(reference) - math.pi), 0.03)
        before = heading.current()["reference_rad"]
        heading.observe(epoch + 4_000_000_000, 7.0, 0.5, 0.0)
        self.assertEqual(before, heading.current()["reference_rad"])

    def test_pid_uses_bottle_site_and_clips_reference_not_measurement(self):
        nominal = np.deg2rad([-4.0, 1.0, 0.0, -7.8, 0.0])
        params = PidParameters.from_mapping(PARAMETERS)
        controller = RightArmHardwarePid(nominal, params, EndpointModel(), 0.02)
        slots = np.zeros(13)
        slots[5:10] = nominal
        controller.set_measured_dq(np.zeros(5))
        # Roll the torso by ten degrees. The bottle gravity error must become
        # nonzero and the generated command must stay inside nominal +/- 5 deg.
        quat = [math.cos(math.radians(5)), math.sin(math.radians(5)), 0, 0]
        q_ref, dq_ref, diagnostics = controller.step(slots, quat, 0.0, 0.02)
        self.assertGreater(np.linalg.norm(diagnostics["gravity_error_before_m_s2"]), 0.1)
        self.assertTrue(np.isfinite(dq_ref).all())
        self.assertTrue(np.all(q_ref >= nominal - np.deg2rad(5) - 1e-12))
        self.assertTrue(np.all(q_ref <= nominal + np.deg2rad(5) + 1e-12))

    def test_plan_holds_nonzero_left_pose_and_pid_through_stop_settle(self):
        target = np.deg2rad([
            -4.0, -1.0, 0.0, -8.1, 0.0,
            -4.0, 1.0, 0.0, -7.8, 0.0,
            0.0, 0.0, 0.0,
        ])
        params = PidParameters.from_mapping(PARAMETERS)
        controller = RightArmHardwarePid(target[5:10], params, EndpointModel(), 0.02)
        plan = HardwarePidPlan(
            np.zeros(13), target, np.r_[np.full(11, 20.0), 0.0, 0.0],
            np.r_[np.full(11, 1.0), 0.0, 0.0], controller,
        )
        quat = [1.0, 0.0, 0.0, 0.0]
        for task_s, expected_stage in (
            (WALK_START_S, "forward_walk"),
            (WALK_STOP_S + 1.0, "stop_settle"),
        ):
            frame = plan.sample(task_s, target, np.zeros(13), quat, 0.0, 0.02)
            np.testing.assert_allclose(frame["q_rad"][:5], target[:5])
            self.assertTrue(frame["diagnostics"]["pid_active"])
            self.assertEqual(frame["stage"], expected_stage)
            self.assertEqual(frame["weight"], 1.0)
        release = plan.sample(RELEASE_START_S + 1.5, target, np.zeros(13), quat, 0.0, 0.02)
        self.assertAlmostEqual(release["weight"], 0.5)
        done = plan.sample(END_S, target, np.zeros(13), quat, 0.0, 0.02)
        self.assertTrue(done["terminal"])
        self.assertEqual(done["weight"], 0.0)

    def test_q_offset_above_five_degrees_is_refused(self):
        bad = dict(PARAMETERS)
        bad["pid_q_offset_limit_deg"] = [5, 5, 5, 5, 5.1]
        with self.assertRaisesRegex(ValueError, "offsets"):
            PidParameters.from_mapping(bad)


if __name__ == "__main__":
    unittest.main()
