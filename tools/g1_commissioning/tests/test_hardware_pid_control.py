"""Offline tests for the real-G1 PID control core; no SDK or DDS."""

import ast
import math
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from endpoint_pose import EndpointModel
from hardware_pid_control import (
    END_S,
    FixedH0Heading,
    HardwarePidPlan,
    linear_weight_release,
    locomotion_setpoint,
    PidParameters,
    RELEASE_START_S,
    RightArmHardwarePid,
    WeightReleaseRamp,
    WALK_START_S,
    WALK_STOP_S,
)
from g1_walk_pid import (
    REQUIRED_CONFIRMATIONS,
    close_sdk_endpoint,
    load_pid_parameters,
    load_profile,
    main as device_main,
)


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
    "hardware_pid_max_dq": 0.07,
    "hardware_pid_max_ddq": 0.20,
}


class HardwarePidControlTest(unittest.TestCase):
    def test_sdk_endpoint_detaches_listener_before_close(self):
        calls = []

        class Entity:
            def set_listener(self, listener):
                calls.append(("listener", listener))

        class Holder:
            pass

        class Channel:
            pass

        class Subscriber:
            def Close(self):
                calls.append(("close", None))

        entity = Entity()
        holder = Holder()
        holder._Reader__reader = entity
        channel = Channel()
        channel._Channel__reader = holder
        subscriber = Subscriber()
        subscriber._ChannelSubscriber__channel = channel

        result = close_sdk_endpoint(subscriber, "subscriber")
        self.assertEqual(result, "listener_detached_then_closed")
        self.assertEqual(calls, [("listener", None), ("close", None)])

    def test_sdk_endpoint_refuses_unknown_sdk_layout(self):
        with self.assertRaisesRegex(RuntimeError, "layout changed"):
            close_sdk_endpoint(object(), "subscriber")

    def test_device_fault_path_contains_no_direct_weight_zero_frame(self):
        source = (Path(__file__).resolve().parents[1] / "g1_walk_pid.py").read_text()
        tree = ast.parse(source)
        direct_zero_lines = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant) and key.value == "weight"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, (int, float))
                    and float(value.value) == 0.0
                ):
                    direct_zero_lines.append(node.lineno)
        self.assertEqual(direct_zero_lines, [])
        self.assertIn("release_result = fallback_weight_release(str(exc))", source)
        self.assertIn('"single_frame_weight_zero_attempted": False', source)

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
        self.assertAlmostEqual(parameters.hardware_max_dq, 0.07)
        self.assertAlmostEqual(parameters.hardware_max_ddq, 0.20)

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

    def test_repaired_device_path_remains_locked_before_output_or_dds(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_exist"
            with mock.patch("g1_walk_pid.load_profile", return_value={}), mock.patch(
                "g1_walk_pid.load_pid_parameters", return_value=(object(), {})
            ):
                result = device_main([
                    "fake0", "--profile", str(Path(directory) / "reviewed.conf"),
                    "--output-dir", str(output),
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

    def test_heading_hold_continues_through_stop_settle(self):
        walking = locomotion_setpoint(14.9, 0.12)
        self.assertEqual(walking["vx_m_s"], 0.5)
        self.assertEqual(walking["yaw_rate_rad_s"], 0.12)
        self.assertTrue(walking["heading_hold_active"])
        self.assertLessEqual(walking["duration_s"], 0.1 + 1e-12)

        settling = locomotion_setpoint(15.0, 0.12)
        self.assertEqual(settling["vx_m_s"], 0.0)
        self.assertEqual(settling["yaw_rate_rad_s"], 0.12)
        self.assertTrue(settling["heading_hold_active"])

        released = locomotion_setpoint(18.0, 0.12)
        self.assertEqual(released["vx_m_s"], 0.0)
        self.assertEqual(released["yaw_rate_rad_s"], 0.0)
        self.assertFalse(released["heading_hold_active"])

        stopped = locomotion_setpoint(10.0, 0.12, inhibited=True)
        self.assertEqual(stopped["vx_m_s"], 0.0)
        self.assertEqual(stopped["yaw_rate_rad_s"], 0.0)

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
        self.assertLessEqual(np.max(np.abs(dq_ref)), 0.004 + 1e-12)
        self.assertLessEqual(
            np.max(np.abs(diagnostics["governed_ddq_ref_rad_s2"])), 0.20 + 1e-12
        )
        self.assertEqual(np.asarray(diagnostics["pid_error_m_s2"]).shape, (2,))
        self.assertEqual(
            np.asarray(diagnostics["pid_gravity_error_jacobian"]).shape, (2, 5)
        )
        self.assertEqual(np.asarray(diagnostics["pid_task_dq_rad_s"]).shape, (5,))
        self.assertEqual(np.asarray(diagnostics["pid_posture_dq_rad_s"]).shape, (5,))

    def test_hardware_governor_prevents_fast_sign_reversal(self):
        nominal = np.deg2rad([-4.0, 1.0, 0.0, -7.8, 0.0])
        params = PidParameters.from_mapping(PARAMETERS)
        controller = RightArmHardwarePid(nominal, params, EndpointModel(), 0.02)
        slots = np.zeros(13)
        slots[5:10] = nominal
        quat = [math.cos(math.radians(5)), math.sin(math.radians(5)), 0, 0]
        previous = np.zeros(5)
        for _ in range(500):
            _, dq_ref, diagnostics = controller.step(slots, quat, 0.0, 0.02)
            self.assertLessEqual(np.max(np.abs(dq_ref)), 0.07 + 1e-12)
            self.assertLessEqual(np.max(np.abs(dq_ref - previous)), 0.004 + 1e-12)
            self.assertLessEqual(
                np.max(np.abs(diagnostics["governed_ddq_ref_rad_s2"])),
                0.20 + 1e-12,
            )
            previous = dq_ref.copy()

    def test_weight_release_never_steps_from_one_to_zero(self):
        samples = [linear_weight_release(1.0, index * 0.02)[0]
                   for index in range(151)]
        self.assertEqual(samples[0], 1.0)
        self.assertGreater(samples[-2], 0.0)
        self.assertEqual(samples[-1], 0.0)
        self.assertTrue(np.all(np.diff(samples) <= 1e-12))
        self.assertLessEqual(np.max(-np.diff(samples)), 0.02 / 3.0 + 1e-12)

        ramp = WeightReleaseRamp(1.0, 0.02)
        actual = [ramp.sample(0.0)[0]]
        # Even a three-second scheduler stall must not create a one-frame drop.
        actual.append(ramp.sample(3.0)[0])
        self.assertAlmostEqual(actual[-1], 1.0 - 0.02 / 3.0)
        for index in range(1, 151):
            actual.append(ramp.sample(3.0 + index * 0.02)[0])
        self.assertEqual(actual[-1], 0.0)
        self.assertLessEqual(np.max(-np.diff(actual)), 0.02 / 3.0 + 1e-12)

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
        release = None
        for task_s in np.arange(RELEASE_START_S, RELEASE_START_S + 1.5001, 0.02):
            release = plan.sample(task_s, target, np.zeros(13), quat, 0.0, 0.02)
        self.assertAlmostEqual(release["weight"], 0.5)
        done = None
        for task_s in np.arange(RELEASE_START_S + 1.52, END_S + 0.0001, 0.02):
            done = plan.sample(task_s, target, np.zeros(13), quat, 0.0, 0.02)
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
