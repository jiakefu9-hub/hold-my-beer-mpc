"""Geometry/time equivalence, deadline handling and bounded 6 ms output."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from endpoint_pose import EndpointModel
from hardware_pid_control import RightArmGravityHelper, RightArmHardwarePid, PidParameters, WeightReleaseRamp
from pid_timing import PeriodicClock, timing_summary
from tools.g1_commissioning.tests.test_hardware_pid_control import PARAMETERS
from arm_pid import ArmPIDPolicy


class PidSixMsTest(unittest.TestCase):
    def test_analytic_jacobian_against_central_difference(self):
        rng = np.random.default_rng(20260924)
        model = EndpointModel()
        reference = ArmPIDPolicy(np.zeros(5))
        for _ in range(120):
            slots = rng.uniform(-0.6, 0.6, 13)
            rotation = Rotation.random(random_state=rng).as_matrix()
            helper = RightArmGravityHelper(model, slots, rng.uniform(-math.pi, math.pi))
            error, jac = helper.compute_gravity_error_and_jacobian(slots[5:10], rotation)
            expected = helper.compute_gravity_error(slots[5:10], rotation)
            numeric = reference._compute_gravity_error_jacobian(
                slots[5:10], rotation, {"compute_gravity_error": helper.compute_gravity_error})
            np.testing.assert_allclose(error, expected, atol=1e-12)
            np.testing.assert_allclose(jac, numeric, atol=3e-8, rtol=3e-7)

    def test_pid_analytic_matches_numeric_at_same_period(self):
        slots = np.zeros(13)
        helper = RightArmGravityHelper(EndpointModel(), slots, 0.4)
        a, b = ArmPIDPolicy(np.zeros(5)), ArmPIDPolicy(np.zeros(5))
        for angle in np.linspace(0.0, 0.1, 30):
            obs = {"current_q": slots[5:10], "current_dq": np.zeros(5),
                   "torso_rotmat": Rotation.from_euler("x", angle).as_matrix(), "dt": 0.006}
            numeric = a.compute_action(obs, {"compute_gravity_error": helper.compute_gravity_error})
            analytic = b.compute_action(obs, {"compute_gravity_error_and_jacobian": helper.compute_gravity_error_and_jacobian})
            np.testing.assert_allclose(analytic, numeric, atol=2e-7)

    def test_physical_filter_and_task_strength_survive_period_change(self):
        def run(dt):
            policy = ArmPIDPolicy(np.zeros(5), kp_pose=1, kd_pose=0, max_dq=100,
                                  de_g_alpha=0.07, task_reference_dt=0.02,
                                  derivative_filter_reference_dt=0.02)
            obs = {"current_q": np.zeros(5), "current_dq": np.zeros(5),
                   "torso_rotmat": np.eye(3), "dt": dt}
            policy.compute_action(obs, {"compute_gravity_error_and_jacobian":
                lambda q, R: (np.array([0.01, 0.02]), np.eye(2, 5))})
            return policy.get_last_diagnostics()
        slow, fast = run(0.02), run(0.006)
        np.testing.assert_allclose(slow["task_dq"], fast["task_dq"])
        self.assertAlmostEqual((1 - slow["derivative_alpha"]) ** 3,
                               (1 - fast["derivative_alpha"]) ** 10)

    def test_deadline_skips_without_catchup(self):
        clock = PeriodicClock(0, 0.006)
        self.assertEqual(clock.advance(2_000_000), 0)
        self.assertEqual(clock.scheduled_ns, 6_000_000)
        self.assertEqual(clock.advance(23_000_000), 2)
        self.assertEqual(clock.scheduled_ns, 24_000_000)
        self.assertEqual(timing_summary([]), {"available": False})

    def test_6ms_governor_and_release_with_stalls(self):
        nominal = np.deg2rad([-4, 1, 0, -7.8, 0])
        controller = RightArmHardwarePid(nominal, PidParameters.from_mapping(PARAMETERS))
        slots = np.zeros(13)
        slots[5:10] = nominal
        previous_dq = np.zeros(5)
        previous_q = None
        for index in range(1800):
            dt = 0.018 if index % 27 == 0 else 0.006
            quat = [math.cos(0.1), math.sin(0.1), 0, 0]
            q, dq, diag = controller.step(slots, quat, 0.0, dt)
            self.assertLessEqual(np.max(np.abs(dq)), 0.07 + 1e-10)
            self.assertLessEqual(np.max(np.abs(dq - previous_dq)), 0.2 * 0.006 + 1e-10)
            self.assertTrue(np.all(np.abs(q - nominal) <= np.deg2rad(5) + 1e-10))
            if previous_q is not None:
                np.testing.assert_allclose((q - previous_q) / 0.006, dq, atol=1e-10)
            previous_q, previous_dq = q, dq
        release = WeightReleaseRamp(1.0)
        self.assertEqual(release.sample(0.0), (1.0, False))
        self.assertAlmostEqual(release.sample(10.0)[0], 0.998)
        for index in range(499):
            weight, done = release.sample(10.0 + (index + 1) * 0.006)
        self.assertEqual((weight, done), (0.0, True))


if __name__ == "__main__":
    unittest.main()
