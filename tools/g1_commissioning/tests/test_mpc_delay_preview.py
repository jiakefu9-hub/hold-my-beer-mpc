"""Offline delay-estimate checks with analytic, not simulated-plant, oracles.

The small mass model below exposes only nominal mass/bias.  The predictor must
derive its state from a measured q/dq and its own issued-command history; no
true plant state, mass mismatch, or real transport-delay measurement is supplied.
"""
from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from disturbance_types import DisturbanceHorizon, DisturbanceInput
from g1_walk_pid import EXPECTED_TARGET_Q
from hardware_mpc_control import HardwareMpcError
from hardware_mpc_delay_preview import HorizonClock, IssuedCommand, RightArmDelayPreviewMpc
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc


def make_horizon(varying=False):
    nodes = []
    for k in range(10):
        t = .006*k
        nodes.append(DisturbanceInput(
            np.array([t, 2*t, -t]) if varying else np.zeros(3),
            np.array([0., 0., 2.]) if varying else np.zeros(3),
            np.array([t, -t, .5*t]) if varying else np.zeros(3),
            Rotation.from_rotvec([0., 0., 2*t]).as_matrix() if varying else np.eye(3)))
    return DisturbanceHorizon(tuple(nodes), tuple(nodes[:-1]))


def analytic_controller(delay=.004, bias=0., limit=1000.):
    """Construct just the nominal propagation interface, without a QP or DDS."""
    controller = object.__new__(RightArmDelayPreviewMpc)
    controller.assumed_command_delay_s = delay
    controller._issued = deque()
    controller._delay_context = None
    controller._last_now = None
    controller._last_observed = None
    controller.inverse = SimpleNamespace(linear_dynamics=lambda q, dq, base:
                                         (np.eye(5), np.full(5, bias)))
    controller.mapper = SimpleNamespace(limit=limit)
    controller.torque_config = dict(kp=np.zeros(5), kd=np.zeros(5))
    return controller


def packet(t, acceleration):
    return IssuedCommand(t, np.full(5, acceleration), np.zeros(5), np.zeros(5))


class HorizonClockTest(unittest.TestCase):
    def test_zero_shift_is_exact_original_forecast(self):
        horizon = make_horizon(True)
        self.assertIs(HorizonClock(horizon).shifted(0.), horizon)

    def test_interpolated_and_extrapolated_rotations_remain_on_so3(self):
        clock = HorizonClock(make_horizon(True))
        for t in (0., .001, .005, .009, .053, .054, .073, .094):
            with self.subTest(t=t):
                d = clock.at(t)
                np.testing.assert_allclose(d.rot_world_body.T@d.rot_world_body, np.eye(3), atol=1e-14)
                self.assertAlmostEqual(np.linalg.det(d.rot_world_body), 1.)
                np.testing.assert_allclose(d.rot_world_body,
                                          Rotation.from_rotvec([0., 0., 2*t]).as_matrix(), atol=1e-14)
                sampled_t = min(t, .054)
                np.testing.assert_allclose(d.acc_world, [sampled_t, 2*sampled_t, -sampled_t], atol=1e-15)

    def test_shift_preserves_node_and_interval_time_conventions(self):
        clock = HorizonClock(make_horizon(True))
        shifted = clock.shifted(.01)
        self.assertEqual((len(shifted.nodes), len(shifted.intervals)), (10, 9))
        for k, d in enumerate(shifted.nodes):
            expected = clock.at(.01+.006*k)
            np.testing.assert_allclose(d.acc_world, expected.acc_world)
            np.testing.assert_allclose(d.rot_world_body, expected.rot_world_body)
        for k, d in enumerate(shifted.intervals):
            values = [clock.at(.01+.006*k+.002*j).acc_world for j in range(3)]
            np.testing.assert_allclose(d.acc_world, np.mean(values, axis=0))
            np.testing.assert_allclose(d.rot_world_body, clock.at(.01+.006*k+.003).rot_world_body)

    def test_invalid_forecast_time_is_rejected(self):
        clock = HorizonClock(make_horizon())
        for t in (-.001, np.nan, np.inf, -np.inf):
            with self.subTest(t=t), self.assertRaises(ValueError):
                clock.at(t)

    def test_vectorized_forecast_matches_independent_scalar_reference(self):
        """Independent scalar interpolation, including noncommuting rotations.

        Use Slerp rather than the implementation's batched relative rotvectors;
        the post-horizon rotation uses a separately constructed Rodrigues map.
        """
        rng = np.random.default_rng(28092026)
        matrices = Rotation.random(10, random_state=rng).as_matrix()
        fields = rng.normal(size=(10, 3, 3))
        horizon = DisturbanceHorizon(
            tuple(DisturbanceInput(*values, matrix) for values, matrix in zip(fields, matrices)),
            tuple(DisturbanceInput(*values, matrix) for values, matrix in zip(fields[:-1], matrices[:-1])))

        def scalar_reference(t):
            if t >= .054:
                vector = fields[-1, 1]*(t-.054)
                angle = np.linalg.norm(vector)
                if angle == 0.:
                    increment = np.eye(3)
                else:
                    x, y, z = vector/angle
                    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                    increment = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
                return fields[-1].copy(), increment@matrices[-1]
            index = max(0, int(np.searchsorted(np.arange(10)*.006, t, side='right')-1))
            index = min(index, 8)
            fraction = (t-.006*index)/.006
            values = fields[index]*(1-fraction)+fields[index+1]*fraction
            attitude = Slerp([0., .006], Rotation.from_matrix(matrices[index:index+2]))
            return values, attitude([t-.006*index]).as_matrix()[0]

        clock = HorizonClock(horizon)
        for t in (0., .0013, .006, .018, .053999, .054, .068, .094):
            with self.subTest(sample_time=t):
                expected, rotation = scalar_reference(t)
                actual = clock.at(t)
                np.testing.assert_allclose([actual.acc_world, actual.omega_world, actual.alpha_world],
                                          expected, atol=1e-13, rtol=0.)
                np.testing.assert_allclose(actual.rot_world_body, rotation, atol=1e-13, rtol=0.)
        for offset in (.0013, .006, .010, .024, .04, .054, .090):
            shifted = clock.shifted(offset)
            for k, actual in enumerate(shifted.nodes):
                with self.subTest(offset=offset, node=k):
                    expected, rotation = scalar_reference(offset+k*.006)
                    np.testing.assert_allclose([actual.acc_world, actual.omega_world, actual.alpha_world],
                                              expected, atol=1e-13, rtol=0.)
                    np.testing.assert_allclose(actual.rot_world_body, rotation, atol=1e-13, rtol=0.)
            for k, actual in enumerate(shifted.intervals):
                with self.subTest(offset=offset, interval=k):
                    values = [scalar_reference(offset+k*.006+j*.002)[0] for j in range(3)]
                    _, rotation = scalar_reference(offset+k*.006+.003)
                    np.testing.assert_allclose([actual.acc_world, actual.omega_world, actual.alpha_world],
                                              np.mean(values, axis=0), atol=1e-13, rtol=0.)
                    np.testing.assert_allclose(actual.rot_world_body, rotation, atol=1e-13, rtol=0.)


class CausalPropagationTest(unittest.TestCase):
    def test_non_grid_command_boundaries_use_history_in_application_order(self):
        controller = analytic_controller()
        controller._issued.extend([packet(.003, 2.), packet(.009, -1.), packet(.030, 900.)])
        q, dq, evidence = controller._predict(np.zeros(5), np.zeros(5),
                                              HorizonClock(make_horizon()), .010, 0.)
        # 0--3 ms hold; 3--9 ms a=2; 9--14 ms a=-1. Future packet is ignored.
        np.testing.assert_allclose(q, np.full(5, .0000835), atol=1e-15)
        np.testing.assert_allclose(dq, np.full(5, .007), atol=1e-15)
        self.assertAlmostEqual(evidence['observation_age_s'], .010)
        self.assertAlmostEqual(evidence['target_minus_observation_s'], .014)
        self.assertAlmostEqual(evidence['command_time_s'], .014)

    def test_history_pruning_preserves_packet_active_at_observation(self):
        controller = analytic_controller()
        controller._issued.extend([packet(-.020, 900.), packet(-.001, 3.), packet(.030, 900.)])
        q, dq, _ = controller._predict(np.zeros(5), np.zeros(5),
                                       HorizonClock(make_horizon()), 0., 0.)
        self.assertEqual(controller._issued[0].apply_s, -.001)
        np.testing.assert_allclose(q, np.full(5, .5*3*.004**2), atol=1e-15)
        np.testing.assert_allclose(dq, np.full(5, 3*.004), atol=1e-15)

    def test_initial_hold_uses_nominal_bias_without_inventing_a_command(self):
        controller = analytic_controller(bias=5.)
        measured_q = np.arange(5.)*.01
        q, dq, _ = controller._predict(measured_q.copy(), np.zeros(5),
                                       HorizonClock(make_horizon()), 0., 0.)
        np.testing.assert_array_equal(q, measured_q)
        np.testing.assert_array_equal(dq, np.zeros(5))
        self.assertEqual(len(controller._issued), 0)

    def test_zero_prediction_span_preserves_measurement_exactly(self):
        controller = analytic_controller(delay=0.)
        q0, dq0 = np.arange(5.)*.02, np.arange(5.)*.03
        q, dq, evidence = controller._predict(q0.copy(), dq0.copy(),
                                              HorizonClock(make_horizon()), 1., 1.)
        np.testing.assert_array_equal(q, q0)
        np.testing.assert_array_equal(dq, dq0)
        self.assertEqual(evidence['prediction_steps'], 0)

    def test_same_model_torque_limit_is_applied_during_prediction(self):
        controller = analytic_controller(limit=3.)
        controller._issued.append(packet(0., 100.))
        q, dq, _ = controller._predict(np.zeros(5), np.zeros(5),
                                       HorizonClock(make_horizon()), 0., 0.)
        np.testing.assert_allclose(q, np.full(5, .5*3*.004**2), atol=1e-15)
        np.testing.assert_allclose(dq, np.full(5, 3*.004), atol=1e-15)

    def test_pd_is_recomputed_once_per_nominal_model_step(self):
        controller = analytic_controller(delay=.006, bias=1.)
        controller.torque_config = dict(kp=np.full(5, 20.), kd=np.ones(5))
        controller._issued.append(IssuedCommand(0., np.full(5, 2.),
                                               np.full(5, .1), np.full(5, .05)))
        expected_q, expected_dq = np.full(5, .01), np.full(5, .02)
        for _ in range(3):
            ddq = 2.+20.*(.1-expected_q)+(.05-expected_dq)-1.
            expected_q = expected_q+.002*expected_dq+.5*.002**2*ddq
            expected_dq = expected_dq+.002*ddq
        q, dq, _ = controller._predict(np.full(5, .01), np.full(5, .02),
                                       HorizonClock(make_horizon()), 0., 0.)
        np.testing.assert_allclose(q, expected_q, atol=1e-15)
        np.testing.assert_allclose(dq, expected_dq, atol=1e-15)

    def test_timestamp_window_and_clock_reversal_fail_closed(self):
        controller = analytic_controller(delay=.006)
        for now, observed in ((0., .001), (1., .965), (np.nan, 0.), (1., np.inf)):
            with self.subTest(now=now, observed=observed), self.assertRaises(ValueError):
                controller.set_delay_context(now, observed)
        controller.set_delay_context(1., .966)  # 34 ms age plus 6 ms command delay.
        self.assertEqual(controller._delay_context, (1., .966))
        controller._last_now = 1.
        with self.assertRaises(ValueError):
            controller.set_delay_context(.999, .99)
        controller._last_observed = .995
        with self.assertRaises(ValueError):
            controller.set_delay_context(1.006, .994)


class DelayControllerIntegrationTest(unittest.TestCase):
    def test_zero_delay_matches_measured_state_controller_and_consumes_context(self):
        original = RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10])
        preview = RightArmDelayPreviewMpc(EXPECTED_TARGET_Q[5:10], assumed_command_delay_s=0.)
        try:
            dq = np.array([.01, -.005, .002, .003, -.002])
            outputs = []
            for controller in (original, preview):
                controller.set_measured_dq(dq)
                controller.set_disturbance_horizon(make_horizon())
                if controller is preview:
                    controller.set_delay_context(1., 1.)
                outputs.append(controller.step(EXPECTED_TARGET_Q, [1., 0., 0., 0.], 0., .006))
            for key in ('mpc_initial_state', 'raw_mpc_ddq_rad_s2', 'tau_ff_candidate_nm'):
                np.testing.assert_allclose(outputs[1][2][key], outputs[0][2][key], rtol=0., atol=1e-11)
            np.testing.assert_allclose(outputs[1][0], outputs[0][0], atol=1e-13)
            np.testing.assert_allclose(outputs[1][1], outputs[0][1], atol=1e-13)
            np.testing.assert_array_equal(preview._measured_dq, dq)
            self.assertTrue(preview.offline_only)
            self.assertFalse(preview.metadata['delay_preview']['field_runner_supported'])
            self.assertFalse(outputs[1][2]['torque_output_authorized'])
            self.assertGreaterEqual(outputs[1][2]['controller_core_ms'],
                                    outputs[1][2]['inner_controller_core_ms'])
            self.assertEqual(len(preview._issued), 1)
            self.assertEqual(preview._issued[0].apply_s, 1.)
            preview.set_disturbance_horizon(make_horizon())
            with self.assertRaises(HardwareMpcError):
                preview.step(EXPECTED_TARGET_Q, [1., 0., 0., 0.], 0., .006)
            for slots, quaternion, yaw, dt in (
                    (EXPECTED_TARGET_Q[:12], [1., 0., 0., 0.], 0., .006),
                    (EXPECTED_TARGET_Q, [0., 0., 0., 0.], 0., .006),
                    (EXPECTED_TARGET_Q, [1., 0., 0., 0.], np.nan, .006),
                    (EXPECTED_TARGET_Q, [1., 0., 0., 0.], 0., 0.)):
                with self.subTest(slots=slots, quaternion=quaternion, yaw=yaw, dt=dt):
                    preview.set_delay_context(1.006, 1.006)
                    with self.assertRaises(ValueError):
                        preview.step(slots, quaternion, yaw, dt)
            self.assertEqual(len(preview._issued), 1)  # No malformed call queued an output.
        finally:
            original.close()
            preview.close()

    def test_delay_assumption_is_explicit_and_bounded(self):
        for delay in (-.001, .021, np.nan, np.inf):
            with self.subTest(delay=delay), self.assertRaises(ValueError):
                RightArmDelayPreviewMpc(EXPECTED_TARGET_Q[5:10], assumed_command_delay_s=delay)


if __name__ == '__main__':
    unittest.main()
