"""Interval semantics, preserved feedback, admission and worker parity; no DDS."""
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from disturbance_types import DisturbanceHorizon, DisturbanceInput
from hardware_mpc_predictor import PredictionResult
from hardware_mpc_forecast_timing import align_acceleration_preview, shifted_interval_means
from g1_walk_mpc import (MpcRuntime, build_parser, validate_accel_forecast_alignment,
                        LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG, LEARNED_TORQUE_CONFIG)


def example(age_ms=1.):
    nodes = tuple(DisturbanceInput(np.ones(3)*k, np.array([.1, .2, k*.01]),
                  np.ones(3)*(-k), Rotation.from_rotvec([.001*k, .002*k, 0]).as_matrix())
                  for k in range(10))
    intervals = tuple(DisturbanceInput(np.ones(3)*k*6., nodes[k].omega_world.copy(),
                      np.ones(3)*k*12., nodes[k].rot_world_body.copy()) for k in range(9))
    return PredictionResult(DisturbanceHorizon(nodes, intervals),
        dict(mode='learned_filtered', anchor_monotonic_ns=100_000_000,
             query_monotonic_ns=100_000_000+round(age_ms*1e6), forecast_y=['preserved']))


class ForecastTimingTests(unittest.TestCase):
    def test_fractional_interval_integral_and_tail_hold(self):
        values = np.arange(9.)[:, None]*6.
        np.testing.assert_allclose(shifted_interval_means(values, 3.)[:, 0],
                                   np.r_[np.arange(8.)*6.+3., 48.])
        np.testing.assert_allclose(shifted_interval_means(values, 6.)[:, 0],
                                   np.r_[np.arange(1.,9.)*6., 48.])
        # A spike is averaged by overlap, not evaluated at the interval midpoint.
        spike = np.zeros((9,1)); spike[1] = 12.
        self.assertAlmostEqual(shifted_interval_means(spike, 1.)[0,0], 2.)
        np.testing.assert_array_equal(shifted_interval_means(values, 0.), values)

    def test_alignment_preserves_all_measured_and_orientation_feedback(self):
        original = example(); aligned = align_acceleration_preview(original)
        self.assertIs(aligned.horizon.nodes, original.horizon.nodes)
        np.testing.assert_allclose(aligned.horizon.intervals[0].acc_world, 4.5)
        np.testing.assert_allclose(aligned.horizon.intervals[0].alpha_world, 9.)
        for old, new in zip(original.horizon.intervals, aligned.horizon.intervals):
            np.testing.assert_array_equal(old.omega_world, new.omega_world)
            np.testing.assert_array_equal(old.rot_world_body, new.rot_world_body)
        self.assertNotIn('acceleration_forecast_alignment', original.diagnostics)
        self.assertEqual(aligned.diagnostics['forecast_y'], ['preserved'])
        evidence = aligned.diagnostics['acceleration_forecast_alignment']
        self.assertEqual(evidence['estimated_host_write_monotonic_ns'], 104_500_000)
        self.assertEqual(evidence['extrapolated_tail_ms'], 4.5)

    def test_hold_current_does_not_apply_alignment(self):
        original = example(); original.diagnostics['mode'] = 'hold_current'
        self.assertIs(align_acceleration_preview(original), original)

    def test_nonfinite_or_noncausal_inputs_are_rejected(self):
        for age in (-.001, 2., 10.):
            with self.assertRaises(ValueError): align_acceleration_preview(example(age))
        for lead in (-1., 12.01, float('nan'), float('inf')):
            with self.assertRaises(ValueError): shifted_interval_means(np.zeros((9,3)), lead)
        with self.assertRaises(ValueError): shifted_interval_means(np.full((9,3),np.nan), 3.)

    def test_flag_off_by_default_and_only_baseline_learned_pair_admitted(self):
        self.assertFalse(build_parser(learned=True).parse_args([]).align_accel_forecast)
        args = [LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG, LEARNED_TORQUE_CONFIG,
                'learned_filtered', 'measured_torque_preview', None]
        validate_accel_forecast_alignment(*args)
        for index, value in [(0, 'configs/hardware_mpc_learned.yaml'), (1,None),
                             (2,'hold_current'), (3,'reference_servo'), (4,0.)]:
            changed = args.copy(); changed[index] = value
            with self.assertRaises(ValueError): validate_accel_forecast_alignment(*changed)

    def test_runtime_only_applies_during_learned_walk_and_preserves_zero_targets(self):
        runtime = MpcRuntime(config=LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG,
            torque_config=LEARNED_TORQUE_CONFIG, zero_arm_neutral=True, align_accel_forecast=True)
        self.addCleanup(runtime.close)
        np.testing.assert_array_equal(runtime.target_q[:10], 0.)
        baseline = example()
        for task, frozen, expected in [(4.999,True,False),(5.,False,False),(5.,True,True),(17.,True,True)]:
            def query(*args, use_learned, **kwargs):
                return PredictionResult(baseline.horizon, dict(baseline.diagnostics,
                    mode='learned_filtered' if use_learned else 'hold_current'))
            with mock.patch.object(runtime.predictor, 'query', side_effect=query):
                runtime.prepare(101_000_000, None, None, 0., task, frozen)
            self.assertEqual('acceleration_forecast_alignment' in runtime.controller._extra, expected)
            for node, old in zip(runtime.controller._horizon.nodes, baseline.horizon.nodes):
                for key in ('acc_world', 'alpha_world', 'omega_world', 'rot_world_body'):
                    np.testing.assert_array_equal(getattr(node,key), getattr(old,key))
        # Disabling the option preserves values (the controller copies its input).
        runtime.align_accel_forecast = False
        with mock.patch.object(runtime.predictor, 'query', return_value=baseline):
            runtime.prepare(101_000_000,None,None,0.,6.)
        self.assertNotIn('acceleration_forecast_alignment',runtime.controller._extra)
        for node, old in zip(runtime.controller._horizon.intervals, baseline.horizon.intervals):
            for key in ('acc_world', 'alpha_world', 'omega_world', 'rot_world_body'):
                np.testing.assert_array_equal(getattr(node,key), getattr(old,key))

    def test_spawned_aligned_worker_matches_direct_and_releases_after_worker_death(self):
        from test_mpc_compute_process import ComputeProcessTests
        case = ComputeProcessTests()
        self.addCleanup(case.doCleanups)
        case.check_runtime_parity(dict(config=LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG,
            torque_config=LEARNED_TORQUE_CONFIG, predictor_mode='learned_filtered',
            stationary=False, field_trial=True, assumed_command_delay_s=None,
            zero_arm_neutral=True, align_accel_forecast=True))


if __name__ == '__main__': unittest.main()
