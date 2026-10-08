"""SDK-free checks of whole-run holdout and reported-torque regression."""
from copy import deepcopy
import unittest

import numpy as np

from analyze_arm_system_identification import NOMINAL
from review_arm_identification_batch import (cross_run_review, features,
    friction_ablation, simple_telemetry_fit)


def synthetic_run(seed):
    rng = np.random.default_rng(seed)
    n = 180
    source_t = np.arange(n + 20)*.006
    source_ff = rng.normal(size=(n + 20, 5))*.1
    qcmd = np.tile(NOMINAL, (n + 20, 1))
    dqcmd = np.zeros_like(qcmd)
    run = dict(time=source_t[10:-10], command_time=source_t,
        source_ff=source_ff, source_qcmd=qcmd, source_dqcmd=dqcmd,
        q_measured_rad=NOMINAL + rng.normal(size=(n, 5))*.02,
        dq_measured_rad_s=rng.normal(size=(n, 5))*.1,
        kp_command=np.full((n, 5), 20.), kd_command=np.ones((n, 5)),
        imu_uncentered=rng.normal(size=(n, 6)),
        q_command_rad=qcmd[10:-10], dq_command_rad_s=dqcmd[10:-10])
    # Fixed positive coupled map; seeds change excitation, not the plant.
    coefficient = np.zeros((26, 5))
    coefficient[:5] = np.diag([2., 3., 4., 5., 6.]) + .05
    coefficient[5:10] = -.4*np.eye(5)
    coefficient[10:15] = -.3*np.eye(5)
    run['qdd'] = features(run, 0) @ coefficient
    request = (source_ff[10:-10]
        + run['kp_command']*(run['q_command_rad']-run['q_measured_rad'])
        - run['kd_command']*run['dq_measured_rad_s'])
    run['tau_est_at_feedback_nm'] = request*np.array([.9, .8, 1., .95, .85]) + .1
    return run


class ArmIdentificationReviewTests(unittest.TestCase):
    def test_third_run_cannot_change_parameters_or_delay(self):
        runs = [synthetic_run(i) for i in range(3)]
        first, _ = cross_run_review(runs)
        changed = deepcopy(runs)
        changed[2]['qdd'] += 10.
        changed[2]['imu_uncentered'] *= 20.
        second, _ = cross_run_review(changed)
        self.assertEqual(first['selected_delay_ms'], second['selected_delay_ms'])
        np.testing.assert_array_equal(first['full_coefficients'], second['full_coefficients'])
        np.testing.assert_array_equal(first['full_intercept'], second['full_intercept'])
        self.assertLess(max(first['full_rmse_rad_s2']), 1e-4)
        self.assertGreater(max(second['full_rmse_rad_s2']), 9.)

    def test_tau_est_fit_is_recoverable_and_independent_of_test_target(self):
        runs = [synthetic_run(i) for i in range(3)]
        fit = simple_telemetry_fit(runs)
        np.testing.assert_allclose(fit['gain'], [.9, .8, 1., .95, .85], atol=1e-12)
        np.testing.assert_allclose(fit['offset_nm'], .1, atol=1e-12)
        self.assertEqual(fit['selected_delay_ms'], 0)
        runs[2]['tau_est_at_feedback_nm'] += 3.
        second = simple_telemetry_fit(runs)
        self.assertEqual(fit['gain'], second['gain'])
        self.assertEqual(fit['offset_nm'], second['offset_nm'])
        self.assertEqual(fit['selected_delay_ms'], second['selected_delay_ms'])
        np.testing.assert_allclose(second['heldout_rmse_nm'], 3.)

    def test_fixed_rigid_model_friction_fit_preserves_nominal_and_holdout(self):
        runs = [synthetic_run(i) for i in range(3)]
        dq = np.vstack([r['dq_measured_rad_s'] for r in runs])
        nominal = np.full_like(dq, .2)
        before = nominal.copy()
        friction = np.array([.15, .25, .2, .3, .1])
        actual = nominal + .1 + np.tanh(dq/.05)*friction
        review = friction_ablation(runs, nominal, actual)
        candidate = review['candidates'][1]
        np.testing.assert_allclose(candidate['train12_coulomb_nm'], friction, atol=1e-10)
        np.testing.assert_allclose(candidate['train12_bias_nm'], .1, atol=1e-10)
        np.testing.assert_allclose(candidate['heldout_bias_and_friction_rmse_nm'], 0., atol=1e-10)
        np.testing.assert_array_equal(nominal, before)
        actual[-len(runs[2]['time']):] += 2.
        updated = friction_ablation(runs, nominal, actual)['candidates'][1]
        self.assertEqual(candidate['train12_coulomb_nm'], updated['train12_coulomb_nm'])
        np.testing.assert_allclose(updated['heldout_bias_and_friction_rmse_nm'], 2.)


if __name__ == '__main__':
    unittest.main()
