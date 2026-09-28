"""Check delay provenance and clocks without changing any hardware capability."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from validate_measured_torque_mpc import (SCENARIOS, Scenario, disturbance,
    failure_evidence, first_contact, resolve_scenario, run_case)


class ScenarioSemanticsTest(unittest.TestCase):
    def test_legacy_boolean_and_named_scenarios(self):
        self.assertEqual(resolve_scenario(None,True),SCENARIOS['combined'])
        self.assertEqual(resolve_scenario(None,False),SCENARIOS['matched'])
        self.assertEqual(resolve_scenario('payload_only').observation_delay_s,0.)
        self.assertEqual(resolve_scenario('observation_only').payload_scale,1.)
        with self.assertRaisesRegex(ValueError,'2 ms'):
            Scenario(observation_delay_s=.003)

    def test_report_first_physical_contact_not_later_control_sample(self):
        values=np.ones((4,5));values[1,3]=-.001;values[3,0]=-.1
        self.assertEqual(first_contact(np.arange(4)*.002,values),
                         dict(time_s=.002,joint_index=3,signed_margin=-.001))

    def test_rotation_omega_alpha_are_same_world_frame(self):
        for t in (.0,.123,.57):
            epsilon=1e-6
            before,base,after=disturbance(t-epsilon),disturbance(t),disturbance(t+epsilon)
            skew=(after.rot_world_body-before.rot_world_body)/(2*epsilon)@base.rot_world_body.T
            np.testing.assert_allclose([skew[2,1],skew[0,2],skew[1,0]],base.omega_world,atol=1e-9)
            np.testing.assert_allclose((after.omega_world-before.omega_world)/(2*epsilon),
                                       base.alpha_world,atol=1e-9)

    def test_delay_provenance_and_metric_identity(self):
        arrays,summary=run_case(duration=.024,scenario='both_delays')
        self.assertEqual(summary['status'],'complete')
        np.testing.assert_array_equal(arrays['interval_active_seq'],[-1,-1,-1,0,0,0,1,1,1,2,2,2])
        np.testing.assert_array_equal(arrays['interval_request_seq'],[0,0,0,1,1,1,2,2,2,3,3,3])
        np.testing.assert_allclose(arrays['interval_observation_age_s'],[0,.002]+[.004]*10)
        np.testing.assert_allclose(arrays['interval_active_desired'][3:6],
                                   np.tile(arrays['desired'][0],(3,1)))
        np.testing.assert_allclose(arrays['current_request_vs_actual'],arrays['actual']-arrays['desired'])
        np.testing.assert_allclose(arrays['actually_active_command_desired_vs_actual'],
                                   arrays['actual']-arrays['active_desired'])
        self.assertAlmostEqual(summary['acceleration_tracking_rmse'],
                               np.sqrt(np.mean((arrays['actual']-arrays['desired'])**2)))
        np.testing.assert_allclose(arrays['physics_t'],np.arange(13)*.002)
        np.testing.assert_allclose(arrays['t'],np.arange(4)*.006)
        self.assertEqual(summary['tracking_2ms']['initial_hold_samples'],3)

    def test_legacy_and_simulation_paths_cannot_silently_ignore_configuration(self):
        for method in ('legacy_reference', 'nominal_inverse', 'simulation_qp_mapper'):
            for options in (dict(torque_config={'recovery_envelope_enabled': True}),
                            dict(assumed_command_delay_s=0.),
                            dict(assumed_command_delay_s=.006),
                            dict(torque_config={}, assumed_command_delay_s=.006)):
                with self.subTest(method=method, options=options):
                    with mock.patch('validate_measured_torque_mpc.RightArmMeasuredTorqueMpc') as measured:
                        with mock.patch('validate_measured_torque_mpc.RightArmHardwareMpc') as legacy:
                            with self.assertRaisesRegex(ValueError, 'no ignored options'):
                                run_case(method=method, duration=.006, **options)
                            measured.assert_not_called()
                            legacy.assert_not_called()

    def test_failure_audit_uses_actual_qp_initial_state_separately_from_measurement(self):
        from hardware_mpc_recovery import RecoveryEnvelopeMpcPolicy
        policy = RecoveryEnvelopeMpcPolicy(np.zeros(5), horizon=9, control_dt=.006,
            max_dq=1., max_ddq=8., joint_limit_margin=np.deg2rad([1., 1., 0., 0., 0.]),
            solver_backend='osqp', solver_time_limit=.5)
        measured_q, measured_dq = np.zeros(5), np.zeros(5)
        true_q, true_dq = np.full(5, -.001), np.zeros(5)
        predicted_q, predicted_dq = measured_q.copy(), np.zeros(5)
        predicted_q[0] = policy.safety_joint_limits[0, 1]+.1
        predicted_dq[0] = 1.
        predicted = np.r_[predicted_q, predicted_dq]
        controller = SimpleNamespace(policy=policy, last_diagnostics={'mpc_initial_state': predicted})
        with tempfile.TemporaryDirectory(prefix='g1-failure-audit-') as directory:
            evidence = failure_evidence(controller, true_q, true_dq, measured_q,
                                        measured_dq, .114, directory)
            self.assertTrue(evidence['independent_constraint_lp']['success'])
            self.assertTrue(evidence['independent_true_state_constraint_lp']['success'])
            self.assertFalse(evidence['independent_qp_initial_constraint_lp']['success'])
            self.assertEqual(evidence['independent_qp_initial_constraint_lp']['status'], 2)
            np.testing.assert_allclose(evidence['qp_initial_q_deg'], np.rad2deg(predicted_q))
            with np.load(Path(directory)/'failure_constraints.npz') as saved:
                for label, initial in (('observed', np.r_[measured_q, measured_dq]),
                                       ('true', np.r_[true_q, true_dq]), ('qp_initial', predicted)):
                    np.testing.assert_array_equal(saved[label+'_lower'][:10], initial)
                    np.testing.assert_array_equal(saved[label+'_upper'][:10], initial)
                # Independently check saved feasible witnesses against all original rows.
                for label in ('observed', 'true'):
                    values = saved['A']@saved[label+'_solution']
                    self.assertLessEqual(np.max(saved[label+'_lower']-values), 1e-7)
                    self.assertLessEqual(np.max(values-saved[label+'_upper']), 1e-7)
                self.assertNotIn('qp_initial_solution', saved.files)
            self.assertTrue((Path(directory)/'failure_state.json').is_file())


if __name__=='__main__':
    unittest.main()
