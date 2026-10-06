"""Recorded-roll failure, finite recovery deadline and final model-state guard.

No SDK participants or robot output. Model-feasible is not physical validation.
"""
import unittest
import numpy as np

from test_mpc_recovery import make_policy, independent_feasibility
from hardware_mpc_torque_control import load_torque_config
from hardware_torque_mapper import LocalTorqueMapper, NoModelTorque

Q=np.array([.03682262742706079,.014208164468590686,.03781628631741912,
            -.07090770205825297,-.02145567535984485])
V=np.array([.2180235653715588,.269350583001252,.0014398090834525099,
            .17857335957469692,.06259325842730817])


class ReentryTests(unittest.TestCase):
    def policy(self):
        p=make_policy(recovery_reentry_enabled=True)
        p.set_recovery_time(0.)
        return p

    def test_recorded_roll_failure_and_fixed_terminal_envelope(self):
        self.assertEqual(independent_feasibility(make_policy(),Q,V).status,2)
        p=self.policy()
        self.assertTrue(independent_feasibility(p,Q,V).success)
        lo,hi,*_=p._build_online_constraint_bounds(Q,V)
        oldlo,oldhi,*_=make_policy()._build_online_constraint_bounds(Q,V)
        np.testing.assert_array_equal(lo[:p.recovery_row_start],oldlo[:p.recovery_row_start])
        np.testing.assert_array_equal(hi[:p.recovery_row_start],oldhi[:p.recovery_row_start])
        np.testing.assert_array_equal(lo[-5:],oldlo[-5:])
        np.testing.assert_array_equal(hi[-5:],oldhi[-5:])
        self.assertTrue(np.all(np.diff(hi[p.recovery_row_start:].reshape(9,5),axis=0)<=0.))
        a,b=p.first_step_acceleration_bounds(Q,V)
        self.assertAlmostEqual(b[1],-2.60892557,places=6)
        self.assertTrue(np.all(a>=-8));self.assertTrue(np.all(b<=8))

    def test_grace_window_cannot_be_restarted_by_new_samples(self):
        p=self.policy();p._build_online_constraint_bounds(Q,V)
        initial=p._reentry_start.copy()
        p.set_recovery_time(.030);_,hi,*_=p._build_online_constraint_bounds(Q,V)
        np.testing.assert_array_equal(p._reentry_start,initial)
        self.assertAlmostEqual(p.reentry_diagnostics['elapsed_s'][1],.03)
        p.set_recovery_time(.054)
        with self.assertRaisesRegex(ValueError,'deadline exceeded'):p._build_online_constraint_bounds(Q,V)

    def test_returns_to_strict_mode_and_rejects_invalid_state_or_time(self):
        p=self.policy();p._build_online_constraint_bounds(Q,V)
        p.set_recovery_time(.018);p._build_online_constraint_bounds(Q,np.zeros(5))
        self.assertFalse(p.reentry_diagnostics['active'])
        with self.assertRaisesRegex(ValueError,'clock'):p.set_recovery_time(.01)
        with self.assertRaisesRegex(ValueError,'stale'):p.first_step_acceleration_bounds(Q,V)
        p.reset()
        with self.assertRaisesRegex(ValueError,'clock'):p._build_online_constraint_bounds(Q,V)
        p.set_recovery_time(0.)
        for q,v in [(Q+np.array([0,.1,0,0,0]),V),(Q,V*10)]:
            with self.assertRaisesRegex(ValueError,'outer'):p._build_online_constraint_bounds(q,v)

    def test_symmetric_lower_reentry(self):
        p=self.policy();q=Q.copy();v=V.copy()
        q[1]=np.deg2rad(-2.81406786);v[1]=-V[1]
        self.assertTrue(independent_feasibility(p,q,v).success)
        a,b=p.first_step_acceleration_bounds(q,v)
        self.assertGreater(a[1],2.6)

    def test_reentry_cannot_be_enabled_without_final_mapper_guard(self):
        with self.assertRaisesRegex(ValueError,'requires'):
            load_torque_config(dict(recovery_envelope_enabled=True,recovery_rate_s_inv=6,
                recovery_guard_deg=[.15]*5,recovery_reentry_enabled=True))

    def test_mapper_must_brake_not_just_stay_below_ten(self):
        m=LocalTorqueMapper(load_torque_config(dict(tau_abs_nm=[5.]*5)))
        z=np.zeros(5);nominal=np.ones(5)
        upper=np.full(5,8.);upper[1]=-2.
        tau,d=m.compute(lambda x:x,np.ones(5),nominal,z,affine_gain=np.eye(5),
            forward_batch=lambda x:x,acceleration_bounds=(-np.full(5,8.),upper))
        self.assertEqual(d['fallback'],'state_envelope_projection_rechecked')
        self.assertAlmostEqual(tau[1],-2.,places=7)
        self.assertTrue(np.all(tau<=upper+1e-8));self.assertTrue(d['model_accepted'])

    def test_unrealizable_braking_still_rejected_without_widening_torque(self):
        m=LocalTorqueMapper(load_torque_config(dict(tau_abs_nm=[.1]*5)))
        z=np.zeros(5);upper=np.full(5,8.);upper[1]=-2.
        with self.assertRaises(NoModelTorque):
            m.compute(lambda x:x,z,z,z,affine_gain=np.eye(5),
                forward_batch=lambda x:x,acceleration_bounds=(-np.full(5,8.),upper))


if __name__=='__main__':unittest.main()
