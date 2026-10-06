"""No-DDS regression: torso bias changes must not freeze support torque."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc, load_torque_config
from hardware_mpc_control import HardwareMpcError
from hardware_torque_mapper import LocalTorqueMapper
from hardware_torque_mapper import NoModelTorque
from audit_measured_torque_replay import check_slew
from test_mpc_recovery import make_policy, independent_feasibility


class BiasSlewTests(unittest.TestCase):
    def controller(self, mode='model_bias_relative', bias=1.):
        c=object.__new__(RightArmMeasuredTorqueMpc)
        c.torque_config=load_torque_config(dict(active_slew_reference=mode,tau_abs_nm=[5.]*5))
        c.mapper=LocalTorqueMapper(c.torque_config)
        def dynamics(q,dq,ddq,base):
            b=np.full(5,bias)
            return dict(tau_model_nm=b+.1*ddq),np.eye(5)*.1,b
        c.inverse=SimpleNamespace(compute_with_linear_dynamics=dynamics)
        c._next_slew_bias=np.zeros(5)
        c._previous_total=np.zeros(5)
        c.last_diagnostics={}
        return c

    def compute(self,c):
        z=np.zeros(5)
        return c._torque(z,z,z,z,z,None,(-np.full(5,.3),np.full(5,.3)))

    def test_support_step_and_residual_are_separate_total_cap_unchanged(self):
        new=self.compute(self.controller())
        old=self.compute(self.controller('total'))
        np.testing.assert_allclose(new['tau_total_estimated_at_feedback_nm'],1.)
        np.testing.assert_allclose(new['mapper']['checked_ddq_rad_s2'],0.,atol=1e-12)
        self.assertGreater(np.max(np.abs(old['mapper']['checked_ddq_rad_s2'])),6.)
        np.testing.assert_allclose(new['torque_slew_bounds_nm'],[np.full(5,.7),np.full(5,1.3)])
        self.assertTrue(np.all(np.abs(new['tau_total_estimated_at_feedback_nm'])<=5))

    def test_no_active_slew_ignores_old_history_box_but_preserves_absolute_limits(self):
        c=self.controller('none');c._next_slew_bias=None
        new=self.compute(c)
        np.testing.assert_allclose(new['tau_total_estimated_at_feedback_nm'],1.)
        self.assertIsNone(new['torque_slew_bounds_nm'])
        self.assertEqual(new['torque_slew_reference'],'none')
        conf=dict(active_slew_reference='none',transition_rate_nm_s=[50.]*5)
        prior=dict(tau_total_estimated_at_feedback_nm=[0.]*5)
        row=dict(new,mpc_active=True,feedback_dt_s=.006)
        check_slew(row,prior,conf)
        row['mpc_active']=False
        with self.assertRaisesRegex(ValueError,'rate envelope'):check_slew(row,prior,conf)

    def test_missing_bias_or_shift_outside_absolute_box_is_rejected(self):
        c=self.controller();c._next_slew_bias=None
        with self.assertRaisesRegex(HardwareMpcError,'missing previous model bias'):
            self.compute(c)
        with self.assertRaisesRegex(ValueError,'empty torque envelope'):
            self.compute(self.controller(bias=6.))
        c=self.controller();c._next_slew_bias[0]=np.nan
        with self.assertRaises(ValueError):self.compute(c)
        with self.assertRaises(ValueError):load_torque_config(dict(active_slew_reference='disable'))

    def test_independent_slew_audit_checks_bias_and_keeps_total_transition_bound(self):
        previous=dict(tau_total_estimated_at_feedback_nm=[0.]*5,torque_model_bias_nm=[0.]*5)
        row=dict(tau_total_estimated_at_feedback_nm=[1.1]*5,torque_model_bias_nm=[1.]*5,
            torque_previous_model_bias_nm=[0.]*5,torque_slew_bias_shift_nm=[1.]*5,
            torque_slew_reference='model_bias_relative',feedback_dt_s=.006,mpc_active=True)
        conf=dict(active_slew_reference='model_bias_relative',transition_rate_nm_s=[50.]*5)
        check_slew(row,previous,conf)
        row['torque_slew_bias_shift_nm']=[.99]*5
        with self.assertRaisesRegex(ValueError,'provenance'):check_slew(row,previous,conf)
        row['torque_slew_bias_shift_nm']=[1.]*5
        row['mpc_active']=False
        with self.assertRaisesRegex(ValueError,'rate envelope'):check_slew(row,previous,conf)
        row['mpc_active']=True;row['tau_total_estimated_at_feedback_nm']=[1.31]*5
        with self.assertRaisesRegex(ValueError,'rate envelope'):check_slew(row,previous,conf)

    def test_mapper_failure_preserves_support_and_slew_diagnostics(self):
        c=self.controller('total',bias=3.)
        with self.assertRaises(NoModelTorque):self.compute(c)
        np.testing.assert_allclose(c.last_diagnostics['torque_model_bias_nm'],3.)
        np.testing.assert_allclose(c.last_diagnostics['torque_slew_bounds_nm'],
                                   [np.full(5,-.3),np.full(5,.3)])
        self.assertFalse(c.last_diagnostics['torque_output_authorized'])

    def test_recorded_failed_state_remains_rejected_not_hidden_by_gate_change(self):
        # 20261006_151622 raw SHA ea74d13c3e4c66621ea98501785146a87639379d071139580e45fb76060ce4f9
        q=np.array([-.021832192737856655,.010212491048300576,.020870893366297476,
                    -.0884221341357576,-.020977771145537664])
        dq=np.array([.6962957496349707,.25289632120123384,.20746681365612657,
                     .4172902025565103,.08769479159594157])
        self.assertEqual(independent_feasibility(make_policy(),q,dq).status,2)
        stop=q[0]+dq[0]**2/(2*6.4)
        self.assertLess(np.rad2deg(stop),1.)  # conservative gate is NOT a physical angle limit


if __name__=='__main__':unittest.main()
