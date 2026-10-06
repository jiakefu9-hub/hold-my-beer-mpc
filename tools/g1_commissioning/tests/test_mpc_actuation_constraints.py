"""No-DDS checks for executor-aware acceleration-MPC constraints."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hardware_mpc_solver import CondensedArmMPCPolicy
from hardware_mpc_torque_control import load_torque_config


class ActuationConstraintTests(unittest.TestCase):
    def policy(self):
        p=CondensedArmMPCPolicy(np.zeros(5),horizon=9,control_dt=.006,
            max_dq=1.,max_ddq=8.,solver_backend='daqp',solver_time_limit=.1)
        p._local_kp=np.full(5,20.);p._local_kd=np.ones(5)
        return p

    def test_absolute_torque_caps_remain_but_large_safe_changes_are_allowed(self):
        p=self.policy();mass=np.diag([.4,.5,.3,.6,.2]);bias=np.array([1.,-.2,.1,-1.,.05])
        p.set_local_actuation_constraints(mass,bias,[5,3,2,5,1.5],[5,3,2,5,1.5],
            np.zeros(5))
        matrix,lo,hi=p._local_actuation_rows()
        self.assertEqual(matrix.shape,(50,45))
        zero=np.zeros(45);value=matrix@zero
        self.assertTrue(np.all(value>=lo-1e-12));self.assertTrue(np.all(value<=hi+1e-12))
        # A 3.2 Nm change exceeds the retired 0.3 Nm-per-tick cap, yet
        # is valid here: all absolute torque/FF limits are respected.
        jump=np.zeros((9,5));jump[0,0]=8.
        value=matrix@(jump/8.).ravel()
        self.assertTrue(np.all(value<=hi+1e-12));self.assertTrue(np.all(value>=lo-1e-12))
        jump[0,0]=11.
        self.assertGreater(np.max(matrix@(jump/8.).ravel()-hi),0.)

    def test_invalid_or_disabled_configuration_is_explicit(self):
        self.assertFalse(load_torque_config()['planning_actuation_constraints_enabled'])
        with self.assertRaisesRegex(ValueError,'boolean'):
            load_torque_config(dict(planning_actuation_constraints_enabled='yes'))
        p=self.policy()
        with self.assertRaises(ValueError):
            p.set_local_actuation_constraints(np.eye(4),np.zeros(5),np.ones(5),np.ones(5),
                np.zeros(5))


if __name__=='__main__':unittest.main()
