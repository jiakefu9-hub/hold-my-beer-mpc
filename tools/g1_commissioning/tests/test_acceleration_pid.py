"""SDK-free C mathematics, field selection and packet execution regression."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from g1_walk_mpc import MpcRuntime, validate_acceleration_pid_selection
from hardware_acceleration_pid import (AccelerationPidPolicy, CONFIG_PATH, VARIANT,
                                       RightArmAccelerationPid, load_pid_config)
from hardware_control_comparison import BASELINE_CONFIG, MODEL_CONFIG, TORQUE_CONFIG, TRIALS
from g1_run_control_comparison import build_command
from test_measured_torque_mpc import horizon
from test_mpc_compute_process import ComputeProcessTests


class AccelerationPidTests(unittest.TestCase):
    def policy(self):
        p = AccelerationPidPolicy(np.zeros(5), control_dt=.006, horizon=9,
            joint_limits=np.tile([-1., 1.], (5,1)), joint_limit_margin=0.,
            max_dq=5., max_ddq=15., solver_time_limit=.1, solver_max_iter=1000,
            solver_backend='daqp')
        p.parameters=load_pid_config(CONFIG_PATH)
        p._local_kp=np.array([20.,20.,2.,20.,20.]);p._local_kd=np.array([1.,1.,.2,1.,1.])
        p.set_local_actuation_constraints(np.eye(5), np.zeros(5), np.ones(5)*100,
                                         np.ones(5)*100, np.zeros(5))
        return p

    def helpers(self, error=(0.,0.), alpha=(0.,0.,0.), acc=(0.,0.,0.)):
        terms=dict(gravity_error=np.array(error)*9.81,
            G_g=np.c_[np.eye(2,5)*9.81, np.zeros((2,5))],
            B_alpha=np.eye(3,5), B_acc=np.c_[np.zeros((3,2)),np.eye(3)],
            C_omega=np.eye(3,5),D_omega=np.zeros(3),C_alpha=np.zeros((3,5)),
            D_alpha=np.array(alpha),C_acc=np.zeros((3,5)),D_acc=np.array(acc))
        return SimpleNamespace(disturbance_prediction=[None]*10,
            interval_disturbance_prediction=[None]*9,
            compute_mpc_terms=lambda *a: terms)

    def action(self, p, helpers, q=None, dq=None):
        return p.compute_action(dict(current_q=np.zeros(5) if q is None else q,
                                    current_dq=np.zeros(5) if dq is None else dq),helpers)[2]

    def test_feedforward_and_feedback_have_correct_sign(self):
        p=self.policy()
        a=self.action(p,self.helpers(alpha=[1.,0.,0.],acc=[0.,1.,0.]))
        self.assertLess(a[0],0.);self.assertLess(a[3],0.)
        p.reset();p.set_local_actuation_constraints(np.eye(5),np.zeros(5),np.ones(5)*100,np.ones(5)*100,np.zeros(5))
        a=self.action(p,self.helpers(error=[.001,0]))
        self.assertLess(a[0],0.)

    def test_unconstrained_result_matches_simulation_equations(self):
        p=self.policy();h=self.helpers(error=[.001,-.002],alpha=[.3,-.2,.4],acc=[1.,2.,3.])
        q=np.array([.02,-.02,.03,.02,-.02]);dq=np.ones(5)*.01
        t=h.compute_mpc_terms();c=p.parameters;e=t['gravity_error']/9.81
        T=(t['G_g'][:,:5]/9.81)@np.linalg.pinv(t['B_alpha'],rcond=1e-5)
        target=np.linalg.pinv(T,rcond=1e-5)@(-c['kp']*e-c['ki']*e*.006)-c['kd']*(t['C_omega']@dq)
        A=np.vstack((t['B_alpha'],np.sqrt(c['acc_weight'])*t['B_acc'],np.sqrt(c['regularization'])*np.eye(5)))
        b=np.r_[target-t['D_alpha'],-np.sqrt(c['acc_weight'])*t['D_acc'],
                 np.sqrt(c['regularization'])*(-c['posture_kp']*q-c['posture_kd']*dq)]
        expected=np.linalg.solve(A.T@A+1e-9*np.eye(5),A.T@b)
        np.testing.assert_allclose(self.action(p,h,q,dq),expected,atol=1e-12)

    def test_torque_rows_match_actual_packet_pd_including_fixed_yaw(self):
        p=self.policy();q=np.arange(5)*.03;dq=np.arange(5)*.1;a=np.array([1.,-2.,3.,-4.,5.])
        m=np.eye(5)*.2+np.ones((5,5))*.01;b=np.arange(5)*.2
        p.set_local_actuation_constraints(m,b,np.ones(5)*10,np.ones(5)*10,dq)
        matrix,lo,hi=p.allocation_constraints(q,dq)
        qr=q+dq*.006+.5*a*.006**2;vr=dq+a*.006
        qr[2]=0.;vr[2]=0.
        pd=p._local_kp*(qr-q)+p._local_kd*(vr-dq)
        np.testing.assert_allclose(matrix[10:]@a-hi[10:],m@a+b-pd-10,atol=1e-12)

    def test_constraints_enforced_and_integrator_freezes_when_limited(self):
        p=self.policy();p.set_local_actuation_constraints(np.eye(5),np.zeros(5),np.ones(5)*.1,np.ones(5)*.1,np.zeros(5))
        a=self.action(p,self.helpers(error=[.3,.2],acc=[3.,2.,1.]))
        self.assertLessEqual(np.max(abs(a)),.100001)
        np.testing.assert_array_equal(p.integral,np.zeros(2))
        self.assertTrue(p.last_diagnostics['integral_frozen'])

    def test_braking_and_next_state_bounds(self):
        p=self.policy();p.set_first_acceleration_bounds(np.full(5,-15.),np.array([-6.,15.,15.,15.,15.]))
        a=self.action(p,self.helpers());self.assertLessEqual(a[0],-6.+1e-7)
        p=self.policy();q=np.zeros(5);dq=np.zeros(5);q[0]=.9999;dq[0]=.1
        with self.assertRaisesRegex(RuntimeError,'infeasible bounds'):
            self.action(p,self.helpers(),q,dq)

    def test_infeasible_torque_rejected_without_fallback(self):
        p=self.policy();p.set_local_actuation_constraints(np.eye(5),np.ones(5)*100,np.ones(5),np.ones(5),np.zeros(5))
        with self.assertRaisesRegex(RuntimeError,'rejected allocation'):
            self.action(p,self.helpers())

    def test_existing_shoulder_margin_allows_braking_without_expanding_outer_limit(self):
        p=self.policy();p.joint_limit_margin=np.array([.1,0,0,0,0])
        p.safety_joint_limits=np.tile([-1.,1.],(5,1));p.joint_limits[0]=[-.9,.9]
        q=np.zeros(5);dq=np.zeros(5);q[0]=.8999;dq[0]=.1
        p.set_first_acceleration_bounds(np.full(5,-15.),np.array([-6.,15.,15.,15.,15.]))
        lo,hi=p.first_step_acceleration_bounds(q,dq)
        self.assertLess(lo[0],hi[0]);self.assertEqual(hi[0],-6.)
        self.assertEqual(p.reentry_diagnostics['upper_rad'][0],1.)
        q[0]=.9999
        lo,hi=p.first_step_acceleration_bounds(q,dq)
        self.assertGreater(lo[0],hi[0])  # no extra room beyond the actual outer box

    def test_future_nodes_do_not_change_single_step_controller(self):
        p=self.policy();h=self.helpers(error=[.001,0]);a=self.action(p,h)
        p=self.policy();h.disturbance_prediction[1:]=[object()]*9
        h.interval_disturbance_prediction[1:]=[object()]*8
        np.testing.assert_array_equal(self.action(p,h),a)

    def test_current_dynamics_packet_closure(self):
        from endpoint_pose import EndpointModel
        c=RightArmAccelerationPid(np.zeros(5),BASELINE_CONFIG,model=EndpointModel(MODEL_CONFIG),
                                  torque_config=TORQUE_CONFIG,pid_config=CONFIG_PATH)
        self.addCleanup(c.close);c.policy.solver_time_limit=.1
        q=np.zeros(13);q[5:10]=[.01,-.02,.03,-.04,.01]
        c.set_measured_dq([.01,-.02,.03,.02,-.01]);c.set_disturbance_horizon(horizon())
        qr,vr,d=c.step(q,[1,0,0,0],0.,.006)
        np.testing.assert_allclose(np.array(d['tau_ff_candidate_nm'])+d['tau_pd_at_feedback_nm'],
                                   d['tau_total_estimated_at_feedback_nm'],atol=1e-12)
        np.testing.assert_allclose(d['mapper']['checked_ddq_rad_s2'],d['raw_mpc_ddq_rad_s2'],atol=1e-10)
        self.assertEqual(qr[2],0.);self.assertEqual(vr[2],0.)
        self.assertEqual(c.metadata['variant'],VARIANT)
        self.assertNotIn('posture_feedback_model',d)

    def test_explicit_dispatch_and_selection_restrictions(self):
        options=dict(config=BASELINE_CONFIG,model_config=MODEL_CONFIG,torque_config=TORQUE_CONFIG,
            zero_arm_neutral=True,align_accel_forecast=True,acceleration_pid_config=CONFIG_PATH)
        c=MpcRuntime(**options);self.addCleanup(c.close)
        self.assertIsInstance(c.controller,RightArmAccelerationPid)
        options.pop('acceleration_pid_config')
        m=MpcRuntime(**options);self.addCleanup(m.close)
        self.assertNotIsInstance(m.controller,RightArmAccelerationPid)
        options['acceleration_pid_config']=CONFIG_PATH;options['zero_arm_neutral']=False
        with self.assertRaises(ValueError):MpcRuntime(**options)

    def test_numbered_command_has_no_implicit_execute_or_unrelated_changes(self):
        for k in TRIALS:
            cmd=build_command(k)
            self.assertIn('--preflight',cmd);self.assertNotIn('--execute',cmd)
            self.assertEqual('--acceleration-pid-config' in cmd,k in ('T03','T05','T08','T09','T11','T12'))
            self.assertEqual('--align-accel-forecast' in cmd,k not in ('T01','T07','T10','T11'))
        self.assertEqual(TRIALS['T03']['pid_config'],TRIALS['T05']['pid_config'])
        self.assertEqual(TRIALS['T02']['config'],TRIALS['T04']['config'])


class AccelerationPidProcessTests(ComputeProcessTests):
    def options(self):
        return dict(config=BASELINE_CONFIG,model_config=MODEL_CONFIG,torque_config=TORQUE_CONFIG,
            predictor_mode='learned_filtered',zero_arm_neutral=True,align_accel_forecast=True,
            acceleration_pid_config=CONFIG_PATH,stationary=False,field_trial=True,
            assumed_command_delay_s=None)


if __name__=='__main__':
    unittest.main()
