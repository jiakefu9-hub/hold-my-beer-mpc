"""Offline dual-arm geometry, reference governor, packets and execution gates."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import g1_dual_upright as entry
from endpoint_pose import rotation
from hardware_dual_upright import (DualUprightController,DualUprightRuntime,effective_profile)
from hardware_pid_control import ARM_MOTOR_INDICES,WEIGHT_MOTOR_INDEX
from g1_walk_pid import load_profile,make_arm_message


def quat(rpy):
    return Rotation.from_euler('xyz',rpy,degrees=True).as_quat()[[3,0,1,2]]


class DualUprightTests(unittest.TestCase):
    def test_both_analytic_jacobians_match_finite_difference(self):
        c=DualUprightController();rng=np.random.default_rng(215)
        for _ in range(25):
            q=np.zeros(13);q[:11]=rng.uniform(-.12,.12,11)
            r=rotation(quat(rng.uniform(-20,20,3)))
            original=c.kinematics.evaluate(q,r)
            for i,side in enumerate(('left','right')):
                for j in range(5):
                    a=q.copy();b=q.copy();a[i*5+j]+=1e-6;b[i*5+j]-=1e-6
                    plus=c.kinematics.evaluate(a,r)[side][0]
                    minus=c.kinematics.evaluate(b,r)[side][0]
                    np.testing.assert_allclose(original[side][1][:,j],((plus-minus)/2e-6)[:2],atol=2e-8)

    def test_yaw_invariance_and_quaternion_sign(self):
        results=[]
        for yaw,sign in ((0,1),(80,1),(-120,-1)):
            c=DualUprightController();c.seed(np.zeros(10))
            for _ in range(200):
                q,dq,_=c.step(np.zeros(13),sign*quat([5,12,yaw]),.006)
            results.append(q)
        for q in results[1:]:np.testing.assert_allclose(q,results[0],atol=1e-12)

    def test_upward_goal_does_not_accept_upside_down_axis(self):
        c=DualUprightController();c.seed(np.zeros(10))
        with self.assertRaisesRegex(ValueError,'upward'):
            c.step(np.zeros(13),quat([180,0,0]),.006)

    def test_both_arms_respond_and_fixed_shoulders_do_not_drift(self):
        c=DualUprightController();c.seed(np.zeros(10))
        for _ in range(500):
            q,dq,d=c.step(np.zeros(13),quat([-5,12,20]),.006)
        self.assertGreater(np.linalg.norm(q[:5]),.05)
        self.assertGreater(np.linalg.norm(q[5:]),.05)
        np.testing.assert_array_equal(q[[1,2,6,7]],np.zeros(4))
        self.assertIn('axis_world',d['left']);self.assertIn('axis_world',d['right'])

    def test_reference_bounds_acceleration_and_delayed_loop(self):
        c=DualUprightController();c.seed(np.zeros(10));previous_q=np.zeros(10);previous_dq=np.zeros(10)
        for k in range(2200):
            dt=.1 if k%13==0 else .006
            q,dq,_=c.step(np.zeros(13),quat([5,35,0]),dt)
            np.testing.assert_allclose(q-previous_q,dq*.006,atol=1e-12)
            self.assertLessEqual(np.max(abs(dq)),.12+1e-10)
            self.assertLessEqual(np.max(abs(dq-previous_dq)),.4*.006+1e-9)
            self.assertTrue(np.all(q<=c.maximum) and np.all(q>=c.minimum))
            previous_q,previous_dq=q,dq

    def test_bad_feedback_and_unseeded_controller_refused(self):
        c=DualUprightController()
        with self.assertRaisesRegex(RuntimeError,'seed'):c.step(np.zeros(13),quat([0,0,0]),.006)
        c.seed(np.zeros(10))
        with self.assertRaises(ValueError):c.step(np.zeros(13),[0,0,0,0],.006)
        q=np.zeros(13);q[0]=np.deg2rad(21)
        with self.assertRaisesRegex(ValueError,'envelope'):c.step(q,quat([0,0,0]),.006)
        with self.assertRaises(ValueError):c.step(np.zeros(13),quat([0,0,0]),float('nan'))

    def test_profile_is_copy_and_both_arm_targets_zero(self):
        p=load_profile(entry.PROFILE);old=p['target_q_array'].copy()
        eff=effective_profile(p)
        np.testing.assert_array_equal(p['target_q_array'],old)
        np.testing.assert_array_equal(eff['target_q_array'],np.zeros(13))
        np.testing.assert_array_equal(eff['kp_array'],p['kp_array'])
        self.assertTrue(eff['dual_upright_experiment']['both_arms_use_imu'])

    def test_entry_has_no_reference_jump_and_control_continues_past_18s(self):
        runtime=DualUprightRuntime();p=effective_profile(load_profile(entry.PROFILE))
        initial=np.zeros(13);initial[3]=.6;initial[8]=.6
        plan=runtime.create_plan(initial,p)
        last=None
        for t in np.arange(0,3,.006):last=plan.sample(t,np.zeros(13),np.zeros(13),quat([0,10,0]),0.,.006)
        frame=plan.sample(3.,np.zeros(13),np.zeros(13),quat([0,10,0]),0.,.006)
        np.testing.assert_allclose(frame['q_rad'][:10]-last['q_rad'][:10],frame['dq_rad_s'][:10]*.006,atol=1e-12)
        first=plan.sample(18.,np.zeros(13),np.zeros(13),quat([0,10,0]),0.,.006)
        late=plan.sample(60.,np.zeros(13),np.zeros(13),quat([0,10,0]),0.,.006)
        self.assertEqual(first['weight'],1.)
        self.assertEqual(late['weight'],1.)
        self.assertFalse(first['terminal']);self.assertFalse(late['terminal'])
        self.assertEqual(first['stage'],'stationary_upright')
        self.assertEqual(late['stage'],'stationary_upright')
        self.assertTrue(runtime.continuous_stationary_hold)

    def test_packet_mapping_both_arms_and_no_feedforward_torque(self):
        rt=DualUprightRuntime();p=effective_profile(load_profile(entry.PROFILE))
        plan=rt.create_plan(np.zeros(13),p)
        frame=plan.sample(3.,np.zeros(13),np.zeros(13),quat([3,10,0]),0.,.006)
        def constructor():
            return SimpleNamespace(motor_cmd=[SimpleNamespace(mode=0,q=0.,dq=0.,tau=0.,kp=0.,kd=0.) for _ in range(35)])
        state=SimpleNamespace(mode_pr=0,mode_machine=4)
        msg=make_arm_message(frame,state,constructor,SimpleNamespace(Crc=lambda m:123))
        for slot,index in enumerate(ARM_MOTOR_INDICES[:11]):
            self.assertEqual(msg.motor_cmd[index].q,frame['q_rad'][slot])
            self.assertEqual(msg.motor_cmd[index].dq,frame['dq_rad_s'][slot])
            self.assertEqual(msg.motor_cmd[index].tau,0.)
        self.assertEqual(msg.motor_cmd[WEIGHT_MOTOR_INDEX].q,1.)
        self.assertEqual(msg.crc,123)

    def test_both_arm_estimated_pd_torque_guard(self):
        rt=DualUprightRuntime();p=effective_profile(load_profile(entry.PROFILE))
        plan=rt.create_plan(np.zeros(13),p)
        frame=plan.sample(3.,np.zeros(13),np.zeros(13),quat([0,10,0]),0.,.006)
        for index in (19,26):
            state=SimpleNamespace(q=np.zeros(35),dq=np.zeros(35));state.q[index]=.1
            with self.assertRaisesRegex(RuntimeError,'PD torque'):
                rt.check_before_write(frame,state,None,0,1_000_000)
        state=SimpleNamespace(q=np.zeros(35),dq=np.zeros(35))
        rt.check_before_write(frame,state,None,0,1_000_000)

    def test_entry_requires_explicit_permission_and_default_is_offline(self):
        with patch.object(entry,'run_device',side_effect=AssertionError('robot output')),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(entry.main(['enx6c1ff701509c','--execute']),1)
        with patch('validate_dual_upright.validate',return_value=dict(passed=True)) as check, \
             patch.object(entry,'run_device',side_effect=AssertionError('robot output')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main([]),0);check.assert_called_once()
        self.assertTrue(DualUprightRuntime.stationary)
        no_capture=entry.build_parser().parse_args(['enx6c1ff701509c','--execute',
            '--permit-real-output',entry.PERMIT,'--no-capture'])
        self.assertTrue(no_capture.no_capture);self.assertIsNone(no_capture.output_dir)
        journal=entry.DiscardJournal();journal.record({'unused':True})
        self.assertTrue(journal.close());self.assertFalse(journal.failed.is_set())
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            entry.build_parser().parse_args(['--task','walk'])


if __name__=='__main__':unittest.main()
