"""Early braking cost plus persistent recovery action, without a new abort gate."""
import unittest
import numpy as np
from hardware_mpc_braking import braking_cost, LatchedPredictiveBrake
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc, load_torque_config
from endpoint_pose import ROOT
from g1_walk_pid import EXPECTED_TARGET_Q
from test_measured_torque_mpc import horizon


class BrakingTests(unittest.TestCase):
    def weights(self, q=0., v=0., **options):
        return braking_cost(np.full(5,q),np.full(5,v),np.full(5,-.1),np.full(5,.1),
            **dict(dict(deceleration=4.,reaction_s=.012,margin_rad=.02,weight=50.),**options))

    def test_inactive_away_from_boundary_or_when_returning(self):
        for q,v in ((0.,.1),(.095,-.1),(-.095,.1),(.099,0.)):
            np.testing.assert_array_equal(self.weights(q,v)[0],np.zeros(5))

    def test_symmetric_smooth_and_monotone_activation(self):
        weights=[self.weights(q,.2)[0][0] for q in (.06,.075,.085,.095)]
        self.assertTrue(np.all(np.diff(weights)>=0))
        self.assertEqual(weights[0],0.);self.assertEqual(weights[-1],50.)
        for q in (.075,.085,.095):
            np.testing.assert_allclose(self.weights(q,.2)[0],self.weights(-q,-.2)[0])

    def test_invalid_tuning_rejected_and_legacy_default_disabled(self):
        self.assertFalse(load_torque_config()['predictive_braking_enabled'])
        for key,value in (('predictive_braking_enabled','yes'),('braking_deceleration_rad_s2',9.),
                          ('braking_velocity_weight',-1.),('braking_reaction_s',float('nan'))):
            with self.assertRaises(ValueError):load_torque_config({key:value})
        with self.assertRaises(ValueError):self.weights(float('nan'),.2)

    def test_cost_changes_without_adding_or_widening_constraints(self):
        c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10])
        try:
            p=c.policy;p.solver_time_limit=.1
            slots=EXPECTED_TARGET_Q.copy()
            q=slots[5:10];v=np.zeros(5)
            q[0]=np.deg2rad(3.5);v[0]=.3
            c.model.data.qpos[:]=c.model.model.qpos0
            c.model.data.qpos[c.model.joint_addresses]=slots[:11]
            h=horizon()
            helpers=c.helper.build_helpers(c.model.data,disturbance_prediction=h.nodes,
                interval_disturbance_prediction=h.intervals,include_kinematics_cache=False)
            lo,hi,*_=p._build_online_constraint_bounds(q,v)
            p.set_braking_velocity_cost(np.zeros(5))
            p.compute_action(dict(current_q=q,current_dq=v,dt=.006),helpers)
            old_blocks,old_tail=[x.copy() for x in p._cost_blocks]
            # Reset the previous trajectory so both builds linearize at the
            # same points. A warm-start change is not a braking-cost change.
            p.reset()
            weights=np.array([50.,0,0,0,0]);p.set_braking_velocity_cost(weights)
            p.compute_action(dict(current_q=q,current_dq=v,dt=.006),helpers)
            new_blocks,new_tail=p._cost_blocks
            expected=np.zeros_like(old_blocks);expected[:,5,5]=100.
            np.testing.assert_allclose(new_blocks-old_blocks,expected,atol=1e-12)
            expected_tail=np.zeros_like(old_tail);expected_tail[5,5]=100.*p.terminal_scale
            np.testing.assert_allclose(new_tail-old_tail,expected_tail,atol=1e-12)
            newlo,newhi,*_=p._build_online_constraint_bounds(q,v)
            np.testing.assert_array_equal(lo,newlo);np.testing.assert_array_equal(hi,newhi)
            p.reset();np.testing.assert_array_equal(p._braking_velocity_cost,np.zeros(5))
        finally:c.close()

    def test_latched_brake_does_not_release_on_one_slower_outward_sample(self):
        brake=LatchedPredictiveBrake()
        options=dict(deceleration=4.,reaction_s=.012,margin_rad=np.deg2rad(1.),
            weight=50.,max_ddq=np.full(5,8.),minimum_deceleration=6.)
        lower=np.deg2rad(np.array([-5.,-5.,-20.,-40.,-40.]))
        upper=np.deg2rad(np.array([5.,3.,5.,40.,40.]))
        q=np.zeros(5);v=np.zeros(5);q[2]=np.deg2rad(3.2);v[2]=.4
        _,bounds,detail=brake.update(q,v,lower,upper,**options)
        self.assertEqual(detail['latch_side'][2],1)
        self.assertEqual(bounds[1][2],-6.)
        # The old stateless cost could disappear here and allow outward +ddq.
        v[2]=.1
        _,bounds,detail=brake.update(q,v,lower,upper,**options)
        self.assertEqual(detail['latch_side'][2],1)
        self.assertEqual(bounds[1][2],-6.)
        v[2]=-.01
        _,_,detail=brake.update(q,v,lower,upper,**options)
        self.assertEqual(detail['latch_side'][2],0)

    def test_wider_field_yaw_box_solves_old_fault_and_keeps_recovery_room(self):
        c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10],
            torque_config=ROOT/'configs/hardware_mpc_torque_field.yaml')
        try:
            np.testing.assert_allclose(np.rad2deg(c.policy.joint_limits[:2]),
                                       [[-15.,15.],[-15.,15.]])
            np.testing.assert_allclose(np.rad2deg(c.policy.safety_joint_limits[:2]),
                                       [[-20.,20.],[-20.,20.]])
            np.testing.assert_allclose(np.rad2deg(c.policy.joint_limits[2]),[-20.,15.])
            np.testing.assert_allclose(np.rad2deg(c.policy.safety_joint_limits[2]),[-25.,20.])
            c.policy.solver_time_limit=.1
            slots=EXPECTED_TARGET_Q.copy()
            slots[5:10]=[.0697122365,.0044101947,.0673513487,-.0629051998,-.0328727290]
            dq=np.array([-.1349903196,.0859029293,.7117671371,.0322135985,.0030679617])
            c.set_measured_dq(dq);c.set_disturbance_horizon(horizon())
            _,_,diag=c.step(slots,[1,0,0,0],0.,.006)
            self.assertTrue(diag['mpc']['solved'])
            self.assertFalse(diag['mpc']['recovery_active'])
            self.assertEqual(diag['predictive_braking']['latch_side'][2],0)

            # Near the new working boundary, braking persists while the policy
            # temporarily uses (but does not target) the +20 deg recovery box.
            c.reset();slots[7]=np.deg2rad(16.);dq[2]=.7
            c.set_measured_dq(dq);c.set_disturbance_horizon(horizon())
            _,_,diag=c.step(slots,[1,0,0,0],0.,.006)
            self.assertTrue(diag['mpc']['solved'])
            self.assertTrue(diag['mpc']['recovery_active'])
            self.assertEqual(diag['predictive_braking']['latch_side'][2],1)
            self.assertLessEqual(diag['raw_mpc_ddq_rad_s2'][2],-6.)
        finally:c.close()

    def test_wider_shoulder_box_solves_recorded_roll_fault_state(self):
        c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10],
            config=ROOT/'configs/hardware_mpc_upright_baseline.yaml',
            torque_config=ROOT/'configs/hardware_mpc_torque_field.yaml')
        try:
            c.policy.solver_time_limit=.1
            slots=EXPECTED_TARGET_Q.copy()
            # 20261007_153046 controller_fault_detail.  With the old
            # shoulder-roll outer upper bound of +3 deg this state could not
            # stop before the horizon crossed the angle row.
            slots[5:10]=[.0205769148,.0176287945,.1525352150,-.0564816520,-.0331363827]
            dq=np.array([.1089126393,1.4588158131,.7040972114,-.0153398085,.0690291375])
            c.set_measured_dq(dq);c.set_disturbance_horizon(horizon())
            _,_,diag=c.step(slots,[1,0,0,0],0.,.005975205)
            self.assertTrue(diag['mpc']['solved'])
            self.assertGreater(diag['mpc']['min_constraint_margins']['q'],0.)
            self.assertGreater(diag['mpc']['min_constraint_margins']['dq'],0.)
            self.assertEqual(diag['predictive_braking']['latch_side'][1],1)
            self.assertLessEqual(diag['raw_mpc_ddq_rad_s2'][1],-6.)
            self.assertLessEqual(np.max(np.abs(diag['raw_mpc_ddq_rad_s2'])),15.+1e-9)
        finally:c.close()


if __name__=='__main__':unittest.main()
