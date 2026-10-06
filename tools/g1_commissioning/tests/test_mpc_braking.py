"""Early braking is a soft QP cost, not a new feasibility/abort gate."""
import unittest
import numpy as np
from hardware_mpc_braking import braking_cost
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc, load_torque_config
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


if __name__=='__main__':unittest.main()
