"""No robot output: feedback-coordinate algebra, torque and runtime regression."""
import contextlib
import io
import os
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from disturbance_types import DisturbanceInput, DisturbanceHorizon
from g1_walk_mpc import (MpcRuntime, build_parser, FIELD_TORQUE_CONFIG,
                         LEARNED_ACC_ALPHA_MPC_CONFIG,
                         LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG, LEARNED_ACC_MPC_CONFIG,
                         LEARNED_MPC_CONFIG, LEARNED_TORQUE_CONFIG)
from g1_walk_pid import EXPECTED_TARGET_Q
from hardware_mpc_control import HardwareMpcError
from hardware_mpc_learned import RightArmLearnedTorqueMpc, YawFeedbackMpcPolicy
from hardware_mpc_solver import CondensedArmMPCPolicy
from hardware_mpc_torque_control import load_torque_config, HardwareTorquePreviewPlan
from test_measured_torque_mpc import horizon


class LearnedMpcTests(unittest.TestCase):
    def controller(self):
        c = RightArmLearnedTorqueMpc(EXPECTED_TARGET_Q[5:10], LEARNED_MPC_CONFIG,
                                    torque_config=LEARNED_TORQUE_CONFIG)
        self.addCleanup(c.close)
        # Mathematical tests, not host real-time assertions.
        c.policy.solver_time_limit = .1
        return c

    def test_physical_config_matches_successful_baseline(self):
        old, new = [load_torque_config(p) for p in (FIELD_TORQUE_CONFIG, LEARNED_TORQUE_CONFIG)]
        for key in ('tau_abs_nm', 'tau_ff_abs_nm', 'kp', 'kd', 'q_min_deg', 'q_max_deg',
                    'q_margin_deg', 'max_dq_rad_s', 'max_ddq_rad_s2', 'max_abs_qacc_rad_s2',
                    'braking_deceleration_rad_s2', 'braking_min_deceleration_rad_s2'):
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)
        self.assertEqual(new['pitch_feedback_kp'], 2.)
        self.assertEqual(new['pitch_feedback_kd'], .2)

    def test_nominal_and_net_coordinates_have_identical_forward_dynamics(self):
        c = self.controller(); p = c.policy
        q = EXPECTED_TARGET_Q[5:10].copy(); q[0] -= .08; q[2] = .13
        dq = np.array([.1, -.04, .09, .2, 0.])
        p.feedback_q = q
        m, b = c.inverse.linear_dynamics(q, dq, horizon().nodes[0])
        p.set_local_actuation_constraints(m, b, c.mapper.limit, c.mapper.limit, dq)
        nominal = np.array([2., -1., 3., 1., -2.]); x = np.r_[q, dq]
        net = nominal + p.feedback_F@x + p.feedback_f
        torque = m@nominal+b+p.feedback_pd
        np.testing.assert_allclose(np.linalg.solve(m, torque-b), net, atol=1e-12)
        np.testing.assert_allclose(p.A@x+p.B@net,
            (p.A+p.B@p.feedback_F)@x+p.B@nominal+p.B@p.feedback_f, atol=1e-12)

    def test_effort_cost_exactly_substitutes_modeled_feedback_over_entire_horizon(self):
        c = self.controller(); p = c.policy
        p.feedback_q = EXPECTED_TARGET_Q[5:10].copy()
        p.set_local_actuation_constraints(np.eye(5)*.2, np.zeros(5), np.ones(5)*10,
                                         np.ones(5)*10, np.zeros(5))
        # Exercise affine constant as well as this robot's zero yaw target.
        p.feedback_f = np.arange(5)*.03
        terms = [dict(C_omega=np.zeros((3,5)), D_omega=np.zeros(3),
            G_g=np.zeros((2,10)), d_g=np.zeros(2), C_acc=np.zeros((3,5)),
            B_acc=np.zeros((3,5)), D_acc=np.zeros(3), C_alpha=np.zeros((3,5)),
            B_alpha=np.zeros((3,5)), D_alpha=np.zeros(3)) for _ in range(10)]
        old_h, old_g = CondensedArmMPCPolicy._build_cost(p, terms)
        old_h = [h.copy() for h in old_h]
        new_h, new_g = p._build_cost(terms)
        rng = np.random.default_rng(72)
        for k in range(9):
            x, a = rng.normal(size=10), rng.normal(size=5)
            z = np.r_[x,a]; sl = slice(k*15,(k+1)*15)
            difference = .5*z@(new_h[k]-old_h[k])@z+(new_g[sl]-old_g[sl])@z
            nominal = a-p.feedback_F@x-p.feedback_f
            expected = nominal@p.R@nominal-a@p.R@a-p.feedback_f@p.R@p.feedback_f
            self.assertAlmostEqual(difference, expected, places=10)
        np.testing.assert_array_equal(new_h[-1], old_h[-1])

    def test_direct_rnea_packet_matches_net_acceleration_without_candidate_search(self):
        c = self.controller(); slots = EXPECTED_TARGET_Q.copy(); slots[7] = .1
        dq = np.array([.03,0,.02,0,0.])
        c.set_measured_dq(dq); c.set_disturbance_horizon(horizon())
        with mock.patch.object(c.mapper, 'compute', side_effect=AssertionError('redundant candidates')):
            qr, dqr, d = c.step(slots, [1,0,0,0], 0., .006)
        total, ff, pd = [np.asarray(d[k]) for k in
            ('tau_total_estimated_at_feedback_nm', 'tau_ff_candidate_nm', 'tau_pd_at_feedback_nm')]
        np.testing.assert_allclose(ff+pd, total, atol=1e-12)
        np.testing.assert_allclose(d['mapper']['checked_ddq_rad_s2'], d['raw_mpc_ddq_rad_s2'], atol=1e-10)
        np.testing.assert_allclose(np.asarray(d['tau_nominal_ff_nm'])+np.asarray(
            d['posture_feedback_model']['current_feedback_torque_nm']), total, atol=1e-12)
        self.assertEqual(d['mapper']['candidate_count'], 0)
        self.assertEqual(qr[2], 0.); self.assertEqual(dqr[2], 0.)
        self.assertFalse(d['mapper']['hardware_certified'])
        m,b = c._prepared_forward
        rnea = c.inverse.compute(slots[5:10], dq, np.asarray(d['raw_mpc_ddq_rad_s2']), horizon().nodes[0])
        np.testing.assert_allclose(rnea['tau_model_nm'], total, atol=1e-10)

    def test_pitch_centering_is_light_restoring_modeled_and_does_not_anchor_packet_reference(self):
        c = self.controller(); p = c.policy
        q = c.nominal.copy(); q[0] -= .1
        dq = np.zeros(5); dq[0] = -.2
        p.feedback_q = q
        m, b = c.inverse.linear_dynamics(q, dq, horizon().nodes[0])
        p.set_local_actuation_constraints(m, b, c.mapper.limit, c.mapper.limit, dq)
        self.assertAlmostEqual(p.feedback_pd[0], .24, places=12)
        self.assertGreater((p.feedback_F@np.r_[q, dq]+p.feedback_f)[0], 0.)
        np.testing.assert_allclose(p.feedback_pd[[1,3,4]], 0.)
        slots = EXPECTED_TARGET_Q.copy(); slots[5:10] = q
        c.set_measured_dq(dq); c.set_disturbance_horizon(horizon())
        qref, _, diagnostic = c.step(slots, [1,0,0,0], 0., .006)
        self.assertNotEqual(qref[0], c.nominal[0])
        self.assertEqual(qref[2], c.nominal[2])
        self.assertEqual(diagnostic['posture_feedback_model']['active_joint_indices'], [0,2])

    def test_affine_torque_matches_independent_rnea_with_moving_base(self):
        c = self.controller(); rng = np.random.default_rng(8)
        for _ in range(15):
            q = c.nominal+rng.normal(0,.06,5); dq=rng.normal(0,.2,5); a=rng.normal(0,4,5)
            base=DisturbanceInput(rng.normal(0,1,3),rng.normal(0,.3,3),rng.normal(0,3,3),
                                  Rotation.from_rotvec(rng.normal(0,.15,3)).as_matrix())
            m,b=c.inverse.linear_dynamics(q,dq,base)
            np.testing.assert_allclose(c.inverse.compute(q,dq,a,base)['tau_model_nm'],m@a+b,atol=1e-9)

    def test_fixed_yaw_ff_constraint_equals_actual_packet_law(self):
        c=self.controller();p=c.policy;q=c.nominal.copy();q[2]=.1;dq=np.ones(5)*.05
        p.feedback_q=q;m,b=c.inverse.linear_dynamics(q,dq,horizon().nodes[0])
        p.set_local_actuation_constraints(m,b,c.mapper.limit,c.mapper.limit,dq)
        matrix,lo,hi=p._local_actuation_rows()
        a=np.array([1.,2.,-3.,2.,1.]); normalized=np.tile(a/p.max_ddq,9)
        row=9*5+2;scale=1/max(abs(m[2]*p.max_ddq))
        expected_ff=(m@a+b)[2]-p.feedback_pd[2]
        self.assertAlmostEqual((matrix@normalized-hi)[row]/scale,
                               expected_ff-c.mapper.limit[2],places=12)

    def test_future_orientation_changes_upright_action_with_zero_dynamic_costs(self):
        c=self.controller(); slots=EXPECTED_TARGET_Q.copy()
        c.set_measured_dq(np.zeros(5));c.set_disturbance_horizon(horizon())
        first=c.step(slots,[1,0,0,0],0.,.006)[2]['raw_mpc_ddq_rad_s2']
        c.reset();c.set_measured_dq(np.zeros(5))
        nodes=tuple(DisturbanceInput(np.zeros(3),np.zeros(3),np.zeros(3),
                    Rotation.from_euler('y',k*.002).as_matrix()) for k in range(10))
        c.set_disturbance_horizon(DisturbanceHorizon(nodes,nodes[:-1]))
        second=c.step(slots,[1,0,0,0],0.,.006)[2]['raw_mpc_ddq_rad_s2']
        self.assertGreater(np.linalg.norm(np.asarray(second)-first),.01)

    def test_direct_executor_rejects_invalid_torque_instead_of_search_or_clipping(self):
        c=self.controller()
        with self.assertRaises(HardwareMpcError):
            c._torque(c.nominal,np.zeros(5),c.nominal,np.zeros(5),np.ones(5)*100,
                      horizon().nodes[0],prepared_dynamics=(np.eye(5),np.zeros(5)))

    def test_learned_entry_defaults_and_no_implicit_output(self):
        args=build_parser(learned=True).parse_args([])
        self.assertFalse(args.execute);self.assertEqual(args.mpc_config,LEARNED_MPC_CONFIG)
        self.assertEqual(args.torque_config,LEARNED_TORQUE_CONFIG)
        self.assertEqual(args.predictor,'learned_filtered')
        self.assertEqual(build_parser().parse_args([]).torque_config,None)
        from g1_walk_mpc_learned import main
        with mock.patch('g1_walk_mpc.preflight',return_value={'passed':True}) as preflight:
            with mock.patch('g1_walk_mpc.run_device',side_effect=AssertionError('hardware')):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(['--cpu',str(min(os.sched_getaffinity(0)))]),0)
        self.assertEqual(preflight.call_args.args[-1],LEARNED_TORQUE_CONFIG)

    def test_acceleration_cost_trial_changes_only_requested_dynamic_weight(self):
        from hardware_mpc_control import load_mpc_config
        baseline = load_mpc_config(LEARNED_MPC_CONFIG)
        trial = load_mpc_config(LEARNED_ACC_MPC_CONFIG)
        changed = {key for key in baseline.keys() | trial.keys()
                   if baseline.get(key) != trial.get(key)}
        self.assertEqual(changed, {'q_ee_acc', 'mpc_start_s', 'mpc_handoff_duration_s'})
        self.assertEqual(trial['q_ee_acc'], .01)
        self.assertEqual(trial['q_ee_alpha'], 0.)
        self.assertEqual(trial['q_ee_omega'], 0.)
        self.assertEqual(trial['mpc_start_s'], 3.3)
        self.assertEqual(trial['mpc_handoff_duration_s'], 1.5)

    def test_angular_acceleration_trial_changes_only_alpha_from_acc_trial(self):
        from hardware_mpc_control import load_mpc_config
        acceleration = load_mpc_config(LEARNED_ACC_MPC_CONFIG)
        angular = load_mpc_config(LEARNED_ACC_ALPHA_MPC_CONFIG)
        changed = {key for key in acceleration.keys() | angular.keys()
                   if acceleration.get(key) != angular.get(key)}
        self.assertEqual(changed, {'q_ee_alpha'})
        self.assertEqual(angular['q_ee_acc'], .01)
        self.assertEqual(angular['q_ee_alpha'], .0005)
        self.assertEqual(angular['q_ee_omega'], 0.)
        self.assertEqual(angular['mpc_start_s'], 3.3)
        self.assertEqual(angular['mpc_handoff_duration_s'], 1.5)

    def test_angular_velocity_trial_changes_only_omega_from_acc_alpha_trial(self):
        from hardware_mpc_control import load_mpc_config
        angular = load_mpc_config(LEARNED_ACC_ALPHA_MPC_CONFIG)
        velocity = load_mpc_config(LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG)
        changed = {key for key in angular.keys() | velocity.keys()
                   if angular.get(key) != velocity.get(key)}
        self.assertEqual(changed, {'q_ee_omega'})
        self.assertEqual(velocity['q_ee_acc'], .01)
        self.assertEqual(velocity['q_ee_alpha'], .0005)
        self.assertEqual(velocity['q_ee_omega'], 1.)
        self.assertEqual(velocity['mpc_start_s'], 3.3)
        self.assertEqual(velocity['mpc_handoff_duration_s'], 1.5)

    def test_acceleration_trial_has_one_time_bumpless_mpc_handoff(self):
        c = RightArmLearnedTorqueMpc(EXPECTED_TARGET_Q[5:10], LEARNED_ACC_MPC_CONFIG,
                                    torque_config=LEARNED_TORQUE_CONFIG)
        self.addCleanup(c.close)
        c.policy.solver_time_limit = .1
        plan = HardwareTorquePreviewPlan(EXPECTED_TARGET_Q, EXPECTED_TARGET_Q,
            np.r_[np.full(11,20.),0,0], np.r_[np.ones(11),0,0], c, mpc_start_s=3.3)
        frames = []
        for task_s in (3.294, 3.3, 4.05, 4.8, 4.9):
            c.set_measured_dq(np.zeros(5)); c.set_disturbance_horizon(horizon())
            frames.append(plan.sample(task_s, EXPECTED_TARGET_Q, np.zeros(13),
                                      [1,0,0,0], 0., .006))
        before, start, middle, end, after = frames
        np.testing.assert_allclose(start['q_rad'], before['q_rad'], atol=1e-12)
        np.testing.assert_allclose(start['dq_rad_s'], before['dq_rad_s'], atol=1e-12)
        np.testing.assert_allclose(start['diagnostics']['tau_total_estimated_at_feedback_nm'],
                                   before['diagnostics']['tau_total_estimated_at_feedback_nm'], atol=1e-12)
        np.testing.assert_allclose([f['diagnostics']['mpc_handoff']['scale']
                                    for f in (start,middle,end,after)],
                                   [0.,.5,1.,1.], atol=1e-12)
        for frame in (start,middle,end,after):
            d=frame['diagnostics']; total=np.asarray(d['tau_total_estimated_at_feedback_nm'])
            pd=(np.asarray(frame['kp'][5:10])*(np.asarray(frame['q_rad'][5:10])-EXPECTED_TARGET_Q[5:10])
                +np.asarray(frame['kd'][5:10])*np.asarray(frame['dq_rad_s'][5:10]))
            np.testing.assert_allclose(np.asarray(d['tau_ff_candidate_nm'])+pd,total,atol=1e-12)
            self.assertLessEqual(np.max(np.abs(d['post_transition_ddq_rad_s2'])),15.+1e-9)
        self.assertEqual(start['diagnostics']['mpc_handoff']['execution_check'],
                         'bounded_interpolation_from_previously_transmitted_packet')
        self.assertFalse(end['diagnostics']['mpc_handoff']['active'])
        self.assertFalse(after['diagnostics']['mpc_handoff']['active'])

    def test_spawned_learned_worker_matches_direct_and_retains_parent_handback(self):
        from test_mpc_compute_process import ComputeProcessTests
        helper=ComputeProcessTests()
        try:
            helper.check_runtime_parity(dict(config=LEARNED_MPC_CONFIG,
                predictor_mode='learned_filtered',stationary=False,field_trial=True,
                torque_config=LEARNED_TORQUE_CONFIG,assumed_command_delay_s=None))
        finally:
            helper.doCleanups()


if __name__=='__main__':unittest.main()
