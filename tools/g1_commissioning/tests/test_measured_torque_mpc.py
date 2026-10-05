"""Physical-state semantics, independent QP, mapper/fallback and packet tests."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm_mpc import ArmMPCPolicy
from disturbance_types import DisturbanceInput, DisturbanceHorizon
from hardware_mpc_torque_control import (RightArmMeasuredTorqueMpc, HardwareTorquePreviewPlan,
    load_torque_config, make_torque_preview_message)
from hardware_torque_mapper import LocalTorqueMapper, NoModelTorque
from g1_walk_pid import EXPECTED_TARGET_Q, run_device
from g1_walk_mpc import main, build_parser


def horizon():
    d = DisturbanceInput(np.zeros(3), np.zeros(3), np.zeros(3), np.eye(3))
    return DisturbanceHorizon((d,)*10, (d,)*9)


class MapperTest(unittest.TestCase):
    def test_exact_affine_gain_preserves_selection_and_forward_checks(self):
        mapper = LocalTorqueMapper(load_torque_config())
        rng = np.random.default_rng(51026)
        for k in range(60):
            a = rng.normal(size=(5,5))
            mass = a@a.T*.05 + np.eye(5)*.02
            gain = np.linalg.inv(mass)
            bias = rng.normal(size=5)
            forward = lambda tau: gain@(tau-bias)
            desired = rng.uniform(-8,8,5)
            nominal = mass@desired+bias+rng.normal(size=5)*.1
            bounds = (-np.ones(5)*4, np.ones(5)*4)
            if k % 3 == 0:
                bounds[0][4] = bounds[1][4] = bias[4]
            options = dict(previous=bias, bounds=bounds)
            try:
                expected, old = mapper.compute(forward,desired,nominal,bias,**options)
            except NoModelTorque:
                with self.assertRaises(NoModelTorque):
                    mapper.compute(forward,desired,nominal,bias,affine_gain=gain,**options)
                continue
            result,new = mapper.compute(forward,desired,nominal,bias,affine_gain=gain,**options)
            np.testing.assert_allclose(result,expected,atol=1e-10,rtol=0)
            self.assertEqual(old['fallback'],new['fallback'])
            self.assertEqual(len(old['passes']),len(new['passes']))
            np.testing.assert_allclose(new['checked_ddq_rad_s2'],forward(result),atol=1e-10)
            self.assertLess(new['forward_calls'],old['forward_calls'])
            batched, trace = mapper.compute(forward,desired,nominal,bias,affine_gain=gain,
                forward_batch=lambda taus:(taus-bias)@gain.T,**options)
            np.testing.assert_allclose(batched,expected,atol=1e-10,rtol=0)
            self.assertEqual(trace['fallback'],old['fallback'])

    def test_original_simulation_mapper_agrees_on_same_mujoco_forward_model(self):
        import mujoco
        from endpoint_pose import EndpointModel
        from sim_support import local_forward_dynamics_torque_mapping
        endpoint=EndpointModel(); model=endpoint.model
        data=mujoco.MjData(model); scratch=mujoco.MjData(model); forward_data=mujoco.MjData(model)
        data.qpos[:]=model.qpos0;data.qpos[2]=1.5
        data.qpos[endpoint.joint_addresses]=EXPECTED_TARGET_Q[:11]
        indices=np.asarray(endpoint.right_dofs)
        joint_ids=[model.joint(name).id for name in
            ('right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint',
             'right_elbow_joint','right_wrist_roll_joint')]
        actuators=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j)[0] for j in joint_ids])
        mapper=LocalTorqueMapper(load_torque_config())
        for scale in (.3,1.,2.):
            desired=np.array([2,-1,.5,1,-.5])*scale
            nominal=np.zeros(5);fixed=np.zeros(model.nu)
            def forward(tau):
                forward_data.qpos[:]=data.qpos;forward_data.qvel[:]=data.qvel
                forward_data.qacc_warmstart[:]=data.qacc_warmstart
                forward_data.ctrl[:]=fixed;forward_data.ctrl[actuators]=tau
                mujoco.mj_forward(model,forward_data)
                return forward_data.qacc[indices].copy()
            expected,original=local_forward_dynamics_torque_mapping(model,data,scratch,fixed,
                desired,nominal,indices,actuators,np.tile([-25.,25.],(5,1)),
                max_abs_qacc=10.,safe_hold_tau=np.zeros(5))
            actual,trace=mapper.compute(forward,desired,nominal,np.zeros(5))
            self.assertTrue(original.final_output_certified)
            np.testing.assert_allclose(actual,expected,atol=1e-8)

    def test_coupled_total_torque_improves_without_double_pd(self):
        c = load_torque_config()
        mapper = LocalTorqueMapper(c)
        mass = np.diag([.4, .3, .25, .12, .03]); mass[0, 1] = mass[1, 0] = .04
        bias = np.array([2, 1, .4, -1, .1])
        desired = np.array([2, -1, 1, 3, -2.])
        nominal = mass@desired+bias+np.array([.3, -.2, .1, .15, .04])
        forward = lambda tau: np.linalg.solve(mass, tau-bias)
        tau, trace = mapper.compute(forward, desired, nominal, bias)
        self.assertLess(trace['error_norm'], trace['nominal_error_norm'])
        self.assertGreaterEqual(len(trace['passes'][0]['candidates']), 2)
        np.testing.assert_allclose(trace['passes'][0]['gain_rad_s2_per_nm'],np.linalg.inv(mass),atol=1e-12)
        np.testing.assert_allclose(trace['checked_ddq_rad_s2'],forward(tau))
        self.assertFalse(trace['hardware_certified'])

    def test_fallback_rechecks_previous_and_fails_without_acceptable_output(self):
        mapper = LocalTorqueMapper(load_torque_config())
        # Every torque is unsafe: a remembered command must NOT pass blindly.
        with self.assertRaises(NoModelTorque) as error:
            mapper.compute(lambda tau: np.ones(5)*20, np.zeros(5),np.zeros(5),np.zeros(5),np.ones(5))
        self.assertFalse(error.exception.trace['model_accepted'])
        self.assertGreater(error.exception.trace['forward_calls'], 10)

    def test_torque_envelope_and_nonlinear_forward_are_actually_evaluated(self):
        mapper = LocalTorqueMapper(load_torque_config())
        checked=[]
        def forward(tau):
            checked.append(tau.copy())
            return 4*tau + .3*tau**3
        tau, trace = mapper.compute(forward,np.ones(5)*5,np.zeros(5),np.zeros(5),
                                    bounds=(-np.ones(5), np.ones(5)))
        self.assertTrue(all(np.max(np.abs(value)) <= 1 for value in checked))
        np.testing.assert_allclose(trace['checked_ddq_rad_s2'],forward(tau))
        self.assertLess(trace['error_norm'],trace['nominal_error_norm'])

    def test_zero_error_nominal_is_preserved_and_bad_forward_rejected(self):
        mapper=LocalTorqueMapper(load_torque_config())
        tau,trace=mapper.compute(lambda value:value,np.ones(5),np.ones(5),np.zeros(5))
        np.testing.assert_array_equal(tau,np.ones(5))
        with self.assertRaises(ValueError):
            mapper.compute(lambda value:np.full(5,np.nan),np.zeros(5),np.zeros(5),np.zeros(5))


class MeasuredMpcTest(unittest.TestCase):
    def setUp(self):
        self.c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10])
        self.c.warmup(EXPECTED_TARGET_Q,[1,0,0,0],count=3)

    def tearDown(self):self.c.close()

    def test_measurement_is_initial_state_even_when_old_reference_is_wrong(self):
        c=self.c; slots=EXPECTED_TARGET_Q.copy(); slots[5:10]+=.005
        dq=np.array([.03,-.02,.01,.02,-.04])
        c._command_q[:]=10.; c._command_dq[:]=20.
        c.set_measured_dq(dq); c.set_disturbance_horizon(horizon())
        qr,dqr,diag=c.step(slots,[1,0,0,0],0,.018)
        np.testing.assert_allclose(diag['mpc_initial_state'],np.r_[slots[5:10],dq])
        ddq=np.asarray(diag['raw_mpc_ddq_rad_s2'])
        np.testing.assert_allclose(qr,slots[5:10]+dq*.006+.5*ddq*.006**2,atol=1e-12)
        np.testing.assert_allclose(dqr,dq+ddq*.006,atol=1e-12)

    def test_uncondensed_simulation_qp_matches_measured_controller(self):
        c=self.c; p=c.policy
        sim=ArmMPCPolicy(c.nominal,horizon=9,control_dt=.006,
            joint_limits=p.safety_joint_limits,joint_limit_margin=p.joint_limit_margin,
            max_dq=1,max_ddq=8,q_ee_acc=.01,q_ee_alpha=.0005,q_ee_omega=8,
            q_gravity=[30,30],q_posture=[4,4,1.5,.2,.2],q_vel=.08,r_ddq=.0025,
            terminal_scale=2,reg=1e-6,solver_eps_abs=1e-8,solver_eps_rel=1e-8,
            solver_max_iter=50000,solver_time_limit=.5)
        slots=EXPECTED_TARGET_Q.copy(); slots[5:10]+=.007
        dq=np.zeros(5); h=horizon()
        c.model.data.qpos[:]=c.model.model.qpos0
        c.model.data.qpos[c.model.joint_addresses]=slots[:11]
        helpers=c.helper.build_helpers(c.model.data,disturbance_prediction=h.nodes,
            interval_disturbance_prediction=h.intervals,include_kinematics_cache=False)
        sim_q,sim_dq,sim_ddq=sim.compute_action({'current_q':slots[5:10],'current_dq':dq,'dt':.006},helpers)
        self.assertTrue(sim.get_last_diagnostics()['solved'])
        c.reset(); c.set_measured_dq(dq); c.set_disturbance_horizon(h)
        qr,dqr,diag=c.step(slots,[1,0,0,0],0,.006)
        np.testing.assert_allclose(diag['raw_mpc_ddq_rad_s2'],sim_ddq,atol=1e-4)
        np.testing.assert_allclose(qr,sim_q,atol=1e-8)

    def test_packet_contains_pd_exactly_once_and_checks_gains(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        c=self.c; c.set_disturbance_horizon(horizon())
        qr,dqr,diag=c.step(EXPECTED_TARGET_Q,[1,0,0,0],0,.006)
        q=EXPECTED_TARGET_Q.copy(); q[5:10]=qr
        dq=np.zeros(13);dq[5:10]=dqr
        frame=dict(q_rad=q,dq_rad_s=dq,kp=np.r_[np.full(11,20.),0,0],
                   kd=np.r_[np.ones(11),0,0],weight=1.,diagnostics=diag)
        args=(frame,SimpleNamespace(mode_pr=0,mode_machine=4),unitree_hg_msg_dds__LowCmd_,CRC())
        msg=make_torque_preview_message(*args)
        ff=np.array([msg.motor_cmd[i].tau for i in range(22,27)])
        np.testing.assert_allclose(ff+diag['tau_pd_at_feedback_nm'],diag['tau_total_estimated_at_feedback_nm'],atol=1e-6)
        host=make_torque_preview_message(*args,host_full_torque=True)
        self.assertTrue(all(host.motor_cmd[i].kp==host.motor_cmd[i].kd==0 for i in range(22,27)))
        np.testing.assert_allclose([host.motor_cmd[i].tau for i in range(22,27)],diag['tau_total_estimated_at_feedback_nm'],atol=1e-6)
        frame['kp'][5]=21
        with self.assertRaises(ValueError):make_torque_preview_message(*args)

    def test_plan_release_retains_feedforward_and_finishes_zero(self):
        c=self.c
        plan=HardwareTorquePreviewPlan(EXPECTED_TARGET_Q,EXPECTED_TARGET_Q,
            np.r_[np.full(11,20.),0,0],np.r_[np.ones(11),0,0],c)
        for t in [2.994,3.,17.994,18.,18.006]:
            c.set_disturbance_horizon(horizon())
            frame=plan.sample(t,EXPECTED_TARGET_Q,np.zeros(13),[1,0,0,0],0,.006)
            self.assertGreater(np.linalg.norm(frame['diagnostics']['tau_ff_candidate_nm']),.1)
        for t in np.arange(18.012,21.024,.006):
            c.set_disturbance_horizon(horizon())
            frame=plan.sample(t,EXPECTED_TARGET_Q,np.zeros(13),[1,0,0,0],0,.006)
        self.assertTrue(frame['terminal'])
        self.assertEqual(frame['weight'],0.)
        np.testing.assert_array_equal(frame['diagnostics']['tau_ff_candidate_nm'],np.zeros(5))

    def test_independent_log_audit_rejects_wrong_initial_state_and_double_pd(self):
        from audit_measured_torque_replay import check_active
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        c=self.c
        plan=HardwareTorquePreviewPlan(EXPECTED_TARGET_Q,EXPECTED_TARGET_Q,
            np.r_[np.full(11,20.),0,0],np.r_[np.ones(11),0,0],c)
        c.set_disturbance_horizon(horizon())
        frame=plan.sample(3.,EXPECTED_TARGET_Q,np.zeros(13),[1,0,0,0],0,.006)
        crc=CRC()
        with mock.patch.object(crc,'Crc',wraps=crc.Crc) as checksum:
            packet=make_torque_preview_message(frame,SimpleNamespace(mode_pr=0,mode_machine=4),
                unitree_hg_msg_dds__LowCmd_,crc)
            self.assertEqual(checksum.call_count,1)
        row=dict(q_measured_rad=EXPECTED_TARGET_Q,dq_measured_rad_s=np.zeros(13),
            q_command_rad=frame['q_rad'],dq_command_rad_s=frame['dq_rad_s'],
            offline_packet_right_tau_nm=[packet.motor_cmd[i].tau for i in range(22,27)],
            **frame['diagnostics'])
        check_active(row,c.torque_config)
        bad=copy.deepcopy(row);bad['mpc_initial_state'][0]+=.1
        with self.assertRaisesRegex(ValueError,'initial state'):check_active(bad,c.torque_config)
        bad=copy.deepcopy(row);bad['offline_packet_right_tau_nm'][0]+=.1
        with self.assertRaisesRegex(ValueError,'packet'):check_active(bad,c.torque_config)

    def test_offline_default_cannot_reach_device(self):
        self.assertEqual(build_parser().parse_args([]).actuation,'measured_torque_preview')
        with mock.patch('g1_walk_mpc.MpcRuntime') as runtime,contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--execute']),1)
            runtime.assert_not_called()
        with self.assertRaisesRegex(ValueError,'offline-only'):
            run_device(None,None,None,None,None,runtime=SimpleNamespace(
                actuation='measured_torque_preview',controller=self.c))


if __name__=='__main__':unittest.main()
