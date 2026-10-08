"""No DDS: identification waveform and offline estimator checks."""
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from g1_arm_system_identification import (CONFIG, IdentificationPlan,
    IdentificationRuntime, load_identification_config, preflight)
from g1_walk_pid import EXPECTED_TARGET_Q
from hardware_pid_control import ARM_MOTOR_INDICES
from analyze_arm_system_identification import NOMINAL, identify, ridge_fit


class ArmIdentificationTests(unittest.TestCase):
    def setUp(self):
        self.config=load_identification_config(CONFIG)
        kp=np.r_[np.full(11,20.),0.,0.];kd=np.r_[np.ones(11),0.,0.]
        self.plan=IdentificationPlan(EXPECTED_TARGET_Q,EXPECTED_TARGET_Q,kp,kd,self.config)

    def test_waveform_is_faded_bounded_unique_and_uses_mpc_packet_gains(self):
        rows=[self.plan.sample(t,EXPECTED_TARGET_Q,np.zeros(13),[1,0,0,0],0.,.006)
              for t in np.arange(0.,18.,.006)]
        torque=np.asarray([r['diagnostics']['tau_ff_candidate_nm'] for r in rows])
        np.testing.assert_array_less(np.max(abs(torque),axis=0),self.config['tau_peak_nm']+1e-9)
        self.assertEqual(len(np.unique(self.config['frequencies_hz'])),15)
        np.testing.assert_allclose(rows[0]['diagnostics']['tau_ff_candidate_nm'],0.)
        np.testing.assert_allclose(rows[-1]['diagnostics']['tau_ff_candidate_nm'],0.)
        active=[r for r in rows if r['stage']=='arm_identification']
        self.assertGreater(len(active),1600)
        np.testing.assert_allclose(active[100]['kp'][5:10],self.config['right_kp'])
        np.testing.assert_allclose(active[100]['kd'][5:10],self.config['right_kd'])

    def test_preflight_is_explicitly_offline(self):
        result=preflight(CONFIG,cpu=min(__import__('os').sched_getaffinity(0)))
        self.assertTrue(result['passed'])
        self.assertFalse(result['dds_initialized']);self.assertFalse(result['publisher_created'])
        np.testing.assert_allclose(result['max_abs_tau_ff_nm'],self.config['tau_peak_nm'],rtol=2e-4)

    def test_packet_contains_only_bounded_right_feedforward_and_state_guard_is_live(self):
        from mpc_crc import PackedCRC
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        profile=dict(target_q_array=EXPECTED_TARGET_Q.copy(),
                     kp_array=np.r_[np.full(11,20.),0.,0.],kd_array=np.r_[np.ones(11),0.,0.])
        runtime=IdentificationRuntime(profile,self.config)
        plan=runtime.create_plan(EXPECTED_TARGET_Q,profile)
        frame=plan.sample(6.,EXPECTED_TARGET_Q,np.zeros(13),[1,0,0,0],0.,.006)
        q=np.zeros(35);dq=np.zeros(35);q[list(ARM_MOTOR_INDICES)]=EXPECTED_TARGET_Q
        state=SimpleNamespace(q=q,dq=dq,mode_pr=0,mode_machine=4)
        packet=runtime.make_message(frame,state,unitree_hg_msg_dds__LowCmd_,PackedCRC())
        expected=frame['diagnostics']['tau_ff_candidate_nm']
        np.testing.assert_allclose([packet.motor_cmd[i].tau for i in range(22,27)],expected,rtol=1e-6)
        np.testing.assert_allclose([packet.motor_cmd[i].tau for i in range(15,20)],0.)
        runtime.accept_packet(frame,packet)
        self.assertEqual(runtime.release_frame(0.)['weight'],1.)
        state.q[22]=EXPECTED_TARGET_Q[5]+np.deg2rad(10.1)
        with self.assertRaisesRegex(RuntimeError,'state envelope'):
            runtime.check_before_write(frame,state,None,0,0)

    def test_ridge_recovers_multivariable_coupling(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(1200,9))
        matrix=rng.normal(size=(9,5));y=x@matrix+.01*rng.normal(size=(1200,5))
        fit=ridge_fit(x,y,np.arange(800),np.arange(800,1200),ridge=1e-8)
        np.testing.assert_allclose(fit['coef'],matrix,atol=.003,rtol=0.)
        self.assertLess(max(fit['rmse']),.012)

    def test_complete_synthetic_capture_reaches_identification_report(self):
        count=1500;dt=.006;t=np.arange(count)*dt
        ff=np.column_stack([.12*np.sin(2*np.pi*(.5+.37*j)*t+.2*j)
                            +.06*np.sin(2*np.pi*(1.7+.11*j)*t) for j in range(5)])
        delayed=np.vstack((np.zeros((1,5)),ff[:-1]))
        plant=np.diag([6.,8.,10.,7.,12.])+np.array([
            [0,.3,-.2,.1,0],[-.2,0,.4,0,.1],[.1,-.3,0,.2,0],
            [.4,0,.1,0,-.2],[0,.2,0,-.1,0]])
        qdd=delayed@plant.T;dq=np.cumsum(qdd,axis=0)*dt
        q=NOMINAL+np.cumsum(dq,axis=0)*dt
        with tempfile.TemporaryDirectory() as directory:
            raw=Path(directory)/'raw.jsonl'
            with raw.open('w') as stream:
                def write(row): stream.write(json.dumps(row)+'\n')
                write(dict(schema='g1_arm_identification_session_v1',event='session_start'))
                for i in range(count):
                    full=lambda right: np.r_[EXPECTED_TARGET_Q[:5],right,EXPECTED_TARGET_Q[10:]]
                    right_only=lambda right: np.r_[np.zeros(5),right,np.zeros(3)]
                    state_ns=1_000_000_000+i*6_000_000
                    # The command is written 3 ms after this feedback.  qdd[i]
                    # uses ff[i-1], whose preceding write is therefore 3 ms old.
                    write_ns=state_ns+3_000_000
                    write(dict(schema='g1_hardware_pid_command_v1',event='dds_write',
                        stage='arm_identification',task_elapsed_s=5+t[i],
                        state_received_monotonic_ns=state_ns,write_begin_monotonic_ns=write_ns,
                        q_measured_rad=full(q[i]).tolist(),dq_measured_rad_s=full(dq[i]).tolist(),
                        q_command_rad=EXPECTED_TARGET_Q.tolist(),dq_command_rad_s=np.zeros(13).tolist(),
                        kp_command=np.r_[np.full(5,20.),self.config['right_kp'],20.,0.,0.].tolist(),
                        kd_command=np.r_[np.ones(5),self.config['right_kd'],1.,0.,0.].tolist(),
                        tau_ff=right_only(ff[i]).tolist(),
                        tau_est_at_feedback_nm=right_only(delayed[i]).tolist()))
                    write(dict(schema='g1_torso_imu_raw_v1',received_monotonic_ns=state_ns,
                        accelerometer_raw_m_s2=[0.,0.,9.81],gyroscope_rad_s=[0.,0.,0.]))
                write(dict(schema='g1_pid_event_v1',event='session_end',
                           outcome='normal_release_completed',final_weight=0.))
            report=identify([raw])
        self.assertEqual(report['acceleration_response']['input_matrix_rank'],5)
        self.assertLessEqual(abs(report['acceleration_response']['selected_common_delay_ms']-3),3)
        self.assertEqual(report['timing_alignment']['feedback'],'state_received_monotonic_ns')
        self.assertFalse(report['acceptance']['automatic_controller_update'])
        self.assertFalse(report['acceptance']['preliminary_candidate'])


if __name__=='__main__': unittest.main()
