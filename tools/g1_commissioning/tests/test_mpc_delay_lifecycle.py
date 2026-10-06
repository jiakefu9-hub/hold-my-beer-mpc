"""Focused offline integration: forecast meaning, final history, task and audit."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from disturbance_types import DisturbanceInput,DisturbanceHorizon
from g1_walk_pid import EXPECTED_TARGET_Q
from g1_walk_mpc import MpcRuntime
from hardware_mpc_control import HardwareMpcError
from hardware_mpc_delay_preview import CommandHistory,IssuedCommand
from hardware_mpc_delay_plan import IntervalHorizonClock,HardwareDelayTorquePreviewPlan
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc,make_torque_preview_message
from audit_measured_torque_replay import check_active


def forecast(varying=False):
    zero=np.zeros(3)
    nodes=tuple(DisturbanceInput(np.full(3,999.) if varying else zero,
        zero,zero,np.eye(3)) for _ in range(10))
    intervals=tuple(DisturbanceInput(np.full(3,k+1.) if varying else zero,
        zero,np.full(3,2*k+1.) if varying else zero,np.eye(3)) for k in range(9))
    return DisturbanceHorizon(nodes,intervals)


class ForecastAndHistoryTest(unittest.TestCase):
    def test_interval_integrals_not_instantaneous_node_values(self):
        original=forecast(True);clock=IntervalHorizonClock(original)
        self.assertIs(clock.shifted(0.),original)
        for offset in (.0015,.003,.006,.013,.053,.065):
            shifted=clock.shifted(offset)
            for k,interval in enumerate(shifted.intervals):
                start=offset+k*.006;end=start+.006
                expected=np.zeros((2,3))
                for j in range(9):
                    right=(j+1)*.006 if j<8 else max(end,.054)
                    overlap=max(0.,min(end,right)-max(start,j*.006))/.006
                    expected+=overlap*np.array([original.intervals[j].acc_world,
                                               original.intervals[j].alpha_world])
                np.testing.assert_allclose(interval.acc_world,expected[0],atol=1e-12)
                np.testing.assert_allclose(interval.alpha_world,expected[1],atol=1e-12)

    def test_weight_transition_is_only_explicit_nominal_bias_blend(self):
        h=CommandHistory();h.reset_history();h.assumed_command_delay_s=.006
        h.inverse=SimpleNamespace(linear_dynamics=lambda q,dq,base:(np.eye(5),np.full(5,2.)))
        h.mapper=SimpleNamespace(limit=25.)
        h.torque_config=dict(kp=np.zeros(5),kd=np.zeros(5))
        h._issued.append(IssuedCommand(0.,np.full(5,10.),np.zeros(5),np.zeros(5),.25))
        q,dq,evidence=h._predict(np.zeros(5),np.zeros(5),IntervalHorizonClock(forecast()),0.,0.)
        # tau=.25*10+.75*2=4; ddq=4-2=2.
        np.testing.assert_allclose(q,np.full(5,.5*2*.006**2))
        np.testing.assert_allclose(dq,np.full(5,2*.006))
        self.assertEqual(evidence['command_time_s'],.006)


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        self.constructor,self.crc=unitree_hg_msg_dds__LowCmd_,CRC()
        self.c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10],
            torque_config=Path(__file__).resolve().parents[3]/'configs/hardware_mpc_torque_recovery.yaml')
        self.c.policy.solver_time_limit=.1
        self.plan=HardwareDelayTorquePreviewPlan(EXPECTED_TARGET_Q,EXPECTED_TARGET_Q,
            np.r_[np.full(11,20.),0,0],np.r_[np.ones(11),0,0],self.c,assumed_command_delay_s=.006)

    def tearDown(self):
        self.c.close()

    def sample(self,t):
        self.c.set_disturbance_horizon(forecast())
        self.plan.set_context(t,t-.004,t-.004)
        return self.plan.sample(t,EXPECTED_TARGET_Q,np.zeros(13),[1,0,0,0],0.,.006)

    def packet(self,frame):
        packet=make_torque_preview_message(frame,SimpleNamespace(mode_pr=0,mode_machine=4),
                                           self.constructor,self.crc)
        return type(packet).deserialize(packet.serialize())

    def test_history_commits_wire_values_only_after_final_packet(self):
        frame=self.sample(3.)
        self.assertEqual(len(self.plan.history._issued),0)
        with self.assertRaises(HardwareMpcError):self.plan.set_context(3.006,3.002,3.002)
        packet=self.packet(frame)
        expected=packet.motor_cmd[22].tau
        packet.motor_cmd[22].tau+=.2
        with self.assertRaises(ValueError):self.plan.commit_packet(frame,packet)
        self.assertEqual(self.plan.committed_packets,0)
        packet.motor_cmd[22].tau=expected
        self.plan.commit_packet(frame,packet)
        self.assertEqual(self.plan.committed_packets,1)
        self.assertEqual(self.plan.history._issued[-1].ff[0],expected)
        packet.motor_cmd[22].tau=100.
        self.assertEqual(self.plan.history._issued[-1].ff[0],expected)
        with self.assertRaises(HardwareMpcError):self.plan.commit_packet(frame,packet)
        next_frame=self.sample(3.006)
        self.assertEqual(next_frame['diagnostics']['delay_preview']['issued_history_size'],1)

    def test_audit_uses_predicted_state_without_discarding_measurement(self):
        frame=self.sample(3.)
        packet=self.packet(frame);self.plan.commit_packet(frame,packet)
        row=dict(q_measured_rad=EXPECTED_TARGET_Q.copy(),dq_measured_rad_s=np.zeros(13),
            q_command_rad=frame['q_rad'],dq_command_rad_s=frame['dq_rad_s'],
            offline_packet_right_tau_nm=[packet.motor_cmd[i].tau for i in range(22,27)],
            **frame['diagnostics'])
        check_active(row,self.c.torque_config,delay_enabled=True)
        bad=copy.deepcopy(row);bad['command_time_q_rad'][0]+=.01
        with self.assertRaises(ValueError):check_active(bad,self.c.torque_config,delay_enabled=True)
        bad=copy.deepcopy(row);bad['raw_observed_q_rad'][0]+=.01
        with self.assertRaises(ValueError):check_active(bad,self.c.torque_config,delay_enabled=True)

    def test_entry_and_complete_release_keep_final_zero_feedforward(self):
        entry=self.sample(0.);packet=self.packet(entry);self.plan.commit_packet(entry,packet)
        self.assertEqual(packet.motor_cmd[29].q,0.)
        self.assertTrue(all(packet.motor_cmd[i].tau==0. for i in range(22,27)))
        # A fresh plan can enter release at its retained initial pose; no
        # physical motion is inferred from this task/packet contract check.
        self.plan._last_total=None
        last=None
        for t in np.arange(18.,21.013,.006):
            frame=self.sample(float(t));packet=self.packet(frame);self.plan.commit_packet(frame,packet)
            if last is not None:
                self.assertLessEqual(last-frame['weight'],.002000001)
            last=frame['weight']
        self.assertTrue(frame['terminal'])
        self.assertEqual(self.plan.history._issued[-1].weight,0.)
        np.testing.assert_array_equal(self.plan.history._issued[-1].ff,np.zeros(5))


class RuntimeTimestampTest(unittest.TestCase):
    def test_startup_without_new_callback_does_not_query_backwards(self):
        runtime=MpcRuntime(predictor_mode='hold_current',assumed_command_delay_s=.006)
        try:
            for stamp in range(0,18_000_001,2_000_000):
                runtime.observe_low(stamp,np.zeros(35),np.zeros(35))
                runtime.observe_imu(stamp,[1,0,0,0],[0,0,0],[0,0,9.81])
            runtime.set_epoch(24_000_000)
            runtime.prepare(25_000_000,None,None,0.,0.)
            self.assertLessEqual(runtime.predictor._anchor_ns,18_000_000)
        finally:runtime.close()

    def test_forecast_anchor_can_precede_raw_arm_observation(self):
        runtime=MpcRuntime(predictor_mode='hold_current',actuation='measured_torque_preview',
                           assumed_command_delay_s=.006)
        try:
            q=np.zeros(35)
            # Use the actual reviewed index mapping, including unused slots.
            from hardware_pid_control import ARM_MOTOR_INDICES
            q[:]=0.;q[list(ARM_MOTOR_INDICES)]=EXPECTED_TARGET_Q
            for t in range(0,20_000_001,2_000_000):
                runtime.observe_low(t,q,np.zeros(35))
                if t<=18_000_000:runtime.observe_imu(t,[1,0,0,0],[0,0,0],[0,0,9.81])
            plan=runtime.create_plan(EXPECTED_TARGET_Q,dict(target_q_array=EXPECTED_TARGET_Q,
                kp_array=np.r_[np.full(11,20.),0,0],kd_array=np.r_[np.ones(11),0,0],
                q_offset_limit_deg_array=np.full(5,5.)))
            runtime.prepare(30_000_000,None,None,0.,0.)
            now,observed,anchor=plan._context
            self.assertAlmostEqual(now-observed,.010)
            self.assertAlmostEqual(observed-anchor,.002)
            self.assertFalse(runtime.controller.metadata['delay_lifecycle']['field_output_supported'])
        finally:runtime.close()


if __name__=='__main__':unittest.main()
