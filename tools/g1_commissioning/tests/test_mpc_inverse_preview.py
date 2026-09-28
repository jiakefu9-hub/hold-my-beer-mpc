"""Candidate feedforward and strict offline boundary checks."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_walk_mpc import main,preflight
from g1_walk_pid import EXPECTED_TARGET_Q,run_device,make_arm_message
from hardware_mpc_inverse_preview import RightArmInversePreviewMpc,make_offline_preview_message
from disturbance_types import DisturbanceInput,DisturbanceHorizon


class InversePreviewTest(unittest.TestCase):
    def test_real_output_rejected_before_runtime_or_sdk(self):
        with mock.patch('g1_walk_mpc.MpcRuntime') as runtime, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--execute','--actuation','inverse_dynamics_preview']),1)
            runtime.assert_not_called()
        with self.assertRaisesRegex(ValueError,'offline-only'):
            run_device(None,None,None,None,None,runtime=SimpleNamespace(
                actuation='inverse_dynamics_preview',controller=SimpleNamespace()))

    def test_preflight_builds_nonzero_torque_without_network(self):
        with mock.patch('socket.socket',side_effect=AssertionError('no network allowed')):
            r=preflight(actuation='inverse_dynamics_preview')
        self.assertTrue(r['passed'])
        self.assertFalse(r['field_output_supported'])
        self.assertFalse(r['publisher_created'])
        self.assertGreater(abs(r['offline_packet_right_tau_nm'][0]),1.)
        self.assertEqual(r['serialized_bytes'],1004)

    def test_feedforward_uses_current_node_actual_qdq_and_governed_ddq(self):
        c=RightArmInversePreviewMpc(EXPECTED_TARGET_Q[5:10])
        try:
            c.warmup(EXPECTED_TARGET_Q,[1,0,0,0],count=3)
            current=DisturbanceInput(np.zeros(3),np.zeros(3),np.zeros(3),np.eye(3))
            future=DisturbanceInput(np.ones(3)*.1,np.zeros(3),np.zeros(3),np.eye(3))
            h=DisturbanceHorizon((current,)+(future,)*9,(future,)*9)
            slots=EXPECTED_TARGET_Q.copy();slots[5:10]+=.03
            c.set_measured_dq(np.ones(5)*.01)
            c.set_disturbance_horizon(h)
            with mock.patch.object(c.inverse,'compute',wraps=c.inverse.compute) as compute:
                q,dq,diag=c.step(slots,[1,0,0,0],0,.006)
                args=compute.call_args.args
                np.testing.assert_allclose(args[0],slots[5:10])
                np.testing.assert_allclose(args[1],.01)
                np.testing.assert_allclose(args[2],diag['governed_ddq_reference_rad_s2'])
                np.testing.assert_allclose(args[3].acc_world,current.acc_world)
            np.testing.assert_allclose(diag['tau_total_estimated_at_feedback_nm'],
                np.array(diag['tau_ff_candidate_nm'])+20*(q-slots[5:10])+(dq-.01))
            self.assertFalse(diag['torque_estimate_feedback_used_for_control'])
            self.assertFalse(diag['torque_output_authorized'])
        finally:c.close()

    def test_candidate_packet_does_not_add_pd_twice_or_change_live_packet(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        frame=dict(q_rad=EXPECTED_TARGET_Q,dq_rad_s=np.zeros(13),
            kp=np.r_[np.full(11,20.),0,0],kd=np.r_[np.ones(11),0,0],weight=1.,
            diagnostics={'controller_kind':'inverse_dynamics_preview',
                         'tau_ff_candidate_nm':[1.,2.,3.,4.,.5]})
        state=SimpleNamespace(mode_pr=0,mode_machine=4)
        p=make_offline_preview_message(frame,state,unitree_hg_msg_dds__LowCmd_,CRC())
        for i in range(35):
            expected=[1.,2.,3.,4.,.5][i-22] if 22<=i<=26 else 0.
            self.assertEqual(p.motor_cmd[i].tau,expected)
        self.assertEqual(p.crc,CRC().Crc(p))
        p=make_arm_message(frame,state,unitree_hg_msg_dds__LowCmd_,CRC())
        self.assertTrue(all(x.tau==0 for x in p.motor_cmd))


if __name__ == '__main__':unittest.main()
