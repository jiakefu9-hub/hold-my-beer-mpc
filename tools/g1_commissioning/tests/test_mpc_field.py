"""No-network first-torque trial contracts and independent numerical checks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_walk_mpc import MpcJournal, MpcRuntime, FIELD_TORQUE_CONFIG, main
from g1_walk_pid import EXPECTED_TARGET_Q
from hardware_mpc_field import TorqueHandback, check_field_packet
from hardware_mpc_control import HardwareMpcError
from hardware_mpc_torque_control import load_torque_config
from hardware_pid_control import ARM_MOTOR_INDICES
from hardware_mpc_predictor import FrozenInnovationBank


def packet():
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    p = unitree_hg_msg_dds__LowCmd_()
    for j,i in enumerate(ARM_MOTOR_INDICES):
        p.motor_cmd[i].q = float(EXPECTED_TARGET_Q[j])
        p.motor_cmd[i].dq = .03 if 5<=j<10 else 0.
        p.motor_cmd[i].kp = 20. if j<11 else 0.
        p.motor_cmd[i].kd = 1. if j<11 else 0.
        p.motor_cmd[i].tau = -1. if 5<=j<10 else 0.
    p.motor_cmd[29].q = 1.
    return p


class TorqueFieldTests(unittest.TestCase):
    def test_handback_preserves_complete_PD_law_for_changed_feedback(self):
        h=TorqueHandback();p=packet();h.accept(p)
        original={k:v.copy() if hasattr(v,'copy') else v for k,v in h.last.items()}
        p.motor_cmd[22].tau=99. # must not mutate latched evidence
        frame=h.apply(dict(weight=1.,diagnostics={}))
        rng=np.random.default_rng(5)
        for _ in range(20):
            q=rng.normal(size=5);dq=rng.normal(size=5)
            prior=original['tau'][5:10]+original['kp'][5:10]*(original['q'][5:10]-q)+original['kd'][5:10]*(original['dq'][5:10]-dq)
            after=np.asarray(frame['diagnostics']['tau_ff_candidate_nm'])+frame['kp'][5:10]*(frame['q_rad'][5:10]-q)-frame['kd'][5:10]*dq
            np.testing.assert_allclose(after,prior,atol=1e-12)
        values=[]
        for t in np.arange(0,3.025,.006):
            frame=h.normal_release(t);values.append(frame['weight'])
        self.assertTrue(frame['terminal']);self.assertEqual(values[-1],0.)
        self.assertLessEqual(np.max(-np.diff(values)),.002000001)
        self.assertEqual(frame['diagnostics']['tau_ff_candidate_nm'],[0.]*5)

    def test_no_successful_packet_no_handback_and_bad_packet_rejected(self):
        h=TorqueHandback()
        with self.assertRaises(HardwareMpcError):h.apply(dict(weight=1))
        p=packet();p.motor_cmd[22].tau=float('nan')
        with self.assertRaises(HardwareMpcError):h.accept(p)

    def test_total_not_just_feedforward_checked(self):
        h=TorqueHandback();h.accept(packet());f=h.apply(dict(weight=1.,diagnostics={}))
        q=np.zeros(35);q[list(ARM_MOTOR_INDICES)]=EXPECTED_TARGET_Q
        low=SimpleNamespace(q=q,dq=np.zeros(35))
        conf=load_torque_config(FIELD_TORQUE_CONFIG)
        check_field_packet(f,low,conf)
        low.q[22]+=1.
        with self.assertRaisesRegex(HardwareMpcError,'torque envelope'):check_field_packet(f,low,conf)

    def test_output_optin_and_stationary_first_before_any_runtime(self):
        for args in [['--execute'],['--execute','--allow-first-torque-field-trial','--task','walk'],
                     ['--execute','--actuation','reference_servo','--allow-first-torque-field-trial']]:
            with mock.patch('g1_walk_mpc.MpcRuntime') as runtime, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args),1);runtime.assert_not_called()

    def test_native_journal_is_immutable_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            j=MpcJournal(Path(d)/'run');a=np.arange(10.)
            row=dict(a=a,nested={'x':[1.,float('nan')]})
            j.record(row);a[:]=9;row['nested']['x'][0]=8
            self.assertTrue(j.close())
            value=json.loads((Path(d)/'run/raw.jsonl').read_text())
            self.assertEqual(value['a'],list(map(float,range(10))))
            self.assertEqual(value['nested']['x'],[1.,None])

    def test_exact_neighbors_match_tree_near_and_far(self):
        b=FrozenInnovationBank();rng=np.random.default_rng(6)
        for shift in (.001,.5,15.):
            for n in rng.integers(0,len(b.train_z),size=12):
                z=b.train_z[n]+rng.normal(size=33)*shift
                f=z/b.feature_scale*b.std+b.mean
                _,diag=b.predict(f,np.zeros(12))
                distances,indices=b.tree.query(z,k=8,workers=1)
                self.assertEqual(diag['neighbor_indices'],indices.tolist())
                np.testing.assert_allclose(diag['neighbor_distances'],distances,atol=1e-11)


if __name__=='__main__':unittest.main()
