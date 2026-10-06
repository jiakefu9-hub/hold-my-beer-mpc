"""Independent official SDK checksum agreement, with no DDS endpoints."""
import unittest
import numpy as np
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from mpc_crc import PackedCRC


class PackedCrcTest(unittest.TestCase):
    def test_identical_to_sdk_on_random_payloads_and_changed_crc_field(self):
        official=CRC();fast=PackedCRC();rng=np.random.default_rng(20261005)
        self.assertIsNot(official,fast)
        for _ in range(60):
            cmd=unitree_hg_msg_dds__LowCmd_();state=unitree_hg_msg_dds__LowState_()
            cmd.mode_pr=state.mode_pr=int(rng.integers(0,2))
            cmd.mode_machine=state.mode_machine=int(rng.integers(0,10))
            state.tick=int(rng.integers(0,2**32))
            for c,s in zip(cmd.motor_cmd,state.motor_state):
                c.mode=s.mode=1
                c.q,c.dq,c.kp,c.kd,c.tau=map(float,rng.normal(size=5))
                s.q,s.dq,s.ddq,s.tau_est,s.vol=map(float,rng.normal(size=5))
                s.temperature=[20,40];s.sensor=[13,37];s.reserve=[1,2,3,4]
            for message in (cmd,state):
                expected=official.Crc(message)
                self.assertEqual(fast.Crc(message),expected)
                message.crc=123456789
                self.assertEqual(fast.Crc(message),expected)
        self.assertIs(type(CRC()),CRC)
        self.assertIsNot(fast.crc_lib,official.crc_lib)


if __name__=='__main__':unittest.main()
