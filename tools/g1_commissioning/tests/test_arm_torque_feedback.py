"""Small raw-log fixtures for torque evidence (no SDK/network)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_arm_torque_feedback import read_samples,describe


def records(future=False,modern=False):
    stamp=4_000_000_000
    command=dict(event='dds_write',weight=1.,write_end_monotonic_ns=stamp-1_000_000,
        schema='g1_arm_static_command_record_v1',q_target=[0.]*13,dq_target=[0.]*13,
        kp=[20.]*13,kd=[1.]*13,tau_ff=[0.]*13)
    if modern:
        command['schema']='g1_hardware_mpc_command_v1'
        for old,new in [('q_target','q_command_rad'),('dq_target','dq_command_rad_s'),
                        ('kp','kp_command'),('kd','kd_command')]:command[new]=command.pop(old)
        command.pop('tau_ff')
    return [dict(task_epoch_monotonic_ns=0),
        dict(schema='g1_torso_imu_raw_v1',received_monotonic_ns=stamp+(1 if future else -1),
             quaternion_wxyz=[1.,0.,0.,0.]),command,
        dict(schema='g1_lowstate_raw_v1',received_monotonic_ns=stamp,crc_valid=True,
             motors=[dict(index=i,q_rad=.01,dq_rad_s=0.,tau_est_nm=-.1875,ddq_raw_rad_s2=0.)
                     for i in range(22,27)])]


class TorqueFeedbackTest(unittest.TestCase):
    def read(self,rows):
        with tempfile.TemporaryDirectory() as folder:
            raw=Path(folder)/'raw.jsonl'
            raw.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
            return read_samples(raw,stride=1)

    def test_legacy_and_modern_command_mapping_and_explicit_estimate(self):
        for modern in (False,True):
            samples,audit=self.read(records(modern=modern))
            self.assertEqual(audit['selected_samples'],1)
            np.testing.assert_allclose(samples[0]['pd'],-.2)
            np.testing.assert_allclose(samples[0]['tau'],-.1875)

    def test_future_invalid_crc_and_non_full_weight_are_not_joined(self):
        fixtures=[records(future=True),records(),records()]
        fixtures[1][-1]['crc_valid']=False
        fixtures[2][2]['weight']=.5
        for rows in fixtures:
            with self.assertRaisesRegex(ValueError,'no aligned'):self.read(rows)


if __name__=='__main__':unittest.main()
