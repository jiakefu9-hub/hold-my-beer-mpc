"""Replay audits independently check consecutive sent references, not self-reports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_hardware_mpc_replays import audit_commands, NOMINAL_RIGHT


class ReplayAuditTest(unittest.TestCase):
    def run_audit(self, tamper=None):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"raw.jsonl"
            with path.open("w") as stream:
                for i in range(2500):
                    row=dict(sequence=i+500,task_elapsed_s=3+i*.006,mpc_active=True,
                        mpc=dict(solved=True,fallback_used=False,max_constraint_violation=0.),
                        q_command_rad=[0.]*5+NOMINAL_RIGHT+[0.]*3,dq_command_rad_s=[0.]*13,
                        governed_ddq_reference_rad_s2=[0.]*5,command_integration_dt_s=.006,
                        feedback_dt_s=.006)
                    if tamper and i==100:
                        tamper(row)
                    stream.write(json.dumps(row)+"\n")
            return audit_commands(path)

    def test_zero_reference_passes(self):
        result=self.run_audit()
        self.assertEqual(result["max_independent_acceleration_rad_s2"],0.)
        self.assertEqual(result["max_position_integration_residual_rad"],0.)

    def test_fake_acceleration_log_cannot_hide_actual_jump(self):
        with self.assertRaisesRegex(ValueError,"max_independent_acceleration"):
            self.run_audit(lambda row:row["dq_command_rad_s"].__setitem__(5,.05))

    def test_angle_jump_and_missing_sequence_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"position_integration"):
            self.run_audit(lambda row:row["q_command_rad"].__setitem__(5,NOMINAL_RIGHT[0]+.001))
        with self.assertRaisesRegex(ValueError,"sequence/time gap"):
            self.run_audit(lambda row:row.update(sequence=9999))


if __name__ == "__main__":
    unittest.main()
