"""Reset, supersede and zero-delay parity details for the L1 scheduler."""

from __future__ import annotations

import unittest

import numpy as np

from right_arm_runtime.sim_mpc_latency import FixedMpcResultDelayLine
from sim_support import ArmCommandDelayLine


class MpcLatencyResetAndParityTest(unittest.TestCase):
    @staticmethod
    def vector(value: float, size: int = 5) -> np.ndarray:
        return np.full(size, value, dtype=np.float64)

    def publish(self, line, sample, anchor, value):
        return line.publish(
            source_time=sample * 0.002,
            source_sample_index=sample,
            source_anchor_index=anchor,
            q_ref=self.vector(value),
            dq_ref=self.vector(value + 1),
            ddq_raw=self.vector(value + 2),
            ddq_des=self.vector(value + 3),
            source_right_arm_q=self.vector(0),
            source_right_arm_dq=self.vector(0),
            source_torso_acc=self.vector(0, 3),
            source_torso_omega=self.vector(0, 3),
            diagnostics={"anchor": anchor},
            packet_ready_wall_ns=100 + anchor,
        )

    def test_reset_clears_active_pending_time_and_ids(self):
        line = FixedMpcResultDelayLine(
            step_dt=0.002, requested_delay_s=0.002, mpc_dt=0.006
        )
        first = self.publish(line, 12, 4, 1)
        line.activate_ready(now=0.026, sample_index=13)
        self.assertIsNotNone(line.active_packet)
        line.reset()
        self.assertIsNone(line.active_packet)
        self.assertIsNone(line.activate_ready(now=0.0, sample_index=0).packet)
        restarted = self.publish(line, 0, 0, 2)
        self.assertEqual(first.command_id, restarted.command_id)

    def test_repeated_or_backward_execution_tick_fails_closed(self):
        line = FixedMpcResultDelayLine(
            step_dt=0.002, requested_delay_s=0.0, mpc_dt=0.006
        )
        line.activate_ready(now=0.004, sample_index=2)
        with self.assertRaises(ValueError):
            line.activate_ready(now=0.004, sample_index=2)
        with self.assertRaises(ValueError):
            line.activate_ready(now=0.002, sample_index=1)

    def test_late_poll_supersedes_older_ready_packet(self):
        line = FixedMpcResultDelayLine(
            step_dt=0.002, requested_delay_s=0.004, mpc_dt=0.006
        )
        self.publish(line, 0, 0, 1)
        newest = self.publish(line, 3, 1, 2)
        activation = line.activate_ready(now=0.010, sample_index=5)
        self.assertEqual(activation.packet.command_id, newest.command_id)
        self.assertEqual(activation.dropped_count, 1)

    def test_experimental_zero_packet_matches_formal_immediate_fields(self):
        formal = ArmCommandDelayLine(
            step_dt=0.002,
            requested_delay=0.0,
            initial_q=self.vector(0),
            initial_dq=self.vector(0),
            initial_ddq=self.vector(0),
        )
        experimental = FixedMpcResultDelayLine(
            step_dt=0.002, requested_delay_s=0.0, mpc_dt=0.006
        )
        formal.publish(
            0.024, self.vector(1), self.vector(2),
            self.vector(3), self.vector(4),
        )
        self.publish(experimental, 12, 4, 1)
        formal_packet = formal.activate_ready(0.024).packet
        experimental_packet = experimental.activate_ready(
            now=0.024, sample_index=12
        ).packet
        np.testing.assert_array_equal(formal_packet.target_q, experimental_packet.q_ref)
        np.testing.assert_array_equal(formal_packet.target_dq, experimental_packet.dq_ref)
        np.testing.assert_array_equal(formal_packet.ddq_raw, experimental_packet.ddq_raw)
        np.testing.assert_array_equal(formal_packet.ddq_des, experimental_packet.ddq_des)
        self.assertEqual(formal_packet.source_time, experimental_packet.source_time)


if __name__ == "__main__":
    unittest.main()
