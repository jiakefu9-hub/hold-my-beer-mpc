"""MPC 命令在 2 ms 虚拟执行网格上的延迟语义测试。"""

import unittest

import numpy as np

from sim_support import ArmCommandDelayLine


class ArmCommandDelayLineTest(unittest.TestCase):
    @staticmethod
    def _vector(value):
        return np.full(5, float(value), dtype=np.float64)

    def _line(self, delay):
        return ArmCommandDelayLine(
            step_dt=0.002,
            requested_delay=delay,
            initial_q=self._vector(0.0),
            initial_dq=self._vector(0.0),
            initial_ddq=self._vector(0.0),
        )

    def test_zero_delay_activates_in_the_publish_tick(self):
        line = self._line(0.0)
        command_id = line.publish(
            0.006,
            self._vector(1.0),
            self._vector(2.0),
            self._vector(3.0),
            self._vector(4.0),
        )
        activation = line.activate_ready(0.006)
        self.assertTrue(activation.activated)
        self.assertEqual(activation.packet.command_id, command_id)
        self.assertEqual(activation.effective_delay, 0.0)
        np.testing.assert_array_equal(
            activation.packet.ddq_des, self._vector(4.0)
        )

    def test_non_grid_delay_rounds_up_to_the_next_physics_tick(self):
        line = self._line(0.003)
        line.publish(
            0.0,
            self._vector(1.0),
            self._vector(0.0),
            self._vector(0.0),
            self._vector(0.0),
        )
        self.assertFalse(line.activate_ready(0.002).activated)
        activation = line.activate_ready(0.004)
        self.assertTrue(activation.activated)
        self.assertAlmostEqual(activation.effective_delay, 0.004)
        self.assertEqual(line.delay_ticks, 2)

    def test_when_multiple_commands_are_ready_only_the_newest_is_used(self):
        line = self._line(0.004)
        first = line.publish(
            0.0,
            self._vector(1.0),
            self._vector(0.0),
            self._vector(0.0),
            self._vector(1.0),
        )
        second = line.publish(
            0.0,
            self._vector(2.0),
            self._vector(0.0),
            self._vector(0.0),
            self._vector(2.0),
        )
        activation = line.activate_ready(0.004)
        self.assertTrue(activation.activated)
        self.assertNotEqual(first, second)
        self.assertEqual(activation.packet.command_id, second)
        self.assertEqual(activation.dropped_count, 1)
        np.testing.assert_array_equal(
            activation.packet.target_q, self._vector(2.0)
        )

    def test_command_fields_are_copied_as_one_immutable_packet(self):
        line = self._line(0.002)
        q = self._vector(1.0)
        dq = self._vector(2.0)
        raw = self._vector(3.0)
        desired = self._vector(4.0)
        line.publish(0.0, q, dq, raw, desired)
        q.fill(9.0)
        dq.fill(9.0)
        raw.fill(9.0)
        desired.fill(9.0)
        packet = line.activate_ready(0.002).packet
        np.testing.assert_array_equal(packet.target_q, self._vector(1.0))
        np.testing.assert_array_equal(packet.target_dq, self._vector(2.0))
        np.testing.assert_array_equal(packet.ddq_raw, self._vector(3.0))
        np.testing.assert_array_equal(packet.ddq_des, self._vector(4.0))


if __name__ == "__main__":
    unittest.main()
