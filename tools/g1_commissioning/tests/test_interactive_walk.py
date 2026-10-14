"""M10 independent timer, terminal restoration and transport lifecycle tests."""
import os
import pty
import termios
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hardware_interactive_walk import PausableWalk, SpacebarInput
import test_hardware_runner as runner_tests
import g1_walk_pid as runner


class InteractiveScheduleTests(unittest.TestCase):
    def test_two_pauses_excluded_and_final_settle_uses_wall_time(self):
        s = PausableWalk()
        self.assertFalse(s.toggle(4.)[0])
        self.assertTrue(s.toggle(7.)[0])
        self.assertEqual(s.snapshot(40.).walking_elapsed_s, 2.)
        self.assertEqual(s.snapshot(40.).phase, 'paused_hold')
        self.assertIsNone(s.snapshot(40.).release_start_s)
        self.assertLess(s.snapshot(40.).plan_time_s, 15.)
        self.assertTrue(s.toggle(40.)[0])
        self.assertTrue(s.toggle(45.)[0])
        self.assertEqual(s.snapshot(50.).walking_elapsed_s, 7.)
        self.assertTrue(s.toggle(50.)[0])
        self.assertEqual(s.snapshot(62.9).phase, 'forward_walk')
        end = s.snapshot(63.)
        self.assertEqual((end.phase, end.walking_elapsed_s, end.plan_time_s, end.release_start_s),
                         ('stop_settle', 20., 15., 66.))
        self.assertFalse(s.toggle(63.)[0])
        self.assertFalse(s.toggle(100.)[0])
        self.assertEqual(s.snapshot(66.).phase, 'release_due')

    def test_command_expiry_pause_zero_and_abort_inhibition(self):
        s = PausableWalk()
        m = s.snapshot(24.95).motion(24.95, .1)
        self.assertAlmostEqual(m['duration_s'], .05)
        self.assertEqual((m['vx_m_s'], m['yaw_rate_rad_s']), (.5, .1))
        self.assertEqual(s.snapshot(25.).motion(25., .1)['vx_m_s'], 0.)
        self.assertEqual(s.snapshot(28.).motion(28., .1)['yaw_rate_rad_s'], 0.)
        s = PausableWalk()
        s.toggle(6.)
        self.assertFalse(s.toggle(6.1)[0])
        m = s.snapshot(9.).motion(9., .1)
        self.assertEqual((m['vx_m_s'], m['yaw_rate_rad_s']), (0., 0.))
        s.toggle(10.)
        m = s.snapshot(11.).motion(11., .1, inhibited=True)
        self.assertEqual((m['vx_m_s'], m['yaw_rate_rad_s']), (0., 0.))
        for bad in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                s.snapshot(bad)

    def test_cbreak_reads_space_without_enter_and_restores_terminal(self):
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        before = termios.tcgetattr(slave)
        reader = SpacebarInput(slave)
        try:
            reader.open()
            active = termios.tcgetattr(slave)
            self.assertTrue(active[3] & termios.ISIG)
            self.assertFalse(active[3] & termios.ICANON)
            self.assertFalse(active[3] & termios.ECHO)
            os.write(master, b' ')
            self.assertEqual(reader.poll(.2), b' ')
        finally:
            reader.close()
        self.assertEqual(termios.tcgetattr(slave), before)
        reader.close()


class InteractiveRunnerTests(unittest.TestCase):
    def run_scenario(self, failure=None, keys=(6., 10., 16., 18.)):
        h = runner_tests.Harness(failure, torque=True)
        events = list(keys)
        opened_closed = []

        class Keyboard:
            def open(self): opened_closed.append('open')
            def close(self): opened_closed.append('close')
            def poll(self, timeout):
                h.clock.sleep(.005)
                if events and h.task_s() >= events[0]:
                    events.pop(0)
                    return b' '
                return b''

        original_run = runner.run_device
        def run_interactive(args, *a, **kw):
            args.space_pause_walk = True
            return original_run(args, *a, **kw)

        with patch.object(runner, 'SpacebarInput', Keyboard), \
                patch.object(runner, 'run_device', side_effect=run_interactive):
            result = h.run()
        self.assertEqual(opened_closed, ['open', 'close'])
        runner_tests.FieldRunnerTest().assert_gradual_release(h)
        self.assertEqual(h.velocities[-1][1], [0., 0., 0.])
        return h, result

    def test_pause_resume_keeps_control_beyond_old_end_and_completes_20s(self):
        h, result = self.run_scenario()
        self.assertEqual(result, 0)
        events = [r for r in h.rows if r.get('event') == 'space_key' and r['accepted']]
        self.assertEqual(len(events), 4)
        frames = [r for r in h.rows if r.get('event') == 'dds_write']
        paused = [r for r in frames if r['stage'] == 'paused_hold']
        self.assertGreater(len(paused), 200)
        self.assertTrue(all(r['weight'] == 1. and r.get('mpc_active') for r in paused))
        after_old_end = [r for r in frames if 22. < r['task_elapsed_s'] < 30.]
        self.assertTrue(after_old_end)
        self.assertTrue(all(r['weight'] == 1. and r.get('mpc_active') for r in after_old_end))
        replies = [r for r in h.rows if r.get('event') == 'velocity_reply']
        self.assertTrue(any(r['walk_schedule']['phase'] == 'paused_hold' for r in replies))
        for r in replies:
            if r['walk_schedule']['phase'] == 'paused_hold':
                self.assertTrue(r['zero_command'])
        end = next(r for r in h.rows if r.get('event') == 'session_end')
        self.assertEqual(end['walk_schedule']['walking_elapsed_s'], 20.)
        released = next(r for r in h.rows if r.get('event') == 'normal_weight_release_started')
        self.assertGreaterEqual(released['task_elapsed_s'], 33.8)
        self.assertTrue(released['zero_velocity_acknowledged'])
        self.assertFalse(any(r.get('event') == 'session_fault' for r in h.rows))

    def test_ctrl_c_while_paused_finishes_without_resuming(self):
        h, result = self.run_scenario('operator', keys=(5.4,))
        self.assertEqual(result, 130)
        self.assertTrue(any(r.get('stage') == 'paused_hold' for r in h.rows))
        self.assertTrue(all(v == [0., 0., 0.] for t, v in h.velocities if t > 5.6))

    def test_velocity_failure_while_paused_releases(self):
        h, result = self.run_scenario('velocity_timeout', keys=(5.4,))
        self.assertEqual(result, 3)
        self.assertTrue(all(v == [0., 0., 0.] for t, v in h.velocities if t > 5.6))


class InteractiveMpcTests(unittest.TestCase):
    def test_real_qp_worker_keeps_fresh_state_during_long_pause(self):
        import g1_run_morning_20261013 as launch
        from g1_walk_mpc import MpcRuntime, apply_zero_arm_neutral_experiment, load_profile
        from hardware_pid_control import ARM_MOTOR_INDICES
        from mpc_compute_process import ProcessMpcRuntime
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        options = dict(config=launch.MPC_CONFIG, model_config=launch.MODEL,
            torque_config=launch.TORQUE_CONFIG, predictor_mode='learned_filtered',
            stationary=False, field_trial=True, zero_arm_neutral=True)
        affinity = set(os.sched_getaffinity(0))
        direct = MpcRuntime(**options)
        self.addCleanup(direct.close)
        worker = ProcessMpcRuntime(compute_cpu=min(affinity), compute_affinity=affinity, **options)
        self.addCleanup(worker.close)
        # Offline parity on synthetic 2 ms observations; not a deadline benchmark.
        worker._step_timeout_s = 2.
        profile = apply_zero_arm_neutral_experiment(load_profile(launch.PROFILE, 'pid'), True)
        plans = [r.create_plan(profile['target_q_array'], profile) for r in (direct, worker)]
        q = np.zeros(35)
        q[list(ARM_MOTOR_INDICES)] = profile['target_q_array']
        state = SimpleNamespace(q=q, dq=np.zeros(35), mode_pr=0, mode_machine=4, received_ns=0)
        epoch = 1_000_000_000
        for r in (direct, worker):
            for t in range(epoch-500_000_000, epoch+1, 2_000_000):
                r.observe_low(t, q, state.dq)
                r.observe_imu(t, [1, 0, 0, 0], [0, 0, 0], [0, 0, 9.81])
            r.set_epoch(epoch)
        schedule = PausableWalk()
        previous = epoch
        toggles = [6., 37.]
        paused_samples = 0
        for index in range(1, 10000):
            wall = index*.006
            if toggles and wall >= toggles[0]:
                schedule.toggle(wall)
                toggles.pop(0)
            snap = schedule.snapshot(wall)
            if snap.phase == 'release_due':
                break
            paused_samples += snap.phase == 'paused_hold'
            stamp = epoch+round(wall*1e9)
            state.received_ns = stamp
            for r in (direct, worker):
                for t in range(previous+2_000_000, stamp+1, 2_000_000):
                    r.observe_low(t, q, state.dq)
                    r.observe_imu(t, [1, 0, 0, 0], [0, 0, 0], [0, 0, 9.81])
                r.prepare(stamp, state, state, 0., wall, heading_frozen=wall >= 5.)
            frames = [p.sample(snap.plan_time_s, profile['target_q_array'], np.zeros(13),
                               [1, 0, 0, 0], 0., .006) for p in plans]
            for k in ('q_rad', 'dq_rad_s', 'kp', 'kd', 'weight'):
                np.testing.assert_allclose(frames[0][k], frames[1][k], atol=1e-12, rtol=0.)
            if wall >= 4.806:
                for f in frames:
                    self.assertEqual(f['weight'], 1.)
                    self.assertTrue(f['diagnostics']['mpc_active'])
                    self.assertFalse(f['diagnostics']['mpc_handoff']['active'])
            for r, f in zip((direct, worker), frames):
                packet = r.make_message(f, state, unitree_hg_msg_dds__LowCmd_, r.create_crc())
                r.accept_packet(f, packet)
            previous = stamp
        self.assertGreater(paused_samples, 5000)
        self.assertEqual(snap.phase, 'release_due')
        self.assertEqual(snap.walking_elapsed_s, 20.)


if __name__ == '__main__':
    unittest.main()
