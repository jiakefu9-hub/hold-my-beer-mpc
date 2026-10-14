"""Morning trial isolation, unchanged MPC admission and zero PID reference safety."""
import contextlib
import io
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import g1_run_morning_20261013 as launch
import g1_walk_mpc
import g1_walk_pid
from hardware_pid_control import RightArmHardwarePid
from hardware_position_pid_zero import CONFIG, RELAXED_CONFIG, EXPECTED, prepare_profile


def parser_args(command, script):
    index = next(i for i, x in enumerate(command) if x.endswith('/'+script))
    return command[index+1:]


class MorningTrials(unittest.TestCase):
    def test_no_implicit_execution_or_batch(self):
        for args in ([], ['M01'], ['M03'], ['M07'], ['M08'], ['M10']):
            with patch('os.execv', side_effect=AssertionError('implicit output')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch.main(args), 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.main(['--execute'])

    def test_learned_and_hold_are_only_predictor_change(self):
        for execute in (False, True):
            a = launch.build_command('M01', execute=execute, output_dir='/tmp/morning_contract_not_created')
            b = launch.build_command('M02', execute=execute, output_dir='/tmp/morning_contract_not_created')
            index = a.index('--predictor')+1
            self.assertEqual(a[index], 'learned_filtered')
            self.assertEqual(b[index], 'hold_current')
            b[index] = a[index]
            self.assertEqual(a, b)
            p = g1_walk_mpc.build_parser(learned=True).parse_args(parser_args(a, 'g1_walk_mpc_learned.py'))
            self.assertFalse(p.align_accel_forecast)
            self.assertTrue(p.zero_arm_neutral)
            self.assertEqual(p.left_pd_gain_scale, 1.)
            self.assertEqual(p.torque_config, launch.TORQUE_CONFIG)
            self.assertIsNone(p.assumed_command_delay_ms)
            self.assertIsNone(p.acceleration_pid_config)

    def test_repeats_are_identical_except_capture_directory(self):
        for a, b in [('M01', 'M06'), ('M02', 'M05'), ('M03', 'M04')]:
            self.assertEqual(launch.build_command(a, execute=True, output_dir='/tmp/morning_contract_not_created'),
                             launch.build_command(b, execute=True, output_dir='/tmp/morning_contract_not_created'))

    def test_m10_is_m01_with_only_space_pause_protocol(self):
        for execute in (False, True):
            a = launch.build_command('M01', execute=execute, output_dir='/tmp/morning_contract_not_created')
            b = launch.build_command('M10', execute=execute, output_dir='/tmp/morning_contract_not_created')
            p = g1_walk_mpc.build_parser(learned=True).parse_args(parser_args(b, 'g1_walk_mpc_learned.py'))
            self.assertTrue(p.space_pause_walk)
            self.assertEqual(p.left_pd_gain_scale, 1.)
            b.remove('--space-pause-walk')
            self.assertEqual(a, b)

    def test_m10_rejects_missing_terminal_before_host_or_robot_setup(self):
        args = parser_args(launch.build_command('M10', execute=True), 'g1_walk_mpc_learned.py')
        with patch('sys.stdin.isatty', return_value=False), \
                patch('field_performance.prepare', side_effect=AssertionError('host setup')), \
                patch.object(g1_walk_mpc, 'run_device', side_effect=AssertionError('robot setup')), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(g1_walk_mpc.main(args, learned=True), 1)

    def test_m08_only_doubles_all_five_left_packet_pd_gains(self):
        for execute in (False, True):
            m01 = launch.build_command(
                'M01', execute=execute, output_dir='/tmp/morning_contract_not_created')
            m08 = launch.build_command(
                'M08', execute=execute, output_dir='/tmp/morning_contract_not_created')
            scale_index = m08.index('--left-pd-gain-scale') + 1
            self.assertEqual(m01[scale_index], '1')
            self.assertEqual(m08[scale_index], '2')
            m08[scale_index] = m01[scale_index]
            self.assertEqual(m01, m08)

        profile = g1_walk_mpc.load_profile(launch.PROFILE, 'pid')
        original_kp = profile['kp_array'].copy()
        original_kd = profile['kd_array'].copy()
        effective = g1_walk_mpc.apply_left_pd_gain_experiment(profile, 2.)
        np.testing.assert_array_equal(effective['kp_array'][:5], 2.*original_kp[:5])
        np.testing.assert_array_equal(effective['kd_array'][:5], 2.*original_kd[:5])
        np.testing.assert_array_equal(effective['kp_array'][5:], original_kp[5:])
        np.testing.assert_array_equal(effective['kd_array'][5:], original_kd[5:])
        np.testing.assert_array_equal(profile['kp_array'], original_kp)
        np.testing.assert_array_equal(profile['kd_array'], original_kd)
        self.assertEqual(effective['left_pd_gain_experiment']['slots'], list(range(5)))

    def test_output_directories_are_unique_and_not_created(self):
        a, b = [launch.build_command('M03', execute=True) for _ in range(2)]
        paths = [Path(x[x.index('--output-dir')+1]) for x in (a, b)]
        self.assertNotEqual(*paths)
        self.assertFalse(any(p.exists() for p in paths))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileExistsError):
            launch.build_command('M01', execute=True, output_dir=tmp)

    def test_pid_execution_gate_and_preflight_parser(self):
        cmd = launch.build_command('M03', execute=True)
        args = parser_args(cmd, 'g1_walk_pid.py')
        parsed = g1_walk_pid.build_parser().parse_args(args)
        self.assertTrue(parsed.zero_arm_neutral)
        self.assertEqual(parsed.permit_real_output, g1_walk_pid.PERMIT)
        self.assertEqual(parsed.position_pid_config, CONFIG)
        del args[args.index('--permit-real-output'):args.index('--permit-real-output')+2]
        with patch.object(g1_walk_pid, 'run_device', side_effect=AssertionError('hardware')), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(g1_walk_pid.main(args), 1)

    def test_zero_profile_copy_and_old_absolute_range_preserved(self):
        original = g1_walk_pid.load_profile(launch.PROFILE)
        old_target = original['target_q_array'].copy()
        effective = prepare_profile(original, CONFIG, launch.MODEL)
        np.testing.assert_array_equal(original['target_q_array'], old_target)
        np.testing.assert_array_equal(effective['target_q_array'][:10], np.zeros(10))
        for key in ('kp_array', 'kd_array'):
            np.testing.assert_array_equal(effective[key], original[key])
        bounds = effective['position_pid_reference_bounds_rad']
        np.testing.assert_allclose(bounds[0], np.minimum(old_target[5:10]-np.deg2rad(5), -np.deg2rad(5)))
        np.testing.assert_allclose(bounds[1], np.maximum(old_target[5:10]+np.deg2rad(5), np.deg2rad(5)))
        params, mapping = g1_walk_pid.load_pid_parameters(launch.MODEL, effective)
        self.assertEqual(params.hardware_max_dq, .07)
        self.assertEqual(params.hardware_max_ddq, .2)
        self.assertEqual(mapping['pid_ki_pose'], [0., 0.])

    def test_new_pid_preflight_cannot_reach_field_or_network(self):
        args = parser_args(launch.build_command('M03'), 'g1_walk_pid.py')
        with patch.object(socket, 'socket', side_effect=AssertionError('network')), \
                patch.object(g1_walk_pid, 'run_device', side_effect=AssertionError('hardware')), \
                patch('field_performance.prepare', side_effect=AssertionError('host mutation')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(g1_walk_pid.main(args), 0)

    def test_relaxed_trial_only_changes_two_governor_parameters(self):
        original = g1_walk_pid.load_profile(launch.PROFILE)
        a = prepare_profile(original, CONFIG, launch.MODEL)
        b = prepare_profile(original, RELAXED_CONFIG, launch.MODEL)
        pa, ma = g1_walk_pid.load_pid_parameters(launch.MODEL, a)
        pb, mb = g1_walk_pid.load_pid_parameters(launch.MODEL, b)
        changed = {k for k in ma if ma[k] != mb[k]}
        self.assertEqual(changed, {'hardware_pid_max_dq', 'hardware_pid_max_ddq'})
        self.assertEqual((pa.hardware_max_dq, pa.hardware_max_ddq), (.07, .2))
        self.assertEqual((pb.hardware_max_dq, pb.hardware_max_ddq), (.14, .4))
        for key in ('target_q_array', 'kp_array', 'kd_array', 'position_pid_reference_bounds_rad'):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertNotIn('position_pid_parameter_overrides', original)
        self.assertTrue(a['position_pid_reference_experiment']['reference_governor_unchanged'])
        self.assertFalse(b['position_pid_reference_experiment']['reference_governor_unchanged'])
        # Reusing the original entry after M07 must not retain relaxed settings.
        again, _ = g1_walk_pid.load_pid_parameters(launch.MODEL,
            prepare_profile(b, CONFIG, launch.MODEL))
        self.assertEqual((again.hardware_max_dq, again.hardware_max_ddq), (.07, .2))
        for execute in (False, True):
            ca = launch.build_command('M03', execute=execute, output_dir='/tmp/morning_contract_not_created')
            cb = launch.build_command('M07', execute=execute, output_dir='/tmp/morning_contract_not_created')
            i = cb.index('--position-pid-config')+1
            self.assertEqual(cb[i], str(RELAXED_CONFIG))
            cb[i] = ca[i]
            self.assertEqual(ca, cb)

    def test_relaxed_config_rejects_unreviewed_limits_and_cannot_skip_output_gate(self):
        profile = g1_walk_pid.load_profile(launch.PROFILE)
        for dq, ddq in ((.48, .4), (.14, 8.)):
            with patch('hardware_position_pid_zero.yaml.safe_load',
                       return_value=dict(EXPECTED, hardware_pid_max_dq=dq, hardware_pid_max_ddq=ddq)), \
                    self.assertRaises(ValueError):
                prepare_profile(profile, RELAXED_CONFIG, launch.MODEL)
        cmd = launch.build_command('M07', execute=True)
        args = parser_args(cmd, 'g1_walk_pid.py')
        del args[args.index('--permit-real-output'):args.index('--permit-real-output')+2]
        with patch.object(g1_walk_pid, 'run_device', side_effect=AssertionError('hardware')), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(g1_walk_pid.main(args), 1)

    def test_relaxed_governor_reversal_boundaries_and_stalls(self):
        from endpoint_pose import EndpointModel
        from hardware_pid_control import HardwarePidPlan
        from scipy.spatial.transform import Rotation
        profile = prepare_profile(g1_walk_pid.load_profile(launch.PROFILE), RELAXED_CONFIG, launch.MODEL)
        params, _ = g1_walk_pid.load_pid_parameters(launch.MODEL, profile)
        bounds = profile['position_pid_reference_bounds_rad']
        controller = RightArmHardwarePid(np.zeros(5), params, model=EndpointModel(launch.MODEL),
                                        reference_bounds_rad=bounds)
        slots = np.zeros(13); previous_q = None; previous_dq = np.zeros(5); peak = 0.
        for i in range(1800):
            dt = .018 if i % 27 == 0 else .006
            angle = 15. if i < 900 else -15.
            qxyzw = Rotation.from_euler('xy', [angle, angle], degrees=True).as_quat()
            q, dq, _ = controller.step(slots, qxyzw[[3, 0, 1, 2]], 0., dt)
            self.assertTrue(np.isfinite(np.r_[q, dq]).all())
            self.assertTrue(np.all(q >= bounds[0]-1e-10) and np.all(q <= bounds[1]+1e-10))
            self.assertLessEqual(np.max(abs(dq)), .14+1e-10)
            self.assertLessEqual(np.max(abs(dq-previous_dq)), .4*.006+1e-10)
            if previous_q is not None:
                np.testing.assert_allclose((q-previous_q)/.006, dq, atol=1e-10)
            peak = max(peak, np.max(abs(dq)))
            previous_q, previous_dq = q.copy(), dq.copy()
        self.assertGreater(peak, .13)  # Prove new limits actually reach the governor.
        plan = HardwarePidPlan(slots, slots, profile['kp_array'], profile['kd_array'], controller)
        for t in np.arange(0, 21.012, .006):
            frame = plan.sample(t, slots, np.zeros(13), [1., 0., 0., 0.], 0., .006)
            np.testing.assert_array_equal(frame['q_rad'][:5], np.zeros(5))
        self.assertTrue(frame['terminal'])
        self.assertEqual(frame['weight'], 0.)

    def test_relaxed_preflight_cannot_reach_network_or_host_setup(self):
        args = parser_args(launch.build_command('M07'), 'g1_walk_pid.py')
        with patch.object(socket, 'socket', side_effect=AssertionError('network')), \
                patch.object(g1_walk_pid, 'run_device', side_effect=AssertionError('hardware')), \
                patch('field_performance.prepare', side_effect=AssertionError('host mutation')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(g1_walk_pid.main(args), 0)

    def test_old_controller_default_matches_explicit_old_box(self):
        profile = g1_walk_pid.load_profile(launch.PROFILE)
        params, _ = g1_walk_pid.load_pid_parameters(launch.MODEL, profile)
        q0 = profile['target_q_array'][5:10]
        a = RightArmHardwarePid(q0, params)
        b = RightArmHardwarePid(q0, params, reference_bounds_rad=[q0-params.q_offset_limit_rad, q0+params.q_offset_limit_rad])
        for i in range(200):
            slots = profile['target_q_array'].copy()
            slots[5:10] += .02*np.sin(i*.03+np.arange(5))
            quat = [np.cos(.03), 0, np.sin(.03), 0]
            qa, da, _ = a.step(slots, quat, 0., .006)
            qb, db, _ = b.step(slots, quat, 0., .006)
            np.testing.assert_array_equal(qa, qb)
            np.testing.assert_array_equal(da, db)

    def test_hold_worker_matches_direct_and_retains_parent_release(self):
        from test_mpc_compute_process import ComputeProcessTests
        case = ComputeProcessTests()
        self.addCleanup(case.doCleanups)
        case.check_runtime_parity(dict(config=launch.MPC_CONFIG, model_config=launch.MODEL,
            torque_config=launch.TORQUE_CONFIG, predictor_mode='hold_current',
            stationary=False, field_trial=True, assumed_command_delay_s=None,
            zero_arm_neutral=True, left_pd_gain_scale=1., align_accel_forecast=False))


if __name__ == '__main__':
    unittest.main()
