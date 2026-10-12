"""Numbered field preparation: frozen candidates, admission and no implicit output."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

import g1_run_control_comparison as launcher
import g1_walk_mpc
from hardware_mpc_control import load_mpc_config
from hardware_control_comparison import (
    ROOT, BASELINE_CONFIG, MODEL_CONFIG, TORQUE_CONFIG, TRIALS,
    CANDIDATE_CONFIGS, ZERO_NEUTRAL_MPC_CONFIGS, validate_trial_config,
    validate_comparison_torque, ORIGINAL_TORQUE_CONFIG, ORIGINAL_MPC_CONFIG,
)


BASELINE_OVERLAY = {
    'schema': 'g1_hardware_mpc_config_v1',
    'q_ee_acc': .01,
    'q_ee_alpha': .0005,
    'q_ee_omega': 1.,
    'mpc_start_s': 3.3,
    'mpc_handoff_duration_s': 1.5,
    'q_gravity': [30., 30.],
    'q_posture': [4., 4., 1.5, .2, .2],
    'q_vel': .08,
    'r_ddq': .0025,
    'terminal_scale': 2.,
}
OVERRIDES = {'T06': {'q_ee_acc': .02}}
PARENTS = {'T01': None, 'T02': 'T01', 'T03': 'T02',
           'T04': 'T02', 'T05': 'T03', 'T06': 'T02'}


def legacy_args(command):
    """Parse the actual downstream runner arguments, regardless of shell prefix."""
    entry = next(i for i, item in enumerate(command)
                 if Path(item).name == 'g1_walk_mpc_learned.py')
    return g1_walk_mpc.build_parser(learned=True).parse_args(command[entry + 1:])


class TrialSeriesTests(unittest.TestCase):
    def test_frozen_baseline_and_candidate_changes_are_explicit(self):
        self.assertEqual(yaml.safe_load(BASELINE_CONFIG.read_text()), BASELINE_OVERLAY)
        self.assertEqual(set(TRIALS), set(PARENTS))
        candidates = set()
        for number, trial in TRIALS.items():
            with self.subTest(trial=number):
                self.assertTrue(trial['name'])
                self.assertEqual(trial['aligned'], number != 'T01')
                expected = dict(BASELINE_OVERLAY, **OVERRIDES.get(number, {}))
                self.assertEqual(yaml.safe_load(trial['config'].read_text()), expected)
                validate_trial_config(trial['config'])
                if number in OVERRIDES:
                    candidates.add(trial['config'].resolve())
                else:
                    self.assertEqual(trial['config'].resolve(), BASELINE_CONFIG.resolve())
        self.assertEqual(set(CANDIDATE_CONFIGS), candidates | {BASELINE_CONFIG.resolve()})
        self.assertEqual(set(ZERO_NEUTRAL_MPC_CONFIGS),
                         candidates | {BASELINE_CONFIG.resolve()})

    def test_each_named_comparison_changes_only_one_factor(self):
        factors = {'T06': 'q_ee_acc'}
        for number, factor in factors.items():
            with self.subTest(trial=number):
                child = load_mpc_config(TRIALS[number]['config'])
                parent = load_mpc_config(TRIALS[PARENTS[number]]['config'])
                self.assertEqual({key for key in child if child[key] != parent[key]}, {factor})
        self.assertEqual(load_mpc_config(TRIALS['T01']['config']),
                         load_mpc_config(TRIALS['T02']['config']))
        self.assertEqual(load_mpc_config(TRIALS['T01']['config']),
                         load_mpc_config(TRIALS['T01']['config']))

    def test_effective_configs_keep_frequency_solver_and_no_velocity_cost(self):
        baseline = load_mpc_config(BASELINE_CONFIG)
        for config in CANDIDATE_CONFIGS:
            with self.subTest(config=config.name):
                current = load_mpc_config(config)
                self.assertEqual(current['control_period_s'], .006)
                self.assertEqual(current['horizon'], 9)
                self.assertEqual(current['q_ee_vel'], 0.)
                self.assertEqual(current['q_ee_omega'], 1.)
                for key in baseline:
                    if (key.startswith('solver_') or key.startswith('reference_')
                            or key in ('regularization', 'q_vel', 'r_ddq', 'terminal_scale')):
                        self.assertEqual(current[key], baseline[key], key)

    def test_modified_candidate_is_rejected_without_modifying_files(self):
        config = TRIALS['T06']['config'].resolve()
        original = yaml.safe_load(config.read_text())
        real_read = Path.read_text
        for changed in (dict(original, q_ee_omega=2.),
                        dict(original, unexpected_extra_parameter=1)):
            def read(path, *args, **kwargs):
                return yaml.safe_dump(changed) if path.resolve() == config else real_read(path, *args, **kwargs)
            with self.subTest(changed=changed), mock.patch.object(Path, 'read_text', read):
                with self.assertRaises(ValueError):
                    validate_trial_config(config)
                with self.assertRaises(ValueError):
                    g1_walk_mpc.validate_accel_forecast_alignment(
                        config, TORQUE_CONFIG, 'learned_filtered', 'measured_torque_preview', None)

    def test_common_torque_matches_successful_baseline(self):
        self.assertEqual(TORQUE_CONFIG.resolve(), ORIGINAL_TORQUE_CONFIG.resolve())
        original = yaml.safe_load(ORIGINAL_TORQUE_CONFIG.read_text())
        current = yaml.safe_load(TORQUE_CONFIG.read_text())
        self.assertEqual(current, original)
        validate_comparison_torque(TORQUE_CONFIG)
        real_read = Path.read_bytes
        def read(path, *args, **kwargs):
            return b'tampered' if path.resolve() == TORQUE_CONFIG else real_read(path, *args, **kwargs)
        with mock.patch.object(Path, 'read_bytes', read):
            with self.assertRaises(ValueError):
                launcher.build_command('T03')
            with self.assertRaises(ValueError):
                g1_walk_mpc.validate_accel_forecast_alignment(
                    BASELINE_CONFIG, TORQUE_CONFIG, 'learned_filtered', 'measured_torque_preview', None)

    def test_candidate_validation_does_not_retire_other_existing_configs(self):
        # This helper owns only the new series; legacy admission owns other paths.
        validate_trial_config(ROOT / 'configs/hardware_mpc_learned.yaml')
        validate_comparison_torque({'legacy_offline_config': True})

    def test_alignment_accepts_candidates_and_rejects_unreviewed_combinations(self):
        for config in CANDIDATE_CONFIGS:
            with self.subTest(config=config.name):
                g1_walk_mpc.validate_accel_forecast_alignment(
                    config, TORQUE_CONFIG, 'learned_filtered', 'measured_torque_preview', None)
        accepted = [TRIALS['T06']['config'], TORQUE_CONFIG,
                    'learned_filtered', 'measured_torque_preview', None]
        for index, value in ((0, ROOT / 'configs/unreviewed_field_candidate.yaml'),
                             (1, g1_walk_mpc.FIELD_TORQUE_CONFIG), (1, None),
                             (2, 'hold_current'), (3, 'reference_servo'),
                             (4, 0.), (4, 6.)):
            changed = accepted.copy()
            changed[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                g1_walk_mpc.validate_accel_forecast_alignment(*changed)

    def test_zero_neutral_and_alignment_admission_reaches_only_offline_preflight(self):
        for config in CANDIDATE_CONFIGS:
            args = ['--preflight', '--zero-arm-neutral', '--align-accel-forecast',
                    '--mpc-config', str(config), '--torque-config', str(TORQUE_CONFIG),
                    '--controller-config', str(MODEL_CONFIG)]
            with self.subTest(config=config.name), \
                    mock.patch.object(g1_walk_mpc, 'select_cpu'), \
                    mock.patch.object(g1_walk_mpc, 'preflight', return_value={'passed': True}) as check, \
                    mock.patch.object(g1_walk_mpc, 'run_device', side_effect=AssertionError('hardware')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(g1_walk_mpc.main(args, learned=True), 0)
                self.assertTrue(check.call_args.kwargs['zero_arm_neutral'])
                self.assertTrue(check.call_args.kwargs['align_accel_forecast'])

    def test_real_output_gates_still_reject_incomplete_execution_requests(self):
        base = ['--execute', '--task', 'walk', '--zero-arm-neutral', '--align-accel-forecast',
                '--mpc-config', str(TRIALS['T06']['config']), '--torque-config', str(TORQUE_CONFIG)]
        extra_cases = ([], ['--allow-first-torque-field-trial'],
                       ['--allow-first-torque-field-trial', '--torque-stationary-validated'])
        for extra in extra_cases:
            with self.subTest(extra=extra), \
                    mock.patch.object(g1_walk_mpc, 'select_cpu'), \
                    mock.patch.object(g1_walk_mpc, 'preflight', side_effect=AssertionError('preflight')) as check, \
                    mock.patch.object(g1_walk_mpc, 'run_device', side_effect=AssertionError('hardware')) as device, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(g1_walk_mpc.main(base + extra, learned=True), 1)
                check.assert_not_called()
                device.assert_not_called()

    def test_zero_neutral_admission_stays_bounded_when_alignment_is_off(self):
        base = ['--preflight', '--zero-arm-neutral',
                '--mpc-config', str(TRIALS['T06']['config']),
                '--torque-config', str(TORQUE_CONFIG)]
        replacements = (
            ['--mpc-config', str(ROOT / 'configs/unreviewed_field_candidate.yaml')],
            ['--torque-config', str(g1_walk_mpc.FIELD_TORQUE_CONFIG)],
            ['--assumed-command-delay-ms', '0'],
        )
        for extra in replacements:
            with self.subTest(extra=extra), \
                    mock.patch.object(g1_walk_mpc, 'select_cpu'), \
                    mock.patch.object(g1_walk_mpc, 'preflight', side_effect=AssertionError('admitted')) as check, \
                    mock.patch.object(g1_walk_mpc, 'run_device', side_effect=AssertionError('hardware')) as device, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(g1_walk_mpc.main(base + extra, learned=True), 1)
                check.assert_not_called()
                device.assert_not_called()

    def test_launcher_pins_shared_payload_zero_neutral_and_unmodified_left_pd(self):
        self.assertEqual(MODEL_CONFIG.resolve(), (ROOT / 'configs/g1_payload75.yaml').resolve())
        for number, trial in TRIALS.items():
            for execute in (False, True):
                with self.subTest(trial=number, execute=execute):
                    command = launcher.build_command(number, execute=execute)
                    parsed = legacy_args(command)
                    self.assertEqual(parsed.execute, execute)
                    self.assertEqual(parsed.preflight, not execute)
                    self.assertEqual(parsed.mpc_config.resolve(), trial['config'].resolve())
                    self.assertEqual(parsed.controller_config.resolve(), MODEL_CONFIG.resolve())
                    self.assertEqual(parsed.torque_config.resolve(), TORQUE_CONFIG.resolve())
                    self.assertTrue(parsed.zero_arm_neutral)
                    self.assertEqual(parsed.left_pd_gain_scale, 1.)
                    self.assertEqual(parsed.align_accel_forecast, trial['aligned'])
                    self.assertEqual(parsed.predictor, 'learned_filtered')
                    self.assertIsNone(parsed.assumed_command_delay_ms)
                    self.assertEqual(parsed.task, 'walk')
                    self.assertEqual(parsed.cpu, 7)
                    if execute:
                        self.assertTrue(parsed.compute_process)
                        self.assertEqual(parsed.rt_priority, 20)
                        self.assertEqual(parsed.nic, 'enx6c1ff701509c')
                        self.assertTrue(parsed.pid_6ms_validated)
                        self.assertTrue(parsed.torque_stationary_validated)
                        self.assertTrue(parsed.allow_first_torque_field_trial)
                        self.assertEqual(parsed.permit_real_output, g1_walk_mpc.PERMIT)
                        self.assertIsNotNone(parsed.profile)
                        self.assertIsNotNone(parsed.output_dir)

    def test_launcher_output_paths_are_new_unique_and_never_created_by_building(self):
        first = legacy_args(launcher.build_command('T06', execute=True)).output_dir
        second = legacy_args(launcher.build_command('T06', execute=True)).output_dir
        self.assertNotEqual(first, second)
        self.assertFalse(first.exists())
        self.assertFalse(second.exists())
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory)
            with self.assertRaises((ValueError, FileExistsError)):
                launcher.build_command('T06', execute=True, output_dir=existing)
            future = existing / 'new_capture'
            parsed = legacy_args(launcher.build_command('T06', execute=True, output_dir=future))
            self.assertEqual(parsed.output_dir.resolve(), future.resolve())
            self.assertFalse(future.exists())

    def test_default_launcher_modes_print_without_executing_anything(self):
        for args in ([], ['T06']):
            with self.subTest(args=args), \
                    mock.patch('os.execv', side_effect=AssertionError('implicit launch')), \
                    mock.patch('subprocess.run', side_effect=AssertionError('implicit launch')), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(launcher.main(args), 0)
                self.assertTrue(output.getvalue().strip())

    def test_spawned_candidate_matches_direct_and_retains_parent_release(self):
        from test_mpc_compute_process import ComputeProcessTests
        case = ComputeProcessTests()
        self.addCleanup(case.doCleanups)
        case.check_runtime_parity(dict(
            config=TRIALS['T06']['config'], model_config=MODEL_CONFIG,
            torque_config=TORQUE_CONFIG, predictor_mode='learned_filtered',
            stationary=False, field_trial=True, assumed_command_delay_s=None,
            zero_arm_neutral=True, left_pd_gain_scale=1., align_accel_forecast=True))


if __name__ == '__main__':
    unittest.main()
