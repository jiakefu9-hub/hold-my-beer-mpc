"""Saved-data comparisons must not turn failed prefixes into complete runs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_torque_robustness import load_run, make_report, markdown


def fixture(directory, duration=.012, complete=True):
    directory.mkdir()
    interval = np.arange(round(duration/.002))*.002
    controls = interval[::3]
    actual = np.ones((len(interval), 5))
    actual[0] = 9.
    seq = np.arange(len(interval))
    seq[0] = -1
    arrays = dict(t=controls, core_ms=np.arange(len(controls))+1., tilt=np.zeros(len(controls)),
        physics_t=np.r_[interval, duration], physics_q=np.zeros((len(interval)+1, 5)),
        physics_dq=np.zeros((len(interval)+1, 5)),
        physics_q_outer_margin=np.full((len(interval)+1, 5), .05),
        interval_t=interval, interval_actual=actual,
        interval_active_seq=seq, interval_active_desired=np.zeros_like(actual),
        interval_actually_active_command_desired_vs_actual=actual.copy())
    np.savez_compressed(directory/'combined.npz', **arrays)
    summary = dict(status='complete' if complete else 'failed',
        failure=None if complete else 'synthetic failure',
        tracking_2ms=dict(actually_active_command_desired_vs_actual_rmse=1.,
                         active_command_samples=len(interval)-1),
        max_abs_qacc_at_2ms=9., minimum_margins_2ms=dict(q_outer_margin=.05),
        core_ms=dict(p99=float(np.percentile(arrays['core_ms'], 99))),
        physical_state_scored_until_s=duration, metrics_cover_requested_duration=complete)
    study = dict(schema='g1_torque_robustness_study_v1',
        duration_s=duration if complete else duration+.01,
        source_unchanged_during_run=True, runs=dict(combined=summary))
    (directory/'summary.json').write_text(json.dumps(study))
    return study


class RobustnessReportTest(unittest.TestCase):
    def test_recompute_excludes_initial_hold_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'complete'
            fixture(directory)
            run = load_run('complete', directory, 'combined')
            report = make_report([run], 'combined')
            metrics = report['runs']['complete']['full_duration_metrics']
            self.assertEqual(metrics['active_command_rmse_rad_s2'], 1.)
            self.assertEqual(metrics['initial_hold_intervals'], 1)
            self.assertEqual(len(run.input_sha256), 2)

    def test_failed_prefix_common_window_and_explicit_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root/'complete')
            fixture(root/'failed', .008, complete=False)
            runs = [load_run(name, root/name, 'combined') for name in ('complete', 'failed')]
            report = make_report(runs, 'combined', .01)
            self.assertIsNone(report['runs']['failed']['full_duration_metrics'])
            self.assertEqual(report['common_prefix_all']['until_s'], .008)
            self.assertEqual(set(report['common_prefix_all']['runs']), {'complete', 'failed'})
            self.assertEqual(set(report['requested_window']['runs']), {'complete'})
            self.assertIn('failed', report['requested_window']['excluded'])
            self.assertIn('FAILED PREFIX ONLY', markdown(report))

    def test_summary_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'tampered'
            study = fixture(directory)
            study['runs']['combined']['tracking_2ms']['actually_active_command_desired_vs_actual_rmse'] = .5
            (directory/'summary.json').write_text(json.dumps(study))
            with self.assertRaisesRegex(ValueError, 'summary/NPZ mismatch'):
                load_run('tampered', directory, 'combined')

    def test_duplicate_label_and_nonpositive_window_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'complete'
            fixture(directory)
            run = load_run('same', directory, 'combined')
            with self.assertRaisesRegex(ValueError, 'unique labels'):
                make_report([run, run], 'combined')
            with self.assertRaisesRegex(ValueError, 'positive and finite'):
                make_report([run], 'combined', 0.)


if __name__ == '__main__':
    unittest.main()
