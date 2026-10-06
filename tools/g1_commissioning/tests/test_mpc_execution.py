"""Synthetic evidence checks only; no DDS participant or robot output."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_mpc_execution import analyze, read_capture, write_outputs, error_stats
from analyze_mpc_field_trial import review
from arm_execution_record import command_evidence
from g1_walk_mpc import main, MpcRuntime
from hardware_pid_control import ARM_MOTOR_INDICES


def fixture():
    epoch = 1_000_000_000
    rows = [dict(event='task_epoch', task_epoch_monotonic_ns=epoch)]
    for n in range(30):
        t = 5.+n*.006
        q, dq = .4*(t-5)**2, .8*(t-5)
        stamp = epoch+round(t*1e9)
        rows.append(dict(schema='g1_hardware_mpc_command_v1', event='dds_write',
            write_end_monotonic_ns=stamp, weight=1., stage='stationary_hold',
            q_command_rad=[.02]*13, dq_command_rad_s=[0.]*13,
            kp_command=[20.]*13, kd_command=[1.]*13, tau_ff=[1.]*13,
            raw_mpc_ddq_rad_s2=[.8]*5, post_transition_ddq_rad_s2=[.75]*5,
            tau_total_estimated_at_feedback_nm=[1.4]*5, mpc_active=True))
        rows.append(dict(schema='g1_lowstate_raw_v1', received_monotonic_ns=stamp,
            crc_valid=True, motors=[dict(index=i, q_rad=q, dq_rad_s=dq,
                tau_est_nm=1.+20*(.02-q)-dq+.125, ddq_raw_rad_s2=0.) for i in range(22,27)]))
    return rows


class ExecutionTest(unittest.TestCase):
    def test_constant_acceleration_request_is_not_hidden_by_noise_statistic(self):
        stats=error_stats(np.zeros((20,5)),np.full((20,5),2.))
        np.testing.assert_allclose(stats['bias'],-2.)
        np.testing.assert_allclose(stats['rmse'],2.)
        np.testing.assert_allclose(stats['centered_rmse'],0.)
        np.testing.assert_allclose(stats['actual_mean'],0.)
        np.testing.assert_allclose(stats['expected_mean'],2.)
        self.assertIsNone(error_stats([],[]))

    def analyze_rows(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'raw.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            return analyze(path, **kwargs)

    def test_packet_not_candidate_and_motor_mapping(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        packet = unitree_hg_msg_dds__LowCmd_()
        for i in ARM_MOTOR_INDICES:
            packet.motor_cmd[i].tau = i+.123456789
        low = SimpleNamespace(crc_valid=True, tau_est=np.arange(35), ddq_raw=np.zeros(35))
        recorded = command_evidence(packet, low)
        wire = type(packet).deserialize(packet.serialize())
        np.testing.assert_array_equal(recorded['tau_ff'],
            [wire.motor_cmd[i].tau for i in ARM_MOTOR_INDICES])
        self.assertEqual(recorded['tau_est_at_feedback_nm'][5:10], list(range(22,27)))
        self.assertTrue(recorded['feedback_precedes_this_write'])

    def test_shared_runtime_builder_preserves_nonzero_torque_and_single_pd(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        runtime = object.__new__(MpcRuntime)
        runtime.actuation = 'measured_torque_preview'
        frame = dict(q_rad=np.zeros(13), dq_rad_s=np.zeros(13), kp=np.full(13,20.),
            kd=np.ones(13), weight=1., diagnostics=dict(controller_kind=runtime.actuation,
                expected_kp=[20.]*5, expected_kd=[1.]*5, tau_ff_candidate_nm=[1.25]*5))
        state = SimpleNamespace(mode_pr=0, mode_machine=4)
        packet = runtime.make_message(frame, state, unitree_hg_msg_dds__LowCmd_, CRC())
        self.assertEqual([packet.motor_cmd[i].tau for i in range(22,27)], [1.25]*5)
        self.assertEqual([packet.motor_cmd[i].kp for i in range(22,27)], [20.]*5)
        self.assertEqual(packet.crc, CRC().Crc(packet))
        frame['weight'] = 0.
        packet = runtime.make_message(frame, state, unitree_hg_msg_dds__LowCmd_, CRC())
        self.assertEqual([packet.motor_cmd[i].tau for i in range(22,27)], [0.]*5)

    def test_known_torque_error_and_acceleration_without_raw_ddq(self):
        data, summary = self.analyze_rows(fixture())
        window = summary['windows']['primary_full_task']
        np.testing.assert_allclose(window['torque_est_minus_feedback_reconstructed_total_nm']['rmse'], .125)
        np.testing.assert_allclose(window['measured_acc_minus_mpc_desired_rad_s2']['rmse'], 0., atol=1e-12)
        np.testing.assert_allclose(window['measured_acc_minus_forward_model_rad_s2']['bias'], .05, atol=1e-12)
        self.assertFalse(summary['absolute_torque_calibrated'])
        self.assertIsNone(summary['hardware_performance_passed'])
        self.assertGreater(len(data['acceleration_t']), 10)

    def test_delay_shift_cannot_erase_constant_acceleration_bias(self):
        rows = fixture()
        for row in rows:
            if row.get('event') == 'dds_write':
                row['raw_mpc_ddq_rad_s2'] = [2.]*5
        for delay in (0., .003, .006, .012):
            with self.subTest(delay=delay):
                _, summary = self.analyze_rows(rows, assumed_delay_s=delay)
                stats = summary['windows']['primary_full_task']['measured_acc_minus_mpc_desired_rad_s2']
                np.testing.assert_allclose(stats['actual_mean'], .8, atol=1e-12)
                np.testing.assert_allclose(stats['expected_mean'], 2., atol=1e-12)
                np.testing.assert_allclose(stats['bias'], -1.2, atol=1e-12)

    def test_field_review_rejects_mixed_capture_and_missing_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, endpoint, out = root/'raw.jsonl', root/'endpoint', root/'review'
            rows = [dict(event='session_start', task='stationary', primary_metric_window_s=[5.,18.])]+fixture()
            for row in rows:
                if row.get('event') == 'dds_write':
                    row['task_elapsed_s'] = (row['write_end_monotonic_ns']-1_000_000_000)*1e-9
            raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            endpoint.mkdir()
            (endpoint/'summary.json').write_text(json.dumps(dict(source_sha256='different')))
            with self.assertRaisesRegex(ValueError, 'different raw capture'):
                review(raw, endpoint, out)
            self.assertFalse(out.exists())
            (endpoint/'summary.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())))
            np.savez(endpoint/'metrics.npz', task_elapsed_s=[5.], left_tilt_deg=[1.], right_tilt_deg=[1.])
            with self.assertRaisesRegex(ValueError, 'missing active commands or full-cycle timing'):
                review(raw, endpoint, out)
            self.assertFalse(out.exists())

    def test_field_review_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with self.assertRaises(FileExistsError):
                review(out/'missing.jsonl', out/'missing_endpoint', out)

    def test_enqueue_order_does_not_change_result(self):
        rows = fixture()
        expected, _ = self.analyze_rows(rows)
        reordered = [rows[0]]+rows[1::2][::-1]+rows[2::2]
        actual, _ = self.analyze_rows(reordered)
        np.testing.assert_array_equal(actual['torque_expected'], expected['torque_expected'])

    def test_missing_tau_is_not_assumed_zero_or_candidate(self):
        rows = fixture()
        for row in rows:
            if row.get('event') == 'dds_write':
                del row['tau_ff']
                row['tau_ff_candidate_nm'] = [1.]*5
        with self.assertRaisesRegex(ValueError, 'explicit tau_ff'):
            self.analyze_rows(rows)

    def test_unknown_intervening_command_is_not_bridged(self):
        rows = fixture()
        del rows[9]['tau_ff']
        with self.assertRaisesRegex(ValueError, 'unknown intervening output'):
            self.analyze_rows(rows)

    def test_offline_packets_cannot_be_reported_as_hardware_response(self):
        rows = fixture()
        for row in rows:
            if row.get('event') == 'dds_write':
                row.pop('event')
                row['schema'] = 'g1_mpc_offline_command_v1'
        with self.assertRaisesRegex(ValueError, 'offline replay'):
            self.analyze_rows(rows)

    def test_partial_weight_future_stale_and_invalid_crc_are_not_success(self):
        for kind in ('weight', 'future', 'stale', 'crc'):
            rows = fixture()
            for row in rows:
                if row.get('event') == 'dds_write':
                    if kind == 'weight': row['weight'] = .5
                    if kind == 'future': row['write_end_monotonic_ns'] += 1_000_000_000
                    if kind == 'stale': row['write_end_monotonic_ns'] -= 1_000_000_000
                elif row.get('schema') == 'g1_lowstate_raw_v1' and kind == 'crc':
                    row['crc_valid'] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.analyze_rows(rows)

    def test_repeated_snapshot_does_not_fabricate_feedback_rate(self):
        rows = fixture()
        low = rows[2]
        for row in rows:
            if row.get('event') == 'dds_write':
                row.update(feedback_crc_valid=True,
                    state_received_monotonic_ns=low['received_monotonic_ns'],
                    q_measured_rad=[0.]*13, dq_measured_rad_s=[0.]*13,
                    tau_est_at_feedback_nm=[1.525]*13)
        _, summary = self.analyze_rows(rows)
        self.assertEqual(summary['audit']['unique_feedback_samples'], 30)
        self.assertEqual(summary['audit']['duplicate_feedback'], 30)

    def test_output_files_and_no_overwrite(self):
        data, summary = self.analyze_rows(fixture())
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'report'
            write_outputs(data, summary, out)
            self.assertEqual({p.name for p in out.iterdir()},
                {'execution.npz', 'summary.json', 'torque.csv', 'acceleration.csv', 'execution.png'})
            with self.assertRaises(FileExistsError): write_outputs(data, summary, out)

    def test_legacy_execute_rejected_before_runtime_or_dds(self):
        with mock.patch('g1_walk_mpc.select_cpu'), mock.patch('g1_walk_mpc.MpcRuntime') as runtime:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['--execute', '--actuation', 'reference_servo']), 1)
            runtime.assert_not_called()


if __name__ == '__main__':
    unittest.main()
