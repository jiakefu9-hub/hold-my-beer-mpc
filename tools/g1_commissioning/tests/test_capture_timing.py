import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('timing', Path(__file__).parents[1] / 'analyze_capture_timing.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class TimingTest(unittest.TestCase):
    def records(self):
        event = lambda name, **kw: dict(schema='g1_capture_event_v1', event=name, **kw)
        start = 1_000_000_000
        result = [event('observation_start', epoch_ns=start, planned_end_ns=start+12_000_000),
                  dict(schema='g1_host_timing_environment_v1', stage='observation_ready',
                       control_cpu=7, main_affinity=[7], main_scheduler=0),
                  event('fsm_reply', monotonic_ns=start-1, request_ns=start-10, reply_ns=start-1, return_code=0),
                  event('fsm_reply', monotonic_ns=start+2_000_000, request_ns=start+1_000_000,
                        reply_ns=start+2_000_000, return_code=0),
                  event('phase_reply', monotonic_ns=start+9_000_000, request_ns=start+3_000_000,
                        reply_ns=start+9_000_000, return_code=7301),
                  event('observation_end', end_ns=start+12_000_000),
                  event('session_end', outcome='observation_completed', imu_continuously_fresh=True,
                        lowstate_continuously_healthy=True, crc_rejected=0, queue_dropped=0)]
        for i in range(2):
            now = start+i*6_000_000
            result += [dict(schema='g1_lowstate_raw_v1', received_monotonic_ns=now,
                            host_callback_sequence=i+1, tick_raw=101, crc_valid=True,
                            journal_enqueued_ns=now+20_000, journal_dequeued_ns=now+30_000,
                            journal_serialized_ns=now+40_000),
                       dict(schema='g1_host_probe_tick_v1', scheduled_ns=now, started_ns=now+50_000,
                            snapshot_done_ns=now+60_000, finished_ns=now+70_000,
                            state_received_ns=now, imu_received_ns=now, state_sequence=i+1,
                            imu_sequence=i+1, missed_slots_after=0, cpu=7, state_valid=True, imu_fresh=True)]
        return result

    def analyze(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            return timing.analyze(path)

    def test_success_window_and_failure_separation(self):
        s = self.analyze(self.records())
        self.assertTrue(s['capture_complete_and_healthy'])
        self.assertEqual(s['distributions']['fsm_success_rpc_rtt_ms']['mean_ms'], 1)
        self.assertEqual(s['distributions']['fsm_success_rpc_rtt_ms']['n'], 1)
        self.assertIsNone(s['distributions']['phase_success_rpc_rtt_ms']['p99_ms'])
        self.assertEqual(s['distributions']['phase_failed_call_duration_ms']['mean_ms'], 6)
        self.assertEqual(s['distributions']['lowstate_host_age_at_snapshot_ms']['mean_ms'], .06)
        self.assertEqual(s['probe']['missing_slots'], 0)
        self.assertEqual(s['tick_repeats'], 1)
        self.assertIn('one_way_dds_transport', s['not_measured'])
        self.assertIn('未取得', timing.report(s))

    def test_missing_end_is_not_completed(self):
        rows = [r for r in self.records() if r.get('event') != 'session_end']
        self.assertFalse(self.analyze(rows)['capture_complete_and_healthy'])

    def test_grid_skips_not_hidden(self):
        rows = self.records()
        rows = [r for r in rows if not (r.get('schema') == 'g1_host_probe_tick_v1' and r['scheduled_ns'] > 1_000_000_000)]
        self.assertEqual(self.analyze(rows)['probe']['missing_slots'], 1)

    def test_bad_clock_and_wrong_cpu(self):
        rows = self.records()
        sample = next(r for r in rows if r.get('schema') == 'g1_host_probe_tick_v1')
        sample['finished_ns'] = sample['started_ns']-1
        sample['cpu'] = 4
        s = self.analyze(rows)
        self.assertFalse(s['capture_complete_and_healthy'])
        self.assertEqual(s['probe']['wrong_cpu_samples'], 1)

    def test_truncated_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in self.records())+'{"partial":')
            with self.assertRaises(ValueError):
                timing.analyze(path)

    def test_percentiles(self):
        self.assertAlmostEqual(timing.describe([0, 10])['p99_ms'], 9.9)
        self.assertIsNone(timing.describe([])['mean_ms'])


if __name__ == '__main__':
    unittest.main()
