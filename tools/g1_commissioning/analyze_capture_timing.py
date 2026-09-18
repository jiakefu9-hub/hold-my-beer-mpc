#!/usr/bin/env python3
"""Offline host/RPC timing only. No DDS imports, device access or control output."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

PERIOD_NS = 6_000_000


def describe(values):
    """Milliseconds; empty distributions remain null, never a fabricated zero."""
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None,
                "p99_ms": None, "max_ms": None, "over_6ms": 0}

    def percentile(p):
        position = (len(ordered) - 1) * p
        lo = math.floor(position)
        hi = math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

    return {"n": len(ordered), "mean_ms": statistics.fmean(ordered),
            "p50_ms": percentile(.5), "p95_ms": percentile(.95),
            "p99_ms": percentile(.99), "max_ms": ordered[-1],
            "over_6ms": sum(x > 6 for x in ordered)}


def analyze(path):
    path = Path(path)
    environment, markers = [], {}
    errors = []
    digest = hashlib.sha256()
    # First pass selects the READY observation window. Startup/discovery traffic
    # is not mixed with steady-window timings. Keep raw file ordering irrelevant.
    with path.open('rb') as source:
        for number, line in enumerate(source, 1):
            digest.update(line)
            if b'g1_capture_event_v1' not in line and b'g1_host_timing_environment_v1' not in line:
                continue
            try:
                record = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'invalid JSON at line {number}: {exc}') from exc
            if record.get('schema') == 'g1_host_timing_environment_v1':
                environment.append(record)
            event = record.get('event')
            if event in ('observation_start', 'observation_end', 'session_end'):
                if event in markers:
                    raise ValueError('multiple sessions/windows in one raw log')
                markers[event] = record
            if event in ('session_error', 'fsm_exception', 'phase_exception'):
                errors.append(record)
    start_marker = markers.get('observation_start', {})
    start = start_marker.get('epoch_ns', start_marker.get('monotonic_ns'))
    if start is None:
        raise ValueError('no READY observation window; this analyzer expects g1_phase_probe')
    planned_end = start_marker.get('planned_end_ns', start + 30_000_000_000)
    observed_end = markers.get('observation_end', {}).get('end_ns')
    end = min(planned_end, observed_end) if observed_end is not None else planned_end
    metrics = defaultdict(list)
    replies = defaultdict(Counter)
    topics = defaultdict(list)
    ticks, tick_rows = [], []
    crc_bad = 0

    def duration(name, begin, finish):
        if not isinstance(begin, int) or not isinstance(finish, int) or finish < begin:
            errors.append({'invalid_timestamps': name, 'begin': begin, 'finish': finish})
            return
        metrics[name].append((finish - begin) / 1e6)

    with path.open() as source:
        for number, line in enumerate(source, 1):
            try:
                r = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'invalid JSON at line {number}: {exc}') from exc
            schema, event = r.get('schema'), r.get('event')
            at = r.get('received_monotonic_ns', r.get('started_ns', r.get('monotonic_ns', 0)))
            if not start <= at < end:
                continue
            if schema in ('g1_torso_imu_raw_v1', 'g1_lowstate_raw_v1'):
                topic = 'torso_imu' if schema == 'g1_torso_imu_raw_v1' else 'lowstate'
                topics[topic].append((at, r['host_callback_sequence']))
                if topic == 'lowstate':
                    ticks.append((at, r['host_callback_sequence'], r['tick_raw']))
                    crc_bad += not r['crc_valid']
                if 'journal_enqueued_ns' in r:
                    # Includes callback copies, inbox CRC/mutex work and enqueue
                    # waiting up to its timestamp; NOT NIC-to-callback transit.
                    duration(topic + '_callback_to_enqueue_ms', at, r['journal_enqueued_ns'])
                    duration(topic + '_logger_queue_wait_ms', r['journal_enqueued_ns'], r['journal_dequeued_ns'])
                    duration(topic + '_logger_serialize_ms', r['journal_dequeued_ns'], r['journal_serialized_ns'])
            if event in ('fsm_reply', 'phase_reply'):
                if not start <= r['request_ns'] <= r['reply_ns'] < end:
                    continue
                name = event.removesuffix('_reply')
                replies[name][str(r['return_code'])] += 1
                status = 'success_rpc_rtt' if r['return_code'] == 0 else 'failed_call_duration'
                duration(f'{name}_{status}_ms', r['request_ns'], r['reply_ns'])
                if name == 'phase' and r.get('return_code') == 0 and not r.get('parse_ok'):
                    replies[name]['success_but_invalid_payload'] += 1
            if schema == 'g1_host_probe_tick_v1':
                tick_rows.append(r)
                duration('probe_wakeup_lateness_ms', r['scheduled_ns'], r['started_ns'])
                duration('probe_snapshot_path_ms', r['started_ns'], r['snapshot_done_ns'])
                duration('probe_full_body_ms', r['started_ns'], r['finished_ns'])
                duration('probe_release_to_finish_ms', r['scheduled_ns'], r['finished_ns'])
                for topic, key in (('lowstate', 'state_received_ns'), ('torso_imu', 'imu_received_ns')):
                    if r[key]:
                        duration(topic + '_host_age_at_snapshot_ms', r[key], r['snapshot_done_ns'])

    topic_report = {}
    for topic, rows in topics.items():
        rows.sort()
        times = [row[0] for row in rows]
        gaps = [(b-a)/1e6 for a, b in zip(times, times[1:])]
        topic_report[topic] = {
            'callbacks': len(rows), 'interarrival_ms': describe(gaps),
            'observed_rate_hz': (len(rows)-1)*1e9/(times[-1]-times[0]) if len(rows)>1 and times[-1]>times[0] else None,
            'host_sequence_gaps': sum(max(0, b[1]-a[1]-1) for a,b in zip(rows,rows[1:])),
            'source_network_loss_known': False}
    tick_rows.sort(key=lambda r: r['started_ns'])
    starts = [r['started_ns'] for r in tick_rows]
    metrics['probe_start_interval_ms'] = [(b-a)/1e6 for a,b in zip(starts,starts[1:])]
    metrics['probe_abs_period_error_ms'] = [abs(x-6) for x in metrics['probe_start_interval_ms']]
    ticks.sort()
    deltas = [((b[2]-a[2]) & 0xffffffff) for a,b in zip(ticks,ticks[1:])]
    # Do NOT subtract robot ticks from the host clock. No offset/drift model is
    # claimed and duplicate ticks are valid for this firmware.
    ready_env = next((e for e in environment if e['stage'] == 'observation_ready'), {})
    cpu = ready_env.get('control_cpu')
    final = markers.get('session_end', {})
    complete = (final.get('outcome') == 'observation_completed' and observed_end is not None and
                observed_end >= planned_end and final.get('imu_continuously_fresh') is True and
                final.get('lowstate_continuously_healthy') is True and final.get('crc_rejected', 0) == 0)
    missing_timing = not tick_rows or not ready_env
    if missing_timing:
        errors.append({'timing_instrumentation_missing': True})
    if ready_env and (ready_env.get('main_scheduler') != 0 or ready_env.get('main_affinity') != [cpu]):
        errors.append({'ordinary_pinned_environment_mismatch': True})
    if any(r['cpu'] != cpu for r in tick_rows):
        errors.append({'control_cpu_mismatch': True})
    if final.get('queue_dropped', 0) != 0:
        errors.append({'logger_queue_dropped': final['queue_dropped']})
    unique_slots = {r['scheduled_ns'] for r in tick_rows}
    expected_slots = max(0, (planned_end-start + PERIOD_NS-1)//PERIOD_NS)
    required = ['fsm_success_rpc_rtt_ms', 'fsm_failed_call_duration_ms',
                'phase_success_rpc_rtt_ms', 'phase_failed_call_duration_ms',
                'probe_wakeup_lateness_ms', 'probe_full_body_ms', 'probe_release_to_finish_ms']
    for key in required:
        metrics[key]  # preserve missing distributions explicitly
    return {
        'schema': 'g1_host_rpc_timing_summary_v1', 'raw_sha256': digest.hexdigest(),
        'window_ns': [start, end], 'planned_period_ms': 6,
        'capture_complete_and_healthy': complete and not errors,
        'environment': environment, 'errors': errors,
        'rpc_return_codes': {k: dict(v) for k,v in replies.items()},
        'distributions': {k: describe(v) for k,v in sorted(metrics.items())},
        'topics': topic_report, 'bad_lowstate_crc': crc_bad,
        'tick_repeats': sum(d == 0 for d in deltas),
        'tick_rollback_or_ambiguous': sum(d >= 2**31 for d in deltas),
        'probe': {'expected_slots': expected_slots, 'recorded_slots': len(tick_rows),
                  'missing_slots': max(0, expected_slots-len(unique_slots)),
                  'skipped_slots_reported': sum(r['missed_slots_after'] for r in tick_rows),
                  'wrong_cpu_samples': sum(r['cpu'] != cpu for r in tick_rows),
                  'invalid_state_samples': sum(not r['state_valid'] for r in tick_rows),
                  'stale_imu_samples': sum(not r['imu_fresh'] for r in tick_rows)},
        'not_measured': ['one_way_dds_transport', 'sensor_to_host_absolute_age',
                         'command_to_firmware_acceptance', 'command_to_motor_response',
                         'hardware_pid_mpc_full_cycle'],
        'interpretation': 'RPC RTT includes host/robot scheduling and service work. Do not halve it or add percentiles to simulation p99.'}


def report(summary):
    lines = ['# G1 只读通信／主机计时报告', '',
             '这是实测日志汇总，不是单向 DDS 或电机响应延迟，也不是 PID/MPC 真机周期验收。', '',
             f"完整且状态健康：{summary['capture_complete_and_healthy']}；原始 SHA-256：`{summary['raw_sha256']}`。", '',
             '| 指标（ms） | n | mean | p95 | p99 | max | >6 ms |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def fmt(value):
        return '未取得' if value is None else f'{value:.6f}'
    for name, d in summary['distributions'].items():
        lines.append(f"| {name} | {d['n']} | {fmt(d['mean_ms'])} | {fmt(d['p95_ms'])} | {fmt(d['p99_ms'])} | {fmt(d['max_ms'])} | {d['over_6ms']} |")
    lines += ['', '`>6 ms` 对 RPC/状态年龄只是参考计数，只有 probe 的 release-to-finish 对应其 6 ms 调度期限。', '',
              '## 环境和数据质量', '', '```json',
              json.dumps({k: summary[k] for k in ('environment', 'rpc_return_codes', 'topics', 'probe',
                         'bad_lowstate_crc', 'tick_repeats', 'tick_rollback_or_ambiguous', 'errors')}, ensure_ascii=False, indent=2),
              '```', '', '未测项目：' + '、'.join(summary['not_measured']) + '。', '',
              'GetPhase 失败与超时单列，不作为成功往返；各项 p99 不可直接与仿真 p99 相加。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    summary = analyze(args.raw)
    args.output_dir.mkdir(parents=False, exist_ok=False)
    (args.output_dir / 'timing_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'timing_report.md').write_text(report(summary))
    print(args.output_dir / 'timing_report.md')
    return 0 if summary['capture_complete_and_healthy'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
