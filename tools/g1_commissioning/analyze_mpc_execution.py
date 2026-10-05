#!/usr/bin/env python3
"""Offline torque/acceleration response audit of successful real command logs.

No DDS imports, no command output, no compensation fit. tau_est is NOT ground
truth. Host receive/write times do not identify motor application timestamps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RIGHT = slice(5, 10)
MOTOR_IDS = list(range(22, 27))


def vector(value, size):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"expected {size} finite values")
    return array


def read_capture(path):
    """Sort by host stamps, not thread-dependent JSONL enqueue order."""
    commands, states, epoch, session, session_end, drain = [], {}, None, {}, {}, {}
    counts = dict(malformed_commands=0, invalid_states=0, failed_writes=0,
                  offline_commands_ignored=0, duplicate_feedback=0)
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for line in stream:
            digest.update(line)
            if not any(key in line for key in
                       (b'dds_write', b'g1_lowstate_raw_v1', b'task_epoch',
                        b'session_start', b'session_end', b'capture_drained',
                        b'g1_mpc_offline_command_v1')):
                continue
            row = json.loads(line)
            if row.get('event') == 'task_epoch':
                if epoch is not None:
                    raise ValueError('multiple task epochs; analyze each capture separately')
                epoch = int(row['task_epoch_monotonic_ns'])
            if row.get('event') == 'session_start':
                session = row
            if row.get('event') == 'session_end':
                session_end = row
            if row.get('event') == 'capture_drained':
                drain = row
            if row.get('schema') == 'g1_mpc_offline_command_v1':
                counts['offline_commands_ignored'] += 1
            if row.get('event') == 'dds_write_failed':
                counts['failed_writes'] += 1
            if row.get('event') == 'dds_write':
                try:
                    # An explicit packet feedforward is mandatory. Never use a
                    # model candidate or infer tau=0 from a historical schema.
                    ff = vector(row['tau_ff'], 13)[RIGHT]
                    fields = [vector(row.get(new, row.get(old)), 13)[RIGHT]
                              for new, old in [('packet_q_rad', 'q_command_rad'),
                                  ('packet_dq_rad_s', 'dq_command_rad_s'),
                                  ('packet_kp', 'kp_command'), ('packet_kd', 'kd_command')]]
                    weight = float(row.get('packet_weight', row['weight']))
                    if not np.isfinite(weight) or not 0 <= weight <= 1:
                        raise ValueError('invalid weight')
                    commands.append(dict(ns=int(row['write_end_monotonic_ns']), ff=ff,
                        qref=fields[0], dqref=fields[1], kp=fields[2], kd=fields[3],
                        weight=weight, stage=row.get('stage'),
                        active=bool(row.get('mpc_active', False)),
                        desired=row.get('raw_mpc_ddq_rad_s2'),
                        model=row.get('post_transition_ddq_rad_s2'),
                        selected_total=row.get('tau_total_estimated_at_feedback_nm')))
                except (KeyError, TypeError, ValueError):
                    counts['malformed_commands'] += 1
                # This is the measurement used BEFORE this write. Its own
                # stamp joins it to an earlier command, never to this row's tau.
                if row.get('feedback_crc_valid') and row.get('tau_est_at_feedback_nm') is not None:
                    try:
                        ns = int(row['state_received_monotonic_ns'])
                        state = dict(q=vector(row['q_measured_rad'], 13)[RIGHT],
                            dq=vector(row['dq_measured_rad_s'], 13)[RIGHT],
                            tau=vector(row['tau_est_at_feedback_nm'], 13)[RIGHT])
                        counts['duplicate_feedback'] += int(ns in states)
                        states[ns] = state
                    except (KeyError, TypeError, ValueError):
                        counts['invalid_states'] += 1
            if row.get('schema') == 'g1_lowstate_raw_v1':
                try:
                    if not row.get('crc_valid'):
                        raise ValueError('invalid CRC')
                    motors = {int(m['index']): m for m in row['motors']}
                    state = {key: vector([motors[i][field] for i in MOTOR_IDS], 5)
                             for key, field in [('q', 'q_rad'), ('dq', 'dq_rad_s'), ('tau', 'tau_est_nm')]}
                    ns = int(row['received_monotonic_ns'])
                    counts['duplicate_feedback'] += int(ns in states)
                    states[ns] = state
                except (KeyError, TypeError, ValueError):
                    counts['invalid_states'] += 1
    if epoch is None or not commands or len(states) < 2:
        raise ValueError('need task epoch, successful writes with explicit tau_ff, and valid feedback; '
                         'offline replay alone is not physical response evidence')
    if counts['malformed_commands']:
        raise ValueError('malformed or torque-ambiguous successful writes; cannot hold an older '
                         'command across an unknown intervening output')
    commands.sort(key=lambda item: item['ns'])
    if any(b['ns'] <= a['ns'] for a, b in zip(commands, commands[1:])):
        raise ValueError('successful command stamps must be unique')
    for c in commands:
        c['t'] = (c['ns']-epoch)*1e-9
    ordered = [dict(t=(ns-epoch)*1e-9, **s) for ns, s in sorted(states.items())]
    return commands, ordered, dict(raw_sha256=digest.hexdigest(), **counts,
        successful_commands=len(commands), unique_feedback_samples=len(ordered),
        session_task=session.get('task', 'unknown'), session_outcome=session_end.get('outcome'),
        final_weight=session_end.get('final_weight'), capture_drain_recorded=bool(drain),
        journal_dropped=drain.get('queue_dropped'))


def error_stats(actual, expected):
    error = np.asarray(actual)-np.asarray(expected)
    if len(error) == 0:
        return None
    return dict(samples=len(error), bias=error.mean(axis=0).tolist(),
        rmse=np.sqrt(np.mean(error**2, axis=0)).tolist(),
        p95_abs=np.percentile(np.abs(error), 95, axis=0).tolist(),
        max_abs=np.max(np.abs(error), axis=0).tolist())


def interval_prediction(commands, times, begin, end, max_age_s):
    """Exact time-weighted mean of held commands; reject gaps/partial weight."""
    index = int(np.searchsorted(times, begin, side='right')-1)
    if index < 0:
        return None
    result = np.zeros((2, 5))
    cursor = begin
    while cursor < end-1e-12:
        c = commands[index]
        stop = min(end, times[index+1] if index+1 < len(times) else end)
        if stop-times[index] > max_age_s or c['weight'] < .999 or not c['active']:
            return None
        try:
            values = np.array([vector(c['desired'], 5), vector(c['model'], 5)])
        except (ValueError, TypeError):
            return None
        result += values*(stop-cursor)
        cursor, index = stop, index+1
    return result/(end-begin)


def analyze(path, assumed_delay_s=0., derivative_window_s=.024, max_age_s=.05):
    if not np.isfinite([assumed_delay_s, derivative_window_s, max_age_s]).all():
        raise ValueError('analysis parameters must be finite')
    if not 0 <= assumed_delay_s <= .05 or not .006 <= derivative_window_s <= .1 or not 0 < max_age_s <= .1:
        raise ValueError('invalid delay, derivative window or maximum sample age')
    commands, states, audit = read_capture(path)
    times = np.array([c['t']+assumed_delay_s for c in commands])
    feedback_t = np.array([s['t'] for s in states])
    rows, acceleration = [], []
    for i, state in enumerate(states):
        t = state['t']
        if not 3 <= t < 18:
            continue
        j = int(np.searchsorted(times, t, side='right')-1)
        if j < 0 or t-times[j] > max_age_s:
            continue
        c = commands[j]
        if c['weight'] < .999:
            continue  # proprietary weight blend has not been identified
        expected = c['ff']+c['kp']*(c['qref']-state['q'])+c['kd']*(c['dqref']-state['dq'])
        rows.append(dict(t=t, expected=expected, ff=c['ff'], estimated=state['tau'],
            q=state['q'], qref=c['qref'], dq=state['dq'], age_ms=(t-times[j])*1000,
            selected_total=c['selected_total']))
        # A finite velocity difference measures the mean acceleration over the
        # SAME interval as the integrated desired/model accelerations. No raw
        # ddq trust, sample-count timing assumption, or future-command lookup.
        k = int(np.searchsorted(feedback_t, t+derivative_window_s, side='left'))
        if k >= len(states) or states[k]['t'] >= 18:
            continue
        end = states[k]['t']
        if end-t > derivative_window_s+max_age_s or np.max(np.diff(feedback_t[i:k+1])) > max_age_s:
            continue
        prediction = interval_prediction(commands, times, t, end, max_age_s)
        if prediction is not None:
            acceleration.append(dict(t=(t+end)/2, span_s=end-t,
                actual=(states[k]['dq']-state['dq'])/(end-t),
                desired=prediction[0], model=prediction[1]))
    if not rows:
        raise ValueError('no full-weight, timestamp-aligned torque pairs; do not infer success')
    windows = {}
    for name, start, stop in [('pre_motion', 3., 5.), ('primary_full_task', 5., 18.)]:
        torque = [r for r in rows if start <= r['t'] < stop]
        accel = [r for r in acceleration if start <= r['t']-r['span_s']/2
                 and r['t']+r['span_s']/2 < stop]
        selected = [r for r in torque if r['selected_total'] is not None
                    and np.asarray(r['selected_total']).shape == (5,)
                    and np.isfinite(r['selected_total']).all()]
        windows[name] = dict(interval_s=[start, stop],
            torque_observed_interval_s=None if not torque else [torque[0]['t'], torque[-1]['t']],
            torque_max_gap_s=None if len(torque) < 2 else float(np.max(np.diff([r['t'] for r in torque]))),
            acceleration_observed_interval_s=None if not accel else
                [accel[0]['t']-accel[0]['span_s']/2, accel[-1]['t']+accel[-1]['span_s']/2],
            torque_est_minus_feedback_reconstructed_total_nm=error_stats(
                [r['estimated'] for r in torque], [r['expected'] for r in torque]),
            torque_est_minus_selected_total_at_command_nm=error_stats(
                [r['estimated'] for r in selected], [r['selected_total'] for r in selected]),
            measured_acc_minus_mpc_desired_rad_s2=error_stats(
                [r['actual'] for r in accel], [r['desired'] for r in accel]),
            measured_acc_minus_forward_model_rad_s2=error_stats(
                [r['actual'] for r in accel], [r['model'] for r in accel]))
    arrays = {f'torque_{key}': np.array([r[key] for r in rows])
              for key in ('t', 'expected', 'ff', 'estimated', 'q', 'qref', 'dq', 'age_ms')}
    arrays.update({f'acceleration_{key}': np.array([r[key] for r in acceleration])
                   for key in ('t', 'span_s', 'actual', 'desired', 'model')})
    summary = dict(schema='g1_mpc_execution_analysis_v1', source=str(Path(path).resolve()),
        program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        audit=audit, right_arm_motor_ids=MOTOR_IDS, windows=windows,
        absolute_torque_calibrated=False, hardware_performance_passed=None,
        assumed_command_to_feedback_delay_s=assumed_delay_s, delay_measured=False,
        velocity_difference_window_s=derivative_window_s, max_sample_age_s=max_age_s,
        unique_torque_est_values=[len(np.unique(arrays['torque_estimated'][:, i])) for i in range(5)],
        warnings=[reason for condition, reason in [
            (audit['failed_writes'] > 0, 'failed command writes recorded'),
            (audit['invalid_states'] > 0, 'invalid feedback was excluded'),
            (not audit['capture_drain_recorded'], 'capture drain marker missing'),
            (audit['journal_dropped'] not in (None, 0), 'recorder dropped rows'),
            (audit['session_outcome'] != 'normal_release_completed', 'normal completion not recorded'),
            (audit['final_weight'] != 0, 'final zero weight not recorded'),
            (rows[0]['t'] > 3.+max_age_s or rows[-1]['t'] < 18.-max_age_s,
             'full-weight response does not cover the complete [3,18) window')]
            if condition],
        limitations=[
            'tau_est is robot-estimated torque, not independent shaft-torque ground truth',
            'expected total = packet tau_ff + kp*(q_ref-q_measured) + kd*(dq_ref-dq_measured)',
            'PD is reconstructed at host feedback time, not observed inside the firmware servo',
            'zero delay is the unshifted comparison, not a claim of zero transport/actuator delay',
            'receive stamps and Write completion do not measure one-way DDS or motor latency',
            'partial-weight transition samples are excluded; blending is not identified',
            'acceleration is noisy interval-average delta(dq)/delta(t), not independent ground truth',
            'overlapping derivative windows are not independent statistical samples',
            'no ddq_raw substitution, force calibration, compensation fit or automatic pass verdict',
            'use the separate endpoint H0 analysis to evaluate actual bottle stabilization'])
    return arrays, summary


def write_outputs(arrays, summary, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory/'execution.npz', **arrays)
    (directory/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    for prefix, keys in [('torque', ('expected', 'ff', 'estimated', 'q', 'qref', 'dq')),
                         ('acceleration', ('actual', 'desired', 'model'))]:
        with (directory/f'{prefix}.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['task_s']+[f'{key}_motor{i}' for key in keys for i in MOTOR_IDS])
            for n, t in enumerate(arrays[f'{prefix}_t']):
                writer.writerow([t]+[v for key in keys for v in arrays[f'{prefix}_{key}'][n]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 3, figsize=(15, 13), sharex='col')
    for j, motor in enumerate(MOTOR_IDS):
        for key, label in [('expected', 'packet FF + reconstructed PD'),
                           ('estimated', 'robot tau_est'), ('ff', 'packet FF only')]:
            axes[j, 0].plot(arrays['torque_t'], arrays[f'torque_{key}'][:, j], label=label)
        for key, label in [('desired', 'MPC desired'), ('model', 'forward model'),
                           ('actual', 'measured velocity difference')]:
            if len(arrays['acceleration_t']):
                axes[j, 1].plot(arrays['acceleration_t'], arrays[f'acceleration_{key}'][:, j], label=label)
        for key in ('q', 'qref'):
            axes[j, 2].plot(arrays['torque_t'], np.rad2deg(arrays[f'torque_{key}'][:, j]), label=key)
        for col, unit in enumerate(('N m (estimate)', 'rad/s^2 (interval mean)', 'deg')):
            axes[j, col].set_ylabel(f'motor {motor}\n{unit}')
            axes[j, col].grid(alpha=.3)
            if j == 0 and axes[j, col].lines:
                axes[j, col].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('task time [s]')
    fig.suptitle('Host-time execution audit; not absolute torque calibration')
    fig.tight_layout()
    fig.savefig(directory/'execution.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_jsonl', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--assumed-delay-ms', type=float, default=0.)
    parser.add_argument('--derivative-window-ms', type=float, default=24.)
    args = parser.parse_args()
    arrays, summary = analyze(args.raw_jsonl, args.assumed_delay_ms/1000, args.derivative_window_ms/1000)
    write_outputs(arrays, summary, args.output_dir)
    print(json.dumps(dict(output=str(args.output_dir), windows=summary['windows'],
                         absolute_torque_calibrated=False), indent=2))


if __name__ == '__main__':
    main()
