#!/usr/bin/env python3
"""Offline field review: timing, pose changes and fixed delay sensitivity.

No SDK import, robot communication, gain fitting or automatic pass verdict.
Uses the recorded [5,18) task contract, not a hand-picked steady-state score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_mpc_execution import analyze, write_outputs


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    if not np.isfinite(a).all():
        raise ValueError('non-finite diagnostic samples')
    return dict(samples=len(a), mean=a.mean(axis=0).tolist(),
                std=a.std(axis=0).tolist(), min=a.min(axis=0).tolist(),
                p99=np.percentile(a, 99, axis=0).tolist(),
                max=a.max(axis=0).tolist())


def review(raw, endpoint_dir, output):
    if output.exists():
        raise FileExistsError(output)
    arrays, execution = analyze(raw)
    endpoint_summary = json.loads((endpoint_dir/'summary.json').read_text())
    if endpoint_summary['source_sha256'] != execution['audit']['raw_sha256']:
        raise ValueError('endpoint analysis belongs to a different raw capture')
    with np.load(endpoint_dir/'metrics.npz', allow_pickle=False) as f:
        endpoint = {key: f[key] for key in ('task_elapsed_s', 'left_tilt_deg', 'right_tilt_deg')}
    writes, cycles, runtime, session = [], [], {}, {}
    with raw.open() as stream:
        for line in stream:
            if not any(key in line for key in ('dds_write"', 'g1_mpc_cycle_complete_v1',
                                              'control_runtime', 'session_start')):
                continue
            r = json.loads(line)
            if r.get('event') == 'session_start':
                session = r
            if r.get('event') == 'control_runtime':
                runtime = r
            if r.get('schema') == 'g1_mpc_cycle_complete_v1':
                cycles.append(r)
            if r.get('event') == 'dds_write':
                writes.append({key: r.get(key) for key in ('sequence', 'task_elapsed_s',
                    'mpc_active', 'q_measured_rad', 'raw_mpc_ddq_rad_s2',
                    'controller_compute_us', 'control_prewrite_us', 'write_duration_us')} |
                    dict(predictor_mode=r.get('predictor', {}).get('mode')))
    if session.get('primary_metric_window_s') != [5., 18.]:
        raise ValueError('this review requires the recorded [5,18) task contract')
    if any(r.get('task_elapsed_s') is None for r in writes+cycles):
        raise ValueError('missing task timestamps; cannot compute field windows')
    active = [r for r in writes if 5 <= r['task_elapsed_s'] < 18 and r['mpc_active']]
    active_cycles = [r for r in cycles if 5 <= r['task_elapsed_s'] < 18]
    if not active or not active_cycles:
        raise ValueError('missing active commands or full-cycle timing')
    if {r['sequence'] for r in active} != {r['sequence'] for r in active_cycles}:
        raise ValueError('active command / complete-cycle sequence mismatch')
    timing = {key: stats([r[key] for r in active_cycles if r.get(key) is not None])
              for key in ('complete_work_ms', 'actual_period_ms', 'wake_lateness_ms',
                          'state_age_at_write_ms', 'imu_age_at_write_ms', 'write_ms')}
    timing.update({key.replace('_us', '_ms'): stats([r[key]/1000 for r in active])
                   for key in ('controller_compute_us', 'control_prewrite_us')})
    timing.update(complete_deadline_misses=sum(r['complete_deadline_missed'] for r in active_cycles),
                  skipped_slots=sum(r['skipped_slots'] for r in active_cycles),
                  active_cycles=len(active_cycles),
                  deadline_miss_percent=100*sum(r['complete_deadline_missed'] for r in active_cycles)/len(active_cycles))
    windows = {}
    mpc_start=float(session.get('mpc_start_s',5.))
    for name, start, stop in [('pre_mpc_last_half_second', mpc_start-.5, mpc_start),
                              ('primary_full_task', 5., 18.),
                              ('diagnostic_entry', mpc_start, mpc_start+1.), ('diagnostic_late', 8., 18.)]:
        rows = [r for r in writes if start <= r['task_elapsed_s'] < stop]
        mask = (endpoint['task_elapsed_s'] >= start) & (endpoint['task_elapsed_s'] < stop)
        windows[name] = dict(interval_s=[start, stop],
            role='primary' if name == 'primary_full_task' else 'diagnostic_only',
            right_q_deg=stats(np.rad2deg([r['q_measured_rad'][5:10] for r in rows])),
            desired_ddq_rad_s2=stats([r['raw_mpc_ddq_rad_s2'] for r in rows if r['mpc_active']]),
            **{side+'_bottle_tilt_deg': stats(endpoint[side+'_tilt_deg'][mask])
               for side in ('left', 'right')})
    sensitivity = {}
    for delay_ms in (0, 3, 6, 12):
        result = execution if delay_ms == 0 else analyze(raw, assumed_delay_s=delay_ms/1000)[1]
        sensitivity[str(delay_ms)] = {key: result['windows'][key]
                                     for key in ('primary_full_task', 'diagnostic_late')}
    summary = dict(schema='g1_mpc_field_review_v1', raw_sha256=execution['audit']['raw_sha256'],
        source=str(raw.resolve()), program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        execution_program_sha256=execution['program_sha256'],
        endpoint_summary_sha256=hashlib.sha256((endpoint_dir/'summary.json').read_bytes()).hexdigest(),
        endpoint_arrays_sha256=hashlib.sha256((endpoint_dir/'metrics.npz').read_bytes()).hexdigest(),
        task=session['task'], mpc_start_s=mpc_start, audit=execution['audit'], runtime_host=runtime,
        timing=timing, windows=windows, actual_predictor_modes=dict(Counter(r['predictor_mode'] for r in active)),
        delay_sensitivity_ms=sensitivity, physical_success_automatically_certified=False,
        limitations=['tau_est is not an independent torque sensor',
            'endpoint pose is model reconstruction from joints and body IMU, not a bottle sensor',
            'delay hypotheses are sensitivity checks, not identified DDS/motor delays',
            'late/entry windows explain behavior and do not replace the full task score',
            'cycle duration excludes final audit-row enqueue; no hard-real-time certificate',
            'static response cannot identify a reliable torque gain or friction compensation'])
    output.mkdir(parents=True, exist_ok=False)
    write_outputs(arrays, execution, output/'execution')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    visible = (endpoint['task_elapsed_s'] >= 3) & (endpoint['task_elapsed_s'] < 18)
    for side in ('left', 'right'):
        axes[0].plot(endpoint['task_elapsed_s'][visible], endpoint[side+'_tilt_deg'][visible], label=side)
    axes[0].set_ylabel('Bottle tilt [deg]\n(model reconstruction)')
    t = np.array([r['task_elapsed_s'] for r in writes])
    q = np.rad2deg([r['q_measured_rad'][5:10] for r in writes])
    visible = (t >= 3) & (t < 18)
    for j, name in enumerate(('shoulder pitch', 'shoulder roll', 'shoulder yaw', 'elbow', 'wrist roll')):
        axes[1].plot(t[visible], q[visible, j], label=name)
    axes[1].set_ylabel('Right measured q [deg]')
    axes[2].plot([r['task_elapsed_s'] for r in active], [r['controller_compute_us']/1000 for r in active], label='controller compute')
    axes[2].plot([r['task_elapsed_s'] for r in active_cycles], [r['complete_work_ms'] for r in active_cycles], label='complete work')
    axes[2].axhline(6., color='black', ls='--', label='6 ms nominal period')
    axes[2].set_ylabel('Host work [ms]')
    for key, label in [('desired', 'MPC desired'), ('model', 'forward model'), ('actual', 'measured delta(dq)/dt')]:
        axes[3].plot(arrays['acceleration_t'], arrays['acceleration_'+key][:, 1], label=label, alpha=.8)
    axes[3].set_ylabel('Shoulder roll\nmean accel [rad/s^2]')
    axes[3].set_xlabel('Task time [s]')
    for ax in axes:
        if mpc_start != 5.:
            ax.axvline(mpc_start,color='gray',ls='--')
        ax.axvline(5, color='gray', ls=':')
        ax.axvline(18, color='gray', ls=':')
        ax.set_xlim(3, 18)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, loc='best', ncol=3)
    fig.suptitle('Recorded '+session['task']+' MPC: response and timing (not torque calibration)')
    fig.tight_layout()
    fig.savefig(output/'overview.png', dpi=160)
    fig.savefig(output/'overview.pdf')
    plt.close(fig)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_jsonl', type=Path)
    parser.add_argument('--endpoint-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.raw_jsonl, args.endpoint_dir, args.output_dir)
    print(json.dumps(dict(output=str(args.output_dir), timing=result['timing'],
                          windows=result['windows']), indent=2))
