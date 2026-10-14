#!/usr/bin/env python3
"""Offline six-run morning comparison; reads captures and never uses robot IO.

First generate each run's endpoint_analysis with analyze_hardware_pid.py.
The same estimator, windows and per-run aggregation are used for all controllers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

os.environ.setdefault('MPLCONFIGDIR', '/tmp/g1_morning_matplotlib')
import numpy as np
import orjson
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from analyze_hardware_pid import _angular_velocity
from g1_run_morning_20261013 import ROOT, validate_baselines

BASE = ROOT/'evaluation/hardware_shadow/commissioning'
GROUPS = {'learned_mpc': ['M01', 'M06'], 'hold_mpc': ['M02', 'M05'],
          'position_pid': ['M03', 'M04']}
RUN_DIRS = {
    'M01': 'morning_M01_mpc_learned_zero_payload75_walk_20261013_150426_750630_999f',
    'M02': 'morning_M02_mpc_hold_zero_payload75_walk_20261013_150615_501041_a64a',
    'M03': 'morning_M03_position_pid_zero_zero_payload75_walk_20261013_150734_982806_6eea',
    'M04': 'morning_M04_position_pid_zero_repeat_zero_payload75_walk_20261013_150910_347989_77e3',
    'M05': 'morning_M05_mpc_hold_repeat_zero_payload75_walk_20261013_151453_637007_fa6e',
    'M06': 'morning_M06_mpc_learned_repeat_zero_payload75_walk_20261013_151617_849525_70f1',
}
WINDOWS = {'primary': (5, 18), 'walk': (5, 15), 'stop': (15, 18),
           'later_walk': (7, 15), 'late_walk': (10, 15)}
METRICS = ('tilt_mean_deg', 'swing_rms_deg', 'acc_rms_m_s2', 'alpha_rms_rad_s2',
           'relative_speed_demeaned_rms_m_s', 'omega_rms_rad_s')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rms(x):
    return float(np.sqrt(np.mean(np.sum(x*x, axis=1))))


def endpoints(f, interval, window=21):
    t = f['task_elapsed_s']
    mask = (t >= interval[0]) & (t < interval[1])
    assert mask.sum() == round((interval[1]-interval[0])*200)
    body_acc = savgol_filter(f['torso_specific_force_h0_m_s2']+[0, 0, -9.81], window, 3, axis=0)
    result = {'torso': {'acc_rms_m_s2': rms(body_acc[mask])}}
    for side in ('left', 'right'):
        rotations = Rotation.from_quat(f[side+'_orientation_h0_quaternion_wxyz'][:, [1,2,3,0]]).as_matrix()
        axes = rotations[mask, :, 2]
        mean_axis = axes.mean(axis=0)
        mean_axis /= np.linalg.norm(mean_axis)
        swing = np.degrees(np.arccos(np.clip(axes@mean_axis, -1, 1)))
        tilt = f[side+'_tilt_deg'][mask]
        pos = f[side+'_position_from_torso_imu_h0_m']
        vel = savgol_filter(pos, window, 3, deriv=1, delta=.005, axis=0)[mask]
        vel -= vel.mean(axis=0)
        omega = savgol_filter(_angular_velocity(rotations, .005), window, 3, axis=0)
        alpha = savgol_filter(omega, window, 3, deriv=1, delta=.005, axis=0)
        acc = body_acc+savgol_filter(pos, window, 3, deriv=2, delta=.005, axis=0)
        result[side] = dict(tilt_mean_deg=float(tilt.mean()), tilt_rms_deg=float(np.sqrt(np.mean(tilt**2))),
            tilt_max_deg=float(tilt.max()), swing_rms_deg=float(np.sqrt(np.mean(swing**2))),
            acc_rms_m_s2=rms(acc[mask]), alpha_rms_rad_s2=rms(alpha[mask]),
            relative_speed_demeaned_rms_m_s=rms(vel), omega_rms_rad_s=rms(omega[mask]))
        if window == 21:
            np.testing.assert_allclose(acc[mask], f[side+'_linear_acceleration_h0_m_s2'][mask], atol=1e-8)
            np.testing.assert_allclose(alpha[mask], f[side+'_angular_acceleration_h0_rad_s2'][mask], atol=1e-8)
    return result


def read_run(key):
    root = BASE/RUN_DIRS[key]
    s = json.loads((root/'endpoint_analysis/summary.json').read_text())
    assert sha(root/'raw.jsonl') == s['source_sha256']
    assert s['capture_quality']['status'] == 'recorded_checks_passed'
    assert s['session_status']['release_completed'] and not s['session_status']['faults']
    assert s['sample_hz'] == 200 and s['derivative_filter']['actual_window_samples'] == 21
    with np.load(root/'endpoint_analysis/metrics.npz') as f:
        windows = {k: endpoints(f, v) for k, v in WINDOWS.items()}
        sensitivity = {str(ms): endpoints(f, WINDOWS['primary'], round(ms*.2)) for ms in (75,135)}
        tilt_at = {str(t): float(f['right_tilt_deg'][np.argmin(abs(f['task_elapsed_s']-t))])
                   for t in (5, 7, 10, 15, 17.9)}
    commands, release, velocities = [], [], []
    session = None
    for line in (root/'raw.jsonl').open('rb'):
        r = orjson.loads(line)
        if r.get('event') == 'session_start': session = r
        if r.get('event') == 'normal_weight_release_started': release.append(r)
        if r.get('event') == 'velocity_reply': velocities.append(r)
        if r.get('event') == 'dds_write' and 5 <= r.get('task_elapsed_s', -1) < 18:
            commands.append(r)
    assert session['control_nominal_period_ms'] == 6
    assert session['zero_arm_neutral_experiment']['enabled']
    assert len(release) == 1 and release[0]['zero_velocity_acknowledged']
    assert all(r['return_code'] == 0 for r in velocities)
    assert all(np.allclose(r['kp_command'][:5], 20) and np.allclose(r['kd_command'][:5], 1)
               and np.allclose(r['q_command_rad'][:5], 0) for r in commands)
    q = np.degrees([r['q_measured_rad'][5:10] for r in commands])
    qc = np.degrees([r['q_command_rad'][5:10] for r in commands])
    e = dict(right_q_min_deg=q.min(axis=0).tolist(), right_q_max_deg=q.max(axis=0).tolist(),
             right_q_command_min_deg=qc.min(axis=0).tolist(), right_q_command_max_deg=qc.max(axis=0).tolist())
    if 'core' in session:
        cfg = session['core']['torque_config']
        tau = np.abs([r['field_total_torque_estimate_at_latest_feedback_nm'] for r in commands])
        ff = np.abs([r['tau_ff'][5:10] for r in commands])
        margin = np.minimum(q.min(0)-cfg['q_min_deg'], np.array(cfg['q_max_deg'])-q.max(0))
        e.update(q_outer_margin_deg=margin.tolist(), torque_model_max_abs_nm=tau.max(0).tolist(),
            torque_model_margin_nm=(np.array(cfg['tau_abs_nm'])-tau.max(0)).tolist(),
            feedforward_margin_nm=(np.array(cfg['tau_ff_abs_nm'])-ff.max(0)).tolist(),
            solved_count=sum(bool(r['mpc'].get('solved')) for r in commands),
            fallback_count=sum(bool(r['mpc'].get('fallback_used')) for r in commands),
            brake_latched_percent=100*float(np.mean([r.get('predictive_braking',{}).get('latched',False) for r in commands])))
        assert margin.min() >= 0 and np.max(tau-np.array(cfg['tau_abs_nm'])) < 1e-5
        assert np.max(ff-np.array(cfg['tau_ff_abs_nm'])) < 1e-5
        assert e['solved_count'] == len(commands) and e['fallback_count'] == 0
        assert not session['align_accel_forecast']
    else:
        cfg = session['position_pid_reference_experiment']
        lo, hi = np.array(cfg['q_reference_min_deg']), np.array(cfg['q_reference_max_deg'])
        dist = np.minimum(qc-lo, hi-qc)
        e.update(reference_min_margin_deg=dist.min(0).tolist(),
                 reference_within_0_1deg_of_limit_percent=(100*(dist<.1).mean(0)).tolist(),
                 governor=s['control']['hardware_governor'])
        assert dist.min() >= -1e-8
    timing = s['control'].get('complete_loop_timing', s['control']['loop_timing'])
    return dict(directory=root.name, windows=windows, sensitivity=sensitivity, execution=e,
        right_tilt_at_task_time_deg=tilt_at,
        timing=timing, lifecycle=s['session_status'], capture_quality=s['capture_quality'],
        provenance=dict(raw_sha256=s['source_sha256'], arrays_sha256=sha(root/'endpoint_analysis/metrics.npz'),
            analysis_source_sha256=sha(Path(__file__)),
            endpoint_analysis_source_sha256=sha(Path(__file__).with_name('analyze_hardware_pid.py')),
            model_config_sha256=session['controller_config_sha256'],
            control_source_sha256=session['control_source_sha256'], predictor=session.get('predictor_mode'),
            mpc_config_sha256=session.get('mpc_config_sha256'), torque_config_sha256=session.get('torque_config_sha256'),
            position_pid_config_sha256=session.get('position_pid_config_sha256')))


def aggregate(runs):
    out = {}
    for g, ids in GROUPS.items():
        out[g] = dict(ids=ids, n=len(ids), windows={})
        for window in WINDOWS:
            out[g]['windows'][window] = {}
            for side in ('left','right','torso'):
                out[g]['windows'][window][side] = {}
                for metric in runs[ids[0]]['windows'][window][side]:
                    a = np.array([runs[k]['windows'][window][side][metric] for k in ids])
                    out[g]['windows'][window][side][metric] = dict(mean=float(a.mean()), min=float(a.min()),
                        max=float(a.max()), sample_std=float(a.std(ddof=1)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    validate_baselines()
    with patch.object(socket, 'socket', side_effect=RuntimeError('offline analysis forbids network')):
        runs = {key: read_run(key) for key in sorted(sum(GROUPS.values(), []))}
        assert len({r['provenance']['model_config_sha256'] for r in runs.values()}) == 1
        for ids in GROUPS.values():
            for field in ('control_source_sha256','mpc_config_sha256','torque_config_sha256','position_pid_config_sha256'):
                assert runs[ids[0]]['provenance'][field] == runs[ids[1]]['provenance'][field], (ids, field)
        groups = aggregate(runs)
        result = dict(window_s=[5,18], sample_hz=200, derivative_window_ms=105, runs=runs, groups=groups,
            semantics=dict(aggregation='Arithmetic mean of per-run metrics, n=2 per controller; no significance claim',
                tilt='Angle between endpoint local +Z and frozen H0/world vertical; mean over samples',
                swing='RMS angular distance from each axis to normalized mean axis of that run/window',
                velocity='Derivative of endpoint-minus-torso IMU position in fixed H0, demeaned per window; not absolute translation',
                acceleration='3D vector norm RMS, gravity removed; angular metrics use full 3D rotation',
                hold='Learning prediction ablation; measured disturbances, gravity and moving-base inverse dynamics retained',
                measurement='Joint/IMU model reconstruction; not external optical or liquid-surface ground truth',
                timing='MPC through timing enqueue; PID through command enqueue; both exclude device actuation delay'))
        (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        lines = ['# 2026-10-13 morning comparison', '', 'All six runs included. Primary [5,18), 200 Hz, 105 ms cubic SG.',
                 'Relative velocity is demeaned per window. Per-group means average TWO independent run metrics.', '',
                 '| Run | Side | Tilt mean deg | Swing RMS deg | Acc RMS m/s² | Alpha RMS rad/s² | Relative speed RMS m/s | Omega RMS rad/s |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        for key,r in runs.items():
            for side in ('left','right'):
                lines.append(f'| {key} | {side} | '+' | '.join(f'{r["windows"]["primary"][side][m]:.5f}' for m in METRICS)+' |')
        for window in WINDOWS:
            lines += ['',f'## Group means: {window}', '', '| Controller | Side | Tilt | Swing | Acc | Alpha | Relative speed | Omega |',
                      '|---|---|---:|---:|---:|---:|---:|---:|']
            for key,g in groups.items():
                for side in ('left','right'):
                    lines.append(f'| {key} | {side} | '+' | '.join(f'{g["windows"][window][side][m]["mean"]:.5f}' for m in METRICS)+' |')
        lines += ['', '## Timing and excitation', '', '| Run | Work P99 ms | Max ms | Deadline misses / samples | Torso acc RMS m/s² |',
                  '|---|---:|---:|---|---:|']
        for key,r in runs.items():
            t=r['timing'];w=t.get('complete_work_ms',t['work_ms'])
            lines.append(f'| {key} | {w["p99"]:.3f} | {w["max"]:.3f} | {t["deadline_misses"]}/{t["samples"]} | {r["windows"]["primary"]["torso"]["acc_rms_m_s2"]:.4f} |')
        lines += ['', '## Semantics', '']+[f'- {k}: {v}' for k,v in result['semantics'].items()]
        lines += ['', 'Detailed joint/reference/torque margins, recording hashes and 75/135 ms sensitivity: summary.json.', '']
        (args.output_dir/'REPORT.md').write_text('\n'.join(lines))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for color, (group, ids) in zip(('tab:blue', 'tab:orange', 'tab:green'), GROUPS.items()):
            for i, key in enumerate(ids):
                with np.load(BASE/RUN_DIRS[key]/'endpoint_analysis/metrics.npz') as f:
                    t = f['task_elapsed_s']; mask = (t >= 5) & (t < 18)
                    ax.plot(t[mask], f['right_tilt_deg'][mask], color=color,
                            linestyle='-' if i == 0 else '--', linewidth=1,
                            label=f'{key}: {group}')
        ax.axvline(15, color='0.5', linestyle=':', label='stop request')
        ax.set(xlabel='Task time (s)', ylabel='Right +Z tilt from vertical (deg)',
               title='All six runs: includes initial convergence, walking and stopping')
        ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(args.output_dir/'right_tilt_traces.png', dpi=180)
        plt.close(fig)
        for key,g in groups.items():
            print(key,{m:round(g['windows']['primary']['right'][m]['mean'],6) for m in METRICS})
    print(args.output_dir/'REPORT.md')


if __name__ == '__main__':
    main()
