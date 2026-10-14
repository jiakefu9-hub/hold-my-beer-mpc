#!/usr/bin/env python3
"""Offline field evidence audit; no DDS, network, fitting or controller changes.

Reuses saved endpoint geometry, checks the original 105 ms results, then changes
only the evaluation window. Joint tracking uses received LowState dq and the
integral of the held model acceleration, with the SAME derivative operator.
This estimates motion consistency, not motor latency or physical torque truth.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
from unittest.mock import patch

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import orjson
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from analyze_hardware_pid import _angular_velocity
from analyze_mpc_forecast_timing import analyze as forecast_analysis
from g1_run_morning_20261013 import ROOT, FROZEN_FILES, validate_baselines
from mpc_host import summarize_timing
from pid_timing import timing_summary

BASE = ROOT/'evaluation/hardware_shadow/commissioning'
REFERENCE = BASE/'final_comparison_20261012/summary.json'
WINDOWS_MS = (75, 105, 135)
JOINTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_pitch', 'wrist_roll')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rms(x):
    return float(np.sqrt(np.mean(np.sum(np.asarray(x)**2, axis=-1))))


def stats(x):
    x = np.asarray(x, dtype=float)
    return dict(zip(('p50', 'p95', 'p99', 'max'), map(float, np.percentile(x, [50,95,99,100]))))


def endpoint_windows(root, reference):
    path = root/'endpoint_analysis/metrics.npz'
    assert sha(path) == reference['provenance']['endpoint_arrays_sha256']
    out = {}
    with np.load(path, allow_pickle=False) as f:
        t = f['task_elapsed_s']
        np.testing.assert_allclose(np.diff(t), .005, atol=1e-10)
        mask = (t >= 5) & (t < 18)
        assert mask.sum() == 2600
        raw_acc = f['torso_specific_force_h0_m_s2'] + [0,0,-9.81]
        for ms in WINDOWS_MS:
            win = round(ms*.001*200)
            assert win % 2 == 1
            body_acc = savgol_filter(raw_acc, win, 3, axis=0)
            sides = {}
            for side in ('left', 'right'):
                pos = f[side+'_position_from_torso_imu_h0_m']
                q = f[side+'_orientation_h0_quaternion_wxyz']
                rot = Rotation.from_quat(q[:, [1,2,3,0]]).as_matrix()
                omega = savgol_filter(_angular_velocity(rot, .005), win, 3, axis=0)
                alpha = savgol_filter(omega, win, 3, deriv=1, delta=.005, axis=0)
                acc = body_acc + savgol_filter(pos, win, 3, deriv=2, delta=.005, axis=0)
                vel = savgol_filter(pos, win, 3, deriv=1, delta=.005, axis=0)[mask]
                vel -= vel.mean(axis=0)
                axes = rot[mask, :, 2]
                axis_mean = axes.mean(axis=0)
                axis_mean /= np.linalg.norm(axis_mean)
                swing = np.degrees(np.arccos(np.clip(axes@axis_mean, -1, 1)))
                tilt = f[side+'_tilt_deg'][mask]
                m = dict(tilt_mean_deg=float(tilt.mean()),
                         swing_rms_deg=float(np.sqrt(np.mean(swing**2))),
                         acc_rms_m_s2=rms(acc[mask]), alpha_rms_rad_s2=rms(alpha[mask]),
                         relative_speed_demeaned_rms_m_s=rms(vel), omega_rms_rad_s=rms(omega[mask]))
                if ms == 105:
                    for key, value in m.items():
                        np.testing.assert_allclose(value, reference[side][key], atol=1e-7, rtol=1e-7)
                sides[side] = m
            out[str(ms)] = sides
    return out


def read_control(root, reference):
    low, commands, timing, complete = [], [], [], []
    digest = hashlib.sha256()
    epoch = session = None
    with (root/'raw.jsonl').open('rb') as stream:
        for line in stream:
            digest.update(line)
            r = orjson.loads(line)
            kind = r.get('schema')
            if r.get('event') == 'task_epoch':
                epoch = r['task_epoch_monotonic_ns']
            if r.get('event') == 'session_start':
                session = r
            if kind == 'g1_lowstate_raw_v1' and r['crc_valid']:
                motors = {m['index']:m for m in r['motors']}
                low.append((r['received_monotonic_ns'], [motors[j]['dq_rad_s'] for j in range(22,27)]))
            if not 5 <= r.get('task_elapsed_s', -1) < 18:
                continue
            if kind == 'g1_hardware_mpc_command_v1' and r.get('event') == 'dds_write':
                commands.append(r)
            elif kind == 'g1_mpc_timing_v1':
                timing.append(r)
            elif kind == 'g1_mpc_cycle_complete_v1':
                complete.append(r)
    assert digest.hexdigest() == reference['provenance']['source_sha256']
    assert epoch and session and len(commands) > 2000 and len(complete) == len(commands)
    commands.sort(key=lambda r:r['write_end_monotonic_ns'])
    low.sort(key=lambda r:r[0])
    full = summarize_timing([{**r, 'full_work_ms':r['complete_work_ms'],
                             'deadline_missed':r['complete_deadline_missed']} for r in complete])
    assert full['deadline_misses'] == reference['timing']['deadline_misses']
    np.testing.assert_allclose(full['complete_work_ms']['p99'], reference['timing']['complete_work_ms']['p99'])
    cfg = session['core']['torque_config']
    q = np.degrees([r['q_measured_rad'][5:10] for r in commands])
    requested = np.array([r['raw_mpc_ddq_rad_s2'] for r in commands])
    modeled = np.array([r['estimated_closed_loop_ddq_rad_s2'] for r in commands])
    tau = np.array([r['field_total_torque_estimate_at_latest_feedback_nm'] for r in commands])
    ff = np.array([r['tau_ff'][5:10] for r in commands])
    bounds = np.array([cfg['q_min_deg'], cfg['q_max_deg']])
    margins = np.minimum(q.min(axis=0)-bounds[0], bounds[1]-q.max(axis=0))
    assert margins.min() >= -1e-6
    assert np.max(np.abs(tau)-cfg['tau_abs_nm']) < 1e-5
    assert np.max(np.abs(ff)-cfg['tau_ff_abs_nm']) < 1e-5
    assert not any(r['mpc'].get('fallback_used') for r in commands)
    constraint = dict(joint_order=JOINTS, q_outer_margin_deg=margins.tolist(),
        acceleration_at_cap_percent=(100*np.mean(np.abs(requested)>=cfg['max_ddq_rad_s2']-1e-3,axis=0)).tolist(),
        model_mapping_change_rms_per_joint=np.sqrt(np.mean((modeled-requested)**2,axis=0)).tolist(),
        torque_model_limit_fraction_per_joint=(np.max(np.abs(tau),axis=0)/cfg['tau_abs_nm']).tolist(),
        feedforward_limit_fraction_per_joint=(np.max(np.abs(ff),axis=0)/cfg['tau_ff_abs_nm']).tolist(),
        brake_active_percent=100*float(np.mean([np.any(np.array(r.get('predictive_braking',{}).get('activation',[]))>0) for r in commands])),
        brake_latched_percent=100*float(np.mean([r.get('predictive_braking',{}).get('latched',False) for r in commands])),
        hardware_torque_margin_certified=False)
    tracking = {}
    lt = np.array([(x[0]-epoch)*1e-9 for x in low])
    dq = np.array([x[1] for x in low])
    ct = np.array([(r['write_end_monotonic_ns']-epoch)*1e-9 for r in commands])
    assert np.all(np.diff(ct)>0) and np.all(np.diff(lt)>0)
    # Restrict to steady walking, away from start/stop and smoothing boundaries.
    grid = np.arange(6., 14.5, .002)
    assert grid[0] > max(lt[0], ct[0]) and grid[-1] < min(lt[-1], ct[-1])
    actual_velocity = np.column_stack([np.interp(grid,lt,dq[:,j]) for j in range(5)])
    index = np.searchsorted(ct, grid, side='right')-1
    # Exact integral of the piecewise-constant model command at host write end.
    integral = np.vstack((np.zeros(5), np.cumsum(modeled[:-1]*np.diff(ct)[:,None],axis=0)))
    modeled_velocity = integral[index]+modeled[index]*(grid-ct[index])[:,None]
    valid = (grid>=6.2)&(grid<14.2)
    for ms in WINDOWS_MS:
        win = round(ms*.001/.002)
        if win%2 == 0: win += 1
        actual = savgol_filter(actual_velocity,win,3,deriv=1,delta=.002,axis=0)[valid]
        expected = savgol_filter(modeled_velocity,win,3,deriv=1,delta=.002,axis=0)[valid]
        corr=[]
        for j in range(5):
            corr.append(float(np.corrcoef(expected[:,j],actual[:,j])[0,1]))
        tracking[str(ms)] = dict(actual_window_samples=win,
            expected_rms_per_joint=np.sqrt(np.mean(expected**2,axis=0)).tolist(),
            observed_rms_per_joint=np.sqrt(np.mean(actual**2,axis=0)).tolist(),
            error_rms_per_joint=np.sqrt(np.mean((actual-expected)**2,axis=0)).tolist(),
            zero_lag_correlation_per_joint=corr,
            rms_ratio_per_joint=(np.sqrt(np.mean(actual**2,axis=0))/np.sqrt(np.mean(expected**2,axis=0))).tolist())
    ages = timing_summary(timing)
    return dict(complete_timing=full, feedback_timing=ages, constraints=constraint,
                tracking=tracking, raw_feedback_acceleration_all_zero_percent=100*float(np.mean([
                    np.max(np.abs(r['ddq_raw_at_feedback_rad_s2'][5:10]))==0 for r in commands])),
                source_sha256=digest.hexdigest())


def make_report(result):
    lines=['# Field readiness audit, 2026-10-13', '',
        'Offline only. Same 12 valid T-series runs and exclusion rules as final comparison.',
        'T05 remains included with operator-observed torso contact; no collision-free claim.',
        'Endpoint window [5,18); group mean of per-run metrics. No independent payload sensor.', '',
        '## Filter window sensitivity', '',
        '| SG ms | Group | N | Left acc | Right acc | Right alpha | Right relative speed | Right omega |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for ms in WINDOWS_MS:
        for group in ('mpc_off','mpc_on','c_on'):
            g=result['groups'][str(ms)][group]
            left,right=g['left'],g['right']
            lines.append(f'|{ms}|{group}|{g["n"]}|{left["acc_rms_m_s2"]:.4f}|{right["acc_rms_m_s2"]:.4f}|{right["alpha_rms_rad_s2"]:.4f}|{right["relative_speed_demeaned_rms_m_s"]:.4f}|{right["omega_rms_rad_s"]:.4f}|')
    lines += ['', '## Execution and deadlines', '',
        '| Run | Complete P99 ms | Deadlines missed | Samples | LowState age at write P99 ms | IMU age P99 ms |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in result['runs'].items():
        t=r['control']['complete_timing']; a=r['control']['feedback_timing']
        lines.append(f'|{name}|{t["complete_work_ms"]["p99"]:.3f}|{t["deadline_misses"]}|{t["samples"]}|{a["state_age_at_write_ms"]["p99"]:.3f}|{a["imu_age_at_write_ms"]["p99"]:.3f}|')
    lines += ['', '## Matched-band joint tracking (105 ms nominal window)', '',
        'Steady walking [6.2,14.2). Both modeled integrated acceleration and received dq use the same SG derivative.',
        'No fitted delay/gain. Physical disturbance, filtering and model error are not separated.', '',
        '| Run | Joint | Model ddq RMS | Observed ddq RMS | Error RMS | Zero-lag correlation |',
        '|---|---|---:|---:|---:|---:|']
    for name,r in result['runs'].items():
        if r['group'] not in ('mpc_off','c_on'): continue
        t=r['control']['tracking']['105']
        for j,label in enumerate(JOINTS):
            lines.append(f'|{name}|{label}|{t["expected_rms_per_joint"][j]:.3f}|{t["observed_rms_per_joint"][j]:.3f}|{t["error_rms_per_joint"][j]:.3f}|{t["zero_lag_correlation_per_joint"][j]:.3f}|')
    lines += ['', '## Forecast labels at host write time', '',
        '| Run | Acc RMSE original / aligned | Alpha RMSE original / aligned |', '|---|---:|---:|']
    for name,r in result['forecast_timing'].items():
        a=r['candidates']['fixed_0ms']['write_plus_0_to_6_ms']['full']
        b=r['candidates']['host_write_aligned']['write_plus_0_to_6_ms']['full']
        lines.append(f'|{name}|{a["acc_vector_rmse_m_s2"]:.4f} / {b["acc_vector_rmse_m_s2"]:.4f}|{a["alpha_vector_rmse_rad_s2"]:.4f} / {b["alpha_vector_rmse_rad_s2"]:.4f}|')
    lines += ['', '## Limitations', '']+['- '+s for s in result['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    validate_baselines()
    ref=json.loads(REFERENCE.read_text())
    result=dict(schema='g1_field_readiness_audit_v1',hardware_output=False,
        window_ms=list(WINDOWS_MS),reference_sha256=sha(REFERENCE),runs={},groups={},forecast_timing={},
        baseline_sha256={str(p.relative_to(ROOT)):h for p,h in FROZEN_FILES.items()},
        limitations=['Offline inputs and host-receive times are not independent physical ground truth.',
        'Changing an offline derivative window does not alter the robot or the online 15 Hz filter.',
        'Model-executed acceleration is an estimate, not measured acceleration.',
        'LowState dq differentiated offline is a bandwidth-limited motion estimate; ddq_raw is not trusted.',
        'Tracking comparison uses no fitted time shift; it does not identify motor or transport latency.',
        'Model force limits do not certify physical torque or collision safety.',
        'Predictor input replay is not closed-loop performance or hardware real-time validation.',
        'Small historical nonrandomized sample; repeat-run variability and contact remain relevant.'])
    with patch.object(socket,'socket',side_effect=RuntimeError('offline audit forbids network')):
        for name,r in ref['runs'].items():
            group=next(g for g,d in ref['groups'].items() if name in d['run_ids'])
            root=BASE/r['provenance']['directory']
            result['runs'][name]=dict(group=group,endpoint=endpoint_windows(root,r),control=read_control(root,r))
            if group=='mpc_off':
                result['forecast_timing'][name]=forecast_analysis(root/'raw.jsonl')
            print(name,'105ms parity and control audit passed',flush=True)
        for ms in WINDOWS_MS:
            result['groups'][str(ms)]={}
            for group,g in ref['groups'].items():
                ids=g['run_ids']; item=dict(n=len(ids),run_ids=ids)
                for side in ('left','right'):
                    item[side]={key:float(np.mean([result['runs'][i]['endpoint'][str(ms)][side][key] for i in ids]))
                        for key in result['runs'][ids[0]]['endpoint'][str(ms)][side]}
                result['groups'][str(ms)][group]=item
    gain=-math.expm1(-2*math.pi*15*.002)
    result['online_filter_math']=dict(gain=gain,low_frequency_acc_delay_ms=1000*.002*(1-gain)/gain,
        low_frequency_alpha_delay_ms=1000*(2*.002*(1-gain)/gain+.001),measured_end_to_end_delay=False)
    validate_baselines()
    result['baseline_unchanged']=True
    result['validation_passed']=True
    result['script_sha256']=sha(__file__)
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'REPORT.md').write_text(make_report(result))
    print('Saved',args.output_dir/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
