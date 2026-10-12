#!/usr/bin/env python3
"""Evaluate saved, causal forecasts against later host-write-aligned IMU labels.

No network, new predictor fitting, controller output, or claim of physical
closed-loop improvement. Raw sensor samples are used only as offline labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import orjson
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation
from hardware_mpc_forecast_timing import QUERY_TO_WRITE_MS, shifted_interval_means
from hardware_mpc_predictor import GAIN, ROOT


def integral_at(times, grid, values):
    """Integral of 2 ms causal-grid samples held to the next grid point."""
    position = (np.asarray(times) - grid[0]) / .002
    index = np.floor(position).astype(int)
    if np.any(index < 0) or np.any(index >= len(values)):
        raise ValueError('label outside recorded interval; no future padding allowed')
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)*.002))
    return cumulative[index] + (position-index)[..., None]*.002*values[index]


def vector_errors(predicted, observed):
    error = predicted-observed
    if not len(error):
        raise ValueError('no matched evaluation samples')
    return dict(samples=len(error),
        acc_vector_rmse_m_s2=float(np.sqrt(np.mean(np.sum(error[:, :3]**2, axis=1)))),
        alpha_vector_rmse_rad_s2=float(np.sqrt(np.mean(np.sum(error[:, 3:]**2, axis=1)))),
        per_axis_rmse=np.sqrt(np.mean(error**2, axis=0)).tolist())


def analyze(source):
    commands, imus = [], []
    with source.open('rb') as stream:
        for line in stream:
            row = orjson.loads(line)
            if (row.get('schema') == 'g1_hardware_mpc_command_v1'
                    and 5 <= row['task_elapsed_s'] < 18
                    and row.get('predictor', {}).get('mode') == 'learned_filtered'):
                if row['predictor'].get('acceleration_forecast_alignment'):
                    raise ValueError('selection study requires unchanged baseline source forecasts')
                commands.append(row)
            elif row.get('schema') == 'g1_mpc_predictor_imu_v1':
                imus.append(row)
    if not commands or not imus:
        raise ValueError('source missing learned forecasts or accepted IMU samples')
    origin = commands[0]['predictor']['anchor_monotonic_ns']
    imus.sort(key=lambda r: r['received_monotonic_ns'])
    times = np.array([(r['received_monotonic_ns']-origin)*1e-9 for r in imus])
    if np.any(np.diff(times) <= 0):
        raise ValueError('IMU receive stamps must be unique and increasing')
    ticks = np.arange(int(np.ceil(times[0]/.002)), int(np.floor(times[-1]/.002))+1)
    grid = ticks*.002
    indices = np.searchsorted(times, grid, side='right')-1
    rotations = Rotation.from_quat(np.array([r['quaternion_wxyz'] for r in imus])[:, [1,2,3,0]])
    acc = rotations.apply(np.array([r['accelerometer_raw_m_s2'] for r in imus])) + [0,0,-9.81]
    omega = rotations.apply(np.array([r['gyroscope_rad_s'] for r in imus]))
    samples = np.c_[acc, omega][indices]
    filtered, _ = lfilter([GAIN], [1, -(1-GAIN)], samples, axis=0,
                          zi=((1-GAIN)*samples[0])[None, :])
    derivative = np.r_[np.zeros((1,3)), np.diff(filtered[:, 3:], axis=0)/.002]
    alpha = lfilter([GAIN], [1, -(1-GAIN)], derivative, axis=0)
    yaw = commands[0]['yaw0_rad']
    if max(abs(r['yaw0_rad']-yaw) for r in commands) > 1e-10:
        raise ValueError('evaluation requires fixed H0 yaw')
    c, s = np.cos(yaw), np.sin(yaw)
    h0 = np.array([[c,s,0],[-s,c,0],[0,0,1]])
    truth_all = np.c_[filtered[:, :3]@h0.T, filtered[:, 3:]@h0.T, alpha@h0.T]
    anchors = np.array([(r['predictor']['anchor_monotonic_ns']-origin)*1e-9 for r in commands])
    anchor_indices = np.rint(anchors/.002).astype(int)-ticks[0]
    current = np.array([r['predictor']['current_y'][:9] for r in commands])
    parity = np.sqrt(np.mean((truth_all[anchor_indices]-current)**2, axis=0))
    if np.max(parity) > .05:
        raise ValueError(f'causal label reconstruction differs materially from logged current_y: {parity}')
    truth = truth_all[:, [0,1,2,6,7,8]]
    forecasts = np.array([r['predictor']['forecast_y'] for r in commands])[:, :, [0,1,2,6,7,8]]
    sends = np.array([(r['write_end_monotonic_ns']-origin)*1e-9 for r in commands])
    queries = np.array([(r['predictor']['query_monotonic_ns']-origin)*1e-9 for r in commands])
    lead_ms = (queries-anchors)*1000.+QUERY_TO_WRITE_MS
    if np.any(lead_ms < 3.5-1e-6) or np.any(lead_ms >= 5.5+1e-6):
        raise ValueError('unexpected causal grid age')
    candidates = {f'fixed_{lead}ms': shifted_interval_means(forecasts, lead)
                  for lead in (0,3,6,9,12)}
    candidates['host_write_aligned'] = np.asarray([
        shifted_interval_means(f, lead) for f, lead in zip(forecasts, lead_ms)])
    elapsed = np.array([r['task_elapsed_s'] for r in commands])
    results = {}
    for name, predicted in candidates.items():
        windows = {}
        for start_ms in (0,18):
            target = (integral_at(sends+(start_ms+6)*.001, grid, truth)
                      - integral_at(sends+start_ms*.001, grid, truth))/.006
            stage = {}
            for phase, mask in [('full', elapsed<18), ('walk', elapsed<15), ('stop', elapsed>=15)]:
                stage[phase] = vector_errors(predicted[mask, start_ms//6], target[mask])
            windows[f'write_plus_{start_ms}_to_{start_ms+6}_ms'] = stage
        results[name] = windows
    return dict(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        samples=len(commands), label_reconstruction_current_y_per_axis_rmse=parity.tolist(),
        anchor_to_write_ms_percentiles=np.percentile((sends-anchors)*1000.,[50,95,99]).tolist(),
        query_to_write_ms_percentiles=np.percentile((sends-queries)*1000.,[50,95,99]).tolist(),
        estimated_write_error_ms_percentiles=np.percentile((queries+.0035-sends)*1000.,[5,50,95]).tolist(),
        candidates=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with patch.object(socket, 'socket', side_effect=RuntimeError('offline analysis forbids network')):
        runs = [analyze(path) for path in args.source]
    result = dict(schema='g1_acceleration_forecast_timing_study_v1', hardware_output=False,
        host_budget_ms=QUERY_TO_WRITE_MS, runs=runs,
        limitations=['Filtered host-receive labels, not raw impact or sensor timestamps',
            'Write completion is not actuator application time; motor/DDS delay unidentified',
            'Saved causal predictions are never recomputed using future label data',
            'Historical episodes have been inspected; not blind validation',
            'Small reconstruction differences can reflect late callback availability',
            'Forecast-error reduction is not endpoint-acceleration or closed-loop improvement'],
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path(__file__).with_name('hardware_mpc_forecast_timing.py'))})
    (args.output_dir/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Acceleration forecast timing: offline label check', '',
        'Vector RMSE against the same filtered IMU interval [host Write end, +6 ms).',
        'This table does not measure endpoint control improvement.', '',
        '| Run | Acc baseline / aligned (m/s²) | Alpha baseline / aligned (rad/s²) |',
        '| --- | ---: | ---: |']
    for r in runs:
        k='write_plus_0_to_6_ms'
        before=r['candidates']['fixed_0ms'][k]['full']
        after=r['candidates']['host_write_aligned'][k]['full']
        a='acc_vector_rmse_m_s2';b='alpha_vector_rmse_rad_s2'
        lines.append(f"| {Path(r['source']).parent.name} | {before[a]:.4f} / {after[a]:.4f} | "
                     f"{before[b]:.4f} / {after[b]:.4f} |")
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__': raise SystemExit(main())
