#!/usr/bin/env python3
"""Offline preparation: three controllers, recorded feedback, no network or DDS.

This is input replay, not a simulated plant or a prediction of field metrics.
Run under taskset -c 0-17 to make the selected validation CPU available.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from endpoint_pose import EndpointModel
from g1_run_morning_20261013 import (ROOT, PROFILE, MODEL, MPC_CONFIG, TORQUE_CONFIG,
                                    POSITION_PID_CONFIG, FROZEN_FILES, validate_baselines)
from g1_walk_pid import load_profile, load_pid_parameters
from hardware_mpc_control import json_values
from hardware_pid_control import RightArmHardwarePid, HardwarePidPlan
from hardware_position_pid_zero import prepare_profile, preflight as pid_preflight
from g1_walk_mpc import preflight as mpc_preflight
from validate_mpc_learned import capture, replay


def replay_pid(data, output):
    commands, low, imu, timing = data
    profile = prepare_profile(load_profile(PROFILE), POSITION_PID_CONFIG, MODEL)
    parameters, _ = load_pid_parameters(MODEL, profile)
    bounds = profile['position_pid_reference_bounds_rad']
    controller = RightArmHardwarePid(np.zeros(5), parameters,
        model=EndpointModel(MODEL), reference_bounds_rad=bounds)
    plan = HardwarePidPlan(commands[0]['q_measured_rad'], profile['target_q_array'],
                          profile['kp_array'], profile['kd_array'], controller)
    imu_times = np.asarray([r[0] for r in imu])
    saved, previous = [], None
    active = 0
    for command in commands:
        t = command['task_elapsed_s']
        i = int(np.searchsorted(imu_times, command['imu_received_monotonic_ns'], side='right'))-1
        if i < 0:
            raise ValueError('no causal recorded IMU sample')
        dt = (command.get('control_actual_period_ms') or 6.)*.001
        frame = plan.sample(t, np.asarray(command['q_measured_rad']),
            np.asarray(command['dq_measured_rad_s']), imu[i][1], command['yaw0_rad'], dt)
        if frame['diagnostics']['pid_active']:
            active += 1
            q, dq = frame['q_rad'][5:10], frame['dq_rad_s'][5:10]
            if np.any(q < bounds[0]-1e-10) or np.any(q > bounds[1]+1e-10):
                raise ValueError('PID reference position limit violated')
            if np.max(np.abs(dq)) > parameters.hardware_max_dq+1e-10:
                raise ValueError('PID reference velocity limit violated')
            if previous is not None and np.max(np.abs(dq-previous)) > parameters.hardware_max_ddq*min(dt,.006)+1e-9:
                raise ValueError('PID reference acceleration limit violated')
            previous = dq.copy()
            np.testing.assert_array_equal(frame['q_rad'][:5], np.zeros(5))
        saved.append(np.r_[t, frame['weight'], frame['q_rad'], frame['dq_rad_s']])
    if not frame['terminal'] or frame['weight'] != 0.:
        raise ValueError('recorded-input PID plan did not finish release')
    trajectory = np.asarray(saved)
    np.savez_compressed(output/'position_pid_replay.npz', columns=trajectory)
    return dict(passed=True, active_samples=active, final_weight=frame['weight'],
        q_command_min_deg=np.rad2deg(trajectory[(trajectory[:,0]>=5)&(trajectory[:,0]<18),7:12].min(axis=0)).tolist(),
        q_command_max_deg=np.rad2deg(trajectory[(trajectory[:,0]>=5)&(trajectory[:,0]<18),7:12].max(axis=0)).tolist(),
        control_performance_not_evaluated=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=7)
    args = parser.parse_args(argv)
    validate_baselines()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(hardware_output=False, network_forbidden=True,
        source=str(args.source.resolve()), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        field_timing_validated=False, closed_loop_validated=False, trials={})
    with patch.object(socket, 'socket', side_effect=RuntimeError('offline validation forbids network')):
        data = capture(args.source)
        for mode in ('learned_filtered', 'hold_current'):
            result = mpc_preflight(config=MPC_CONFIG, model_config=MODEL, mode=mode,
                cpu=args.cpu, torque_config=TORQUE_CONFIG, zero_arm_neutral=True)
            replay_result, _ = replay(data, args.output_dir, mode, args.cpu,
                config=MPC_CONFIG, model_config=MODEL, torque_config=TORQUE_CONFIG,
                predictor_mode=mode, zero_arm_neutral=True, use_logged_query_times=True)
            report['trials'][mode] = dict(preflight=result, replay=replay_result)
            print(mode, replay_result['status'], 'final_weight', replay_result['final_weight'], flush=True)
        profile = prepare_profile(load_profile(PROFILE), POSITION_PID_CONFIG, MODEL)
        params, _ = load_pid_parameters(MODEL, profile)
        report['trials']['position_pid'] = dict(preflight=pid_preflight(profile, params, MODEL),
                                              replay=replay_pid(data, args.output_dir))
    validate_baselines()
    report['baseline_files_unchanged'] = True
    report['baseline_sha256'] = {str(p.relative_to(ROOT)): h for p,h in FROZEN_FILES.items()}
    report['passed'] = all(t['preflight']['passed'] and
        (t['replay'].get('passed') or t['replay'].get('status') == 'passed') for t in report['trials'].values())
    (args.output_dir/'summary.json').write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+'\n')
    print('offline preparation passed:', report['passed'], flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
