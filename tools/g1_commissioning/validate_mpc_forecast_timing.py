#!/usr/bin/env python3
"""Replay baseline/aligned variants through the existing offline packet lifecycle."""
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
from g1_walk_mpc import LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG, ROOT
from validate_mpc_learned import capture, replay
from hardware_mpc_control import json_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--math-only', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(schema='g1_acceleration_forecast_replay_v1', hardware_output=False,
                  network_forbidden=True, math_only=args.math_only, sources=[])
    with patch.object(socket, 'socket', side_effect=RuntimeError('offline replay forbids network')):
        for source in args.source:
            data = capture(source)
            with source.open('rb') as stream:
                session = next(orjson.loads(line) for line in stream
                               if orjson.loads(line).get('schema') == 'g1_mpc_session_v1')
            zero = session.get('zero_arm_neutral_experiment', {}).get('enabled', False)
            directory = args.output_dir/source.parent.name
            directory.mkdir()
            reports, torques = {}, {}
            for name, aligned in [('baseline', False), ('aligned', True)]:
                reports[name], torques[name] = replay(data, directory, name, args.cpu,
                    math_only=args.math_only, config=LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG,
                    predictor_mode='learned_filtered', zero_arm_neutral=zero,
                    align_accel_forecast=aligned, use_logged_query_times=True)
                print(source.parent.name, name, reports[name]['status'],
                      reports[name]['active_commands'], reports[name]['errors'], flush=True)
            keys = sorted(torques['baseline'].keys() & torques['aligned'].keys())
            delta = np.asarray([np.asarray(torques['aligned'][k])-torques['baseline'][k] for k in keys])
            result['sources'].append(dict(source=str(source.resolve()),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                zero_arm_neutral=zero, variants=reports, paired_commands=len(keys),
                per_joint_mean_abs_tau_change_nm=np.mean(abs(delta),axis=0).tolist() if keys else None,
                per_joint_max_abs_tau_change_nm=np.max(abs(delta),axis=0).tolist() if keys else None))
    result['limitations'] = [
        'Recorded q/dq and torso motion do not respond to new commands: not closed-loop validation',
        'No network, DDS output or live compute-process scheduling in this replay',
        'Math-only, if selected, raises the local QP time allowance to 100 ms; not a timing pass',
        'New option preserves current inverse dynamics and measured state; no complete latency model']
    paths = [Path(__file__), Path(__file__).with_name('validate_mpc_learned.py'),
             Path(__file__).with_name('hardware_mpc_forecast_timing.py'),
             Path(__file__).with_name('g1_walk_mpc.py'), LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG,
             ROOT/'configs/hardware_mpc_torque_learned.yaml']
    result['code_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output_dir/'summary.json').write_text(json.dumps(json_values(result),indent=2,allow_nan=False)+'\n')
    return 0 if all(v['status']=='passed' and v['final_weight']==0 for r in result['sources']
                    for v in r['variants'].values()) else 1


if __name__ == '__main__': raise SystemExit(main())
