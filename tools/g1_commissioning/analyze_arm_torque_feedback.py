#!/usr/bin/env python3
"""Read saved logs: torque estimates, PD tracking and model gravity, no DDS.

This cannot calibrate absolute shaft torque. Static/gravity residuals include
payload, inertia, friction, sensor scaling, timing and firmware differences.
No fitted compensation is exported to a controller. Downsampling is explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from endpoint_pose import EndpointModel, ROOT, rotation
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from disturbance_types import DisturbanceInput
from hardware_arm_inverse_dynamics import RightArmInverseDynamics
from robot_model_backend.cpp_rnea_backend import CppRightArmRneaBackend


def read_samples(raw, stride=100):
    """Causal file-order join, requiring timestamp age <=100 ms for IMU/cmd."""
    if not isinstance(stride,int) or stride < 1:
        raise ValueError('stride must be a positive integer')
    latest_imu = command = None
    epoch = None
    counter = 0
    skipped = 0
    samples = []
    digest = hashlib.sha256()
    with Path(raw).open('rb') as stream:
        for line in stream:
            digest.update(line)
            # The raw recorder writes compact JSON. Parsing only selected
            # LowState rows avoids retaining multi-hundred-MB logs in memory.
            if b'g1_torso_imu_raw_v1' in line:
                latest_imu = line
            elif b'"task_epoch_monotonic_ns"' in line:
                epoch = int(json.loads(line)['task_epoch_monotonic_ns'])
            elif b'"dds_write"' in line:
                candidate = json.loads(line)
                if candidate.get('event') == 'dds_write': command = candidate
            elif b'g1_lowstate_raw_v1' in line:
                counter += 1
                if counter % stride or epoch is None: continue
                low = json.loads(line)
                stamp = int(low['received_monotonic_ns'])
                task_s = (stamp-epoch)*1e-9
                if not 3. <= task_s < 18.: continue
                if latest_imu is None or command is None or not low.get('crc_valid'):
                    skipped += 1; continue
                imu = json.loads(latest_imu)
                command_stamp = command.get('write_end_monotonic_ns', command.get('write_end_ns'))
                if command_stamp is None or not all(0 <= stamp-int(ns) <= 100_000_000
                    for ns in (command_stamp,imu['received_monotonic_ns'])):
                    skipped += 1; continue
                if float(command.get('weight',0)) < .999:
                    skipped += 1; continue
                motors = {int(m['index']):m for m in low['motors']}
                q = np.array([motors[i]['q_rad'] for i in range(22,27)])
                dq = np.array([motors[i]['dq_rad_s'] for i in range(22,27)])
                tau = np.array([motors[i]['tau_est_nm'] for i in range(22,27)])
                qref = np.array(command.get('q_target',command.get('q_command_rad')))[5:10]
                dqref = np.array(command.get('dq_target',command.get('dq_command_rad_s')))[5:10]
                kp = np.array(command.get('kp',command.get('kp_command')))[5:10]
                kd = np.array(command.get('kd',command.get('kd_command')))[5:10]
                # Current field PID/MPC schemas send tau=0; old A3/capture
                # records include tau_ff explicitly. Do not assume unknowns.
                if 'tau_ff' in command:
                    ff=np.array(command['tau_ff'])[5:10]
                elif command.get('schema') in {'g1_hardware_pid_command_v1','g1_hardware_mpc_command_v1'}:
                    ff=np.zeros(5)
                else:
                    skipped += 1; continue
                if not np.isfinite(np.r_[q,dq,tau,qref,dqref,kp,kd,ff]).all():
                    skipped += 1; continue
                samples.append(dict(t=task_s,q=q,dq=dq,tau=tau,qref=qref,
                    pd=ff+kp*(qref-q)+kd*(dqref-dq),
                    ddq_raw=np.array([motors[i]['ddq_raw_rad_s2'] for i in range(22,27)]),
                    R=rotation(imu['quaternion_wxyz']),
                    imu_age_ms=(stamp-int(imu['received_monotonic_ns']))*1e-6,
                    command_age_ms=(stamp-int(command_stamp))*1e-6))
    if not samples: raise ValueError(f'no aligned full-weight samples: {raw}')
    return samples,dict(raw_sha256=digest.hexdigest(),lowstate_rows=counter,
                        lowstate_stride=stride,selected_samples=len(samples),skipped_samples=skipped)


def describe(samples, inverse):
    for sample in samples:
        sample['gravity'] = inverse.compute(sample['q'],np.zeros(5),np.zeros(5),
            DisturbanceInput(np.zeros(3),np.zeros(3),np.zeros(3),sample['R']))['tau_model_nm']
    windows={
        'quiet_pre_walk_3p5_5': [s for s in samples if 3.5<=s['t']<5 and max(abs(s['dq']))<.05],
        'walk_5_15': [s for s in samples if 5<=s['t']<15],
        'late_settle_17_18': [s for s in samples if 17<=s['t']<18],
    }
    report={}
    for key,rows in windows.items():
        if not rows:
            report[key]={'samples':0};continue
        values={key:np.array([s[key] for s in rows]) for key in ('q','dq','tau','qref','pd','gravity','ddq_raw')}
        q,dq,tau,qref,pd,g,ddq=(values[k] for k in ('q','dq','tau','qref','pd','gravity','ddq_raw'))
        report[key]={
            'samples':len(rows),'motor_indices':list(range(22,27)),
            'q_mean_deg':np.rad2deg(q.mean(axis=0)).tolist(),
            'q_target_mean_deg':np.rad2deg(qref.mean(axis=0)).tolist(),
            'tracking_error_mean_deg':np.rad2deg((q-qref).mean(axis=0)).tolist(),
            'tau_est_mean_nm':tau.mean(axis=0).tolist(),
            'tau_est_min_nm':tau.min(axis=0).tolist(),'tau_est_max_nm':tau.max(axis=0).tolist(),
            'tau_est_nonzero_fraction':(np.abs(tau)>1e-9).mean(axis=0).tolist(),
            'tau_est_unique_values':[int(len(np.unique(tau[:,i]))) for i in range(5)],
            'tau_est_on_0p0625_grid':bool(np.all(np.abs(tau/.0625-np.round(tau/.0625))<1e-5)),
            'pd_estimated_at_feedback_mean_nm':pd.mean(axis=0).tolist(),
            'tau_est_minus_pd_rmse_nm':np.sqrt(np.mean((tau-pd)**2,axis=0)).tolist(),
            'model_gravity_only_mean_nm':g.mean(axis=0).tolist(),
            'tau_est_minus_gravity_rmse_nm':np.sqrt(np.mean((tau-g)**2,axis=0)).tolist(),
            'ddq_raw_all_zero':bool(np.all(ddq==0)),
            'max_imu_age_ms':max(s['imu_age_ms'] for s in rows),
            'max_command_age_ms':max(s['command_age_ms'] for s in rows),
        }
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_logs',nargs='+',type=Path)
    parser.add_argument('--stride',type=int,default=100)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('output must be a new file')
    model=EndpointModel()
    backend=CppRightArmRneaBackend(model.xml,library_path=ROOT/'build/right_arm_rnea/libright_arm_rnea.so')
    try:
        inverse=RightArmInverseDynamics(model,backend)
        reports=[]
        for raw in args.raw_logs:
            samples,audit=read_samples(raw,args.stride)
            reports.append(dict(source=str(raw),**audit,windows=describe(samples,inverse)))
        result=dict(schema='g1_arm_torque_feedback_analysis_v1',runs=reports,
            model=inverse.metadata,xml_sha256=model.xml_hashes(),
            program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            inverse_source_sha256=hashlib.sha256(Path(__file__).with_name('hardware_arm_inverse_dynamics.py').read_bytes()).hexdigest(),
            hardware_output=False,absolute_torque_calibration=False,
            limitations=[
                'tau_est is an estimate, not independent shaft-torque ground truth',
                '0.0625 grid is an observation of sampled values, not an accuracy specification',
                'ddq_raw zero does not mean actual angular acceleration is zero',
                'PD estimate uses latest prior command and feedback, not internal motor-loop timestamps',
                'gravity-only model comparison is interpretable primarily during quiet standing',
                'unknown actual payload and unmodeled friction/firmware can confound residuals',
                'selected stride does not describe unsampled spikes; no compensation is fitted or enabled'])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
        print(json.dumps({'output':str(args.output),'runs':len(reports),
                          'samples':[r['selected_samples'] for r in reports]}))
    finally:backend.close()


if __name__=='__main__':main()
