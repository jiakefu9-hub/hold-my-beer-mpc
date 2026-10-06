#!/usr/bin/env python3
"""Reproducible, no-socket day review and measured-state counterfactual replay.

Replays ALL saved active states and fault states, never invented future walking.
Two input variants share the same new controller/limits: saved command-time
prediction versus received feedback. Neither is a physical closed-loop trial.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import orjson
from disturbance_types import DisturbanceInput,DisturbanceHorizon
from scipy.spatial.transform import Rotation
from hardware_mpc_delay_plan import IntervalHorizonClock
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc,load_torque_config
from hardware_mpc_control import json_values
from endpoint_pose import EndpointModel,ROOT
from g1_walk_pid import EXPECTED_TARGET_Q
from investigate_mpc_walk_start import feasibility

FIELD=ROOT/'configs/hardware_mpc_torque_field.yaml'


def horizon_from(sample,predicted):
    current=np.asarray(sample['current']);forecast=np.asarray(sample['forecast'])
    rotation=Rotation.from_euler('xyz',current[9:12]).as_matrix()
    nodes=[DisturbanceInput(current[:3],current[3:6],current[6:9],rotation)]
    intervals=[]
    for future in forecast:
        omega=.5*(nodes[-1].omega_world+future[3:6])
        midpoint=Rotation.from_rotvec(omega*.003).as_matrix()@rotation
        rotation=Rotation.from_rotvec(omega*.006).as_matrix()@rotation
        intervals.append(DisturbanceInput(future[:3],omega,future[6:9],midpoint))
        nodes.append(DisturbanceInput(future[:3],future[3:6],future[6:9],rotation))
    result=DisturbanceHorizon(tuple(nodes),tuple(intervals))
    if predicted:
        d=sample['delay']
        result=IntervalHorizonClock(result).shifted(d['command_time_s']-d['forecast_s'])
    return result


def read_capture(path):
    summary=dict(raw=str(path.relative_to(ROOT)),commands=0,active_commands=0,
                 faults=[],events=[],last_active_s=None)
    samples=[];last=None;sha=hashlib.sha256()
    with path.open('rb') as stream:
        for line in stream:
            sha.update(line);row=orjson.loads(line);event=row.get('event')
            if event=='session_start':
                summary.update(task=row['task'],delay_ms=row.get('assumed_command_delay_ms'),
                               saved_config=row['core']['torque_config'])
            if event in ('session_end','session_fault','local_failure','capture_drained','velocity_final_zero'):
                summary['events'].append(row)
            if event=='dds_write':
                summary['commands']+=1
                if row.get('mpc_active'):
                    summary['active_commands']+=1;summary['last_active_s']=row['task_elapsed_s']
                    d=row
                else:
                    last=row;continue
            elif event=='controller_fault_detail':
                summary['faults'].append(row['reason']);d=row['diagnostics']
                if 'predictor' not in d or last is None:continue
            else:
                continue
            pred=d['predictor']
            slots=d.get('q_measured_rad',last['q_measured_rad'] if last else None)
            speeds=d.get('dq_measured_rad_s',last['dq_measured_rad_s'] if last else None)
            samples.append(dict(kind='fault' if event=='controller_fault_detail' else 'issued',
                task_s=d.get('task_elapsed_s'),slots=slots,
                q=d.get('raw_observed_q_rad',slots[5:10]),
                dq=d.get('raw_observed_dq_rad_s',speeds[5:10]),
                predicted_q=d.get('command_time_q_rad',slots[5:10]),
                predicted_dq=d.get('command_time_dq_rad_s',speeds[5:10]),
                current=pred['current_y'],forecast=pred['forecast_y'],
                delay=d.get('delay_preview'),dt=d['feedback_dt_s']))
            if event=='dds_write':last=row
    summary['sha256']=sha.hexdigest()
    return summary,samples


def replay(folder,samples,predicted):
    c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10],folder/'mpc_config.yaml',
        model=EndpointModel(folder/'controller_config.yaml'),torque_config=FIELD)
    c.policy.solver_time_limit=.1  # math replay; live timing tested separately
    result=dict(samples=len(samples),passed=0,failures=[],max_model_tracking_error=0.,
                max_abs_tau_nm=np.zeros(5),final_fault_lp=None)
    try:
        for index,sample in enumerate(samples):
            q=np.asarray(sample['predicted_q'] if predicted else sample['q'])
            dq=np.asarray(sample['predicted_dq'] if predicted else sample['dq'])
            slots=np.asarray(sample['slots']).copy();slots[5:10]=q
            c.set_measured_dq(dq)
            h=horizon_from(sample,predicted and sample['delay'] is not None)
            c.set_disturbance_horizon(h,{'mode':'captured_body_counterfactual'})
            try:
                qr,vr,d=c.step(slots,[1,0,0,0],0.,sample['dt'])
                tau=np.asarray(d['tau_total_estimated_at_feedback_nm'])
                ff=np.asarray(d['tau_ff_candidate_nm'])
                conf=c.torque_config
                if (d['torque_slew_bounds_nm'] is not None or d['torque_slew_reference']!='none'
                        or np.any(abs(tau)>c.mapper.limit+1e-6)
                        or np.any(abs(ff)>np.asarray(conf['tau_ff_abs_nm'])+1e-6)
                        or np.any(qr<np.deg2rad(conf['q_min_deg'])-1e-7)
                        or np.any(qr>np.deg2rad(conf['q_max_deg'])+1e-7)):
                    raise AssertionError('independent packet/limit audit failed')
                result['passed']+=1
                result['max_abs_tau_nm']=np.maximum(result['max_abs_tau_nm'],abs(tau))
                result['max_model_tracking_error']=max(result['max_model_tracking_error'],
                    float(np.max(abs(np.asarray(d['mapper']['checked_ddq_rad_s2'])-
                                     np.asarray(d['raw_mpc_ddq_rad_s2'])))))
            except (RuntimeError,ValueError) as exc:
                result['failures'].append(dict(index=index,kind=sample['kind'],task_s=sample['task_s'],
                    reason=str(exc),q_deg=np.rad2deg(q),dq_rad_s=dq))
            if sample['kind']=='fault':
                # Independent full-state/input LP: original q/dq/ddq limits,
                # without extra braking or ANY torque rows. Disambiguates why
                # a QP fails instead of calling every failure "too conservative".
                result['final_fault_lp']=feasibility(c.policy,q,dq)
    finally:c.close()
    result['all_saved_states_pass']=not result['failures']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    files=sorted((ROOT/'evaluation/hardware_shadow/commissioning').glob('mpc_torque_*20261006_*/raw.jsonl'))
    results=[]
    with patch.object(socket,'socket',side_effect=RuntimeError('review forbids sockets')):
        for path in files:
            summary,samples=read_capture(path)
            if samples:
                summary['new_measured_state']=replay(path.parent,samples,False)
                summary['new_with_saved_extrapolation']=replay(path.parent,samples,True)
                print(path.parent.name,len(samples),
                      'measured pass',summary['new_measured_state']['passed'],
                      'predicted pass',summary['new_with_saved_extrapolation']['passed'],flush=True)
            else:print(path.parent.name,'no active MPC records',flush=True)
            results.append(summary)
    sources=[Path(__file__),*[Path(__file__).with_name(name) for name in
        ('hardware_mpc_solver.py','hardware_mpc_torque_control.py','hardware_torque_mapper.py',
         'hardware_mpc_delay_plan.py','investigate_mpc_walk_start.py')],ROOT/'arm_mpc.py',FIELD,
         ROOT/'configs/hardware_mpc_torque_preview.yaml']
    result=json_values(dict(schema='g1_mpc_day_review_v1',hardware_output=False,dds_initialized=False,
        field_config=load_torque_config(FIELD),runs=results,offline_solver_time_budget_s=.1,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        limitations=['saved states under OLD commands, not new closed-loop robot response',
            'body forecast reused at its recorded anchor; no future data or template refit',
            'no data beyond failed walking prefixes; cannot certify full walking',
            'predicted variant uses same new no-slew controller, not reconstruction of old binaries',
            '0.1 s offline solver budget is not live timing evidence']))
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
