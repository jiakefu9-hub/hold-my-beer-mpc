#!/usr/bin/env python3
"""No-socket counterfactual: early braking versus raising absolute torque caps.

Retains every computed command and source hashes. Saved-state replay cannot
prove real closed-loop behavior; the additional integration is a MODEL test.
"""
import argparse
from collections import deque
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
from review_mpc_20261006 import read_capture, horizon_from, FIELD
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc, load_torque_config
from hardware_mpc_control import json_values
from endpoint_pose import EndpointModel, ROOT
from g1_walk_pid import EXPECTED_TARGET_Q


def controller(folder, braking, wide):
    config=load_torque_config(FIELD)
    config['predictive_braking_enabled']=braking
    # Freeze the historical comparator even after the live field overlay is
    # raised. This study asks old cap versus 25, not today's moving defaults.
    cap=[25.]*5 if wide else [5.,3.,2.,5.,1.5]
    config.update(tau_abs_nm=cap,tau_ff_abs_nm=cap)
    c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10],folder/'mpc_config.yaml',
        model=EndpointModel(folder/'controller_config.yaml'),torque_config=config)
    c.policy.solver_time_limit=.1  # Mathematical replay, NOT a timing benchmark.
    return c


def step(c,sample,q=None,dq=None):
    q=np.asarray(sample['q'] if q is None else q)
    dq=np.asarray(sample['dq'] if dq is None else dq)
    slots=np.asarray(sample['slots']).copy();slots[5:10]=q
    c.set_measured_dq(dq)
    c.set_disturbance_horizon(horizon_from(sample,False),{'mode':'captured_body_counterfactual'})
    return c.step(slots,[1,0,0,0],0.,.006)[2]


def replay(folder,samples,braking,wide,stream):
    c=controller(folder,braking,wide)
    report=dict(samples=len(samples),passed=0,failures=[],peak_total_nm=np.zeros(5),
                braking_activated_count=np.zeros(5,dtype=int))
    commands=[]
    try:
        for index,sample in enumerate(samples):
            row=dict(run=folder.name,braking=braking,wide_25_nm=wide,index=index,
                task_s=sample['task_s'],q_deg=np.rad2deg(sample['q']),dq=sample['dq'])
            try:
                d=step(c,sample)
                tau=np.asarray(d['tau_total_estimated_at_feedback_nm'])
                activation=np.asarray(d['predictive_braking'].get('activation',[0.]*5))
                row.update(ddq=d['raw_mpc_ddq_rad_s2'],tau=tau,
                    checked_ddq=d['mapper']['checked_ddq_rad_s2'],braking_detail=d['predictive_braking'])
                report['peak_total_nm']=np.maximum(report['peak_total_nm'],abs(tau))
                report['braking_activated_count']+=activation>0
                report['passed']+=1
                commands.append(row)
            except (ValueError,RuntimeError) as exc:
                row['error']=str(exc);report['failures'].append(row);commands.append(None)
            stream.write(json.dumps(json_values(row),allow_nan=False)+'\n')
    finally:c.close()
    return report,commands


def model_braking(folder,sample,braking,gain,delay_steps):
    """Toy response: gain times model-checked acceleration, with FIFO delay.

    Gravity/support compensation is assumed exact. Torso forecast is frozen
    to this captured prefix. This probes early braking, NOT full walking.
    """
    c=controller(folder,braking,False)
    q=np.asarray(sample['q']).copy();v=np.asarray(sample['dq']).copy()
    queue=deque([np.zeros(5) for _ in range(delay_steps)])
    trace=[];fault=None
    try:
        for k in range(40):  # 240 ms, 6 ms grid
            trace.append(dict(t=k*.006,q_deg=np.rad2deg(q),dq=v.copy()))
            if np.any(q<c.minimum-1e-7) or np.any(q>c.maximum+1e-7):
                fault='model crossed original angle boundary';break
            try:d=step(c,sample,q,v)
            except (ValueError,RuntimeError) as exc:fault=str(exc);break
            requested=np.asarray(d['mapper']['checked_ddq_rad_s2'])
            queue.append(gain*requested);a=queue.popleft()
            trace[-1].update(requested_ddq=requested,applied_ddq=a)
            q=q+v*.006+.5*a*.006**2;v=v+a*.006
        if fault is None:
            trace.append(dict(t=len(trace)*.006,q_deg=np.rad2deg(q),dq=v.copy()))
    finally:c.close()
    return dict(braking=braking,response_gain=gain,delay_s=delay_steps*.006,
                fault=fault,peak_shoulder_pitch_deg=max(r['q_deg'][0] for r in trace),trace=trace)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    runs=[]
    root=ROOT/'evaluation/hardware_shadow/commissioning'
    with (patch.object(socket,'socket',side_effect=RuntimeError('study forbids network')),
          (args.output_dir/'commands.jsonl').open('w') as stream):
        for path in sorted(root.glob('mpc_torque_*20261006_*/raw.jsonl')):
            info,samples=read_capture(path)
            if not samples:continue
            info=dict(raw=info['raw'],sha256=info['sha256'],variants={})
            variants={}
            for braking,wide in ((False,False),(False,True),(True,False),(True,True)):
                key=f'brake_{int(braking)}_wide_{int(wide)}'
                info['variants'][key],variants[key]=replay(path.parent,samples,braking,wide,stream)
            comparisons={}
            for braking in (0,1):
                pairs=list(zip(variants[f'brake_{braking}_wide_0'],variants[f'brake_{braking}_wide_1']))
                differences=[np.max(abs(np.asarray(a['tau'])-b['tau'])) for a,b in pairs if a and b]
                comparisons[f'cap_effect_brake_{braking}']=dict(
                    compared=len(differences),max_abs_command_difference_nm=max(differences,default=None),
                    changed_over_0_01_nm=sum(x>.01 for x in differences))
            info['comparisons']=comparisons
            if path.parent.name.endswith('171242'):
                # First actually recorded walking state where pitch's soft
                # braking cost activates. No synthetic favorable initial q.
                case=next((i for i,row in enumerate(variants['brake_1_wide_0'])
                           if row and row['task_s'] is not None and row['task_s']>=5
                           and row['braking_detail']['activation'][0]>0),None)
                info['model_probe_start_index']=case
                if case is not None:
                    info['model_probes']=[model_braking(path.parent,samples[case],b,g,d)
                        for g,d in ((1.,0),(.5,2)) for b in (False,True)]
            runs.append(info)
            print(path.parent.name,len(samples),'variants',
                  [v['passed'] for v in info['variants'].values()],comparisons,flush=True)
    paths=[Path(__file__),Path(__file__).with_name('review_mpc_20261006.py'),
        *[Path(__file__).with_name(n) for n in ('hardware_mpc_braking.py','hardware_mpc_solver.py',
            'hardware_mpc_torque_control.py','hardware_torque_mapper.py')],ROOT/'arm_mpc.py',FIELD,
        ROOT/'configs/hardware_mpc_torque_preview.yaml']
    result=dict(hardware_output=False,field_config=load_torque_config(FIELD),runs=runs,
        compared_caps_nm=dict(historical=[5,3,2,5,1.5],wide=[25]*5),
        source_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
        limitations=['Recorded states under OLD commands, not physical closed-loop results',
            'No new torque cap is applied to hardware',
            'Model probes freeze the recorded torso forecast; response gain/delay are assumptions',
            'Model probes assume exact support compensation; not a robustness proof',
            '0.1 s offline solver allowance is not a timing pass'])
    (args.output_dir/'summary.json').write_text(json.dumps(json_values(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
