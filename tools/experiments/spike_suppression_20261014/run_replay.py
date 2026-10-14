"""Recorded-input numerical validation. No socket, DDS, robot, or plant.

New commands do not change old observations. This checks consistency, command
increments and bounds, never physical vibration reduction or field timing.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,json,socket,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent),str(ROOT/'tools/g1_commissioning')]
import numpy as np
import orjson
from validate_mpc_learned import capture,replay
from hardware_mpc_learned import RightArmLearnedTorqueMpc,YawFeedbackMpcPolicy
from g1_walk_mpc import MpcRuntime
from hardware_mpc_control import json_values
from delta_acceleration import hardware_policy,algebra_check


def stats(x):
    x=np.asarray(x)
    return dict(rms=float(np.sqrt(np.mean(x*x))),p99=float(np.percentile(abs(x),99)),max=float(np.max(abs(x))))


def compact(path):
    rows=[]
    for line in path.open('rb'):
        r=orjson.loads(line);d=r.get('diagnostics',{})
        if not d.get('mpc_active'):continue
        rows.append(dict(t=r['task_s'],a=d['estimated_closed_loop_ddq_rad_s2'],
            tau=d['tau_total_estimated_at_feedback_nm'],b=d['torque_model_bias_nm'],
            predicted_acc=d['mpc']['one_step_prediction']['ee_lin_acc'],
            full_compute_ms=r['full_compute_ms']))
    if not rows:return dict(samples=0)
    arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    mask=(arrays['t']>=5)&(arrays['t']<15)
    x={k:v[mask] for k,v in arrays.items()};a=x['a'];tau=x['tau']
    out=dict(samples=len(a),modeled_acceleration_jump=stats(np.diff(a,axis=0)),
        modeled_torque_jump=stats(np.diff(tau,axis=0)),
        pitch_torque_jump=stats(np.diff(tau[:,0])),
        expected_endpoint_acceleration_rms=float(np.sqrt(np.mean(np.sum(x['predicted_acc']**2,axis=1)))),
        acceleration_sign_flips_over10=int(np.sum((a[1:]*a[:-1]<0)&(abs(a[1:])>10)&(abs(a[:-1])>10))))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=4)
    parser.add_argument('--weights',type=float,nargs='+',default=[0,.00025,.001,.004,.016])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    data=capture(args.source)
    paths=[ROOT/p for p in ('arm_mpc.py','tools/g1_commissioning/hardware_mpc_learned.py',
        'tools/g1_commissioning/hardware_mpc_solver.py','configs/hardware_mpc_learned_acc001_alpha0005_omega1.yaml',
        'configs/hardware_mpc_torque_learned.yaml','configs/g1_payload75.yaml')]
    report=dict(source=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        hardware_output=False,network_forbidden=True,closed_loop=False,
        algebra_check=algebra_check(),variants={},
        baseline_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    original_accept=MpcRuntime.accept_packet
    def accepted(runtime,frame,packet):
        original_accept(runtime,frame,packet)
        d=frame.get('diagnostics',{})
        if d.get('mpc_active'):
            runtime.controller.policy.accept_acceleration(d['estimated_closed_loop_ddq_rad_s2'])
    with patch.object(socket,'socket',side_effect=RuntimeError('research replay forbids networking')):
        for weight in args.weights:
            name='delta_'+str(weight).replace('.','p')
            with patch.object(RightArmLearnedTorqueMpc,'policy_type',hardware_policy(YawFeedbackMpcPolicy,weight)),patch.object(MpcRuntime,'accept_packet',accepted):
                result,_=replay(data,args.output,name,args.cpu,config=paths[3],model_config=paths[5],torque_config=paths[4],
                    predictor_mode='learned_filtered',zero_arm_neutral=True,use_logged_query_times=True)
            result['delta_weight']=weight
            path=args.output/(name+'.jsonl')
            result['commands']=compact(path)
            report['variants'][name]=result
            (args.output/'summary.json').write_text(json.dumps(json_values(report),indent=2)+'\n')
            print(name,result['status'],result['commands'],result['errors'],flush=True)
            # Only newly generated replay diagnostics: compact commands, exact
            # source and hashes remain; never remove physical captures.
            path.unlink()
    for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==report['baseline_sha256'][str(p.relative_to(ROOT))]
    return 0

if __name__=='__main__':raise SystemExit(main())
