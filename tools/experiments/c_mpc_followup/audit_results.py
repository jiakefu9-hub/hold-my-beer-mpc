"""Retain complete metrics, matched-period timing and post-hoc arm contacts."""
from pathlib import Path
import json
import sys

import mujoco
import numpy as np
import yaml

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ROOT=REPO/'evaluation/c_mpc_followup_20261012'
sys.path.insert(0,str(HERE.parent/'mpc_comprehensive'))
from contact_audit import audit


def enrich(result):
    if not result.get('complete'):
        log=(ROOT/'logs'/(result['name']+'.log')).read_text()
        result['failure_details']=[line for line in log.splitlines()
            if any(s in line for s in ['Error:', 'NO_SAFE_TORQUE', 'infeasible'])][-3:]
        return result
    if 'contact_audit' not in result:
        result['contact_audit']=audit(result)
    config=yaml.safe_load(Path(result['config']).read_text())
    dt=config['simulation_dt']*config['arm_control_decimation']
    perf=json.loads((Path(result['path'])/'perf_summary.json').read_text())['total']
    timing=perf['real_hardware_control']['right_arm_interval']
    assert abs(timing['budget_ms']-1000*dt)<1e-9
    result['full_interval_ms']=timing
    result['control_period_s']=dt
    result['timing_environment']=perf['timing_scope']['runtime_environment']
    model=mujoco.MjModel.from_xml_path(str(REPO/config['xml_path']))
    names=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll']
    indices=[model.joint('right_'+name+'_joint').qposadr[0] for name in names]
    with np.load(Path(result['path'])/'compact_metrics.npz') as tr:
        mask=(tr['time']>=0)&(tr['time']<8.-1e-10)
        q=np.rad2deg(tr['qpos'][mask][:,indices])
    margin=np.array(config['mpc_q_operating_margin_deg'])
    lo=np.array(config['mpc_q_min_deg'])+margin
    hi=np.array(config['mpc_q_max_deg'])-margin
    result['measured_joint_degrees']=dict(order=names,minimum=q.min(0).tolist(),
        maximum=q.max(0).tolist(),inner_box_minimum_margin=np.minimum(q-lo,hi-q).min(0).tolist(),
        measured_inner_box_exceedance_frames=int(np.sum(np.any((q<lo-1e-6)|(q>hi+1e-6),axis=1))))
    return result


if __name__=='__main__':
    all_results=[]
    for path in sorted((ROOT/'results').glob('*.json')):
        result=enrich(json.loads(path.read_text()))
        path.write_text(json.dumps(result,indent=2)+'\n')
        all_results.append(result)
        print(result['name'],result.get('contact_audit',result.get('failure_details')),flush=True)
    (ROOT/'all_results.json').write_text(json.dumps(all_results,indent=2)+'\n')
