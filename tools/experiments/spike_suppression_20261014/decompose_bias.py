"""Decompose the logged arm-model bias, not measured physical joint torque."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import orjson
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tools/g1_commissioning')]
from endpoint_pose import EndpointModel
from hardware_arm_inverse_dynamics import RightArmInverseDynamics
from disturbance_types import DisturbanceInput


def analyze(path):
    model=EndpointModel(path.parent/'controller_config.yaml')
    backend=SimpleNamespace(nq=model.model.nq,nv=model.model.nv,scene_mjcf_path=model.xml.resolve())
    dyn=RightArmInverseDynamics(model,backend);zero=np.zeros(3);rows=[]
    for line in path.open('rb'):
        r=orjson.loads(line)
        if r.get('schema')!='g1_hardware_mpc_command_v1' or not r.get('mpc_active') or not 5<=r['task_elapsed_s']<15:continue
        y=np.asarray(r['predictor']['current_y']);R=Rotation.from_euler('xyz',y[9:12]).as_matrix()
        q=np.asarray(r['measured_q_rad']);dq=np.asarray(r['measured_dq_rad_s'])
        def bias(a,w,alpha,vel):return dyn.linear_dynamics(q,vel,DisturbanceInput(a,w,alpha,R))[1]
        gravity=bias(zero,zero,zero,np.zeros(5))
        lin=bias(y[:3],zero,zero,np.zeros(5))-gravity
        ang=bias(zero,zero,y[6:9],np.zeros(5))-gravity
        vel=bias(zero,y[3:6],zero,dq)-gravity
        b=np.asarray(r['torque_model_bias_nm']);tau=np.asarray(r['tau_total_estimated_at_feedback_nm'])
        rows.append(dict(t=r['task_elapsed_s'],gravity=gravity,base_linear=lin,
            base_angular_acceleration=ang,velocity_terms=vel,b=b,tau=tau,inertia_action=tau-b,
            closure_max=float(np.max(abs(gravity+lin+ang+vel-b)))))
    data={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    closure=float(data['closure_max'].max())
    if closure>1e-5:raise ValueError(f'bias decomposition does not match capture: {closure}')
    groups=['gravity','base_linear','base_angular_acceleration','velocity_terms','inertia_action']
    changes={k:np.diff(data[k],axis=0) for k in [*groups,'b','tau']}
    worst=np.argsort(abs(changes['tau'][:,0]))[-5:][::-1]
    return dict(source=str(path),samples=len(rows),closure_max=closure,
        per_joint_increment_rms_nm={k:np.sqrt(np.mean(v*v,axis=0)).tolist() for k,v in changes.items()},
        largest_pitch_changes=[dict(t=float(data['t'][i+1]),delta={k:float(v[i,0]) for k,v in changes.items()}) for i in worst])


if __name__=='__main__':
    results=[analyze(Path(s)) for s in sys.argv[1:]]
    out=ROOT/'evaluation/spike_suppression_20261014/bias_decomposition.json'
    out.write_text(json.dumps(results,indent=2)+'\n')
    print(out)
    for r in results:print(r,flush=True)
