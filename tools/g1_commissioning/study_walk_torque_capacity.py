#!/usr/bin/env python3
"""Full recorded H0 walking: model torque demand, NOT MPC/hardware success.

Evaluate tau=M*a+b at all 6 ms samples of [5,18) in the twelve September
captures. The maximum over the whole acceleration box [-8,8]^5 is exact for
this frozen local affine model; actual controller demands are usually smaller.
No templates are trained, no SDK is imported, no commands are sent.
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
from scipy.spatial.transform import Rotation
from disturbance_types import DisturbanceInput
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc
from g1_walk_pid import EXPECTED_TARGET_Q
from endpoint_pose import ROOT
from hardware_mpc_control import json_values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    dataset=ROOT/'evaluation/hardware_shadow/commissioning/walk_h0_predictor_study_20260925/data'
    old=np.array([5.,3.,2.,5.,1.5]);results=[]
    with patch.object(socket,'socket',side_effect=RuntimeError('capacity study forbids network')):
        c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10])
        try:
            for trial in range(1,13):
                source=dataset/f'trial{trial:02d}_prepared.npz'
                audit_path=dataset/f'trial{trial:02d}_h0_audit.json'
                audit=json.loads(audit_path.read_text())
                with np.load(source,allow_pickle=False) as archive:d=dict(archive)
                # Prepared files use the original 35-slot motor indexing.
                selected=np.flatnonzero((d['t']>=5)&(d['t']<18))[::3]
                if len(selected)<2100:raise ValueError('incomplete full walking window')
                rotations=Rotation.from_quat(d['quaternion_h0_xyzw'][selected]).as_matrix()
                arrays={key:[] for key in ('measured','nominal')}
                for mode in arrays:
                    for row,k in enumerate(selected):
                        q=d['q'][k,22:27] if mode=='measured' else EXPECTED_TARGET_Q[5:10]
                        v=d['dq'][k,22:27] if mode=='measured' else np.zeros(5)
                        base=DisturbanceInput(d['acc'][k],d['omega'][k],d['alpha'][k],rotations[row])
                        mass,bias=c.inverse.linear_dynamics(q,v,base)
                        # Exact per-axis maximum absolute total torque over
                        # all 32 acceleration-box corners; not a fitted score.
                        total_bound=abs(bias)+8.*np.sum(abs(mass),axis=1)
                        kp,kd=c.torque_config['kp'],c.torque_config['kd'];dt=.006
                        ff_bias=bias-kp*dt*v
                        ff_mass=mass-np.diag(.5*kp*dt**2+kd*dt)
                        ff_bound=abs(ff_bias)+8.*np.sum(abs(ff_mass),axis=1)
                        arrays[mode].append(np.r_[bias,total_bound,ff_bound])
                arrays={key:np.asarray(value) for key,value in arrays.items()}
                np.savez_compressed(args.output_dir/f'trial{trial:02d}.npz',
                    t=d['t'][selected],q=d['q'][selected,22:27],dq=d['dq'][selected,22:27],**arrays)
                result=dict(trial=trial,samples=len(selected),source=str(source.relative_to(ROOT)),
                    prepared_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                    raw_sha256_as_recorded_by_original_audit=audit['sha256'],
                    supplemental=trial==6,modes={})
                for mode,values in arrays.items():
                    all_bound=np.maximum(values[:,5:10],values[:,10:15])
                    result['modes'][mode]=dict(peak_support_nm=np.max(abs(values[:,:5]),axis=0),
                        peak_total_box_nm=np.max(values[:,5:10],axis=0),
                        peak_ff_box_nm=np.max(values[:,10:15],axis=0),
                        old_limit_exceeded_count=np.sum(all_bound>old,axis=0),
                        peak_time_s=[float(d['t'][selected[np.argmax(all_bound[:,j])]]) for j in range(5)])
                results.append(result);print(trial,json_values(result['modes']),flush=True)
        finally:c.close()
    files=[Path(__file__),ROOT/'configs/hardware_mpc_torque_preview.yaml',
        Path(__file__).with_name('hardware_arm_inverse_dynamics.py')]
    primary=[r for r in results if not r['supplemental']]
    result=dict(hardware_output=False,runs=results,model_bottle_mass_kg=.25,
        maximum_allowed_ddq_rad_s2=8.,old_limits_nm=old,
        window_s=[5,18],sample_spacing_s=.006,body_filter_hz=15.,
        primary_peak_bound={mode:np.max([np.maximum(r['modes'][mode]['peak_total_box_nm'],
            r['modes'][mode]['peak_ff_box_nm']) for r in primary],axis=0) for mode in ('measured','nominal')},
        source_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},
        limitations=['Model-based torque box, not actual MPC torque or calibrated motor output',
            'Recorded body/arm motions under PD; no MPC reaction on torso is simulated',
            'Per-axis extreme torques can correspond to different acceleration corners',
            '6 ms samples and filtered IMU do not cover every instantaneous impact',
            'Trial 6 is supplemental due to missing final records; other 11 are primary',
            'Previously held-out data used for actuator preparation, not fresh predictor validation',
            'No raw re-extraction; prepared arrays are hashed and original audit hashes retained'])
    (args.output_dir/'summary.json').write_text(json.dumps(json_values(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
