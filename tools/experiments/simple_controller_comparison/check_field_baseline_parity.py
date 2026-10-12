"""Compare unchanged MPC numerics against this task's saved pre-edit runtime."""
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/g1_commissioning'))
from g1_walk_mpc import MpcRuntime
from hardware_control_comparison import BASELINE_CONFIG,MODEL_CONFIG,TORQUE_CONFIG
import numpy as np


def main():
    old=runpy.run_path(str(ROOT/'evaluation/acceleration_pid_refinement_20261012/hardware_before/tools/g1_commissioning/g1_walk_mpc.py'))['MpcRuntime']
    options=dict(config=BASELINE_CONFIG,model_config=MODEL_CONFIG,torque_config=TORQUE_CONFIG,
                 zero_arm_neutral=True,align_accel_forecast=True)
    runtimes=[old(**options),MpcRuntime(**options)]
    diffs=[]
    try:
        for r in runtimes:
            r.predictor.set_grid_origin(0)
            r.controller.policy.solver_time_limit=.1
        for i in range(601):
            ns=i*2_000_000;t=ns*1e-9
            q=np.zeros(35);dq=np.zeros(35)
            q[:12]=.01*np.sin(t+np.arange(12));dq[:12]=.01*np.cos(t+np.arange(12))
            quat=[np.cos(.01*np.sin(t)),np.sin(.01*np.sin(t)),0,0]
            gyro=[.02*np.cos(t),.01*np.sin(t),0];acc=[.2*np.sin(t),.1*np.cos(t),9.81]
            for r in runtimes:r.observe_low(ns,q,dq);r.observe_imu(ns,quat,gyro,acc)
            if i < 3 or i%3:continue
            outputs=[]
            for r in runtimes:
                r.prepare(ns+197000,None,None,0.,5.+t)
                qr,vr,d=r.controller.step(np.zeros(13),quat,0.,.006)
                outputs.append(np.r_[qr,vr,d['raw_mpc_ddq_rad_s2'],d['tau_ff_candidate_nm']])
            diffs.append(float(np.max(abs(outputs[0]-outputs[1]))))
    finally:
        for r in runtimes:r.close()
    result=dict(samples=len(diffs),max_absolute_command_difference=max(diffs),passed=max(diffs)==0,
                original_runtime_snapshot='hardware_before',hardware_output=False)
    dest=ROOT/'evaluation/acceleration_pid_refinement_20261012/baseline_parity.json'
    dest.write_text(json.dumps(result,indent=2)+'\n');print(result)
    assert result['passed']


if __name__=='__main__':main()
