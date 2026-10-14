#!/usr/bin/env python3
"""Offline geometry and reduced reference-servo tests, not a whole-robot sim.

The artificial plant has a first-order position response and fixed load sag.
It verifies correction sign, bounded references and continuous hold, not field success.
"""
import argparse
import json
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from hardware_dual_upright import CONFIG, MODEL, DualUprightRuntime
from hardware_pid_control import CONTROL_PERIOD_S, ARM_MOTOR_INDICES


def quaternion(rpy_deg):
    return Rotation.from_euler('xyz',rpy_deg,degrees=True).as_quat()[[3,0,1,2]]


def validate(config=CONFIG,model_config=MODEL):
    cases=[]
    with patch.object(socket,'socket',side_effect=RuntimeError('offline dual validation forbids network')):
        for name,rpy,sag_deg,sway in [
            ('upright',[0,0,0],0,False),
            ('standing_pitch14',[0,14,0],0,False),
            ('roll_minus8_pitch12_yaw65',[-8,12,65],0,False),
            ('roll8_pitch_minus12_yaw_minus110',[8,-12,-110],0,False),
            ('standing_with_fixed_load_sag',[3,12,40],3,False),
            ('slow_standing_sway',[0,10,25],2,True),
        ]:
            runtime=DualUprightRuntime(config,model_config)
            c=runtime.controller
            profile=dict(target_q_array=np.zeros(13),kp_array=np.r_[np.full(11,20.),0.,0.],kd_array=np.r_[np.ones(11),0.,0.])
            state=np.zeros(13);state_velocity=np.zeros(13)
            plan=runtime.create_plan(state,profile)
            sag=np.zeros(13);sag[[0,3,5,8]]=np.deg2rad(sag_deg)
            last_q=last_dq=None
            tilts=[];times=[];started=time.perf_counter()
            max_reference_speed=max_reference_acceleration=0.
            for k in range(5001):
                t=k*CONTROL_PERIOD_S
                angles=np.asarray(rpy,dtype=float)
                if sway:
                    angles += [1.5*np.sin(2*np.pi*.15*t),2*np.sin(2*np.pi*.12*t),0]
                qimu=quaternion(angles)
                tick=time.perf_counter()
                frame=plan.sample(t,state,state_velocity,qimu,0.,CONTROL_PERIOD_S)
                times.append((time.perf_counter()-tick)*1e3)
                if frame['diagnostics'].get('dual_upright_active'):
                    low=SimpleNamespace(q=np.zeros(35),dq=np.zeros(35))
                    low.q[list(ARM_MOTOR_INDICES)]=state
                    low.dq[list(ARM_MOTOR_INDICES)]=state_velocity
                    runtime.check_before_write(frame,low,None,0,1_000_000)
                    q,dq=frame['q_rad'][:10],frame['dq_rad_s'][:10]
                    assert np.all(q>=c.minimum-1e-12) and np.all(q<=c.maximum+1e-12)
                    max_reference_speed=max(max_reference_speed,float(np.max(abs(dq))))
                    if last_dq is not None:
                        max_reference_acceleration=max(max_reference_acceleration,float(np.max(abs(dq-last_dq))/CONTROL_PERIOD_S))
                        np.testing.assert_allclose(q-last_q,dq*CONTROL_PERIOD_S,atol=1e-12)
                    last_q,last_dq=q.copy(),dq.copy()
                    if 26<=t<30:
                        tilts.append([frame['diagnostics'][s]['tilt_deg'] for s in ('left','right')])
                # Explicitly a reduced artificial servo, not MuJoCo dynamics.
                state_velocity=(frame['q_rad']-sag-state)/.08
                state += CONTROL_PERIOD_S*state_velocity
                state[10:]=0.
            limits=c.config
            assert max_reference_speed<=limits['max_reference_speed_rad_s']+1e-9
            assert max_reference_acceleration<=limits['max_reference_acceleration_rad_s2']+1e-7
            assert not frame['terminal'] and frame['weight']==1.
            assert frame['stage']=='stationary_upright'
            tail=np.asarray(tilts)
            threshold=2. if sway else .25
            success=bool(np.max(tail.mean(axis=0))<threshold)
            cases.append(dict(name=name,passed=success,body_rpy_deg=rpy,
                tail_mean_tilt_deg=tail.mean(axis=0).tolist(),tail_max_tilt_deg=tail.max(axis=0).tolist(),
                max_reference_speed_rad_s=max_reference_speed,
                max_reference_acceleration_rad_s2=max_reference_acceleration,
                final_weight=frame['weight'],continuous_hold_verified=True,
                compute_p99_ms=float(np.percentile(times,99)),
                elapsed_s=time.perf_counter()-started))
    return dict(schema='g1_dual_upright_offline_validation_v1',passed=all(c['passed'] for c in cases),
        cases=cases,hardware_output=False,dds_initialized=False,field_validated=False,
        both_arm_prewrite_pd_envelopes_checked=True,
        plant='reduced first-order position servo plus fixed artificial sag, NOT whole-robot dynamics',
        limitations='No contact/dynamics/robot transport validation; runtime timing here excludes DDS and logging.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=validate()
    output=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as f: f.write(output)
    print(output)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
