#!/usr/bin/env python3
"""Offline entry ablation seeded from a field pose; no DDS or robot output.

Prescribed upright stationary torso, 250 g model payload, conditional arm plant.
Weight blending, 4 ms observation delay and 6 ms application delay are assumed,
not identified. This tests entry feasibility, NOT whole-robot safety/tracking.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from disturbance_types import DisturbanceInput, DisturbanceHorizon
from g1_walk_mpc import MpcRuntime, FIELD_TORQUE_CONFIG
from g1_walk_pid import EXPECTED_TARGET_Q
from hardware_arm_inverse_dynamics import RightArmInverseDynamics
from hardware_pid_control import ARM_MOTOR_INDICES


def run(initial, velocity, start_s):
    runtime=MpcRuntime(predictor_mode='hold_current',stationary=True,field_trial=True,
                       torque_config=FIELD_TORQUE_CONFIG,assumed_command_delay_s=.006)
    rows=[];fault=None
    try:
        profile=dict(target_q_array=EXPECTED_TARGET_Q,kp_array=np.r_[np.full(11,20.),0,0],
                     kd_array=np.r_[np.ones(11),0,0],q_offset_limit_deg_array=np.full(5,5.))
        plan=runtime.create_plan(initial,profile);plan.mpc_start_s=start_s
        controller=runtime.controller
        # Pure numerical experiment: no wall deadline claim. All physical
        # packet, torque, rate and model acceleration envelopes are unchanged.
        controller.policy.solver_time_limit=.1
        plant=RightArmInverseDynamics(controller.model,controller.backend)
        base=DisturbanceInput(np.zeros(3),np.zeros(3),np.zeros(3),np.eye(3))
        horizon=DisturbanceHorizon((base,)*10,(base,)*9)
        q=initial[5:10].copy();v=velocity[5:10].copy()
        observations=deque();pending=deque();active=None
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        crc=runtime.create_crc()
        for step in range(3501):
            t=step*.002
            observations.append((t,q.copy(),v.copy()))
            while len(observations)>1 and observations[1][0]<=t-.004+1e-10:observations.popleft()
            if step%3==0:
                stamp,mq,mv=observations[0]
                slots=initial.copy();slots[5:10]=mq
                speeds=np.zeros(13);speeds[5:10]=mv
                controller.set_disturbance_horizon(horizon)
                plan.set_context(t,stamp,stamp)
                try:
                    frame=plan.sample(t,slots,speeds,[1,0,0,0],0.,.006)
                    fullq=np.zeros(35);fullv=np.zeros(35)
                    fullq[list(ARM_MOTOR_INDICES)]=slots;fullv[list(ARM_MOTOR_INDICES)]=speeds
                    state=SimpleNamespace(q=fullq,dq=fullv,mode_pr=0,mode_machine=4)
                    packet=runtime.make_message(frame,state,unitree_hg_msg_dds__LowCmd_,crc)
                    # Actual wire float32 rounding, no DDS initialization.
                    packet=type(packet).deserialize(packet.serialize())
                    runtime.accept_packet(frame,packet)
                    motors=[packet.motor_cmd[i] for i in range(22,27)]
                    command=dict(q=np.array([m.q for m in motors]),dq=np.array([m.dq for m in motors]),
                                 ff=np.array([m.tau for m in motors]),weight=packet.motor_cmd[29].q)
                    pending.append((t+.006,command))
                except Exception as exc:
                    fault=dict(time_s=t,reason=str(exc),diagnostics=controller.last_diagnostics)
                    break
                rows.append(dict(time_s=t,q_deg=np.rad2deg(q).tolist(),dq_rad_s=v.tolist(),
                                 weight=frame['weight'],mpc_active=frame['diagnostics'].get('mpc_active',False),
                                 fallback=frame['diagnostics'].get('mapper',{}).get('fallback')))
            while pending and pending[0][0]<=t+1e-10:_,active=pending.popleft()
            mass,bias=plant.linear_dynamics(q,v,base)
            total=bias if active is None else (active['weight']*(active['ff']+
                20.*(active['q']-q)+(active['dq']-v))+(1.-active['weight'])*bias)
            acc=np.linalg.solve(mass,total-bias)
            q=q+v*.002+.5*acc*.002**2;v=v+acc*.002
            if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('plant nonfinite')
        snapshots={str(target):min(rows,key=lambda r:abs(r['time_s']-target))
                   for target in (3.,5.,7.) if rows and rows[-1]['time_s']>=target-.006}
        return dict(mpc_start_s=start_s,completed_7s=fault is None,fault=fault,
                    snapshots=snapshots,rows=rows)
    finally:runtime.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.raw.open() as source:
        first=next(json.loads(line) for line in source if '"g1_hardware_mpc_command_v1"' in line)
    with patch.object(socket,'socket',side_effect=RuntimeError('offline entry study forbids sockets')):
        results=[run(np.array(first['q_measured_rad']),np.array(first['dq_measured_rad_s']),s) for s in (3.,5.)]
    result=dict(raw_source=str(args.raw),raw_sha256=hashlib.sha256(args.raw.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                hardware_output=False,solver_time_limit_s=.1,
                assumptions=['upright stationary prescribed torso; not recorded torso trajectory',
                  'nominal conditional MuJoCo arm plant, 250 g payload, no additional friction',
                  'unidentified weight blend with bias support; 4 ms observation, 6 ms application delay',
                  'not whole-body gait, real actuator calibration or timing validation'],results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=2)
    for r in results:print(json.dumps({k:v for k,v in r.items() if k!='rows'}))


if __name__=='__main__':main()
