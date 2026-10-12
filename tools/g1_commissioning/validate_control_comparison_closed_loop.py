#!/usr/bin/env python3
"""Conditional-arm MODEL check with recorded torso/leg inputs, never a robot.

Unlike open-loop replay, virtual q/dq respond to C's packets here. The plant is
the same uncalibrated conditional arm model, with no contacts, sensor noise or
arm-to-body response. This is an integration/stability check, not field proof.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from g1_walk_mpc import MpcRuntime
from hardware_acceleration_pid import CONFIG_PATH
from hardware_control_comparison import BASELINE_CONFIG,MODEL_CONFIG,TORQUE_CONFIG,TRIALS
from native_arm_delay import NativeArmDelay
from validate_mpc_learned import capture
from hardware_control_comparison import ROOT

def sampled_contacts(states):
    """Post-check right-arm poses; other joints at qpos0, no contact dynamics."""
    import mujoco
    m=mujoco.MjModel.from_xml_path(str(ROOT / "resources/g1_description/scene_payload75.xml"))
    d=mujoco.MjData(m)
    names=["right_shoulder_pitch_joint","right_shoulder_roll_joint",
           "right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint"]
    addresses=[m.jnt_qposadr[m.joint(n).id] for n in names]
    bodies={m.body("right_shoulder_pitch_link").id}
    for bid in range(m.nbody):
        if int(m.body_parentid[bid]) in bodies:bodies.add(bid)
    count=0;pairs=set()
    for state in states:
        d.qpos[:]=m.qpos0;d.qpos[addresses]=state[:5]
        mujoco.mj_fwdPosition(m,d);hit=False
        for c in d.contact:
            if c.dist<=0 and any(int(m.geom_bodyid[g]) in bodies for g in (c.geom1,c.geom2)):
                hit=True
                pairs.add(tuple(m.body(int(m.geom_bodyid[g])).name for g in (c.geom1,c.geom2)))
        count+=hit
    return dict(frames=count,pairs=sorted(pairs),other_joints_at_qpos0=True,
                exhaustive_collision_validation=False)

from control_comparison_evidence import DEFAULT_SOURCES,sha256_file


def check(data, delay_s, trial_id="T03"):
    spec=TRIALS[trial_id]
    commands,low,imu,timing=data
    r=MpcRuntime(config=spec["config"],model_config=MODEL_CONFIG,torque_config=TORQUE_CONFIG,
        zero_arm_neutral=True,align_accel_forecast=spec["aligned"],
        acceleration_pid_config=spec["pid_config"],field_trial=True)
    native=NativeArmDelay(r.controller.inverse)
    profile=dict(target_q_array=r.target_q,kp_array=np.r_[np.full(11,20.),0,0],
                 kd_array=np.r_[np.ones(11),0,0],q_offset_limit_deg_array=np.full(5,5.))
    plan=r.create_plan(np.array(commands[0]['q_measured_rad']),profile)
    epoch=timing[commands[0]['sequence']]-round(commands[0]['task_elapsed_s']*1e9)
    anchor=next(c['predictor']['anchor_monotonic_ns'] for c in commands if 'predictor' in c)
    epoch=anchor-round((anchor-epoch)/2_000_000)*2_000_000
    r.predictor.set_grid_origin(epoch)
    li=ii=0;previous_t=None;active=None;pending=deque();previous_base=None
    q=dq=None;states=[];error=None
    try:
        for c in commands:
            t=c['task_elapsed_s']
            if t>=18:break
            now=c.get('predictor',{}).get('query_monotonic_ns',max(timing[c['sequence']],c['state_received_monotonic_ns'],c['imu_received_monotonic_ns']))
            while li<len(low) and low[li][0]<=now:r.observe_low(*low[li]);li+=1
            while ii<len(imu) and imu[ii][0]<=now:r.observe_imu(*imu[ii]);ii+=1
            if t<2.99:continue
            if previous_t is not None:
                cursor=previous_t
                while cursor<t-1e-12:
                    while pending and pending[0][0]<=cursor+1e-12:_,active=pending.popleft()
                    end=min(t,cursor+.002,pending[0][0] if pending else t)
                    q,dq=native.predict(q,dq,[(cursor,end-cursor,active)],[previous_base],
                                        r.controller.torque_config,r.controller.mapper.limit)
                    cursor=end
            else:
                q=np.array(c['q_measured_rad'][5:10]);dq=np.array(c['dq_measured_rad_s'][5:10])
            r.prepare(now,None,None,c['yaw0_rad'],t,c['heading_reference_frozen'])
            previous_base=r.controller._horizon.nodes[0]
            slots=np.array(c['q_measured_rad']);speeds=np.array(c['dq_measured_rad_s'])
            slots[5:10]=q;speeds[5:10]=dq
            try:
                frame=plan.sample(t,slots,speeds,imu[ii-1][1],c['yaw0_rad'],.006)
            except Exception as exc:
                error=dict(task_s=t,error=str(exc),q_deg=np.rad2deg(q).tolist(),dq=dq.tolist());break
            packet=SimpleNamespace(ff=np.array(frame['diagnostics']['tau_ff_candidate_nm']),
                qref=np.array(frame['q_rad'][5:10]),dqref=np.array(frame['dq_rad_s'][5:10]),weight=frame['weight'])
            pending.append((t+delay_s,packet))
            states.append(np.r_[q,dq]);previous_t=t
    finally:
        native.close();r.close()
    states=np.array(states)
    contacts=sampled_contacts(states)
    return dict(passed=error is None and contacts["frames"]==0,
                sampled_geometry=contacts, delay_ms=delay_s*1000,samples=len(states),error=error,
                last_task_s=previous_t,q_absmax_deg=np.rad2deg(np.max(abs(states[:,:5]),axis=0)).tolist(),
                q_min_deg=np.rad2deg(states[:,:5].min(axis=0)).tolist(),
                q_max_deg=np.rad2deg(states[:,:5].max(axis=0)).tolist(),
                dq_absmax=np.max(abs(states[:,5:]),axis=0).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trial', choices=('T01','T02','T03','T06'), default='T03')
    a=p.parse_args()
    if a.output.exists():p.error('output already exists; choose a new path')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(hardware_output=False,conditional_model_only=True,
        config_sha256={str(p):sha256_file(p) for p in (BASELINE_CONFIG, MODEL_CONFIG, TORQUE_CONFIG, CONFIG_PATH, TRIALS[a.trial]['config'])},
        source_sha256=sha256_file(Path(__file__)),
        model_matches_controller=True,physical_performance_validated=False,results=[])
    with patch.object(socket,'socket',side_effect=RuntimeError('offline check forbids network')):
        for source in DEFAULT_SOURCES:
            data=capture(source)
            for delay in (0.,.004):
                result=dict(source=str(source),source_sha256=sha256_file(source),trial=a.trial,**check(data,delay,a.trial))
                report['results'].append(result)
                a.output.write_text(json.dumps(report,indent=2)+'\n');print(result,flush=True)
    return 0 if all(r['passed'] for r in report['results']) else 1


if __name__=='__main__':raise SystemExit(main())
