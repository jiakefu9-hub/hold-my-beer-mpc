#!/usr/bin/env python3
"""Causal recorded-input comparison; no DDS or physical closed-loop claim.

Replays the first complete MPC run through the frozen baseline, feedback-aware
hold-current, and feedback-aware learned controller. Uses recorded loop/receive
timestamps, all stages, real IDL/CRC serialization, and final packet checks.
Output contains every computed command; never fits a predictor on this run.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import orjson
from g1_walk_mpc import MpcRuntime, FIELD_TORQUE_CONFIG, LEARNED_MPC_CONFIG, LEARNED_TORQUE_CONFIG
from g1_walk_pid import ROOT, EXPECTED_TARGET_Q
from hardware_pid_control import ARM_MOTOR_INDICES
from hardware_mpc_control import json_values
from mpc_host import host_evidence


def capture(path):
    commands, low, imu, timing = [], [], [], {}
    with path.open('rb') as stream:
        for line in stream:
            row = orjson.loads(line); kind = row.get('schema')
            if kind == 'g1_hardware_mpc_command_v1':
                commands.append(row)
            elif kind == 'g1_mpc_predictor_low_v1':
                low.append((row['received_monotonic_ns'], np.asarray(row['q_rad']),
                            np.asarray(row['dq_rad_s'])))
            elif kind == 'g1_mpc_predictor_imu_v1':
                imu.append((row['received_monotonic_ns'], row['quaternion_wxyz'],
                            row['gyroscope_rad_s'], row['accelerometer_raw_m_s2']))
            elif kind == 'g1_mpc_timing_v1':
                timing[row['sequence']] = row['loop_begin_monotonic_ns']
    if not commands or commands[-1]['weight'] != 0 or commands[-1]['task_elapsed_s'] < 21:
        raise ValueError('source must include complete entry, 5..18s control and release')
    for rows in (low, imu):
        rows.sort(key=lambda r:r[0])
        if any(b[0] <= a[0] for a,b in zip(rows, rows[1:])):
            raise ValueError('duplicate/reversed source receive timestamps')
    if any(c['sequence'] not in timing for c in commands):
        raise ValueError('missing source cycle timestamp')
    return commands, low, imu, timing


def stats(values):
    x = np.asarray(values)
    return dict(count=len(x), mean=float(x.mean()), p50=float(np.median(x)),
                p99=float(np.percentile(x,99)), maximum=float(x.max())) if len(x) else None


def replay(data, output, name, cpu, *, math_only=False):
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    commands, low, imu, timing = data
    learned_variant = name != 'frozen_baseline'
    mode = 'learned_filtered' if name == 'learned' else 'hold_current'
    runtime = MpcRuntime(config=LEARNED_MPC_CONFIG if learned_variant else
                         ROOT/'configs/hardware_mpc_upright_baseline.yaml',
        torque_config=LEARNED_TORQUE_CONFIG if learned_variant else FIELD_TORQUE_CONFIG,
        predictor_mode=mode, field_trial=True, stationary=False)
    if math_only:
        runtime.controller.policy.solver_time_limit = .1
    initial = np.asarray(commands[0]['q_measured_rad'])
    profile = dict(target_q_array=EXPECTED_TARGET_Q, kp_array=np.r_[np.full(11,20.),0,0],
                   kd_array=np.r_[np.ones(11),0,0], q_offset_limit_deg_array=np.full(5,5.))
    plan = runtime.create_plan(initial, profile)
    crc = runtime.create_crc()
    epoch = timing[commands[0]['sequence']]-round(commands[0]['task_elapsed_s']*1e9)
    runtime.predictor.set_grid_origin(epoch)
    li = max(0,int(np.searchsorted([r[0] for r in low],epoch-500_000_000))-1)
    ii = max(0,int(np.searchsorted([r[0] for r in imu],epoch-500_000_000))-1)
    records, errors, active_times, predictor_times, core_times, packet_times = [], [], [], [], [], []
    closure_max = 0.; active_count = learned_count = 0; last_frame = None
    learned_qp = runtime.controller.policy
    cpu_before = set(os.sched_getaffinity(0)); gc_before=gc.isenabled()
    try:
        os.sched_setaffinity(0,{cpu}); gc.collect();gc.disable()
        with (output/f'{name}.jsonl').open('xb') as stream:
            for c in commands:
                t=c['task_elapsed_s']
                # A selected snapshot may arrive just after cycle wakeup. Use
                # its actual availability, never future samples/interpolation.
                now=max(timing[c['sequence']],c['state_received_monotonic_ns'],c['imu_received_monotonic_ns'])
                start=time.perf_counter_ns()
                while li<len(low) and low[li][0]<=now:
                    runtime.observe_low(*low[li]);li+=1
                while ii<len(imu) and imu[ii][0]<=now:
                    runtime.observe_imu(*imu[ii]);ii+=1
                before_prediction=time.perf_counter_ns()
                row=dict(sequence=c['sequence'],task_s=t,variant=name,query_received_ns=now)
                try:
                    if t < 18:
                        runtime.prepare(now,None,None,c['yaw0_rad'],t,c['heading_reference_frozen'])
                    after_prediction=time.perf_counter_ns()
                    if t < 18:
                        frame=plan.sample(t,np.asarray(c['q_measured_rad']),np.asarray(c['dq_measured_rad_s']),
                            imu[ii-1][1],c['yaw0_rad'],(c['control_actual_period_ms'] or 6.)*.001)
                    elif c['stage'] == 'normal_stop_wait':
                        frame=runtime.handback.apply(dict(stage=c['stage'],weight=1.,terminal=False,diagnostics={}))
                    else:
                        # Real field transport bypasses numerical Plan.sample
                        # after 18s: preserve the last sent PD law and fade it.
                        frame=runtime.release_frame(c.get('normal_release_elapsed_s',t-18.))
                    after_control=time.perf_counter_ns()
                    state=SimpleNamespace(q=low[li-1][1].copy(),dq=low[li-1][2].copy(),mode_pr=0,mode_machine=4)
                    state.q[list(ARM_MOTOR_INDICES)]=c['q_measured_rad']
                    state.dq[list(ARM_MOTOR_INDICES)]=c['dq_measured_rad_s']
                    packet=runtime.make_message(frame,state,unitree_hg_msg_dds__LowCmd_,crc)
                    packet.serialize();runtime.accept_packet(frame,packet)
                    after_packet=time.perf_counter_ns()
                    d=frame['diagnostics']
                    row.update(weight=frame['weight'],q_command_rad=frame['q_rad'],dq_command_rad_s=frame['dq_rad_s'],
                        q_measured_rad=c['q_measured_rad'],dq_measured_rad_s=c['dq_measured_rad_s'],
                        packet_right_tau=[packet.motor_cmd[i].tau for i in range(22,27)],
                        diagnostics=d, full_compute_ms=(after_packet-start)*1e-6,
                        predictor_ms=(after_prediction-before_prediction)*1e-6,
                        core_plan_ms=(after_control-after_prediction)*1e-6,
                        packet_ms=(after_packet-after_control)*1e-6)
                    if d.get('mpc_active'):
                        active_count+=1
                        learned_count+=d['predictor']['mode']=='learned_filtered'
                        pd=np.asarray(d['tau_pd_at_feedback_nm'])
                        np.testing.assert_allclose(np.asarray(d['tau_ff_candidate_nm'])+pd,
                                                   d['tau_total_estimated_at_feedback_nm'],atol=1e-10)
                        if learned_variant:
                            closure=float(np.max(abs(np.asarray(d['estimated_closed_loop_ddq_rad_s2'])-
                                                      d['raw_mpc_ddq_rad_s2'])))
                            closure_max=max(closure_max,closure)
                            if closure>1e-6:raise ValueError('planned acceleration != final torque acceleration')
                            if d['mapper']['candidate_count']!=0:raise ValueError('unexpected candidate search')
                        if 5<=t<18:
                            active_times.append(row['full_compute_ms']);predictor_times.append(row['predictor_ms'])
                            core_times.append(row['core_plan_ms']);packet_times.append(row['packet_ms'])
                        records.append((c['sequence'],d['tau_total_estimated_at_feedback_nm']))
                    last_frame=frame
                except Exception as exc:
                    row.update(error=str(exc),diagnostics=runtime.controller.last_diagnostics)
                    errors.append(dict(sequence=c['sequence'],task_s=t,error=str(exc)))
                stream.write(orjson.dumps(row,option=orjson.OPT_SERIALIZE_NUMPY,default=json_values)+b'\n')
                if errors:break  # do not reset and pretend a failed replay completed
    finally:
        runtime.close();os.sched_setaffinity(0,cpu_before)
        if gc_before:gc.enable()
    result=dict(status='passed' if not errors else 'failed',errors=errors,active_commands=active_count,
        learned_commands=learned_count,final_weight=None if last_frame is None else last_frame['weight'],
        variant_metadata=runtime.controller.metadata,
        planned_vs_final_acceleration_max_error_rad_s2=closure_max if learned_variant else None,
        full_compute_ms=stats(active_times),predictor_ms=stats(predictor_times),core_plan_ms=stats(core_times),
        packet_ms=stats(packet_times),over_6ms_count=sum(x>6 for x in active_times),
        solver_time_limit_s=learned_qp.solver_time_limit,
        raw_sha256=hashlib.sha256((output/f'{name}.jsonl').read_bytes()).hexdigest())
    return result,dict(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'evaluation/hardware_shadow/commissioning/'
                   'mpc_torque_walk_reboot_performance_20261007_172732/raw.jsonl')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cpu',type=int,default=2)
    p.add_argument('--math-only',action='store_true',help='explicit 100ms QP allowance; NOT a timing pass')
    args=p.parse_args();data=capture(args.source)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    result=dict(schema='g1_learned_mpc_validation_v1',hardware_output=False,dds_initialized=False,
        network_forbidden=True,source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        host=host_evidence(args.cpu),math_only=args.math_only,variants={},comparisons={})
    trajectories={}
    with patch.object(socket,'socket',side_effect=RuntimeError('offline validation forbids network')):
        for name in ('frozen_baseline','yaw_aware_hold','learned'):
            report,trajectories[name]=replay(data,args.output_dir,name,args.cpu,math_only=args.math_only)
            result['variants'][name]=report
            print(name,report['status'],report['active_commands'],report['full_compute_ms'],report['errors'],flush=True)
    for a,b in (('frozen_baseline','yaw_aware_hold'),('yaw_aware_hold','learned')):
        keys=sorted(trajectories[a].keys()&trajectories[b].keys())
        difference=np.array([np.asarray(trajectories[b][k])-trajectories[a][k] for k in keys])
        result['comparisons'][f'{a}_vs_{b}']=dict(paired_states=len(keys),
            per_joint_mean_abs_tau_change_nm=np.mean(abs(difference),axis=0).tolist() if keys else None,
            per_joint_max_abs_tau_change_nm=np.max(abs(difference),axis=0).tolist() if keys else None)
    paths=[Path(__file__),LEARNED_MPC_CONFIG,LEARNED_TORQUE_CONFIG,*[Path(__file__).with_name(name)
        for name in ('g1_walk_mpc.py','g1_walk_mpc_learned.py','hardware_mpc_learned.py',
                     'hardware_mpc_torque_control.py','hardware_mpc_solver.py','hardware_mpc_predictor.py')]]
    result['code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result['limitations']=['Recorded states do not respond to these new commands: not physical validation',
        'Logical receive-time replay, no real-time pacing or DDS, RPC, worker IPC, audit-IO timing included',
        'Host powersave/SCHED_OTHER measurements are not tomorrow CPU7/FIFO certification',
        'Only conditional arm model consistency checked, no measured torque calibration',
        'Learned forecasts are filtered H0 estimates, not exact raw landing impulses']
    (args.output_dir/'summary.json').write_text(json.dumps(json_values(result),indent=2,allow_nan=False)+'\n')
    return 0 if all(r['status']=='passed' for r in result['variants'].values()) else 1


if __name__=='__main__':raise SystemExit(main())
