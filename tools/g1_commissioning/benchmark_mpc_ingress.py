#!/usr/bin/env python3
"""Offline MPC timing with real SDK IDL decoding and concurrent callbacks.

No DDS participant, RPC or publisher is constructed. Synthetic stationary
feedback does NOT respond to the computed commands. This measures contention,
not hardware tracking or end-to-end network delay. Guards are evaluated and
counted without sending any command, so a slow sample does not hide the tail.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from unittest.mock import patch

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from g1_walk_mpc import MpcRuntime, MpcJournal, FIELD_TORQUE_CONFIG
from g1_walk_pid import Streams, Interlock, EXPECTED_TARGET_Q
from hardware_pid_control import ARM_MOTOR_INDICES
from mpc_host import ControlThreadScope, summarize_timing, percentiles
from pid_timing import PeriodicClock


def run(output, *, cpu=7, rt_priority=0, switch_ms=5., threaded=True, duration=21., isolated=False):
    from unitree_sdk2py.idl.default import (unitree_hg_msg_dds__LowState_,
        unitree_hg_msg_dds__IMUState_, unitree_hg_msg_dds__LowCmd_)
    scope = ControlThreadScope(cpu, rt_priority)
    original_switch = sys.getswitchinterval()
    gc_enabled = gc.isenabled()
    runtime = journal = None
    workers = []
    stop = threading.Event()
    worker_errors, callback_rows, rows = [], [], []
    root=Path(__file__).resolve().parents[2]
    sources=[*Path(__file__).parent.glob('*.py'),root/'arm_mpc.py',
             root/'robot_model_backend/cpp_rnea_backend.py',root/'cpp/g1_arm_delay/delay.cpp',
             root/'build/g1_arm_delay/libg1_arm_delay.so',root/'configs/hardware_mpc_torque_field.yaml']
    source_hash = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    try:
        scope.prepare_workers()
        runtime_type=MpcRuntime;options={}
        if isolated:
            from mpc_compute_process import ProcessMpcRuntime
            runtime_type=ProcessMpcRuntime
            options=dict(compute_cpu=cpu,compute_priority=rt_priority,compute_affinity=scope.affinity)
        runtime = runtime_type(predictor_mode='hold_current', stationary=True,
            torque_config=FIELD_TORQUE_CONFIG, assumed_command_delay_s=.006, field_trial=True,**options)
        runtime.host_scope=scope
        journal = MpcJournal(output)
        runtime.journal = journal
        crc = runtime.create_crc()
        streams = Streams(journal, Interlock(), crc, observer=runtime)
        low = unitree_hg_msg_dds__LowState_()
        low.mode_machine = 4
        for i, q in zip(ARM_MOTOR_INDICES, EXPECTED_TARGET_Q):
            low.motor_state[i].q = float(q)
        low.imu_state.quaternion = [1., 0., 0., 0.]
        low.imu_state.accelerometer = [0., 0., 9.81]
        low.crc = crc.Crc(low)
        imu = unitree_hg_msg_dds__IMUState_()
        imu.quaternion = [1., 0., 0., 0.]
        imu.accelerometer = [0., 0., 9.81]
        payloads = [(type(low), low.serialize(), streams.low_callback),
                    (type(imu), imu.serialize(), streams.imu_callback)]
        profile = dict(target_q_array=EXPECTED_TARGET_Q, kp_array=np.r_[np.full(11,20.),0,0],
                       kd_array=np.r_[np.ones(11),0,0], q_offset_limit_deg_array=np.full(5,5.))
        plan = runtime.create_plan(EXPECTED_TARGET_Q, profile)

        def receive_once(item):
            cls, payload, callback = item
            begin, cpu_begin = time.perf_counter_ns(), time.thread_time_ns()
            callback(cls.deserialize(payload))
            callback_rows.append(dict(topic=cls.__name__, wall_ms=(time.perf_counter_ns()-begin)*1e-6,
                                      cpu_ms=(time.thread_time_ns()-cpu_begin)*1e-6))

        def receive_loop(item):
            try:
                clock = PeriodicClock(time.monotonic_ns(), .002)
                while not stop.is_set():
                    clock.wait()
                    receive_once(item)
                    clock.advance(time.monotonic_ns())
            except Exception as exc:
                worker_errors.append(repr(exc))
                stop.set()

        for item in payloads:
            receive_once(item)  # initialize serially before competing threads
        runtime.prepare_startup()
        if threaded:
            for item in payloads:
                worker = threading.Thread(target=receive_loop, args=(item,), daemon=True)
                worker.start(); workers.append(worker)
        host = runtime.enter_control_thread(cpu)
        sys.setswitchinterval(switch_ms*.001)
        host['python_thread_switch_interval_ms'] = sys.getswitchinterval()*1000
        epoch = time.monotonic_ns()
        streams.set_epoch(epoch)
        runtime.set_epoch(epoch)
        clock = PeriodicClock(time.monotonic_ns(), .006)
        previous = None
        last_frame = None
        release_start=None
        while True:
            clock.wait()
            begin, cpu_begin = time.monotonic_ns(), time.thread_time_ns()
            task_s = (begin-epoch)*1e-9
            if task_s >= duration or stop.is_set():
                break
            if not threaded:
                for item in payloads: receive_once(item)
            low_state, imu_state = streams.latest()
            actual_dt = .006 if previous is None else (begin-previous)*1e-9
            heading = streams.heading_current()
            if task_s >= 5 and not heading['reference_frozen']:
                streams.freeze_heading(); heading = streams.heading_current()
            yaw0 = heading['reference_rad']
            after_ingress = time.monotonic_ns()
            if task_s<18:
                runtime.prepare(after_ingress, low_state, imu_state, yaw0, task_s,
                                heading_frozen=heading['reference_frozen'])
            after_predict = time.monotonic_ns()
            if task_s<18:
                frame = plan.sample(task_s, low_state.q[list(ARM_MOTOR_INDICES)],
                    low_state.dq[list(ARM_MOTOR_INDICES)], imu_state.quaternion, yaw0, actual_dt)
            else:
                if release_start is None: release_start=task_s
                # Same parent-only release. No synthetic network/RPC success
                # is claimed; zero-speed acknowledgment waiting is excluded.
                frame=runtime.release_frame(task_s-release_start)
            after_control = time.monotonic_ns()
            packet = runtime.make_message(frame, low_state, unitree_hg_msg_dds__LowCmd_, crc)
            packet.serialize()  # same local CDR construction; NEVER Write
            prewrite = time.monotonic_ns()
            rejection = None
            try:
                runtime.check_before_write(frame, low_state, imu_state, begin, prewrite)
            except RuntimeError as exc:
                rejection = str(exc)
            # Offline hypothetical command history, including rejected times:
            # allows timing stress to continue; NOT the field send contract.
            runtime.accept_packet(frame, packet)
            journal.record(dict(schema='g1_mpc_contention_command_v1',task_elapsed_s=task_s,
                                output_sent=False, guard_rejection=rejection, **frame['diagnostics']))
            finished = time.monotonic_ns()
            row = dict(task_elapsed_s=task_s, stage=frame['stage'],
                ingress_ms=(after_ingress-begin)*1e-6, predictor_ms=(after_predict-after_ingress)*1e-6,
                controller_ms=(after_control-after_predict)*1e-6, packet_ms=(prewrite-after_control)*1e-6,
                prewrite_ms=(prewrite-begin)*1e-6, full_work_ms=(finished-begin)*1e-6,
                thread_cpu_ms=(time.thread_time_ns()-cpu_begin)*1e-6,
                wake_lateness_ms=(begin-clock.scheduled_ns)*1e-6,
                deadline_missed=finished>clock.scheduled_ns+clock.period_ns,
                guard_rejection=rejection, mapper_fallback=frame['diagnostics'].get('mapper',{}).get('fallback'))
            row['skipped_slots'] = clock.advance(finished)
            rows.append(row); previous, last_frame = begin, frame
            if frame['terminal']:break
        if worker_errors:
            raise RuntimeError(str(worker_errors))
        result = dict(schema='g1_mpc_ingress_benchmark_v1', host=host, threaded=threaded,isolated=isolated,
            source_sha256=source_hash, switch_ms=switch_ms,
            hardware_output=False, dds_initialized=False, source='synthetic_stationary_not_closed_loop',
            all_stages=summarize_timing(rows), active=summarize_timing([r for r in rows if 5<=r['task_elapsed_s']<18]),
            prewrite_guard_rejections=sum(r['guard_rejection'] is not None for r in rows),
            terminal=bool(last_frame and last_frame['terminal']),
            final_weight=None if last_frame is None else float(last_frame['weight']),
            callbacks_wall_ms=percentiles(r['wall_ms'] for r in callback_rows),
            callbacks_cpu_ms=percentiles(r['cpu_ms'] for r in callback_rows),
            limitations=['IDL decoding and actual callbacks included; no network receive, Write or RPC',
                         'rejected offline candidates committed only to continue timing, never real output',
                         'no hardware response, latency identification or hard realtime claim',
                         'parent thread CPU excludes child compute CPU when isolated',
                         'zero-speed RPC acknowledgement and full field health queries excluded'])
    finally:
        stop.set()
        for worker in workers: worker.join(2.)
        if journal is not None: journal.close()
        if runtime is not None: runtime.close()
        scope.restore()
        sys.setswitchinterval(original_switch)
        if gc_enabled: gc.enable()
    result.update(journal_failed=journal.failed.is_set(),journal_dropped=journal.dropped)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'timing.json').write_text(json.dumps(rows)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=7)
    parser.add_argument('--rt-priority',type=int,default=0)
    parser.add_argument('--switch-ms',type=float,default=5.)
    parser.add_argument('--serial',action='store_true')
    parser.add_argument('--isolated',action='store_true')
    parser.add_argument('--duration',type=float,default=24.)
    args=parser.parse_args()
    if not .05<=args.switch_ms<=5 or not 4<=args.duration<=30:
        parser.error('switch-ms must be .05..5 and duration 4..30 seconds')
    with patch.object(socket,'socket',side_effect=RuntimeError('offline benchmark forbids sockets')):
        result=run(args.output_dir,cpu=args.cpu,rt_priority=args.rt_priority,switch_ms=args.switch_ms,
                   threaded=not args.serial,duration=args.duration,isolated=args.isolated)
    print(json.dumps({k:result[k] for k in ('threaded','switch_ms','active','prewrite_guard_rejections',
                                          'journal_failed','journal_dropped')}),flush=True)


if __name__=='__main__':
    main()
