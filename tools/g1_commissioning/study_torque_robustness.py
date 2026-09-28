#!/usr/bin/env python3
"""Offline single-factor torque MPC failure attribution; never creates DDS.

Matched, payload, friction, observation delay, actuation delay, both delays,
and combined scenarios retain historical request error alongside active-command
error. Physical-time traces include the failure state and all 2 ms intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Set before importing NumPy/SciPy through the validation module.
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

from validate_measured_torque_mpc import (METHODS, ROOT, SCENARIOS, grid_steps,
    host_evidence, json_values, np, run_case)


def source_hashes(torque_config=None):
    names = ('study_torque_robustness.py','validate_measured_torque_mpc.py',
        'hardware_mpc_torque_control.py','hardware_torque_mapper.py',
        'hardware_arm_inverse_dynamics.py','hardware_mpc_control.py',
        'hardware_mpc_solver.py','endpoint_pose.py','hardware_mpc_delay_preview.py')
    files=[Path(__file__).with_name(name) for name in names]
    optional=Path(__file__).with_name('hardware_mpc_recovery.py')
    if optional.exists():
        files.append(optional)
    files += [ROOT/name for name in ('arm_mpc.py','kinematics_helper.py',
        'robot_model_backend/cpp_rnea_backend.py','configs/hardware_mpc.yaml',
        'configs/hardware_mpc_torque_preview.yaml','configs/g1.yaml',
        'build/right_arm_rnea/libright_arm_rnea.so')]
    if torque_config:
        files.append(Path(torque_config).resolve())
    # Include actual MJCF inputs, including nested includes and meshes only
    # insofar as they define the model XML; native library is hashed above.
    import xml.etree.ElementTree as ET
    from endpoint_pose import EndpointModel
    pending=[EndpointModel().xml.resolve()]
    seen=set()
    while pending:
        file=pending.pop()
        if file in seen:
            continue
        seen.add(file); files.append(file)
        tree=ET.parse(file)
        pending.extend((file.parent/node.attrib['file']).resolve()
                       for node in tree.iter('include'))
    return {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def save_plot(output_dir, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for name,(a,summary) in runs.items():
        if not len(a.get('interval_t',[])):
            continue
        label=name+(' (STOPPED)' if summary['status']!='complete' else '')
        axes[0,0].plot(a['t'],a['tilt'],label=label)
        axes[0,1].plot(a['physics_t'],np.rad2deg(a['physics_q_outer_margin'].min(axis=1)),label=label)
        axes[1,0].plot(a['interval_t'],np.linalg.norm(a['interval_current_request_vs_actual'],axis=1),label=label)
        valid=a['interval_active_seq']>=0
        axes[1,1].plot(a['interval_t'][valid],np.linalg.norm(
            a['interval_actually_active_command_desired_vs_actual'][valid],axis=1),label=label)
    labels=('bottle tilt (deg)','minimum outer position margin (deg)',
            'actual - current request norm (rad/s²)',
            'actual - active command desired norm (rad/s²)')
    for ax,label in zip(axes.flat,labels):
        ax.set_ylabel(label);ax.set_xlabel('physical time (s)');ax.grid(True)
    axes[0,1].axhline(0,color='black',linewidth=.8)
    axes[0,0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(output_dir/'comparison.png',dpi=160);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=3.)
    parser.add_argument('--cpu',type=int,default=4)
    parser.add_argument('--method',choices=METHODS,default='measured_mapper')
    parser.add_argument('--scenarios',nargs='+',choices=tuple(SCENARIOS),default=list(SCENARIOS))
    parser.add_argument('--torque-config',type=Path)
    parser.add_argument('--assumed-command-delay-ms',type=float,
        help='Explicit offline predictor assumption, NOT taken from the plant; omit to disable compensation')
    args=parser.parse_args(argv)
    if not .1<=args.duration<=20:
        parser.error('duration must be .1..20 s')
    grid_steps(args.duration,'duration')
    if args.cpu not in os.sched_getaffinity(0):
        parser.error('requested CPU is outside the current affinity')
    os.sched_setaffinity(0,{args.cpu})
    args.output_dir.mkdir(parents=True,exist_ok=False)
    start_hashes=source_hashes(args.torque_config)
    result=dict(schema='g1_torque_robustness_study_v1',dds=False,hardware_output=False,
        torque_output_authorized=False,duration_s=args.duration,cpu=args.cpu,method=args.method,
        torque_config=str(args.torque_config) if args.torque_config else None,
        scope='offline conditional five-joint arm dynamics, analytically prescribed moving torso',
        forecast=('exact analytic forecast at observed timestamp; causal nominal command-time propagation and horizon shift'
                  if args.assumed_command_delay_ms is not None else
                  'exact analytic forecast at the observed timestamp; no timestamp compensation'),
        assumed_command_delay_ms=args.assumed_command_delay_ms,
        ordering='record physical state, acquire delayed observation, solve if due, activate due commands, integrate 2ms',
        limits='signed margins: positive inside, zero touching, negative outside; touches use 1e-10 tolerance',
        metrics='historical 6ms request RMSE preserved; both named errors also scored at 2ms; active RMSE excludes seq=-1 hold',
        limitations=['not hardware or full locomotion evidence','2ms PD is a simulated assumption',
            'sampled first boundary contact is resolved to 2ms, not interpolated',
            'failed runs include their completed prefix and terminal observed physical state',
            'loop timing excludes plant stepping, observation transport and packet writes'],
        source_sha256_before=start_hashes,host=host_evidence(args.cpu),runs={})
    runs={}
    for name in args.scenarios:
        directory=args.output_dir/name
        a,summary=run_case(args.method,False,args.duration,scenario=name,
            failure_output_dir=directory,torque_config=args.torque_config,
            assumed_command_delay_s=(args.assumed_command_delay_ms*.001
                                    if args.assumed_command_delay_ms is not None else None))
        np.savez_compressed(args.output_dir/f'{name}.npz',**a)
        result['runs'][name]=summary;runs[name]=(a,summary)
        result['source_sha256_after']=source_hashes(args.torque_config)
        result['source_unchanged_during_run']=start_hashes==result['source_sha256_after']
        (args.output_dir/'summary.json').write_text(json.dumps(json_values(result),indent=2)+'\n')
        print(json.dumps(json_values(dict(scenario=name,status=summary['status'],
            failure=summary['failure'],scored_until_s=summary.get('physical_state_scored_until_s'),
            historical_request_rmse=summary.get('acceleration_tracking_rmse'),
            tracking_2ms=summary.get('tracking_2ms'),
            first_contact_2ms=summary.get('first_contact_2ms'),
            saturated_intervals=summary.get('saturated_2ms_intervals')))),flush=True)
    save_plot(args.output_dir,runs)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
