#!/usr/bin/env python3
"""Replay recorded QP states and torque candidates offline, never open DDS.

Recorded states do not respond to the counterfactual commands. This diagnoses
the logged failure and verifies constraints; it does not simulate future walking.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings
import socket
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from disturbance_types import DisturbanceInput, DisturbanceHorizon
from hardware_mpc_delay_plan import IntervalHorizonClock
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc, load_torque_config
from hardware_mpc_control import json_values
from endpoint_pose import EndpointModel
from g1_walk_pid import EXPECTED_TARGET_Q


def recorded_horizon(d):
    """Reconstruct logged predictor SO(3) integration, then its time shift."""
    pred = d['predictor']
    if pred['rotation_policy'] != 'measured_node0_plus_world_left_trapezoidal_omega_SO3_integration':
        raise ValueError('unsupported recorded rotation policy')
    current, forecast = np.asarray(pred['current_y']), np.asarray(pred['forecast_y'])
    rotation = Rotation.from_euler('xyz', current[9:12]).as_matrix()
    nodes = [DisturbanceInput(current[:3],current[3:6],current[6:9],rotation)]
    intervals = []
    for future in forecast:
        omega = .5*(nodes[-1].omega_world+future[3:6])
        midpoint = Rotation.from_rotvec(omega*.003).as_matrix()@rotation
        rotation = Rotation.from_rotvec(omega*.006).as_matrix()@rotation
        intervals.append(DisturbanceInput(future[:3],omega,future[6:9],midpoint))
        nodes.append(DisturbanceInput(future[:3],future[3:6],future[6:9],rotation))
    delay = d['delay_preview']
    return IntervalHorizonClock(DisturbanceHorizon(tuple(nodes),tuple(intervals))).shifted(
        delay['command_time_s']-delay['forecast_s'])


def feasibility(policy, q, dq, omit_extra=False):
    lower, upper, *_ = policy._build_online_constraint_bounds(q,dq)
    lower[:policy.nx] = upper[:policy.nx] = np.r_[q,dq]
    end = policy.recovery_row_start if omit_extra else len(lower)
    matrix = policy._A_cons[:end].toarray()
    lo, hi = lower[:end], upper[:end]
    finite_lo, finite_hi = np.isfinite(lo), np.isfinite(hi)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
        result = linprog(np.zeros(policy.num_variables), A_ub=np.r_[matrix[finite_hi],-matrix[finite_lo]],
            b_ub=np.r_[hi[finite_hi],-lo[finite_lo]],bounds=(None,None),method='highs',options={'threads':1})
    residual = None if result.x is None else float(max(0.,np.max(lo-matrix@result.x),np.max(matrix@result.x-hi)))
    return dict(success=bool(result.success),status=int(result.status),max_constraint_residual=residual)


def short_model_replay(samples, initial, directory, slew_reference):
    """Nominal one-step forward-acceleration feedback, NOT hardware response.

    Apply the selected model acceleration until the next recorded query time.
    This intentionally small double-integrator plant tests the startup cause,
    not friction, contact dynamics, communication or a complete walking trial.
    """
    config=load_torque_config(directory/'torque_config.yaml')
    config['active_slew_reference']=slew_reference
    c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10],directory/'mpc_config.yaml',
        model=EndpointModel(directory/'controller_config.yaml'),torque_config=config)
    c.policy.solver_time_limit=.1
    q=np.asarray(initial['q_measured_rad'][5:10]).copy()
    dq=np.asarray(initial['dq_measured_rad_s'][5:10]).copy()
    total=np.asarray(initial['tau_total_estimated_at_feedback_nm'])
    if initial.get('mpc_active'):
        previous_bias=c.inverse.linear_dynamics(np.asarray(initial['command_time_q_rad']),
            np.asarray(initial['command_time_dq_rad_s']),recorded_horizon(initial).nodes[0])[1]
    else:
        previous_bias=np.asarray(initial['inverse_dynamics']['tau_model_nm'])
    records=[]
    try:
        for index,(d,_,kind) in enumerate(samples):
            slots=np.asarray(initial['q_measured_rad']).copy();slots[5:10]=q
            delta=config['transition_rate_nm_s']*min(float(d['feedback_dt_s']),.006)
            c._previous_total=total.copy()
            c._next_total_bounds=(total-delta,total+delta)
            c._next_slew_bias=previous_bias.copy()
            c.set_measured_dq(dq);c.set_disturbance_horizon(recorded_horizon(d),d['predictor'])
            try:
                _,_,diag=c.step(slots,[1,0,0,0],0.,d['feedback_dt_s'])
            except Exception as exc:
                records.append(dict(index=index,success=False,reason=str(exc),diagnostics=c.last_diagnostics))
                break
            dt=(samples[index+1][0]['delay_preview']['command_time_s']-d['delay_preview']['command_time_s']
                if index+1<len(samples) else 0.)
            if not 0<=dt<=.03:
                raise ValueError('invalid recorded command-time interval')
            acc=np.asarray(diag['mapper']['checked_ddq_rad_s2'])
            new_total=np.asarray(diag['tau_total_estimated_at_feedback_nm'])
            bias=np.asarray(diag['torque_model_bias_nm'])
            residual_change=(new_total-bias)-(total-previous_bias)
            records.append(dict(index=index,success=True,diagnostics=diag,model_q_rad=q.copy(),
                model_dq_rad_s=dq.copy(),propagation_s=dt,total_step_nm=new_total-total,
                bias_relative_step_nm=residual_change,step_bound_nm=delta))
            q=q+dt*dq+.5*dt**2*acc;dq=dq+dt*acc
            total,previous_bias=new_total,bias
        successful=[r for r in records if r['success']]
        step_key='bias_relative_step_nm' if slew_reference=='model_bias_relative' else 'total_step_nm'
        checks=dict(
            rate_bound=all(np.all(np.abs(r[step_key])<=r['step_bound_nm']+1e-8) for r in successful),
            absolute_total=all(np.all(np.abs(r['diagnostics']['tau_total_estimated_at_feedback_nm'])
                                     <=c.mapper.limit+1e-8) for r in successful),
            forward_acceleration=all(np.max(np.abs(r['diagnostics']['mapper']['checked_ddq_rad_s2']))
                                     <=c.mapper.acc_limit+1e-8 for r in successful),
            qp_solved=all(r['diagnostics']['mpc']['solved'] and not r['diagnostics']['mpc']['fallback_used']
                          for r in successful))
        if not all(checks.values()):
            raise ValueError(f'short model constraint audit failed: {checks}')
        return dict(completed=len(records)==len(samples) and all(r['success'] for r in records),
            independent_output_checks=checks,
            records=records,final_q_rad=q,final_dq_rad_s=dq,
            scope='short nominal forward-acceleration integrator; not full dynamics or field validation')
    finally:
        c.close()


def replay(raw, output, stationary_reference=None):
    if output.exists():
        raise FileExistsError(output)
    directory = raw.parent
    samples, lows, fault, previous, session = [], [], None, None, None
    with raw.open() as stream:
        for line in stream:
            r = json.loads(line)
            if r.get('event') == 'session_start':
                session = r
            if r.get('schema') == 'g1_mpc_predictor_low_v1':
                lows.append(r)
            if r.get('event') == 'dds_write' and r.get('task_elapsed_s') is not None and fault is None:
                if r.get('mpc_active'):
                    samples.append((r,previous,'successful_write'))
                previous = r
            if r.get('event') == 'controller_fault_detail':
                fault = r['diagnostics']
                samples.append((fault,previous,'failed_solve'))
    if fault is None or previous is None or session is None:
        raise ValueError('need a captured controller failure and preceding successful output')
    variants = {}
    for omit_envelope in (False,True):
        config = load_torque_config(directory/'torque_config.yaml')
        config['active_slew_reference'] = 'total'  # reproduce the captured old controller
        c = RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10],directory/'mpc_config.yaml',
             model=EndpointModel(directory/'controller_config.yaml'),torque_config=config)
        # Offline only: isolate mathematical failure from cold-start OS jitter.
        c.policy.solver_time_limit = .1
        if omit_envelope:
            # Diagnostic ablation ONLY in this output-absent program. Never
            # change the field config or interpret success as permission.
            c.policy._l_template[c.policy.recovery_row_start:] = -np.inf
            c.policy._u_template[c.policy.recovery_row_start:] = np.inf
        records = []
        try:
            for d, prev, kind in samples:
                q, dq = np.asarray(d['command_time_q_rad']),np.asarray(d['command_time_dq_rad_s'])
                slots = np.asarray(d.get('q_measured_rad',prev['q_measured_rad'])).copy()
                slots[5:10] = q
                last_total = np.asarray(prev['tau_total_estimated_at_feedback_nm'])
                delta = config['transition_rate_nm_s']*min(float(d['feedback_dt_s']),.006)
                c._previous_total = last_total.copy()
                c._next_total_bounds = (last_total-delta,last_total+delta)
                c.set_measured_dq(dq)
                c.set_disturbance_horizon(recorded_horizon(d),d['predictor'])
                success, reason = True, None
                try:
                    c.step(slots,[1,0,0,0],0.,d['feedback_dt_s'])
                except Exception as exc:
                    success, reason = False,str(exc)
                diag = c.last_diagnostics
                records.append(dict(kind=kind,sequence=d.get('sequence'),success=success,reason=reason,
                    diagnostics=diag, prior_recorded_total_nm=last_total, torque_step_bound_nm=delta))
            q, dq = np.asarray(fault['command_time_q_rad']),np.asarray(fault['command_time_dq_rad_s'])
            lps = dict(predicted=feasibility(c.policy,q,dq),
                measured=feasibility(c.policy,np.asarray(fault['raw_observed_q_rad']),np.asarray(fault['raw_observed_dq_rad_s'])),
                predicted_without_extra_envelope=feasibility(c.policy,q,dq,True))
            mass,bias=c.inverse.linear_dynamics(q,dq,recorded_horizon(fault).nodes[0])
            gain=np.linalg.inv(mass)
            delta=config['transition_rate_nm_s']*min(float(fault['feedback_dt_s']),.006)
            last_total=np.asarray(previous['tau_total_estimated_at_feedback_nm'])
            lo=np.maximum(last_total-delta,-c.mapper.limit);hi=np.minimum(last_total+delta,c.mapper.limit)
            min_acc=np.where(gain>=0,gain*lo,gain*hi).sum(axis=1)-gain@bias
            variants['without_extra_envelope' if omit_envelope else 'original'] = dict(
                records=records,independent_full_state_lp=lps,
                old_slew_box_componentwise_min_acc_rad_s2=min_acc,
                fault_model_gain=gain,fault_model_bias_nm=bias,old_torque_lower_nm=lo,old_torque_upper_nm=hi)
        finally:
            c.close()
    # The delay clock has its OWN origin, not task_epoch. Recover it from an
    # actual command snapshot before comparing the later received measurement.
    origin = previous['state_received_monotonic_ns']-round(previous['delay_preview']['observed_s']*1e9)
    target = origin+round(fault['delay_preview']['command_time_s']*1e9)
    nearby = sorted(lows,key=lambda r:abs(r['received_monotonic_ns']-target))[:2]
    temporal = dict(command_time_monotonic_ns=target,neighbouring_received_states=[dict(
        offset_ms=(r['received_monotonic_ns']-target)*1e-6,
        q_rad=r['q_rad'][22:27],dq_rad_s=r['dq_rad_s'][22:27]) for r in nearby],
        predicted_q_rad=fault['command_time_q_rad'],predicted_dq_rad_s=fault['command_time_dq_rad_s'],
        note='receive-time comparison, not a measured motor application or sensor timestamp')
    starts={'simultaneous':samples[0][1]}
    if stationary_reference is not None:
        with stationary_reference.open() as stream:
            for line in stream:
                r=json.loads(line)
                if r.get('event')=='dds_write' and r.get('task_elapsed_s') is not None and 16<r['task_elapsed_s']<18:
                    starts['settled_reference']=r
        if 'settled_reference' not in starts:
            raise ValueError('missing stationary reference window')
    model_cases={name+'_'+mode:short_model_replay(samples,initial,directory,mode)
                 for name,initial in starts.items() for mode in ('total','model_bias_relative')}
    summary = json_values(dict(schema='g1_mpc_walk_start_investigation_v1',source=str(raw.resolve()),
        raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),variants=variants,delay_comparison=temporal,
        short_model_cases=model_cases,
        stationary_reference_sha256=None if stationary_reference is None else hashlib.sha256(stationary_reference.read_bytes()).hexdigest(),
        offline_solver_time_budget_s=.1,field_solver_time_budget_unchanged=True,
        dds_initialized=False,hardware_output=False,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),*[Path(__file__).with_name(name) for name in
                ('hardware_mpc_recovery.py','hardware_mpc_torque_control.py','hardware_torque_mapper.py')]]},
        limitation='recorded-state/counterfactual command check only; not closed-loop hardware or timing proof'))
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for label,result in variants.items():
        print('recorded-state variant',label,'LP',result['independent_full_state_lp'])
        for row in result['records']:
            print(row['kind'],row['sequence'],row['success'],row['reason'],
                  row['diagnostics'].get('raw_mpc_ddq_rad_s2'),
                  row['diagnostics'].get('mapper',{}).get('checked_ddq_rad_s2'))
    print('temporal',temporal)
    for label,result in model_cases.items():
        print('short_model',label,'completed',result['completed'],'steps',len(result['records']))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--stationary-reference',type=Path)
    args=parser.parse_args()
    with patch.object(socket,'socket',side_effect=RuntimeError('offline investigation forbids sockets')):
        replay(args.raw,args.output_dir,args.stationary_reference)
