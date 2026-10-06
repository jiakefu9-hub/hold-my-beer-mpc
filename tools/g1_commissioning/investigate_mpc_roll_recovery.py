#!/usr/bin/env python3
"""No-output replay of the 2026-10-06 roll rejection and bounded reentry.

Recorded-state replay is counterfactual, not closed-loop validation. The separate
nominal test integrates selected model acceleration; it is NOT a hardware plant.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc, load_torque_config
from hardware_mpc_control import json_values
from investigate_mpc_walk_start import recorded_horizon, feasibility
from endpoint_pose import EndpointModel
from g1_walk_pid import EXPECTED_TARGET_Q
from audit_measured_torque_replay import check_active, check_slew


def samples_from(raw):
    samples=[];previous=None;fault_seen=False
    for line in raw.open():
        r=json.loads(line)
        if r.get('event')=='dds_write':
            if r.get('mpc_active') and not fault_seen:samples.append((r,previous))
            previous=r
        if r.get('event')=='controller_fault_detail':
            samples.append((r['diagnostics'],previous));fault_seen=True
    if not fault_seen or not samples or samples[-1][1] is None:
        raise ValueError('need a captured failure with preceding command')
    return samples


def trial(raw,samples,*,new,nominal=False,planning=False,envelope=True,raw_state=False):
    folder=raw.parent
    conf=load_torque_config(folder/'torque_config.yaml')
    # Only these two new switches change; original torque/PD/slew limits remain.
    conf.update(recovery_envelope_enabled=envelope,recovery_reentry_enabled=new,
                enforce_mapper_state_envelope=new,
                planning_actuation_constraints_enabled=planning)
    c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10],folder/'mpc_config.yaml',
        model=EndpointModel(folder/'controller_config.yaml'),torque_config=conf)
    c.policy.solver_time_limit=.1  # isolate math from cold-start scheduling
    records=[];q=v=total=bias=None
    try:
        for i,(d,prev) in enumerate(samples):
            if not nominal or q is None:
                q=np.asarray(d['raw_observed_q_rad'] if raw_state else d['command_time_q_rad']).copy()
                v=np.asarray(d['raw_observed_dq_rad_s'] if raw_state else d['command_time_dq_rad_s']).copy()
                total=np.asarray(prev['tau_total_estimated_at_feedback_nm']).copy()
                bias=np.asarray(prev['torque_model_bias_nm']).copy()
            slots=np.asarray(d.get('q_measured_rad',prev['q_measured_rad'])).copy();slots[5:10]=q
            delta=conf['transition_rate_nm_s']*min(d['feedback_dt_s'],.006)
            c._previous_total=total.copy();c._next_total_bounds=(total-delta,total+delta)
            c._next_slew_bias=bias.copy();c.set_measured_dq(v)
            c.set_disturbance_horizon(recorded_horizon(d),d['predictor'])
            try:
                qr,vr,diag=c.step(slots,[1,0,0,0],0.,d['feedback_dt_s'])
            except Exception as exc:
                records.append(dict(index=i,sequence=d.get('sequence'),success=False,reason=str(exc),
                                    diagnostics=c.last_diagnostics))
                break
            selected=np.asarray(diag['tau_total_estimated_at_feedback_nm'])
            b=np.asarray(diag['torque_model_bias_nm']);a=np.asarray(diag['mapper']['checked_ddq_rad_s2'])
            residual_step=(selected-b)-(total-bias)
            if (np.any(np.abs(residual_step)>delta+1e-7)
                    or np.any(np.abs(selected)>np.asarray(conf['tau_abs_nm'])+1e-7)):
                raise ValueError('independent torque/slew check failed')
            next_q=q+.006*v+.5*.006**2*a;next_v=v+.006*a
            next_h=next_q+next_v/conf['recovery_rate_s_inv']
            # Audit a purely LOCAL preview packet. Never label this as a DDS
            # write or a physical torque response.
            audit=dict(diag,q_measured_rad=slots,dq_measured_rad_s=np.r_[np.zeros(5),v,np.zeros(3)],
                q_command_rad=np.r_[slots[:5],qr,slots[10:]],
                dq_command_rad_s=np.r_[np.zeros(5),vr,np.zeros(3)],
                offline_packet_right_tau_nm=diag['tau_ff_candidate_nm'],post_transition_ddq_rad_s2=a)
            check_active(audit,conf)
            check_slew(audit,dict(tau_total_estimated_at_feedback_nm=total,torque_model_bias_nm=bias),conf)
            if new:
                lo,hi=np.asarray(diag['final_acceleration_bounds_rad_s2'])
                reentry=diag['reentry']
                if (np.any(a<lo-1e-7) or np.any(a>hi+1e-7)
                        or np.any(next_q<np.deg2rad(conf['q_min_deg'])-1e-8)
                        or np.any(next_q>np.deg2rad(conf['q_max_deg'])+1e-8)
                        or np.any(np.abs(next_v)>conf['max_dq_rad_s']+1e-8)
                        or np.any(next_h<np.asarray(reentry['first_lower_rad'])-1e-8)
                        or np.any(next_h>np.asarray(reentry['first_upper_rad'])+1e-8)):
                    raise ValueError('independent final model-state check failed')
            records.append(dict(index=i,sequence=d.get('sequence'),success=True,
                diagnostics=diag,next_q_rad=next_q,next_dq_rad_s=next_v,next_h_rad=next_h,
                residual_step_nm=residual_step,independent_checks_passed=True))
            if nominal and i+1<len(samples):
                dt=samples[i+1][0]['delay_preview']['command_time_s']-d['delay_preview']['command_time_s']
                if not 0<dt<.03:raise ValueError('invalid model propagation interval')
                q=q+dt*v+.5*dt**2*a;v=v+dt*a
                total,bias=selected,b
        return dict(completed=len(records)==len(samples) and all(r['success'] for r in records),records=records,
            mode='nominal acceleration integrator' if nominal else 'recorded states / counterfactual outputs')
    finally:c.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw',type=Path);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    with patch.object(socket,'socket',side_effect=RuntimeError('offline analysis forbids sockets')):
        samples=samples_from(args.raw)
        results={}
        for new in (False,True):
            label='fixed' if new else 'original'
            results[label+'_fault_only']=trial(args.raw,samples[-1:],new=new)
            results[label+'_recorded']=trial(args.raw,samples,new=new)
            results[label+'_nominal']=trial(args.raw,samples,new=new,nominal=True)
        results['actuation_aware_recorded']=trial(args.raw,samples,new=True,planning=True)
        results['actuation_aware_nominal']=trial(args.raw,samples,new=True,planning=True,nominal=True)
        results['base_envelope_actuation_aware_recorded']=trial(
            args.raw,samples,new=False,planning=True,envelope=False)
        results['base_envelope_actuation_aware_nominal']=trial(
            args.raw,samples,new=False,planning=True,envelope=False,nominal=True)
        results['base_envelope_original_mapper_recorded']=trial(
            args.raw,samples,new=False,planning=False,envelope=False)
        results['base_envelope_actuation_aware_raw_recorded']=trial(
            args.raw,samples,new=False,planning=True,envelope=False,raw_state=True)
        # A separate nominal recovery from the ACTUAL failed state. Hold its
        # body forecast, not an invented continuation of the physical walk.
        recovery=[]
        for k in range(21):
            d=copy.deepcopy(samples[-1][0]);d['feedback_dt_s']=.006
            for key in ('command_time_s','observed_s','now_s','forecast_s'):
                d['delay_preview'][key]+=k*.006
            recovery.append((d,samples[-1][1]))
        results['fixed_fault_frozen_body_nominal']=trial(args.raw,recovery,new=True,nominal=True)
    result=json_values(dict(schema='g1_mpc_roll_recovery_investigation_v1',
        raw_sha256=hashlib.sha256(args.raw.read_bytes()).hexdigest(),raw=str(args.raw.resolve()),
        results=results,hardware_output=False,dds_initialized=False,offline_solver_time_budget_s=.1,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),*[Path(__file__).with_name(name) for name in
                ('hardware_mpc_recovery.py','hardware_mpc_torque_control.py','hardware_torque_mapper.py',
                 'investigate_mpc_walk_start.py','audit_measured_torque_replay.py')]]},
        limitations=['not physical torque/latency validation','nominal uses recorded body disturbance',
                     'short captured prefix only, not full walking','offline solver budget is not field timing']))
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,r in results.items():
        print(name,'completed',r['completed'],'steps',len(r['records']),
              'last_reason',r['records'][-1].get('reason'))


if __name__=='__main__':main()
