#!/usr/bin/env python3
"""Reproducible moving-base arm CLOSED-LOOP migration comparison; no SDK/DDS.

This isolates arm execution with prescribed torso motion, not whole-G1 gait.
Compare legacy references, measured MPC + nominal ID, mapped measured MPC,
and the original uncondensed simulation QP with the SAME conditional mapper.
Perturbed plant trials expose mismatch/delay rather than validate hardware.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial.transform import Rotation
from arm_mpc import ArmMPCPolicy
from disturbance_types import DisturbanceInput, DisturbanceHorizon
from endpoint_pose import EndpointModel, ROOT
from hardware_arm_inverse_dynamics import RightArmInverseDynamics
from hardware_mpc_control import RightArmHardwareMpc, json_values
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc
from robot_model_backend.cpp_rnea_backend import CppRightArmRneaBackend
from g1_walk_pid import EXPECTED_TARGET_Q
from mpc_host import host_evidence


def failure_evidence(controller, q, dq, measured_q, measured_dq, t):
    """Independent feasibility LP; diagnostics only, never a control fallback."""
    from scipy.optimize import linprog
    p = controller.policy
    lower, upper, *_ = p._build_online_constraint_bounds(measured_q, measured_dq)
    lower[:p.nx] = upper[:p.nx] = np.r_[measured_q, measured_dq]
    matrix = p._A_cons.toarray()
    lo, hi = np.isfinite(lower), np.isfinite(upper)
    lp = linprog(np.zeros(p.num_variables),
        A_ub=np.r_[matrix[hi], -matrix[lo]], b_ub=np.r_[upper[hi], -lower[lo]],
        bounds=[(None, None)]*p.num_variables, method='highs')
    return json_values(dict(time_s=t, true_q_deg=np.rad2deg(q), true_dq=dq,
        observed_q_deg=np.rad2deg(measured_q), observed_dq=measured_dq,
        outer_q_bounds_deg=np.rad2deg(p.safety_joint_limits),
        independent_constraint_lp=dict(status=int(lp.status), success=bool(lp.success),
                                       message=str(lp.message)),
        diagnostics=controller.last_diagnostics))


def disturbance(t):
    # Ground-truth prescribed torso motion. An exact forecast intentionally
    # isolates execution; no predictor accuracy is claimed by these trials.
    phase = 2*np.pi*t
    roll, pitch = .025*np.sin(phase), .015*np.sin(phase*2)
    dr, dp = .025*2*np.pi*np.cos(phase), .015*4*np.pi*np.cos(phase*2)
    ddr, ddp = -.025*(2*np.pi)**2*np.sin(phase), -.015*(4*np.pi)**2*np.sin(phase*2)
    rot = Rotation.from_euler('yx',[pitch,roll]).as_matrix()  # Rx(roll) Ry(pitch)
    # Spatial omega for Rx(r)Ry(p): [dr, cos(r)dp, sin(r)dp].
    w=np.array([dr,np.cos(roll)*dp,np.sin(roll)*dp])
    alpha=np.array([ddr,np.cos(roll)*ddp-np.sin(roll)*dr*dp,
                    np.sin(roll)*ddp+np.cos(roll)*dr*dp])
    acc=np.array([.5*np.sin(phase),.4*np.cos(phase),.6*np.sin(2*phase)])
    return DisturbanceInput(acc,w,alpha,rot)


def horizon(t):
    nodes=tuple(disturbance(t+k*.006) for k in range(10))
    intervals=[]
    for k in range(9):
        values=[disturbance(t+k*.006+j*.002) for j in range(3)]
        intervals.append(DisturbanceInput(np.mean([v.acc_world for v in values],axis=0),
            np.mean([v.omega_world for v in values],axis=0),
            np.mean([v.alpha_world for v in values],axis=0),values[0].rot_world_body))
    return DisturbanceHorizon(nodes,tuple(intervals))


def simulation_policy(c):
    config=c.config
    keys=("q_ee_acc","q_ee_alpha","q_ee_omega","q_gravity","q_posture",
          "q_vel","r_ddq","terminal_scale")
    return ArmMPCPolicy(c.nominal,control_dt=.006,horizon=9,
        joint_limits=np.column_stack((c.minimum,c.maximum)),
        joint_limit_margin=c.policy.joint_limit_margin,max_dq=c.max_dq,max_ddq=c.max_ddq,
        reg=config['regularization'],**{k:config[k] for k in keys},
        solver_eps_abs=1e-7,solver_eps_rel=1e-7,solver_max_iter=20000,solver_time_limit=.1)


def run_case(method, perturbed, duration):
    measured=method!='legacy_reference'
    c=(RightArmMeasuredTorqueMpc if measured else RightArmHardwareMpc)(EXPECTED_TARGET_Q[5:10])
    # This test measures wall time, but does not discard useful dynamics data
    # due to host scheduling. The real preview keeps its original QP deadline.
    c.policy.solver_time_limit=.1
    if method=='simulation_qp_mapper':
        c.policy=simulation_policy(c)
    if measured:
        # warmup inherited method assumes condensed solver; sim policy gets
        # explicit identical warmup without changing its solver implementation.
        for _ in range(5):
            c.set_disturbance_horizon(horizon(0))
            c.step(EXPECTED_TARGET_Q,[1,0,0,0],0,.006)
        c.reset()
    else:c.warmup(EXPECTED_TARGET_Q,[1,0,0,0],count=5)
    plant_model=EndpointModel()
    if perturbed:
        bottle=plant_model.model.body('right_bottle').id
        plant_model.model.body_mass[bottle] *= 1.2
        plant_model.model.body_inertia[bottle] *= 1.2
        # Change only plant parameters; controller retains nominal XML.
    backend=CppRightArmRneaBackend(plant_model.xml,library_path=ROOT/'build/right_arm_rnea/libright_arm_rnea.so')
    plant=RightArmInverseDynamics(plant_model,backend)
    q=EXPECTED_TARGET_Q[5:10].copy();dq=np.zeros(5)
    slots=EXPECTED_TARGET_Q.copy()
    sample_delay=2 if perturbed else 0  # 4 ms delayed observation
    actuation_delay=.006 if perturbed else 0.
    observations=deque(maxlen=20); pending=deque()
    _,bias=plant.linear_dynamics(q,dq,disturbance(0))
    active=(bias.copy(),q.copy(),np.zeros(5))
    rows=[]; status='complete';reason=None;failure_state=None;desired=np.zeros(5);last_diag={}
    max_plant_qacc = 0.
    metadata = c.metadata.copy()
    try:
        for index in range(round(duration/.002)):
            t=index*.002; base=disturbance(t)
            observations.append((q.copy(),dq.copy(),t))
            measured_q,measured_dq,stamp=observations[max(0,len(observations)-1-sample_delay)]
            if index%3==0:
                slots[5:10]=measured_q
                c.set_measured_dq(measured_dq)
                h=horizon(stamp)
                c.set_disturbance_horizon(h)
                quat=Rotation.from_matrix(h.nodes[0].rot_world_body).as_quat(scalar_first=True)
                started=time.perf_counter_ns()
                qref,dqref,diag=c.step(slots,quat,0,.006)
                elapsed=(time.perf_counter_ns()-started)*1e-6
                last_diag=diag
                desired=np.asarray(diag['raw_mpc_ddq_rad_s2'])
                if method=='legacy_reference':ff=np.zeros(5)
                elif method=='nominal_inverse':ff=np.asarray(diag['tau_nominal_ff_nm'])
                else:ff=np.asarray(diag['tau_ff_candidate_nm'])
                pending.append((t+actuation_delay,ff.copy(),qref.copy(),dqref.copy()))
            while pending and pending[0][0]<=t+1e-12:
                _,ff,qref,dqref=pending.popleft();active=ff,qref,dqref
            ff,qref,dqref=active
            # Firmware-like PD updated at physical 2 ms in this test only.
            tau=np.clip(ff+20*(qref-q)+(dqref-dq),-25,25)
            mass,bias=plant.linear_dynamics(q,dq,base)
            friction=.03*dq+.015*np.tanh(dq/.02) if perturbed else np.zeros(5)
            accel=np.linalg.solve(mass,tau-bias-friction)
            max_plant_qacc=max(max_plant_qacc,float(np.max(np.abs(accel))))
            if index%3==0:
                slots[5:10]=q
                relative=plant_model.relative(slots)['right'][1]
                axis=base.rot_world_body@relative[:,2]
                tilt=float(np.rad2deg(np.arccos(np.clip(axis[2],-1,1))))
                # Measured joint/torso state with the true plant acceleration
                # evaluates the same endpoint task equations as MPC.
                plant_model.data.qpos[:]=plant_model.model.qpos0
                plant_model.data.qpos[plant_model.joint_addresses]=slots[:11]
                helpers=c.helper.build_helpers(plant_model.data,
                    disturbance_prediction=horizon(t).nodes,
                    interval_disturbance_prediction=horizon(t).intervals,include_kinematics_cache=False)
                terms=helpers.compute_mpc_terms(q,dq,base,base,True)
                endpoint_acc=terms['C_acc']@dq+terms['B_acc']@accel+terms['D_acc']
                rows.append(dict(t=t,q=q.copy(),dq=dq.copy(),desired=desired.copy(),actual=accel.copy(),
                    tau=tau.copy(),tilt=tilt,endpoint_acc=endpoint_acc,core_ms=elapsed,
                    model_ddq=last_diag.get('mapper',{}).get('checked_ddq_rad_s2',np.full(5,np.nan))))
            q += dq*.002+.5*accel*.002**2
            dq += accel*.002
            if not np.isfinite(q).all() or np.max(np.abs(dq))>20:
                raise RuntimeError('plant numerical/velocity divergence')
    except Exception as exc:
        status='failed'; reason=f'{type(exc).__name__}: {exc}'
        failure_state=failure_evidence(c,q,dq,measured_q,measured_dq,t)
    finally:
        c.close();backend.close()
    arrays={key:np.asarray([row[key] for row in rows]) for key in rows[0]} if rows else {}
    if rows:
        err=arrays['actual']-arrays['desired']
        summary=dict(status=status,failure=reason,failure_state=failure_state,samples=len(rows),
            scored_until_s=float(arrays['t'][-1]),
            metrics_cover_requested_duration=status=='complete',
            acceleration_tracking_rmse=float(np.sqrt(np.mean(err**2))),
            tilt_rms_deg=float(np.sqrt(np.mean(arrays['tilt']**2))),
            endpoint_acceleration_rms_m_s2=float(np.sqrt(np.mean(np.sum(arrays['endpoint_acc']**2,axis=1)))),
            max_abs_qacc=float(np.max(np.abs(arrays['actual']))),
            max_abs_qacc_at_2ms=float(max_plant_qacc),
            core_ms={k:float(v) for k,v in zip(('p50','p95','p99','max'),
                np.r_[np.percentile(arrays['core_ms'],[50,95,99]),np.max(arrays['core_ms'])])})
    else:summary=dict(status=status,failure=reason,failure_state=failure_state,samples=0)
    summary['controller_metadata']=metadata
    return arrays,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=3.)
    parser.add_argument('--cpu',type=int,default=2)
    args=parser.parse_args()
    if not .1<=args.duration<=20:parser.error('duration must be .1..20 s')
    os.sched_setaffinity(0,{args.cpu})
    args.output_dir.mkdir(parents=True,exist_ok=False)
    summaries={};all_arrays={}
    for perturbed in (False,True):
        for method in ('legacy_reference','nominal_inverse','measured_mapper','simulation_qp_mapper'):
            name=f'{"perturbed" if perturbed else "matched"}_{method}'
            arrays,summary=run_case(method,perturbed,args.duration)
            np.savez_compressed(args.output_dir/f'{name}.npz',**arrays)
            summaries[name]=summary;all_arrays[name]=arrays
            print(name,json.dumps({k:v for k,v in summary.items()
                  if k not in {'controller_metadata','failure_state'}}),flush=True)
            if summary.get('failure_state'):
                state=summary['failure_state']
                print('failure_state',json.dumps({k:v for k,v in state.items() if k!='diagnostics'}),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for name,a in all_arrays.items():
        if not a:continue
        row=int(name.startswith('perturbed'))
        label=name.split('_',1)[1]
        if summaries[name]['status']!='complete':label+=' (STOPPED)'
        axes[row,0].plot(a['t'],a['tilt'],label=label)
        axes[row,1].plot(a['t'],np.linalg.norm(a['actual']-a['desired'],axis=1),label=label)
    for row in range(2):
        axes[row,0].set_ylabel(('mismatched' if row else 'matched')+' plant: tilt (deg)')
        axes[row,1].set_ylabel('joint acceleration error norm (rad/s²)')
        for ax in axes[row]:ax.grid(True);ax.legend(fontsize=8);ax.set_xlabel('time (s)')
    fig.tight_layout();fig.savefig(args.output_dir/'comparison.png',dpi=150);plt.close(fig)
    sources=[Path(__file__),*[Path(__file__).with_name(name) for name in
        ('hardware_mpc_torque_control.py','hardware_torque_mapper.py','hardware_arm_inverse_dynamics.py',
         'hardware_mpc_control.py','hardware_mpc_solver.py','endpoint_pose.py')],
        ROOT/'arm_mpc.py',ROOT/'kinematics_helper.py',ROOT/'robot_model_backend/cpp_rnea_backend.py',
        ROOT/'configs/hardware_mpc.yaml',ROOT/'configs/hardware_mpc_torque_preview.yaml']
    result=dict(schema='g1_measured_torque_closed_loop_v1',dds=False,hardware_output=False,
        scope='prescribed moving torso, closed-loop 5-DoF arm; no ground/contact or full locomotion',
        forecast='analytic exact torso forecast, isolates execution from learned-predictor error',
        perturbation='plant bottle mass/inertia +20%, small friction, observation 4ms and command 6ms delay',
        simulation_comparison='original uncondensed ArmMPCPolicy + same conditional mapper, not full contact mapper',
        metrics='entire simulated interval; failed runs cover only their completed prefix',
        limitations=['not the full hardware 5..18s task; starts at nominal posture and weight=1',
            'closed-loop comparison omits lifecycle torque slew; separate full-plan replay checks it',
            'legacy uses old reference governor/bounds; not an isolated mapper ablation',
            'nominal_inverse computes but discards mapper output; its timing is not bare ID cost',
            '2ms simulated PD is an assumption, not a measured firmware update rate',
            'perfect forecast, no IMU noise; CPU core timing here excludes ingress/packet/logging'],
        duration_s=args.duration,cpu=args.cpu,host=host_evidence(args.cpu),runs=summaries,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output_dir/'summary.json').write_text(json.dumps(json_values(result),indent=2)+'\n')


if __name__=='__main__':main()
