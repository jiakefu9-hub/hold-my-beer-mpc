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
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
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


PHYSICS_DT = .002
CONTROL_STEPS = 3
METHODS = ('legacy_reference', 'nominal_inverse', 'measured_mapper', 'simulation_qp_mapper')


@dataclass(frozen=True)
class Scenario:
    name: str = 'matched'
    payload_scale: float = 1.
    viscous_friction: float = 0.
    coulomb_friction: float = 0.
    observation_delay_s: float = 0.
    actuation_delay_s: float = 0.

    def __post_init__(self):
        for key in ('payload_scale', 'viscous_friction', 'coulomb_friction',
                    'observation_delay_s', 'actuation_delay_s'):
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0 or (key == 'payload_scale' and value == 0):
                raise ValueError(f'invalid scenario {key}')
        for key in ('observation_delay_s', 'actuation_delay_s'):
            grid_steps(getattr(self, key), key)


def grid_steps(seconds, name):
    steps = round(seconds / PHYSICS_DT)
    if not np.isfinite(seconds) or seconds < 0 or not np.isclose(
            steps * PHYSICS_DT, seconds, atol=1e-12, rtol=0):
        raise ValueError(f'{name} must be a nonnegative multiple of 2 ms')
    return steps


SCENARIOS = {
    'matched': Scenario(),
    'payload_only': Scenario('payload_only', payload_scale=1.2),
    'friction_only': Scenario('friction_only', viscous_friction=.03, coulomb_friction=.015),
    'observation_only': Scenario('observation_only', observation_delay_s=.004),
    'actuation_only': Scenario('actuation_only', actuation_delay_s=.006),
    'both_delays': Scenario('both_delays', observation_delay_s=.004, actuation_delay_s=.006),
    'combined': Scenario('combined', 1.2, .03, .015, .004, .006),
}


def resolve_scenario(scenario=None, perturbed=False):
    # An explicit scenario takes precedence; old positional calls are unchanged.
    if scenario is None:
        return SCENARIOS['combined' if perturbed else 'matched']
    if isinstance(scenario, str):
        return SCENARIOS[scenario]
    if isinstance(scenario, dict):
        return Scenario(**scenario)
    if not isinstance(scenario, Scenario):
        raise TypeError('scenario must be a name, dict, or Scenario')
    return scenario


@dataclass(frozen=True)
class PendingCommand:
    apply_step: int
    seq: int
    issued_s: float
    observed_s: float
    ff: np.ndarray
    qref: np.ndarray
    dqref: np.ndarray
    desired_ddq: np.ndarray


def physical_margins(policy, q, dq):
    return dict(q_inner_margin=np.minimum(q-policy.joint_limits[:, 0],
                                         policy.joint_limits[:, 1]-q),
                q_outer_margin=np.minimum(q-policy.safety_joint_limits[:, 0],
                                         policy.safety_joint_limits[:, 1]-q),
                dq_margin=policy.max_dq-np.abs(dq))


def first_contact(times, margins, *, tolerance=1e-10):
    hits = np.argwhere(np.asarray(margins) <= tolerance)
    if not len(hits):
        return None
    row, joint = hits[0]
    return dict(time_s=float(times[row]), joint_index=int(joint),
                signed_margin=float(margins[row, joint]))


def failure_evidence(controller, q, dq, measured_q, measured_dq, t, output_dir=None):
    """Independent feasibility LP; diagnostics only, never a control fallback."""
    from scipy.optimize import linprog
    p = controller.policy
    matrix = p._A_cons.toarray()
    results, arrays = {}, dict(A=matrix, objective=np.zeros(p.num_variables))
    initial=np.asarray(controller.last_diagnostics.get('mpc_initial_state',np.r_[measured_q,measured_dq]))
    for label, state_q, state_dq in (('observed', measured_q, measured_dq), ('true', q, dq),
                                     ('qp_initial', initial[:5], initial[5:])):
        lower, upper, *_ = p._build_online_constraint_bounds(state_q, state_dq)
        lower[:p.nx] = upper[:p.nx] = np.r_[state_q, state_dq]
        lo, hi = np.isfinite(lower), np.isfinite(upper)
        # HiGHS defaults to a worker pool; an explicit single-thread option is
        # passed through SciPy so diagnostic LPs cannot disturb CPU 2 timing.
        import warnings
        from scipy.optimize import OptimizeWarning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=OptimizeWarning,
                                    message='Unrecognized options detected.*')
            lp = linprog(np.zeros(p.num_variables),
                A_ub=np.r_[matrix[hi], -matrix[lo]], b_ub=np.r_[upper[hi], -lower[lo]],
                bounds=[(None, None)]*p.num_variables, method='highs', options={'threads': 1})
        results[label] = dict(status=int(lp.status), success=bool(lp.success), message=str(lp.message))
        arrays[label+'_lower'], arrays[label+'_upper'] = lower, upper
        if lp.x is not None:
            arrays[label+'_solution'] = lp.x
    evidence = json_values(dict(time_s=t, true_q_deg=np.rad2deg(q), true_dq=dq,
        observed_q_deg=np.rad2deg(measured_q), observed_dq=measured_dq,
        outer_q_bounds_deg=np.rad2deg(p.safety_joint_limits),
        independent_constraint_lp=results['observed'],
        independent_true_state_constraint_lp=results['true'],
        independent_qp_initial_constraint_lp=results['qp_initial'],
        qp_initial_q_deg=np.rad2deg(initial[:5]), qp_initial_dq=initial[5:],
        lp_variable_bounds='all variables unbounded; lower <= A @ z <= upper',
        diagnostics=controller.last_diagnostics))
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir/'failure_constraints.npz', **arrays)
        evidence['lp_artifact'] = str(output_dir/'failure_constraints.npz')
        (output_dir/'failure_state.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence


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


def run_case(method='measured_mapper', perturbed=False, duration=3., *, scenario=None,
             failure_output_dir=None, torque_config=None, assumed_command_delay_s=None):
    scenario = resolve_scenario(scenario, perturbed)
    if method not in METHODS:
        raise ValueError(f'unknown method: {method}')
    if method != 'measured_mapper' and (torque_config is not None or assumed_command_delay_s is not None):
        raise ValueError('torque configuration and delay preview require measured_mapper; no ignored options')
    steps = grid_steps(duration, 'duration')
    if steps <= 0:
        raise ValueError('duration must be positive')
    measured=method!='legacy_reference'
    controller_type=RightArmMeasuredTorqueMpc
    delay_options={}
    if assumed_command_delay_s is not None:
        from hardware_mpc_delay_preview import RightArmDelayPreviewMpc
        controller_type=RightArmDelayPreviewMpc
        delay_options=dict(assumed_command_delay_s=assumed_command_delay_s)
    c=(controller_type(EXPECTED_TARGET_Q[5:10],torque_config=torque_config,**delay_options)
       if measured else RightArmHardwareMpc(EXPECTED_TARGET_Q[5:10]))
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
            if assumed_command_delay_s is not None:
                c.set_delay_context(0.,0.)
            c.step(EXPECTED_TARGET_Q,[1,0,0,0],0,.006)
        c.reset()
    else:c.warmup(EXPECTED_TARGET_Q,[1,0,0,0],count=5)
    plant_model=EndpointModel()
    if scenario.payload_scale != 1.:
        bottle=plant_model.model.body('right_bottle').id
        plant_model.model.body_mass[bottle] *= scenario.payload_scale
        plant_model.model.body_inertia[bottle] *= scenario.payload_scale
        # Change only plant parameters; controller retains nominal XML.
    backend=CppRightArmRneaBackend(plant_model.xml,library_path=ROOT/'build/right_arm_rnea/libright_arm_rnea.so')
    plant=RightArmInverseDynamics(plant_model,backend)
    q=EXPECTED_TARGET_Q[5:10].copy();dq=np.zeros(5)
    slots=EXPECTED_TARGET_Q.copy()
    sample_delay=grid_steps(scenario.observation_delay_s, 'observation delay')
    actuation_delay=grid_steps(scenario.actuation_delay_s, 'actuation delay')
    observations=deque(maxlen=sample_delay+1); pending=deque()
    _,bias=plant.linear_dynamics(q,dq,disturbance(0))
    active=PendingCommand(0, -1, 0., 0., bias.copy(), q.copy(), np.zeros(5), np.zeros(5))
    rows=[]; status='complete';reason=None;failure_state=None;desired=np.zeros(5);last_diag={}
    physical_rows=[];interval_rows=[]; seq=-1
    max_plant_qacc = 0.
    metadata = c.metadata.copy()
    try:
        for index in range(steps):
            t=index*PHYSICS_DT; base=disturbance(t)
            physical_rows.append(dict(t=t, q=q.copy(), dq=dq.copy(), **physical_margins(c.policy,q,dq)))
            observations.append((q.copy(),dq.copy(),t))
            measured_q,measured_dq,stamp=observations[max(0,len(observations)-1-sample_delay)]
            if index%CONTROL_STEPS==0:
                slots[5:10]=measured_q
                c.set_measured_dq(measured_dq)
                h=horizon(stamp)
                c.set_disturbance_horizon(h)
                if assumed_command_delay_s is not None:
                    c.set_delay_context(t,stamp)
                quat=Rotation.from_matrix(h.nodes[0].rot_world_body).as_quat(scalar_first=True)
                started=time.perf_counter_ns()
                qref,dqref,diag=c.step(slots,quat,0,.006)
                elapsed=(time.perf_counter_ns()-started)*1e-6
                last_diag=diag
                desired=np.asarray(diag['raw_mpc_ddq_rad_s2'])
                if method=='legacy_reference':ff=np.zeros(5)
                elif method=='nominal_inverse':ff=np.asarray(diag['tau_nominal_ff_nm'])
                else:ff=np.asarray(diag['tau_ff_candidate_nm'])
                seq += 1
                pending.append(PendingCommand(index+actuation_delay, seq, t, stamp,
                    ff.copy(), qref.copy(), dqref.copy(), desired.copy()))
            while pending and pending[0].apply_step<=index:
                active=pending.popleft()
            ff,qref,dqref=active.ff,active.qref,active.dqref
            # Firmware-like PD updated at physical 2 ms in this test only.
            requested_tau=ff+20*(qref-q)+(dqref-dq)
            tau=np.clip(requested_tau,-25,25)
            mass,bias=plant.linear_dynamics(q,dq,base)
            friction=scenario.viscous_friction*dq+scenario.coulomb_friction*np.tanh(dq/.02)
            accel=np.linalg.solve(mass,tau-bias-friction)
            max_plant_qacc=max(max_plant_qacc,float(np.max(np.abs(accel))))
            errors=dict(current_request_vs_actual=accel-desired,
                        actually_active_command_desired_vs_actual=accel-active.desired_ddq)
            command_trace=dict(request_seq=seq, active_seq=active.seq,
                active_desired=active.desired_ddq.copy(), active_issued_s=active.issued_s,
                active_observed_s=active.observed_s, observed_s=stamp,
                command_age_s=t-active.issued_s, observation_age_s=t-stamp)
            interval_rows.append(dict(t=t, actual=accel.copy(), desired=desired.copy(),
                tau=tau.copy(), requested_tau=requested_tau.copy(),
                torque_margin=25-np.abs(requested_tau), saturated=np.abs(requested_tau)>25,
                acceleration_margin=c.policy.max_ddq-np.abs(accel),
                mapper_acceleration_margin=(c.mapper.acc_limit if measured else 8.)-np.abs(accel),
                **command_trace, **errors))
            if index%CONTROL_STEPS==0:
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
                    observed_q=measured_q.copy(),observed_dq=measured_dq.copy(),
                    mpc_initial_state=last_diag.get('mpc_initial_state',np.r_[measured_q,measured_dq]),
                    predicted_command_time_s=last_diag.get('delay_preview',{}).get('command_time_s',stamp),
                    **command_trace, **errors,
                    model_ddq=last_diag.get('mapper',{}).get('checked_ddq_rad_s2',np.full(5,np.nan))))
            q += dq*PHYSICS_DT+.5*accel*PHYSICS_DT**2
            dq += accel*PHYSICS_DT
            if not np.isfinite(q).all() or np.max(np.abs(dq))>20:
                raise RuntimeError('plant numerical/velocity divergence')
    except Exception as exc:
        status='failed'; reason=f'{type(exc).__name__}: {exc}'
        failure_state=failure_evidence(c,q,dq,measured_q,measured_dq,t,failure_output_dir)
    finally:
        c.close();backend.close()
    arrays={key:np.asarray([row[key] for row in rows]) for key in rows[0]} if rows else {}
    if status=='complete':
        physical_rows.append(dict(t=steps*PHYSICS_DT,q=q.copy(),dq=dq.copy(),
                                  **physical_margins(c.policy,q,dq)))
    for prefix, trace in (('physics_',physical_rows),('interval_',interval_rows)):
        if trace:
            arrays.update({prefix+key:np.asarray([row[key] for row in trace]) for key in trace[0]})
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
    summary['scenario']=asdict(scenario)
    summary['clock']=dict(physics_dt_s=PHYSICS_DT,control_dt_s=CONTROL_STEPS*PHYSICS_DT,
        scheduling='integer physics-step queue; observation includes current state before solve',
        startup='oldest available observation until delay history fills; initial hold has active_seq=-1',
        integration='explicit constant acceleration over each 2ms interval')
    summary['acceleration_tracking_rmse_semantics']='historical: actual minus latest request at 6ms samples'
    if interval_rows:
        valid=arrays['interval_active_seq']>=0
        summary['tracking_2ms']=dict(
            current_request_vs_actual_rmse=float(np.sqrt(np.mean(arrays['interval_current_request_vs_actual']**2))),
            actually_active_command_desired_vs_actual_rmse=(float(np.sqrt(np.mean(
                arrays['interval_actually_active_command_desired_vs_actual'][valid]**2))) if valid.any() else None),
            active_command_samples=int(valid.sum()),initial_hold_samples=int((~valid).sum()),
            active_metric_excludes_initial_hold=True)
        summary['first_contact_2ms']={key:first_contact(arrays['physics_t'],arrays['physics_'+key])
            for key in ('q_inner_margin','q_outer_margin','dq_margin')}
        summary['first_contact_2ms'].update({key:first_contact(arrays['interval_t'],arrays['interval_'+key])
            for key in ('acceleration_margin','mapper_acceleration_margin','torque_margin')})
        summary['minimum_margins_2ms']={key:float(arrays['physics_'+key].min())
            for key in ('q_inner_margin','q_outer_margin','dq_margin')}
        summary['minimum_margins_2ms'].update({key:float(arrays['interval_'+key].min())
            for key in ('acceleration_margin','mapper_acceleration_margin','torque_margin')})
        summary['saturated_2ms_intervals']=int(arrays['interval_saturated'].any(axis=1).sum())
        summary['physical_state_scored_until_s']=float(arrays['physics_t'][-1])
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
