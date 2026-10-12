"""C: task PID + one-step constrained acceleration allocation, SDK-free.

Shares measured H0 kinematics, conditional arm dynamics, limits and lifecycle
with field MPC. There is no predicted joint trajectory or MPC horizon solve.
The five-variable constrained least squares replaces simulation's final clip:
packet feedforward and total torque must be feasible before execution.
"""
from pathlib import Path
import time

import numpy as np
import yaml

from endpoint_pose import ROOT
from hardware_mpc_control import HardwareMpcError, json_values
from hardware_mpc_solver import CondensedArmMPCPolicy
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc

CONFIG_PATH = ROOT / 'configs/hardware_acceleration_pid_20261012.yaml'
VARIANT = 'feedforward_acceleration_pid_v1'


def load_pid_config(path):
    c = yaml.safe_load(Path(path).read_text())
    keys = {'schema', 'kp', 'ki', 'kd', 'acc_weight', 'regularization',
            'posture_kp', 'posture_kd', 'integral_limit'}
    if set(c) != keys or c['schema'] != 'g1_acceleration_pid_v1':
        raise ValueError('invalid acceleration PID config keys/schema')
    for key in keys - {'schema'}:
        c[key] = float(c[key])
        if not np.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f'invalid acceleration PID {key}')
    if c['regularization'] <= 0 or c['integral_limit'] <= 0 or c['kp'] <= 0:
        raise ValueError('positive PID proportional gain, regularization and integral limit required')
    return c


class AccelerationPidPolicy:
    """One current acceleration allocation; forecast uses only interval zero."""
    # Reuse validation and storage for the exact executor constraints.
    set_local_actuation_constraints = CondensedArmMPCPolicy.set_local_actuation_constraints
    clear_local_actuation_constraints = CondensedArmMPCPolicy.clear_local_actuation_constraints
    set_first_acceleration_bounds = CondensedArmMPCPolicy.set_first_acceleration_bounds
    set_braking_velocity_cost = CondensedArmMPCPolicy.set_braking_velocity_cost

    def __init__(self, default_q, *, control_dt, horizon, joint_limits,
                 joint_limit_margin, max_dq, max_ddq, solver_time_limit,
                 solver_max_iter, solver_backend, **unused_mpc_costs):
        if solver_backend != 'daqp' or control_dt != .006 or horizon != 9:
            raise ValueError('C requires the established DAQP / 6 ms forecast interface')
        import daqp
        self._daqp = daqp
        self.nu = 5
        self.default_q = np.asarray(default_q, dtype=float).copy()
        self.control_dt = float(control_dt)
        self.horizon = 1  # one-step allocation, not the 9-interval predictor contract
        margin = np.asarray(joint_limit_margin)
        self.joint_limit_margin = np.broadcast_to(margin, (5,)).copy()
        self.safety_joint_limits = np.asarray(joint_limits, dtype=float).copy()
        self.joint_limits = np.asarray(joint_limits, dtype=float).copy()
        self.joint_limits[:, 0] += margin
        self.joint_limits[:, 1] -= margin
        self.max_dq = np.broadcast_to(max_dq, (5,)).copy()
        self.max_ddq = np.broadcast_to(max_ddq, (5,)).copy()
        self.solver_time_limit = float(solver_time_limit)
        self.solver_max_iter = int(solver_max_iter)
        self._condensed_solver = None  # shared warmup interface; never a horizon solver
        self.parameters = None
        self.reset()

    def reset(self):
        self.integral = np.zeros(2)
        self._local_actuation = None
        self._last_local_actuation = None
        self._first_acceleration_bounds = None
        self._braking_velocity_cost = np.zeros(5)
        self.last_diagnostics = {}

    def get_cost_definition(self):
        return dict(control_family=VARIANT, optimized_variables=5,
                    prediction_horizon_optimized=False, parameters=self.parameters,
                    equation='||Jw*a+b_alpha-alpha_PID||^2 + wa*||Jv*a+b_acc||^2 + reg*||a-a_posture||^2',
                    constraints='next q/dq, acceleration, total and packet torque, existing latched braking')

    def get_last_diagnostics(self, copy_data=False):
        return self.last_diagnostics.copy() if copy_data else self.last_diagnostics

    def first_step_acceleration_bounds(self, q, dq):
        dt = self.control_dt
        # Same existing MPC shoulder recovery margin: when the modeled stop
        # cannot stay in the inner box, use the ORIGINAL outer safety box on
        # that side while the unchanged latched brake acts. Never extend a
        # physical limit or force an impossible return within one 6 ms tick.
        qlo, qhi = self.joint_limits[:, 0].copy(), self.joint_limits[:, 1].copy()
        stopping_lo = q-np.maximum(-dq, 0.)**2/(1.6*self.max_ddq)-1e-4
        stopping_hi = q+np.maximum(dq, 0.)**2/(1.6*self.max_ddq)+1e-4
        use_lo = (self.joint_limit_margin > 0.) & (stopping_lo < qlo)
        use_hi = (self.joint_limit_margin > 0.) & (stopping_hi > qhi)
        qlo[use_lo] = self.safety_joint_limits[use_lo, 0]
        qhi[use_hi] = self.safety_joint_limits[use_hi, 1]
        self.reentry_diagnostics = dict(active=bool(np.any(use_lo | use_hi)),
            lower_rad=qlo, upper_rad=qhi, original_outer_limits_unchanged=True)
        lo = np.maximum(-self.max_ddq, (-self.max_dq-dq)/dt)
        hi = np.minimum(self.max_ddq, (self.max_dq-dq)/dt)
        lo = np.maximum(lo, 2*(qlo-q-dq*dt)/dt**2)
        hi = np.minimum(hi, 2*(qhi-q-dq*dt)/dt**2)
        if self._first_acceleration_bounds is not None:
            lo = np.maximum(lo, self._first_acceleration_bounds[0])
            hi = np.minimum(hi, self._first_acceleration_bounds[1])
        return lo, hi

    def allocation_constraints(self, q, dq):
        if self._local_actuation is None:
            raise HardwareMpcError('C requires current measured-state torque constraints')
        c = self._local_actuation
        dt = self.control_dt
        lo, hi = self.first_step_acceleration_bounds(q, dq)
        # For four axes: PD=kp*(dq*dt+0.5*a*dt^2)+kd*a*dt.
        # The existing yaw packet instead holds the nominal angle and zero dq.
        pd_gain = .5*self._local_kp*dt**2 + self._local_kd*dt
        pd_offset = self._local_kp*dq*dt
        pd_gain[2] = 0.
        pd_offset[2] = self._local_kp[2]*(self.default_q[2]-q[2])-self._local_kd[2]*dq[2]
        matrix = np.vstack((np.eye(5), c['mass'], c['mass']-np.diag(pd_gain)))
        lower = np.r_[lo, -c['total_limit']-c['bias'], -c['ff_limit']-c['bias']+pd_offset]
        upper = np.r_[hi, c['total_limit']-c['bias'], c['ff_limit']-c['bias']+pd_offset]
        return matrix, lower, upper

    def compute_action(self, observation, helpers):
        started = time.perf_counter()
        q, dq = (np.asarray(observation[k], dtype=float) for k in ('current_q', 'current_dq'))
        if q.shape != (5,) or dq.shape != (5,) or not np.isfinite(np.r_[q, dq]).all():
            raise HardwareMpcError('invalid measured C state')
        c = self.parameters
        if c is None:
            raise HardwareMpcError('C parameters not explicitly configured')
        t = helpers.compute_mpc_terms(q, dq, helpers.disturbance_prediction[0],
                                     helpers.interval_disturbance_prediction[0], True)
        e = np.asarray(t['gravity_error']) / 9.81
        Jg = t['G_g'][:, :5] / 9.81
        Jw, Jv = t['B_alpha'], t['B_acc']
        T = Jg @ np.linalg.pinv(Jw, rcond=1e-5)
        omega = t['D_omega'] + t['C_omega'] @ dq
        proposed_integral = np.clip(self.integral + e*self.control_dt,
                                    -c['integral_limit'], c['integral_limit'])
        alpha_pid = np.linalg.pinv(T, rcond=1e-5) @ (-c['kp']*e-c['ki']*proposed_integral)-c['kd']*omega
        ba = t['D_acc'] + t['C_acc'] @ dq
        balpha = t['D_alpha'] + t['C_alpha'] @ dq
        posture = c['posture_kp']*(self.default_q-q)-c['posture_kd']*dq
        A = np.vstack((Jw, np.sqrt(c['acc_weight'])*Jv, np.sqrt(c['regularization'])*np.eye(5)))
        b = np.r_[alpha_pid-balpha, -np.sqrt(c['acc_weight'])*ba, np.sqrt(c['regularization'])*posture]
        H = A.T@A + 1e-9*np.eye(5)
        f = -A.T@b
        # Existing near-boundary brake also discourages outward joint speed.
        H += np.diag(self._braking_velocity_cost*self.control_dt**2)
        f += self._braking_velocity_cost*self.control_dt*dq
        raw = np.linalg.solve(H, -f)
        matrix, lo, hi = self.allocation_constraints(q, dq)
        self._last_local_actuation = (matrix, lo, hi)
        if not np.isfinite(np.r_[H.ravel(), f, matrix.ravel(), lo, hi]).all() or np.any(lo > hi):
            raise HardwareMpcError('C allocation has invalid or infeasible bounds')
        solve_start = time.perf_counter()
        value = matrix @ raw
        if np.all(value >= lo) and np.all(value <= hi):
            a, flag, iterations, method = raw, 1, 0, 'unconstrained_least_squares_feasible'
        else:
            scale = 1/np.maximum(np.max(np.abs(matrix), axis=1), 1e-12)
            a, _, flag, info = self._daqp.solve(np.ascontiguousarray(H), np.ascontiguousarray(f),
                np.ascontiguousarray(matrix*scale[:, None]), np.ascontiguousarray(hi*scale),
                np.ascontiguousarray(lo*scale), primal_tol=1e-9, dual_tol=1e-10, eps_prox=0.,
                iter_limit=self.solver_max_iter, time_limit=self.solver_time_limit)
            iterations, method = int(info['iterations']), 'five_variable_constrained_least_squares'
        elapsed = time.perf_counter()-solve_start
        good = flag == 1 and np.isfinite(a).all()
        violation = float(max(0., np.max(lo-matrix@a), np.max(matrix@a-hi))) if good else float('inf')
        good = good and violation <= 1e-6
        constrained = bool(np.max(np.abs(a-raw)) > 1e-6) if good else True
        if good and not constrained:
            self.integral = proposed_integral
        self.last_diagnostics = dict(solved=good, success=good, fallback_used=False,
            solver_status=method if good else f'allocation_rejected_{flag}', solver_status_val=int(flag),
            control_family=VARIANT, gravity_error=t['gravity_error'],
            iterations=iterations, solve_time=elapsed, max_constraint_violation=violation,
            raw_allocation_ddq=raw, ddq_des=a, integral=self.integral.copy(),
            integral_frozen=constrained, alpha_pid=alpha_pid, predicted_linear_offset=ba,
            predicted_angular_offset=balpha, predicted_endpoint_acceleration=Jv@a+ba,
            predicted_endpoint_angular_acceleration=Jw@a+balpha,
            feedback_state_extrapolated=False, forecast_interval_used=0,
            policy_core_time=time.perf_counter()-started)
        if not good:
            raise HardwareMpcError(f'C rejected allocation: flag={flag}, violation={violation}')
        return q+dq*self.control_dt+.5*a*self.control_dt**2, dq+a*self.control_dt, a


class RightArmAccelerationPid(RightArmMeasuredTorqueMpc):
    policy_type = AccelerationPidPolicy

    def __init__(self, *args, pid_config, **kwargs):
        parameters = load_pid_config(pid_config)
        super().__init__(*args, **kwargs)
        self.policy.parameters = parameters
        c = self.torque_config
        if (not c['planning_actuation_constraints_enabled'] or not c['anchor_shoulder_yaw_reference']
                or c['active_slew_reference'] != 'none' or c['recovery_envelope_enabled']):
            raise ValueError('C requires original bounded direct torque execution')
        self.metadata.update(variant=VARIANT, model='measured_state_task_acceleration_pid',
            solver='5_variable_constrained_least_squares_DAQP_only_if_needed',
            cost=self.policy.get_cost_definition(), pid_config_path=str(pid_config),
            state='measured q,dq and current measured body attitude/angular velocity; no state extrapolation',
            tracking_offset_model='none', forecast_use='first acc/alpha interval only; no horizon optimization',
            inherited_mpc_costs_used=False,
            planning_actuation_constraints=dict(enabled=True, horizon_steps=1),
            output_semantics='tau=(M*a+b)-packet_PD; firmware adds packet_PD once',
            mpc_pitch_feedback_prior_used=False, candidate_search_enabled=False)

    def step(self, *args, **kwargs):
        try:
            result = super().step(*args, **kwargs)
        except Exception:
            self.last_diagnostics['acceleration_pid'] = json_values(self.policy.last_diagnostics)
            self.last_diagnostics['actual_control_family'] = VARIANT
            raise
        self.last_diagnostics['actual_control_family'] = VARIANT
        return result

    def _torque(self, q, dq, qref, dqref, ddq, base, bounds=None, prepared_dynamics=None):
        if bounds is not None:
            raise HardwareMpcError('C does not accept unplanned torque slew bounds')
        mass, bias = prepared_dynamics or self.inverse.linear_dynamics(q, dq, base)
        total = mass@ddq+bias
        pd = self.torque_config['kp']*(qref-q)+self.torque_config['kd']*(dqref-dq)
        ff = total-pd
        checked = np.linalg.solve(mass, total-bias)
        lo, hi = self.policy.first_step_acceleration_bounds(q, dq)
        if (not np.isfinite(np.r_[total, ff, checked]).all()
                or np.any(np.abs(total) > self.mapper.limit+1e-6)
                or np.any(np.abs(ff) > np.asarray(self.torque_config['tau_ff_abs_nm'])+1e-6)
                or np.any(checked < lo-1e-6) or np.any(checked > hi+1e-6)
                or np.max(np.abs(checked-ddq)) > 1e-6):
            raise HardwareMpcError('C final torque violates allocation envelope')
        self._prepared_forward = (mass, bias)
        self._previous_total = total.copy()
        return dict(inverse_dynamics=dict(tau_model_nm=total, method='conditional affine RNEA M*a+b'),
            mapper=dict(method='direct_affine_no_candidate_search', candidate_count=0,
                        model_accepted=True, hardware_certified=False, checked_ddq_rad_s2=checked),
            final_acceleration_bounds_rad_s2=[lo, hi], torque_model_bias_nm=bias,
            torque_slew_reference='none', tau_nominal_ff_nm=total,
            tau_pd_at_feedback_nm=pd, tau_model_selected_nm=total,
            retained_firmware_pd_nm=np.zeros(5), tau_ff_candidate_nm=ff,
            tau_total_estimated_at_feedback_nm=total, expected_kp=self.torque_config['kp'],
            expected_kd=self.torque_config['kd'], torque_output_authorized=False,
            torque_estimate_feedback_used_for_control=False)
