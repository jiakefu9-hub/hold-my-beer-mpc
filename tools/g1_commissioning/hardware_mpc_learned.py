"""Learned-forecast experiment: yaw-aware MPC and direct conditional RNEA.

The successful baseline is deliberately left in hardware_mpc_torque_control.
No SDK, publishers, model fitting, or hardware authority in this module.
"""
from __future__ import annotations

import numpy as np

from hardware_mpc_control import HardwareMpcError
from hardware_mpc_solver import CondensedArmMPCPolicy
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc


class YawFeedbackMpcPolicy(CondensedArmMPCPolicy):
    """Exact input change of coordinates for a frozen local yaw-PD model.

    Original nominal acceleration u, retained yaw torque p(x):
        a = u + inv(M) @ p(x) = u + F x + f
        x_next = A x + B (u + F x + f).
    Optimize NET acceleration a instead: the existing integrator, physical
    q/dq/a limits, endpoint costs and condensed matrices stay unchanged.
    Substitute u = a - F x - f in the nominal-acceleration effort cost.
    This is algebraically the same feedback-aware prediction, not a second
    PD added after planning, nor an empirical model of the real servo.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_q = self.default_q.copy()
        self.feedback_F = np.zeros((self.nu, self.nx))
        self.feedback_f = np.zeros(self.nu)
        self.feedback_pd = np.zeros(self.nu)
        # A valid solved QP that was descheduled is not an infeasible QP.
        # Native solve budget remains; the parent enforces the bounded total
        # latency, fresh feedback and repeated-lateness rules before any Write.
        self.allow_solved_wall_overrun = True

    def set_local_actuation_constraints(self, mass, bias, total_limit, ff_limit, feedback_dq):
        super().set_local_actuation_constraints(mass, bias, total_limit, ff_limit, feedback_dq)
        direction = np.linalg.solve(mass, np.eye(self.nu)[:, 2])
        kp, kd = float(self._local_kp[2]), float(self._local_kd[2])
        self.feedback_F.fill(0.)
        self.feedback_F[:, 2] = -direction * kp
        self.feedback_F[:, self.nu+2] = -direction * kd
        self.feedback_f = direction * kp * self.default_q[2]
        self.feedback_pd.fill(0.)
        self.feedback_pd[2] = kp*(self.default_q[2]-self.feedback_q[2])-kd*feedback_dq[2]

    def _build_cost(self, step_terms):
        blocks, linear = super()._build_cost(step_terms)
        stages, terminal = self._cost_blocks
        transform = np.hstack((-self.feedback_F, np.eye(self.nu)))
        delta = transform.T @ self.R @ transform
        delta[self.nx:, self.nx:] -= self.R
        stages += 2*delta[None]
        offset = -2*transform.T @ self.R @ self.feedback_f
        linear[:self.horizon*self.stage_dim].reshape(self.horizon, self.stage_dim)[:] += offset
        self._cost_blocks = (stages, terminal)
        return [*stages, terminal], linear

    def _local_actuation_rows(self):
        matrix, lo, hi = super()._local_actuation_rows()
        # Replace just the yaw FF row: its packet reference is fixed at nominal,
        # unlike the one-step references on the other four axes. All total
        # torque rows already constrain M*a+b, INCLUDING predicted yaw feedback.
        row_index = self.horizon*self.nu + 2
        row = np.zeros(self.horizon*self.nu)
        row[:self.nu] = self._local_actuation['mass'][2]*self.max_ddq
        offset = self._local_actuation['bias'][2]-self.feedback_pd[2]
        limit = self._local_actuation['ff_limit'][2]
        scale = 1/max(float(np.max(np.abs(row))), 1e-12)
        matrix[row_index] = row*scale
        lo[row_index], hi[row_index] = (-limit-offset)*scale, (limit-offset)*scale
        return matrix, lo, hi


class RightArmLearnedTorqueMpc(RightArmMeasuredTorqueMpc):
    policy_type = YawFeedbackMpcPolicy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = self.torque_config
        if (not c.get('yaw_feedback_in_prediction', False)
                or not c['anchor_shoulder_yaw_reference']
                or not c['planning_actuation_constraints_enabled']
                or c['active_slew_reference'] != 'none'
                or c['recovery_envelope_enabled']):
            raise ValueError('learned variant requires its explicit yaw-aware direct-torque configuration')
        self.metadata.update(
            variant='learned_yaw_aware_direct_v1',
            forward_model='conditional rigid right arm with observed moving torso, not full-body contact dynamics',
            torque_mapping='direct M*a+b; one final affine consistency/envelope check; no candidate search',
            output_semantics='a includes yaw feedback; packet tau=(M*a+b)-PD, device adds PD once',
            tracking_offset_model='yaw PD represented across horizon by exact input-coordinate substitution',
            yaw_feedback_prediction=dict(enabled=True, kp=float(c['kp'][2]), kd=float(c['kd'][2]),
                target_rad=float(self.nominal[2]), physical_identification=False,
                equation='a=u+F*x+f; effort=(a-F*x-f)^T R (a-F*x-f)',
                acceleration_limits_apply_to='net model acceleration including yaw PD'),
            candidate_search_enabled=False)
        self.metadata['shoulder_yaw_execution']['semantics'] = (
            'fixed nominal packet q/dq; PD modeled inside prediction, not added after final torque check')

    def step(self, arm_slots, *args, **kwargs):
        self.policy.feedback_q = np.asarray(arm_slots, dtype=float)[5:10].copy()
        return super().step(arm_slots, *args, **kwargs)

    def _torque(self, q, dq, qref, dqref, ddq, base, bounds=None, prepared_dynamics=None):
        if bounds is not None:
            raise HardwareMpcError('direct torque variant does not support an unplanned slew constraint')
        mass, bias = (self.inverse.linear_dynamics(q, dq, base)
                      if prepared_dynamics is None else prepared_dynamics)
        # RNEA's affine form, already used inside this QP. Re-running inverse
        # and forward dynamics with different labels adds no physical evidence.
        total = mass@ddq+bias
        pd = self.torque_config['kp']*(qref-q)+self.torque_config['kd']*(dqref-dq)
        ff = total-pd
        checked = np.linalg.solve(mass, total-bias)
        limits = self._active_braking_acceleration_bounds
        ff_limit = np.asarray(self.torque_config['tau_ff_abs_nm'])
        if (not np.isfinite(np.r_[total, ff, checked]).all()
                or np.any(np.abs(total) > self.mapper.limit+1e-6)
                or np.any(np.abs(ff) > ff_limit+1e-6)
                or np.any(np.abs(checked) > self.mapper.acc_limit+1e-6)
                or np.max(np.abs(checked-ddq)) > 1e-6
                or (limits is not None and
                    (np.any(checked < limits[0]-1e-6) or np.any(checked > limits[1]+1e-6)))):
            raise HardwareMpcError('direct final torque disagrees with planned physical envelope')
        self._prepared_forward = (mass, bias)
        self._previous_total = total.copy()
        yaw_acc = self.policy.feedback_F@np.r_[q, dq]+self.policy.feedback_f
        nominal = ddq-yaw_acc
        qp = self.policy.get_last_diagnostics(copy_data=False)
        # Small, auditable 9x5 evidence: no recomputation or file IO in the core.
        predicted_states = qp['predicted_states'][:-1]
        predicted_net = qp['predicted_inputs']
        predicted_yaw = predicted_states@self.policy.feedback_F.T+self.policy.feedback_f
        return dict(
            solver_wall_over_budget=bool(getattr(self.policy,'_last_solved_wall_over_budget',False)),
            inverse_dynamics=dict(tau_model_nm=total, method='conditional affine RNEA M*a+b'),
            mapper=dict(method='direct_affine_no_candidate_search', candidate_count=0,
                forward_calls=1, model_accepted=True, hardware_certified=False,
                tau_total_nm=total, checked_ddq_rad_s2=checked,
                error_norm=float(np.linalg.norm(checked-ddq))),
            yaw_feedback_model=dict(F_rad_s2_per_state=self.policy.feedback_F,
                f_rad_s2=self.policy.feedback_f,
                current_yaw_pd_nm=self.policy.feedback_pd,
                current_yaw_acceleration_rad_s2=yaw_acc,
                nominal_mpc_acceleration_rad_s2=nominal,
                horizon_net_acceleration_rad_s2=predicted_net,
                horizon_yaw_acceleration_rad_s2=predicted_yaw,
                horizon_nominal_acceleration_rad_s2=predicted_net-predicted_yaw),
            final_acceleration_bounds_rad_s2=limits,
            torque_slew_reference='none', torque_model_bias_nm=bias,
            torque_previous_model_bias_nm=None, torque_slew_bias_shift_nm=np.zeros(5),
            torque_slew_bounds_nm=None, tau_nominal_ff_nm=mass@nominal+bias,
            tau_pd_at_feedback_nm=pd, tau_model_selected_nm=total,
            retained_firmware_pd_nm=np.zeros(5),  # no EXTRA PD after model torque
            tau_ff_candidate_nm=ff, tau_total_estimated_at_feedback_nm=total,
            expected_kp=self.torque_config['kp'], expected_kd=self.torque_config['kd'],
            torque_output_authorized=False, torque_estimate_feedback_used_for_control=False)
