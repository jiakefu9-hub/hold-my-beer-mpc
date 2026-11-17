"""Simulation-only PID/position-MPC comparison policies; no robot SDK.

The full-task recorder has a three-output acceleration-MPC wire interface.
For POSITION policies the third output is an UNUSED zero placeholder.
AccelerationPID instead uses the existing acceleration-to-torque execution layer.
Actual acceleration is always measured from the MuJoCo plant for evaluation.
"""
from dataclasses import replace
import time

import numpy as np
import osqp
from scipy import sparse


class PositionPolicy:
    def __init__(self, config, neutral, dt):
        self.config = config
        self.neutral = np.asarray(neutral, dtype=float).copy()
        self.dt = float(dt)
        self.N = int(config.get('mpc_horizon', 9))
        self.horizon = self.N
        self.control_dt = self.dt
        margin = np.asarray(config.get('mpc_q_operating_margin_deg', [0.] * 5))
        self.lo = np.deg2rad(np.asarray(config['mpc_q_min_deg']) + margin)
        self.hi = np.deg2rad(np.asarray(config['mpc_q_max_deg']) - margin)
        self.vmax = float(config.get('simple_max_dq', config.get('mpc_max_dq', 2.)))
        self.amax = float(config.get('simple_max_ddq', 15.))
        self.ff = bool(config.get('simple_feedforward', True))
        self.reset()

    def reset(self):
        self.ref = None
        self.vref = np.zeros(5)
        self.integral = np.zeros(2)
        self.derivative = np.zeros(2)
        self.last_error = None
        self.solver = None
        self.servo_bias = np.zeros(5)
        self.previous_q = None
        self.last_diagnostics = {}

    def get_cost_definition(self):
        # Acceleration-MPC cost diagnostics do not describe these controllers.
        return None

    def get_last_diagnostics(self, copy_data=False):
        return self.last_diagnostics.copy()

    def terms(self, q, dq, helpers):
        nodes = tuple(helpers.disturbance_prediction)
        intervals = tuple(helpers.interval_disturbance_prediction)
        if not self.ff:
            nodes = (helpers.disturbance,) * (self.N + 1)
            intervals = (helpers.disturbance,) * self.N
        return helpers.compute_mpc_terms_batch(
            np.tile(q, (self.N + 1, 1)), np.tile(dq, (self.N + 1, 1)),
            nodes, intervals, np.ones(self.N + 1, dtype=bool))

    def finish(self, qref, dqref, started, **diagnostics):
        self.ref = np.asarray(qref).copy()
        self.vref = np.asarray(dqref).copy()
        self.last_diagnostics = dict(
            success=True, solved=True, fallback_used=False, fallback_feasible=True,
            solver_status='not_applicable_position_control', solver_status_val=1,
            ddq_raw=np.zeros(5), ddq_des=np.zeros(5), ddq_output_unused=True,
            control_family=self.config['simple_controller'],
            policy_core_time=time.perf_counter() - started,
            max_constraint_violation=0., iterations=0, solve_time=0.)
        self.last_diagnostics.update(diagnostics)
        return self.ref.copy(), self.vref.copy(), np.zeros(5)

    def bounded_velocity(self, desired):
        velocity = np.clip(desired, -self.vmax, self.vmax)
        velocity = np.clip(velocity, self.vref - self.amax * self.dt,
                           self.vref + self.amax * self.dt)
        ref = np.clip(self.ref + self.dt * velocity, self.lo, self.hi)
        return ref, (ref - self.ref) / self.dt


class FeedforwardPID(PositionPolicy):
    """Resolved-rate tilt PID plus predicted base-rotation feedforward.

    e is signed gravity projection / g (approximately signed tilt in radians).
    dq = J#[-Kp e - Ki integral(e) - Kd filtered(de/dt) - ff de_base/dt]
         + (I-J#J) Kposture (q_neutral-q).
    The FF term evaluates future body orientation with *fixed current q*;
    it does not read future simulator state. References are rate/accel bounded.
    """
    def compute_action(self, observation, helpers):
        start = time.perf_counter()
        q, dq = np.asarray(observation['current_q']), np.asarray(observation['current_dq'])
        if self.ref is None:
            self.ref = np.clip(q.copy(), self.lo, self.hi)
        terms = self.terms(q, dq, helpers)
        error = terms[0]['gravity_error'] / 9.81
        J = terms[0]['G_g'][:, :5] / 9.81
        inverse = J.T @ np.linalg.inv(J @ J.T + .02**2 * np.eye(2))
        raw_derivative = np.zeros(2) if self.last_error is None else (error-self.last_error)/self.dt
        self.last_error = error.copy()
        alpha = 1. - np.exp(-self.dt / float(self.config.get('simple_pid_filter_s', .018)))
        self.derivative += alpha * (raw_derivative - self.derivative)
        old_integral = self.integral.copy()
        self.integral = np.clip(self.integral + error*self.dt, -.1, .1)
        k = min(self.N, max(1, int(self.config.get('simple_pid_preview_steps', 1))))
        base_rate = (terms[k]['gravity_error']/9.81-error)/(k*self.dt) if self.ff else np.zeros(2)
        task = (-float(self.config.get('simple_pid_kp', 16.))*error
                -float(self.config.get('simple_pid_ki', 4.))*self.integral
                -float(self.config.get('simple_pid_kd', .12))*self.derivative
                -float(self.config.get('simple_ff_gain', 1.))*base_rate)
        desired = (inverse @ task + (np.eye(5)-inverse@J)
                   @ (float(self.config.get('simple_posture_gain', 1.))*(self.neutral-q)))
        if self.config.get('simple_pid_full_rotation', False):
            # Also cancel spin around the payload axis, which a two-axis tilt
            # PID alone does not regulate. Still a rate PID, no optimization.
            Jw = terms[0]['B_alpha']
            inv_w = Jw.T@np.linalg.inv(Jw@Jw.T+.02**2*np.eye(3))
            tilt_rate_map = J@np.linalg.pinv(Jw, rcond=1e-5)
            feedback = task + float(self.config.get('simple_ff_gain', 1.))*base_rate
            omega_feedforward = terms[k]['D_omega'] if self.ff else np.zeros(3)
            desired = (inv_w @ (np.linalg.pinv(tilt_rate_map, rcond=1e-5)@feedback
                       -float(self.config.get('simple_ff_gain', 1.))*omega_feedforward)
                       +(np.eye(5)-inv_w@Jw)
                       @(float(self.config.get('simple_posture_gain', 1.))*(self.neutral-q)))
        ref, velocity = self.bounded_velocity(desired)
        if np.max(np.abs(velocity-desired)) > 1e-8:
            self.integral = old_integral  # conditional integration anti-windup
        return self.finish(ref, velocity, start, gravity_error=terms[0]['gravity_error'],
                           base_tilt_rate_ff=base_rate, desired_reference_velocity=desired)


class OriginalPIDAdapter(PositionPolicy):
    """Unchanged repository PID equations/gains, with the shared position bounds."""
    def __init__(self, config, neutral, dt, policy):
        super().__init__(config, neutral, dt)
        self.policy = policy

    def compute_action(self, observation, helpers):
        start = time.perf_counter()
        obs = dict(observation, torso_rotmat=helpers.disturbance.rot_world_body)
        ref, velocity = self.policy.compute_action(obs, helpers)
        ref = np.clip(ref, self.lo, self.hi)
        return self.finish(ref, velocity, start)


class PositionMPC(PositionPolicy):
    """Five-state first-order position-servo MPC with position-reference inputs.

    q[k+1] = (1-beta) q[k] + beta r[k]. No dq state or inverse dynamics.
    Velocity/acceleration task costs, if enabled, are finite differences of
    predicted q. Measured dq supplies only the first acceleration boundary.
    This is an approximate position-servo model, not the actual robot dynamics.
    """
    def __init__(self, config, neutral, dt):
        super().__init__(config, neutral, dt)
        self.beta = float(config.get('simple_position_beta', .5))
        if not 0. < self.beta <= 1.:
            raise ValueError('position beta must be in (0, 1]')
        n = 5*self.N
        a = 1.-self.beta
        scalar = np.array([[self.beta*a**(k-j) if j <= k else 0.
                            for j in range(self.N)] for k in range(self.N)])
        self.L = np.kron(scalar, np.eye(5))
        self.F = np.kron((a**np.arange(1, self.N+1))[:, None], np.eye(5))
        self.diff = np.eye(n) - np.eye(n, k=-5)
        self.diff2 = self.diff@self.diff
        self.constraint_matrix = sparse.csc_matrix(np.vstack((np.eye(n), self.diff, self.diff2)))
        self.p_rows, self.p_cols = np.triu_indices(n)
        # OSQP uses CSC column order, not numpy's row-major triu_indices order.
        order = np.lexsort((self.p_rows, self.p_cols))
        self.p_rows, self.p_cols = self.p_rows[order], self.p_cols[order]

    def compute_action(self, observation, helpers):
        start = time.perf_counter()
        q, dq = np.asarray(observation['current_q']), np.asarray(observation['current_dq'])
        if self.ref is None:
            self.ref = np.clip(q.copy(), self.lo, self.hi)
        terms = self.terms(q, dq, helpers)
        n = 5*self.N
        H, h = np.eye(n)*1e-9, np.zeros(n)
        def cost(M, b, weight):
            nonlocal H, h
            if weight:
                H += float(weight)*(M.T@M)
                h += float(weight)*(M.T@b)
        gain = float(self.config.get('simple_servo_bias_gain', 0.))
        if self.previous_q is not None:
            innovation = q-((1-self.beta)*self.previous_q+self.beta*self.ref)
            self.servo_bias += gain*(innovation-self.servo_bias)
            self.servo_bias = np.clip(self.servo_bias, -.03, .03)
        self.previous_q = q.copy()
        # Known, causal model offset, held constant in the prediction horizon.
        # Only q is an optimized state; bias is estimated outside the QP.
        offset = self.F@q + self.L@np.tile(self.servo_bias/self.beta, self.N)
        vmap = self.diff@self.L/self.dt
        voffset = (self.diff@offset - np.r_[q, np.zeros(n-5)])/self.dt
        amap = self.diff@vmap/self.dt
        aoffset = (self.diff@voffset - np.r_[dq, np.zeros(n-5)])/self.dt
        for k in range(self.N):
            sl = slice(5*k, 5*k+5)
            t = terms[k+1]
            J = t['G_g'][:, :5]/9.81
            cost(J@self.L[sl], J@(offset[sl]-q)+t['gravity_error']/9.81,
                 self.config.get('simple_q_tilt', 1.))
            cost(self.L[sl], offset[sl]-self.neutral,
                 self.config.get('simple_q_posture', .002))
            cost(t['C_omega']@vmap[sl], t['D_omega']+t['C_omega']@voffset[sl],
                 self.config.get('simple_q_omega', .001))
            # Derived acceleration costs use only predicted q differences.
            it = terms[k]
            cost(it['B_acc']@amap[sl]+it['C_acc']@vmap[sl],
                 it['D_acc']+it['B_acc']@aoffset[sl]+it['C_acc']@voffset[sl],
                 self.config.get('simple_q_acc', 0.))
            cost(it['B_alpha']@amap[sl]+it['C_alpha']@vmap[sl],
                 it['D_alpha']+it['B_alpha']@aoffset[sl]+it['C_alpha']@voffset[sl],
                 self.config.get('simple_q_alpha', 0.))
        cost(self.diff/self.dt, -np.r_[self.ref, np.zeros(n-5)]/self.dt,
             self.config.get('simple_r_rate', .0001))
        P = sparse.csc_matrix((2*H[self.p_rows, self.p_cols],
                              (self.p_rows, self.p_cols)), shape=(n, n))
        delta = np.r_[self.ref, np.zeros(n-5)]
        accel_offset = np.r_[self.ref+self.dt*self.vref, -self.ref, np.zeros(n-10)]
        lower = np.r_[np.tile(self.lo, self.N), delta-self.vmax*self.dt,
                      accel_offset-self.amax*self.dt**2]
        upper = np.r_[np.tile(self.hi, self.N), delta+self.vmax*self.dt,
                      accel_offset+self.amax*self.dt**2]
        if self.solver is None:
            self.solver = osqp.OSQP()
            self.solver.setup(P=P, q=2*h, A=self.constraint_matrix, l=lower, u=upper,
                              verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=1500,
                              polishing=False, warm_starting=True)
        else:
            self.solver.update(Px=P.data, q=2*h, l=lower, u=upper)
        result = self.solver.solve(raise_error=False)
        good = result.info.status_val in (1, 2) and result.x is not None and np.isfinite(result.x).all()
        if good:
            rawref = result.x[:5]
            violation = max(0., float(np.max(lower-self.constraint_matrix@result.x)),
                            float(np.max(self.constraint_matrix@result.x-upper)))
            good = violation < 1e-4
        else:
            violation = float('inf')
        desired = (rawref-self.ref)/self.dt if good else np.zeros(5)
        # Same bounded reference actuator used by the resolved-rate PID.
        ref, velocity = self.bounded_velocity(desired)
        return self.finish(ref, velocity, start, success=good, solved=good,
                           fallback_used=not good, solver_status=result.info.status,
                           solver_status_val=result.info.status_val, iterations=result.info.iter,
                           solve_time=result.info.solve_time, max_constraint_violation=violation,
                           reference_rate_limiter_active=bool(np.max(np.abs(desired-velocity))>1e-7))


class AccelerationPID(PositionPolicy):
    """Resolved-acceleration tilt PID with inertial disturbance feedforward.

    One 5x5 damped least-squares solve per tick, no horizon optimization.
    The resulting ddq uses EXACTLY the acceleration-MPC inverse-dynamics and
    candidate-validation execution chain. This separates optimizer from executor.
    """
    def reset(self):
        super().reset()
        self.predictive_brake = None

    def compute_action(self, observation, helpers):
        start = time.perf_counter()
        q, dq = np.asarray(observation['current_q']), np.asarray(observation['current_dq'])
        terms = self.terms(q, dq, helpers)
        t = terms[0]
        e = t['gravity_error']/9.81
        J = t['G_g'][:, :5]/9.81
        # Gravity projection rate due to a world angular velocity.
        T = J @ np.linalg.pinv(t['B_alpha'], rcond=1e-5)
        omega = t['D_omega'] + t['C_omega']@dq
        de = T@omega
        previous_integral = self.integral.copy()
        self.integral = np.clip(self.integral + e*self.dt, -.1, .1)
        target = (-float(self.config.get('simple_apid_kp', 400.))*e
                  -float(self.config.get('simple_apid_kd', 40.))*de
                  -float(self.config.get('simple_apid_ki', 50.))*self.integral)
        k = min(self.N-1, max(0, int(self.config.get('simple_acc_preview_steps', 0))))
        ff = terms[k] if self.ff else None
        angular_offset = (ff['D_alpha']+ff['C_alpha']@dq) if self.ff else np.zeros(3)
        linear_offset = (ff['D_acc']+ff['C_acc']@dq) if self.ff else np.zeros(3)
        reg = float(self.config.get('simple_apid_regularization', .003))
        wa = float(self.config.get('simple_apid_acc_weight', .1))
        posture = (float(self.config.get('simple_apid_posture_kp', 4.))*(self.neutral-q)
                   -float(self.config.get('simple_apid_posture_kd', 2.))*dq)
        A = np.vstack((J, np.sqrt(wa)*t['B_acc'], np.sqrt(reg)*np.eye(5)))
        b = np.r_[target-T@angular_offset, -np.sqrt(wa)*linear_offset, np.sqrt(reg)*posture]
        if self.config.get('simple_apid_full_rotation', False):
            target_world = (np.linalg.pinv(T, rcond=1e-5)@(
                -float(self.config.get('simple_apid_kp', 400.))*e
                -float(self.config.get('simple_apid_ki', 50.))*self.integral)
                -float(self.config.get('simple_apid_kd', 40.))*omega)
            A = np.vstack((t['B_alpha'], np.sqrt(wa)*t['B_acc'], np.sqrt(reg)*np.eye(5)))
            b = np.r_[target_world-angular_offset, -np.sqrt(wa)*linear_offset, np.sqrt(reg)*posture]
        raw = np.linalg.solve(A.T@A+1e-9*np.eye(5), A.T@b)
        low = np.maximum(-self.amax, (-self.vmax-dq)/self.dt)
        high = np.minimum(self.amax, (self.vmax-dq)/self.dt)
        low = np.maximum(low, 2*(self.lo-q-dq*self.dt)/self.dt**2)
        high = np.minimum(high, 2*(self.hi-q-dq*self.dt)/self.dt**2)
        feasible = low <= high
        ddq = np.where(feasible, np.clip(raw, low, high), np.clip(-dq/self.dt, -self.amax, self.amax))
        bounded_allocation = bool(self.config.get('simple_apid_bounded_allocation', False))
        allocation_iterations = 0
        if bounded_allocation:
            # Optional C refinement: solve the same task allocation WITH the
            # acceleration/state box instead of clipping each joint afterward.
            # Actual torque/contact checks remain in the unchanged executor.
            import daqp
            if self.config.get('simple_apid_predictive_brake', False):
                from tools.g1_commissioning.hardware_mpc_braking import LatchedPredictiveBrake
                if self.predictive_brake is None:
                    self.predictive_brake = LatchedPredictiveBrake(5)
                weights, (blo, bhi), _ = self.predictive_brake.update(q, dq, self.lo, self.hi,
                    deceleration=4., reaction_s=.012, margin_rad=np.deg2rad(1.), weight=50.,
                    max_ddq=np.full(5, self.amax), minimum_deceleration=6.)
                low, high = np.maximum(low, blo), np.minimum(high, bhi)
                feasible = low <= high
                A = np.vstack((A, np.diag(np.sqrt(weights))*self.dt))
                b = np.r_[b, -np.sqrt(weights)*dq]
                raw = np.linalg.solve(A.T@A+1e-9*np.eye(5), A.T@b)
                ddq = raw.copy()
            if not np.all(feasible):
                raise RuntimeError('C bounded allocation has infeasible next-state bounds')
            if np.any(raw < low) or np.any(raw > high):
                ddq, _, flag, info = daqp.solve(np.ascontiguousarray(A.T@A+1e-9*np.eye(5)),
                    np.ascontiguousarray(-A.T@b), np.eye(5), np.ascontiguousarray(high),
                    np.ascontiguousarray(low), primal_tol=1e-9, dual_tol=1e-10, eps_prox=0.,
                    iter_limit=500)
                allocation_iterations = int(info['iterations'])
                if (flag != 1 or not np.isfinite(ddq).all()
                        or np.any(ddq < low-1e-6) or np.any(ddq > high+1e-6)):
                    raise RuntimeError(f'C bounded allocation rejected: {flag}')
            if np.max(np.abs(ddq-raw)) > 1e-6:
                self.integral = previous_integral
        ref = np.clip(q+dq*self.dt+.5*ddq*self.dt**2, self.lo, self.hi)
        velocity = np.clip(dq+ddq*self.dt, -self.vmax, self.vmax)
        self.finish(ref, velocity, start, gravity_error=t['gravity_error'],
                    ddq_raw=raw, ddq_des=ddq, ddq_output_unused=False,
                    bounded_allocation=bounded_allocation, allocation_iterations=allocation_iterations)
        return ref, velocity, ddq


def make_setup(factory, config, controller_name, neutral, dt):
    setup = factory(config, controller_name, neutral, dt)
    family = config.get('simple_controller', 'acceleration_mpc')
    if family == 'acceleration_mpc':
        return setup
    classes = {'ff_pid': FeedforwardPID, 'position_mpc': PositionMPC,
               'acceleration_pid': AccelerationPID}
    if family == 'original_pid':
        policy = OriginalPIDAdapter(config, neutral, dt, factory(config, 'pid', neutral, dt).policy)
    else:
        policy = classes[family](config, neutral, dt)
    acceleration = family == 'acceleration_pid'
    metadata = dict(setup.metadata, actual_control_family=family,
                    simulation_only=True, inverse_dynamics_used=acceleration,
                    unused_zero_ddq_wire_placeholder=not acceleration,
                    research_parameters={k: v for k, v in config.items() if k.startswith('simple_')})
    return replace(setup, policy=policy, policy_type=family,
                   notes='simulation controller comparison',
                   metadata=metadata, acceleration_controller=acceleration)
