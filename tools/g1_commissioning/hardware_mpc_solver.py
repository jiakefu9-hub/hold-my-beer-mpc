"""Exact state elimination of the existing MPC QP; hardware-only solve adapter.

For nine intervals, optimize 45 acceleration variables rather than 145 state+
input variables. z=E*x0+T*u enforces the integrator equalities algebraically.
Simulation cost terms and bounds are preserved; the field controller can add
a soft early-braking velocity cost. The full trajectory is reconstructed and
checked by ArmMPCPolicy. No relaxed safety constraints.
"""
from types import SimpleNamespace
import time
import numpy as np
from scipy import sparse
import osqp
from arm_mpc import ArmMPCPolicy


class CondensedArmMPCPolicy(ArmMPCPolicy):
    def __init__(self, *args, solver_backend="daqp", q_ee_vel=0.0, **kwargs):
        super().__init__(*args, q_ee_vel=q_ee_vel, **kwargs)
        if solver_backend not in {"daqp", "osqp"}:
            raise ValueError("unknown explicit QP backend")
        self.solver_backend = solver_backend
        self._daqp = None
        if solver_backend == "daqp":
            import daqp
            self._daqp = daqp
        n = self.horizon * self.nu
        self._input_scale = np.tile(self.max_ddq, self.horizon)
        self._E = np.zeros((self.num_variables, self.nx))
        self._T = np.zeros((self.num_variables, n))
        state_e, state_t = np.eye(self.nx), np.zeros((self.nx, n))
        for k in range(self.horizon + 1):
            sl = slice(self._cx(k), self._cx(k) + self.nx)
            self._E[sl], self._T[sl] = state_e, state_t
            if k < self.horizon:
                control = slice(k*self.nu, (k+1)*self.nu)
                self._T[self._cu(k):self._cu(k)+self.nu, control] = np.diag(self.max_ddq)
                state_e = self.A @ state_e
                state_t = self.A @ state_t
                state_t[:, control] += self.B * self.max_ddq[None, :]
        self._ineq_start = (self.horizon + 1)*self.nx
        self._constraint = self._A_cons[self._ineq_start:]
        self._constraint_e = self._constraint @ self._E
        self._stage_t = self._T[:self.horizon*self.stage_dim].reshape(self.horizon,self.stage_dim,n)
        self._stage_tt = self._stage_t.transpose(0,2,1)
        self._terminal_t = self._T[self.horizon*self.stage_dim:]
        self._cost_blocks = None
        ac = self._constraint @ self._T
        # Normalize each inequality's coefficient magnitude, including the
        # microradian position increments near a reference boundary. Otherwise
        # small q-row coefficients can be effectively ignored by the ADMM
        # residual test even while input-bound residuals remain significant.
        self._row_scale = 1 / np.maximum(np.max(np.abs(ac), axis=1), 1e-12)
        self._ac = sparse.csc_matrix(ac*self._row_scale[:, None])
        self._ac_dense = np.ascontiguousarray(self._ac.toarray())
        self._cols = np.repeat(np.arange(n), np.arange(1, n+1))
        self._rows = np.concatenate([np.arange(k+1) for k in range(n)])
        self._input_rows = np.concatenate([np.arange(self._cu(k), self._cu(k)+self.nu)
                                           for k in range(self.horizon)])
        self._condensed_solver = None
        self._local_actuation = None
        self._last_local_actuation = None
        self._first_acceleration_bounds = None
        self._braking_velocity_cost = np.zeros(self.nu)

    def set_local_actuation_constraints(self, mass, bias, total_limit, ff_limit, feedback_dq):
        """Expose the executor's existing local torque envelope to the QP.

        The conditional arm model is frozen for this 54 ms horizon.  This is
        the same local approximation used by the final forward check, not a
        new actuator limit. Only absolute total and feedforward limits are
        planned. There are no inter-tick or future torque-rate constraints.
        """
        vectors = [np.asarray(x, dtype=np.float64) for x in
                   (bias, total_limit, ff_limit, feedback_dq)]
        if any(x.shape != (self.nu,) or not np.all(np.isfinite(x)) for x in vectors):
            raise ValueError('local actuation vectors must be finite five-vectors')
        bias, total_limit, ff_limit, feedback_dq = vectors
        mass = np.asarray(mass, dtype=np.float64)
        if (mass.shape != (self.nu, self.nu) or not np.all(np.isfinite(mass))
                or np.any(total_limit <= 0) or np.any(ff_limit <= 0)):
            raise ValueError('invalid local actuation model/envelope')
        np.linalg.cholesky(.5*(mass+mass.T))
        self._local_actuation = dict(mass=mass.copy(), bias=bias.copy(),
            total_limit=total_limit.copy(), ff_limit=ff_limit.copy(),
            feedback_dq=feedback_dq.copy())

    def clear_local_actuation_constraints(self):
        self._local_actuation = None

    def set_first_acceleration_bounds(self, lower=None, upper=None):
        """Optionally restrict only the acceleration that will be transmitted."""
        if lower is None and upper is None:
            self._first_acceleration_bounds = None
            return
        lower, upper = [np.asarray(x, dtype=np.float64) for x in (lower, upper)]
        if (lower.shape != (self.nu,) or upper.shape != (self.nu,)
                or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
                or np.any(lower > upper) or np.any(lower < -self.max_ddq)
                or np.any(upper > self.max_ddq)):
            raise ValueError('invalid first acceleration bounds')
        self._first_acceleration_bounds = (lower.copy(), upper.copy())

    def set_braking_velocity_cost(self, weights):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.nu,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError('braking weights must be finite nonnegative five-vector')
        self._braking_velocity_cost = weights.copy()

    def _local_actuation_rows(self):
        """Return normalized-input rows for torque and optional first-action braking."""
        data = self._local_actuation
        if data is None and self._first_acceleration_bounds is None:
            return None
        n, m, dt = self.horizon, self.nu, self.control_dt
        scale = np.diag(self.max_ddq)
        rows, lower, upper = [], [], []

        if data is not None:
            mass = data['mass']
            # Frozen local total-torque model: tau_k = M*a_k + b.
            for k in range(n):
                row = np.zeros((m, n*m))
                row[:, k*m:(k+1)*m] = mass @ scale
                rows.append(row)
                lower.append(-data['total_limit']-data['bias'])
                upper.append(data['total_limit']-data['bias'])

            # Packet feedforward after the device-side one-step PD law.  Only the
            # first action is transmitted; the next solve rechecks its own first
            # action against fresh feedback.  Keeping this to one row block avoids
            # burdening the 6 ms solve with limits on commands that are never sent.
            kp = np.asarray(getattr(self, '_local_kp', np.zeros(m)), dtype=np.float64)
            kd = np.asarray(getattr(self, '_local_kd', np.zeros(m)), dtype=np.float64)
            pd_a = np.diag(.5*kp*dt**2+kd*dt)
            current = (mass-pd_a) @ scale
            offset = data['bias']-kp*dt*data['feedback_dq']
            row = np.zeros((m, n*m)); row[:, :m] = current
            rows.append(row)
            lower.append(-data['ff_limit']-offset)
            upper.append(data['ff_limit']-offset)

        if self._first_acceleration_bounds is not None:
            # u is normalized; multiplying by max_ddq gives physical rad/s^2.
            row = np.zeros((m, n*m)); row[:, :m] = scale
            rows.append(row)
            lower.append(self._first_acceleration_bounds[0])
            upper.append(self._first_acceleration_bounds[1])

        matrix = np.vstack(rows)
        lo, hi = np.concatenate(lower), np.concatenate(upper)
        row_scale = 1/np.maximum(np.max(np.abs(matrix), axis=1), 1e-12)
        return (np.ascontiguousarray(matrix*row_scale[:, None]),
                np.ascontiguousarray(lo*row_scale),
                np.ascontiguousarray(hi*row_scale))

    def _build_cost(self, step_terms):
        """Batch simulation terms, plus optional soft velocity braking cost."""
        terms = {key: np.stack([t[key] for t in step_terms]) for key in
                 ("C_omega", "D_omega", "G_g", "d_g", "C_acc", "B_acc", "D_acc",
                  "C_alpha", "B_alpha", "D_alpha")}
        transpose = lambda a: a.transpose(0, 2, 1)
        ew, gg = terms["C_omega"] @ self.Sv, terms["G_g"]
        qxx = (self._state_regularization_hessian[None] + transpose(ew) @ self.Q_ee_omega @ ew
               + transpose(gg) @ self.Qg @ gg)
        velocity_indices = np.arange(self.nu, self.nx)
        qxx[:, velocity_indices, velocity_indices] += self._braking_velocity_cost
        fx = (self._posture_linear_cost[None, :, None]
              + transpose(ew) @ self.Q_ee_omega @ terms["D_omega"][..., None]
              + transpose(gg) @ self.Qg @ terms["d_g"][..., None])
        if self._linear_velocity_cost_active:
            ev = np.stack([t['C_vel'] for t in step_terms]) @ self.Sv
            dv = np.stack([t['D_vel'] for t in step_terms])[..., None]
            qxx += transpose(ev) @ self.Q_ee_vel @ ev
            fx += transpose(ev) @ self.Q_ee_vel @ dv
        terminal_h = 2*self.terminal_scale*qxx[-1] + self.reg*np.eye(self.nx)
        terminal_f = 2*self.terminal_scale*fx[-1, :, 0]
        qxx, fx = qxx[:-1].copy(), fx[:-1].copy()
        qxu = np.zeros((self.horizon,self.nx,self.nu))
        quu = np.tile(self.R, (self.horizon,1,1))
        fu = np.zeros((self.horizon,self.nu,1))
        for suffix, weight, active in (("acc",self.Q_ee_acc,self._linear_acceleration_cost_active),
                                       ("alpha",self.Q_ee_alpha,self._angular_acceleration_cost_active)):
            if not active:
                continue
            e=terms["C_"+suffix][:-1]@self.Sv
            b=terms["B_"+suffix][:-1]
            d=terms["D_"+suffix][:-1,:,None]
            eq,bq=transpose(e)@weight,transpose(b)@weight
            qxx+=eq@e; qxu+=eq@b; quu+=bq@b; fx+=eq@d; fu+=bq@d
        blocks=np.empty((self.horizon,self.stage_dim,self.stage_dim))
        blocks[:,:self.nx,:self.nx]=qxx
        blocks[:,:self.nx,self.nx:]=qxu
        blocks[:,self.nx:,:self.nx]=transpose(qxu)
        blocks[:,self.nx:,self.nx:]=quu
        blocks=blocks+transpose(blocks)+self.reg*np.eye(self.stage_dim)[None]
        linear=np.r_[2*np.concatenate((fx,fu),axis=1).reshape(-1),terminal_f]
        self._cost_blocks = (blocks, terminal_h)
        return [*blocks,terminal_h],linear

    def get_cost_definition(self):
        definition = super().get_cost_definition()
        definition['linear_velocity_semantics'] = 'v_E-v_IMU = omega_IMU x r_IMU_E + J_v*dq; fixed H0'
        return definition

    def _build_one_step_diagnostics(self, q, dq, q_ref, dq_ref, ddq,
                                    acceleration_terms, end_state_terms):
        result = super()._build_one_step_diagnostics(
            q, dq, q_ref, dq_ref, ddq, acceleration_terms, end_state_terms)
        if self._linear_velocity_cost_active:
            result['ee_lin_vel_relative_imu_h0_m_s'] = result.pop('ee_lin_vel_relative_imu_m_s')
            result['ee_lin_vel_relative_imu_offset_h0_m_s'] = result.pop('ee_lin_vel_relative_imu_offset_m_s')
        return result

    def reset(self):
        super().reset()
        self._braking_velocity_cost = np.zeros(self.nu)
        self._local_actuation = None
        self._last_local_actuation = None
        self._first_acceleration_bounds = None
        if getattr(self, "_condensed_solver", None) is not None:
            self._condensed_solver.warm_start(x=np.zeros(self.horizon*self.nu),
                                              y=np.zeros(self._ac.shape[0]))

    def _cost_matrix_for_solver(self, p_values):
        # The condensed solver consumes the exact stage blocks directly.
        # Do not rebuild and then densify the unused 145x145 sparse matrix.
        if self._cost_blocks is None:
            raise RuntimeError('condensed stage costs were not assembled')
        return None

    def _objective(self, solution, linear, full_p):
        if full_p is not None:
            return float(.5*solution@full_p@solution+linear@solution)
        blocks, terminal = self._cost_blocks
        end = self.horizon*self.stage_dim
        stages = solution[:end].reshape(self.horizon,self.stage_dim)
        tail = solution[end:]
        return float(.5*(np.einsum('ki,kij,kj->',stages,blocks,stages)
                          + tail@terminal@tail)+linear@solution)

    def condense(self, P, linear, lower, upper, *, cost_blocks=None):
        full_p = None
        if P is not None:
            upper_triangle = P.toarray()
            full_p = upper_triangle + upper_triangle.T - np.diag(np.diag(upper_triangle))
        offset = self._E @ lower[:self.nx]
        if cost_blocks is None:
            if full_p is None:
                raise ValueError('condensing needs either a Hessian or exact stage blocks')
            pc = self._T.T @ full_p @ self._T
            qc = self._T.T @ (full_p @ offset + linear)
        else:
            blocks, terminal = cost_blocks
            # Same block-diagonal Hessian; do not multiply its large zero
            # off-diagonal regions every 6 ms. Full trajectory checks remain.
            ht = blocks @ self._stage_t
            pc = np.sum(self._stage_tt @ ht, axis=0) + self._terminal_t.T @ terminal @ self._terminal_t
            stage_end = self.horizon*self.stage_dim
            sx = offset[:stage_end].reshape(self.horizon,self.stage_dim,1)
            sf = linear[:stage_end].reshape(self.horizon,self.stage_dim,1)
            qc = np.sum(self._stage_tt @ (blocks@sx+sf),axis=0)[:,0]
            qc += self._terminal_t.T @ (terminal@offset[stage_end:]+linear[stage_end:])
        pc = .5*(pc+pc.T)
        constraint_offset = self._constraint_e @ lower[:self.nx]
        lc = (lower[self._ineq_start:]-constraint_offset)*self._row_scale
        uc = (upper[self._ineq_start:]-constraint_offset)*self._row_scale
        return pc, qc, lc, uc, offset, full_p

    def _solve_qp(self, P, p_values, linear, lower, upper, warm_start):
        try:
            pc, qc, lc, uc, offset, full_p = self.condense(P, linear, lower, upper,
                                                        cost_blocks=self._cost_blocks)
            local = self._local_actuation_rows()
            self._last_local_actuation = local
            solve_matrix = self._ac_dense
            if local is not None:
                if self.solver_backend != 'daqp':
                    raise ValueError('local actuation constraints require the DAQP backend')
                solve_matrix = np.ascontiguousarray(np.vstack((solve_matrix, local[0])))
                lc = np.ascontiguousarray(np.r_[lc, local[1]])
                uc = np.ascontiguousarray(np.r_[uc, local[2]])
            if self.solver_backend == "daqp":
                begin = time.perf_counter()
                u, _, flag, details = self._daqp.solve(
                    np.ascontiguousarray(pc), np.ascontiguousarray(qc), solve_matrix,
                    np.ascontiguousarray(uc), np.ascontiguousarray(lc),
                    primal_tol=1e-7, dual_tol=1e-10, eps_prox=0.,
                    iter_limit=self.solver_max_iter, time_limit=self.solver_time_limit)
                wall = time.perf_counter()-begin
                wall_over_budget = wall > self.solver_time_limit
                self._last_solved_wall_over_budget = flag == 1 and wall_over_budget
                success = flag == 1 and (not wall_over_budget or
                                         getattr(self, 'allow_solved_wall_overrun', False))
                full_solution = offset + self._T @ u
                residual = solve_matrix @ u
                primal = float(max(np.max(lc-residual), np.max(residual-uc), 0.))
                dual = float(np.max(np.abs(pc@u+qc+solve_matrix.T@details["lam"])))
                status = "solved" if success else (
                    "daqp_wall_budget_exceeded" if flag == 1 else f"daqp_exitflag_{flag}")
                info = SimpleNamespace(status=status, status_val=1 if success else -1,
                    obj_val=self._objective(full_solution,linear,full_p),
                    iter=int(details["iterations"]), prim_res=primal, dual_res=dual,
                    run_time=wall, setup_time=float(details["setup_time"]),
                    solve_time=float(details["solve_time"]), update_time=0.)
                return SimpleNamespace(x=full_solution,info=info),None
            values = pc[self._rows, self._cols]
            if self._condensed_solver is None:
                self._condensed_solver = osqp.OSQP()
                matrix = sparse.csc_matrix((values, (self._rows, self._cols)), shape=pc.shape)
                self._condensed_solver.setup(P=matrix, q=qc, A=self._ac, l=lc, u=uc,
                    verbose=False, eps_abs=self.solver_eps_abs, eps_rel=self.solver_eps_rel,
                    max_iter=self.solver_max_iter, check_termination=self.solver_check_termination,
                    rho=self.solver_rho, adaptive_rho=self.solver_adaptive_rho,
                    adaptive_rho_interval=25,
                    time_limit=self.solver_time_limit, polishing=False, warm_starting=True)
            else:
                self._condensed_solver.update(Px=values, q=qc, l=lc, u=uc)
            self._condensed_solver.warm_start(x=warm_start[self._input_rows]/self._input_scale)
            result = self._condensed_solver.solve(raise_error=False)
            full_solution = None if result.x is None else offset + self._T @ result.x
            info = SimpleNamespace(**vars(result.info))
            if full_solution is not None and np.isfinite(full_solution).all():
                info.obj_val = self._objective(full_solution,linear,full_p)
            return SimpleNamespace(x=full_solution, info=info), None
        except Exception as exc:
            return None, exc

    def _check_result(self, result, lower, upper):
        solved, solution, status, status_val, violation = super()._check_result(result, lower, upper)
        if solved and self._last_local_actuation is not None:
            matrix, lo, hi = self._last_local_actuation
            normalized = solution[self._input_rows]/self._input_scale
            value = matrix @ normalized
            local_violation = float(max(np.max(lo-value), np.max(value-hi), 0.))
            violation = max(violation, local_violation)
            if local_violation > 1e-6:
                return False, solution, status+':local_actuation_residual', status_val, violation
        if solved and violation > 1e-6:
            return False, solution, status+":full_constraint_residual", status_val, violation
        return solved, solution, status, status_val, violation
