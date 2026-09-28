"""Exact state elimination of the existing MPC QP; hardware-only solve adapter.

For nine intervals, optimize 45 acceleration variables rather than 145 state+
input variables. z=E*x0+T*u enforces the integrator equalities algebraically.
Cost terms and inequality bounds are unchanged, then the full trajectory is
reconstructed and checked by ArmMPCPolicy. No relaxed safety constraints.
"""
from types import SimpleNamespace
import time
import numpy as np
from scipy import sparse
import osqp
from arm_mpc import ArmMPCPolicy


class CondensedArmMPCPolicy(ArmMPCPolicy):
    def __init__(self, *args, solver_backend="daqp", **kwargs):
        super().__init__(*args, **kwargs)
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

    def _build_cost(self, step_terms):
        """Batch the unchanged seven objective terms across the nine stages."""
        terms = {key: np.stack([t[key] for t in step_terms]) for key in
                 ("C_omega", "D_omega", "G_g", "d_g", "C_acc", "B_acc", "D_acc",
                  "C_alpha", "B_alpha", "D_alpha")}
        transpose = lambda a: a.transpose(0, 2, 1)
        ew, gg = terms["C_omega"] @ self.Sv, terms["G_g"]
        qxx = (self._state_regularization_hessian[None] + transpose(ew) @ self.Q_ee_omega @ ew
               + transpose(gg) @ self.Qg @ gg)
        fx = (self._posture_linear_cost[None, :, None]
              + transpose(ew) @ self.Q_ee_omega @ terms["D_omega"][..., None]
              + transpose(gg) @ self.Qg @ terms["d_g"][..., None])
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
        return [*blocks,terminal_h],linear

    def reset(self):
        super().reset()
        if getattr(self, "_condensed_solver", None) is not None:
            self._condensed_solver.warm_start(x=np.zeros(self.horizon*self.nu),
                                              y=np.zeros(self._ac.shape[0]))

    def condense(self, P, linear, lower, upper):
        upper_triangle = P.toarray()
        full_p = upper_triangle + upper_triangle.T - np.diag(np.diag(upper_triangle))
        offset = self._E @ lower[:self.nx]
        pc = self._T.T @ full_p @ self._T
        pc = .5*(pc+pc.T)
        qc = self._T.T @ (full_p @ offset + linear)
        constraint_offset = self._constraint @ offset
        lc = (lower[self._ineq_start:]-constraint_offset)*self._row_scale
        uc = (upper[self._ineq_start:]-constraint_offset)*self._row_scale
        return pc, qc, lc, uc, offset, full_p

    def _solve_qp(self, P, p_values, linear, lower, upper, warm_start):
        try:
            pc, qc, lc, uc, offset, full_p = self.condense(P, linear, lower, upper)
            if self.solver_backend == "daqp":
                begin = time.perf_counter()
                u, _, flag, details = self._daqp.solve(
                    np.ascontiguousarray(pc), np.ascontiguousarray(qc), self._ac_dense,
                    np.ascontiguousarray(uc), np.ascontiguousarray(lc),
                    primal_tol=1e-7, dual_tol=1e-10, eps_prox=0.,
                    iter_limit=self.solver_max_iter, time_limit=self.solver_time_limit)
                wall = time.perf_counter()-begin
                success = flag == 1 and wall <= self.solver_time_limit
                full_solution = offset + self._T @ u
                residual = self._ac_dense @ u
                primal = float(max(np.max(lc-residual), np.max(residual-uc), 0.))
                dual = float(np.max(np.abs(pc@u+qc+self._ac_dense.T@details["lam"])))
                status = "solved" if success else (
                    "daqp_wall_budget_exceeded" if flag == 1 else f"daqp_exitflag_{flag}")
                info = SimpleNamespace(status=status, status_val=1 if success else -1,
                    obj_val=float(.5*full_solution@full_p@full_solution+linear@full_solution),
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
                info.obj_val = float(.5*full_solution @ full_p @ full_solution + linear @ full_solution)
            return SimpleNamespace(x=full_solution, info=info), None
        except Exception as exc:
            return None, exc

    def _check_result(self, result, lower, upper):
        solved, solution, status, status_val, violation = super()._check_result(result, lower, upper)
        if solved and violation > 1e-6:
            return False, solution, status+":full_constraint_residual", status_val, violation
        return solved, solution, status, status_val, violation
