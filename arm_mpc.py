import time

import numpy as np
from scipy import sparse

try:
    import osqp
except ImportError:  # PID/LQR 模式不应因为缺少 MPC 可选依赖而无法启动。
    osqp = None


class ArmMPCPolicy:
    """右臂 5 关节线性时变 MPC。

    状态为 ``x = [q, dq]``，控制输入为 ``u = ddq``。控制器每次只执行
    QP 解的第一拍，并返回 ``(q_ref, dq_ref, ddq_des)`` 给下层力矩执行链。
    """

    COST_TERM_NAMES = (
        "linear_acceleration",
        "angular_acceleration",
        "gravity",
        "posture",
        "velocity",
        "control",
    )

    # 关节顺序：shoulder pitch / roll / yaw / elbow pitch / wrist roll。
    DEFAULT_JOINT_LIMITS = np.deg2rad(
        np.array(
            [
                [-5.0, 5.0],
                [-5.0, 3.0],
                [-20.0, 5.0],
                [-40.0, 40.0],
                [-40.0, 40.0],
            ],
            dtype=np.float64,
        )
    )

    def __init__(
        self,
        default_q,
        control_dt=0.006,
        horizon=12,
        q_ee_acc=1.0,
        q_ee_alpha=0.075,
        q_gravity=30.0,
        q_posture=0.05,
        q_vel=0.02,
        r_ddq=0.25,
        terminal_scale=2.0,
        reg=1e-6,
        max_dq=1.0,
        max_ddq=8.0,
        failure_braking_gain=4.0,
        failure_posture_gain=4.0,
        failure_max_ddq_scale=0.5,
        joint_limits=None,
        joint_limit_margin=0.0,
        solver_eps_abs=1e-4,
        solver_eps_rel=1e-4,
        solver_max_iter=1000,
        solver_rho=0.1,
        solver_adaptive_rho=True,
        solver_scaled_termination=False,
        solver_polishing=False,
        solver_verbose=False,
        solver_time_limit=None,
    ):
        # 【非核心代码】参数检查和权重整理。
        if osqp is None:
            raise ImportError(
                "MPC 模式需要 OSQP；请在 g1_mpc 环境执行 `pip install osqp`。"
            )
        self.default_q = np.asarray(default_q, dtype=np.float64).reshape(-1).copy()
        self.n = self.default_q.size
        if self.n != 5:
            raise ValueError(f"ArmMPCPolicy 固定控制 5 个右臂关节，当前得到 {self.n} 个。")

        self.nx = 2 * self.n
        self.nu = self.n
        self.stage_dim = self.nx + self.nu
        self.control_dt = float(control_dt)
        self.horizon = int(horizon)
        if not np.isfinite(self.control_dt) or self.control_dt <= 0.0:
            raise ValueError("control_dt 必须是正数。")
        if self.horizon < 1:
            raise ValueError("horizon 必须至少为 1。")

        self.Q_ee_acc = self._make_weight(q_ee_acc, 3, "q_ee_acc")
        self.Q_ee_alpha = self._make_weight(q_ee_alpha, 3, "q_ee_alpha")
        # MPC 使用末端系重力向量的 x/y 分量，因此这里严格是 2x2。
        self.Qg = self._make_weight(q_gravity, 2, "q_gravity")
        self.Qq = self._make_weight(q_posture, self.n, "q_posture")
        self.Qv = self._make_weight(q_vel, self.n, "q_vel")
        self.R = self._make_weight(r_ddq, self.nu, "r_ddq", positive_definite=True)

        self.terminal_scale = float(terminal_scale)
        self.reg = float(reg)
        if not np.isfinite(self.terminal_scale) or self.terminal_scale < 0.0:
            raise ValueError("terminal_scale 必须是非负数。")
        if not np.isfinite(self.reg) or self.reg < 0.0:
            raise ValueError("reg 必须是非负数。")

        self.max_dq = self._make_positive_vector(max_dq, self.n, "max_dq")
        self.max_ddq = self._make_positive_vector(max_ddq, self.nu, "max_ddq")
        self.failure_braking_gain = float(failure_braking_gain)
        if not np.isfinite(self.failure_braking_gain) or self.failure_braking_gain <= 0.0:
            raise ValueError("failure_braking_gain 必须是正数。")
        self.failure_posture_gain = float(failure_posture_gain)
        if not np.isfinite(self.failure_posture_gain) or self.failure_posture_gain < 0.0:
            raise ValueError("failure_posture_gain 必须是非负数。")
        self.failure_max_ddq_scale = float(failure_max_ddq_scale)
        if (
            not np.isfinite(self.failure_max_ddq_scale)
            or self.failure_max_ddq_scale <= 0.0
            or self.failure_max_ddq_scale > 1.0
        ):
            raise ValueError("failure_max_ddq_scale 必须位于 (0, 1]。")
        self.joint_limit_margin = self._make_nonnegative_vector(
            joint_limit_margin, self.n, "joint_limit_margin"
        )
        self.safety_joint_limits = None
        self.joint_limits = None
        self.set_joint_limits(
            self.DEFAULT_JOINT_LIMITS if joint_limits is None else joint_limits
        )

        self.solver_eps_abs = float(solver_eps_abs)
        self.solver_eps_rel = float(solver_eps_rel)
        self.solver_max_iter = int(solver_max_iter)
        self.solver_rho = float(solver_rho)
        self.solver_adaptive_rho = bool(solver_adaptive_rho)
        self.solver_scaled_termination = bool(solver_scaled_termination)
        self.solver_polishing = bool(solver_polishing)
        self.solver_verbose = bool(solver_verbose)
        self.solver_time_limit = (
            None if solver_time_limit is None else float(solver_time_limit)
        )
        if self.solver_eps_abs <= 0.0 or self.solver_eps_rel <= 0.0:
            raise ValueError("OSQP 的 eps_abs / eps_rel 必须是正数。")
        if self.solver_max_iter < 1:
            raise ValueError("solver_max_iter 必须至少为 1。")
        if not np.isfinite(self.solver_rho) or self.solver_rho <= 0.0:
            raise ValueError("solver_rho 必须是正数。")
        if self.solver_time_limit is not None and self.solver_time_limit <= 0.0:
            raise ValueError("solver_time_limit 必须是正数或 None。")

        # 【半核心代码】二阶积分模型和固定选择矩阵。
        eye = np.eye(self.n, dtype=np.float64)
        zeros = np.zeros_like(eye)
        dt = self.control_dt
        self.A = np.block([[eye, dt * eye], [zeros, eye]])
        self.B = np.vstack([0.5 * dt * dt * eye, dt * eye])
        self.Sq = np.hstack([eye, zeros])
        self.Sv = np.hstack([zeros, eye])

        self.num_variables = (self.horizon + 1) * self.nx + self.horizon * self.nu
        self._solver = None
        self._last_u_plan = None
        self._last_solution = None

        # A_cons 的稀疏结构恒定；P 的每个阶段块使用固定的上三角稀疏结构。
        self._A_cons, self._l_template, self._u_template = self._build_constraints()
        self._p_rows, self._p_cols = self._build_hessian_pattern()
        self._last_diagnostics = self._empty_diagnostics()

    def compute_action(self, arm_obs, helpers=None):
        """求解一轮 MPC，返回 ``q_ref, dq_ref, ddq_des``。

        ``helpers.compute_mpc_terms(q, dq, disturbance)`` 必须返回：
        ``D_acc/C_acc/B_acc/D_alpha/C_alpha/B_alpha/G_g/d_g/gravity_error``。
        其中 ``G_g`` 为 2x10，``d_g`` 和 ``gravity_error`` 均为 2 维。
        """
        q = np.asarray(self._obs_get(arm_obs, "current_q"), dtype=np.float64).reshape(-1)
        dq = np.asarray(self._obs_get(arm_obs, "current_dq"), dtype=np.float64).reshape(-1)
        if q.shape != (self.n,) or dq.shape != (self.n,):
            raise ValueError(
                f"current_q/current_dq 必须都是 shape={(self.n,)}，"
                f"当前为 {q.shape}/{dq.shape}。"
            )
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            raise ValueError("current_q/current_dq 包含 NaN 或 Inf。")

        observed_dt = float(self._obs_get(arm_obs, "dt", self.control_dt))
        if not np.isclose(observed_dt, self.control_dt, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"观测 dt={observed_dt} 与 MPC 固定离散周期 "
                f"{self.control_dt} 不一致。"
            )

        terms_fn, disturbance_prediction = self._resolve_terms_helper(helpers)
        x_measured = np.concatenate([q, dq])

        start_time = time.perf_counter()

        # 【核心代码】平移上一拍输入，并严格按二阶积分器生成本拍工作轨迹。
        working_inputs = self._shift_input_plan()
        working_states = self._rollout(x_measured, working_inputs)

        # 【核心代码】关闭前馈时序列是当前测量的零阶保持；开启时每步使用
        # 相位模板预测，并带有从当前姿态积分得到的 R_B,k。
        step_terms = []
        for k in range(self.horizon + 1):
            raw_terms = terms_fn(
                working_states[k, : self.n],
                working_states[k, self.n :],
                disturbance_prediction[k],
            )
            step_terms.append(self._validate_step_terms(raw_terms, k))

        # 【核心代码】组装各阶段二次代价；这里没有 torso-relative 位置项。
        stage_hessians, linear_cost = self._build_cost(step_terms)
        p_values = self._pack_upper_triangles(stage_hessians)
        P = sparse.csc_matrix(
            (p_values, (self._p_rows, self._p_cols)),
            shape=(self.num_variables, self.num_variables),
        )
        if P.nnz != len(self._p_rows):
            raise RuntimeError("MPC Hessian 的固定稀疏结构发生变化。")

        (
            lower,
            upper,
            recovery_states,
            recovery_inputs,
            recovery_active,
        ) = self._build_online_constraint_bounds(q, dq)
        lower[: self.nx] = x_measured
        upper[: self.nx] = x_measured
        # 状态已在正常运行盒外时，用确定性的恢复轨迹 warm-start；它与本轮
        # 时变硬边界完全一致，避免继续拿越界的上一拍轨迹初始化 OSQP。
        warm_start = self._pack_trajectory(
            recovery_states if recovery_active else working_states,
            recovery_inputs if recovery_active else working_inputs,
        )
        assembly_time = time.perf_counter() - start_time

        result, solver_error = self._solve_qp(
            P, p_values, linear_cost, lower, upper, warm_start
        )
        (
            solved,
            solution,
            solver_status,
            solver_status_val,
            max_violation,
        ) = self._check_result(result, lower, upper)

        fallback_used = not solved
        fallback_feasible = True
        if solved:
            predicted_states, predicted_inputs = self._unpack_solution(solution)
            ddq_raw = predicted_inputs[0].copy()
            ddq_des = np.clip(ddq_raw, -self.max_ddq, self.max_ddq)
            self._last_u_plan = predicted_inputs.copy()
            self._last_solution = solution.copy()
        else:
            # 【半核心代码】求解失败时使用有边界的关节制动，不执行无效 QP 解。
            ddq_des, fallback_feasible = self._braking_fallback(q, dq)
            ddq_raw = ddq_des.copy()
            fallback_inputs = working_inputs.copy()
            fallback_inputs[0] = ddq_des
            predicted_states = self._rollout(x_measured, fallback_inputs)
            predicted_inputs = fallback_inputs
            if solver_error is not None:
                solver_status = f"osqp_exception:{type(solver_error).__name__}"

        q_ref = q + dq * self.control_dt + 0.5 * ddq_des * self.control_dt**2
        dq_ref = dq + ddq_des * self.control_dt
        # 正常解只会有求解容差级偏差；fallback 时裁剪可避免 PD 参考继续越界。
        q_ref = np.clip(
            q_ref,
            self.safety_joint_limits[:, 0],
            self.safety_joint_limits[:, 1],
        )
        dq_ref = np.clip(dq_ref, -self.max_dq, self.max_dq)

        one_step_prediction = self._build_one_step_diagnostics(
            q,
            dq,
            q_ref,
            dq_ref,
            ddq_des,
            acceleration_terms=step_terms[0],
            end_state_terms=step_terms[1],
        )
        min_constraint_margins = self._constraint_margins(
            predicted_states, predicted_inputs
        )
        info = None if result is None else result.info
        self._last_diagnostics = {
            "solved": bool(solved),
            "success": bool(solved),
            "fallback_used": bool(fallback_used),
            "fallback_feasible": bool(fallback_feasible),
            "solver_status": str(solver_status),
            "solver_status_val": int(solver_status_val),
            "objective": self._info_value(info, "obj_val", np.nan),
            "iterations": int(self._info_value(info, "iter", 0)),
            "primal_residual": self._info_value(info, "prim_res", np.nan),
            "dual_residual": self._info_value(info, "dual_res", np.nan),
            "prim_res": self._info_value(info, "prim_res", np.nan),
            "dual_res": self._info_value(info, "dual_res", np.nan),
            "max_constraint_violation": float(max_violation),
            "min_constraint_margins": min_constraint_margins,
            "assembly_time": float(assembly_time),
            "solver_run_time": self._info_value(info, "run_time", 0.0),
            "solver_setup_time": self._info_value(info, "setup_time", 0.0),
            "solver_update_time": self._info_value(info, "update_time", 0.0),
            "solver_solve_time": self._info_value(info, "solve_time", 0.0),
            "solve_time": self._info_value(info, "solve_time", 0.0),
            "ddq_raw": ddq_raw.copy(),
            "ddq_des": ddq_des.copy(),
            "working_states": working_states.copy(),
            "working_inputs": working_inputs.copy(),
            "predicted_states": predicted_states.copy(),
            "predicted_inputs": predicted_inputs.copy(),
            "gravity_error": step_terms[0]["gravity_error"].copy(),
            "disturbance_prediction": self._disturbance_diagnostics(
                disturbance_prediction
            ),
            "current_q_violation": float(
                max(
                    np.max(self.joint_limits[:, 0] - q),
                    np.max(q - self.joint_limits[:, 1]),
                    0.0,
                )
            ),
            "current_q_safety_violation": float(
                max(
                    np.max(self.safety_joint_limits[:, 0] - q),
                    np.max(q - self.safety_joint_limits[:, 1]),
                    0.0,
                )
            ),
            "recovery_active": bool(recovery_active),
            "one_step_prediction": one_step_prediction,
        }

        return (
            q_ref.astype(np.float32),
            dq_ref.astype(np.float32),
            ddq_des.astype(np.float32),
        )

    def get_last_diagnostics(self):
        """【非核心代码】返回最近一次 MPC 的求解、预测和回退信息。"""
        return self._copy_nested(self._last_diagnostics)

    def get_cost_definition(self):
        """【非核心代码】返回实际使用的代价权重，便于实验记录。"""
        return {
            "term_names": self.COST_TERM_NAMES,
            "Q_ee_acc": self.Q_ee_acc.copy(),
            "Q_ee_alpha": self.Q_ee_alpha.copy(),
            "Qg": self.Qg.copy(),
            "Qq": self.Qq.copy(),
            "Qv": self.Qv.copy(),
            "R": self.R.copy(),
            "posture_reference": self.default_q.copy(),
            "terminal_scale": self.terminal_scale,
        }

    def set_joint_limits(self, joint_limits):
        """更新外层安全边界，并由配置裕量得到正常运行边界。"""
        limits = np.asarray(joint_limits, dtype=np.float64)
        if limits.shape != (self.n, 2):
            raise ValueError(
                f"joint_limits 必须为 shape={(self.n, 2)}，当前为 {limits.shape}。"
            )
        if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("joint_limits 必须有限，且每个下界都严格小于上界。")
        operating_limits = limits.copy()
        operating_limits[:, 0] += self.joint_limit_margin
        operating_limits[:, 1] -= self.joint_limit_margin
        if np.any(operating_limits[:, 0] >= operating_limits[:, 1]):
            raise ValueError("joint_limit_margin 过大，导致正常运行关节盒为空。")
        self.safety_joint_limits = limits.copy()
        # joint_limits 保留为正常 MPC 运行盒，避免改变已有代价与诊断接口。
        self.joint_limits = operating_limits
        if hasattr(self, "_A_cons"):
            self._A_cons, self._l_template, self._u_template = self._build_constraints()

    def reset(self):
        """【非核心代码】清除跨控制周期的 warm-start 状态。"""
        self._last_u_plan = None
        self._last_solution = None
        self._last_diagnostics = self._empty_diagnostics()
        if self._solver is not None:
            self._solver.warm_start(
                x=np.zeros(self.num_variables, dtype=np.float64),
                y=np.zeros(self._A_cons.shape[0], dtype=np.float64),
            )

    # ------------------------------------------------------------------
    # 【核心代码】QP 代价
    # ------------------------------------------------------------------
    def _build_cost(self, step_terms):
        hessians = []
        linear = np.zeros(self.num_variables, dtype=np.float64)

        for k in range(self.horizon):
            terms = step_terms[k]
            E_acc = terms["C_acc"] @ self.Sv
            E_alpha = terms["C_alpha"] @ self.Sv

            Qxx = (
                E_acc.T @ self.Q_ee_acc @ E_acc
                + E_alpha.T @ self.Q_ee_alpha @ E_alpha
                + terms["G_g"].T @ self.Qg @ terms["G_g"]
                + self.Sq.T @ self.Qq @ self.Sq
                + self.Sv.T @ self.Qv @ self.Sv
            )
            Qxu = (
                E_acc.T @ self.Q_ee_acc @ terms["B_acc"]
                + E_alpha.T @ self.Q_ee_alpha @ terms["B_alpha"]
            )
            Quu = (
                terms["B_acc"].T @ self.Q_ee_acc @ terms["B_acc"]
                + terms["B_alpha"].T @ self.Q_ee_alpha @ terms["B_alpha"]
                + self.R
            )
            fx = (
                E_acc.T @ self.Q_ee_acc @ terms["D_acc"]
                + E_alpha.T @ self.Q_ee_alpha @ terms["D_alpha"]
                + terms["G_g"].T @ self.Qg @ terms["d_g"]
                - self.Sq.T @ self.Qq @ self.default_q
            )
            fu = (
                terms["B_acc"].T @ self.Q_ee_acc @ terms["D_acc"]
                + terms["B_alpha"].T @ self.Q_ee_alpha @ terms["D_alpha"]
            )

            local_hessian = 2.0 * np.block([[Qxx, Qxu], [Qxu.T, Quu]])
            local_hessian = 0.5 * (local_hessian + local_hessian.T)
            local_hessian += self.reg * np.eye(self.stage_dim, dtype=np.float64)
            local_linear = 2.0 * np.concatenate([fx, fu])

            hessians.append(local_hessian)
            start = self._cx(k)
            linear[start : start + self.stage_dim] = local_linear

        # 终端没有 u_N，也没有末端加速度项。
        terminal = step_terms[self.horizon]
        Qg_terminal = self.terminal_scale * self.Qg
        Qq_terminal = self.terminal_scale * self.Qq
        Qv_terminal = self.terminal_scale * self.Qv
        Qxx_terminal = (
            terminal["G_g"].T @ Qg_terminal @ terminal["G_g"]
            + self.Sq.T @ Qq_terminal @ self.Sq
            + self.Sv.T @ Qv_terminal @ self.Sv
        )
        fx_terminal = (
            terminal["G_g"].T @ Qg_terminal @ terminal["d_g"]
            - self.Sq.T @ Qq_terminal @ self.default_q
        )
        H_terminal = 2.0 * Qxx_terminal
        H_terminal = 0.5 * (H_terminal + H_terminal.T)
        H_terminal += self.reg * np.eye(self.nx, dtype=np.float64)
        h_terminal = 2.0 * fx_terminal

        hessians.append(H_terminal)
        linear[self._cx(self.horizon) : self._cx(self.horizon) + self.nx] = h_terminal
        return hessians, linear

    # ------------------------------------------------------------------
    # 【半核心代码】固定稀疏结构、求解和 warm-start
    # ------------------------------------------------------------------
    def _build_constraints(self):
        equality_rows = (self.horizon + 1) * self.nx
        box_rows = 3 * self.horizon * self.n
        total_rows = equality_rows + box_rows
        constraints = sparse.lil_matrix(
            (total_rows, self.num_variables), dtype=np.float64
        )
        lower = np.zeros(total_rows, dtype=np.float64)
        upper = np.zeros(total_rows, dtype=np.float64)

        # x_0 = x_measured；实际数值在每次 compute_action 中更新。
        constraints[: self.nx, self._cx(0) : self._cx(0) + self.nx] = np.eye(
            self.nx
        )

        # -A x_k - B u_k + x_{k+1} = 0。
        for k in range(self.horizon):
            row = (k + 1) * self.nx
            constraints[row : row + self.nx, self._cx(k) : self._cx(k) + self.nx] = -self.A
            constraints[row : row + self.nx, self._cu(k) : self._cu(k) + self.nu] = -self.B
            constraints[
                row : row + self.nx,
                self._cx(k + 1) : self._cx(k + 1) + self.nx,
            ] = np.eye(self.nx)

        row = equality_rows
        for k in range(1, self.horizon + 1):
            constraints[row : row + self.n, self._cx(k) : self._cx(k) + self.nx] = self.Sq
            lower[row : row + self.n] = self.joint_limits[:, 0]
            upper[row : row + self.n] = self.joint_limits[:, 1]
            row += self.n

        for k in range(1, self.horizon + 1):
            constraints[row : row + self.n, self._cx(k) : self._cx(k) + self.nx] = self.Sv
            lower[row : row + self.n] = -self.max_dq
            upper[row : row + self.n] = self.max_dq
            row += self.n

        for k in range(self.horizon):
            constraints[row : row + self.nu, self._cu(k) : self._cu(k) + self.nu] = np.eye(
                self.nu
            )
            lower[row : row + self.nu] = -self.max_ddq
            upper[row : row + self.nu] = self.max_ddq
            row += self.nu

        if row != total_rows:
            raise RuntimeError("MPC 约束行数内部组装错误。")
        return constraints.tocsc(), lower, upper

    def _build_online_constraint_bounds(self, q, dq):
        """【核心代码】构造正常运行盒或跨控制拍收回的恢复硬边界。

        正常状态严格使用内层运行盒。若当前状态或其制动距离已经在运行盒
        外，shoulder 对应方向临时开放到外层安全盒；每个真实控制拍重新
        计算，状态返回后自动恢复内层边界。这里没有 QP 松弛变量。
        """
        recovery_states, recovery_inputs = self._build_recovery_trajectory(q, dq)
        lower = self._l_template.copy()
        upper = self._u_template.copy()
        q_row_start = (self.horizon + 1) * self.nx
        recovery_active = False
        # 按 80% 可用 DDQ 计算向外速度的制动距离，给真正的最大制动保留
        # 20% 可行性余量；numerical_margin 约 0.006°，远小于 1° 运行裕量。
        braking_acceleration = 0.8 * self.max_ddq
        numerical_margin = 1e-4
        upper_stopping_envelope = (
            q
            + np.maximum(dq, 0.0) ** 2 / (2.0 * braking_acceleration)
            + numerical_margin
        )
        lower_stopping_envelope = (
            q
            - np.maximum(-dq, 0.0) ** 2 / (2.0 * braking_acceleration)
            - numerical_margin
        )
        recovery_lower = np.minimum(
            self.joint_limits[:, 0], lower_stopping_envelope
        )
        recovery_upper = np.maximum(
            self.joint_limits[:, 1], upper_stopping_envelope
        )
        lower_recovery_needed = (
            lower_stopping_envelope < self.joint_limits[:, 0]
        )
        upper_recovery_needed = (
            upper_stopping_envelope > self.joint_limits[:, 1]
        )
        has_operating_margin = self.joint_limit_margin > 0.0
        # 对设置了内层裕量的 shoulder，一旦需要恢复就开放到对应的外层
        # 安全边界，避免制动轨迹再次贴住一个几乎无内部空间的数值边界。
        recovery_lower[lower_recovery_needed & has_operating_margin] = (
            self.safety_joint_limits[lower_recovery_needed & has_operating_margin, 0]
        )
        recovery_upper[upper_recovery_needed & has_operating_margin] = (
            self.safety_joint_limits[upper_recovery_needed & has_operating_margin, 1]
        )

        currently_safe = (
            (q >= self.safety_joint_limits[:, 0])
            & (q <= self.safety_joint_limits[:, 1])
        )
        recovery_lower[currently_safe] = np.maximum(
            recovery_lower[currently_safe],
            self.safety_joint_limits[currently_safe, 0],
        )
        recovery_upper[currently_safe] = np.minimum(
            recovery_upper[currently_safe],
            self.safety_joint_limits[currently_safe, 1],
        )
        recovery_active = bool(
            np.any(recovery_lower < self.joint_limits[:, 0] - 1e-12)
            or np.any(recovery_upper > self.joint_limits[:, 1] + 1e-12)
        )

        for k in range(1, self.horizon + 1):
            row = q_row_start + (k - 1) * self.n
            lower[row : row + self.n] = recovery_lower
            upper[row : row + self.n] = recovery_upper

        return lower, upper, recovery_states, recovery_inputs, recovery_active

    def _build_recovery_trajectory(self, q, dq):
        """【半核心代码】生成满足 DDQ/DQ 上限的确定性关节恢复轨迹。"""
        states = np.empty((self.horizon + 1, self.nx), dtype=np.float64)
        inputs = np.empty((self.horizon, self.nu), dtype=np.float64)
        states[0] = np.concatenate([q, dq])
        target_q = np.clip(
            self.default_q,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1],
        )
        dt = self.control_dt

        for k in range(self.horizon):
            q_k = states[k, : self.n]
            dq_k = states[k, self.n :]
            desired = 24.0 * (target_q - q_k) - 6.0 * dq_k

            # 已经越界或下一拍按当前速度将越界时，优先使用最大向内加速度。
            q_zero_input = q_k + dt * dq_k
            upper_recovery = (q_k > self.joint_limits[:, 1]) | (
                q_zero_input > self.joint_limits[:, 1]
            )
            lower_recovery = (q_k < self.joint_limits[:, 0]) | (
                q_zero_input < self.joint_limits[:, 0]
            )
            desired[upper_recovery] = -self.max_ddq[upper_recovery]
            desired[lower_recovery] = self.max_ddq[lower_recovery]

            input_lower = np.maximum(
                -self.max_ddq, (-self.max_dq - dq_k) / dt
            )
            input_upper = np.minimum(
                self.max_ddq, (self.max_dq - dq_k) / dt
            )
            inputs[k] = np.clip(desired, input_lower, input_upper)
            states[k + 1] = self.A @ states[k] + self.B @ inputs[k]

        return states, inputs

    def _build_hessian_pattern(self):
        rows = []
        cols = []
        blocks = [
            (self._cx(k), self.stage_dim) for k in range(self.horizon)
        ] + [(self._cx(self.horizon), self.nx)]
        for start, size in blocks:
            for local_col in range(size):
                for local_row in range(local_col + 1):
                    rows.append(start + local_row)
                    cols.append(start + local_col)
        return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)

    @staticmethod
    def _pack_upper_triangles(hessians):
        values = []
        for hessian in hessians:
            for col in range(hessian.shape[1]):
                values.extend(hessian[: col + 1, col])
        return np.asarray(values, dtype=np.float64)

    def _solve_qp(self, P, p_values, linear, lower, upper, warm_start):
        try:
            if self._solver is None:
                settings = {
                    "verbose": self.solver_verbose,
                    "eps_abs": self.solver_eps_abs,
                    "eps_rel": self.solver_eps_rel,
                    "max_iter": self.solver_max_iter,
                    "rho": self.solver_rho,
                    "adaptive_rho": self.solver_adaptive_rho,
                    "scaled_termination": self.solver_scaled_termination,
                    "polishing": self.solver_polishing,
                    "warm_starting": True,
                }
                if self.solver_time_limit is not None:
                    settings["time_limit"] = self.solver_time_limit
                self._solver = osqp.OSQP()
                self._solver.setup(
                    P=P,
                    q=linear,
                    A=self._A_cons,
                    l=lower,
                    u=upper,
                    **settings,
                )
            else:
                self._solver.update(Px=p_values, q=linear, l=lower, u=upper)

            self._solver.warm_start(x=warm_start)
            return self._solver.solve(raise_error=False), None
        except Exception as exc:  # 求解器异常也必须转入安全回退。
            return None, exc

    def _check_result(self, result, lower, upper):
        if result is None:
            return False, None, "no_result", -1, np.inf

        status = str(result.info.status)
        status_val = int(result.info.status_val)
        solution = None if result.x is None else np.asarray(result.x, dtype=np.float64)
        accepted_status = status_val in (1, 2)
        # infeasible/max-iter 状态下 result.x 可能是证书或未收敛迭代量，
        # 不能把它当作可执行轨迹计算约束违反量。
        if not accepted_status:
            return False, None, status, status_val, np.inf
        finite_solution = (
            solution is not None
            and solution.shape == (self.num_variables,)
            and np.all(np.isfinite(solution))
        )
        if not finite_solution:
            return False, solution, status, status_val, np.inf

        constraint_value = self._A_cons @ solution
        max_violation = float(
            max(
                np.max(lower - constraint_value),
                np.max(constraint_value - upper),
                0.0,
            )
        )
        tolerance = max(1e-3, 2.0 * self.solver_eps_abs)
        solved = accepted_status and max_violation <= tolerance
        return solved, solution, status, status_val, max_violation

    def _shift_input_plan(self):
        if self._last_u_plan is None:
            return np.zeros((self.horizon, self.nu), dtype=np.float64)
        shifted = np.empty_like(self._last_u_plan)
        if self.horizon > 1:
            shifted[:-1] = self._last_u_plan[1:]
        shifted[-1] = self._last_u_plan[-1]
        return np.clip(shifted, -self.max_ddq, self.max_ddq)

    def _rollout(self, x0, inputs):
        states = np.empty((self.horizon + 1, self.nx), dtype=np.float64)
        states[0] = x0
        for k in range(self.horizon):
            states[k + 1] = self.A @ states[k] + self.B @ inputs[k]
        return states

    def _pack_trajectory(self, states, inputs):
        packed = np.empty(self.num_variables, dtype=np.float64)
        for k in range(self.horizon):
            packed[self._cx(k) : self._cx(k) + self.nx] = states[k]
            packed[self._cu(k) : self._cu(k) + self.nu] = inputs[k]
        packed[self._cx(self.horizon) : self._cx(self.horizon) + self.nx] = states[-1]
        return packed

    def _unpack_solution(self, solution):
        states = np.empty((self.horizon + 1, self.nx), dtype=np.float64)
        inputs = np.empty((self.horizon, self.nu), dtype=np.float64)
        for k in range(self.horizon):
            states[k] = solution[self._cx(k) : self._cx(k) + self.nx]
            inputs[k] = solution[self._cu(k) : self._cu(k) + self.nu]
        states[-1] = solution[
            self._cx(self.horizon) : self._cx(self.horizon) + self.nx
        ]
        return states, inputs

    # ------------------------------------------------------------------
    # 【半核心代码】失败回退、helper 校验和诊断
    # ------------------------------------------------------------------
    def _braking_fallback(self, q, dq):
        """QP 失败时温和制动，并缓慢把关节拉回运行盒中央。

        正常回退最多只使用 ``failure_max_ddq_scale`` 比例的 DDQ。只有
        下一控制拍已经会越过外层安全盒时，后面的硬安全裁剪才允许给出
        更强的向内加速度；这是安全兜底，不是常规恢复命令。
        """
        target_q = np.clip(
            self.default_q,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1],
        )
        fallback_limit = self.failure_max_ddq_scale * self.max_ddq
        desired = np.clip(
            self.failure_posture_gain * (target_q - q)
            - self.failure_braking_gain * dq,
            -fallback_limit,
            fallback_limit,
        )
        dt = self.control_dt
        lower = -self.max_ddq.copy()
        upper = self.max_ddq.copy()

        lower = np.maximum(lower, (-self.max_dq - dq) / dt)
        upper = np.minimum(upper, (self.max_dq - dq) / dt)
        lower = np.maximum(
            lower,
            2.0
            * (self.safety_joint_limits[:, 0] - q - dt * dq)
            / (dt * dt),
        )
        upper = np.minimum(
            upper,
            2.0
            * (self.safety_joint_limits[:, 1] - q - dt * dq)
            / (dt * dt),
        )

        feasible = bool(np.all(lower <= upper))
        if feasible:
            return np.clip(desired, lower, upper), True
        return desired, False

    def _constraint_margins(self, states, inputs):
        predicted_q = states[1:, : self.n]
        predicted_dq = states[1:, self.n :]
        q_margin = np.minimum(
            predicted_q - self.joint_limits[:, 0],
            self.joint_limits[:, 1] - predicted_q,
        )
        dq_margin = self.max_dq - np.abs(predicted_dq)
        ddq_margin = self.max_ddq - np.abs(inputs)
        return {
            "q": float(np.min(q_margin)),
            "dq": float(np.min(dq_margin)),
            "ddq": float(np.min(ddq_margin)),
        }

    def _resolve_terms_helper(self, helpers):
        if isinstance(helpers, dict):
            terms_fn = helpers.get("compute_mpc_terms")
            disturbance = helpers.get("disturbance")
            prediction = helpers.get("disturbance_prediction")
        else:
            terms_fn = None if helpers is None else getattr(helpers, "compute_mpc_terms", None)
            disturbance = None if helpers is None else getattr(helpers, "disturbance", None)
            prediction = (
                None
                if helpers is None
                else getattr(helpers, "disturbance_prediction", None)
            )
        if not callable(terms_fn):
            raise ValueError("ArmMPCPolicy 需要 helpers.compute_mpc_terms(...) 支持。")
        if prediction is None:
            prediction = (disturbance,) * (self.horizon + 1)
        else:
            prediction = tuple(prediction)
            if len(prediction) != self.horizon + 1:
                raise ValueError(
                    "disturbance_prediction 必须包含 horizon+1 个预测步，"
                    f"当前为 {len(prediction)}。"
                )
        return terms_fn, prediction

    @staticmethod
    def _disturbance_diagnostics(prediction):
        def vector(item, name):
            value = None if item is None else getattr(item, name, None)
            return (
                np.zeros(3, dtype=np.float64)
                if value is None
                else np.asarray(value, dtype=np.float64).copy()
            )

        def rotation(item):
            value = None if item is None else getattr(item, "rot_world_body", None)
            return (
                np.full((3, 3), np.nan, dtype=np.float64)
                if value is None
                else np.asarray(value, dtype=np.float64).copy()
            )

        return {
            "acc_world": np.stack([vector(item, "acc_world") for item in prediction]),
            "omega_world": np.stack(
                [vector(item, "omega_world") for item in prediction]
            ),
            "alpha_world": np.stack(
                [vector(item, "alpha_world") for item in prediction]
            ),
            "rot_world_body": np.stack([rotation(item) for item in prediction]),
        }

    def _validate_step_terms(self, terms, step):
        if not isinstance(terms, dict):
            raise ValueError(f"第 {step} 步 compute_mpc_terms 必须返回 dict。")
        expected_shapes = {
            "D_acc": (3,),
            "C_acc": (3, self.n),
            "B_acc": (3, self.nu),
            "D_alpha": (3,),
            "C_alpha": (3, self.n),
            "B_alpha": (3, self.nu),
            "G_g": (2, self.nx),
            "d_g": (2,),
            "gravity_error": (2,),
        }
        validated = {}
        for name, expected_shape in expected_shapes.items():
            if name not in terms:
                raise ValueError(f"第 {step} 步 compute_mpc_terms 缺少字段 {name}。")
            value = np.asarray(terms[name], dtype=np.float64)
            if value.shape != expected_shape:
                raise ValueError(
                    f"第 {step} 步 {name} 应为 shape={expected_shape}，"
                    f"当前为 {value.shape}。"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"第 {step} 步 {name} 包含 NaN 或 Inf。")
            validated[name] = value.copy()
        return validated

    def _build_one_step_diagnostics(
        self,
        q,
        dq,
        q_ref,
        dq_ref,
        ddq,
        acceleration_terms,
        end_state_terms,
    ):
        x1 = np.concatenate([q_ref, dq_ref])
        # 区间加速度使用 k=0 的模型；区间末端重力使用 k=1 的预测姿态。
        linear_acc = (
            acceleration_terms["D_acc"]
            + acceleration_terms["C_acc"] @ dq
            + acceleration_terms["B_acc"] @ ddq
        )
        angular_acc = (
            acceleration_terms["D_alpha"]
            + acceleration_terms["C_alpha"] @ dq
            + acceleration_terms["B_alpha"] @ ddq
        )
        gravity_error = (
            end_state_terms["G_g"] @ x1 + end_state_terms["d_g"]
        )
        posture_error = q_ref - self.default_q
        costs = {
            "linear_acceleration": float(
                linear_acc @ self.Q_ee_acc @ linear_acc
            ),
            "angular_acceleration": float(
                angular_acc @ self.Q_ee_alpha @ angular_acc
            ),
            "gravity": float(gravity_error @ self.Qg @ gravity_error),
            "posture": float(posture_error @ self.Qq @ posture_error),
            "velocity": float(dq_ref @ self.Qv @ dq_ref),
            "control": float(ddq @ self.R @ ddq),
        }
        return {
            "q": q_ref.copy(),
            "dq": dq_ref.copy(),
            "ee_lin_acc": linear_acc.copy(),
            "ee_ang_acc": angular_acc.copy(),
            "gravity_error": gravity_error.copy(),
            # 保存仿射模型，评估阶段可把 ddq_des 换成 ddq_real，
            # 从而区分“DDQ 没执行出来”和“任务模型本身不准”。
            "ee_lin_acc_offset": (
                acceleration_terms["D_acc"]
                + acceleration_terms["C_acc"] @ dq
            ),
            "ee_lin_acc_ddq_map": acceleration_terms["B_acc"].copy(),
            "ee_ang_acc_offset": (
                acceleration_terms["D_alpha"]
                + acceleration_terms["C_alpha"] @ dq
            ),
            "ee_ang_acc_ddq_map": acceleration_terms["B_alpha"].copy(),
            "cost_terms": costs,
        }

    def _empty_diagnostics(self):
        return {
            "solved": False,
            "success": False,
            "fallback_used": False,
            "fallback_feasible": False,
            "solver_status": "not_run",
            "solver_status_val": 0,
            "objective": np.nan,
            "iterations": 0,
            "primal_residual": np.nan,
            "dual_residual": np.nan,
            "prim_res": np.nan,
            "dual_res": np.nan,
            "max_constraint_violation": np.nan,
            "min_constraint_margins": {
                "q": np.nan,
                "dq": np.nan,
                "ddq": np.nan,
            },
            "assembly_time": 0.0,
            "solver_run_time": 0.0,
            "solver_setup_time": 0.0,
            "solver_update_time": 0.0,
            "solver_solve_time": 0.0,
            "solve_time": 0.0,
            "ddq_raw": np.zeros(self.nu, dtype=np.float64),
            "ddq_des": np.zeros(self.nu, dtype=np.float64),
            "working_states": np.zeros((self.horizon + 1, self.nx), dtype=np.float64),
            "working_inputs": np.zeros((self.horizon, self.nu), dtype=np.float64),
            "predicted_states": np.zeros(
                (self.horizon + 1, self.nx), dtype=np.float64
            ),
            "predicted_inputs": np.zeros((self.horizon, self.nu), dtype=np.float64),
            "gravity_error": np.zeros(2, dtype=np.float64),
            "disturbance_prediction": {
                "acc_world": np.zeros((self.horizon + 1, 3), dtype=np.float64),
                "omega_world": np.zeros(
                    (self.horizon + 1, 3), dtype=np.float64
                ),
                "alpha_world": np.zeros(
                    (self.horizon + 1, 3), dtype=np.float64
                ),
                "rot_world_body": np.full(
                    (self.horizon + 1, 3, 3), np.nan, dtype=np.float64
                ),
            },
            "current_q_violation": 0.0,
            "current_q_safety_violation": 0.0,
            "recovery_active": False,
            "one_step_prediction": {
                "q": np.zeros(self.n, dtype=np.float64),
                "dq": np.zeros(self.n, dtype=np.float64),
                "ee_lin_acc": np.zeros(3, dtype=np.float64),
                "ee_ang_acc": np.zeros(3, dtype=np.float64),
                "gravity_error": np.zeros(2, dtype=np.float64),
                "ee_lin_acc_offset": np.zeros(3, dtype=np.float64),
                "ee_lin_acc_ddq_map": np.zeros((3, self.nu), dtype=np.float64),
                "ee_ang_acc_offset": np.zeros(3, dtype=np.float64),
                "ee_ang_acc_ddq_map": np.zeros((3, self.nu), dtype=np.float64),
                "cost_terms": {name: 0.0 for name in self.COST_TERM_NAMES},
            },
        }

    def _cx(self, step):
        if step == self.horizon:
            return self.horizon * (self.nx + self.nu)
        return step * (self.nx + self.nu)

    def _cu(self, step):
        return self._cx(step) + self.nx

    @staticmethod
    def _make_weight(value, size, name, positive_definite=False):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            matrix = np.eye(size, dtype=np.float64) * float(array)
        elif array.shape == (size,):
            matrix = np.diag(array)
        elif array.shape == (size, size):
            matrix = array.copy()
        else:
            raise ValueError(
                f"{name} 必须是标量、长度 {size} 的向量或 {size}x{size} 矩阵。"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} 包含 NaN 或 Inf。")
        if not np.allclose(matrix, matrix.T, atol=1e-12):
            raise ValueError(f"{name} 必须是对称矩阵。")
        eigenvalues = np.linalg.eigvalsh(matrix)
        threshold = 1e-12 if positive_definite else -1e-12
        if positive_definite and np.min(eigenvalues) <= threshold:
            raise ValueError(f"{name} 必须正定。")
        if not positive_definite and np.min(eigenvalues) < threshold:
            raise ValueError(f"{name} 必须半正定。")
        return matrix

    @staticmethod
    def _make_positive_vector(value, size, name):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(size, float(array), dtype=np.float64)
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} 必须是正标量或长度 {size} 的正向量。")
        if np.any(array <= 0.0):
            raise ValueError(f"{name} 的所有分量都必须大于 0。")
        return array.copy()

    @staticmethod
    def _make_nonnegative_vector(value, size, name):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(size, float(array), dtype=np.float64)
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} 必须是非负标量或长度 {size} 的非负向量。")
        if np.any(array < 0.0):
            raise ValueError(f"{name} 的所有分量都必须大于等于 0。")
        return array.copy()

    @staticmethod
    def _obs_get(observation, key, default=None):
        if isinstance(observation, dict):
            return observation.get(key, default)
        return getattr(observation, key, default)

    @staticmethod
    def _info_value(info, name, default):
        if info is None:
            return default
        value = getattr(info, name, default)
        if isinstance(default, float):
            return float(value)
        return value

    @classmethod
    def _copy_nested(cls, value):
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: cls._copy_nested(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._copy_nested(item) for item in value]
        return value
