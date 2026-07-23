import numpy as np


class ArmLQRPolicy:
    """有限时域时变 LQR。输出右臂 5 维 q_ref / dq_ref / ddq_des。"""

    COST_TERM_NAMES = (
        "linear_acceleration",
        "angular_acceleration",
        "position",
        "gravity",
        "posture",
        "velocity",
        "control",
    )

    def __init__(
        self,
        default_q,
        control_dt=0.02,
        horizon=12,
        q_acc=1.0,
        q_alpha=0.05,
        q_position=20.0,
        q_gravity=30.0,
        q_posture=0.4,
        q_vel=0.02,
        q_ee_velocity=0.0,
        r_ddq=0.25,
        terminal_scale=2.0,
        reg=1e-6,
        max_ddq=3.0,
        max_dq=1.0,
        ddq_rate_limit=350.0,
        ddq_smoothing_alpha=0.45,
        joint_limits=None,
        joint_limit_margin=0.25,
        joint_limit_stiffness=8.0,
        joint_limit_damping=2.0,
    ):
        # 右臂标称关节角 q_nom（5 维），用作姿态正则参考位形
        self.default_q = np.asarray(default_q, dtype=np.float64).copy()  # right arm only (5 DoF)
        self.n = self.default_q.shape[0]
        self.nx = 2 * self.n
        # 控制周期；主程序 main_sim.py 中传入手臂控制更新周期 arm_control_dt。
        self.control_dt = float(control_dt)
        self.horizon = int(horizon)
        self.q_acc = float(q_acc)
        self.q_alpha = float(q_alpha)
        self.q_position = float(q_position)
        self.q_gravity = float(q_gravity)
        q_posture = np.asarray(q_posture, dtype=np.float64)
        if q_posture.ndim == 0:
            q_posture = np.full(self.n, float(q_posture), dtype=np.float64)
        if q_posture.shape != (self.n,) or np.any(q_posture < 0.0):
            raise ValueError(f"q_posture 必须是非负标量或长度为 {self.n} 的向量。")
        self.q_posture = q_posture.copy()
        self.q_vel = float(q_vel)
        self.q_ee_velocity = float(q_ee_velocity)
        self.r_ddq = float(r_ddq)
        self.Qa = np.eye(3) * float(q_acc)
        self.Qalpha = np.eye(3) * float(q_alpha)
        self.Qp = np.eye(3) * float(q_position)
        self.Qg = np.eye(3) * float(q_gravity)
        # 分关节姿态代价：允许单独约束 shoulder pitch/roll，保留其余关节调姿自由度。
        self.Qq = np.diag(self.q_posture)
        self.Qv = np.eye(self.n) * float(q_vel)
        self.Qeev = np.eye(3) * float(q_ee_velocity)
        # 控制代价权重 R：惩罚关节加速度 u（ddq），u 越大代价越高，R 越大动作越平滑
        self.R = np.eye(self.n) * float(r_ddq)
        # 终端状态代价权重 QN：惩罚姿态偏差 q_nom - q_ref 和关节速度偏差 dq_nom - dq_ref
        self.QN = self._blk(float(terminal_scale) * self.Qq, float(terminal_scale) * self.Qv)
        self.terminal_scale = float(terminal_scale)
        self.reg = float(reg)
        self.max_ddq = float(max_ddq)
        self.max_dq = float(max_dq)
        self.ddq_rate_limit = None if ddq_rate_limit is None else float(ddq_rate_limit)
        self.ddq_smoothing_alpha = float(np.clip(ddq_smoothing_alpha, 0.0, 1.0))
        self.joint_limit_margin = float(joint_limit_margin)
        self.joint_limit_stiffness = float(joint_limit_stiffness)
        self.joint_limit_damping = float(joint_limit_damping)
        self.prev_u = None
        self.joint_limits = None
        self.set_joint_limits(joint_limits)
        self.last_u_raw = np.zeros(self.n, dtype=np.float64)
        self.last_u_command = np.zeros(self.n, dtype=np.float64)
        self.last_position = np.zeros(3, dtype=np.float64)
        self.last_position_reference = np.zeros(3, dtype=np.float64)
        self.last_position_error = np.zeros(3, dtype=np.float64)
        self.last_gravity_error = np.zeros(3, dtype=np.float64)
        self.last_one_step_prediction = {
            "q": np.zeros(self.n, dtype=np.float64),
            "dq": np.zeros(self.n, dtype=np.float64),
            "ee_lin_acc": np.zeros(3, dtype=np.float64),
            "ee_ang_acc": np.zeros(3, dtype=np.float64),
            "position_error": np.zeros(3, dtype=np.float64),
            "gravity_error": np.zeros(3, dtype=np.float64),
            "cost_terms": {name: 0.0 for name in self.COST_TERM_NAMES},
        }

    def compute_action(self, arm_obs, helpers=None):
        q = np.asarray(self._obs_get(arm_obs, "current_q"), dtype=np.float64)
        dq = np.asarray(self._obs_get(arm_obs, "current_dq"), dtype=np.float64)
        dt = float(self._obs_get(arm_obs, "dt", self.control_dt))
        x0 = np.concatenate([q, dq])
        A, B, S_q, S_v = self._discrete(dt)
        # 从 helpers 取局部线性化回调，返回每步的 C/B/D、重力项等
        terms_fn = None if helpers is None else getattr(helpers, "compute_lqr_terms", None)
        if not callable(terms_fn):
            raise ValueError("ArmLQRPolicy 需要 helpers.compute_lqr_terms(...) 支持。")
        step_terms = []
        q_bar = q.copy()
        dq_bar = dq.copy()
        torso_rotmat = self._obs_get(arm_obs, "torso_rotmat", None)
        disturbance = getattr(helpers, "disturbance", None)
        for _ in range(self.horizon):
            step_terms.append(terms_fn(q_bar, dq_bar, torso_rotmat, disturbance))
            q_bar = q_bar + dq_bar * dt
        # 每步包含加速度、torso-relative 位置和三维重力误差的局部仿射系数。
        P = self.QN.copy()
        p = np.zeros(self.nx, dtype=np.float64)
        K0 = None
        k0 = None
        # 倒序遍历预测步：k 从 horizon-1, horizon-2, ... 一直到 0。
        for k in range(self.horizon - 1, -1, -1):
            t = step_terms[k]
            # torso-relative 末端速度为 J_p(q) dq；它直接抑制位置持续单向漂移。
            G_pv = np.hstack([np.zeros((3, self.n), dtype=np.float64), t["G_p"][:, :self.n]])
            # 先整理当前第 k 步的单步代价 l_k 参数：关于状态 x 和控制 u 的二次项 / 一次项。
            Qxx = S_v.T @ t["C_acc"].T @ self.Qa @ t["C_acc"] @ S_v + S_v.T @ t["C_alpha"].T @ self.Qalpha @ t["C_alpha"] @ S_v + t["G_p"].T @ self.Qp @ t["G_p"] + G_pv.T @ self.Qeev @ G_pv + t["G_g"].T @ self.Qg @ t["G_g"] + S_q.T @ self.Qq @ S_q + S_v.T @ self.Qv @ S_v
            Qxu = S_v.T @ t["C_acc"].T @ self.Qa @ t["B_acc"] + S_v.T @ t["C_alpha"].T @ self.Qalpha @ t["B_alpha"]
            Quu = t["B_acc"].T @ self.Qa @ t["B_acc"] + t["B_alpha"].T @ self.Qalpha @ t["B_alpha"] + self.R
            fx = S_v.T @ t["C_acc"].T @ self.Qa @ t["D_acc"] + S_v.T @ t["C_alpha"].T @ self.Qalpha @ t["D_alpha"] + t["G_p"].T @ self.Qp @ t["d_p"] + t["G_g"].T @ self.Qg @ t["d_g"] - S_q.T @ self.Qq @ self.default_q
            fu = t["B_acc"].T @ self.Qa @ t["D_acc"] + t["B_alpha"].T @ self.Qalpha @ t["D_alpha"]
            # 这里进入第 k 步时，P/p 是值函数 V_{k+1} 的参数。
            # 把单步代价加上 V_{k+1}，得到当前 Q_k(x,u) 的参数。
            F = Qxx + A.T @ P @ A
            M = Qxu + A.T @ P @ B
            H_raw = Quu + B.T @ P @ B
            H = 0.5 * (H_raw + H_raw.T) + self.reg * np.eye(self.n) # 控制 Hessian：对称化消除浮点误差，再加 reg 正则保证可逆、数值稳定
            h = fx + A.T @ p
            g = fu + B.T @ p
            # 对 Q_k(x,u) 关于 u 求最优，得到反馈项 K 和前馈项 kk。
            K = np.linalg.solve(H, M.T)
            kk = np.linalg.solve(H, g)
            # 把最优 u 代回 Q_k，得到当前值函数 V_k 的二次项 P 和一次项 p。
            P = F - M @ K
            p = h - M @ kk
            K0, k0 = K, kk
        u_raw = -(K0 @ x0 + k0)  # 第一拍最优控制量 u=ddq。
        # 当前对比实验完全旁路 ddq 后处理，直接把 Riccati 输出送入逆动力学。
        u = u_raw.copy()
        dq_ref = np.clip(dq + u * dt, -self.max_dq, self.max_dq)  # 用 ddq 积分一步得到目标关节速度，并限速。
        q_ref = q + dq * dt + 0.5 * u * dt * dt  # 匀加速积分一步得到目标关节位置。
        if self.joint_limits is not None:
            q_ref = np.clip(q_ref, self.joint_limits[:, 0], self.joint_limits[:, 1])
        first_terms = step_terms[0]
        # 预测一个完整手臂控制周期后的状态，供离线与真实轨迹严格对齐比较。
        x1_model = A @ x0 + B @ u
        q1_model = x1_model[:self.n]
        dq1_model = x1_model[self.n:]
        ee_lin_acc_model = first_terms["C_acc"] @ dq + first_terms["B_acc"] @ u + first_terms["D_acc"]
        ee_ang_acc_model = first_terms["C_alpha"] @ dq + first_terms["B_alpha"] @ u + first_terms["D_alpha"]
        position_error_model = first_terms["G_p"] @ x1_model + first_terms["d_p"]
        gravity_error_model = first_terms["G_g"] @ x1_model + first_terms["d_g"]
        ee_position_velocity_model = first_terms["G_p"][:, :self.n] @ dq1_model
        posture_error_model = q1_model - self.default_q
        cost_terms = {
            "linear_acceleration": float(ee_lin_acc_model @ self.Qa @ ee_lin_acc_model),
            "angular_acceleration": float(ee_ang_acc_model @ self.Qalpha @ ee_ang_acc_model),
            "position": float(position_error_model @ self.Qp @ position_error_model),
            "gravity": float(gravity_error_model @ self.Qg @ gravity_error_model),
            "posture": float(posture_error_model @ self.Qq @ posture_error_model),
            "velocity": float(
                dq1_model @ self.Qv @ dq1_model
                + ee_position_velocity_model @ self.Qeev @ ee_position_velocity_model
            ),
            "control": float(u @ self.R @ u),
        }
        self.last_u_raw = u_raw.copy()
        self.last_u_command = u.copy()
        self.last_position = first_terms["position"].copy()
        self.last_position_reference = first_terms["position_reference"].copy()
        self.last_position_error = first_terms["position_error"].copy()
        self.last_gravity_error = first_terms["gravity_error"].copy()
        self.last_one_step_prediction = {
            "q": q1_model.copy(),
            "dq": dq1_model.copy(),
            "ee_lin_acc": ee_lin_acc_model.copy(),
            "ee_ang_acc": ee_ang_acc_model.copy(),
            "position_error": position_error_model.copy(),
            "gravity_error": gravity_error_model.copy(),
            "cost_terms": cost_terms,
        }
        return q_ref.astype(np.float32), dq_ref.astype(np.float32), u.astype(np.float32)

    def get_last_diagnostics(self):
        """返回最近一次 LQR 更新的关键中间量，供主程序记录。"""
        return {
            "ddq_raw": self.last_u_raw.copy(),
            "ddq_command": self.last_u_command.copy(),
            "position": self.last_position.copy(),
            "position_reference": self.last_position_reference.copy(),
            "position_error": self.last_position_error.copy(),
            "gravity_error": self.last_gravity_error.copy(),
            "one_step_prediction": {
                key: value.copy() if isinstance(value, np.ndarray) else dict(value)
                for key, value in self.last_one_step_prediction.items()
            },
        }

    def get_cost_definition(self):
        """返回实际轨迹代价重算所需的权重和姿态参考。"""
        return {
            "term_names": self.COST_TERM_NAMES,
            "Qa": self.Qa.copy(),
            "Qalpha": self.Qalpha.copy(),
            "Qp": self.Qp.copy(),
            "Qg": self.Qg.copy(),
            "Qq": self.Qq.copy(),
            "Qv": self.Qv.copy(),
            "Qeev": self.Qeev.copy(),
            "R": self.R.copy(),
            "posture_reference": self.default_q.copy(),
        }

    def set_joint_limits(self, joint_limits):
        if joint_limits is None:
            self.joint_limits = None
            return
        joint_limits = np.asarray(joint_limits, dtype=np.float64)
        if joint_limits.shape != (self.n, 2):
            raise ValueError(f"joint_limits shape {joint_limits.shape} 与右臂维度 {(self.n, 2)} 不一致。")
        self.joint_limits = joint_limits.copy()

    def _apply_ddq_safety(self, q, dq, u_raw):
        """保留供后续对比的可选安全层；当前 compute_action() 不调用。"""
        u = np.clip(u_raw, -self.max_ddq, self.max_ddq)
        u = self._apply_joint_limit_guard(q, dq, u)
        return np.clip(u, -self.max_ddq, self.max_ddq)

    def _post_process_ddq(self, q, dq, u_raw, dt):
        # 完整后处理暂时保留用于后续对比实验；当前 compute_action() 不调用它。
        # 第 1 层保护：先对 LQR 原始输出 u_raw 做硬限幅，避免 ddq_des 一上来就超过 self.max_ddq。
        u = np.clip(u_raw, -self.max_ddq, self.max_ddq)

        # 第 2 层保护：限制相邻两拍 ddq 的变化率，避免这一拍相比上一拍跳得过猛。
        # 只有从第二次控制开始（prev_u 不为 None）并且配置了 ddq_rate_limit 时才会生效。
        if self.prev_u is not None and self.ddq_rate_limit is not None:
            max_delta = max(0.0, self.ddq_rate_limit) * dt
            u = np.clip(u, self.prev_u - max_delta, self.prev_u + max_delta)

        # 第 3 层保护：用上一拍输出做一阶平滑，进一步减小抖动。
        # 同样只会在 prev_u 已存在时生效；alpha 越小，保留上一拍输出的比例越大。
        if self.prev_u is not None and self.ddq_smoothing_alpha < 1.0:
            alpha = self.ddq_smoothing_alpha
            u = alpha * u + (1.0 - alpha) * self.prev_u

        # 第 4 层保护：如果关节快靠近上下限，则对 ddq 加一个“往回拉”的安全修正。
        u = self._apply_joint_limit_guard(q, dq, u)

        # 第 5 层保护：关节限位修正后再做一次 ddq 硬限幅，防止安全修正本身把某一维推得过大。
        u = np.clip(u, -self.max_ddq, self.max_ddq)

        # 记录当前输出，供下一拍做 rate limit 和 smoothing。
        self.prev_u = u.copy()
        return u

    def _apply_joint_limit_guard(self, q, dq, u):
        if self.joint_limits is None or self.joint_limit_margin <= 0.0:
            return u
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        margin = self.joint_limit_margin
        u_safe = u.copy()

        lower_distance = q - lower
        upper_distance = upper - q
        lower_zone = lower_distance < margin
        upper_zone = upper_distance < margin

        lower_acc = self.joint_limit_stiffness * (margin - lower_distance) - self.joint_limit_damping * np.minimum(dq, 0.0)
        upper_acc = -self.joint_limit_stiffness * (margin - upper_distance) - self.joint_limit_damping * np.maximum(dq, 0.0)
        u_safe[lower_zone] = np.maximum(u_safe[lower_zone], lower_acc[lower_zone])
        u_safe[upper_zone] = np.minimum(u_safe[upper_zone], upper_acc[upper_zone])
        return u_safe

    @staticmethod
    def _blk(a, b):
        z1 = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64); z2 = np.zeros((b.shape[0], a.shape[1]), dtype=np.float64)
        return np.block([[a, z1], [z2, b]])

    def _discrete(self, dt):
        I = np.eye(self.n, dtype=np.float64)
        A = np.block([[I, dt * I], [np.zeros_like(I), I]])
        B = np.vstack([0.5 * dt * dt * I, dt * I])
        S_q = np.hstack([I, np.zeros_like(I)]); S_v = np.hstack([np.zeros_like(I), I])
        return A, B, S_q, S_v

    @staticmethod
    def _obs_get(arm_obs, key, default=None):
        return arm_obs.get(key, default) if isinstance(arm_obs, dict) else getattr(arm_obs, key, default)
