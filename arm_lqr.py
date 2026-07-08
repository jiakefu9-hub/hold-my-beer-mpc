import numpy as np


class ArmLQRPolicy:
    """有限时域时变 LQR。统一输出右臂 5 维 q_ref / dq_ref。"""

    def __init__(self, default_q, control_dt=0.02, horizon=12, q_acc=1.0, q_alpha=0.05, q_gravity=30.0, q_posture=0.4, q_vel=0.02, r_ddq=1e-2, terminal_scale=2.0, reg=1e-6, max_ddq=8.0, max_dq=2.0):
        # 右臂标称关节角 q_nom（5 维），用作姿态正则参考位形
        self.default_q = np.asarray(default_q, dtype=np.float64).copy()  # right arm only (5 DoF)
        self.n = self.default_q.shape[0]
        self.nx = 2 * self.n
        # 控制周期；主程序 main_sim.py 中传入手臂控制更新周期 arm_control_dt。
        self.control_dt = float(control_dt)
        self.horizon = int(horizon)
        self.Qa = np.eye(3) * float(q_acc)
        self.Qalpha = np.eye(3) * float(q_alpha)
        self.Qg = np.eye(2) * float(q_gravity)
        self.Qq = np.eye(self.n) * float(q_posture)
        self.Qv = np.eye(self.n) * float(q_vel)
        # 控制代价权重 R：惩罚关节加速度 u（ddq），u 越大代价越高，R 越大动作越平滑
        self.R = np.eye(self.n) * float(r_ddq)
        # 终端状态代价权重 QN：惩罚姿态偏差 q_nom - q_ref 和关节速度偏差 dq_nom - dq_ref
        self.QN = self._blk(float(terminal_scale) * self.Qq, float(terminal_scale) * self.Qv)
        self.reg = float(reg)
        self.max_ddq = float(max_ddq)
        self.max_dq = float(max_dq)

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
        # step_terms 最后是长度 horizon 的 list；每个元素是 dict，含 D_acc(3,), C_acc(3,5), B_acc(3,5), G_g(2,10) 等。
        P = self.QN.copy()
        p = np.zeros(self.nx, dtype=np.float64)
        K0 = None
        k0 = None
        # 倒序遍历预测步：k 从 horizon-1, horizon-2, ... 一直到 0。
        for k in range(self.horizon - 1, -1, -1):
            t = step_terms[k]
            # 先整理当前第 k 步的单步代价 l_k 参数：关于状态 x 和控制 u 的二次项 / 一次项。
            Qxx = S_v.T @ t["C_acc"].T @ self.Qa @ t["C_acc"] @ S_v + S_v.T @ t["C_alpha"].T @ self.Qalpha @ t["C_alpha"] @ S_v + t["G_g"].T @ self.Qg @ t["G_g"] + S_q.T @ self.Qq @ S_q + S_v.T @ self.Qv @ S_v
            Qxu = S_v.T @ t["C_acc"].T @ self.Qa @ t["B_acc"] + S_v.T @ t["C_alpha"].T @ self.Qalpha @ t["B_alpha"]
            Quu = t["B_acc"].T @ self.Qa @ t["B_acc"] + t["B_alpha"].T @ self.Qalpha @ t["B_alpha"] + self.R
            fx = S_v.T @ t["C_acc"].T @ self.Qa @ t["D_acc"] + S_v.T @ t["C_alpha"].T @ self.Qalpha @ t["D_alpha"] + t["G_g"].T @ self.Qg @ t["d_g"] - S_q.T @ self.Qq @ self.default_q
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
        u = np.clip(-(K0 @ x0 + k0), -self.max_ddq, self.max_ddq)  # 第一拍最优控制量 u=ddq，并限制最大关节加速度。
        dq_ref = np.clip(dq + u * dt, -self.max_dq, self.max_dq)  # 用 ddq 积分一步得到目标关节速度，并限速。
        q_ref = q + dq * dt + 0.5 * u * dt * dt  # 匀加速积分一步得到目标关节位置。
        return q_ref.astype(np.float32), dq_ref.astype(np.float32)

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
