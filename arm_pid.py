import numpy as np


class ArmPIDPolicy:
    """右臂持杯 PID 基线控制器。

    统一输出右臂 5 维 ``q_ref`` / ``dq_ref``，供下层 PD 跟踪。

    这里实现的是：
    1. 读取当前重力方向误差 e_g
    2. 对 e_g 做任务空间 PID
    3. 用中心差分构造 J_g = de_g / dq
    4. 用阻尼伪逆把任务空间修正映射到关节空间
    5. 叠加关节零位形正则项，防止手臂跑飞
    """

    def __init__(
        self,
        default_q,
        kp_pose=None,
        kd_pose=None,
        ki_pose=0.0,
        posture_gain=0.6,
        control_dt=0.01,
        finite_diff_eps=1e-4,
        damping=1e-3,
        integral_limit=0.20,
        max_dq=2.0,
        de_g_alpha=0.2,
    ):
        self.default_q = np.asarray(default_q, dtype=np.float64).copy()  # right arm only (5 DoF)
        self.n = self.default_q.shape[0]

        # 任务空间误差 e_g = [e_x, e_y]^T，对应 2 维倾斜误差
        self.kp_pose = self._make_diag(kp_pose, 2, default_value=4.0)
        self.kd_pose = self._make_diag(kd_pose, 2, default_value=0.4)
        self.ki_pose = self._make_diag(ki_pose, 2, default_value=0.0)

        # 关节空间姿态正则，让右臂不要跑离 q_nom = default_q 太远
        self.posture_gain = self._make_diag(posture_gain, self.n, default_value=0.6)

        self.control_dt = float(control_dt)
        self.finite_diff_eps = float(finite_diff_eps)
        self.damping = float(damping)
        self.integral_limit = float(integral_limit)
        self.max_dq = float(max_dq)
        self.de_g_alpha = float(de_g_alpha)

        self.integral_error = np.zeros(2, dtype=np.float64)
        self.prev_e_g = None
        self.filtered_de_g = np.zeros(2, dtype=np.float64)
        self.q_ref_state = None
        self._warned_missing_helper = False

    def compute_action(self, arm_obs, helpers=None):
        """输入 ``arm_obs`` / ``helpers``，输出 ``(q_ref, dq_ref)``。

        arm_obs 最少应提供：
        - current_q: 当前右臂 5 维关节角
        - current_dq: 当前右臂 5 维关节角速度
        - torso_quat 或 torso_rotmat: 当前躯干 / IMU 姿态
        - dt: 当前控制周期（可选，未提供则回退到 self.control_dt）

        helpers 里最关键的是一个可调用对象 ``compute_gravity_error(q, W_R_I)``：
        - 输入当前右臂关节角 q 和当前躯干姿态 W_R_I
        - 输出 e_g = [e_x, e_y]^T
        """
        q = np.asarray(self._obs_get(arm_obs, "current_q"), dtype=np.float64).copy()
        dq = np.asarray(self._obs_get(arm_obs, "current_dq"), dtype=np.float64).copy()
        dt = float(self._obs_get(arm_obs, "dt", self.control_dt))
        W_R_I = self._get_world_from_imu_rotmat(arm_obs)

        if self.q_ref_state is None or self.q_ref_state.shape != q.shape:
            # 第一次进入时，把内部 q_ref 初始化到当前测量值附近，避免突然跳变
            self.q_ref_state = q.copy()

        # ------------------------------------------------------------------
        # 1) 计算当前的重力方向误差 e_g
        #    e_g = P_xy(^E R_W g^W)
        #    这里不在 arm_pid.py 里直接做正运动学，而是通过 helper 回调计算，
        #    这样主逻辑清楚，也方便你后面替换成真正的 MuJoCo / Pinocchio 实现。
        # ------------------------------------------------------------------
        e_g = self._compute_gravity_error(q, W_R_I, helpers)

        # e_g 的时间导数：先差分，再做一阶低通滤波，避免 walking 中差分噪声直接放大到 Kd 项
        if self.prev_e_g is None or dt <= 1e-9:
            de_g_raw = np.zeros_like(e_g)
        else:
            de_g_raw = (e_g - self.prev_e_g) / dt
        alpha = float(np.clip(self.de_g_alpha, 0.0, 1.0))
        self.filtered_de_g = alpha * de_g_raw + (1.0 - alpha) * self.filtered_de_g
        de_g = self.filtered_de_g.copy()
        self.prev_e_g = e_g.copy()

        # 积分项需要限幅，避免 walking 场景中 windup
        self.integral_error = self.integral_error + e_g * dt
        self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)

        # ------------------------------------------------------------------
        # 2) 任务空间 PID
        #    u_g 不是角度，也不是末端角速度；它表示“希望二维误差往哪个方向减小”
        # ------------------------------------------------------------------
        u_g = (
            -self.kp_pose @ e_g
            - self.kd_pose @ de_g
            - self.ki_pose @ self.integral_error
        )

        # ------------------------------------------------------------------
        # 3) 在当前工作点，用中心差分构造 J_g = de_g / dq
        #    J_g(:, i) ≈ (e_g(q + eps e_i) - e_g(q - eps e_i)) / (2 eps)
        # ------------------------------------------------------------------
        J_g = self._compute_gravity_error_jacobian(q, W_R_I, helpers)

        # ------------------------------------------------------------------
        # 4) 用阻尼伪逆把任务空间修正映射到关节空间
        #    δq_task = J_g^† u_g
        #    dq_task = δq_task / dt
        # ------------------------------------------------------------------
        J_pinv = self._damped_pinv(J_g, self.damping)
        delta_q_task = J_pinv @ u_g
        dq_task = delta_q_task / max(dt, 1e-6)

        # ------------------------------------------------------------------
        # 5) 关节空间姿态正则：拉回到 q_nom = default_q
        #    dq_posture = K_posture (q_nom - q)
        # ------------------------------------------------------------------
        dq_posture = self.posture_gain @ (self.default_q - q)

        dq_ref = dq_task + dq_posture
        dq_ref = np.clip(dq_ref, -self.max_dq, self.max_dq)

        # 这里沿用 PID_DESIGN.md 的思路：积分关节速度参考得到位置参考
        self.q_ref_state = self.q_ref_state + dq_ref * dt
        q_ref = self.q_ref_state.copy()

        return q_ref.astype(np.float32), dq_ref.astype(np.float32)

    def _compute_gravity_error(self, q, W_R_I, helpers):
        """通过 helper 计算 e_g。

        约定 helper 提供：
        - compute_gravity_error(q, W_R_I)
        或
        - compute_eg(q, W_R_I)

        如果当前还没有接好 kinematics helper，这里先回退成 0，
        这样至少不会让程序直接炸掉，但会退化成“只有姿态正则项”的 baseline。
        """
        fn = None
        if helpers is not None:
            if isinstance(helpers, dict):
                fn = helpers.get("compute_gravity_error", helpers.get("compute_eg", None))
            else:
                fn = getattr(helpers, "compute_gravity_error", None)
                if fn is None:
                    fn = getattr(helpers, "compute_eg", None)

        if callable(fn):
            e_g = np.asarray(fn(q, W_R_I), dtype=np.float64).reshape(-1)
            if e_g.shape[0] != 2:
                raise ValueError(f"compute_gravity_error 必须返回 shape=(2,) 的误差，当前得到 {e_g.shape}")
            return e_g

        if not self._warned_missing_helper:
            print("[ArmPIDPolicy] 未提供 compute_gravity_error helper，当前退化为仅使用关节姿态正则项。")
            self._warned_missing_helper = True
        return np.zeros(2, dtype=np.float64)

    def _compute_gravity_error_jacobian(self, q_star, W_R_I_star, helpers):
        """用中心差分计算 J_g。

        第 i 列表示：保持当前躯干姿态 W_R_I 不变，仅扰动第 i 个关节时，
        e_g = [e_x, e_y]^T 会如何变化。
        """
        J_g = np.zeros((2, self.n), dtype=np.float64)
        eps = self.finite_diff_eps

        for i in range(self.n):
            q_plus = q_star.copy()
            q_minus = q_star.copy()
            q_plus[i] += eps
            q_minus[i] -= eps

            e_plus = self._compute_gravity_error(q_plus, W_R_I_star, helpers)
            e_minus = self._compute_gravity_error(q_minus, W_R_I_star, helpers)

            # 中心差分比单边差分更稳，也更符合 PID_DESIGN.md 的说明
            J_g[:, i] = (e_plus - e_minus) / (2.0 * eps)

        return J_g

    @staticmethod
    def _damped_pinv(J, damping):
        """阻尼伪逆：J^† = J^T (J J^T + λ^2 I)^(-1)"""
        J = np.asarray(J, dtype=np.float64)
        if J.ndim != 2:
            raise ValueError("J 必须是二维矩阵。")
        m = J.shape[0]
        reg = (damping ** 2) * np.eye(m, dtype=np.float64)
        return J.T @ np.linalg.inv(J @ J.T + reg)

    @staticmethod
    def _make_diag(value, size, default_value):
        if value is None:
            return np.eye(size, dtype=np.float64) * default_value
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.eye(size, dtype=np.float64) * float(arr)
        if arr.ndim == 1 and arr.shape[0] == size:
            return np.diag(arr)
        if arr.shape == (size, size):
            return arr.astype(np.float64)
        raise ValueError(f"无法把输入 {value} 转成 {size}x{size} 增益矩阵。")

    @staticmethod
    def _obs_get(arm_obs, key, default=None):
        if isinstance(arm_obs, dict):
            return arm_obs.get(key, default)
        return getattr(arm_obs, key, default)

    def _get_world_from_imu_rotmat(self, arm_obs):
        torso_rotmat = self._obs_get(arm_obs, "torso_rotmat", None)
        if torso_rotmat is not None:
            R = np.asarray(torso_rotmat, dtype=np.float64)
            if R.shape != (3, 3):
                raise ValueError(f"torso_rotmat 需要 shape=(3,3)，当前得到 {R.shape}")
            return R

        torso_quat = self._obs_get(arm_obs, "torso_quat", None)
        if torso_quat is None:
            raise ValueError("arm_obs 必须提供 torso_rotmat 或 torso_quat。")
        return self._quat_wxyz_to_rotmat(torso_quat)

    @staticmethod
    def _quat_wxyz_to_rotmat(q_wxyz):
        q = np.asarray(q_wxyz, dtype=np.float64).reshape(-1)
        if q.shape[0] != 4:
            raise ValueError(f"torso_quat 需要 4 维 wxyz 四元数，当前得到 {q.shape}")
        n = np.linalg.norm(q)
        if n < 1e-12:
            raise ValueError("收到零范数四元数，无法构造旋转矩阵。")
        w, x, y, z = q / n
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float64)