import numpy as np

class ArmMPCPolicy:
    """MPC 控制器接口骨架。统一输出 q_ref, dq_ref。"""

    def __init__(self, default_q, horizon, dt, Q_weights=None, R_weights=None):
        self.default_q = np.array(default_q, dtype=np.float32)
        self.horizon = horizon
        self.dt = dt
        self.Q = Q_weights
        self.R = R_weights
        self.solver = None

    def compute_action(self, arm_obs, helpers=None):
        """输入 arm_obs / helpers，输出 (q_ref, dq_ref)。"""
        q_ref = self.default_q.copy()
        dq_ref = np.zeros_like(q_ref)
        return q_ref, dq_ref
