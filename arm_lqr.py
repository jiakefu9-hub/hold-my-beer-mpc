import numpy as np

class ArmLQRPolicy:
    """LQR 控制器接口骨架。统一输出 q_ref, dq_ref。"""

    def __init__(self, default_q, K_matrix=None, dt=0.002):
        self.default_q = np.array(default_q, dtype=np.float32)
        self.K = None if K_matrix is None else np.array(K_matrix, dtype=np.float32)
        self.dt = dt

    def compute_action(self, arm_obs, helpers=None):
        """输入 arm_obs / helpers，输出 (q_ref, dq_ref)。"""
        q_ref = self.default_q.copy()
        dq_ref = np.zeros_like(q_ref)
        return q_ref, dq_ref