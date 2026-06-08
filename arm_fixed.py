import numpy as np

class ArmFixedPolicy:
    """固定姿态基线接口骨架。统一输出 q_ref, dq_ref。"""

    def __init__(self, target_q):
        self.target_q = np.array(target_q, dtype=np.float32)

    def compute_action(self, arm_obs, helpers=None):
        """输入 arm_obs / helpers，输出 (q_ref, dq_ref)。"""
        q_ref = self.target_q.copy()
        dq_ref = np.zeros_like(q_ref)
        return q_ref, dq_ref