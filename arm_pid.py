import numpy as np

class ArmPIDPolicy:
    """PID 控制器接口骨架。统一输出 q_ref, dq_ref。"""

    def __init__(self, default_q, kp_pose=None, kd_pose=None, ki_pose=0.0):
        self.default_q = np.array(default_q, dtype=np.float32)
        self.kp_pose = kp_pose
        self.kd_pose = kd_pose
        self.ki_pose = ki_pose

    def compute_action(self, arm_obs, helpers=None):
        """输入 arm_obs / helpers，输出 (q_ref, dq_ref)。"""
        q_ref = self.default_q.copy()
        dq_ref = np.zeros_like(q_ref)
        return q_ref, dq_ref