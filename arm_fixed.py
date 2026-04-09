import numpy as np

class ArmFixedPolicy:
    """
    固定姿态策略 (Baseline)。
    不计算任何反馈力矩，仅仅输出固定的目标关节角度 (target_q)，
    用于将腰部和双臂固定在指定的位置。
    """
    def __init__(self, target_q):
        """
        初始化策略
        :param target_q: 期望固定的关节角度 (长度 11)
        """
        self.target_q = np.array(target_q, dtype=np.float32)
        
    def compute_action(self, torso_quat, torso_omega, current_q, current_dq):
        """
        计算控制动作 (符合 Sim2Real 标准)
        :param torso_quat: 躯干 (torso_link) 的四元数 [w, x, y, z]
        :param torso_omega: 躯干 (torso_link) 的角速度 [wx, wy, wz]
        :param current_q: 当前手臂/腰部关节角度 (长度 11)
        :param current_dq: 当前手臂/腰部关节角速度 (长度 11)
        :return: 期望关节位置 target_q (长度 11)
        """
        # 因为是固定姿态策略，无视传感器输入，永远返回设定好的目标角度
        return self.target_q