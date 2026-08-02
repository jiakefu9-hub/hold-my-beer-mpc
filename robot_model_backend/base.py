"""MPC 预测运动学后端的最小统一接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictionKinematics:
    """一个 MPC 预测节点需要的几何量。

    Jacobian 和 Jacobian 导数都用世界系表达，且只保留
    ``joint_names`` 指定的右臂关节列。
    """

    ee_position_world: np.ndarray
    ee_rotation_world: np.ndarray
    imu_position_world: np.ndarray
    imu_rotation_world: np.ndarray
    J_v_world: np.ndarray
    J_w_world: np.ndarray
    dJ_v_world: np.ndarray
    dJ_w_world: np.ndarray


class PredictionKinematicsBackend(ABC):
    """MuJoCo/Pinocchio 预测运动学的共同协议。"""

    backend_name = "abstract"

    @abstractmethod
    def evaluate(
        self,
        qpos_reference: np.ndarray,
        q_arm: np.ndarray,
        dq_arm: np.ndarray,
        *,
        acceleration_required: bool = True,
    ) -> PredictionKinematics:
        """在冻结整机其他自由度时，计算右臂预测节点运动学。"""
