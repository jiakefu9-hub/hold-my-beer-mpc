from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class ControlObservation:
    current_q: np.ndarray
    current_dq: np.ndarray
    torso_quat: np.ndarray
    torso_omega: np.ndarray
    torso_acc: Optional[np.ndarray] = None
    torso_alpha: Optional[np.ndarray] = None
    torso_rotmat: Optional[np.ndarray] = None
    phase: Optional[float] = None
    dt: Optional[float] = None


@dataclass
class DisturbanceInput:
    acc_world: Optional[np.ndarray] = None
    omega_world: Optional[np.ndarray] = None
    alpha_world: Optional[np.ndarray] = None
    rot_world_body: Optional[np.ndarray] = None


@dataclass
class KinematicsCache:
    J_v: Optional[np.ndarray] = None
    dJ_v: Optional[np.ndarray] = None
    J_w: Optional[np.ndarray] = None
    dJ_w: Optional[np.ndarray] = None


@dataclass
class ControllerHelpers:
    model: Any = None
    data: Any = None
    disturbance: Optional[DisturbanceInput] = None
    kinematics: Optional[KinematicsCache] = None


class KinematicsHelper:
    """只定义接口，不实现具体运动学/动力学计算。"""

    def __init__(self, model: Any, ee_body_name: str, joint_indices: np.ndarray):
        self.model = model
        self.ee_body_name = ee_body_name
        self.joint_indices = np.array(joint_indices, dtype=np.int32)

    def build_observation(
        self,
        current_q: np.ndarray,
        current_dq: np.ndarray,
        torso_quat: np.ndarray,
        torso_omega: np.ndarray,
        torso_acc: Optional[np.ndarray] = None,
        torso_alpha: Optional[np.ndarray] = None,
        torso_rotmat: Optional[np.ndarray] = None,
        phase: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> ControlObservation:
        raise NotImplementedError

    def compute_kinematics_cache(self, data: Any) -> KinematicsCache:
        raise NotImplementedError

    def build_disturbance_input(
        self,
        acc_world: Optional[np.ndarray] = None,
        omega_world: Optional[np.ndarray] = None,
        alpha_world: Optional[np.ndarray] = None,
        rot_world_body: Optional[np.ndarray] = None,
    ) -> DisturbanceInput:
        raise NotImplementedError

    def build_helpers(
        self,
        data: Any,
        disturbance: Optional[DisturbanceInput] = None,
    ) -> ControllerHelpers:
        raise NotImplementedError