from dataclasses import dataclass
from typing import Any, Callable, Optional

import mujoco
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
    compute_gravity_error: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None


class KinematicsHelper:
    """为上层控制器提供最基本的运动学辅助接口。"""

    def __init__(self, model: Any, ee_site_name: str, joint_indices: np.ndarray, imu_site_name: str = "imu_in_torso"):
        self.model = model
        self.ee_site_name = ee_site_name
        self.imu_site_name = imu_site_name
        self.joint_indices = np.array(joint_indices, dtype=np.int32)
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        self.imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, imu_site_name)
        self._scratch = mujoco.MjData(model)

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
        return ControlObservation(
            current_q=np.asarray(current_q),
            current_dq=np.asarray(current_dq),
            torso_quat=np.asarray(torso_quat),
            torso_omega=np.asarray(torso_omega),
            torso_acc=None if torso_acc is None else np.asarray(torso_acc),
            torso_alpha=None if torso_alpha is None else np.asarray(torso_alpha),
            torso_rotmat=None if torso_rotmat is None else np.asarray(torso_rotmat),
            phase=phase,
            dt=dt,
        )

    def compute_kinematics_cache(self, data: Any) -> KinematicsCache:
        return KinematicsCache()

    def build_disturbance_input(
        self,
        acc_world: Optional[np.ndarray] = None,
        omega_world: Optional[np.ndarray] = None,
        alpha_world: Optional[np.ndarray] = None,
        rot_world_body: Optional[np.ndarray] = None,
    ) -> DisturbanceInput:
        return DisturbanceInput(acc_world, omega_world, alpha_world, rot_world_body)

    def build_helpers(
        self,
        data: Any,
        disturbance: Optional[DisturbanceInput] = None,
    ) -> ControllerHelpers:
        qpos_ref = np.asarray(data.qpos, dtype=np.float64).copy()
        return ControllerHelpers(
            model=self.model,
            data=data,
            disturbance=disturbance,
            kinematics=self.compute_kinematics_cache(data),
            compute_gravity_error=lambda q, W_R_I: self.compute_gravity_error(q, W_R_I, qpos_ref),
        )

    def compute_gravity_error(self, q_right_arm: np.ndarray, W_R_I: np.ndarray, qpos_reference: np.ndarray) -> np.ndarray:
        # 1) 冻结当前整机姿态：把当前仿真 qpos 拷贝到 scratch data 中
        self._scratch.qpos[:] = qpos_reference
        # 2) 仅替换右臂 5 个关节，模拟“在当前躯干姿态不变条件下扰动右臂”
        self._scratch.qpos[self.joint_indices] = np.asarray(q_right_arm, dtype=np.float64)
        mujoco.mj_forward(self.model, self._scratch)

        # 3) 从正运动学结果读取末端姿态：site_xmat 给的是 ^W R_E
        W_R_E = self._scratch.site_xmat[self.ee_site_id].reshape(3, 3).copy()

        # 4) 任务空间重力误差定义：e_g = P_xy(^E R_W g^W)
        #    这里 ^E R_W = (^W R_E)^T，g^W = [0, 0, -9.81]^T
        g_W = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        E_R_W = W_R_E.T
        g_E = E_R_W @ g_W
        return g_E[:2].copy()