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
    # 该姿态由当前实测值锚定模板相对姿态得到。
    rot_world_body: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DisturbanceHorizon:
    """MPC 的节点扰动与控制区间扰动，两者时间语义不可混用。"""

    nodes: tuple
    intervals: tuple


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
    disturbance_prediction: Optional[tuple] = None
    interval_disturbance_prediction: Optional[tuple] = None
    kinematics: Optional[KinematicsCache] = None
    torso_relative_position_reference: Optional[np.ndarray] = None
    compute_gravity_error: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    compute_lqr_terms: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, Optional[DisturbanceInput]], dict]] = None
    compute_mpc_terms: Optional[
        Callable[
            [
                np.ndarray,
                np.ndarray,
                Optional[DisturbanceInput],
                Optional[DisturbanceInput],
                bool,
            ],
            dict,
        ]
    ] = None


class KinematicsHelper:
    """为上层控制器提供最基本的运动学辅助接口。"""

    def __init__(
        self,
        model: Any,
        ee_site_name: str,
        joint_indices: np.ndarray,
        imu_site_name: str = "imu_in_torso",
        position_reference_q: Optional[np.ndarray] = None,
        prediction_backend: Optional[Any] = None,
    ):
        self.model = model
        self.ee_site_name = ee_site_name
        self.imu_site_name = imu_site_name
        self.joint_indices = np.array(joint_indices, dtype=np.int32)
        self.qvel_indices = self.joint_indices - 1
        # 【核心代码】MPC 预测运动学可以由 MuJoCo 或 Pinocchio 提供；
        # LQR/PID 的既有 MuJoCo 路径保持不变。
        self.prediction_backend = prediction_backend
        self.position_reference_q = None if position_reference_q is None else np.asarray(position_reference_q, dtype=np.float64).copy()
        if self.position_reference_q is not None and self.position_reference_q.shape != self.joint_indices.shape:
            raise ValueError("position_reference_q 必须与被控关节数量一致。")
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        self.imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, imu_site_name)
        # 初始化 MuJoCo 数据结构 data，用于临时计算雅可比矩阵
        self._scratch = mujoco.MjData(model)
        # 预测窗口内会重复计算同一 site 的 Jacobian；复用工作数组，避免
        # 每个节点重新分配四个 3 x nv 数组。
        self._jacp_workspace = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jacr_workspace = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jacp_dot_workspace = np.zeros(
            (3, self.model.nv), dtype=np.float64
        )
        self._jacr_dot_workspace = np.zeros(
            (3, self.model.nv), dtype=np.float64
        )
        # 当前模型中 IMU 与右肩固连在同一运动链段，名义右臂角确定后该相对位置就是常量。
        self.torso_relative_position_reference = (
            None
            if self.position_reference_q is None
            else self.compute_torso_relative_position(self.position_reference_q)
        )

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
        qpos_reference = np.asarray(data.qpos, dtype=np.float64).copy()
        q_right_arm = qpos_reference[self.joint_indices].copy()
        dq_right_arm = np.asarray(data.qvel, dtype=np.float64)[self.qvel_indices].copy()
        self._set_scratch_state(qpos_reference, q_right_arm, dq_right_arm)
        J_v, J_w = self._site_jacobians_world()
        dJ_v, dJ_w = self._site_jacobian_dots_world()
        return KinematicsCache(J_v=J_v, dJ_v=dJ_v, J_w=J_w, dJ_w=dJ_w)

    def build_disturbance_input(
        self,
        acc_world: Optional[np.ndarray] = None,
        omega_world: Optional[np.ndarray] = None,
        alpha_world: Optional[np.ndarray] = None,
        rot_world_body: Optional[np.ndarray] = None,
    ) -> DisturbanceInput:
        return DisturbanceInput(
            acc_world, omega_world, alpha_world, rot_world_body
        )

    def build_helpers(
        self,
        data: Any,
        disturbance: Optional[DisturbanceInput] = None,
        disturbance_prediction: Optional[tuple] = None,
        interval_disturbance_prediction: Optional[tuple] = None,
        include_kinematics_cache: bool = True,
    ) -> ControllerHelpers:
        qpos_ref = np.asarray(data.qpos, dtype=np.float64).copy()
        position_reference = (
            self.compute_torso_relative_position(qpos_ref[self.joint_indices])
            if self.torso_relative_position_reference is None
            else self.torso_relative_position_reference.copy()
        )
        # 【半核心代码】统一打包各控制器所需的运动学回调。
        # PID/MPC 使用二维重力误差；现有 LQR 仍通过 compute_lqr_terms 使用三维误差，
        # 因此不能直接把底层三维函数改掉。
        return ControllerHelpers(
            model=self.model,  # right_arm_helper 初始化时的 MuJoCo 模型 m
            data=data,  # 当前仿真步的机器人状态 d（qpos/qvel 等）
            disturbance=disturbance,  # main_sim 中由世界系躯干 acc/omega/alpha/R 构建的完整扰动
            disturbance_prediction=disturbance_prediction,
            interval_disturbance_prediction=(
                interval_disturbance_prediction
            ),
            kinematics=(
                self.compute_kinematics_cache(data)
                if include_kinematics_cache
                else None
            ),
            torso_relative_position_reference=position_reference,
            compute_gravity_error=lambda q, W_R_I: self.compute_tilt_error(q, W_R_I, qpos_ref),
            compute_lqr_terms=lambda q, dq, W_R_I, dist=None: self.compute_lqr_terms(
                q, dq, W_R_I, qpos_ref, dist, position_reference
            ),
            compute_mpc_terms=lambda q, dq, node_dist=None, interval_dist=None, acceleration_required=True: self.compute_mpc_terms(
                q,
                dq,
                qpos_ref,
                node_dist,
                interval_dist,
                acceleration_required,
            ),
        )

    def compute_gravity_error(self, q_right_arm: np.ndarray, _W_R_I: np.ndarray, qpos_reference: np.ndarray) -> np.ndarray:
        """现有 LQR 使用的三维有方向重力误差。"""
        # q_right_arm: shape=(5,), right arm qpos[25:30]; qpos_reference: shape=(model.nq,), full robot qpos.
        self._set_scratch_state(qpos_reference, q_right_arm, np.zeros(len(self.qvel_indices), dtype=np.float64))
        W_R_E = self._scratch.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        gravity_world = np.asarray(self.model.opt.gravity, dtype=np.float64)
        gravity_reference_end = np.array([0.0, 0.0, -np.linalg.norm(gravity_world)], dtype=np.float64)
        # 三维有方向误差：倒立时 z 误差约为 2g，不再与正立同为零。
        return W_R_E.T @ gravity_world - gravity_reference_end

    def compute_tilt_error(self, q_right_arm: np.ndarray, _W_R_I: np.ndarray, qpos_reference: np.ndarray) -> np.ndarray:
        """PID/MPC 使用的二维有符号倾斜误差。"""
        self._set_scratch_state(
            qpos_reference,
            q_right_arm,
            np.zeros(len(self.qvel_indices), dtype=np.float64),
        )
        W_R_E = self._scratch.site_xmat[self.ee_site_id].reshape(3, 3)
        gravity_world = np.asarray(self.model.opt.gravity, dtype=np.float64)
        return (W_R_E.T @ gravity_world)[:2].copy()

    def compute_gravity_error_jacobian(self, q_right_arm: np.ndarray, W_R_I: np.ndarray, qpos_reference: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        error_dim = self.compute_gravity_error(q_right_arm, W_R_I, qpos_reference).shape[0]
        J_g = np.zeros((error_dim, len(q_right_arm)), dtype=np.float64)
        for i in range(len(q_right_arm)):
            q_plus = np.asarray(q_right_arm, dtype=np.float64).copy(); q_plus[i] += eps
            q_minus = np.asarray(q_right_arm, dtype=np.float64).copy(); q_minus[i] -= eps
            e_plus = self.compute_gravity_error(q_plus, W_R_I, qpos_reference)
            e_minus = self.compute_gravity_error(q_minus, W_R_I, qpos_reference)
            J_g[:, i] = (e_plus - e_minus) / (2.0 * eps)
        return J_g

    def compute_torso_relative_position(self, q_right_arm: np.ndarray) -> np.ndarray:
        """仅由右臂关节角计算抓持点在 torso IMU 坐标系中的位置。"""
        self._set_scratch_state(self.model.qpos0, q_right_arm, np.zeros(len(self.qvel_indices), dtype=np.float64))
        W_R_B = self._scratch.site_xmat[self.imu_site_id].reshape(3, 3).copy()
        p_E = self._scratch.site_xpos[self.ee_site_id].copy()
        p_B = self._scratch.site_xpos[self.imu_site_id].copy()
        return W_R_B.T @ (p_E - p_B)

    def _set_scratch_state(self, qpos_reference: np.ndarray, q_right_arm: np.ndarray, dq_right_arm: np.ndarray):
        self._scratch.qpos[:] = np.asarray(qpos_reference, dtype=np.float64)
        self._scratch.qvel[:] = 0.0
        self._scratch.qpos[self.joint_indices] = np.asarray(q_right_arm, dtype=np.float64)
        self._scratch.qvel[self.qvel_indices] = np.asarray(dq_right_arm, dtype=np.float64)
        # 【核心代码】这里只读取位姿、Jacobian 和 Jacobian 导数，不需要
        # 执行动力学、接触求解或传感器阶段。位置+速度前向阶段与完整
        # mj_forward 对这些量逐元素一致，但离线基准约少 18% 调用耗时。
        mujoco.mj_fwdPosition(self.model, self._scratch)
        mujoco.mj_fwdVelocity(self.model, self._scratch)

    def _site_jacobians_world(self):
        jacp = self._jacp_workspace
        jacr = self._jacr_workspace
        jacp.fill(0.0)
        jacr.fill(0.0)
        mujoco.mj_jacSite(self.model, self._scratch, jacp, jacr, self.ee_site_id)
        # 这个模型里：floating base: nv = 6，23 个 hinge 关节: nv = 23，所以：model.nv = 6 + 23 = 29
        # jacp/jacr 原本是整机雅可比 shape=(3, model.nv)；这里只取右臂 qvel 列，变成 shape=(3, 5)。
        return jacp[:, self.qvel_indices].copy(), jacr[:, self.qvel_indices].copy()

    def _site_jacobian_dots_world(self):
        """在当前 scratch 状态解析计算抓持点 Jacobian 的时间导数。"""
        jacp_dot = self._jacp_dot_workspace
        jacr_dot = self._jacr_dot_workspace
        jacp_dot.fill(0.0)
        jacr_dot.fill(0.0)
        site_position = self._scratch.site_xpos[self.ee_site_id]
        site_body_id = int(self.model.site_bodyid[self.ee_site_id])
        # 【核心代码】mj_jacDot 直接使用当前 qpos/qvel，不再构造 q+、q-。
        # 因此每个工作点只需前面的一次 mj_forward，也无需恢复 scratch。
        mujoco.mj_jacDot(
            self.model,
            self._scratch,
            jacp_dot,
            jacr_dot,
            site_position,
            site_body_id,
        )
        return (
            jacp_dot[:, self.qvel_indices].copy(),
            jacr_dot[:, self.qvel_indices].copy(),
        )

    def compute_lqr_terms(
        self,
        q_right_arm: np.ndarray,
        dq_right_arm: np.ndarray,
        _W_R_I: np.ndarray,
        qpos_reference: np.ndarray,
        disturbance: Optional[DisturbanceInput] = None,
        torso_relative_position_reference: Optional[np.ndarray] = None,
    ) -> dict:
        q = np.asarray(q_right_arm, dtype=np.float64)
        dq = np.asarray(dq_right_arm, dtype=np.float64)
        self._set_scratch_state(qpos_reference, q, dq)
        J_v, J_w = self._site_jacobians_world()
        dJ_v, dJ_w = self._site_jacobian_dots_world()
        p_E = self._scratch.site_xpos[self.ee_site_id].copy()
        p_B = self._scratch.site_xpos[self.imu_site_id].copy()
        W_R_B = self._scratch.site_xmat[self.imu_site_id].reshape(3, 3).copy()
        omega_B = np.zeros(3, dtype=np.float64) if disturbance is None or disturbance.omega_world is None else np.asarray(disturbance.omega_world, dtype=np.float64)
        a_B = np.zeros(3, dtype=np.float64) if disturbance is None or disturbance.acc_world is None else np.asarray(disturbance.acc_world, dtype=np.float64)
        alpha_B = np.zeros(3, dtype=np.float64) if disturbance is None or disturbance.alpha_world is None else np.asarray(disturbance.alpha_world, dtype=np.float64)
        r_BE = p_E - p_B
        position = W_R_B.T @ r_BE
        position_reference = position.copy() if torso_relative_position_reference is None else np.asarray(torso_relative_position_reference, dtype=np.float64)
        position_error = position - position_reference
        # torso site 位于右臂运动链上游，因此只需把世界系末端 Jacobian 旋转到 torso 系。
        J_p = W_R_B.T @ J_v
        J_g = self.compute_gravity_error_jacobian(q, np.eye(3), qpos_reference)
        e_g = self.compute_gravity_error(q, np.eye(3), qpos_reference)
        return {
            "D_acc": a_B
            + np.cross(alpha_B, r_BE)
            + np.cross(
                omega_B, np.cross(omega_B, r_BE)
            ),
            "C_acc": 2.0 * self._skew(omega_B) @ J_v + dJ_v,
            "B_acc": J_v,
            "D_alpha": alpha_B,
            "C_alpha": self._skew(omega_B) @ J_w + dJ_w,
            "B_alpha": J_w,
            "d_p": position_error - J_p @ q,
            "G_p": np.hstack([J_p, np.zeros_like(J_p)]),
            "d_g": e_g - J_g @ q,
            "G_g": np.hstack([J_g, np.zeros_like(J_g)]),
            "position": position.copy(),
            "position_reference": position_reference.copy(),
            "position_error": position_error.copy(),
            "gravity_error": e_g.copy(),
        }

    def compute_mpc_terms(
        self,
        q_right_arm: np.ndarray,
        dq_right_arm: np.ndarray,
        qpos_reference: np.ndarray,
        node_disturbance: Optional[DisturbanceInput] = None,
        interval_disturbance: Optional[DisturbanceInput] = None,
        acceleration_required: bool = True,
    ) -> dict:
        """构造 MPC 单个预测步的局部仿射观测模型。

        【核心代码】这里只提供末端线/角加速度、世界系角速度与二维重力误差。
        torso-relative 位置既不进入代价，也不进入约束。
        """
        q = np.asarray(q_right_arm, dtype=np.float64)
        dq = np.asarray(dq_right_arm, dtype=np.float64)
        if self.prediction_backend is None:
            # 兼容旧调用：没有显式后端时继续使用本类原有的 MuJoCo scratch。
            self._set_scratch_state(qpos_reference, q, dq)
            J_v, J_w = self._site_jacobians_world()
            if acceleration_required:
                dJ_v, dJ_w = self._site_jacobian_dots_world()
            else:
                dJ_v = np.zeros_like(J_v)
                dJ_w = np.zeros_like(J_w)
            p_E = self._scratch.site_xpos[self.ee_site_id].copy()
            p_B = self._scratch.site_xpos[self.imu_site_id].copy()
            W_R_B_current = (
                self._scratch.site_xmat[self.imu_site_id]
                .reshape(3, 3)
                .copy()
            )
            W_R_E_current = (
                self._scratch.site_xmat[self.ee_site_id]
                .reshape(3, 3)
                .copy()
            )
        else:
            # 【核心代码】统一后端只返回同一组几何量，下面的扰动模型、
            # 重力线性化和代价定义不因换库而改变。
            prediction = self.prediction_backend.evaluate(
                qpos_reference,
                q,
                dq,
                acceleration_required=acceleration_required,
            )
            J_v = prediction.J_v_world
            J_w = prediction.J_w_world
            dJ_v = prediction.dJ_v_world
            dJ_w = prediction.dJ_w_world
            p_E = prediction.ee_position_world
            p_B = prediction.imu_position_world
            W_R_B_current = prediction.imu_rotation_world
            W_R_E_current = prediction.ee_rotation_world

        # 【核心代码】节点量描述 t_k；区间量描述 u_k 将执行的
        # [t_k,t_{k+1})。终端没有 u_N，未给区间量时兼容回退到节点量。
        interval = (
            node_disturbance
            if interval_disturbance is None
            else interval_disturbance
        )
        omega_B_node = self._disturbance_vector(
            node_disturbance, "omega_world"
        )
        if acceleration_required:
            omega_B_interval = self._disturbance_vector(
                interval, "omega_world"
            )
            a_B_interval = self._disturbance_vector(
                interval, "acc_world"
            )
            alpha_B_interval = self._disturbance_vector(
                interval, "alpha_world"
            )
        else:
            omega_B_interval = np.zeros(3, dtype=np.float64)
            a_B_interval = np.zeros(3, dtype=np.float64)
            alpha_B_interval = np.zeros(3, dtype=np.float64)
        W_R_B_predicted = self._disturbance_rotation(
            node_disturbance, W_R_B_current
        )

        # 【核心代码】scratch 中的 p_E-p_B 已是当前 base 姿态下的世界系杠杆臂。
        # base 平移严格相消，不需要进入模板。前馈只需用预测姿态相对当前姿态
        # 的旋转增量修正它；关闭前馈时 delta_R=I，退化为 LQR 的直接差值写法。
        delta_R = W_R_B_predicted @ W_R_B_current.T
        r_BE = (
            delta_R @ (p_E - p_B)
            if acceleration_required
            else np.zeros(3, dtype=np.float64)
        )
        if acceleration_required:
            J_v = delta_R @ J_v
            dJ_v = delta_R @ dJ_v
        else:
            # Qa/Qalpha 关闭或终端节点时，这些量不进入代价；保留正确
            # shape 的零矩阵供统一校验，避免无意义的 cross/skew 运算。
            J_v = np.zeros_like(J_v)
            dJ_v = np.zeros_like(dJ_v)
        J_w = delta_R @ J_w
        dJ_w = (
            delta_R @ dJ_w
            if acceleration_required
            else np.zeros_like(dJ_w)
        )
        W_R_E = delta_R @ W_R_E_current

        # 【核心代码】解析式二维重力 Jacobian。
        # 相比逐关节中心差分，每个预测步可少做 10 次 mj_forward。
        gravity_world = np.asarray(self.model.opt.gravity, dtype=np.float64)
        gravity_end = W_R_E.T @ gravity_world
        J_g = (W_R_E.T @ self._skew(gravity_world) @ J_w)[:2, :]
        gravity_error = gravity_end[:2]

        if acceleration_required:
            D_acc = a_B_interval + np.cross(
                alpha_B_interval, r_BE
            ) + np.cross(
                omega_B_interval,
                np.cross(omega_B_interval, r_BE),
            )
            C_acc = (
                2.0 * self._skew(omega_B_interval) @ J_v + dJ_v
            )
            D_alpha = alpha_B_interval
            C_alpha = (
                self._skew(omega_B_interval) @ J_w + dJ_w
            )
        else:
            D_acc = np.zeros(3, dtype=np.float64)
            C_acc = np.zeros_like(J_v)
            D_alpha = np.zeros(3, dtype=np.float64)
            C_alpha = np.zeros_like(J_w)

        return {
            "D_acc": D_acc,
            "C_acc": C_acc,
            "B_acc": J_v,
            "D_alpha": D_alpha,
            "C_alpha": C_alpha,
            "B_alpha": J_w,
            # 【核心代码】世界系末端角速度：
            # omega_E = omega_B + J_omega(q) dq。
            # J_w 已在上面用 delta_R 修正到该预测步的 base 姿态。
            "D_omega": omega_B_node,
            "C_omega": J_w,
            "d_g": gravity_error - J_g @ q,
            "G_g": np.hstack([J_g, np.zeros_like(J_g)]),
            "gravity_error": gravity_error.copy(),
        }

    @staticmethod
    def _disturbance_vector(disturbance: Optional[DisturbanceInput], name: str) -> np.ndarray:
        if disturbance is None:
            return np.zeros(3, dtype=np.float64)
        value = getattr(disturbance, name, None)
        return np.zeros(3, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)

    @staticmethod
    def _disturbance_rotation(
        disturbance: Optional[DisturbanceInput],
        default_rotation: np.ndarray,
    ) -> np.ndarray:
        if disturbance is None or disturbance.rot_world_body is None:
            return np.asarray(default_rotation, dtype=np.float64)
        rotation = np.asarray(disturbance.rot_world_body, dtype=np.float64)
        if (
            rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ValueError(
                "disturbance.rot_world_body 必须是有效的 3x3 旋转矩阵。"
            )
        return rotation

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(v, dtype=np.float64)
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
