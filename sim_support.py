import csv
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import mujoco
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)
RIGHT_ARM_QPOS_SLICE = slice(25, 30)
RIGHT_ARM_QVEL_SLICE = slice(24, 29)
RIGHT_ARM_CTRL_SLICE = slice(18, 23)
RIGHT_ARM_DDQ_SATURATION_EPS = 1e-2
RIGHT_ARM_TAU_SATURATION_EPS = 1e-6
LQR_COST_TERM_NAMES = (
    "linear_acceleration",
    "angular_acceleration",
    "position",
    "gravity",
    "posture",
    "velocity",
    "control",
)
CONTACT_CONSTRAINT_TYPES = np.array(
    [
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC),
    ],
    dtype=np.int32,
)
FRICTION_CONSTRAINT_TYPES = np.array(
    [
        int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF),
        int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON),
    ],
    dtype=np.int32,
)


@dataclass
class SceneIds:
    torso_id: int
    imu_site_id: int
    torso_acc_sensor_id: int
    left_grasp_site_id: int
    right_grasp_site_id: int


@dataclass
class TorsoMotionState:
    quat: np.ndarray
    rotmat: np.ndarray
    lin_vel: np.ndarray
    ang_vel: np.ndarray
    lin_acc: np.ndarray
    ang_acc: np.ndarray


@dataclass(frozen=True)
class HeadingControlOutput:
    reference_world: float
    yaw_unwrapped: float
    yaw_filtered: float
    yaw_error: float
    yaw_rate_filtered: float
    yaw_rate_command: float
    yaw_rate_correction: float
    command_saturated: bool


class HeadingHoldController:
    """用一个步态周期的均值抑制躯干摆动，并闭环保持世界系航向。"""

    def __init__(
        self,
        sample_dt,
        averaging_window,
        kp,
        kd,
        yaw_rate_feedforward,
        max_abs_yaw_rate,
    ):
        self.kp = float(kp)
        self.kd = float(kd)
        self.yaw_rate_feedforward = float(yaw_rate_feedforward)
        self.max_abs_yaw_rate = float(max_abs_yaw_rate)
        self.window_samples = max(1, int(round(float(averaging_window) / float(sample_dt))))
        self._yaw_history = deque(maxlen=self.window_samples)
        self._yaw_rate_history = deque(maxlen=self.window_samples)
        self._previous_wrapped_yaw = None
        self._unwrapped_yaw = None
        self._reference_world = None
        self.last_output = HeadingControlOutput(
            reference_world=np.nan,
            yaw_unwrapped=np.nan,
            yaw_filtered=np.nan,
            yaw_error=np.nan,
            yaw_rate_filtered=np.nan,
            yaw_rate_command=self.yaw_rate_feedforward,
            yaw_rate_correction=0.0,
            command_saturated=False,
        )

    @staticmethod
    def _wrap_to_pi(angle):
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    def update(self, yaw_world, yaw_rate_world):
        yaw_world = float(yaw_world)
        yaw_rate_world = float(yaw_rate_world)
        if self._previous_wrapped_yaw is None:
            self._unwrapped_yaw = yaw_world
            self._reference_world = yaw_world
        else:
            self._unwrapped_yaw += self._wrap_to_pi(yaw_world - self._previous_wrapped_yaw)
        self._previous_wrapped_yaw = yaw_world

        self._yaw_history.append(self._unwrapped_yaw)
        self._yaw_rate_history.append(yaw_rate_world)
        yaw_filtered = float(np.mean(self._yaw_history))
        yaw_rate_filtered = float(np.mean(self._yaw_rate_history))
        yaw_error = float(self._reference_world - yaw_filtered)
        correction = self.kp * yaw_error - self.kd * yaw_rate_filtered
        command_raw = self.yaw_rate_feedforward + correction
        command = float(np.clip(command_raw, -self.max_abs_yaw_rate, self.max_abs_yaw_rate))
        self.last_output = HeadingControlOutput(
            reference_world=float(self._reference_world),
            yaw_unwrapped=float(self._unwrapped_yaw),
            yaw_filtered=yaw_filtered,
            yaw_error=yaw_error,
            yaw_rate_filtered=yaw_rate_filtered,
            yaw_rate_command=command,
            yaw_rate_correction=correction,
            command_saturated=not np.isclose(command, command_raw),
        )
        return self.last_output


@dataclass
class DirectDriveJointGroup:
    joint_names: tuple
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    ctrl_indices: np.ndarray
    joint_ids: np.ndarray
    actuator_ids: np.ndarray
    torque_limits: np.ndarray
    inverse_dynamics_data: mujoco.MjData
    forward_dynamics_data: mujoco.MjData


@dataclass
class InverseDynamicsResult:
    tau_ff: np.ndarray
    tau_inverse: np.ndarray
    tau_contact: np.ndarray
    tau_constraint_total: np.ndarray
    tau_constraint_noncontact: np.ndarray
    tau_constraint_nonfriction: np.ndarray
    tau_constraint_friction: np.ndarray


@dataclass
class ForwardDynamicsMappingResult:
    tau_nominal: np.ndarray
    tau_correction_raw: np.ndarray
    tau_correction: np.ndarray
    tau_cmd_raw: np.ndarray
    qacc_baseline: np.ndarray
    qacc_predicted: np.ndarray
    qacc_prediction_error: np.ndarray
    qacc_validated: np.ndarray
    qacc_validation_error: np.ndarray
    qacc_linearization_error: np.ndarray
    gain_matrix: np.ndarray
    singular_values: np.ndarray
    condition_number: float
    validation_scale: float
    validation_attempts: int
    validation_improved: bool
    validation_tracking_safety_satisfied: bool
    validation_qacc_safety_satisfied: bool
    validation_safe_candidate_count: int
    validation_total_error_rejections: int
    validation_joint_error_rejections: int
    validation_qacc_limit_rejections: int
    first_pass_qacc_validated: np.ndarray
    first_pass_qacc_validation_error: np.ndarray
    second_pass_triggered: bool
    second_pass_accepted: bool
    second_pass_tracking_safety_satisfied: bool
    second_pass_qacc_safety_satisfied: bool
    second_pass_tau_correction_raw: np.ndarray
    second_pass_tau_correction: np.ndarray
    second_pass_qacc_predicted: np.ndarray
    second_pass_qacc_validated: np.ndarray
    second_pass_qacc_validation_error: np.ndarray
    second_pass_qacc_linearization_error: np.ndarray
    second_pass_gain_matrix: np.ndarray
    second_pass_singular_values: np.ndarray
    second_pass_condition_number: float
    second_pass_validation_scale: float
    second_pass_validation_attempts: int
    second_pass_safe_candidate_count: int
    second_pass_total_error_rejections: int
    second_pass_joint_error_rejections: int
    second_pass_qacc_limit_rejections: int
    safety_fallback_used: bool
    safety_fallback_satisfied: bool
    safety_fallback_attempts: int


@dataclass
class EvalBuffers:
    eval_data: dict
    trajectory_data: dict
    prev_left_lin_vel: np.ndarray
    prev_left_ang_vel: np.ndarray
    prev_right_lin_vel: np.ndarray
    prev_right_ang_vel: np.ndarray
    prev_torso_lin_vel: np.ndarray
    prev_torso_ang_vel: np.ndarray
    torso_xy_start: Optional[np.ndarray] = None


# ==============================
# 核心代码：主控制链直接依赖的支持函数
# 这部分最值得优先阅读，主要服务 main_sim.py 的右臂控制与状态构建。
# ==============================
def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_site_vel(model, data, site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp @ data.qvel, jacr @ data.qvel


def update_torso_motion_state(model, data, scene_ids, buffers, counter, simulation_dt):
    lin_vel, ang_vel = get_site_vel(model, data, scene_ids.imu_site_id)
    imu_rot_world = data.site_xmat[scene_ids.imu_site_id].reshape(3, 3).copy()
    accel_adr = int(model.sensor_adr[scene_ids.torso_acc_sensor_id])
    accel_dim = int(model.sensor_dim[scene_ids.torso_acc_sensor_id])
    if accel_dim != 3:
        raise ValueError(f"torso accelerometer 维度应为 3，实际为 {accel_dim}。")
    specific_force_imu = data.sensordata[accel_adr:accel_adr + accel_dim].copy()
    # MuJoCo accelerometer 输出 IMU 局部系比力；旋转到世界系并加回世界系重力，得到平动加速度。
    lin_acc = np.zeros(3) if counter == 0 else imu_rot_world @ specific_force_imu + model.opt.gravity
    ang_acc = np.zeros(3) if counter == 0 else (ang_vel - buffers.prev_torso_ang_vel) / simulation_dt
    buffers.prev_torso_lin_vel, buffers.prev_torso_ang_vel = lin_vel.copy(), ang_vel.copy()
    return TorsoMotionState(
        quat=data.xquat[scene_ids.torso_id].copy(),
        rotmat=imu_rot_world,
        lin_vel=lin_vel.copy(),
        ang_vel=ang_vel.copy(),
        lin_acc=lin_acc,
        ang_acc=ang_acc,
    )


def build_right_arm_observation(current_q, current_dq, torso_state, dt):
    return {
        "current_q": current_q,
        "current_dq": current_dq,
        "torso_quat": torso_state.quat,
        "torso_omega": torso_state.ang_vel,
        "torso_acc": torso_state.lin_acc,
        "torso_alpha": torso_state.ang_acc,
        "torso_rotmat": torso_state.rotmat,
        "dt": dt,
    }


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


# ==============================
# 核心代码：右臂执行层与逆动力学前馈（这一整段是 sim_support.py 最值得优先阅读的部分）
# 这部分负责把 right_arm 的索引上下文、ddq_des 和 tau_pd 变成最终执行力矩。
# ==============================
def resolve_direct_drive_joint_group(
    model,
    joint_names,
    expected_qpos_indices,
    expected_qvel_indices,
    expected_ctrl_indices,
    group_label="关节组",
):
    joint_names = tuple(joint_names)
    qpos_indices = np.asarray(expected_qpos_indices, dtype=np.int32)
    qvel_indices = np.asarray(expected_qvel_indices, dtype=np.int32)
    ctrl_indices = np.asarray(expected_ctrl_indices, dtype=np.int32)

    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names],
        dtype=np.int32,
    )
    actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in joint_names],
        dtype=np.int32,
    )
    if len(joint_names) != len(qpos_indices) or len(joint_names) != len(qvel_indices) or len(joint_names) != len(ctrl_indices):
        raise ValueError(f"{group_label} joint/qpos/qvel/ctrl 数量不一致。")
    if np.any(joint_ids < 0) or np.any(actuator_ids < 0):
        missing_joints = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
        missing_actuators = [name for name, actuator_id in zip(joint_names, actuator_ids) if actuator_id < 0]
        raise ValueError(f"{group_label} 找不到 joint/actuator: joints={missing_joints}, actuators={missing_actuators}")
    if not np.array_equal(model.jnt_qposadr[joint_ids], qpos_indices):
        raise ValueError(f"{group_label} qpos 索引与预期不一致。")
    if not np.array_equal(model.jnt_dofadr[joint_ids], qvel_indices):
        raise ValueError(f"{group_label} qvel 索引与预期不一致。")
    if not np.array_equal(actuator_ids, ctrl_indices):
        raise ValueError(f"{group_label} ctrl 索引与预期不一致。")
    if not np.array_equal(model.actuator_trnid[actuator_ids, 0], joint_ids):
        raise ValueError(f"{group_label} actuator 没有一一驱动对应 joint。")
    if not np.allclose(model.actuator_gear[actuator_ids, 0], 1.0):
        raise ValueError(f"{group_label} actuator 不是 gear=1 的 direct-drive 映射。")

    torque_limits = model.jnt_actfrcrange[joint_ids].copy()
    if not np.all(torque_limits[:, 0] < torque_limits[:, 1]):
        raise ValueError(f"{group_label} 必须在 XML 中配置有效的 actuatorfrcrange。")

    return DirectDriveJointGroup(
        joint_names=joint_names,
        qpos_indices=qpos_indices,
        qvel_indices=qvel_indices,
        ctrl_indices=ctrl_indices,
        joint_ids=joint_ids,
        actuator_ids=actuator_ids,
        torque_limits=torque_limits,
        inverse_dynamics_data=mujoco.MjData(model),
        forward_dynamics_data=mujoco.MjData(model),
    )


def resolve_right_arm_control_context(model, joint_names):
    return resolve_direct_drive_joint_group(
        model,
        joint_names,
        expected_qpos_indices=np.arange(RIGHT_ARM_QPOS_SLICE.start, RIGHT_ARM_QPOS_SLICE.stop, dtype=np.int32),
        expected_qvel_indices=np.arange(RIGHT_ARM_QVEL_SLICE.start, RIGHT_ARM_QVEL_SLICE.stop, dtype=np.int32),
        expected_ctrl_indices=np.arange(RIGHT_ARM_CTRL_SLICE.start, RIGHT_ARM_CTRL_SLICE.stop, dtype=np.int32),
        group_label="右臂逆动力学",
    )


def _constraint_generalized_force(model, scratch, constraint_types):
    """从 MuJoCo 约束行中重建指定类型的广义力。"""
    if scratch.nefc == 0:
        return np.zeros(model.nv, dtype=np.float64)
    efc_jacobian = np.asarray(scratch.efc_J, dtype=np.float64).reshape(-1, model.nv)[:scratch.nefc]
    efc_type = np.asarray(scratch.efc_type[:scratch.nefc], dtype=np.int32)
    selected_rows = np.isin(efc_type, constraint_types)
    if not np.any(selected_rows):
        return np.zeros(model.nv, dtype=np.float64)
    return efc_jacobian[selected_rows].T @ scratch.efc_force[:scratch.nefc][selected_rows]


def inverse_dynamics_feedforward(model, data, scratch, desired_qacc, qvel_indices):
    """计算不对抗非摩擦约束、同时保留摩擦补偿的逆动力学前馈。"""
    qvel_indices = np.asarray(qvel_indices, dtype=np.int32)
    desired_qacc = np.asarray(desired_qacc, dtype=np.float64)
    if desired_qacc.shape != qvel_indices.shape:
        raise ValueError(
            f"desired_qacc shape {desired_qacc.shape} 与 qvel_indices shape "
            f"{qvel_indices.shape} 不一致。"
        )

    scratch.time = data.time
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = data.qvel
    scratch.qacc[:] = 0.0
    scratch.qacc[qvel_indices] = desired_qacc
    scratch.qfrc_applied[:] = data.qfrc_applied
    scratch.xfrc_applied[:] = data.xfrc_applied
    if model.nmocap:
        scratch.mocap_pos[:] = data.mocap_pos
        scratch.mocap_quat[:] = data.mocap_quat

    mujoco.mj_inverse(model, scratch)
    tau_inverse = scratch.qfrc_inverse[qvel_indices].copy()
    tau_constraint_total = scratch.qfrc_constraint[qvel_indices].copy()
    tau_contact = _constraint_generalized_force(model, scratch, CONTACT_CONSTRAINT_TYPES)[qvel_indices]
    tau_constraint_friction = _constraint_generalized_force(
        model,
        scratch,
        FRICTION_CONSTRAINT_TYPES,
    )[qvel_indices]
    tau_constraint_noncontact = tau_constraint_total - tau_contact
    tau_constraint_nonfriction = tau_constraint_total - tau_constraint_friction
    # 加回 contact、limit、equality 等非摩擦约束，不让执行器主动抵消它们。
    # frictionloss 仍保留在 qfrc_inverse 中，由前馈力矩正常克服。
    tau_ff = tau_inverse + tau_constraint_nonfriction
    return InverseDynamicsResult(
        tau_ff=tau_ff,
        tau_inverse=tau_inverse,
        tau_contact=tau_contact,
        tau_constraint_total=tau_constraint_total,
        tau_constraint_noncontact=tau_constraint_noncontact,
        tau_constraint_nonfriction=tau_constraint_nonfriction,
        tau_constraint_friction=tau_constraint_friction,
    )


def local_forward_dynamics_torque_mapping(
    model,
    data,
    scratch,
    fixed_ctrl,
    desired_qacc,
    tau_nominal,
    qvel_indices,
    ctrl_indices,
    torque_limits,
    perturbation=0.1,
    regularization=5.0,
    validation_scales=(1.0, 0.5, 0.25, 0.125),
    second_pass_error_threshold=5.0,
    max_joint_error=4.0,
    max_abs_qacc=8.0,
):
    """局部线性求力矩；高残差时在已验收力矩处重线性化一次。"""
    desired_qacc = np.asarray(desired_qacc, dtype=np.float64)
    tau_nominal = np.asarray(tau_nominal, dtype=np.float64)
    qvel_indices = np.asarray(qvel_indices, dtype=np.int32)
    ctrl_indices = np.asarray(ctrl_indices, dtype=np.int32)
    torque_limits = np.asarray(torque_limits, dtype=np.float64)
    fixed_ctrl = np.asarray(fixed_ctrl, dtype=np.float64)
    joint_count = len(qvel_indices)
    if desired_qacc.shape != (joint_count,) or tau_nominal.shape != (joint_count,):
        raise ValueError("前向动力学映射的 desired_qacc/tau_nominal 维度不正确。")
    if fixed_ctrl.shape != (model.nu,) or torque_limits.shape != (joint_count, 2):
        raise ValueError("前向动力学映射的 ctrl/torque_limits 维度不正确。")
    if (
        perturbation <= 0.0
        or regularization < 0.0
        or second_pass_error_threshold < 0.0
        or max_joint_error <= 0.0
        or max_abs_qacc <= 0.0
    ):
        raise ValueError("扰动和验收安全阈值必须大于 0，正则化与二次修正阈值不能小于 0。")
    validation_scales = tuple(float(scale) for scale in validation_scales)
    if not validation_scales or any(scale <= 0.0 or scale > 1.0 for scale in validation_scales):
        raise ValueError("validation_scales 必须是位于 (0, 1] 的非空序列。")

    tau_nominal = np.clip(tau_nominal, torque_limits[:, 0], torque_limits[:, 1])
    baseline_ctrl = fixed_ctrl.copy()
    baseline_ctrl[ctrl_indices] = tau_nominal
    mujoco.mj_copyData(scratch, model, data)
    qacc_warmstart = scratch.qacc_warmstart.copy()
    scratch.ctrl[:] = baseline_ctrl
    scratch.qacc_warmstart[:] = qacc_warmstart
    mujoco.mj_forward(model, scratch)
    qacc_baseline = scratch.qacc[qvel_indices].copy()

    def solve_validated_pass(base_tau, base_qacc):
        """在指定力矩工作点重线性化，完整检查候选并返回安全候选中的最优项。"""
        pass_ctrl = fixed_ctrl.copy()
        pass_ctrl[ctrl_indices] = base_tau
        gain_matrix = np.zeros((joint_count, joint_count), dtype=np.float64)
        for column in range(joint_count):
            signed_perturbation = float(perturbation)
            if base_tau[column] + signed_perturbation > torque_limits[column, 1]:
                signed_perturbation = -signed_perturbation
            scratch.ctrl[:] = pass_ctrl
            scratch.ctrl[ctrl_indices[column]] += signed_perturbation
            scratch.qacc_warmstart[:] = qacc_warmstart
            mujoco.mj_forwardSkip(model, scratch, mujoco.mjtStage.mjSTAGE_VEL, 1)
            gain_matrix[:, column] = (
                scratch.qacc[qvel_indices] - base_qacc
            ) / signed_perturbation

        u_matrix, singular_values, vt_matrix = np.linalg.svd(gain_matrix, full_matrices=False)
        acceleration_error = desired_qacc - base_qacc
        damped_inverse = singular_values / (singular_values ** 2 + float(regularization))
        correction_raw = vt_matrix.T @ (damped_inverse * (u_matrix.T @ acceleration_error))
        base_error_norm = float(np.linalg.norm(base_qacc - desired_qacc))
        validation_scale = 0.0
        validation_attempts = 0
        safe_candidate_count = 0
        total_error_rejections = 0
        joint_error_rejections = 0
        qacc_limit_rejections = 0
        tau_cmd = base_tau.copy()
        tau_cmd_raw = base_tau.copy()
        qacc_validated = base_qacc.copy()
        best_error_norm = np.inf
        best_qacc_safe = None
        best_progress = None
        for scale in validation_scales:
            validation_attempts += 1
            candidate_tau_raw = base_tau + scale * correction_raw
            candidate_tau = np.clip(candidate_tau_raw, torque_limits[:, 0], torque_limits[:, 1])
            candidate_ctrl = pass_ctrl.copy()
            candidate_ctrl[ctrl_indices] = candidate_tau
            scratch.ctrl[:] = candidate_ctrl
            scratch.qacc_warmstart[:] = qacc_warmstart
            mujoco.mj_forward(model, scratch)
            candidate_qacc = scratch.qacc[qvel_indices].copy()
            candidate_error = candidate_qacc - desired_qacc
            candidate_error_norm = float(np.linalg.norm(candidate_error))
            total_error_improved = candidate_error_norm < base_error_norm
            joint_error_safe = float(np.max(np.abs(candidate_error))) <= float(max_joint_error)
            qacc_safe = float(np.max(np.abs(candidate_qacc))) <= float(max_abs_qacc)
            total_error_rejections += int(not total_error_improved)
            joint_error_rejections += int(not joint_error_safe)
            qacc_limit_rejections += int(not qacc_safe)
            if total_error_improved and joint_error_safe and qacc_safe:
                safe_candidate_count += 1
            if (
                total_error_improved
                and joint_error_safe
                and qacc_safe
                and candidate_error_norm < best_error_norm
            ):
                validation_scale = scale
                tau_cmd = candidate_tau
                tau_cmd_raw = candidate_tau_raw
                qacc_validated = candidate_qacc
                best_error_norm = candidate_error_norm
            if total_error_improved and qacc_safe:
                qacc_safe_key = (float(np.max(np.abs(candidate_error))), candidate_error_norm)
                if best_qacc_safe is None or qacc_safe_key < best_qacc_safe[0]:
                    best_qacc_safe = (
                        qacc_safe_key,
                        scale,
                        candidate_tau,
                        candidate_tau_raw,
                        candidate_qacc,
                    )
            if total_error_improved and (
                best_progress is None or candidate_error_norm < best_progress[0]
            ):
                best_progress = (
                    candidate_error_norm,
                    scale,
                    candidate_tau,
                    candidate_tau_raw,
                    candidate_qacc,
                )

        # 严格候选不可行时仍沿改善方向建立第二轮工作点；优先保住 qacc 硬上限。
        tracking_safety_satisfied = safe_candidate_count > 0
        if not tracking_safety_satisfied and best_qacc_safe is not None:
            _, validation_scale, tau_cmd, tau_cmd_raw, qacc_validated = best_qacc_safe
        elif not tracking_safety_satisfied and best_progress is not None:
            _, validation_scale, tau_cmd, tau_cmd_raw, qacc_validated = best_progress
        qacc_safety_satisfied = float(np.max(np.abs(qacc_validated))) <= float(max_abs_qacc)

        correction = tau_cmd - base_tau
        qacc_predicted = base_qacc + gain_matrix @ correction
        if singular_values.size == 0 or singular_values[-1] <= np.finfo(np.float64).eps:
            condition_number = np.inf
        else:
            condition_number = float(singular_values[0] / singular_values[-1])
        return {
            "tau_cmd": tau_cmd,
            "tau_cmd_raw": tau_cmd_raw,
            "correction_raw": correction_raw,
            "correction": correction,
            "qacc_predicted": qacc_predicted,
            "qacc_validated": qacc_validated,
            "qacc_validation_error": qacc_validated - desired_qacc,
            "qacc_linearization_error": qacc_validated - qacc_predicted,
            "gain_matrix": gain_matrix,
            "singular_values": singular_values,
            "condition_number": condition_number,
            "validation_scale": validation_scale,
            "validation_attempts": validation_attempts,
            "improved": validation_scale > 0.0,
            "tracking_safety_satisfied": tracking_safety_satisfied,
            "qacc_safety_satisfied": qacc_safety_satisfied,
            "safe_candidate_count": safe_candidate_count,
            "total_error_rejections": total_error_rejections,
            "joint_error_rejections": joint_error_rejections,
            "qacc_limit_rejections": qacc_limit_rejections,
        }

    first_pass = solve_validated_pass(tau_nominal, qacc_baseline)
    first_pass_residual_norm = float(np.linalg.norm(first_pass["qacc_validation_error"]))
    second_pass_triggered = bool(
        first_pass["improved"]
        and (
            not first_pass["tracking_safety_satisfied"]
            or first_pass_residual_norm > float(second_pass_error_threshold)
        )
    )
    second_pass = None
    if second_pass_triggered:
        # 第一轮候选可能已经进入新的摩擦/接触模式；在该力矩处重算 G，再修正一次剩余残差。
        second_pass = solve_validated_pass(
            first_pass["tau_cmd"],
            first_pass["qacc_validated"],
        )

    if second_pass is not None and second_pass["tracking_safety_satisfied"]:
        final_pass = second_pass
    elif first_pass["tracking_safety_satisfied"]:
        final_pass = first_pass
    elif (
        second_pass is not None
        and second_pass["improved"]
        and second_pass["qacc_safety_satisfied"]
    ):
        final_pass = second_pass
    else:
        final_pass = first_pass
    second_pass_accepted = bool(second_pass is not None and final_pass is second_pass)

    safety_fallback_used = False
    safety_fallback_satisfied = final_pass["qacc_safety_satisfied"]
    safety_fallback_attempts = 0
    if not safety_fallback_satisfied:
        # 接触瞬态下固定回退力矩也可能更危险；在当前最佳点最多再重线性化两次。
        safety_fallback_used = True
        for _ in range(2):
            safety_fallback_attempts += 1
            rescue_pass = solve_validated_pass(
                final_pass["tau_cmd"],
                final_pass["qacc_validated"],
            )
            if not rescue_pass["improved"]:
                break
            final_pass = rescue_pass
            if final_pass["qacc_safety_satisfied"]:
                break
        safety_fallback_satisfied = final_pass["qacc_safety_satisfied"]
    zero_vector = np.zeros(joint_count, dtype=np.float64)
    zero_matrix = np.zeros((joint_count, joint_count), dtype=np.float64)
    tau_cmd = final_pass["tau_cmd"]
    tau_correction = tau_cmd - tau_nominal
    qacc_predicted = final_pass["qacc_predicted"]
    qacc_validated = final_pass["qacc_validated"]
    qacc_prediction_error = qacc_predicted - desired_qacc
    qacc_validation_error = qacc_validated - desired_qacc
    qacc_linearization_error = final_pass["qacc_linearization_error"]
    gain_matrix = final_pass["gain_matrix"]
    singular_values = final_pass["singular_values"]
    condition_number = final_pass["condition_number"]
    return tau_cmd, ForwardDynamicsMappingResult(
        tau_nominal=tau_nominal,
        tau_correction_raw=first_pass["correction_raw"],
        tau_correction=tau_correction,
        tau_cmd_raw=final_pass["tau_cmd_raw"],
        qacc_baseline=qacc_baseline,
        qacc_predicted=qacc_predicted,
        qacc_prediction_error=qacc_prediction_error,
        qacc_validated=qacc_validated,
        qacc_validation_error=qacc_validation_error,
        qacc_linearization_error=qacc_linearization_error,
        gain_matrix=gain_matrix,
        singular_values=singular_values,
        condition_number=condition_number,
        validation_scale=first_pass["validation_scale"],
        validation_attempts=first_pass["validation_attempts"],
        validation_improved=first_pass["improved"],
        validation_tracking_safety_satisfied=first_pass["tracking_safety_satisfied"],
        validation_qacc_safety_satisfied=first_pass["qacc_safety_satisfied"],
        validation_safe_candidate_count=first_pass["safe_candidate_count"],
        validation_total_error_rejections=first_pass["total_error_rejections"],
        validation_joint_error_rejections=first_pass["joint_error_rejections"],
        validation_qacc_limit_rejections=first_pass["qacc_limit_rejections"],
        first_pass_qacc_validated=first_pass["qacc_validated"],
        first_pass_qacc_validation_error=first_pass["qacc_validation_error"],
        second_pass_triggered=second_pass_triggered,
        second_pass_accepted=second_pass_accepted,
        second_pass_tracking_safety_satisfied=(
            second_pass["tracking_safety_satisfied"] if second_pass is not None else False
        ),
        second_pass_qacc_safety_satisfied=(
            second_pass["qacc_safety_satisfied"] if second_pass is not None else False
        ),
        second_pass_tau_correction_raw=(
            second_pass["correction_raw"] if second_pass is not None else zero_vector
        ),
        second_pass_tau_correction=(
            second_pass["correction"] if second_pass is not None else zero_vector
        ),
        second_pass_qacc_predicted=(
            second_pass["qacc_predicted"] if second_pass is not None else zero_vector
        ),
        second_pass_qacc_validated=(
            second_pass["qacc_validated"] if second_pass is not None else zero_vector
        ),
        second_pass_qacc_validation_error=(
            second_pass["qacc_validation_error"] if second_pass is not None else zero_vector
        ),
        second_pass_qacc_linearization_error=(
            second_pass["qacc_linearization_error"] if second_pass is not None else zero_vector
        ),
        second_pass_gain_matrix=(
            second_pass["gain_matrix"] if second_pass is not None else zero_matrix
        ),
        second_pass_singular_values=(
            second_pass["singular_values"] if second_pass is not None else zero_vector
        ),
        second_pass_condition_number=(
            second_pass["condition_number"] if second_pass is not None else np.inf
        ),
        second_pass_validation_scale=(
            second_pass["validation_scale"] if second_pass is not None else 0.0
        ),
        second_pass_validation_attempts=(
            second_pass["validation_attempts"] if second_pass is not None else 0
        ),
        second_pass_safe_candidate_count=(
            second_pass["safe_candidate_count"] if second_pass is not None else 0
        ),
        second_pass_total_error_rejections=(
            second_pass["total_error_rejections"] if second_pass is not None else 0
        ),
        second_pass_joint_error_rejections=(
            second_pass["joint_error_rejections"] if second_pass is not None else 0
        ),
        second_pass_qacc_limit_rejections=(
            second_pass["qacc_limit_rejections"] if second_pass is not None else 0
        ),
        safety_fallback_used=safety_fallback_used,
        safety_fallback_satisfied=safety_fallback_satisfied,
        safety_fallback_attempts=safety_fallback_attempts,
    )


def apply_computed_torque_control(
    model,
    data,
    id_index_scratch,
    desired_qacc,
    tau_pd,
    fixed_ctrl,
    forward_dynamics_perturbation=0.1,
    forward_dynamics_regularization=5.0,
    forward_dynamics_second_pass_error_threshold=5.0,
    forward_dynamics_max_joint_error=4.0,
    forward_dynamics_max_abs_qacc=8.0,
):
    tau_pd = np.asarray(tau_pd, dtype=np.float64)
    desired_qacc = np.asarray(desired_qacc, dtype=np.float64)
    if tau_pd.shape != id_index_scratch.qvel_indices.shape:
        raise ValueError(f"tau_pd shape {tau_pd.shape} 与控制关节数量 {id_index_scratch.qvel_indices.shape} 不一致。")

    inverse_result = inverse_dynamics_feedforward(
        model,
        data,
        id_index_scratch.inverse_dynamics_data,
        desired_qacc,
        id_index_scratch.qvel_indices,
    )
    tau_nominal = np.clip(
        inverse_result.tau_ff + tau_pd,
        id_index_scratch.torque_limits[:, 0],
        id_index_scratch.torque_limits[:, 1],
    )
    tau_cmd, mapping_result = local_forward_dynamics_torque_mapping(
        model,
        data,
        id_index_scratch.forward_dynamics_data,
        fixed_ctrl,
        desired_qacc,
        tau_nominal,
        id_index_scratch.qvel_indices,
        id_index_scratch.ctrl_indices,
        id_index_scratch.torque_limits,
        perturbation=forward_dynamics_perturbation,
        regularization=forward_dynamics_regularization,
        second_pass_error_threshold=forward_dynamics_second_pass_error_threshold,
        max_joint_error=forward_dynamics_max_joint_error,
        max_abs_qacc=forward_dynamics_max_abs_qacc,
    )
    return tau_cmd, inverse_result, mapping_result


# ==============================
# 非核心代码：性能统计与实验辅助（建议放在文件后半部分）
# 这部分主要服务调试、测速和结果保存，不是控制数学核心。
# 如果继续做第二轮整理，应将整个 PerformanceMonitor 区块后移，
# 让 right_arm 执行层与逆动力学前馈整体进入文件前半部分。
# ==============================
@dataclass
class PerformanceMonitor:
    step_budget: float
    arm_budget: Optional[float] = None
    warn_interval: Optional[int] = None
    step_start: float = 0.0
    arm_control_start: float = 0.0
    arm_control_elapsed: float = 0.0
    computed_torque_start: float = 0.0
    computed_torque_elapsed: float = 0.0
    mj_step_start: float = 0.0
    mj_step_elapsed: float = 0.0
    arm_control_ran: bool = False
    computed_torque_ran: bool = False
    total_steps: int = 0
    total_arm_updates: int = 0
    total_arm_elapsed: float = 0.0
    total_arm_total_elapsed: float = 0.0
    total_computed_torque_updates: int = 0
    total_computed_torque_elapsed: float = 0.0
    total_mj_step_elapsed: float = 0.0
    total_other_elapsed: float = 0.0
    total_loop_elapsed: float = 0.0
    max_arm_elapsed: float = 0.0
    max_arm_total_elapsed: float = 0.0
    max_computed_torque_elapsed: float = 0.0
    max_mj_step_elapsed: float = 0.0
    max_other_elapsed: float = 0.0
    max_loop_elapsed: float = 0.0
    arm_overruns: int = 0
    arm_total_overruns: int = 0
    computed_torque_overruns: int = 0
    loop_overruns: int = 0
    window_steps: int = 0
    window_arm_elapsed: float = 0.0
    window_arm_total_elapsed: float = 0.0
    window_computed_torque_elapsed: float = 0.0
    window_mj_step_elapsed: float = 0.0
    window_other_elapsed: float = 0.0
    window_loop_elapsed: float = 0.0
    window_max_arm_elapsed: float = 0.0
    window_max_arm_total_elapsed: float = 0.0
    window_max_computed_torque_elapsed: float = 0.0
    window_max_mj_step_elapsed: float = 0.0
    window_max_other_elapsed: float = 0.0
    window_max_loop_elapsed: float = 0.0
    window_arm_overruns: int = 0
    window_arm_total_overruns: int = 0
    window_computed_torque_overruns: int = 0
    window_loop_overruns: int = 0
    window_arm_updates: int = 0
    window_computed_torque_updates: int = 0
    window_reports: list = field(default_factory=list)

    def __post_init__(self):
        if self.arm_budget is None:
            self.arm_budget = self.step_budget
        if self.warn_interval is None:
            self.warn_interval = max(1, int(round(1.0 / self.step_budget)))

    def start_step(self):
        self.step_start = time.perf_counter()

    def start_arm_control(self):
        self.arm_control_start = time.perf_counter()
        self.arm_control_ran = True

    def finish_arm_control(self):
        self.arm_control_elapsed = time.perf_counter() - self.arm_control_start

    def start_computed_torque_control(self):
        self.computed_torque_start = time.perf_counter()
        self.computed_torque_ran = True

    def finish_computed_torque_control(self):
        self.computed_torque_elapsed += time.perf_counter() - self.computed_torque_start

    def start_mj_step(self):
        self.mj_step_start = time.perf_counter()

    def finish_mj_step(self):
        self.mj_step_elapsed = time.perf_counter() - self.mj_step_start

    def finish_step(self, counter, sleep=True):
        loop_elapsed = time.perf_counter() - self.step_start
        self.record_step(loop_elapsed)
        self.print_window_summary_if_needed(counter)
        if sleep:
            time_until_next_step = self.step_budget - loop_elapsed
            if time_until_next_step > 0.0:
                time.sleep(time_until_next_step)
        return loop_elapsed

    def record_step(self, loop_elapsed):
        other_elapsed = max(0.0, loop_elapsed - self.arm_control_elapsed - self.computed_torque_elapsed - self.mj_step_elapsed)
        self.total_steps += 1
        self.total_mj_step_elapsed += self.mj_step_elapsed
        self.total_other_elapsed += other_elapsed
        self.total_loop_elapsed += loop_elapsed
        self.max_mj_step_elapsed = max(self.max_mj_step_elapsed, self.mj_step_elapsed)
        self.max_other_elapsed = max(self.max_other_elapsed, other_elapsed)
        self.max_loop_elapsed = max(self.max_loop_elapsed, loop_elapsed)
        if loop_elapsed > self.step_budget:
            self.loop_overruns += 1

        self.window_steps += 1
        self.window_mj_step_elapsed += self.mj_step_elapsed
        self.window_other_elapsed += other_elapsed
        self.window_loop_elapsed += loop_elapsed
        self.window_max_mj_step_elapsed = max(self.window_max_mj_step_elapsed, self.mj_step_elapsed)
        self.window_max_other_elapsed = max(self.window_max_other_elapsed, other_elapsed)
        self.window_max_loop_elapsed = max(self.window_max_loop_elapsed, loop_elapsed)
        if loop_elapsed > self.step_budget:
            self.window_loop_overruns += 1

        if self.arm_control_ran:
            arm_total_elapsed = self.arm_control_elapsed + self.computed_torque_elapsed
            self.total_arm_updates += 1
            self.total_arm_elapsed += self.arm_control_elapsed
            self.total_arm_total_elapsed += arm_total_elapsed
            self.max_arm_elapsed = max(self.max_arm_elapsed, self.arm_control_elapsed)
            self.max_arm_total_elapsed = max(self.max_arm_total_elapsed, arm_total_elapsed)
            self.window_arm_updates += 1
            self.window_arm_elapsed += self.arm_control_elapsed
            self.window_arm_total_elapsed += arm_total_elapsed
            self.window_max_arm_elapsed = max(self.window_max_arm_elapsed, self.arm_control_elapsed)
            self.window_max_arm_total_elapsed = max(self.window_max_arm_total_elapsed, arm_total_elapsed)
            if self.arm_control_elapsed > self.arm_budget:
                self.arm_overruns += 1
                self.window_arm_overruns += 1
            if arm_total_elapsed > self.arm_budget:
                self.arm_total_overruns += 1
                self.window_arm_total_overruns += 1

        if self.computed_torque_ran:
            self.total_computed_torque_updates += 1
            self.total_computed_torque_elapsed += self.computed_torque_elapsed
            self.max_computed_torque_elapsed = max(self.max_computed_torque_elapsed, self.computed_torque_elapsed)
            self.window_computed_torque_updates += 1
            self.window_computed_torque_elapsed += self.computed_torque_elapsed
            self.window_max_computed_torque_elapsed = max(self.window_max_computed_torque_elapsed, self.computed_torque_elapsed)
            if self.computed_torque_elapsed > self.step_budget:
                self.computed_torque_overruns += 1
                self.window_computed_torque_overruns += 1

        self.arm_control_ran = False
        self.computed_torque_ran = False
        self.arm_control_elapsed = 0.0
        self.computed_torque_elapsed = 0.0
        self.mj_step_elapsed = 0.0

    def print_window_summary_if_needed(self, counter):
        if counter % self.warn_interval != 0 or self.window_steps == 0:
            return

        report = self._build_report(
            "perf",
            self.window_steps,
            self.window_arm_updates,
            self.window_arm_elapsed,
            self.window_arm_total_elapsed,
            self.window_computed_torque_updates,
            self.window_computed_torque_elapsed,
            self.window_mj_step_elapsed,
            self.window_other_elapsed,
            self.window_loop_elapsed,
            self.window_max_arm_elapsed,
            self.window_max_arm_total_elapsed,
            self.window_max_computed_torque_elapsed,
            self.window_max_mj_step_elapsed,
            self.window_max_other_elapsed,
            self.window_max_loop_elapsed,
            self.window_arm_overruns,
            self.window_arm_total_overruns,
            self.window_computed_torque_overruns,
            self.window_loop_overruns,
        )
        report["end_step"] = int(counter)
        self.window_reports.append(report)
        self._print_report(report)
        self.window_steps = 0
        self.window_arm_elapsed = 0.0
        self.window_arm_total_elapsed = 0.0
        self.window_computed_torque_elapsed = 0.0
        self.window_mj_step_elapsed = 0.0
        self.window_other_elapsed = 0.0
        self.window_loop_elapsed = 0.0
        self.window_arm_updates = 0
        self.window_computed_torque_updates = 0
        self.window_max_arm_elapsed = 0.0
        self.window_max_arm_total_elapsed = 0.0
        self.window_max_computed_torque_elapsed = 0.0
        self.window_max_mj_step_elapsed = 0.0
        self.window_max_other_elapsed = 0.0
        self.window_max_loop_elapsed = 0.0
        self.window_arm_overruns = 0
        self.window_arm_total_overruns = 0
        self.window_computed_torque_overruns = 0
        self.window_loop_overruns = 0

    def print_summary(self):
        if self.total_steps == 0:
            return
        self._print_report(self.build_total_report())

    def build_total_report(self):
        return self._build_report(
            "perf total",
            self.total_steps,
            self.total_arm_updates,
            self.total_arm_elapsed,
            self.total_arm_total_elapsed,
            self.total_computed_torque_updates,
            self.total_computed_torque_elapsed,
            self.total_mj_step_elapsed,
            self.total_other_elapsed,
            self.total_loop_elapsed,
            self.max_arm_elapsed,
            self.max_arm_total_elapsed,
            self.max_computed_torque_elapsed,
            self.max_mj_step_elapsed,
            self.max_other_elapsed,
            self.max_loop_elapsed,
            self.arm_overruns,
            self.arm_total_overruns,
            self.computed_torque_overruns,
            self.loop_overruns,
        )

    def save_report(self, run_dir):
        total_report = self.build_total_report()
        summary_path = os.path.join(run_dir, "perf_summary.json")
        windows_path = os.path.join(run_dir, "perf_windows.csv")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"total": total_report, "warn_interval": self.warn_interval, "window_count": len(self.window_reports)}, f, indent=2, ensure_ascii=False)
        if self.window_reports:
            with open(windows_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.window_reports[0].keys()))
                writer.writeheader()
                writer.writerows(self.window_reports)
        return summary_path, windows_path if self.window_reports else None

    def _build_report(
        self,
        label,
        steps,
        arm_updates,
        arm_elapsed,
        arm_total_elapsed,
        computed_torque_updates,
        computed_torque_elapsed,
        mj_step_elapsed,
        other_elapsed,
        loop_elapsed,
        max_arm_elapsed,
        max_arm_total_elapsed,
        max_computed_torque_elapsed,
        max_mj_step_elapsed,
        max_other_elapsed,
        max_loop_elapsed,
        arm_overruns,
        arm_total_overruns,
        computed_torque_overruns,
        loop_overruns,
    ):
        budget_ms = self.step_budget * 1000.0
        arm_budget_ms = self.arm_budget * 1000.0
        arm_avg_ms = 0.0 if arm_updates == 0 else arm_elapsed / arm_updates * 1000.0
        arm_total_avg_ms = 0.0 if arm_updates == 0 else arm_total_elapsed / arm_updates * 1000.0
        computed_torque_avg_ms = 0.0 if computed_torque_updates == 0 else computed_torque_elapsed / computed_torque_updates * 1000.0
        return {
            "label": label,
            "steps": int(steps),
            "arm_updates": int(arm_updates),
            "computed_torque_updates": int(computed_torque_updates),
            "budget_ms": float(budget_ms),
            "arm_budget_ms": float(arm_budget_ms),
            "arm_avg_ms": float(arm_avg_ms),
            "arm_max_ms": float(max_arm_elapsed * 1000.0),
            "arm_overruns": int(arm_overruns),
            "arm_total_avg_ms": float(arm_total_avg_ms),
            "arm_total_max_ms": float(max_arm_total_elapsed * 1000.0),
            "arm_total_overruns": int(arm_total_overruns),
            "computed_torque_avg_ms": float(computed_torque_avg_ms),
            "computed_torque_max_ms": float(max_computed_torque_elapsed * 1000.0),
            "computed_torque_overruns": int(computed_torque_overruns),
            "mj_step_avg_ms": float(mj_step_elapsed / steps * 1000.0),
            "mj_step_max_ms": float(max_mj_step_elapsed * 1000.0),
            "other_avg_ms": float(other_elapsed / steps * 1000.0),
            "other_max_ms": float(max_other_elapsed * 1000.0),
            "loop_avg_ms": float(loop_elapsed / steps * 1000.0),
            "loop_max_ms": float(max_loop_elapsed * 1000.0),
            "loop_overruns": int(loop_overruns),
        }

    def _print_report(self, report):
        level = "WARN" if report["arm_overruns"] or report["arm_total_overruns"] or report["computed_torque_overruns"] or report["loop_overruns"] else "INFO"
        print(
            f"[{level}] {report['label']}: steps={report['steps']}, budget={report['budget_ms']:.2f} ms, arm_budget={report['arm_budget_ms']:.2f} ms, arm_updates={report['arm_updates']}, "
            f"arm policy avg/max={report['arm_avg_ms']:.2f}/{report['arm_max_ms']:.2f} ms, arm policy overruns={report['arm_overruns']}, "
            f"arm total avg/max={report['arm_total_avg_ms']:.2f}/{report['arm_total_max_ms']:.2f} ms, arm total overruns={report['arm_total_overruns']}, "
            f"computed torque updates={report['computed_torque_updates']}, avg/max={report['computed_torque_avg_ms']:.3f}/{report['computed_torque_max_ms']:.3f} ms, overruns={report['computed_torque_overruns']}, "
            f"mj_step avg/max={report['mj_step_avg_ms']:.2f}/{report['mj_step_max_ms']:.2f} ms, other avg/max={report['other_avg_ms']:.2f}/{report['other_max_ms']:.2f} ms, "
            f"loop avg/max={report['loop_avg_ms']:.2f}/{report['loop_max_ms']:.2f} ms, loop overruns={report['loop_overruns']}"
        )


def tilt_error_from_rot(rot, gravity_world=None):
    """兼容旧字段名，实际返回有方向的三维末端重力误差。"""
    gravity_world = np.array([0.0, 0.0, -9.81]) if gravity_world is None else np.asarray(gravity_world, dtype=np.float64)
    gravity_reference_end = np.array([0.0, 0.0, -np.linalg.norm(gravity_world)], dtype=np.float64)
    return np.asarray(rot, dtype=np.float64).T @ gravity_world - gravity_reference_end


def upright_alignment_from_rot(rot, gravity_world=None):
    """1 表示末端 z 轴正立，0 表示水平，-1 表示倒立。"""
    gravity_world = np.array([0.0, 0.0, -9.81]) if gravity_world is None else np.asarray(gravity_world, dtype=np.float64)
    world_up = -gravity_world / max(np.linalg.norm(gravity_world), 1e-12)
    return float(np.dot(np.asarray(rot, dtype=np.float64)[:, 2], world_up))


def quat_to_yaw_wxyz(quaternion):
    qw, qx, qy, qz = quaternion
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


# ==============================
# 非核心代码：调试可视化、评估与实验保存
# 这部分对复现实验很重要，但不属于控制器本体逻辑。
# ==============================
def print_model_mappings(model):
    print("=" * 50)
    print("关节 (Joints - 对应 qpos/qvel):")
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    for i, name in enumerate(joint_names):
        print(f"  Joint ID: {i:2d}, Name: {name}")

    print("\n驱动器 (Actuators - 对应 ctrl):")
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    for i, name in enumerate(actuator_names):
        print(f"  Actuator ID: {i:2d}, Name: {name}")
    print("=" * 50)


def resolve_scene_ids(model):
    return SceneIds(
        torso_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"),
        imu_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso"),
        torso_acc_sensor_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu-torso-linear-acceleration"),
        left_grasp_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site"),
        right_grasp_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site"),
    )


def add_axis_visual(scene, pos, rot, sphere_radius=0.02, axis_length=0.20, axis_radius=0.008, origin_rgba=None):
    if origin_rgba is None:
        origin_rgba = np.array([1.0, 1.0, 0.0, 0.9])

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([sphere_radius, 0.0, 0.0]),
        pos,
        np.eye(3).reshape(-1),
        origin_rgba,
    )
    scene.ngeom += 1

    axis_colors = [
        np.array([1.0, 0.0, 0.0, 0.9]),
        np.array([0.0, 1.0, 0.0, 0.9]),
        np.array([0.0, 0.0, 1.0, 0.9]),
    ]
    for i in range(3):
        end = pos + rot[:, i] * axis_length
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            axis_colors[i],
        )
        mujoco.mjv_connector(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, axis_radius, pos, end)
        scene.ngeom += 1


def draw_debug_axes(scene, data, scene_ids):
    scene.ngeom = 0

    add_axis_visual(
        scene,
        np.array([0.0, 0.0, 0.0]),
        np.eye(3),
        sphere_radius=0.025,
        axis_length=0.25,
        axis_radius=0.010,
        origin_rgba=np.array([1.0, 1.0, 1.0, 0.95]),
    )

    imu_pos = data.site_xpos[scene_ids.imu_site_id].copy()
    imu_rot = data.site_xmat[scene_ids.imu_site_id].reshape(3, 3).copy()
    add_axis_visual(scene, imu_pos, imu_rot, sphere_radius=0.02, axis_length=0.20, axis_radius=0.008)

    left_pos = data.site_xpos[scene_ids.left_grasp_site_id].copy()
    left_rot = data.site_xmat[scene_ids.left_grasp_site_id].reshape(3, 3).copy()
    add_axis_visual(
        scene,
        left_pos,
        left_rot,
        sphere_radius=0.015,
        axis_length=0.08,
        axis_radius=0.006,
        origin_rgba=np.array([1.0, 0.5, 0.0, 0.9]),
    )

    right_pos = data.site_xpos[scene_ids.right_grasp_site_id].copy()
    right_rot = data.site_xmat[scene_ids.right_grasp_site_id].reshape(3, 3).copy()
    add_axis_visual(
        scene,
        right_pos,
        right_rot,
        sphere_radius=0.015,
        axis_length=0.08,
        axis_radius=0.006,
        origin_rgba=np.array([0.0, 1.0, 1.0, 0.9]),
    )


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def create_eval_run_dir(base_dir, experiment_name, run_metadata):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, experiment_name, timestamp)
    os.makedirs(run_dir, exist_ok=False)
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(_to_serializable(run_metadata), f, indent=2, ensure_ascii=False)
    return run_dir


def build_run_metadata(config_file, experiment_name, policy_type, controller_notes, controller_meta, cmd_nominal, simulation_dt, gait_period, warmup_cycles, evaluation_cycles, cooldown_cycles):
    return {
        "config_file": config_file,
        "experiment_name": experiment_name,
        "policy_type": policy_type,
        "right_arm_joint_names": list(RIGHT_ARM_JOINT_NAMES),
        "notes": controller_notes,
        "cmd_nominal": cmd_nominal,
        "simulation_dt": simulation_dt,
        "gait_period": gait_period,
        "warmup_cycles": warmup_cycles,
        "evaluation_cycles": evaluation_cycles,
        "cooldown_cycles": cooldown_cycles,
        **controller_meta,
    }


def init_eval_buffers():
    return EvalBuffers(
        eval_data={
            "time": [],
            "torso_yaw": [],
            "left_ee_lin_acc_world": [],
            "left_ee_ang_acc_world": [],
            "left_ee_tilt_error": [],
            "left_ee_upright_alignment": [],
            "right_ee_lin_acc_world": [],
            "right_ee_ang_acc_world": [],
            "right_ee_tilt_error": [],
            "right_ee_upright_alignment": [],
        },
        trajectory_data={
            "time": [],
            "qpos": [],
            "qvel": [],
            "qacc": [],
            "ctrl": [],
            "right_arm_q": [],
            "right_arm_dq": [],
            "right_arm_qacc": [],
            "right_arm_ctrl": [],
            "right_arm_target_q": [],
            "right_arm_target_dq": [],
            "right_arm_ddq_raw": [],
            "right_arm_ddq_des": [],
            "right_arm_ddq_saturation_limit": [],
            "right_arm_ddq_saturation_mask": [],
            "right_arm_tau_inverse": [],
            # 兼容旧字段：从本版开始 tau_constraint 与 tau_contact 含义相同，均仅表示 contact 分量。
            "right_arm_tau_constraint": [],
            "right_arm_tau_contact": [],
            "right_arm_tau_constraint_total": [],
            "right_arm_tau_constraint_noncontact": [],
            "right_arm_tau_constraint_nonfriction": [],
            "right_arm_tau_constraint_friction": [],
            "right_arm_tau_ff": [],
            "right_arm_tau_pd": [],
            "right_arm_tau_nominal": [],
            "right_arm_tau_mapping_correction_raw": [],
            "right_arm_tau_mapping_correction": [],
            "right_arm_tau_cmd_raw": [],
            "right_arm_tau_limit_lower": [],
            "right_arm_tau_limit_upper": [],
            "right_arm_tau_saturation_mask": [],
            "right_arm_actual_qfrc_bias": [],
            "right_arm_actual_qfrc_passive": [],
            "right_arm_actual_qfrc_constraint": [],
            "right_arm_qacc_mapping_baseline": [],
            "right_arm_qacc_mapping_predicted": [],
            "right_arm_qacc_mapping_prediction_error": [],
            "right_arm_qacc_mapping_validated": [],
            "right_arm_qacc_mapping_validation_error": [],
            "right_arm_qacc_mapping_linearization_error": [],
            "right_arm_qacc_mapping_model_error": [],
            "right_arm_forward_dynamics_gain": [],
            "right_arm_forward_dynamics_singular_values": [],
            "right_arm_forward_dynamics_condition_number": [],
            "right_arm_forward_dynamics_validation_scale": [],
            "right_arm_forward_dynamics_validation_attempts": [],
            "right_arm_forward_dynamics_validation_improved": [],
            "right_arm_forward_dynamics_tracking_safety_satisfied": [],
            "right_arm_forward_dynamics_qacc_safety_satisfied": [],
            "right_arm_forward_dynamics_safe_candidate_count": [],
            "right_arm_forward_dynamics_total_error_rejections": [],
            "right_arm_forward_dynamics_joint_error_rejections": [],
            "right_arm_forward_dynamics_qacc_limit_rejections": [],
            "right_arm_first_pass_qacc_validated": [],
            "right_arm_first_pass_qacc_validation_error": [],
            "right_arm_forward_dynamics_second_pass_triggered": [],
            "right_arm_forward_dynamics_second_pass_accepted": [],
            "right_arm_second_pass_tracking_safety_satisfied": [],
            "right_arm_second_pass_qacc_safety_satisfied": [],
            "right_arm_second_pass_tau_correction_raw": [],
            "right_arm_second_pass_tau_correction": [],
            "right_arm_second_pass_qacc_predicted": [],
            "right_arm_second_pass_qacc_validated": [],
            "right_arm_second_pass_qacc_validation_error": [],
            "right_arm_second_pass_qacc_linearization_error": [],
            "right_arm_second_pass_forward_dynamics_gain": [],
            "right_arm_second_pass_singular_values": [],
            "right_arm_second_pass_condition_number": [],
            "right_arm_second_pass_validation_scale": [],
            "right_arm_second_pass_validation_attempts": [],
            "right_arm_second_pass_safe_candidate_count": [],
            "right_arm_second_pass_total_error_rejections": [],
            "right_arm_second_pass_joint_error_rejections": [],
            "right_arm_second_pass_qacc_limit_rejections": [],
            "right_arm_forward_dynamics_safety_fallback_used": [],
            "right_arm_forward_dynamics_safety_fallback_satisfied": [],
            "right_arm_forward_dynamics_safety_fallback_attempts": [],
            "torso_lin_vel_world": [],
            "torso_ang_vel_world": [],
            "torso_acc_world_raw": [],
            "torso_acc_world_used": [],
            "torso_alpha_world_raw": [],
            "torso_alpha_world_used": [],
            "heading_reference_world": [],
            "heading_yaw_unwrapped": [],
            "heading_yaw_filtered": [],
            "heading_yaw_error": [],
            "heading_yaw_rate_filtered": [],
            "heading_yaw_rate_correction": [],
            "heading_yaw_rate_command": [],
            "heading_command_saturated": [],
            "right_ee_lin_vel_world": [],
            "right_ee_ang_vel_world": [],
            "right_ee_position_torso": [],
            "right_ee_position_reference_torso": [],
            "right_ee_position_error_torso": [],
            "right_ee_gravity_error_end": [],
            "right_ee_upright_alignment": [],
            "right_lqr_one_step_q_model": [],
            "right_lqr_one_step_dq_model": [],
            "right_lqr_one_step_ee_lin_acc_model": [],
            "right_lqr_one_step_ee_ang_acc_model": [],
            "right_lqr_one_step_position_error_model": [],
            "right_lqr_one_step_gravity_error_model": [],
            "right_lqr_one_step_cost_model": [],
            "right_lqr_one_step_prediction_valid": [],
            "arm_policy_updated": [],
            "contact_count": [],
        },
        prev_left_lin_vel=np.zeros(3),
        prev_left_ang_vel=np.zeros(3),
        prev_right_lin_vel=np.zeros(3),
        prev_right_ang_vel=np.zeros(3),
        prev_torso_lin_vel=np.zeros(3),
        prev_torso_ang_vel=np.zeros(3),
    )


def make_video_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([2.2, 0.0, 0.9])
    cam.distance = 5.2
    cam.azimuth = 120.0
    cam.elevation = -25.0
    return cam


def make_video_renderer(model, preferred_width=1280, preferred_height=720):
    if imageio is None:
        return None, None, None

    vis_global = getattr(model.vis, "global_", None)
    if vis_global is not None:
        try:
            vis_global.offwidth = max(int(vis_global.offwidth), preferred_width)
            vis_global.offheight = max(int(vis_global.offheight), preferred_height)
        except Exception:
            pass

    offwidth = int(getattr(vis_global, "offwidth", preferred_width))
    offheight = int(getattr(vis_global, "offheight", preferred_height))
    width = min(preferred_width, offwidth)
    height = min(preferred_height, offheight)

    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        return renderer, width, height
    except Exception as exc:
        print(f"[video] Renderer 初始化失败，已跳过视频保存: {exc}")
        return None, width, height


def add_lqr_tracking_trajectory_data(trajectory_data, simulation_dt, cost_definition):
    """把相邻两次手臂更新之间的真实响应与一步预测对齐到前一次更新时间。"""
    time_values = np.asarray(trajectory_data.get("time", []), dtype=np.float64)
    sample_count = len(time_values)
    joint_count = len(RIGHT_ARM_JOINT_NAMES)
    cost_count = len(LQR_COST_TERM_NAMES)

    def empty(width):
        return np.full((sample_count, width), np.nan, dtype=np.float64)

    derived = {
        "right_arm_ddq_real": empty(joint_count),
        "right_arm_ddq_tracking_error": empty(joint_count),
        "right_lqr_one_step_q_actual": empty(joint_count),
        "right_lqr_one_step_dq_actual": empty(joint_count),
        "right_lqr_one_step_ee_lin_acc_actual": empty(3),
        "right_lqr_one_step_ee_ang_acc_actual": empty(3),
        "right_lqr_one_step_position_error_actual": empty(3),
        "right_lqr_one_step_gravity_error_actual": empty(3),
        "right_lqr_one_step_cost_actual": empty(cost_count),
        "right_lqr_one_step_cost_error": empty(cost_count),
        "right_lqr_tracking_interval_dt": np.full(sample_count, np.nan, dtype=np.float64),
        "right_lqr_tracking_valid": np.zeros(sample_count, dtype=bool),
    }
    if sample_count == 0 or cost_definition is None:
        trajectory_data.update(derived)
        return

    prediction_valid = np.asarray(
        trajectory_data.get("right_lqr_one_step_prediction_valid", []),
        dtype=bool,
    )
    right_dq = np.asarray(trajectory_data.get("right_arm_dq", []), dtype=np.float64)
    right_q = np.asarray(trajectory_data.get("right_arm_q", []), dtype=np.float64)
    ddq_des = np.asarray(trajectory_data.get("right_arm_ddq_des", []), dtype=np.float64)
    ee_lin_vel = np.asarray(trajectory_data.get("right_ee_lin_vel_world", []), dtype=np.float64)
    ee_ang_vel = np.asarray(trajectory_data.get("right_ee_ang_vel_world", []), dtype=np.float64)
    position_error = np.asarray(trajectory_data.get("right_ee_position_error_torso", []), dtype=np.float64)
    gravity_error = np.asarray(trajectory_data.get("right_ee_gravity_error_end", []), dtype=np.float64)
    model_cost = np.asarray(trajectory_data.get("right_lqr_one_step_cost_model", []), dtype=np.float64)
    expected_shapes = (
        prediction_valid.shape == (sample_count,),
        right_dq.shape == (sample_count, joint_count),
        right_q.shape == (sample_count, joint_count),
        ddq_des.shape == (sample_count, joint_count),
        ee_lin_vel.shape == (sample_count, 3),
        ee_ang_vel.shape == (sample_count, 3),
        position_error.shape == (sample_count, 3),
        gravity_error.shape == (sample_count, 3),
        model_cost.shape == (sample_count, cost_count),
    )
    if not all(expected_shapes):
        trajectory_data.update(derived)
        return

    Qa = np.asarray(cost_definition["Qa"], dtype=np.float64)
    Qalpha = np.asarray(cost_definition["Qalpha"], dtype=np.float64)
    Qp = np.asarray(cost_definition["Qp"], dtype=np.float64)
    Qg = np.asarray(cost_definition["Qg"], dtype=np.float64)
    Qq = np.asarray(cost_definition["Qq"], dtype=np.float64)
    Qv = np.asarray(cost_definition["Qv"], dtype=np.float64)
    R = np.asarray(cost_definition["R"], dtype=np.float64)
    posture_reference = np.asarray(cost_definition["posture_reference"], dtype=np.float64)

    update_indices = np.flatnonzero(prediction_valid)
    for start_index, next_index in zip(update_indices[:-1], update_indices[1:]):
        before_index = start_index - 1
        end_index = next_index - 1
        interval_dt = (next_index - start_index) * float(simulation_dt)
        if before_index < 0 or end_index <= before_index or interval_dt <= 0.0:
            continue

        ddq_real = (right_dq[end_index] - right_dq[before_index]) / interval_dt
        ee_lin_acc_real = (ee_lin_vel[end_index] - ee_lin_vel[before_index]) / interval_dt
        ee_ang_acc_real = (ee_ang_vel[end_index] - ee_ang_vel[before_index]) / interval_dt
        q_actual = right_q[end_index]
        dq_actual = right_dq[end_index]
        position_actual = position_error[end_index]
        gravity_actual = gravity_error[end_index]
        posture_error = q_actual - posture_reference
        control = ddq_des[start_index]
        actual_cost = np.array(
            [
                ee_lin_acc_real @ Qa @ ee_lin_acc_real,
                ee_ang_acc_real @ Qalpha @ ee_ang_acc_real,
                position_actual @ Qp @ position_actual,
                gravity_actual @ Qg @ gravity_actual,
                posture_error @ Qq @ posture_error,
                dq_actual @ Qv @ dq_actual,
                control @ R @ control,
            ],
            dtype=np.float64,
        )

        derived["right_arm_ddq_real"][start_index] = ddq_real
        derived["right_arm_ddq_tracking_error"][start_index] = ddq_real - control
        derived["right_lqr_one_step_q_actual"][start_index] = q_actual
        derived["right_lqr_one_step_dq_actual"][start_index] = dq_actual
        derived["right_lqr_one_step_ee_lin_acc_actual"][start_index] = ee_lin_acc_real
        derived["right_lqr_one_step_ee_ang_acc_actual"][start_index] = ee_ang_acc_real
        derived["right_lqr_one_step_position_error_actual"][start_index] = position_actual
        derived["right_lqr_one_step_gravity_error_actual"][start_index] = gravity_actual
        derived["right_lqr_one_step_cost_actual"][start_index] = actual_cost
        derived["right_lqr_one_step_cost_error"][start_index] = actual_cost - model_cost[start_index]
        derived["right_lqr_tracking_interval_dt"][start_index] = interval_dt
        derived["right_lqr_tracking_valid"][start_index] = (
            np.all(np.isfinite(model_cost[start_index]))
            and np.all(np.isfinite(actual_cost))
            and np.all(np.isfinite(ddq_real))
        )

    trajectory_data.update(derived)


def _component_tracking_metrics(reference, actual):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.ndim != 2 or actual.shape != reference.shape or reference.shape[0] == 0:
        width = reference.shape[1] if reference.ndim == 2 else 0
        return {name: np.zeros(width).tolist() for name in (
            "reference_rms", "actual_rms", "rmse", "mae", "abs_max", "normalized_rmse", "correlation", "gain"
        )}
    error = actual - reference
    reference_rms = np.sqrt(np.mean(reference ** 2, axis=0))
    actual_rms = np.sqrt(np.mean(actual ** 2, axis=0))
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    correlation = np.zeros(reference.shape[1], dtype=np.float64)
    gain = np.zeros(reference.shape[1], dtype=np.float64)
    for component in range(reference.shape[1]):
        ref_component = reference[:, component]
        actual_component = actual[:, component]
        if np.std(ref_component) > 1e-12 and np.std(actual_component) > 1e-12:
            correlation[component] = np.corrcoef(ref_component, actual_component)[0, 1]
        denominator = ref_component @ ref_component
        if denominator > 1e-12:
            gain[component] = (ref_component @ actual_component) / denominator
    return {
        "reference_rms": reference_rms.tolist(),
        "actual_rms": actual_rms.tolist(),
        "rmse": rmse.tolist(),
        "mae": np.mean(np.abs(error), axis=0).tolist(),
        "abs_max": np.max(np.abs(error), axis=0).tolist(),
        "normalized_rmse": (rmse / np.maximum(reference_rms, 1e-12)).tolist(),
        "correlation": correlation.tolist(),
        "gain": gain.tolist(),
    }


def _validation_scale_summary(scale_values, selection_mask):
    scale_values = np.asarray(scale_values, dtype=np.float64)
    selection_mask = np.asarray(selection_mask, dtype=bool)
    if scale_values.shape != selection_mask.shape:
        scale_values = np.zeros_like(selection_mask, dtype=np.float64)
        selection_mask = np.zeros_like(selection_mask, dtype=bool)
    selected = scale_values[selection_mask]
    labels = ("first", "second", "third", "fourth", "fallback")
    scales = (1.0, 0.5, 0.25, 0.125, 0.0)
    total = int(selected.size)
    selections = []
    for index, (label, scale) in enumerate(zip(labels, scales), start=1):
        count = int(np.sum(np.isclose(selected, scale)))
        selections.append({
            "candidate_index": index if scale > 0.0 else 0,
            "label": label,
            "scale": scale,
            "count": count,
            "fraction": count / float(total) if total else 0.0,
        })
    return {"sample_count": total, "selections": selections}


def _validation_safety_summary(trajectory_data, prefix, selection_mask):
    selection_mask = np.asarray(selection_mask, dtype=bool)
    field_names = (
        "safe_candidate_count",
        "total_error_rejections",
        "joint_error_rejections",
        "qacc_limit_rejections",
    )
    summary = {}
    for field_name in field_names:
        values = np.asarray(
            trajectory_data.get(f"{prefix}_{field_name}", []),
            dtype=np.float64,
        )
        selected = values[selection_mask] if values.shape == selection_mask.shape else np.zeros(0)
        summary[field_name] = {
            "total": int(np.sum(selected)) if selected.size else 0,
            "mean_per_pass": float(np.mean(selected)) if selected.size else 0.0,
        }
        if field_name == "safe_candidate_count":
            summary[field_name]["zero_candidate_passes"] = int(np.sum(selected == 0))
            summary[field_name]["zero_candidate_fraction"] = (
                float(np.mean(selected == 0)) if selected.size else 0.0
            )
    return summary


def _masked_vector_norm_stats(values, mask, width=5):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != (len(mask), width):
        return {"sample_count": 0, "mean": 0.0, "rms": 0.0, "p95": 0.0, "max": 0.0}
    norms = np.linalg.norm(values[mask], axis=1)
    if not norms.size:
        return {"sample_count": 0, "mean": 0.0, "rms": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "sample_count": int(norms.size),
        "mean": float(np.mean(norms)),
        "rms": float(np.sqrt(np.mean(norms ** 2))),
        "p95": float(np.percentile(norms, 95.0)),
        "max": float(np.max(norms)),
    }


def compute_lqr_tracking_diagnostics(trajectory_data, eval_start_time, eval_end_time):
    time_values = np.asarray(trajectory_data.get("time", []), dtype=np.float64)
    valid = np.asarray(trajectory_data.get("right_lqr_tracking_valid", []), dtype=bool).copy()
    if valid.shape != time_values.shape:
        valid = np.zeros_like(time_values, dtype=bool)
    interval_dt = np.asarray(trajectory_data.get("right_lqr_tracking_interval_dt", []), dtype=np.float64)
    if interval_dt.shape != time_values.shape:
        valid = np.zeros_like(time_values, dtype=bool)
        interval_dt = np.full_like(time_values, np.nan)
    valid &= (time_values >= eval_start_time) & (time_values + interval_dt <= eval_end_time + 1e-12)
    ddq_des = np.asarray(trajectory_data.get("right_arm_ddq_des", []), dtype=np.float64)
    ddq_real = np.asarray(trajectory_data.get("right_arm_ddq_real", []), dtype=np.float64)
    model_cost = np.asarray(trajectory_data.get("right_lqr_one_step_cost_model", []), dtype=np.float64)
    actual_cost = np.asarray(trajectory_data.get("right_lqr_one_step_cost_actual", []), dtype=np.float64)
    sample_count = int(np.sum(valid))
    diagnostics = {
        "definition": {
            "alignment": "one arm-control interval: prediction at update k versus response immediately before update k+1",
            "ddq_real": "(dq[k+1] - dq[k]) / arm_control_interval",
            "model_cost": "one-step model acceleration and predicted end-of-interval state",
            "actual_cost": "interval-average end-effector acceleration, measured end-of-interval state, and commanded ddq control cost",
            "cost_error": "actual_cost - one_step_model_cost",
            "evaluation_window": [float(eval_start_time), float(eval_end_time)],
        },
        "sample_count": sample_count,
        "joint_names": list(RIGHT_ARM_JOINT_NAMES),
        "cost_term_names": list(LQR_COST_TERM_NAMES),
        "interval_dt_mean": float(np.mean(interval_dt[valid])) if sample_count else 0.0,
    }
    eval_step_mask = (time_values >= eval_start_time) & (time_values < eval_end_time)
    first_scale = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_validation_scale", []),
        dtype=np.float64,
    )
    second_triggered = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_second_pass_triggered", []),
        dtype=bool,
    )
    second_accepted = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_second_pass_accepted", []),
        dtype=bool,
    )
    second_scale = np.asarray(
        trajectory_data.get("right_arm_second_pass_validation_scale", []),
        dtype=np.float64,
    )
    first_tracking_safe = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_tracking_safety_satisfied", []),
        dtype=bool,
    )
    first_qacc_safe = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_qacc_safety_satisfied", []),
        dtype=bool,
    )
    second_tracking_safe = np.asarray(
        trajectory_data.get("right_arm_second_pass_tracking_safety_satisfied", []),
        dtype=bool,
    )
    second_qacc_safe = np.asarray(
        trajectory_data.get("right_arm_second_pass_qacc_safety_satisfied", []),
        dtype=bool,
    )
    safety_fallback_used = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_safety_fallback_used", []),
        dtype=bool,
    )
    safety_fallback_satisfied = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_safety_fallback_satisfied", []),
        dtype=bool,
    )
    safety_fallback_attempts = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_safety_fallback_attempts", []),
        dtype=np.int64,
    )
    if second_triggered.shape != time_values.shape:
        second_triggered = np.zeros_like(time_values, dtype=bool)
    if second_accepted.shape != time_values.shape:
        second_accepted = np.zeros_like(time_values, dtype=bool)
    second_eval_mask = eval_step_mask & second_triggered
    first_pass_validation = _validation_scale_summary(first_scale, eval_step_mask)
    second_pass_validation = _validation_scale_summary(second_scale, second_eval_mask)
    first_pass_safety = _validation_safety_summary(
        trajectory_data,
        "right_arm_forward_dynamics",
        eval_step_mask,
    )
    second_pass_safety = _validation_safety_summary(
        trajectory_data,
        "right_arm_second_pass",
        second_eval_mask,
    )
    eval_step_count = int(np.sum(eval_step_mask))
    triggered_count = int(np.sum(second_eval_mask))
    accepted_count = int(np.sum(eval_step_mask & second_accepted))
    def masked_fraction(values, mask):
        return float(np.mean(values[mask])) if values.shape == mask.shape and np.any(mask) else 0.0

    diagnostics["forward_dynamics_validation"] = {
        "definition": "all candidate scales are evaluated; select the minimum-total-error candidate that improves the baseline and satisfies per-joint-error/qacc safety limits",
        "first_pass": {
            **first_pass_validation,
            "safety": first_pass_safety,
            "tracking_safety_satisfied_fraction": masked_fraction(first_tracking_safe, eval_step_mask),
            "qacc_safety_satisfied_fraction": masked_fraction(first_qacc_safe, eval_step_mask),
        },
        "second_pass": {
            **second_pass_validation,
            "safety": second_pass_safety,
            "tracking_safety_satisfied_fraction": masked_fraction(second_tracking_safe, second_eval_mask),
            "qacc_safety_satisfied_fraction": masked_fraction(second_qacc_safe, second_eval_mask),
            "triggered_count": triggered_count,
            "triggered_fraction_of_evaluation_steps": (
                triggered_count / float(eval_step_count) if eval_step_count else 0.0
            ),
            "accepted_count": accepted_count,
            "accepted_fraction_given_triggered": (
                accepted_count / float(triggered_count) if triggered_count else 0.0
            ),
        },
        "final_safety_fallback": {
            "used_count": int(np.sum(safety_fallback_used[eval_step_mask]))
            if safety_fallback_used.shape == eval_step_mask.shape
            else 0,
            "used_fraction": masked_fraction(safety_fallback_used, eval_step_mask),
            "satisfied_fraction_when_used": (
                masked_fraction(safety_fallback_satisfied, eval_step_mask & safety_fallback_used)
                if safety_fallback_used.shape == eval_step_mask.shape
                else 0.0
            ),
            "attempt_count": int(np.sum(safety_fallback_attempts[eval_step_mask]))
            if safety_fallback_attempts.shape == eval_step_mask.shape
            else 0,
        },
        "first_pass_residual_norm": _masked_vector_norm_stats(
            trajectory_data.get("right_arm_first_pass_qacc_validation_error", []),
            eval_step_mask,
        ),
        "final_residual_norm": _masked_vector_norm_stats(
            trajectory_data.get("right_arm_qacc_mapping_validation_error", []),
            eval_step_mask,
        ),
    }
    if (
        sample_count == 0
        or ddq_des.shape != ddq_real.shape
        or ddq_des.shape[0] != len(time_values)
        or model_cost.shape != actual_cost.shape
        or model_cost.shape[0] != len(time_values)
    ):
        diagnostics["ddq_tracking"] = _component_tracking_metrics(np.zeros((0, 5)), np.zeros((0, 5)))
        diagnostics["cost_tracking"] = _component_tracking_metrics(np.zeros((0, 7)), np.zeros((0, 7)))
        return diagnostics

    ddq_metrics = _component_tracking_metrics(ddq_des[valid], ddq_real[valid])
    ddq_error_norm = np.linalg.norm(ddq_real[valid] - ddq_des[valid], axis=1)
    ddq_metrics["error_norm_rms"] = float(np.sqrt(np.mean(ddq_error_norm ** 2)))
    ddq_metrics["error_norm_p95"] = float(np.percentile(ddq_error_norm, 95.0))
    ddq_metrics["error_norm_max"] = float(np.max(ddq_error_norm))
    diagnostics["ddq_tracking"] = ddq_metrics

    cost_metrics = _component_tracking_metrics(model_cost[valid], actual_cost[valid])
    model_total = np.sum(model_cost[valid], axis=1, keepdims=True)
    actual_total = np.sum(actual_cost[valid], axis=1, keepdims=True)
    cost_metrics["total"] = _component_tracking_metrics(model_total, actual_total)
    diagnostics["cost_tracking"] = cost_metrics
    return diagnostics


def compute_right_arm_trajectory_diagnostics(trajectory_data):
    n = len(RIGHT_ARM_JOINT_NAMES)
    def matrix(name, width):
        value = np.asarray(trajectory_data.get(name, []), dtype=np.float64)
        return value if value.ndim == 2 and value.shape[1] == width else np.zeros((0, width), dtype=np.float64)

    def component_stats(result, prefix, value, width):
        valid = value.ndim == 2 and value.shape[0] > 0 and value.shape[1] == width
        result[f"{prefix}_rms"] = np.sqrt(np.mean(value ** 2, axis=0)) if valid else np.zeros(width)
        result[f"{prefix}_abs_max"] = np.max(np.abs(value), axis=0) if valid else np.zeros(width)

    ddq_raw = matrix("right_arm_ddq_raw", n)
    ddq_des = matrix("right_arm_ddq_des", n)
    ddq_mask = np.asarray(trajectory_data.get("right_arm_ddq_saturation_mask", []), dtype=bool)
    qacc = matrix("right_arm_qacc", n)
    ctrl = matrix("right_arm_ctrl", n)
    ddq_limits = np.asarray(trajectory_data.get("right_arm_ddq_saturation_limit", []), dtype=np.float64)
    tau_inverse = matrix("right_arm_tau_inverse", n)
    tau_constraint = matrix("right_arm_tau_constraint", n)
    tau_contact = matrix("right_arm_tau_contact", n)
    tau_constraint_total = matrix("right_arm_tau_constraint_total", n)
    tau_constraint_noncontact = matrix("right_arm_tau_constraint_noncontact", n)
    tau_constraint_nonfriction = matrix("right_arm_tau_constraint_nonfriction", n)
    tau_constraint_friction = matrix("right_arm_tau_constraint_friction", n)
    tau_ff = matrix("right_arm_tau_ff", n)
    tau_pd = matrix("right_arm_tau_pd", n)
    tau_nominal = matrix("right_arm_tau_nominal", n)
    tau_mapping_correction_raw = matrix("right_arm_tau_mapping_correction_raw", n)
    tau_mapping_correction = matrix("right_arm_tau_mapping_correction", n)
    qacc_mapping_baseline = matrix("right_arm_qacc_mapping_baseline", n)
    qacc_mapping_predicted = matrix("right_arm_qacc_mapping_predicted", n)
    qacc_mapping_prediction_error = matrix("right_arm_qacc_mapping_prediction_error", n)
    qacc_mapping_validated = matrix("right_arm_qacc_mapping_validated", n)
    qacc_mapping_validation_error = matrix("right_arm_qacc_mapping_validation_error", n)
    qacc_mapping_linearization_error = matrix("right_arm_qacc_mapping_linearization_error", n)
    qacc_mapping_model_error = matrix("right_arm_qacc_mapping_model_error", n)
    forward_dynamics_singular_values = matrix("right_arm_forward_dynamics_singular_values", n)
    tau_raw = matrix("right_arm_tau_cmd_raw", n)
    tau_low = matrix("right_arm_tau_limit_lower", n)
    tau_high = matrix("right_arm_tau_limit_upper", n)
    tau_mask = np.asarray(trajectory_data.get("right_arm_tau_saturation_mask", []), dtype=bool)
    if ddq_mask.shape != ddq_des.shape:
        ddq_mask = np.zeros_like(ddq_des, dtype=bool)
    if tau_raw.shape != ctrl.shape:
        tau_raw = np.zeros_like(ctrl)
    if tau_low.shape != ctrl.shape:
        tau_low = np.full_like(ctrl, -np.inf)
    if tau_high.shape != ctrl.shape:
        tau_high = np.full_like(ctrl, np.inf)
    if tau_mask.shape != ctrl.shape:
        tau_mask = (tau_raw < (tau_low + RIGHT_ARM_TAU_SATURATION_EPS)) | (tau_raw > (tau_high - RIGHT_ARM_TAU_SATURATION_EPS))
    ddq_n = int(ddq_des.shape[0])
    tau_n = int(ctrl.shape[0])
    ddq_count = ddq_mask.sum(axis=0).astype(np.int64)
    ddq_frac = np.zeros(n, dtype=np.float64) if ddq_n == 0 else ddq_count / float(ddq_n)
    tau_count = tau_mask.sum(axis=0).astype(np.int64)
    tau_frac = np.zeros(n, dtype=np.float64) if tau_n == 0 else tau_count / float(tau_n)
    finite_limits = ddq_limits[np.isfinite(ddq_limits)]
    ddq_limit = float(finite_limits[-1]) if finite_limits.size else np.inf
    ddq_thr = ddq_limit - RIGHT_ARM_DDQ_SATURATION_EPS if np.isfinite(ddq_limit) else np.inf
    tau_low_last = tau_low[-1].copy() if tau_n > 0 else np.full(n, -np.inf)
    tau_high_last = tau_high[-1].copy() if tau_n > 0 else np.full(n, np.inf)
    diagnostics = {
        "right_arm_joint_names": np.asarray(RIGHT_ARM_JOINT_NAMES),
        "right_arm_ddq_saturation_limit": np.array(ddq_limit),
        "right_arm_ddq_saturation_threshold": np.array(ddq_thr),
        "right_arm_ddq_saturation_count": ddq_count,
        "right_arm_ddq_saturation_fraction": ddq_frac,
        "right_arm_ddq_saturation_percent": ddq_frac * 100.0,
        "right_arm_ddq_saturation_any_fraction": np.array(float(np.mean(np.any(ddq_mask, axis=1))) if ddq_n > 0 else 0.0),
        "right_arm_tau_limit_lower": tau_low_last,
        "right_arm_tau_limit_upper": tau_high_last,
        "right_arm_tau_saturation_count": tau_count,
        "right_arm_tau_saturation_fraction": tau_frac,
        "right_arm_tau_saturation_percent": tau_frac * 100.0,
        "right_arm_tau_saturation_any_fraction": np.array(float(np.mean(np.any(tau_mask, axis=1))) if tau_n > 0 else 0.0),
        "right_arm_tau_clip_delta_abs_max": np.max(np.abs(ctrl - tau_raw), axis=0) if tau_n > 0 else np.zeros(n),
    }
    for prefix, value in [
        ("right_arm_ddq_raw", ddq_raw),
        ("right_arm_ddq", ddq_des),
        ("right_arm_qacc", qacc),
        ("right_arm_tau_inverse", tau_inverse),
        ("right_arm_tau_constraint", tau_constraint),
        ("right_arm_tau_contact", tau_contact),
        ("right_arm_tau_constraint_total", tau_constraint_total),
        ("right_arm_tau_constraint_noncontact", tau_constraint_noncontact),
        ("right_arm_tau_constraint_nonfriction", tau_constraint_nonfriction),
        ("right_arm_tau_constraint_friction", tau_constraint_friction),
        ("right_arm_tau_ff", tau_ff),
        ("right_arm_tau_pd", tau_pd),
        ("right_arm_tau_nominal", tau_nominal),
        ("right_arm_tau_mapping_correction_raw", tau_mapping_correction_raw),
        ("right_arm_tau_mapping_correction", tau_mapping_correction),
        ("right_arm_tau_raw", tau_raw),
        ("right_arm_ctrl", ctrl),
        ("right_arm_qacc_mapping_baseline", qacc_mapping_baseline),
        ("right_arm_qacc_mapping_predicted", qacc_mapping_predicted),
        ("right_arm_qacc_mapping_prediction_error", qacc_mapping_prediction_error),
        ("right_arm_qacc_mapping_validated", qacc_mapping_validated),
        ("right_arm_qacc_mapping_validation_error", qacc_mapping_validation_error),
        ("right_arm_qacc_mapping_linearization_error", qacc_mapping_linearization_error),
        ("right_arm_qacc_mapping_model_error", qacc_mapping_model_error),
        ("right_arm_forward_dynamics_singular_values", forward_dynamics_singular_values),
    ]:
        component_stats(diagnostics, prefix, value, n)
    ddq_postprocess_delta = ddq_des - ddq_raw if ddq_des.shape == ddq_raw.shape else np.zeros_like(ddq_des)
    component_stats(diagnostics, "right_arm_ddq_postprocess_delta", ddq_postprocess_delta, n)
    mapping_error_norm = np.linalg.norm(qacc_mapping_prediction_error, axis=1) if qacc_mapping_prediction_error.shape[0] else np.zeros(0)
    diagnostics["right_arm_qacc_mapping_prediction_error_norm_rms"] = np.array(
        np.sqrt(np.mean(mapping_error_norm ** 2)) if mapping_error_norm.size else 0.0
    )
    diagnostics["right_arm_qacc_mapping_prediction_error_norm_max"] = np.array(
        np.max(mapping_error_norm) if mapping_error_norm.size else 0.0
    )
    validation_error_norm = np.linalg.norm(qacc_mapping_validation_error, axis=1) if qacc_mapping_validation_error.shape[0] else np.zeros(0)
    diagnostics["right_arm_qacc_mapping_validation_error_norm_rms"] = np.array(
        np.sqrt(np.mean(validation_error_norm ** 2)) if validation_error_norm.size else 0.0
    )
    diagnostics["right_arm_qacc_mapping_validation_error_norm_max"] = np.array(
        np.max(validation_error_norm) if validation_error_norm.size else 0.0
    )
    linearization_error_norm = np.linalg.norm(qacc_mapping_linearization_error, axis=1) if qacc_mapping_linearization_error.shape[0] else np.zeros(0)
    diagnostics["right_arm_qacc_mapping_linearization_error_norm_rms"] = np.array(
        np.sqrt(np.mean(linearization_error_norm ** 2)) if linearization_error_norm.size else 0.0
    )
    diagnostics["right_arm_qacc_mapping_linearization_error_norm_max"] = np.array(
        np.max(linearization_error_norm) if linearization_error_norm.size else 0.0
    )
    mapping_model_error_norm = np.linalg.norm(qacc_mapping_model_error, axis=1) if qacc_mapping_model_error.shape[0] else np.zeros(0)
    diagnostics["right_arm_qacc_mapping_model_error_norm_rms"] = np.array(
        np.sqrt(np.mean(mapping_model_error_norm ** 2)) if mapping_model_error_norm.size else 0.0
    )
    diagnostics["right_arm_qacc_mapping_model_error_norm_max"] = np.array(
        np.max(mapping_model_error_norm) if mapping_model_error_norm.size else 0.0
    )
    condition_number = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_condition_number", []),
        dtype=np.float64,
    )
    finite_condition = condition_number[np.isfinite(condition_number)]
    diagnostics["right_arm_forward_dynamics_condition_number_mean"] = np.array(
        np.mean(finite_condition) if finite_condition.size else np.inf
    )
    diagnostics["right_arm_forward_dynamics_condition_number_p95"] = np.array(
        np.percentile(finite_condition, 95.0) if finite_condition.size else np.inf
    )
    diagnostics["right_arm_forward_dynamics_condition_number_max"] = np.array(
        np.max(finite_condition) if finite_condition.size else np.inf
    )
    validation_scale = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_validation_scale", []),
        dtype=np.float64,
    )
    validation_attempts = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_validation_attempts", []),
        dtype=np.float64,
    )
    validation_improved = np.asarray(
        trajectory_data.get("right_arm_forward_dynamics_validation_improved", []),
        dtype=bool,
    )
    diagnostics["right_arm_forward_dynamics_validation_scale_mean"] = np.array(
        np.mean(validation_scale) if validation_scale.size else 0.0
    )
    diagnostics["right_arm_forward_dynamics_validation_attempts_mean"] = np.array(
        np.mean(validation_attempts) if validation_attempts.size else 0.0
    )
    diagnostics["right_arm_forward_dynamics_validation_attempts_max"] = np.array(
        np.max(validation_attempts) if validation_attempts.size else 0.0
    )
    diagnostics["right_arm_forward_dynamics_validation_fallback_fraction"] = np.array(
        np.mean(~validation_improved) if validation_improved.size else 0.0
    )
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0):
        label = str(scale).replace(".", "p")
        diagnostics[f"right_arm_forward_dynamics_validation_scale_{label}_fraction"] = np.array(
            np.mean(np.isclose(validation_scale, scale)) if validation_scale.size else 0.0
        )

    if qacc.shape == ddq_des.shape and qacc.shape[0] > 0:
        tracking_error = qacc - ddq_des
        diagnostics["right_arm_qacc_tracking_error_rms"] = np.sqrt(np.mean(tracking_error ** 2, axis=0))
        diagnostics["right_arm_qacc_tracking_error_abs_max"] = np.max(np.abs(tracking_error), axis=0)
        diagnostics["right_arm_qacc_instantaneous_tracking_error_rms"] = diagnostics["right_arm_qacc_tracking_error_rms"].copy()
        diagnostics["right_arm_qacc_instantaneous_tracking_error_abs_max"] = diagnostics["right_arm_qacc_tracking_error_abs_max"].copy()
    else:
        diagnostics["right_arm_qacc_tracking_error_rms"] = np.zeros(n)
        diagnostics["right_arm_qacc_tracking_error_abs_max"] = np.zeros(n)
        diagnostics["right_arm_qacc_instantaneous_tracking_error_rms"] = np.zeros(n)
        diagnostics["right_arm_qacc_instantaneous_tracking_error_abs_max"] = np.zeros(n)

    for name in [
        "torso_acc_world_raw",
        "torso_acc_world_used",
        "torso_alpha_world_raw",
        "torso_alpha_world_used",
        "right_ee_position_error_torso",
        "right_ee_gravity_error_end",
    ]:
        value = matrix(name, 3)
        component_stats(diagnostics, name, value, 3)
        norm = np.linalg.norm(value, axis=1) if value.shape[0] else np.zeros(0)
        diagnostics[f"{name}_norm_rms"] = np.array(np.sqrt(np.mean(norm ** 2)) if norm.size else 0.0)
        diagnostics[f"{name}_norm_max"] = np.array(np.max(norm) if norm.size else 0.0)

    alignment = np.asarray(trajectory_data.get("right_ee_upright_alignment", []), dtype=np.float64)
    diagnostics["right_ee_upright_alignment_mean"] = np.array(float(np.mean(alignment)) if alignment.size else 0.0)
    diagnostics["right_ee_upright_alignment_min"] = np.array(float(np.min(alignment)) if alignment.size else 0.0)
    diagnostics["right_ee_inverted_fraction"] = np.array(float(np.mean(alignment < 0.0)) if alignment.size else 0.0)
    contact_count = np.asarray(trajectory_data.get("contact_count", []), dtype=np.int64)
    diagnostics["any_contact_fraction"] = np.array(float(np.mean(contact_count > 0)) if contact_count.size else 0.0)
    constraint_norm = np.linalg.norm(tau_constraint, axis=1) if tau_constraint.shape[0] else np.zeros(0)
    diagnostics["right_arm_constraint_active_fraction"] = np.array(float(np.mean(constraint_norm > 1e-6)) if constraint_norm.size else 0.0)
    diagnostics["right_arm_contact_constraint_active_fraction"] = diagnostics["right_arm_constraint_active_fraction"].copy()
    total_constraint_norm = np.linalg.norm(tau_constraint_total, axis=1) if tau_constraint_total.shape[0] else np.zeros(0)
    diagnostics["right_arm_total_constraint_active_fraction"] = np.array(
        float(np.mean(total_constraint_norm > 1e-6)) if total_constraint_norm.size else 0.0
    )
    return diagnostics


def save_trajectory(trajectory_path, trajectory_data, xml_path, simulation_dt):
    right_arm_diagnostics = compute_right_arm_trajectory_diagnostics(trajectory_data)
    arrays = {name: np.asarray(values) for name, values in trajectory_data.items()}
    if arrays["right_arm_ddq_raw"].shape == arrays["right_arm_ddq_des"].shape:
        arrays["right_arm_ddq_postprocess_delta"] = arrays["right_arm_ddq_des"] - arrays["right_arm_ddq_raw"]
    arrays["right_arm_ddq_saturation_limit_history"] = arrays.pop("right_arm_ddq_saturation_limit")
    arrays["right_arm_tau_limit_lower_history"] = arrays.pop("right_arm_tau_limit_lower")
    arrays["right_arm_tau_limit_upper_history"] = arrays.pop("right_arm_tau_limit_upper")
    np.savez(
        trajectory_path,
        **arrays,
        **right_arm_diagnostics,
        lqr_cost_term_names=np.asarray(LQR_COST_TERM_NAMES),
        xml_path=np.array(xml_path),
        simulation_dt=np.array(simulation_dt),
    )
    return right_arm_diagnostics


def save_right_arm_diagnostics(run_dir, diagnostics):
    diagnostics_path = os.path.join(run_dir, "right_arm_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(diagnostics), f, indent=2, ensure_ascii=False)
    return diagnostics_path


def save_lqr_tracking_diagnostics(run_dir, diagnostics):
    diagnostics_path = os.path.join(run_dir, "lqr_tracking_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(diagnostics), f, indent=2, ensure_ascii=False)
    return diagnostics_path


def save_lqr_ddq_tracking_plot(run_dir, trajectory_data, diagnostics, eval_start_time, eval_end_time):
    """绘制评估区间内五关节 ddq_des 与 6 ms 速度差分 ddq_real。"""
    plot_path = os.path.join(run_dir, "ddq_tracking.png")
    time_values = np.asarray(trajectory_data.get("time", []), dtype=np.float64)
    valid = np.asarray(trajectory_data.get("right_lqr_tracking_valid", []), dtype=bool)
    interval_dt = np.asarray(trajectory_data.get("right_lqr_tracking_interval_dt", []), dtype=np.float64)
    ddq_des = np.asarray(trajectory_data.get("right_arm_ddq_des", []), dtype=np.float64)
    ddq_real = np.asarray(trajectory_data.get("right_arm_ddq_real", []), dtype=np.float64)
    expected_shape = (len(time_values), len(RIGHT_ARM_JOINT_NAMES))
    if (
        valid.shape != time_values.shape
        or interval_dt.shape != time_values.shape
        or ddq_des.shape != expected_shape
        or ddq_real.shape != expected_shape
    ):
        return None
    valid &= (time_values >= eval_start_time) & (
        time_values + interval_dt <= eval_end_time + 1e-12
    )
    if not np.any(valid):
        return None

    metrics = diagnostics.get("ddq_tracking", {})
    correlation = np.asarray(metrics.get("correlation", np.zeros(5)), dtype=np.float64)
    gain = np.asarray(metrics.get("gain", np.zeros(5)), dtype=np.float64)
    rmse = np.asarray(metrics.get("rmse", np.zeros(5)), dtype=np.float64)
    labels = tuple(name.removeprefix("right_").removesuffix("_joint") for name in RIGHT_ARM_JOINT_NAMES)
    fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True)
    plot_time = time_values[valid]
    for joint, (axis, label) in enumerate(zip(axes, labels)):
        axis.plot(plot_time, ddq_des[valid, joint], color="tab:blue", lw=1.2, label="ddq_des")
        axis.plot(plot_time, ddq_real[valid, joint], color="tab:orange", lw=1.0, alpha=0.9, label="ddq_real")
        axis.axhline(0.0, color="black", lw=0.6, alpha=0.4)
        axis.grid(True, alpha=0.3)
        axis.set_ylabel("rad/s^2")
        axis.set_title(label)
        axis.text(
            0.99,
            0.95,
            f"corr={correlation[joint]:.3f}\ngain={gain[joint]:.3f}\nRMSE={rmse[joint]:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("LQR DDQ tracking: desired versus realized 6 ms average acceleration")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    return plot_path


def save_lqr_tracking_preview(run_dir, trajectory_data, eval_start_time, eval_end_time):
    """保存严格按相邻手臂控制更新对齐的 DDQ 与一步代价跟踪表。"""
    preview_path = os.path.join(run_dir, "lqr_tracking_preview.csv")
    time_values = np.asarray(trajectory_data.get("time", []), dtype=np.float64)
    valid = np.asarray(trajectory_data.get("right_lqr_tracking_valid", []), dtype=bool)
    ddq_des = np.asarray(trajectory_data.get("right_arm_ddq_des", []), dtype=np.float64)
    ddq_real = np.asarray(trajectory_data.get("right_arm_ddq_real", []), dtype=np.float64)
    ddq_error = np.asarray(trajectory_data.get("right_arm_ddq_tracking_error", []), dtype=np.float64)
    model_cost = np.asarray(trajectory_data.get("right_lqr_one_step_cost_model", []), dtype=np.float64)
    actual_cost = np.asarray(trajectory_data.get("right_lqr_one_step_cost_actual", []), dtype=np.float64)
    cost_error = np.asarray(trajectory_data.get("right_lqr_one_step_cost_error", []), dtype=np.float64)
    interval_dt = np.asarray(trajectory_data.get("right_lqr_tracking_interval_dt", []), dtype=np.float64)
    sample_count = len(time_values)
    expected = (
        valid.shape == (sample_count,),
        ddq_des.shape == (sample_count, len(RIGHT_ARM_JOINT_NAMES)),
        ddq_real.shape == ddq_des.shape,
        ddq_error.shape == ddq_des.shape,
        model_cost.shape == (sample_count, len(LQR_COST_TERM_NAMES)),
        actual_cost.shape == model_cost.shape,
        cost_error.shape == model_cost.shape,
        interval_dt.shape == (sample_count,),
    )
    if not all(expected):
        valid = np.zeros(sample_count, dtype=bool)

    joint_labels = tuple(name.removeprefix("right_").removesuffix("_joint") for name in RIGHT_ARM_JOINT_NAMES)
    headers = ["time", "interval_dt", "in_evaluation"]
    for joint in joint_labels:
        headers.extend((f"ddq_des_{joint}", f"ddq_real_{joint}", f"ddq_error_{joint}"))
    for term in LQR_COST_TERM_NAMES:
        headers.extend((f"cost_model_{term}", f"cost_actual_{term}", f"cost_error_{term}"))
    headers.extend(("cost_model_total", "cost_actual_total", "cost_error_total"))

    with open(preview_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for index in np.flatnonzero(valid):
            row = [
                time_values[index],
                interval_dt[index],
                bool(
                    eval_start_time <= time_values[index]
                    and time_values[index] + interval_dt[index] <= eval_end_time + 1e-12
                ),
            ]
            for joint in range(len(joint_labels)):
                row.extend((ddq_des[index, joint], ddq_real[index, joint], ddq_error[index, joint]))
            for term in range(len(LQR_COST_TERM_NAMES)):
                row.extend((model_cost[index, term], actual_cost[index, term], cost_error[index, term]))
            model_total = float(np.sum(model_cost[index]))
            actual_total = float(np.sum(actual_cost[index]))
            row.extend((model_total, actual_total, actual_total - model_total))
            writer.writerow(row)
    return preview_path


def save_control_preview(run_dir, trajectory_data):
    """把最关键的高频控制信号另存为可直接查看的 CSV。"""
    preview_path = os.path.join(run_dir, "control_preview.csv")
    time_values = np.asarray(trajectory_data.get("time", []), dtype=np.float64)
    joint_labels = tuple(name.removeprefix("right_").removesuffix("_joint") for name in RIGHT_ARM_JOINT_NAMES)
    vector_signals = [
        ("right_arm_q", joint_labels),
        ("right_arm_dq", joint_labels),
        ("right_arm_qacc", joint_labels),
        ("right_arm_target_q", joint_labels),
        ("right_arm_target_dq", joint_labels),
        ("right_arm_ddq_raw", joint_labels),
        ("right_arm_ddq_des", joint_labels),
        ("right_arm_ddq_real", joint_labels),
        ("right_arm_ddq_tracking_error", joint_labels),
        ("right_arm_tau_inverse", joint_labels),
        ("right_arm_tau_constraint", joint_labels),
        ("right_arm_tau_contact", joint_labels),
        ("right_arm_tau_constraint_total", joint_labels),
        ("right_arm_tau_constraint_noncontact", joint_labels),
        ("right_arm_tau_constraint_nonfriction", joint_labels),
        ("right_arm_tau_constraint_friction", joint_labels),
        ("right_arm_tau_ff", joint_labels),
        ("right_arm_tau_pd", joint_labels),
        ("right_arm_tau_nominal", joint_labels),
        ("right_arm_tau_mapping_correction_raw", joint_labels),
        ("right_arm_tau_mapping_correction", joint_labels),
        ("right_arm_tau_cmd_raw", joint_labels),
        ("right_arm_ctrl", joint_labels),
        ("right_arm_qacc_mapping_baseline", joint_labels),
        ("right_arm_qacc_mapping_predicted", joint_labels),
        ("right_arm_qacc_mapping_prediction_error", joint_labels),
        ("right_arm_qacc_mapping_validated", joint_labels),
        ("right_arm_qacc_mapping_validation_error", joint_labels),
        ("right_arm_qacc_mapping_linearization_error", joint_labels),
        ("right_arm_qacc_mapping_model_error", joint_labels),
        ("right_arm_forward_dynamics_singular_values", joint_labels),
        ("right_arm_first_pass_qacc_validated", joint_labels),
        ("right_arm_first_pass_qacc_validation_error", joint_labels),
        ("right_arm_second_pass_tau_correction_raw", joint_labels),
        ("right_arm_second_pass_tau_correction", joint_labels),
        ("right_arm_second_pass_qacc_predicted", joint_labels),
        ("right_arm_second_pass_qacc_validated", joint_labels),
        ("right_arm_second_pass_qacc_validation_error", joint_labels),
        ("right_arm_second_pass_qacc_linearization_error", joint_labels),
        ("right_arm_second_pass_singular_values", joint_labels),
        ("right_arm_actual_qfrc_bias", joint_labels),
        ("right_arm_actual_qfrc_passive", joint_labels),
        ("right_arm_actual_qfrc_constraint", joint_labels),
        ("torso_lin_vel_world", ("x", "y", "z")),
        ("torso_ang_vel_world", ("x", "y", "z")),
        ("torso_acc_world_raw", ("x", "y", "z")),
        ("torso_acc_world_used", ("x", "y", "z")),
        ("torso_alpha_world_raw", ("x", "y", "z")),
        ("torso_alpha_world_used", ("x", "y", "z")),
        ("right_ee_position_torso", ("x", "y", "z")),
        ("right_ee_position_reference_torso", ("x", "y", "z")),
        ("right_ee_position_error_torso", ("x", "y", "z")),
        ("right_ee_gravity_error_end", ("x", "y", "z")),
    ]
    scalar_signals = [
        "right_arm_forward_dynamics_condition_number",
        "right_arm_forward_dynamics_validation_scale",
        "right_arm_forward_dynamics_validation_attempts",
        "right_arm_forward_dynamics_validation_improved",
        "right_arm_forward_dynamics_tracking_safety_satisfied",
        "right_arm_forward_dynamics_qacc_safety_satisfied",
        "right_arm_forward_dynamics_safe_candidate_count",
        "right_arm_forward_dynamics_total_error_rejections",
        "right_arm_forward_dynamics_joint_error_rejections",
        "right_arm_forward_dynamics_qacc_limit_rejections",
        "right_arm_forward_dynamics_second_pass_triggered",
        "right_arm_forward_dynamics_second_pass_accepted",
        "right_arm_second_pass_tracking_safety_satisfied",
        "right_arm_second_pass_qacc_safety_satisfied",
        "right_arm_second_pass_condition_number",
        "right_arm_second_pass_validation_scale",
        "right_arm_second_pass_validation_attempts",
        "right_arm_second_pass_safe_candidate_count",
        "right_arm_second_pass_total_error_rejections",
        "right_arm_second_pass_joint_error_rejections",
        "right_arm_second_pass_qacc_limit_rejections",
        "right_arm_forward_dynamics_safety_fallback_used",
        "right_arm_forward_dynamics_safety_fallback_satisfied",
        "right_arm_forward_dynamics_safety_fallback_attempts",
        "heading_reference_world",
        "heading_yaw_unwrapped",
        "heading_yaw_filtered",
        "heading_yaw_error",
        "heading_yaw_rate_filtered",
        "heading_yaw_rate_correction",
        "heading_yaw_rate_command",
        "heading_command_saturated",
        "right_ee_upright_alignment",
        "arm_policy_updated",
        "contact_count",
    ]
    arrays = []
    headers = ["time"]
    for name, labels in vector_signals:
        value = np.asarray(trajectory_data.get(name, []), dtype=np.float64)
        if value.shape != (len(time_values), len(labels)):
            value = np.full((len(time_values), len(labels)), np.nan)
        arrays.append(value)
        headers.extend(f"{name}_{label}" for label in labels)
    scalars = []
    for name in scalar_signals:
        value = np.asarray(trajectory_data.get(name, []), dtype=np.float64)
        if value.shape != (len(time_values),):
            value = np.full(len(time_values), np.nan)
        scalars.append(value)
        headers.append(name)

    with open(preview_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, timestamp in enumerate(time_values):
            row = [timestamp]
            for value in arrays:
                row.extend(value[i].tolist())
            row.extend(value[i] for value in scalars)
            writer.writerow(row)
    return preview_path


def write_video(video_path, video_frames, video_fps):
    if imageio is None or not video_frames:
        return
    imageio.mimwrite(video_path, video_frames, fps=video_fps, quality=8, macro_block_size=None)


def close_renderer(renderer):
    if renderer is None:
        return
    close_fn = getattr(renderer, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def record_eval_step(model, data, counter, simulation_dt, scene_ids, buffers, right_arm_control=None):
    if buffers.torso_xy_start is None:
        buffers.torso_xy_start = data.xpos[scene_ids.torso_id][:2].copy()
    left_rot = data.site_xmat[scene_ids.left_grasp_site_id].reshape(3, 3).copy()
    right_rot = data.site_xmat[scene_ids.right_grasp_site_id].reshape(3, 3).copy()
    torso_yaw = quat_to_yaw_wxyz(data.xquat[scene_ids.torso_id].copy())
    left_lin_vel, left_ang_vel = get_site_vel(model, data, scene_ids.left_grasp_site_id)
    right_lin_vel, right_ang_vel = get_site_vel(model, data, scene_ids.right_grasp_site_id)
    left_lin_acc = np.zeros(3) if counter == 0 else (left_lin_vel - buffers.prev_left_lin_vel) / simulation_dt
    left_ang_acc = np.zeros(3) if counter == 0 else (left_ang_vel - buffers.prev_left_ang_vel) / simulation_dt
    right_lin_acc = np.zeros(3) if counter == 0 else (right_lin_vel - buffers.prev_right_lin_vel) / simulation_dt
    right_ang_acc = np.zeros(3) if counter == 0 else (right_ang_vel - buffers.prev_right_ang_vel) / simulation_dt
    buffers.prev_left_lin_vel, buffers.prev_left_ang_vel = left_lin_vel.copy(), left_ang_vel.copy()
    buffers.prev_right_lin_vel, buffers.prev_right_ang_vel = right_lin_vel.copy(), right_ang_vel.copy()
    t = counter * simulation_dt
    buffers.trajectory_data["time"].append(t)
    buffers.trajectory_data["qpos"].append(data.qpos.copy())
    buffers.trajectory_data["qvel"].append(data.qvel.copy())
    buffers.trajectory_data["qacc"].append(data.qacc.copy())
    buffers.trajectory_data["ctrl"].append(data.ctrl.copy())
    if right_arm_control is None:
        right_arm_control = {}
    def control_vector(name, size, default=0.0):
        value = np.asarray(right_arm_control.get(name, np.full(size, default)), dtype=np.float64)
        return value.copy() if value.shape == (size,) else np.full(size, default, dtype=np.float64)

    def control_scalar(name, default=np.nan):
        value = np.asarray(right_arm_control.get(name, default), dtype=np.float64)
        return float(value) if value.shape == () else float(default)

    target_q = control_vector("target_q", 5)
    target_dq = control_vector("target_dq", 5)
    ddq_raw = control_vector("ddq_raw", 5)
    ddq_des = np.asarray(right_arm_control.get("ddq_des", np.zeros(5)), dtype=np.float64).copy()
    ddq_saturation_limit = float(right_arm_control.get("ddq_saturation_limit", np.inf))
    ddq_saturation_threshold = ddq_saturation_limit - RIGHT_ARM_DDQ_SATURATION_EPS
    ddq_saturation_mask = np.zeros_like(ddq_des, dtype=bool) if (not np.isfinite(ddq_saturation_threshold) or ddq_saturation_threshold <= 0.0) else (np.abs(ddq_des) >= ddq_saturation_threshold)
    tau_inverse = control_vector("tau_inverse", 5)
    tau_contact = control_vector("tau_contact", 5)
    tau_constraint_total = control_vector("tau_constraint_total", 5)
    tau_constraint_noncontact = control_vector("tau_constraint_noncontact", 5)
    tau_constraint_nonfriction = control_vector("tau_constraint_nonfriction", 5)
    tau_constraint_friction = control_vector("tau_constraint_friction", 5)
    tau_ff = control_vector("tau_ff", 5)
    tau_pd = control_vector("tau_pd", 5)
    tau_nominal = control_vector("tau_nominal", 5)
    tau_mapping_correction_raw = control_vector("tau_mapping_correction_raw", 5)
    tau_mapping_correction = control_vector("tau_mapping_correction", 5)
    qacc_mapping_baseline = control_vector("qacc_mapping_baseline", 5)
    qacc_mapping_predicted = control_vector("qacc_mapping_predicted", 5)
    qacc_mapping_prediction_error = control_vector("qacc_mapping_prediction_error", 5)
    qacc_mapping_validated = control_vector("qacc_mapping_validated", 5)
    qacc_mapping_validation_error = control_vector("qacc_mapping_validation_error", 5)
    qacc_mapping_linearization_error = control_vector("qacc_mapping_linearization_error", 5)
    # mj_step 后的实际瞬时 qacc 应与验收用完整 mj_forward 一致；该误差用于检查时序和求解一致性。
    qacc_mapping_model_error = data.qacc[RIGHT_ARM_QVEL_SLICE].copy() - qacc_mapping_validated
    forward_dynamics_singular_values = control_vector("forward_dynamics_singular_values", 5)
    forward_dynamics_gain = np.asarray(
        right_arm_control.get("forward_dynamics_gain", np.zeros((5, 5))),
        dtype=np.float64,
    )
    if forward_dynamics_gain.shape != (5, 5):
        forward_dynamics_gain = np.zeros((5, 5), dtype=np.float64)
    forward_dynamics_condition_number = float(
        right_arm_control.get("forward_dynamics_condition_number", np.inf)
    )
    forward_dynamics_validation_scale = float(
        right_arm_control.get("forward_dynamics_validation_scale", 0.0)
    )
    forward_dynamics_validation_attempts = int(
        right_arm_control.get("forward_dynamics_validation_attempts", 0)
    )
    forward_dynamics_validation_improved = bool(
        right_arm_control.get("forward_dynamics_validation_improved", False)
    )
    forward_dynamics_tracking_safety_satisfied = bool(
        right_arm_control.get("forward_dynamics_tracking_safety_satisfied", False)
    )
    forward_dynamics_qacc_safety_satisfied = bool(
        right_arm_control.get("forward_dynamics_qacc_safety_satisfied", False)
    )
    forward_dynamics_safe_candidate_count = int(
        right_arm_control.get("forward_dynamics_safe_candidate_count", 0)
    )
    forward_dynamics_total_error_rejections = int(
        right_arm_control.get("forward_dynamics_total_error_rejections", 0)
    )
    forward_dynamics_joint_error_rejections = int(
        right_arm_control.get("forward_dynamics_joint_error_rejections", 0)
    )
    forward_dynamics_qacc_limit_rejections = int(
        right_arm_control.get("forward_dynamics_qacc_limit_rejections", 0)
    )
    first_pass_qacc_validated = control_vector("first_pass_qacc_validated", 5)
    first_pass_qacc_validation_error = control_vector("first_pass_qacc_validation_error", 5)
    forward_dynamics_second_pass_triggered = bool(
        right_arm_control.get("forward_dynamics_second_pass_triggered", False)
    )
    forward_dynamics_second_pass_accepted = bool(
        right_arm_control.get("forward_dynamics_second_pass_accepted", False)
    )
    second_pass_tracking_safety_satisfied = bool(
        right_arm_control.get("second_pass_tracking_safety_satisfied", False)
    )
    second_pass_qacc_safety_satisfied = bool(
        right_arm_control.get("second_pass_qacc_safety_satisfied", False)
    )
    second_pass_tau_correction_raw = control_vector("second_pass_tau_correction_raw", 5)
    second_pass_tau_correction = control_vector("second_pass_tau_correction", 5)
    second_pass_qacc_predicted = control_vector("second_pass_qacc_predicted", 5)
    second_pass_qacc_validated = control_vector("second_pass_qacc_validated", 5)
    second_pass_qacc_validation_error = control_vector("second_pass_qacc_validation_error", 5)
    second_pass_qacc_linearization_error = control_vector("second_pass_qacc_linearization_error", 5)
    second_pass_singular_values = control_vector("second_pass_singular_values", 5)
    second_pass_forward_dynamics_gain = np.asarray(
        right_arm_control.get("second_pass_forward_dynamics_gain", np.zeros((5, 5))),
        dtype=np.float64,
    )
    if second_pass_forward_dynamics_gain.shape != (5, 5):
        second_pass_forward_dynamics_gain = np.zeros((5, 5), dtype=np.float64)
    second_pass_condition_number = float(
        right_arm_control.get("second_pass_condition_number", np.inf)
    )
    second_pass_validation_scale = float(
        right_arm_control.get("second_pass_validation_scale", 0.0)
    )
    second_pass_validation_attempts = int(
        right_arm_control.get("second_pass_validation_attempts", 0)
    )
    second_pass_safe_candidate_count = int(
        right_arm_control.get("second_pass_safe_candidate_count", 0)
    )
    second_pass_total_error_rejections = int(
        right_arm_control.get("second_pass_total_error_rejections", 0)
    )
    second_pass_joint_error_rejections = int(
        right_arm_control.get("second_pass_joint_error_rejections", 0)
    )
    second_pass_qacc_limit_rejections = int(
        right_arm_control.get("second_pass_qacc_limit_rejections", 0)
    )
    forward_dynamics_safety_fallback_used = bool(
        right_arm_control.get("forward_dynamics_safety_fallback_used", False)
    )
    forward_dynamics_safety_fallback_satisfied = bool(
        right_arm_control.get("forward_dynamics_safety_fallback_satisfied", False)
    )
    forward_dynamics_safety_fallback_attempts = int(
        right_arm_control.get("forward_dynamics_safety_fallback_attempts", 0)
    )
    tau_cmd_raw = np.asarray(right_arm_control.get("tau_cmd_raw", tau_ff + tau_pd), dtype=np.float64).copy()
    tau_limit_lower = np.asarray(right_arm_control.get("tau_limit_lower", np.full(5, -np.inf)), dtype=np.float64).copy()
    tau_limit_upper = np.asarray(right_arm_control.get("tau_limit_upper", np.full(5, np.inf)), dtype=np.float64).copy()
    if tau_cmd_raw.shape != tau_ff.shape:
        tau_cmd_raw = tau_ff + tau_pd
    if tau_limit_lower.shape != tau_ff.shape:
        tau_limit_lower = np.full_like(tau_ff, -np.inf)
    if tau_limit_upper.shape != tau_ff.shape:
        tau_limit_upper = np.full_like(tau_ff, np.inf)
    tau_saturation_mask = (tau_cmd_raw < (tau_limit_lower + RIGHT_ARM_TAU_SATURATION_EPS)) | (tau_cmd_raw > (tau_limit_upper - RIGHT_ARM_TAU_SATURATION_EPS))
    torso_lin_vel = control_vector("torso_lin_vel_world", 3)
    torso_ang_vel = control_vector("torso_ang_vel_world", 3)
    torso_acc_raw = control_vector("torso_acc_world_raw", 3)
    torso_acc_used = control_vector("torso_acc_world_used", 3)
    torso_alpha_raw = control_vector("torso_alpha_world_raw", 3)
    torso_alpha_used = control_vector("torso_alpha_world_used", 3)
    heading_reference_world = control_scalar("heading_reference_world")
    heading_yaw_unwrapped = control_scalar("heading_yaw_unwrapped")
    heading_yaw_filtered = control_scalar("heading_yaw_filtered")
    heading_yaw_error = control_scalar("heading_yaw_error")
    heading_yaw_rate_filtered = control_scalar("heading_yaw_rate_filtered")
    heading_yaw_rate_correction = control_scalar("heading_yaw_rate_correction")
    heading_yaw_rate_command = control_scalar("heading_yaw_rate_command")
    heading_command_saturated = bool(right_arm_control.get("heading_command_saturated", False))
    position_reference = control_vector("ee_position_reference_torso", 3)
    lqr_prediction = right_arm_control.get("lqr_one_step_prediction")
    lqr_prediction_valid = isinstance(lqr_prediction, dict)

    def prediction_vector(name, size):
        if not lqr_prediction_valid:
            return np.full(size, np.nan, dtype=np.float64)
        value = np.asarray(lqr_prediction.get(name, np.full(size, np.nan)), dtype=np.float64)
        return value.copy() if value.shape == (size,) else np.full(size, np.nan, dtype=np.float64)

    prediction_costs = {} if not lqr_prediction_valid else lqr_prediction.get("cost_terms", {})
    lqr_cost_model = np.array(
        [float(prediction_costs.get(name, np.nan)) for name in LQR_COST_TERM_NAMES],
        dtype=np.float64,
    )
    torso_rot = data.site_xmat[scene_ids.imu_site_id].reshape(3, 3).copy()
    position_torso = torso_rot.T @ (
        data.site_xpos[scene_ids.right_grasp_site_id] - data.site_xpos[scene_ids.imu_site_id]
    )
    position_error = position_torso - position_reference
    gravity_error = tilt_error_from_rot(right_rot, model.opt.gravity)
    upright_alignment = upright_alignment_from_rot(right_rot, model.opt.gravity)
    arm_policy_updated = bool(right_arm_control.get("arm_policy_updated", False))
    buffers.trajectory_data["right_arm_q"].append(data.qpos[RIGHT_ARM_QPOS_SLICE].copy())
    buffers.trajectory_data["right_arm_dq"].append(data.qvel[RIGHT_ARM_QVEL_SLICE].copy())
    buffers.trajectory_data["right_arm_qacc"].append(data.qacc[RIGHT_ARM_QVEL_SLICE].copy())
    buffers.trajectory_data["right_arm_ctrl"].append(data.ctrl[RIGHT_ARM_CTRL_SLICE].copy())
    buffers.trajectory_data["right_arm_target_q"].append(target_q)
    buffers.trajectory_data["right_arm_target_dq"].append(target_dq)
    buffers.trajectory_data["right_arm_ddq_raw"].append(ddq_raw)
    buffers.trajectory_data["right_arm_ddq_des"].append(ddq_des)
    buffers.trajectory_data["right_arm_ddq_saturation_limit"].append(ddq_saturation_limit)
    buffers.trajectory_data["right_arm_ddq_saturation_mask"].append(ddq_saturation_mask)
    buffers.trajectory_data["right_arm_tau_inverse"].append(tau_inverse)
    buffers.trajectory_data["right_arm_tau_constraint"].append(tau_contact)
    buffers.trajectory_data["right_arm_tau_contact"].append(tau_contact)
    buffers.trajectory_data["right_arm_tau_constraint_total"].append(tau_constraint_total)
    buffers.trajectory_data["right_arm_tau_constraint_noncontact"].append(tau_constraint_noncontact)
    buffers.trajectory_data["right_arm_tau_constraint_nonfriction"].append(tau_constraint_nonfriction)
    buffers.trajectory_data["right_arm_tau_constraint_friction"].append(tau_constraint_friction)
    buffers.trajectory_data["right_arm_tau_ff"].append(tau_ff)
    buffers.trajectory_data["right_arm_tau_pd"].append(tau_pd)
    buffers.trajectory_data["right_arm_tau_nominal"].append(tau_nominal)
    buffers.trajectory_data["right_arm_tau_mapping_correction_raw"].append(tau_mapping_correction_raw)
    buffers.trajectory_data["right_arm_tau_mapping_correction"].append(tau_mapping_correction)
    buffers.trajectory_data["right_arm_tau_cmd_raw"].append(tau_cmd_raw)
    buffers.trajectory_data["right_arm_tau_limit_lower"].append(tau_limit_lower)
    buffers.trajectory_data["right_arm_tau_limit_upper"].append(tau_limit_upper)
    buffers.trajectory_data["right_arm_tau_saturation_mask"].append(tau_saturation_mask)
    buffers.trajectory_data["right_arm_actual_qfrc_bias"].append(data.qfrc_bias[RIGHT_ARM_QVEL_SLICE].copy())
    buffers.trajectory_data["right_arm_actual_qfrc_passive"].append(data.qfrc_passive[RIGHT_ARM_QVEL_SLICE].copy())
    buffers.trajectory_data["right_arm_actual_qfrc_constraint"].append(data.qfrc_constraint[RIGHT_ARM_QVEL_SLICE].copy())
    buffers.trajectory_data["right_arm_qacc_mapping_baseline"].append(qacc_mapping_baseline)
    buffers.trajectory_data["right_arm_qacc_mapping_predicted"].append(qacc_mapping_predicted)
    buffers.trajectory_data["right_arm_qacc_mapping_prediction_error"].append(qacc_mapping_prediction_error)
    buffers.trajectory_data["right_arm_qacc_mapping_validated"].append(qacc_mapping_validated)
    buffers.trajectory_data["right_arm_qacc_mapping_validation_error"].append(qacc_mapping_validation_error)
    buffers.trajectory_data["right_arm_qacc_mapping_linearization_error"].append(qacc_mapping_linearization_error)
    buffers.trajectory_data["right_arm_qacc_mapping_model_error"].append(qacc_mapping_model_error)
    buffers.trajectory_data["right_arm_forward_dynamics_gain"].append(forward_dynamics_gain.copy())
    buffers.trajectory_data["right_arm_forward_dynamics_singular_values"].append(forward_dynamics_singular_values)
    buffers.trajectory_data["right_arm_forward_dynamics_condition_number"].append(forward_dynamics_condition_number)
    buffers.trajectory_data["right_arm_forward_dynamics_validation_scale"].append(forward_dynamics_validation_scale)
    buffers.trajectory_data["right_arm_forward_dynamics_validation_attempts"].append(forward_dynamics_validation_attempts)
    buffers.trajectory_data["right_arm_forward_dynamics_validation_improved"].append(forward_dynamics_validation_improved)
    buffers.trajectory_data["right_arm_forward_dynamics_tracking_safety_satisfied"].append(forward_dynamics_tracking_safety_satisfied)
    buffers.trajectory_data["right_arm_forward_dynamics_qacc_safety_satisfied"].append(forward_dynamics_qacc_safety_satisfied)
    buffers.trajectory_data["right_arm_forward_dynamics_safe_candidate_count"].append(forward_dynamics_safe_candidate_count)
    buffers.trajectory_data["right_arm_forward_dynamics_total_error_rejections"].append(forward_dynamics_total_error_rejections)
    buffers.trajectory_data["right_arm_forward_dynamics_joint_error_rejections"].append(forward_dynamics_joint_error_rejections)
    buffers.trajectory_data["right_arm_forward_dynamics_qacc_limit_rejections"].append(forward_dynamics_qacc_limit_rejections)
    buffers.trajectory_data["right_arm_first_pass_qacc_validated"].append(first_pass_qacc_validated)
    buffers.trajectory_data["right_arm_first_pass_qacc_validation_error"].append(first_pass_qacc_validation_error)
    buffers.trajectory_data["right_arm_forward_dynamics_second_pass_triggered"].append(forward_dynamics_second_pass_triggered)
    buffers.trajectory_data["right_arm_forward_dynamics_second_pass_accepted"].append(forward_dynamics_second_pass_accepted)
    buffers.trajectory_data["right_arm_second_pass_tracking_safety_satisfied"].append(second_pass_tracking_safety_satisfied)
    buffers.trajectory_data["right_arm_second_pass_qacc_safety_satisfied"].append(second_pass_qacc_safety_satisfied)
    buffers.trajectory_data["right_arm_second_pass_tau_correction_raw"].append(second_pass_tau_correction_raw)
    buffers.trajectory_data["right_arm_second_pass_tau_correction"].append(second_pass_tau_correction)
    buffers.trajectory_data["right_arm_second_pass_qacc_predicted"].append(second_pass_qacc_predicted)
    buffers.trajectory_data["right_arm_second_pass_qacc_validated"].append(second_pass_qacc_validated)
    buffers.trajectory_data["right_arm_second_pass_qacc_validation_error"].append(second_pass_qacc_validation_error)
    buffers.trajectory_data["right_arm_second_pass_qacc_linearization_error"].append(second_pass_qacc_linearization_error)
    buffers.trajectory_data["right_arm_second_pass_forward_dynamics_gain"].append(second_pass_forward_dynamics_gain.copy())
    buffers.trajectory_data["right_arm_second_pass_singular_values"].append(second_pass_singular_values)
    buffers.trajectory_data["right_arm_second_pass_condition_number"].append(second_pass_condition_number)
    buffers.trajectory_data["right_arm_second_pass_validation_scale"].append(second_pass_validation_scale)
    buffers.trajectory_data["right_arm_second_pass_validation_attempts"].append(second_pass_validation_attempts)
    buffers.trajectory_data["right_arm_second_pass_safe_candidate_count"].append(second_pass_safe_candidate_count)
    buffers.trajectory_data["right_arm_second_pass_total_error_rejections"].append(second_pass_total_error_rejections)
    buffers.trajectory_data["right_arm_second_pass_joint_error_rejections"].append(second_pass_joint_error_rejections)
    buffers.trajectory_data["right_arm_second_pass_qacc_limit_rejections"].append(second_pass_qacc_limit_rejections)
    buffers.trajectory_data["right_arm_forward_dynamics_safety_fallback_used"].append(forward_dynamics_safety_fallback_used)
    buffers.trajectory_data["right_arm_forward_dynamics_safety_fallback_satisfied"].append(forward_dynamics_safety_fallback_satisfied)
    buffers.trajectory_data["right_arm_forward_dynamics_safety_fallback_attempts"].append(forward_dynamics_safety_fallback_attempts)
    buffers.trajectory_data["torso_lin_vel_world"].append(torso_lin_vel)
    buffers.trajectory_data["torso_ang_vel_world"].append(torso_ang_vel)
    buffers.trajectory_data["torso_acc_world_raw"].append(torso_acc_raw)
    buffers.trajectory_data["torso_acc_world_used"].append(torso_acc_used)
    buffers.trajectory_data["torso_alpha_world_raw"].append(torso_alpha_raw)
    buffers.trajectory_data["torso_alpha_world_used"].append(torso_alpha_used)
    buffers.trajectory_data["heading_reference_world"].append(heading_reference_world)
    buffers.trajectory_data["heading_yaw_unwrapped"].append(heading_yaw_unwrapped)
    buffers.trajectory_data["heading_yaw_filtered"].append(heading_yaw_filtered)
    buffers.trajectory_data["heading_yaw_error"].append(heading_yaw_error)
    buffers.trajectory_data["heading_yaw_rate_filtered"].append(heading_yaw_rate_filtered)
    buffers.trajectory_data["heading_yaw_rate_correction"].append(heading_yaw_rate_correction)
    buffers.trajectory_data["heading_yaw_rate_command"].append(heading_yaw_rate_command)
    buffers.trajectory_data["heading_command_saturated"].append(heading_command_saturated)
    buffers.trajectory_data["right_ee_lin_vel_world"].append(right_lin_vel.copy())
    buffers.trajectory_data["right_ee_ang_vel_world"].append(right_ang_vel.copy())
    buffers.trajectory_data["right_ee_position_torso"].append(position_torso.copy())
    buffers.trajectory_data["right_ee_position_reference_torso"].append(position_reference)
    buffers.trajectory_data["right_ee_position_error_torso"].append(position_error)
    buffers.trajectory_data["right_ee_gravity_error_end"].append(gravity_error)
    buffers.trajectory_data["right_ee_upright_alignment"].append(upright_alignment)
    buffers.trajectory_data["right_lqr_one_step_q_model"].append(prediction_vector("q", 5))
    buffers.trajectory_data["right_lqr_one_step_dq_model"].append(prediction_vector("dq", 5))
    buffers.trajectory_data["right_lqr_one_step_ee_lin_acc_model"].append(prediction_vector("ee_lin_acc", 3))
    buffers.trajectory_data["right_lqr_one_step_ee_ang_acc_model"].append(prediction_vector("ee_ang_acc", 3))
    buffers.trajectory_data["right_lqr_one_step_position_error_model"].append(prediction_vector("position_error", 3))
    buffers.trajectory_data["right_lqr_one_step_gravity_error_model"].append(prediction_vector("gravity_error", 3))
    buffers.trajectory_data["right_lqr_one_step_cost_model"].append(lqr_cost_model)
    buffers.trajectory_data["right_lqr_one_step_prediction_valid"].append(lqr_prediction_valid)
    buffers.trajectory_data["arm_policy_updated"].append(arm_policy_updated)
    buffers.trajectory_data["contact_count"].append(int(data.ncon))
    buffers.eval_data["time"].append(t)
    buffers.eval_data["torso_yaw"].append(torso_yaw)
    buffers.eval_data["left_ee_lin_acc_world"].append(left_lin_acc)
    buffers.eval_data["left_ee_ang_acc_world"].append(left_ang_acc)
    buffers.eval_data["left_ee_tilt_error"].append(tilt_error_from_rot(left_rot, model.opt.gravity))
    buffers.eval_data["left_ee_upright_alignment"].append(upright_alignment_from_rot(left_rot, model.opt.gravity))
    buffers.eval_data["right_ee_lin_acc_world"].append(right_lin_acc)
    buffers.eval_data["right_ee_ang_acc_world"].append(right_ang_acc)
    buffers.eval_data["right_ee_tilt_error"].append(gravity_error)
    buffers.eval_data["right_ee_upright_alignment"].append(upright_alignment)


def finalize_run(run_dir, buffers, xml_path, simulation_dt, video_path, video_frames, video_fps, has_renderer, video_width, video_height, data, scene_ids, eval_start_time, eval_end_time, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles, gait_period, experiment_name, perf_monitor=None, lqr_cost_definition=None):
    trajectory_path = os.path.join(run_dir, "trajectory.npz")
    add_lqr_tracking_trajectory_data(buffers.trajectory_data, simulation_dt, lqr_cost_definition)
    lqr_tracking_diagnostics = compute_lqr_tracking_diagnostics(
        buffers.trajectory_data,
        eval_start_time,
        eval_end_time,
    )
    right_arm_diagnostics = save_trajectory(trajectory_path, buffers.trajectory_data, xml_path, simulation_dt)
    right_arm_diagnostics_path = save_right_arm_diagnostics(run_dir, right_arm_diagnostics)
    lqr_tracking_diagnostics_path = save_lqr_tracking_diagnostics(run_dir, lqr_tracking_diagnostics)
    lqr_ddq_tracking_plot_path = save_lqr_ddq_tracking_plot(
        run_dir,
        buffers.trajectory_data,
        lqr_tracking_diagnostics,
        eval_start_time,
        eval_end_time,
    )
    lqr_tracking_preview_path = save_lqr_tracking_preview(
        run_dir,
        buffers.trajectory_data,
        eval_start_time,
        eval_end_time,
    )
    control_preview_path = save_control_preview(run_dir, buffers.trajectory_data)
    heading_stats, heading_plot_path, heading_diagnostics_path = save_heading_control_diagnostics(
        run_dir,
        buffers.trajectory_data,
        eval_start_time,
        eval_end_time,
    )
    write_video(video_path, video_frames, video_fps)
    perf_summary_path, perf_windows_path = (None, None) if perf_monitor is None else perf_monitor.save_report(run_dir)
    walk_distance = float(np.linalg.norm(data.xpos[scene_ids.torso_id][:2] - buffers.torso_xy_start)) if buffers.torso_xy_start is not None else 0.0
    stats, saved_paths = save_eval(run_dir, buffers.eval_data, eval_start_time, eval_end_time, walk_distance, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles, gait_period, experiment_name, extra_stats=heading_stats)
    saved_paths["perf_summary"] = perf_summary_path
    saved_paths["perf_windows"] = perf_windows_path
    saved_paths["right_arm_diagnostics"] = right_arm_diagnostics_path
    saved_paths["lqr_tracking_diagnostics"] = lqr_tracking_diagnostics_path
    saved_paths["lqr_ddq_tracking_plot"] = lqr_ddq_tracking_plot_path
    saved_paths["lqr_tracking_preview"] = lqr_tracking_preview_path
    saved_paths["control_preview"] = control_preview_path
    saved_paths["heading_control_plot"] = heading_plot_path
    saved_paths["heading_control_diagnostics"] = heading_diagnostics_path
    print_run_summary(stats, saved_paths, trajectory_path, video_path, has_renderer, video_frames, video_width, video_height, walk_distance, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles)
    if heading_stats:
        print(
            "Heading hold | filtered error rms/max = "
            f"{heading_stats['heading_error_rms']:.5f}/{heading_stats['heading_error_max_abs']:.5f} rad, "
            f"yaw-rate cmd mean/range = {heading_stats['heading_yaw_rate_command_mean']:.5f}/"
            f"[{heading_stats['heading_yaw_rate_command_min']:.5f}, {heading_stats['heading_yaw_rate_command_max']:.5f}] rad/s"
        )
    if lqr_tracking_diagnostics["sample_count"]:
        ddq_tracking = lqr_tracking_diagnostics["ddq_tracking"]
        print(f"LQR DDQ tracking RMSE = {np.asarray(ddq_tracking['rmse']).round(4).tolist()}")
        print(f"LQR DDQ tracking correlation = {np.asarray(ddq_tracking['correlation']).round(4).tolist()}")
        print(f"LQR DDQ tracking gain = {np.asarray(ddq_tracking['gain']).round(4).tolist()}")
        cost_rmse = lqr_tracking_diagnostics["cost_tracking"]["rmse"]
        print(f"LQR one-step cost tracking RMSE = {dict(zip(LQR_COST_TERM_NAMES, np.asarray(cost_rmse).round(4).tolist()))}")
        validation = lqr_tracking_diagnostics.get("forward_dynamics_validation", {})
        first_selections = validation.get("first_pass", {}).get("selections", [])
        if first_selections:
            selection_text = ", ".join(
                f"{item['label']}({item['scale']:g})={item['count']} ({item['fraction'] * 100.0:.2f}%)"
                for item in first_selections
            )
            print(f"LQR forward-dynamics first-pass selections: {selection_text}")
        second = validation.get("second_pass", {})
        if second:
            print(
                "LQR forward-dynamics second pass: "
                f"triggered={second.get('triggered_count', 0)} "
                f"({second.get('triggered_fraction_of_evaluation_steps', 0.0) * 100.0:.2f}%), "
                f"accepted={second.get('accepted_count', 0)} "
                f"({second.get('accepted_fraction_given_triggered', 0.0) * 100.0:.2f}% of triggered)"
            )


def _fmt3(v):
    return f"[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]"


def _fmt2(v):
    return f"[{v[0]:.4f}, {v[1]:.4f}]"


def save_yaw_diagnostics(run_dir, time_values, yaw_values, eval_start_time, eval_end_time):
    t = np.asarray(time_values)
    yaw = np.asarray(yaw_values)
    yaw_unwrapped = np.unwrap(yaw)
    yaw_error = yaw_unwrapped - yaw_unwrapped[0]
    mask = (t >= eval_start_time) & (t < eval_end_time)
    if mask.sum() < 2:
        mask = np.ones_like(t, dtype=bool)
    slope, intercept = np.polyfit(t[mask], yaw_error[mask], 1)
    stats = {
        "yaw_mean": float(np.mean(yaw_error[mask])),
        "yaw_slope": float(slope),
        "yaw_final_drift": float(yaw_error[mask][-1]),
        "max_abs_yaw_error": float(np.max(np.abs(yaw_error[mask]))),
    }
    yaw_png = os.path.join(run_dir, "yaw.png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax in axes:
        ax.axvline(eval_start_time, color="gray", ls="--")
        ax.axvline(eval_end_time, color="gray", ls="--")
        ax.grid(True, alpha=0.3)
    axes[0].plot(t, yaw, lw=1.2); axes[0].set_title("torso yaw [rad]")
    axes[1].plot(t, yaw_error, lw=1.2); axes[1].plot(t, slope * t + intercept, ls="--", lw=1.0); axes[1].set_title("yaw error from first sample [rad]")
    axes[2].plot(t, np.abs(yaw_error), lw=1.2); axes[2].set_title("|yaw error| [rad]")
    axes[2].set_xlabel("time [s]")
    axes[1].text(0.98, 0.95, f"mean={stats['yaw_mean']:.6f}\nslope={stats['yaw_slope']:.6f}\nfinal={stats['yaw_final_drift']:.6f}\nmax_abs={stats['max_abs_yaw_error']:.6f}", transform=axes[1].transAxes, ha="right", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    fig.tight_layout(); fig.savefig(yaw_png, dpi=160); plt.close(fig)
    return stats, yaw_png


def save_heading_control_diagnostics(run_dir, trajectory_data, eval_start_time, eval_end_time):
    t = np.asarray(trajectory_data.get("time", []), dtype=np.float64)
    reference = np.asarray(trajectory_data.get("heading_reference_world", []), dtype=np.float64)
    yaw = np.asarray(trajectory_data.get("heading_yaw_unwrapped", []), dtype=np.float64)
    yaw_filtered = np.asarray(trajectory_data.get("heading_yaw_filtered", []), dtype=np.float64)
    error = np.asarray(trajectory_data.get("heading_yaw_error", []), dtype=np.float64)
    yaw_rate = np.asarray(trajectory_data.get("heading_yaw_rate_filtered", []), dtype=np.float64)
    correction = np.asarray(trajectory_data.get("heading_yaw_rate_correction", []), dtype=np.float64)
    command = np.asarray(trajectory_data.get("heading_yaw_rate_command", []), dtype=np.float64)
    saturated = np.asarray(trajectory_data.get("heading_command_saturated", []), dtype=bool)
    arrays = (reference, yaw, yaw_filtered, error, yaw_rate, correction, command, saturated)
    if not len(t) or any(value.shape != t.shape for value in arrays):
        return {}, None, None

    valid = (
        (t >= eval_start_time)
        & (t < eval_end_time)
        & np.isfinite(reference)
        & np.isfinite(yaw_filtered)
        & np.isfinite(error)
        & np.isfinite(command)
    )
    if np.count_nonzero(valid) < 2:
        return {}, None, None
    slope = float(np.polyfit(t[valid], yaw_filtered[valid] - reference[valid], 1)[0])
    stats = {
        "heading_sample_count": int(np.count_nonzero(valid)),
        "heading_error_mean": float(np.mean(error[valid])),
        "heading_error_rms": float(np.sqrt(np.mean(error[valid] ** 2))),
        "heading_error_max_abs": float(np.max(np.abs(error[valid]))),
        "heading_filtered_drift_slope": slope,
        "heading_yaw_rate_command_mean": float(np.mean(command[valid])),
        "heading_yaw_rate_command_min": float(np.min(command[valid])),
        "heading_yaw_rate_command_max": float(np.max(command[valid])),
        "heading_command_saturation_fraction": float(np.mean(saturated[valid])),
    }

    diagnostics_path = os.path.join(run_dir, "heading_control_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(stats), f, indent=2, ensure_ascii=False)

    plot_path = os.path.join(run_dir, "heading_control.png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t, yaw, lw=0.8, alpha=0.5, label="raw unwrapped yaw")
    axes[0].plot(t, yaw_filtered, lw=1.4, label="one-cycle mean yaw")
    axes[0].plot(t, reference, ls="--", lw=1.0, label="world heading reference")
    axes[0].set_ylabel("yaw [rad]")
    axes[0].legend(loc="best")
    axes[1].plot(t, error, lw=1.2, label="heading error")
    axes[1].plot(t, yaw_rate, lw=0.9, alpha=0.75, label="mean yaw rate")
    axes[1].set_ylabel("error / rate")
    axes[1].legend(loc="best")
    axes[2].plot(t, command, lw=1.2, label="runtime yaw-rate command")
    axes[2].plot(t, correction, lw=0.9, alpha=0.75, label="feedback correction")
    axes[2].set_ylabel("command [rad/s]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="best")
    for ax in axes:
        ax.axvline(eval_start_time, color="gray", ls="--")
        ax.axvline(eval_end_time, color="gray", ls="--")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return stats, plot_path, diagnostics_path


def save_eval(
    run_dir,
    data,
    eval_start_time,
    eval_end_time,
    walk_distance,
    total_cycles,
    warmup_cycles,
    evaluation_cycles,
    cooldown_cycles,
    gait_period,
    experiment_name,
    extra_stats=None,
):
    t = np.asarray(data["time"])
    mask = (t >= eval_start_time) & (t < eval_end_time)

    stats = {
        "gait_period": gait_period,
        "total_cycles": total_cycles,
        "warmup_cycles": warmup_cycles,
        "evaluation_cycles": evaluation_cycles,
        "cooldown_cycles": cooldown_cycles,
        "eval_start_time": eval_start_time,
        "eval_end_time": eval_end_time,
        "walk_distance_xy": walk_distance,
    }
    yaw_png_path = None
    if "torso_yaw" in data and len(data["torso_yaw"]) > 0:
        yaw_stats, yaw_png_path = save_yaw_diagnostics(run_dir, data["time"], data["torso_yaw"], eval_start_time, eval_end_time)
        stats.update(yaw_stats)
    if extra_stats:
        stats.update(extra_stats)

    sides = ["left", "right"]
    fig, axes = plt.subplots(6, 2, figsize=(20, 12), sharex=True)

    csv_path = os.path.join(run_dir, "metrics_preview.csv")
    png_path = os.path.join(run_dir, "metrics.png")
    npz_path = os.path.join(run_dir, "metrics.npz")
    summary_path = os.path.join(run_dir, "summary.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "side",
                "acc_x",
                "acc_y",
                "acc_z",
                "acc_norm",
                "alpha_x",
                "alpha_y",
                "alpha_z",
                "alpha_norm",
                "tilt_x",
                "tilt_y",
                "tilt_z",
                "tilt_norm",
                "upright_alignment",
            ]
        )

        for c, side in enumerate(sides):
            acc = np.asarray(data[f"{side}_ee_lin_acc_world"])
            alpha = np.asarray(data[f"{side}_ee_ang_acc_world"])
            tilt = np.asarray(data[f"{side}_ee_tilt_error"])
            alignment = np.asarray(data[f"{side}_ee_upright_alignment"])

            acc_n = np.linalg.norm(acc, axis=1)
            alpha_n = np.linalg.norm(alpha, axis=1)
            tilt_n = np.linalg.norm(tilt, axis=1)

            for i in range(len(t)):
                writer.writerow(
                    [
                        t[i],
                        side,
                        acc[i, 0],
                        acc[i, 1],
                        acc[i, 2],
                        acc_n[i],
                        alpha[i, 0],
                        alpha[i, 1],
                        alpha[i, 2],
                        alpha_n[i],
                        tilt[i, 0],
                        tilt[i, 1],
                        tilt[i, 2],
                        tilt_n[i],
                        alignment[i],
                    ]
                )

            for key, arr in [("acc", acc_n), ("alpha", alpha_n), ("tilt", tilt_n)]:
                stats[f"{side}_{key}_mean"] = arr[mask].mean()
                stats[f"{side}_{key}_std"] = arr[mask].std()
                stats[f"{side}_{key}_rms"] = np.sqrt(np.mean(arr[mask] ** 2))

            stats[f"{side}_acc_xyz_mean"] = acc[mask].mean(axis=0)
            stats[f"{side}_acc_xyz_std"] = acc[mask].std(axis=0)
            stats[f"{side}_acc_xyz_rms"] = np.sqrt(np.mean(acc[mask] ** 2, axis=0))

            stats[f"{side}_alpha_xyz_mean"] = alpha[mask].mean(axis=0)
            stats[f"{side}_alpha_xyz_std"] = alpha[mask].std(axis=0)
            stats[f"{side}_alpha_xyz_rms"] = np.sqrt(np.mean(alpha[mask] ** 2, axis=0))

            stats[f"{side}_tilt_xyz_mean"] = tilt[mask].mean(axis=0)
            stats[f"{side}_tilt_xyz_std"] = tilt[mask].std(axis=0)
            stats[f"{side}_tilt_xyz_rms"] = np.sqrt(np.mean(tilt[mask] ** 2, axis=0))
            stats[f"{side}_upright_alignment_mean"] = alignment[mask].mean()
            stats[f"{side}_upright_alignment_min"] = alignment[mask].min()
            stats[f"{side}_inverted_fraction"] = np.mean(alignment[mask] < 0.0)

            cols = ["r", "g", "b"]
            labels = ["x", "y", "z"]
            styles = ["-", "--", ":"]

            for j in range(3):
                axes[0, c].plot(t, acc[:, j], color=cols[j], ls=styles[j], lw=1.2, alpha=0.9, label=labels[j])
                axes[2, c].plot(t, alpha[:, j], color=cols[j], ls=styles[j], lw=1.2, alpha=0.9, label=labels[j])

            for j in range(3):
                axes[4, c].plot(t, tilt[:, j], color=cols[j], ls=styles[j], lw=1.2, alpha=0.9, label=f"gravity_error_{labels[j]}")

            titles = [
                f"{side} acc xyz",
                f"{side} acc norm",
                f"{side} alpha xyz",
                f"{side} alpha norm",
                f"{side} directed gravity error xyz",
                f"{side} gravity error norm",
            ]

            for r in [0, 2, 4]:
                axes[r, c].axvline(eval_start_time, color="gray", ls="--")
                axes[r, c].axvline(eval_end_time, color="gray", ls="--")
                axes[r, c].legend(loc="upper left", fontsize=8)
                axes[r, c].grid(True, alpha=0.3)

            axes[0, c].text(
                0.98,
                0.95,
                f"mean={_fmt3(stats[f'{side}_acc_xyz_mean'])}\nstd={_fmt3(stats[f'{side}_acc_xyz_std'])}\nrms={_fmt3(stats[f'{side}_acc_xyz_rms'])}",
                transform=axes[0, c].transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )
            axes[2, c].text(
                0.98,
                0.95,
                f"mean={_fmt3(stats[f'{side}_alpha_xyz_mean'])}\nstd={_fmt3(stats[f'{side}_alpha_xyz_std'])}\nrms={_fmt3(stats[f'{side}_alpha_xyz_rms'])}",
                transform=axes[2, c].transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )
            axes[4, c].text(
                0.98,
                0.95,
                f"mean={_fmt3(stats[f'{side}_tilt_xyz_mean'])}\nstd={_fmt3(stats[f'{side}_tilt_xyz_std'])}\nrms={_fmt3(stats[f'{side}_tilt_xyz_rms'])}\nalign min={stats[f'{side}_upright_alignment_min']:.3f}",
                transform=axes[4, c].transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )

            for r, y, key in [(1, acc_n, "acc"), (3, alpha_n, "alpha"), (5, tilt_n, "tilt")]:
                axes[r, c].plot(t, y, lw=1.2)
                axes[r, c].axvline(eval_start_time, color="gray", ls="--")
                axes[r, c].axvline(eval_end_time, color="gray", ls="--")
                axes[r, c].axhline(stats[f"{side}_{key}_mean"], color="r", ls="--")
                axes[r, c].grid(True, alpha=0.3)
                axes[r, c].text(
                    0.98,
                    0.95,
                    f"mean={stats[f'{side}_{key}_mean']:.6f}\nstd={stats[f'{side}_{key}_std']:.6f}\nrms={stats[f'{side}_{key}_rms']:.6f}",
                    transform=axes[r, c].transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                )

            for r in range(6):
                axes[r, c].set_title(titles[r])

    axes[5, 0].set_xlabel("time [s]")
    axes[5, 1].set_xlabel("time [s]")
    fig.suptitle(
        f"{experiment_name} | left/right palm grasp sites | "
        f"{warmup_cycles}+{evaluation_cycles}+{cooldown_cycles} cycles\n"
        f"walk distance xy = {walk_distance:.3f} m"
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    np.savez(npz_path, **data, **stats)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(stats), f, indent=2, ensure_ascii=False)

    return stats, {
        "run_dir": run_dir,
        "csv": csv_path,
        "png": png_path,
        "yaw_png": yaw_png_path,
        "npz": npz_path,
        "summary": summary_path,
    }


def print_run_summary(
    stats,
    saved_paths,
    trajectory_path,
    video_path,
    has_renderer,
    video_frames,
    video_width,
    video_height,
    walk_distance,
    total_cycles,
    warmup_cycles,
    evaluation_cycles,
    cooldown_cycles,
):
    print(f"评估已保存到目录: {saved_paths['run_dir']}")
    extra_video = video_path if video_frames else "未保存（缺少 imageio、Renderer 初始化失败或无帧）"
    extra_perf = saved_paths.get("perf_summary") if saved_paths.get("perf_summary") is not None else "未保存 perf 概览"
    extra_right_arm = saved_paths.get("right_arm_diagnostics") if saved_paths.get("right_arm_diagnostics") is not None else "未保存右臂诊断"
    extra_control_preview = saved_paths.get("control_preview") if saved_paths.get("control_preview") is not None else "未保存控制 CSV"
    extra_lqr_tracking = saved_paths.get("lqr_tracking_diagnostics") if saved_paths.get("lqr_tracking_diagnostics") is not None else "未保存 LQR tracking 诊断"
    extra_lqr_tracking_preview = saved_paths.get("lqr_tracking_preview") if saved_paths.get("lqr_tracking_preview") is not None else "未保存 LQR tracking CSV"
    extra_lqr_tracking_plot = saved_paths.get("lqr_ddq_tracking_plot") if saved_paths.get("lqr_ddq_tracking_plot") is not None else "未保存 LQR tracking 图片"
    print(
        f"文件: {saved_paths['npz']} | {saved_paths['csv']} | {saved_paths['png']} | "
        f"{saved_paths['summary']} | {extra_perf} | {extra_right_arm} | {extra_control_preview} | "
        f"{extra_lqr_tracking} | {extra_lqr_tracking_preview} | {extra_lqr_tracking_plot} | "
        f"{trajectory_path} | {extra_video}"
    )

    if has_renderer:
        print(f"视频分辨率 = {video_width}x{video_height} (受 MuJoCo offscreen framebuffer 限制)")

    for side in ["left", "right"]:
        print(f"{side} | acc mean/std/rms = {stats[f'{side}_acc_mean']:.4f}/{stats[f'{side}_acc_std']:.4f}/{stats[f'{side}_acc_rms']:.4f}")
        print(f"{side} | alpha mean/std/rms = {stats[f'{side}_alpha_mean']:.4f}/{stats[f'{side}_alpha_std']:.4f}/{stats[f'{side}_alpha_rms']:.4f}")
        print(
            f"{side} | gravity error mean/std/rms = "
            f"{stats[f'{side}_tilt_mean']:.4f}/{stats[f'{side}_tilt_std']:.4f}/{stats[f'{side}_tilt_rms']:.4f}, "
            f"upright min/inverted = {stats[f'{side}_upright_alignment_min']:.4f}/{stats[f'{side}_inverted_fraction'] * 100.0:.2f}%"
        )

    print(
        f"总周期数 = {total_cycles}, warm-up = {warmup_cycles}, evaluation = {evaluation_cycles}, "
        f"cooldown = {cooldown_cycles}, 本次仿真 torso xy 行走距离 = {walk_distance:.3f} m"
    )
