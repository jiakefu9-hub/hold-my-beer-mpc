import csv
import hashlib
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
MPC_COST_TERM_NAMES = (
    "linear_acceleration",
    "angular_acceleration",
    "angular_velocity",
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
    hold_last_safe_available: bool
    hold_last_safe_used: bool
    hold_last_safe_satisfied: bool
    hold_last_safe_qacc: np.ndarray


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


@dataclass
class ArmControllerSetup:
    """右臂控制器及其运行期配置；用于把参数装配从主循环下沉。"""

    policy: object
    policy_type: str
    notes: str
    metadata: dict
    acceleration_controller: bool
    lqr_cost_definition: Optional[dict]
    execution_perturbation: float
    execution_regularization: float
    execution_second_pass_error_threshold: float
    execution_max_joint_error: float
    execution_max_abs_qacc: float
    execution_enable_second_pass: bool
    execution_safety_rescue_passes: int
    execution_hold_last_safe: bool
    ddq_saturation_limit: float


class TorsoAccelerationFilter:
    """LQR/MPC 共用的 torso 加速度限幅与一阶滤波。"""

    def __init__(
        self,
        config,
        enabled,
        acc_alpha_key="ddq_torso_acc_filter_alpha",
        alpha_alpha_key="ddq_torso_alpha_filter_alpha",
    ):
        self.enabled = bool(enabled)
        self.acc_alpha = float(
            np.clip(config.get(acc_alpha_key, 0.20), 0.0, 1.0)
        )
        self.alpha_alpha = float(
            np.clip(config.get(alpha_alpha_key, 0.20), 0.0, 1.0)
        )
        self.acc_limit = float(config.get("ddq_torso_acc_limit", 30.0))
        self.alpha_limit = float(config.get("ddq_torso_alpha_limit", 40.0))
        self.filtered_acc = np.zeros(3, dtype=np.float64)
        self.filtered_alpha = np.zeros(3, dtype=np.float64)

    def update(self, torso_state):
        """返回原始加速度，并在启用时把 state 中的加速度替换为滤波值。"""
        raw_acc = torso_state.lin_acc.copy()
        raw_alpha = torso_state.ang_acc.copy()
        if not self.enabled:
            return raw_acc, raw_alpha

        limited_acc = np.clip(torso_state.lin_acc, -self.acc_limit, self.acc_limit)
        limited_alpha = np.clip(
            torso_state.ang_acc, -self.alpha_limit, self.alpha_limit
        )
        self.filtered_acc = (
            self.acc_alpha * limited_acc
            + (1.0 - self.acc_alpha) * self.filtered_acc
        )
        self.filtered_alpha = (
            self.alpha_alpha * limited_alpha
            + (1.0 - self.alpha_alpha) * self.filtered_alpha
        )
        torso_state.lin_acc = self.filtered_acc.copy()
        torso_state.ang_acc = self.filtered_alpha.copy()
        return raw_acc, raw_alpha


class PhaseDisturbancePredictor:
    """按步态相位插值 H 系模板，并输出世界系 MPC 扰动预测。"""

    TEMPLATE_FILES = {
        "raw": "heading_disturbance_template.npz",
        "half_smoothed": "heading_disturbance_template_half_smoothed.npz",
        "fully_smoothed": "heading_disturbance_template_fully_smoothed.npz",
    }

    def __init__(
        self,
        template_dir,
        variant,
        control_dt,
        horizon,
        acc_limit=np.inf,
        alpha_limit=np.inf,
        slow_bias_enabled=True,
        slow_bias_time_constant=0.4,
    ):
        from kinematics_helper import DisturbanceHorizon, DisturbanceInput

        self._disturbance_type = DisturbanceInput
        self._horizon_type = DisturbanceHorizon
        self.variant = str(variant).strip().lower()
        if self.variant not in self.TEMPLATE_FILES:
            raise ValueError(
                "mpc_disturbance_template 必须是 raw、half_smoothed "
                "或 fully_smoothed。"
            )
        self.path = os.path.abspath(
            os.path.join(template_dir, self.TEMPLATE_FILES[self.variant])
        )
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"找不到 MPC 扰动模板: {self.path}")
        with open(self.path, "rb") as template_file:
            self.template_sha256 = hashlib.sha256(
                template_file.read()
            ).hexdigest()

        with np.load(self.path, allow_pickle=False) as source:
            schema_version = int(
                np.asarray(source["template_schema_version"]).item()
            )
            frame_name = str(np.asarray(source["frame_name"]).item())
            stored_variant = str(np.asarray(source["template_variant"]).item())
            heading_definition = str(
                np.asarray(source["heading_definition"]).item()
            )
            self.period = float(np.asarray(source["period"]).item())
            self.source_dt = float(np.asarray(source["source_dt"]).item())
            self.interval_dt = float(
                np.asarray(source["interval_dt"]).item()
            )
            phase_reference = str(
                np.asarray(source["phase_reference"]).item()
            )
            phase_grid_convention = str(
                np.asarray(source["phase_grid_convention"]).item()
            )
            self.phase_centers = np.asarray(
                source["phase_centers"], dtype=np.float64
            ).copy()
            valid_bins = np.asarray(source["valid_bins"], dtype=bool)
            interval_valid_bins = np.asarray(
                source["interval_valid_bins"], dtype=bool
            )
            self.node_acc_template = np.asarray(
                source["torso_linear_acceleration_node_template"],
                dtype=np.float64,
            ).copy()
            self.node_omega_template = np.asarray(
                source["torso_angular_velocity_node_template"],
                dtype=np.float64,
            ).copy()
            self.node_alpha_template = np.asarray(
                source["torso_angular_acceleration_node_template"],
                dtype=np.float64,
            ).copy()
            self.interval_acc_template = np.asarray(
                source["torso_linear_acceleration_interval_template"],
                dtype=np.float64,
            ).copy()
            self.interval_omega_template = np.asarray(
                source["torso_angular_velocity_interval_template"],
                dtype=np.float64,
            ).copy()
            self.interval_alpha_template = np.asarray(
                source["torso_angular_acceleration_interval_template"],
                dtype=np.float64,
            ).copy()
            self.orientation_quaternion_template = np.asarray(
                source["torso_orientation_quaternion_template"],
                dtype=np.float64,
            ).copy()
            self.orientation_rotation_template = np.asarray(
                source["torso_orientation_rotation_matrix_template"],
                dtype=np.float64,
            ).copy()

        expected_shape = (len(self.phase_centers), 3)
        expected_quaternion_shape = (len(self.phase_centers), 4)
        expected_rotation_shape = (len(self.phase_centers), 3, 3)
        expected_heading_definition = (
            "previous_complete_gait_cycle_circular_mean_torso_yaw"
        )
        if schema_version != 2:
            raise ValueError(
                f"MPC 区间模板 schema 必须为 2，当前为 {schema_version}。"
            )
        if frame_name != "heading":
            raise ValueError(
                f"扰动模板必须使用 heading 坐标系，当前为 {frame_name!r}。"
            )
        if heading_definition != expected_heading_definition:
            raise ValueError(
                "H 系定义不匹配：模板必须使用上一完整步态周期 torso yaw "
                "的圆周均值。"
            )
        if stored_variant != self.variant:
            raise ValueError(
                f"模板类型不匹配：配置={self.variant!r}，文件={stored_variant!r}。"
            )
        if (
            phase_reference != "interval_start"
            or phase_grid_convention != "uniform_start_grid"
        ):
            raise ValueError("模板相位必须表示控制区间起点。")
        if (
            self.period <= 0.0
            or self.source_dt <= 0.0
            or not np.isclose(
                self.interval_dt,
                float(control_dt),
                rtol=1e-6,
                atol=1e-9,
            )
            or len(self.phase_centers) < 2
            or not np.all(valid_bins)
            or not np.all(interval_valid_bins)
            or self.node_acc_template.shape != expected_shape
            or self.node_omega_template.shape != expected_shape
            or self.node_alpha_template.shape != expected_shape
            or self.interval_acc_template.shape != expected_shape
            or self.interval_omega_template.shape != expected_shape
            or self.interval_alpha_template.shape != expected_shape
            or self.orientation_quaternion_template.shape
            != expected_quaternion_shape
            or self.orientation_rotation_template.shape != expected_rotation_shape
        ):
            raise ValueError("扰动模板的周期、有效 bin 或核心数组格式无效。")
        if not all(
            np.all(np.isfinite(value))
            for value in (
                self.phase_centers,
                self.node_acc_template,
                self.node_omega_template,
                self.node_alpha_template,
                self.interval_acc_template,
                self.interval_omega_template,
                self.interval_alpha_template,
                self.orientation_quaternion_template,
                self.orientation_rotation_template,
            )
        ):
            raise ValueError("扰动模板包含 NaN 或 Inf。")
        quaternion_norm_error = np.max(
            np.abs(
                np.linalg.norm(
                    self.orientation_quaternion_template, axis=1
                )
                - 1.0
            )
        )
        rotation_orthogonality_error = np.max(
            np.linalg.norm(
                np.transpose(self.orientation_rotation_template, (0, 2, 1))
                @ self.orientation_rotation_template
                - np.eye(3),
                axis=(1, 2),
            )
        )
        rotation_det_error = np.max(
            np.abs(np.linalg.det(self.orientation_rotation_template) - 1.0)
        )
        rotations_from_quaternion = np.stack(
            [
                self._quaternion_to_rotation(quaternion)
                for quaternion in self.orientation_quaternion_template
            ]
        )
        quaternion_rotation_error = np.max(
            np.linalg.norm(
                rotations_from_quaternion - self.orientation_rotation_template,
                axis=(1, 2),
            )
        )
        if (
            quaternion_norm_error > 1e-6
            or rotation_orthogonality_error > 1e-6
            or rotation_det_error > 1e-6
            or quaternion_rotation_error > 1e-6
        ):
            raise ValueError("扰动模板中的姿态四元数或旋转矩阵无效。")

        self.control_dt = float(control_dt)
        self.horizon = int(horizon)
        self.acc_limit = float(acc_limit)
        self.alpha_limit = float(alpha_limit)
        self.slow_bias_enabled = bool(slow_bias_enabled)
        self.slow_bias_time_constant = float(slow_bias_time_constant)
        if self.control_dt <= 0.0 or self.horizon < 1:
            raise ValueError("扰动预测器要求 control_dt>0 且 horizon>=1。")
        if (
            not np.isfinite(self.slow_bias_time_constant)
            or self.slow_bias_time_constant <= 0.0
        ):
            raise ValueError("slow_bias_time_constant 必须是正数。")
        self.slow_bias_update_alpha = (
            1.0 - np.exp(-self.control_dt / self.slow_bias_time_constant)
            if self.slow_bias_enabled
            else 0.0
        )
        self._slow_bias_acc = np.zeros(3, dtype=np.float64)
        self._slow_bias_omega = np.zeros(3, dtype=np.float64)
        self._slow_bias_alpha = np.zeros(3, dtype=np.float64)

        # 【核心代码】H_j 使用上一完整周期 C_{j-1} 的 torso yaw 圆周均值。
        # 第一周期只积累历史；从第二周期开始，每个周期边界更新一次并整周期保持。
        self._cycle_index = None
        self._cycle_first_time = None
        self._cycle_last_time = None
        self._yaw_sine_sum = 0.0
        self._yaw_cosine_sum = 0.0
        self._yaw_sample_count = 0
        self._heading_yaw_world = None
        self._heading_concentration = np.nan
        self._heading_activation_time = None
        self._previous_one_step_prediction = None
        self._previous_prediction_used_template = False
        self._last_prediction_diagnostics = self._empty_prediction_diagnostics()

    def predict(self, simulation_time, measured_disturbance):
        """一次生成 N+1 个节点扰动和 N 个未来 6 ms 区间扰动。"""
        measured_acc = self._vector(measured_disturbance, "acc_world")
        measured_omega = self._vector(measured_disturbance, "omega_world")
        measured_alpha = self._vector(measured_disturbance, "alpha_world")
        measured_rotation_value = getattr(
            measured_disturbance, "rot_world_body", None
        )
        if measured_rotation_value is None:
            raise ValueError("完整当前扰动 d_0 缺少 torso 姿态 R_B,0。")
        measured_rotation = np.asarray(
            measured_rotation_value, dtype=np.float64
        ).copy()
        if (
            measured_rotation.shape != (3, 3)
            or not np.all(np.isfinite(measured_rotation))
            or not np.allclose(
                measured_rotation.T @ measured_rotation,
                np.eye(3),
                atol=1e-6,
            )
            or not np.isclose(
                np.linalg.det(measured_rotation), 1.0, atol=1e-6
            )
        ):
            raise ValueError("当前 torso 姿态必须是有效的 3x3 旋转矩阵。")

        self._update_heading_frame(
            simulation_time=float(simulation_time),
            measured_rotation=measured_rotation,
        )
        if self._heading_yaw_world is None:
            prediction = self._zero_order_hold(
                measured_acc,
                measured_omega,
                measured_alpha,
                measured_rotation,
            )
            self._last_prediction_diagnostics = self._build_prediction_diagnostics(
                simulation_time=simulation_time,
                measured_acc=measured_acc,
                measured_omega=measured_omega,
                measured_alpha=measured_alpha,
                measured_rotation=measured_rotation,
                template_acc_world=None,
                template_omega_world=None,
                template_alpha_world=None,
                template_rotation_world=None,
            )
            self._remember_one_step_prediction(
                prediction.nodes[1], used_template=False
            )
            return prediction

        # ^W R_H 只含上一完整周期平均 yaw，z 轴始终与 W 系重力轴重合。
        rotation_world_heading = self._rotation_z(
            self._heading_yaw_world
        )
        phase_now = (float(simulation_time) / self.period) % 1.0
        node_acc_now = rotation_world_heading @ self._sample(
            self.node_acc_template, phase_now
        )
        node_omega_now = rotation_world_heading @ self._sample(
            self.node_omega_template, phase_now
        )
        node_alpha_now = rotation_world_heading @ self._sample(
            self.node_alpha_template, phase_now
        )
        anchor_orientation_world = rotation_world_heading @ (
            self._quaternion_to_rotation(
                self._sample_quaternion(
                    self.orientation_quaternion_template, phase_now
                )
            )
        )

        # 【核心代码】模板保留步态冲击等快速周期项；实测与 node 模板之差
        # 只经过慢 EMA 形成长期偏差，不再用滞后的 d0 平移整条预测曲线。
        beta = self.slow_bias_update_alpha
        self._slow_bias_acc = (
            (1.0 - beta) * self._slow_bias_acc
            + beta * (measured_acc - node_acc_now)
        )
        self._slow_bias_omega = (
            (1.0 - beta) * self._slow_bias_omega
            + beta * (measured_omega - node_omega_now)
        )
        self._slow_bias_alpha = (
            (1.0 - beta) * self._slow_bias_alpha
            + beta * (measured_alpha - node_alpha_now)
        )

        node_prediction = []
        for step in range(self.horizon + 1):
            phase = (
                phase_now + step * self.control_dt / self.period
            ) % 1.0
            template_acc_world = rotation_world_heading @ self._sample(
                self.node_acc_template, phase
            )
            template_omega_world = rotation_world_heading @ self._sample(
                self.node_omega_template, phase
            )
            template_alpha_world = rotation_world_heading @ self._sample(
                self.node_alpha_template, phase
            )
            acc = template_acc_world + self._slow_bias_acc
            omega = template_omega_world + self._slow_bias_omega
            alpha = template_alpha_world + self._slow_bias_alpha
            # 当前节点仍严格等于实测；只有未来节点不再被瞬时误差整体平移。
            if step == 0:
                acc = measured_acc.copy()
                omega = measured_omega.copy()
                alpha = measured_alpha.copy()
            acc = np.clip(acc, -self.acc_limit, self.acc_limit)
            alpha = np.clip(alpha, -self.alpha_limit, self.alpha_limit)

            template_orientation_world = rotation_world_heading @ (
                self._quaternion_to_rotation(
                    self._sample_quaternion(
                        self.orientation_quaternion_template, phase
                    )
                )
            )
            # 姿态也先做 H→W，再以当前实测姿态锚定模板相对转动。
            # 同一个 ^W R_H 在相对转动中会严格抵消，但保留显式转换更易核对。
            rotation = (
                measured_rotation
                @ anchor_orientation_world.T
                @ template_orientation_world
            )

            node_prediction.append(
                self._disturbance_type(
                    acc_world=acc,
                    omega_world=omega,
                    alpha_world=alpha,
                    rot_world_body=rotation.copy(),
                )
            )

        interval_prediction = []
        for step in range(self.horizon):
            phase = (
                phase_now + step * self.control_dt / self.period
            ) % 1.0
            midpoint_phase = (
                phase + 0.5 * self.control_dt / self.period
            ) % 1.0
            interval_acc = rotation_world_heading @ self._sample(
                self.interval_acc_template, phase
            )
            interval_omega = rotation_world_heading @ self._sample(
                self.interval_omega_template, phase
            )
            interval_alpha = rotation_world_heading @ self._sample(
                self.interval_alpha_template, phase
            )
            interval_acc = np.clip(
                interval_acc + self._slow_bias_acc,
                -self.acc_limit,
                self.acc_limit,
            )
            interval_omega = interval_omega + self._slow_bias_omega
            interval_alpha = np.clip(
                interval_alpha + self._slow_bias_alpha,
                -self.alpha_limit,
                self.alpha_limit,
            )
            midpoint_orientation_world = rotation_world_heading @ (
                self._quaternion_to_rotation(
                    self._sample_quaternion(
                        self.orientation_quaternion_template,
                        midpoint_phase,
                    )
                )
            )
            interval_rotation = (
                measured_rotation
                @ anchor_orientation_world.T
                @ midpoint_orientation_world
            )
            interval_prediction.append(
                self._disturbance_type(
                    acc_world=interval_acc,
                    omega_world=interval_omega,
                    alpha_world=interval_alpha,
                    rot_world_body=interval_rotation,
                )
            )

        prediction = self._horizon_type(
            nodes=tuple(node_prediction),
            intervals=tuple(interval_prediction),
        )
        self._last_prediction_diagnostics = self._build_prediction_diagnostics(
            simulation_time=simulation_time,
            measured_acc=measured_acc,
            measured_omega=measured_omega,
            measured_alpha=measured_alpha,
            measured_rotation=measured_rotation,
            template_acc_world=node_acc_now,
            template_omega_world=node_omega_now,
            template_alpha_world=node_alpha_now,
            template_rotation_world=anchor_orientation_world,
        )
        self._remember_one_step_prediction(
            prediction.nodes[1], used_template=True
        )
        return prediction

    def metadata(self):
        return {
            "enabled": True,
            "variant": self.variant,
            "path": self.path,
            "sha256": self.template_sha256,
            "template_frame": "heading",
            "controller_output_frame": "world",
            "period": self.period,
            "source_dt": self.source_dt,
            "interval_dt": self.interval_dt,
            "num_bins": int(len(self.phase_centers)),
            "heading_definition": (
                "previous_complete_gait_cycle_circular_mean_torso_yaw"
            ),
            "heading_update": "once_per_gait_cycle_and_held_within_cycle",
            "initialization": (
                "zero_order_hold_until_first_complete_gait_cycle"
            ),
            "prediction": (
                "node_and_future_interval_template_plus_slow_bias"
                if self.slow_bias_enabled
                else "node_and_future_interval_template_without_slow_bias"
            ),
            "node_definition": "instantaneous_at_t_k",
            "interval_definition": "average_over_[t_k,t_k+control_dt)",
            "anchor_mode": (
                "exact_measured_node0_plus_slow_bias"
                if self.slow_bias_enabled
                else "exact_measured_node0_template_only_future"
            ),
            "slow_bias_enabled": self.slow_bias_enabled,
            "slow_bias_time_constant": self.slow_bias_time_constant,
            "slow_bias_update_alpha": self.slow_bias_update_alpha,
            "phase_source": "counter_times_simulation_dt",
            "periodic_interpolation": "linear_on_interval_start_grid",
            "rotation_prediction": "measurement_anchored_quaternion_template",
            "orientation_interpolation": "shortest_path_slerp",
        }

    def runtime_state(self):
        """【非核心代码】返回 H 系初始化状态，便于日志与诊断。"""
        return {
            "heading_ready": self._heading_yaw_world is not None,
            "heading_yaw_world": (
                np.nan
                if self._heading_yaw_world is None
                else float(self._heading_yaw_world)
            ),
            "heading_concentration": float(self._heading_concentration),
            "heading_activation_time": (
                np.nan
                if self._heading_activation_time is None
                else float(self._heading_activation_time)
            ),
            "slow_bias_acc_world": self._slow_bias_acc.copy(),
            "slow_bias_omega_world": self._slow_bias_omega.copy(),
            "slow_bias_alpha_world": self._slow_bias_alpha.copy(),
        }

    def get_last_diagnostics(self):
        """【非核心代码】返回模板当前偏差和上一拍的一步预测误差。"""
        copied = {}
        for name, value in self._last_prediction_diagnostics.items():
            copied[name] = value.copy() if isinstance(value, np.ndarray) else value
        return copied

    def _build_prediction_diagnostics(
        self,
        simulation_time,
        measured_acc,
        measured_omega,
        measured_alpha,
        measured_rotation,
        template_acc_world,
        template_omega_world,
        template_alpha_world,
        template_rotation_world,
    ):
        """【半核心代码】区分模板绝对偏差与真正的一步预测误差。"""
        nan_vector = np.full(3, np.nan, dtype=np.float64)
        template_ready = template_acc_world is not None
        previous = self._previous_one_step_prediction
        one_step_valid = (
            previous is not None and self._previous_prediction_used_template
        )

        diagnostics = {
            "heading_ready": bool(template_ready),
            "phase": (float(simulation_time) / self.period) % 1.0,
            "heading_yaw_world": (
                np.nan
                if self._heading_yaw_world is None
                else float(self._heading_yaw_world)
            ),
            "template_acc_world": (
                nan_vector.copy()
                if template_acc_world is None
                else np.asarray(template_acc_world, dtype=np.float64).copy()
            ),
            "template_omega_world": (
                nan_vector.copy()
                if template_omega_world is None
                else np.asarray(template_omega_world, dtype=np.float64).copy()
            ),
            "template_alpha_world": (
                nan_vector.copy()
                if template_alpha_world is None
                else np.asarray(template_alpha_world, dtype=np.float64).copy()
            ),
            "anchor_acc_error": (
                nan_vector.copy()
                if template_acc_world is None
                else measured_acc - template_acc_world
            ),
            "anchor_omega_error": (
                nan_vector.copy()
                if template_omega_world is None
                else measured_omega - template_omega_world
            ),
            "anchor_alpha_error": (
                nan_vector.copy()
                if template_alpha_world is None
                else measured_alpha - template_alpha_world
            ),
            "anchor_rotation_error_angle": (
                np.nan
                if template_rotation_world is None
                else self._rotation_error_angle(
                    measured_rotation, template_rotation_world
                )
            ),
            "slow_bias_acc_world": self._slow_bias_acc.copy(),
            "slow_bias_omega_world": self._slow_bias_omega.copy(),
            "slow_bias_alpha_world": self._slow_bias_alpha.copy(),
            "one_step_prediction_valid": bool(one_step_valid),
            "one_step_acc_error": (
                measured_acc - previous["acc_world"]
                if one_step_valid
                else nan_vector.copy()
            ),
            "one_step_omega_error": (
                measured_omega - previous["omega_world"]
                if one_step_valid
                else nan_vector.copy()
            ),
            "one_step_alpha_error": (
                measured_alpha - previous["alpha_world"]
                if one_step_valid
                else nan_vector.copy()
            ),
            "one_step_rotation_error_angle": (
                self._rotation_error_angle(
                    measured_rotation, previous["rot_world_body"]
                )
                if one_step_valid
                else np.nan
            ),
        }
        return diagnostics

    def _remember_one_step_prediction(self, prediction, used_template):
        self._previous_one_step_prediction = {
            "acc_world": np.asarray(prediction.acc_world, dtype=np.float64).copy(),
            "omega_world": np.asarray(
                prediction.omega_world, dtype=np.float64
            ).copy(),
            "alpha_world": np.asarray(
                prediction.alpha_world, dtype=np.float64
            ).copy(),
            "rot_world_body": np.asarray(
                prediction.rot_world_body, dtype=np.float64
            ).copy(),
        }
        self._previous_prediction_used_template = bool(used_template)

    @staticmethod
    def _rotation_error_angle(measured_rotation, predicted_rotation):
        relative = np.asarray(measured_rotation) @ np.asarray(predicted_rotation).T
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.arccos(cosine))

    @staticmethod
    def _empty_prediction_diagnostics():
        nan_vector = np.full(3, np.nan, dtype=np.float64)
        return {
            "heading_ready": False,
            "phase": np.nan,
            "heading_yaw_world": np.nan,
            "template_acc_world": nan_vector.copy(),
            "template_omega_world": nan_vector.copy(),
            "template_alpha_world": nan_vector.copy(),
            "anchor_acc_error": nan_vector.copy(),
            "anchor_omega_error": nan_vector.copy(),
            "anchor_alpha_error": nan_vector.copy(),
            "anchor_rotation_error_angle": np.nan,
            "slow_bias_acc_world": nan_vector.copy(),
            "slow_bias_omega_world": nan_vector.copy(),
            "slow_bias_alpha_world": nan_vector.copy(),
            "one_step_prediction_valid": False,
            "one_step_acc_error": nan_vector.copy(),
            "one_step_omega_error": nan_vector.copy(),
            "one_step_alpha_error": nan_vector.copy(),
            "one_step_rotation_error_angle": np.nan,
        }

    def _update_heading_frame(self, simulation_time, measured_rotation):
        """在周期边界用刚结束的完整周期更新当前 H 系。"""
        if not np.isfinite(simulation_time) or simulation_time < 0.0:
            raise ValueError("simulation_time 必须是非负有限数。")
        cycle_index = int(np.floor(simulation_time / self.period + 1e-12))
        yaw_world = float(
            np.arctan2(measured_rotation[1, 0], measured_rotation[0, 0])
        )

        if self._cycle_index is None:
            self._cycle_index = cycle_index
            self._cycle_first_time = simulation_time
        elif cycle_index < self._cycle_index:
            raise ValueError("扰动预测器不支持仿真时间倒退。")
        elif cycle_index > self._cycle_index:
            covered_duration = (
                0.0
                if self._cycle_last_time is None
                else self._cycle_last_time
                - self._cycle_first_time
                + self.control_dt
            )
            enough_samples = self._yaw_sample_count >= max(
                2, int(np.floor(0.9 * self.period / self.control_dt))
            )
            if enough_samples and covered_duration >= 0.9 * self.period:
                concentration = np.hypot(
                    self._yaw_sine_sum, self._yaw_cosine_sum
                ) / self._yaw_sample_count
                if concentration >= 1e-6:
                    self._heading_yaw_world = float(
                        np.arctan2(
                            self._yaw_sine_sum,
                            self._yaw_cosine_sum,
                        )
                    )
                    self._heading_concentration = float(concentration)
                    if self._heading_activation_time is None:
                        self._heading_activation_time = simulation_time

            self._cycle_index = cycle_index
            self._cycle_first_time = simulation_time
            self._cycle_last_time = None
            self._yaw_sine_sum = 0.0
            self._yaw_cosine_sum = 0.0
            self._yaw_sample_count = 0

        self._yaw_sine_sum += np.sin(yaw_world)
        self._yaw_cosine_sum += np.cos(yaw_world)
        self._yaw_sample_count += 1
        self._cycle_last_time = simulation_time

    def _zero_order_hold(
        self,
        measured_acc,
        measured_omega,
        measured_alpha,
        measured_rotation,
    ):
        nodes = tuple(
            self._disturbance_type(
                acc_world=measured_acc.copy(),
                omega_world=measured_omega.copy(),
                alpha_world=measured_alpha.copy(),
                rot_world_body=measured_rotation.copy(),
            )
            for _ in range(self.horizon + 1)
        )
        intervals = tuple(
            self._disturbance_type(
                acc_world=measured_acc.copy(),
                omega_world=measured_omega.copy(),
                alpha_world=measured_alpha.copy(),
                rot_world_body=measured_rotation.copy(),
            )
            for _ in range(self.horizon)
        )
        return self._horizon_type(nodes=nodes, intervals=intervals)

    def _sample(self, values, phase):
        lower, upper, fraction = self._phase_bracket(phase)
        return (1.0 - fraction) * values[lower] + fraction * values[upper]

    def _sample_quaternion(self, quaternions, phase):
        # 四元数不能直接线性插值；使用最短路径 SLERP，并处理 q/-q 二义性。
        lower, upper, fraction = self._phase_bracket(phase)
        q0 = np.asarray(quaternions[lower], dtype=np.float64)
        q1 = np.asarray(quaternions[upper], dtype=np.float64)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            result = (1.0 - fraction) * q0 + fraction * q1
            return result / np.linalg.norm(result)
        angle = np.arccos(dot)
        scale = np.sin(angle)
        return (
            np.sin((1.0 - fraction) * angle) * q0
            + np.sin(fraction * angle) * q1
        ) / scale

    def _phase_bracket(self, phase):
        """周期插值任意相位，当前时刻无需对齐到 2 ms 模板网格。"""
        centers = self.phase_centers
        value = float(phase) % 1.0
        upper_unwrapped = int(
            np.searchsorted(centers, value, side="right")
        )
        lower = (upper_unwrapped - 1) % len(centers)
        upper = upper_unwrapped % len(centers)
        lower_phase = float(centers[lower])
        upper_phase = float(centers[upper])
        if upper_unwrapped == 0:
            lower_phase -= 1.0
        elif upper_unwrapped == len(centers):
            upper_phase += 1.0
        width = upper_phase - lower_phase
        if width <= 0.0:
            raise ValueError("模板 phase_centers 必须严格递增。")
        fraction = (value - lower_phase) / width
        return lower, upper, float(np.clip(fraction, 0.0, 1.0))

    @staticmethod
    def _vector(disturbance, name):
        value = None if disturbance is None else getattr(disturbance, name, None)
        vector = (
            np.zeros(3, dtype=np.float64)
            if value is None
            else np.asarray(value, dtype=np.float64)
        )
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"measured_disturbance.{name} 必须是有限的三维向量。")
        return vector

    @staticmethod
    def _quaternion_to_rotation(quaternion):
        quaternion = np.asarray(quaternion, dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("姿态四元数必须是有限的 wxyz 四维向量。")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            raise ValueError("姿态四元数范数不能为零。")
        w, x, y, z = quaternion / norm
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_z(angle):
        cosine = np.cos(float(angle))
        sine = np.sin(float(angle))
        return np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def create_arm_controller(config, controller_name, default_q, control_dt):
    """【半核心代码】从 YAML 参数创建 PID/LQR/MPC，并集中生成实验元数据。"""
    from arm_lqr import ArmLQRPolicy
    from arm_mpc import ArmMPCPolicy
    from arm_pid import ArmPIDPolicy

    controller_name = str(controller_name).lower()
    if controller_name not in {"pid", "lqr", "mpc"}:
        raise ValueError(
            f"arm_controller={controller_name!r} 无效，只能选择 'pid'、'lqr' 或 'mpc'。"
        )

    perturbation = float(config.get("ddq_execution_perturbation", 0.1))
    regularization = float(config.get("ddq_execution_regularization", 5.0))
    second_pass_threshold = float(
        config.get("ddq_execution_second_pass_error_threshold", 5.0)
    )
    max_joint_error = float(config.get("ddq_execution_max_joint_error", 4.0))
    max_abs_qacc = float(config.get("ddq_execution_max_abs_qacc", 8.0))
    enable_second_pass = bool(
        config.get(
            f"{controller_name}_execution_enable_second_pass",
            controller_name == "lqr",
        )
    )
    safety_rescue_passes = int(
        config.get(
            f"{controller_name}_execution_safety_rescue_passes",
            2 if controller_name == "lqr" else 0,
        )
    )
    hold_last_safe = bool(
        config.get(
            f"{controller_name}_execution_hold_last_safe",
            controller_name == "mpc",
        )
    )
    acc_filter_key = (
        "mpc_torso_acc_filter_alpha"
        if controller_name == "mpc"
        else "ddq_torso_acc_filter_alpha"
    )
    alpha_filter_key = (
        "mpc_torso_alpha_filter_alpha"
        if controller_name == "mpc"
        else "ddq_torso_alpha_filter_alpha"
    )
    torso_metadata = {
        "torso_acc_filter_alpha": float(
            config.get(acc_filter_key, 0.20)
        ),
        "torso_alpha_filter_alpha": float(
            config.get(alpha_filter_key, 0.20)
        ),
        "torso_acc_limit": float(config.get("ddq_torso_acc_limit", 30.0)),
        "torso_alpha_limit": float(config.get("ddq_torso_alpha_limit", 40.0)),
    }
    execution_metadata = {
        "forward_dynamics_perturbation_nm": perturbation,
        "forward_dynamics_regularization": regularization,
        "forward_dynamics_second_pass_error_threshold": second_pass_threshold,
        "forward_dynamics_max_joint_error": max_joint_error,
        "forward_dynamics_max_abs_qacc": max_abs_qacc,
        "forward_dynamics_enable_second_pass": enable_second_pass,
        "forward_dynamics_safety_rescue_passes": safety_rescue_passes,
        "forward_dynamics_hold_last_safe": hold_last_safe,
    }

    lqr_cost_definition = None
    ddq_saturation_limit = np.inf
    if controller_name == "lqr":
        kwargs = {
            "horizon": int(config.get("lqr_horizon", 12)),
            "q_ee_acc": float(config.get("lqr_q_ee_acc", 1.0)),
            "q_ee_alpha": float(config.get("lqr_q_ee_alpha", 0.05)),
            "q_position": float(config.get("lqr_q_position", 20.0)),
            "q_gravity": float(config.get("lqr_q_gravity", 30.0)),
            "q_posture": config.get("lqr_q_posture", 0.4),
            "q_vel": float(config.get("lqr_q_vel", 0.02)),
            "r_ddq": float(config.get("lqr_r_ddq", 0.25)),
            "terminal_scale": float(config.get("lqr_terminal_scale", 2.0)),
            "reg": float(config.get("lqr_reg", 1e-6)),
            "max_ddq": float(config.get("lqr_max_ddq", 3.0)),
            "max_dq": float(config.get("lqr_max_dq", 1.0)),
            "ddq_rate_limit": float(config.get("lqr_ddq_rate_limit", 350.0)),
            "ddq_smoothing_alpha": float(
                config.get("lqr_ddq_smoothing_alpha", 0.45)
            ),
            "joint_limit_margin": float(config.get("lqr_joint_limit_margin", 0.25)),
            "joint_limit_stiffness": float(
                config.get("lqr_joint_limit_stiffness", 8.0)
            ),
            "joint_limit_damping": float(
                config.get("lqr_joint_limit_damping", 2.0)
            ),
        }
        policy = ArmLQRPolicy(default_q=default_q, control_dt=control_dt, **kwargs)
        lqr_cost_definition = policy.get_cost_definition()
        policy_type = "ArmLQRPolicy"
        notes = "finite-horizon time-varying LQR with per-joint posture weights, torso-relative end-effector position cost, directed 3D gravity error, fully bypassed ddq post-processing, protected torso disturbance inputs, and validated local MuJoCo forward-dynamics torque mapping initialized by non-friction-constraint-aware inverse dynamics plus joint-space PD"
        metadata = {
            "lqr_config": {
                **kwargs,
                "control_dt": policy.control_dt,
                **torso_metadata,
                **execution_metadata,
                "torque_control": "mujoco_local_forward_dynamics_mapping_from_inverse_dynamics_nominal",
                "constraint_aware_tau_formula": "qfrc_inverse + qfrc_constraint_nonfriction",
                "constraints_added_back": "contact + joint/tendon limit + equality",
                "constraints_excluded_from_addback": "FRICTION_DOF + FRICTION_TENDON",
                "forward_dynamics_mapping": "ddq_right ~= ddq_baseline + G_tau * delta_tau, then evaluate every full mj_forward candidate and select the minimum-error safe candidate",
                "forward_dynamics_validation_scales": [1.0, 0.5, 0.25, 0.125],
                "forward_dynamics_candidate_selection": "minimum total error among candidates that improve total error and satisfy joint-error/qacc limits",
                "forward_dynamics_evaluations_per_step": "10; plus 9 when the second pass triggers; up to two additional 9-evaluation safety-rescue passes only if final qacc exceeds the limit",
                "uncontrolled_qacc_assumption": 0.0,
                "gravity_error": "directed_3d",
                "position_reference_q": np.asarray(default_q).copy(),
                "torso_acc_source": "mujoco_imu_accelerometer_world_without_gravity",
                "torso_alpha_source": "finite_difference_world_angular_velocity",
                "position_reference_frame": "torso_imu",
                "end_effector_velocity_cost_enabled": False,
                "ddq_post_process": "fully_bypassed",
                "ddq_hard_clip_enabled": False,
                "joint_limit_guard": "disabled",
                "ddq_rate_limit_enabled": False,
                "ddq_smoothing_enabled": False,
                "ddq_tracking": "6ms_velocity_difference_aligned_between_consecutive_arm_updates",
                "cost_tracking": "one_step_model_vs_realized_next_arm_update",
            }
        }
    elif controller_name == "mpc":
        feedforward_enabled = bool(
            config.get("mpc_disturbance_feedforward_enabled", False)
        )
        q_min = np.deg2rad(
            np.asarray(
                config.get(
                    "mpc_q_min_deg", [-20.0, -10.0, -20.0, -20.0, -20.0]
                ),
                dtype=np.float64,
            )
        )
        q_max = np.deg2rad(
            np.asarray(
                config.get("mpc_q_max_deg", [10.0, 5.0, 5.0, 20.0, 20.0]),
                dtype=np.float64,
            )
        )
        if q_min.shape != (5,) or q_max.shape != (5,) or np.any(q_min >= q_max):
            raise ValueError(
                "mpc_q_min_deg/mpc_q_max_deg 必须是长度为 5 且下界小于上界的数组。"
            )
        q_operating_margin = np.deg2rad(
            np.asarray(
                config.get("mpc_q_operating_margin_deg", np.zeros(5)),
                dtype=np.float64,
            )
        )
        if (
            q_operating_margin.shape != (5,)
            or np.any(q_operating_margin < 0.0)
            or np.any(2.0 * q_operating_margin >= q_max - q_min)
        ):
            raise ValueError(
                "mpc_q_operating_margin_deg 必须是长度为 5 的非负数组，"
                "且两侧裕量之和必须小于对应安全范围。"
            )
        kwargs = {
            "horizon": int(config.get("mpc_horizon", 12)),
            "q_ee_acc": config.get("mpc_q_ee_acc", 1.0),
            "q_ee_alpha": config.get("mpc_q_ee_alpha", 0.075),
            # 默认关闭；需要时可用标量、3 维对角权重或 3x3 矩阵配置。
            "q_ee_omega": config.get("mpc_q_ee_omega", 0.0),
            "q_gravity": config.get("mpc_q_gravity", 30.0),
            "q_posture": config.get("mpc_q_posture", 0.05),
            "q_vel": config.get("mpc_q_vel", 0.02),
            "r_ddq": config.get("mpc_r_ddq", 0.25),
            "terminal_scale": float(config.get("mpc_terminal_scale", 2.0)),
            "reg": float(config.get("mpc_qp_regularization", 1e-6)),
            "max_dq": float(config.get("mpc_max_dq", 1.0)),
            "max_ddq": float(config.get("mpc_max_ddq", 8.0)),
            "joint_limits": np.column_stack([q_min, q_max]),
            "joint_limit_margin": q_operating_margin,
            "solver_max_iter": int(config.get("mpc_osqp_max_iter", 400)),
            "solver_rho": float(config.get("mpc_osqp_rho", 0.1)),
            "solver_adaptive_rho": bool(
                config.get("mpc_osqp_adaptive_rho", True)
            ),
            "solver_scaled_termination": bool(
                config.get("mpc_osqp_scaled_termination", False)
            ),
            "solver_eps_abs": float(config.get("mpc_osqp_eps_abs", 1e-3)),
            "solver_eps_rel": float(config.get("mpc_osqp_eps_rel", 1e-3)),
            "solver_polishing": bool(config.get("mpc_osqp_polishing", False)),
            "failure_braking_gain": float(
                config.get("mpc_failure_braking_gain", 4.0)
            ),
            "failure_posture_gain": float(
                config.get("mpc_failure_posture_gain", 4.0)
            ),
            "failure_max_ddq_scale": float(
                config.get("mpc_failure_max_ddq_scale", 0.5)
            ),
        }
        policy = ArmMPCPolicy(default_q=default_q, control_dt=control_dt, **kwargs)
        policy_type = "ArmMPCPolicy"
        notes = (
            "sparse constrained right-arm MPC with ddq input, "
            + (
                "heading-frame phase-template base-disturbance feedforward, "
                if feedforward_enabled
                else "zero-order-held measured base motion, "
            )
            + "configurable world-frame end-effector angular-velocity cost, "
            + "2D gravity error, no end-effector position cost, and validated "
            + "inverse/forward-dynamics torque execution"
        )
        ddq_saturation_limit = kwargs["max_ddq"]
        metadata = {
            "mpc_config": {
                **kwargs,
                "control_dt": control_dt,
                **torso_metadata,
                **execution_metadata,
                "solver": "OSQP",
                "base_prediction": (
                    "heading_to_world_measurement_anchored_phase_template"
                    if feedforward_enabled
                    else "zero_order_hold_current_measurement"
                ),
                "gravity_error": "signed_2d_xy",
                "torso_relative_position_cost": False,
            }
        }
    else:
        kwargs = {
            "kp_pose": config.get("pid_kp_pose", [1.20, 1.20]),
            "kd_pose": config.get("pid_kd_pose", [1.20, 1.20]),
            "ki_pose": config.get("pid_ki_pose", [0.0, 0.0]),
            "posture_gain": config.get(
                "pid_posture_gain", [1.15, 1.15, 2.10, 1.15, 0.95]
            ),
            "finite_diff_eps": float(config.get("pid_finite_diff_eps", 1e-4)),
            "damping": float(config.get("pid_damping", 0.15)),
            "integral_limit": float(config.get("pid_integral_limit", 0.20)),
            "max_dq": float(config.get("pid_max_dq", 0.48)),
            "de_g_alpha": float(config.get("pid_de_g_alpha", 0.07)),
        }
        policy = ArmPIDPolicy(default_q=default_q, control_dt=control_dt, **kwargs)
        policy_type = "ArmPIDPolicy"
        notes = "tuning_v7: continue monotonic conservative tuning with a slightly lower Kp, larger damped-pinv damping, tighter max_dq, slightly stronger de_g filtering, and a mild posture-regularization increase to further reduce right-hand acceleration while keeping tilt small"
        metadata = {"pid_config": {"default_q": np.asarray(default_q).copy(), **kwargs}}

    return ArmControllerSetup(
        policy=policy,
        policy_type=policy_type,
        notes=notes,
        metadata=metadata,
        acceleration_controller=controller_name in {"lqr", "mpc"},
        lqr_cost_definition=lqr_cost_definition,
        execution_perturbation=perturbation,
        execution_regularization=regularization,
        execution_second_pass_error_threshold=second_pass_threshold,
        execution_max_joint_error=max_joint_error,
        execution_max_abs_qacc=max_abs_qacc,
        execution_enable_second_pass=enable_second_pass,
        execution_safety_rescue_passes=safety_rescue_passes,
        execution_hold_last_safe=hold_last_safe,
        ddq_saturation_limit=ddq_saturation_limit,
    )


def build_right_arm_control_record(
    *,
    arm_policy_updated,
    target_q,
    target_dq,
    ddq_raw,
    ddq_des,
    ddq_saturation_limit,
    tau_pd,
    torque_limits,
    torso_state,
    raw_torso_acc,
    raw_torso_alpha,
    heading_state,
    heading_yaw_rate_command,
    ee_position_reference_torso,
    inverse_result=None,
    mapping_result=None,
    lqr_one_step_prediction=None,
    mpc_diagnostics=None,
):
    """【非核心代码】把执行结果展开成 record_eval_step 所需的日志字典。"""
    zeros = np.zeros(5, dtype=np.float64)
    zero_matrix = np.zeros((5, 5), dtype=np.float64)

    if inverse_result is None:
        inverse_values = {
            "tau_inverse": zeros,
            "tau_contact": zeros,
            "tau_constraint_total": zeros,
            "tau_constraint_noncontact": zeros,
            "tau_constraint_nonfriction": zeros,
            "tau_constraint_friction": zeros,
            "tau_ff": zeros,
        }
    else:
        inverse_values = {
            "tau_inverse": inverse_result.tau_inverse,
            "tau_contact": inverse_result.tau_contact,
            "tau_constraint_total": inverse_result.tau_constraint_total,
            "tau_constraint_noncontact": inverse_result.tau_constraint_noncontact,
            "tau_constraint_nonfriction": inverse_result.tau_constraint_nonfriction,
            "tau_constraint_friction": inverse_result.tau_constraint_friction,
            "tau_ff": inverse_result.tau_ff,
        }

    if mapping_result is None:
        mapping_values = {
            "tau_nominal": tau_pd,
            "tau_mapping_correction_raw": zeros,
            "tau_mapping_correction": zeros,
            "tau_cmd_raw": tau_pd,
            "qacc_mapping_baseline": zeros,
            "qacc_mapping_predicted": zeros,
            "qacc_mapping_prediction_error": zeros,
            "qacc_mapping_validated": zeros,
            "qacc_mapping_validation_error": zeros,
            "qacc_mapping_linearization_error": zeros,
            "forward_dynamics_gain": zero_matrix,
            "forward_dynamics_singular_values": zeros,
            "forward_dynamics_condition_number": np.inf,
            "forward_dynamics_validation_scale": 0.0,
            "forward_dynamics_validation_attempts": 0,
            "forward_dynamics_validation_improved": False,
            "forward_dynamics_tracking_safety_satisfied": False,
            "forward_dynamics_qacc_safety_satisfied": False,
            "forward_dynamics_safe_candidate_count": 0,
            "forward_dynamics_total_error_rejections": 0,
            "forward_dynamics_joint_error_rejections": 0,
            "forward_dynamics_qacc_limit_rejections": 0,
            "first_pass_qacc_validated": zeros,
            "first_pass_qacc_validation_error": zeros,
            "forward_dynamics_second_pass_triggered": False,
            "forward_dynamics_second_pass_accepted": False,
            "second_pass_tracking_safety_satisfied": False,
            "second_pass_qacc_safety_satisfied": False,
            "second_pass_tau_correction_raw": zeros,
            "second_pass_tau_correction": zeros,
            "second_pass_qacc_predicted": zeros,
            "second_pass_qacc_validated": zeros,
            "second_pass_qacc_validation_error": zeros,
            "second_pass_qacc_linearization_error": zeros,
            "second_pass_forward_dynamics_gain": zero_matrix,
            "second_pass_singular_values": zeros,
            "second_pass_condition_number": np.inf,
            "second_pass_validation_scale": 0.0,
            "second_pass_validation_attempts": 0,
            "second_pass_safe_candidate_count": 0,
            "second_pass_total_error_rejections": 0,
            "second_pass_joint_error_rejections": 0,
            "second_pass_qacc_limit_rejections": 0,
            "forward_dynamics_safety_fallback_used": False,
            "forward_dynamics_safety_fallback_satisfied": False,
            "forward_dynamics_safety_fallback_attempts": 0,
            "forward_dynamics_hold_last_safe_available": False,
            "forward_dynamics_hold_last_safe_used": False,
            "forward_dynamics_hold_last_safe_satisfied": False,
            "forward_dynamics_hold_last_safe_qacc": zeros,
        }
    else:
        mapping_values = {
            "tau_nominal": mapping_result.tau_nominal,
            "tau_mapping_correction_raw": mapping_result.tau_correction_raw,
            "tau_mapping_correction": mapping_result.tau_correction,
            "tau_cmd_raw": mapping_result.tau_cmd_raw,
            "qacc_mapping_baseline": mapping_result.qacc_baseline,
            "qacc_mapping_predicted": mapping_result.qacc_predicted,
            "qacc_mapping_prediction_error": mapping_result.qacc_prediction_error,
            "qacc_mapping_validated": mapping_result.qacc_validated,
            "qacc_mapping_validation_error": mapping_result.qacc_validation_error,
            "qacc_mapping_linearization_error": mapping_result.qacc_linearization_error,
            "forward_dynamics_gain": mapping_result.gain_matrix,
            "forward_dynamics_singular_values": mapping_result.singular_values,
            "forward_dynamics_condition_number": mapping_result.condition_number,
            "forward_dynamics_validation_scale": mapping_result.validation_scale,
            "forward_dynamics_validation_attempts": mapping_result.validation_attempts,
            "forward_dynamics_validation_improved": mapping_result.validation_improved,
            "forward_dynamics_tracking_safety_satisfied": mapping_result.validation_tracking_safety_satisfied,
            "forward_dynamics_qacc_safety_satisfied": mapping_result.validation_qacc_safety_satisfied,
            "forward_dynamics_safe_candidate_count": mapping_result.validation_safe_candidate_count,
            "forward_dynamics_total_error_rejections": mapping_result.validation_total_error_rejections,
            "forward_dynamics_joint_error_rejections": mapping_result.validation_joint_error_rejections,
            "forward_dynamics_qacc_limit_rejections": mapping_result.validation_qacc_limit_rejections,
            "first_pass_qacc_validated": mapping_result.first_pass_qacc_validated,
            "first_pass_qacc_validation_error": mapping_result.first_pass_qacc_validation_error,
            "forward_dynamics_second_pass_triggered": mapping_result.second_pass_triggered,
            "forward_dynamics_second_pass_accepted": mapping_result.second_pass_accepted,
            "second_pass_tracking_safety_satisfied": mapping_result.second_pass_tracking_safety_satisfied,
            "second_pass_qacc_safety_satisfied": mapping_result.second_pass_qacc_safety_satisfied,
            "second_pass_tau_correction_raw": mapping_result.second_pass_tau_correction_raw,
            "second_pass_tau_correction": mapping_result.second_pass_tau_correction,
            "second_pass_qacc_predicted": mapping_result.second_pass_qacc_predicted,
            "second_pass_qacc_validated": mapping_result.second_pass_qacc_validated,
            "second_pass_qacc_validation_error": mapping_result.second_pass_qacc_validation_error,
            "second_pass_qacc_linearization_error": mapping_result.second_pass_qacc_linearization_error,
            "second_pass_forward_dynamics_gain": mapping_result.second_pass_gain_matrix,
            "second_pass_singular_values": mapping_result.second_pass_singular_values,
            "second_pass_condition_number": mapping_result.second_pass_condition_number,
            "second_pass_validation_scale": mapping_result.second_pass_validation_scale,
            "second_pass_validation_attempts": mapping_result.second_pass_validation_attempts,
            "second_pass_safe_candidate_count": mapping_result.second_pass_safe_candidate_count,
            "second_pass_total_error_rejections": mapping_result.second_pass_total_error_rejections,
            "second_pass_joint_error_rejections": mapping_result.second_pass_joint_error_rejections,
            "second_pass_qacc_limit_rejections": mapping_result.second_pass_qacc_limit_rejections,
            "forward_dynamics_safety_fallback_used": mapping_result.safety_fallback_used,
            "forward_dynamics_safety_fallback_satisfied": mapping_result.safety_fallback_satisfied,
            "forward_dynamics_safety_fallback_attempts": mapping_result.safety_fallback_attempts,
            "forward_dynamics_hold_last_safe_available": mapping_result.hold_last_safe_available,
            "forward_dynamics_hold_last_safe_used": mapping_result.hold_last_safe_used,
            "forward_dynamics_hold_last_safe_satisfied": mapping_result.hold_last_safe_satisfied,
            "forward_dynamics_hold_last_safe_qacc": mapping_result.hold_last_safe_qacc,
        }

    return {
        "arm_policy_updated": bool(arm_policy_updated),
        "target_q": target_q,
        "target_dq": target_dq,
        "ddq_raw": ddq_raw,
        "ddq_des": ddq_des,
        "ddq_saturation_limit": ddq_saturation_limit,
        **inverse_values,
        "tau_pd": tau_pd,
        **mapping_values,
        "tau_limit_lower": torque_limits[:, 0],
        "tau_limit_upper": torque_limits[:, 1],
        "torso_lin_vel_world": torso_state.lin_vel,
        "torso_ang_vel_world": torso_state.ang_vel,
        "torso_acc_world_raw": raw_torso_acc,
        "torso_acc_world_used": torso_state.lin_acc,
        "torso_alpha_world_raw": raw_torso_alpha,
        "torso_alpha_world_used": torso_state.ang_acc,
        "heading_reference_world": heading_state.reference_world,
        "heading_yaw_unwrapped": heading_state.yaw_unwrapped,
        "heading_yaw_filtered": heading_state.yaw_filtered,
        "heading_yaw_error": heading_state.yaw_error,
        "heading_yaw_rate_filtered": heading_state.yaw_rate_filtered,
        "heading_yaw_rate_correction": heading_state.yaw_rate_correction,
        "heading_yaw_rate_command": heading_yaw_rate_command,
        "heading_command_saturated": heading_state.command_saturated,
        "ee_position_reference_torso": ee_position_reference_torso,
        "lqr_one_step_prediction": lqr_one_step_prediction,
        "mpc_diagnostics": mpc_diagnostics,
    }


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
    enable_second_pass=True,
    max_safety_rescue_passes=2,
    previous_executed_tau=None,
):
    """局部线性求力矩；高残差时在已验收力矩处重线性化一次。"""
    desired_qacc = np.asarray(desired_qacc, dtype=np.float64)
    tau_nominal = np.asarray(tau_nominal, dtype=np.float64)
    qvel_indices = np.asarray(qvel_indices, dtype=np.int32)
    ctrl_indices = np.asarray(ctrl_indices, dtype=np.int32)
    torque_limits = np.asarray(torque_limits, dtype=np.float64)
    fixed_ctrl = np.asarray(fixed_ctrl, dtype=np.float64)
    previous_executed_tau = (
        None
        if previous_executed_tau is None
        else np.asarray(previous_executed_tau, dtype=np.float64)
    )
    joint_count = len(qvel_indices)
    if desired_qacc.shape != (joint_count,) or tau_nominal.shape != (joint_count,):
        raise ValueError("前向动力学映射的 desired_qacc/tau_nominal 维度不正确。")
    if fixed_ctrl.shape != (model.nu,) or torque_limits.shape != (joint_count, 2):
        raise ValueError("前向动力学映射的 ctrl/torque_limits 维度不正确。")
    if (
        previous_executed_tau is not None
        and previous_executed_tau.shape != (joint_count,)
    ):
        raise ValueError("previous_executed_tau 维度不正确。")
    if (
        previous_executed_tau is not None
        and not np.all(np.isfinite(previous_executed_tau))
    ):
        raise ValueError("previous_executed_tau 包含 NaN 或 Inf。")
    if (
        perturbation <= 0.0
        or regularization < 0.0
        or second_pass_error_threshold < 0.0
        or max_joint_error <= 0.0
        or max_abs_qacc <= 0.0
    ):
        raise ValueError("扰动和验收安全阈值必须大于 0，正则化与二次修正阈值不能小于 0。")
    max_safety_rescue_passes = int(max_safety_rescue_passes)
    if max_safety_rescue_passes < 0:
        raise ValueError("max_safety_rescue_passes 不能小于 0。")
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
        enable_second_pass
        and first_pass["improved"]
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
    if not safety_fallback_satisfied and max_safety_rescue_passes > 0:
        # 【半核心代码】可选安全救援；MPC 初版在配置中设为 0，只记录第一轮结果。
        safety_fallback_used = True
        for _ in range(max_safety_rescue_passes):
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

    # 【核心安全代码】救援仍失败时，重新验收上一仿真步实际执行的力矩。
    # 状态和接触模式可能已经改变，所以不能未经本拍 mj_forward 就直接复用。
    # 这里的 available 表示“本拍确实需要并能够尝试”，便于统计救援失败次数，
    # 而不是笼统表示上一拍力矩存在（除首拍外它几乎总是存在）。
    hold_last_safe_available = bool(
        not safety_fallback_satisfied and previous_executed_tau is not None
    )
    hold_last_safe_used = False
    hold_last_safe_satisfied = False
    hold_last_safe_qacc = np.zeros(joint_count, dtype=np.float64)
    if not safety_fallback_satisfied and hold_last_safe_available:
        hold_tau = np.clip(
            previous_executed_tau,
            torque_limits[:, 0],
            torque_limits[:, 1],
        )
        hold_ctrl = fixed_ctrl.copy()
        hold_ctrl[ctrl_indices] = hold_tau
        scratch.ctrl[:] = hold_ctrl
        scratch.qacc_warmstart[:] = qacc_warmstart
        mujoco.mj_forward(model, scratch)
        hold_last_safe_qacc = scratch.qacc[qvel_indices].copy()
        hold_last_safe_satisfied = bool(
            np.max(np.abs(hold_last_safe_qacc)) <= float(max_abs_qacc)
        )
        if hold_last_safe_satisfied:
            hold_last_safe_used = True
            safety_fallback_satisfied = True

    zero_vector = np.zeros(joint_count, dtype=np.float64)
    zero_matrix = np.zeros((joint_count, joint_count), dtype=np.float64)
    tau_cmd = (
        np.clip(previous_executed_tau, torque_limits[:, 0], torque_limits[:, 1])
        if hold_last_safe_used
        else final_pass["tau_cmd"]
    )
    tau_correction = tau_cmd - tau_nominal
    qacc_predicted = (
        hold_last_safe_qacc.copy()
        if hold_last_safe_used
        else final_pass["qacc_predicted"]
    )
    qacc_validated = (
        hold_last_safe_qacc.copy()
        if hold_last_safe_used
        else final_pass["qacc_validated"]
    )
    qacc_prediction_error = qacc_predicted - desired_qacc
    qacc_validation_error = qacc_validated - desired_qacc
    qacc_linearization_error = (
        zero_vector.copy()
        if hold_last_safe_used
        else final_pass["qacc_linearization_error"]
    )
    gain_matrix = final_pass["gain_matrix"]
    singular_values = final_pass["singular_values"]
    condition_number = final_pass["condition_number"]
    return tau_cmd, ForwardDynamicsMappingResult(
        tau_nominal=tau_nominal,
        tau_correction_raw=first_pass["correction_raw"],
        tau_correction=tau_correction,
        tau_cmd_raw=(
            tau_cmd.copy() if hold_last_safe_used else final_pass["tau_cmd_raw"]
        ),
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
        hold_last_safe_available=hold_last_safe_available,
        hold_last_safe_used=hold_last_safe_used,
        hold_last_safe_satisfied=hold_last_safe_satisfied,
        hold_last_safe_qacc=hold_last_safe_qacc,
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
    forward_dynamics_enable_second_pass=True,
    forward_dynamics_max_safety_rescue_passes=2,
    forward_dynamics_enable_hold_last_safe=False,
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
    previous_executed_tau = (
        data.ctrl[id_index_scratch.ctrl_indices].copy()
        if forward_dynamics_enable_hold_last_safe and data.time > 0.0
        else None
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
        enable_second_pass=forward_dynamics_enable_second_pass,
        max_safety_rescue_passes=forward_dynamics_max_safety_rescue_passes,
        previous_executed_tau=previous_executed_tau,
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
            "right_arm_forward_dynamics_hold_last_safe_available": [],
            "right_arm_forward_dynamics_hold_last_safe_used": [],
            "right_arm_forward_dynamics_hold_last_safe_satisfied": [],
            "right_arm_forward_dynamics_hold_last_safe_qacc": [],
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
            "right_mpc_solver_success": [],
            "right_mpc_solver_status_val": [],
            "right_mpc_solver_iterations": [],
            "right_mpc_primal_residual": [],
            "right_mpc_dual_residual": [],
            "right_mpc_objective": [],
            "right_mpc_assembly_time": [],
            "right_mpc_solve_time": [],
            "right_mpc_max_constraint_violation": [],
            "right_mpc_fallback_used": [],
            "right_mpc_fallback_feasible": [],
            "right_mpc_current_q_violation": [],
            "right_mpc_current_q_safety_violation": [],
            "right_mpc_recovery_active": [],
            "right_mpc_q_margin_min": [],
            "right_mpc_dq_margin_min": [],
            "right_mpc_ddq_margin_min": [],
            "right_mpc_one_step_q_model": [],
            "right_mpc_one_step_dq_model": [],
            "right_mpc_one_step_ee_lin_acc_model": [],
            "right_mpc_one_step_ee_ang_acc_model": [],
            "right_mpc_one_step_ee_ang_vel_model": [],
            "right_mpc_one_step_gravity_error_model": [],
            "right_mpc_one_step_ee_lin_acc_offset": [],
            "right_mpc_one_step_ee_lin_acc_ddq_map": [],
            "right_mpc_one_step_ee_ang_acc_offset": [],
            "right_mpc_one_step_ee_ang_acc_ddq_map": [],
            "right_mpc_one_step_cost_model": [],
            "right_mpc_one_step_prediction_valid": [],
            "right_mpc_disturbance_acc_k0": [],
            "right_mpc_disturbance_acc_k1": [],
            "right_mpc_disturbance_acc_terminal": [],
            "right_mpc_disturbance_omega_k0": [],
            "right_mpc_disturbance_omega_k1": [],
            "right_mpc_disturbance_omega_terminal": [],
            "right_mpc_disturbance_alpha_k0": [],
            "right_mpc_disturbance_alpha_k1": [],
            "right_mpc_disturbance_alpha_terminal": [],
            "right_mpc_interval_acc_k0": [],
            "right_mpc_interval_omega_k0": [],
            "right_mpc_interval_alpha_k0": [],
            "right_mpc_disturbance_rotation_terminal_angle": [],
            "right_mpc_template_heading_ready": [],
            "right_mpc_template_phase": [],
            "right_mpc_template_heading_yaw_world": [],
            "right_mpc_template_acc_world": [],
            "right_mpc_template_omega_world": [],
            "right_mpc_template_alpha_world": [],
            "right_mpc_template_anchor_acc_error": [],
            "right_mpc_template_anchor_omega_error": [],
            "right_mpc_template_anchor_alpha_error": [],
            "right_mpc_template_slow_bias_acc": [],
            "right_mpc_template_slow_bias_omega": [],
            "right_mpc_template_slow_bias_alpha": [],
            "right_mpc_template_anchor_rotation_error_angle": [],
            "right_mpc_template_one_step_prediction_valid": [],
            "right_mpc_template_one_step_acc_error": [],
            "right_mpc_template_one_step_omega_error": [],
            "right_mpc_template_one_step_alpha_error": [],
            "right_mpc_template_one_step_rotation_error_angle": [],
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
    """对齐相邻手臂更新间的 DDQ 响应；有 LQR 代价定义时再计算一步代价。"""
    time_values = np.asarray(trajectory_data.get("time", []), dtype=np.float64)
    sample_count = len(time_values)
    joint_count = len(RIGHT_ARM_JOINT_NAMES)
    cost_count = len(LQR_COST_TERM_NAMES)

    def empty(width):
        return np.full((sample_count, width), np.nan, dtype=np.float64)

    derived = {
        "right_arm_ddq_real": empty(joint_count),
        "right_arm_ddq_tracking_error": empty(joint_count),
        "right_arm_ddq_tracking_interval_dt": np.full(sample_count, np.nan, dtype=np.float64),
        "right_arm_ddq_tracking_valid": np.zeros(sample_count, dtype=bool),
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
    if sample_count == 0:
        trajectory_data.update(derived)
        return

    # 【非核心代码】DDQ 实际响应是 PID/LQR/MPC 共用指标，不应依赖 LQR 代价定义。
    arm_updated = np.asarray(trajectory_data.get("arm_policy_updated", []), dtype=bool)
    right_dq = np.asarray(trajectory_data.get("right_arm_dq", []), dtype=np.float64)
    ddq_des = np.asarray(trajectory_data.get("right_arm_ddq_des", []), dtype=np.float64)
    if (
        arm_updated.shape == (sample_count,)
        and right_dq.shape == (sample_count, joint_count)
        and ddq_des.shape == (sample_count, joint_count)
    ):
        update_indices = np.flatnonzero(arm_updated)
        for start_index, next_index in zip(update_indices[:-1], update_indices[1:]):
            before_index = start_index - 1
            end_index = next_index - 1
            interval_dt = (next_index - start_index) * float(simulation_dt)
            if before_index < 0 or end_index <= before_index or interval_dt <= 0.0:
                continue
            ddq_real = (right_dq[end_index] - right_dq[before_index]) / interval_dt
            derived["right_arm_ddq_real"][start_index] = ddq_real
            derived["right_arm_ddq_tracking_error"][start_index] = (
                ddq_real - ddq_des[start_index]
            )
            derived["right_arm_ddq_tracking_interval_dt"][start_index] = interval_dt
            derived["right_arm_ddq_tracking_valid"][start_index] = (
                np.all(np.isfinite(ddq_des[start_index]))
                and np.all(np.isfinite(ddq_real))
            )

    if cost_definition is None:
        trajectory_data.update(derived)
        return

    prediction_valid = np.asarray(
        trajectory_data.get("right_lqr_one_step_prediction_valid", []),
        dtype=bool,
    )
    right_q = np.asarray(trajectory_data.get("right_arm_q", []), dtype=np.float64)
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

    Q_ee_acc = np.asarray(cost_definition["Q_ee_acc"], dtype=np.float64)
    Q_ee_alpha = np.asarray(cost_definition["Q_ee_alpha"], dtype=np.float64)
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
                ee_lin_acc_real @ Q_ee_acc @ ee_lin_acc_real,
                ee_ang_acc_real @ Q_ee_alpha @ ee_ang_acc_real,
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


def add_mpc_tracking_trajectory_data(trajectory_data, simulation_dt, cost_definition):
    """【半核心代码】把 MPC 的一步模型结果与下一控制拍前的真实响应严格对齐。"""
    time_values = np.asarray(trajectory_data.get("time", []), dtype=np.float64)
    sample_count = len(time_values)
    joint_count = len(RIGHT_ARM_JOINT_NAMES)
    cost_count = len(MPC_COST_TERM_NAMES)

    def empty(width):
        return np.full((sample_count, width), np.nan, dtype=np.float64)

    derived = {
        "right_mpc_one_step_q_actual": empty(joint_count),
        "right_mpc_one_step_dq_actual": empty(joint_count),
        "right_mpc_one_step_ee_lin_acc_actual": empty(3),
        "right_mpc_one_step_ee_ang_acc_actual": empty(3),
        "right_mpc_one_step_ee_ang_vel_actual": empty(3),
        "right_mpc_one_step_gravity_error_actual": empty(2),
        "right_mpc_one_step_ee_lin_acc_realized_ddq_model": empty(3),
        "right_mpc_one_step_ee_ang_acc_realized_ddq_model": empty(3),
        "right_mpc_one_step_ee_lin_acc_error": empty(3),
        "right_mpc_one_step_ee_ang_acc_error": empty(3),
        "right_mpc_one_step_ee_ang_vel_error": empty(3),
        "right_mpc_one_step_gravity_error": empty(2),
        "right_mpc_one_step_cost_actual": empty(cost_count),
        "right_mpc_one_step_cost_error": empty(cost_count),
        "right_mpc_interval_acc_actual": empty(3),
        "right_mpc_interval_omega_actual": empty(3),
        "right_mpc_interval_alpha_actual": empty(3),
        "right_mpc_interval_acc_error": empty(3),
        "right_mpc_interval_omega_error": empty(3),
        "right_mpc_interval_alpha_error": empty(3),
        "right_mpc_tracking_interval_dt": np.full(
            sample_count, np.nan, dtype=np.float64
        ),
        "right_mpc_tracking_valid": np.zeros(sample_count, dtype=bool),
    }
    if sample_count == 0 or cost_definition is None:
        trajectory_data.update(derived)
        return

    prediction_valid = np.asarray(
        trajectory_data.get("right_mpc_one_step_prediction_valid", []),
        dtype=bool,
    )
    right_q = np.asarray(
        trajectory_data.get("right_arm_q", []), dtype=np.float64
    )
    right_dq = np.asarray(
        trajectory_data.get("right_arm_dq", []), dtype=np.float64
    )
    ddq_des = np.asarray(
        trajectory_data.get("right_arm_ddq_des", []), dtype=np.float64
    )
    ee_lin_vel = np.asarray(
        trajectory_data.get("right_ee_lin_vel_world", []), dtype=np.float64
    )
    ee_ang_vel = np.asarray(
        trajectory_data.get("right_ee_ang_vel_world", []), dtype=np.float64
    )
    torso_lin_vel = np.asarray(
        trajectory_data.get("torso_lin_vel_world", []), dtype=np.float64
    )
    torso_ang_vel = np.asarray(
        trajectory_data.get("torso_ang_vel_world", []), dtype=np.float64
    )
    interval_acc_model = np.asarray(
        trajectory_data.get("right_mpc_interval_acc_k0", []),
        dtype=np.float64,
    )
    interval_omega_model = np.asarray(
        trajectory_data.get("right_mpc_interval_omega_k0", []),
        dtype=np.float64,
    )
    interval_alpha_model = np.asarray(
        trajectory_data.get("right_mpc_interval_alpha_k0", []),
        dtype=np.float64,
    )
    gravity_error = np.asarray(
        trajectory_data.get("right_ee_gravity_error_end", []),
        dtype=np.float64,
    )
    model_linear_acc = np.asarray(
        trajectory_data.get("right_mpc_one_step_ee_lin_acc_model", []),
        dtype=np.float64,
    )
    model_angular_acc = np.asarray(
        trajectory_data.get("right_mpc_one_step_ee_ang_acc_model", []),
        dtype=np.float64,
    )
    model_angular_vel = np.asarray(
        trajectory_data.get("right_mpc_one_step_ee_ang_vel_model", []),
        dtype=np.float64,
    )
    model_gravity = np.asarray(
        trajectory_data.get("right_mpc_one_step_gravity_error_model", []),
        dtype=np.float64,
    )
    linear_offset = np.asarray(
        trajectory_data.get("right_mpc_one_step_ee_lin_acc_offset", []),
        dtype=np.float64,
    )
    linear_ddq_map = np.asarray(
        trajectory_data.get("right_mpc_one_step_ee_lin_acc_ddq_map", []),
        dtype=np.float64,
    )
    angular_offset = np.asarray(
        trajectory_data.get("right_mpc_one_step_ee_ang_acc_offset", []),
        dtype=np.float64,
    )
    angular_ddq_map = np.asarray(
        trajectory_data.get("right_mpc_one_step_ee_ang_acc_ddq_map", []),
        dtype=np.float64,
    )
    model_cost = np.asarray(
        trajectory_data.get("right_mpc_one_step_cost_model", []),
        dtype=np.float64,
    )
    expected_shapes = (
        prediction_valid.shape == (sample_count,),
        right_q.shape == (sample_count, joint_count),
        right_dq.shape == (sample_count, joint_count),
        ddq_des.shape == (sample_count, joint_count),
        ee_lin_vel.shape == (sample_count, 3),
        ee_ang_vel.shape == (sample_count, 3),
        torso_lin_vel.shape == (sample_count, 3),
        torso_ang_vel.shape == (sample_count, 3),
        interval_acc_model.shape == (sample_count, 3),
        interval_omega_model.shape == (sample_count, 3),
        interval_alpha_model.shape == (sample_count, 3),
        gravity_error.shape == (sample_count, 3),
        model_linear_acc.shape == (sample_count, 3),
        model_angular_acc.shape == (sample_count, 3),
        model_angular_vel.shape == (sample_count, 3),
        model_gravity.shape == (sample_count, 2),
        linear_offset.shape == (sample_count, 3),
        linear_ddq_map.shape == (sample_count, 3, joint_count),
        angular_offset.shape == (sample_count, 3),
        angular_ddq_map.shape == (sample_count, 3, joint_count),
        model_cost.shape == (sample_count, cost_count),
    )
    if not all(expected_shapes):
        trajectory_data.update(derived)
        return

    Q_ee_acc = np.asarray(cost_definition["Q_ee_acc"], dtype=np.float64)
    Q_ee_alpha = np.asarray(cost_definition["Q_ee_alpha"], dtype=np.float64)
    Q_ee_omega = np.asarray(
        cost_definition["Q_ee_omega"], dtype=np.float64
    )
    Qg = np.asarray(cost_definition["Qg"], dtype=np.float64)
    Qq = np.asarray(cost_definition["Qq"], dtype=np.float64)
    Qv = np.asarray(cost_definition["Qv"], dtype=np.float64)
    R = np.asarray(cost_definition["R"], dtype=np.float64)
    posture_reference = np.asarray(
        cost_definition["posture_reference"], dtype=np.float64
    )

    update_indices = np.flatnonzero(prediction_valid)
    for start_index, next_index in zip(update_indices[:-1], update_indices[1:]):
        before_index = start_index - 1
        end_index = next_index - 1
        interval_dt = (next_index - start_index) * float(simulation_dt)
        if before_index < 0 or end_index <= before_index or interval_dt <= 0.0:
            continue

        ee_lin_acc_actual = (
            ee_lin_vel[end_index] - ee_lin_vel[before_index]
        ) / interval_dt
        ee_ang_acc_actual = (
            ee_ang_vel[end_index] - ee_ang_vel[before_index]
        ) / interval_dt
        ddq_real = (
            right_dq[end_index] - right_dq[before_index]
        ) / interval_dt
        # torso_state 在 mj_step 前写入当前行，因此同一物理区间严格使用
        # start_index→next_index；EE 速度则在 step 后记录，使用上面的
        # before_index→end_index。
        interval_acc_actual = (
            torso_lin_vel[next_index] - torso_lin_vel[start_index]
        ) / interval_dt
        interval_alpha_actual = (
            torso_ang_vel[next_index] - torso_ang_vel[start_index]
        ) / interval_dt
        interval_samples = torso_ang_vel[start_index : next_index + 1]
        weights = np.ones(len(interval_samples), dtype=np.float64)
        weights[[0, -1]] = 0.5
        weights /= np.sum(weights)
        interval_omega_actual = weights @ interval_samples
        linear_realized_ddq_model = (
            linear_offset[start_index]
            + linear_ddq_map[start_index] @ ddq_real
        )
        angular_realized_ddq_model = (
            angular_offset[start_index]
            + angular_ddq_map[start_index] @ ddq_real
        )
        q_actual = right_q[end_index]
        dq_actual = right_dq[end_index]
        # 与 x1 模型预测对齐：取下一次控制更新前的世界系末端角速度。
        ee_ang_vel_actual = ee_ang_vel[end_index]
        gravity_actual = gravity_error[end_index, :2]
        posture_error = q_actual - posture_reference
        control = ddq_des[start_index]
        actual_cost = np.array(
            [
                ee_lin_acc_actual @ Q_ee_acc @ ee_lin_acc_actual,
                ee_ang_acc_actual @ Q_ee_alpha @ ee_ang_acc_actual,
                ee_ang_vel_actual @ Q_ee_omega @ ee_ang_vel_actual,
                gravity_actual @ Qg @ gravity_actual,
                posture_error @ Qq @ posture_error,
                dq_actual @ Qv @ dq_actual,
                control @ R @ control,
            ],
            dtype=np.float64,
        )

        derived["right_mpc_one_step_q_actual"][start_index] = q_actual
        derived["right_mpc_one_step_dq_actual"][start_index] = dq_actual
        derived["right_mpc_one_step_ee_lin_acc_actual"][
            start_index
        ] = ee_lin_acc_actual
        derived["right_mpc_one_step_ee_ang_acc_actual"][
            start_index
        ] = ee_ang_acc_actual
        derived["right_mpc_one_step_ee_ang_vel_actual"][
            start_index
        ] = ee_ang_vel_actual
        derived["right_mpc_one_step_gravity_error_actual"][
            start_index
        ] = gravity_actual
        derived["right_mpc_interval_acc_actual"][
            start_index
        ] = interval_acc_actual
        derived["right_mpc_interval_omega_actual"][
            start_index
        ] = interval_omega_actual
        derived["right_mpc_interval_alpha_actual"][
            start_index
        ] = interval_alpha_actual
        derived["right_mpc_interval_acc_error"][start_index] = (
            interval_acc_actual - interval_acc_model[start_index]
        )
        derived["right_mpc_interval_omega_error"][start_index] = (
            interval_omega_actual - interval_omega_model[start_index]
        )
        derived["right_mpc_interval_alpha_error"][start_index] = (
            interval_alpha_actual - interval_alpha_model[start_index]
        )
        derived["right_mpc_one_step_ee_lin_acc_realized_ddq_model"][
            start_index
        ] = linear_realized_ddq_model
        derived["right_mpc_one_step_ee_ang_acc_realized_ddq_model"][
            start_index
        ] = angular_realized_ddq_model
        derived["right_mpc_one_step_ee_lin_acc_error"][start_index] = (
            ee_lin_acc_actual - model_linear_acc[start_index]
        )
        derived["right_mpc_one_step_ee_ang_acc_error"][start_index] = (
            ee_ang_acc_actual - model_angular_acc[start_index]
        )
        derived["right_mpc_one_step_ee_ang_vel_error"][start_index] = (
            ee_ang_vel_actual - model_angular_vel[start_index]
        )
        derived["right_mpc_one_step_gravity_error"][start_index] = (
            gravity_actual - model_gravity[start_index]
        )
        derived["right_mpc_one_step_cost_actual"][start_index] = actual_cost
        derived["right_mpc_one_step_cost_error"][start_index] = (
            actual_cost - model_cost[start_index]
        )
        derived["right_mpc_tracking_interval_dt"][start_index] = interval_dt
        derived["right_mpc_tracking_valid"][start_index] = all(
            np.all(np.isfinite(value))
            for value in (
                model_linear_acc[start_index],
                model_angular_acc[start_index],
                model_angular_vel[start_index],
                model_gravity[start_index],
                model_cost[start_index],
                linear_realized_ddq_model,
                angular_realized_ddq_model,
                actual_cost,
            )
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


def _contact_mode_changed_during_intervals(
    time_values, interval_dt, valid, contact_count
):
    """标记每个手臂控制区间内 MuJoCo 接触数量是否发生变化。"""
    changed = np.zeros_like(valid, dtype=bool)
    if (
        contact_count.shape != valid.shape
        or interval_dt.shape != valid.shape
        or time_values.shape != valid.shape
    ):
        return changed
    for index in np.flatnonzero(valid):
        interval_end = time_values[index] + interval_dt[index] + 1e-12
        end_index = int(np.searchsorted(time_values, interval_end, side="right"))
        values = contact_count[index:max(index + 1, end_index)]
        changed[index] = bool(
            values.size > 1 and np.any(values[1:] != values[0])
        )
    return changed


def compute_lqr_tracking_diagnostics(trajectory_data, eval_start_time, eval_end_time):
    time_values = np.asarray(trajectory_data.get("time", []), dtype=np.float64)
    # 【半核心代码】DDQ 跟踪是三种控制器共用指标，不依赖 LQR 一步代价是否存在。
    valid = np.asarray(trajectory_data.get("right_arm_ddq_tracking_valid", []), dtype=bool).copy()
    if valid.shape != time_values.shape:
        valid = np.zeros_like(time_values, dtype=bool)
    interval_dt = np.asarray(
        trajectory_data.get("right_arm_ddq_tracking_interval_dt", []),
        dtype=np.float64,
    )
    if interval_dt.shape != time_values.shape:
        valid = np.zeros_like(time_values, dtype=bool)
        interval_dt = np.full_like(time_values, np.nan)
    valid &= (time_values >= eval_start_time) & (time_values + interval_dt <= eval_end_time + 1e-12)

    cost_valid = np.asarray(
        trajectory_data.get("right_lqr_tracking_valid", []), dtype=bool
    ).copy()
    cost_interval_dt = np.asarray(
        trajectory_data.get("right_lqr_tracking_interval_dt", []), dtype=np.float64
    )
    if cost_valid.shape != time_values.shape or cost_interval_dt.shape != time_values.shape:
        cost_valid = np.zeros_like(time_values, dtype=bool)
    else:
        cost_valid &= (time_values >= eval_start_time) & (
            time_values + cost_interval_dt <= eval_end_time + 1e-12
        )
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
        "cost_sample_count": int(np.sum(cost_valid)),
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
    hold_last_safe_available = np.asarray(
        trajectory_data.get(
            "right_arm_forward_dynamics_hold_last_safe_available", []
        ),
        dtype=bool,
    )
    hold_last_safe_used = np.asarray(
        trajectory_data.get(
            "right_arm_forward_dynamics_hold_last_safe_used", []
        ),
        dtype=bool,
    )
    hold_last_safe_satisfied = np.asarray(
        trajectory_data.get(
            "right_arm_forward_dynamics_hold_last_safe_satisfied", []
        ),
        dtype=bool,
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
        "hold_last_safe": {
            "available_fraction": masked_fraction(
                hold_last_safe_available, eval_step_mask
            ),
            "used_count": int(np.sum(hold_last_safe_used[eval_step_mask]))
            if hold_last_safe_used.shape == eval_step_mask.shape
            else 0,
            "used_fraction": masked_fraction(
                hold_last_safe_used, eval_step_mask
            ),
            "satisfied_fraction_when_available": (
                masked_fraction(
                    hold_last_safe_satisfied,
                    eval_step_mask & hold_last_safe_available,
                )
                if hold_last_safe_available.shape == eval_step_mask.shape
                else 0.0
            ),
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
    if sample_count == 0 or ddq_des.shape != ddq_real.shape or ddq_des.shape[0] != len(time_values):
        diagnostics["ddq_tracking"] = _component_tracking_metrics(np.zeros((0, 5)), np.zeros((0, 5)))
    else:
        ddq_metrics = _component_tracking_metrics(ddq_des[valid], ddq_real[valid])
        ddq_error_norm = np.linalg.norm(ddq_real[valid] - ddq_des[valid], axis=1)
        ddq_metrics["error_norm_rms"] = float(np.sqrt(np.mean(ddq_error_norm ** 2)))
        ddq_metrics["error_norm_p95"] = float(np.percentile(ddq_error_norm, 95.0))
        ddq_metrics["error_norm_max"] = float(np.max(ddq_error_norm))
        ddq_limits = np.asarray(
            trajectory_data.get("right_arm_ddq_saturation_limit", []),
            dtype=np.float64,
        )
        if ddq_limits.shape == valid.shape:
            ddq_limits = ddq_limits[:, None]
        if ddq_limits.shape in (ddq_real.shape, (len(valid), 1)):
            spike_mask = valid & np.any(
                np.abs(ddq_real) > ddq_limits + 1e-9, axis=1
            )
            qacc_safe = np.asarray(
                trajectory_data.get(
                    "right_arm_forward_dynamics_safety_fallback_satisfied", []
                ),
                dtype=bool,
            )
            tau_contact = np.asarray(
                trajectory_data.get("right_arm_tau_contact", []),
                dtype=np.float64,
            )
            recovery_active = np.asarray(
                trajectory_data.get("right_mpc_recovery_active", []),
                dtype=bool,
            )
            contact_count = np.asarray(
                trajectory_data.get("contact_count", []), dtype=np.int64
            )
            contact_mode_changed = _contact_mode_changed_during_intervals(
                time_values,
                interval_dt,
                valid,
                contact_count,
            )
            final_validated_unsafe = (
                ~qacc_safe
                if qacc_safe.shape == valid.shape
                else np.zeros_like(valid)
            )
            arm_contact = (
                np.linalg.norm(tau_contact, axis=1) > 1e-6
                if tau_contact.shape == ddq_real.shape
                else np.zeros_like(valid)
            )
            if recovery_active.shape != valid.shape:
                recovery_active = np.zeros_like(valid)
            spike_count = int(np.count_nonzero(spike_mask))
            ddq_metrics["spike_diagnostics"] = {
                "definition": (
                    "an arm interval is a spike when any |ddq_real_j| exceeds "
                    "that joint's configured ddq limit"
                ),
                "count": spike_count,
                "fraction": float(spike_count / sample_count),
                "max_abs_ddq_real": float(
                    np.max(np.abs(ddq_real[valid]))
                ),
                "final_validated_unsafe_fraction_given_spike": (
                    float(np.mean(final_validated_unsafe[spike_mask]))
                    if spike_count
                    else 0.0
                ),
                "right_arm_contact_fraction_given_spike": (
                    float(np.mean(arm_contact[spike_mask]))
                    if spike_count
                    else 0.0
                ),
                "contact_count_changed_fraction_given_spike": (
                    float(np.mean(contact_mode_changed[spike_mask]))
                    if spike_count
                    else 0.0
                ),
                "recovery_active_fraction_given_spike": (
                    float(np.mean(recovery_active[spike_mask]))
                    if spike_count
                    else 0.0
                ),
            }
        diagnostics["ddq_tracking"] = ddq_metrics

    if (
        np.any(cost_valid)
        and model_cost.shape == actual_cost.shape
        and model_cost.shape[0] == len(time_values)
    ):
        cost_metrics = _component_tracking_metrics(
            model_cost[cost_valid], actual_cost[cost_valid]
        )
        model_total = np.sum(model_cost[cost_valid], axis=1, keepdims=True)
        actual_total = np.sum(actual_cost[cost_valid], axis=1, keepdims=True)
        cost_metrics["total"] = _component_tracking_metrics(model_total, actual_total)
        diagnostics["cost_tracking"] = cost_metrics
    else:
        diagnostics["cost_tracking"] = _component_tracking_metrics(
            np.zeros((0, 7)), np.zeros((0, 7))
        )
    return diagnostics


def compute_mpc_diagnostics(trajectory_data, eval_start_time, eval_end_time):
    """汇总“模板→QP 一步理想量→仿真实际量”三层 MPC 诊断。"""
    time_values = np.asarray(
        trajectory_data.get("time", []), dtype=np.float64
    )
    sample_count = len(time_values)
    eval_mask = (
        (time_values >= eval_start_time) & (time_values < eval_end_time)
        if sample_count
        else np.zeros(0, dtype=bool)
    )
    tracking_valid = np.asarray(
        trajectory_data.get("right_mpc_tracking_valid", []), dtype=bool
    )
    interval_dt = np.asarray(
        trajectory_data.get("right_mpc_tracking_interval_dt", []),
        dtype=np.float64,
    )
    if (
        tracking_valid.shape != (sample_count,)
        or interval_dt.shape != (sample_count,)
    ):
        tracking_valid = np.zeros(sample_count, dtype=bool)
        interval_dt = np.full(sample_count, np.nan, dtype=np.float64)
    else:
        tracking_valid &= eval_mask
        tracking_valid &= (
            time_values + interval_dt <= eval_end_time + 1e-12
        )

    def matrix(name, width):
        values = np.asarray(
            trajectory_data.get(name, []), dtype=np.float64
        )
        if values.shape != (sample_count, width):
            return np.full((sample_count, width), np.nan, dtype=np.float64)
        return values

    model_linear = matrix("right_mpc_one_step_ee_lin_acc_model", 3)
    actual_linear = matrix("right_mpc_one_step_ee_lin_acc_actual", 3)
    realized_ddq_linear = matrix(
        "right_mpc_one_step_ee_lin_acc_realized_ddq_model", 3
    )
    model_angular = matrix("right_mpc_one_step_ee_ang_acc_model", 3)
    actual_angular = matrix("right_mpc_one_step_ee_ang_acc_actual", 3)
    realized_ddq_angular = matrix(
        "right_mpc_one_step_ee_ang_acc_realized_ddq_model", 3
    )
    model_angular_velocity = matrix(
        "right_mpc_one_step_ee_ang_vel_model", 3
    )
    actual_angular_velocity = matrix(
        "right_mpc_one_step_ee_ang_vel_actual", 3
    )
    model_gravity = matrix("right_mpc_one_step_gravity_error_model", 2)
    actual_gravity = matrix("right_mpc_one_step_gravity_error_actual", 2)
    model_cost = matrix(
        "right_mpc_one_step_cost_model", len(MPC_COST_TERM_NAMES)
    )
    actual_cost = matrix(
        "right_mpc_one_step_cost_actual", len(MPC_COST_TERM_NAMES)
    )

    diagnostics = {
        "definition": {
            "alignment": (
                "model at arm update k versus response immediately before "
                "arm update k+1"
            ),
            "model_acceleration": (
                "affine task model evaluated with the selected first input; "
                "QP u_0 when solved, configured fallback input otherwise"
            ),
            "actual_acceleration": (
                "end-effector velocity difference over the same arm-control interval"
            ),
            "gravity": (
                "model x_1 versus measured end-of-interval signed xy gravity error"
            ),
            "angular_velocity": (
                "model x_1 versus measured end-of-interval world-frame "
                "end-effector angular velocity"
            ),
            "weighted_cost": (
                "the same Q_A, Q_alpha, Q_omega, Q_G, Q_q, Q_v and R "
                "applied to "
                "model and actual interval outcomes; fallback samples are not "
                "optimized MPC ideals"
            ),
            "template_anchor_error": (
                "current measured node d_0 minus H node-template value at "
                "the same phase; this residual enters only through slow bias"
            ),
            "template_one_step_error": (
                "current measured node disturbance minus the previous "
                "update's node k=1 prediction"
            ),
            "interval_disturbance_error": (
                "predicted interval[k=0] versus the realized average over "
                "the same future arm-control interval"
            ),
            "evaluation_window": [
                float(eval_start_time),
                float(eval_end_time),
            ],
        },
        "task_sample_count": int(np.sum(tracking_valid)),
        "cost_term_names": list(MPC_COST_TERM_NAMES),
        "interval_dt_mean": (
            float(np.mean(interval_dt[tracking_valid]))
            if np.any(tracking_valid)
            else 0.0
        ),
    }
    arm_updated = np.asarray(
        trajectory_data.get("arm_policy_updated", []), dtype=bool
    )
    solver_success = np.asarray(
        trajectory_data.get("right_mpc_solver_success", []), dtype=bool
    )
    fallback_used = np.asarray(
        trajectory_data.get("right_mpc_fallback_used", []), dtype=bool
    )
    fallback_feasible = np.asarray(
        trajectory_data.get("right_mpc_fallback_feasible", []), dtype=bool
    )
    solver_status = np.asarray(
        trajectory_data.get("right_mpc_solver_status_val", []),
        dtype=np.float64,
    )
    current_q_violation = np.asarray(
        trajectory_data.get("right_mpc_current_q_violation", []),
        dtype=np.float64,
    )
    current_q_safety_violation = np.asarray(
        trajectory_data.get("right_mpc_current_q_safety_violation", []),
        dtype=np.float64,
    )
    recovery_active = np.asarray(
        trajectory_data.get("right_mpc_recovery_active", []), dtype=bool
    )
    expected_scalar_shape = (sample_count,)
    if all(
        value.shape == expected_scalar_shape
        for value in (
            arm_updated,
            solver_success,
            fallback_used,
            fallback_feasible,
            solver_status,
            current_q_violation,
            current_q_safety_violation,
            recovery_active,
        )
    ):
        update_mask = eval_mask & arm_updated
        status_values = solver_status[update_mask]
        finite_status = status_values[np.isfinite(status_values)].astype(int)
        unique_status, status_counts = np.unique(
            finite_status, return_counts=True
        )
        violation_values = current_q_violation[update_mask]
        finite_violation = violation_values[np.isfinite(violation_values)]
        safety_violation_values = current_q_safety_violation[update_mask]
        finite_safety_violation = safety_violation_values[
            np.isfinite(safety_violation_values)
        ]
        diagnostics["solver"] = {
            "update_count": int(np.sum(update_mask)),
            "success_fraction": (
                float(np.mean(solver_success[update_mask]))
                if np.any(update_mask)
                else 0.0
            ),
            "fallback_fraction": (
                float(np.mean(fallback_used[update_mask]))
                if np.any(update_mask)
                else 0.0
            ),
            "fallback_feasible_fraction_given_fallback": (
                float(
                    np.mean(
                        fallback_feasible[update_mask & fallback_used]
                    )
                )
                if np.any(update_mask & fallback_used)
                else 0.0
            ),
            "status_val_counts": {
                str(value): int(count)
                for value, count in zip(unique_status, status_counts)
            },
            "current_q_violation_fraction": (
                float(np.mean(finite_violation > 1e-9))
                if finite_violation.size
                else 0.0
            ),
            "current_q_violation_max_rad": (
                float(np.max(finite_violation))
                if finite_violation.size
                else 0.0
            ),
            "current_q_safety_violation_fraction": (
                float(np.mean(finite_safety_violation > 1e-9))
                if finite_safety_violation.size
                else 0.0
            ),
            "current_q_safety_violation_max_rad": (
                float(np.max(finite_safety_violation))
                if finite_safety_violation.size
                else 0.0
            ),
            "recovery_active_fraction": (
                float(np.mean(recovery_active[update_mask]))
                if np.any(update_mask)
                else 0.0
            ),
        }
    else:
        diagnostics["solver"] = {
            "update_count": 0,
            "success_fraction": 0.0,
            "fallback_fraction": 0.0,
            "fallback_feasible_fraction_given_fallback": 0.0,
            "status_val_counts": {},
            "current_q_violation_fraction": 0.0,
            "current_q_violation_max_rad": 0.0,
            "current_q_safety_violation_fraction": 0.0,
            "current_q_safety_violation_max_rad": 0.0,
            "recovery_active_fraction": 0.0,
        }
    for name, model_values, actual_values in (
        ("linear_acceleration", model_linear, actual_linear),
        ("angular_acceleration", model_angular, actual_angular),
        (
            "angular_velocity",
            model_angular_velocity,
            actual_angular_velocity,
        ),
        ("gravity_error", model_gravity, actual_gravity),
        ("weighted_cost", model_cost, actual_cost),
    ):
        diagnostics[name] = _component_tracking_metrics(
            model_values[tracking_valid], actual_values[tracking_valid]
        )
        if name != "weighted_cost" and np.any(tracking_valid):
            model_norm_rms = float(
                np.sqrt(
                    np.mean(
                        np.sum(model_values[tracking_valid] ** 2, axis=1)
                    )
                )
            )
            actual_norm_rms = float(
                np.sqrt(
                    np.mean(
                        np.sum(actual_values[tracking_valid] ** 2, axis=1)
                    )
                )
            )
            error_norm_rms = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                actual_values[tracking_valid]
                                - model_values[tracking_valid]
                            )
                            ** 2,
                            axis=1,
                        )
                    )
                )
            )
            diagnostics[name]["norm_summary"] = {
                "model_rms": model_norm_rms,
                "actual_rms": actual_norm_rms,
                "actual_minus_model_percent": float(
                    100.0
                    * (actual_norm_rms - model_norm_rms)
                    / max(model_norm_rms, 1e-12)
                ),
                "tracking_error_percent_of_actual_rms": float(
                    100.0 * error_norm_rms / max(actual_norm_rms, 1e-12)
                ),
            }
    diagnostics["linear_acceleration_decomposition"] = {
        "ddq_execution_effect": _component_tracking_metrics(
            model_linear[tracking_valid],
            realized_ddq_linear[tracking_valid],
        ),
        "remaining_task_model_error": _component_tracking_metrics(
            realized_ddq_linear[tracking_valid],
            actual_linear[tracking_valid],
        ),
    }
    diagnostics["angular_acceleration_decomposition"] = {
        "ddq_execution_effect": _component_tracking_metrics(
            model_angular[tracking_valid],
            realized_ddq_angular[tracking_valid],
        ),
        "remaining_task_model_error": _component_tracking_metrics(
            realized_ddq_angular[tracking_valid],
            actual_angular[tracking_valid],
        ),
    }

    interval_values = {
        "linear_acceleration": (
            matrix("right_mpc_interval_acc_k0", 3),
            matrix("right_mpc_interval_acc_actual", 3),
            matrix("right_mpc_interval_acc_error", 3),
        ),
        "angular_velocity": (
            matrix("right_mpc_interval_omega_k0", 3),
            matrix("right_mpc_interval_omega_actual", 3),
            matrix("right_mpc_interval_omega_error", 3),
        ),
        "angular_acceleration": (
            matrix("right_mpc_interval_alpha_k0", 3),
            matrix("right_mpc_interval_alpha_actual", 3),
            matrix("right_mpc_interval_alpha_error", 3),
        ),
    }
    interval_diagnostics = {
        "sample_count": int(np.sum(tracking_valid)),
        "definition": "future 6 ms interval prediction versus realized interval average",
    }
    peak_mask = tracking_valid & (
        np.linalg.norm(interval_values["linear_acceleration"][1], axis=1)
        > 5.0
    )
    interval_diagnostics["peak_sample_count"] = int(np.sum(peak_mask))
    for name, (prediction, actual, error) in interval_values.items():
        tracking = _component_tracking_metrics(
            prediction[tracking_valid], actual[tracking_valid]
        )
        tracking["error_norm"] = _masked_vector_norm_stats(
            error, tracking_valid, width=3
        )
        tracking["peak_error_norm"] = _masked_vector_norm_stats(
            error, peak_mask, width=3
        )
        tracking["actual_norm"] = _masked_vector_norm_stats(
            actual, tracking_valid, width=3
        )
        interval_diagnostics[name] = tracking
    diagnostics["interval_disturbance"] = interval_diagnostics

    if np.any(tracking_valid):
        task_term_count = MPC_COST_TERM_NAMES.index("posture")
        model_task_total = np.sum(
            model_cost[tracking_valid, :task_term_count], axis=1
        )
        actual_task_total = np.sum(
            actual_cost[tracking_valid, :task_term_count], axis=1
        )
        diagnostics["weighted_task_cost_total"] = {
            "model_mean": float(np.mean(model_task_total)),
            "model_rms": float(np.sqrt(np.mean(model_task_total**2))),
            "actual_mean": float(np.mean(actual_task_total)),
            "actual_rms": float(np.sqrt(np.mean(actual_task_total**2))),
            "ratio_actual_to_model_mean": float(
                np.mean(actual_task_total)
                / max(np.mean(model_task_total), 1e-12)
            ),
        }
    else:
        diagnostics["weighted_task_cost_total"] = {
            "model_mean": 0.0,
            "model_rms": 0.0,
            "actual_mean": 0.0,
            "actual_rms": 0.0,
            "ratio_actual_to_model_mean": 0.0,
        }

    heading_ready = np.asarray(
        trajectory_data.get("right_mpc_template_heading_ready", []),
        dtype=bool,
    )
    one_step_valid = np.asarray(
        trajectory_data.get(
            "right_mpc_template_one_step_prediction_valid", []
        ),
        dtype=bool,
    )
    if heading_ready.shape != (sample_count,):
        heading_ready = np.zeros(sample_count, dtype=bool)
    if one_step_valid.shape != (sample_count,):
        one_step_valid = np.zeros(sample_count, dtype=bool)
    anchor_mask = eval_mask & heading_ready
    forecast_mask = eval_mask & one_step_valid

    template_values = {
        "acceleration": (
            matrix("right_mpc_template_acc_world", 3),
            matrix("torso_acc_world_used", 3),
            matrix("right_mpc_template_anchor_acc_error", 3),
            matrix("right_mpc_template_one_step_acc_error", 3),
        ),
        "angular_velocity": (
            matrix("right_mpc_template_omega_world", 3),
            matrix("torso_ang_vel_world", 3),
            matrix("right_mpc_template_anchor_omega_error", 3),
            matrix("right_mpc_template_one_step_omega_error", 3),
        ),
        "angular_acceleration": (
            matrix("right_mpc_template_alpha_world", 3),
            matrix("torso_alpha_world_used", 3),
            matrix("right_mpc_template_anchor_alpha_error", 3),
            matrix("right_mpc_template_one_step_alpha_error", 3),
        ),
    }
    template_diagnostics = {
        "anchor_sample_count": int(np.sum(anchor_mask)),
        "one_step_sample_count": int(np.sum(forecast_mask)),
    }
    for name, (template, measured, anchor_error, forecast_error) in template_values.items():
        anchor_tracking = _component_tracking_metrics(
            template[anchor_mask], measured[anchor_mask]
        )
        forecast_prediction = measured[forecast_mask] - forecast_error[
            forecast_mask
        ]
        forecast_tracking = _component_tracking_metrics(
            forecast_prediction, measured[forecast_mask]
        )
        anchor_tracking["error_norm"] = _masked_vector_norm_stats(
            anchor_error, anchor_mask, width=3
        )
        forecast_tracking["error_norm"] = _masked_vector_norm_stats(
            forecast_error, forecast_mask, width=3
        )
        measured_anchor_norm_rms = float(
            np.sqrt(np.mean(np.sum(measured[anchor_mask] ** 2, axis=1)))
        ) if np.any(anchor_mask) else 0.0
        measured_forecast_norm_rms = float(
            np.sqrt(np.mean(np.sum(measured[forecast_mask] ** 2, axis=1)))
        ) if np.any(forecast_mask) else 0.0
        anchor_tracking["measured_norm_rms"] = measured_anchor_norm_rms
        anchor_tracking["error_percent_of_measured_rms"] = float(
            100.0
            * anchor_tracking["error_norm"]["rms"]
            / max(measured_anchor_norm_rms, 1e-12)
        )
        forecast_tracking["measured_norm_rms"] = measured_forecast_norm_rms
        forecast_tracking["error_percent_of_measured_rms"] = float(
            100.0
            * forecast_tracking["error_norm"]["rms"]
            / max(measured_forecast_norm_rms, 1e-12)
        )
        template_diagnostics[name] = {
            "current_phase_absolute_match": anchor_tracking,
            "anchored_one_step_prediction": forecast_tracking,
        }

    anchor_rotation_error = np.asarray(
        trajectory_data.get(
            "right_mpc_template_anchor_rotation_error_angle", []
        ),
        dtype=np.float64,
    )
    forecast_rotation_error = np.asarray(
        trajectory_data.get(
            "right_mpc_template_one_step_rotation_error_angle", []
        ),
        dtype=np.float64,
    )

    def scalar_error_stats(values, mask):
        if values.shape != (sample_count,) or not np.any(mask):
            return {"mean": 0.0, "rms": 0.0, "p95": 0.0, "max": 0.0}
        selected = np.abs(values[mask])
        return {
            "mean": float(np.mean(selected)),
            "rms": float(np.sqrt(np.mean(selected**2))),
            "p95": float(np.percentile(selected, 95.0)),
            "max": float(np.max(selected)),
        }

    template_diagnostics["orientation_error_angle_rad"] = {
        "current_phase_absolute_match": scalar_error_stats(
            anchor_rotation_error, anchor_mask
        ),
        "anchored_one_step_prediction": scalar_error_stats(
            forecast_rotation_error, forecast_mask
        ),
    }
    template_diagnostics["slow_bias_norm"] = {
        "acceleration": _masked_vector_norm_stats(
            matrix("right_mpc_template_slow_bias_acc", 3),
            anchor_mask,
            width=3,
        ),
        "angular_velocity": _masked_vector_norm_stats(
            matrix("right_mpc_template_slow_bias_omega", 3),
            anchor_mask,
            width=3,
        ),
        "angular_acceleration": _masked_vector_norm_stats(
            matrix("right_mpc_template_slow_bias_alpha", 3),
            anchor_mask,
            width=3,
        ),
    }
    diagnostics["disturbance_template"] = template_diagnostics
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
        mpc_cost_term_names=np.asarray(MPC_COST_TERM_NAMES),
        xml_path=np.array(xml_path),
        simulation_dt=np.array(simulation_dt),
    )
    return right_arm_diagnostics


def save_right_arm_diagnostics(run_dir, diagnostics):
    diagnostics_path = os.path.join(run_dir, "right_arm_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(diagnostics), f, indent=2, ensure_ascii=False)
    return diagnostics_path


def save_lqr_tracking_diagnostics(run_dir, diagnostics, controller_name="lqr"):
    diagnostics_path = os.path.join(
        run_dir, f"{controller_name}_tracking_diagnostics.json"
    )
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(diagnostics), f, indent=2, ensure_ascii=False)
    return diagnostics_path


def save_mpc_diagnostics(run_dir, diagnostics):
    diagnostics_path = os.path.join(run_dir, "mpc_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(diagnostics), f, indent=2, ensure_ascii=False)
    return diagnostics_path


def save_mpc_diagnostics_plot(
    run_dir,
    trajectory_data,
    eval_start_time,
    eval_end_time,
):
    """【非核心代码】比较控制器一步模型预测和同区间实际量。"""
    time_values = np.asarray(
        trajectory_data.get("time", []), dtype=np.float64
    )
    sample_count = len(time_values)
    tracking_valid = np.asarray(
        trajectory_data.get("right_mpc_tracking_valid", []), dtype=bool
    )
    if tracking_valid.shape != (sample_count,):
        return None
    eval_mask = (time_values >= eval_start_time) & (
        time_values < eval_end_time
    )
    tracking_valid &= eval_mask
    if not np.any(tracking_valid):
        return None
    solver_success = np.asarray(
        trajectory_data.get("right_mpc_solver_success", []), dtype=bool
    )
    if solver_success.shape != (sample_count,):
        solver_success = np.zeros(sample_count, dtype=bool)
    fallback_in_plot = ~solver_success[tracking_valid]

    def matrix(name, width):
        values = np.asarray(
            trajectory_data.get(name, []), dtype=np.float64
        )
        return (
            values
            if values.shape == (sample_count, width)
            else np.full((sample_count, width), np.nan, dtype=np.float64)
        )

    task_signals = (
        (
            "Linear acceleration norm [m/s²]",
            matrix("right_mpc_one_step_ee_lin_acc_model", 3),
            matrix("right_mpc_one_step_ee_lin_acc_actual", 3),
        ),
        (
            "Angular acceleration norm [rad/s²]",
            matrix("right_mpc_one_step_ee_ang_acc_model", 3),
            matrix("right_mpc_one_step_ee_ang_acc_actual", 3),
        ),
        (
            "Angular velocity norm [rad/s]",
            matrix("right_mpc_one_step_ee_ang_vel_model", 3),
            matrix("right_mpc_one_step_ee_ang_vel_actual", 3),
        ),
        (
            "2D gravity-error norm [m/s²]",
            matrix("right_mpc_one_step_gravity_error_model", 2),
            matrix("right_mpc_one_step_gravity_error_actual", 2),
        ),
    )
    plot_path = os.path.join(run_dir, "mpc_model_vs_actual.png")
    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    for axis, (title, model_values, actual_values) in zip(axes, task_signals):
        model_norm = np.linalg.norm(model_values[tracking_valid], axis=1)
        actual_norm = np.linalg.norm(actual_values[tracking_valid], axis=1)
        model_rms = np.sqrt(np.mean(model_norm**2))
        actual_rms = np.sqrt(np.mean(actual_norm**2))
        level_difference_percent = (
            100.0 * (actual_rms - model_rms) / max(model_rms, 1e-12)
        )
        axis.plot(
            time_values[tracking_valid],
            model_norm,
            lw=1.2,
            label="controller one-step model",
        )
        axis.plot(
            time_values[tracking_valid],
            actual_norm,
            lw=1.0,
            label="actual",
        )
        if np.any(fallback_in_plot):
            axis.plot(
                time_values[tracking_valid][fallback_in_plot],
                model_norm[fallback_in_plot],
                linestyle="none",
                marker="x",
                markersize=3,
                alpha=0.45,
                color="tab:red",
                label="QP fallback sample",
            )
        axis.set_title(
            f"{title} | RMS: model={model_rms:.3f}, actual={actual_rms:.3f}, "
            f"actual is {level_difference_percent:+.1f}%"
        )
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(
        "Controller one-step model versus actual task quantities "
        "(red x = QP fallback, not an optimized MPC solution)"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    return plot_path


def save_mpc_template_tracking_plot(
    run_dir,
    trajectory_data,
    eval_start_time,
    eval_end_time,
    gait_period,
):
    """用中间一个步态周期显示当前模板、一步预测与同拍实测量。"""
    time_values = np.asarray(
        trajectory_data.get("time", []), dtype=np.float64
    )
    sample_count = len(time_values)
    heading_ready = np.asarray(
        trajectory_data.get("right_mpc_template_heading_ready", []),
        dtype=bool,
    )
    forecast_valid = np.asarray(
        trajectory_data.get(
            "right_mpc_template_one_step_prediction_valid", []
        ),
        dtype=bool,
    )
    if (
        heading_ready.shape != (sample_count,)
        or forecast_valid.shape != (sample_count,)
    ):
        return None
    period = float(gait_period)
    if not np.isfinite(period) or period <= 0.0:
        return None
    complete_cycles = max(
        1, int(np.floor((eval_end_time - eval_start_time) / period))
    )
    middle_cycle = complete_cycles // 2
    plot_start_time = eval_start_time + middle_cycle * period
    plot_end_time = min(plot_start_time + period, eval_end_time)
    cycle_mask = (time_values >= plot_start_time) & (
        time_values < plot_end_time
    )
    valid = cycle_mask & heading_ready & forecast_valid
    if not np.any(valid):
        return None

    def matrix(name):
        values = np.asarray(
            trajectory_data.get(name, []), dtype=np.float64
        )
        return (
            values
            if values.shape == (sample_count, 3)
            else np.full((sample_count, 3), np.nan, dtype=np.float64)
        )

    signals = (
        (
            "Base acceleration norm [m/s²]",
            matrix("torso_acc_world_used"),
            matrix("right_mpc_template_acc_world"),
            matrix("right_mpc_template_one_step_acc_error"),
        ),
        (
            "Base angular velocity norm [rad/s]",
            matrix("torso_ang_vel_world"),
            matrix("right_mpc_template_omega_world"),
            matrix("right_mpc_template_one_step_omega_error"),
        ),
        (
            "Base angular acceleration norm [rad/s²]",
            matrix("torso_alpha_world_used"),
            matrix("right_mpc_template_alpha_world"),
            matrix("right_mpc_template_one_step_alpha_error"),
        ),
    )

    plot_path = os.path.join(run_dir, "mpc_template_tracking.png")
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for axis, (title, measured, current_template, forecast_error) in zip(
        axes, signals
    ):
        one_step_prediction = measured - forecast_error
        measured_norm = np.linalg.norm(measured[valid], axis=1)
        template_norm = np.linalg.norm(current_template[valid], axis=1)
        forecast_norm = np.linalg.norm(one_step_prediction[valid], axis=1)
        measured_rms = np.sqrt(np.mean(measured_norm**2))
        current_error_rms = np.sqrt(
            np.mean(np.sum((measured[valid] - current_template[valid]) ** 2, axis=1))
        )
        forecast_error_rms = np.sqrt(
            np.mean(np.sum(forecast_error[valid] ** 2, axis=1))
        )
        current_percent = 100.0 * current_error_rms / max(measured_rms, 1e-12)
        forecast_percent = 100.0 * forecast_error_rms / max(measured_rms, 1e-12)
        axis.plot(
            time_values[valid],
            measured_norm,
            lw=1.2,
            label="measured at current update",
        )
        axis.plot(
            time_values[valid],
            template_norm,
            lw=1.0,
            label="template at current phase",
        )
        axis.plot(
            time_values[valid],
            forecast_norm,
            lw=1.0,
            label="previous update's k=1 prediction for current update",
        )
        axis.set_title(
            f"{title} | error/measured RMS: "
            f"current={current_percent:.1f}%, one-step={forecast_percent:.1f}%"
        )
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(
        "Disturbance template versus measured base motion — "
        f"middle gait cycle [{plot_start_time:.3f}, {plot_end_time:.3f}) s"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    return plot_path


def save_mpc_interval_disturbance_plot(
    run_dir,
    trajectory_data,
    eval_start_time,
    eval_end_time,
    gait_period,
):
    """【非核心代码】用中间一个步态周期核对 6 ms 区间扰动预测。"""
    time_values = np.asarray(
        trajectory_data.get("time", []), dtype=np.float64
    )
    sample_count = len(time_values)
    tracking_valid = np.asarray(
        trajectory_data.get("right_mpc_tracking_valid", []), dtype=bool
    )
    heading_ready = np.asarray(
        trajectory_data.get("right_mpc_template_heading_ready", []),
        dtype=bool,
    )
    if (
        tracking_valid.shape != (sample_count,)
        or heading_ready.shape != (sample_count,)
    ):
        return None

    period = float(gait_period)
    if not np.isfinite(period) or period <= 0.0:
        return None
    complete_cycles = max(
        1, int(np.floor((eval_end_time - eval_start_time) / period))
    )
    middle_cycle = complete_cycles // 2
    plot_start_time = eval_start_time + middle_cycle * period
    plot_end_time = min(plot_start_time + period, eval_end_time)
    valid = (
        tracking_valid
        & heading_ready
        & (time_values >= plot_start_time)
        & (time_values < plot_end_time)
    )
    if not np.any(valid):
        return None

    def matrix(name):
        values = np.asarray(
            trajectory_data.get(name, []), dtype=np.float64
        )
        if values.shape == (sample_count, 3):
            return values
        return np.full((sample_count, 3), np.nan, dtype=np.float64)

    signals = (
        (
            "Base linear acceleration [m/s²]",
            matrix("right_mpc_interval_acc_k0"),
            matrix("right_mpc_interval_acc_actual"),
        ),
        (
            "Base angular velocity [rad/s]",
            matrix("right_mpc_interval_omega_k0"),
            matrix("right_mpc_interval_omega_actual"),
        ),
        (
            "Base angular acceleration [rad/s²]",
            matrix("right_mpc_interval_alpha_k0"),
            matrix("right_mpc_interval_alpha_actual"),
        ),
    )

    plot_path = os.path.join(
        run_dir, "mpc_interval_disturbance_tracking.png"
    )
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for axis, (title, prediction, actual) in zip(axes, signals):
        prediction_norm = np.linalg.norm(prediction[valid], axis=1)
        actual_norm = np.linalg.norm(actual[valid], axis=1)
        actual_rms = np.sqrt(np.mean(actual_norm**2))
        error_rms = np.sqrt(
            np.mean(np.sum((actual[valid] - prediction[valid]) ** 2, axis=1))
        )
        error_percent = 100.0 * error_rms / max(actual_rms, 1e-12)
        axis.plot(
            time_values[valid],
            prediction_norm,
            lw=1.2,
            label="predicted average for following 6 ms",
        )
        axis.plot(
            time_values[valid],
            actual_norm,
            lw=1.0,
            label="realized average over following 6 ms",
        )
        axis.set_title(
            f"{title} | vector-error RMS / actual RMS = "
            f"{error_rms:.3f} / {actual_rms:.3f} ({error_percent:.1f}%)"
        )
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("control-interval start time [s]")
    fig.suptitle(
        "MPC interval disturbance prediction versus realization — "
        f"middle gait cycle [{plot_start_time:.3f}, {plot_end_time:.3f}) s"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    return plot_path


def save_mpc_diagnostics_preview(
    run_dir,
    trajectory_data,
    eval_start_time,
    eval_end_time,
):
    """保存每个 MPC 更新点最直接的模型/实际代价与模板误差。"""
    preview_path = os.path.join(run_dir, "mpc_diagnostics_preview.csv")
    time_values = np.asarray(
        trajectory_data.get("time", []), dtype=np.float64
    )
    sample_count = len(time_values)
    tracking_valid = np.asarray(
        trajectory_data.get("right_mpc_tracking_valid", []), dtype=bool
    )
    interval_dt = np.asarray(
        trajectory_data.get("right_mpc_tracking_interval_dt", []),
        dtype=np.float64,
    )
    model_cost = np.asarray(
        trajectory_data.get("right_mpc_one_step_cost_model", []),
        dtype=np.float64,
    )
    actual_cost = np.asarray(
        trajectory_data.get("right_mpc_one_step_cost_actual", []),
        dtype=np.float64,
    )
    anchor_rotation = np.asarray(
        trajectory_data.get(
            "right_mpc_template_anchor_rotation_error_angle", []
        ),
        dtype=np.float64,
    )
    forecast_rotation = np.asarray(
        trajectory_data.get(
            "right_mpc_template_one_step_rotation_error_angle", []
        ),
        dtype=np.float64,
    )

    def matrix(name, width):
        values = np.asarray(
            trajectory_data.get(name, []), dtype=np.float64
        )
        return (
            values
            if values.shape == (sample_count, width)
            else np.full((sample_count, width), np.nan, dtype=np.float64)
        )

    anchor_errors = (
        matrix("right_mpc_template_anchor_acc_error", 3),
        matrix("right_mpc_template_anchor_omega_error", 3),
        matrix("right_mpc_template_anchor_alpha_error", 3),
    )
    forecast_errors = (
        matrix("right_mpc_template_one_step_acc_error", 3),
        matrix("right_mpc_template_one_step_omega_error", 3),
        matrix("right_mpc_template_one_step_alpha_error", 3),
    )
    task_norm_values = (
        (
            "linear_acceleration",
            matrix("right_mpc_one_step_ee_lin_acc_model", 3),
            matrix(
                "right_mpc_one_step_ee_lin_acc_realized_ddq_model", 3
            ),
            matrix("right_mpc_one_step_ee_lin_acc_actual", 3),
        ),
        (
            "angular_acceleration",
            matrix("right_mpc_one_step_ee_ang_acc_model", 3),
            matrix(
                "right_mpc_one_step_ee_ang_acc_realized_ddq_model", 3
            ),
            matrix("right_mpc_one_step_ee_ang_acc_actual", 3),
        ),
    )
    state_norm_values = (
        (
            "angular_velocity",
            matrix("right_mpc_one_step_ee_ang_vel_model", 3),
            matrix("right_mpc_one_step_ee_ang_vel_actual", 3),
        ),
    )
    expected = (
        tracking_valid.shape == (sample_count,),
        interval_dt.shape == (sample_count,),
        model_cost.shape == (sample_count, len(MPC_COST_TERM_NAMES)),
        actual_cost.shape == model_cost.shape,
        anchor_rotation.shape == (sample_count,),
        forecast_rotation.shape == (sample_count,),
    )
    if not all(expected):
        tracking_valid = np.zeros(sample_count, dtype=bool)

    headers = ["time", "interval_dt", "in_evaluation"]
    for name, _, _, _ in task_norm_values:
        headers.extend(
            (
                f"{name}_model_norm",
                f"{name}_realized_ddq_model_norm",
                f"{name}_actual_norm",
            )
        )
    for name, _, _ in state_norm_values:
        headers.extend(
            (
                f"{name}_model_norm",
                f"{name}_actual_norm",
            )
        )
    for term in MPC_COST_TERM_NAMES:
        headers.extend(
            (
                f"cost_model_{term}",
                f"cost_actual_{term}",
                f"cost_error_{term}",
            )
        )
    for prefix in ("acc", "omega", "alpha"):
        headers.extend(
            (
                f"template_anchor_{prefix}_error_norm",
                f"template_one_step_{prefix}_error_norm",
            )
        )
    headers.extend(
        (
            "template_anchor_rotation_error_angle",
            "template_one_step_rotation_error_angle",
        )
    )

    with open(preview_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for index in np.flatnonzero(tracking_valid):
            row = [
                time_values[index],
                interval_dt[index],
                bool(
                    eval_start_time <= time_values[index]
                    and time_values[index] + interval_dt[index]
                    <= eval_end_time + 1e-12
                ),
            ]
            for _, model_value, realized_value, actual_value in task_norm_values:
                row.extend(
                    (
                        np.linalg.norm(model_value[index]),
                        np.linalg.norm(realized_value[index]),
                        np.linalg.norm(actual_value[index]),
                    )
                )
            for _, model_value, actual_value in state_norm_values:
                row.extend(
                    (
                        np.linalg.norm(model_value[index]),
                        np.linalg.norm(actual_value[index]),
                    )
                )
            for term in range(len(MPC_COST_TERM_NAMES)):
                row.extend(
                    (
                        model_cost[index, term],
                        actual_cost[index, term],
                        actual_cost[index, term] - model_cost[index, term],
                    )
                )
            for anchor_error, forecast_error in zip(
                anchor_errors, forecast_errors
            ):
                row.extend(
                    (
                        np.linalg.norm(anchor_error[index]),
                        np.linalg.norm(forecast_error[index]),
                    )
                )
            row.extend((anchor_rotation[index], forecast_rotation[index]))
            writer.writerow(row)
    return preview_path


def save_lqr_ddq_tracking_plot(
    run_dir,
    trajectory_data,
    diagnostics,
    eval_start_time,
    eval_end_time,
    controller_name="lqr",
):
    """绘制评估区间内五关节 ddq_des 与 6 ms 速度差分 ddq_real。"""
    plot_path = os.path.join(run_dir, "ddq_tracking.png")
    time_values = np.asarray(trajectory_data.get("time", []), dtype=np.float64)
    valid = np.asarray(
        trajectory_data.get("right_arm_ddq_tracking_valid", []), dtype=bool
    )
    interval_dt = np.asarray(
        trajectory_data.get("right_arm_ddq_tracking_interval_dt", []),
        dtype=np.float64,
    )
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
    qacc_safe = np.asarray(
        trajectory_data.get(
            "right_arm_forward_dynamics_safety_fallback_satisfied", []
        ),
        dtype=bool,
    )
    tau_contact = np.asarray(
        trajectory_data.get("right_arm_tau_contact", []), dtype=np.float64
    )
    recovery_active = np.asarray(
        trajectory_data.get("right_mpc_recovery_active", []), dtype=bool
    )
    hold_last_safe_used = np.asarray(
        trajectory_data.get(
            "right_arm_forward_dynamics_hold_last_safe_used", []
        ),
        dtype=bool,
    )
    contact_count = np.asarray(
        trajectory_data.get("contact_count", []), dtype=np.int64
    )
    unsafe = (
        ~qacc_safe
        if qacc_safe.shape == time_values.shape
        else np.zeros_like(valid)
    )
    arm_contact = (
        np.linalg.norm(tau_contact, axis=1) > 1e-6
        if tau_contact.shape == expected_shape
        else np.zeros_like(valid)
    )
    if recovery_active.shape != time_values.shape:
        recovery_active = np.zeros_like(valid)
    if hold_last_safe_used.shape != time_values.shape:
        hold_last_safe_used = np.zeros_like(valid)
    contact_mode_changed = _contact_mode_changed_during_intervals(
        time_values,
        interval_dt,
        valid,
        contact_count,
    )
    labels = tuple(name.removeprefix("right_").removesuffix("_joint") for name in RIGHT_ARM_JOINT_NAMES)
    fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True)
    plot_time = time_values[valid]
    for joint, (axis, label) in enumerate(zip(axes, labels)):
        axis.plot(plot_time, ddq_des[valid, joint], color="tab:blue", lw=1.2, label="ddq_des")
        axis.plot(plot_time, ddq_real[valid, joint], color="tab:orange", lw=1.0, alpha=0.9, label="ddq_real")
        axis.axhline(0.0, color="black", lw=0.6, alpha=0.4)
        # 图顶短标记用于区分执行安全、接触切换、右臂接触和 MPC 恢复盒，
        # 从而避免把所有 ddq_real 尖峰都误判为关节恢复命令。
        marker_specs = (
            (unsafe, 0.98, "tab:red", "final validated qacc unsafe"),
            (
                contact_mode_changed,
                0.95,
                "tab:cyan",
                "contact count changed in interval",
            ),
            (arm_contact, 0.92, "tab:purple", "right-arm contact torque"),
            (hold_last_safe_used, 0.89, "tab:orange", "hold-last-safe torque"),
            (recovery_active, 0.86, "tab:green", "MPC recovery box active"),
        )
        for marker_mask, y_position, color, marker_label in marker_specs:
            marker_valid = valid & marker_mask
            if np.any(marker_valid):
                axis.plot(
                    time_values[marker_valid],
                    np.full(np.count_nonzero(marker_valid), y_position),
                    linestyle="none",
                    marker="|",
                    markersize=5,
                    color=color,
                    transform=axis.get_xaxis_transform(),
                    label=marker_label if joint == 0 else None,
                )
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
    valid_count = max(int(np.count_nonzero(valid)), 1)
    fig.suptitle(
        f"{controller_name.upper()} DDQ tracking: "
        "desired versus realized arm-interval average acceleration\n"
        f"final-validated-unsafe={np.count_nonzero(valid & unsafe) / valid_count:.1%}, "
        f"contact-change={np.count_nonzero(valid & contact_mode_changed) / valid_count:.1%}, "
        f"right-arm-contact={np.count_nonzero(valid & arm_contact) / valid_count:.1%}, "
        f"hold-last-safe={np.count_nonzero(valid & hold_last_safe_used) / valid_count:.1%}, "
        f"recovery-box={np.count_nonzero(valid & recovery_active) / valid_count:.1%}"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    return plot_path


def save_lqr_tracking_preview(
    run_dir,
    trajectory_data,
    eval_start_time,
    eval_end_time,
    controller_name="lqr",
):
    """保存严格按相邻手臂控制更新对齐的 DDQ 与一步代价跟踪表。"""
    preview_path = os.path.join(run_dir, f"{controller_name}_tracking_preview.csv")
    time_values = np.asarray(trajectory_data.get("time", []), dtype=np.float64)
    valid = np.asarray(
        trajectory_data.get("right_arm_ddq_tracking_valid", []), dtype=bool
    )
    ddq_des = np.asarray(trajectory_data.get("right_arm_ddq_des", []), dtype=np.float64)
    ddq_real = np.asarray(trajectory_data.get("right_arm_ddq_real", []), dtype=np.float64)
    ddq_error = np.asarray(trajectory_data.get("right_arm_ddq_tracking_error", []), dtype=np.float64)
    model_cost = np.asarray(trajectory_data.get("right_lqr_one_step_cost_model", []), dtype=np.float64)
    actual_cost = np.asarray(trajectory_data.get("right_lqr_one_step_cost_actual", []), dtype=np.float64)
    cost_error = np.asarray(trajectory_data.get("right_lqr_one_step_cost_error", []), dtype=np.float64)
    interval_dt = np.asarray(
        trajectory_data.get("right_arm_ddq_tracking_interval_dt", []),
        dtype=np.float64,
    )
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
        ("right_arm_forward_dynamics_hold_last_safe_qacc", joint_labels),
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
        "right_arm_forward_dynamics_hold_last_safe_available",
        "right_arm_forward_dynamics_hold_last_safe_used",
        "right_arm_forward_dynamics_hold_last_safe_satisfied",
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
    forward_dynamics_hold_last_safe_available = bool(
        right_arm_control.get(
            "forward_dynamics_hold_last_safe_available", False
        )
    )
    forward_dynamics_hold_last_safe_used = bool(
        right_arm_control.get("forward_dynamics_hold_last_safe_used", False)
    )
    forward_dynamics_hold_last_safe_satisfied = bool(
        right_arm_control.get(
            "forward_dynamics_hold_last_safe_satisfied", False
        )
    )
    forward_dynamics_hold_last_safe_qacc = control_vector(
        "forward_dynamics_hold_last_safe_qacc", 5
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
    mpc_diagnostics = right_arm_control.get("mpc_diagnostics")
    mpc_diagnostics_valid = isinstance(mpc_diagnostics, dict)

    def mpc_scalar(name):
        if not mpc_diagnostics_valid:
            return np.nan
        return float(mpc_diagnostics.get(name, np.nan))

    mpc_margins = (
        mpc_diagnostics.get("min_constraint_margins", {})
        if mpc_diagnostics_valid
        else {}
    )
    mpc_disturbance_prediction = (
        mpc_diagnostics.get("disturbance_prediction", {})
        if mpc_diagnostics_valid
        else {}
    )
    mpc_interval_disturbance_prediction = (
        mpc_diagnostics.get("interval_disturbance_prediction", {})
        if mpc_diagnostics_valid
        else {}
    )
    mpc_prediction = (
        mpc_diagnostics.get("one_step_prediction")
        if mpc_diagnostics_valid
        else None
    )
    mpc_prediction_valid = isinstance(mpc_prediction, dict)
    mpc_template_diagnostics = (
        mpc_diagnostics.get("disturbance_template_diagnostics", {})
        if mpc_diagnostics_valid
        else {}
    )

    def mpc_disturbance_vector(name, index, interval=False):
        source = (
            mpc_interval_disturbance_prediction
            if interval
            else mpc_disturbance_prediction
        )
        values = np.asarray(
            source.get(name, []), dtype=np.float64
        )
        if values.ndim != 2 or values.shape[1] != 3 or not len(values):
            return np.full(3, np.nan, dtype=np.float64)
        resolved_index = index if index >= 0 else len(values) + index
        if not 0 <= resolved_index < len(values):
            return np.full(3, np.nan, dtype=np.float64)
        return values[resolved_index].copy()

    def mpc_terminal_rotation_angle():
        values = np.asarray(
            mpc_disturbance_prediction.get("rot_world_body", []),
            dtype=np.float64,
        )
        if values.ndim != 3 or values.shape[1:] != (3, 3) or len(values) < 2:
            return np.nan
        relative = values[-1] @ values[0].T
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.arccos(cosine))

    def dictionary_vector(dictionary, name, size):
        if not isinstance(dictionary, dict):
            return np.full(size, np.nan, dtype=np.float64)
        value = np.asarray(
            dictionary.get(name, np.full(size, np.nan)), dtype=np.float64
        )
        return value.copy() if value.shape == (size,) else np.full(size, np.nan, dtype=np.float64)

    def dictionary_matrix(dictionary, name, shape):
        if not isinstance(dictionary, dict):
            return np.full(shape, np.nan, dtype=np.float64)
        value = np.asarray(
            dictionary.get(name, np.full(shape, np.nan)), dtype=np.float64
        )
        return value.copy() if value.shape == shape else np.full(
            shape, np.nan, dtype=np.float64
        )

    prediction_costs = {} if not lqr_prediction_valid else lqr_prediction.get("cost_terms", {})
    lqr_cost_model = np.array(
        [float(prediction_costs.get(name, np.nan)) for name in LQR_COST_TERM_NAMES],
        dtype=np.float64,
    )
    mpc_prediction_costs = (
        {} if not mpc_prediction_valid else mpc_prediction.get("cost_terms", {})
    )
    mpc_cost_model = np.array(
        [
            float(mpc_prediction_costs.get(name, np.nan))
            for name in MPC_COST_TERM_NAMES
        ],
        dtype=np.float64,
    )

    def template_scalar(name):
        return float(mpc_template_diagnostics.get(name, np.nan))

    def template_vector(name):
        return dictionary_vector(mpc_template_diagnostics, name, 3)
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
    buffers.trajectory_data["right_arm_forward_dynamics_hold_last_safe_available"].append(forward_dynamics_hold_last_safe_available)
    buffers.trajectory_data["right_arm_forward_dynamics_hold_last_safe_used"].append(forward_dynamics_hold_last_safe_used)
    buffers.trajectory_data["right_arm_forward_dynamics_hold_last_safe_satisfied"].append(forward_dynamics_hold_last_safe_satisfied)
    buffers.trajectory_data["right_arm_forward_dynamics_hold_last_safe_qacc"].append(forward_dynamics_hold_last_safe_qacc)
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
    buffers.trajectory_data["right_lqr_one_step_q_model"].append(dictionary_vector(lqr_prediction, "q", 5))
    buffers.trajectory_data["right_lqr_one_step_dq_model"].append(dictionary_vector(lqr_prediction, "dq", 5))
    buffers.trajectory_data["right_lqr_one_step_ee_lin_acc_model"].append(dictionary_vector(lqr_prediction, "ee_lin_acc", 3))
    buffers.trajectory_data["right_lqr_one_step_ee_ang_acc_model"].append(dictionary_vector(lqr_prediction, "ee_ang_acc", 3))
    buffers.trajectory_data["right_lqr_one_step_position_error_model"].append(dictionary_vector(lqr_prediction, "position_error", 3))
    buffers.trajectory_data["right_lqr_one_step_gravity_error_model"].append(dictionary_vector(lqr_prediction, "gravity_error", 3))
    buffers.trajectory_data["right_lqr_one_step_cost_model"].append(lqr_cost_model)
    buffers.trajectory_data["right_lqr_one_step_prediction_valid"].append(lqr_prediction_valid)
    buffers.trajectory_data["right_mpc_solver_success"].append(
        bool(mpc_diagnostics.get("success", False)) if mpc_diagnostics_valid else False
    )
    buffers.trajectory_data["right_mpc_solver_status_val"].append(mpc_scalar("solver_status_val"))
    buffers.trajectory_data["right_mpc_solver_iterations"].append(mpc_scalar("iterations"))
    buffers.trajectory_data["right_mpc_primal_residual"].append(mpc_scalar("primal_residual"))
    buffers.trajectory_data["right_mpc_dual_residual"].append(mpc_scalar("dual_residual"))
    buffers.trajectory_data["right_mpc_objective"].append(mpc_scalar("objective"))
    buffers.trajectory_data["right_mpc_assembly_time"].append(mpc_scalar("assembly_time"))
    buffers.trajectory_data["right_mpc_solve_time"].append(mpc_scalar("solve_time"))
    buffers.trajectory_data["right_mpc_max_constraint_violation"].append(
        mpc_scalar("max_constraint_violation")
    )
    buffers.trajectory_data["right_mpc_fallback_used"].append(
        bool(mpc_diagnostics.get("fallback_used", False)) if mpc_diagnostics_valid else False
    )
    buffers.trajectory_data["right_mpc_fallback_feasible"].append(
        bool(mpc_diagnostics.get("fallback_feasible", False))
        if mpc_diagnostics_valid
        else False
    )
    buffers.trajectory_data["right_mpc_current_q_violation"].append(
        mpc_scalar("current_q_violation")
    )
    buffers.trajectory_data["right_mpc_current_q_safety_violation"].append(
        mpc_scalar("current_q_safety_violation")
    )
    buffers.trajectory_data["right_mpc_recovery_active"].append(
        bool(mpc_diagnostics.get("recovery_active", False))
        if mpc_diagnostics_valid
        else False
    )
    buffers.trajectory_data["right_mpc_q_margin_min"].append(
        float(mpc_margins.get("q", np.nan))
    )
    buffers.trajectory_data["right_mpc_dq_margin_min"].append(
        float(mpc_margins.get("dq", np.nan))
    )
    buffers.trajectory_data["right_mpc_ddq_margin_min"].append(
        float(mpc_margins.get("ddq", np.nan))
    )
    buffers.trajectory_data["right_mpc_one_step_q_model"].append(
        dictionary_vector(mpc_prediction, "q", 5)
    )
    buffers.trajectory_data["right_mpc_one_step_dq_model"].append(
        dictionary_vector(mpc_prediction, "dq", 5)
    )
    buffers.trajectory_data["right_mpc_one_step_ee_lin_acc_model"].append(
        dictionary_vector(mpc_prediction, "ee_lin_acc", 3)
    )
    buffers.trajectory_data["right_mpc_one_step_ee_ang_acc_model"].append(
        dictionary_vector(mpc_prediction, "ee_ang_acc", 3)
    )
    buffers.trajectory_data["right_mpc_one_step_ee_ang_vel_model"].append(
        dictionary_vector(mpc_prediction, "ee_ang_vel", 3)
    )
    buffers.trajectory_data["right_mpc_one_step_gravity_error_model"].append(
        dictionary_vector(mpc_prediction, "gravity_error", 2)
    )
    buffers.trajectory_data["right_mpc_one_step_ee_lin_acc_offset"].append(
        dictionary_vector(mpc_prediction, "ee_lin_acc_offset", 3)
    )
    buffers.trajectory_data["right_mpc_one_step_ee_lin_acc_ddq_map"].append(
        dictionary_matrix(mpc_prediction, "ee_lin_acc_ddq_map", (3, 5))
    )
    buffers.trajectory_data["right_mpc_one_step_ee_ang_acc_offset"].append(
        dictionary_vector(mpc_prediction, "ee_ang_acc_offset", 3)
    )
    buffers.trajectory_data["right_mpc_one_step_ee_ang_acc_ddq_map"].append(
        dictionary_matrix(mpc_prediction, "ee_ang_acc_ddq_map", (3, 5))
    )
    buffers.trajectory_data["right_mpc_one_step_cost_model"].append(mpc_cost_model)
    buffers.trajectory_data["right_mpc_one_step_prediction_valid"].append(
        mpc_prediction_valid
    )
    for field, source_name, index in (
        ("right_mpc_disturbance_acc_k0", "acc_world", 0),
        ("right_mpc_disturbance_acc_k1", "acc_world", 1),
        ("right_mpc_disturbance_acc_terminal", "acc_world", -1),
        ("right_mpc_disturbance_omega_k0", "omega_world", 0),
        ("right_mpc_disturbance_omega_k1", "omega_world", 1),
        ("right_mpc_disturbance_omega_terminal", "omega_world", -1),
        ("right_mpc_disturbance_alpha_k0", "alpha_world", 0),
        ("right_mpc_disturbance_alpha_k1", "alpha_world", 1),
        ("right_mpc_disturbance_alpha_terminal", "alpha_world", -1),
    ):
        buffers.trajectory_data[field].append(
            mpc_disturbance_vector(source_name, index)
        )
    for field, source_name in (
        ("right_mpc_interval_acc_k0", "acc_world"),
        ("right_mpc_interval_omega_k0", "omega_world"),
        ("right_mpc_interval_alpha_k0", "alpha_world"),
    ):
        buffers.trajectory_data[field].append(
            mpc_disturbance_vector(source_name, 0, interval=True)
        )
    buffers.trajectory_data[
        "right_mpc_disturbance_rotation_terminal_angle"
    ].append(mpc_terminal_rotation_angle())
    buffers.trajectory_data["right_mpc_template_heading_ready"].append(
        bool(mpc_template_diagnostics.get("heading_ready", False))
    )
    buffers.trajectory_data["right_mpc_template_phase"].append(
        template_scalar("phase")
    )
    buffers.trajectory_data["right_mpc_template_heading_yaw_world"].append(
        template_scalar("heading_yaw_world")
    )
    for field, source_name in (
        ("right_mpc_template_acc_world", "template_acc_world"),
        ("right_mpc_template_omega_world", "template_omega_world"),
        ("right_mpc_template_alpha_world", "template_alpha_world"),
        ("right_mpc_template_anchor_acc_error", "anchor_acc_error"),
        ("right_mpc_template_anchor_omega_error", "anchor_omega_error"),
        ("right_mpc_template_anchor_alpha_error", "anchor_alpha_error"),
        ("right_mpc_template_slow_bias_acc", "slow_bias_acc_world"),
        (
            "right_mpc_template_slow_bias_omega",
            "slow_bias_omega_world",
        ),
        (
            "right_mpc_template_slow_bias_alpha",
            "slow_bias_alpha_world",
        ),
        ("right_mpc_template_one_step_acc_error", "one_step_acc_error"),
        ("right_mpc_template_one_step_omega_error", "one_step_omega_error"),
        ("right_mpc_template_one_step_alpha_error", "one_step_alpha_error"),
    ):
        buffers.trajectory_data[field].append(template_vector(source_name))
    buffers.trajectory_data[
        "right_mpc_template_anchor_rotation_error_angle"
    ].append(template_scalar("anchor_rotation_error_angle"))
    buffers.trajectory_data[
        "right_mpc_template_one_step_prediction_valid"
    ].append(
        bool(
            mpc_template_diagnostics.get(
                "one_step_prediction_valid", False
            )
        )
    )
    buffers.trajectory_data[
        "right_mpc_template_one_step_rotation_error_angle"
    ].append(template_scalar("one_step_rotation_error_angle"))
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


def finalize_run(run_dir, buffers, xml_path, simulation_dt, video_path, video_frames, video_fps, has_renderer, video_width, video_height, data, scene_ids, eval_start_time, eval_end_time, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles, gait_period, experiment_name, perf_monitor=None, lqr_cost_definition=None, mpc_cost_definition=None, arm_controller="lqr"):
    trajectory_path = os.path.join(run_dir, "trajectory.npz")
    add_lqr_tracking_trajectory_data(buffers.trajectory_data, simulation_dt, lqr_cost_definition)
    add_mpc_tracking_trajectory_data(
        buffers.trajectory_data, simulation_dt, mpc_cost_definition
    )
    lqr_tracking_diagnostics = compute_lqr_tracking_diagnostics(
        buffers.trajectory_data,
        eval_start_time,
        eval_end_time,
    )
    mpc_diagnostics = compute_mpc_diagnostics(
        buffers.trajectory_data,
        eval_start_time,
        eval_end_time,
    )
    right_arm_diagnostics = save_trajectory(trajectory_path, buffers.trajectory_data, xml_path, simulation_dt)
    right_arm_diagnostics_path = save_right_arm_diagnostics(run_dir, right_arm_diagnostics)
    lqr_tracking_diagnostics_path = save_lqr_tracking_diagnostics(
        run_dir, lqr_tracking_diagnostics, arm_controller
    )
    mpc_diagnostics_path = save_mpc_diagnostics(run_dir, mpc_diagnostics)
    mpc_diagnostics_plot_path = save_mpc_diagnostics_plot(
        run_dir,
        buffers.trajectory_data,
        eval_start_time,
        eval_end_time,
    )
    mpc_template_plot_path = save_mpc_template_tracking_plot(
        run_dir,
        buffers.trajectory_data,
        eval_start_time,
        eval_end_time,
        gait_period,
    )
    mpc_interval_plot_path = save_mpc_interval_disturbance_plot(
        run_dir,
        buffers.trajectory_data,
        eval_start_time,
        eval_end_time,
        gait_period,
    )
    mpc_diagnostics_preview_path = save_mpc_diagnostics_preview(
        run_dir,
        buffers.trajectory_data,
        eval_start_time,
        eval_end_time,
    )
    lqr_ddq_tracking_plot_path = save_lqr_ddq_tracking_plot(
        run_dir,
        buffers.trajectory_data,
        lqr_tracking_diagnostics,
        eval_start_time,
        eval_end_time,
        arm_controller,
    )
    lqr_tracking_preview_path = save_lqr_tracking_preview(
        run_dir,
        buffers.trajectory_data,
        eval_start_time,
        eval_end_time,
        arm_controller,
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
    saved_paths["mpc_diagnostics"] = mpc_diagnostics_path
    saved_paths["mpc_diagnostics_plot"] = mpc_diagnostics_plot_path
    saved_paths["mpc_template_plot"] = mpc_template_plot_path
    saved_paths["mpc_interval_disturbance_plot"] = mpc_interval_plot_path
    saved_paths["mpc_diagnostics_preview"] = mpc_diagnostics_preview_path
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
        controller_label = arm_controller.upper()
        print(f"{controller_label} DDQ tracking RMSE = {np.asarray(ddq_tracking['rmse']).round(4).tolist()}")
        print(f"{controller_label} DDQ tracking correlation = {np.asarray(ddq_tracking['correlation']).round(4).tolist()}")
        print(f"{controller_label} DDQ tracking gain = {np.asarray(ddq_tracking['gain']).round(4).tolist()}")
        if lqr_tracking_diagnostics.get("cost_sample_count", 0):
            cost_rmse = lqr_tracking_diagnostics["cost_tracking"]["rmse"]
            print(f"LQR one-step cost tracking RMSE = {dict(zip(LQR_COST_TERM_NAMES, np.asarray(cost_rmse).round(4).tolist()))}")
        validation = lqr_tracking_diagnostics.get("forward_dynamics_validation", {})
        first_selections = validation.get("first_pass", {}).get("selections", [])
        if first_selections:
            selection_text = ", ".join(
                f"{item['label']}({item['scale']:g})={item['count']} ({item['fraction'] * 100.0:.2f}%)"
                for item in first_selections
            )
            print(f"{controller_label} forward-dynamics first-pass selections: {selection_text}")
        second = validation.get("second_pass", {})
        if second:
            print(
                f"{controller_label} forward-dynamics second pass: "
                f"triggered={second.get('triggered_count', 0)} "
                f"({second.get('triggered_fraction_of_evaluation_steps', 0.0) * 100.0:.2f}%), "
                f"accepted={second.get('accepted_count', 0)} "
                f"({second.get('accepted_fraction_given_triggered', 0.0) * 100.0:.2f}% of triggered)"
            )
    if arm_controller == "mpc" and mpc_diagnostics["task_sample_count"]:
        solver = mpc_diagnostics["solver"]
        print(
            "MPC solver success/fallback/operating-box-outside fraction = "
            f"{solver['success_fraction']:.3f}/"
            f"{solver['fallback_fraction']:.3f}/"
            f"{solver['current_q_violation_fraction']:.3f}"
        )
        print(
            "MPC safety-box-outside/recovery-active fraction = "
            f"{solver['current_q_safety_violation_fraction']:.3f}/"
            f"{solver['recovery_active_fraction']:.3f}"
        )
        task_total = mpc_diagnostics["weighted_task_cost_total"]
        print(
            "MPC one-step weighted task cost mean (model/actual/ratio) = "
            f"{task_total['model_mean']:.4f}/"
            f"{task_total['actual_mean']:.4f}/"
            f"{task_total['ratio_actual_to_model_mean']:.3f}"
        )
        for name, label in (
            ("linear_acceleration", "linear acceleration"),
            ("angular_acceleration", "angular acceleration"),
            ("angular_velocity", "angular velocity"),
            ("gravity_error", "2D gravity error"),
        ):
            norm_summary = mpc_diagnostics[name]["norm_summary"]
            print(
                f"MPC {label} RMS (model/actual/actual-minus-model) = "
                f"{norm_summary['model_rms']:.4f}/"
                f"{norm_summary['actual_rms']:.4f}/"
                f"{norm_summary['actual_minus_model_percent']:+.1f}%"
            )
        for name, label in (
            ("linear_acceleration_decomposition", "linear acceleration"),
            ("angular_acceleration_decomposition", "angular acceleration"),
        ):
            decomposition = mpc_diagnostics[name]
            execution_rmse = np.linalg.norm(
                decomposition["ddq_execution_effect"]["rmse"]
            )
            model_rmse = np.linalg.norm(
                decomposition["remaining_task_model_error"]["rmse"]
            )
            print(
                f"MPC {label} error-norm RMS "
                f"(DDQ execution/task-model remainder) = "
                f"{execution_rmse:.4f}/{model_rmse:.4f}"
            )
        template = mpc_diagnostics["disturbance_template"]
        for name, label in (
            ("acceleration", "a_B"),
            ("angular_velocity", "omega_B"),
            ("angular_acceleration", "alpha_B"),
        ):
            item = template[name]
            anchor_rms = item["current_phase_absolute_match"][
                "error_norm"
            ]["rms"]
            forecast_rms = item["anchored_one_step_prediction"][
                "error_norm"
            ]["rms"]
            anchor_percent = item["current_phase_absolute_match"][
                "error_percent_of_measured_rms"
            ]
            forecast_percent = item["anchored_one_step_prediction"][
                "error_percent_of_measured_rms"
            ]
            print(
                f"MPC template {label} error-norm RMS "
                f"(current match/one-step) = "
                f"{anchor_rms:.4f}/{forecast_rms:.4f}; "
                f"relative to measured = "
                f"{anchor_percent:.1f}%/{forecast_percent:.1f}%"
            )
        interval = mpc_diagnostics["interval_disturbance"]
        for name, label in (
            ("linear_acceleration", "a_B"),
            ("angular_velocity", "omega_B"),
            ("angular_acceleration", "alpha_B"),
        ):
            item = interval[name]
            error_rms = item["error_norm"]["rms"]
            actual_rms = item["actual_norm"]["rms"]
            error_percent = (
                100.0 * error_rms / max(actual_rms, 1e-12)
            )
            peak_error_rms = item["peak_error_norm"]["rms"]
            print(
                f"MPC 6 ms interval {label} prediction "
                f"(error RMS/actual RMS/relative/peak-error RMS) = "
                f"{error_rms:.4f}/{actual_rms:.4f}/"
                f"{error_percent:.1f}%/{peak_error_rms:.4f}"
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
    # MPC 的重力任务严格只使用末端系 x/y；LQR 保持原有三维定义。
    gravity_norm_width = (
        2 if str(experiment_name).lower().endswith("_mpc") else 3
    )

    stats = {
        "gait_period": gait_period,
        "total_cycles": total_cycles,
        "warmup_cycles": warmup_cycles,
        "evaluation_cycles": evaluation_cycles,
        "cooldown_cycles": cooldown_cycles,
        "eval_start_time": eval_start_time,
        "eval_end_time": eval_end_time,
        "walk_distance_xy": walk_distance,
        "gravity_error_norm_definition": (
            "signed_xy_used_by_mpc_cost"
            if gravity_norm_width == 2
            else "directed_xyz"
        ),
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
                "gravity_error_cost_norm",
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
            tilt_n = np.linalg.norm(tilt[:, :gravity_norm_width], axis=1)
            tilt_xyz_n = np.linalg.norm(tilt, axis=1)

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
            stats[f"{side}_tilt_xyz_norm_mean"] = tilt_xyz_n[mask].mean()
            stats[f"{side}_tilt_xyz_norm_std"] = tilt_xyz_n[mask].std()
            stats[f"{side}_tilt_xyz_norm_rms"] = np.sqrt(
                np.mean(tilt_xyz_n[mask] ** 2)
            )
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
                (
                    f"{side} directed gravity error xyz "
                    "(z diagnostic only)"
                    if gravity_norm_width == 2
                    else f"{side} directed gravity error xyz"
                ),
                (
                    (
                        f"{side} gravity error xy norm (MPC cost)"
                        if side == "right"
                        else f"{side} gravity error xy norm (diagnostic)"
                    )
                    if gravity_norm_width == 2
                    else f"{side} gravity error xyz norm"
                ),
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
    extra_lqr_tracking = saved_paths.get("lqr_tracking_diagnostics") if saved_paths.get("lqr_tracking_diagnostics") is not None else "未保存 DDQ tracking 诊断"
    extra_lqr_tracking_preview = saved_paths.get("lqr_tracking_preview") if saved_paths.get("lqr_tracking_preview") is not None else "未保存 DDQ tracking CSV"
    extra_lqr_tracking_plot = saved_paths.get("lqr_ddq_tracking_plot") if saved_paths.get("lqr_ddq_tracking_plot") is not None else "未保存 DDQ tracking 图片"
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
        gravity_components = (
            (
                "xy (MPC cost)"
                if side == "right"
                else "xy (diagnostic)"
            )
            if stats.get("gravity_error_norm_definition")
            == "signed_xy_used_by_mpc_cost"
            else "xyz"
        )
        print(
            f"{side} | gravity error {gravity_components} norm mean/std/rms = "
            f"{stats[f'{side}_tilt_mean']:.4f}/{stats[f'{side}_tilt_std']:.4f}/{stats[f'{side}_tilt_rms']:.4f}, "
            f"upright min/inverted = {stats[f'{side}_upright_alignment_min']:.4f}/{stats[f'{side}_inverted_fraction'] * 100.0:.2f}%"
        )

    print(
        f"总周期数 = {total_cycles}, warm-up = {warmup_cycles}, evaluation = {evaluation_cycles}, "
        f"cooldown = {cooldown_cycles}, 本次仿真 torso xy 行走距离 = {walk_distance:.3f} m"
    )
