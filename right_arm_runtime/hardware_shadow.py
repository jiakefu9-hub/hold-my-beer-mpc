"""Fail-closed Unitree G1 state conversion and MPC shadow orchestration.

This module never imports a Unitree command publisher and never writes the
command shared-memory slot.  It converts a read-only LowState snapshot into
the existing predictor/MPC inputs and builds an in-memory, output-disabled
command proposal for inspection only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Optional, Protocol

import mujoco
import numpy as np
import yaml

from disturbance_predictor import (
    DisturbancePredictorObservation,
    create_disturbance_predictor,
)
from disturbance_template.full_task_protocol import (
    DEFAULT_FULL_TASK_PROTOCOL,
    FixedStartupPdHandoff,
)
from kinematics_helper import KinematicsHelper
from robot_model_backend import create_prediction_backend
from right_arm_control_setup import RIGHT_ARM_JOINT_NAMES, create_arm_controller

from .control_contracts import ControlStateCapabilities, TaskClockEvent
from .full_task_control_core import (
    FullTaskControlIntent,
    FullTaskControlObservation,
    FullTaskRightArmControlCore,
)
from .hardware_output_contract import FutureCommandMode, HardwareControlProposal

from .unitree_shm import (
    ARM_SDK_JOINT_COUNT,
    CommandMode,
    RobotStateSnapshot,
    StateIngressFlags,
)


G1_23DOF_MOTOR_TO_JOINT = {
    0: "left_hip_pitch_joint",
    1: "left_hip_roll_joint",
    2: "left_hip_yaw_joint",
    3: "left_knee_joint",
    4: "left_ankle_pitch_joint",
    5: "left_ankle_roll_joint",
    6: "right_hip_pitch_joint",
    7: "right_hip_roll_joint",
    8: "right_hip_yaw_joint",
    9: "right_knee_joint",
    10: "right_ankle_pitch_joint",
    11: "right_ankle_roll_joint",
    12: "waist_yaw_joint",
    15: "left_shoulder_pitch_joint",
    16: "left_shoulder_roll_joint",
    17: "left_shoulder_yaw_joint",
    18: "left_elbow_joint",
    19: "left_wrist_roll_joint",
    22: "right_shoulder_pitch_joint",
    23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint",
    25: "right_elbow_joint",
    26: "right_wrist_roll_joint",
}
LOWER_BODY_MOTOR_INDICES = tuple(range(12))
RIGHT_ARM_MOTOR_INDICES = (22, 23, 24, 25, 26)
# Official G1 arm5 SDK ordering: left arm5, right arm5, waist3.
ARM_SDK_MOTOR_INDICES = (
    15,
    16,
    17,
    18,
    19,
    22,
    23,
    24,
    25,
    26,
    12,
    13,
    14,
)
PROTOCOL_V3_REQUIRED_INGRESS_FLAGS = int(
    StateIngressFlags.LOW_STATE_CRC_VALID
    | StateIngressFlags.PAIRED_INGRESS_VALIDATED
    | StateIngressFlags.TORSO_IMU_PRESENT
)
PROTOCOL_V3_MAX_SOURCE_SKEW_NS = 5_000_000


class HardwareContractError(RuntimeError):
    """Declared robot/message contract is missing or not verified."""


class HardwareStateError(RuntimeError):
    """A state snapshot is stale, inconsistent, or outside declared bounds."""


def validate_protocol_v3_ingress(
    state: RobotStateSnapshot,
    *,
    expected_session_nonce: int,
    now_ns: int,
    future_tolerance_ns: int,
) -> None:
    """Fail closed on unpaired, replayed-session, or forged state provenance."""

    expected_nonce = int(expected_session_nonce)
    if expected_nonce <= 0:
        raise HardwareContractError(
            "expected protocol-v3 ingress session nonce must be nonzero"
        )
    if int(state.ingress_session_nonce) != expected_nonce:
        raise HardwareStateError("protocol-v3 ingress session nonce mismatch")
    flags = int(state.ingress_flags)
    if (flags & PROTOCOL_V3_REQUIRED_INGRESS_FLAGS) != (
        PROTOCOL_V3_REQUIRED_INGRESS_FLAGS
    ):
        raise HardwareStateError("protocol-v3 ingress validation flags missing")
    if flags & int(StateIngressFlags.SYNTHETIC_FIXTURE):
        raise HardwareStateError(
            "synthetic protocol-v3 ingress is forbidden in hardware shadow"
        )
    low_state_ns = int(state.low_state_timestamp_ns)
    torso_imu_ns = int(state.torso_imu_timestamp_ns)
    source_ns = int(state.monotonic_timestamp_ns)
    validated_ns = int(state.validated_timestamp_ns)
    source_skew_ns = int(state.source_skew_ns)
    if min(low_state_ns, torso_imu_ns, source_ns, validated_ns) <= 0:
        raise HardwareStateError("protocol-v3 ingress timestamps must be positive")
    expected_source_ns = min(low_state_ns, torso_imu_ns)
    expected_skew_ns = abs(low_state_ns - torso_imu_ns)
    if source_ns != expected_source_ns or source_skew_ns != expected_skew_ns:
        raise HardwareStateError("protocol-v3 ingress timestamp/skew mismatch")
    if source_skew_ns > PROTOCOL_V3_MAX_SOURCE_SKEW_NS:
        raise HardwareStateError("protocol-v3 paired source skew exceeds 5 ms")
    if validated_ns < max(low_state_ns, torso_imu_ns):
        raise HardwareStateError("protocol-v3 validation predates source samples")
    if validated_ns > int(now_ns) + int(future_tolerance_ns):
        raise HardwareStateError("protocol-v3 validation timestamp is in the future")


class HardwareStateSource(Protocol):
    """Minimum source boundary used by the hardware shadow runner."""

    def read_state(self, *, max_attempts: int = 100) -> RobotStateSnapshot:
        """Return one internally consistent, monotonic state snapshot."""


@dataclass(frozen=True)
class HardwareFrameContract:
    robot_model: str
    joint_mapping: str
    joint_mapping_verified: bool
    robot_tick_monotonic_verified: bool
    imu_contract_verified: bool
    imu_source_topic: str
    imu_model_site: str
    quaternion_order: str
    orientation_semantics: str
    angular_velocity_frame: str
    linear_acceleration_frame: str
    linear_acceleration_semantics: str
    torso_from_imu_rotation: np.ndarray
    allowed_mode_pr: tuple[int, ...]
    allowed_mode_machine: tuple[int, ...]
    state_timeout_ns: int
    future_tolerance_ns: int
    state_dt_min_s: float
    state_dt_max_s: float
    joint_position_tolerance_rad: float
    joint_velocity_abs_max_rad_s: float
    gyroscope_abs_max_rad_s: float
    accelerometer_abs_max_m_s2: float
    angular_acceleration_abs_max_rad_s2: float
    motor_case_temperature_max_c: float
    motor_winding_temperature_max_c: float

    @classmethod
    def from_mapping(
        cls, value: dict, *, require_verified: bool = True
    ) -> "HardwareFrameContract":
        if not isinstance(value, dict):
            raise HardwareContractError("hardware_shadow config must be a mapping")
        imu = value.get("imu", {})
        if not isinstance(imu, dict):
            raise HardwareContractError("hardware_shadow.imu must be a mapping")
        rotation = np.asarray(
            imu.get("torso_from_imu_rotation", np.eye(3)), dtype=np.float64
        )
        if (
            rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise HardwareContractError(
                "torso_from_imu_rotation must be a proper 3x3 rotation"
            )
        contract = cls(
            robot_model=str(value.get("robot_model", "")),
            joint_mapping=str(value.get("joint_mapping", "")),
            joint_mapping_verified=bool(
                value.get("joint_mapping_verified", False)
            ),
            robot_tick_monotonic_verified=bool(
                value.get("robot_tick_monotonic_verified", False)
            ),
            imu_contract_verified=bool(
                imu.get("contract_verified", False)
            ),
            imu_source_topic=str(imu.get("source_topic", "")),
            imu_model_site=str(imu.get("model_site", "")),
            quaternion_order=str(imu.get("quaternion_order", "")),
            orientation_semantics=str(
                imu.get("orientation_semantics", "")
            ),
            angular_velocity_frame=str(
                imu.get("angular_velocity_frame", "")
            ),
            linear_acceleration_frame=str(
                imu.get("linear_acceleration_frame", "")
            ),
            linear_acceleration_semantics=str(
                imu.get("linear_acceleration_semantics", "")
            ),
            torso_from_imu_rotation=rotation.copy(),
            allowed_mode_pr=tuple(int(x) for x in value.get("allowed_mode_pr", ())),
            allowed_mode_machine=tuple(
                int(x) for x in value.get("allowed_mode_machine", ())
            ),
            state_timeout_ns=int(
                round(float(value.get("state_timeout_ms", 20.0)) * 1e6)
            ),
            future_tolerance_ns=int(
                round(float(value.get("future_tolerance_ms", 1.0)) * 1e6)
            ),
            state_dt_min_s=float(value.get("state_dt_min_ms", 0.2)) * 1e-3,
            state_dt_max_s=float(value.get("state_dt_max_ms", 20.0)) * 1e-3,
            joint_position_tolerance_rad=float(
                value.get("joint_position_tolerance_rad", 0.02)
            ),
            joint_velocity_abs_max_rad_s=float(
                value.get("joint_velocity_abs_max_rad_s", 20.0)
            ),
            gyroscope_abs_max_rad_s=float(
                value.get("gyroscope_abs_max_rad_s", 20.0)
            ),
            accelerometer_abs_max_m_s2=float(
                value.get("accelerometer_abs_max_m_s2", 100.0)
            ),
            angular_acceleration_abs_max_rad_s2=float(
                value.get("angular_acceleration_abs_max_rad_s2", 200.0)
            ),
            motor_case_temperature_max_c=float(
                value.get("motor_case_temperature_max_c", 85.0)
            ),
            motor_winding_temperature_max_c=float(
                value.get("motor_winding_temperature_max_c", 120.0)
            ),
        )
        contract.validate(require_verified=require_verified)
        return contract

    def validate(self, *, require_verified: bool) -> None:
        expected = {
            "robot_model": (self.robot_model, "g1_23dof_rev_1_0"),
            "joint_mapping": (
                self.joint_mapping,
                "unitree_sdk2_g1_23dof_arm5",
            ),
            "quaternion_order": (self.quaternion_order, "wxyz"),
            "imu_source_topic": (
                self.imu_source_topic,
                "rt/secondary_imu",
            ),
            "imu_model_site": (self.imu_model_site, "imu_in_torso"),
            "orientation_semantics": (
                self.orientation_semantics,
                "world_from_imu",
            ),
            "angular_velocity_frame": (
                self.angular_velocity_frame,
                "imu",
            ),
            "linear_acceleration_frame": (
                self.linear_acceleration_frame,
                "imu",
            ),
        }
        errors = [
            f"{name}={actual!r}, expected {wanted!r}"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if self.linear_acceleration_semantics not in {
            "specific_force",
            "linear_acceleration",
        }:
            errors.append(
                "linear_acceleration_semantics must be specific_force or "
                "linear_acceleration"
            )
        positive = {
            "state_timeout_ns": self.state_timeout_ns,
            "future_tolerance_ns": self.future_tolerance_ns,
            "state_dt_min_s": self.state_dt_min_s,
            "state_dt_max_s": self.state_dt_max_s,
            "joint_velocity_abs_max_rad_s": self.joint_velocity_abs_max_rad_s,
            "gyroscope_abs_max_rad_s": self.gyroscope_abs_max_rad_s,
            "accelerometer_abs_max_m_s2": self.accelerometer_abs_max_m_s2,
            "angular_acceleration_abs_max_rad_s2": (
                self.angular_acceleration_abs_max_rad_s2
            ),
            "motor_case_temperature_max_c": (
                self.motor_case_temperature_max_c
            ),
            "motor_winding_temperature_max_c": (
                self.motor_winding_temperature_max_c
            ),
        }
        errors.extend(
            f"{name} must be positive"
            for name, number in positive.items()
            if not math.isfinite(float(number)) or float(number) <= 0.0
        )
        if self.state_dt_min_s >= self.state_dt_max_s:
            errors.append("state_dt_min_s must be smaller than state_dt_max_s")
        if (
            not math.isfinite(self.joint_position_tolerance_rad)
            or self.joint_position_tolerance_rad < 0.0
        ):
            errors.append("joint_position_tolerance_rad must be nonnegative")
        if any(not 0 <= mode <= 255 for mode in self.allowed_mode_pr):
            errors.append("allowed_mode_pr values must fit uint8")
        if any(not 0 <= mode <= 255 for mode in self.allowed_mode_machine):
            errors.append("allowed_mode_machine values must fit uint8")
        if require_verified:
            if not self.joint_mapping_verified:
                errors.append("joint_mapping_verified is false")
            if not self.robot_tick_monotonic_verified:
                errors.append("robot_tick_monotonic_verified is false")
            if not self.imu_contract_verified:
                errors.append("imu.contract_verified is false")
            if not self.allowed_mode_pr:
                errors.append("allowed_mode_pr is empty")
            if not self.allowed_mode_machine:
                errors.append("allowed_mode_machine is empty")
        if errors:
            raise HardwareContractError("; ".join(errors))


def load_hardware_shadow_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("hardware_shadow"), dict
    ):
        raise HardwareContractError(
            f"{config_path} must contain hardware_shadow mapping"
        )
    if bool(payload["hardware_shadow"].get("output_enabled", True)):
        raise HardwareContractError(
            "hardware shadow output_enabled must remain false"
        )
    return payload


@dataclass(frozen=True)
class HardwareObservation:
    monotonic_timestamp_ns: int
    sample_id: int
    state_age_ns: int
    validated_timestamp_ns: int
    capabilities: ControlStateCapabilities
    qpos_mujoco: np.ndarray
    qvel_mujoco: np.ndarray
    right_arm_q: np.ndarray
    right_arm_dq: np.ndarray
    lower_body_q: np.ndarray
    lower_body_dq: np.ndarray
    torso_rotation_world: np.ndarray
    torso_angular_velocity_world: np.ndarray
    torso_linear_acceleration_world: np.ndarray
    torso_angular_acceleration_world: np.ndarray
    gravity_direction_torso: np.ndarray
    derivative_ready: bool


class G1HardwareStateAdapter:
    """Convert one verified Unitree LowState snapshot to controller state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        contract: HardwareFrameContract,
        *,
        expected_ingress_session_nonce: int,
    ):
        self.model = model
        self.contract = contract
        self.expected_ingress_session_nonce = int(
            expected_ingress_session_nonce
        )
        if self.expected_ingress_session_nonce <= 0:
            raise HardwareContractError(
                "expected protocol-v3 ingress session nonce must be nonzero"
            )
        self._mapping = self._resolve_model_mapping()
        self._scratch = mujoco.MjData(model)
        self._imu_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, contract.imu_model_site
        )
        if self._imu_site_id < 0:
            raise HardwareContractError(
                f"MJCF missing IMU site {contract.imu_model_site!r}"
            )
        self._previous_timestamp_ns: Optional[int] = None
        self._previous_sample_id: Optional[int] = None
        self._previous_robot_tick: Optional[int] = None
        self._previous_omega_world: Optional[np.ndarray] = None

    def _resolve_model_mapping(self) -> tuple[tuple[int, int, int], ...]:
        mapping = []
        for motor, joint_name in G1_23DOF_MOTOR_TO_JOINT.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise HardwareContractError(
                    f"MJCF missing mapped joint {joint_name!r}"
                )
            mapping.append(
                (
                    motor,
                    int(self.model.jnt_qposadr[joint_id]),
                    int(self.model.jnt_dofadr[joint_id]),
                )
            )
        if len(mapping) != 23:
            raise HardwareContractError("G1 23-DOF mapping is incomplete")
        return tuple(mapping)

    @staticmethod
    def _rotation_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
        w, x, y, z = quaternion
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, np.asarray(rotation).reshape(-1))
        if quaternion[0] < 0.0:
            quaternion *= -1.0
        return quaternion

    def inspect_snapshot(
        self, state: RobotStateSnapshot, *, now_ns: Optional[int] = None
    ) -> dict:
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        quaternion = np.asarray(state.imu_quaternion_wxyz, dtype=np.float64)
        temperatures = np.asarray(
            state.motor_temperature_c, dtype=np.float64
        )
        return {
            "sample_id": int(state.sample_id),
            "robot_tick": int(state.robot_tick),
            "state_age_ms": (now - int(state.monotonic_timestamp_ns)) * 1e-6,
            "mode_pr": int(state.mode_pr),
            "mode_machine": int(state.mode_machine),
            "quaternion_wxyz": quaternion.tolist(),
            "quaternion_norm": float(np.linalg.norm(quaternion)),
            "gyroscope_raw": list(state.imu_gyroscope),
            "accelerometer_raw": list(state.imu_accelerometer),
            "motor_case_temperature_c_max": float(
                np.max(temperatures[:, 0])
            ),
            "motor_winding_temperature_c_max": float(
                np.max(temperatures[:, 1])
            ),
            "right_arm_q_rad": [state.q[index] for index in RIGHT_ARM_MOTOR_INDICES],
            "right_arm_dq_rad_s": [state.dq[index] for index in RIGHT_ARM_MOTOR_INDICES],
        }

    def convert(
        self, state: RobotStateSnapshot, *, now_ns: Optional[int] = None
    ) -> HardwareObservation:
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        validate_protocol_v3_ingress(
            state,
            expected_session_nonce=self.expected_ingress_session_nonce,
            now_ns=now,
            future_tolerance_ns=self.contract.future_tolerance_ns,
        )
        timestamp = int(state.monotonic_timestamp_ns)
        sample_id = int(state.sample_id)
        robot_tick = int(state.robot_tick)
        if timestamp <= 0 or sample_id <= 0:
            raise HardwareStateError("state timestamp/sample_id must be positive")
        if not 0 <= robot_tick <= 0xFFFFFFFF:
            raise HardwareStateError("robot tick must fit uint32")
        if timestamp > now + self.contract.future_tolerance_ns:
            raise HardwareStateError("state timestamp is in the future")
        age = max(0, now - timestamp)
        if age > self.contract.state_timeout_ns:
            raise HardwareStateError(
                f"state is stale: {age * 1e-6:.3f} ms"
            )
        if int(state.mode_pr) not in self.contract.allowed_mode_pr:
            raise HardwareStateError(f"unexpected mode_pr={state.mode_pr}")
        if int(state.mode_machine) not in self.contract.allowed_mode_machine:
            raise HardwareStateError(
                f"unexpected mode_machine={state.mode_machine}"
            )
        if self._previous_sample_id is not None:
            if sample_id <= self._previous_sample_id:
                raise HardwareStateError("state sample_id repeated or regressed")
            if timestamp <= int(self._previous_timestamp_ns):
                raise HardwareStateError("state timestamp repeated or regressed")
            tick_delta = (
                robot_tick - int(self._previous_robot_tick)
            ) & 0xFFFFFFFF
            if tick_delta == 0 or tick_delta >= (1 << 31):
                raise HardwareStateError("robot tick repeated or regressed")

        arrays = {
            "q": np.asarray(state.q, dtype=np.float64),
            "dq": np.asarray(state.dq, dtype=np.float64),
            "ddq": np.asarray(state.ddq, dtype=np.float64),
            "tau_est": np.asarray(state.tau_est, dtype=np.float64),
            "temperature": np.asarray(
                state.motor_temperature_c, dtype=np.float64
            ),
            "quaternion": np.asarray(
                state.imu_quaternion_wxyz, dtype=np.float64
            ),
            "gyroscope": np.asarray(state.imu_gyroscope, dtype=np.float64),
            "accelerometer": np.asarray(
                state.imu_accelerometer, dtype=np.float64
            ),
            "rpy": np.asarray(state.imu_rpy, dtype=np.float64),
        }
        expected = {
            "q": (35,),
            "dq": (35,),
            "ddq": (35,),
            "tau_est": (35,),
            "temperature": (35, 2),
            "quaternion": (4,),
            "gyroscope": (3,),
            "accelerometer": (3,),
            "rpy": (3,),
        }
        for name, value in arrays.items():
            if value.shape != expected[name] or not np.all(np.isfinite(value)):
                raise HardwareStateError(
                    f"{name} has invalid shape or nonfinite values"
                )
        if np.max(np.abs(arrays["dq"])) > self.contract.joint_velocity_abs_max_rad_s:
            raise HardwareStateError("joint velocity exceeds declared bound")
        if np.max(np.abs(arrays["gyroscope"])) > self.contract.gyroscope_abs_max_rad_s:
            raise HardwareStateError("gyroscope exceeds declared bound")
        if np.max(np.abs(arrays["accelerometer"])) > self.contract.accelerometer_abs_max_m_s2:
            raise HardwareStateError("accelerometer exceeds declared bound")
        if np.max(arrays["temperature"][:, 0]) > self.contract.motor_case_temperature_max_c:
            raise HardwareStateError("motor case temperature exceeds declared bound")
        if np.max(arrays["temperature"][:, 1]) > self.contract.motor_winding_temperature_max_c:
            raise HardwareStateError("motor winding temperature exceeds declared bound")

        quaternion = arrays["quaternion"]
        quaternion_norm = float(np.linalg.norm(quaternion))
        if not 0.8 <= quaternion_norm <= 1.2:
            raise HardwareStateError("IMU quaternion norm is invalid")
        quaternion = quaternion / quaternion_norm
        world_from_imu = self._rotation_from_wxyz(quaternion)
        torso_from_imu = self.contract.torso_from_imu_rotation
        world_from_torso = world_from_imu @ torso_from_imu.T
        omega_world = world_from_imu @ arrays["gyroscope"]
        acceleration_world = world_from_imu @ arrays["accelerometer"]
        if self.contract.linear_acceleration_semantics == "specific_force":
            acceleration_world += np.asarray(self.model.opt.gravity)

        derivative_ready = self._previous_omega_world is not None
        if derivative_ready:
            dt = (timestamp - int(self._previous_timestamp_ns)) * 1e-9
            if not self.contract.state_dt_min_s <= dt <= self.contract.state_dt_max_s:
                raise HardwareStateError(f"state sample dt={dt:.6f} s is invalid")
            alpha_world = (omega_world - self._previous_omega_world) / dt
            if np.max(np.abs(alpha_world)) > self.contract.angular_acceleration_abs_max_rad_s2:
                raise HardwareStateError("derived angular acceleration exceeds bound")
        else:
            alpha_world = np.zeros(3, dtype=np.float64)

        qpos = np.asarray(self.model.qpos0, dtype=np.float64).copy()
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        # LowState gives torso/IMU orientation, while the MJCF free joint is
        # the pelvis.  First evaluate the measured waist/joints with an
        # identity pelvis, then solve the pelvis rotation that makes the
        # model IMU frame exactly match the verified hardware torso frame.
        qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
        for motor, q_index, v_index in self._mapping:
            qpos[q_index] = arrays["q"][motor]
            qvel[v_index] = arrays["dq"][motor]
            joint_id = int(self.model.dof_jntid[v_index])
            if bool(self.model.jnt_limited[joint_id]):
                lower, upper = self.model.jnt_range[joint_id]
                tolerance = self.contract.joint_position_tolerance_rad
                if not lower - tolerance <= qpos[q_index] <= upper + tolerance:
                    name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                    )
                    raise HardwareStateError(
                        f"joint {name} outside MJCF physical range"
                    )
        self._scratch.qpos[:] = qpos
        self._scratch.qvel[:] = qvel
        mujoco.mj_fwdPosition(self.model, self._scratch)
        pelvis_from_model_imu = self._scratch.site_xmat[
            self._imu_site_id
        ].reshape(3, 3)
        world_from_pelvis = (
            world_from_torso @ pelvis_from_model_imu.T
        )
        qpos[3:7] = self._wxyz_from_rotation(world_from_pelvis)

        self._previous_timestamp_ns = timestamp
        self._previous_sample_id = sample_id
        self._previous_robot_tick = robot_tick
        self._previous_omega_world = omega_world.copy()
        return HardwareObservation(
            monotonic_timestamp_ns=timestamp,
            sample_id=sample_id,
            state_age_ns=age,
            validated_timestamp_ns=int(state.validated_timestamp_ns),
            capabilities=ControlStateCapabilities(
                right_arm_joint_state=True,
                torso_rotation=True,
                torso_angular_velocity=True,
                torso_linear_acceleration=True,
                torso_angular_acceleration=derivative_ready,
                # The paired LowState/torso-IMU ingress does not provide these
                # state families. Keep them explicitly unknown instead of
                # presenting nominal MJCF/zero-filled values as measurements.
                floating_base_translation=False,
                floating_base_velocity=False,
                foot_contacts=False,
                external_forces=False,
            ),
            qpos_mujoco=qpos,
            qvel_mujoco=qvel,
            right_arm_q=arrays["q"][list(RIGHT_ARM_MOTOR_INDICES)].copy(),
            right_arm_dq=arrays["dq"][list(RIGHT_ARM_MOTOR_INDICES)].copy(),
            lower_body_q=arrays["q"][list(LOWER_BODY_MOTOR_INDICES)].copy(),
            lower_body_dq=arrays["dq"][list(LOWER_BODY_MOTOR_INDICES)].copy(),
            torso_rotation_world=world_from_torso,
            torso_angular_velocity_world=omega_world,
            torso_linear_acceleration_world=acceleration_world,
            torso_angular_acceleration_world=alpha_world,
            gravity_direction_torso=(
                world_from_torso.T @ np.asarray([0.0, 0.0, -1.0])
            ),
            derivative_ready=derivative_ready,
        )


@dataclass(frozen=True)
class ShadowArmCommand:
    source_state_timestamp_ns: int
    source_sample_id: int
    command_mode: int
    arm_weight: float
    q_ref: tuple[float, ...]
    dq_ref: tuple[float, ...]
    ddq_des: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    tau_ff: tuple[float, ...]
    request_output: bool = False
    publish_performed: bool = False
    ready_for_output: bool = False
    torque_semantics: str = "not_computed_missing_full_state_estimator"


class ShadowCommandBuilder:
    """Build protocol-shaped command proposals without any write capability."""

    def __init__(self, config: dict):
        kp_config = np.asarray(config["arm_waist_kps"], dtype=np.float64)
        kd_config = np.asarray(config["arm_waist_kds"], dtype=np.float64)
        if kp_config.shape != (11,) or kd_config.shape != (11,):
            raise HardwareContractError("arm_waist_kps/kds must have length 11")
        if (
            not np.all(np.isfinite(kp_config))
            or not np.all(np.isfinite(kd_config))
            or np.any(kp_config < 0.0)
            or np.any(kd_config < 0.0)
        ):
            raise HardwareContractError("arm_waist_kps/kds must be finite and nonnegative")
        self.kp = np.zeros(ARM_SDK_JOINT_COUNT, dtype=np.float64)
        self.kd = np.zeros(ARM_SDK_JOINT_COUNT, dtype=np.float64)
        self.kp[:5], self.kd[:5] = kp_config[1:6], kd_config[1:6]
        self.kp[5:10], self.kd[5:10] = kp_config[6:11], kd_config[6:11]
        self.kp[10], self.kd[10] = kp_config[0], kd_config[0]

    def build(
        self,
        observation: HardwareObservation,
        q_ref_right: np.ndarray,
        dq_ref_right: np.ndarray,
        ddq_des_right: np.ndarray,
        raw_state: RobotStateSnapshot,
    ) -> ShadowArmCommand:
        vectors = [
            np.asarray(value, dtype=np.float64)
            for value in (q_ref_right, dq_ref_right, ddq_des_right)
        ]
        if any(value.shape != (5,) for value in vectors) or not all(
            np.all(np.isfinite(value)) for value in vectors
        ):
            raise HardwareStateError("MPC command vectors are invalid")
        q_ref = np.asarray(
            [raw_state.q[index] for index in ARM_SDK_MOTOR_INDICES],
            dtype=np.float64,
        )
        dq_ref = np.zeros(ARM_SDK_JOINT_COUNT, dtype=np.float64)
        ddq_des = np.zeros(ARM_SDK_JOINT_COUNT, dtype=np.float64)
        q_ref[5:10] = vectors[0]
        dq_ref[5:10] = vectors[1]
        ddq_des[5:10] = vectors[2]
        return ShadowArmCommand(
            source_state_timestamp_ns=observation.monotonic_timestamp_ns,
            source_sample_id=observation.sample_id,
            command_mode=int(CommandMode.ROBOT_PD_PLUS_FEEDFORWARD),
            # A shadow proposal never asks the firmware to take arm control.
            arm_weight=0.0,
            q_ref=tuple(q_ref),
            dq_ref=tuple(dq_ref),
            ddq_des=tuple(ddq_des),
            kp=tuple(self.kp),
            kd=tuple(self.kd),
            tau_ff=(0.0,) * ARM_SDK_JOINT_COUNT,
        )


@dataclass(frozen=True)
class ShadowCycleResult:
    source_sample_id: int
    logical_time_s: float
    command: ShadowArmCommand
    mpc_success: bool
    predictor_requested: str
    predictor_used: str
    predictor_fallback_reason: str
    timing_s: dict
    diagnostics: dict


@dataclass(frozen=True)
class FullTaskShadowCycleResult:
    """One final-controller H3-offline cycle with no output capability."""

    source_sample_id: int
    task_time_s: float
    task_anchor: int
    intent: FullTaskControlIntent
    command: ShadowArmCommand
    proposal: HardwareControlProposal
    timing_s: dict
    diagnostics: dict


class HardwareShadowController:
    """Run predictor + MPC + command build with no output sink."""

    def __init__(
        self,
        *,
        repo_dir: str | Path,
        controller_config: dict,
        contract: HardwareFrameContract,
        expected_ingress_session_nonce: int,
        predictor_name: Optional[str] = None,
    ):
        self.repo_dir = Path(repo_dir).resolve()
        self.config = dict(controller_config)
        self.control_dt = float(self.config["simulation_dt"]) * int(
            self.config["arm_control_decimation"]
        )
        if not np.isclose(self.control_dt, 0.006, atol=1e-12):
            raise HardwareContractError("hardware shadow control_dt must be 6 ms")
        xml_path = Path(self.config["xml_path"])
        if not xml_path.is_absolute():
            xml_path = self.repo_dir / xml_path
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.state_adapter = G1HardwareStateAdapter(
            self.model,
            contract,
            expected_ingress_session_nonce=expected_ingress_session_nonce,
        )
        self._right_joint_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                for name in RIGHT_ARM_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self._right_joint_ids < 0):
            raise HardwareContractError("right-arm MJCF joint mapping failed")
        self._right_q_indices = self.model.jnt_qposadr[self._right_joint_ids]
        right_target = np.asarray(
            self.config["arm_waist_target"], dtype=np.float64
        )[6:11]
        self.policy = create_arm_controller(
            self.config, "mpc", right_target, self.control_dt
        ).policy
        backend_name = str(
            self.config.get("mpc_prediction_kinematics_backend", "cpp_pinocchio")
        )
        self.prediction_backend = create_prediction_backend(
            backend_name,
            mujoco_model=self.model,
            joint_names=RIGHT_ARM_JOINT_NAMES,
            mjcf_path=xml_path,
            ee_name="right_grasp_site",
            imu_name="imu_in_torso",
        )
        self.helper = KinematicsHelper(
            self.model,
            ee_site_name="right_grasp_site",
            joint_indices=self._right_q_indices,
            imu_site_name="imu_in_torso",
            position_reference_q=right_target,
            prediction_backend=self.prediction_backend,
        )
        requested_config = dict(self.config)
        requested_name = (
            "template" if predictor_name is None else str(predictor_name)
        ).strip().lower()
        if requested_name not in {"template", "full_task_template"}:
            raise HardwareContractError(
                "hardware shadow predictor must be template or full_task_template"
            )
        requested_config["disturbance_predictor"] = requested_name
        self.predictor_requested = requested_name
        self.predictor = create_disturbance_predictor(
            requested_config,
            repo_dir=str(self.repo_dir),
            control_dt=self.control_dt,
            horizon=self.policy.horizon,
            acc_limit=float(self.config["ddq_torso_acc_limit"]),
            alpha_limit=float(self.config["ddq_torso_alpha_limit"]),
        )
        self.command_builder = ShadowCommandBuilder(self.config)
        self._right_target = right_target.copy()
        self._full_task_core = (
            FullTaskRightArmControlCore(
                predictor=self.predictor,
                arm_policy=self.policy,
                nominal_command=np.asarray(self.config["cmd_init"], dtype=np.float64),
                protocol=DEFAULT_FULL_TASK_PROTOCOL,
                startup_handoff=FixedStartupPdHandoff(
                    0.024, DEFAULT_FULL_TASK_PROTOCOL
                ),
            )
            if requested_name == "full_task_template"
            else None
        )
        self._full_task_session_nonce: Optional[str] = None
        self._full_task_epoch_id: Optional[str] = None
        self._full_task_proposal_count = 0
        self._next_control_timestamp_ns: Optional[int] = None
        self._control_index = 0
        self._previous_mpc_success: Optional[bool] = None
        self._filtered_acc = np.zeros(3, dtype=np.float64)
        self._filtered_alpha = np.zeros(3, dtype=np.float64)
        self._timing_samples: list[dict] = []

    def close(self) -> None:
        close = getattr(self.prediction_backend, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "HardwareShadowController":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def inspect_snapshot(self, *args, **kwargs) -> dict:
        return self.state_adapter.inspect_snapshot(*args, **kwargs)

    def prime_full_task_state(
        self,
        raw_state: RobotStateSnapshot,
        *,
        now_ns: Optional[int] = None,
    ) -> HardwareObservation:
        """Consume pre-task history without starting or advancing task time."""

        if self._full_task_core is None:
            raise HardwareContractError(
                "prime_full_task_state requires predictor=full_task_template"
            )
        if self._full_task_core.ready:
            raise HardwareContractError(
                "full-task state priming is forbidden after the task epoch starts"
            )
        return self.state_adapter.convert(raw_state, now_ns=now_ns)

    def reset_full_task_epoch(
        self, *, session_nonce: str, task_epoch_id: str
    ) -> None:
        """Start only the explicit controller epoch; never infer it from state."""

        if self._full_task_core is None:
            raise HardwareContractError(
                "reset_full_task_epoch requires predictor=full_task_template"
            )
        self._full_task_core.reset(
            session_nonce=session_nonce, task_epoch_id=task_epoch_id
        )
        self._full_task_session_nonce = str(session_nonce)
        self._full_task_epoch_id = str(task_epoch_id)
        self._full_task_proposal_count = 0

    def _filter_disturbance(
        self, acceleration: np.ndarray, alpha: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        acc_gain = float(self.config.get("mpc_torso_acc_filter_alpha", 0.5))
        alpha_gain = float(
            self.config.get("mpc_torso_alpha_filter_alpha", 0.5)
        )
        acc_limit = float(self.config["ddq_torso_acc_limit"])
        alpha_limit = float(self.config["ddq_torso_alpha_limit"])
        self._filtered_acc += acc_gain * (
            np.clip(acceleration, -acc_limit, acc_limit) - self._filtered_acc
        )
        self._filtered_alpha += alpha_gain * (
            np.clip(alpha, -alpha_limit, alpha_limit) - self._filtered_alpha
        )
        return self._filtered_acc.copy(), self._filtered_alpha.copy()

    def process(
        self,
        raw_state: RobotStateSnapshot,
        *,
        now_ns: Optional[int] = None,
        state_read_time_s: float = 0.0,
        task_clock_event: Optional[TaskClockEvent] = None,
    ) -> Optional[ShadowCycleResult | FullTaskShadowCycleResult]:
        started = time.perf_counter()
        observation = self.state_adapter.convert(raw_state, now_ns=now_ns)
        conversion_time = time.perf_counter() - started
        if self._full_task_core is not None:
            if task_clock_event is None:
                raise HardwareContractError(
                    "full_task_template requires an explicit TaskClockEvent"
                )
            if not observation.derivative_ready:
                raise HardwareStateError(
                    "full-task anchor state lacks causal angular acceleration; "
                    "prime pre-task state before resetting the task epoch"
                )
            return self._process_full_task(
                raw_state,
                observation,
                task_clock_event,
                started=started,
                state_read_time_s=state_read_time_s,
                conversion_time=conversion_time,
            )
        if not observation.derivative_ready:
            return None
        timestamp = observation.monotonic_timestamp_ns
        if self._next_control_timestamp_ns is None:
            self._next_control_timestamp_ns = timestamp
        tolerance_ns = 1_000_000
        if timestamp + tolerance_ns < self._next_control_timestamp_ns:
            return None
        while self._next_control_timestamp_ns <= timestamp + tolerance_ns:
            self._next_control_timestamp_ns += int(round(self.control_dt * 1e9))

        logical_time = self._control_index * self.control_dt
        self._control_index += 1
        filtered_acc, filtered_alpha = self._filter_disturbance(
            observation.torso_linear_acceleration_world,
            observation.torso_angular_acceleration_world,
        )
        disturbance = self.helper.build_disturbance_input(
            acc_world=filtered_acc,
            omega_world=observation.torso_angular_velocity_world,
            alpha_world=filtered_alpha,
            rot_world_body=observation.torso_rotation_world,
        )

        predictor_started = time.perf_counter()
        self.predictor.update(
            DisturbancePredictorObservation(
                simulation_time=logical_time,
                measured_disturbance=disturbance,
            )
        )
        horizon = self.predictor.predict(self.policy.horizon, self.control_dt)
        predictor_time = time.perf_counter() - predictor_started

        helper_started = time.perf_counter()
        model_state = SimpleNamespace(
            qpos=observation.qpos_mujoco,
            qvel=observation.qvel_mujoco,
        )
        helpers = self.helper.build_helpers(
            model_state,
            disturbance=disturbance,
            disturbance_prediction=horizon.nodes,
            interval_disturbance_prediction=horizon.intervals,
            include_kinematics_cache=False,
        )
        helper_time = time.perf_counter() - helper_started

        mpc_started = time.perf_counter()
        control_observation = self.helper.build_observation(
            current_q=observation.right_arm_q,
            current_dq=observation.right_arm_dq,
            torso_quat=observation.qpos_mujoco[3:7],
            torso_omega=observation.torso_angular_velocity_world,
            torso_acc=filtered_acc,
            torso_alpha=filtered_alpha,
            torso_rotmat=observation.torso_rotation_world,
            dt=self.control_dt,
        )
        q_ref, dq_ref, ddq_des = self.policy.compute_action(
            control_observation, helpers
        )
        mpc_time = time.perf_counter() - mpc_started
        diagnostics = self.policy.get_last_diagnostics(copy_data=False)

        command_started = time.perf_counter()
        command = self.command_builder.build(
            observation, q_ref, dq_ref, ddq_des, raw_state
        )
        command_time = time.perf_counter() - command_started
        compute_time = time.perf_counter() - started
        total_time = float(state_read_time_s) + compute_time
        timing = {
            "state_read": float(state_read_time_s),
            "source_state_age": float(observation.state_age_ns * 1e-9),
            "source_to_command_age": float(
                observation.state_age_ns * 1e-9 + compute_time
            ),
            "state_conversion": float(conversion_time),
            "predictor": float(predictor_time),
            "helper": float(helper_time),
            "mpc": float(mpc_time),
            "command_build": float(command_time),
            "complete_shadow_path": float(total_time),
            "budget": self.control_dt,
            "overrun": bool(total_time > self.control_dt),
        }
        self._timing_samples.append(timing)
        self._previous_mpc_success = bool(diagnostics.get("success", False))
        return ShadowCycleResult(
            source_sample_id=observation.sample_id,
            logical_time_s=logical_time,
            command=command,
            mpc_success=self._previous_mpc_success,
            predictor_requested=self.predictor_requested,
            predictor_used=self.predictor_requested,
            predictor_fallback_reason="none",
            timing_s=timing,
            diagnostics={
                "solver_status": diagnostics.get("solver_status"),
                "fallback_used": bool(diagnostics.get("fallback_used", False)),
                "current_q_safety_violation": float(
                    diagnostics.get("current_q_safety_violation", np.nan)
                ),
                "predictor": self.predictor.get_last_diagnostics(),
            },
        )

    def _process_full_task(
        self,
        raw_state: RobotStateSnapshot,
        observation: HardwareObservation,
        task_event: TaskClockEvent,
        *,
        started: float,
        state_read_time_s: float,
        conversion_time: float,
    ) -> FullTaskShadowCycleResult:
        """Run one explicit full-task anchor and build an offline proposal."""

        if not self._full_task_core.ready:
            raise HardwareContractError(
                "reset_full_task_epoch must precede the first TaskClockEvent"
            )
        filtered_acc, filtered_alpha = self._filter_disturbance(
            observation.torso_linear_acceleration_world,
            observation.torso_angular_acceleration_world,
        )
        disturbance = self.helper.build_disturbance_input(
            acc_world=filtered_acc,
            omega_world=observation.torso_angular_velocity_world,
            alpha_world=filtered_alpha,
            rot_world_body=observation.torso_rotation_world,
        )
        control_observation = FullTaskControlObservation(
            session_nonce=str(self._full_task_session_nonce),
            source_sample_id=observation.sample_id,
            source_timestamp_ns=observation.monotonic_timestamp_ns,
            validated_timestamp_ns=observation.validated_timestamp_ns,
            state_source="unitree_lowstate_secondary_imu_paired_protocol_v3",
            state_valid=True,
            capabilities=observation.capabilities,
            current_q=observation.right_arm_q,
            current_dq=observation.right_arm_dq,
            measured_disturbance=disturbance,
        )
        model_state = SimpleNamespace(
            qpos=observation.qpos_mujoco,
            qvel=observation.qvel_mujoco,
        )

        def build_helpers(measurement, horizon):
            return self.helper.build_helpers(
                model_state,
                disturbance=measurement,
                disturbance_prediction=horizon.nodes,
                interval_disturbance_prediction=horizon.intervals,
                include_kinematics_cache=False,
            )

        intent = self._full_task_core.step(
            control_observation, task_event, build_helpers
        )
        command_started = time.perf_counter()
        if intent.mpc_output_enabled:
            q_ref = intent.generated_q_ref
            dq_ref = intent.generated_dq_ref
            ddq_des = intent.generated_ddq_des
        else:
            q_ref = self._right_target
            dq_ref = np.zeros(5, dtype=np.float64)
            ddq_des = np.zeros(5, dtype=np.float64)
        command = self.command_builder.build(
            observation, q_ref, dq_ref, ddq_des, raw_state
        )
        active_mask = tuple(5 <= index < 10 for index in range(ARM_SDK_JOINT_COUNT))
        kp = tuple(value if active else 0.0 for value, active in zip(command.kp, active_mask))
        kd = tuple(value if active else 0.0 for value, active in zip(command.kd, active_mask))
        generated_ns = max(
            int(observation.validated_timestamp_ns),
            int(task_event.event_monotonic_timestamp_ns),
        )
        self._full_task_proposal_count += 1
        proposal = HardwareControlProposal(
            session_nonce=intent.session_nonce,
            proposal_id=self._full_task_proposal_count,
            source_sample_id=intent.source_sample_id,
            source_timestamp_ns=intent.source_timestamp_ns,
            task_epoch_id=intent.task_epoch_id,
            task_time_ns=intent.task_time_ns,
            full_task_anchor=intent.full_task_anchor,
            generated_timestamp_ns=generated_ns,
            expires_timestamp_ns=generated_ns + int(round(self.control_dt * 1e9)),
            mode=FutureCommandMode.ROBOT_PD_PLUS_FEEDFORWARD,
            # H3-offline never asks firmware to take arm ownership.
            arm_weight=0.0,
            active_mask=active_mask,
            q_ref=command.q_ref,
            dq_ref=command.dq_ref,
            ddq_des=command.ddq_des,
            kp=kp,
            kd=kd,
            tau=command.tau_ff,
            diagnostics={
                "state_source": control_observation.state_source,
                "mpc_output_enabled": bool(intent.mpc_output_enabled),
                "hardware_torque_state_complete": bool(
                    intent.hardware_torque_state_complete
                ),
                "torque_semantics": command.torque_semantics,
                "mpc_success": bool(
                    intent.controller_diagnostics.get("success", False)
                ),
                "predictor_fallback": bool(
                    intent.predictor_diagnostics.get("fallback_used", False)
                ),
            },
        )
        command_time = time.perf_counter() - command_started
        compute_time = time.perf_counter() - started
        timing = {
            "state_read": float(state_read_time_s),
            "source_state_age": float(observation.state_age_ns * 1e-9),
            "source_to_command_age": float(
                observation.state_age_ns * 1e-9 + compute_time
            ),
            "state_conversion": float(conversion_time),
            "predictor": float(intent.timing.predictor_s),
            "helper": float(intent.timing.helper_s),
            "mpc": float(intent.timing.mpc_s),
            "command_build": float(command_time),
            "complete_shadow_path": float(state_read_time_s) + compute_time,
            "budget": self.control_dt,
            "overrun": bool(float(state_read_time_s) + compute_time > self.control_dt),
        }
        self._timing_samples.append(timing)
        self._previous_mpc_success = bool(
            intent.controller_diagnostics.get("success", False)
        )
        return FullTaskShadowCycleResult(
            source_sample_id=observation.sample_id,
            task_time_s=intent.task_time_ns * 1e-9,
            task_anchor=intent.full_task_anchor,
            intent=intent,
            command=command,
            proposal=proposal,
            timing_s=timing,
            diagnostics={
                "solver_status": intent.controller_diagnostics.get("solver_status"),
                "fallback_used": bool(
                    intent.controller_diagnostics.get("fallback_used", False)
                ),
                "predictor": dict(intent.predictor_diagnostics),
                "write_count": 0,
                "publish_count": 0,
            },
        )

    def summary(self) -> dict:
        samples = self._timing_samples
        values = np.asarray(
            [item["complete_shadow_path"] for item in samples],
            dtype=np.float64,
        )
        timing = {
            "count": int(values.size),
            "mean_ms": float(np.mean(values) * 1e3) if values.size else None,
            "p95_ms": float(np.quantile(values, 0.95) * 1e3)
            if values.size
            else None,
            "p99_ms": float(np.quantile(values, 0.99) * 1e3)
            if values.size
            else None,
            "max_ms": float(np.max(values) * 1e3) if values.size else None,
            "overrun_count": int(
                sum(bool(item["overrun"]) for item in samples)
            ),
        }
        stage_names = (
            "state_read",
            "source_state_age",
            "source_to_command_age",
            "state_conversion",
            "predictor",
            "helper",
            "mpc",
            "command_build",
        )
        timing["stages_ms"] = {
            name: {
                "mean": float(np.mean([item[name] for item in samples]) * 1e3),
                "p99": float(
                    np.quantile([item[name] for item in samples], 0.99) * 1e3
                ),
                "max": float(np.max([item[name] for item in samples]) * 1e3),
            }
            for name in stage_names
        } if samples else {}
        return {
            "mode": "hardware_shadow_read_only",
            "output_capability": "absent",
            "hardware_output_authorized": False,
            "proposal_count": int(self._full_task_proposal_count),
            "command_write_count": 0,
            "publish_count": 0,
            "control_period_ms": self.control_dt * 1e3,
            "predictor_requested": self.predictor_requested,
            "timing": timing,
        }


__all__ = (
    "ARM_SDK_MOTOR_INDICES",
    "G1_23DOF_MOTOR_TO_JOINT",
    "HardwareContractError",
    "HardwareFrameContract",
    "HardwareObservation",
    "HardwareShadowController",
    "FullTaskShadowCycleResult",
    "HardwareStateError",
    "HardwareStateSource",
    "ShadowArmCommand",
    "ShadowCommandBuilder",
    "ShadowCycleResult",
    "load_hardware_shadow_config",
)
