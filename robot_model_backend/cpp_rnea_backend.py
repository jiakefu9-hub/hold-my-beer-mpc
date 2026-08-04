"""C++ Pinocchio RNEA 的窄 C ABI Python 适配层。"""

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import time

import numpy as np

from .base import (
    PredictionKinematics,
    PredictionKinematicsBackend,
    PredictionKinematicsBatch,
)


JOINT_COUNT = 5
KINEMATICS_MAX_STATES = 32
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)
EE_FRAME_NAME = "right_grasp_site"
IMU_FRAME_NAME = "imu_in_torso"
DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)
BYTE_POINTER = ctypes.POINTER(ctypes.c_uint8)
DEFAULT_LIBRARY_PATH = Path(
    "/tmp/hold-my-beer-mpc-right-arm-rnea-build/libright_arm_rnea.so"
)


class _NativeOutput(ctypes.Structure):
    _fields_ = [
        ("tau_rnea", ctypes.c_double * JOINT_COUNT),
        ("tau_constraint_friction", ctypes.c_double * JOINT_COUNT),
        ("tau_ff", ctypes.c_double * JOINT_COUNT),
        ("core_elapsed_ns", ctypes.c_uint64),
        ("rnea_elapsed_ns", ctypes.c_uint64),
    ]


class _NativeKinematicsBatchOutput(ctypes.Structure):
    """必须与 C ABI v2 的 ``RightArmKinematicsBatchOutput`` 完全一致。"""

    _fields_ = [
        ("state_count", ctypes.c_int32),
        (
            "ee_position_world",
            ctypes.c_double * (KINEMATICS_MAX_STATES * 3),
        ),
        (
            "ee_rotation_world",
            ctypes.c_double * (KINEMATICS_MAX_STATES * 9),
        ),
        (
            "imu_position_world",
            ctypes.c_double * (KINEMATICS_MAX_STATES * 3),
        ),
        (
            "imu_rotation_world",
            ctypes.c_double * (KINEMATICS_MAX_STATES * 9),
        ),
        (
            "J_v_world",
            ctypes.c_double
            * (KINEMATICS_MAX_STATES * 3 * JOINT_COUNT),
        ),
        (
            "J_w_world",
            ctypes.c_double
            * (KINEMATICS_MAX_STATES * 3 * JOINT_COUNT),
        ),
        (
            "dJ_v_world",
            ctypes.c_double
            * (KINEMATICS_MAX_STATES * 3 * JOINT_COUNT),
        ),
        (
            "dJ_w_world",
            ctypes.c_double
            * (KINEMATICS_MAX_STATES * 3 * JOINT_COUNT),
        ),
        ("core_elapsed_ns", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class CppRneaResult:
    tau_rnea: np.ndarray
    tau_constraint_friction: np.ndarray
    tau_ff: np.ndarray
    wall_elapsed_time: float
    core_elapsed_time: float
    rnea_elapsed_time: float


class CppRightArmRneaBackend(PredictionKinematicsBackend):
    """【半核心】管理 C++ model/Data 生命周期和 ctypes 边界。"""

    backend_name = "cpp_pinocchio"

    def __init__(self, scene_mjcf_path, library_path=None):
        configured_path = library_path or os.environ.get(
            "RIGHT_ARM_RNEA_LIBRARY", str(DEFAULT_LIBRARY_PATH)
        )
        self.library_path = Path(configured_path).expanduser().resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(
                "C++ RNEA 共享库不存在，请先运行 "
                "./cpp/right_arm_rnea/build_and_test.sh："
                f"{self.library_path}"
            )
        self.scene_mjcf_path = Path(scene_mjcf_path).expanduser().resolve()
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_common_signatures()
        abi_version = int(self._library.right_arm_rnea_abi_version())
        if abi_version != 3:
            raise RuntimeError(
                "C++ RNEA ABI 版本不匹配："
                f"期望 3，当前 {abi_version}。"
            )
        self._configure_v2_signatures()

        self._error = ctypes.create_string_buffer(1024)
        self._handle = self._library.right_arm_rnea_create(
            str(self.scene_mjcf_path).encode("utf-8"),
            self._error,
            len(self._error),
        )
        if not self._handle:
            raise RuntimeError(self._error_text("创建 C++ RNEA 失败"))
        self.nq = int(
            self._library.right_arm_rnea_mujoco_nq(self._handle)
        )
        self.nv = int(
            self._library.right_arm_rnea_mujoco_nv(self._handle)
        )
        self._output = _NativeOutput()
        self._kinematics_output = _NativeKinematicsBatchOutput()

    def _configure_common_signatures(self):
        library = self._library
        library.right_arm_rnea_abi_version.argtypes = []
        library.right_arm_rnea_abi_version.restype = ctypes.c_uint32
        library.right_arm_rnea_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.right_arm_rnea_create.restype = ctypes.c_void_p
        library.right_arm_rnea_destroy.argtypes = [ctypes.c_void_p]
        library.right_arm_rnea_destroy.restype = None
        library.right_arm_rnea_mujoco_nq.argtypes = [ctypes.c_void_p]
        library.right_arm_rnea_mujoco_nq.restype = ctypes.c_size_t
        library.right_arm_rnea_mujoco_nv.argtypes = [ctypes.c_void_p]
        library.right_arm_rnea_mujoco_nv.restype = ctypes.c_size_t
        library.right_arm_rnea_compute.argtypes = [
            ctypes.c_void_p,
            DOUBLE_POINTER,
            ctypes.c_size_t,
            DOUBLE_POINTER,
            ctypes.c_size_t,
            DOUBLE_POINTER,
            ctypes.c_size_t,
            DOUBLE_POINTER,
            ctypes.c_size_t,
            DOUBLE_POINTER,
            ctypes.c_size_t,
            DOUBLE_POINTER,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(_NativeOutput),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.right_arm_rnea_compute.restype = ctypes.c_int
        library.right_arm_rnea_status_string.argtypes = [ctypes.c_int]
        library.right_arm_rnea_status_string.restype = ctypes.c_char_p

    def _configure_v2_signatures(self):
        library = self._library
        library.right_arm_kinematics_batch_compute.argtypes = [
            ctypes.c_void_p,
            DOUBLE_POINTER,
            ctypes.c_size_t,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
            BYTE_POINTER,
            ctypes.c_size_t,
            ctypes.POINTER(_NativeKinematicsBatchOutput),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.right_arm_kinematics_batch_compute.restype = ctypes.c_int

    @staticmethod
    def _array(values, expected_shape, name):
        result = np.ascontiguousarray(values, dtype=np.float64)
        if result.shape != expected_shape:
            raise ValueError(
                f"{name} 维度应为 {expected_shape}，当前 {result.shape}。"
            )
        return result

    @staticmethod
    def _pointer(values):
        return values.ctypes.data_as(DOUBLE_POINTER)

    @staticmethod
    def _byte_pointer(values):
        return values.ctypes.data_as(BYTE_POINTER)

    def _error_text(self, prefix):
        detail = self._error.value.decode("utf-8", errors="replace")
        return f"{prefix}: {detail}" if detail else prefix

    def _raise_status(self, status, prefix):
        status_name = self._library.right_arm_rnea_status_string(status)
        name = (
            status_name.decode("ascii", errors="replace")
            if status_name
            else str(status)
        )
        raise RuntimeError(self._error_text(f"{prefix} {name}"))

    @staticmethod
    def _batch_array(native_values, state_count, trailing_shape):
        element_count = state_count * int(np.prod(trailing_shape))
        return (
            np.ctypeslib.as_array(native_values)[:element_count]
            .copy()
            .reshape((state_count, *trailing_shape))
        )

    def evaluate_batch(
        self,
        qpos_reference,
        q_arm,
        dq_arm,
        *,
        acceleration_required=True,
    ):
        """【核心桥接】一次 C ABI 调用计算整个 MPC 预测窗口。"""

        qpos = self._array(qpos_reference, (self.nq,), "qpos_reference")
        q = np.ascontiguousarray(q_arm, dtype=np.float64)
        dq = np.ascontiguousarray(dq_arm, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != JOINT_COUNT:
            raise ValueError(
                "q_arm 必须为 (state_count, 5)，"
                f"当前 {q.shape}。"
            )
        if dq.shape != q.shape:
            raise ValueError(
                f"dq_arm 必须与 q_arm 同形，当前 {dq.shape}/{q.shape}。"
            )
        state_count = int(q.shape[0])
        if not 1 <= state_count <= KINEMATICS_MAX_STATES:
            raise ValueError(
                "state_count 必须为 1.."
                f"{KINEMATICS_MAX_STATES}，当前 {state_count}。"
            )
        if not all(np.all(np.isfinite(value)) for value in (qpos, q, dq)):
            raise ValueError("预测运动学输入包含 NaN 或 Inf。")

        required = np.asarray(acceleration_required)
        if required.ndim == 0:
            required = np.full(
                state_count, bool(required), dtype=np.uint8
            )
        else:
            if required.shape != (state_count,):
                raise ValueError(
                    "acceleration_required 必须是布尔标量或 "
                    f"({state_count},) 数组，当前 {required.shape}。"
                )
            required = np.ascontiguousarray(required, dtype=np.uint8)

        self._error[0] = 0
        start = time.perf_counter_ns()
        status = self._library.right_arm_kinematics_batch_compute(
            self._handle,
            self._pointer(qpos),
            qpos.size,
            self._pointer(q),
            self._pointer(dq),
            self._byte_pointer(required),
            state_count,
            ctypes.byref(self._kinematics_output),
            self._error,
            len(self._error),
        )
        wall_elapsed = (time.perf_counter_ns() - start) * 1e-9
        if status != 0:
            self._raise_status(status, "C++ 批量运动学")
        if int(self._kinematics_output.state_count) != state_count:
            raise RuntimeError(
                "C++ 批量运动学返回的 state_count 不匹配："
                f"期望 {state_count}，当前 "
                f"{self._kinematics_output.state_count}。"
            )

        output = self._kinematics_output
        jacobian_shape = (3, JOINT_COUNT)
        return PredictionKinematicsBatch(
            ee_position_world=self._batch_array(
                output.ee_position_world, state_count, (3,)
            ),
            ee_rotation_world=self._batch_array(
                output.ee_rotation_world, state_count, (3, 3)
            ),
            imu_position_world=self._batch_array(
                output.imu_position_world, state_count, (3,)
            ),
            imu_rotation_world=self._batch_array(
                output.imu_rotation_world, state_count, (3, 3)
            ),
            J_v_world=self._batch_array(
                output.J_v_world, state_count, jacobian_shape
            ),
            J_w_world=self._batch_array(
                output.J_w_world, state_count, jacobian_shape
            ),
            dJ_v_world=self._batch_array(
                output.dJ_v_world, state_count, jacobian_shape
            ),
            dJ_w_world=self._batch_array(
                output.dJ_w_world, state_count, jacobian_shape
            ),
            wall_elapsed_time=wall_elapsed,
            core_elapsed_time=float(output.core_elapsed_ns) * 1e-9,
        )

    def evaluate(
        self,
        qpos_reference,
        q_arm,
        dq_arm,
        *,
        acceleration_required=True,
    ):
        """保持单节点后端接口兼容，内部仍走 ABI v2 批处理。"""

        batch = self.evaluate_batch(
            qpos_reference,
            np.asarray(q_arm, dtype=np.float64)[None, :],
            np.asarray(dq_arm, dtype=np.float64)[None, :],
            acceleration_required=acceleration_required,
        )
        return PredictionKinematics(
            ee_position_world=batch.ee_position_world[0].copy(),
            ee_rotation_world=batch.ee_rotation_world[0].copy(),
            imu_position_world=batch.imu_position_world[0].copy(),
            imu_rotation_world=batch.imu_rotation_world[0].copy(),
            J_v_world=batch.J_v_world[0].copy(),
            J_w_world=batch.J_w_world[0].copy(),
            dJ_v_world=batch.dJ_v_world[0].copy(),
            dJ_w_world=batch.dJ_w_world[0].copy(),
        )

    def compute_feedforward(
        self,
        qpos_mujoco,
        qvel_mujoco,
        desired_right_arm_ddq,
        tau_passive,
        friction_loss,
        timestep,
        friction_breakaway_steps,
        reference_qacc=None,
    ):
        """【核心桥接】一次返回 RNEA、摩擦项和名义前馈。"""

        qpos = self._array(qpos_mujoco, (self.nq,), "qpos")
        qvel = self._array(qvel_mujoco, (self.nv,), "qvel")
        reference = self._array(
            np.zeros(self.nv, dtype=np.float64)
            if reference_qacc is None
            else reference_qacc,
            (self.nv,),
            "reference_qacc",
        )
        desired = self._array(
            desired_right_arm_ddq, (JOINT_COUNT,), "desired_ddq"
        )
        passive = self._array(
            tau_passive, (JOINT_COUNT,), "tau_passive"
        )
        friction = self._array(
            friction_loss, (JOINT_COUNT,), "friction_loss"
        )
        self._error[0] = 0
        start = time.perf_counter_ns()
        status = self._library.right_arm_rnea_compute(
            self._handle,
            self._pointer(qpos),
            qpos.size,
            self._pointer(qvel),
            qvel.size,
            self._pointer(reference),
            reference.size,
            self._pointer(desired),
            desired.size,
            self._pointer(passive),
            passive.size,
            self._pointer(friction),
            friction.size,
            float(timestep),
            float(friction_breakaway_steps),
            ctypes.byref(self._output),
            self._error,
            len(self._error),
        )
        wall_elapsed = (time.perf_counter_ns() - start) * 1e-9
        if status != 0:
            self._raise_status(status, "C++ RNEA")
        return CppRneaResult(
            tau_rnea=np.ctypeslib.as_array(
                self._output.tau_rnea
            ).copy(),
            tau_constraint_friction=np.ctypeslib.as_array(
                self._output.tau_constraint_friction
            ).copy(),
            tau_ff=np.ctypeslib.as_array(self._output.tau_ff).copy(),
            wall_elapsed_time=wall_elapsed,
            core_elapsed_time=float(self._output.core_elapsed_ns) * 1e-9,
            rnea_elapsed_time=float(self._output.rnea_elapsed_ns) * 1e-9,
        )

    def close(self):
        if getattr(self, "_handle", None):
            self._library.right_arm_rnea_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
