"""C++ Pinocchio RNEA 的窄 C ABI Python 适配层。"""

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import time

import numpy as np


JOINT_COUNT = 5
DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)
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


@dataclass(frozen=True)
class CppRneaResult:
    tau_rnea: np.ndarray
    tau_constraint_friction: np.ndarray
    tau_ff: np.ndarray
    wall_elapsed_time: float
    core_elapsed_time: float
    rnea_elapsed_time: float


class CppRightArmRneaBackend:
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
        self._configure_signatures()
        if int(self._library.right_arm_rnea_abi_version()) != 1:
            raise RuntimeError("C++ RNEA ABI 版本不匹配。")

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

    def _configure_signatures(self):
        library = self._library
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
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(_NativeOutput),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.right_arm_rnea_compute.restype = ctypes.c_int
        library.right_arm_rnea_status_string.argtypes = [ctypes.c_int]
        library.right_arm_rnea_status_string.restype = ctypes.c_char_p
        library.right_arm_rnea_abi_version.argtypes = []
        library.right_arm_rnea_abi_version.restype = ctypes.c_uint32

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

    def _error_text(self, prefix):
        detail = self._error.value.decode("utf-8", errors="replace")
        return f"{prefix}: {detail}" if detail else prefix

    def compute_feedforward(
        self,
        qpos_mujoco,
        qvel_mujoco,
        desired_right_arm_ddq,
        tau_passive,
        friction_loss,
        timestep,
        friction_breakaway_steps,
    ):
        """【核心桥接】一次返回 RNEA、摩擦项和名义前馈。"""

        qpos = self._array(qpos_mujoco, (self.nq,), "qpos")
        qvel = self._array(qvel_mujoco, (self.nv,), "qvel")
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
            status_name = self._library.right_arm_rnea_status_string(status)
            name = (
                status_name.decode("ascii", errors="replace")
                if status_name
                else str(status)
            )
            raise RuntimeError(self._error_text(f"C++ RNEA {name}"))
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
