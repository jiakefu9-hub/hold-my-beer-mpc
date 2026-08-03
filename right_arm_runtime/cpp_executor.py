"""C++ 右臂 2 ms 安全执行器的窄 ctypes 适配层。"""

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import time

import numpy as np


JOINT_COUNT = 5
ABI_VERSION = 1
HOST_FULL_TORQUE = 0
DEVICE_PD = 1
DEFAULT_LIBRARY_PATH = Path(
    "/tmp/hold-my-beer-mpc-right-arm-executor-build/"
    "libright_arm_executor.so"
)
MODE_NAMES = {
    0: "active",
    1: "command_timed_out",
    2: "state_timed_out",
    3: "invalid_command",
    4: "invalid_state",
}


class _Config(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("output_semantics", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("command_timeout_ns", ctypes.c_int64),
        ("state_timeout_ns", ctypes.c_int64),
        ("kp", ctypes.c_double * JOINT_COUNT),
        ("kd", ctypes.c_double * JOINT_COUNT),
        ("timeout_damping", ctypes.c_double * JOINT_COUNT),
        ("q_ref_min", ctypes.c_double * JOINT_COUNT),
        ("q_ref_max", ctypes.c_double * JOINT_COUNT),
        ("dq_ref_abs_max", ctypes.c_double * JOINT_COUNT),
        ("tau_min", ctypes.c_double * JOINT_COUNT),
        ("tau_max", ctypes.c_double * JOINT_COUNT),
    ]


class _Input(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("command_timestamp_ns", ctypes.c_int64),
        ("state_timestamp_ns", ctypes.c_int64),
        ("q", ctypes.c_double * JOINT_COUNT),
        ("dq", ctypes.c_double * JOINT_COUNT),
        ("q_ref", ctypes.c_double * JOINT_COUNT),
        ("dq_ref", ctypes.c_double * JOINT_COUNT),
        ("tau_ff", ctypes.c_double * JOINT_COUNT),
    ]


class _Output(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("executor_mode", ctypes.c_uint32),
        ("output_semantics", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("command_age_ns", ctypes.c_int64),
        ("state_age_ns", ctypes.c_int64),
        ("core_elapsed_ns", ctypes.c_uint64),
        ("effective_q_ref", ctypes.c_double * JOINT_COUNT),
        ("effective_dq_ref", ctypes.c_double * JOINT_COUNT),
        ("predicted_pd_tau", ctypes.c_double * JOINT_COUNT),
        ("predicted_total_tau_raw", ctypes.c_double * JOINT_COUNT),
        ("predicted_total_tau_limited", ctypes.c_double * JOINT_COUNT),
        ("actuator_q_ref", ctypes.c_double * JOINT_COUNT),
        ("actuator_dq_ref", ctypes.c_double * JOINT_COUNT),
        ("actuator_kp", ctypes.c_double * JOINT_COUNT),
        ("actuator_kd", ctypes.c_double * JOINT_COUNT),
        ("actuator_tau_ff", ctypes.c_double * JOINT_COUNT),
    ]


@dataclass(frozen=True)
class CppExecutorResult:
    mode: str
    output_semantics: str
    flags: int
    command_age_ns: int
    state_age_ns: int
    effective_q_ref: np.ndarray
    effective_dq_ref: np.ndarray
    predicted_pd_tau: np.ndarray
    predicted_total_tau_raw: np.ndarray
    predicted_total_tau_limited: np.ndarray
    actuator_q_ref: np.ndarray
    actuator_dq_ref: np.ndarray
    actuator_kp: np.ndarray
    actuator_kd: np.ndarray
    actuator_tau_ff: np.ndarray
    wall_elapsed_time: float
    core_elapsed_time: float


class CppRightArmExecutor:
    """【半核心】复用 C++ 执行器对象和固定 ABI 缓冲区。"""

    def __init__(
        self,
        *,
        kp,
        kd,
        timeout_damping,
        q_ref_min,
        q_ref_max,
        dq_ref_abs_max,
        tau_min,
        tau_max,
        command_timeout_ms=30.0,
        state_timeout_ms=10.0,
        output_semantics="host_full_torque",
        library_path=None,
    ):
        configured_path = library_path or os.environ.get(
            "RIGHT_ARM_EXECUTOR_LIBRARY", str(DEFAULT_LIBRARY_PATH)
        )
        self.library_path = Path(configured_path).expanduser().resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(
                "C++ 右臂执行器共享库不存在，请先运行 "
                "./cpp/right_arm_executor/build_and_test.sh："
                f"{self.library_path}"
            )
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_signatures()
        if int(self._library.rae_abi_version()) != ABI_VERSION:
            raise RuntimeError("C++ 右臂执行器 ABI 版本不匹配。")

        semantics = {
            "host_full_torque": HOST_FULL_TORQUE,
            "device_pd": DEVICE_PD,
        }.get(str(output_semantics).strip().lower())
        if semantics is None:
            raise ValueError(
                "output_semantics 必须是 host_full_torque 或 device_pd。"
            )
        config = _Config()
        self._check(
            self._library.rae_get_default_config_v1(
                semantics, ctypes.byref(config)
            ),
            "读取默认配置",
        )
        config.command_timeout_ns = self._milliseconds_to_ns(
            command_timeout_ms, "command_timeout_ms"
        )
        config.state_timeout_ns = self._milliseconds_to_ns(
            state_timeout_ms, "state_timeout_ms"
        )
        for name, values in (
            ("kp", kp),
            ("kd", kd),
            ("timeout_damping", timeout_damping),
            ("q_ref_min", q_ref_min),
            ("q_ref_max", q_ref_max),
            ("dq_ref_abs_max", dq_ref_abs_max),
            ("tau_min", tau_min),
            ("tau_max", tau_max),
        ):
            self._copy_into_c_array(
                getattr(config, name), self._vector(values, name)
            )

        handle = ctypes.c_void_p()
        self._check(
            self._library.rae_create_v1(
                ctypes.byref(config), ctypes.byref(handle)
            ),
            "创建执行器",
        )
        if not handle:
            raise RuntimeError("C++ 右臂执行器返回空 handle。")
        self._handle = handle
        self._semantics_name = str(output_semantics).strip().lower()
        self._input = _Input(
            struct_size=ctypes.sizeof(_Input), abi_version=ABI_VERSION
        )
        self._output = _Output()

    def _configure_signatures(self):
        library = self._library
        library.rae_abi_version.argtypes = []
        library.rae_abi_version.restype = ctypes.c_uint32
        library.rae_get_default_config_v1.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_Config),
        ]
        library.rae_get_default_config_v1.restype = ctypes.c_int32
        library.rae_create_v1.argtypes = [
            ctypes.POINTER(_Config),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.rae_create_v1.restype = ctypes.c_int32
        library.rae_destroy.argtypes = [ctypes.c_void_p]
        library.rae_destroy.restype = None
        library.rae_step_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Input),
            ctypes.c_int64,
            ctypes.POINTER(_Output),
        ]
        library.rae_step_v1.restype = ctypes.c_int32
        library.rae_status_string.argtypes = [ctypes.c_int32]
        library.rae_status_string.restype = ctypes.c_char_p

    @staticmethod
    def _milliseconds_to_ns(value, name):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须是有限正数。")
        return int(round(value * 1e6))

    @staticmethod
    def _vector(values, name):
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (JOINT_COUNT,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} 必须是有限的 5 维数组。")
        return result

    @staticmethod
    def _copy_into_c_array(destination, source):
        np.copyto(np.ctypeslib.as_array(destination), source)

    @staticmethod
    def _copy_from_c_array(source):
        return np.ctypeslib.as_array(source).copy()

    def _check(self, status, operation):
        if status == 0:
            return
        text = self._library.rae_status_string(status)
        detail = (
            text.decode("ascii", errors="replace") if text else str(status)
        )
        raise RuntimeError(f"C++ 右臂执行器{operation}失败：{detail}")

    def step(
        self,
        *,
        now_ns,
        command_timestamp_ns,
        state_timestamp_ns,
        q,
        dq,
        q_ref,
        dq_ref,
        tau_ff,
    ):
        """【核心桥接】在一个 2 ms 控制拍中执行限幅、PD 和超时保护。"""
        wall_start = time.perf_counter_ns()
        self._input.command_timestamp_ns = int(command_timestamp_ns)
        self._input.state_timestamp_ns = int(state_timestamp_ns)
        for name, values in (
            ("q", q),
            ("dq", dq),
            ("q_ref", q_ref),
            ("dq_ref", dq_ref),
            ("tau_ff", tau_ff),
        ):
            self._copy_into_c_array(
                getattr(self._input, name), self._runtime_vector(values, name)
            )
        self._check(
            self._library.rae_step_v1(
                self._handle,
                ctypes.byref(self._input),
                int(now_ns),
                ctypes.byref(self._output),
            ),
            "单步计算",
        )
        # 先完成所有 ABI 输出复制，再结束计时；该墙钟因此包含完整 Python
        # 桥接成本，而 core_elapsed_time 只包含 C++ Step。
        effective_q_ref = self._copy_from_c_array(
            self._output.effective_q_ref
        )
        effective_dq_ref = self._copy_from_c_array(
            self._output.effective_dq_ref
        )
        predicted_pd_tau = self._copy_from_c_array(
            self._output.predicted_pd_tau
        )
        predicted_total_tau_raw = self._copy_from_c_array(
            self._output.predicted_total_tau_raw
        )
        predicted_total_tau_limited = self._copy_from_c_array(
            self._output.predicted_total_tau_limited
        )
        actuator_q_ref = self._copy_from_c_array(
            self._output.actuator_q_ref
        )
        actuator_dq_ref = self._copy_from_c_array(
            self._output.actuator_dq_ref
        )
        actuator_kp = self._copy_from_c_array(self._output.actuator_kp)
        actuator_kd = self._copy_from_c_array(self._output.actuator_kd)
        actuator_tau_ff = self._copy_from_c_array(
            self._output.actuator_tau_ff
        )
        wall_elapsed_time = (time.perf_counter_ns() - wall_start) * 1e-9
        return CppExecutorResult(
            mode=MODE_NAMES.get(int(self._output.executor_mode), "unknown"),
            output_semantics=self._semantics_name,
            flags=int(self._output.flags),
            command_age_ns=int(self._output.command_age_ns),
            state_age_ns=int(self._output.state_age_ns),
            effective_q_ref=effective_q_ref,
            effective_dq_ref=effective_dq_ref,
            predicted_pd_tau=predicted_pd_tau,
            predicted_total_tau_raw=predicted_total_tau_raw,
            predicted_total_tau_limited=predicted_total_tau_limited,
            actuator_q_ref=actuator_q_ref,
            actuator_dq_ref=actuator_dq_ref,
            actuator_kp=actuator_kp,
            actuator_kd=actuator_kd,
            actuator_tau_ff=actuator_tau_ff,
            wall_elapsed_time=wall_elapsed_time,
            core_elapsed_time=float(self._output.core_elapsed_ns) * 1e-9,
        )

    @staticmethod
    def _runtime_vector(values, name):
        """运行期只检查布局；NaN/Inf 必须交给 C++ 安全层拒绝。"""
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (JOINT_COUNT,):
            raise ValueError(f"{name} 必须是 5 维数组。")
        return result

    def close(self):
        if getattr(self, "_handle", None):
            self._library.rae_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
