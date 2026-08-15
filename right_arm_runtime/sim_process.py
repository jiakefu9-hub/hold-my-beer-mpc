"""仿真专用独立 C++ 右臂执行进程客户端。

这个模块只负责确定性的 external-step IPC：Python 发布一份完整的
MuJoCo 状态和右臂命令，C++ 完成 RNEA、候选验收和最终执行器计算，
Python 收到同一 ``request_id`` 的结果后才允许推进下一物理步。
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import functools
import json
import mmap
import os
from pathlib import Path
import select
import subprocess
import time
import uuid

import numpy as np

from robot_model_backend.cpp_rnea_backend import _NativeOutput as _RneaOutput
from .cpp_ddq_mapper import _Output as _MapperOutput
from .cpp_executor import (
    ABI_VERSION as _EXECUTOR_ABI_VERSION,
    DEVICE_PD,
    HOST_FULL_TORQUE,
    MODE_NAMES,
    CppExecutorResult,
    _Config as _ExecutorConfig,
    _Output as _ExecutorOutput,
)
from .atomic_seqlock import (
    _ATOMIC_LOAD_8,
    _ATOMIC_STORE_8,
    _MEMORY_ORDER_ACQUIRE,
    _MEMORY_ORDER_RELEASE,
)


PROTOCOL_MAGIC = 0x475253494D525431
PROTOCOL_VERSION = 2
LAYOUT_SIZE = 9984
REQUEST_SLOT_OFFSET = 64
REQUEST_PAYLOAD_OFFSET = 72
RESPONSE_SLOT_OFFSET = 7168
RESPONSE_PAYLOAD_OFFSET = 7176
MAX_NQ = 64
MAX_NV = 64
MAX_NU = 64
MAX_XFRC = 384
ARM_DOF = 5

REQUEST_MAPPING_UPDATE_DUE = 1 << 0
REQUEST_SHUTDOWN = 1 << 1
REQUEST_HAS_PREVIOUS_EXECUTED_TAU = 1 << 2

RESPONSE_MAPPING_UPDATED = 1 << 0
RESPONSE_CACHED_FEEDFORWARD_REUSED = 1 << 1
RESPONSE_EXECUTOR_FALLBACK_ACTIVE = 1 << 2
RESPONSE_FINAL_TORQUE_FINITE = 1 << 3

STATUS_NAMES = {
    0: "ok",
    1: "shutdown",
    2: "invalid_request",
    3: "model_dimension_mismatch",
    4: "executor_config_error",
    5: "rnea_error",
    6: "mapper_error",
    7: "executor_error",
    8: "no_cached_feedforward",
    9: "internal_error",
    10: "NO_SAFE_TORQUE",
}

DEFAULT_WORKER_PATH = Path(
    "/tmp/hold-my-beer-mpc-right-arm-sim-runtime-build/"
    "right_arm_sim_runtime_worker"
)


class SimRuntimeError(RuntimeError):
    """独立仿真执行进程没有安全完成当前请求。"""


class SimRuntimeLayoutError(SimRuntimeError):
    """Python 与 C++ 固定 ABI 不一致。"""


def _poison_process_on_failure(method):
    """任何一次请求失败后立即销毁通道，禁止读取迟到响应。"""

    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        if self._failed:
            raise SimRuntimeError(
                "C++仿真执行进程已因上一次失败而失效，不能复用："
                f"{self._failure_reason}"
            )
        if self._closed:
            raise SimRuntimeError("C++仿真执行进程已经关闭，不能继续使用。")
        try:
            return method(self, *args, **kwargs)
        except BaseException as error:
            # request_id 只有在成功响应后才前移。若超时响应稍后到达，
            # 继续复用同一管道就可能把旧响应误认为新请求，因此失败后
            # 必须直接终止 worker，而不是尝试恢复当前会话。
            self._poison(f"{type(error).__name__}: {error}")
            raise

    return guarded


class _MapperConfig(ctypes.Structure):
    _fields_ = [
        ("perturbation", ctypes.c_double),
        ("regularization", ctypes.c_double),
        ("validation_scales", ctypes.c_double * 8),
        ("validation_scale_count", ctypes.c_int32),
        ("enable_second_pass", ctypes.c_int32),
        ("max_safety_rescue_passes", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("second_pass_error_threshold", ctypes.c_double),
        ("max_joint_error", ctypes.c_double),
        ("max_abs_qacc", ctypes.c_double),
    ]


class _Request(ctypes.Structure):
    _fields_ = [
        ("session_id", ctypes.c_uint64),
        ("request_id", ctypes.c_uint64),
        ("command_id", ctypes.c_uint64),
        ("command_source_state_id", ctypes.c_uint64),
        ("execution_state_id", ctypes.c_uint64),
        ("publish_monotonic_ns", ctypes.c_uint64),
        ("command_timestamp_ns", ctypes.c_uint64),
        ("state_timestamp_ns", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("nq", ctypes.c_uint32),
        ("nv", ctypes.c_uint32),
        ("nu", ctypes.c_uint32),
        ("nbody", ctypes.c_uint32),
        ("simulation_time", ctypes.c_double),
        ("mujoco_timestep", ctypes.c_double),
        ("friction_breakaway_steps", ctypes.c_double),
        ("qpos", ctypes.c_double * MAX_NQ),
        ("qvel", ctypes.c_double * MAX_NV),
        ("reference_qacc", ctypes.c_double * MAX_NV),
        ("fixed_ctrl", ctypes.c_double * MAX_NU),
        ("qacc_warmstart", ctypes.c_double * MAX_NV),
        ("qfrc_applied", ctypes.c_double * MAX_NV),
        ("xfrc_applied", ctypes.c_double * MAX_XFRC),
        ("right_arm_q", ctypes.c_double * ARM_DOF),
        ("right_arm_dq", ctypes.c_double * ARM_DOF),
        ("q_ref", ctypes.c_double * ARM_DOF),
        ("dq_ref", ctypes.c_double * ARM_DOF),
        ("ddq_des", ctypes.c_double * ARM_DOF),
        ("tau_passive", ctypes.c_double * ARM_DOF),
        ("friction_loss", ctypes.c_double * ARM_DOF),
        ("tau_pd", ctypes.c_double * ARM_DOF),
        ("previous_executed_tau", ctypes.c_double * ARM_DOF),
        ("mapper_config", _MapperConfig),
        ("executor_config", _ExecutorConfig),
    ]


class _Response(ctypes.Structure):
    _fields_ = [
        ("session_id", ctypes.c_uint64),
        ("request_id", ctypes.c_uint64),
        ("command_id", ctypes.c_uint64),
        ("command_source_state_id", ctypes.c_uint64),
        ("execution_state_id", ctypes.c_uint64),
        ("request_publish_monotonic_ns", ctypes.c_uint64),
        ("worker_start_monotonic_ns", ctypes.c_uint64),
        ("worker_finish_monotonic_ns", ctypes.c_uint64),
        ("total_elapsed_ns", ctypes.c_uint64),
        ("status", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("rnea_output", _RneaOutput),
        ("mapper_output", _MapperOutput),
        ("executor_output", _ExecutorOutput),
        ("validated_tau_ff", ctypes.c_double * ARM_DOF),
        ("final_tau", ctypes.c_double * ARM_DOF),
        ("error", ctypes.c_char * 512),
    ]


_EXPECTED_LAYOUT = {
    "request_payload_size": 7088,
    "response_payload_size": 2752,
    "request.qpos_offset": 112,
    "request.qvel_offset": 624,
    "request.reference_qacc_offset": 1136,
    "request.fixed_ctrl_offset": 1648,
    "request.xfrc_applied_offset": 3184,
    "request.executor_config_offset": 6736,
    "response.rnea_output_offset": 80,
    "response.mapper_output_offset": 216,
    "response.executor_output_offset": 1712,
    "response.final_tau_offset": 2200,
}


def python_layout_report() -> dict[str, int]:
    """返回可与 worker ``--print-layout`` 逐项比较的 ABI。"""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "layout_size": LAYOUT_SIZE,
        "request_offset": REQUEST_SLOT_OFFSET,
        "request_payload_size": ctypes.sizeof(_Request),
        "response_offset": RESPONSE_SLOT_OFFSET,
        "response_payload_size": ctypes.sizeof(_Response),
        "request.qpos_offset": _Request.qpos.offset,
        "request.qvel_offset": _Request.qvel.offset,
        "request.reference_qacc_offset": _Request.reference_qacc.offset,
        "request.fixed_ctrl_offset": _Request.fixed_ctrl.offset,
        "request.xfrc_applied_offset": _Request.xfrc_applied.offset,
        "request.executor_config_offset": _Request.executor_config.offset,
        "response.rnea_output_offset": _Response.rnea_output.offset,
        "response.mapper_output_offset": _Response.mapper_output.offset,
        "response.executor_output_offset": _Response.executor_output.offset,
        "response.final_tau_offset": _Response.final_tau.offset,
    }


def _validate_python_layout() -> None:
    report = python_layout_report()
    for name, expected in _EXPECTED_LAYOUT.items():
        if report[name] != expected:
            raise SimRuntimeLayoutError(
                f"Python/C++ ABI字段不一致：{name}={report[name]}，"
                f"C++={expected}。"
            )


_validate_python_layout()


def _copy_array(destination, values, count: int, name: str) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.size != count:
        raise ValueError(f"{name} 必须包含 {count} 个数，当前 {array.shape}。")
    flat = np.ascontiguousarray(array.reshape(-1))
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"{name} 包含 NaN 或 Inf。")
    target = np.ctypeslib.as_array(destination)
    target[:count] = flat
    if count < target.size:
        target[count:] = 0.0


def _copy_executor_output(
    output: _ExecutorOutput, *, wall_elapsed_time: float
) -> CppExecutorResult:
    vector = lambda field: np.array(field, dtype=np.float64, copy=True)
    return CppExecutorResult(
        mode=MODE_NAMES.get(int(output.executor_mode), "unknown"),
        output_semantics=(
            "host_full_torque"
            if int(output.output_semantics) == HOST_FULL_TORQUE
            else "device_pd"
        ),
        flags=int(output.flags),
        command_age_ns=int(output.command_age_ns),
        state_age_ns=int(output.state_age_ns),
        effective_q_ref=vector(output.effective_q_ref),
        effective_dq_ref=vector(output.effective_dq_ref),
        predicted_pd_tau=vector(output.predicted_pd_tau),
        predicted_total_tau_raw=vector(output.predicted_total_tau_raw),
        predicted_total_tau_limited=vector(
            output.predicted_total_tau_limited
        ),
        actuator_q_ref=vector(output.actuator_q_ref),
        actuator_dq_ref=vector(output.actuator_dq_ref),
        actuator_kp=vector(output.actuator_kp),
        actuator_kd=vector(output.actuator_kd),
        actuator_tau_ff=vector(output.actuator_tau_ff),
        wall_elapsed_time=float(wall_elapsed_time),
        core_elapsed_time=float(output.core_elapsed_ns) * 1e-9,
    )


@dataclass(frozen=True)
class SimProcessResult:
    session_id: int
    request_id: int
    command_id: int
    command_source_state_id: int
    execution_state_id: int
    mapping_updated: bool
    cached_feedforward_reused: bool
    rnea_output: _RneaOutput
    mapper_output: _MapperOutput
    executor_result: CppExecutorResult
    validated_tau_ff: np.ndarray
    final_tau: np.ndarray
    roundtrip_elapsed_time: float
    worker_elapsed_time: float
    queue_elapsed_time: float


class SimProcessShadowValidator:
    """逐拍证明独立进程与冻结同步执行链产生相同结果。"""

    def __init__(self, absolute_tolerance=1e-9):
        self.absolute_tolerance = float(absolute_tolerance)
        self.sample_count = 0
        self.mapping_sample_count = 0
        self.max_abs_error = {
            "rnea": 0.0,
            "validated_feedforward": 0.0,
            "mapper_tau": 0.0,
            "final_tau": 0.0,
        }

    @staticmethod
    def _max_error(left, right) -> float:
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.shape != right.shape:
            raise SimRuntimeError(
                f"shadow数组维度不一致：{left.shape}/{right.shape}。"
            )
        return float(np.max(np.abs(left - right), initial=0.0))

    def _record(self, name, left, right) -> None:
        error = self._max_error(left, right)
        self.max_abs_error[name] = max(self.max_abs_error[name], error)
        if not np.isfinite(error) or error > self.absolute_tolerance:
            raise SimRuntimeError(
                f"独立进程shadow不一致：{name}最大误差={error:.3e}，"
                f"容差={self.absolute_tolerance:.3e}。"
            )

    def validate(
        self,
        process_result,
        *,
        inverse_result,
        mapping_result,
        pre_executor_tau,
        final_tau,
        tau_pd,
        mapping_update_due,
    ) -> None:
        self.sample_count += 1
        self._record(
            "validated_feedforward",
            process_result.validated_tau_ff,
            np.asarray(pre_executor_tau) - np.asarray(tau_pd),
        )
        self._record("final_tau", process_result.final_tau, final_tau)
        if not mapping_update_due:
            return
        self.mapping_sample_count += 1
        self._record(
            "rnea", process_result.rnea_output.tau_rnea, inverse_result.tau_rnea
        )
        self._record(
            "rnea", process_result.rnea_output.tau_ff, inverse_result.tau_ff
        )
        self._record(
            "mapper_tau", process_result.mapper_output.tau_cmd, pre_executor_tau
        )
        vector_fields = (
            "tau_nominal",
            "tau_correction_raw",
            "tau_correction",
            "tau_cmd_raw",
            "qacc_baseline",
            "qacc_predicted",
            "qacc_validation_error",
            "qacc_validated",
            "first_pass_qacc_validated",
            "second_pass_qacc_validated",
            "hold_last_safe_qacc",
        )
        for name in vector_fields:
            self._record(
                "mapper_tau",
                getattr(process_result.mapper_output, name),
                getattr(mapping_result, name),
            )
        branch_fields = (
            "validation_attempts",
            "validation_improved",
            "validation_tracking_safety_satisfied",
            "validation_qacc_safety_satisfied",
            "second_pass_triggered",
            "second_pass_accepted",
            "second_pass_validation_attempts",
            "safety_fallback_used",
            "safety_fallback_satisfied",
            "safety_fallback_attempts",
            "hold_last_safe_available",
            "hold_last_safe_used",
            "hold_last_safe_satisfied",
            "safe_hold_used",
            "safety_line_search_used",
            "safety_line_search_attempts",
            "final_output_certified",
            "no_safe_torque",
            "full_forward_calls",
            "forward_skip_calls",
            "validated_pass_count",
        )
        mismatches = []
        for name in branch_fields:
            native_value = int(getattr(process_result.mapper_output, name))
            python_value = int(getattr(mapping_result, name))
            if native_value != python_value:
                mismatches.append(f"{name}:{native_value}!={python_value}")
        if mismatches:
            raise SimRuntimeError(
                "独立进程shadow分支不一致：" + ", ".join(mismatches)
            )

    def summary(self) -> dict:
        return {
            "definition": (
                "synchronous C ABI execution versus independent C++ process"
            ),
            "absolute_tolerance": self.absolute_tolerance,
            "sample_count": self.sample_count,
            "mapping_sample_count": self.mapping_sample_count,
            "max_abs_error": dict(self.max_abs_error),
            "passed": bool(
                self.sample_count > 0
                and max(self.max_abs_error.values())
                <= self.absolute_tolerance
            ),
        }

    def save(self, run_dir) -> Path:
        path = Path(run_dir) / "right_arm_process_shadow_parity.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(self.summary(), stream, indent=2, ensure_ascii=False)
        return path


class RightArmSimProcess:
    """【核心桥接】管理一个仿真专用、锁步运行的 C++ 子进程。"""

    def __init__(
        self,
        scene_path,
        *,
        nq: int,
        nv: int,
        nu: int,
        nbody: int,
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
        mapper_perturbation=0.1,
        mapper_regularization=5.0,
        mapper_validation_scales=(1.0, 0.5, 0.25, 0.125),
        mapper_second_pass_error_threshold=5.0,
        mapper_max_joint_error=4.0,
        mapper_max_abs_qacc=10.0,
        mapper_enable_second_pass=True,
        mapper_max_safety_rescue_passes=2,
        worker_path=None,
        response_timeout_s=5.0,
    ):
        self.scene_path = Path(scene_path).expanduser().resolve()
        configured_worker = worker_path or os.environ.get(
            "RIGHT_ARM_SIM_RUNTIME_WORKER", str(DEFAULT_WORKER_PATH)
        )
        self.worker_path = Path(configured_worker).expanduser().resolve()
        if not self.scene_path.is_file():
            raise FileNotFoundError(self.scene_path)
        if not self.worker_path.is_file():
            raise FileNotFoundError(
                "独立C++仿真执行器不存在，请先运行 ./cpp/build_runtime.sh："
                f"{self.worker_path}"
            )
        self.nq, self.nv, self.nu, self.nbody = map(
            int, (nq, nv, nu, nbody)
        )
        if not (
            0 < self.nq <= MAX_NQ
            and 0 < self.nv <= MAX_NV
            and 0 < self.nu <= MAX_NU
            and 0 < 6 * self.nbody <= MAX_XFRC
        ):
            raise ValueError("MuJoCo模型维度超过独立进程协议上限。")
        self.response_timeout_s = float(response_timeout_s)
        if not np.isfinite(self.response_timeout_s) or self.response_timeout_s <= 0:
            raise ValueError("response_timeout_s 必须是有限正数。")

        self._executor_config = self._make_executor_config(
            kp=kp,
            kd=kd,
            timeout_damping=timeout_damping,
            q_ref_min=q_ref_min,
            q_ref_max=q_ref_max,
            dq_ref_abs_max=dq_ref_abs_max,
            tau_min=tau_min,
            tau_max=tau_max,
            command_timeout_ms=command_timeout_ms,
            state_timeout_ms=state_timeout_ms,
            output_semantics=output_semantics,
        )
        self._mapper_config = self._make_mapper_config(
            perturbation=mapper_perturbation,
            regularization=mapper_regularization,
            validation_scales=mapper_validation_scales,
            second_pass_error_threshold=mapper_second_pass_error_threshold,
            max_joint_error=mapper_max_joint_error,
            max_abs_qacc=mapper_max_abs_qacc,
            enable_second_pass=mapper_enable_second_pass,
            max_safety_rescue_passes=mapper_max_safety_rescue_passes,
        )
        self._session_id = uuid.uuid4().int & ((1 << 64) - 1) or 1
        self._next_request_id = 1
        self._mapping = None
        self._request_sequence = None
        self._response_sequence = None
        self._process = None
        self._request_write_fd = None
        self._response_read_fd = None
        self._closed = False
        self._failed = False
        self._failure_reason = ""
        self._shm_name = (
            f"/g1_right_arm_sim_{os.getpid()}_{uuid.uuid4().hex[:12]}"
        )
        self._shm_path = Path("/dev/shm") / self._shm_name[1:]
        try:
            self._start()
        except BaseException as error:
            # 构造函数失败时 __del__ 不可靠；这里同步回收已经启动的
            # 子进程、pipe 和共享内存，保证不会留下孤儿 worker。
            self._poison(f"startup {type(error).__name__}: {error}")
            raise

    @staticmethod
    def _make_executor_config(**values) -> _ExecutorConfig:
        semantics_name = str(values.pop("output_semantics")).strip().lower()
        semantics = {
            "host_full_torque": HOST_FULL_TORQUE,
            "device_pd": DEVICE_PD,
        }.get(semantics_name)
        if semantics is None:
            raise ValueError("output_semantics 必须是 host_full_torque 或 device_pd。")
        config = _ExecutorConfig()
        config.struct_size = ctypes.sizeof(_ExecutorConfig)
        config.abi_version = _EXECUTOR_ABI_VERSION
        config.output_semantics = semantics
        config.command_timeout_ns = int(
            round(float(values.pop("command_timeout_ms")) * 1e6)
        )
        config.state_timeout_ns = int(
            round(float(values.pop("state_timeout_ms")) * 1e6)
        )
        if config.command_timeout_ns <= 0 or config.state_timeout_ns <= 0:
            raise ValueError("执行器超时必须为有限正数。")
        for name, source in values.items():
            _copy_array(getattr(config, name), source, ARM_DOF, name)
        return config

    @staticmethod
    def _make_mapper_config(
        *,
        perturbation,
        regularization,
        validation_scales,
        second_pass_error_threshold,
        max_joint_error,
        max_abs_qacc,
        enable_second_pass,
        max_safety_rescue_passes,
    ) -> _MapperConfig:
        scales = tuple(float(value) for value in validation_scales)
        if not 0 < len(scales) <= 8 or not all(np.isfinite(scales)):
            raise ValueError("validation_scales 必须包含1到8个有限数。")
        config = _MapperConfig()
        config.perturbation = float(perturbation)
        config.regularization = float(regularization)
        config.validation_scale_count = len(scales)
        config.enable_second_pass = int(bool(enable_second_pass))
        config.max_safety_rescue_passes = int(max_safety_rescue_passes)
        config.second_pass_error_threshold = float(
            second_pass_error_threshold
        )
        config.max_joint_error = float(max_joint_error)
        config.max_abs_qacc = float(max_abs_qacc)
        for index, value in enumerate(scales):
            config.validation_scales[index] = value
        return config

    @staticmethod
    def _sequence_address(sequence_object) -> int:
        return ctypes.addressof(sequence_object)

    def _start(self) -> None:
        request_read_fd = None
        response_write_fd = None
        try:
            request_read_fd, self._request_write_fd = os.pipe()
            self._response_read_fd, response_write_fd = os.pipe()
            command = [
                str(self.worker_path),
                "--scene",
                str(self.scene_path),
                "--shm-name",
                self._shm_name,
                "--request-fd",
                str(request_read_fd),
                "--response-fd",
                str(response_write_fd),
                "--reset-shm",
                "--unlink-on-exit",
            ]
            self._process = subprocess.Popen(
                command,
                pass_fds=(request_read_fd, response_write_fd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        finally:
            # 这两端只属于 child；Popen 失败时也必须由 parent 关闭。
            for descriptor in (request_read_fd, response_write_fd):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
        deadline = time.monotonic() + self.response_timeout_s
        while True:
            if self._process.poll() is not None:
                raise SimRuntimeError(self._worker_failure("启动失败"))
            try:
                descriptor = os.open(self._shm_path, os.O_RDWR)
                break
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise SimRuntimeError("等待C++共享内存超时。")
                time.sleep(0.001)
        try:
            # shm_open makes the name visible before ftruncate necessarily
            # publishes the final size.  This race is observable when both
            # processes share one CPU, so wait within the same startup budget.
            observed_size = int(os.fstat(descriptor).st_size)
            while observed_size != LAYOUT_SIZE:
                if self._process.poll() is not None:
                    raise SimRuntimeError(self._worker_failure("启动失败"))
                if time.monotonic() >= deadline:
                    raise SimRuntimeLayoutError(
                        "C++共享内存大小不匹配："
                        f"expected={LAYOUT_SIZE}, observed={observed_size}。"
                    )
                time.sleep(0.001)
                observed_size = int(os.fstat(descriptor).st_size)
            self._mapping = mmap.mmap(
                descriptor,
                LAYOUT_SIZE,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(descriptor)
        # shm_open/ftruncate makes the path and final size visible before the
        # worker has necessarily initialized the header.  With parent and
        # worker intentionally pinned to one CPU, the parent can otherwise see
        # the transient all-zero header and misreport an ABI mismatch.  Wait
        # only for the startup header; the ready pipe below remains the model-
        # load barrier and no control-loop timing is affected.
        observed_header = (0, 0, 0)
        expected_header = (PROTOCOL_MAGIC, PROTOCOL_VERSION, LAYOUT_SIZE)
        while True:
            observed_header = (
                int.from_bytes(self._mapping[0:8], "little"),
                int.from_bytes(self._mapping[8:12], "little"),
                int.from_bytes(self._mapping[12:16], "little"),
            )
            if observed_header == expected_header:
                break
            if self._process.poll() is not None:
                raise SimRuntimeError(self._worker_failure("启动失败"))
            if time.monotonic() >= deadline:
                raise SimRuntimeLayoutError(
                    "C++共享内存magic/version/layout_size不匹配："
                    f"expected={expected_header}, observed={observed_header}。"
                )
            time.sleep(0.001)
        # 共享内存会在模型加载前出现；必须等到Pinocchio/MuJoCo句柄都
        # 构造完成，避免把约0.4 s的一次性启动成本误算进首个控制拍。
        ready, _, _ = select.select(
            [self._process.stdout], [], [], max(0.0, deadline - time.monotonic())
        )
        if not ready:
            raise SimRuntimeError("等待C++ worker ready握手超时。")
        ready_line = self._process.stdout.readline().strip()
        if "external-step ready" not in ready_line:
            raise SimRuntimeError(
                f"C++ worker ready握手无效：{ready_line!r}。"
            )
        self._process.stdout.close()
        self._request_sequence = ctypes.c_uint64.from_buffer(
            self._mapping, REQUEST_SLOT_OFFSET
        )
        self._response_sequence = ctypes.c_uint64.from_buffer(
            self._mapping, RESPONSE_SLOT_OFFSET
        )

    def _worker_failure(self, prefix: str) -> str:
        process = self._process
        status = None if process is None else process.poll()
        stderr = ""
        if process is not None and status is not None and process.stderr:
            stderr = process.stderr.read().strip()
        return f"{prefix}（exit={status}）" + (f"：{stderr}" if stderr else "")

    def scheduling_snapshot(self) -> dict:
        """Read the live worker scheduling state for timing evidence."""

        process = self._process
        if process is None or process.poll() is not None:
            raise SimRuntimeError(self._worker_failure("C++进程不可用"))
        policy = int(os.sched_getscheduler(process.pid))
        policy_names = {
            getattr(os, name): name
            for name in (
                "SCHED_OTHER",
                "SCHED_BATCH",
                "SCHED_IDLE",
                "SCHED_FIFO",
                "SCHED_RR",
            )
            if hasattr(os, name)
        }
        return {
            "pid": int(process.pid),
            "policy": policy,
            "policy_name": policy_names.get(policy, str(policy)),
            "priority": int(
                os.sched_getparam(process.pid).sched_priority
            ),
            "nice": int(os.getpriority(os.PRIO_PROCESS, process.pid)),
            "cpu_affinity": sorted(
                int(cpu) for cpu in os.sched_getaffinity(process.pid)
            ),
        }

    def _write_request(self, request: _Request) -> None:
        sequence_address = self._sequence_address(self._request_sequence)
        sequence = int(
            _ATOMIC_LOAD_8(sequence_address, _MEMORY_ORDER_ACQUIRE)
        )
        if sequence & 1:
            sequence += 1
        _ATOMIC_STORE_8(
            sequence_address, sequence + 1, _MEMORY_ORDER_RELEASE
        )
        ctypes.memmove(
            ctypes.addressof(self._request_sequence)
            + (REQUEST_PAYLOAD_OFFSET - REQUEST_SLOT_OFFSET),
            ctypes.byref(request),
            ctypes.sizeof(request),
        )
        _ATOMIC_STORE_8(
            sequence_address, sequence + 2, _MEMORY_ORDER_RELEASE
        )

    def _read_response(self, max_attempts=100) -> _Response:
        sequence_address = self._sequence_address(self._response_sequence)
        for _ in range(int(max_attempts)):
            before = int(
                _ATOMIC_LOAD_8(sequence_address, _MEMORY_ORDER_ACQUIRE)
            )
            if before & 1:
                continue
            response = _Response.from_buffer_copy(
                self._mapping[
                    RESPONSE_PAYLOAD_OFFSET:
                    RESPONSE_PAYLOAD_OFFSET + ctypes.sizeof(_Response)
                ]
            )
            after = int(
                _ATOMIC_LOAD_8(sequence_address, _MEMORY_ORDER_ACQUIRE)
            )
            if before == after and not (after & 1):
                return response
        raise SimRuntimeError("未能读取稳定的C++响应快照。")

    @_poison_process_on_failure
    def execute(
        self,
        *,
        simulation_time,
        command_timestamp,
        command_id,
        command_source_state_id,
        execution_state_id,
        mapping_update_due,
        mujoco_timestep,
        friction_breakaway_steps,
        qpos,
        qvel,
        reference_qacc,
        fixed_ctrl,
        qacc_warmstart,
        qfrc_applied,
        xfrc_applied,
        right_arm_q,
        right_arm_dq,
        q_ref,
        dq_ref,
        ddq_des,
        tau_passive,
        friction_loss,
        tau_pd,
        previous_executed_tau=None,
    ) -> SimProcessResult:
        """发布一个虚拟2 ms状态，等待同ID最终力矩；失败时不返回旧力矩。"""

        if self._process is None or self._process.poll() is not None:
            raise SimRuntimeError(self._worker_failure("C++进程不可用"))
        request = _Request()
        request.session_id = self._session_id
        request.request_id = self._next_request_id
        request.command_id = int(command_id)
        request.command_source_state_id = int(command_source_state_id)
        request.execution_state_id = int(execution_state_id)
        request.command_timestamp_ns = int(round(float(command_timestamp) * 1e9))
        request.state_timestamp_ns = int(round(float(simulation_time) * 1e9))
        request.flags = int(bool(mapping_update_due)) * REQUEST_MAPPING_UPDATE_DUE
        if previous_executed_tau is not None:
            request.flags |= REQUEST_HAS_PREVIOUS_EXECUTED_TAU
        request.nq, request.nv = self.nq, self.nv
        request.nu, request.nbody = self.nu, self.nbody
        request.simulation_time = float(simulation_time)
        request.mujoco_timestep = float(mujoco_timestep)
        request.friction_breakaway_steps = float(friction_breakaway_steps)
        _copy_array(request.qpos, qpos, self.nq, "qpos")
        _copy_array(request.qvel, qvel, self.nv, "qvel")
        _copy_array(
            request.reference_qacc,
            reference_qacc,
            self.nv,
            "reference_qacc",
        )
        _copy_array(request.fixed_ctrl, fixed_ctrl, self.nu, "fixed_ctrl")
        _copy_array(
            request.qacc_warmstart,
            qacc_warmstart,
            self.nv,
            "qacc_warmstart",
        )
        _copy_array(
            request.qfrc_applied,
            qfrc_applied,
            self.nv,
            "qfrc_applied",
        )
        _copy_array(
            request.xfrc_applied,
            xfrc_applied,
            6 * self.nbody,
            "xfrc_applied",
        )
        for name, values in (
            ("right_arm_q", right_arm_q),
            ("right_arm_dq", right_arm_dq),
            ("q_ref", q_ref),
            ("dq_ref", dq_ref),
            ("ddq_des", ddq_des),
            ("tau_passive", tau_passive),
            ("friction_loss", friction_loss),
            ("tau_pd", tau_pd),
        ):
            _copy_array(getattr(request, name), values, ARM_DOF, name)
        if previous_executed_tau is not None:
            _copy_array(
                request.previous_executed_tau,
                previous_executed_tau,
                ARM_DOF,
                "previous_executed_tau",
            )
        request.mapper_config = self._mapper_config
        request.executor_config = self._executor_config
        request.publish_monotonic_ns = time.monotonic_ns()

        roundtrip_start = time.perf_counter_ns()
        self._write_request(request)
        try:
            os.write(self._request_write_fd, b"\x01")
        except (BrokenPipeError, OSError) as error:
            raise SimRuntimeError(self._worker_failure("通知C++失败")) from error
        ready, _, _ = select.select(
            [self._response_read_fd], [], [], self.response_timeout_s
        )
        if not ready:
            raise SimRuntimeError("等待C++响应超时，拒绝复用旧力矩。")
        byte = os.read(self._response_read_fd, 1)
        if byte != b"\x01":
            raise SimRuntimeError(self._worker_failure("C++响应管道关闭"))
        response = self._read_response()
        roundtrip = (time.perf_counter_ns() - roundtrip_start) * 1e-9
        expected_ids = (
            self._session_id,
            self._next_request_id,
            int(command_id),
            int(command_source_state_id),
            int(execution_state_id),
        )
        actual_ids = (
            int(response.session_id),
            int(response.request_id),
            int(response.command_id),
            int(response.command_source_state_id),
            int(response.execution_state_id),
        )
        if actual_ids != expected_ids:
            raise SimRuntimeError(
                f"C++响应错帧：期望{expected_ids}，收到{actual_ids}。"
            )
        self._next_request_id += 1
        status = int(response.status)
        if status != 0:
            error_text = bytes(response.error).split(b"\0", 1)[0].decode(
                "utf-8", errors="replace"
            )
            raise SimRuntimeError(
                f"C++执行失败[{STATUS_NAMES.get(status, status)}]：{error_text}"
            )
        if (
            int(response.mapper_output.final_output_certified) != 1
            or int(response.mapper_output.no_safe_torque) != 0
        ):
            raise SimRuntimeError(
                "NO_SAFE_TORQUE: C++响应没有认证的 mapper 最终输出。"
            )
        final_tau = np.array(response.final_tau, dtype=np.float64, copy=True)
        if not (
            int(response.flags) & RESPONSE_FINAL_TORQUE_FINITE
            and np.all(np.isfinite(final_tau))
        ):
            raise SimRuntimeError("C++响应未确认最终力矩有限。")
        worker_elapsed = float(response.total_elapsed_ns) * 1e-9
        queue_elapsed = max(
            0.0,
            (int(response.worker_start_monotonic_ns)
             - int(response.request_publish_monotonic_ns))
            * 1e-9,
        )
        return SimProcessResult(
            session_id=actual_ids[0],
            request_id=actual_ids[1],
            command_id=actual_ids[2],
            command_source_state_id=actual_ids[3],
            execution_state_id=actual_ids[4],
            mapping_updated=bool(
                int(response.flags) & RESPONSE_MAPPING_UPDATED
            ),
            cached_feedforward_reused=bool(
                int(response.flags) & RESPONSE_CACHED_FEEDFORWARD_REUSED
            ),
            rnea_output=response.rnea_output,
            mapper_output=response.mapper_output,
            executor_result=_copy_executor_output(
                response.executor_output,
                wall_elapsed_time=roundtrip,
            ),
            validated_tau_ff=np.array(
                response.validated_tau_ff, dtype=np.float64, copy=True
            ),
            final_tau=final_tau,
            roundtrip_elapsed_time=roundtrip,
            worker_elapsed_time=worker_elapsed,
            queue_elapsed_time=queue_elapsed,
        )

    def _poison(self, reason: str) -> None:
        """把当前IPC会话标记为永久失效，并强制回收所有资源。"""

        if not self._failed:
            self._failure_reason = str(reason)
        self._failed = True
        self._close_resources(graceful=False)

    def _close_resources(self, *, graceful: bool) -> None:
        process = self._process
        if (
            graceful
            and process is not None
            and process.poll() is None
            and self._mapping is not None
            and self._request_write_fd is not None
            and self._response_read_fd is not None
        ):
            try:
                request = _Request()
                request.session_id = self._session_id
                request.request_id = self._next_request_id
                request.flags = REQUEST_SHUTDOWN
                self._write_request(request)
                os.write(self._request_write_fd, b"\x01")
                ready, _, _ = select.select(
                    [self._response_read_fd], [], [], 1.0
                )
                if ready:
                    os.read(self._response_read_fd, 1)
            except (OSError, SimRuntimeError):
                pass
        for descriptor_name in ("_request_write_fd", "_response_read_fd"):
            descriptor = getattr(self, descriptor_name)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                setattr(self, descriptor_name, None)

        if process is not None:
            if process.poll() is None and not graceful:
                try:
                    process.terminate()
                except OSError:
                    pass
            try:
                process.wait(timeout=1.0 if graceful else 0.25)
            except subprocess.TimeoutExpired:
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
        self._process = None
        # from_buffer 对象必须先释放，否则 mmap.close 会报 exported pointer。
        self._request_sequence = None
        self._response_sequence = None
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        try:
            self._shm_path.unlink()
        except FileNotFoundError:
            pass
        self._closed = True

    def close(self) -> None:
        """正常发送 shutdown；若会话已失败则保持强制关闭状态。"""

        if self._closed:
            return
        self._close_resources(graceful=not self._failed)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = (
    "RightArmSimProcess",
    "SimProcessResult",
    "SimProcessShadowValidator",
    "SimRuntimeError",
    "SimRuntimeLayoutError",
    "python_layout_report",
)
