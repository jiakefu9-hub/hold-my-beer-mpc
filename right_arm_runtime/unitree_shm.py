"""Unitree 右臂 2 ms 适配器的 POSIX 共享内存客户端。

该模块只负责跨进程数据交换，不创建 DDS publisher，也不直接接触机器人。
布局严格对应 ``cpp/unitree_arm_adapter`` 的 protocol v3。
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from enum import IntEnum, IntFlag
import hashlib
import math
import mmap
import os
from pathlib import Path
import time
from typing import Iterable

from .atomic_seqlock import (
    _ATOMIC_LOAD_8,
    _ATOMIC_STORE_8,
    _MEMORY_ORDER_ACQUIRE,
    _MEMORY_ORDER_RELEASE,
)


PROTOCOL_MAGIC = 0x473141524D504331
PROTOCOL_VERSION = 3
MOTOR_COUNT = 35
ARM_SDK_JOINT_COUNT = 13
SHA256_BYTES = 32
DEFAULT_SHARED_MEMORY_NAME = "/g1_arm_mpc"


class CommandMode(IntEnum):
    """两种互斥的力矩/PD 执行语义。"""

    ROBOT_PD_PLUS_FEEDFORWARD = 1
    DIRECT_TORQUE = 2


class CommandFlags(IntFlag):
    REQUEST_OUTPUT = 1 << 0
    REQUEST_ARMING_PD = 1 << 1
    REQUEST_ACTIVE = 1 << 2
    REQUEST_RELEASE = 1 << 3


class RequestedLifecycle(IntEnum):
    ARMING_PD = 1
    ACTIVE = 2
    RELEASE = 3


class AdapterMode(IntEnum):
    STARTUP = 0
    ACTIVE_ROBOT_PD = 1
    ACTIVE_DIRECT_TORQUE = 2
    DRY_RUN = 3
    SAFE_RELEASE_NO_COMMAND = 4
    SAFE_RELEASE_COMMAND_STALE = 5
    SAFE_RELEASE_STATE_STALE = 6
    SAFE_RELEASE_INVALID_COMMAND = 7
    SAFE_RELEASE_INVALID_STATE = 8
    SAFE_RELEASE_DEADLINE = 9
    SAFE_RELEASE_OVERTEMPERATURE = 10


class AdapterStatusFlags(IntFlag):
    OUTPUT_ENABLED = 1 << 0
    DDS_WRITE_PERFORMED = 1 << 1
    COMMAND_SNAPSHOT_VALID = 1 << 2
    STATE_SNAPSHOT_VALID = 1 << 3
    COMMAND_CLAMPED = 1 << 4
    DEADLINE_HEALTHY = 1 << 5
    COMMAND_ACCEPTED_BY_SAFETY = 1 << 6
    RECEIPT_IDENTITY_VALID = 1 << 7
    PRE_SINK_DEADLINE_HEALTHY = 1 << 8
    PRE_SINK_EXPIRY_HEALTHY = 1 << 9
    SINK_WRITE_PERFORMED = 1 << 10


class ReceiptReason(IntEnum):
    NONE = 0
    ACCEPTED_OUTPUT_DISABLED = 1
    DDS_WRITE_PERFORMED = 2
    SAFE_RELEASE_NO_COMMAND = 10
    SAFE_RELEASE_COMMAND_STALE = 11
    SAFE_RELEASE_STATE_STALE = 12
    SAFE_RELEASE_INVALID_COMMAND = 13
    SAFE_RELEASE_INVALID_STATE = 14
    SAFE_RELEASE_DEADLINE = 15
    SAFE_RELEASE_OVERTEMPERATURE = 16
    OUTPUT_ENABLED_BUT_NOT_WRITTEN = 17


class StateIngressFlags(IntFlag):
    LOW_STATE_CRC_VALID = 1 << 0
    PAIRED_INGRESS_VALIDATED = 1 << 1
    TORSO_IMU_PRESENT = 1 << 2
    SYNTHETIC_FIXTURE = 1 << 31


class LayoutMismatchError(RuntimeError):
    """共享内存 ABI 与本模块的 protocol v3 不一致。"""


class SeqlockReadError(RuntimeError):
    """在限定重试次数内没有取得一致快照。"""


_Double13 = ctypes.c_double * ARM_SDK_JOINT_COUNT
_Double35 = ctypes.c_double * MOTOR_COUNT
_TemperaturePair = ctypes.c_int16 * 2
_Sha256 = ctypes.c_uint8 * SHA256_BYTES


class _ArmCommandPayload(ctypes.Structure):
    _fields_ = [
        ("monotonic_timestamp_ns", ctypes.c_uint64),
        ("producer_sequence", ctypes.c_uint64),
        ("command_id", ctypes.c_uint64),
        ("source_sample_id", ctypes.c_uint64),
        ("source_timestamp_ns", ctypes.c_uint64),
        ("task_time_ns", ctypes.c_uint64),
        ("full_task_anchor", ctypes.c_uint64),
        ("expires_timestamp_ns", ctypes.c_uint64),
        ("session_nonce", ctypes.c_uint64),
        ("task_epoch_id", ctypes.c_uint64),
        ("safety_policy_id", ctypes.c_uint64),
        ("mode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("active_mask", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("arm_weight", ctypes.c_double),
        ("safety_policy_sha256", _Sha256),
        ("q_ref", _Double13),
        ("dq_ref", _Double13),
        ("ddq_des", _Double13),
        ("kp", _Double13),
        ("kd", _Double13),
        ("tau", _Double13),
    ]


class _RobotStatePayload(ctypes.Structure):
    _fields_ = [
        ("monotonic_timestamp_ns", ctypes.c_uint64),
        ("validated_timestamp_ns", ctypes.c_uint64),
        ("ingress_session_nonce", ctypes.c_uint64),
        ("low_state_timestamp_ns", ctypes.c_uint64),
        ("torso_imu_timestamp_ns", ctypes.c_uint64),
        ("source_skew_ns", ctypes.c_uint64),
        ("sample_id", ctypes.c_uint64),
        ("robot_tick", ctypes.c_uint32),
        ("ingress_flags", ctypes.c_uint32),
        ("mode_pr", ctypes.c_uint8),
        ("mode_machine", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 2),
        ("q", _Double35),
        ("dq", _Double35),
        ("ddq", _Double35),
        ("tau_est", _Double35),
        ("motor_temperature_c", _TemperaturePair * MOTOR_COUNT),
        ("imu_quaternion_wxyz", ctypes.c_double * 4),
        ("imu_gyroscope", ctypes.c_double * 3),
        ("imu_accelerometer", ctypes.c_double * 3),
        ("imu_rpy", ctypes.c_double * 3),
    ]


class _AdapterStatusPayload(ctypes.Structure):
    _fields_ = [
        ("monotonic_timestamp_ns", ctypes.c_uint64),
        ("loop_count", ctypes.c_uint64),
        ("receipt_id", ctypes.c_uint64),
        ("producer_sequence", ctypes.c_uint64),
        ("command_id", ctypes.c_uint64),
        ("source_sample_id", ctypes.c_uint64),
        ("source_timestamp_ns", ctypes.c_uint64),
        ("observed_state_sample_id", ctypes.c_uint64),
        ("observed_state_timestamp_ns", ctypes.c_uint64),
        ("task_time_ns", ctypes.c_uint64),
        ("full_task_anchor", ctypes.c_uint64),
        ("command_timestamp_ns", ctypes.c_uint64),
        ("expires_timestamp_ns", ctypes.c_uint64),
        ("dds_write_timestamp_ns", ctypes.c_uint64),
        ("sink_write_timestamp_ns", ctypes.c_uint64),
        ("pre_sink_check_timestamp_ns", ctypes.c_uint64),
        ("pre_sink_deadline_ns", ctypes.c_uint64),
        ("session_nonce", ctypes.c_uint64),
        ("task_epoch_id", ctypes.c_uint64),
        ("safety_policy_id", ctypes.c_uint64),
        ("command_age_ns", ctypes.c_uint64),
        ("state_age_ns", ctypes.c_uint64),
        ("wake_lateness_ns", ctypes.c_uint64),
        ("execution_time_ns", ctypes.c_uint64),
        ("deadline_miss_count", ctypes.c_uint64),
        ("command_stale_count", ctypes.c_uint64),
        ("state_stale_count", ctypes.c_uint64),
        ("overtemperature_count", ctypes.c_uint64),
        ("mode", ctypes.c_uint32),
        ("requested_command_mode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("receipt_reason", ctypes.c_uint32),
        ("guard_reason", ctypes.c_uint32),
        ("requested_active_mask", ctypes.c_uint32),
        ("executed_active_mask", ctypes.c_uint32),
        ("requested_arm_weight", ctypes.c_double),
        ("executed_arm_weight", ctypes.c_double),
        ("safety_policy_sha256", _Sha256),
        ("selected_q", _Double13),
        ("selected_dq", _Double13),
        ("selected_ddq_des", _Double13),
        ("selected_kp", _Double13),
        ("selected_kd", _Double13),
        ("selected_tau", _Double13),
    ]


# ctypes 不能可靠地声明 C++ alignas(64)，因此显式写出每个槽的尾部 padding。
# 这些结构只允许映射 protocol v3，不能作为一个可自动兼容未来版本的 ABI。
class _CommandSlot(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("payload", _ArmCommandPayload),
        ("padding", ctypes.c_uint8 * 56),
    ]


class _StateSlot(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("payload", _RobotStatePayload),
        ("padding", ctypes.c_uint8 * 24),
    ]


class _StatusSlot(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("payload", _AdapterStatusPayload),
        ("padding", ctypes.c_uint8 * 24),
    ]


class _SharedMemoryLayout(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint64),
        ("version", ctypes.c_uint32),
        ("layout_size", ctypes.c_uint32),
        ("header_padding", ctypes.c_uint8 * 48),
        ("command", _CommandSlot),
        ("state", _StateSlot),
        ("status", _StatusSlot),
    ]


_EXPECTED_LAYOUT = {
    "protocol_version": PROTOCOL_VERSION,
    "layout_size": 3328,
    "command_offset": 64,
    "command_payload_size": 768,
    "state_offset": 896,
    "state_payload_size": 1440,
    "status_offset": 2368,
    "status_payload_size": 928,
}

_EXPECTED_FIELD_OFFSETS = {
    "command.monotonic_timestamp_ns": 0,
    "command.producer_sequence": 8,
    "command.command_id": 16,
    "command.source_sample_id": 24,
    "command.source_timestamp_ns": 32,
    "command.task_time_ns": 40,
    "command.full_task_anchor": 48,
    "command.expires_timestamp_ns": 56,
    "command.session_nonce": 64,
    "command.task_epoch_id": 72,
    "command.safety_policy_id": 80,
    "command.mode": 88,
    "command.flags": 92,
    "command.active_mask": 96,
    "command.reserved": 100,
    "command.arm_weight": 104,
    "command.safety_policy_sha256": 112,
    "command.q_ref": 144,
    "command.dq_ref": 248,
    "command.ddq_des": 352,
    "command.kp": 456,
    "command.kd": 560,
    "command.tau": 664,
    "state.monotonic_timestamp_ns": 0,
    "state.validated_timestamp_ns": 8,
    "state.ingress_session_nonce": 16,
    "state.low_state_timestamp_ns": 24,
    "state.torso_imu_timestamp_ns": 32,
    "state.source_skew_ns": 40,
    "state.sample_id": 48,
    "state.robot_tick": 56,
    "state.ingress_flags": 60,
    "state.mode_pr": 64,
    "state.mode_machine": 65,
    "state.reserved": 66,
    "state.q": 72,
    "state.dq": 352,
    "state.ddq": 632,
    "state.tau_est": 912,
    "state.motor_temperature_c": 1192,
    "state.imu_quaternion_wxyz": 1336,
    "state.imu_gyroscope": 1368,
    "state.imu_accelerometer": 1392,
    "state.imu_rpy": 1416,
    "status.monotonic_timestamp_ns": 0,
    "status.loop_count": 8,
    "status.receipt_id": 16,
    "status.producer_sequence": 24,
    "status.command_id": 32,
    "status.source_sample_id": 40,
    "status.source_timestamp_ns": 48,
    "status.observed_state_sample_id": 56,
    "status.observed_state_timestamp_ns": 64,
    "status.task_time_ns": 72,
    "status.full_task_anchor": 80,
    "status.command_timestamp_ns": 88,
    "status.expires_timestamp_ns": 96,
    "status.dds_write_timestamp_ns": 104,
    "status.sink_write_timestamp_ns": 112,
    "status.pre_sink_check_timestamp_ns": 120,
    "status.pre_sink_deadline_ns": 128,
    "status.session_nonce": 136,
    "status.task_epoch_id": 144,
    "status.safety_policy_id": 152,
    "status.command_age_ns": 160,
    "status.state_age_ns": 168,
    "status.wake_lateness_ns": 176,
    "status.execution_time_ns": 184,
    "status.deadline_miss_count": 192,
    "status.command_stale_count": 200,
    "status.state_stale_count": 208,
    "status.overtemperature_count": 216,
    "status.mode": 224,
    "status.requested_command_mode": 228,
    "status.flags": 232,
    "status.receipt_reason": 236,
    "status.guard_reason": 240,
    "status.requested_active_mask": 244,
    "status.executed_active_mask": 248,
    "status.requested_arm_weight": 256,
    "status.executed_arm_weight": 264,
    "status.safety_policy_sha256": 272,
    "status.selected_q": 304,
    "status.selected_dq": 408,
    "status.selected_ddq_des": 512,
    "status.selected_kp": 616,
    "status.selected_kd": 720,
    "status.selected_tau": 824,
}


def python_layout_report() -> dict[str, int]:
    """返回与 C++ ``--print-layout`` 同名的 Python ABI 信息。"""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "layout_size": ctypes.sizeof(_SharedMemoryLayout),
        "command_offset": _SharedMemoryLayout.command.offset,
        "command_payload_size": ctypes.sizeof(_ArmCommandPayload),
        "state_offset": _SharedMemoryLayout.state.offset,
        "state_payload_size": ctypes.sizeof(_RobotStatePayload),
        "status_offset": _SharedMemoryLayout.status.offset,
        "status_payload_size": ctypes.sizeof(_AdapterStatusPayload),
    }


def _validate_python_layout() -> None:
    report = python_layout_report()
    if report != _EXPECTED_LAYOUT:
        raise LayoutMismatchError(
            f"Python ctypes 顶层布局错误：{report} != {_EXPECTED_LAYOUT}"
        )
    payloads = {
        "command": _ArmCommandPayload,
        "state": _RobotStatePayload,
        "status": _AdapterStatusPayload,
    }
    actual_offsets = {}
    for name, structure in payloads.items():
        for field_name, _ in structure._fields_:
            actual_offsets[f"{name}.{field_name}"] = getattr(
                structure, field_name
            ).offset
    if actual_offsets != _EXPECTED_FIELD_OFFSETS:
        raise LayoutMismatchError(
            "Python ctypes 字段偏移与 protocol v3 不一致。"
        )


_validate_python_layout()


_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class CommandWriteReceipt:
    command_id: int
    producer_sequence: int
    session_nonce: int
    task_epoch_id: int
    source_sample_id: int
    source_timestamp_ns: int
    task_time_ns: int
    full_task_anchor: int
    expires_timestamp_ns: int
    active_mask_bits: int
    safety_policy_id: int
    safety_policy_sha256: str
    monotonic_timestamp_ns: int
    published_sequence: int
    request_output: bool


@dataclass(frozen=True)
class ProtocolV3CommandIdentity:
    """完整command/source/task/policy绑定；字符串使用稳定SHA256映射。

    ``hash()`` 受Python进程随机种子影响，绝不能用于跨进程ABI identity。
    ``stable_identity_u64`` 定义为UTF-8字符串SHA256的前8字节大端整数。
    """

    session_nonce: str | int
    producer_sequence: int
    command_id: int
    source_sample_id: int
    source_timestamp_ns: int
    task_epoch_id: str | int
    task_time_ns: int
    full_task_anchor: int
    expires_timestamp_ns: int
    active_mask: tuple[bool, ...]
    safety_policy_id: str | int
    safety_policy_sha256: str | bytes
    requested_lifecycle: RequestedLifecycle


@dataclass(frozen=True)
class CertifiedHilCommandEnvelope:
    """Protocol-only metadata around an offline-certified command.

    ``command`` must be an actual ``CertifiedHardwareCommand`` produced by the
    hardware-output contract.  This wrapper adds ABI policy/sequence fields; it
    cannot authorize real output.
    """

    command: object
    producer_sequence: int
    safety_policy_id: str | int
    safety_policy_sha256: str | bytes
    requested_lifecycle: RequestedLifecycle


@dataclass(frozen=True)
class RobotStateSnapshot:
    monotonic_timestamp_ns: int
    sample_id: int
    robot_tick: int
    mode_pr: int
    mode_machine: int
    q: tuple[float, ...]
    dq: tuple[float, ...]
    ddq: tuple[float, ...]
    tau_est: tuple[float, ...]
    motor_temperature_c: tuple[tuple[int, int], ...]
    imu_quaternion_wxyz: tuple[float, ...]
    imu_gyroscope: tuple[float, ...]
    imu_accelerometer: tuple[float, ...]
    imu_rpy: tuple[float, ...]
    # Defaults preserve source compatibility for old synthetic unit fixtures;
    # hardware/HIL validation never treats zero provenance as valid.
    validated_timestamp_ns: int = 0
    ingress_session_nonce: int = 0
    low_state_timestamp_ns: int = 0
    torso_imu_timestamp_ns: int = 0
    source_skew_ns: int = 0
    ingress_flags: int = 0


@dataclass(frozen=True)
class AdapterExecutionReceipt:
    monotonic_timestamp_ns: int
    loop_count: int
    receipt_id: int
    producer_sequence: int
    command_id: int
    source_sample_id: int
    source_timestamp_ns: int
    observed_state_sample_id: int
    observed_state_timestamp_ns: int
    task_time_ns: int
    full_task_anchor: int
    command_timestamp_ns: int
    expires_timestamp_ns: int
    dds_write_timestamp_ns: int
    sink_write_timestamp_ns: int
    pre_sink_check_timestamp_ns: int
    pre_sink_deadline_ns: int
    session_nonce: int
    task_epoch_id: int
    safety_policy_id: int
    command_age_ns: int
    state_age_ns: int
    wake_lateness_ns: int
    execution_time_ns: int
    deadline_miss_count: int
    command_stale_count: int
    state_stale_count: int
    overtemperature_count: int
    mode: int
    requested_command_mode: int
    flags: int
    receipt_reason: int
    guard_reason: int
    requested_active_mask_bits: int
    executed_active_mask_bits: int
    requested_arm_weight: float
    executed_arm_weight: float
    safety_policy_sha256: str
    selected_q: tuple[float, ...]
    selected_dq: tuple[float, ...]
    selected_ddq_des: tuple[float, ...]
    selected_kp: tuple[float, ...]
    selected_kd: tuple[float, ...]
    selected_tau: tuple[float, ...]

    @property
    def mode_name(self) -> str:
        try:
            return AdapterMode(self.mode).name.lower()
        except ValueError:
            return f"unknown_{self.mode}"

    @property
    def receipt_reason_name(self) -> str:
        try:
            return ReceiptReason(self.receipt_reason).name.lower()
        except ValueError:
            return f"unknown_{self.receipt_reason}"

    @property
    def requested_active_mask(self) -> tuple[bool, ...]:
        return _unpack_active_mask(self.requested_active_mask_bits)

    @property
    def executed_active_mask(self) -> tuple[bool, ...]:
        return _unpack_active_mask(self.executed_active_mask_bits)


# Backward-compatible public name.  In protocol v3 the status slot is a full
# execution receipt, not merely a timing snapshot.
AdapterStatusSnapshot = AdapterExecutionReceipt


def _vector(values: Iterable[float] | None, name: str) -> tuple[float, ...]:
    if values is None:
        return (0.0,) * ARM_SDK_JOINT_COUNT
    result = tuple(float(value) for value in values)
    if len(result) != ARM_SDK_JOINT_COUNT:
        raise ValueError(f"{name} 必须恰好包含 13 个数。")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} 必须全部为有限数。")
    return result


def _uint64(value: int, name: str) -> int:
    result = int(value)
    if result < 0 or result > _UINT64_MASK:
        raise ValueError(f"{name} 必须位于 uint64 范围内。")
    return result


def stable_identity_u64(value: str | int, name: str = "identity") -> int:
    """Return the repository-wide deterministic uint64 identity mapping."""

    if isinstance(value, str):
        if not value:
            raise ValueError(f"{name} 不能为空。")
        result = int.from_bytes(
            hashlib.sha256(value.encode("utf-8")).digest()[:8], "big"
        )
    else:
        result = _uint64(value, name)
    if result == 0:
        raise ValueError(f"{name} 映射后不能为0。")
    return result


def _sha256_bytes(value: str | bytes) -> bytes:
    if isinstance(value, str):
        try:
            result = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError("safety_policy_sha256 必须是64位十六进制。") from exc
    else:
        result = bytes(value)
    if len(result) != SHA256_BYTES:
        raise ValueError("safety_policy_sha256 必须恰好包含32字节。")
    if not any(result):
        raise ValueError("safety_policy_sha256 不能是全零摘要。")
    return result


def _pack_active_mask(values: Iterable[bool]) -> int:
    mask = tuple(bool(value) for value in values)
    if len(mask) != ARM_SDK_JOINT_COUNT:
        raise ValueError("active_mask 必须恰好包含13个布尔值。")
    result = 0
    for index, active in enumerate(mask):
        if active:
            result |= 1 << index
    return result


def _unpack_active_mask(bits: int) -> tuple[bool, ...]:
    bits = int(bits)
    return tuple(bool(bits & (1 << index)) for index in range(13))


def _validate_protocol_v3_identity(
    identity: ProtocolV3CommandIdentity,
    *,
    generated_timestamp_ns: int,
) -> tuple[int, int, int, int, bytes, RequestedLifecycle]:
    session_nonce = stable_identity_u64(identity.session_nonce, "session_nonce")
    task_epoch_id = stable_identity_u64(identity.task_epoch_id, "task_epoch_id")
    policy_id = stable_identity_u64(identity.safety_policy_id, "safety_policy_id")
    command_id = _uint64(identity.command_id, "command_id")
    source_sample_id = _uint64(identity.source_sample_id, "source_sample_id")
    source_timestamp_ns = _uint64(
        identity.source_timestamp_ns, "source_timestamp_ns"
    )
    producer_sequence = _uint64(
        identity.producer_sequence, "producer_sequence"
    )
    task_time_ns = _uint64(identity.task_time_ns, "task_time_ns")
    anchor = _uint64(identity.full_task_anchor, "full_task_anchor")
    expiry = _uint64(identity.expires_timestamp_ns, "expires_timestamp_ns")
    if command_id == 0 or source_sample_id == 0 or source_timestamp_ns == 0:
        raise ValueError("command/source identity 必须为非零。")
    if source_timestamp_ns > generated_timestamp_ns:
        raise ValueError("source_timestamp_ns 不能晚于command生成时间。")
    if anchor > _UINT64_MASK // 6_000_000 or task_time_ns != anchor * 6_000_000:
        raise ValueError("task_time_ns 必须是full_task_anchor的精确6 ms时间。")
    if expiry <= generated_timestamp_ns:
        raise ValueError("expires_timestamp_ns 必须晚于command生成时间。")
    mask_bits = _pack_active_mask(identity.active_mask)
    try:
        lifecycle = RequestedLifecycle(identity.requested_lifecycle)
    except ValueError as exc:
        raise ValueError("requested_lifecycle 无效。") from exc
    if lifecycle is RequestedLifecycle.RELEASE:
        if mask_bits != 0:
            raise ValueError("release command 的active_mask必须为空。")
    elif mask_bits == 0:
        raise ValueError("arming/active command 至少需要一个active slot。")
    digest = _sha256_bytes(identity.safety_policy_sha256)
    # producer_sequence在task anchor 0时允许为0；读取变量避免误删验证。
    _ = producer_sequence
    return session_nonce, task_epoch_id, policy_id, mask_bits, digest, lifecycle


def _shared_memory_path(name: str) -> Path:
    if len(name) < 2 or not name.startswith("/") or "/" in name[1:]:
        raise ValueError("POSIX 共享内存名必须形如 /g1_arm_mpc。")
    return Path("/dev/shm") / name[1:]


class UnitreeArmSharedMemoryClient:
    """Python MPC 与 C++ 2 ms 适配器之间的 protocol v3 客户端。"""

    def __init__(
        self,
        name: str = DEFAULT_SHARED_MEMORY_NAME,
        *,
        wait_timeout_s: float = 0.0,
        read_only: bool = False,
    ):
        self.name = name
        self.path = _shared_memory_path(name)
        self.read_only = bool(read_only)
        self._mapping: mmap.mmap | None = None
        self._layout: _SharedMemoryLayout | None = None
        self._next_command_id = 1
        self._open(float(wait_timeout_s))

    def _open(self, wait_timeout_s: float) -> None:
        if not math.isfinite(wait_timeout_s) or wait_timeout_s < 0.0:
            raise ValueError("wait_timeout_s 必须是有限非负数。")
        deadline = time.monotonic() + wait_timeout_s
        while True:
            try:
                descriptor = os.open(
                    self.path, os.O_RDONLY if self.read_only else os.O_RDWR
                )
                break
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))

        try:
            size = os.fstat(descriptor).st_size
            if size != _EXPECTED_LAYOUT["layout_size"]:
                raise LayoutMismatchError(
                    f"共享内存大小为 {size}，protocol v3 应为 "
                    f"{_EXPECTED_LAYOUT['layout_size']}。"
                )
            mapping = mmap.mmap(
                descriptor,
                _EXPECTED_LAYOUT["layout_size"],
                # ctypes.from_buffer needs a writable local buffer.  A
                # read-only client therefore uses a private COW mapping:
                # accidental Python writes cannot reach the shared object.
                flags=mmap.MAP_PRIVATE if self.read_only else mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(descriptor)

        layout = _SharedMemoryLayout.from_buffer(mapping)
        try:
            if (
                layout.magic != PROTOCOL_MAGIC
                or layout.version != PROTOCOL_VERSION
                or layout.layout_size != _EXPECTED_LAYOUT["layout_size"]
            ):
                raise LayoutMismatchError(
                    "共享内存 magic/version/layout_size 与 protocol v3 不一致："
                    f"magic=0x{layout.magic:x}, version={layout.version}, "
                    f"layout_size={layout.layout_size}。"
                )
            # 【半核心】mmap 本身页对齐；仍显式检查三个 alignas(64) 槽。
            base_address = ctypes.addressof(layout)
            for slot_name in ("command", "state", "status"):
                slot_address = base_address + getattr(
                    _SharedMemoryLayout, slot_name
                ).offset
                if slot_address % 64 != 0:
                    raise LayoutMismatchError(
                        f"{slot_name} 槽没有按 64 字节对齐。"
                    )
        except Exception:
            del layout
            mapping.close()
            raise
        self._mapping = mapping
        self._layout = layout

    def close(self) -> None:
        if self._mapping is None:
            return
        self._layout = None
        self._mapping.close()
        self._mapping = None

    def __enter__(self) -> "UnitreeArmSharedMemoryClient":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def _require_layout(self) -> _SharedMemoryLayout:
        if self._layout is None:
            raise RuntimeError("共享内存客户端已经关闭。")
        return self._layout

    @staticmethod
    def _sequence_address(slot) -> int:
        return ctypes.addressof(slot) + type(slot).sequence.offset

    @staticmethod
    def _payload_address(slot) -> int:
        return ctypes.addressof(slot) + type(slot).payload.offset

    @classmethod
    def _read_slot(cls, slot, payload_type, max_attempts: int):
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须为正整数。")
        sequence_address = cls._sequence_address(slot)
        payload_address = cls._payload_address(slot)
        for _ in range(int(max_attempts)):
            before = int(
                _ATOMIC_LOAD_8(sequence_address, _MEMORY_ORDER_ACQUIRE)
            )
            if before & 1:
                continue
            snapshot = payload_type.from_buffer_copy(
                ctypes.string_at(payload_address, ctypes.sizeof(payload_type))
            )
            after = int(
                _ATOMIC_LOAD_8(sequence_address, _MEMORY_ORDER_ACQUIRE)
            )
            if before == after and not (after & 1):
                return snapshot
        raise SeqlockReadError(
            f"连续 {max_attempts} 次未能取得稳定的共享内存快照。"
        )

    @classmethod
    def _write_slot(cls, slot, payload) -> int:
        sequence_address = cls._sequence_address(slot)
        sequence = int(
            _ATOMIC_LOAD_8(sequence_address, _MEMORY_ORDER_ACQUIRE)
        )
        if sequence & 1:
            sequence = (sequence + 1) & _UINT64_MASK
        writing_sequence = (sequence + 1) & _UINT64_MASK
        published_sequence = (sequence + 2) & _UINT64_MASK
        # 【核心代码】奇数封住 C++ 读者；payload 完整复制后才发布偶数版本。
        _ATOMIC_STORE_8(
            sequence_address, writing_sequence, _MEMORY_ORDER_RELEASE
        )
        ctypes.memmove(
            cls._payload_address(slot),
            ctypes.byref(payload),
            ctypes.sizeof(payload),
        )
        _ATOMIC_STORE_8(
            sequence_address, published_sequence, _MEMORY_ORDER_RELEASE
        )
        return published_sequence

    def read_state(self, *, max_attempts: int = 100) -> RobotStateSnapshot:
        """【核心】读取 C++/DDS 发布的同一版本完整状态快照。"""

        payload = self._read_slot(
            self._require_layout().state,
            _RobotStatePayload,
            max_attempts,
        )
        return RobotStateSnapshot(
            monotonic_timestamp_ns=int(payload.monotonic_timestamp_ns),
            validated_timestamp_ns=int(payload.validated_timestamp_ns),
            ingress_session_nonce=int(payload.ingress_session_nonce),
            low_state_timestamp_ns=int(payload.low_state_timestamp_ns),
            torso_imu_timestamp_ns=int(payload.torso_imu_timestamp_ns),
            source_skew_ns=int(payload.source_skew_ns),
            sample_id=int(payload.sample_id),
            robot_tick=int(payload.robot_tick),
            ingress_flags=int(payload.ingress_flags),
            mode_pr=int(payload.mode_pr),
            mode_machine=int(payload.mode_machine),
            q=tuple(payload.q),
            dq=tuple(payload.dq),
            ddq=tuple(payload.ddq),
            tau_est=tuple(payload.tau_est),
            motor_temperature_c=tuple(
                (int(pair[0]), int(pair[1]))
                for pair in payload.motor_temperature_c
            ),
            imu_quaternion_wxyz=tuple(payload.imu_quaternion_wxyz),
            imu_gyroscope=tuple(payload.imu_gyroscope),
            imu_accelerometer=tuple(payload.imu_accelerometer),
            imu_rpy=tuple(payload.imu_rpy),
        )

    def read_status(self, *, max_attempts: int = 100) -> AdapterExecutionReceipt:
        """读取C++完整receipt、循环耗时、数据年龄和安全模式。"""

        payload = self._read_slot(
            self._require_layout().status,
            _AdapterStatusPayload,
            max_attempts,
        )
        return AdapterExecutionReceipt(
            monotonic_timestamp_ns=int(payload.monotonic_timestamp_ns),
            loop_count=int(payload.loop_count),
            receipt_id=int(payload.receipt_id),
            producer_sequence=int(payload.producer_sequence),
            command_id=int(payload.command_id),
            source_sample_id=int(payload.source_sample_id),
            source_timestamp_ns=int(payload.source_timestamp_ns),
            observed_state_sample_id=int(payload.observed_state_sample_id),
            observed_state_timestamp_ns=int(
                payload.observed_state_timestamp_ns
            ),
            task_time_ns=int(payload.task_time_ns),
            full_task_anchor=int(payload.full_task_anchor),
            command_timestamp_ns=int(payload.command_timestamp_ns),
            expires_timestamp_ns=int(payload.expires_timestamp_ns),
            dds_write_timestamp_ns=int(payload.dds_write_timestamp_ns),
            sink_write_timestamp_ns=int(payload.sink_write_timestamp_ns),
            pre_sink_check_timestamp_ns=int(
                payload.pre_sink_check_timestamp_ns
            ),
            pre_sink_deadline_ns=int(payload.pre_sink_deadline_ns),
            session_nonce=int(payload.session_nonce),
            task_epoch_id=int(payload.task_epoch_id),
            safety_policy_id=int(payload.safety_policy_id),
            command_age_ns=int(payload.command_age_ns),
            state_age_ns=int(payload.state_age_ns),
            wake_lateness_ns=int(payload.wake_lateness_ns),
            execution_time_ns=int(payload.execution_time_ns),
            deadline_miss_count=int(payload.deadline_miss_count),
            command_stale_count=int(payload.command_stale_count),
            state_stale_count=int(payload.state_stale_count),
            overtemperature_count=int(payload.overtemperature_count),
            mode=int(payload.mode),
            requested_command_mode=int(payload.requested_command_mode),
            flags=int(payload.flags),
            receipt_reason=int(payload.receipt_reason),
            guard_reason=int(payload.guard_reason),
            requested_active_mask_bits=int(payload.requested_active_mask),
            executed_active_mask_bits=int(payload.executed_active_mask),
            requested_arm_weight=float(payload.requested_arm_weight),
            executed_arm_weight=float(payload.executed_arm_weight),
            safety_policy_sha256=bytes(
                payload.safety_policy_sha256
            ).hex(),
            selected_q=tuple(payload.selected_q),
            selected_dq=tuple(payload.selected_dq),
            selected_ddq_des=tuple(payload.selected_ddq_des),
            selected_kp=tuple(payload.selected_kp),
            selected_kd=tuple(payload.selected_kd),
            selected_tau=tuple(payload.selected_tau),
        )

    read_receipt = read_status

    def _write_command(
        self,
        *,
        mode: CommandMode,
        arm_weight: float,
        q_ref: Iterable[float] | None,
        dq_ref: Iterable[float] | None,
        ddq_des: Iterable[float] | None,
        kp: Iterable[float] | None,
        kd: Iterable[float] | None,
        tau: Iterable[float] | None,
        command_id: int | None,
        request_output: bool,
        identity: ProtocolV3CommandIdentity | None = None,
    ) -> CommandWriteReceipt:
        if self.read_only:
            raise PermissionError(
                "read-only Unitree shared-memory client cannot write commands"
            )
        arm_weight = float(arm_weight)
        if not math.isfinite(arm_weight):
            raise ValueError("arm_weight 必须是有限数。")
        timestamp_ns = time.monotonic_ns()
        session_nonce = 0
        producer_sequence = 0
        source_sample_id = 0
        source_timestamp_ns = 0
        task_epoch_id = 0
        task_time_ns = 0
        full_task_anchor = 0
        expires_timestamp_ns = 0
        active_mask_bits = 0
        safety_policy_id = 0
        safety_digest = bytes(SHA256_BYTES)
        lifecycle_flag = 0
        if identity is None:
            if request_output:
                raise PermissionError(
                    "普通write_* API禁止REQUEST_OUTPUT；必须经过未来现场授权路径"
                )
            if command_id is None:
                command_id = self._next_command_id
                self._next_command_id += 1
            command_id = _uint64(command_id, "command_id")
        else:
            if command_id is not None and int(command_id) != identity.command_id:
                raise ValueError("command_id 与protocol-v3 identity不一致。")
            command_id = _uint64(identity.command_id, "command_id")
            (
                session_nonce,
                task_epoch_id,
                safety_policy_id,
                active_mask_bits,
                safety_digest,
                lifecycle,
            ) = _validate_protocol_v3_identity(
                identity, generated_timestamp_ns=timestamp_ns
            )
            producer_sequence = _uint64(
                identity.producer_sequence, "producer_sequence"
            )
            source_sample_id = _uint64(
                identity.source_sample_id, "source_sample_id"
            )
            source_timestamp_ns = _uint64(
                identity.source_timestamp_ns, "source_timestamp_ns"
            )
            task_time_ns = _uint64(identity.task_time_ns, "task_time_ns")
            full_task_anchor = _uint64(
                identity.full_task_anchor, "full_task_anchor"
            )
            expires_timestamp_ns = _uint64(
                identity.expires_timestamp_ns, "expires_timestamp_ns"
            )
            lifecycle_flag = {
                RequestedLifecycle.ARMING_PD:
                    int(CommandFlags.REQUEST_ARMING_PD),
                RequestedLifecycle.ACTIVE:
                    int(CommandFlags.REQUEST_ACTIVE),
                RequestedLifecycle.RELEASE:
                    int(CommandFlags.REQUEST_RELEASE),
            }[lifecycle]
            # Stage 2 deliberately has no authorized real-output Python API.
            # A later site-gated implementation must add a separate capability,
            # not flip this transport helper's default.
            if request_output:
                raise PermissionError(
                    "protocol-v3 offline writer禁止REQUEST_OUTPUT；真实hardware output尚未授权"
                )

        payload = _ArmCommandPayload()
        payload.monotonic_timestamp_ns = timestamp_ns
        payload.producer_sequence = producer_sequence
        payload.command_id = command_id
        payload.source_sample_id = source_sample_id
        payload.source_timestamp_ns = source_timestamp_ns
        payload.task_time_ns = task_time_ns
        payload.full_task_anchor = full_task_anchor
        payload.expires_timestamp_ns = expires_timestamp_ns
        payload.session_nonce = session_nonce
        payload.task_epoch_id = task_epoch_id
        payload.safety_policy_id = safety_policy_id
        payload.mode = int(mode)
        # 默认不请求输出；只有调用者每拍显式传 True 才置位。
        payload.flags = lifecycle_flag | (
            int(CommandFlags.REQUEST_OUTPUT) if request_output else 0
        )
        payload.active_mask = active_mask_bits
        payload.arm_weight = arm_weight
        for index, value in enumerate(safety_digest):
            payload.safety_policy_sha256[index] = value
        for field_name, values in (
            ("q_ref", _vector(q_ref, "q_ref")),
            ("dq_ref", _vector(dq_ref, "dq_ref")),
            ("ddq_des", _vector(ddq_des, "ddq_des")),
            ("kp", _vector(kp, "kp")),
            ("kd", _vector(kd, "kd")),
            ("tau", _vector(tau, "tau")),
        ):
            destination = getattr(payload, field_name)
            for index, value in enumerate(values):
                destination[index] = value

        sequence = self._write_slot(
            self._require_layout().command, payload
        )
        return CommandWriteReceipt(
            command_id=command_id,
            producer_sequence=producer_sequence,
            session_nonce=session_nonce,
            task_epoch_id=task_epoch_id,
            source_sample_id=source_sample_id,
            source_timestamp_ns=source_timestamp_ns,
            task_time_ns=task_time_ns,
            full_task_anchor=full_task_anchor,
            expires_timestamp_ns=expires_timestamp_ns,
            active_mask_bits=active_mask_bits,
            safety_policy_id=safety_policy_id,
            safety_policy_sha256=safety_digest.hex(),
            monotonic_timestamp_ns=timestamp_ns,
            published_sequence=sequence,
            request_output=bool(request_output),
        )

    def _write_bound_protocol_v3_command(
        self,
        *,
        identity: ProtocolV3CommandIdentity,
        mode: CommandMode,
        arm_weight: float,
        q_ref: Iterable[float],
        dq_ref: Iterable[float],
        ddq_des: Iterable[float],
        kp: Iterable[float],
        kd: Iterable[float],
        tau: Iterable[float],
    ) -> CommandWriteReceipt:
        """Write a fully bound command for offline/HIL transport only.

        This stage structurally leaves ``REQUEST_OUTPUT`` clear.  The method
        validates all protocol-v3 identities but does not claim site hardware
        safety or authorize a DDS publisher.
        """

        try:
            mode = CommandMode(mode)
        except ValueError as exc:
            raise ValueError("mode不是受支持的protocol-v3 command mode。") from exc
        return self._write_command(
            mode=mode,
            arm_weight=arm_weight,
            q_ref=q_ref,
            dq_ref=dq_ref,
            ddq_des=ddq_des,
            kp=kp,
            kd=kd,
            tau=tau,
            command_id=identity.command_id,
            request_output=False,
            identity=identity,
        )

    def write_certified_hil_command(
        self,
        envelope: CertifiedHilCommandEnvelope,
    ) -> CommandWriteReceipt:
        """Transport one contract-certified command to publisher-absent HIL.

        The only public fully-bound writer accepts the offline contract type,
        verifies its immutable no-output scope, and still leaves
        ``REQUEST_OUTPUT`` clear.
        """

        from .hardware_output_contract import (
            CertifiedHardwareCommand,
            FutureCommandMode,
        )

        if not isinstance(envelope, CertifiedHilCommandEnvelope):
            raise TypeError("envelope必须是CertifiedHilCommandEnvelope。")
        command = envelope.command
        if not isinstance(command, CertifiedHardwareCommand):
            raise TypeError("envelope.command不是CertifiedHardwareCommand。")
        if command.certification_scope != "offline_transport_contract_only":
            raise PermissionError("不支持的command certification scope。")
        if command.hardware_safety_certified:
            raise PermissionError("离线HIL不能接受真机安全认证声明。")
        if command.hardware_output_authorized:
            raise PermissionError("离线HIL不能接受真实输出授权。")

        mode = {
            FutureCommandMode.ROBOT_PD_PLUS_FEEDFORWARD:
                CommandMode.ROBOT_PD_PLUS_FEEDFORWARD,
            FutureCommandMode.DIRECT_TORQUE:
                CommandMode.DIRECT_TORQUE,
        }.get(command.mode)
        if mode is None:
            raise ValueError("CertifiedHardwareCommand mode不受支持。")
        identity = ProtocolV3CommandIdentity(
            session_nonce=command.session_nonce,
            producer_sequence=envelope.producer_sequence,
            command_id=command.command_id,
            source_sample_id=command.source_sample_id,
            source_timestamp_ns=command.source_timestamp_ns,
            task_epoch_id=command.task_epoch_id,
            task_time_ns=command.task_time_ns,
            full_task_anchor=command.full_task_anchor,
            expires_timestamp_ns=command.expires_timestamp_ns,
            active_mask=command.active_mask,
            safety_policy_id=envelope.safety_policy_id,
            safety_policy_sha256=envelope.safety_policy_sha256,
            requested_lifecycle=envelope.requested_lifecycle,
        )
        return self._write_bound_protocol_v3_command(
            identity=identity,
            mode=mode,
            arm_weight=command.arm_weight,
            q_ref=command.q_ref,
            dq_ref=command.dq_ref,
            ddq_des=command.ddq_des,
            kp=command.kp,
            kd=command.kd,
            tau=command.tau,
        )

    def write_robot_pd_plus_feedforward(
        self,
        *,
        arm_weight: float,
        q_ref: Iterable[float],
        dq_ref: Iterable[float],
        kp: Iterable[float],
        kd: Iterable[float],
        tau_ff: Iterable[float],
        ddq_des: Iterable[float] | None = None,
        command_id: int | None = None,
        request_output: bool = False,
    ) -> CommandWriteReceipt:
        """写入底层 PD + 纯前馈力矩命令。

        ``tau_ff`` 不得包含 PD；否则机器人底层会把同一 PD 计算两遍。
        """

        return self._write_command(
            mode=CommandMode.ROBOT_PD_PLUS_FEEDFORWARD,
            arm_weight=arm_weight,
            q_ref=q_ref,
            dq_ref=dq_ref,
            ddq_des=ddq_des,
            kp=kp,
            kd=kd,
            tau=tau_ff,
            command_id=command_id,
            request_output=request_output,
        )

    def write_direct_torque(
        self,
        *,
        arm_weight: float,
        tau_cmd: Iterable[float],
        ddq_des: Iterable[float] | None = None,
        command_id: int | None = None,
        request_output: bool = False,
    ) -> CommandWriteReceipt:
        """写入已经包含反馈项的最终力矩。

        C++ 会从最新状态填 q，并强制发送 kp=kd=0，避免重复 PD。
        """

        return self._write_command(
            mode=CommandMode.DIRECT_TORQUE,
            arm_weight=arm_weight,
            q_ref=None,
            dq_ref=None,
            ddq_des=ddq_des,
            kp=None,
            kd=None,
            tau=tau_cmd,
            command_id=command_id,
            request_output=request_output,
        )


__all__ = (
    "ARM_SDK_JOINT_COUNT",
    "AdapterExecutionReceipt",
    "AdapterMode",
    "AdapterStatusFlags",
    "AdapterStatusSnapshot",
    "CommandMode",
    "CommandWriteReceipt",
    "CertifiedHilCommandEnvelope",
    "DEFAULT_SHARED_MEMORY_NAME",
    "LayoutMismatchError",
    "MOTOR_COUNT",
    "PROTOCOL_MAGIC",
    "PROTOCOL_VERSION",
    "ProtocolV3CommandIdentity",
    "ReceiptReason",
    "RequestedLifecycle",
    "RobotStateSnapshot",
    "SeqlockReadError",
    "UnitreeArmSharedMemoryClient",
    "python_layout_report",
    "stable_identity_u64",
)
