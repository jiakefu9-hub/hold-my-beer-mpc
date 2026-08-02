"""Unitree 右臂 2 ms 适配器的 POSIX 共享内存客户端。

该模块只负责跨进程数据交换，不创建 DDS publisher，也不直接接触机器人。
布局严格对应 ``cpp/unitree_arm_adapter`` 的 protocol v2。
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import ctypes.util
from enum import IntEnum, IntFlag
import math
import mmap
import os
from pathlib import Path
import time
from typing import Iterable


PROTOCOL_MAGIC = 0x473141524D504331
PROTOCOL_VERSION = 2
MOTOR_COUNT = 35
ARM_SDK_JOINT_COUNT = 13
DEFAULT_SHARED_MEMORY_NAME = "/g1_arm_mpc"


class CommandMode(IntEnum):
    """两种互斥的力矩/PD 执行语义。"""

    ROBOT_PD_PLUS_FEEDFORWARD = 1
    DIRECT_TORQUE = 2


class CommandFlags(IntFlag):
    REQUEST_OUTPUT = 1 << 0


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


class LayoutMismatchError(RuntimeError):
    """共享内存 ABI 与本模块的 protocol v2 不一致。"""


class SeqlockReadError(RuntimeError):
    """在限定重试次数内没有取得一致快照。"""


_Double13 = ctypes.c_double * ARM_SDK_JOINT_COUNT
_Double35 = ctypes.c_double * MOTOR_COUNT
_TemperaturePair = ctypes.c_int16 * 2


class _ArmCommandPayload(ctypes.Structure):
    _fields_ = [
        ("monotonic_timestamp_ns", ctypes.c_uint64),
        ("command_id", ctypes.c_uint64),
        ("mode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("arm_weight", ctypes.c_double),
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
        ("sample_id", ctypes.c_uint64),
        ("robot_tick", ctypes.c_uint32),
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
        ("command_id", ctypes.c_uint64),
        ("command_age_ns", ctypes.c_uint64),
        ("state_age_ns", ctypes.c_uint64),
        ("wake_lateness_ns", ctypes.c_uint64),
        ("execution_time_ns", ctypes.c_uint64),
        ("deadline_miss_count", ctypes.c_uint64),
        ("command_stale_count", ctypes.c_uint64),
        ("state_stale_count", ctypes.c_uint64),
        ("overtemperature_count", ctypes.c_uint64),
        ("mode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


# ctypes 不能可靠地声明 C++ alignas(64)，因此显式写出每个槽的尾部 padding。
# 这些结构只允许映射 protocol v2，不能作为一个可自动兼容未来版本的 ABI。
class _CommandSlot(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("payload", _ArmCommandPayload),
        ("padding", ctypes.c_uint8 * 40),
    ]


class _StateSlot(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("payload", _RobotStatePayload),
        ("padding", ctypes.c_uint8 * 8),
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
    "layout_size": 2304,
    "command_offset": 64,
    "command_payload_size": 656,
    "state_offset": 768,
    "state_payload_size": 1392,
    "status_offset": 2176,
    "status_payload_size": 96,
}

_EXPECTED_FIELD_OFFSETS = {
    "command.monotonic_timestamp_ns": 0,
    "command.command_id": 8,
    "command.mode": 16,
    "command.flags": 20,
    "command.arm_weight": 24,
    "command.q_ref": 32,
    "command.dq_ref": 136,
    "command.ddq_des": 240,
    "command.kp": 344,
    "command.kd": 448,
    "command.tau": 552,
    "state.monotonic_timestamp_ns": 0,
    "state.sample_id": 8,
    "state.robot_tick": 16,
    "state.mode_pr": 20,
    "state.mode_machine": 21,
    "state.reserved": 22,
    "state.q": 24,
    "state.dq": 304,
    "state.ddq": 584,
    "state.tau_est": 864,
    "state.motor_temperature_c": 1144,
    "state.imu_quaternion_wxyz": 1288,
    "state.imu_gyroscope": 1320,
    "state.imu_accelerometer": 1344,
    "state.imu_rpy": 1368,
    "status.monotonic_timestamp_ns": 0,
    "status.loop_count": 8,
    "status.command_id": 16,
    "status.command_age_ns": 24,
    "status.state_age_ns": 32,
    "status.wake_lateness_ns": 40,
    "status.execution_time_ns": 48,
    "status.deadline_miss_count": 56,
    "status.command_stale_count": 64,
    "status.state_stale_count": 72,
    "status.overtemperature_count": 80,
    "status.mode": 88,
    "status.flags": 92,
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
            "Python ctypes 字段偏移与 protocol v2 不一致。"
        )


_validate_python_layout()


def _load_libatomic():
    library_name = ctypes.util.find_library("atomic") or "libatomic.so.1"
    library = ctypes.CDLL(library_name)
    atomic_load = getattr(library, "__atomic_load_8")
    atomic_load.argtypes = [ctypes.c_void_p, ctypes.c_int]
    atomic_load.restype = ctypes.c_uint64
    atomic_store = getattr(library, "__atomic_store_8")
    atomic_store.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    atomic_store.restype = None
    return library, atomic_load, atomic_store


_LIBATOMIC, _ATOMIC_LOAD_8, _ATOMIC_STORE_8 = _load_libatomic()
_MEMORY_ORDER_ACQUIRE = 2
_MEMORY_ORDER_RELEASE = 3
_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class CommandWriteReceipt:
    command_id: int
    monotonic_timestamp_ns: int
    published_sequence: int
    request_output: bool


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


@dataclass(frozen=True)
class AdapterStatusSnapshot:
    monotonic_timestamp_ns: int
    loop_count: int
    command_id: int
    command_age_ns: int
    state_age_ns: int
    wake_lateness_ns: int
    execution_time_ns: int
    deadline_miss_count: int
    command_stale_count: int
    state_stale_count: int
    overtemperature_count: int
    mode: int
    flags: int

    @property
    def mode_name(self) -> str:
        try:
            return AdapterMode(self.mode).name.lower()
        except ValueError:
            return f"unknown_{self.mode}"


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


def _shared_memory_path(name: str) -> Path:
    if len(name) < 2 or not name.startswith("/") or "/" in name[1:]:
        raise ValueError("POSIX 共享内存名必须形如 /g1_arm_mpc。")
    return Path("/dev/shm") / name[1:]


class UnitreeArmSharedMemoryClient:
    """Python MPC 与 C++ 2 ms 适配器之间的 protocol v2 客户端。"""

    def __init__(
        self,
        name: str = DEFAULT_SHARED_MEMORY_NAME,
        *,
        wait_timeout_s: float = 0.0,
    ):
        self.name = name
        self.path = _shared_memory_path(name)
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
                descriptor = os.open(self.path, os.O_RDWR)
                break
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))

        try:
            size = os.fstat(descriptor).st_size
            if size != _EXPECTED_LAYOUT["layout_size"]:
                raise LayoutMismatchError(
                    f"共享内存大小为 {size}，protocol v2 应为 2304。"
                )
            mapping = mmap.mmap(
                descriptor,
                _EXPECTED_LAYOUT["layout_size"],
                flags=mmap.MAP_SHARED,
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
                    "共享内存 magic/version/layout_size 与 protocol v2 不一致："
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
            sample_id=int(payload.sample_id),
            robot_tick=int(payload.robot_tick),
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

    def read_status(self, *, max_attempts: int = 100) -> AdapterStatusSnapshot:
        """【半核心】读取 C++ 循环耗时、数据年龄和安全模式。"""

        payload = self._read_slot(
            self._require_layout().status,
            _AdapterStatusPayload,
            max_attempts,
        )
        return AdapterStatusSnapshot(
            monotonic_timestamp_ns=int(payload.monotonic_timestamp_ns),
            loop_count=int(payload.loop_count),
            command_id=int(payload.command_id),
            command_age_ns=int(payload.command_age_ns),
            state_age_ns=int(payload.state_age_ns),
            wake_lateness_ns=int(payload.wake_lateness_ns),
            execution_time_ns=int(payload.execution_time_ns),
            deadline_miss_count=int(payload.deadline_miss_count),
            command_stale_count=int(payload.command_stale_count),
            state_stale_count=int(payload.state_stale_count),
            overtemperature_count=int(payload.overtemperature_count),
            mode=int(payload.mode),
            flags=int(payload.flags),
        )

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
    ) -> CommandWriteReceipt:
        arm_weight = float(arm_weight)
        if not math.isfinite(arm_weight):
            raise ValueError("arm_weight 必须是有限数。")
        if command_id is None:
            command_id = self._next_command_id
            self._next_command_id += 1
        command_id = _uint64(command_id, "command_id")
        timestamp_ns = time.monotonic_ns()

        payload = _ArmCommandPayload()
        payload.monotonic_timestamp_ns = timestamp_ns
        payload.command_id = command_id
        payload.mode = int(mode)
        # 默认不请求输出；只有调用者每拍显式传 True 才置位。
        payload.flags = (
            int(CommandFlags.REQUEST_OUTPUT) if request_output else 0
        )
        payload.arm_weight = arm_weight
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
            monotonic_timestamp_ns=timestamp_ns,
            published_sequence=sequence,
            request_output=bool(request_output),
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
    "AdapterMode",
    "AdapterStatusFlags",
    "AdapterStatusSnapshot",
    "CommandMode",
    "CommandWriteReceipt",
    "DEFAULT_SHARED_MEMORY_NAME",
    "LayoutMismatchError",
    "MOTOR_COUNT",
    "PROTOCOL_MAGIC",
    "PROTOCOL_VERSION",
    "RobotStateSnapshot",
    "SeqlockReadError",
    "UnitreeArmSharedMemoryClient",
    "python_layout_report",
)
