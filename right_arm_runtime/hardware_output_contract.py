"""Offline-only contract preparation for future Unitree hardware output.

This module deliberately has no Unitree SDK, DDS, shared-memory writer, or
publisher dependency.  It validates state/command binding and transport
semantics for an in-memory fake sink.  Passing these checks is *not* hardware
safety certification and never authorizes a real command write.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Mapping

from right_arm_runtime.control_contracts import TaskClockEvent


ARM_SDK_JOINT_COUNT = 13


class HardwareOutputContractError(RuntimeError):
    """A proposal violates the offline future-output contract."""


class FutureCommandMode(str, Enum):
    ROBOT_PD_PLUS_FEEDFORWARD = "robot_pd_plus_feedforward"
    DIRECT_TORQUE = "direct_torque"


class SafetyClass(str, Enum):
    HARD_STOP = "hard_stop"
    SOFT_GUARD = "soft_guard"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class ValidatedStateIdentity:
    """Identity of one state that passed the hardware-ingress contract."""

    session_nonce: str
    sample_id: int
    source_timestamp_ns: int
    validated_timestamp_ns: int
    arm_sdk_q: tuple[float, ...]


@dataclass(frozen=True)
class HardwareControlProposal:
    """Controller result before any hardware-output certification."""

    session_nonce: str
    proposal_id: int
    source_sample_id: int
    source_timestamp_ns: int
    task_epoch_id: str
    task_time_ns: int
    full_task_anchor: int
    generated_timestamp_ns: int
    expires_timestamp_ns: int
    mode: FutureCommandMode
    arm_weight: float
    active_mask: tuple[bool, ...]
    q_ref: tuple[float, ...]
    dq_ref: tuple[float, ...]
    ddq_des: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    tau: tuple[float, ...]
    diagnostics: Mapping[str, float | int | bool | str]


@dataclass(frozen=True)
class CertifiedHardwareCommand:
    """Command accepted only for offline fake-sink contract testing.

    ``hardware_output_authorized`` is structurally fixed to false.  A future
    real-output stage needs a separate site-verified safety policy and explicit
    authorization; this class cannot provide either.
    """

    session_nonce: str
    command_id: int
    source_sample_id: int
    source_timestamp_ns: int
    task_epoch_id: str
    task_time_ns: int
    full_task_anchor: int
    certified_timestamp_ns: int
    expires_timestamp_ns: int
    mode: FutureCommandMode
    arm_weight: float
    active_mask: tuple[bool, ...]
    q_ref: tuple[float, ...]
    dq_ref: tuple[float, ...]
    ddq_des: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    tau: tuple[float, ...]
    safety_events: tuple[tuple[SafetyClass, str], ...]
    certification_scope: str = "offline_transport_contract_only"
    hardware_safety_certified: bool = False
    hardware_output_authorized: bool = False


@dataclass(frozen=True)
class ExecutionReceipt:
    command_id: int
    source_sample_id: int
    accepted: bool
    sink: str
    receipt_timestamp_ns: int
    dds_write_performed: bool
    hardware_output_performed: bool
    reason: str


def _vector(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != ARM_SDK_JOINT_COUNT:
        raise HardwareOutputContractError(f"{name} must contain 13 values")
    if not all(math.isfinite(value) for value in result):
        raise HardwareOutputContractError(f"{name} contains NaN/Inf")
    return result


def _validate_state_identity(state: ValidatedStateIdentity) -> None:
    if not state.session_nonce:
        raise HardwareOutputContractError("state session_nonce is empty")
    if state.sample_id <= 0 or state.source_timestamp_ns <= 0:
        raise HardwareOutputContractError("state identity is invalid")
    if state.validated_timestamp_ns < state.source_timestamp_ns:
        raise HardwareOutputContractError("state validation predates source")
    _vector(state.arm_sdk_q, "state.arm_sdk_q")


def certify_for_offline_fake_sink(
    proposal: HardwareControlProposal,
    state: ValidatedStateIdentity,
    *,
    now_ns: int,
) -> CertifiedHardwareCommand:
    """Validate binding and command semantics without claiming hardware safety."""

    _validate_state_identity(state)
    now_ns = int(now_ns)
    if proposal.session_nonce != state.session_nonce:
        raise HardwareOutputContractError("proposal/state session mismatch")
    if proposal.source_sample_id != state.sample_id:
        raise HardwareOutputContractError("proposal/state sample mismatch")
    if proposal.source_timestamp_ns != state.source_timestamp_ns:
        raise HardwareOutputContractError("proposal/state timestamp mismatch")
    if not proposal.task_epoch_id:
        raise HardwareOutputContractError("task_epoch_id is empty")
    if proposal.task_time_ns < 0 or proposal.full_task_anchor < 0:
        raise HardwareOutputContractError("task time/anchor is invalid")
    if proposal.task_time_ns != proposal.full_task_anchor * 6_000_000:
        raise HardwareOutputContractError("task time is not the exact 6 ms anchor")
    if proposal.proposal_id <= 0:
        raise HardwareOutputContractError("proposal_id must be positive")
    if proposal.generated_timestamp_ns < state.validated_timestamp_ns:
        raise HardwareOutputContractError("proposal predates state validation")
    if proposal.expires_timestamp_ns <= proposal.generated_timestamp_ns:
        raise HardwareOutputContractError("proposal expiry is invalid")
    if now_ns > proposal.expires_timestamp_ns:
        raise HardwareOutputContractError("proposal is expired")
    if not math.isfinite(float(proposal.arm_weight)) or not (
        0.0 <= float(proposal.arm_weight) <= 1.0
    ):
        raise HardwareOutputContractError("arm_weight is outside [0, 1]")
    if len(proposal.active_mask) != ARM_SDK_JOINT_COUNT:
        raise HardwareOutputContractError("active_mask must contain 13 values")

    q_ref = _vector(proposal.q_ref, "q_ref")
    dq_ref = _vector(proposal.dq_ref, "dq_ref")
    ddq_des = _vector(proposal.ddq_des, "ddq_des")
    kp = _vector(proposal.kp, "kp")
    kd = _vector(proposal.kd, "kd")
    tau = _vector(proposal.tau, "tau")
    state_q = _vector(state.arm_sdk_q, "state.arm_sdk_q")
    if any(value < 0.0 for value in (*kp, *kd)):
        raise HardwareOutputContractError("kp/kd must be nonnegative")

    if proposal.mode is FutureCommandMode.DIRECT_TORQUE:
        if any(value != 0.0 for value in (*kp, *kd)):
            raise HardwareOutputContractError(
                "direct torque requires robot-side kp/kd to be zero"
            )
    elif proposal.mode is not FutureCommandMode.ROBOT_PD_PLUS_FEEDFORWARD:
        raise HardwareOutputContractError("unsupported command mode")

    for index, active in enumerate(proposal.active_mask):
        if active:
            continue
        if (
            q_ref[index] != state_q[index]
            or dq_ref[index] != 0.0
            or ddq_des[index] != 0.0
            or kp[index] != 0.0
            or kd[index] != 0.0
            or tau[index] != 0.0
        ):
            raise HardwareOutputContractError(
                f"inactive slot {index} is not a zero-action hold"
            )

    events: list[tuple[SafetyClass, str]] = []
    if "predicted_max_abs_qacc" in proposal.diagnostics:
        value = float(proposal.diagnostics["predicted_max_abs_qacc"])
        if not math.isfinite(value):
            raise HardwareOutputContractError(
                "predicted_max_abs_qacc diagnostic is nonfinite"
            )
        # MuJoCo qacc is retained as evidence only.  No real-hardware hard
        # threshold is invented by this offline preparation stage.
        events.append((SafetyClass.DIAGNOSTIC, "PREDICTED_QACC_RECORDED"))

    return CertifiedHardwareCommand(
        session_nonce=proposal.session_nonce,
        command_id=proposal.proposal_id,
        source_sample_id=proposal.source_sample_id,
        source_timestamp_ns=proposal.source_timestamp_ns,
        task_epoch_id=proposal.task_epoch_id,
        task_time_ns=proposal.task_time_ns,
        full_task_anchor=proposal.full_task_anchor,
        certified_timestamp_ns=now_ns,
        expires_timestamp_ns=proposal.expires_timestamp_ns,
        mode=proposal.mode,
        arm_weight=float(proposal.arm_weight),
        active_mask=tuple(bool(value) for value in proposal.active_mask),
        q_ref=q_ref,
        dq_ref=dq_ref,
        ddq_des=ddq_des,
        kp=kp,
        kd=kd,
        tau=tau,
        safety_events=tuple(events),
    )


class FakeHardwareCommandSink:
    """In-memory O1 sink with expiry, session, replay and watchdog checks."""

    def __init__(self, *, session_nonce: str, watchdog_timeout_ns: int):
        if not session_nonce:
            raise ValueError("session_nonce must be nonempty")
        if int(watchdog_timeout_ns) <= 0:
            raise ValueError("watchdog_timeout_ns must be positive")
        self.session_nonce = session_nonce
        self.watchdog_timeout_ns = int(watchdog_timeout_ns)
        self._last_command_id = 0
        self._last_source_sample_id = 0
        self._last_accept_ns: int | None = None

    def submit(
        self, command: CertifiedHardwareCommand, *, now_ns: int
    ) -> ExecutionReceipt:
        now_ns = int(now_ns)
        reason = "ACCEPTED_OFFLINE_FAKE_SINK"
        accepted = True
        if command.hardware_output_authorized:
            accepted, reason = False, "REAL_OUTPUT_AUTHORIZATION_FORBIDDEN"
        elif command.hardware_safety_certified:
            accepted, reason = False, "UNSUPPORTED_HARDWARE_SAFETY_CLAIM"
        elif command.certification_scope != "offline_transport_contract_only":
            accepted, reason = False, "INVALID_CERTIFICATION_SCOPE"
        elif command.session_nonce != self.session_nonce:
            accepted, reason = False, "SESSION_MISMATCH"
        elif now_ns > command.expires_timestamp_ns:
            accepted, reason = False, "COMMAND_EXPIRED"
        elif command.command_id <= self._last_command_id:
            accepted, reason = False, "COMMAND_REPLAY_OR_REGRESSION"
        elif command.source_sample_id <= self._last_source_sample_id:
            accepted, reason = False, "SOURCE_STATE_REPLAY_OR_REGRESSION"
        if accepted:
            self._last_command_id = command.command_id
            self._last_source_sample_id = command.source_sample_id
            self._last_accept_ns = now_ns
        return ExecutionReceipt(
            command_id=command.command_id,
            source_sample_id=command.source_sample_id,
            accepted=accepted,
            sink="offline_in_memory_fake_sink",
            receipt_timestamp_ns=now_ns,
            dds_write_performed=False,
            hardware_output_performed=False,
            reason=reason,
        )

    def watchdog_receipt(self, *, now_ns: int) -> ExecutionReceipt:
        now_ns = int(now_ns)
        healthy = (
            self._last_accept_ns is not None
            and now_ns - self._last_accept_ns <= self.watchdog_timeout_ns
        )
        return ExecutionReceipt(
            command_id=self._last_command_id,
            source_sample_id=self._last_source_sample_id,
            accepted=healthy,
            sink="offline_in_memory_fake_sink",
            receipt_timestamp_ns=now_ns,
            dds_write_performed=False,
            hardware_output_performed=False,
            reason="WATCHDOG_HEALTHY" if healthy else "WATCHDOG_EXPIRED",
        )


__all__ = (
    "ARM_SDK_JOINT_COUNT",
    "CertifiedHardwareCommand",
    "ExecutionReceipt",
    "FakeHardwareCommandSink",
    "FutureCommandMode",
    "HardwareControlProposal",
    "HardwareOutputContractError",
    "SafetyClass",
    "TaskClockEvent",
    "ValidatedStateIdentity",
    "certify_for_offline_fake_sink",
)
