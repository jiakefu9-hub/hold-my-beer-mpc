"""Platform-neutral orchestration for the frozen full-task right-arm control.

The core owns only the authoritative 6 ms task-anchor sequence, predictor/H
progression, and MPC proposal generation.  It deliberately does not read
MuJoCo state, write ``d.ctrl``, access DDS/shared memory, or certify hardware
output.  Platform adapters provide one already-validated observation, one
explicit task-clock event, and a helper factory for their kinematics backend.

During ``[0, 24 ms)`` the predictor and MPC still run at every anchor to keep
the accepted warm-start/cache semantics.  The returned intent is nevertheless
marked ``mpc_output_enabled=False``; only the adapter's fixed-posture PD may be
executed during that prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from disturbance_predictor import (
    DisturbancePredictorObservation,
    FullTaskPredictorError,
)
from disturbance_template.full_task_protocol import (
    DEFAULT_FULL_TASK_PROTOCOL,
    FixedStartupPdHandoff,
    FullTaskProtocol,
    direct_step_planned_command,
    is_valid_rotation_batch,
)
from disturbance_types import DisturbanceHorizon, DisturbanceInput
from right_arm_runtime.control_contracts import (
    ControlStateCapabilities,
    TaskClockEvent,
)


RIGHT_ARM_DOF = 5


class FullTaskControlCoreError(RuntimeError):
    """Fail-closed shared-control error with a stable reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = str(reason_code)
        super().__init__(f"{self.reason_code}: {message}")


@dataclass(frozen=True)
class FullTaskControlObservation:
    """One validated, causal state consumed at an exact 6 ms anchor.

    ``source_timestamp_ns`` identifies when the state was sampled;
    ``validated_timestamp_ns`` identifies when the ingress contract accepted
    it.  ``state_valid`` is explicit so an adapter cannot represent an unknown
    or rejected state by silently filling its fields with zeros.
    """

    session_nonce: str
    source_sample_id: int
    source_timestamp_ns: int
    validated_timestamp_ns: int
    state_source: str
    state_valid: bool
    capabilities: ControlStateCapabilities
    current_q: np.ndarray
    current_dq: np.ndarray
    measured_disturbance: DisturbanceInput

    def __post_init__(self) -> None:
        q = np.asarray(self.current_q, dtype=np.float64).copy()
        dq = np.asarray(self.current_dq, dtype=np.float64).copy()
        if q.shape != (RIGHT_ARM_DOF,) or dq.shape != (RIGHT_ARM_DOF,):
            raise ValueError("current_q/current_dq must both have shape (5,)")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            raise ValueError("current_q/current_dq contain NaN/Inf")
        object.__setattr__(self, "current_q", q)
        object.__setattr__(self, "current_dq", dq)


@dataclass(frozen=True)
class FullTaskControlTiming:
    predictor_s: float
    helper_s: float
    mpc_s: float
    diagnostics_s: float


@dataclass(frozen=True)
class FullTaskControlIntent:
    """MPC result before platform execution or hardware certification."""

    session_nonce: str
    task_epoch_id: str
    task_time_ns: int
    full_task_anchor: int
    source_sample_id: int
    source_timestamp_ns: int
    state_capabilities: ControlStateCapabilities
    hardware_torque_state_complete: bool
    mpc_output_enabled: bool
    first_mpc_anchor: bool
    generated_q_ref: np.ndarray
    generated_dq_ref: np.ndarray
    generated_ddq_des: np.ndarray
    generated_ddq_raw: np.ndarray
    disturbance_horizon: DisturbanceHorizon
    predictor_diagnostics: Mapping[str, Any]
    controller_diagnostics: Mapping[str, Any]
    torso_relative_position_reference: np.ndarray | None
    timing: FullTaskControlTiming


class _Predictor(Protocol):
    def reset(self) -> None: ...

    def update(self, observation: DisturbancePredictorObservation) -> None: ...

    def predict(self, horizon: int, dt: float) -> DisturbanceHorizon: ...

    def get_last_diagnostics(self, copy_data: bool = True) -> dict: ...


class _ArmPolicy(Protocol):
    horizon: int
    control_dt: float

    def reset(self) -> None: ...

    def compute_action(self, arm_obs: Mapping[str, Any], helpers: Any) -> tuple: ...

    def get_last_diagnostics(self, copy_data: bool = True) -> dict: ...


HelperFactory = Callable[[DisturbanceInput, DisturbanceHorizon], Any]


class FullTaskRightArmControlCore:
    """Shared final-controller state machine driven by explicit task events."""

    def __init__(
        self,
        *,
        predictor: _Predictor,
        arm_policy: _ArmPolicy,
        nominal_command: np.ndarray,
        protocol: FullTaskProtocol = DEFAULT_FULL_TASK_PROTOCOL,
        startup_handoff: FixedStartupPdHandoff | None = None,
    ) -> None:
        self.predictor = predictor
        self.arm_policy = arm_policy
        self.protocol = protocol
        self.startup_handoff = (
            FixedStartupPdHandoff(0.024, protocol)
            if startup_handoff is None
            else startup_handoff
        )
        nominal = np.asarray(nominal_command, dtype=np.float64).copy()
        if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal_command must be finite with shape (3,)")
        if not np.isclose(
            float(arm_policy.control_dt), protocol.mpc_dt, atol=1e-12, rtol=0.0
        ):
            raise ValueError("arm policy control_dt must match the 6 ms protocol")
        if int(arm_policy.horizon) != protocol.horizon:
            raise ValueError("arm policy horizon must match the full-task protocol")
        if self.startup_handoff.protocol != protocol:
            raise ValueError("startup handoff must use the same protocol")
        self.nominal_command = nominal
        self._session_nonce: str | None = None
        self._task_epoch_id: str | None = None
        self._last_anchor: int | None = None
        self._last_source_sample_id: int | None = None
        self._last_source_timestamp_ns: int | None = None
        self._last_event_sequence: int | None = None
        self._last_event_timestamp_ns: int | None = None

    @property
    def ready(self) -> bool:
        return self._session_nonce is not None

    def reset(self, *, session_nonce: str, task_epoch_id: str) -> None:
        session = str(session_nonce)
        epoch = str(task_epoch_id)
        if not session or not epoch:
            raise ValueError("session_nonce and task_epoch_id must be nonempty")
        self.predictor.reset()
        self.arm_policy.reset()
        self._session_nonce = session
        self._task_epoch_id = epoch
        self._last_anchor = None
        self._last_source_sample_id = None
        self._last_source_timestamp_ns = None
        self._last_event_sequence = None
        self._last_event_timestamp_ns = None

    def _fail(self, reason_code: str, message: str) -> None:
        raise FullTaskControlCoreError(reason_code, message)

    @staticmethod
    def _finite_vector(values: tuple[float, ...], name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise FullTaskControlCoreError(
                "task_command_invalid", f"{name} must be finite with shape (3,)"
            )
        return result

    def _validate_event_and_observation(
        self,
        observation: FullTaskControlObservation,
        event: TaskClockEvent,
    ) -> tuple[int, float]:
        if not self.ready:
            self._fail("reset_required", "reset must establish session and task epoch")
        if not isinstance(observation, FullTaskControlObservation):
            self._fail("observation_type_invalid", "unexpected observation type")
        if not isinstance(event, TaskClockEvent):
            self._fail("task_clock_event_invalid", "explicit TaskClockEvent is required")
        if not observation.state_valid:
            self._fail("state_not_validated", "ingress marked the source state invalid")
        if not observation.state_source:
            self._fail("state_provenance_missing", "state_source must be nonempty")
        if not isinstance(observation.capabilities, ControlStateCapabilities):
            self._fail("state_capabilities_missing", "typed state capabilities are required")
        if not observation.capabilities.mpc_observation_complete:
            self._fail("mpc_state_incomplete", "required right-arm/torso state is unavailable")
        if (
            event.session_nonce != self._session_nonce
            or observation.session_nonce != self._session_nonce
        ):
            self._fail("session_mismatch", "state/task event is from another session")
        if event.task_epoch_id != self._task_epoch_id:
            self._fail("task_epoch_mismatch", "task event is from another epoch")
        if observation.source_sample_id < 0:
            self._fail("source_identity_invalid", "source sample id must be nonnegative")
        if observation.source_timestamp_ns < 0:
            self._fail("source_identity_invalid", "source timestamp must be nonnegative")
        if observation.validated_timestamp_ns < observation.source_timestamp_ns:
            self._fail("source_identity_invalid", "state validation predates sampling")
        if event.source_sample_id != observation.source_sample_id:
            self._fail("task_event_source_mismatch", "task event is bound to another source sample")
        if event.producer_sequence < 0 or event.event_monotonic_timestamp_ns < 0:
            self._fail("task_event_identity_invalid", "task event identity must be nonnegative")
        if event.event_monotonic_timestamp_ns < observation.validated_timestamp_ns:
            self._fail(
                "task_event_future_state",
                "task event predates availability of its bound source state",
            )
        if self._last_event_sequence is not None:
            if event.producer_sequence != self._last_event_sequence + 1:
                self._fail("task_event_sequence_gap_or_replay", "task event sequence must advance by exactly one")
            if event.event_monotonic_timestamp_ns <= self._last_event_timestamp_ns:
                self._fail("task_event_time_repeated_or_backward", "task event timestamp did not advance")
        if self._last_source_sample_id is not None:
            if observation.source_sample_id <= self._last_source_sample_id:
                self._fail("source_sample_repeated_or_backward", "source sample did not advance")
            if observation.source_timestamp_ns <= self._last_source_timestamp_ns:
                self._fail("source_time_repeated_or_backward", "source time did not advance")

        anchor = int(event.full_task_anchor)
        task_time_ns = int(event.task_time_ns)
        if anchor < 0 or task_time_ns < 0:
            self._fail("task_time_invalid", "task anchor/time must be nonnegative")
        anchor_period_ns = int(round(self.protocol.mpc_dt * 1e9))
        if task_time_ns != anchor * anchor_period_ns:
            self._fail("task_time_not_on_anchor", "task time must equal anchor * 6 ms")
        task_time = task_time_ns * 1e-9
        try:
            protocol_anchor = self.protocol.anchor_index(task_time)
        except ValueError as exc:
            self._fail("task_time_out_of_protocol", str(exc))
        if protocol_anchor != anchor:
            self._fail("task_anchor_mismatch", "task time and anchor disagree")
        if self._last_anchor is None:
            if anchor != 0:
                self._fail("task_epoch_did_not_start_at_zero", "first event must be anchor 0")
        elif anchor != self._last_anchor + 1:
            self._fail("task_anchor_gap_or_replay", "every 6 ms anchor is required once")

        planned = self._finite_vector(
            event.planned_command_vx_vy_wz, "planned command"
        )
        runtime = self._finite_vector(
            event.runtime_command_vx_vy_wz, "runtime command"
        )
        expected = direct_step_planned_command(
            task_time, self.nominal_command, self.protocol
        ).planned_command
        if not np.allclose(planned, expected, atol=1e-12, rtol=0.0):
            self._fail("planned_command_protocol_mismatch", "planned command violates direct-step protocol")
        if not np.allclose(runtime[:2], planned[:2], atol=1e-12, rtol=0.0):
            self._fail("runtime_translation_mismatch", "heading correction may only change runtime wz")
        if not np.isfinite(float(event.heading_reference_rad)):
            self._fail("heading_reference_invalid", "heading reference must be finite")
        return anchor, task_time

    @staticmethod
    def _policy_diagnostics(policy: _ArmPolicy) -> dict:
        try:
            return dict(policy.get_last_diagnostics(copy_data=False))
        except TypeError:
            return dict(policy.get_last_diagnostics())

    @staticmethod
    def _predictor_diagnostics(predictor: _Predictor) -> dict:
        try:
            return dict(predictor.get_last_diagnostics(copy_data=False))
        except TypeError:
            return dict(predictor.get_last_diagnostics())

    @staticmethod
    def _vector5(value: Any, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64).copy()
        if result.shape != (RIGHT_ARM_DOF,) or not np.all(np.isfinite(result)):
            raise FullTaskControlCoreError(
                "mpc_output_invalid", f"{name} must be finite with shape (5,)"
            )
        return result

    def step(
        self,
        observation: FullTaskControlObservation,
        task_event: TaskClockEvent,
        helper_factory: HelperFactory,
    ) -> FullTaskControlIntent:
        """Consume exactly one authoritative anchor and generate one intent."""

        anchor, task_time = self._validate_event_and_observation(
            observation, task_event
        )
        if not callable(helper_factory):
            self._fail("helper_factory_invalid", "helper_factory must be callable")

        predictor_start = time.perf_counter()
        try:
            self.predictor.update(
                DisturbancePredictorObservation(
                    # The shared core drives the predictor with authoritative
                    # task time.  It cannot infer or rebase an epoch from the
                    # platform's simulation/wall-clock timestamp.
                    simulation_time=task_time,
                    measured_disturbance=observation.measured_disturbance,
                )
            )
            horizon = self.predictor.predict(
                self.protocol.horizon, self.protocol.mpc_dt
            )
        except FullTaskPredictorError as exc:
            self._fail(f"predictor_{exc.reason_code}", str(exc))
        predictor_s = time.perf_counter() - predictor_start
        if (
            not isinstance(horizon, DisturbanceHorizon)
            or len(horizon.nodes) != self.protocol.horizon + 1
            or len(horizon.intervals) != self.protocol.horizon
        ):
            self._fail("predictor_horizon_invalid", "predictor returned the wrong horizon shape")
        rotations = np.stack(
            [np.asarray(node.rot_world_body, dtype=np.float64) for node in horizon.nodes]
        )
        if not np.all(is_valid_rotation_batch(rotations)):
            self._fail("predictor_rotation_invalid", "predictor returned a non-SO(3) node")

        helper_start = time.perf_counter()
        helpers = helper_factory(observation.measured_disturbance, horizon)
        helper_s = time.perf_counter() - helper_start
        if helpers is None:
            self._fail("helper_construction_failed", "helper factory returned None")

        mpc_start = time.perf_counter()
        q_ref, dq_ref, ddq_des = self.arm_policy.compute_action(
            {
                "current_q": observation.current_q,
                "current_dq": observation.current_dq,
                "dt": self.protocol.mpc_dt,
            },
            helpers,
        )
        mpc_s = time.perf_counter() - mpc_start

        diagnostics_start = time.perf_counter()
        predictor_diagnostics = self._predictor_diagnostics(self.predictor)
        controller_diagnostics = self._policy_diagnostics(self.arm_policy)
        try:
            ddq_raw = controller_diagnostics["ddq_raw"]
        except KeyError:
            self._fail("mpc_diagnostics_invalid", "controller diagnostics omit ddq_raw")
        q_ref = self._vector5(q_ref, "q_ref")
        dq_ref = self._vector5(dq_ref, "dq_ref")
        ddq_des = self._vector5(ddq_des, "ddq_des")
        ddq_raw = self._vector5(ddq_raw, "ddq_raw")
        controller_diagnostics["disturbance_predictor_diagnostics"] = (
            predictor_diagnostics
        )
        reference = getattr(helpers, "torso_relative_position_reference", None)
        reference = (
            None
            if reference is None
            else np.asarray(reference, dtype=np.float64).copy()
        )
        diagnostics_s = time.perf_counter() - diagnostics_start

        sample_index = anchor * self.protocol.mpc_stride
        decision = self.startup_handoff.decision(sample_index)
        intent = FullTaskControlIntent(
            session_nonce=self._session_nonce,
            task_epoch_id=self._task_epoch_id,
            task_time_ns=int(task_event.task_time_ns),
            full_task_anchor=anchor,
            source_sample_id=int(observation.source_sample_id),
            source_timestamp_ns=int(observation.source_timestamp_ns),
            state_capabilities=observation.capabilities,
            hardware_torque_state_complete=(
                observation.capabilities.hardware_torque_state_complete
            ),
            mpc_output_enabled=bool(decision.mpc_control_enabled),
            first_mpc_anchor=bool(decision.first_mpc_anchor),
            generated_q_ref=q_ref,
            generated_dq_ref=dq_ref,
            generated_ddq_des=ddq_des,
            generated_ddq_raw=ddq_raw,
            disturbance_horizon=horizon,
            predictor_diagnostics=predictor_diagnostics,
            controller_diagnostics=controller_diagnostics,
            torso_relative_position_reference=reference,
            timing=FullTaskControlTiming(
                predictor_s=predictor_s,
                helper_s=helper_s,
                mpc_s=mpc_s,
                diagnostics_s=diagnostics_s,
            ),
        )
        # Commit sequence state only after all predictor/helper/MPC outputs
        # have passed structural validation.  A failed cycle cannot be treated
        # as a successfully consumed anchor by the adapter.
        self._last_anchor = anchor
        self._last_source_sample_id = int(observation.source_sample_id)
        self._last_source_timestamp_ns = int(observation.source_timestamp_ns)
        self._last_event_sequence = int(task_event.producer_sequence)
        self._last_event_timestamp_ns = int(task_event.event_monotonic_timestamp_ns)
        return intent


__all__ = (
    "FullTaskControlCoreError",
    "FullTaskControlIntent",
    "FullTaskControlObservation",
    "FullTaskControlTiming",
    "FullTaskRightArmControlCore",
    "HelperFactory",
    "TaskClockEvent",
)
