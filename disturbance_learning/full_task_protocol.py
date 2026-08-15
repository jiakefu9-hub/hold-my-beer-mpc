"""Frozen T1 full-task timing, command, anchor, and causal-H semantics.

This module is deliberately independent from the legacy ramp schedule and the
four existing disturbance predictors.  The offline collector and the future
online full-task predictors must share these definitions instead of copying
time/index arithmetic into separate call sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PROTOCOL_NAME = "full_task_direct_step"
PROTOCOL_VERSION = "full_task_direct_step_v1"
RAW_SCHEMA_VERSION = "full_task_raw_v1"

# This is the one authoritative event order for T1 full-task data.  At a
# boundary such as 6.4 s, the policy update was completed after the preceding
# [t-2 ms, t) physics step, so the next strict pre-step sample at t already
# contains the newly consumed planned/runtime command.
PRE_STEP_EVENT_ORDER = (
    "boundary_command_and_policy_state_already_committed_from_previous_step",
    "compute_leg_pd_from_latest_policy_target",
    "sample_torso_and_joint_state_for_current_pre_step",
    "run_predictor_and_mpc_when_current_time_is_a_6ms_anchor",
    "run_ddq_to_torque_process_chain_and_write_actuator_ctrl",
    "append_strict_pre_step_raw_sample_for_[t,t+2ms)",
    "mujoco_mj_step_advances_[t,t+2ms)",
    "legacy_post_step_evaluation_recording",
    "increment_physics_counter",
    "at_20ms_boundary_update_planned_command_heading_and_policy_for_next_step",
)


@dataclass(frozen=True)
class FullTaskProtocol:
    """Versioned direct-step task and its exact discrete grids."""

    physics_dt: float = 0.002
    mpc_dt: float = 0.006
    policy_dt: float = 0.020
    gait_period: float = 0.8
    stop_time: float = 6.4
    headline_end: float = 8.0
    record_end: float = 8.06
    horizon: int = 9
    protocol_name: str = PROTOCOL_NAME
    protocol_version: str = PROTOCOL_VERSION
    raw_schema_version: str = RAW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        scalar_values = (
            self.physics_dt,
            self.mpc_dt,
            self.policy_dt,
            self.gait_period,
            self.stop_time,
            self.headline_end,
            self.record_end,
        )
        if not all(np.isfinite(scalar_values)):
            raise ValueError("full-task protocol times must be finite")
        if not (
            0.0 < self.physics_dt <= self.mpc_dt <= self.policy_dt
            and 0.0 < self.stop_time < self.headline_end < self.record_end
            and self.horizon > 0
        ):
            raise ValueError("full-task protocol times are not ordered")
        for name, numerator, denominator in (
            ("mpc_dt/physics_dt", self.mpc_dt, self.physics_dt),
            ("policy_dt/physics_dt", self.policy_dt, self.physics_dt),
            ("stop_time/policy_dt", self.stop_time, self.policy_dt),
            ("record_end/physics_dt", self.record_end, self.physics_dt),
        ):
            ratio = numerator / denominator
            if not np.isclose(ratio, round(ratio), atol=1e-10, rtol=0.0):
                raise ValueError(f"{name} must be an integer grid ratio")
        if self.last_horizon_node_time > self.record_end + 1e-12:
            raise ValueError("record tail does not cover the final horizon node")

    @property
    def physics_steps(self) -> int:
        """Number of [t,t+physics_dt) intervals ending at record_end."""
        return int(round(self.record_end / self.physics_dt))

    @property
    def mpc_stride(self) -> int:
        return int(round(self.mpc_dt / self.physics_dt))

    @property
    def policy_stride(self) -> int:
        return int(round(self.policy_dt / self.physics_dt))

    @staticmethod
    def _half_open_count(end_time: float, dt: float) -> int:
        return int(np.ceil(float(end_time) / float(dt) - 1e-12))

    @property
    def headline_anchor_count(self) -> int:
        return self._half_open_count(self.headline_end, self.mpc_dt)

    @property
    def recorded_anchor_count(self) -> int:
        return self._half_open_count(self.record_end, self.mpc_dt)

    @property
    def last_headline_anchor_time(self) -> float:
        return (self.headline_anchor_count - 1) * self.mpc_dt

    @property
    def last_horizon_node_time(self) -> float:
        return self.last_headline_anchor_time + self.horizon * self.mpc_dt

    @property
    def headline_anchor_times(self) -> np.ndarray:
        return np.arange(self.headline_anchor_count, dtype=np.float64) * self.mpc_dt

    def validate_task_time(self, task_time: float, *, allow_endpoint: bool = True) -> float:
        value = float(task_time)
        upper = self.record_end
        if not np.isfinite(value) or value < -1e-12:
            raise ValueError("task time must be finite and nonnegative")
        if value > upper + 1e-12 or (not allow_endpoint and value >= upper - 1e-12):
            raise ValueError(
                f"task time {value:.9f} is outside the protocol range "
                f"[0,{upper:.9f}{']' if allow_endpoint else ')'}"
            )
        return 0.0 if abs(value) <= 1e-12 else value

    def sample_time(self, sample_index: int) -> float:
        index = int(sample_index)
        if index < 0 or index >= self.physics_steps:
            raise ValueError("physics sample index is outside the raw interval grid")
        return index * self.physics_dt

    def is_mpc_anchor_sample(self, sample_index: int) -> bool:
        index = int(sample_index)
        return index >= 0 and index % self.mpc_stride == 0

    def anchor_index_from_sample(self, sample_index: int) -> int:
        index = int(sample_index)
        if not self.is_mpc_anchor_sample(index):
            raise ValueError("physics sample is not on the 6 ms MPC grid")
        return index // self.mpc_stride

    def anchor_index(self, task_time: float) -> int:
        value = self.validate_task_time(task_time)
        index = int(round(value / self.mpc_dt))
        if not np.isclose(value, index * self.mpc_dt, atol=1e-10, rtol=0.0):
            raise ValueError("task time is not on the 6 ms MPC anchor grid")
        return index

    def future_window_sample_indices(self, anchor_index: int) -> dict[str, np.ndarray]:
        """Return exact raw indices for a 10-node/9-interval future window."""
        anchor = int(anchor_index)
        if anchor < 0 or anchor >= self.headline_anchor_count:
            raise ValueError("anchor is outside the [0, headline_end) grid")
        start = anchor * self.mpc_stride
        node = start + self.mpc_stride * np.arange(self.horizon + 1, dtype=np.int64)
        interval_start = node[:-1].copy()
        interval_end = node[1:].copy()
        if int(node[-1]) >= self.physics_steps:
            raise ValueError("raw tail does not cover this future window")
        return {
            "node": node,
            "interval_start": interval_start,
            "interval_end": interval_end,
        }


DEFAULT_FULL_TASK_PROTOCOL = FullTaskProtocol()


@dataclass(frozen=True)
class DirectStepCommand:
    planned_command: np.ndarray
    segment_id: int
    segment_name: str


def direct_step_planned_command(
    task_time: float,
    nominal_command: np.ndarray,
    protocol: FullTaskProtocol = DEFAULT_FULL_TASK_PROTOCOL,
) -> DirectStepCommand:
    """Return the independent no-ramp full-task planned command.

    Only planned translational velocity switches to zero.  Planned yaw-rate
    feedforward remains the configured nominal value; heading control may
    replace it with a small closed-loop runtime yaw-rate.
    """
    value = protocol.validate_task_time(task_time)
    nominal = np.asarray(nominal_command, dtype=np.float64)
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("nominal command must be finite [vx, vy, wz]")
    command = nominal.copy()
    if value >= protocol.stop_time - 1e-12:
        command[:2] = 0.0
        return DirectStepCommand(command, 1, "direct_stopped_translation")
    return DirectStepCommand(command, 0, "direct_walking")


class FullTaskClock:
    """Map monotonic simulation time to a resettable, non-wrapping task time."""

    def __init__(self, protocol: FullTaskProtocol = DEFAULT_FULL_TASK_PROTOCOL):
        self.protocol = protocol
        self._epoch_index = -1
        self._epoch_label: str | None = None
        self._origin_simulation_time: float | None = None
        self._last_simulation_time: float | None = None

    @property
    def epoch_index(self) -> int:
        if self._origin_simulation_time is None:
            raise RuntimeError("full-task clock has not been reset")
        return self._epoch_index

    @property
    def epoch_label(self) -> str:
        if self._epoch_label is None:
            raise RuntimeError("full-task clock has not been reset")
        return self._epoch_label

    @property
    def origin_simulation_time(self) -> float:
        if self._origin_simulation_time is None:
            raise RuntimeError("full-task clock has not been reset")
        return self._origin_simulation_time

    def reset(self, simulation_time: float = 0.0, *, epoch_label: str | None = None) -> None:
        origin = float(simulation_time)
        if not np.isfinite(origin):
            raise ValueError("task epoch simulation time must be finite")
        self._epoch_index += 1
        self._epoch_label = (
            str(epoch_label)
            if epoch_label is not None
            else f"task_epoch_{self._epoch_index}"
        )
        if not self._epoch_label:
            raise ValueError("task epoch label must be nonempty")
        self._origin_simulation_time = origin
        self._last_simulation_time = origin

    def observe(self, simulation_time: float) -> float:
        if self._origin_simulation_time is None or self._last_simulation_time is None:
            raise RuntimeError("full-task clock must be reset before use")
        current = float(simulation_time)
        if not np.isfinite(current):
            raise ValueError("simulation time must be finite")
        if current < self._last_simulation_time - 1e-12:
            raise ValueError("simulation time moved backward within one task epoch")
        task_time = current - self._origin_simulation_time
        value = self.protocol.validate_task_time(task_time)
        self._last_simulation_time = max(self._last_simulation_time, current)
        return value


def _validate_rotation(rotation: np.ndarray, *, name: str = "rotation") -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} is not orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} is not a proper SO(3) rotation")
    return matrix


def rotation_z(yaw: float) -> np.ndarray:
    cosine = np.cos(float(yaw))
    sine = np.sin(float(yaw))
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CausalHeadingFrameState:
    anchor_index: int
    task_time: float
    current_cycle_index: int
    source_cycle_index: int
    source_sample_count: int
    source: str
    yaw_world: float
    rotation_world_heading: np.ndarray
    concentration: float


class FullTaskCausalHeadingFrame:
    """Causal 6 ms H-frame estimator for future offline/online parity."""

    DEFINITION_VERSION = "full_task_cycle_held_heading_v1"
    SOURCE_CODES = {
        "first_cycle_causal_prefix": 0,
        "previous_complete_cycle": 1,
    }

    def __init__(self, protocol: FullTaskProtocol = DEFAULT_FULL_TASK_PROTOCOL):
        self.protocol = protocol
        self.reset()

    def reset(self) -> None:
        self._last_anchor_index: int | None = None
        self._current_cycle_index: int | None = None
        self._sine_sum = 0.0
        self._cosine_sum = 0.0
        self._sample_count = 0
        self._previous_heading: float | None = None
        self._previous_concentration = np.nan
        self._previous_sample_count = 0
        self._previous_cycle_index = -1
        self.last_state: CausalHeadingFrameState | None = None

    @staticmethod
    def _circular_mean(sine_sum: float, cosine_sum: float, count: int) -> tuple[float, float]:
        if count <= 0:
            raise ValueError("cannot form a heading frame from zero samples")
        concentration = float(np.hypot(sine_sum, cosine_sum) / count)
        if concentration < 1e-8:
            raise ValueError("yaw samples do not define a stable circular mean")
        return float(np.arctan2(sine_sum, cosine_sum)), concentration

    def update(self, task_time: float, rotation_world_body: np.ndarray) -> CausalHeadingFrameState:
        anchor_index = self.protocol.anchor_index(task_time)
        if self._last_anchor_index is not None:
            if anchor_index <= self._last_anchor_index:
                raise ValueError("causal-H anchor time repeated or moved backward")
            if anchor_index != self._last_anchor_index + 1:
                raise ValueError("causal-H requires every 6 ms anchor without gaps")
        rotation = _validate_rotation(rotation_world_body, name="rotation_world_body")
        yaw_world = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        cycle_index = int(np.floor(float(task_time) / self.protocol.gait_period + 1e-12))

        if self._current_cycle_index is None:
            self._current_cycle_index = cycle_index
        elif cycle_index < self._current_cycle_index:
            raise ValueError("causal-H cycle index moved backward")
        elif cycle_index > self._current_cycle_index:
            if cycle_index != self._current_cycle_index + 1:
                raise ValueError("causal-H skipped an entire gait cycle")
            heading, concentration = self._circular_mean(
                self._sine_sum, self._cosine_sum, self._sample_count
            )
            self._previous_heading = heading
            self._previous_concentration = concentration
            self._previous_sample_count = self._sample_count
            self._previous_cycle_index = self._current_cycle_index
            self._current_cycle_index = cycle_index
            self._sine_sum = 0.0
            self._cosine_sum = 0.0
            self._sample_count = 0

        self._sine_sum += float(np.sin(yaw_world))
        self._cosine_sum += float(np.cos(yaw_world))
        self._sample_count += 1

        if cycle_index == 0:
            heading, concentration = self._circular_mean(
                self._sine_sum, self._cosine_sum, self._sample_count
            )
            source = "first_cycle_causal_prefix"
            source_cycle_index = 0
            source_sample_count = self._sample_count
        else:
            if self._previous_heading is None:
                raise ValueError("previous complete gait cycle is unavailable")
            heading = self._previous_heading
            concentration = self._previous_concentration
            source = "previous_complete_cycle"
            source_cycle_index = self._previous_cycle_index
            source_sample_count = self._previous_sample_count

        state = CausalHeadingFrameState(
            anchor_index=anchor_index,
            task_time=float(task_time),
            current_cycle_index=cycle_index,
            source_cycle_index=source_cycle_index,
            source_sample_count=source_sample_count,
            source=source,
            yaw_world=heading,
            rotation_world_heading=rotation_z(heading),
            concentration=float(concentration),
        )
        _validate_rotation(state.rotation_world_heading, name="rotation_world_heading")
        self._last_anchor_index = anchor_index
        self.last_state = state
        return state

    def rotation_heading_body(self, rotation_world_body: np.ndarray) -> np.ndarray:
        if self.last_state is None:
            raise RuntimeError("causal-H has not received its first anchor")
        rotation = _validate_rotation(rotation_world_body, name="rotation_world_body")
        relative = self.last_state.rotation_world_heading.T @ rotation
        return _validate_rotation(relative, name="rotation_heading_body").copy()


class FullTaskContinuousHeadingFrame:
    """Continuous, causal full-task H frame used by template v2.

    The estimator consumes every 6 ms anchor exactly once.  Before 0.8 s it
    uses the causal prefix, then it uses the inclusive set of available
    anchors in ``[t - 0.8 s, t]``.  The first anchor at or after the 6.4 s
    direct stop reuses the last state established strictly before the stop.
    No future yaw, interpolation, or additional low-pass filter is involved.
    """

    DEFINITION_VERSION = "full_task_continuous_heading_v2"
    SOURCE_CODES = {
        "causal_prefix": 0,
        "rolling_0p8s": 1,
        "frozen_pre_stop": 2,
    }

    def __init__(self, protocol: FullTaskProtocol = DEFAULT_FULL_TASK_PROTOCOL):
        self.protocol = protocol
        self.reset()

    def reset(self) -> None:
        self._last_anchor_index: int | None = None
        self._window: list[tuple[float, float, float]] = []
        self._sine_sum = 0.0
        self._cosine_sum = 0.0
        self.last_state: CausalHeadingFrameState | None = None

    @staticmethod
    def _circular_mean(sine_sum: float, cosine_sum: float, count: int) -> tuple[float, float]:
        return FullTaskCausalHeadingFrame._circular_mean(sine_sum, cosine_sum, count)

    def _validate_next_anchor(self, task_time: float) -> int:
        anchor_index = self.protocol.anchor_index(task_time)
        if self._last_anchor_index is not None:
            if anchor_index <= self._last_anchor_index:
                raise ValueError("continuous-H anchor time repeated or moved backward")
            if anchor_index != self._last_anchor_index + 1:
                raise ValueError("continuous-H requires every 6 ms anchor without gaps")
        return anchor_index

    def update(self, task_time: float, rotation_world_body: np.ndarray) -> CausalHeadingFrameState:
        anchor_index = self._validate_next_anchor(task_time)
        rotation = _validate_rotation(rotation_world_body, name="rotation_world_body")
        current_cycle = int(np.floor(float(task_time) / self.protocol.gait_period + 1e-12))

        if float(task_time) >= self.protocol.stop_time - 1e-12:
            if self.last_state is None:
                raise ValueError("continuous-H cannot freeze before a pre-stop anchor exists")
            state = CausalHeadingFrameState(
                anchor_index=anchor_index,
                task_time=float(task_time),
                current_cycle_index=current_cycle,
                source_cycle_index=self.last_state.source_cycle_index,
                source_sample_count=self.last_state.source_sample_count,
                source="frozen_pre_stop",
                yaw_world=self.last_state.yaw_world,
                rotation_world_heading=self.last_state.rotation_world_heading.copy(),
                concentration=self.last_state.concentration,
            )
            self._last_anchor_index = anchor_index
            self.last_state = state
            return state

        yaw_world = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        sine = float(np.sin(yaw_world))
        cosine = float(np.cos(yaw_world))
        self._window.append((float(task_time), sine, cosine))
        self._sine_sum += sine
        self._cosine_sum += cosine

        if float(task_time) >= self.protocol.gait_period - 1e-12:
            cutoff = float(task_time) - self.protocol.gait_period
            while self._window and self._window[0][0] < cutoff - 1e-12:
                _, old_sine, old_cosine = self._window.pop(0)
                self._sine_sum -= old_sine
                self._cosine_sum -= old_cosine
            source = "rolling_0p8s"
        else:
            source = "causal_prefix"

        heading, concentration = self._circular_mean(
            self._sine_sum, self._cosine_sum, len(self._window)
        )
        state = CausalHeadingFrameState(
            anchor_index=anchor_index,
            task_time=float(task_time),
            current_cycle_index=current_cycle,
            source_cycle_index=-1,
            source_sample_count=len(self._window),
            source=source,
            yaw_world=heading,
            rotation_world_heading=rotation_z(heading),
            concentration=float(concentration),
        )
        _validate_rotation(state.rotation_world_heading, name="rotation_world_heading")
        self._last_anchor_index = anchor_index
        self.last_state = state
        return state

    def rotation_heading_body(self, rotation_world_body: np.ndarray) -> np.ndarray:
        if self.last_state is None:
            raise RuntimeError("continuous-H has not received its first anchor")
        rotation = _validate_rotation(rotation_world_body, name="rotation_world_body")
        relative = self.last_state.rotation_world_heading.T @ rotation
        return _validate_rotation(relative, name="rotation_heading_body").copy()


def recompute_continuous_heading_frames(
    anchor_task_time: np.ndarray,
    rotation_world_body: np.ndarray,
    protocol: FullTaskProtocol = DEFAULT_FULL_TASK_PROTOCOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay the shared v2 helper for collector/builder/parity validation."""
    times = np.asarray(anchor_task_time, dtype=np.float64)
    rotations = np.asarray(rotation_world_body, dtype=np.float64)
    if times.ndim != 1 or rotations.shape != (len(times), 3, 3):
        raise ValueError("continuous-H replay expects [anchor] times and [anchor,3,3] rotations")
    helper = FullTaskContinuousHeadingFrame(protocol)
    states = [helper.update(float(time), rotation) for time, rotation in zip(times, rotations)]
    return (
        np.asarray([state.yaw_world for state in states], dtype=np.float64),
        np.stack([state.rotation_world_heading for state in states]),
        np.asarray([state.concentration for state in states], dtype=np.float64),
        np.asarray([FullTaskContinuousHeadingFrame.SOURCE_CODES[state.source] for state in states], dtype=np.int64),
    )


def rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Convert one or more proper rotations to world-frame roll/pitch/yaw."""
    matrices = np.asarray(rotation, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError("rotation array must end in shape (3,3)")
    pitch = np.arcsin(np.clip(-matrices[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(matrices[..., 2, 1], matrices[..., 2, 2])
    yaw = np.arctan2(matrices[..., 1, 0], matrices[..., 0, 0])
    return np.stack((roll, pitch, yaw), axis=-1)


def is_valid_rotation_batch(rotation: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    matrices = np.asarray(rotation, dtype=np.float64)
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3):
        raise ValueError("rotation batch must end in shape (3,3)")
    orthogonal = np.all(
        np.isclose(
            np.swapaxes(matrices, -1, -2) @ matrices,
            np.eye(3),
            atol=atol,
            rtol=0.0,
        ),
        axis=(-2, -1),
    )
    determinant = np.isclose(
        np.linalg.det(matrices), 1.0, atol=atol, rtol=0.0
    )
    finite = np.all(np.isfinite(matrices), axis=(-2, -1))
    return finite & orthogonal & determinant
