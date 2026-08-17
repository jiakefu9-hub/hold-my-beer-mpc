"""Strict pre-step T1 raw recording, validation, manifest, and smoke plots."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from disturbance_template.full_task_protocol import (
    PRE_STEP_EVENT_ORDER,
    DirectStepCommand,
    FullTaskCausalHeadingFrame,
    FullTaskClock,
    FullTaskContinuousHeadingFrame,
    FullTaskProtocol,
    direct_step_planned_command,
    is_valid_rotation_batch,
    rotation_matrix_to_rpy,
)


PRE_STEP_SAMPLE_DEFINITION = (
    "state_timestamp_and_commands_describe_the_interval_[t,t+physics_dt);"
    "sample_is_appended_after_ctrl_is_finalized_and_before_mujoco_mj_step"
)
FALL_MIN_TORSO_HEIGHT_M = 0.45
FALL_MAX_ABS_ROLL_PITCH_RAD = float(np.deg2rad(60.0))


RAW_FIELD_DEFINITIONS: dict[str, dict[str, str]] = {
    "task_epoch": {"shape": "[]", "unit": "index", "frame": "task", "meaning": "resettable task epoch index"},
    "task_time": {"shape": "[]", "unit": "s", "frame": "task", "meaning": "time since the current task epoch"},
    "simulation_time": {"shape": "[]", "unit": "s", "frame": "MuJoCo", "meaning": "monotonic simulation time at strict pre-step"},
    "physics_sample_index": {"shape": "[]", "unit": "index", "frame": "2ms_grid", "meaning": "zero-based raw interval index"},
    "planned_command": {"shape": "[3]", "unit": "m/s,m/s,rad/s", "frame": "locomotion_command", "meaning": "direct-step command before heading correction"},
    "runtime_command": {"shape": "[3]", "unit": "m/s,m/s,rad/s", "frame": "locomotion_command", "meaning": "command actually consumed by the lower-body policy"},
    "policy_update_applied": {"shape": "[]", "unit": "bool", "frame": "20ms_grid", "meaning": "policy consumed command at this boundary"},
    "policy_command_consumed_time": {"shape": "[]", "unit": "s", "frame": "task", "meaning": "latest policy update task time; NaN before first update"},
    "mpc_anchor": {"shape": "[]", "unit": "bool", "frame": "6ms_grid", "meaning": "predictor/MPC updated at this pre-step"},
    "mpc_anchor_index": {"shape": "[]", "unit": "index", "frame": "6ms_grid", "meaning": "anchor index or -1 off-grid"},
    "gait_phase": {"shape": "[]", "unit": "cycle_fraction", "frame": "locomotion_policy", "meaning": "physical gait time modulo gait period"},
    "gait_cycle_index": {"shape": "[]", "unit": "index", "frame": "locomotion_policy", "meaning": "floor(task_time/gait_period)"},
    "torso_position_world": {"shape": "[3]", "unit": "m", "frame": "world", "meaning": "torso body position"},
    "torso_rotation_world": {"shape": "[3,3]", "unit": "SO(3)", "frame": "R_WB", "meaning": "world-from-torso orientation"},
    "torso_linear_velocity_world": {"shape": "[3]", "unit": "m/s", "frame": "world", "meaning": "torso IMU-site linear velocity"},
    "torso_angular_velocity_world": {"shape": "[3]", "unit": "rad/s", "frame": "world", "meaning": "torso IMU-site angular velocity"},
    "torso_linear_acceleration_world_raw": {"shape": "[3]", "unit": "m/s^2", "frame": "world", "meaning": "causal IMU acceleration before MPC filter"},
    "torso_linear_acceleration_world_used": {"shape": "[3]", "unit": "m/s^2", "frame": "world", "meaning": "acceleration consumed by current predictor/MPC"},
    "torso_angular_acceleration_world_raw": {"shape": "[3]", "unit": "rad/s^2", "frame": "world", "meaning": "causal finite-difference alpha before MPC filter"},
    "torso_angular_acceleration_world_used": {"shape": "[3]", "unit": "rad/s^2", "frame": "world", "meaning": "alpha consumed by current predictor/MPC"},
    "lower_body_q": {"shape": "[12]", "unit": "rad", "frame": "joint", "meaning": "lower-body joint positions"},
    "lower_body_dq": {"shape": "[12]", "unit": "rad/s", "frame": "joint", "meaning": "lower-body joint velocities"},
    "lower_body_policy_target": {"shape": "[12]", "unit": "rad", "frame": "joint", "meaning": "latest lower-body policy target"},
    "right_arm_q": {"shape": "[5]", "unit": "rad", "frame": "joint", "meaning": "right-arm joint positions"},
    "right_arm_dq": {"shape": "[5]", "unit": "rad/s", "frame": "joint", "meaning": "right-arm joint velocities"},
    "right_arm_ddq_des": {"shape": "[5]", "unit": "rad/s^2", "frame": "joint", "meaning": "currently active MPC acceleration command"},
    "generalized_qpos": {"shape": "[nq]", "unit": "MuJoCo_qpos", "frame": "model", "meaning": "complete pre-step generalized position"},
    "generalized_qvel": {"shape": "[nv]", "unit": "MuJoCo_qvel", "frame": "model", "meaning": "complete pre-step generalized velocity"},
    "generalized_qacc": {"shape": "[nv]", "unit": "MuJoCo_qacc", "frame": "model", "meaning": "latest causally available acceleration"},
    "actuator_ctrl": {"shape": "[nu]", "unit": "actuator_input", "frame": "model", "meaning": "final control for upcoming physics interval"},
    "heading_reference_world": {"shape": "[]", "unit": "rad", "frame": "world", "meaning": "heading controller reference; NaN before first 20ms update"},
    "heading_measurement_world": {"shape": "[]", "unit": "rad", "frame": "world", "meaning": "instantaneous wrapped torso yaw"},
    "heading_yaw_filtered": {"shape": "[]", "unit": "rad", "frame": "world", "meaning": "heading controller moving-average yaw"},
    "heading_yaw_error": {"shape": "[]", "unit": "rad", "frame": "world", "meaning": "reference minus filtered yaw"},
    "heading_yaw_rate_correction": {"shape": "[]", "unit": "rad/s", "frame": "world", "meaning": "heading feedback correction"},
    "heading_yaw_rate_command": {"shape": "[]", "unit": "rad/s", "frame": "world", "meaning": "runtime wz after heading control"},
    "heading_command_saturated": {"shape": "[]", "unit": "bool", "frame": "controller", "meaning": "heading command saturation flag"},
    "causal_h_yaw_world": {"shape": "[]", "unit": "rad", "frame": "world", "meaning": "held H yaw available at the latest MPC anchor"},
    "causal_h_rotation_world": {"shape": "[3,3]", "unit": "SO(3)", "frame": "R_WH", "meaning": "world-from-heading rotation"},
    "torso_rotation_heading": {"shape": "[3,3]", "unit": "SO(3)", "frame": "R_HB", "meaning": "torso orientation in the held anchor H frame"},
    "causal_h_source_code": {"shape": "[]", "unit": "enum", "frame": "task", "meaning": "0 first-cycle causal prefix; 1 previous complete cycle"},
    "causal_h_source_cycle_index": {"shape": "[]", "unit": "index", "frame": "task", "meaning": "cycle used to define H"},
    "causal_h_source_sample_count": {"shape": "[]", "unit": "count", "frame": "6ms_grid", "meaning": "anchor yaw samples used by H"},
    "mpc_diagnostics_valid": {"shape": "[]", "unit": "bool", "frame": "6ms_grid", "meaning": "QP status belongs to this anchor"},
    "mpc_success": {"shape": "[]", "unit": "bool", "frame": "6ms_grid", "meaning": "OSQP solution accepted"},
    "mpc_fallback_used": {"shape": "[]", "unit": "bool", "frame": "6ms_grid", "meaning": "MPC braking fallback used"},
    "mpc_solver_status_val": {"shape": "[]", "unit": "OSQP_status", "frame": "solver", "meaning": "numeric OSQP status"},
    "runtime_mapping_safety_fallback_used": {"shape": "[]", "unit": "bool", "frame": "2ms_execution", "meaning": "DDQ-to-torque safety rescue used"},
    "runtime_executor_flags": {"shape": "[]", "unit": "bitmask", "frame": "2ms_execution", "meaning": "C++ executor status flags"},
}


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.copy()


class FullTaskRawRecorder:
    """Append-only strict pre-step recorder used only by the T1-A entrypoint."""

    def __init__(
        self,
        *,
        protocol: FullTaskProtocol,
        clock: FullTaskClock,
        nominal_command: np.ndarray,
        heading_frame_version: str = FullTaskCausalHeadingFrame.DEFINITION_VERSION,
    ) -> None:
        self.protocol = protocol
        self.clock = clock
        self.nominal_command = _finite_vector(nominal_command, (3,), "nominal_command")
        frame_types = {
            FullTaskCausalHeadingFrame.DEFINITION_VERSION: FullTaskCausalHeadingFrame,
            FullTaskContinuousHeadingFrame.DEFINITION_VERSION: FullTaskContinuousHeadingFrame,
        }
        try:
            frame_type = frame_types[str(heading_frame_version)]
        except KeyError as error:
            raise ValueError(f"unsupported full-task heading frame: {heading_frame_version}") from error
        self.heading_frame_version = str(heading_frame_version)
        self.causal_h = frame_type(protocol)
        self._data: dict[str, list[Any]] = {name: [] for name in RAW_FIELD_DEFINITIONS}
        self._dynamic_shapes: dict[str, tuple[int, ...]] = {}

    def _dynamic_vector(self, value: Any, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite vector")
        expected = self._dynamic_shapes.setdefault(name, array.shape)
        if array.shape != expected:
            raise ValueError(f"{name} changed shape from {expected} to {array.shape}")
        return array.copy()

    def append(
        self,
        *,
        simulation_time: float,
        sample_index: int,
        planned_command: np.ndarray,
        runtime_command: np.ndarray,
        policy_update_applied: bool,
        policy_command_consumed_time: float,
        mpc_anchor: bool,
        torso_position_world: np.ndarray,
        torso_rotation_world: np.ndarray,
        torso_linear_velocity_world: np.ndarray,
        torso_angular_velocity_world: np.ndarray,
        torso_linear_acceleration_world_raw: np.ndarray,
        torso_linear_acceleration_world_used: np.ndarray,
        torso_angular_acceleration_world_raw: np.ndarray,
        torso_angular_acceleration_world_used: np.ndarray,
        lower_body_q: np.ndarray,
        lower_body_dq: np.ndarray,
        lower_body_policy_target: np.ndarray,
        right_arm_q: np.ndarray,
        right_arm_dq: np.ndarray,
        right_arm_ddq_des: np.ndarray,
        generalized_qpos: np.ndarray,
        generalized_qvel: np.ndarray,
        generalized_qacc: np.ndarray,
        actuator_ctrl: np.ndarray,
        heading_state: Any,
        mpc_diagnostics: dict[str, Any] | None,
        runtime_mapping_safety_fallback_used: bool,
        runtime_executor_flags: int,
    ) -> None:
        index = int(sample_index)
        if index != len(self._data["task_time"]):
            raise ValueError("strict pre-step samples must be contiguous from index zero")
        task_time = self.clock.observe(simulation_time)
        expected_time = self.protocol.sample_time(index)
        if not np.isclose(task_time, expected_time, atol=1e-10, rtol=0.0):
            raise ValueError("task time is not aligned with the 2 ms sample index")
        expected_anchor = self.protocol.is_mpc_anchor_sample(index)
        if bool(mpc_anchor) != expected_anchor:
            raise ValueError("MPC anchor flag disagrees with the shared 6 ms grid")

        planned = _finite_vector(planned_command, (3,), "planned_command")
        expected_command: DirectStepCommand = direct_step_planned_command(
            task_time, self.nominal_command, self.protocol
        )
        if not np.allclose(planned, expected_command.planned_command, atol=1e-8, rtol=0.0):
            raise ValueError("planned command disagrees with direct-step protocol")
        runtime = _finite_vector(runtime_command, (3,), "runtime_command")
        if not np.allclose(runtime[:2], planned[:2], atol=1e-8, rtol=0.0):
            raise ValueError("heading control may only change runtime wz")

        rotation = _finite_vector(torso_rotation_world, (3, 3), "torso_rotation_world")
        if not bool(is_valid_rotation_batch(rotation)):
            raise ValueError("torso_rotation_world is not a proper SO(3) rotation")
        if expected_anchor:
            h_state = self.causal_h.update(task_time, rotation)
        else:
            h_state = self.causal_h.last_state
            if h_state is None:
                raise RuntimeError("first raw sample must also be the first MPC anchor")
        rotation_heading_body = self.causal_h.rotation_heading_body(rotation)

        measured_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        cycle_index = int(np.floor(task_time / self.protocol.gait_period + 1e-12))
        gait_phase = float((task_time / self.protocol.gait_period) % 1.0)
        anchor_index = (
            self.protocol.anchor_index_from_sample(index) if expected_anchor else -1
        )
        diagnostics_valid = bool(expected_anchor and mpc_diagnostics is not None)
        mpc_success = bool(mpc_diagnostics.get("success", False)) if diagnostics_valid else False
        mpc_fallback = bool(mpc_diagnostics.get("fallback_used", False)) if diagnostics_valid else False
        solver_status_val = int(mpc_diagnostics.get("solver_status_val", 0)) if diagnostics_valid else 0

        sample = {
            "task_epoch": self.clock.epoch_index,
            "task_time": task_time,
            "simulation_time": float(simulation_time),
            "physics_sample_index": index,
            "planned_command": planned,
            "runtime_command": runtime,
            "policy_update_applied": bool(policy_update_applied),
            "policy_command_consumed_time": float(policy_command_consumed_time),
            "mpc_anchor": expected_anchor,
            "mpc_anchor_index": anchor_index,
            "gait_phase": gait_phase,
            "gait_cycle_index": cycle_index,
            "torso_position_world": _finite_vector(torso_position_world, (3,), "torso_position_world"),
            "torso_rotation_world": rotation,
            "torso_linear_velocity_world": _finite_vector(torso_linear_velocity_world, (3,), "torso_linear_velocity_world"),
            "torso_angular_velocity_world": _finite_vector(torso_angular_velocity_world, (3,), "torso_angular_velocity_world"),
            "torso_linear_acceleration_world_raw": _finite_vector(torso_linear_acceleration_world_raw, (3,), "torso_linear_acceleration_world_raw"),
            "torso_linear_acceleration_world_used": _finite_vector(torso_linear_acceleration_world_used, (3,), "torso_linear_acceleration_world_used"),
            "torso_angular_acceleration_world_raw": _finite_vector(torso_angular_acceleration_world_raw, (3,), "torso_angular_acceleration_world_raw"),
            "torso_angular_acceleration_world_used": _finite_vector(torso_angular_acceleration_world_used, (3,), "torso_angular_acceleration_world_used"),
            "lower_body_q": _finite_vector(lower_body_q, (12,), "lower_body_q"),
            "lower_body_dq": _finite_vector(lower_body_dq, (12,), "lower_body_dq"),
            "lower_body_policy_target": _finite_vector(lower_body_policy_target, (12,), "lower_body_policy_target"),
            "right_arm_q": _finite_vector(right_arm_q, (5,), "right_arm_q"),
            "right_arm_dq": _finite_vector(right_arm_dq, (5,), "right_arm_dq"),
            "right_arm_ddq_des": _finite_vector(right_arm_ddq_des, (5,), "right_arm_ddq_des"),
            "generalized_qpos": self._dynamic_vector(generalized_qpos, "generalized_qpos"),
            "generalized_qvel": self._dynamic_vector(generalized_qvel, "generalized_qvel"),
            "generalized_qacc": self._dynamic_vector(generalized_qacc, "generalized_qacc"),
            "actuator_ctrl": self._dynamic_vector(actuator_ctrl, "actuator_ctrl"),
            "heading_reference_world": float(heading_state.reference_world),
            "heading_measurement_world": measured_yaw,
            "heading_yaw_filtered": float(heading_state.yaw_filtered),
            "heading_yaw_error": float(heading_state.yaw_error),
            "heading_yaw_rate_correction": float(heading_state.yaw_rate_correction),
            "heading_yaw_rate_command": float(runtime[2]),
            "heading_command_saturated": bool(heading_state.command_saturated),
            "causal_h_yaw_world": float(h_state.yaw_world),
            "causal_h_rotation_world": h_state.rotation_world_heading.copy(),
            "torso_rotation_heading": rotation_heading_body,
            "causal_h_source_code": self.causal_h.SOURCE_CODES[h_state.source],
            "causal_h_source_cycle_index": int(h_state.source_cycle_index),
            "causal_h_source_sample_count": int(h_state.source_sample_count),
            "mpc_diagnostics_valid": diagnostics_valid,
            "mpc_success": mpc_success,
            "mpc_fallback_used": mpc_fallback,
            "mpc_solver_status_val": solver_status_val,
            "runtime_mapping_safety_fallback_used": bool(runtime_mapping_safety_fallback_used),
            "runtime_executor_flags": int(runtime_executor_flags),
        }
        for name, value in sample.items():
            self._data[name].append(value)

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays = {name: np.asarray(values) for name, values in self._data.items()}
        arrays.update(
            {
                "raw_schema_version": np.array(self.protocol.raw_schema_version),
                "protocol_name": np.array(self.protocol.protocol_name),
                "protocol_version": np.array(self.protocol.protocol_version),
                "sample_timing": np.array(PRE_STEP_SAMPLE_DEFINITION),
                "task_epoch_label": np.array(self.clock.epoch_label),
                "task_epoch_origin_simulation_time": np.array(
                    self.clock.origin_simulation_time, dtype=np.float64
                ),
                "physics_dt": np.array(self.protocol.physics_dt, dtype=np.float64),
                "mpc_dt": np.array(self.protocol.mpc_dt, dtype=np.float64),
                "policy_dt": np.array(self.protocol.policy_dt, dtype=np.float64),
                "gait_period": np.array(self.protocol.gait_period, dtype=np.float64),
                "stop_time": np.array(self.protocol.stop_time, dtype=np.float64),
                "headline_end": np.array(self.protocol.headline_end, dtype=np.float64),
                "record_end": np.array(self.protocol.record_end, dtype=np.float64),
                "horizon": np.array(self.protocol.horizon, dtype=np.int64),
                "nominal_command": self.nominal_command.copy(),
                "pre_step_event_order": np.asarray(PRE_STEP_EVENT_ORDER),
                "causal_h_source_names": np.asarray(
                    tuple(self.causal_h.SOURCE_CODES)
                ),
                "heading_frame_version": np.array(self.heading_frame_version),
            }
        )
        return arrays


def validate_full_task_raw(
    raw: dict[str, np.ndarray],
    protocol: FullTaskProtocol,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    missing = sorted(set(RAW_FIELD_DEFINITIONS) - set(raw))
    if missing:
        raise ValueError(f"raw schema is missing fields: {missing}")
    task_time = np.asarray(raw["task_time"], dtype=np.float64)
    sample_index = np.asarray(raw["physics_sample_index"], dtype=np.int64)
    count = len(task_time)
    if count == 0:
        raise ValueError("raw episode is empty")
    np.testing.assert_array_equal(sample_index, np.arange(count, dtype=np.int64))
    np.testing.assert_allclose(
        task_time,
        sample_index * protocol.physics_dt,
        atol=1e-10,
        rtol=0.0,
    )
    simulation_time = np.asarray(raw["simulation_time"], dtype=np.float64)
    origin = float(np.asarray(raw["task_epoch_origin_simulation_time"]))
    np.testing.assert_allclose(simulation_time - task_time, origin, atol=1e-10, rtol=0.0)
    if str(np.asarray(raw["sample_timing"])) != PRE_STEP_SAMPLE_DEFINITION:
        raise ValueError("raw sample timing definition is not strict pre-step")

    expected_anchor = sample_index % protocol.mpc_stride == 0
    np.testing.assert_array_equal(np.asarray(raw["mpc_anchor"], dtype=bool), expected_anchor)
    expected_policy = (sample_index > 0) & (sample_index % protocol.policy_stride == 0)
    planned = np.asarray(raw["planned_command"], dtype=np.float64)
    runtime = np.asarray(raw["runtime_command"], dtype=np.float64)
    nominal = np.asarray(raw["nominal_command"], dtype=np.float64)
    before_stop = task_time < protocol.stop_time - 1e-12
    after_stop = ~before_stop
    np.testing.assert_allclose(
        planned[before_stop],
        np.broadcast_to(nominal, planned[before_stop].shape),
        atol=1e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        planned[after_stop, :2],
        np.zeros_like(planned[after_stop, :2]),
        atol=1e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        planned[after_stop, 2],
        np.full_like(planned[after_stop, 2], nominal[2]),
        atol=1e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(runtime[:, :2], planned[:, :2], atol=1e-8, rtol=0.0)

    if not np.all(is_valid_rotation_batch(raw["torso_rotation_world"])):
        raise ValueError("raw torso rotations contain invalid SO(3) values")
    if not np.all(is_valid_rotation_batch(raw["causal_h_rotation_world"])):
        raise ValueError("raw causal-H rotations contain invalid SO(3) values")
    if not np.all(is_valid_rotation_batch(raw["torso_rotation_heading"])):
        raise ValueError("raw H-relative rotations contain invalid SO(3) values")

    finite_fields = tuple(
        name
        for name in RAW_FIELD_DEFINITIONS
        if name
        not in {
            "policy_command_consumed_time",
            "heading_reference_world",
            "heading_yaw_filtered",
            "heading_yaw_error",
        }
    )
    nonfinite_count = int(
        sum(np.size(raw[name]) - np.count_nonzero(np.isfinite(raw[name])) for name in finite_fields)
    )
    if nonfinite_count:
        raise ValueError(f"raw episode contains {nonfinite_count} unexpected NaN/Inf values")
    recorded_policy = np.asarray(raw["policy_update_applied"], dtype=bool)
    np.testing.assert_array_equal(recorded_policy, expected_policy)
    first_policy_index = int(np.flatnonzero(expected_policy)[0])
    pre_policy = np.arange(count) < first_policy_index
    for name in (
        "policy_command_consumed_time",
        "heading_reference_world",
        "heading_yaw_filtered",
        "heading_yaw_error",
    ):
        values = np.asarray(raw[name], dtype=np.float64)
        if not np.all(np.isnan(values[pre_policy])) or not np.all(np.isfinite(values[~pre_policy])):
            raise ValueError(f"{name} initialization boundary is inconsistent")
    np.testing.assert_allclose(
        np.asarray(raw["gait_phase"], dtype=np.float64),
        (task_time / protocol.gait_period) % 1.0,
        atol=1e-12,
        rtol=0.0,
    )

    headline_anchor = expected_anchor & (task_time < protocol.headline_end - 1e-12)
    headline_times = task_time[headline_anchor]
    tail_indices = protocol.future_window_sample_indices(
        protocol.headline_anchor_count - 1
    )
    tail_complete = bool(
        int(tail_indices["node"][-1]) < count
        and np.isclose(
            task_time[int(tail_indices["node"][-1])],
            protocol.last_horizon_node_time,
            atol=1e-10,
            rtol=0.0,
        )
    )
    report = {
        "raw_sample_count": count,
        "first_raw_time": float(task_time[0]),
        "last_raw_time": float(task_time[-1]),
        "covered_interval_end": float(task_time[-1] + protocol.physics_dt),
        "headline_anchor_count": int(np.count_nonzero(headline_anchor)),
        "recorded_anchor_count": int(np.count_nonzero(expected_anchor)),
        "last_headline_anchor": float(headline_times[-1]),
        "last_horizon_node": float(protocol.last_horizon_node_time),
        "tail_complete": tail_complete,
        "first_policy_command_consumed_time": float(task_time[first_policy_index]),
        "unexpected_nonfinite_count": nonfinite_count,
        "rotation_valid": True,
        "strict_pre_step": True,
    }
    if require_complete:
        expected = {
            "raw_sample_count": protocol.physics_steps,
            "headline_anchor_count": protocol.headline_anchor_count,
            "recorded_anchor_count": protocol.recorded_anchor_count,
        }
        for name, value in expected.items():
            if report[name] != value:
                raise ValueError(f"{name}={report[name]} but expected {value}")
        if not np.isclose(report["covered_interval_end"], protocol.record_end, atol=1e-10):
            raise ValueError("raw intervals do not cover record_end")
        if not np.isclose(report["last_headline_anchor"], protocol.last_headline_anchor_time, atol=1e-10):
            raise ValueError("last headline anchor is incorrect")
        if not tail_complete:
            raise ValueError("last 54 ms horizon is not covered by raw data")
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_asset(path: Path, repo_dir: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        relative = resolved.relative_to(repo_dir.resolve())
        portable = relative.as_posix()
    except ValueError:
        portable = str(resolved)
    return {
        "path": portable,
        "absolute_path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def current_git_state(repo_dir: Path) -> dict[str, str]:
    def read(*args: str) -> str:
        return subprocess.check_output(
            ("git", *args), cwd=repo_dir, text=True
        ).strip()

    return {
        "head": read("rev-parse", "HEAD"),
        "branch": read("rev-parse", "--abbrev-ref", "HEAD"),
        "status_short": read("status", "--short"),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def compute_smoke_summary(
    raw: dict[str, np.ndarray],
    protocol: FullTaskProtocol,
    validation: dict[str, Any],
    *,
    heading_enabled: bool,
) -> dict[str, Any]:
    time = np.asarray(raw["task_time"], dtype=np.float64)
    position = np.asarray(raw["torso_position_world"], dtype=np.float64)
    rpy = rotation_matrix_to_rpy(raw["torso_rotation_world"])
    headline_end_index = int(round(protocol.headline_end / protocol.physics_dt))
    headline_position = position[: headline_end_index + 1, :2]
    xy_displacement = float(np.linalg.norm(headline_position[-1] - headline_position[0]))
    xy_arc_length = float(
        np.sum(np.linalg.norm(np.diff(headline_position, axis=0), axis=1))
    )
    valid_heading = np.isfinite(raw["heading_reference_world"])
    heading_error = np.asarray(raw["heading_yaw_error"], dtype=np.float64)
    raw_heading_difference = np.arctan2(
        np.sin(
            np.asarray(raw["heading_measurement_world"])[valid_heading]
            - np.asarray(raw["heading_reference_world"])[valid_heading]
        ),
        np.cos(
            np.asarray(raw["heading_measurement_world"])[valid_heading]
            - np.asarray(raw["heading_reference_world"])[valid_heading]
        ),
    )
    max_filtered_heading_error = float(np.max(np.abs(heading_error[valid_heading])))
    max_raw_yaw_deviation = float(np.max(np.abs(raw_heading_difference)))
    max_abs_roll = float(np.max(np.abs(rpy[:, 0])))
    max_abs_pitch = float(np.max(np.abs(rpy[:, 1])))
    min_height = float(np.min(position[:, 2]))
    max_height = float(np.max(position[:, 2]))
    fallen = bool(
        min_height < FALL_MIN_TORSO_HEIGHT_M
        or max(max_abs_roll, max_abs_pitch) > FALL_MAX_ABS_ROLL_PITCH_RAD
    )

    mpc_mask = np.asarray(raw["mpc_diagnostics_valid"], dtype=bool)
    mpc_success = np.asarray(raw["mpc_success"], dtype=bool)
    qp_failure_count = int(np.count_nonzero(mpc_mask & ~mpc_success))
    qp_fallback_count = int(
        np.count_nonzero(mpc_mask & np.asarray(raw["mpc_fallback_used"], dtype=bool))
    )
    mapping_fallback_count = int(
        np.count_nonzero(raw["runtime_mapping_safety_fallback_used"])
    )

    def command_at(value: float) -> dict[str, list[float]]:
        index = int(round(value / protocol.physics_dt))
        return {
            "planned": np.asarray(raw["planned_command"])[index].tolist(),
            "runtime": np.asarray(raw["runtime_command"])[index].tolist(),
        }

    command_boundary = {
        "t_6_398": command_at(6.398),
        "t_6_400": command_at(6.400),
        "t_6_402": command_at(6.402),
        "anchor_t_6_396": command_at(6.396),
    }
    first_policy_time = float(validation["first_policy_command_consumed_time"])
    direct_step_effective = bool(
        np.allclose(command_boundary["t_6_398"]["planned"][:2], raw["nominal_command"][:2])
        and np.allclose(command_boundary["t_6_400"]["planned"][:2], 0.0)
        and np.allclose(command_boundary["t_6_402"]["planned"][:2], 0.0)
    )
    distance_within_gate = 2.8 <= xy_displacement <= 3.6
    nominal_mapping_path_passed = mapping_fallback_count == 0
    smoke_passed = bool(
        heading_enabled
        and direct_step_effective
        and validation["strict_pre_step"]
        and validation["tail_complete"]
        and validation["unexpected_nonfinite_count"] == 0
        and not fallen
        and qp_failure_count == 0
        and distance_within_gate
    )
    warnings = (
        ["MAPPING_SAFETY_FALLBACK_USED"]
        if not nominal_mapping_path_passed
        else []
    )
    return {
        "status": "PASS" if smoke_passed else "FAIL",
        "smoke_passed": smoke_passed,
        "nominal_mapping_path_passed": nominal_mapping_path_passed,
        "warnings": warnings,
        "xy_start_world_m": position[0, :2].tolist(),
        "xy_at_headline_end_world_m": position[headline_end_index, :2].tolist(),
        "xy_displacement_m": xy_displacement,
        "xy_arc_length_m": xy_arc_length,
        "expected_displacement_m": 3.2,
        "displacement_difference_m": xy_displacement - 3.2,
        "distance_gate_m": [2.8, 3.6],
        "distance_within_gate": distance_within_gate,
        "max_abs_filtered_heading_error_rad": max_filtered_heading_error,
        "max_abs_raw_yaw_deviation_rad": max_raw_yaw_deviation,
        "max_abs_roll_rad": max_abs_roll,
        "max_abs_pitch_rad": max_abs_pitch,
        "torso_height_min_m": min_height,
        "torso_height_max_m": max_height,
        "fall_definition": {
            "min_torso_height_m": FALL_MIN_TORSO_HEIGHT_M,
            "max_abs_roll_pitch_rad": FALL_MAX_ABS_ROLL_PITCH_RAD,
        },
        "fallen": fallen,
        "nan_inf_count": int(validation["unexpected_nonfinite_count"]),
        "qp_update_count": int(np.count_nonzero(mpc_mask)),
        "qp_failure_count": qp_failure_count,
        "qp_fallback_count": qp_fallback_count,
        "runtime_mapping_safety_fallback_count": mapping_fallback_count,
        "runtime_executor_nonzero_flag_count": int(
            np.count_nonzero(raw["runtime_executor_flags"])
        ),
        "direct_step_effective": direct_step_effective,
        "heading_enabled": bool(heading_enabled),
        "command_boundary": command_boundary,
        "first_frame_command_visible_time_s": float(time[0]),
        "first_policy_command_consumed_time_s": first_policy_time,
        **validation,
    }


def save_smoke_plots(
    raw: dict[str, np.ndarray], protocol: FullTaskProtocol, output_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    time = np.asarray(raw["task_time"], dtype=np.float64)
    position = np.asarray(raw["torso_position_world"], dtype=np.float64)
    planned = np.asarray(raw["planned_command"], dtype=np.float64)
    runtime = np.asarray(raw["runtime_command"], dtype=np.float64)
    rpy = rotation_matrix_to_rpy(raw["torso_rotation_world"])
    paths: list[Path] = []
    linewidth = 0.8

    path = output_dir / "full_task_xy_trajectory.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(position[:, 0], position[:, 1], lw=linewidth)
    ax.scatter(position[0, 0], position[0, 1], label="t=0", zorder=3)
    end_index = int(round(protocol.headline_end / protocol.physics_dt))
    ax.scatter(position[end_index, 0], position[end_index, 1], label="t=8.0", zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "full_task_planned_runtime_commands.png"
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for column, (axis, label, unit) in enumerate(
        zip(axes, ("vx", "vy", "wz"), ("m/s", "m/s", "rad/s"))
    ):
        axis.plot(time, planned[:, column], label="planned", lw=linewidth)
        axis.plot(time, runtime[:, column], label="runtime", lw=linewidth, alpha=0.8)
        axis.axvline(protocol.stop_time, color="red", ls="--", lw=linewidth)
        axis.set_ylabel(f"{label} [{unit}]")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("task time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "full_task_torso_rpy.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    for column, label in enumerate(("roll", "pitch", "yaw")):
        ax.plot(time, rpy[:, column], label=label, lw=linewidth)
    ax.axvline(protocol.stop_time, color="red", ls="--", lw=linewidth)
    ax.set_xlabel("task time [s]")
    ax.set_ylabel("angle [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "full_task_event_grid.png"
    fig, ax = plt.subplots(figsize=(12, 4))
    physics = time
    mpc = time[np.asarray(raw["mpc_anchor"], dtype=bool)]
    policy = time[np.asarray(raw["policy_update_applied"], dtype=bool)]
    ax.scatter(physics, np.zeros_like(physics), s=1, label="2 ms pre-step sample")
    ax.scatter(mpc, np.ones_like(mpc), s=3, label="6 ms MPC anchor")
    ax.scatter(policy, np.full_like(policy, 2.0), s=8, label="20 ms policy update")
    ax.axvline(protocol.stop_time, color="red", ls="--", lw=linewidth, label="direct stop")
    ax.axvline(protocol.headline_end, color="black", ls=":", lw=linewidth, label="headline end")
    ax.set_yticks((0, 1, 2), ("physics", "MPC", "policy"))
    ax.set_xlabel("task time [s]")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "full_task_tail_coverage.png"
    fig, ax = plt.subplots(figsize=(12, 4))
    tail = time >= 7.95
    raw_tail = time[tail]
    anchor_tail = time[tail & np.asarray(raw["mpc_anchor"], dtype=bool)]
    last_window = protocol.future_window_sample_indices(
        protocol.headline_anchor_count - 1
    )
    node_times = np.asarray(raw["task_time"])[last_window["node"]]
    ax.scatter(raw_tail, np.zeros_like(raw_tail), s=8, label="2 ms raw")
    ax.scatter(anchor_tail, np.ones_like(anchor_tail), s=16, label="6 ms anchors")
    ax.scatter(node_times, np.full_like(node_times, 2.0), marker="x", s=35, label="last horizon nodes")
    ax.axvline(protocol.last_headline_anchor_time, color="tab:green", ls="--", lw=linewidth, label="last headline anchor")
    ax.axvline(protocol.headline_end, color="black", ls=":", lw=linewidth, label="headline end")
    ax.axvline(protocol.last_horizon_node_time, color="red", ls="--", lw=linewidth, label="last horizon node")
    ax.axvline(protocol.record_end, color="tab:purple", ls=":", lw=linewidth, label="covered interval end")
    ax.set_xlim(7.95, protocol.record_end + 0.002)
    ax.set_yticks((0, 1, 2), ("raw", "anchor", "node"))
    ax.set_xlabel("task time [s]")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def save_full_task_smoke_artifacts(
    *,
    recorder: FullTaskRawRecorder,
    run_dir: Path,
    repo_dir: Path,
    config_path: Path,
    policy_path: Path,
    xml_path: Path,
    legacy_template_path: Path,
    predictor_metadata: dict[str, Any],
    control_chain: dict[str, Any],
    initial_lower_q_offset: np.ndarray,
    initial_lower_dq: np.ndarray,
    heading_enabled: bool,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    repo_dir = repo_dir.resolve()
    raw = recorder.to_arrays()
    validation = validate_full_task_raw(
        raw, recorder.protocol, require_complete=True
    )
    summary = compute_smoke_summary(
        raw, recorder.protocol, validation, heading_enabled=heading_enabled
    )

    raw_path = run_dir / "full_task_nominal_raw.npz"
    np.savez_compressed(raw_path, **raw)
    plots = save_smoke_plots(raw, recorder.protocol, run_dir)
    summary_path = run_dir / "full_task_smoke_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_value(summary), stream, indent=2, ensure_ascii=False)

    protocol = recorder.protocol
    manifest = {
        "schema_version": protocol.raw_schema_version,
        "protocol": {
            "name": protocol.protocol_name,
            "version": protocol.protocol_version,
            "task_epoch_label": recorder.clock.epoch_label,
            "task_epoch_index": recorder.clock.epoch_index,
            "task_epoch_origin_simulation_time": recorder.clock.origin_simulation_time,
            "physics_dt": protocol.physics_dt,
            "mpc_dt": protocol.mpc_dt,
            "policy_dt": protocol.policy_dt,
            "gait_period": protocol.gait_period,
            "stop_time": protocol.stop_time,
            "headline": [0.0, protocol.headline_end],
            "headline_interval": "[0.0,8.0)",
            "record_end": protocol.record_end,
            "horizon": protocol.horizon,
            "horizon_duration": protocol.horizon * protocol.mpc_dt,
            "command": "nominal vx/vy until 6.4 s, then direct zero translation; nominal wz remains heading feedforward",
            "out_of_range": "raise ValueError; never wrap to task start",
            "reset": "increments task epoch and restarts task time at zero",
        },
        "time_semantics": {
            "sample_timing": PRE_STEP_SAMPLE_DEFINITION,
            "event_order": PRE_STEP_EVENT_ORDER,
            "task_t0": "planned command is readable by recorder/predictor; no claim of physical motion",
            "first_policy_consumption": validation["first_policy_command_consumed_time"],
            "anchor_grid": "task_time = anchor_index * 0.006",
            "headline_anchor_count": protocol.headline_anchor_count,
            "last_headline_anchor": protocol.last_headline_anchor_time,
            "last_horizon_node": protocol.last_horizon_node_time,
        },
        "causal_h": {
            "version": recorder.heading_frame_version,
            "update_grid": "6ms MPC anchors only",
            "definition": (
                "causal prefix for t<0.8s; trailing causal 0.8s circular mean for "
                "0.8<=t<6.4s; freeze the final pre-stop H for t>=6.4s"
                if recorder.heading_frame_version
                == FullTaskContinuousHeadingFrame.DEFINITION_VERSION
                else "first-cycle causal prefix then previous complete 0.8s cycle mean"
            ),
            "within_horizon": "H fixed at anchor; future yaw is never read",
            "rotation": "R_WH is an analytic yaw SO(3) rotation; R_HB=R_WH.T@R_WB",
            "source_codes": recorder.causal_h.SOURCE_CODES,
            "additional_low_pass_filter": "none",
        },
        "episode": {
            "identity": "nominal",
            "dataset_role": "nominal_smoke_only_not_a_template",
            "future_pair_id": None,
            "initial_lower_q_offset_rad": np.asarray(initial_lower_q_offset).tolist(),
            "initial_lower_dq_rad_s": np.asarray(initial_lower_dq).tolist(),
        },
        "git": current_git_state(repo_dir),
        "control_chain": _json_value(control_chain),
        "predictor": _json_value(predictor_metadata),
        "assets": {
            "config": portable_asset(config_path, repo_dir),
            "policy": portable_asset(policy_path, repo_dir),
            "simulation_and_pinocchio_model": portable_asset(xml_path, repo_dir),
            "predictor_asset": portable_asset(legacy_template_path, repo_dir),
            "main_control_source": portable_asset(repo_dir / "main_sim.py", repo_dir),
            "protocol_source": portable_asset(
                repo_dir / "disturbance_template/full_task_protocol.py", repo_dir
            ),
            "recording_source": portable_asset(
                repo_dir / "disturbance_template/full_task_recording.py", repo_dir
            ),
            "raw": portable_asset(raw_path, repo_dir),
            "smoke_summary": portable_asset(summary_path, repo_dir),
            "plots": [portable_asset(path, repo_dir) for path in plots],
        },
        "raw_schema": RAW_FIELD_DEFINITIONS,
        "validation": summary,
        "scope": {
            "final_full_task_template_generated": False,
            "batch_11_plus_4_run": False,
            "full_task_online_predictor_added": (
                predictor_metadata.get("predictor_type")
                == "full_task_template"
            ),
            "t2_started": True,
            "n1_n2_started": False,
        },
    }
    if predictor_metadata.get("predictor_type") == "template":
        manifest["assets"]["legacy_phase_template"] = portable_asset(
            legacy_template_path, repo_dir
        )
    manifest_path = run_dir / "full_task_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_value(manifest), stream, indent=2, ensure_ascii=False)
    return {
        "raw_path": raw_path,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "plot_paths": plots,
        "summary": summary,
    }
