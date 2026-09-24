#!/usr/bin/env python3
"""Pure control/geometry core for the isolated G1 Arm SDK PID field tool.

This module has no Unitree SDK import and never opens DDS.  It is shared by the
offline tests, the opt-in device runner, and the offline result analyser.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

from endpoint_pose import EndpointModel, rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARM_MOTOR_INDICES = (15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 12, 13, 14)
WEIGHT_MOTOR_INDEX = 29
VALID_ARM_SLOTS = tuple(range(11))

ENTRY_END_S = 3.0
WALK_START_S = 5.0
WALK_STOP_S = 15.0
RELEASE_START_S = 18.0
END_S = 21.0
FORWARD_SPEED_M_S = 0.5
WEIGHT_RELEASE_DURATION_S = 3.0
CONTROL_PERIOD_S = 0.006
PID_REFERENCE_PERIOD_S = 0.020
WEIGHT_RELEASE_NOMINAL_PERIOD_S = CONTROL_PERIOD_S


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def yaw_from_quaternion(quaternion_wxyz) -> float:
    matrix = rotation(quaternion_wxyz)
    return math.atan2(matrix[1, 0], matrix[0, 0])


def vertical_angular_rate(rpy_rad, gyro_imu_rad_s) -> float:
    """Project local IMU angular velocity onto navigation-frame vertical."""
    roll, pitch = np.asarray(rpy_rad, dtype=float)[:2]
    gx, gy, gz = np.asarray(gyro_imu_rad_s, dtype=float)
    return float(
        -math.sin(pitch) * gx
        + math.cos(pitch) * math.sin(roll) * gy
        + math.cos(pitch) * math.cos(roll) * gz
    )


class FixedH0Heading:
    """Fixed run heading plus a causal one-second heading-feedback filter."""

    reference_start_s = 3.0
    reference_end_s = 5.0
    minimum_reference_span_s = 1.0
    control_window_s = 1.0
    kp = 1.0
    kd = 0.1
    max_rate = 0.25

    def __init__(self):
        self._samples = deque()
        self._reference_sine = 0.0
        self._reference_cosine = 0.0
        self._reference_times = []
        self.reference = None

    def observe(self, received_ns: int, task_s: float, yaw: float, vertical_rate: float):
        values = (task_s, yaw, vertical_rate)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("heading observation must be finite")
        received_ns = int(received_ns)
        if self._samples and received_ns <= self._samples[-1][0]:
            return
        self._samples.append((received_ns, float(yaw), float(vertical_rate)))
        window_ns = int(self.control_window_s * 1e9)
        while len(self._samples) > 1 and received_ns - self._samples[0][0] > window_ns:
            self._samples.popleft()
        if (
            self.reference is None
            and self.reference_start_s <= task_s < self.reference_end_s
        ):
            self._reference_sine += math.sin(yaw)
            self._reference_cosine += math.cos(yaw)
            self._reference_times.append(float(task_s))

    def freeze(self) -> float:
        if self.reference is not None:
            return self.reference
        if len(self._reference_times) < 2:
            raise RuntimeError("H0 needs pre-walk torso-yaw samples")
        span = self._reference_times[-1] - self._reference_times[0]
        if span < self.minimum_reference_span_s:
            raise RuntimeError("H0 needs at least one second of pre-walk yaw samples")
        self.reference = math.atan2(self._reference_sine, self._reference_cosine)
        return self.reference

    def current(self) -> dict:
        if not self._samples:
            raise RuntimeError("heading needs torso IMU samples")
        filtered_yaw = math.atan2(
            sum(math.sin(sample[1]) for sample in self._samples),
            sum(math.cos(sample[1]) for sample in self._samples),
        )
        filtered_rate = float(np.mean([sample[2] for sample in self._samples]))
        frozen = self.reference is not None
        relative = wrap_angle(filtered_yaw - self.reference) if frozen else 0.0
        error = -relative if frozen else 0.0
        correction = (
            float(np.clip(self.kp * error - self.kd * filtered_rate,
                          -self.max_rate, self.max_rate))
            if frozen else 0.0
        )
        first = self._reference_times[0] if self._reference_times else None
        last = self._reference_times[-1] if self._reference_times else None
        return {
            "reference_frozen": frozen,
            "reference_samples": len(self._reference_times),
            "reference_first_s": first,
            "reference_last_s": last,
            "reference_span_s": 0.0 if first is None else last - first,
            "reference_rad": 0.0 if self.reference is None else self.reference,
            "filtered_yaw_rad": filtered_yaw,
            "relative_yaw_rad": relative,
            "error_rad": error,
            "filtered_vertical_rate_rad_s": filtered_rate,
            "correction_rad_s": correction,
        }


@dataclass(frozen=True)
class PidParameters:
    kp_pose: np.ndarray
    kd_pose: np.ndarray
    ki_pose: np.ndarray
    posture_gain: np.ndarray
    finite_diff_eps: float
    damping: float
    integral_limit: float
    max_dq: float
    de_g_alpha: float
    q_offset_limit_rad: np.ndarray
    hardware_max_dq: float
    hardware_max_ddq: float

    @classmethod
    def from_mapping(cls, values: Mapping):
        def vector(name, count):
            result = np.asarray(values[name], dtype=float)
            if result.shape != (count,) or not np.isfinite(result).all():
                raise ValueError(f"{name} must contain {count} finite values")
            return result

        result = cls(
            kp_pose=vector("pid_kp_pose", 2),
            kd_pose=vector("pid_kd_pose", 2),
            ki_pose=vector("pid_ki_pose", 2),
            posture_gain=vector("pid_posture_gain", 5),
            finite_diff_eps=float(values["pid_finite_diff_eps"]),
            damping=float(values["pid_damping"]),
            integral_limit=float(values["pid_integral_limit"]),
            max_dq=float(values["pid_max_dq"]),
            de_g_alpha=float(values["pid_de_g_alpha"]),
            q_offset_limit_rad=np.deg2rad(vector("pid_q_offset_limit_deg", 5)),
            hardware_max_dq=float(values["hardware_pid_max_dq"]),
            hardware_max_ddq=float(values["hardware_pid_max_ddq"]),
        )
        scalars = (
            result.finite_diff_eps,
            result.damping,
            result.integral_limit,
            result.max_dq,
            result.de_g_alpha,
            result.hardware_max_dq,
            result.hardware_max_ddq,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in scalars):
            raise ValueError("PID scalar parameters must be finite and non-negative")
        if result.finite_diff_eps <= 0.0 or result.damping <= 0.0 or result.max_dq <= 0.0:
            raise ValueError("PID finite-difference, damping and max_dq must be positive")
        if result.hardware_max_dq <= 0.0 or result.hardware_max_ddq <= 0.0:
            raise ValueError("hardware PID velocity and acceleration limits must be positive")
        if result.hardware_max_dq > result.max_dq:
            raise ValueError("hardware PID velocity limit cannot exceed raw PID max_dq")
        if not 0.0 <= result.de_g_alpha <= 1.0:
            raise ValueError("pid_de_g_alpha must be in [0,1]")
        if np.any(result.q_offset_limit_rad <= 0.0) or np.any(
            result.q_offset_limit_rad > np.deg2rad(5.0) + 1e-12
        ):
            raise ValueError("PID q-reference offsets must be in (0,5] degrees")
        return result


class RightArmGravityHelper:
    """MuJoCo FK helper using the same bottle-center site as simulation."""

    gravity_h0 = np.array([0.0, 0.0, -9.81], dtype=float)

    def __init__(self, model: EndpointModel, arm_slots, yaw0_rad: float):
        self.model = model
        self.arm_slots = np.asarray(arm_slots, dtype=float).copy()
        if self.arm_slots.shape != (13,):
            raise ValueError("arm_slots must have shape (13,)")
        self.yaw0_rad = float(yaw0_rad)

    def compute_gravity_error(self, right_q, world_from_body):
        slots = self.arm_slots.copy()
        slots[5:10] = np.asarray(right_q, dtype=float)
        _, body_from_endpoint = self.model.relative(slots)["right"]
        c, s = math.cos(self.yaw0_rad), math.sin(self.yaw0_rad)
        h0_from_world = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        h0_from_endpoint = h0_from_world @ np.asarray(world_from_body) @ body_from_endpoint
        return (h0_from_endpoint.T @ self.gravity_h0)[:2]

    def compute_gravity_error_and_jacobian(self, right_q, world_from_body):
        slots = self.arm_slots.copy()
        slots[5:10] = right_q
        return self.model.right_gravity_error_and_jacobian(slots, world_from_body)


class RightArmHardwarePid:
    """Simulation PID reused with hardware FK and a reviewed q-reference box."""

    def __init__(self, nominal_right_q, parameters: PidParameters,
                 model: EndpointModel | None = None, control_dt: float = CONTROL_PERIOD_S):
        from arm_pid import ArmPIDPolicy

        self.nominal = np.asarray(nominal_right_q, dtype=float).copy()
        if self.nominal.shape != (5,) or not np.isfinite(self.nominal).all():
            raise ValueError("nominal_right_q must be five finite angles")
        self.parameters = parameters
        self.control_dt = float(control_dt)
        if not math.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be positive and finite")
        self.model = EndpointModel() if model is None else model
        self.policy = ArmPIDPolicy(
            default_q=self.nominal,
            kp_pose=parameters.kp_pose,
            kd_pose=parameters.kd_pose,
            ki_pose=parameters.ki_pose,
            posture_gain=parameters.posture_gain,
            control_dt=control_dt,
            finite_diff_eps=parameters.finite_diff_eps,
            damping=parameters.damping,
            integral_limit=parameters.integral_limit,
            max_dq=parameters.max_dq,
            de_g_alpha=parameters.de_g_alpha,
            task_reference_dt=PID_REFERENCE_PERIOD_S,
            derivative_filter_reference_dt=PID_REFERENCE_PERIOD_S,
        )
        self.minimum = self.nominal - parameters.q_offset_limit_rad
        self.maximum = self.nominal + parameters.q_offset_limit_rad
        self._last_dq = np.zeros(5, dtype=float)
        self._command_q = None
        self._command_dq = np.zeros(5, dtype=float)

    def reset(self):
        self.policy.integral_error.fill(0.0)
        self.policy.prev_e_g = None
        self.policy.filtered_de_g.fill(0.0)
        self.policy.q_ref_state = None
        self._command_q = None
        self._command_dq.fill(0.0)

    def step(self, arm_slots, imu_quaternion_wxyz, yaw0_rad: float, dt: float):
        feedback_dt = float(dt)
        if not math.isfinite(feedback_dt) or feedback_dt <= 0:
            raise ValueError("PID feedback dt must be positive and finite")
        slots = np.asarray(arm_slots, dtype=float)
        if slots.shape != (13,) or not np.isfinite(slots).all():
            raise ValueError("arm feedback must be a finite 13-slot vector")
        world_from_body = rotation(imu_quaternion_wxyz)
        helper = RightArmGravityHelper(self.model, slots, yaw0_rad)
        raw_q_ref, raw_dq_ref = self.policy.compute_action(
            {
                "current_q": slots[5:10],
                "current_dq": np.asarray(self._last_dq, dtype=float),
                "torso_rotmat": world_from_body,
                "dt": float(dt),
            },
            {"compute_gravity_error_and_jacobian": helper.compute_gravity_error_and_jacobian},
        )
        policy_diagnostics = self.policy.get_last_diagnostics()
        raw_q_ref = np.asarray(raw_q_ref, dtype=float)
        raw_dq_ref = np.asarray(raw_dq_ref, dtype=float)
        if self._command_q is None:
            # The first hardware PID reference starts from measured q, not from
            # an internal simulation state that may differ from the robot.
            self._command_q = np.clip(slots[5:10], self.minimum, self.maximum)
            self._command_dq.fill(0.0)

        acceleration_limit = self.parameters.hardware_max_ddq
        # Missed deadlines do not authorize a larger command jump.
        dt = min(feedback_dt, self.control_dt)
        # Limit speed early enough that the reference can decelerate to zero
        # before reaching either +/-5 degree position boundary.  Without this
        # viability limit, clipping q at the boundary would require dq to jump
        # abruptly to zero and violate the acceleration contract.
        distance_to_min = np.maximum(self._command_q - self.minimum, 0.0)
        distance_to_max = np.maximum(self.maximum - self._command_q, 0.0)
        safe_speed_to_min = (
            -acceleration_limit * dt
            + np.sqrt(
                (acceleration_limit * dt) ** 2
                + 2.0 * acceleration_limit * distance_to_min
            )
        )
        safe_speed_to_max = (
            -acceleration_limit * dt
            + np.sqrt(
                (acceleration_limit * dt) ** 2
                + 2.0 * acceleration_limit * distance_to_max
            )
        )
        velocity_target = np.clip(
            raw_dq_ref,
            -np.minimum(self.parameters.hardware_max_dq, safe_speed_to_min),
            np.minimum(self.parameters.hardware_max_dq, safe_speed_to_max),
        )
        previous_dq = self._command_dq.copy()
        previous_q = self._command_q.copy()
        max_delta_dq = acceleration_limit * dt
        governed_dq = np.clip(
            velocity_target,
            previous_dq - max_delta_dq,
            previous_dq + max_delta_dq,
        )
        unclipped = previous_q + governed_dq * dt
        clipped = np.clip(unclipped, self.minimum, self.maximum)
        was_clipped = np.abs(clipped - unclipped) > 1e-12
        # Keep sent q and dq mutually consistent.  The stopping-distance limit
        # above makes this final numerical projection acceleration-safe.
        governed_dq = (clipped - previous_q) / max(dt, 1e-9)
        self._command_q = clipped.copy()
        self._command_dq = governed_dq.copy()
        # Keep the reused simulation policy's integrator aligned with what was
        # actually sent; otherwise its internal q_ref can drift behind the
        # hardware-only governor even though the robot never received it.
        self.policy.q_ref_state = clipped.copy()
        return clipped, governed_dq, {
            "gravity_error_before_m_s2": policy_diagnostics["error"].tolist(),
            "feedback_dt_s": feedback_dt,
            "command_integration_dt_s": dt,
            "derivative_alpha": policy_diagnostics["derivative_alpha"],
            "jacobian_method": "analytic_mujoco_site",
            "pid_error_m_s2": policy_diagnostics["error"].tolist(),
            "pid_error_derivative_raw_m_s3": policy_diagnostics[
                "error_derivative_raw"
            ].tolist(),
            "pid_error_derivative_filtered_m_s3": policy_diagnostics[
                "error_derivative_filtered"
            ].tolist(),
            "pid_integral_error_m_s": policy_diagnostics[
                "integral_error"
            ].tolist(),
            "pid_task_correction": policy_diagnostics[
                "task_correction"
            ].tolist(),
            "pid_gravity_error_jacobian": policy_diagnostics[
                "gravity_error_jacobian"
            ].tolist(),
            "pid_task_dq_rad_s": policy_diagnostics["task_dq"].tolist(),
            "pid_posture_dq_rad_s": policy_diagnostics[
                "posture_dq"
            ].tolist(),
            "raw_pid_q_ref_rad": raw_q_ref.tolist(),
            "raw_pid_dq_ref_rad_s": raw_dq_ref.tolist(),
            "governed_dq_ref_rad_s": governed_dq.tolist(),
            "governed_ddq_ref_rad_s2": (
                (governed_dq - previous_dq) / max(float(dt), 1e-9)
            ).tolist(),
            "q_ref_unclipped_rad": unclipped.tolist(),
            "q_reference_clipped": was_clipped.tolist(),
            "raw_velocity_limited": (
                np.abs(raw_dq_ref) > self.parameters.hardware_max_dq + 1e-12
            ).tolist(),
            "position_safe_speed_to_min_rad_s": safe_speed_to_min.tolist(),
            "position_safe_speed_to_max_rad_s": safe_speed_to_max.tolist(),
        }

    def set_measured_dq(self, right_dq):
        right_dq = np.asarray(right_dq, dtype=float)
        if right_dq.shape != (5,) or not np.isfinite(right_dq).all():
            raise ValueError("right_dq must contain five finite values")
        self._last_dq = right_dq.copy()


def stage(task_s: float) -> str:
    if task_s < ENTRY_END_S:
        return "arm_ramp_in"
    if task_s < WALK_START_S:
        return "stationary_baseline"
    if task_s < WALK_STOP_S:
        return "forward_walk"
    if task_s < RELEASE_START_S:
        return "stop_settle"
    if task_s < END_S:
        return "arm_ramp_out"
    return "complete"


def locomotion_setpoint(task_s: float, heading_correction_rad_s: float,
                        inhibited: bool = False) -> dict:
    """Return the reviewed forward/yaw command schedule.

    Forward motion ends at 15 s, while heading hold remains active through the
    complete stop-settle window and ends at 18 s.  The RPC duration is clipped
    at each boundary so an earlier command cannot remain active across it.
    """
    task_s = float(task_s)
    correction = float(heading_correction_rad_s)
    if not math.isfinite(task_s) or not math.isfinite(correction):
        raise ValueError("locomotion setpoint inputs must be finite")
    enabled = not bool(inhibited)
    walking = enabled and WALK_START_S <= task_s < WALK_STOP_S
    heading_hold = enabled and WALK_START_S <= task_s < RELEASE_START_S
    vx = FORWARD_SPEED_M_S if walking else 0.0
    wz = correction if heading_hold else 0.0
    if walking:
        boundary_s = WALK_STOP_S
    elif heading_hold:
        boundary_s = RELEASE_START_S
    else:
        boundary_s = task_s + 0.2
    duration_s = min(0.2, max(0.001, boundary_s - task_s))
    return {
        "vx_m_s": vx,
        "yaw_rate_rad_s": wz,
        "duration_s": duration_s,
        "walking_active": walking,
        "heading_hold_active": heading_hold,
    }


def linear_weight_release(start_weight: float, elapsed_s: float) -> tuple[float, bool]:
    """Return a monotone three-second Arm SDK hand-back.

    The first sample preserves ``start_weight``.  Weight reaches zero only at
    or after the full release duration; callers must never replace this with a
    one-frame weight-zero cleanup.
    """
    start_weight = float(start_weight)
    elapsed_s = float(elapsed_s)
    if not math.isfinite(start_weight) or not 0.0 <= start_weight <= 1.0:
        raise ValueError("release start_weight must be finite and in [0,1]")
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ValueError("release elapsed_s must be finite and non-negative")
    ratio = np.clip(elapsed_s / WEIGHT_RELEASE_DURATION_S, 0.0, 1.0)
    weight = start_weight * float(1.0 - ratio)
    return weight, elapsed_s >= WEIGHT_RELEASE_DURATION_S


class WeightReleaseRamp:
    """Three-second hand-back with a per-published-frame decrement limit.

    Wall-clock interpolation alone can jump straight to zero if the process is
    suspended during release.  This stateful limiter follows the same linear
    schedule during normal operation, but never reduces weight by more
    than one nominal control-frame step.  A delayed loop therefore extends the
    hand-back instead of producing an abrupt ownership change.
    """

    def __init__(self, start_weight: float,
                 nominal_period_s: float = WEIGHT_RELEASE_NOMINAL_PERIOD_S):
        start_weight = float(start_weight)
        nominal_period_s = float(nominal_period_s)
        if not math.isfinite(start_weight) or not 0.0 <= start_weight <= 1.0:
            raise ValueError("release start_weight must be finite and in [0,1]")
        if not math.isfinite(nominal_period_s) or nominal_period_s <= 0.0:
            raise ValueError("release nominal period must be positive and finite")
        self.start_weight = start_weight
        self.current_weight = start_weight
        self.max_step = (
            start_weight * nominal_period_s / WEIGHT_RELEASE_DURATION_S
        )
        self._first_sample = True

    def sample(self, elapsed_s: float) -> tuple[float, bool]:
        desired, _ = linear_weight_release(self.start_weight, elapsed_s)
        if self._first_sample:
            self._first_sample = False
        else:
            self.current_weight = max(
                desired,
                self.current_weight - self.max_step,
            )
        if self.current_weight <= 1e-12:
            self.current_weight = 0.0
        return self.current_weight, self.current_weight == 0.0


class HardwarePidPlan:
    """Arm plan: nonzero A3 left pose, right PID, then weight hand-back."""

    def __init__(self, initial_slots, target_slots, kp, kd, controller):
        self.initial = np.asarray(initial_slots, dtype=float).copy()
        self.target = np.asarray(target_slots, dtype=float).copy()
        self.kp = np.asarray(kp, dtype=float).copy()
        self.kd = np.asarray(kd, dtype=float).copy()
        if any(array.shape != (13,) for array in
               (self.initial, self.target, self.kp, self.kd)):
            raise ValueError("plan arrays must have 13 slots")
        self.controller = controller
        self._release_q = None
        self._release_ramp = None
        self._last_q = self.initial.copy()

    def sample(self, task_s, measured_slots, measured_dq, imu_quaternion, yaw0_rad, dt):
        task_s = float(task_s)
        measured_slots = np.asarray(measured_slots, dtype=float)
        q = self.target.copy()
        dq = np.zeros(13)
        diagnostics = {"pid_active": False}
        if task_s < ENTRY_END_S:
            ratio = np.clip(task_s / ENTRY_END_S, 0.0, 1.0)
            q = self.initial + ratio * (self.target - self.initial)
            weight = float(ratio)
        elif task_s < RELEASE_START_S:
            self.controller.set_measured_dq(np.asarray(measured_dq)[5:10])
            right_q, right_dq, pid = self.controller.step(
                measured_slots, imu_quaternion, yaw0_rad, dt
            )
            q[5:10] = right_q
            dq[5:10] = right_dq
            weight = 1.0
            diagnostics = {"pid_active": True, **pid}
        elif task_s < END_S:
            if self._release_q is None:
                self._release_q = self._last_q.copy()
                self._release_q[11:] = 0.0
                self._release_ramp = WeightReleaseRamp(1.0, self.controller.control_dt)
            q = self._release_q.copy()
            weight, terminal = self._release_ramp.sample(task_s - RELEASE_START_S)
        else:
            if self._release_q is None:
                self._release_q = self._last_q.copy()
                self._release_q[11:] = 0.0
                self._release_ramp = WeightReleaseRamp(1.0, self.controller.control_dt)
            q = self._release_q.copy()
            weight, terminal = self._release_ramp.sample(task_s - RELEASE_START_S)
        q[11:] = 0.0
        if task_s < RELEASE_START_S:
            self._last_q = q.copy()
        stage_name = stage(task_s)
        if stage_name == "complete" and not terminal:
            stage_name = "arm_ramp_out"
        return {
            "stage": stage_name,
            "q_rad": q,
            "dq_rad_s": dq,
            "kp": self.kp,
            "kd": self.kd,
            "weight": weight,
            "terminal": task_s >= END_S and terminal,
            "diagnostics": diagnostics,
        }
