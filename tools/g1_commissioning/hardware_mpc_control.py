#!/usr/bin/env python3
"""SDK-free 6 ms reference-servo MPC for the G1 Arm SDK field runner.

This reuses the simulation's actual constrained MPC and C++ task kinematics,
NOT its MuJoCo inverse-dynamics/torque mapper. The QP state is the persistent
command reference. Measured tracking offsets anchor task predictions to the
physical arm. Ideal acceleration tracking is an approximation, not a proven
model of the firmware PD servo. No DDS, locomotion or mode changes occur here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from pathlib import Path
import time

import numpy as np
import yaml

from endpoint_pose import EndpointModel, ROOT, rotation
from hardware_pid_control import CONTROL_PERIOD_S
from arm_mpc import ArmMPCPolicy
from hardware_mpc_solver import CondensedArmMPCPolicy
from disturbance_types import DisturbanceHorizon, DisturbanceInput
from kinematics_helper import KinematicsHelper
from robot_model_backend.cpp_rnea_backend import (
    CppRightArmRneaBackend, RIGHT_ARM_JOINT_NAMES,
)

DEFAULT_CONFIG = ROOT / "configs/hardware_mpc.yaml"


class HardwareMpcError(RuntimeError):
    """No new output is authorized; the runner must stop and hand back."""


def json_values(value):
    """Keep failure diagnostics valid JSON without hiding solver failures."""
    value_type = type(value)
    if value_type is float:
        return value if math.isfinite(value) else None
    if value_type in (int, bool, str, type(None)):
        return value
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_values(item) for item in value]
    if isinstance(value, np.ndarray):
        # These dtypes produce only native JSON scalars. Check floating values
        # once in NumPy instead of recursively inspecting every list element.
        # Extended floats and object arrays still need scalar conversion below.
        if (value.dtype.kind in "biu" or
                (value.dtype.kind == "f" and value.dtype.itemsize <= 8
                 and np.isfinite(value).all())):
            return value.tolist()
        return json_values(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_mpc_config(config=None):
    values = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if config is not None:
        values.update(yaml.safe_load(Path(config).read_text())
                      if isinstance(config, (str, Path)) else dict(config))
    if values.get("schema") != "g1_hardware_mpc_config_v1":
        raise ValueError("unexpected hardware MPC configuration schema")
    if values["prediction_backend"] != "cpp_pinocchio":
        raise ValueError("hardware MPC requires the explicit C++ kinematics backend")
    if values["horizon"] != 9 or abs(float(values["control_period_s"]) - .006) > 1e-12:
        raise ValueError("hardware MPC contract is nine 6 ms control intervals")
    bounds = np.asarray(values["reference_offset_limit_deg"], dtype=float)
    if bounds.shape != (5,) or not np.isfinite(bounds).all() or np.any(bounds <= 0) or np.any(bounds > 5):
        raise ValueError("reference offsets must contain five values in (0,5] degrees")
    for key, upper in (("reference_max_dq_rad_s", .07),
                       ("reference_max_ddq_rad_s2", .20)):
        number = float(values[key])
        if not math.isfinite(number) or not 0 < number <= upper:
            raise ValueError(f"{key} exceeds the field reference envelope")
    limit = float(values["solver_time_limit_s"])
    if not math.isfinite(limit) or not 0 < limit <= .004:
        raise ValueError("QP time budget must be positive and at most 4 ms")
    return values


def _validated_horizon(horizon):
    if not isinstance(horizon, DisturbanceHorizon) or len(horizon.nodes) != 10 or len(horizon.intervals) != 9:
        raise ValueError("MPC needs ten nodes and nine following intervals")
    samples = (*horizon.nodes, *horizon.intervals)
    vectors = np.asarray([[x.acc_world, x.omega_world, x.alpha_world] for x in samples], dtype=float)
    matrices = np.asarray([x.rot_world_body for x in samples], dtype=float)
    if vectors.shape != (19, 3, 3) or not np.isfinite(vectors).all():
        raise ValueError("invalid H0 disturbance vectors")
    if (matrices.shape != (19, 3, 3) or not np.isfinite(matrices).all()
        or np.max(np.abs(matrices.transpose(0, 2, 1) @ matrices - np.eye(3))) > 1e-6
        or np.max(np.abs(np.linalg.det(matrices) - 1)) > 1e-6):
        raise ValueError("H0 torso orientation is not a proper rotation")
    checked = [DisturbanceInput(v[0], v[1], v[2], r) for v, r in zip(vectors, matrices)]
    return DisturbanceHorizon(tuple(checked[:10]), tuple(checked[10:]))


class RightArmHardwareMpc:
    """Persistent bounded q/dq references from an actual nine-interval QP.

    Compatible with HardwarePidPlan's controller interface. The runner supplies
    a fresh, causally produced horizon before each sample; it also owns freshness
    checks and exception cleanup. Command initialization is the nominal pose
    already reached by the three-second ramp, not a jump to measured q.
    """

    control_dt = CONTROL_PERIOD_S

    def __init__(self, nominal_right_q, config=None, model=None, library_path=None):
        self.config = load_mpc_config(config)
        self.nominal = np.asarray(nominal_right_q, dtype=float).copy()
        if self.nominal.shape != (5,) or not np.isfinite(self.nominal).all():
            raise ValueError("nominal_right_q must contain five finite angles")
        self.model = EndpointModel() if model is None else model
        offsets = np.deg2rad(self.config["reference_offset_limit_deg"])
        self.minimum, self.maximum = self.nominal - offsets, self.nominal + offsets
        self.max_dq = float(self.config["reference_max_dq_rad_s"])
        self.max_ddq = float(self.config["reference_max_ddq_rad_s2"])
        path = Path(library_path or self.config["kinematics_library"])
        if not path.is_absolute():
            path = ROOT / path
        self.backend = CppRightArmRneaBackend(self.model.xml, library_path=path, retain_gil=True)
        if (self.backend.nq, self.backend.nv) != (self.model.model.nq, self.model.model.nv):
            self.backend.close()
            raise ValueError("C++ and endpoint model dimensions differ")
        indices = np.array([self.model.model.joint(name).qposadr[0]
                            for name in RIGHT_ARM_JOINT_NAMES])
        self.helper = KinematicsHelper(
            self.model.model, "right_grasp_site", indices,
            position_reference_q=self.nominal, prediction_backend=self.backend,
        )
        keys = ("q_ee_acc", "q_ee_alpha", "q_ee_omega", "q_gravity", "q_posture",
                "q_vel", "r_ddq", "terminal_scale", "solver_eps_abs", "solver_eps_rel",
                "solver_max_iter", "solver_check_termination", "solver_rho",
                "solver_adaptive_rho")
        self.policy = CondensedArmMPCPolicy(
            self.nominal, control_dt=self.control_dt, horizon=9,
            solver_backend=self.config["solver_backend"],
            joint_limits=np.column_stack((self.minimum, self.maximum)),
            joint_limit_margin=0.0, max_dq=self.max_dq, max_ddq=self.max_ddq,
            reg=float(self.config["regularization"]),
            solver_time_limit=float(self.config["solver_time_limit_s"]),
            **{key: self.config[key] for key in keys},
        )
        self.metadata = {
            "schema": "g1_hardware_mpc_core_v1", "model": "reference_servo_mpc",
            "solver": self.config["solver_backend"]+"_exact_state_elimination_45_inputs_full_QP_checked",
            "solver_package_version": importlib.metadata.version(self.config["solver_backend"]),
            "state": "persistent sent q_reference,dq_reference, not measured joint state",
            "tracking_offset_model": "q_actual(k)=q_virtual(k)+q_error+k*dt*dq_error; dq_actual=dq_virtual+dq_error",
            "actuation": "Arm SDK firmware PD; tau_ff=0; no inverse dynamics",
            "frame": "fixed H0", "config": self.config,
            "library": str(path.resolve()),
            "library_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "xml_sha256": self.model.xml_hashes(),
            "cost": self.policy.get_cost_definition(),
        }
        self.reset()

    def close(self):
        self.backend.close()

    def reset(self):
        self.policy.reset()
        self._command_q = self.nominal.copy()
        self._command_dq = np.zeros(5)
        self._measured_dq = np.zeros(5)
        self._horizon = None
        self._extra = {}
        self.last_diagnostics = {}

    def set_measured_dq(self, right_dq):
        value = np.asarray(right_dq, dtype=float)
        if value.shape != (5,) or not np.isfinite(value).all():
            raise ValueError("right_dq must contain five finite measurements")
        self._measured_dq = value.copy()

    def set_disturbance_horizon(self, horizon, extra_diagnostics=None):
        self._horizon = _validated_horizon(horizon)
        self._extra = dict(extra_diagnostics or {})

    @staticmethod
    def _shift_terms(terms, q_offset, dq_offset):
        result = dict(terms)
        for suffix in ("acc", "alpha", "omega"):
            result["D_" + suffix] = terms["D_" + suffix] + terms["C_" + suffix] @ dq_offset
        result["d_g"] = terms["d_g"] + terms["G_g"] @ np.r_[q_offset, dq_offset]
        return result

    def _task_helpers(self, slots, q_error, dq_error, horizon):
        # Model base pose is arbitrary: KinematicsHelper rotates its current
        # torso frame to each supplied H0 orientation. No fabricated global
        # translation, leg contact or acceleration is used.
        self.model.data.qpos[:] = self.model.model.qpos0
        self.model.data.qpos[self.model.joint_addresses] = slots[:11]
        helpers = self.helper.build_helpers(
            self.model.data, disturbance_prediction=horizon.nodes,
            interval_disturbance_prediction=horizon.intervals,
            include_kinematics_cache=False,
        )
        scalar, batch = helpers.compute_mpc_terms, helpers.compute_mpc_terms_batch

        def shifted_scalar(q, dq, node, interval=None, acceleration_required=True):
            terms = scalar(q + q_error, dq + dq_error, node, interval, acceleration_required)
            return self._shift_terms(terms, q_error, dq_error)

        def shifted_batch(q, dq, nodes, intervals, required):
            q_offsets = q_error + np.arange(len(q))[:, None] * self.control_dt * dq_error
            terms = batch(q + q_offsets, dq + dq_error, nodes, intervals, required)
            return tuple(self._shift_terms(row, offset, dq_error)
                         for row, offset in zip(terms, q_offsets))

        helpers.compute_mpc_terms = shifted_scalar
        helpers.compute_mpc_terms_batch = shifted_batch
        return helpers

    def _govern(self, desired_dq, dt):
        # Same stopping-distance and rate envelope as the established PID.
        previous_q, previous_dq = self._command_q.copy(), self._command_dq.copy()
        distances = (np.maximum(previous_q - self.minimum, 0),
                     np.maximum(self.maximum - previous_q, 0))
        # Reserve one additional discrete interval of braking distance. The
        # QP uses a double integrator whereas sent references use semi-implicit
        # integration: the continuous v²/(2a) envelope alone can leave the next
        # discrete QP infeasible within microradians of a position boundary.
        safe = [-2*self.max_ddq * dt + np.sqrt((2*self.max_ddq * dt) ** 2 + 2 * self.max_ddq * d)
                for d in distances]
        target = np.clip(desired_dq, -np.minimum(self.max_dq, safe[0]), np.minimum(self.max_dq, safe[1]))
        velocity = np.clip(target, previous_dq - self.max_ddq * dt, previous_dq + self.max_ddq * dt)
        q = np.clip(previous_q + dt * velocity, self.minimum, self.maximum)
        velocity = (q - previous_q) / dt
        if np.max(np.abs(velocity - previous_dq)) > self.max_ddq * dt + 1e-9:
            raise HardwareMpcError("reference governor lost acceleration viability")
        return q, velocity

    def step(self, arm_slots, imu_quaternion_wxyz, yaw0_rad, dt):
        started = time.perf_counter_ns()
        slots = np.asarray(arm_slots, dtype=float)
        if slots.shape != (13,) or not np.isfinite(slots).all():
            raise ValueError("arm slots must contain thirteen finite angles")
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(float(yaw0_rad)):
            raise ValueError("feedback dt and H0 yaw must be finite; dt positive")
        rotation(imu_quaternion_wxyz)  # validate even though the horizon supplies R
        if self._horizon is None:
            raise HardwareMpcError("a fresh disturbance horizon is required")
        horizon, self._horizon = self._horizon, None  # stale reuse is not implicit
        previous_q, previous_dq = self._command_q.copy(), self._command_dq.copy()
        q_error, dq_error = slots[5:10] - previous_q, self._measured_dq - previous_dq
        helpers = self._task_helpers(slots, q_error, dq_error, horizon)
        raw_q, raw_dq, ddq = self.policy.compute_action(
            {"current_q": previous_q, "current_dq": previous_dq, "dt": self.control_dt}, helpers,
        )
        diagnostics = self.policy.get_last_diagnostics(copy_data=False)
        # Every-cycle audit retains inputs/forecast, actual sent references,
        # first MPC action, objective, constraints and all solver timing. Large
        # duplicated work/solution rollouts add no sensor evidence; keep them
        # only for a failed solve, outside the normal high-rate journal path.
        omitted = {"working_states", "working_inputs", "predicted_states", "predicted_inputs",
                   "disturbance_prediction", "interval_disturbance_prediction"}
        recorded = diagnostics if not diagnostics["solved"] else {
            key: value for key, value in diagnostics.items() if key not in omitted}
        self.last_diagnostics = {
            "mpc_active": True, "controller_kind": "reference_servo_mpc",
            "gravity_error_before_m_s2": diagnostics["gravity_error"].tolist(),
            "mpc": json_values(recorded), "predictor": self._extra,
            "measured_q_rad": slots[5:10].tolist(),
            "measured_dq_rad_s": self._measured_dq.tolist(),
            "reference_tracking_error_rad": q_error.tolist(),
            "velocity_tracking_error_rad_s": dq_error.tolist(),
            "raw_mpc_q_reference_rad": raw_q.tolist(),
            "raw_mpc_dq_reference_rad_s": raw_dq.tolist(),
            "raw_mpc_ddq_rad_s2": ddq.tolist(),
            "feedback_dt_s": dt, "command_integration_dt_s": min(dt, self.control_dt),
        }
        if not diagnostics["solved"] or diagnostics["fallback_used"]:
            raise HardwareMpcError(f"MPC rejected solution: {diagnostics['solver_status']}")
        command_dt = min(dt, self.control_dt)
        q, dq = self._govern(raw_dq, command_dt)
        self._command_q, self._command_dq = q.copy(), dq.copy()
        self.last_diagnostics.update({
            "q_reference_clipped": (np.abs(q - self.nominal) >= self.maximum - self.nominal - 1e-9).tolist(),
            "governed_ddq_reference_rad_s2": ((dq - previous_dq) / command_dt).tolist(),
            "governor_changed_solution": bool(np.max(np.abs(dq - raw_dq)) > 1e-7),
            "governor_q_minus_qp_q_rad": (q - raw_q).tolist(),
            "controller_core_ms": (time.perf_counter_ns() - started) * 1e-6,
        })
        return q.copy(), dq.copy(), self.last_diagnostics

    def warmup(self, arm_slots, imu_quaternion_wxyz, yaw0_rad=0.0, count=30):
        """Solve before publisher creation, then reset all reference state."""
        c, s = math.cos(yaw0_rad), math.sin(yaw0_rad)
        h0_from_world = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        matrix = h0_from_world @ rotation(imu_quaternion_wxyz)
        disturbance = DisturbanceInput(np.zeros(3), np.zeros(3), np.zeros(3), matrix)
        horizon = DisturbanceHorizon((disturbance,) * 10, (disturbance,) * 9)
        durations = []
        live_limit = self.policy.solver_time_limit
        # First allocation/page faults happen before DDS, not against the live
        # solve budget. Restore the reviewed limit before any plan can execute.
        self.policy.solver_time_limit = .1
        if self.policy._condensed_solver is not None:
            self.policy._condensed_solver.update_settings(time_limit=.1)
        try:
            for _ in range(int(count)):
                self.set_disturbance_horizon(horizon, {"mode": "offline_warmup_zoh"})
                _, _, diagnostics = self.step(arm_slots, imu_quaternion_wxyz, yaw0_rad, self.control_dt)
                durations.append(diagnostics["controller_core_ms"])
        finally:
            self.policy.solver_time_limit = live_limit
            if self.policy._condensed_solver is not None:
                self.policy._condensed_solver.update_settings(time_limit=live_limit)
            self.reset()
        return {"samples": len(durations), "max_core_ms": max(durations, default=0),
                "median_core_ms": float(np.median(durations)) if durations else 0.0}
