"""Measured-state MPC and model-checked torque execution, offline migration.

The legacy reference-servo controller remains an explicit comparison path.
This controller consumes real q/dq, produces a one-step PD reference, and
checks TOTAL torque candidates. Field output remains unsupported until the
physical torque envelope and response have been commissioned.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
import numpy as np
import yaml

from endpoint_pose import ROOT, rotation
from hardware_arm_inverse_dynamics import RightArmInverseDynamics, finite_vector
from hardware_mpc_control import RightArmHardwareMpc, HardwareMpcError, json_values
from hardware_mpc_solver import CondensedArmMPCPolicy
from hardware_torque_mapper import LocalTorqueMapper, NoModelTorque
from hardware_pid_control import HardwarePidPlan

ACTUATION = "measured_torque_preview"
TORQUE_CONFIG = ROOT / "configs/hardware_mpc_torque_preview.yaml"


def load_torque_config(config=None):
    values = yaml.safe_load(TORQUE_CONFIG.read_text())
    if config is not None:
        values.update(yaml.safe_load(Path(config).read_text())
                      if isinstance(config, (str, Path)) else config)
    if values.get("schema") != "g1_measured_torque_preview_v1":
        raise ValueError("unexpected measured torque configuration")
    for key in ("q_min_deg", "q_max_deg", "q_margin_deg", "kp", "kd", "transition_rate_nm_s"):
        values[key] = finite_vector(values[key], 5, key)
    if (np.any(values["q_min_deg"]+2*values["q_margin_deg"] >= values["q_max_deg"])
            or np.any(values["q_margin_deg"] < 0)
            or np.any(values["kp"] < 0) or np.any(values["kd"] < 0)
            or np.any(values["transition_rate_nm_s"] <= 0)):
        raise ValueError("invalid physical-state/PD/transition configuration")
    for key in ("max_dq_rad_s", "max_ddq_rad_s2"):
        if not math.isfinite(float(values[key])) or float(values[key]) <= 0:
            raise ValueError(f"invalid {key}")
    enabled = values.setdefault("recovery_envelope_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("recovery_envelope_enabled must be boolean")
    if enabled:
        values["recovery_guard_deg"] = finite_vector(values["recovery_guard_deg"], 5, "recovery guard")
        rate = float(values["recovery_rate_s_inv"])
        if (not math.isfinite(rate) or rate <= 0 or np.any(values["recovery_guard_deg"] < 0)
                or np.any(2*values["recovery_guard_deg"] >= values["q_max_deg"]-values["q_min_deg"])):
            raise ValueError("invalid recovery envelope")
    return values


class RightArmMeasuredTorqueMpc(RightArmHardwareMpc):
    offline_only = True

    def __init__(self, *args, torque_config=None, **kwargs):
        c = load_torque_config(torque_config)
        mapper = LocalTorqueMapper(c)
        super().__init__(*args, **kwargs)
        self.torque_config = c
        self.minimum, self.maximum = np.deg2rad(c["q_min_deg"]), np.deg2rad(c["q_max_deg"])
        self.max_dq, self.max_ddq = float(c["max_dq_rad_s"]), float(c["max_ddq_rad_s2"])
        keys = ("q_ee_acc", "q_ee_alpha", "q_ee_omega", "q_gravity", "q_posture",
                "q_vel", "r_ddq", "terminal_scale", "solver_eps_abs", "solver_eps_rel",
                "solver_max_iter", "solver_check_termination", "solver_rho", "solver_adaptive_rho")
        policy_type = CondensedArmMPCPolicy
        recovery_options = {}
        if c["recovery_envelope_enabled"]:
            from hardware_mpc_recovery import RecoveryEnvelopeMpcPolicy
            policy_type = RecoveryEnvelopeMpcPolicy
            recovery_options = dict(recovery_rate_s_inv=c["recovery_rate_s_inv"],
                                    recovery_guard_rad=np.deg2rad(c["recovery_guard_deg"]))
        self.policy = policy_type(
            self.nominal, control_dt=.006, horizon=9, solver_backend=self.config["solver_backend"],
            joint_limits=np.column_stack((self.minimum, self.maximum)),
            joint_limit_margin=np.deg2rad(c["q_margin_deg"]),
            max_dq=self.max_dq, max_ddq=self.max_ddq, reg=float(self.config["regularization"]),
            solver_time_limit=float(self.config["solver_time_limit_s"]),
            **{key: self.config[key] for key in keys}, **recovery_options)
        self.inverse = RightArmInverseDynamics(self.model, self.backend)
        self.mapper = mapper
        self._prepared_forward = None
        self.metadata.update(model="measured_state_acceleration_mpc",
            state="measured q,dq; previous solution only warm-starts optimization",
            tracking_offset_model="none; measured initial state, no persistent-reference governor",
            actuation=ACTUATION, offline_only=True, field_output_supported=False,
            torque_config=json_values(c), inverse_dynamics=self.inverse.metadata,
            recovery_envelope=(dict(enabled=True, rate_s_inv=c["recovery_rate_s_inv"],
                guard_deg=c["recovery_guard_deg"].tolist(),
                semantics="extra predicted q+dq/rate bounds inside unchanged outer joint limits",
                physical_safety_certified=False) if c["recovery_envelope_enabled"] else dict(enabled=False)),
            forward_model="MuJoCo conditional right-arm mass/bias with prescribed observed torso motion",
            unmodelled=["unknown ground/contact reactions", "actuator delay and torque gain",
                        "physical friction and payload inertia error", "future arm-to-base reaction"],
            output_semantics="selected total torque minus current PD; device adds PD once",
            cost=self.policy.get_cost_definition())
        self.reset()

    def reset(self):
        super().reset()
        self._previous_total = None
        self._current_base = None
        self._next_total_bounds = None

    def _torque(self, q, dq, qref, dqref, ddq, base, bounds=None):
        inverse, mass, bias = self.inverse.compute_with_linear_dynamics(q, dq, ddq, base)
        gain = np.linalg.solve(mass, np.eye(5))
        forward = lambda tau: gain @ (tau-bias)
        pd = self.torque_config["kp"]*(qref-q)+self.torque_config["kd"]*(dqref-dq)
        try:
            total, mapping = self.mapper.compute(forward, ddq, inverse["tau_model_nm"]+pd,
                                                safe_hold=bias-self.torque_config["kd"]*dq,
                                                previous=self._previous_total, bounds=bounds)
        except NoModelTorque as exc:
            self.last_diagnostics.update(mapper=exc.trace, torque_output_authorized=False)
            self.last_diagnostics = json_values(self.last_diagnostics)
            raise
        self._previous_total = total.copy()
        self._prepared_forward = (mass, bias)
        return dict(inverse_dynamics=inverse, mapper=mapping,
                    tau_nominal_ff_nm=inverse["tau_model_nm"], tau_pd_at_feedback_nm=pd,
                    tau_ff_candidate_nm=total-pd, tau_total_estimated_at_feedback_nm=total,
                    expected_kp=self.torque_config["kp"], expected_kd=self.torque_config["kd"],
                    torque_output_authorized=False, torque_estimate_feedback_used_for_control=False)

    def step(self, arm_slots, imu_quaternion_wxyz, yaw0_rad, dt):
        start = time.perf_counter_ns()
        slots = finite_vector(arm_slots, 13, "arm slots")
        rotation(imu_quaternion_wxyz)
        if not math.isfinite(float(dt)) or dt <= 0 or not math.isfinite(float(yaw0_rad)):
            raise ValueError("invalid feedback time/frame")
        if self._horizon is None:
            raise HardwareMpcError("a fresh disturbance horizon is required")
        horizon, self._horizon = self._horizon, None
        self._current_base = horizon.nodes[0]
        q, dq = slots[5:10].copy(), self._measured_dq.copy()
        # Use the SAME task kinematics as simulation without tracking offsets.
        self.model.data.qpos[:] = self.model.model.qpos0
        self.model.data.qpos[self.model.joint_addresses] = slots[:11]
        helpers = self.helper.build_helpers(self.model.data, disturbance_prediction=horizon.nodes,
            interval_disturbance_prediction=horizon.intervals, include_kinematics_cache=False)
        qref, dqref, ddq = self.policy.compute_action({"current_q": q, "current_dq": dq, "dt": .006}, helpers)
        qp = self.policy.get_last_diagnostics(copy_data=False)
        self.last_diagnostics = dict(mpc_active=True, controller_kind=ACTUATION,
            measured_q_rad=q, measured_dq_rad_s=dq, mpc_initial_state=np.r_[q, dq],
            raw_mpc_ddq_rad_s2=ddq, one_step_q_reference_rad=qref,
            one_step_dq_reference_rad_s=dqref, predictor=self._extra,
            gravity_error_before_m_s2=qp["gravity_error"], feedback_dt_s=float(dt),
            command_integration_dt_s=.006, mpc={key: value for key, value in qp.items()
                if key not in {"working_states", "working_inputs", "predicted_states", "predicted_inputs",
                               "disturbance_prediction", "interval_disturbance_prediction"}})
        if not qp["solved"] or qp["fallback_used"]:
            self.last_diagnostics = json_values(self.last_diagnostics)
            raise HardwareMpcError(f"measured-state MPC rejected: {qp['solver_status']}")
        bounds, self._next_total_bounds = self._next_total_bounds, None
        self.last_diagnostics.update(self._torque(q, dq, qref, dqref, ddq, self._current_base, bounds))
        self.last_diagnostics["controller_core_ms"] = (time.perf_counter_ns()-start)*1e-6
        self.last_diagnostics = json_values(self.last_diagnostics)
        return qref.copy(), dqref.copy(), self.last_diagnostics


class HardwareTorquePreviewPlan(HardwarePidPlan):
    """Offline complete entry/control/release plan; no DDS authority.

    Total right-arm torque is rate-bounded through transitions; the final
    command is checked again after this projection. Weight release retains
    support feedforward until weight reaches zero, avoiding a torque drop at 18s.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (not np.allclose(self.kp[5:10], self.controller.torque_config["kp"])
                or not np.allclose(self.kd[5:10], self.controller.torque_config["kd"])):
            raise ValueError("plan PD must match the evaluated torque model")
        self._last_total = None

    def sample(self, task_s, measured_slots, measured_dq, imu_quaternion, yaw0_rad, dt):
        c = self.controller
        if not math.isfinite(float(dt)) or dt <= 0 or not math.isfinite(float(task_s)):
            raise ValueError("invalid plan time")
        if c._horizon is None:
            raise HardwareMpcError("fresh torso horizon required in entry and release too")
        base = c._horizon.nodes[0]
        q = finite_vector(measured_slots, 13, "q")[5:10]
        dq = finite_vector(measured_dq, 13, "dq")[5:10]
        c._next_total_bounds = None
        if self._last_total is not None and 3 <= task_s < 18:
            delta = c.torque_config["transition_rate_nm_s"]*min(float(dt), .006)
            c._next_total_bounds = (self._last_total-delta, self._last_total+delta)
        frame = super().sample(task_s, measured_slots, measured_dq, imu_quaternion, yaw0_rad, dt)
        pd = self.kp[5:10]*(frame["q_rad"][5:10]-q)+self.kd[5:10]*(frame["dq_rad_s"][5:10]-dq)
        if not 3 <= task_s < 18:
            c._horizon = None
            inverse = c.inverse.compute(q, dq, np.zeros(5), base)
            entry = float(np.clip(task_s/3., 0., 1.))
            total = inverse["tau_model_nm"]*entry+pd
            frame["diagnostics"].update(controller_kind=ACTUATION, inverse_dynamics=inverse,
                torque_output_authorized=False, transition_model=True)
        else:
            total = np.asarray(frame["diagnostics"]["tau_total_estimated_at_feedback_nm"])
        limit = c.mapper.limit
        total = np.clip(total, -limit, limit)
        if self._last_total is not None and not 3 <= task_s < 18:
            delta = c.torque_config["transition_rate_nm_s"]*min(float(dt), .006)
            # Scalar interpolation preserves coupled forward-model behaviour;
            # clipping five axes separately can create an unchecked direction.
            difference = total-self._last_total
            ratio = min(1., float(np.min(delta/np.maximum(np.abs(difference), 1e-12))))
            total = self._last_total+ratio*difference
        self._last_total = total.copy()
        c._previous_total = total.copy()
        mass, bias = (c._prepared_forward if 3 <= task_s < 18 else
                      c.inverse.linear_dynamics(q, dq, base))
        acceleration = np.linalg.solve(mass, total-bias)
        # Entry is the existing position ramp, not acceleration-controlled MPC.
        # During active/release, reject a transition that invalidates the model
        # envelope instead of calling the pre-projection result accepted.
        if frame["weight"] > 0 and task_s >= 3 and np.max(np.abs(acceleration)) > c.mapper.acc_limit+1e-9:
            raise HardwareMpcError("post-transition total torque fails forward-model envelope")
        frame["diagnostics"].update(tau_pd_at_feedback_nm=pd.tolist(),
            feedback_dt_s=float(dt),
            tau_total_estimated_at_feedback_nm=total.tolist(),
            tau_ff_candidate_nm=(total-pd).tolist(), expected_kp=self.kp[5:10].tolist(),
            expected_kd=self.kd[5:10].tolist(), post_transition_ddq_rad_s2=acceleration.tolist(),
            acceptance="conditional_forward_model_only", torque_output_authorized=False)
        # No residual torque command after ownership has been fully released.
        if frame["weight"] == 0:
            frame["diagnostics"]["tau_ff_candidate_nm"] = [0.]*5
        return frame


def make_torque_preview_message(frame, state, constructor, crc, *, host_full_torque=False):
    from g1_walk_pid import make_arm_message
    message = make_arm_message(frame, state, constructor, crc, finalize_crc=False)
    diag = frame["diagnostics"]
    if diag.get("controller_kind") != ACTUATION:
        raise ValueError("measured torque packet requires explicit controller provenance")
    if not (np.allclose(frame["kp"][5:10], diag["expected_kp"])
            and np.allclose(frame["kd"][5:10], diag["expected_kd"])):
        raise ValueError("packet PD differs from evaluated PD")
    key = "tau_total_estimated_at_feedback_nm" if host_full_torque else "tau_ff_candidate_nm"
    tau = finite_vector(diag[key], 5, "packet torque")
    for i, value in zip(range(22, 27), tau):
        message.motor_cmd[i].tau = float(value) if frame["weight"] > 0 else 0.
        if host_full_torque:
            message.motor_cmd[i].kp = message.motor_cmd[i].kd = 0.
    message.crc = crc.Crc(message)
    return message
