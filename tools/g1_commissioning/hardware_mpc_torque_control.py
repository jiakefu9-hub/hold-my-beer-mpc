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
import orjson
import yaml

from endpoint_pose import ROOT, rotation
from hardware_arm_inverse_dynamics import RightArmInverseDynamics, finite_vector
from hardware_mpc_control import RightArmHardwareMpc, HardwareMpcError, json_values
from hardware_mpc_solver import CondensedArmMPCPolicy
from hardware_mpc_braking import LatchedPredictiveBrake
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
    if values.setdefault('active_slew_reference', 'total') not in ('none', 'total', 'model_bias_relative'):
        raise ValueError('unknown active torque slew reference')
    for key in ('recovery_reentry_enabled','enforce_mapper_state_envelope'):
        if not isinstance(values.setdefault(key,False),bool):
            raise ValueError(f'{key} must be boolean')
    if (values['recovery_reentry_enabled'] and not values['enforce_mapper_state_envelope']
            or values['enforce_mapper_state_envelope'] and not enabled):
            raise ValueError('bounded reentry requires both recovery and final mapper state checks')
    if not isinstance(values.setdefault('planning_actuation_constraints_enabled',False),bool):
        raise ValueError('planning_actuation_constraints_enabled must be boolean')
    if not isinstance(values.setdefault('anchor_shoulder_yaw_reference',False),bool):
        raise ValueError('anchor_shoulder_yaw_reference must be boolean')
    if not isinstance(values.setdefault('predictive_braking_enabled',False),bool):
        raise ValueError('predictive_braking_enabled must be boolean')
    for key, default in (('braking_deceleration_rad_s2',4.),('braking_reaction_s',.012),
                         ('braking_margin_deg',1.),('braking_velocity_weight',50.)):
        values[key] = float(values.get(key,default))
        if not math.isfinite(values[key]) or values[key] < 0 or (key != 'braking_reaction_s' and values[key] == 0):
            raise ValueError(f'invalid {key}')
    if (values['braking_deceleration_rad_s2'] > values['max_ddq_rad_s2']
            or np.any(2*values['braking_margin_deg'] >= values['q_max_deg']-values['q_min_deg'])):
        raise ValueError('invalid predictive braking tuning')
    values['braking_min_deceleration_rad_s2'] = float(
        values.get('braking_min_deceleration_rad_s2', 6.))
    if (not math.isfinite(values['braking_min_deceleration_rad_s2'])
            or not 0 < values['braking_min_deceleration_rad_s2'] <= values['max_ddq_rad_s2']):
        raise ValueError('invalid braking_min_deceleration_rad_s2')
    if enabled:
        values["recovery_guard_deg"] = finite_vector(values["recovery_guard_deg"], 5, "recovery guard")
        rate = float(values["recovery_rate_s_inv"])
        if (not math.isfinite(rate) or rate <= 0 or np.any(values["recovery_guard_deg"] < 0)
                or np.any(2*values["recovery_guard_deg"] >= values["q_max_deg"]-values["q_min_deg"])):
            raise ValueError("invalid recovery envelope")
    return values


class RightArmMeasuredTorqueMpc(RightArmHardwareMpc):
    offline_only = True
    policy_type = CondensedArmMPCPolicy

    def __init__(self, *args, torque_config=None, **kwargs):
        c = load_torque_config(torque_config)
        mapper = LocalTorqueMapper(c)
        # The base constructor calls the overridden reset(), so create this
        # state before delegating to it.
        self.brake = LatchedPredictiveBrake(5)
        super().__init__(*args, **kwargs)
        self.torque_config = c
        self.minimum, self.maximum = np.deg2rad(c["q_min_deg"]), np.deg2rad(c["q_max_deg"])
        self.max_dq, self.max_ddq = float(c["max_dq_rad_s"]), float(c["max_ddq_rad_s2"])
        keys = ("q_ee_acc", "q_ee_alpha", "q_ee_omega", "q_gravity", "q_posture",
                "q_vel", "r_ddq", "terminal_scale", "solver_eps_abs", "solver_eps_rel",
                "solver_max_iter", "solver_check_termination", "solver_rho", "solver_adaptive_rho")
        policy_type = self.policy_type
        recovery_options = {}
        if c["recovery_envelope_enabled"]:
            from hardware_mpc_recovery import RecoveryEnvelopeMpcPolicy
            policy_type = RecoveryEnvelopeMpcPolicy
            recovery_options = dict(recovery_rate_s_inv=c["recovery_rate_s_inv"],
                                    recovery_reentry_enabled=c['recovery_reentry_enabled'],
                                    recovery_guard_rad=np.deg2rad(c["recovery_guard_deg"]))
        self.policy = policy_type(
            self.nominal, control_dt=.006, horizon=9, solver_backend=self.config["solver_backend"],
            joint_limits=np.column_stack((self.minimum, self.maximum)),
            joint_limit_margin=np.deg2rad(c["q_margin_deg"]),
            max_dq=self.max_dq, max_ddq=self.max_ddq, reg=float(self.config["regularization"]),
            solver_time_limit=float(self.config["solver_time_limit_s"]),
            **{key: self.config[key] for key in keys}, **recovery_options)
        self.policy._local_kp = c['kp'].copy()
        self.policy._local_kd = c['kd'].copy()
        self.inverse = RightArmInverseDynamics(self.model, self.backend)
        self.mapper = mapper
        self._active_braking_acceleration_bounds = None
        self._prepared_forward = None
        self.metadata.update(model="measured_state_acceleration_mpc",
            state="measured q,dq; previous solution only warm-starts optimization",
            tracking_offset_model="none; measured initial state, no persistent-reference governor",
            actuation=ACTUATION, offline_only=True, field_output_supported=False,
            torque_config=json_values(c), inverse_dynamics=self.inverse.metadata,
            recovery_envelope=(dict(enabled=True, rate_s_inv=c["recovery_rate_s_inv"],
                bounded_reentry_enabled=c['recovery_reentry_enabled'],
                reentry_deadline_s=.054, final_mapper_state_checked=c['enforce_mapper_state_envelope'],
                guard_deg=c["recovery_guard_deg"].tolist(),
                semantics="extra predicted q+dq/rate bounds inside unchanged outer joint limits",
                physical_safety_certified=False) if c["recovery_envelope_enabled"] else dict(enabled=False)),
            forward_model="MuJoCo conditional right-arm mass/bias with prescribed observed torso motion",
            unmodelled=["unknown ground/contact reactions", "actuator delay and torque gain",
                        "physical friction and payload inertia error", "future arm-to-base reaction"],
            output_semantics="selected total torque minus current PD; device adds PD once",
            shoulder_yaw_execution=dict(
                nominal_reference_rad=float(self.nominal[2]),
                reference_anchored=c['anchor_shoulder_yaw_reference'],
                packet_kp=float(c['kp'][2]), packet_kd=float(c['kd'][2]),
                semantics=('fixed nominal packet q/dq reference; field-configured light firmware PD '
                           'is retained once on shoulder yaw while other axes retain model-total execution')),
            planning_actuation_constraints=dict(
                enabled=c['planning_actuation_constraints_enabled'],
                semantics='absolute total/feedforward limits inside the acceleration MPC; no torque-rate rows',
                local_model='current conditional arm mass and bias frozen across 54 ms horizon'),
            predictive_braking=dict(enabled=c['predictive_braking_enabled'],
                semantics='soft horizon cost plus latched first-action braking; no abort gate',
                deceleration_rad_s2=c['braking_deceleration_rad_s2'],
                reaction_s=c['braking_reaction_s'], margin_deg=c['braking_margin_deg'],
                velocity_weight=c['braking_velocity_weight'],
                minimum_deceleration_rad_s2=c['braking_min_deceleration_rad_s2'],
                physical_guarantee=False),
            cost=self.policy.get_cost_definition())
        self.reset()

    def reset(self):
        super().reset()
        self._previous_total = None
        self._current_base = None
        self._next_total_bounds = None
        self._next_slew_bias = None
        self._recovery_elapsed_s = 0.
        self._active_braking_acceleration_bounds = None
        self.brake.reset()

    def _torque(self, q, dq, qref, dqref, ddq, base, bounds=None, prepared_dynamics=None):
        if prepared_dynamics is None:
            inverse, mass, bias = self.inverse.compute_with_linear_dynamics(q, dq, ddq, base)
        else:
            mass, bias = prepared_dynamics
            inverse = self.inverse.compute(q, dq, ddq, base)
        previous_bias, self._next_slew_bias = self._next_slew_bias, None
        if self.torque_config['active_slew_reference'] == 'none':
            bounds = None
        shift = np.zeros(5)
        if self.torque_config['active_slew_reference'] == 'model_bias_relative' and bounds is not None:
            if previous_bias is None:
                raise HardwareMpcError('missing previous model bias for active torque slew')
            # tau=M*ddq+b. Slew-limit the acceleration-producing residual, not
            # changes of b caused by torso motion. Absolute total/FF limits
            # and the final forward-model acceleration guard remain enforced.
            shift = bias-finite_vector(previous_bias,5,'previous model bias')
            bounds = (bounds[0]+shift,bounds[1]+shift)
        slew_diagnostics = dict(
            torque_slew_reference=self.torque_config['active_slew_reference'],
            torque_model_bias_nm=bias, torque_previous_model_bias_nm=previous_bias,
            torque_slew_bias_shift_nm=shift,
            torque_slew_bounds_nm=None if bounds is None else np.asarray(bounds))
        # Preserve the evaluated support/bounds even when every candidate is
        # rejected; a fault must remain independently diagnosable.
        self.last_diagnostics.update(slew_diagnostics)
        gain = np.linalg.solve(mass, np.eye(5))
        forward = lambda tau: gain @ (tau-bias)
        pd = self.torque_config["kp"]*(qref-q)+self.torque_config["kd"]*(dqref-dq)
        retained_pd = np.zeros(5)
        if self.torque_config['anchor_shoulder_yaw_reference']:
            retained_pd[2] = pd[2]
        acceleration_bounds = (self.policy.first_step_acceleration_bounds(q,dq)
                               if self.torque_config['enforce_mapper_state_envelope'] else None)
        active_brake_bounds = getattr(self, '_active_braking_acceleration_bounds', None)
        if active_brake_bounds is not None:
            brake_lo, brake_hi = active_brake_bounds
            if acceleration_bounds is None:
                acceleration_bounds = (brake_lo.copy(), brake_hi.copy())
            else:
                acceleration_bounds = (np.maximum(acceleration_bounds[0], brake_lo),
                                       np.minimum(acceleration_bounds[1], brake_hi))
        self.last_diagnostics['final_acceleration_bounds_rad_s2'] = acceleration_bounds
        try:
            # The optimizer requests acceleration.  M*ddq+b is therefore the
            # desired TOTAL torque.  Packet feedforward subtracts PD below so
            # firmware-side PD is added exactly once.
            total, mapping = self.mapper.compute(forward, ddq, mass@ddq+bias,
                                                safe_hold=bias-self.torque_config["kd"]*dq,
                                                previous=self._previous_total, bounds=bounds,
                                                acceleration_bounds=acceleration_bounds,
                                                affine_gain=gain,
                                                forward_batch=lambda taus: (taus-bias)@gain.T)
        except NoModelTorque as exc:
            self.last_diagnostics.update(mapper=exc.trace, torque_output_authorized=False)
            self.last_diagnostics = json_values(self.last_diagnostics)
            raise
        self._previous_total = total.copy()
        self._prepared_forward = (mass, bias)
        executed_total = total + retained_pd
        return dict(inverse_dynamics=inverse, mapper=mapping,
                    final_acceleration_bounds_rad_s2=acceleration_bounds,
                    **slew_diagnostics,
                    tau_nominal_ff_nm=inverse["tau_model_nm"], tau_pd_at_feedback_nm=pd,
                    tau_model_selected_nm=total,
                    retained_firmware_pd_nm=retained_pd,
                    tau_ff_candidate_nm=executed_total-pd,
                    tau_total_estimated_at_feedback_nm=executed_total,
                    expected_kp=self.torque_config["kp"], expected_kd=self.torque_config["kd"],
                    torque_output_authorized=False, torque_estimate_feedback_used_for_control=False)

    def step(self, arm_slots, imu_quaternion_wxyz, yaw0_rad, dt):
        start = time.perf_counter_ns()
        start_cpu = time.thread_time_ns()
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
        self._recovery_elapsed_s += float(dt)
        if self.torque_config['recovery_envelope_enabled']:
            self.policy.set_recovery_time(self._recovery_elapsed_s)
        bounds = self._next_total_bounds
        prepared_dynamics = self.inverse.linear_dynamics(q, dq, self._current_base)
        if self.torque_config['planning_actuation_constraints_enabled']:
            self.policy.set_local_actuation_constraints(
                *prepared_dynamics, self.mapper.limit,
                finite_vector(self.torque_config.get('tau_ff_abs_nm',self.mapper.limit),5,'FF limit'),
                dq)
        else:
            self.policy.clear_local_actuation_constraints()
        braking = dict(enabled=self.torque_config['predictive_braking_enabled'])
        braking_weights = np.zeros(5)
        self._active_braking_acceleration_bounds = None
        if braking['enabled']:
            c = self.torque_config
            # The trigger uses the normal operating box. The policy may open
            # its separately configured outer box only while recovering.
            braking_weights, brake_bounds, detail = self.brake.update(
                q,dq,self.policy.joint_limits[:,0],self.policy.joint_limits[:,1],
                deceleration=c['braking_deceleration_rad_s2'], reaction_s=c['braking_reaction_s'],
                margin_rad=np.deg2rad(c['braking_margin_deg']), weight=c['braking_velocity_weight'],
                max_ddq=self.policy.max_ddq,
                minimum_deceleration=c['braking_min_deceleration_rad_s2'])
            self._active_braking_acceleration_bounds = brake_bounds if detail['latched'] else None
            braking.update(detail)
        self.policy.set_first_acceleration_bounds(
            *(self._active_braking_acceleration_bounds or (None,None)))
        self.policy.set_braking_velocity_cost(braking_weights)
        try:
            qref, dqref, ddq = self.policy.compute_action({"current_q": q, "current_dq": dq, "dt": .006}, helpers)
        except Exception:
            self.last_diagnostics = json_values(dict(mpc_active=True, measured_q_rad=q,
                measured_dq_rad_s=dq, torque_output_authorized=False,
                predictive_braking=braking,
                reentry=getattr(self.policy,'reentry_diagnostics',None)))
            raise
        if self.torque_config['anchor_shoulder_yaw_reference']:
            # The endpoint-upright objective is almost insensitive to rotation
            # about the arm/bottle vertical axis.  Do not let the one-step
            # packet reference follow a measured shoulder-yaw drift.  This is
            # the established nominal A3 angle (zero for the current profile),
            # while the acceleration MPC still uses the measured state.
            qref = np.asarray(qref, dtype=float).copy()
            dqref = np.asarray(dqref, dtype=float).copy()
            qref[2] = self.nominal[2]
            dqref[2] = 0.0
        qp = self.policy.get_last_diagnostics(copy_data=False)
        self.last_diagnostics = dict(mpc_active=True, controller_kind=ACTUATION,
            measured_q_rad=q, measured_dq_rad_s=dq, mpc_initial_state=np.r_[q, dq],
            raw_mpc_ddq_rad_s2=ddq, one_step_q_reference_rad=qref,
            predictive_braking=braking,
            one_step_dq_reference_rad_s=dqref, predictor=self._extra,
            shoulder_yaw_reference_anchored=
                self.torque_config['anchor_shoulder_yaw_reference'],
            gravity_error_before_m_s2=qp["gravity_error"], feedback_dt_s=float(dt),
            command_integration_dt_s=.006, mpc={key: value for key, value in qp.items()
                if key not in {"working_states", "working_inputs", "predicted_states", "predicted_inputs",
                               "disturbance_prediction", "interval_disturbance_prediction"}})
        self.last_diagnostics['reentry'] = getattr(self.policy,'reentry_diagnostics',None)
        self.last_diagnostics['actuation_planning'] = dict(
            absolute_torque_constraints=self.torque_config['planning_actuation_constraints_enabled'],
            torque_rate_constraints=False,
            local_rows=0 if self.policy._last_local_actuation is None else len(self.policy._last_local_actuation[1]))
        if not qp["solved"] or qp["fallback_used"]:
            self.last_diagnostics = json_values(self.last_diagnostics)
            raise HardwareMpcError(f"measured-state MPC rejected: {qp['solver_status']}")
        bounds, self._next_total_bounds = self._next_total_bounds, None
        self.last_diagnostics.update(self._torque(q, dq, qref, dqref, ddq, self._current_base,
                                                  bounds, prepared_dynamics))
        self.last_diagnostics["controller_core_ms"] = (time.perf_counter_ns()-start)*1e-6
        self.last_diagnostics["controller_thread_cpu_ms"] = (time.thread_time_ns()-start_cpu)*1e-6
        # Immutable JSON-native audit snapshot, including every candidate.
        # Native encoding avoids a Python recursive walk on the 6 ms thread;
        # nonfinite diagnostics remain null, exactly as in json_values.
        self.last_diagnostics = orjson.loads(orjson.dumps(self.last_diagnostics,
            option=orjson.OPT_SERIALIZE_NUMPY, default=json_values))
        return qref.copy(), dqref.copy(), self.last_diagnostics


class HardwareTorquePreviewPlan(HardwarePidPlan):
    """Offline complete entry/control/release plan; no DDS authority.

    Total right-arm torque is rate-bounded through entry/release transitions;
    active control optionally bounds the residual relative to model bias.
    The final command is checked again after projection. Weight release retains
    support feedforward until weight reaches zero, avoiding a torque drop at 18s.
    """
    def __init__(self, *args, mpc_start_s=3., **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(mpc_start_s) or not 3.<=mpc_start_s<=5.:
            raise ValueError('MPC start must be within the existing 3..5s baseline')
        self.mpc_start_s = float(mpc_start_s)
        # The field yaw anchor deliberately uses a much softer packet PD than
        # the A3 posture hold.  Apply that single-axis overlay to the actual
        # command gains before checking that execution matches the evaluated
        # torque model; the other twelve slots are unchanged.
        if self.controller.torque_config['anchor_shoulder_yaw_reference']:
            self.kp[7] = self.controller.torque_config['kp'][2]
            self.kd[7] = self.controller.torque_config['kd'][2]
        if (not np.allclose(self.kp[5:10], self.controller.torque_config["kp"])
                or not np.allclose(self.kd[5:10], self.controller.torque_config["kd"])):
            raise ValueError("plan PD must match the evaluated torque model")
        self._last_total = None
        self._last_bias = None

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
        c._next_slew_bias = None
        active = self.mpc_start_s <= task_s < 18
        if (self._last_total is not None and active
                and c.torque_config['active_slew_reference'] != 'none'):
            delta = c.torque_config["transition_rate_nm_s"]*min(float(dt), .006)
            c._next_total_bounds = (self._last_total-delta, self._last_total+delta)
            c._next_slew_bias = self._last_bias
        if 3. <= task_s < self.mpc_start_s:
            # Fixed-posture settling before the configured MPC handoff:
            # fixed posture plus the same bounded support torque as entry.
            # Do not jump straight from a moving arm into acceleration MPC.
            frame = dict(stage='stationary_baseline',q_rad=self.target.copy(),dq_rad_s=np.zeros(13),
                         kp=self.kp,kd=self.kd,weight=1.,terminal=False,
                         diagnostics=dict(pid_active=False,mpc_active=False,entry_settle=True))
            self._last_q = self.target.copy()
        else:
            frame = super().sample(task_s, measured_slots, measured_dq, imu_quaternion, yaw0_rad, dt)
        pd = self.kp[5:10]*(frame["q_rad"][5:10]-q)+self.kd[5:10]*(frame["dq_rad_s"][5:10]-dq)
        if not active:
            c._horizon = None
            inverse = c.inverse.compute(q, dq, np.zeros(5), base)
            entry = float(np.clip(task_s/3., 0., 1.))
            total = inverse["tau_model_nm"]*entry+pd
            model_total = total.copy()
            frame["diagnostics"].update(controller_kind=ACTUATION, inverse_dynamics=inverse,
                torque_output_authorized=False, transition_model=True)
        else:
            total = np.asarray(frame["diagnostics"]["tau_total_estimated_at_feedback_nm"])
            model_total = np.asarray(frame["diagnostics"]["tau_model_selected_nm"])
        limit = c.mapper.limit
        total = np.clip(total, -limit, limit)
        if self._last_total is not None and not active:
            delta = c.torque_config["transition_rate_nm_s"]*min(float(dt), .006)
            # Scalar interpolation preserves coupled forward-model behaviour;
            # clipping five axes separately can create an unchecked direction.
            difference = total-self._last_total
            ratio = min(1., float(np.min(delta/np.maximum(np.abs(difference), 1e-12))))
            total = self._last_total+ratio*difference
        self._last_total = total.copy()
        c._previous_total = model_total.copy()
        mass, bias = (c._prepared_forward if active else
                      c.inverse.linear_dynamics(q, dq, base))
        model_acceleration = np.linalg.solve(mass, model_total-bias)
        closed_loop_acceleration = np.linalg.solve(mass, total-bias)
        self._last_bias = bias.copy()
        # Entry is the existing position ramp, not acceleration-controlled MPC.
        # During active/release, reject a transition that invalidates the model
        # envelope instead of calling the pre-projection result accepted.
        if (frame["weight"] > 0 and task_s >= self.mpc_start_s
                and np.max(np.abs(model_acceleration)) > c.mapper.acc_limit+1e-9):
            raise HardwareMpcError("post-transition total torque fails forward-model envelope")
        retained = total-model_total if active else np.zeros(5)
        frame["diagnostics"].update(tau_pd_at_feedback_nm=pd.tolist(),
            torque_model_bias_nm=bias.tolist(),
            feedback_dt_s=float(dt),
            tau_model_selected_nm=model_total.tolist(),
            retained_firmware_pd_nm=retained.tolist(),
            tau_total_estimated_at_feedback_nm=total.tolist(),
            tau_ff_candidate_nm=(total-pd).tolist(), expected_kp=self.kp[5:10].tolist(),
            expected_kd=self.kd[5:10].tolist(),
            post_transition_ddq_rad_s2=model_acceleration.tolist(),
            estimated_closed_loop_ddq_rad_s2=closed_loop_acceleration.tolist(),
            acceptance="conditional_forward_model_only", torque_output_authorized=False)
        frame['diagnostics']['mpc_start_s'] = self.mpc_start_s
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
