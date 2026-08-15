"""Shared right-arm controller construction without platform adapter imports.

This module is deliberately independent of MuJoCo simulation helpers and the
Unitree hardware adapter.  Both adapters consume the same controller factory;
``sim_support`` re-exports these symbols for compatibility with existing
simulation callers.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


@dataclass
class ArmControllerSetup:
    """右臂控制器及其运行期配置；用于把参数装配从主循环下沉。"""

    policy: object
    policy_type: str
    notes: str
    metadata: dict
    acceleration_controller: bool
    lqr_cost_definition: Optional[dict]
    execution_perturbation: float
    execution_regularization: float
    execution_second_pass_error_threshold: float
    execution_max_joint_error: float
    execution_max_abs_qacc: float
    execution_enable_second_pass: bool
    execution_safety_rescue_passes: int
    execution_hold_last_safe: bool
    ddq_saturation_limit: float


def create_arm_controller(config, controller_name, default_q, control_dt):
    """【半核心代码】从 YAML 参数创建 PID/LQR/MPC，并集中生成实验元数据。"""
    from arm_lqr import ArmLQRPolicy
    from arm_mpc import ArmMPCPolicy
    from arm_pid import ArmPIDPolicy

    controller_name = str(controller_name).lower()
    if controller_name not in {"pid", "lqr", "mpc"}:
        raise ValueError(
            f"arm_controller={controller_name!r} 无效，只能选择 'pid'、'lqr' 或 'mpc'。"
        )

    perturbation = float(config.get("ddq_execution_perturbation", 0.1))
    regularization = float(config.get("ddq_execution_regularization", 5.0))
    second_pass_threshold = float(
        config.get("ddq_execution_second_pass_error_threshold", 5.0)
    )
    max_joint_error = float(config.get("ddq_execution_max_joint_error", 4.0))
    max_abs_qacc = float(config.get("ddq_execution_max_abs_qacc", 8.0))
    enable_second_pass = bool(
        config.get(
            f"{controller_name}_execution_enable_second_pass",
            controller_name == "lqr",
        )
    )
    safety_rescue_passes = int(
        config.get(
            f"{controller_name}_execution_safety_rescue_passes",
            2 if controller_name == "lqr" else 0,
        )
    )
    hold_last_safe = bool(
        config.get(
            f"{controller_name}_execution_hold_last_safe",
            controller_name == "mpc",
        )
    )
    acc_filter_key = (
        "mpc_torso_acc_filter_alpha"
        if controller_name == "mpc"
        else "ddq_torso_acc_filter_alpha"
    )
    alpha_filter_key = (
        "mpc_torso_alpha_filter_alpha"
        if controller_name == "mpc"
        else "ddq_torso_alpha_filter_alpha"
    )
    torso_metadata = {
        "torso_acc_filter_alpha": float(
            config.get(acc_filter_key, 0.20)
        ),
        "torso_alpha_filter_alpha": float(
            config.get(alpha_filter_key, 0.20)
        ),
        "torso_acc_limit": float(config.get("ddq_torso_acc_limit", 30.0)),
        "torso_alpha_limit": float(config.get("ddq_torso_alpha_limit", 40.0)),
    }
    execution_metadata = {
        "forward_dynamics_perturbation_nm": perturbation,
        "forward_dynamics_regularization": regularization,
        "forward_dynamics_second_pass_error_threshold": second_pass_threshold,
        "forward_dynamics_max_joint_error": max_joint_error,
        "forward_dynamics_max_abs_qacc": max_abs_qacc,
        "forward_dynamics_enable_second_pass": enable_second_pass,
        "forward_dynamics_safety_rescue_passes": safety_rescue_passes,
        "forward_dynamics_hold_last_safe": hold_last_safe,
        "forward_dynamics_candidate_evaluation": "mj_forwardSkip_mjSTAGE_VEL_skip_sensors",
        "forward_dynamics_candidate_selection": "model_ranked_then_best_real_safe_candidate_among_at_least_two_validations",
        "forward_dynamics_evaluations_per_pass": "5 torque perturbations plus 1-4 on-demand candidate validations",
        "forward_dynamics_scratch_copy": "physical_state_warmstart_and_external_inputs_only_before_full_mj_forward",
        "constraint_force_reconstruction": "mj_mulJacTVec_with_selected_constraint_rows",
    }

    lqr_cost_definition = None
    ddq_saturation_limit = np.inf
    if controller_name == "lqr":
        kwargs = {
            "horizon": int(config.get("lqr_horizon", 12)),
            "q_ee_acc": float(config.get("lqr_q_ee_acc", 1.0)),
            "q_ee_alpha": float(config.get("lqr_q_ee_alpha", 0.05)),
            "q_position": float(config.get("lqr_q_position", 20.0)),
            "q_gravity": float(config.get("lqr_q_gravity", 30.0)),
            "q_posture": config.get("lqr_q_posture", 0.4),
            "q_vel": float(config.get("lqr_q_vel", 0.02)),
            "r_ddq": float(config.get("lqr_r_ddq", 0.25)),
            "terminal_scale": float(config.get("lqr_terminal_scale", 2.0)),
            "reg": float(config.get("lqr_reg", 1e-6)),
            "max_ddq": float(config.get("lqr_max_ddq", 3.0)),
            "max_dq": float(config.get("lqr_max_dq", 1.0)),
            "ddq_rate_limit": float(config.get("lqr_ddq_rate_limit", 350.0)),
            "ddq_smoothing_alpha": float(
                config.get("lqr_ddq_smoothing_alpha", 0.45)
            ),
            "joint_limit_margin": float(config.get("lqr_joint_limit_margin", 0.25)),
            "joint_limit_stiffness": float(
                config.get("lqr_joint_limit_stiffness", 8.0)
            ),
            "joint_limit_damping": float(
                config.get("lqr_joint_limit_damping", 2.0)
            ),
        }
        policy = ArmLQRPolicy(default_q=default_q, control_dt=control_dt, **kwargs)
        lqr_cost_definition = policy.get_cost_definition()
        policy_type = "ArmLQRPolicy"
        notes = "finite-horizon time-varying LQR with per-joint posture weights, torso-relative end-effector position cost, directed 3D gravity error, fully bypassed ddq post-processing, protected torso disturbance inputs, and validated local MuJoCo forward-dynamics torque mapping initialized by non-friction-constraint-aware inverse dynamics plus joint-space PD"
        metadata = {
            "lqr_config": {
                **kwargs,
                "control_dt": policy.control_dt,
                **torso_metadata,
                **execution_metadata,
                "torque_control": "mujoco_local_forward_dynamics_mapping_from_inverse_dynamics_nominal",
                "constraint_aware_tau_formula": "qfrc_inverse + qfrc_constraint_nonfriction",
                "constraints_added_back": "contact + joint/tendon limit + equality",
                "constraints_excluded_from_addback": "FRICTION_DOF + FRICTION_TENDON",
                "forward_dynamics_mapping": "ddq_right ~= ddq_baseline + G_tau * delta_tau, then rank candidates with the local model and validate on demand with mj_forwardSkip",
                "forward_dynamics_validation_scales": [1.0, 0.5, 0.25, 0.125],
                "forward_dynamics_candidate_selection": "rank with local model, validate at least two candidates, then select the minimum-error real-safe candidate; continue only when both fail",
                "forward_dynamics_evaluations_per_step": "8-10 in the first pass (1 full baseline plus 5 perturbations plus 2-4 validations); plus 7-9 when the second pass triggers; up to two additional 7-9-evaluation safety-rescue passes only if final qacc exceeds the limit",
                "uncontrolled_qacc_assumption": 0.0,
                "gravity_error": "directed_3d",
                "position_reference_q": np.asarray(default_q).copy(),
                "torso_acc_source": "mujoco_imu_accelerometer_world_without_gravity",
                "torso_alpha_source": "finite_difference_world_angular_velocity",
                "position_reference_frame": "torso_imu",
                "end_effector_velocity_cost_enabled": False,
                "ddq_post_process": "fully_bypassed",
                "ddq_hard_clip_enabled": False,
                "joint_limit_guard": "disabled",
                "ddq_rate_limit_enabled": False,
                "ddq_smoothing_enabled": False,
                "ddq_tracking": "6ms_velocity_difference_aligned_between_consecutive_arm_updates",
                "cost_tracking": "one_step_model_vs_realized_next_arm_update",
            }
        }
    elif controller_name == "mpc":
        configured_predictor = config.get("disturbance_predictor")
        predictor_name = (
            "template"
            if configured_predictor is None
            and bool(config.get("mpc_disturbance_feedforward_enabled", False))
            else (
                "zoh"
                if configured_predictor is None
                else str(configured_predictor).strip().lower()
            )
        )
        feedforward_enabled = (
            predictor_name in {"template", "full_task_template"}
        )
        q_min = np.deg2rad(
            np.asarray(
                config.get(
                    "mpc_q_min_deg", [-20.0, -10.0, -20.0, -20.0, -20.0]
                ),
                dtype=np.float64,
            )
        )
        q_max = np.deg2rad(
            np.asarray(
                config.get("mpc_q_max_deg", [10.0, 5.0, 5.0, 20.0, 20.0]),
                dtype=np.float64,
            )
        )
        if q_min.shape != (5,) or q_max.shape != (5,) or np.any(q_min >= q_max):
            raise ValueError(
                "mpc_q_min_deg/mpc_q_max_deg 必须是长度为 5 且下界小于上界的数组。"
            )
        q_operating_margin = np.deg2rad(
            np.asarray(
                config.get("mpc_q_operating_margin_deg", np.zeros(5)),
                dtype=np.float64,
            )
        )
        if (
            q_operating_margin.shape != (5,)
            or np.any(q_operating_margin < 0.0)
            or np.any(2.0 * q_operating_margin >= q_max - q_min)
        ):
            raise ValueError(
                "mpc_q_operating_margin_deg 必须是长度为 5 的非负数组，"
                "且两侧裕量之和必须小于对应安全范围。"
            )
        kwargs = {
            "horizon": int(config.get("mpc_horizon", 12)),
            "q_ee_acc": config.get("mpc_q_ee_acc", 1.0),
            "q_ee_alpha": config.get("mpc_q_ee_alpha", 0.075),
            # 默认关闭；需要时可用标量、3 维对角权重或 3x3 矩阵配置。
            "q_ee_omega": config.get("mpc_q_ee_omega", 0.0),
            "q_gravity": config.get("mpc_q_gravity", 30.0),
            "q_posture": config.get("mpc_q_posture", 0.05),
            "q_vel": config.get("mpc_q_vel", 0.02),
            "r_ddq": config.get("mpc_r_ddq", 0.25),
            "terminal_scale": float(config.get("mpc_terminal_scale", 2.0)),
            "reg": float(config.get("mpc_qp_regularization", 1e-6)),
            "max_dq": float(config.get("mpc_max_dq", 1.0)),
            "max_ddq": float(config.get("mpc_max_ddq", 8.0)),
            "joint_limits": np.column_stack([q_min, q_max]),
            "joint_limit_margin": q_operating_margin,
            "solver_max_iter": int(config.get("mpc_osqp_max_iter", 400)),
            "solver_check_termination": int(
                config.get("mpc_osqp_check_termination", 25)
            ),
            "solver_rho": float(config.get("mpc_osqp_rho", 0.1)),
            "solver_adaptive_rho": bool(
                config.get("mpc_osqp_adaptive_rho", True)
            ),
            "solver_scaled_termination": bool(
                config.get("mpc_osqp_scaled_termination", False)
            ),
            "solver_eps_abs": float(config.get("mpc_osqp_eps_abs", 1e-3)),
            "solver_eps_rel": float(config.get("mpc_osqp_eps_rel", 1e-3)),
            "solver_polishing": bool(config.get("mpc_osqp_polishing", False)),
            "failure_braking_gain": float(
                config.get("mpc_failure_braking_gain", 4.0)
            ),
            "failure_posture_gain": float(
                config.get("mpc_failure_posture_gain", 4.0)
            ),
            "failure_max_ddq_scale": float(
                config.get("mpc_failure_max_ddq_scale", 0.5)
            ),
        }
        policy = ArmMPCPolicy(default_q=default_q, control_dt=control_dt, **kwargs)
        policy_type = "ArmMPCPolicy"
        notes = (
            "sparse constrained right-arm MPC with ddq input, "
            + (
                "absolute-task-time full-task disturbance feedforward, "
                if predictor_name == "full_task_template"
                else "heading-frame phase-template base-disturbance feedforward, "
                if feedforward_enabled
                else "zero-order-held measured base motion, "
            )
            + "configurable world-frame end-effector angular-velocity cost, "
            + "2D gravity error, no end-effector position cost, and validated "
            + "inverse/forward-dynamics torque execution"
        )
        ddq_saturation_limit = kwargs["max_ddq"]
        metadata = {
            "mpc_config": {
                **kwargs,
                "control_dt": control_dt,
                **torso_metadata,
                **execution_metadata,
                "solver": "OSQP",
                "base_prediction": (
                    "continuous_heading_to_world_absolute_task_template_v2"
                    if predictor_name == "full_task_template"
                    else (
                        "heading_to_world_measurement_anchored_phase_template"
                        if feedforward_enabled
                        else "zero_order_hold_current_measurement"
                    )
                ),
                "gravity_error": "signed_2d_xy",
                "torso_relative_position_cost": False,
            }
        }
    else:
        kwargs = {
            "kp_pose": config.get("pid_kp_pose", [1.20, 1.20]),
            "kd_pose": config.get("pid_kd_pose", [1.20, 1.20]),
            "ki_pose": config.get("pid_ki_pose", [0.0, 0.0]),
            "posture_gain": config.get(
                "pid_posture_gain", [1.15, 1.15, 2.10, 1.15, 0.95]
            ),
            "finite_diff_eps": float(config.get("pid_finite_diff_eps", 1e-4)),
            "damping": float(config.get("pid_damping", 0.15)),
            "integral_limit": float(config.get("pid_integral_limit", 0.20)),
            "max_dq": float(config.get("pid_max_dq", 0.48)),
            "de_g_alpha": float(config.get("pid_de_g_alpha", 0.07)),
        }
        policy = ArmPIDPolicy(default_q=default_q, control_dt=control_dt, **kwargs)
        policy_type = "ArmPIDPolicy"
        notes = "tuning_v7: continue monotonic conservative tuning with a slightly lower Kp, larger damped-pinv damping, tighter max_dq, slightly stronger de_g filtering, and a mild posture-regularization increase to further reduce right-hand acceleration while keeping tilt small"
        metadata = {"pid_config": {"default_q": np.asarray(default_q).copy(), **kwargs}}

    return ArmControllerSetup(
        policy=policy,
        policy_type=policy_type,
        notes=notes,
        metadata=metadata,
        acceleration_controller=controller_name in {"lqr", "mpc"},
        lqr_cost_definition=lqr_cost_definition,
        execution_perturbation=perturbation,
        execution_regularization=regularization,
        execution_second_pass_error_threshold=second_pass_threshold,
        execution_max_joint_error=max_joint_error,
        execution_max_abs_qacc=max_abs_qacc,
        execution_enable_second_pass=enable_second_pass,
        execution_safety_rescue_passes=safety_rescue_passes,
        execution_hold_last_safe=hold_last_safe,
        ddq_saturation_limit=ddq_saturation_limit,
    )
