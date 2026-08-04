import gc
import os
import time
from contextlib import nullcontext

# viewer 使用 GLFW，离屏视频渲染改用独立 EGL 上下文，避免退出时 GLFW 重复销毁。
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

# run.sh 已在导入原生数值库前固定线程环境；这里再明确限制 Torch
# intra-op/inter-op，避免策略推理为很小的网络启动额外工作线程。
_runtime_torch_threads = max(
    1, int(os.environ.get("MPC_CONTROL_NUM_THREADS", "1"))
)
torch.set_num_threads(_runtime_torch_threads)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # 同一解释器中若已有并行工作启动，Torch 不允许再次修改 inter-op。
    pass

from kinematics_helper import KinematicsHelper
from robot_model_backend import CppRightArmRneaBackend, create_prediction_backend
from right_arm_runtime import (
    CppDdqTorqueMapper,
    CppRightArmExecutor,
    RightArmSimProcess,
    SimProcessShadowValidator,
)
from sim_support import (
    ArmCommandDelayLine,
    HeadingHoldController,
    PhaseDisturbancePredictor,
    PerformanceMonitor,
    RneaOtherAccelerationEstimator,
    TorsoAccelerationFilter,
    apply_computed_torque_control,
    build_performance_runtime_config,
    build_right_arm_control_record,
    build_right_arm_observation,
    build_run_metadata,
    close_renderer,
    create_arm_controller,
    create_eval_run_dir,
    draw_debug_axes,
    finalize_run,
    forward_dynamics_result_from_cpp_mapper_response,
    get_gravity_orientation,
    init_eval_buffers,
    inverse_dynamics_result_from_sim_process,
    make_video_camera,
    make_video_renderer,
    pd_control,
    quat_to_yaw_wxyz,
    record_eval_step,
    resolve_right_arm_control_context,
    resolve_scene_ids,
    update_torso_motion_state,
)


if __name__ == "__main__":
    # ==============================
    # 1. 读取配置【非核心代码】
    # 决定仿真模型、RL 行走策略、右臂控制器类型和控制频率。
    # 这部分更像实验入口与参数装配；需要知道有哪些配置项，但不用逐行死记。
    # ==============================
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument("--headless", action="store_true", help="不启动交互 viewer，适合服务器和自动测试")
    parser.add_argument("--no-video", action="store_true", help="不创建离屏视频")
    parser.add_argument("--smoke-test", action="store_true", help="只运行一个步态周期并关闭 viewer/video")
    parser.add_argument("--run-label", default="", help="追加到评估目录名的简短实验标签")
    parser.add_argument(
        "--evaluation-group",
        default="",
        help="把多组 A/B 运行收进同一个 evaluation 子目录",
    )
    parser.add_argument(
        "--mpc-command-delay-ms",
        type=float,
        default=None,
        help="覆盖 MPC 命令从状态采样到 2 ms 执行拍激活的仿真延迟",
    )
    parser.add_argument(
        "--right-arm-runtime-mode",
        choices=("sync", "process", "shadow"),
        default=None,
        help="覆盖右臂执行结构：同步C ABI、独立C++进程或逐拍shadow",
    )
    args = parser.parse_args()
    config_file = args.config_file
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = config_file if os.path.isabs(config_file) else os.path.join(repo_dir, "configs", config_file)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        arm_waist_kps = np.array(config["arm_waist_kps"], dtype=np.float32)
        arm_waist_kds = np.array(config["arm_waist_kds"], dtype=np.float32)
        arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        arm_controller = config.get("arm_controller", "pid").lower()
        if arm_controller not in {"pid", "lqr", "mpc"}:
            raise ValueError(
                f"arm_controller={arm_controller!r} 无效，只能选择 'pid'、'lqr' 或 'mpc'。"
            )
        arm_control_decimation = int(config.get("arm_control_decimation", 2))
        mpc_ddq_execution_mode = str(
            config.get("mpc_ddq_execution_mode", "every_step")
        ).lower()
        mpc_prediction_kinematics_backend = str(
            config.get("mpc_prediction_kinematics_backend", "mujoco")
        ).strip().lower()
        ddq_nominal_inverse_dynamics_backend = str(
            config.get("ddq_nominal_inverse_dynamics_backend", "mujoco")
        ).strip().lower()
        ddq_pinocchio_friction_breakaway_steps = float(
            config.get("ddq_pinocchio_friction_breakaway_steps", 5.0)
        )
        ddq_rnea_other_qacc_mode = str(
            config.get("ddq_rnea_other_qacc_mode", "zero")
        ).strip().lower()
        ddq_rnea_other_qacc_filter_alpha = float(
            config.get("ddq_rnea_other_qacc_filter_alpha", 0.5)
        )
        ddq_rnea_other_qacc_trend_window = int(
            config.get("ddq_rnea_other_qacc_trend_window", 4)
        )
        ddq_rnea_other_qacc_trend_lead_steps = float(
            config.get("ddq_rnea_other_qacc_trend_lead_steps", 1.0)
        )
        ddq_rnea_other_qacc_limit = float(
            config.get("ddq_rnea_other_qacc_limit", 100.0)
        )
        ddq_rnea_other_qacc_blend = float(
            config.get("ddq_rnea_other_qacc_blend", 1.0)
        )
        right_arm_executor_backend = str(
            config.get("right_arm_executor_backend", "python")
        ).strip().lower()
        requested_right_arm_execution_runtime = str(
            config.get("right_arm_execution_runtime", "sync")
            if args.right_arm_runtime_mode is None
            else args.right_arm_runtime_mode
        ).strip().lower()
        ddq_forward_dynamics_backend = str(
            config.get("ddq_forward_dynamics_backend", "python")
        ).strip().lower()
        if ddq_forward_dynamics_backend not in {"python", "cpp"}:
            raise ValueError(
                "ddq_forward_dynamics_backend 必须是 python 或 cpp。"
            )
        right_arm_executor_output_semantics = str(
            config.get(
                "right_arm_executor_output_semantics",
                "host_full_torque",
            )
        ).strip().lower()
        mpc_command_delay_ms = float(
            config.get("mpc_command_delay_ms", 0.0)
            if args.mpc_command_delay_ms is None
            else args.mpc_command_delay_ms
        )
        if right_arm_executor_backend not in {"python", "cpp"}:
            raise ValueError(
                "right_arm_executor_backend 必须是 python 或 cpp。"
            )
        if requested_right_arm_execution_runtime not in {
            "sync",
            "process",
            "shadow",
        }:
            raise ValueError(
                "right_arm_execution_runtime 必须是 sync、process 或 shadow。"
            )
        if right_arm_executor_output_semantics not in {
            "host_full_torque",
            "device_pd",
        }:
            raise ValueError(
                "right_arm_executor_output_semantics 必须是 "
                "host_full_torque 或 device_pd。"
            )
        if not np.isfinite(mpc_command_delay_ms) or mpc_command_delay_ms < 0.0:
            raise ValueError("mpc_command_delay_ms 必须是有限非负数。")
        if arm_controller != "mpc" and mpc_command_delay_ms > 0.0:
            raise ValueError("当前命令延迟实验只允许 arm_controller=mpc。")
        if (
            not np.isfinite(ddq_pinocchio_friction_breakaway_steps)
            or ddq_pinocchio_friction_breakaway_steps < 0.0
        ):
            raise ValueError(
                "ddq_pinocchio_friction_breakaway_steps 必须是有限非负数。"
            )
        if mpc_prediction_kinematics_backend not in {
            "mujoco",
            "pinocchio",
            "cpp_pinocchio",
        }:
            raise ValueError(
                "mpc_prediction_kinematics_backend 必须是 "
                "mujoco、pinocchio 或 cpp_pinocchio。"
            )
        if ddq_nominal_inverse_dynamics_backend not in {
            "mujoco",
            "pinocchio_shadow",
            "pinocchio",
            "cpp_pinocchio",
        }:
            raise ValueError(
                "ddq_nominal_inverse_dynamics_backend 必须是 "
                "mujoco、pinocchio_shadow、pinocchio 或 cpp_pinocchio。"
            )
        valid_execution_modes = {
            "every_step",
            "twice_per_interval",
            "policy_update",
        }
        if mpc_ddq_execution_mode not in valid_execution_modes:
            raise ValueError(
                "mpc_ddq_execution_mode 必须是 every_step、"
                "twice_per_interval 或 policy_update。"
            )
        cmd_nominal = np.array(config["cmd_init"], dtype=np.float32)
        heading_control_enabled = bool(config.get("heading_control_enabled", True))
        heading_filter_cycles = float(config.get("heading_filter_cycles", 1.0))
        heading_kp = float(config.get("heading_kp", 0.6))
        heading_kd = float(config.get("heading_kd", 0.1))
        heading_max_yaw_rate = float(config.get("heading_max_yaw_rate", 0.25))

    # ==============================
    # 2. 初始化主循环状态【非核心代码】
    # RL 动作缓存、腿部目标、观测向量、实验时序参数。
    # 这部分属于运行准备；知道变量用途即可，不是控制算法本身的核心。
    # ==============================
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    gait_period = 0.8
    warmup_cycles = int(config.get("warmup_cycles", 3))
    evaluation_cycles = int(config.get("evaluation_cycles", 8))
    cooldown_cycles = int(config.get("cooldown_cycles", 2))
    viewer_enabled = bool(config.get("viewer_enabled", True)) and not args.headless
    record_video = bool(config.get("record_video", True)) and not args.no_video
    if args.smoke_test:
        warmup_cycles = 0
        evaluation_cycles = 1
        cooldown_cycles = 0
        viewer_enabled = False
        record_video = False
    if warmup_cycles < 0 or evaluation_cycles <= 0 or cooldown_cycles < 0:
        raise ValueError("warmup/evaluation/cooldown 周期必须分别满足 >=0、>0、>=0。")
    total_cycles = warmup_cycles + evaluation_cycles + cooldown_cycles
    eval_start_time = warmup_cycles * gait_period
    eval_end_time = (warmup_cycles + evaluation_cycles) * gait_period
    eval_duration = total_cycles * gait_period
    run_label = "".join(
        character
        for character in str(args.run_label).strip().lower().replace(" ", "_")
        if character.isalnum() or character in {"_", "-"}
    )
    experiment_name = f"left_fixed_right_{arm_controller}"
    evaluation_group = "".join(
        character
        for character in str(args.evaluation_group).strip().lower().replace(" ", "_")
        if character.isalnum() or character in {"_", "-"}
    ) or experiment_name
    buffers = init_eval_buffers()

    # ==============================
    # 3. 加载 MuJoCo 模型与 RL 行走策略【半核心】
    # MuJoCo 负责整机物理推进；TorchScript policy 负责下肢 locomotion。
    # 对论文/面试来说，需要知道“谁负责物理、谁负责走路”，但不用纠结加载语法细节。
    # ==============================
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    # 初始化 site_xmat/xpos 等派生运动学量；扰动前馈第 0 拍需要有效的 torso 姿态。
    mujoco.mj_forward(m, d)
    free_joint_ids = np.flatnonzero(
        m.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    if free_joint_ids.size != 1:
        raise ValueError("当前 RNEA 加速度实验要求模型恰好包含一个 floating base。")
    free_v_start = int(m.jnt_dofadr[int(free_joint_ids[0])])
    rnea_other_acceleration_estimator = RneaOtherAccelerationEstimator(
        m.nv,
        mode=ddq_rnea_other_qacc_mode,
        filter_alpha=ddq_rnea_other_qacc_filter_alpha,
        trend_window=ddq_rnea_other_qacc_trend_window,
        trend_lead_steps=ddq_rnea_other_qacc_trend_lead_steps,
        acceleration_limit=ddq_rnea_other_qacc_limit,
        base_qacc_indices=np.arange(free_v_start, free_v_start + 6),
        measured_blend=ddq_rnea_other_qacc_blend,
    )

    # --- 调试打印：如需检查关节、速度和驱动器的索引映射，可在这里临时添加打印 ---
    
    # 加载策略
    policy = torch.jit.load(policy_path)

    # ==============================
    # 4. 实例化右臂控制器【核心代码】
    # 右臂可在 PID / LQR / MPC 之间切换；这里决定本次实验使用哪个控制器。
    # 这部分要重点理解：控制器输入输出、控制周期，以及加速度控制器如何接入力矩执行链。
    # ==============================
    right_arm_target = arm_waist_target[6:11].copy()
    arm_control_dt = simulation_dt * arm_control_decimation
    target_right_arm_q = right_arm_target.copy()
    target_right_arm_dq = np.zeros(5, dtype=np.float32)
    desired_right_arm_ddq = np.zeros(5, dtype=np.float32)
    raw_right_arm_ddq = np.zeros(5, dtype=np.float64)
    right_ee_position_reference_torso = np.zeros(3, dtype=np.float64)
    lqr_one_step_prediction = None
    mpc_diagnostics = None
    cached_right_arm_tau_ff = None
    cached_inverse_result = None
    cached_mapping_result = None
    mpc_command_delay_line = (
        ArmCommandDelayLine(
            step_dt=simulation_dt,
            requested_delay=mpc_command_delay_ms * 1e-3,
            initial_q=target_right_arm_q,
            initial_dq=target_right_arm_dq,
            initial_ddq=desired_right_arm_ddq,
        )
        if arm_controller == "mpc"
        else None
    )
    active_command_source_time = 0.0
    active_acceleration_command_id = 0
    command_activation = None
    last_mpc_command_activation_counter = None
    controller_setup = create_arm_controller(
        config, arm_controller, right_arm_target, arm_control_dt
    )
    arm_policy = controller_setup.policy
    acceleration_controller = controller_setup.acceleration_controller
    # process/shadow 承载的是 ddq_des -> 力矩链，只适用于 LQR/MPC。
    # PID 没有 ddq_des，继续走原同步 C++ PD 执行器，避免全局配置为
    # process 时意外绕过 PID 的既有最终力矩路径。
    right_arm_execution_runtime = (
        requested_right_arm_execution_runtime
        if acceleration_controller
        else "sync"
    )
    lqr_cost_definition = controller_setup.lqr_cost_definition
    mpc_cost_definition = (
        arm_policy.get_cost_definition() if arm_controller == "mpc" else None
    )
    controller_meta = controller_setup.metadata
    if arm_controller == "mpc":
        controller_meta["mpc_config"]["ddq_execution_mode"] = (
            mpc_ddq_execution_mode
        )
        controller_meta["mpc_config"]["command_delay"] = (
            mpc_command_delay_line.metadata()
        )
    controller_meta["robot_model_backends"] = {
        "mpc_prediction_kinematics": (
            mpc_prediction_kinematics_backend
            if arm_controller == "mpc"
            else "not_used"
        ),
        "ddq_nominal_inverse_dynamics": (
            ddq_nominal_inverse_dynamics_backend
            if acceleration_controller
            else "not_used"
        ),
        "model_source": "matching_simulation_mjcf",
        "mujoco_contact_validation_retained": True,
        "pinocchio_friction_breakaway_steps": (
            ddq_pinocchio_friction_breakaway_steps
        ),
        "nominal_inverse_dynamics_other_qacc": {
            "mode": ddq_rnea_other_qacc_mode,
            "source": "latest_causal_mujoco_qacc_from_previous_physics_step",
            "filter_alpha": ddq_rnea_other_qacc_filter_alpha,
            "trend_window": ddq_rnea_other_qacc_trend_window,
            "trend_lead_steps": ddq_rnea_other_qacc_trend_lead_steps,
            "absolute_limit": ddq_rnea_other_qacc_limit,
            "measured_blend": ddq_rnea_other_qacc_blend,
        },
        "right_arm_executor": right_arm_executor_backend,
        "right_arm_execution_runtime": right_arm_execution_runtime,
        "right_arm_execution_runtime_requested": (
            requested_right_arm_execution_runtime
        ),
        "ddq_forward_dynamics_mapping": ddq_forward_dynamics_backend,
        "right_arm_executor_output_semantics": (
            right_arm_executor_output_semantics
        ),
    }
    filter_keys = (
        {
            "acc_alpha_key": "mpc_torso_acc_filter_alpha",
            "alpha_alpha_key": "mpc_torso_alpha_filter_alpha",
        }
        if arm_controller == "mpc"
        else {}
    )
    torso_acceleration_filter = TorsoAccelerationFilter(
        config, enabled=acceleration_controller, **filter_keys
    )
    disturbance_predictor = None
    if arm_controller == "mpc" and bool(
        config.get("mpc_disturbance_feedforward_enabled", False)
    ):
        disturbance_predictor = PhaseDisturbancePredictor(
            template_dir=os.path.join(
                repo_dir,
                config.get(
                    "mpc_disturbance_template_dir",
                    "disturbance_model_new_heading/templates_heading_interval",
                ),
            ),
            variant=config.get(
                "mpc_disturbance_template", "fully_smoothed"
            ),
            control_dt=arm_control_dt,
            horizon=arm_policy.horizon,
            acc_limit=torso_acceleration_filter.acc_limit,
            alpha_limit=torso_acceleration_filter.alpha_limit,
            slow_bias_enabled=bool(
                config.get("mpc_disturbance_slow_bias_enabled", True)
            ),
            slow_bias_time_constant=float(
                config.get(
                    "mpc_disturbance_slow_bias_time_constant", 0.4
                )
            ),
        )
        controller_meta["mpc_config"][
            "disturbance_feedforward"
        ] = disturbance_predictor.metadata()
    elif arm_controller == "mpc":
        controller_meta["mpc_config"]["disturbance_feedforward"] = {
            "enabled": False,
            "variant": str(
                config.get("mpc_disturbance_template", "fully_smoothed")
            ),
            "prediction": "zero_order_hold_current_measurement",
        }
    controller_meta["heading_control"] = {
        "enabled": heading_control_enabled,
        "reference_frame": "world",
        "reference_source": "initial_torso_yaw",
        "filter_cycles": heading_filter_cycles,
        "kp": heading_kp,
        "kd": heading_kd,
        "yaw_rate_feedforward": float(cmd_nominal[2]),
        "max_abs_yaw_rate": heading_max_yaw_rate,
    }
    performance_runtime = build_performance_runtime_config(
        config=config,
        arm_control_dt=arm_control_dt,
        eval_start_time=eval_start_time,
        eval_end_time=eval_end_time,
        run_end_time=eval_duration,
        warmup_cycles=warmup_cycles,
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
    )
    controller_meta["runtime_timing_environment"] = (
        performance_runtime.metadata
    )
    # ==============================
    # 5. 创建实验输出目录与视频录制器【非核心代码】
    # 每次 run 都单独保存 metadata、轨迹、评估图和视频，方便横向对比。
    # 这是实验管理与结果保存，不是控制数学核心。
    # ==============================
    run_metadata = build_run_metadata(
        config_file,
        experiment_name,
        controller_setup.policy_type,
        controller_setup.notes,
        controller_meta,
        cmd_nominal,
        simulation_dt,
        gait_period,
        warmup_cycles,
        evaluation_cycles,
        cooldown_cycles,
    )
    run_dir = create_eval_run_dir(
        os.path.join(repo_dir, "evaluation"),
        evaluation_group,
        run_metadata,
        run_label=run_label,
    )
    video_path = os.path.join(run_dir, "rollout.mp4")
    video_fps = 30
    video_stride = max(1, int(round(1.0 / (simulation_dt * video_fps))))
    video_frames = []
    video_camera = make_video_camera()
    renderer = None
    video_width = None
    video_height = None
    video_renderer_available = False

    # ==============================
    # 6. 解析主循环要用到的模型上下文【半核心代码】
    # resolve 这里可理解为“按名字/约定查出来并整理好”。
    # 这一节本身主要是在装配上下文，不是控制公式本体；但它会把后面要用到的核心支持模块接进主循环。
    # 这几类对象的区别是：
    # scene_ids：只负责“场景对象 id”；它是一个轻量 id 容器，里面包含 torso body、IMU site、左手抓持点 site、右手抓持点 site 的 MuJoCo id。
    #            主循环后面会反复用这些 id 读取姿态/速度/位置，并在 viewer 里画调试坐标轴。
    # right_arm_id_index_scratch：只负责“右臂 ids / indices / scratch 上下文”；它关注的是右臂 5 个关节和执行器在 MuJoCo 里的索引、力矩约束，以及逆动力学 scratch data，里面包含：
    #            - qpos_indices / qvel_indices / ctrl_indices：右臂 5 个关节在 qpos / qvel / ctrl 中的索引
    #            - joint_ids / actuator_ids：按名字查到的 MuJoCo joint / actuator id
    #            - torque_limits：从 XML 读取的右臂力矩上下限
    #            - inverse_dynamics_data：专门给 mj_inverse() 做逆动力学前馈计算用的 scratch MjData
    # right_arm_helper：只负责“右臂运动学/建模辅助”；它知道右臂末端 site、右臂关节索引、IMU site，
    #            后面用它来构建 torso_disturbance、Jacobian、重力误差和 LQR 线性化所需接口。
    # perf_monitor：记录每拍的时间统计，当前会分别跟踪 arm_control、mj_step、other_overhead 和 loop_total。
    # ==============================
    scene_ids = resolve_scene_ids(m)

    right_arm_id_index_scratch = resolve_right_arm_control_context(m, run_metadata["right_arm_joint_names"])
    if arm_controller == "lqr":
        arm_policy.set_joint_limits(m.jnt_range[right_arm_id_index_scratch.joint_ids])
    # 【核心代码】预测后端与名义逆动力学后端共用同一份 matching MJCF。
    # Pinocchio 只替代无接触的模型计算；后面的 MuJoCo 候选验收不变。
    prediction_backend = create_prediction_backend(
        (
            mpc_prediction_kinematics_backend
            if arm_controller == "mpc"
            else "mujoco"
        ),
        mujoco_model=m,
        joint_names=right_arm_id_index_scratch.joint_names,
        mjcf_path=xml_path,
        ee_name="right_grasp_site",
        imu_name="imu_in_torso",
    )
    pinocchio_backend = (
        prediction_backend
        if prediction_backend.backend_name == "pinocchio"
        else None
    )
    if (
        acceleration_controller
        and ddq_nominal_inverse_dynamics_backend.startswith("pinocchio")
        and pinocchio_backend is None
    ):
        pinocchio_backend = create_prediction_backend(
            "pinocchio",
            mujoco_model=m,
            joint_names=right_arm_id_index_scratch.joint_names,
            mjcf_path=xml_path,
            ee_name="right_grasp_site",
            imu_name="imu_in_torso",
        )
    # 预测窗口和 RNEA 都选择 C++ Pinocchio 时复用同一个 Model/Data
    # handle，避免重复解析模型和维护两套本地状态。
    cpp_rnea_backend = (
        prediction_backend
        if prediction_backend.backend_name == "cpp_pinocchio"
        else None
    )
    if (
        acceleration_controller
        and ddq_nominal_inverse_dynamics_backend == "cpp_pinocchio"
        and cpp_rnea_backend is None
    ):
        cpp_rnea_backend = CppRightArmRneaBackend(xml_path)
    cpp_ddq_mapper = (
        CppDdqTorqueMapper(xml_path)
        if acceleration_controller
        and ddq_forward_dynamics_backend == "cpp"
        and right_arm_execution_runtime in {"sync", "shadow"}
        else None
    )
    if arm_controller == "mpc":
        executor_q_min = np.deg2rad(
            np.asarray(config["mpc_q_min_deg"], dtype=np.float64)
        )
        executor_q_max = np.deg2rad(
            np.asarray(config["mpc_q_max_deg"], dtype=np.float64)
        )
    else:
        executor_q_min = m.jnt_range[
            right_arm_id_index_scratch.joint_ids, 0
        ].copy()
        executor_q_max = m.jnt_range[
            right_arm_id_index_scratch.joint_ids, 1
        ].copy()
    executor_dq_ref_abs_max = np.asarray(
        config.get("right_arm_executor_dq_ref_abs_max", [1.0] * 5),
        dtype=np.float64,
    )
    if executor_dq_ref_abs_max.shape != (5,):
        raise ValueError(
            "right_arm_executor_dq_ref_abs_max 必须是长度 5 的数组。"
        )
    if arm_controller == "mpc":
        mpc_dq_limit = np.broadcast_to(
            np.asarray(config.get("mpc_max_dq", 1.0), dtype=np.float64),
            (5,),
        )
        if np.any(executor_dq_ref_abs_max + 1e-12 < mpc_dq_limit):
            raise ValueError(
                "C++ 执行器 dq_ref 限幅不能小于 MPC 速度约束，"
                "否则最终 PD 会偏离已验收力矩。"
            )
    cpp_right_arm_executor = (
        CppRightArmExecutor(
            kp=arm_waist_kps[6:11],
            kd=arm_waist_kds[6:11],
            timeout_damping=np.asarray(
                config.get(
                    "right_arm_executor_timeout_damping",
                    arm_waist_kds[6:11],
                ),
                dtype=np.float64,
            ),
            q_ref_min=executor_q_min,
            q_ref_max=executor_q_max,
            dq_ref_abs_max=executor_dq_ref_abs_max,
            tau_min=right_arm_id_index_scratch.torque_limits[:, 0],
            tau_max=right_arm_id_index_scratch.torque_limits[:, 1],
            command_timeout_ms=float(
                config.get("right_arm_executor_command_timeout_ms", 30.0)
            ),
            state_timeout_ms=float(
                config.get("right_arm_executor_state_timeout_ms", 10.0)
            ),
            output_semantics=right_arm_executor_output_semantics,
        )
        if right_arm_executor_backend == "cpp"
        and right_arm_execution_runtime in {"sync", "shadow"}
        else None
    )
    if (
        acceleration_controller
        and right_arm_execution_runtime in {"process", "shadow"}
        and (
            ddq_nominal_inverse_dynamics_backend != "cpp_pinocchio"
            or ddq_forward_dynamics_backend != "cpp"
            or right_arm_executor_backend != "cpp"
        )
    ):
        raise ValueError(
            "独立C++进程当前要求 cpp_pinocchio RNEA、cpp mapper "
            "和cpp executor，避免出现两套不同执行定义。"
        )
    right_arm_sim_process = (
        RightArmSimProcess(
            xml_path,
            nq=m.nq,
            nv=m.nv,
            nu=m.nu,
            nbody=m.nbody,
            kp=arm_waist_kps[6:11],
            kd=arm_waist_kds[6:11],
            timeout_damping=np.asarray(
                config.get(
                    "right_arm_executor_timeout_damping",
                    arm_waist_kds[6:11],
                ),
                dtype=np.float64,
            ),
            q_ref_min=executor_q_min,
            q_ref_max=executor_q_max,
            dq_ref_abs_max=executor_dq_ref_abs_max,
            tau_min=right_arm_id_index_scratch.torque_limits[:, 0],
            tau_max=right_arm_id_index_scratch.torque_limits[:, 1],
            command_timeout_ms=float(
                config.get("right_arm_executor_command_timeout_ms", 30.0)
            ),
            state_timeout_ms=float(
                config.get("right_arm_executor_state_timeout_ms", 10.0)
            ),
            output_semantics=right_arm_executor_output_semantics,
            mapper_perturbation=controller_setup.execution_perturbation,
            mapper_regularization=controller_setup.execution_regularization,
            mapper_second_pass_error_threshold=(
                controller_setup.execution_second_pass_error_threshold
            ),
            mapper_max_joint_error=(
                controller_setup.execution_max_joint_error
            ),
            mapper_max_abs_qacc=(
                controller_setup.execution_max_abs_qacc
            ),
            mapper_enable_second_pass=(
                controller_setup.execution_enable_second_pass
            ),
            mapper_max_safety_rescue_passes=(
                controller_setup.execution_safety_rescue_passes
            ),
        )
        if acceleration_controller
        and right_arm_execution_runtime in {"process", "shadow"}
        else None
    )
    process_shadow_validator = (
        SimProcessShadowValidator(absolute_tolerance=1e-9)
        if acceleration_controller
        and right_arm_execution_runtime == "shadow"
        else None
    )
    right_arm_helper = KinematicsHelper(
        m,
        ee_site_name="right_grasp_site",
        joint_indices=right_arm_id_index_scratch.qpos_indices,
        imu_site_name="imu_in_torso",
        position_reference_q=right_arm_target,
        prediction_backend=(
            prediction_backend if arm_controller == "mpc" else None
        ),
    )

    # 行走策略仍接收 yaw-rate 命令；这里在它外层增加世界系航向保持。
    heading_controller = HeadingHoldController(
        sample_dt=simulation_dt * control_decimation,
        averaging_window=heading_filter_cycles * gait_period,
        kp=heading_kp,
        kd=heading_kd,
        yaw_rate_feedforward=float(cmd_nominal[2]),
        max_abs_yaw_rate=heading_max_yaw_rate,
    )
    heading_state = heading_controller.last_output
    cmd_runtime = cmd_nominal.copy()
    heading_yaw_rate_command_runtime = float(cmd_runtime[2])

    perf_monitor = PerformanceMonitor(
        step_budget=simulation_dt,
        arm_budget=arm_control_dt,
        measurement_start_time=performance_runtime.measurement_start_time,
        measurement_end_time=performance_runtime.measurement_end_time,
        runtime_environment=performance_runtime.metadata,
    )

    # ==============================
    # 7. 主仿真循环【核心代码】
    # 每一拍的总体顺序是：
    #   腿部控制 -> 上肢状态读取与右臂控制 -> 写入力矩 -> mj_step 推进物理 -> 记录评估 -> RL 更新 -> 可视化与计时
    # 这里是整个文件最需要看懂的部分，因为真正的控制数据流都发生在这里。
    # ==============================
    gc_was_enabled = gc.isenabled()
    if performance_runtime.disable_gc_during_control:
        gc.collect()
        gc.disable()
    viewer_context = mujoco.viewer.launch_passive(m, d) if viewer_enabled else nullcontext(None)
    with viewer_context as viewer:
        # viewer 和离屏 renderer 分别使用 GLFW/EGL。后创建 renderer，
        # 并在退出 viewer 前先关闭它，保证两个 OpenGL 上下文按反序释放。
        if record_video:
            renderer, video_width, video_height = make_video_renderer(
                m,
                preferred_width=1280,
                preferred_height=720,
            )
        video_renderer_available = renderer is not None
        display_mode = "交互显示" if viewer is not None else "headless"
        print(f"运行模式 = {display_mode} | 实验 = {experiment_name} | 运行 {total_cycles} 个周期 = {eval_duration:.1f}s，其中 warm-up {warmup_cycles} 周期、evaluation {evaluation_cycles} 周期、cooldown {cooldown_cycles} 周期")
        while (viewer is None or viewer.is_running()) and counter * simulation_dt < eval_duration:
            perf_monitor.start_step(counter * simulation_dt)

            # --- 7.1 腿部控制【半核心】---
            # 这部分是已有 locomotion 执行链；需要知道它负责下肢，不需要像右臂控制那样细抠每个细节。
            # 位置向量第 7 到 18 项为腿部的 12 个关节，速度向量第 6 到 17 项为对应的速度
            leg_q = d.qpos[7:19]
            leg_dq = d.qvel[6:18]
            tau_leg = pd_control(target_dof_pos, leg_q, kps, np.zeros_like(kds), leg_dq, kds)
            d.ctrl[:12] = tau_leg

            # --- 7.2 上肢状态读取与右臂控制【核心代码】---
            # 顺序: 腰部(1)、左臂(5)、右臂(5)
            # 这一段分成两层：
            # 1) 上层控制器（PID / LQR / MPC）根据当前右臂状态和 torso 扰动，生成右臂参考。
            #    - PID 路径：直接输出 right_arm 的 q_ref / dq_ref
            #    - LQR/MPC 路径：除了输出 q_ref / dq_ref，还会额外输出期望关节加速度 ddq_des
            # 2) 下层执行层把参考转成真正施加到 MuJoCo 的力矩：
            #    - 基础项：所有上肢统一先经过 joint-space PD，得到 tau_pd
            #    - LQR/MPC 额外项：配置的逆动力学先生成名义力矩，再用局部前向动力学
            #      映射反求使右臂实际加速度接近 ddq_des 的最终力矩。

            arm_policy_update_due = counter % arm_control_decimation == 0
            if arm_policy_update_due:
                perf_monitor.begin_arm_interval(counter * simulation_dt)
            # 真机相关右臂路径从整机参考加速度和当前状态处理开始，
            # 到最终力矩写入为止；不包含腿部 PD、MuJoCo 物理推进和画图。
            perf_monitor.start_right_arm_path()
            rnea_reference_qacc = rnea_other_acceleration_estimator.update(
                d.qacc
            )

            # 当前上肢的真实状态（腰 + 左臂 + 右臂），后面 PD 会用它和目标状态做误差反馈
            arm_waist_q = d.qpos[19:30]
            arm_waist_dq = d.qvel[18:29]

            # 腰和左臂在本实验里不做在线优化，始终保持固定目标位形
            waist_left_target_q = arm_waist_target[:6].copy()
            waist_left_target_dq = np.zeros(6, dtype=np.float32)

            # 从上肢状态里切出右臂 5 个关节，作为右臂控制器的当前状态输入
            right_arm_q = arm_waist_q[6:11]
            right_arm_dq = arm_waist_dq[6:11]

            # torso 线加速度直接读取 MuJoCo IMU accelerometer，并转换到去重力后的世界系；
            # torso 角加速度仍由世界系角速度有限差分得到。
            # 这些量一方面用于构建扰动输入，另一方面也会进入 right_arm_obs 给右臂控制器使用。
            torso_state = update_torso_motion_state(m, d, scene_ids, buffers, counter, simulation_dt)
            raw_torso_acc, raw_torso_alpha = torso_acceleration_filter.update(
                torso_state
            )

            # 把 torso 世界系运动量和当前姿态打包，供 KinematicsHelper 以及
            # LQR/MPC 局部线性化使用。MPC 前馈先把 H 模板旋到 W，
            # 再以该 d_0（包括 R_B,0）锚定模板相对变化。
            torso_disturbance = right_arm_helper.build_disturbance_input(
                acc_world=torso_state.lin_acc,
                omega_world=torso_state.ang_vel,
                alpha_world=torso_state.ang_acc,
                rot_world_body=torso_state.rotmat,
            )
            
            # right_arm_obs 是上层右臂控制器看到的“当前观测”；其中包含右臂状态、torso 姿态与运动信息，以及右臂控制周期。
            right_arm_obs = build_right_arm_observation(right_arm_q, right_arm_dq, torso_state, arm_control_dt)
            if arm_policy_update_due:
                perf_monitor.start_arm_control()
                disturbance_prediction_start = time.perf_counter()
                disturbance_horizon = (
                    None
                    if disturbance_predictor is None
                    else disturbance_predictor.predict(
                        counter * simulation_dt,
                        torso_disturbance,
                    )
                )
                disturbance_prediction_time = (
                    time.perf_counter() - disturbance_prediction_start
                )
                disturbance_prediction = (
                    None
                    if disturbance_horizon is None
                    else disturbance_horizon.nodes
                )
                interval_disturbance_prediction = (
                    None
                    if disturbance_horizon is None
                    else disturbance_horizon.intervals
                )
                # right_arm_helpers 封装当前步的运动学量，以及 PID/LQR/MPC 各自的线性化回调。
                helper_construction_start = time.perf_counter()
                right_arm_helpers = right_arm_helper.build_helpers(
                    d,
                    disturbance=torso_disturbance,
                    disturbance_prediction=disturbance_prediction,
                    interval_disturbance_prediction=(
                        interval_disturbance_prediction
                    ),
                    # MPC 直接按预测窗口求运动学；旧的单点 cache 没有消费者。
                    include_kinematics_cache=(arm_controller != "mpc"),
                )
                helper_construction_time = (
                    time.perf_counter() - helper_construction_start
                )
                right_ee_position_reference_torso = right_arm_helpers.torso_relative_position_reference.copy()
                if acceleration_controller:
                    # 【核心代码】LQR/MPC 都输出 ddq_des，并复用同一条力矩执行链。
                    # - target_right_arm_q / dq：给下层 PD 跟踪的参考轨迹
                    # - desired_right_arm_ddq：期望关节加速度，后面用于 computed torque 前馈
                    controller_compute_start = time.perf_counter()
                    (
                        generated_target_right_arm_q,
                        generated_target_right_arm_dq,
                        generated_desired_right_arm_ddq,
                    ) = arm_policy.compute_action(
                        right_arm_obs, right_arm_helpers
                    )
                    controller_compute_action_time = (
                        time.perf_counter() - controller_compute_start
                    )
                    diagnostics_start = time.perf_counter()
                    controller_diagnostics = (
                        arm_policy.get_last_diagnostics(copy_data=False)
                        if arm_controller == "mpc"
                        else arm_policy.get_last_diagnostics()
                    )
                    generated_raw_right_arm_ddq = controller_diagnostics[
                        "ddq_raw"
                    ]
                    if arm_controller == "lqr":
                        # 同一次 LQR 更新在随后三个 2 ms 执行拍中必须保留
                        # 相同的来源时间和命令编号，C++ 才能看到真实命令年龄。
                        active_command_source_time = counter * simulation_dt
                        active_acceleration_command_id += 1
                        target_right_arm_q = generated_target_right_arm_q
                        target_right_arm_dq = generated_target_right_arm_dq
                        desired_right_arm_ddq = (
                            generated_desired_right_arm_ddq
                        )
                        raw_right_arm_ddq = generated_raw_right_arm_ddq
                        lqr_one_step_prediction = controller_diagnostics["one_step_prediction"]
                    else:
                        # 【核心延迟模型】同一次 MPC 求解的 q/dq/ddq 必须
                        # 作为一个命令包发布，不能只延迟其中某一项。
                        mpc_command_delay_line.publish(
                            counter * simulation_dt,
                            generated_target_right_arm_q,
                            generated_target_right_arm_dq,
                            generated_raw_right_arm_ddq,
                            generated_desired_right_arm_ddq,
                        )
                        mpc_diagnostics = controller_diagnostics
                        if disturbance_predictor is not None:
                            mpc_diagnostics["disturbance_template_diagnostics"] = (
                                disturbance_predictor.get_last_diagnostics()
                            )
                    diagnostics_time = time.perf_counter() - diagnostics_start
                    if arm_controller == "mpc":
                        # 【非核心诊断】把上层控制拍拆开计时，确认优化真正作用于
                        # 真机也会存在的扰动预测、helper 构造和 MPC 计算路径。
                        controller_diagnostics.update(
                            {
                                "disturbance_prediction_time": (
                                    disturbance_prediction_time
                                ),
                                "helper_construction_time": (
                                    helper_construction_time
                                ),
                                "controller_compute_action_time": (
                                    controller_compute_action_time
                                ),
                                "diagnostics_time": diagnostics_time,
                            }
                        )
                else:
                    # PID 路径只输出右臂参考轨迹，不单独生成期望加速度
                    target_right_arm_q, target_right_arm_dq = arm_policy.compute_action(right_arm_obs, right_arm_helpers)
                perf_monitor.finish_arm_control()
                if arm_controller == "mpc":
                    perf_monitor.record_mpc_timing(controller_diagnostics)

            if mpc_command_delay_line is not None:
                command_activation = mpc_command_delay_line.activate_ready(
                    counter * simulation_dt
                )
                active_packet = command_activation.packet
                target_right_arm_q = active_packet.target_q
                target_right_arm_dq = active_packet.target_dq
                raw_right_arm_ddq = active_packet.ddq_raw
                desired_right_arm_ddq = active_packet.ddq_des
                active_command_source_time = active_packet.source_time
                if command_activation.activated:
                    last_mpc_command_activation_counter = counter

            # 把“固定的腰/左臂目标”和“在线计算的右臂目标”拼回完整上肢目标
            target_arm_waist_q = np.concatenate([waist_left_target_q, target_right_arm_q])
            target_arm_waist_dq = np.concatenate([waist_left_target_dq, target_right_arm_dq])

            # 第一层执行：统一用 joint-space PD 把目标状态转成上肢控制力矩 tau_pd
            tau_arm_waist = pd_control(
                target_arm_waist_q, arm_waist_q, arm_waist_kps,
                target_arm_waist_dq, arm_waist_dq, arm_waist_kds
            )
            # 只切出右臂 5 维的 PD 力矩；LQR/MPC 后面还要和逆动力学前馈叠加
            right_arm_tau_pd = tau_arm_waist[6:11].copy()
            inverse_result = None
            mapping_result = None
            cpp_executor_result = None
            process_result = None
            ddq_execution_updated = False
            if acceleration_controller:
                # 【核心代码】第二层执行（LQR/MPC 共用）：
                # 用 desired_right_arm_ddq 作为右臂期望加速度，由配置选择
                # MuJoCo inverse 或 Pinocchio RNEA 计算 tau_ff。
                # apply_computed_torque_control() 内部会：
                # 1) 复制当前整机 qpos / qvel 到 scratch data
                # 2) 按配置填入上一物理拍的整机参考加速度，再把右臂 5 维
                #    覆盖为 desired_right_arm_ddq；zero 模式则退化为旧实现
                # 3) 生成“非摩擦约束不对抗”的名义力矩
                # 4) 固定腿、腰和左臂力矩：1 次完整 mj_forward 建立基线，
                #    再用 5 次 mj_forwardSkip（每个右臂力矩各扰动一次）构建 G_tau
                # 5) 通过一次阻尼最小二乘求右臂力矩修正
                # 6) 局部模型先对 1.0/0.5/0.25/0.125 四个修正尺度排序，
                #    用 mj_forwardSkip 至少验收预测最优的两个，选真实误差更小者；都失败才继续
                # 7) 验收后残差仍大于阈值时，在已接受力矩处重算 G_tau，并做一次同样受验收的二次修正
                execution_update_due = True
                if arm_controller == "mpc":
                    delayed_grid_active = (
                        mpc_command_delay_line.quantized_delay > 0.0
                    )
                    if delayed_grid_active:
                        # 命令可能在原 6 ms 区间的 2 ms 相位到达。到达
                        # 当拍必须立刻重算，第二次验收相对该激活拍后移 4 ms。
                        activated_now = bool(
                            command_activation is not None
                            and command_activation.activated
                        )
                        steps_since_activation = (
                            None
                            if last_mpc_command_activation_counter is None
                            else counter - last_mpc_command_activation_counter
                        )
                        if mpc_ddq_execution_mode == "policy_update":
                            execution_update_due = activated_now
                        elif mpc_ddq_execution_mode == "twice_per_interval":
                            execution_update_due = activated_now or (
                                steps_since_activation
                                == arm_control_decimation - 1
                            )
                    elif mpc_ddq_execution_mode == "policy_update":
                        execution_update_due = arm_policy_update_due
                    elif mpc_ddq_execution_mode == "twice_per_interval":
                        interval_phase = counter % arm_control_decimation
                        execution_update_due = (
                            interval_phase == 0
                            or interval_phase == arm_control_decimation - 1
                        )
                if cached_right_arm_tau_ff is None:
                    execution_update_due = True

                fixed_ctrl_for_mapping = d.ctrl.copy()
                fixed_ctrl_for_mapping[12:18] = tau_arm_waist[:6]
                previous_executed_tau = (
                    d.ctrl[right_arm_id_index_scratch.ctrl_indices].copy()
                    if controller_setup.execution_hold_last_safe
                    and d.time > 0.0
                    else None
                )

                # sync是冻结基线；shadow先跑sync再逐拍核对独立进程。
                if right_arm_execution_runtime in {"sync", "shadow"}:
                    if execution_update_due:
                        perf_monitor.start_computed_torque_control()
                        right_arm_tau, inverse_result, mapping_result = apply_computed_torque_control(
                            m,
                            d,
                            right_arm_id_index_scratch,
                            desired_right_arm_ddq,
                            right_arm_tau_pd,
                            fixed_ctrl_for_mapping,
                            forward_dynamics_perturbation=controller_setup.execution_perturbation,
                            forward_dynamics_regularization=controller_setup.execution_regularization,
                            forward_dynamics_second_pass_error_threshold=(
                                controller_setup.execution_second_pass_error_threshold
                            ),
                            forward_dynamics_max_joint_error=controller_setup.execution_max_joint_error,
                            forward_dynamics_max_abs_qacc=controller_setup.execution_max_abs_qacc,
                            forward_dynamics_enable_second_pass=controller_setup.execution_enable_second_pass,
                            forward_dynamics_max_safety_rescue_passes=(
                                controller_setup.execution_safety_rescue_passes
                            ),
                            forward_dynamics_enable_hold_last_safe=(
                                controller_setup.execution_hold_last_safe
                            ),
                            inverse_dynamics_backend=(
                                ddq_nominal_inverse_dynamics_backend
                            ),
                            pinocchio_backend=pinocchio_backend,
                            cpp_rnea_backend=cpp_rnea_backend,
                            forward_dynamics_backend=(
                                ddq_forward_dynamics_backend
                            ),
                            cpp_ddq_mapper=cpp_ddq_mapper,
                            pinocchio_friction_breakaway_steps=(
                                ddq_pinocchio_friction_breakaway_steps
                            ),
                            inverse_dynamics_reference_qacc=(
                                rnea_reference_qacc
                            ),
                        )
                        ddq_execution_elapsed = (
                            perf_monitor.finish_computed_torque_control()
                        )
                        perf_monitor.record_ddq_execution_timing(
                            inverse_result,
                            mapping_result,
                            call_elapsed=ddq_execution_elapsed,
                        )
                        ddq_execution_updated = True
                        cached_right_arm_tau_ff = (
                            right_arm_tau - right_arm_tau_pd
                        )
                        cached_inverse_result = inverse_result
                        cached_mapping_result = mapping_result
                    else:
                        right_arm_tau = (
                            cached_right_arm_tau_ff + right_arm_tau_pd
                        )
                        inverse_result = cached_inverse_result
                        mapping_result = cached_mapping_result
                    tau_arm_waist[6:11] = right_arm_tau

                if right_arm_sim_process is not None:
                    process_command_timestamp = (
                        float(active_command_source_time)
                        if mpc_command_delay_line is not None
                        else float(active_command_source_time)
                    )
                    process_command_id = (
                        int(active_packet.command_id) + 1
                        if mpc_command_delay_line is not None
                        else active_acceleration_command_id
                    )
                    process_source_state_id = (
                        int(round(process_command_timestamp / simulation_dt))
                        + 1
                    )
                    if (
                        right_arm_execution_runtime == "process"
                        and execution_update_due
                    ):
                        perf_monitor.start_computed_torque_control()
                    process_result = right_arm_sim_process.execute(
                        simulation_time=d.time,
                        command_timestamp=process_command_timestamp,
                        command_id=process_command_id,
                        command_source_state_id=process_source_state_id,
                        execution_state_id=counter + 1,
                        mapping_update_due=execution_update_due,
                        mujoco_timestep=m.opt.timestep,
                        friction_breakaway_steps=(
                            ddq_pinocchio_friction_breakaway_steps
                        ),
                        qpos=d.qpos,
                        qvel=d.qvel,
                        reference_qacc=rnea_reference_qacc,
                        fixed_ctrl=fixed_ctrl_for_mapping,
                        qacc_warmstart=d.qacc_warmstart,
                        qfrc_applied=d.qfrc_applied,
                        xfrc_applied=d.xfrc_applied,
                        right_arm_q=right_arm_q,
                        right_arm_dq=right_arm_dq,
                        q_ref=target_right_arm_q,
                        dq_ref=target_right_arm_dq,
                        ddq_des=desired_right_arm_ddq,
                        tau_passive=d.qfrc_passive[
                            right_arm_id_index_scratch.qvel_indices
                        ],
                        friction_loss=m.dof_frictionloss[
                            right_arm_id_index_scratch.qvel_indices
                        ],
                        tau_pd=right_arm_tau_pd,
                        previous_executed_tau=previous_executed_tau,
                    )
                    perf_monitor.record_sim_process_timing(
                        roundtrip_elapsed_time=(
                            process_result.roundtrip_elapsed_time
                        ),
                        worker_elapsed_time=(
                            process_result.worker_elapsed_time
                        ),
                        queue_elapsed_time=(
                            process_result.queue_elapsed_time
                        ),
                        executor_core_elapsed_time=(
                            process_result.executor_result.core_elapsed_time
                        ),
                        mapping_updated=process_result.mapping_updated,
                        include_in_interval_composition=(
                            right_arm_execution_runtime == "process"
                        ),
                    )
                    if right_arm_execution_runtime == "process":
                        if execution_update_due:
                            inverse_result = (
                                inverse_dynamics_result_from_sim_process(
                                    m,
                                    d,
                                    right_arm_id_index_scratch.qvel_indices,
                                    process_result,
                                )
                            )
                            mapper_core_elapsed = (
                                float(
                                    process_result.mapper_output.total_elapsed_ns
                                )
                                * 1e-9
                            )
                            right_arm_tau, mapping_result = (
                                forward_dynamics_result_from_cpp_mapper_response(
                                    process_result,
                                    wall_elapsed_time=mapper_core_elapsed,
                                    backend="cpp_process_mujoco",
                                )
                            )
                            cached_right_arm_tau_ff = (
                                process_result.validated_tau_ff.copy()
                            )
                            cached_inverse_result = inverse_result
                            cached_mapping_result = mapping_result
                            ddq_execution_elapsed = (
                                perf_monitor.finish_computed_torque_control()
                            )
                            perf_monitor.record_ddq_execution_timing(
                                inverse_result,
                                mapping_result,
                                call_elapsed=ddq_execution_elapsed,
                            )
                            ddq_execution_updated = True
                        else:
                            right_arm_tau = (
                                process_result.validated_tau_ff
                                + right_arm_tau_pd
                            )
                            inverse_result = cached_inverse_result
                            mapping_result = cached_mapping_result
                        tau_arm_waist[6:11] = right_arm_tau
                        cpp_executor_result = process_result.executor_result
            if cpp_right_arm_executor is not None:
                # 【核心代码】C++ 每个 2 ms 仿真拍都读取最新右臂 q/dq，
                # 只在这一处合成 PD、执行参考/力矩限幅及超时/NaN 保护。
                # 局部 MuJoCo 映射给出的最终力矩先减去本拍 Python PD，
                # 作为纯前馈传入；C++ 再加一次同参数 PD，避免重复计算。
                pre_executor_tau = tau_arm_waist[6:11].copy()
                executor_tau_ff = pre_executor_tau - right_arm_tau_pd
                simulated_now_ns = int(round(float(d.time) * 1e9))
                command_source_ns = (
                    int(round(float(active_command_source_time) * 1e9))
                    if acceleration_controller
                    else simulated_now_ns
                )
                cpp_executor_result = cpp_right_arm_executor.step(
                    now_ns=simulated_now_ns,
                    command_timestamp_ns=command_source_ns,
                    state_timestamp_ns=simulated_now_ns,
                    q=arm_waist_q[6:11],
                    dq=arm_waist_dq[6:11],
                    q_ref=target_right_arm_q,
                    dq_ref=target_right_arm_dq,
                    tau_ff=executor_tau_ff,
                )
                perf_monitor.record_cpp_executor_timing(
                    cpp_executor_result.wall_elapsed_time,
                    cpp_executor_result.core_elapsed_time,
                    cpp_executor_result.mode,
                )
                if right_arm_executor_output_semantics == "host_full_torque":
                    tau_arm_waist[6:11] = (
                        cpp_executor_result.actuator_tau_ff
                    )
                else:
                    # 仿真 direct-drive actuator 不自带 Unitree PD；device_pd
                    # 只用于字段语义验证，仿真仍执行其预计的总力矩。
                    tau_arm_waist[6:11] = (
                        cpp_executor_result.predicted_total_tau_limited
                    )
            if process_result is not None:
                if process_shadow_validator is not None:
                    # 【核心验收】先比较三层力矩和mapper分支；任何一拍
                    # 超过1e-9都立即停止，不能用相似轨迹掩盖错帧或错分支。
                    process_shadow_validator.validate(
                        process_result,
                        inverse_result=inverse_result,
                        mapping_result=mapping_result,
                        pre_executor_tau=pre_executor_tau,
                        final_tau=tau_arm_waist[6:11],
                        tau_pd=right_arm_tau_pd,
                        mapping_update_due=execution_update_due,
                    )
                # process和shadow最终都执行独立进程返回的力矩；shadow已在
                # 上一行证明它与冻结同步链相同。
                tau_arm_waist[6:11] = process_result.final_tau
                cpp_executor_result = process_result.executor_result
            # 最终把完整的上肢力矩（腰 + 左臂 + 右臂）写进 d.ctrl[12:23]
            d.ctrl[12:23] = tau_arm_waist
            perf_monitor.finish_right_arm_path()

            # --- 7.3 写入力矩并推进物理【核心代码】---
            # 到这里 d.ctrl 已经准备好；mj_step 会真正让整机往前走一拍。
            # 这一小段很重要，因为它定义了“控制输出最终如何进入仿真执行层”。
            perf_monitor.start_mj_step()
            mujoco.mj_step(m, d)
            perf_monitor.finish_mj_step()
            record_eval_step(
                m,
                d,
                counter,
                simulation_dt,
                scene_ids,
                buffers,
                right_arm_control=build_right_arm_control_record(
                    arm_policy_updated=arm_policy_update_due,
                    ddq_execution_updated=ddq_execution_updated,
                    target_q=target_right_arm_q,
                    target_dq=target_right_arm_dq,
                    ddq_raw=raw_right_arm_ddq,
                    ddq_des=desired_right_arm_ddq,
                    ddq_saturation_limit=controller_setup.ddq_saturation_limit,
                    tau_pd=right_arm_tau_pd,
                    torque_limits=right_arm_id_index_scratch.torque_limits,
                    torso_state=torso_state,
                    raw_torso_acc=raw_torso_acc,
                    raw_torso_alpha=raw_torso_alpha,
                    heading_state=heading_state,
                    heading_yaw_rate_command=heading_yaw_rate_command_runtime,
                    ee_position_reference_torso=right_ee_position_reference_torso,
                    inverse_result=inverse_result,
                    mapping_result=mapping_result,
                    cpp_executor_result=cpp_executor_result,
                    lqr_one_step_prediction=(
                        lqr_one_step_prediction
                        if arm_controller == "lqr"
                        and arm_policy_update_due
                        else None
                    ),
                    mpc_diagnostics=(
                        mpc_diagnostics
                        if arm_controller == "mpc"
                        and arm_policy_update_due
                        else None
                    ),
                ),
            )
            if renderer is not None and counter % video_stride == 0:
                renderer.update_scene(d, camera=video_camera)
                video_frames.append(renderer.render().copy())

            # --- 7.4 locomotion policy 更新【半核心】---
            # 下肢 RL 不需要每个 physics step 都更新，而是按 control_decimation 低频更新一次。
            # 需要知道它和右臂控制是并行存在的两条链，但不必像右臂那样细看实现细节。
            counter += 1
            if counter % control_decimation == 0:
                # 在这里施加控制信号。

                # 创建观测（强化学习策略只观测腿部状态，需要截取前 12 个关节）
                qj = d.qpos[7:19]
                dqj = d.qvel[6:18]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                count = counter * simulation_dt
                phase = count % gait_period / gait_period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                cmd_active = cmd_nominal.copy()
                if heading_control_enabled:
                    torso_yaw_world = quat_to_yaw_wxyz(d.xquat[scene_ids.torso_id].copy())
                    heading_state = heading_controller.update(torso_yaw_world, torso_state.ang_vel[2])
                    cmd_active[2] = heading_state.yaw_rate_command
                if count < eval_end_time:
                    command_scale = 1.0
                else:
                    cooldown_ratio = np.clip((count - eval_end_time) / max(eval_duration - eval_end_time, 1e-8), 0.0, 1.0)
                    command_scale = 1.0 - cooldown_ratio
                cmd_runtime = command_scale * cmd_active
                heading_yaw_rate_command_runtime = float(cmd_runtime[2])
                obs[6:9] = cmd_runtime * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # 策略推理
                action = policy(obs_tensor).detach().numpy().squeeze()
                # 将动作转换为目标关节位置
                target_dof_pos = action * action_scale + default_angles

            # --- 7.5 调试可视化与步时统计【非核心代码】---
            if viewer is not None:
                draw_debug_axes(viewer.user_scn, d, scene_ids)
                viewer.sync()
            perf_monitor.finish_step(counter)

        perf_monitor.finish_pending_arm_interval()
        perf_monitor.print_summary()
        close_renderer(renderer)
        renderer = None

    if performance_runtime.disable_gc_during_control and gc_was_enabled:
        gc.enable()

    # ==============================
    # 8. 收尾保存【非核心代码】
    # renderer 和 viewer 都已按正确顺序释放，这里只做轨迹、评估指标、视频和性能统计的文件保存。
    # 这是实验收尾，不是控制逻辑核心。
    # ==============================
    finalize_run(
        run_dir,
        buffers,
        xml_path,
        simulation_dt,
        video_path,
        video_frames,
        video_fps,
        video_renderer_available,
        video_width,
        video_height,
        d,
        scene_ids,
        eval_start_time,
        eval_end_time,
        total_cycles,
        warmup_cycles,
        evaluation_cycles,
        cooldown_cycles,
        gait_period,
        experiment_name,
        perf_monitor=perf_monitor,
        lqr_cost_definition=lqr_cost_definition,
        mpc_cost_definition=mpc_cost_definition,
        arm_controller=arm_controller,
    )
    if mpc_command_delay_line is not None:
        mpc_command_delay_line.save_report(
            run_dir, eval_start_time, eval_end_time
        )
    if process_shadow_validator is not None:
        shadow_path = process_shadow_validator.save(run_dir)
        print(
            "独立C++进程逐拍shadow通过："
            f"{process_shadow_validator.summary()} | {shadow_path}"
        )
    # 【非核心收尾】显式释放原生 handle，避免同一 Python 进程
    # 重复创建仿真时依赖解析器退出顺序或进程回收。
    closed_native_backends = set()
    for native_backend in (
        cpp_ddq_mapper,
        cpp_rnea_backend,
        prediction_backend,
        cpp_right_arm_executor,
        right_arm_sim_process,
    ):
        if (
            native_backend is not None
            and hasattr(native_backend, "close")
            and id(native_backend) not in closed_native_backends
        ):
            native_backend.close()
            closed_native_backends.add(id(native_backend))
