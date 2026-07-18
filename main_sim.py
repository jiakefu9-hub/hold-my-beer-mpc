import os

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

# --- 引入我们独立的策略 ---
from arm_pid import ArmPIDPolicy
from arm_lqr import ArmLQRPolicy
# 可选：导入手臂模型预测控制策略
from kinematics_helper import KinematicsHelper
from sim_support import (
    PerformanceMonitor,
    apply_computed_torque_control,
    build_right_arm_observation,
    build_run_metadata,
    close_renderer,
    create_eval_run_dir,
    draw_debug_axes,
    finalize_run,
    get_gravity_orientation,
    init_eval_buffers,
    make_video_camera,
    make_video_renderer,
    pd_control,
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
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"/home/fjk/g1_ws/hold-my-beer-mpc/configs/{config_file}", "r") as f:
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
        arm_control_decimation = int(config.get("arm_control_decimation", 2))
        lqr_torso_acc_filter_alpha = float(config.get("lqr_torso_acc_filter_alpha", 0.20))
        lqr_torso_alpha_filter_alpha = float(config.get("lqr_torso_alpha_filter_alpha", 0.20))
        lqr_torso_acc_limit = float(config.get("lqr_torso_acc_limit", 30.0))
        lqr_torso_alpha_limit = float(config.get("lqr_torso_alpha_limit", 40.0))
        cmd_nominal = np.array(config["cmd_init"], dtype=np.float32)

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
    warmup_cycles = 3
    evaluation_cycles = 8
    cooldown_cycles = 2
    total_cycles = warmup_cycles + evaluation_cycles + cooldown_cycles
    eval_start_time = warmup_cycles * gait_period
    eval_end_time = (warmup_cycles + evaluation_cycles) * gait_period
    eval_duration = total_cycles * gait_period
    experiment_name = f"left_fixed_right_{arm_controller}"
    buffers = init_eval_buffers()

    # ==============================
    # 3. 加载 MuJoCo 模型与 RL 行走策略【半核心】
    # MuJoCo 负责整机物理推进；TorchScript policy 负责下肢 locomotion。
    # 对论文/面试来说，需要知道“谁负责物理、谁负责走路”，但不用纠结加载语法细节。
    # ==============================
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # --- 调试打印：如需检查关节、速度和驱动器的索引映射，可在这里临时添加打印 ---
    
    # 加载策略
    policy = torch.jit.load(policy_path)

    # ==============================
    # 4. 实例化右臂控制器【核心代码】
    # 右臂可在 PID / LQR 之间切换；这里决定了本次实验到底在比较什么控制器。
    # 这部分要重点理解：控制器输入输出是什么、控制周期是多少、PID 与 LQR 的执行链有什么区别。
    # ==============================
    right_arm_target = arm_waist_target[6:11].copy()
    arm_control_dt = simulation_dt * arm_control_decimation
    target_right_arm_q = right_arm_target.copy()
    target_right_arm_dq = np.zeros(5, dtype=np.float32)
    desired_right_arm_ddq = np.zeros(5, dtype=np.float32)
    raw_right_arm_ddq = np.zeros(5, dtype=np.float64)
    right_ee_position_reference_torso = np.zeros(3, dtype=np.float64)
    filtered_torso_acc = np.zeros(3, dtype=np.float64)
    filtered_torso_alpha = np.zeros(3, dtype=np.float64)
    lqr_one_step_prediction = None
    lqr_cost_definition = None
    if arm_controller == "lqr":
        lqr_kwargs = {
            "horizon": int(config.get("lqr_horizon", 12)),
            "q_acc": float(config.get("lqr_q_acc", 1.0)),
            "q_alpha": float(config.get("lqr_q_alpha", 0.05)),
            "q_position": float(config.get("lqr_q_position", 20.0)),
            "q_gravity": float(config.get("lqr_q_gravity", 30.0)),
            "q_posture": float(config.get("lqr_q_posture", 0.4)),
            "q_vel": float(config.get("lqr_q_vel", 0.02)),
            "r_ddq": float(config.get("lqr_r_ddq", 0.25)),
            "terminal_scale": float(config.get("lqr_terminal_scale", 2.0)),
            "reg": float(config.get("lqr_reg", 1e-6)),
            "max_ddq": float(config.get("lqr_max_ddq", 3.0)),
            "max_dq": float(config.get("lqr_max_dq", 1.0)),
            "ddq_rate_limit": float(config.get("lqr_ddq_rate_limit", 350.0)),
            "ddq_smoothing_alpha": float(config.get("lqr_ddq_smoothing_alpha", 0.45)),
            "joint_limit_margin": float(config.get("lqr_joint_limit_margin", 0.25)),
            "joint_limit_stiffness": float(config.get("lqr_joint_limit_stiffness", 8.0)),
            "joint_limit_damping": float(config.get("lqr_joint_limit_damping", 2.0)),
        }
        arm_policy = ArmLQRPolicy(default_q=right_arm_target, control_dt=arm_control_dt, **lqr_kwargs)
        lqr_cost_definition = arm_policy.get_cost_definition()
        policy_type = "ArmLQRPolicy"
        controller_notes = "finite-horizon time-varying LQR with torso-relative end-effector position cost, directed 3D gravity error, fully bypassed ddq post-processing, protected torso disturbance inputs, and contact-aware MuJoCo inverse-dynamics feedforward plus joint-space PD feedback"
        controller_meta = {
            "lqr_config": {
                **lqr_kwargs,
                "control_dt": arm_policy.control_dt,
                "torque_control": "mujoco_contact_aware_inverse_dynamics_plus_pd",
                "contact_aware_tau_formula": "qfrc_inverse + qfrc_contact_only",
                "noncontact_constraint_compensation": "retained_in_qfrc_inverse",
                "uncontrolled_qacc_assumption": 0.0,
                "ddq_tracking": "6ms_velocity_difference_aligned_between_consecutive_arm_updates",
                "cost_tracking": "one_step_model_vs_realized_next_arm_update",
                "torso_acc_filter_alpha": lqr_torso_acc_filter_alpha,
                "torso_alpha_filter_alpha": lqr_torso_alpha_filter_alpha,
                "torso_acc_limit": lqr_torso_acc_limit,
                "torso_alpha_limit": lqr_torso_alpha_limit,
                "torso_acc_source": "mujoco_imu_accelerometer_world_without_gravity",
                "torso_alpha_source": "finite_difference_world_angular_velocity",
                "position_reference_frame": "torso_imu",
                "position_reference_q": right_arm_target.copy(),
                "end_effector_velocity_cost_enabled": False,
                "gravity_error": "directed_3d",
                "ddq_post_process": "fully_bypassed",
                "ddq_hard_clip_enabled": False,
                "joint_limit_guard": "disabled",
                "ddq_rate_limit_enabled": False,
                "ddq_smoothing_enabled": False,
            }
        }
    else:
        arm_policy = ArmPIDPolicy(default_q=right_arm_target, kp_pose=np.array([1.20, 1.20], dtype=np.float64), kd_pose=np.array([1.2, 1.2], dtype=np.float64), ki_pose=0.0, posture_gain=np.array([1.15, 1.15, 2.10, 1.15, 0.95], dtype=np.float64), control_dt=arm_control_dt, damping=1.5e-1, max_dq=0.48, de_g_alpha=0.07)
        policy_type = "ArmPIDPolicy"
        controller_notes = "tuning_v7: continue monotonic conservative tuning with a slightly lower Kp, larger damped-pinv damping, tighter max_dq, slightly stronger de_g filtering, and a mild posture-regularization increase to further reduce right-hand acceleration while keeping tilt small"
        controller_meta = {"pid_config": {"default_q": right_arm_target, "kp_pose_diag": np.diag(arm_policy.kp_pose), "kd_pose_diag": np.diag(arm_policy.kd_pose), "ki_pose_diag": np.diag(arm_policy.ki_pose), "posture_gain_diag": np.diag(arm_policy.posture_gain), "finite_diff_eps": arm_policy.finite_diff_eps, "damping": arm_policy.damping, "integral_limit": arm_policy.integral_limit, "max_dq": arm_policy.max_dq, "de_g_alpha": arm_policy.de_g_alpha}}
    # ==============================
    # 5. 创建实验输出目录与视频录制器【非核心代码】
    # 每次 run 都单独保存 metadata、轨迹、评估图和视频，方便横向对比。
    # 这是实验管理与结果保存，不是控制数学核心。
    # ==============================
    run_metadata = build_run_metadata(config_file, experiment_name, policy_type, controller_notes, controller_meta, cmd_nominal, simulation_dt, gait_period, warmup_cycles, evaluation_cycles, cooldown_cycles)
    run_dir = create_eval_run_dir("/home/fjk/g1_ws/hold-my-beer-mpc/evaluation", experiment_name, run_metadata)
    video_path = os.path.join(run_dir, "rollout.mp4")
    video_fps = 30
    video_stride = max(1, int(round(1.0 / (simulation_dt * video_fps))))
    video_frames = []
    video_camera = make_video_camera()
    renderer, video_width, video_height = make_video_renderer(m, preferred_width=1280, preferred_height=720)

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
    right_arm_helper = KinematicsHelper(
        m,
        ee_site_name="right_grasp_site",
        joint_indices=right_arm_id_index_scratch.qpos_indices,
        imu_site_name="imu_in_torso",
        position_reference_q=right_arm_target,
    )

    perf_monitor = PerformanceMonitor(step_budget=simulation_dt, arm_budget=arm_control_dt)

    # ==============================
    # 7. 主仿真循环【核心代码】
    # 每一拍的总体顺序是：
    #   腿部控制 -> 上肢状态读取与右臂控制 -> 写入力矩 -> mj_step 推进物理 -> 记录评估 -> RL 更新 -> 可视化与计时
    # 这里是整个文件最需要看懂的部分，因为真正的控制数据流都发生在这里。
    # ==============================
    with mujoco.viewer.launch_passive(m, d) as viewer:
        print(f"坐标轴可视化已开启：世界系/IMU/左右手均显示红=X轴，绿=Y轴，蓝=Z轴 | 实验 = {experiment_name} | 运行 {total_cycles} 个周期 = {eval_duration:.1f}s，其中 warm-up {warmup_cycles} 周期、evaluation {evaluation_cycles} 周期、cooldown {cooldown_cycles} 周期")
        while viewer.is_running() and counter * simulation_dt < eval_duration:
            perf_monitor.start_step()
            
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
            # 1) 上层控制器（PID / LQR）根据当前右臂状态和 torso 扰动，生成右臂参考。
            #    - PID 路径：直接输出 right_arm 的 q_ref / dq_ref
            #    - LQR 路径：除了输出 q_ref / dq_ref，还会额外输出期望关节加速度 ddq_des
            # 2) 下层执行层把参考转成真正施加到 MuJoCo 的力矩：
            #    - 基础项：所有上肢统一先经过 joint-space PD，得到 tau_pd
            #    - LQR 额外项：根据 ddq_des 调用 mj_inverse()，并消去接触约束反力项得到 tau_ff，
            #      最终右臂实际执行的是 tau = tau_ff + tau_pd

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
            raw_torso_acc = torso_state.lin_acc.copy()
            raw_torso_alpha = torso_state.ang_acc.copy()
            if arm_controller == "lqr":
                acc_alpha = float(np.clip(lqr_torso_acc_filter_alpha, 0.0, 1.0))
                alpha_alpha = float(np.clip(lqr_torso_alpha_filter_alpha, 0.0, 1.0))
                torso_acc_limited = np.clip(torso_state.lin_acc, -lqr_torso_acc_limit, lqr_torso_acc_limit)
                torso_alpha_limited = np.clip(torso_state.ang_acc, -lqr_torso_alpha_limit, lqr_torso_alpha_limit)
                filtered_torso_acc = acc_alpha * torso_acc_limited + (1.0 - acc_alpha) * filtered_torso_acc
                filtered_torso_alpha = alpha_alpha * torso_alpha_limited + (1.0 - alpha_alpha) * filtered_torso_alpha
                torso_state.lin_acc = filtered_torso_acc.copy()
                torso_state.ang_acc = filtered_torso_alpha.copy()

            # 把 torso 的世界系运动量打包成 torso_disturbance，供 KinematicsHelper / LQR 局部线性化使用。
            torso_disturbance = right_arm_helper.build_disturbance_input(
                acc_world=torso_state.lin_acc,
                omega_world=torso_state.ang_vel,
                alpha_world=torso_state.ang_acc,
                rot_world_body=torso_state.rotmat,
            )
            
            # right_arm_obs 是上层右臂控制器看到的“当前观测”；其中包含右臂状态、torso 姿态与运动信息，以及右臂控制周期。
            right_arm_obs = build_right_arm_observation(right_arm_q, right_arm_dq, torso_state, arm_control_dt)
            if counter % arm_control_decimation == 0:
                perf_monitor.start_arm_control()
                # right_arm_helpers 里封装了当前步右臂控制要用到的运动学量、重力误差计算和 LQR 线性化接口
                right_arm_helpers = right_arm_helper.build_helpers(d, disturbance=torso_disturbance)
                right_ee_position_reference_torso = right_arm_helpers.torso_relative_position_reference.copy()
                if arm_controller == "lqr":
                    # LQR 先在上层求解：
                    # - target_right_arm_q / dq：给下层 PD 跟踪的参考轨迹
                    # - desired_right_arm_ddq：期望关节加速度，后面用于 computed torque 前馈
                    target_right_arm_q, target_right_arm_dq, desired_right_arm_ddq = arm_policy.compute_action(right_arm_obs, right_arm_helpers)
                    lqr_diagnostics = arm_policy.get_last_diagnostics()
                    raw_right_arm_ddq = lqr_diagnostics["ddq_raw"]
                    lqr_one_step_prediction = lqr_diagnostics["one_step_prediction"]
                else:
                    # PID 路径只输出右臂参考轨迹，不单独生成期望加速度
                    target_right_arm_q, target_right_arm_dq = arm_policy.compute_action(right_arm_obs, right_arm_helpers)
                perf_monitor.finish_arm_control()

            # 把“固定的腰/左臂目标”和“在线计算的右臂目标”拼回完整上肢目标
            target_arm_waist_q = np.concatenate([waist_left_target_q, target_right_arm_q])
            target_arm_waist_dq = np.concatenate([waist_left_target_dq, target_right_arm_dq])

            # 第一层执行：统一用 joint-space PD 把目标状态转成上肢控制力矩 tau_pd
            tau_arm_waist = pd_control(
                target_arm_waist_q, arm_waist_q, arm_waist_kps,
                target_arm_waist_dq, arm_waist_dq, arm_waist_kds
            )
            # 只切出右臂 5 维的 PD 力矩；如果是 LQR，后面还要和逆动力学前馈叠加
            right_arm_tau_pd = tau_arm_waist[6:11].copy()
            right_arm_tau_ff = np.zeros(5, dtype=np.float64)
            right_arm_tau_inverse = np.zeros(5, dtype=np.float64)
            right_arm_tau_contact = np.zeros(5, dtype=np.float64)
            right_arm_tau_constraint_total = np.zeros(5, dtype=np.float64)
            right_arm_tau_constraint_noncontact = np.zeros(5, dtype=np.float64)
            right_arm_tau_cmd_raw = right_arm_tau_pd.copy()
            right_arm_tau_limit_lower = right_arm_id_index_scratch.torque_limits[:, 0].copy()
            right_arm_tau_limit_upper = right_arm_id_index_scratch.torque_limits[:, 1].copy()
            if arm_controller == "lqr":
                # 第二层执行（仅 LQR）：
                # 用 desired_right_arm_ddq 作为右臂的期望关节加速度，调用 mj_inverse() 计算 tau_ff。
                # apply_computed_torque_control() 内部会：
                # 1) 复制当前整机 qpos / qvel 到 scratch data
                # 2) 在 scratch.qacc 中只给右臂 5 个自由度填入 desired_right_arm_ddq
                # 3) 调用 mj_inverse()，从全部约束中只重建 contact 广义力
                # 4) qfrc_inverse 只加 contact，保留 frictionloss 等非接触约束补偿
                perf_monitor.start_computed_torque_control()
                right_arm_tau, inverse_result = apply_computed_torque_control(
                    m,
                    d,
                    right_arm_id_index_scratch,
                    desired_right_arm_ddq,
                    right_arm_tau_pd,
                )
                perf_monitor.finish_computed_torque_control()
                right_arm_tau_ff = inverse_result.tau_ff
                right_arm_tau_inverse = inverse_result.tau_inverse
                right_arm_tau_contact = inverse_result.tau_contact
                right_arm_tau_constraint_total = inverse_result.tau_constraint_total
                right_arm_tau_constraint_noncontact = inverse_result.tau_constraint_noncontact
                right_arm_tau_cmd_raw = right_arm_tau_ff + right_arm_tau_pd
                # 用 computed torque 的结果替换右臂原来的纯 PD 力矩
                tau_arm_waist[6:11] = right_arm_tau
            # 最终把完整的上肢力矩（腰 + 左臂 + 右臂）写进 d.ctrl[12:23]
            d.ctrl[12:23] = tau_arm_waist

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
                right_arm_control={
                    "arm_policy_updated": counter % arm_control_decimation == 0,
                    "target_q": target_right_arm_q,
                    "target_dq": target_right_arm_dq,
                    "ddq_raw": raw_right_arm_ddq,
                    "ddq_des": desired_right_arm_ddq,
                    "ddq_saturation_limit": np.inf,
                    "tau_inverse": right_arm_tau_inverse,
                    "tau_contact": right_arm_tau_contact,
                    "tau_constraint_total": right_arm_tau_constraint_total,
                    "tau_constraint_noncontact": right_arm_tau_constraint_noncontact,
                    "tau_ff": right_arm_tau_ff,
                    "tau_pd": right_arm_tau_pd,
                    "tau_cmd_raw": right_arm_tau_cmd_raw,
                    "tau_limit_lower": right_arm_tau_limit_lower,
                    "tau_limit_upper": right_arm_tau_limit_upper,
                    "torso_lin_vel_world": torso_state.lin_vel,
                    "torso_ang_vel_world": torso_state.ang_vel,
                    "torso_acc_world_raw": raw_torso_acc,
                    "torso_acc_world_used": torso_state.lin_acc,
                    "torso_alpha_world_raw": raw_torso_alpha,
                    "torso_alpha_world_used": torso_state.ang_acc,
                    "ee_position_reference_torso": right_ee_position_reference_torso,
                    "lqr_one_step_prediction": (
                        lqr_one_step_prediction
                        if arm_controller == "lqr" and counter % arm_control_decimation == 0
                        else None
                    ),
                },
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
                if count < eval_end_time:
                    cmd_runtime = cmd_nominal
                else:
                    cooldown_ratio = np.clip((count - eval_end_time) / max(eval_duration - eval_end_time, 1e-8), 0.0, 1.0)
                    cmd_runtime = (1.0 - cooldown_ratio) * cmd_nominal
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
            draw_debug_axes(viewer.user_scn, d, scene_ids)
            viewer.sync()
            perf_monitor.finish_step(counter)

        perf_monitor.print_summary()

    # ==============================
    # 8. 收尾保存【非核心代码】
    # 退出 viewer 后统一保存轨迹、评估指标、视频和性能统计，再释放 renderer。
    # 这是实验收尾，不是控制逻辑核心。
    # ==============================
        finalize_run(run_dir, buffers, xml_path, simulation_dt, video_path, video_frames, video_fps, renderer is not None, video_width, video_height, d, scene_ids, eval_start_time, eval_end_time, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles, gait_period, experiment_name, perf_monitor=perf_monitor, lqr_cost_definition=lqr_cost_definition)
    close_renderer(renderer)
    renderer = None
