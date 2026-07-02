import os
import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

# --- 引入我们独立的策略 ---
from arm_pid import ArmPIDPolicy
from arm_lqr import ArmLQRPolicy
# from arm_mpc import ArmMPCPolicy
from kinematics_helper import KinematicsHelper
from sim_support import build_run_metadata, create_eval_run_dir, draw_debug_axes, finalize_run, get_gravity_orientation, get_site_vel, init_eval_buffers, make_video_camera, make_video_renderer, pd_control, record_eval_step, resolve_scene_ids


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"/home/fjk/g1_ws/hold-my-beer-mpc/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
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
        cmd_nominal = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
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
    eval_duration = min(simulation_duration, total_cycles * gait_period)
    experiment_name = f"left_fixed_right_{arm_controller}"
    buffers = init_eval_buffers()

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # --- 调试打印：输出关节和驱动器的映射关系 ---
    # print("="*50)
    # print("关节 (Joints - 对应 qpos/qvel):")
    # joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    # for i, name in enumerate(joint_names):
    #     print(f"  Joint ID: {i:2d}, Name: {name}")
    
    # print("\n驱动器 (Actuators - 对应 ctrl):")
    # actuator_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    # for i, name in enumerate(actuator_names):
    #     print(f"  Actuator ID: {i:2d}, Name: {name}")
    # print("="*50)
    
    # load policy
    policy = torch.jit.load(policy_path)

    # --- 实例化右臂控制策略 ---
    right_arm_target = arm_waist_target[6:11].copy()
    if arm_controller == "lqr":
        arm_policy = ArmLQRPolicy(default_q=right_arm_target, control_dt=simulation_dt, horizon=int(config.get("lqr_horizon", 12)))
        policy_type = "ArmLQRPolicy"
        controller_notes = "finite-horizon time-varying LQR baseline with local kinematic linearization and torso-disturbance feedforward terms"
        controller_meta = {"lqr_config": {"horizon": arm_policy.horizon, "control_dt": arm_policy.control_dt, "max_ddq": arm_policy.max_ddq, "max_dq": arm_policy.max_dq}}
    else:
        arm_policy = ArmPIDPolicy(default_q=right_arm_target, kp_pose=np.array([1.20, 1.20], dtype=np.float64), kd_pose=np.array([1.2, 1.2], dtype=np.float64), ki_pose=0.0, posture_gain=np.array([1.15, 1.15, 2.10, 1.15, 0.95], dtype=np.float64), control_dt=simulation_dt, damping=1.5e-1, max_dq=0.48, de_g_alpha=0.07)
        policy_type = "ArmPIDPolicy"
        controller_notes = "tuning_v7: continue monotonic conservative tuning with a slightly lower Kp, larger damped-pinv damping, tighter max_dq, slightly stronger de_g filtering, and a mild posture-regularization increase to further reduce right-hand acceleration while keeping tilt small"
        controller_meta = {"pid_config": {"default_q": right_arm_target, "kp_pose_diag": np.diag(arm_policy.kp_pose), "kd_pose_diag": np.diag(arm_policy.kd_pose), "ki_pose_diag": np.diag(arm_policy.ki_pose), "posture_gain_diag": np.diag(arm_policy.posture_gain), "finite_diff_eps": arm_policy.finite_diff_eps, "damping": arm_policy.damping, "integral_limit": arm_policy.integral_limit, "max_dq": arm_policy.max_dq, "de_g_alpha": arm_policy.de_g_alpha}}
    run_metadata = build_run_metadata(config_file, experiment_name, policy_type, controller_notes, controller_meta, cmd_nominal, simulation_dt, gait_period, warmup_cycles, evaluation_cycles, cooldown_cycles)
    run_dir = create_eval_run_dir("/home/fjk/g1_ws/hold-my-beer-mpc/evaluation", experiment_name, run_metadata)
    video_path = os.path.join(run_dir, "rollout.mp4")
    video_fps = 30
    video_stride = max(1, int(round(1.0 / (simulation_dt * video_fps))))
    video_frames = []
    video_camera = make_video_camera()
    renderer, video_width, video_height = make_video_renderer(m, preferred_width=1280, preferred_height=720)

    # 预先找到主循环中会直接使用的 torso/IMU/左右手 grasp site ID，方便后续读取和可视化
    scene_ids = resolve_scene_ids(m)

    # 右臂 5 个关节在 qpos 中对应索引 25:30；helper 用它来“冻结当前整机，只扰动右臂”
    right_arm_qpos_indices = np.arange(25, 30, dtype=np.int32)
    # 末端名字，5个关节索引，IMU名字
    right_arm_helper = KinematicsHelper(m, ee_site_name="right_grasp_site", joint_indices=right_arm_qpos_indices, imu_site_name="imu_in_torso")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        print(f"坐标轴可视化已开启：世界系/IMU/左右手均显示红=X轴，绿=Y轴，蓝=Z轴 | 实验 = {experiment_name} | 运行 {total_cycles} 个周期 = {eval_duration:.1f}s，其中 warm-up {warmup_cycles} 周期、evaluation {evaluation_cycles} 周期、cooldown {cooldown_cycles} 周期")
        while viewer.is_running() and counter * simulation_dt < eval_duration:
            step_start = time.time()
            
            # --- 1. 腿部控制 (0~11) ---
            # qpos[7:19] 为腿部的 12 个关节，qvel[6:18] 为对应的速度
            leg_q = d.qpos[7:19]
            leg_dq = d.qvel[6:18]
            tau_leg = pd_control(target_dof_pos, leg_q, kps, np.zeros_like(kds), leg_dq, kds)
            d.ctrl[:12] = tau_leg

            # --- 2. 腰部与手臂控制 (12~22) ---
            # 顺序: waist(1), left_arm(5), right_arm(5)
            # 获取了上肢的目前关节位置和速度
            arm_waist_q = d.qpos[19:30]
            arm_waist_dq = d.qvel[18:29]

            # 获取了腰部和左臂的target位置和速度
            waist_left_target_q = arm_waist_target[:6].copy()
            waist_left_target_dq = np.zeros(6, dtype=np.float32)

            # 获取了右臂的目前关节位置和速度
            right_arm_q = arm_waist_q[6:11]
            right_arm_dq = arm_waist_dq[6:11]

            # 获取了躯干的四元数和R
            torso_quat = d.xquat[scene_ids.torso_id]
            torso_rotmat = d.site_xmat[scene_ids.imu_site_id].reshape(3, 3).copy()
            torso_lin_vel, torso_ang_vel = get_site_vel(m, d, scene_ids.imu_site_id)
            torso_acc = np.zeros(3) if counter == 0 else (torso_lin_vel - buffers.prev_torso_lin_vel) / simulation_dt
            torso_alpha = np.zeros(3) if counter == 0 else (torso_ang_vel - buffers.prev_torso_ang_vel) / simulation_dt
            buffers.prev_torso_lin_vel, buffers.prev_torso_ang_vel = torso_lin_vel.copy(), torso_ang_vel.copy()

            # 构建了扰动输入，扰动由躯干的线加速度、角速度、角加速度和R矩阵组成
            disturbance = right_arm_helper.build_disturbance_input(acc_world=torso_acc, omega_world=torso_ang_vel, alpha_world=torso_alpha, rot_world_body=torso_rotmat)
            
            # 构建了右臂的观测
            arm_obs = {
                "current_q": right_arm_q,
                "current_dq": right_arm_dq,
                "torso_quat": torso_quat,
                "torso_omega": torso_ang_vel,
                "torso_acc": torso_acc,
                "torso_alpha": torso_alpha,
                "torso_rotmat": torso_rotmat,
                "dt": simulation_dt,
            }
            helpers = right_arm_helper.build_helpers(d, disturbance=disturbance)
            target_right_arm_q, target_right_arm_dq = arm_policy.compute_action(arm_obs, helpers)

            target_arm_waist_q = np.concatenate([waist_left_target_q, target_right_arm_q])
            target_arm_waist_dq = np.concatenate([waist_left_target_dq, target_right_arm_dq])

            tau_arm_waist = pd_control(
                target_arm_waist_q, arm_waist_q, arm_waist_kps,
                target_arm_waist_dq, arm_waist_dq, arm_waist_kds
            )
            d.ctrl[12:23] = tau_arm_waist

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            record_eval_step(m, d, counter, simulation_dt, scene_ids, buffers)
            if renderer is not None and counter % video_stride == 0:
                renderer.update_scene(d, camera=video_camera)
                video_frames.append(renderer.render().copy())

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation (RL策略只观测腿部状态，需要截取前12个关节)
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
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # 在 viewer 中画出世界系、torso IMU、左手抓持点、右手抓持点的坐标轴
            draw_debug_axes(viewer.user_scn, d, scene_ids)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        finalize_run(run_dir, buffers, xml_path, simulation_dt, video_path, video_frames, video_fps, renderer, video_width, video_height, d, scene_ids, eval_start_time, eval_end_time, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles, gait_period, experiment_name)
