import csv
import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

# --- 引入我们独立的策略 (所有的算法最终只输出 target_q 目标位置！) ---
from arm_fixed import ArmFixedPolicy
# from arm_pid import ArmPIDPolicy
# from arm_lqr import ArmLQRPolicy
# from arm_mpc import ArmMPCPolicy


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


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
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    period = 0.8  # 步态周期

    # --- 数据采集设置 ---
    # 目前 config 中 simulation_duration 是 60s。对于扰动采集，我们运行 20 秒即可。
    # 20 秒 = 25 个 phase (20 / 0.8 = 25)。其中前 4 秒（5 个 phase）作为启动阶段，后续可丢弃。
    collection_duration = 20.0
    
    disturbance_data = {
        "count": [],
        "phase": [],
        "torso_linear_acceleration": [],
        "torso_angular_velocity": [],
        "torso_angular_acceleration": [],
        "torso_quaternion": [],
        "left_foot_z": [],
        "right_foot_z": []
    }
    prev_omega = np.zeros(3)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # 获取传感器和连杆的 ID 及内存地址 (必须在模型 m 加载后执行)
    accel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu-torso-linear-acceleration")
    gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu-torso-angular-velocity")
    accel_adr = m.sensor_adr[accel_id]
    gyro_adr = m.sensor_adr[gyro_id]

    left_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    # --- 调试打印：输出关节和驱动器的映射关系 ---
    print("="*50)
    print("关节 (Joints - 对应 qpos/qvel):")
    joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    for i, name in enumerate(joint_names):
        print(f"  Joint ID: {i:2d}, Name: {name}")
    
    print("\n驱动器 (Actuators - 对应 ctrl):")
    actuator_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    for i, name in enumerate(actuator_names):
        print(f"  Actuator ID: {i:2d}, Name: {name}")
    print("="*50)
    
    # load policy
    policy = torch.jit.load(policy_path)

    # --- 实例化手臂控制策略 (这里我们传入要锁死的 target_q 数组) ---
    arm_policy = ArmFixedPolicy(target_q=arm_waist_target)

    # 预先找到 torso_link 的 ID，方便后续每步读取它的姿态和角速度
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after collection_duration simulation-seconds.
        print(f"\n--- 开始采集扰动数据 (总时长 {collection_duration} 秒仿真时间, 约 {collection_duration/period:.1f} 个步态周期) ---")
        while viewer.is_running() and counter * simulation_dt < collection_duration:
            step_start = time.time()
            count = counter * simulation_dt
            
            # --- 1. 腿部控制 (0~11) ---
            # qpos[7:19] 为腿部的 12 个关节，qvel[6:18] 为对应的速度
            leg_q = d.qpos[7:19]
            leg_dq = d.qvel[6:18]
            tau_leg = pd_control(target_dof_pos, leg_q, kps, np.zeros_like(kds), leg_dq, kds)
            d.ctrl[:12] = tau_leg

            # --- 2. 腰部与手臂控制 (12~22) ---
            # 1. 提取当前上肢关节状态
            arm_waist_q = d.qpos[19:30]
            arm_waist_dq = d.qvel[18:29]
            
            # 2. 提取躯干(torso_link)的姿态和角速度（这是上肢控制最重要的反馈！）
            torso_quat = d.xquat[torso_id]
            # 恢复为：统一使用局部传感器读取角速度
            torso_omega = d.sensordata[gyro_adr:gyro_adr+3].copy() 
            
            # 3. --- 核心：在这里调用控制策略！---
            # 所有未来的策略 (PID, LQR, MPC) 都会用这个标准接口，吃状态，吐目标角度
            target_arm_waist_q = arm_policy.compute_action(
                torso_quat=torso_quat, 
                torso_omega=torso_omega, 
                current_q=arm_waist_q, 
                current_dq=arm_waist_dq
            )
            
            # 4. 统一执行 PD 控制计算最终力矩 (完全模拟真机底层的电机闭环)
            tau_arm_waist = pd_control(
                target_arm_waist_q, arm_waist_q, arm_waist_kps, 
                np.zeros_like(arm_waist_kds), arm_waist_dq, arm_waist_kds
            )
            d.ctrl[12:23] = tau_arm_waist

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # --- 收集扰动数据 ---
            phase = count % period / period

            # 采集最原始的局部坐标系传感器数据 (带重力偏置，后续统一在后处理阶段转换)
            linear_acc = d.sensordata[accel_adr:accel_adr+3].copy()
            angular_vel = d.sensordata[gyro_adr:gyro_adr+3].copy()
            
            # 计算角加速度 (局部坐标系下的一阶差分)
            if counter == 0:
                angular_acc = np.zeros(3)
            else:
                angular_acc = (angular_vel - prev_omega) / simulation_dt
            prev_omega = angular_vel.copy()
            
            # 读取绝对姿态和双脚高度
            quat = d.xquat[torso_id].copy()
            lf_z = d.xpos[left_foot_id][2]
            rf_z = d.xpos[right_foot_id][2]

            # 存入字典
            disturbance_data["count"].append(count)
            disturbance_data["phase"].append(phase)
            disturbance_data["torso_linear_acceleration"].append(linear_acc)
            disturbance_data["torso_angular_velocity"].append(angular_vel)
            disturbance_data["torso_angular_acceleration"].append(angular_acc)
            disturbance_data["torso_quaternion"].append(quat)
            disturbance_data["left_foot_z"].append(lf_z)
            disturbance_data["right_foot_z"].append(rf_z)

            # 每经过一个完整的周期 (0.8s) 打印一次日志
            if counter > 0 and counter % int(period / simulation_dt) == 0:
                print(f"[数据采集] Time: {count:.2f}s | Phase: {phase:.2f} | Torso Acc-Z: {linear_acc[2]:.2f} m/s² | L-Foot Z: {lf_z:.3f} | R-Foot Z: {rf_z:.3f}")

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

                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        # --- 仿真结束，保存数据 ---
        print("\n--- 仿真结束，正在保存采集的数据 ---")
        npz_path = "/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_data.npz"
        csv_path = "/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_data_preview.csv"
        np.savez(npz_path, **disturbance_data)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "count", "phase",
                "lin_acc_x", "lin_acc_y", "lin_acc_z",
                "ang_vel_x", "ang_vel_y", "ang_vel_z",
                "ang_acc_x", "ang_acc_y", "ang_acc_z",
                "quat_w", "quat_x", "quat_y", "quat_z",
                "left_foot_z", "right_foot_z",
            ])
            for i in range(len(disturbance_data["count"])):
                writer.writerow([
                    disturbance_data["count"][i], disturbance_data["phase"][i],
                    *disturbance_data["torso_linear_acceleration"][i],
                    *disturbance_data["torso_angular_velocity"][i],
                    *disturbance_data["torso_angular_acceleration"][i],
                    *disturbance_data["torso_quaternion"][i],
                    disturbance_data["left_foot_z"][i], disturbance_data["right_foot_z"][i],
                ])

        print(f"数据已成功保存至: {npz_path}")
        print(f"预览 CSV 已导出至: {csv_path}")
        print(f"总计收集了 {len(disturbance_data['count'])} 帧 (每帧包含 8 项物理量)")
        print(f"对应仿真时长: {len(disturbance_data['count']) * simulation_dt:.3f} s")

