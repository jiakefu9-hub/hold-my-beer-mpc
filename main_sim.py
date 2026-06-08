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


def add_imu_axis_visual(scene, pos, rot, axis_length=0.20, axis_radius=0.008):
    """在 viewer.user_scn 中画出 IMU 的位置和局部坐标轴。
    红=X, 绿=Y, 蓝=Z。
    注意：交点严格位于 XML 中 `imu_in_torso` 的 site 位置；如果 IMU 在机身内部，靠拉长坐标轴来显示其位置。
    """
    scene.ngeom = 0

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.02, 0.0, 0.0]),
        pos,
        np.eye(3).reshape(-1),
        np.array([1.0, 1.0, 0.0, 0.9]),
    )
    scene.ngeom += 1

    axis_colors = [
        np.array([1.0, 0.0, 0.0, 0.9]),
        np.array([0.0, 1.0, 0.0, 0.9]),
        np.array([0.0, 0.0, 1.0, 0.9]),
    ]

    for i in range(3):
        end = pos + rot[:, i] * axis_length
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            axis_colors[i],
        )
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            axis_radius,
            pos,
            end,
        )
        scene.ngeom += 1


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

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

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

    # 预先找到 torso_link 和 torso IMU 的 ID，方便后续读取和可视化
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    imu_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        print("IMU 可视化已开启：红=X轴，绿=Y轴，蓝=Z轴")
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
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
            torso_omega = d.cvel[torso_id][3:6] # cvel 前3位角速度，后3位线速度
            
            # 3. --- 核心：在这里调用统一控制策略接口 ---
            # 所有策略统一输入 arm_obs/helpers，统一输出 q_ref, dq_ref
            arm_obs = {
                "current_q": arm_waist_q,
                "current_dq": arm_waist_dq,
                "torso_quat": torso_quat,
                "torso_omega": torso_omega,
            }
            helpers = None
            target_arm_waist_q, target_arm_waist_dq = arm_policy.compute_action(arm_obs, helpers)
            
            # 4. 统一执行 PD 控制计算最终力矩 (模拟真机底层位置/速度闭环)
            tau_arm_waist = pd_control(
                target_arm_waist_q, arm_waist_q, arm_waist_kps,
                target_arm_waist_dq, arm_waist_dq, arm_waist_kds
            )
            d.ctrl[12:23] = tau_arm_waist

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

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

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
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

            # 在 viewer 中画出 torso 上的 IMU 位置与局部坐标轴
            imu_pos = d.site_xpos[imu_site_id].copy()
            imu_rot = d.site_xmat[imu_site_id].reshape(3, 3).copy()
            add_imu_axis_visual(viewer.user_scn, imu_pos, imu_rot)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
