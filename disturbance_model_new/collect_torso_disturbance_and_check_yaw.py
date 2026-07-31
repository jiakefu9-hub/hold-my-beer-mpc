import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_site_velocity_world(model, data, site_id):
    """与主仿真一致地计算 IMU site 的世界系线速度和角速度。"""
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp @ data.qvel, jacr @ data.qvel


def normalize_quaternion_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("遇到零范数四元数，无法转换。")
    return q / n


def quat_to_yaw_wxyz(q):
    w, x, y, z = normalize_quaternion_wxyz(q)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def quat_wxyz_to_rotmat_world_from_imu(q_wxyz):
    w, x, y, z = normalize_quaternion_wxyz(q_wxyz)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def fit_line(x, y):
    coeff = np.polyfit(x, y, deg=1)
    slope, intercept = coeff[0], coeff[1]
    return slope, intercept, slope * x + intercept


def phase_boundary_times(count, phase):
    idx = np.where(np.diff(phase) < -0.5)[0] + 1
    return count[idx]


def save_csv(csv_path, data, yaw_fit):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "count", "phase",
            "quat_w", "quat_x", "quat_y", "quat_z",
            "yaw", "yaw_unwrapped", "yaw_fit_line",
            "lin_acc_local_x", "lin_acc_local_y", "lin_acc_local_z",
            "ang_vel_local_x", "ang_vel_local_y", "ang_vel_local_z",
            "ang_acc_local_x", "ang_acc_local_y", "ang_acc_local_z",
            "lin_vel_world_x", "lin_vel_world_y", "lin_vel_world_z",
            "lin_acc_world_x", "lin_acc_world_y", "lin_acc_world_z",
            "ang_vel_world_x", "ang_vel_world_y", "ang_vel_world_z",
            "ang_acc_world_x", "ang_acc_world_y", "ang_acc_world_z",
            "left_foot_z", "right_foot_z",
        ])
        n = len(data["count"])
        for i in range(n):
            writer.writerow([
                data["count"][i], data["phase"][i], *data["torso_quaternion"][i],
                data["yaw"][i], data["yaw_unwrapped"][i], yaw_fit[i],
                *data["torso_linear_acceleration_local"][i],
                *data["torso_angular_velocity_local"][i],
                *data["torso_angular_acceleration_local"][i],
                *data["torso_linear_velocity_world"][i],
                *data["torso_linear_acceleration_world"][i],
                *data["torso_angular_velocity_world"][i],
                *data["torso_angular_acceleration_world"][i],
                data["left_foot_z"][i], data["right_foot_z"][i],
            ])


def make_plot(png_path, data, discard_time, yaw_mean_stable, yaw_std_stable, yaw_slope_stable, yaw_fit, frame_tag="local"):
    count = np.asarray(data["count"])
    phase = np.asarray(data["phase"])
    yaw = np.asarray(data["yaw"])
    yaw_unwrapped = np.asarray(data["yaw_unwrapped"])
    lin_acc = np.asarray(data[f"torso_linear_acceleration_{frame_tag}"])
    ang_vel = np.asarray(data[f"torso_angular_velocity_{frame_tag}"])
    ang_acc = np.asarray(data[f"torso_angular_acceleration_{frame_tag}"])

    boundaries = phase_boundary_times(count, phase)
    labels = ["x", "y", "z"]
    colors = ["r", "g", "b"]

    fig, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=True)
    fig.canvas.manager.set_window_title("Torso Disturbance Collection + Yaw Check")

    axes[0].plot(count, phase, color="black", label="phase")
    axes[0].set_ylabel("phase")
    axes[0].set_title("Phase vs Time")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(count, yaw, color="tab:blue", label="yaw")
    axes[1].plot(count, yaw_unwrapped, color="tab:orange", label="yaw_unwrapped")
    axes[1].axvline(discard_time, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("rad")
    axes[1].set_title("Yaw and Yaw Unwrapped")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(count, yaw_unwrapped, color="tab:gray", alpha=0.7, label="yaw_unwrapped")
    axes[2].plot(count, yaw_fit, color="tab:red", linewidth=2.0, label="linear fit")
    axes[2].axvline(discard_time, color="gray", linestyle="--", linewidth=1.0)
    axes[2].axhline(yaw_mean_stable, color="tab:green", linestyle="--", linewidth=1.2, label="stable mean yaw")
    axes[2].set_ylabel("rad")
    axes[2].set_title("Yaw Straightness Check")
    axes[2].text(
        0.98, 0.95,
        f"stable mean yaw = {yaw_mean_stable:.6f} rad\n"
        f"stable std = {yaw_std_stable:.6f} rad\n"
        f"stable fitted slope = {yaw_slope_stable:.6f} rad/s",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    for i in range(3):
        axes[3].plot(count, lin_acc[:, i], color=colors[i], label=f"acc_{labels[i]}")
    axes[3].axvline(discard_time, color="gray", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("m/s²")
    axes[3].set_title(f"Torso Linear Acceleration ({frame_tag} frame)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    for i in range(3):
        axes[4].plot(count, ang_vel[:, i], color=colors[i], label=f"omega_{labels[i]}")
    axes[4].axvline(discard_time, color="gray", linestyle="--", linewidth=1.0)
    axes[4].set_ylabel("rad/s")
    axes[4].set_title(f"Torso Angular Velocity ({frame_tag} frame)")
    axes[4].legend(loc="upper right")
    axes[4].grid(True, alpha=0.3)

    for i in range(3):
        axes[5].plot(count, ang_acc[:, i], color=colors[i], label=f"alpha_{labels[i]}")
    axes[5].axvline(discard_time, color="gray", linestyle="--", linewidth=1.0)
    axes[5].set_ylabel("rad/s²")
    axes[5].set_xlabel("time [s]")
    axes[5].set_title(f"Torso Angular Acceleration ({frame_tag} frame)")
    axes[5].legend(loc="upper right")
    axes[5].grid(True, alpha=0.3)

    for ax in axes:
        for t in boundaries:
            ax.axvline(t, color="gray", linestyle="--", linewidth=0.6, alpha=0.35)

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="重新收集 torso 扰动数据，并直接做 yaw 直线性检查")
    parser.add_argument("config_file", type=str, help="config file name in configs/")
    parser.add_argument("--duration", type=float, default=20.0, help="采集总时长，默认 20s")
    parser.add_argument("--discard-time", type=float, default=4.0, help="统计 yaw 时丢弃前几秒，默认 4s")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model_new/torso_disturbance_straight",
        help="输出前缀，默认保存到 disturbance_model_new/",
    )
    args = parser.parse_args()

    project_root = "/home/fjk/g1_ws/hold-my-beer-mpc"
    config_path = os.path.join(project_root, "configs", args.config_file)
    with open(config_path, "r") as f:
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
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    period = 0.8
    counter = 0

    data = {
        "count": [],
        "phase": [],
        "torso_linear_acceleration_local": [],
        "torso_angular_velocity_local": [],
        "torso_angular_acceleration_local": [],
        "torso_linear_velocity_world": [],
        "torso_linear_acceleration_world": [],
        "torso_angular_velocity_world": [],
        "torso_angular_acceleration_world": [],
        "R_world_from_imu": [],
        "torso_quaternion": [],
        "yaw": [],
        "yaw_unwrapped": [],
        "left_foot_z": [],
        "right_foot_z": [],
        "cmd_used": cmd.copy(),
        "discard_time": args.discard_time,
        "collection_duration": args.duration,
        "gait_period": period,
    }

    prev_omega_local = np.zeros(3)
    prev_omega_world = np.zeros(3)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    accel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu-torso-linear-acceleration")
    gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu-torso-angular-velocity")
    accel_adr = m.sensor_adr[accel_id]
    gyro_adr = m.sensor_adr[gyro_id]

    left_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    imu_site_id = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso"
    )
    if imu_site_id < 0:
        raise ValueError("模型中找不到 imu_in_torso site。")
    # 第一个 step 前样本也必须具有有效的 sensor/site 派生量。
    mujoco.mj_forward(m, d)

    policy = torch.jit.load(policy_path)

    print("=" * 60)
    print("开始重新收集 torso 扰动数据，并检查 yaw 是否足够接近直线")
    print(f"cmd_init = {cmd.tolist()}")
    print(f"总时长 = {args.duration:.3f} s, 丢弃前 = {args.discard_time:.3f} s")
    print("=" * 60)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and counter * simulation_dt < args.duration:
            step_start = time.time()
            count = counter * simulation_dt

            # 腿部控制
            leg_q = d.qpos[7:19]
            leg_dq = d.qvel[6:18]
            tau_leg = pd_control(target_dof_pos, leg_q, kps, np.zeros_like(kds), leg_dq, kds)
            d.ctrl[:12] = tau_leg

            # 上肢锁定
            arm_waist_q = d.qpos[19:30]
            arm_waist_dq = d.qvel[18:29]
            target_arm_waist_q = arm_waist_target.copy()
            target_arm_waist_dq = np.zeros_like(target_arm_waist_q)
            tau_arm_waist = pd_control(
                target_arm_waist_q,
                arm_waist_q,
                arm_waist_kps,
                target_arm_waist_dq,
                arm_waist_dq,
                arm_waist_kds,
            )
            d.ctrl[12:23] = tau_arm_waist

            # 【核心数据】在 mj_step 前、与主仿真 MPC 读取扰动完全相同的
            # 物理时刻采样。旧采集器在 step 后读取却沿用 step 前相位，
            # 会给模板带来固定 2 ms 相位偏移。
            phase = count % period / period
            linear_acc_local = d.sensordata[accel_adr:accel_adr + 3].copy()
            linear_vel_world, angular_vel_world = get_site_velocity_world(
                m, d, imu_site_id
            )
            R_W_IMU = d.site_xmat[imu_site_id].reshape(3, 3).copy()
            angular_vel_local = R_W_IMU.T @ angular_vel_world
            angular_acc_local = np.zeros(3) if counter == 0 else (angular_vel_local - prev_omega_local) / simulation_dt
            prev_omega_local = angular_vel_local.copy()

            quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, R_W_IMU.reshape(-1))
            yaw = quat_to_yaw_wxyz(quat)
            linear_acc_world = R_W_IMU @ linear_acc_local + m.opt.gravity
            angular_acc_world = np.zeros(3) if counter == 0 else (angular_vel_world - prev_omega_world) / simulation_dt
            prev_omega_world = angular_vel_world.copy()

            data["count"].append(count)
            data["phase"].append(phase)
            data["torso_linear_acceleration_local"].append(linear_acc_local)
            data["torso_angular_velocity_local"].append(angular_vel_local)
            data["torso_angular_acceleration_local"].append(angular_acc_local)
            data["torso_linear_velocity_world"].append(linear_vel_world)
            data["torso_linear_acceleration_world"].append(linear_acc_world)
            data["torso_angular_velocity_world"].append(angular_vel_world)
            data["torso_angular_acceleration_world"].append(angular_acc_world)
            data["R_world_from_imu"].append(R_W_IMU)
            data["torso_quaternion"].append(quat)
            data["yaw"].append(yaw)
            data["left_foot_z"].append(d.xpos[left_foot_id][2])
            data["right_foot_z"].append(d.xpos[right_foot_id][2])

            if counter > 0 and counter % int(period / simulation_dt) == 0:
                print(
                    f"[采集中] t={count:.2f}s | phase={phase:.2f} | "
                    f"yaw={yaw:.4f} rad | acc_local_z={linear_acc_local[2]:.2f} m/s² | acc_world_z={linear_acc_world[2]:.2f} m/s²"
                )

            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[7:19]
                dqj = d.qvel[6:18]
                quat_base = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat_base)
                omega = omega * ang_vel_scale

                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9:9 + num_actions] = qj
                obs[9 + num_actions:9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions:9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # 后处理 yaw
    data["count"] = np.asarray(data["count"], dtype=np.float64)
    data["phase"] = np.asarray(data["phase"], dtype=np.float64)
    data["torso_linear_acceleration_local"] = np.asarray(data["torso_linear_acceleration_local"], dtype=np.float64)
    data["torso_angular_velocity_local"] = np.asarray(data["torso_angular_velocity_local"], dtype=np.float64)
    data["torso_angular_acceleration_local"] = np.asarray(data["torso_angular_acceleration_local"], dtype=np.float64)
    data["torso_linear_velocity_world"] = np.asarray(data["torso_linear_velocity_world"], dtype=np.float64)
    data["torso_linear_acceleration_world"] = np.asarray(data["torso_linear_acceleration_world"], dtype=np.float64)
    data["torso_angular_velocity_world"] = np.asarray(data["torso_angular_velocity_world"], dtype=np.float64)
    data["torso_angular_acceleration_world"] = np.asarray(data["torso_angular_acceleration_world"], dtype=np.float64)
    data["R_world_from_imu"] = np.asarray(data["R_world_from_imu"], dtype=np.float64)
    data["torso_quaternion"] = np.asarray(data["torso_quaternion"], dtype=np.float64)
    data["yaw"] = np.asarray(data["yaw"], dtype=np.float64)
    data["left_foot_z"] = np.asarray(data["left_foot_z"], dtype=np.float64)
    data["right_foot_z"] = np.asarray(data["right_foot_z"], dtype=np.float64)

    data["yaw_unwrapped"] = np.unwrap(data["yaw"])

    stable_mask = data["count"] >= args.discard_time
    if stable_mask.sum() < 10:
        raise RuntimeError("丢弃前几秒后剩余样本太少，请增大 duration 或减小 discard-time。")

    stable_t = data["count"][stable_mask]
    stable_yaw = data["yaw_unwrapped"][stable_mask]
    yaw_mean_stable = float(stable_yaw.mean())
    yaw_std_stable = float(stable_yaw.std())
    yaw_slope_stable, yaw_intercept_stable, yaw_fit_all = fit_line(data["count"], data["yaw_unwrapped"])
    yaw_slope_stable_refit, yaw_intercept_stable_refit, yaw_fit_stable_part = fit_line(stable_t, stable_yaw)

    # 用稳定段的拟合结果延拓到全时域，更符合你的“舍弃前几秒后再判断”
    yaw_fit_all = yaw_slope_stable_refit * data["count"] + yaw_intercept_stable_refit

    # 调整建议
    current_yaw_cmd = float(cmd[2])
    if abs(yaw_slope_stable_refit) < 0.003:
        suggestion = "当前 yaw 漂移已经很小，cmd[2] 基本可以保持不变。"
        suggested_cmd2 = current_yaw_cmd
    elif yaw_slope_stable_refit < 0.0:
        suggestion = "稳定段 yaw 拟合斜率为负，说明仍在往负方向缓慢偏；建议把 cmd[2] 再略微调大一点。"
        suggested_cmd2 = current_yaw_cmd + 0.5 * abs(yaw_slope_stable_refit)
    else:
        suggestion = "稳定段 yaw 拟合斜率为正，说明补偿略大；建议把 cmd[2] 再略微调小一点。"
        suggested_cmd2 = current_yaw_cmd - 0.5 * abs(yaw_slope_stable_refit)

    prefix = args.output_prefix
    npz_path = prefix + ".npz"
    csv_path = prefix + "_preview.csv"
    png_local_path = prefix + "_local.png"
    png_world_path = prefix + "_world.png"

    np.savez(
        npz_path,
        **data,
        yaw_mean_stable=yaw_mean_stable,
        yaw_std_stable=yaw_std_stable,
        yaw_slope_stable=yaw_slope_stable_refit,
        yaw_intercept_stable=yaw_intercept_stable_refit,
        yaw_fit_line=yaw_fit_all,
        suggested_cmd2=suggested_cmd2,
    )
    save_csv(csv_path, data, yaw_fit_all)
    make_plot(
        png_local_path,
        data,
        args.discard_time,
        yaw_mean_stable,
        yaw_std_stable,
        yaw_slope_stable_refit,
        yaw_fit_all,
        frame_tag="local",
    )
    make_plot(
        png_world_path,
        data,
        args.discard_time,
        yaw_mean_stable,
        yaw_std_stable,
        yaw_slope_stable_refit,
        yaw_fit_all,
        frame_tag="world",
    )

    print("\n" + "=" * 60)
    print("采集完成")
    print("=" * 60)
    print(f"数据已保存: {npz_path}")
    print(f"预览 CSV: {csv_path}")
    print(f"local 图像已保存        : {png_local_path}")
    print(f"world 图像已保存        : {png_world_path}")
    print(f"当前 cmd[2]                : {current_yaw_cmd:.6f}")
    print(f"稳定段平均 yaw            : {yaw_mean_stable:.6f} rad")
    print(f"稳定段 yaw 标准差         : {yaw_std_stable:.6f} rad")
    print(f"稳定段 yaw 线性拟合斜率   : {yaw_slope_stable_refit:.6f} rad/s")
    print(f"建议尝试的 cmd[2]         : {suggested_cmd2:.6f}")
    print(f"判断                      : {suggestion}")
    print("=" * 60)


if __name__ == "__main__":
    main()
