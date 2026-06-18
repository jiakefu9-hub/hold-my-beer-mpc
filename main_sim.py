import csv
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

# --- 引入我们独立的策略 ---
from arm_fixed import ArmFixedPolicy
from arm_pid import ArmPIDPolicy
# from arm_lqr import ArmLQRPolicy
# from arm_mpc import ArmMPCPolicy
from kinematics_helper import KinematicsHelper


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


def add_axis_visual(scene, pos, rot, sphere_radius=0.02, axis_length=0.20, axis_radius=0.008, origin_rgba=None):
    """在 viewer.user_scn 中画出任意局部坐标轴。红=X, 绿=Y, 蓝=Z。"""
    if origin_rgba is None:
        origin_rgba = np.array([1.0, 1.0, 0.0, 0.9])
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([sphere_radius, 0.0, 0.0]), pos, np.eye(3).reshape(-1), origin_rgba)
    scene.ngeom += 1
    axis_colors = [np.array([1.0, 0.0, 0.0, 0.9]), np.array([0.0, 1.0, 0.0, 0.9]), np.array([0.0, 0.0, 1.0, 0.9])]
    for i in range(3):
        end = pos + rot[:, i] * axis_length
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), axis_colors[i])
        mujoco.mjv_connector(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, axis_radius, pos, end)
        scene.ngeom += 1


def get_site_vel(m, d, site_id):
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
    return jacp @ d.qvel, jacr @ d.qvel


def tilt_error_from_rot(rot):
    return (rot.T @ np.array([0.0, 0.0, -9.81]))[:2]


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def create_eval_run_dir(base_dir, experiment_name, run_metadata):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, experiment_name, timestamp)
    os.makedirs(run_dir, exist_ok=False)
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(_to_serializable(run_metadata), f, indent=2, ensure_ascii=False)
    return run_dir


def make_video_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # 录像相机固定在更远的位置，并把注视点放到机器人行进路径中段，避免后半程走出画面。
    cam.lookat[:] = np.array([2.2, 0.0, 0.9])
    cam.distance = 5.2
    cam.azimuth = 150.0
    cam.elevation = -18.0
    return cam


def make_video_renderer(model, preferred_width=1280, preferred_height=720):
    if imageio is None:
        return None, None, None
    vis_global = getattr(model.vis, "global_", None)
    if vis_global is not None:
        try:
            vis_global.offwidth = max(int(vis_global.offwidth), preferred_width)
            vis_global.offheight = max(int(vis_global.offheight), preferred_height)
        except Exception:
            pass
    offwidth = int(getattr(vis_global, "offwidth", preferred_width))
    offheight = int(getattr(vis_global, "offheight", preferred_height))
    width = min(preferred_width, offwidth)
    height = min(preferred_height, offheight)
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        return renderer, width, height
    except Exception as exc:
        print(f"[video] Renderer 初始化失败，已跳过视频保存: {exc}")
        return None, width, height


def save_eval(run_dir, data, eval_start_time, eval_end_time, walk_distance, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles, gait_period, experiment_name):
    def fmt3(v):
        return f"[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]"

    def fmt2(v):
        return f"[{v[0]:.4f}, {v[1]:.4f}]"

    t = np.asarray(data["time"])
    mask = (t >= eval_start_time) & (t < eval_end_time)
    stats = {"gait_period": gait_period, "total_cycles": total_cycles, "warmup_cycles": warmup_cycles, "evaluation_cycles": evaluation_cycles, "cooldown_cycles": cooldown_cycles, "eval_start_time": eval_start_time, "eval_end_time": eval_end_time, "walk_distance_xy": walk_distance}
    sides = ["left", "right"]
    fig, axes = plt.subplots(6, 2, figsize=(20, 12), sharex=True)
    csv_path = os.path.join(run_dir, "metrics_preview.csv")
    png_path = os.path.join(run_dir, "metrics.png")
    npz_path = os.path.join(run_dir, "metrics.npz")
    summary_path = os.path.join(run_dir, "summary.json")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time","side","acc_x","acc_y","acc_z","acc_norm","alpha_x","alpha_y","alpha_z","alpha_norm","tilt_x","tilt_y","tilt_norm"])
        for c, side in enumerate(sides):
            acc = np.asarray(data[f"{side}_ee_lin_acc_world"]); alpha = np.asarray(data[f"{side}_ee_ang_acc_world"]); tilt = np.asarray(data[f"{side}_ee_tilt_error"])
            acc_n, alpha_n, tilt_n = np.linalg.norm(acc, axis=1), np.linalg.norm(alpha, axis=1), np.linalg.norm(tilt, axis=1)
            for i in range(len(t)): w.writerow([t[i], side, acc[i,0], acc[i,1], acc[i,2], acc_n[i], alpha[i,0], alpha[i,1], alpha[i,2], alpha_n[i], tilt[i,0], tilt[i,1], tilt_n[i]])
            for key, arr in [("acc", acc_n), ("alpha", alpha_n), ("tilt", tilt_n)]:
                stats[f"{side}_{key}_mean"] = arr[mask].mean(); stats[f"{side}_{key}_std"] = arr[mask].std(); stats[f"{side}_{key}_rms"] = np.sqrt(np.mean(arr[mask] ** 2))
            stats[f"{side}_acc_xyz_mean"] = acc[mask].mean(axis=0); stats[f"{side}_acc_xyz_std"] = acc[mask].std(axis=0); stats[f"{side}_acc_xyz_rms"] = np.sqrt(np.mean(acc[mask] ** 2, axis=0))
            stats[f"{side}_alpha_xyz_mean"] = alpha[mask].mean(axis=0); stats[f"{side}_alpha_xyz_std"] = alpha[mask].std(axis=0); stats[f"{side}_alpha_xyz_rms"] = np.sqrt(np.mean(alpha[mask] ** 2, axis=0))
            stats[f"{side}_tilt_xy_mean"] = tilt[mask].mean(axis=0); stats[f"{side}_tilt_xy_std"] = tilt[mask].std(axis=0); stats[f"{side}_tilt_xy_rms"] = np.sqrt(np.mean(tilt[mask] ** 2, axis=0))
            cols = ["r", "g", "b"]; labels = ["x", "y", "z"]; styles = ["-", "--", ":"]
            for j in range(3):
                axes[0,c].plot(t, acc[:,j], color=cols[j], ls=styles[j], lw=1.2, alpha=0.9, label=labels[j])
                axes[2,c].plot(t, alpha[:,j], color=cols[j], ls=styles[j], lw=1.2, alpha=0.9, label=labels[j])
            axes[4,c].plot(t, tilt[:,0], color="m", ls="-", lw=1.2, alpha=0.9, label="tilt_x")
            axes[4,c].plot(t, tilt[:,1], color="c", ls="--", lw=1.2, alpha=0.9, label="tilt_y")
            titles = [f"{side} acc xyz", f"{side} acc norm", f"{side} alpha xyz", f"{side} alpha norm", f"{side} tilt x/y", f"{side} tilt norm"]
            for r in [0,2,4]:
                axes[r,c].axvline(eval_start_time, color="gray", ls="--"); axes[r,c].axvline(eval_end_time, color="gray", ls="--"); axes[r,c].legend(loc="upper left", fontsize=8); axes[r,c].grid(True, alpha=0.3)
            axes[0,c].text(0.98,0.95,f"mean={fmt3(stats[f'{side}_acc_xyz_mean'])}\nstd={fmt3(stats[f'{side}_acc_xyz_std'])}\nrms={fmt3(stats[f'{side}_acc_xyz_rms'])}", transform=axes[0,c].transAxes, ha="right", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
            axes[2,c].text(0.98,0.95,f"mean={fmt3(stats[f'{side}_alpha_xyz_mean'])}\nstd={fmt3(stats[f'{side}_alpha_xyz_std'])}\nrms={fmt3(stats[f'{side}_alpha_xyz_rms'])}", transform=axes[2,c].transAxes, ha="right", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
            axes[4,c].text(0.98,0.95,f"mean={fmt2(stats[f'{side}_tilt_xy_mean'])}\nstd={fmt2(stats[f'{side}_tilt_xy_std'])}\nrms={fmt2(stats[f'{side}_tilt_xy_rms'])}", transform=axes[4,c].transAxes, ha="right", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
            norms = [(1, acc_n, "acc"), (3, alpha_n, "alpha"), (5, tilt_n, "tilt")]
            for r, y, key in norms:
                axes[r,c].plot(t, y, lw=1.2); axes[r,c].axvline(eval_start_time, color="gray", ls="--"); axes[r,c].axvline(eval_end_time, color="gray", ls="--"); axes[r,c].axhline(stats[f"{side}_{key}_mean"], color="r", ls="--"); axes[r,c].grid(True, alpha=0.3)
                axes[r,c].text(0.98,0.95,f"mean={stats[f'{side}_{key}_mean']:.6f}\nstd={stats[f'{side}_{key}_std']:.6f}\nrms={stats[f'{side}_{key}_rms']:.6f}", transform=axes[r,c].transAxes, ha="right", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
            for r in range(6): axes[r,c].set_title(titles[r])
        axes[5,0].set_xlabel("time [s]"); axes[5,1].set_xlabel("time [s]")
        fig.suptitle(f"{experiment_name} | left/right palm grasp sites | {warmup_cycles}+{evaluation_cycles}+{cooldown_cycles} cycles\nwalk distance xy = {walk_distance:.3f} m")
        fig.tight_layout(); fig.savefig(png_path, dpi=160); plt.close(fig)
    np.savez(npz_path, **data, **stats)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(stats), f, indent=2, ensure_ascii=False)
    return stats, {"run_dir": run_dir, "csv": csv_path, "png": png_path, "npz": npz_path, "summary": summary_path}


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
        
        cmd_nominal = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    gait_period = 0.8
    warmup_cycles = 2
    evaluation_cycles = 8
    cooldown_cycles = 2
    total_cycles = warmup_cycles + evaluation_cycles + cooldown_cycles
    eval_start_time = warmup_cycles * gait_period
    eval_end_time = (warmup_cycles + evaluation_cycles) * gait_period
    eval_duration = min(simulation_duration, total_cycles * gait_period)
    experiment_name = "left_fixed_right_pid"
    run_dir = None
    eval_data = {"time": [], "left_ee_lin_acc_world": [], "left_ee_ang_acc_world": [], "left_ee_tilt_error": [], "right_ee_lin_acc_world": [], "right_ee_ang_acc_world": [], "right_ee_tilt_error": []}
    prev_left_lin_vel = np.zeros(3); prev_left_ang_vel = np.zeros(3)
    prev_right_lin_vel = np.zeros(3); prev_right_ang_vel = np.zeros(3); torso_xy_start = None

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

    # --- 实例化右臂控制策略（回到 20260618_225317 的 tuning_v7 参数） ---
    right_arm_target = arm_waist_target[6:11].copy()
    arm_policy = ArmPIDPolicy(
        default_q=right_arm_target,
        kp_pose=np.array([1.20, 1.20], dtype=np.float64),
        kd_pose=np.array([1.2, 1.2], dtype=np.float64),
        ki_pose=0.0,
        posture_gain=np.array([1.15, 1.15, 2.10, 1.15, 0.95], dtype=np.float64),
        control_dt=simulation_dt,
        damping=1.5e-1,
        max_dq=0.48,
        de_g_alpha=0.07,
    )
    run_metadata = {
        "config_file": config_file,
        "experiment_name": experiment_name,
        "policy_type": "ArmPIDPolicy",
        "right_arm_joint_names": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"],
        "notes": "tuning_v7: continue monotonic conservative tuning with a slightly lower Kp, larger damped-pinv damping, tighter max_dq, slightly stronger de_g filtering, and a mild posture-regularization increase to further reduce right-hand acceleration while keeping tilt small",
        "cmd_nominal": cmd_nominal,
        "simulation_dt": simulation_dt,
        "gait_period": gait_period,
        "warmup_cycles": warmup_cycles,
        "evaluation_cycles": evaluation_cycles,
        "cooldown_cycles": cooldown_cycles,
        "pid_config": {
            "default_q": right_arm_target,
            "kp_pose_diag": np.diag(arm_policy.kp_pose),
            "kd_pose_diag": np.diag(arm_policy.kd_pose),
            "ki_pose_diag": np.diag(arm_policy.ki_pose),
            "posture_gain_diag": np.diag(arm_policy.posture_gain),
            "finite_diff_eps": arm_policy.finite_diff_eps,
            "damping": arm_policy.damping,
            "integral_limit": arm_policy.integral_limit,
            "max_dq": arm_policy.max_dq,
            "de_g_alpha": arm_policy.de_g_alpha,
        },
    }
    run_dir = create_eval_run_dir("/home/fjk/g1_ws/hold-my-beer-mpc/evaluation", experiment_name, run_metadata)
    trajectory_path = os.path.join(run_dir, "trajectory.npz")
    video_path = os.path.join(run_dir, "rollout.mp4")
    trajectory_data = {"time": [], "qpos": [], "qvel": [], "ctrl": []}
    video_fps = 30
    video_stride = max(1, int(round(1.0 / (simulation_dt * video_fps))))
    video_frames = []
    video_camera = make_video_camera()
    renderer, video_width, video_height = make_video_renderer(m, preferred_width=1280, preferred_height=720)

    # 预先找到 torso_link、torso IMU 和任务末端 grasp site 的 ID，方便后续读取和可视化
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    imu_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso")
    left_grasp_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site")
    right_grasp_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site")

    # 右臂 5 个关节在 qpos 中对应索引 25:30；helper 用它来“冻结当前整机，只扰动右臂”
    right_arm_qpos_indices = np.arange(25, 30, dtype=np.int32)
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
            arm_waist_q = d.qpos[19:30]
            arm_waist_dq = d.qvel[18:29]

            waist_left_target_q = arm_waist_target[:6].copy()
            waist_left_target_dq = np.zeros(6, dtype=np.float32)

            right_arm_q = arm_waist_q[6:11]
            right_arm_dq = arm_waist_dq[6:11]

            torso_quat = d.xquat[torso_id]
            torso_omega = d.cvel[torso_id][3:6] # cvel 前3位角速度，后3位线速度

            arm_obs = {
                "current_q": right_arm_q,
                "current_dq": right_arm_dq,
                "torso_quat": torso_quat,
                "torso_omega": torso_omega,
                "torso_rotmat": d.site_xmat[imu_site_id].reshape(3, 3).copy(),
                "dt": simulation_dt,
            }
            helpers = right_arm_helper.build_helpers(d)
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
            if torso_xy_start is None:
                torso_xy_start = d.xpos[torso_id][:2].copy()
            left_rot = d.site_xmat[left_grasp_site_id].reshape(3, 3).copy()
            right_rot = d.site_xmat[right_grasp_site_id].reshape(3, 3).copy()
            left_lin_vel, left_ang_vel = get_site_vel(m, d, left_grasp_site_id)
            right_lin_vel, right_ang_vel = get_site_vel(m, d, right_grasp_site_id)
            left_lin_acc = np.zeros(3) if counter == 0 else (left_lin_vel - prev_left_lin_vel) / simulation_dt
            left_ang_acc = np.zeros(3) if counter == 0 else (left_ang_vel - prev_left_ang_vel) / simulation_dt
            right_lin_acc = np.zeros(3) if counter == 0 else (right_lin_vel - prev_right_lin_vel) / simulation_dt
            right_ang_acc = np.zeros(3) if counter == 0 else (right_ang_vel - prev_right_ang_vel) / simulation_dt
            prev_left_lin_vel, prev_left_ang_vel = left_lin_vel.copy(), left_ang_vel.copy()
            prev_right_lin_vel, prev_right_ang_vel = right_lin_vel.copy(), right_ang_vel.copy()
            trajectory_data["time"].append(counter * simulation_dt)
            trajectory_data["qpos"].append(d.qpos.copy())
            trajectory_data["qvel"].append(d.qvel.copy())
            trajectory_data["ctrl"].append(d.ctrl.copy())
            if renderer is not None and counter % video_stride == 0:
                renderer.update_scene(d, camera=video_camera)
                video_frames.append(renderer.render().copy())
            eval_data["time"].append(counter * simulation_dt)
            eval_data["left_ee_lin_acc_world"].append(left_lin_acc); eval_data["left_ee_ang_acc_world"].append(left_ang_acc); eval_data["left_ee_tilt_error"].append(tilt_error_from_rot(left_rot))
            eval_data["right_ee_lin_acc_world"].append(right_lin_acc); eval_data["right_ee_ang_acc_world"].append(right_ang_acc); eval_data["right_ee_tilt_error"].append(tilt_error_from_rot(right_rot))

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
            viewer.user_scn.ngeom = 0

            world_pos = np.array([0.0, 0.0, 0.0])
            world_rot = np.eye(3)
            add_axis_visual(viewer.user_scn, world_pos, world_rot, sphere_radius=0.025, axis_length=0.25, axis_radius=0.010, origin_rgba=np.array([1.0, 1.0, 1.0, 0.95]))

            imu_pos = d.site_xpos[imu_site_id].copy()
            imu_rot = d.site_xmat[imu_site_id].reshape(3, 3).copy()
            add_axis_visual(viewer.user_scn, imu_pos, imu_rot, sphere_radius=0.02, axis_length=0.20, axis_radius=0.008, origin_rgba=np.array([1.0, 1.0, 0.0, 0.9]))

            left_grasp_pos = d.site_xpos[left_grasp_site_id].copy()
            left_grasp_rot = d.site_xmat[left_grasp_site_id].reshape(3, 3).copy()
            add_axis_visual(viewer.user_scn, left_grasp_pos, left_grasp_rot, sphere_radius=0.015, axis_length=0.08, axis_radius=0.006, origin_rgba=np.array([1.0, 0.5, 0.0, 0.9]))

            right_grasp_pos = d.site_xpos[right_grasp_site_id].copy()
            right_grasp_rot = d.site_xmat[right_grasp_site_id].reshape(3, 3).copy()
            add_axis_visual(viewer.user_scn, right_grasp_pos, right_grasp_rot, sphere_radius=0.015, axis_length=0.08, axis_radius=0.006, origin_rgba=np.array([0.0, 1.0, 1.0, 0.9]))

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        np.savez(
            trajectory_path,
            time=np.asarray(trajectory_data["time"]),
            qpos=np.asarray(trajectory_data["qpos"]),
            qvel=np.asarray(trajectory_data["qvel"]),
            ctrl=np.asarray(trajectory_data["ctrl"]),
            xml_path=np.array(xml_path),
            simulation_dt=np.array(simulation_dt),
        )
        if renderer is not None and video_frames:
            imageio.mimwrite(video_path, video_frames, fps=video_fps, quality=8, macro_block_size=None)
        walk_distance = float(np.linalg.norm(d.xpos[torso_id][:2] - torso_xy_start)) if torso_xy_start is not None else 0.0
        stats, saved_paths = save_eval(run_dir, eval_data, eval_start_time, eval_end_time, walk_distance, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles, gait_period, experiment_name)
        print(f"评估已保存到目录: {saved_paths['run_dir']}")
        extra_video = video_path if renderer is not None and video_frames else "未保存（缺少 imageio、Renderer 初始化失败或无帧）"
        print(f"文件: {saved_paths['npz']} | {saved_paths['csv']} | {saved_paths['png']} | {saved_paths['summary']} | {trajectory_path} | {extra_video}")
        if renderer is not None:
            print(f"视频分辨率 = {video_width}x{video_height} (受 MuJoCo offscreen framebuffer 限制)")
        for side in ["left", "right"]:
            print(f"{side} | acc mean/std/rms = {stats[f'{side}_acc_mean']:.4f}/{stats[f'{side}_acc_std']:.4f}/{stats[f'{side}_acc_rms']:.4f}")
            print(f"{side} | alpha mean/std/rms = {stats[f'{side}_alpha_mean']:.4f}/{stats[f'{side}_alpha_std']:.4f}/{stats[f'{side}_alpha_rms']:.4f}")
            print(f"{side} | tilt mean/std/rms = {stats[f'{side}_tilt_mean']:.4f}/{stats[f'{side}_tilt_std']:.4f}/{stats[f'{side}_tilt_rms']:.4f}")
        print(f"总周期数 = {total_cycles}, warm-up = {warmup_cycles}, evaluation = {evaluation_cycles}, cooldown = {cooldown_cycles}, 本次仿真 torso xy 行走距离 = {walk_distance:.3f} m")
