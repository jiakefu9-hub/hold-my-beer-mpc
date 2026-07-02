import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import mujoco
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


@dataclass
class SceneIds:
    torso_id: int
    imu_site_id: int
    left_grasp_site_id: int
    right_grasp_site_id: int


@dataclass
class EvalBuffers:
    eval_data: dict
    trajectory_data: dict
    prev_left_lin_vel: np.ndarray
    prev_left_ang_vel: np.ndarray
    prev_right_lin_vel: np.ndarray
    prev_right_ang_vel: np.ndarray
    prev_torso_lin_vel: np.ndarray
    prev_torso_ang_vel: np.ndarray
    torso_xy_start: Optional[np.ndarray] = None


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_site_vel(model, data, site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp @ data.qvel, jacr @ data.qvel


def tilt_error_from_rot(rot):
    return (rot.T @ np.array([0.0, 0.0, -9.81]))[:2]


def quat_to_yaw_wxyz(quaternion):
    qw, qx, qy, qz = quaternion
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def print_model_mappings(model):
    print("=" * 50)
    print("关节 (Joints - 对应 qpos/qvel):")
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    for i, name in enumerate(joint_names):
        print(f"  Joint ID: {i:2d}, Name: {name}")

    print("\n驱动器 (Actuators - 对应 ctrl):")
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    for i, name in enumerate(actuator_names):
        print(f"  Actuator ID: {i:2d}, Name: {name}")
    print("=" * 50)


def resolve_scene_ids(model):
    return SceneIds(
        torso_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"),
        imu_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso"),
        left_grasp_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site"),
        right_grasp_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site"),
    )


def add_axis_visual(scene, pos, rot, sphere_radius=0.02, axis_length=0.20, axis_radius=0.008, origin_rgba=None):
    if origin_rgba is None:
        origin_rgba = np.array([1.0, 1.0, 0.0, 0.9])

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([sphere_radius, 0.0, 0.0]),
        pos,
        np.eye(3).reshape(-1),
        origin_rgba,
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
        mujoco.mjv_connector(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, axis_radius, pos, end)
        scene.ngeom += 1


def draw_debug_axes(scene, data, scene_ids):
    scene.ngeom = 0

    add_axis_visual(
        scene,
        np.array([0.0, 0.0, 0.0]),
        np.eye(3),
        sphere_radius=0.025,
        axis_length=0.25,
        axis_radius=0.010,
        origin_rgba=np.array([1.0, 1.0, 1.0, 0.95]),
    )

    imu_pos = data.site_xpos[scene_ids.imu_site_id].copy()
    imu_rot = data.site_xmat[scene_ids.imu_site_id].reshape(3, 3).copy()
    add_axis_visual(scene, imu_pos, imu_rot, sphere_radius=0.02, axis_length=0.20, axis_radius=0.008)

    left_pos = data.site_xpos[scene_ids.left_grasp_site_id].copy()
    left_rot = data.site_xmat[scene_ids.left_grasp_site_id].reshape(3, 3).copy()
    add_axis_visual(
        scene,
        left_pos,
        left_rot,
        sphere_radius=0.015,
        axis_length=0.08,
        axis_radius=0.006,
        origin_rgba=np.array([1.0, 0.5, 0.0, 0.9]),
    )

    right_pos = data.site_xpos[scene_ids.right_grasp_site_id].copy()
    right_rot = data.site_xmat[scene_ids.right_grasp_site_id].reshape(3, 3).copy()
    add_axis_visual(
        scene,
        right_pos,
        right_rot,
        sphere_radius=0.015,
        axis_length=0.08,
        axis_radius=0.006,
        origin_rgba=np.array([0.0, 1.0, 1.0, 0.9]),
    )


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


def build_run_metadata(config_file, experiment_name, policy_type, controller_notes, controller_meta, cmd_nominal, simulation_dt, gait_period, warmup_cycles, evaluation_cycles, cooldown_cycles):
    return {
        "config_file": config_file,
        "experiment_name": experiment_name,
        "policy_type": policy_type,
        "right_arm_joint_names": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"],
        "notes": controller_notes,
        "cmd_nominal": cmd_nominal,
        "simulation_dt": simulation_dt,
        "gait_period": gait_period,
        "warmup_cycles": warmup_cycles,
        "evaluation_cycles": evaluation_cycles,
        "cooldown_cycles": cooldown_cycles,
        **controller_meta,
    }


def init_eval_buffers():
    return EvalBuffers(
        eval_data={"time": [], "torso_yaw": [], "left_ee_lin_acc_world": [], "left_ee_ang_acc_world": [], "left_ee_tilt_error": [], "right_ee_lin_acc_world": [], "right_ee_ang_acc_world": [], "right_ee_tilt_error": []},
        trajectory_data={"time": [], "qpos": [], "qvel": [], "ctrl": []},
        prev_left_lin_vel=np.zeros(3),
        prev_left_ang_vel=np.zeros(3),
        prev_right_lin_vel=np.zeros(3),
        prev_right_ang_vel=np.zeros(3),
        prev_torso_lin_vel=np.zeros(3),
        prev_torso_ang_vel=np.zeros(3),
    )


def make_video_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([2.2, 0.0, 0.9])
    cam.distance = 5.2
    cam.azimuth = 120.0
    cam.elevation = -25.0
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


def save_trajectory(trajectory_path, trajectory_data, xml_path, simulation_dt):
    np.savez(
        trajectory_path,
        time=np.asarray(trajectory_data["time"]),
        qpos=np.asarray(trajectory_data["qpos"]),
        qvel=np.asarray(trajectory_data["qvel"]),
        ctrl=np.asarray(trajectory_data["ctrl"]),
        xml_path=np.array(xml_path),
        simulation_dt=np.array(simulation_dt),
    )


def write_video(video_path, video_frames, video_fps):
    if imageio is None or not video_frames:
        return
    imageio.mimwrite(video_path, video_frames, fps=video_fps, quality=8, macro_block_size=None)


def record_eval_step(model, data, counter, simulation_dt, scene_ids, buffers):
    if buffers.torso_xy_start is None:
        buffers.torso_xy_start = data.xpos[scene_ids.torso_id][:2].copy()
    left_rot = data.site_xmat[scene_ids.left_grasp_site_id].reshape(3, 3).copy()
    right_rot = data.site_xmat[scene_ids.right_grasp_site_id].reshape(3, 3).copy()
    torso_yaw = quat_to_yaw_wxyz(data.xquat[scene_ids.torso_id].copy())
    left_lin_vel, left_ang_vel = get_site_vel(model, data, scene_ids.left_grasp_site_id)
    right_lin_vel, right_ang_vel = get_site_vel(model, data, scene_ids.right_grasp_site_id)
    left_lin_acc = np.zeros(3) if counter == 0 else (left_lin_vel - buffers.prev_left_lin_vel) / simulation_dt
    left_ang_acc = np.zeros(3) if counter == 0 else (left_ang_vel - buffers.prev_left_ang_vel) / simulation_dt
    right_lin_acc = np.zeros(3) if counter == 0 else (right_lin_vel - buffers.prev_right_lin_vel) / simulation_dt
    right_ang_acc = np.zeros(3) if counter == 0 else (right_ang_vel - buffers.prev_right_ang_vel) / simulation_dt
    buffers.prev_left_lin_vel, buffers.prev_left_ang_vel = left_lin_vel.copy(), left_ang_vel.copy()
    buffers.prev_right_lin_vel, buffers.prev_right_ang_vel = right_lin_vel.copy(), right_ang_vel.copy()
    t = counter * simulation_dt
    buffers.trajectory_data["time"].append(t)
    buffers.trajectory_data["qpos"].append(data.qpos.copy())
    buffers.trajectory_data["qvel"].append(data.qvel.copy())
    buffers.trajectory_data["ctrl"].append(data.ctrl.copy())
    buffers.eval_data["time"].append(t)
    buffers.eval_data["torso_yaw"].append(torso_yaw)
    buffers.eval_data["left_ee_lin_acc_world"].append(left_lin_acc)
    buffers.eval_data["left_ee_ang_acc_world"].append(left_ang_acc)
    buffers.eval_data["left_ee_tilt_error"].append(tilt_error_from_rot(left_rot))
    buffers.eval_data["right_ee_lin_acc_world"].append(right_lin_acc)
    buffers.eval_data["right_ee_ang_acc_world"].append(right_ang_acc)
    buffers.eval_data["right_ee_tilt_error"].append(tilt_error_from_rot(right_rot))


def finalize_run(run_dir, buffers, xml_path, simulation_dt, video_path, video_frames, video_fps, renderer, video_width, video_height, data, scene_ids, eval_start_time, eval_end_time, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles, gait_period, experiment_name):
    trajectory_path = os.path.join(run_dir, "trajectory.npz")
    save_trajectory(trajectory_path, buffers.trajectory_data, xml_path, simulation_dt)
    write_video(video_path, video_frames, video_fps)
    walk_distance = float(np.linalg.norm(data.xpos[scene_ids.torso_id][:2] - buffers.torso_xy_start)) if buffers.torso_xy_start is not None else 0.0
    stats, saved_paths = save_eval(run_dir, buffers.eval_data, eval_start_time, eval_end_time, walk_distance, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles, gait_period, experiment_name)
    print_run_summary(stats, saved_paths, trajectory_path, video_path, renderer, video_frames, video_width, video_height, walk_distance, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles)


def _fmt3(v):
    return f"[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]"


def _fmt2(v):
    return f"[{v[0]:.4f}, {v[1]:.4f}]"


def save_yaw_diagnostics(run_dir, time_values, yaw_values, eval_start_time, eval_end_time):
    t = np.asarray(time_values)
    yaw = np.asarray(yaw_values)
    yaw_unwrapped = np.unwrap(yaw)
    yaw_error = yaw_unwrapped - yaw_unwrapped[0]
    mask = (t >= eval_start_time) & (t < eval_end_time)
    if mask.sum() < 2:
        mask = np.ones_like(t, dtype=bool)
    slope, intercept = np.polyfit(t[mask], yaw_error[mask], 1)
    stats = {
        "yaw_mean": float(np.mean(yaw_error[mask])),
        "yaw_slope": float(slope),
        "yaw_final_drift": float(yaw_error[mask][-1]),
        "max_abs_yaw_error": float(np.max(np.abs(yaw_error[mask]))),
    }
    yaw_png = os.path.join(run_dir, "yaw.png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax in axes:
        ax.axvline(eval_start_time, color="gray", ls="--")
        ax.axvline(eval_end_time, color="gray", ls="--")
        ax.grid(True, alpha=0.3)
    axes[0].plot(t, yaw, lw=1.2); axes[0].set_title("torso yaw [rad]")
    axes[1].plot(t, yaw_error, lw=1.2); axes[1].plot(t, slope * t + intercept, ls="--", lw=1.0); axes[1].set_title("yaw error from first sample [rad]")
    axes[2].plot(t, np.abs(yaw_error), lw=1.2); axes[2].set_title("|yaw error| [rad]")
    axes[2].set_xlabel("time [s]")
    axes[1].text(0.98, 0.95, f"mean={stats['yaw_mean']:.6f}\nslope={stats['yaw_slope']:.6f}\nfinal={stats['yaw_final_drift']:.6f}\nmax_abs={stats['max_abs_yaw_error']:.6f}", transform=axes[1].transAxes, ha="right", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    fig.tight_layout(); fig.savefig(yaw_png, dpi=160); plt.close(fig)
    return stats, yaw_png


def save_eval(
    run_dir,
    data,
    eval_start_time,
    eval_end_time,
    walk_distance,
    total_cycles,
    warmup_cycles,
    evaluation_cycles,
    cooldown_cycles,
    gait_period,
    experiment_name,
):
    t = np.asarray(data["time"])
    mask = (t >= eval_start_time) & (t < eval_end_time)

    stats = {
        "gait_period": gait_period,
        "total_cycles": total_cycles,
        "warmup_cycles": warmup_cycles,
        "evaluation_cycles": evaluation_cycles,
        "cooldown_cycles": cooldown_cycles,
        "eval_start_time": eval_start_time,
        "eval_end_time": eval_end_time,
        "walk_distance_xy": walk_distance,
    }
    yaw_png_path = None
    if "torso_yaw" in data and len(data["torso_yaw"]) > 0:
        yaw_stats, yaw_png_path = save_yaw_diagnostics(run_dir, data["time"], data["torso_yaw"], eval_start_time, eval_end_time)
        stats.update(yaw_stats)

    sides = ["left", "right"]
    fig, axes = plt.subplots(6, 2, figsize=(20, 12), sharex=True)

    csv_path = os.path.join(run_dir, "metrics_preview.csv")
    png_path = os.path.join(run_dir, "metrics.png")
    npz_path = os.path.join(run_dir, "metrics.npz")
    summary_path = os.path.join(run_dir, "summary.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "side",
                "acc_x",
                "acc_y",
                "acc_z",
                "acc_norm",
                "alpha_x",
                "alpha_y",
                "alpha_z",
                "alpha_norm",
                "tilt_x",
                "tilt_y",
                "tilt_norm",
            ]
        )

        for c, side in enumerate(sides):
            acc = np.asarray(data[f"{side}_ee_lin_acc_world"])
            alpha = np.asarray(data[f"{side}_ee_ang_acc_world"])
            tilt = np.asarray(data[f"{side}_ee_tilt_error"])

            acc_n = np.linalg.norm(acc, axis=1)
            alpha_n = np.linalg.norm(alpha, axis=1)
            tilt_n = np.linalg.norm(tilt, axis=1)

            for i in range(len(t)):
                writer.writerow(
                    [
                        t[i],
                        side,
                        acc[i, 0],
                        acc[i, 1],
                        acc[i, 2],
                        acc_n[i],
                        alpha[i, 0],
                        alpha[i, 1],
                        alpha[i, 2],
                        alpha_n[i],
                        tilt[i, 0],
                        tilt[i, 1],
                        tilt_n[i],
                    ]
                )

            for key, arr in [("acc", acc_n), ("alpha", alpha_n), ("tilt", tilt_n)]:
                stats[f"{side}_{key}_mean"] = arr[mask].mean()
                stats[f"{side}_{key}_std"] = arr[mask].std()
                stats[f"{side}_{key}_rms"] = np.sqrt(np.mean(arr[mask] ** 2))

            stats[f"{side}_acc_xyz_mean"] = acc[mask].mean(axis=0)
            stats[f"{side}_acc_xyz_std"] = acc[mask].std(axis=0)
            stats[f"{side}_acc_xyz_rms"] = np.sqrt(np.mean(acc[mask] ** 2, axis=0))

            stats[f"{side}_alpha_xyz_mean"] = alpha[mask].mean(axis=0)
            stats[f"{side}_alpha_xyz_std"] = alpha[mask].std(axis=0)
            stats[f"{side}_alpha_xyz_rms"] = np.sqrt(np.mean(alpha[mask] ** 2, axis=0))

            stats[f"{side}_tilt_xy_mean"] = tilt[mask].mean(axis=0)
            stats[f"{side}_tilt_xy_std"] = tilt[mask].std(axis=0)
            stats[f"{side}_tilt_xy_rms"] = np.sqrt(np.mean(tilt[mask] ** 2, axis=0))

            cols = ["r", "g", "b"]
            labels = ["x", "y", "z"]
            styles = ["-", "--", ":"]

            for j in range(3):
                axes[0, c].plot(t, acc[:, j], color=cols[j], ls=styles[j], lw=1.2, alpha=0.9, label=labels[j])
                axes[2, c].plot(t, alpha[:, j], color=cols[j], ls=styles[j], lw=1.2, alpha=0.9, label=labels[j])

            axes[4, c].plot(t, tilt[:, 0], color="m", ls="-", lw=1.2, alpha=0.9, label="tilt_x")
            axes[4, c].plot(t, tilt[:, 1], color="c", ls="--", lw=1.2, alpha=0.9, label="tilt_y")

            titles = [
                f"{side} acc xyz",
                f"{side} acc norm",
                f"{side} alpha xyz",
                f"{side} alpha norm",
                f"{side} tilt x/y",
                f"{side} tilt norm",
            ]

            for r in [0, 2, 4]:
                axes[r, c].axvline(eval_start_time, color="gray", ls="--")
                axes[r, c].axvline(eval_end_time, color="gray", ls="--")
                axes[r, c].legend(loc="upper left", fontsize=8)
                axes[r, c].grid(True, alpha=0.3)

            axes[0, c].text(
                0.98,
                0.95,
                f"mean={_fmt3(stats[f'{side}_acc_xyz_mean'])}\nstd={_fmt3(stats[f'{side}_acc_xyz_std'])}\nrms={_fmt3(stats[f'{side}_acc_xyz_rms'])}",
                transform=axes[0, c].transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )
            axes[2, c].text(
                0.98,
                0.95,
                f"mean={_fmt3(stats[f'{side}_alpha_xyz_mean'])}\nstd={_fmt3(stats[f'{side}_alpha_xyz_std'])}\nrms={_fmt3(stats[f'{side}_alpha_xyz_rms'])}",
                transform=axes[2, c].transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )
            axes[4, c].text(
                0.98,
                0.95,
                f"mean={_fmt2(stats[f'{side}_tilt_xy_mean'])}\nstd={_fmt2(stats[f'{side}_tilt_xy_std'])}\nrms={_fmt2(stats[f'{side}_tilt_xy_rms'])}",
                transform=axes[4, c].transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )

            for r, y, key in [(1, acc_n, "acc"), (3, alpha_n, "alpha"), (5, tilt_n, "tilt")]:
                axes[r, c].plot(t, y, lw=1.2)
                axes[r, c].axvline(eval_start_time, color="gray", ls="--")
                axes[r, c].axvline(eval_end_time, color="gray", ls="--")
                axes[r, c].axhline(stats[f"{side}_{key}_mean"], color="r", ls="--")
                axes[r, c].grid(True, alpha=0.3)
                axes[r, c].text(
                    0.98,
                    0.95,
                    f"mean={stats[f'{side}_{key}_mean']:.6f}\nstd={stats[f'{side}_{key}_std']:.6f}\nrms={stats[f'{side}_{key}_rms']:.6f}",
                    transform=axes[r, c].transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                )

            for r in range(6):
                axes[r, c].set_title(titles[r])

    axes[5, 0].set_xlabel("time [s]")
    axes[5, 1].set_xlabel("time [s]")
    fig.suptitle(
        f"{experiment_name} | left/right palm grasp sites | "
        f"{warmup_cycles}+{evaluation_cycles}+{cooldown_cycles} cycles\n"
        f"walk distance xy = {walk_distance:.3f} m"
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    np.savez(npz_path, **data, **stats)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(stats), f, indent=2, ensure_ascii=False)

    return stats, {
        "run_dir": run_dir,
        "csv": csv_path,
        "png": png_path,
        "yaw_png": yaw_png_path,
        "npz": npz_path,
        "summary": summary_path,
    }


def print_run_summary(
    stats,
    saved_paths,
    trajectory_path,
    video_path,
    renderer,
    video_frames,
    video_width,
    video_height,
    walk_distance,
    total_cycles,
    warmup_cycles,
    evaluation_cycles,
    cooldown_cycles,
):
    print(f"评估已保存到目录: {saved_paths['run_dir']}")
    extra_video = video_path if video_frames else "未保存（缺少 imageio、Renderer 初始化失败或无帧）"
    print(
        f"文件: {saved_paths['npz']} | {saved_paths['csv']} | {saved_paths['png']} | "
        f"{saved_paths['summary']} | {trajectory_path} | {extra_video}"
    )

    if renderer is not None:
        print(f"视频分辨率 = {video_width}x{video_height} (受 MuJoCo offscreen framebuffer 限制)")

    for side in ["left", "right"]:
        print(f"{side} | acc mean/std/rms = {stats[f'{side}_acc_mean']:.4f}/{stats[f'{side}_acc_std']:.4f}/{stats[f'{side}_acc_rms']:.4f}")
        print(f"{side} | alpha mean/std/rms = {stats[f'{side}_alpha_mean']:.4f}/{stats[f'{side}_alpha_std']:.4f}/{stats[f'{side}_alpha_rms']:.4f}")
        print(f"{side} | tilt mean/std/rms = {stats[f'{side}_tilt_mean']:.4f}/{stats[f'{side}_tilt_std']:.4f}/{stats[f'{side}_tilt_rms']:.4f}")

    print(
        f"总周期数 = {total_cycles}, warm-up = {warmup_cycles}, evaluation = {evaluation_cycles}, "
        f"cooldown = {cooldown_cycles}, 本次仿真 torso xy 行走距离 = {walk_distance:.3f} m"
    )