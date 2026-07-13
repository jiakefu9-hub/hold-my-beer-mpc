import csv
import json
import os
import time
from dataclasses import dataclass, field
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
class TorsoMotionState:
    quat: np.ndarray
    rotmat: np.ndarray
    lin_vel: np.ndarray
    ang_vel: np.ndarray
    lin_acc: np.ndarray
    ang_acc: np.ndarray


@dataclass
class DirectDriveJointGroup:
    joint_names: tuple
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    ctrl_indices: np.ndarray
    joint_ids: np.ndarray
    actuator_ids: np.ndarray
    torque_limits: np.ndarray
    inverse_dynamics_data: mujoco.MjData


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


# ==============================
# 核心代码：主控制链直接依赖的支持函数
# 这部分最值得优先阅读，主要服务 main_sim.py 的右臂控制与状态构建。
# ==============================
def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_site_vel(model, data, site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp @ data.qvel, jacr @ data.qvel


def update_torso_motion_state(model, data, scene_ids, buffers, counter, simulation_dt):
    lin_vel, ang_vel = get_site_vel(model, data, scene_ids.imu_site_id)
    lin_acc = np.zeros(3) if counter == 0 else (lin_vel - buffers.prev_torso_lin_vel) / simulation_dt
    ang_acc = np.zeros(3) if counter == 0 else (ang_vel - buffers.prev_torso_ang_vel) / simulation_dt
    buffers.prev_torso_lin_vel, buffers.prev_torso_ang_vel = lin_vel.copy(), ang_vel.copy()
    return TorsoMotionState(
        quat=data.xquat[scene_ids.torso_id].copy(),
        rotmat=data.site_xmat[scene_ids.imu_site_id].reshape(3, 3).copy(),
        lin_vel=lin_vel.copy(),
        ang_vel=ang_vel.copy(),
        lin_acc=lin_acc,
        ang_acc=ang_acc,
    )


def build_right_arm_observation(current_q, current_dq, torso_state, dt):
    return {
        "current_q": current_q,
        "current_dq": current_dq,
        "torso_quat": torso_state.quat,
        "torso_omega": torso_state.ang_vel,
        "torso_acc": torso_state.lin_acc,
        "torso_alpha": torso_state.ang_acc,
        "torso_rotmat": torso_state.rotmat,
        "dt": dt,
    }


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


# ==============================
# 核心代码：右臂执行层与逆动力学前馈（这一整段是 sim_support.py 最值得优先阅读的部分）
# 这部分负责把 right_arm 的索引上下文、ddq_des 和 tau_pd 变成最终执行力矩。
# ==============================
def resolve_direct_drive_joint_group(
    model,
    joint_names,
    expected_qpos_indices,
    expected_qvel_indices,
    expected_ctrl_indices,
    group_label="关节组",
):
    joint_names = tuple(joint_names)
    qpos_indices = np.asarray(expected_qpos_indices, dtype=np.int32)
    qvel_indices = np.asarray(expected_qvel_indices, dtype=np.int32)
    ctrl_indices = np.asarray(expected_ctrl_indices, dtype=np.int32)

    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names],
        dtype=np.int32,
    )
    actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in joint_names],
        dtype=np.int32,
    )
    if len(joint_names) != len(qpos_indices) or len(joint_names) != len(qvel_indices) or len(joint_names) != len(ctrl_indices):
        raise ValueError(f"{group_label} joint/qpos/qvel/ctrl 数量不一致。")
    if np.any(joint_ids < 0) or np.any(actuator_ids < 0):
        missing_joints = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
        missing_actuators = [name for name, actuator_id in zip(joint_names, actuator_ids) if actuator_id < 0]
        raise ValueError(f"{group_label} 找不到 joint/actuator: joints={missing_joints}, actuators={missing_actuators}")
    if not np.array_equal(model.jnt_qposadr[joint_ids], qpos_indices):
        raise ValueError(f"{group_label} qpos 索引与预期不一致。")
    if not np.array_equal(model.jnt_dofadr[joint_ids], qvel_indices):
        raise ValueError(f"{group_label} qvel 索引与预期不一致。")
    if not np.array_equal(actuator_ids, ctrl_indices):
        raise ValueError(f"{group_label} ctrl 索引与预期不一致。")
    if not np.array_equal(model.actuator_trnid[actuator_ids, 0], joint_ids):
        raise ValueError(f"{group_label} actuator 没有一一驱动对应 joint。")
    if not np.allclose(model.actuator_gear[actuator_ids, 0], 1.0):
        raise ValueError(f"{group_label} actuator 不是 gear=1 的 direct-drive 映射。")

    torque_limits = model.jnt_actfrcrange[joint_ids].copy()
    if not np.all(torque_limits[:, 0] < torque_limits[:, 1]):
        raise ValueError(f"{group_label} 必须在 XML 中配置有效的 actuatorfrcrange。")

    return DirectDriveJointGroup(
        joint_names=joint_names,
        qpos_indices=qpos_indices,
        qvel_indices=qvel_indices,
        ctrl_indices=ctrl_indices,
        joint_ids=joint_ids,
        actuator_ids=actuator_ids,
        torque_limits=torque_limits,
        inverse_dynamics_data=mujoco.MjData(model),
    )


def resolve_right_arm_control_context(model, joint_names):
    return resolve_direct_drive_joint_group(
        model,
        joint_names,
        expected_qpos_indices=np.arange(25, 30, dtype=np.int32),
        expected_qvel_indices=np.arange(24, 29, dtype=np.int32),
        expected_ctrl_indices=np.arange(18, 23, dtype=np.int32),
        group_label="右臂逆动力学",
    )


def inverse_dynamics_feedforward(model, data, scratch, desired_qacc, qvel_indices):
    """计算指定自由度期望加速度对应的 MuJoCo 逆动力学广义力。"""
    qvel_indices = np.asarray(qvel_indices, dtype=np.int32)
    desired_qacc = np.asarray(desired_qacc, dtype=np.float64)
    if desired_qacc.shape != qvel_indices.shape:
        raise ValueError(
            f"desired_qacc shape {desired_qacc.shape} 与 qvel_indices shape "
            f"{qvel_indices.shape} 不一致。"
        )

    scratch.time = data.time
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = data.qvel
    scratch.qacc[:] = 0.0
    scratch.qacc[qvel_indices] = desired_qacc
    if model.nmocap:
        scratch.mocap_pos[:] = data.mocap_pos
        scratch.mocap_quat[:] = data.mocap_quat

    mujoco.mj_inverse(model, scratch)
    return scratch.qfrc_inverse[qvel_indices].copy()


def apply_computed_torque_control(model, data, id_index_scratch, desired_qacc, tau_pd):
    tau_pd = np.asarray(tau_pd, dtype=np.float64)
    desired_qacc = np.asarray(desired_qacc, dtype=np.float64)
    if tau_pd.shape != id_index_scratch.qvel_indices.shape:
        raise ValueError(f"tau_pd shape {tau_pd.shape} 与控制关节数量 {id_index_scratch.qvel_indices.shape} 不一致。")

    tau_ff = inverse_dynamics_feedforward(
        model,
        data,
        id_index_scratch.inverse_dynamics_data,
        desired_qacc,
        id_index_scratch.qvel_indices,
    )
    tau_cmd = np.clip(
        tau_ff + tau_pd,
        id_index_scratch.torque_limits[:, 0],
        id_index_scratch.torque_limits[:, 1],
    )
    return tau_cmd, tau_ff


# ==============================
# 非核心代码：性能统计与实验辅助（建议放在文件后半部分）
# 这部分主要服务调试、测速和结果保存，不是控制数学核心。
# 如果继续做第二轮整理，应将整个 PerformanceMonitor 区块后移，
# 让 right_arm 执行层与逆动力学前馈整体进入文件前半部分。
# ==============================
@dataclass
class PerformanceMonitor:
    step_budget: float
    arm_budget: Optional[float] = None
    warn_interval: Optional[int] = None
    step_start: float = 0.0
    arm_control_start: float = 0.0
    arm_control_elapsed: float = 0.0
    mj_step_start: float = 0.0
    mj_step_elapsed: float = 0.0
    arm_control_ran: bool = False
    total_steps: int = 0
    total_arm_updates: int = 0
    total_arm_elapsed: float = 0.0
    total_mj_step_elapsed: float = 0.0
    total_other_elapsed: float = 0.0
    total_loop_elapsed: float = 0.0
    max_arm_elapsed: float = 0.0
    max_mj_step_elapsed: float = 0.0
    max_other_elapsed: float = 0.0
    max_loop_elapsed: float = 0.0
    arm_overruns: int = 0
    loop_overruns: int = 0
    window_steps: int = 0
    window_arm_elapsed: float = 0.0
    window_mj_step_elapsed: float = 0.0
    window_other_elapsed: float = 0.0
    window_loop_elapsed: float = 0.0
    window_max_arm_elapsed: float = 0.0
    window_max_mj_step_elapsed: float = 0.0
    window_max_other_elapsed: float = 0.0
    window_max_loop_elapsed: float = 0.0
    window_arm_overruns: int = 0
    window_loop_overruns: int = 0
    window_arm_updates: int = 0
    window_reports: list = field(default_factory=list)

    def __post_init__(self):
        if self.arm_budget is None:
            self.arm_budget = self.step_budget
        if self.warn_interval is None:
            self.warn_interval = max(1, int(round(1.0 / self.step_budget)))

    def start_step(self):
        self.step_start = time.perf_counter()

    def start_arm_control(self):
        self.arm_control_start = time.perf_counter()
        self.arm_control_ran = True

    def finish_arm_control(self):
        self.arm_control_elapsed = time.perf_counter() - self.arm_control_start

    def start_mj_step(self):
        self.mj_step_start = time.perf_counter()

    def finish_mj_step(self):
        self.mj_step_elapsed = time.perf_counter() - self.mj_step_start

    def finish_step(self, counter, sleep=True):
        loop_elapsed = time.perf_counter() - self.step_start
        self.record_step(loop_elapsed)
        self.print_window_summary_if_needed(counter)
        if sleep:
            time_until_next_step = self.step_budget - loop_elapsed
            if time_until_next_step > 0.0:
                time.sleep(time_until_next_step)
        return loop_elapsed

    def record_step(self, loop_elapsed):
        other_elapsed = max(0.0, loop_elapsed - self.arm_control_elapsed - self.mj_step_elapsed)
        self.total_steps += 1
        self.total_mj_step_elapsed += self.mj_step_elapsed
        self.total_other_elapsed += other_elapsed
        self.total_loop_elapsed += loop_elapsed
        self.max_mj_step_elapsed = max(self.max_mj_step_elapsed, self.mj_step_elapsed)
        self.max_other_elapsed = max(self.max_other_elapsed, other_elapsed)
        self.max_loop_elapsed = max(self.max_loop_elapsed, loop_elapsed)
        if loop_elapsed > self.step_budget:
            self.loop_overruns += 1

        self.window_steps += 1
        self.window_mj_step_elapsed += self.mj_step_elapsed
        self.window_other_elapsed += other_elapsed
        self.window_loop_elapsed += loop_elapsed
        self.window_max_mj_step_elapsed = max(self.window_max_mj_step_elapsed, self.mj_step_elapsed)
        self.window_max_other_elapsed = max(self.window_max_other_elapsed, other_elapsed)
        self.window_max_loop_elapsed = max(self.window_max_loop_elapsed, loop_elapsed)
        if loop_elapsed > self.step_budget:
            self.window_loop_overruns += 1

        if self.arm_control_ran:
            self.total_arm_updates += 1
            self.total_arm_elapsed += self.arm_control_elapsed
            self.max_arm_elapsed = max(self.max_arm_elapsed, self.arm_control_elapsed)
            self.window_arm_updates += 1
            self.window_arm_elapsed += self.arm_control_elapsed
            self.window_max_arm_elapsed = max(self.window_max_arm_elapsed, self.arm_control_elapsed)
            if self.arm_control_elapsed > self.arm_budget:
                self.arm_overruns += 1
                self.window_arm_overruns += 1

        self.arm_control_ran = False
        self.arm_control_elapsed = 0.0
        self.mj_step_elapsed = 0.0

    def print_window_summary_if_needed(self, counter):
        if counter % self.warn_interval != 0 or self.window_steps == 0:
            return

        report = self._build_report("perf", self.window_steps, self.window_arm_updates, self.window_arm_elapsed, self.window_mj_step_elapsed, self.window_other_elapsed, self.window_loop_elapsed, self.window_max_arm_elapsed, self.window_max_mj_step_elapsed, self.window_max_other_elapsed, self.window_max_loop_elapsed, self.window_arm_overruns, self.window_loop_overruns)
        report["end_step"] = int(counter)
        self.window_reports.append(report)
        self._print_report(report)
        self.window_steps = 0
        self.window_arm_elapsed = 0.0
        self.window_mj_step_elapsed = 0.0
        self.window_other_elapsed = 0.0
        self.window_loop_elapsed = 0.0
        self.window_arm_updates = 0
        self.window_max_arm_elapsed = 0.0
        self.window_max_mj_step_elapsed = 0.0
        self.window_max_other_elapsed = 0.0
        self.window_max_loop_elapsed = 0.0
        self.window_arm_overruns = 0
        self.window_loop_overruns = 0

    def print_summary(self):
        if self.total_steps == 0:
            return
        self._print_report(self.build_total_report())

    def build_total_report(self):
        return self._build_report("perf total", self.total_steps, self.total_arm_updates, self.total_arm_elapsed, self.total_mj_step_elapsed, self.total_other_elapsed, self.total_loop_elapsed, self.max_arm_elapsed, self.max_mj_step_elapsed, self.max_other_elapsed, self.max_loop_elapsed, self.arm_overruns, self.loop_overruns)

    def save_report(self, run_dir):
        total_report = self.build_total_report()
        summary_path = os.path.join(run_dir, "perf_summary.json")
        windows_path = os.path.join(run_dir, "perf_windows.csv")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"total": total_report, "warn_interval": self.warn_interval, "window_count": len(self.window_reports)}, f, indent=2, ensure_ascii=False)
        if self.window_reports:
            with open(windows_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.window_reports[0].keys()))
                writer.writeheader()
                writer.writerows(self.window_reports)
        return summary_path, windows_path if self.window_reports else None

    def _build_report(self, label, steps, arm_updates, arm_elapsed, mj_step_elapsed, other_elapsed, loop_elapsed, max_arm_elapsed, max_mj_step_elapsed, max_other_elapsed, max_loop_elapsed, arm_overruns, loop_overruns):
        budget_ms = self.step_budget * 1000.0
        arm_budget_ms = self.arm_budget * 1000.0
        arm_avg_ms = 0.0 if arm_updates == 0 else arm_elapsed / arm_updates * 1000.0
        return {
            "label": label,
            "steps": int(steps),
            "arm_updates": int(arm_updates),
            "budget_ms": float(budget_ms),
            "arm_budget_ms": float(arm_budget_ms),
            "arm_avg_ms": float(arm_avg_ms),
            "arm_max_ms": float(max_arm_elapsed * 1000.0),
            "arm_overruns": int(arm_overruns),
            "mj_step_avg_ms": float(mj_step_elapsed / steps * 1000.0),
            "mj_step_max_ms": float(max_mj_step_elapsed * 1000.0),
            "other_avg_ms": float(other_elapsed / steps * 1000.0),
            "other_max_ms": float(max_other_elapsed * 1000.0),
            "loop_avg_ms": float(loop_elapsed / steps * 1000.0),
            "loop_max_ms": float(max_loop_elapsed * 1000.0),
            "loop_overruns": int(loop_overruns),
        }

    def _print_report(self, report):
        level = "WARN" if report["arm_overruns"] or report["loop_overruns"] else "INFO"
        print(
            f"[{level}] {report['label']}: steps={report['steps']}, budget={report['budget_ms']:.2f} ms, arm_budget={report['arm_budget_ms']:.2f} ms, arm_updates={report['arm_updates']}, "
            f"arm avg/max={report['arm_avg_ms']:.2f}/{report['arm_max_ms']:.2f} ms, arm overruns={report['arm_overruns']}, "
            f"mj_step avg/max={report['mj_step_avg_ms']:.2f}/{report['mj_step_max_ms']:.2f} ms, other avg/max={report['other_avg_ms']:.2f}/{report['other_max_ms']:.2f} ms, "
            f"loop avg/max={report['loop_avg_ms']:.2f}/{report['loop_max_ms']:.2f} ms, loop overruns={report['loop_overruns']}"
        )


def tilt_error_from_rot(rot):
    return (rot.T @ np.array([0.0, 0.0, -9.81]))[:2]


def quat_to_yaw_wxyz(quaternion):
    qw, qx, qy, qz = quaternion
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


# ==============================
# 非核心代码：调试可视化、评估与实验保存
# 这部分对复现实验很重要，但不属于控制器本体逻辑。
# ==============================
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
        trajectory_data={
            "time": [],
            "qpos": [],
            "qvel": [],
            "qacc": [],
            "ctrl": [],
            "right_arm_ddq_des": [],
            "right_arm_tau_ff": [],
            "right_arm_tau_pd": [],
        },
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
        qacc=np.asarray(trajectory_data["qacc"]),
        ctrl=np.asarray(trajectory_data["ctrl"]),
        right_arm_ddq_des=np.asarray(trajectory_data["right_arm_ddq_des"]),
        right_arm_tau_ff=np.asarray(trajectory_data["right_arm_tau_ff"]),
        right_arm_tau_pd=np.asarray(trajectory_data["right_arm_tau_pd"]),
        xml_path=np.array(xml_path),
        simulation_dt=np.array(simulation_dt),
    )


def write_video(video_path, video_frames, video_fps):
    if imageio is None or not video_frames:
        return
    imageio.mimwrite(video_path, video_frames, fps=video_fps, quality=8, macro_block_size=None)


def close_renderer(renderer):
    if renderer is None:
        return
    close_fn = getattr(renderer, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def record_eval_step(model, data, counter, simulation_dt, scene_ids, buffers, right_arm_control=None):
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
    buffers.trajectory_data["qacc"].append(data.qacc.copy())
    buffers.trajectory_data["ctrl"].append(data.ctrl.copy())
    if right_arm_control is None:
        right_arm_control = {}
    buffers.trajectory_data["right_arm_ddq_des"].append(
        np.asarray(right_arm_control.get("ddq_des", np.zeros(5)), dtype=np.float64).copy()
    )
    buffers.trajectory_data["right_arm_tau_ff"].append(
        np.asarray(right_arm_control.get("tau_ff", np.zeros(5)), dtype=np.float64).copy()
    )
    buffers.trajectory_data["right_arm_tau_pd"].append(
        np.asarray(right_arm_control.get("tau_pd", np.zeros(5)), dtype=np.float64).copy()
    )
    buffers.eval_data["time"].append(t)
    buffers.eval_data["torso_yaw"].append(torso_yaw)
    buffers.eval_data["left_ee_lin_acc_world"].append(left_lin_acc)
    buffers.eval_data["left_ee_ang_acc_world"].append(left_ang_acc)
    buffers.eval_data["left_ee_tilt_error"].append(tilt_error_from_rot(left_rot))
    buffers.eval_data["right_ee_lin_acc_world"].append(right_lin_acc)
    buffers.eval_data["right_ee_ang_acc_world"].append(right_ang_acc)
    buffers.eval_data["right_ee_tilt_error"].append(tilt_error_from_rot(right_rot))


def finalize_run(run_dir, buffers, xml_path, simulation_dt, video_path, video_frames, video_fps, has_renderer, video_width, video_height, data, scene_ids, eval_start_time, eval_end_time, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles, gait_period, experiment_name, perf_monitor=None):
    trajectory_path = os.path.join(run_dir, "trajectory.npz")
    save_trajectory(trajectory_path, buffers.trajectory_data, xml_path, simulation_dt)
    write_video(video_path, video_frames, video_fps)
    perf_summary_path, perf_windows_path = (None, None) if perf_monitor is None else perf_monitor.save_report(run_dir)
    walk_distance = float(np.linalg.norm(data.xpos[scene_ids.torso_id][:2] - buffers.torso_xy_start)) if buffers.torso_xy_start is not None else 0.0
    stats, saved_paths = save_eval(run_dir, buffers.eval_data, eval_start_time, eval_end_time, walk_distance, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles, gait_period, experiment_name)
    saved_paths["perf_summary"] = perf_summary_path
    saved_paths["perf_windows"] = perf_windows_path
    print_run_summary(stats, saved_paths, trajectory_path, video_path, has_renderer, video_frames, video_width, video_height, walk_distance, total_cycles, warmup_cycles, evaluation_cycles, cooldown_cycles)


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
    has_renderer,
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
    extra_perf = saved_paths.get("perf_summary") if saved_paths.get("perf_summary") is not None else "未保存 perf 概览"
    print(
        f"文件: {saved_paths['npz']} | {saved_paths['csv']} | {saved_paths['png']} | "
        f"{saved_paths['summary']} | {extra_perf} | {trajectory_path} | {extra_video}"
    )

    if has_renderer:
        print(f"视频分辨率 = {video_width}x{video_height} (受 MuJoCo offscreen framebuffer 限制)")

    for side in ["left", "right"]:
        print(f"{side} | acc mean/std/rms = {stats[f'{side}_acc_mean']:.4f}/{stats[f'{side}_acc_std']:.4f}/{stats[f'{side}_acc_rms']:.4f}")
        print(f"{side} | alpha mean/std/rms = {stats[f'{side}_alpha_mean']:.4f}/{stats[f'{side}_alpha_std']:.4f}/{stats[f'{side}_alpha_rms']:.4f}")
        print(f"{side} | tilt mean/std/rms = {stats[f'{side}_tilt_mean']:.4f}/{stats[f'{side}_tilt_std']:.4f}/{stats[f'{side}_tilt_rms']:.4f}")

    print(
        f"总周期数 = {total_cycles}, warm-up = {warmup_cycles}, evaluation = {evaluation_cycles}, "
        f"cooldown = {cooldown_cycles}, 本次仿真 torso xy 行走距离 = {walk_distance:.3f} m"
    )
