# 怎么用
# - 使用默认输入输出路径：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model_new/build_world_disturbance_templates.py
#
# - 指定输入文件：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model_new/build_world_disturbance_templates.py \
#       --input /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model_new/torso_disturbance_straight.npz
#
# - 指定丢弃时长、bin 数、平滑窗口：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model_new/build_world_disturbance_templates.py \
#       --discard-time 4.0 --num-bins 100 --window-size 5
#
# - 指定输出目录：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model_new/build_world_disturbance_templates.py \
#       --output-dir /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model_new/templates_world

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_KEYS = [
    "count",
    "phase",
    "torso_quaternion",
    "R_world_from_imu",
    "torso_linear_acceleration_world",
    "torso_angular_velocity_world",
    "left_foot_z",
    "right_foot_z",
    "gait_period",
]


def load_npz(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"输入文件不存在: {npz_path}")
    data = np.load(npz_path)
    keys = list(data.keys())
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        raise KeyError(f"输入 npz 缺少必要字段: {missing}")
    return {k: data[k] for k in keys}


def circular_central_difference(values, dt):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("circular_central_difference 需要 shape=(N, D) 的数组。")
    forward = np.roll(values, -1, axis=0)
    backward = np.roll(values, 1, axis=0)
    return (forward - backward) / (2.0 * dt)


def circular_moving_average(values, window_size):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("circular_moving_average 需要 shape=(N, D) 的数组。")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size 必须为正奇数。")
    if window_size == 1:
        return values.copy()

    pad = window_size // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]], axis=0)
    kernel = np.ones(window_size, dtype=np.float64) / window_size

    smoothed = np.zeros_like(values)
    for d in range(values.shape[1]):
        smoothed[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
    return smoothed


def normalize_quaternions_wxyz(quaternions):
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("四元数数组必须为 shape=(N, 4)，顺序为 wxyz。")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-12) or not np.all(np.isfinite(quaternions)):
        raise ValueError("四元数包含零范数、NaN 或 Inf。")
    return quaternions / norms


def markley_quaternion_mean_wxyz(quaternions):
    """在 SO(3) 上做符号不变的四元数均值，而不是逐元素平均。"""
    quaternions = normalize_quaternions_wxyz(quaternions)
    scatter = np.einsum("ni,nj->ij", quaternions, quaternions)
    _, eigenvectors = np.linalg.eigh(scatter)
    mean = eigenvectors[:, -1]
    if np.dot(mean, quaternions[0]) < 0.0:
        mean = -mean
    return mean / np.linalg.norm(mean)


def align_quaternion_sequence_wxyz(quaternions):
    """统一相邻 bin 的 q/-q 符号，便于保存、画图和 SLERP。"""
    aligned = normalize_quaternions_wxyz(quaternions).copy()
    if aligned[0, 0] < 0.0:
        aligned[0] *= -1.0
    for i in range(1, len(aligned)):
        if np.dot(aligned[i - 1], aligned[i]) < 0.0:
            aligned[i] *= -1.0
    return aligned


def quaternion_wxyz_to_rotmat(quaternions):
    quaternions = normalize_quaternions_wxyz(quaternions)
    w, x, y, z = quaternions.T
    rotations = np.empty((len(quaternions), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations


def circular_quaternion_moving_average(quaternions, window_size):
    quaternions = align_quaternion_sequence_wxyz(quaternions)
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size 必须为正奇数。")
    if window_size == 1:
        return quaternions.copy()
    pad = window_size // 2
    smoothed = np.empty_like(quaternions)
    for i in range(len(quaternions)):
        indices = (np.arange(i - pad, i + pad + 1) % len(quaternions)).astype(int)
        smoothed[i] = markley_quaternion_mean_wxyz(quaternions[indices])
    return align_quaternion_sequence_wxyz(smoothed)


def build_raw_template(source, discard_time, num_bins):
    count = np.asarray(source["count"], dtype=np.float64)
    phase = np.asarray(source["phase"], dtype=np.float64)
    orientation_W = normalize_quaternions_wxyz(source["torso_quaternion"])
    orientation_rotmat_W = np.asarray(
        source["R_world_from_imu"], dtype=np.float64
    )
    rotations_from_quaternion = quaternion_wxyz_to_rotmat(orientation_W)
    if (
        orientation_rotmat_W.shape != rotations_from_quaternion.shape
        or not np.all(np.isfinite(orientation_rotmat_W))
        or np.max(
            np.linalg.norm(
                orientation_rotmat_W - rotations_from_quaternion,
                axis=(1, 2),
            )
        )
        > 1e-8
    ):
        raise ValueError("原始四元数与 R_world_from_imu 不一致。")
    acc_W = np.asarray(source["torso_linear_acceleration_world"], dtype=np.float64)
    omega_W = np.asarray(source["torso_angular_velocity_world"], dtype=np.float64)
    left_foot_z = np.asarray(source["left_foot_z"], dtype=np.float64)
    right_foot_z = np.asarray(source["right_foot_z"], dtype=np.float64)
    period = float(source["gait_period"])

    if not (
        len(count)
        == len(phase)
        == len(orientation_W)
        == len(acc_W)
        == len(omega_W)
        == len(left_foot_z)
        == len(right_foot_z)
    ):
        raise ValueError("输入数据长度不一致。")

    mask = count >= discard_time
    count = count[mask]
    phase = phase[mask]
    orientation_W = orientation_W[mask]
    acc_W = acc_W[mask]
    omega_W = omega_W[mask]
    left_foot_z = left_foot_z[mask]
    right_foot_z = right_foot_z[mask]

    if len(count) == 0:
        raise ValueError("丢弃启动段后没有剩余数据。")

    bin_ids = np.floor(phase * num_bins).astype(int)
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)
    phase_centers = (np.arange(num_bins, dtype=np.float64) + 0.5) / num_bins
    dt_bin = period / num_bins

    bin_counts = np.zeros(num_bins, dtype=np.int64)
    valid_bins = np.zeros(num_bins, dtype=bool)

    acc_mean = np.zeros((num_bins, 3), dtype=np.float64)
    acc_std = np.zeros((num_bins, 3), dtype=np.float64)
    omega_mean = np.zeros((num_bins, 3), dtype=np.float64)
    omega_std = np.zeros((num_bins, 3), dtype=np.float64)
    orientation_mean = np.zeros((num_bins, 4), dtype=np.float64)
    orientation_dispersion = np.zeros(num_bins, dtype=np.float64)

    left_mean = np.zeros(num_bins, dtype=np.float64)
    left_std = np.zeros(num_bins, dtype=np.float64)
    right_mean = np.zeros(num_bins, dtype=np.float64)
    right_std = np.zeros(num_bins, dtype=np.float64)

    for b in range(num_bins):
        idx = bin_ids == b
        c = np.count_nonzero(idx)
        bin_counts[b] = c
        if c == 0:
            continue

        valid_bins[b] = True
        acc_mean[b] = acc_W[idx].mean(axis=0)
        acc_std[b] = acc_W[idx].std(axis=0)

        omega_mean[b] = omega_W[idx].mean(axis=0)
        omega_std[b] = omega_W[idx].std(axis=0)
        orientation_mean[b] = markley_quaternion_mean_wxyz(orientation_W[idx])
        dots = np.clip(
            np.abs(orientation_W[idx] @ orientation_mean[b]), 0.0, 1.0
        )
        orientation_errors = 2.0 * np.arccos(dots)
        orientation_dispersion[b] = np.sqrt(np.mean(orientation_errors**2))

        left_mean[b] = left_foot_z[idx].mean()
        left_std[b] = left_foot_z[idx].std()
        right_mean[b] = right_foot_z[idx].mean()
        right_std[b] = right_foot_z[idx].std()

    if not np.all(valid_bins):
        missing = np.where(~valid_bins)[0]
        raise ValueError(f"存在空 bin，无法构建完整周期模板。空 bin 索引: {missing.tolist()}")

    orientation_mean = align_quaternion_sequence_wxyz(orientation_mean)
    orientation_rotmat = quaternion_wxyz_to_rotmat(orientation_mean)
    alpha_template = circular_central_difference(omega_mean, dt_bin)

    return {
        "frame_name": np.array("world"),
        "period": np.array(period, dtype=np.float64),
        "discard_time": np.array(discard_time, dtype=np.float64),
        "num_bins": np.array(num_bins, dtype=np.int64),
        "dt_bin": np.array(dt_bin, dtype=np.float64),
        "phase_centers": phase_centers,
        "bin_counts": bin_counts,
        "valid_bins": valid_bins,
        "torso_linear_acceleration_template": acc_mean,
        "torso_linear_acceleration_std": acc_std,
        "torso_angular_velocity_template": omega_mean,
        "torso_angular_velocity_std": omega_std,
        "torso_angular_acceleration_template": alpha_template,
        "torso_orientation_quaternion_template": orientation_mean,
        "torso_orientation_rotation_matrix_template": orientation_rotmat,
        "torso_orientation_dispersion_rad": orientation_dispersion,
        "orientation_average_method": np.array("markley_quaternion_mean"),
        "orientation_quaternion_order": np.array("wxyz"),
        "orientation_rotation_convention": np.array("world_from_torso_imu"),
        "left_foot_z_mean": left_mean,
        "left_foot_z_std": left_std,
        "right_foot_z_mean": right_mean,
        "right_foot_z_std": right_std,
    }


def build_half_smoothed_template(raw_template, window_size):
    acc_raw = np.asarray(raw_template["torso_linear_acceleration_template"], dtype=np.float64)
    acc_std = np.asarray(raw_template["torso_linear_acceleration_std"], dtype=np.float64)
    omega_raw = np.asarray(raw_template["torso_angular_velocity_template"], dtype=np.float64)
    omega_std = np.asarray(raw_template["torso_angular_velocity_std"], dtype=np.float64)
    orientation_raw = np.asarray(
        raw_template["torso_orientation_quaternion_template"],
        dtype=np.float64,
    )
    dt_bin = float(raw_template["dt_bin"])

    omega_smoothed = circular_moving_average(omega_raw, window_size)
    alpha_from_smoothed = circular_central_difference(omega_smoothed, dt_bin)
    orientation_smoothed = circular_quaternion_moving_average(
        orientation_raw, window_size
    )

    out = {k: raw_template[k] for k in raw_template.keys()}
    out["torso_linear_acceleration_template"] = acc_raw
    out["torso_linear_acceleration_std"] = acc_std
    out["torso_angular_velocity_template_raw"] = omega_raw
    out["torso_angular_velocity_template"] = omega_smoothed
    out["torso_angular_velocity_std"] = omega_std
    out["torso_angular_acceleration_template"] = alpha_from_smoothed
    out["torso_orientation_quaternion_template_raw"] = orientation_raw
    out["torso_orientation_rotation_matrix_template_raw"] = (
        quaternion_wxyz_to_rotmat(orientation_raw)
    )
    out["torso_orientation_quaternion_template"] = orientation_smoothed
    out["torso_orientation_rotation_matrix_template"] = (
        quaternion_wxyz_to_rotmat(orientation_smoothed)
    )
    out["smoothing_window_size"] = np.array(window_size, dtype=np.int64)
    out["template_variant"] = np.array("half_smoothed")
    return out


def build_fully_smoothed_template(raw_template, window_size):
    acc_raw = np.asarray(raw_template["torso_linear_acceleration_template"], dtype=np.float64)
    acc_std = np.asarray(raw_template["torso_linear_acceleration_std"], dtype=np.float64)
    omega_raw = np.asarray(raw_template["torso_angular_velocity_template"], dtype=np.float64)
    omega_std = np.asarray(raw_template["torso_angular_velocity_std"], dtype=np.float64)
    orientation_raw = np.asarray(
        raw_template["torso_orientation_quaternion_template"],
        dtype=np.float64,
    )
    dt_bin = float(raw_template["dt_bin"])

    acc_smoothed = circular_moving_average(acc_raw, window_size)
    omega_smoothed = circular_moving_average(omega_raw, window_size)
    alpha_from_smoothed = circular_central_difference(omega_smoothed, dt_bin)
    orientation_smoothed = circular_quaternion_moving_average(
        orientation_raw, window_size
    )

    out = {k: raw_template[k] for k in raw_template.keys()}
    out["torso_linear_acceleration_template_raw"] = acc_raw
    out["torso_linear_acceleration_template"] = acc_smoothed
    out["torso_linear_acceleration_std"] = acc_std
    out["torso_angular_velocity_template_raw"] = omega_raw
    out["torso_angular_velocity_template"] = omega_smoothed
    out["torso_angular_velocity_std"] = omega_std
    out["torso_angular_acceleration_template"] = alpha_from_smoothed
    out["torso_orientation_quaternion_template_raw"] = orientation_raw
    out["torso_orientation_rotation_matrix_template_raw"] = (
        quaternion_wxyz_to_rotmat(orientation_raw)
    )
    out["torso_orientation_quaternion_template"] = orientation_smoothed
    out["torso_orientation_rotation_matrix_template"] = (
        quaternion_wxyz_to_rotmat(orientation_smoothed)
    )
    out["smoothing_window_size"] = np.array(window_size, dtype=np.int64)
    out["template_variant"] = np.array("fully_smoothed")
    return out


def save_template_npz(template, npz_path):
    np.savez(npz_path, **template)


def save_template_csv(template, csv_path):
    phase_centers = np.asarray(template["phase_centers"])
    bin_counts = np.asarray(template["bin_counts"])
    acc = np.asarray(template["torso_linear_acceleration_template"])
    acc_std = np.asarray(template["torso_linear_acceleration_std"])
    omega = np.asarray(template["torso_angular_velocity_template"])
    omega_std = np.asarray(template["torso_angular_velocity_std"])
    alpha = np.asarray(template["torso_angular_acceleration_template"])
    orientation_quat = np.asarray(
        template["torso_orientation_quaternion_template"]
    )
    orientation_rotmat = np.asarray(
        template["torso_orientation_rotation_matrix_template"]
    )
    orientation_dispersion = np.asarray(
        template["torso_orientation_dispersion_rad"]
    )
    left_mean = np.asarray(template["left_foot_z_mean"])
    left_std = np.asarray(template["left_foot_z_std"])
    right_mean = np.asarray(template["right_foot_z_mean"])
    right_std = np.asarray(template["right_foot_z_std"])

    acc_raw = template["torso_linear_acceleration_template_raw"] if "torso_linear_acceleration_template_raw" in template else None
    omega_raw = template["torso_angular_velocity_template_raw"] if "torso_angular_velocity_template_raw" in template else None

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        header = [
            "bin_id", "phase_center", "bin_count",
            "acc_W_x", "acc_W_y", "acc_W_z",
            "acc_W_std_x", "acc_W_std_y", "acc_W_std_z",
            "omega_W_x", "omega_W_y", "omega_W_z",
            "omega_W_std_x", "omega_W_std_y", "omega_W_std_z",
            "alpha_W_x", "alpha_W_y", "alpha_W_z",
            "orientation_quat_w", "orientation_quat_x",
            "orientation_quat_y", "orientation_quat_z",
            "orientation_R_00", "orientation_R_01", "orientation_R_02",
            "orientation_R_10", "orientation_R_11", "orientation_R_12",
            "orientation_R_20", "orientation_R_21", "orientation_R_22",
            "orientation_dispersion_rad",
            "left_foot_z_mean", "left_foot_z_std",
            "right_foot_z_mean", "right_foot_z_std",
        ]
        if acc_raw is not None:
            header += ["acc_W_raw_x", "acc_W_raw_y", "acc_W_raw_z"]
        if omega_raw is not None:
            header += ["omega_W_raw_x", "omega_W_raw_y", "omega_W_raw_z"]
        writer.writerow(header)

        for i in range(len(phase_centers)):
            row = [
                i, phase_centers[i], bin_counts[i],
                *acc[i], *acc_std[i],
                *omega[i], *omega_std[i],
                *alpha[i],
                *orientation_quat[i],
                *orientation_rotmat[i].reshape(-1),
                orientation_dispersion[i],
                left_mean[i], left_std[i], right_mean[i], right_std[i],
            ]
            if acc_raw is not None:
                row += list(acc_raw[i])
            if omega_raw is not None:
                row += list(omega_raw[i])
            writer.writerow(row)


def print_summary(name, template, npz_path, csv_path):
    acc_std = np.asarray(template["torso_linear_acceleration_std"])
    omega_std = np.asarray(template["torso_angular_velocity_std"])
    orientation_quat = np.asarray(
        template["torso_orientation_quaternion_template"]
    )
    orientation_rotmat = np.asarray(
        template["torso_orientation_rotation_matrix_template"]
    )
    orientation_dispersion = np.asarray(
        template["torso_orientation_dispersion_rad"]
    )
    bin_counts = np.asarray(template["bin_counts"])

    print("=" * 72)
    print(f"{name} 构建完成")
    print("=" * 72)
    print(f"输出 npz                : {npz_path}")
    print(f"输出 csv                : {csv_path}")
    print(f"period                  : {float(template['period']):.6f} s")
    print(f"num_bins                : {int(template['num_bins'])}")
    print(f"dt_bin                  : {float(template['dt_bin']):.6f} s")
    print(f"min / max / mean count  : {bin_counts.min()} / {bin_counts.max()} / {bin_counts.mean():.2f}")
    print(f"acc_W std mean          : {acc_std.mean(axis=0)}")
    print(f"omega_W std mean        : {omega_std.mean(axis=0)}")
    print(
        "orientation dispersion : "
        f"{np.rad2deg(orientation_dispersion.mean()):.4f} deg mean"
    )
    print(
        "quaternion norm error   : "
        f"{np.max(np.abs(np.linalg.norm(orientation_quat, axis=1) - 1.0)):.3e}"
    )
    orthogonality_error = np.max(
        np.linalg.norm(
            np.transpose(orientation_rotmat, (0, 2, 1))
            @ orientation_rotmat
            - np.eye(3),
            axis=(1, 2),
        )
    )
    print(f"rotation orth error      : {orthogonality_error:.3e}")
    if "smoothing_window_size" in template:
        print(f"smoothing window        : {int(template['smoothing_window_size'])} bins")


def plot_template_comparison(raw_data, half_data, full_data, png_path):
    labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    rpy_labels = ["roll", "pitch", "yaw"]
    col_titles = ["Raw Template", "Half Smoothed Template", "Fully Smoothed Template"]
    row_titles = [
        "World-Frame Linear Acceleration",
        "World-Frame Angular Velocity",
        "World-Frame Angular Acceleration",
        "World-from-Torso Orientation (deg, visualization only)",
    ]

    fig, axes = plt.subplots(4, 3, figsize=(26, 15), sharex=True)
    fig.canvas.manager.set_window_title("World Disturbance Template Comparison")

    for c, (title, data) in enumerate(zip(col_titles, [raw_data, half_data, full_data])):
        phase = np.asarray(data["phase_centers"])
        acc = np.asarray(data["torso_linear_acceleration_template"])
        acc_std = np.asarray(data["torso_linear_acceleration_std"])
        omega = np.asarray(data["torso_angular_velocity_template"])
        omega_std = np.asarray(data["torso_angular_velocity_std"])
        alpha = np.asarray(data["torso_angular_acceleration_template"])
        orientation_rotmat = np.asarray(
            data["torso_orientation_rotation_matrix_template"]
        )
        orientation_rpy = np.rad2deg(
            np.column_stack(
                [
                    np.arctan2(
                        orientation_rotmat[:, 2, 1],
                        orientation_rotmat[:, 2, 2],
                    ),
                    np.arcsin(
                        np.clip(-orientation_rotmat[:, 2, 0], -1.0, 1.0)
                    ),
                    np.arctan2(
                        orientation_rotmat[:, 1, 0],
                        orientation_rotmat[:, 0, 0],
                    ),
                ]
            )
        )

        for i in range(3):
            axes[0, c].plot(phase, acc[:, i], color=colors[i], label=f"acc_W_{labels[i]}")
            axes[0, c].fill_between(phase, acc[:, i] - acc_std[:, i], acc[:, i] + acc_std[:, i], color=colors[i], alpha=0.18)
        for i in range(3):
            axes[1, c].plot(phase, omega[:, i], color=colors[i], label=f"omega_W_{labels[i]}")
            axes[1, c].fill_between(phase, omega[:, i] - omega_std[:, i], omega[:, i] + omega_std[:, i], color=colors[i], alpha=0.18)
        for i in range(3):
            axes[2, c].plot(phase, alpha[:, i], color=colors[i], label=f"alpha_W_{labels[i]}")
        for i in range(3):
            axes[3, c].plot(
                phase,
                orientation_rpy[:, i],
                color=colors[i],
                label=rpy_labels[i],
            )

        for r in range(4):
            axes[r, c].set_xlim(0.0, 1.0)
            axes[r, c].grid(True, axis="y", alpha=0.3)
            axes[r, c].legend(loc="upper right", fontsize=8)
            if r == 0:
                axes[r, c].set_title(title)
            if c == 0:
                axes[r, c].set_ylabel(row_titles[r])
            if r == 3:
                axes[r, c].set_xlabel("phase")

    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="基于 torso_disturbance_straight.npz 的 world 数据生成 raw / half / full 三种模板并保存对比图")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model_new/torso_disturbance_straight.npz",
        help="输入 npz，使用其中的 world 相关数据构建模板",
    )
    parser.add_argument(
        "--discard-time",
        type=float,
        default=None,
        help="丢弃启动段时长；默认优先读取输入 npz 中的 discard_time，没有则用 4.0",
    )
    parser.add_argument("--num-bins", type=int, default=100, help="phase 分 bin 数")
    parser.add_argument("--window-size", type=int, default=5, help="平滑窗口大小，奇数，单位 bin")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model_new/templates_world",
        help="输出目录",
    )
    args = parser.parse_args()

    try:
        source = load_npz(args.input)
    except Exception as e:
        print(f"读取输入失败: {e}")
        sys.exit(1)

    discard_time = float(source["discard_time"]) if args.discard_time is None and "discard_time" in source else (4.0 if args.discard_time is None else args.discard_time)
    os.makedirs(args.output_dir, exist_ok=True)

    raw_template = build_raw_template(source, discard_time=discard_time, num_bins=args.num_bins)
    raw_template["template_variant"] = np.array("raw")

    half_template = build_half_smoothed_template(raw_template, args.window_size)
    full_template = build_fully_smoothed_template(raw_template, args.window_size)

    raw_prefix = os.path.join(args.output_dir, "world_disturbance_template")
    half_prefix = os.path.join(args.output_dir, "world_disturbance_template_half_smoothed")
    full_prefix = os.path.join(args.output_dir, "world_disturbance_template_fully_smoothed")
    compare_png = os.path.join(args.output_dir, "World_Disturbance_Template_Comparison.png")

    save_template_npz(raw_template, raw_prefix + ".npz")
    save_template_csv(raw_template, raw_prefix + "_preview.csv")

    save_template_npz(half_template, half_prefix + ".npz")
    save_template_csv(half_template, half_prefix + "_preview.csv")

    save_template_npz(full_template, full_prefix + ".npz")
    save_template_csv(full_template, full_prefix + "_preview.csv")

    plot_template_comparison(raw_template, half_template, full_template, compare_png)

    print_summary("World Raw Template", raw_template, raw_prefix + ".npz", raw_prefix + "_preview.csv")
    print_summary("World Half Smoothed Template", half_template, half_prefix + ".npz", half_prefix + "_preview.csv")
    print_summary("World Fully Smoothed Template", full_template, full_prefix + ".npz", full_prefix + "_preview.csv")
    print(f"\n对比图已保存            : {compare_png}")


if __name__ == "__main__":
    main()
