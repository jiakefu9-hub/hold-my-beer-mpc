# 怎么用
# - 使用默认输入输出路径：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/build_fully_smoothed_disturbance_template.py
#
# - 指定输入模板：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/build_fully_smoothed_disturbance_template.py \
#       --input /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_template.npz
#
# - 指定平滑窗口（奇数，单位：bin）：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/build_fully_smoothed_disturbance_template.py \
#       --window-size 5
#
# - 指定输出前缀：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/build_fully_smoothed_disturbance_template.py \
#       --output-prefix /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_template_fully_smoothed

import argparse
import csv
import os
import sys

import numpy as np


REQUIRED_KEYS = [
    "period",
    "discard_time",
    "num_bins",
    "dt_bin",
    "phase_centers",
    "bin_counts",
    "valid_bins",
    "torso_linear_acceleration_template",
    "torso_linear_acceleration_std",
    "torso_angular_velocity_template",
    "torso_angular_velocity_std",
    "torso_angular_acceleration_template",
    "left_foot_z_mean",
    "left_foot_z_std",
    "right_foot_z_mean",
    "right_foot_z_std",
]


def load_npz(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"输入文件不存在: {npz_path}")

    data = np.load(npz_path)
    keys = list(data.keys())
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        raise KeyError(f"模板文件缺少必要字段: {missing}")

    return {k: data[k] for k in keys}


def circular_moving_average(values, window_size):
    """
    对周期模板做环形滑动平均。

    使用环形（circular）滑动平均的原因：
    - phase 模板是周期性的
    - 最后一个 bin 和第一个 bin 在物理上是首尾相接的

    在这个“全轻度平滑模板”版本里：
    - acc_H 做轻度平滑
    - omega_H 做轻度平滑
    - alpha_H 不直接平滑，而是由平滑后的 omega_H 再求导得到
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError("circular_moving_average 需要 shape=(N, D) 的数组。")
    if window_size < 1:
        raise ValueError("window_size 必须 >= 1")
    if window_size % 2 == 0:
        raise ValueError("window_size 必须为奇数")

    if window_size == 1:
        return values.copy()

    pad = window_size // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]], axis=0)
    kernel = np.ones(window_size, dtype=np.float64) / window_size

    smoothed = np.zeros_like(values)
    for d in range(values.shape[1]):
        smoothed[:, d] = np.convolve(padded[:, d], kernel, mode="valid")

    return smoothed


def circular_central_difference(values, dt):
    """
    对周期模板做环形中心差分：
        d/dt f_i ≈ (f_{i+1} - f_{i-1}) / (2 dt)

    这里用于：
    - alpha_H = d/dt (omega_H_smoothed)
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("circular_central_difference 需要 shape=(N, D) 的数组。")

    forward = np.roll(values, -1, axis=0)
    backward = np.roll(values, 1, axis=0)
    return (forward - backward) / (2.0 * dt)


def build_fully_smoothed_template(template, window_size):
    period = float(template["period"])
    discard_time = float(template["discard_time"])
    num_bins = int(template["num_bins"])
    dt_bin = float(template["dt_bin"])

    phase_centers = np.asarray(template["phase_centers"], dtype=np.float64)
    bin_counts = np.asarray(template["bin_counts"], dtype=np.int64)
    valid_bins = np.asarray(template["valid_bins"], dtype=bool)

    # 原始模板
    acc_raw = np.asarray(template["torso_linear_acceleration_template"], dtype=np.float64)
    acc_std = np.asarray(template["torso_linear_acceleration_std"], dtype=np.float64)

    omega_raw = np.asarray(template["torso_angular_velocity_template"], dtype=np.float64)
    omega_std = np.asarray(template["torso_angular_velocity_std"], dtype=np.float64)

    # 全轻度平滑版本：
    # 1) acc_H 做轻度平滑
    acc_smoothed = circular_moving_average(acc_raw, window_size)

    # 2) omega_H 做轻度平滑
    omega_smoothed = circular_moving_average(omega_raw, window_size)

    # 3) alpha_H 从平滑后的 omega_H 求导
    alpha_from_smoothed_omega = circular_central_difference(omega_smoothed, dt_bin)

    left_foot_z_mean = np.asarray(template["left_foot_z_mean"], dtype=np.float64)
    left_foot_z_std = np.asarray(template["left_foot_z_std"], dtype=np.float64)
    right_foot_z_mean = np.asarray(template["right_foot_z_mean"], dtype=np.float64)
    right_foot_z_std = np.asarray(template["right_foot_z_std"], dtype=np.float64)

    smoothed_template = {
        "period": np.array(period, dtype=np.float64),
        "discard_time": np.array(discard_time, dtype=np.float64),
        "num_bins": np.array(num_bins, dtype=np.int64),
        "dt_bin": np.array(dt_bin, dtype=np.float64),
        "phase_centers": phase_centers,
        "bin_counts": bin_counts,
        "valid_bins": valid_bins,

        # 版本 C 的核心模板
        "torso_linear_acceleration_template_raw": acc_raw,
        "torso_linear_acceleration_template": acc_smoothed,
        "torso_linear_acceleration_std": acc_std,

        "torso_angular_velocity_template_raw": omega_raw,
        "torso_angular_velocity_template": omega_smoothed,
        "torso_angular_velocity_std": omega_std,

        "torso_angular_acceleration_template": alpha_from_smoothed_omega,

        # 保留唯一性验证信息
        "left_foot_z_mean": left_foot_z_mean,
        "left_foot_z_std": left_foot_z_std,
        "right_foot_z_mean": right_foot_z_mean,
        "right_foot_z_std": right_foot_z_std,

        # 记录平滑信息
        "smoothing_window_size": np.array(window_size, dtype=np.int64),
    }

    return smoothed_template


def save_npz(template, npz_path):
    np.savez(npz_path, **template)


def save_csv(template, csv_path):
    phase_centers = template["phase_centers"]
    bin_counts = template["bin_counts"]

    acc_raw = template["torso_linear_acceleration_template_raw"]
    acc = template["torso_linear_acceleration_template"]
    acc_std = template["torso_linear_acceleration_std"]

    omega_raw = template["torso_angular_velocity_template_raw"]
    omega = template["torso_angular_velocity_template"]
    omega_std = template["torso_angular_velocity_std"]

    alpha = template["torso_angular_acceleration_template"]

    left_mean = template["left_foot_z_mean"]
    left_std = template["left_foot_z_std"]
    right_mean = template["right_foot_z_mean"]
    right_std = template["right_foot_z_std"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bin_id",
                "phase_center",
                "bin_count",

                "acc_H_raw_x", "acc_H_raw_y", "acc_H_raw_z",
                "acc_H_smooth_x", "acc_H_smooth_y", "acc_H_smooth_z",
                "acc_H_std_x", "acc_H_std_y", "acc_H_std_z",

                "omega_H_raw_x", "omega_H_raw_y", "omega_H_raw_z",
                "omega_H_smooth_x", "omega_H_smooth_y", "omega_H_smooth_z",
                "omega_H_std_x", "omega_H_std_y", "omega_H_std_z",

                "alpha_H_x", "alpha_H_y", "alpha_H_z",

                "left_foot_z_mean", "left_foot_z_std",
                "right_foot_z_mean", "right_foot_z_std",
            ]
        )

        for i in range(len(phase_centers)):
            writer.writerow(
                [
                    i,
                    phase_centers[i],
                    bin_counts[i],

                    *acc_raw[i],
                    *acc[i],
                    *acc_std[i],

                    *omega_raw[i],
                    *omega[i],
                    *omega_std[i],

                    *alpha[i],

                    left_mean[i],
                    left_std[i],
                    right_mean[i],
                    right_std[i],
                ]
            )


def print_summary(template, npz_path, csv_path):
    acc_std = np.asarray(template["torso_linear_acceleration_std"])
    omega_std = np.asarray(template["torso_angular_velocity_std"])
    window_size = int(template["smoothing_window_size"])

    print("=" * 72)
    print("全轻度平滑扰动模板（版本 C）构建完成")
    print("=" * 72)
    print(f"输出 npz                : {npz_path}")
    print(f"输出 csv                : {csv_path}")
    print(f"acc_H 平滑方法          : 环形滑动平均 (circular moving average)")
    print(f"omega_H 平滑方法        : 环形滑动平均 (circular moving average)")
    print(f"平滑窗口                : {window_size} bins")

    print("\n版本 C 规则：")
    print("- acc_H   : 使用轻度平滑模板")
    print("- omega_H : 使用轻度平滑模板")
    print("- alpha_H : 由平滑后的 omega_H 做周期中心差分得到")

    print("\n模板稳定性（原始 std，仅供参考）：")
    print(f"  acc_H std mean        : {acc_std.mean(axis=0)}")
    print(f"  omega_H std mean      : {omega_std.mean(axis=0)}")


def main():
    parser = argparse.ArgumentParser(description="基于 disturbance_template.npz 生成版本 C 全轻度平滑模板")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_template.npz",
        help="输入模板 npz 文件路径",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="acc_H 与 omega_H 的平滑窗口大小（奇数，单位：bin）",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_template_fully_smoothed",
        help="输出文件前缀",
    )
    args = parser.parse_args()

    if args.window_size % 2 == 0:
        print("错误：--window-size 必须为奇数。")
        sys.exit(1)

    try:
        template = load_npz(args.input)
        smoothed_template = build_fully_smoothed_template(template, args.window_size)
    except Exception as e:
        print(f"生成全轻度平滑模板失败: {e}")
        sys.exit(1)

    npz_path = args.output_prefix + ".npz"
    csv_path = args.output_prefix + "_preview.csv"

    save_npz(smoothed_template, npz_path)
    save_csv(smoothed_template, csv_path)
    print_summary(smoothed_template, npz_path, csv_path)


if __name__ == "__main__":
    main()