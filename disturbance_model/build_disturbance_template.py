# 怎么用
# - 使用默认输入输出路径：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/build_disturbance_template.py
#
# - 指定输入文件：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/build_disturbance_template.py \
#       --heading /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_data_heading.npz \
#       --processed /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_data_processed.npz
#
# - 指定 bin 数：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/build_disturbance_template.py --num-bins 100
#
# - 指定输出前缀：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/build_disturbance_template.py \
#       --output-prefix /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_template

import argparse
import csv
import os
import sys

import numpy as np


HEADING_REQUIRED_KEYS = [
    "count",
    "phase",
    "torso_linear_acceleration_heading",
    "torso_angular_velocity_heading",
]

PROCESSED_REQUIRED_KEYS = [
    "count",
    "phase",
    "left_foot_z",
    "right_foot_z",
]


def load_npz(npz_path, required_keys):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"文件不存在: {npz_path}")

    data = np.load(npz_path)
    keys = list(data.keys())
    missing = [k for k in required_keys if k not in keys]
    if missing:
        raise KeyError(f"{npz_path} 缺少必要字段: {missing}")

    return {k: data[k] for k in keys}


def circular_central_difference(values, dt):
    """
    对周期模板做环形中心差分：
        d/dt f_i ≈ (f_{i+1} - f_{i-1}) / (2 dt)

    这里之所以使用“环形”差分，是因为 phase 模板本身是周期的，
    第 0 个 bin 和最后一个 bin 在物理上是首尾相接的。
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("circular_central_difference 需要输入 shape=(N, D) 的数组。")

    forward = np.roll(values, -1, axis=0)
    backward = np.roll(values, 1, axis=0)
    return (forward - backward) / (2.0 * dt)


def build_template(
    heading_data,
    processed_data,
    discard_time=4.0,
    period=0.8,
    num_bins=100,
):
    count_h = np.asarray(heading_data["count"], dtype=np.float64)
    phase_h = np.asarray(heading_data["phase"], dtype=np.float64)
    lin_acc_H = np.asarray(heading_data["torso_linear_acceleration_heading"], dtype=np.float64)
    ang_vel_H = np.asarray(heading_data["torso_angular_velocity_heading"], dtype=np.float64)

    count_p = np.asarray(processed_data["count"], dtype=np.float64)
    phase_p = np.asarray(processed_data["phase"], dtype=np.float64)
    left_foot_z = np.asarray(processed_data["left_foot_z"], dtype=np.float64)
    right_foot_z = np.asarray(processed_data["right_foot_z"], dtype=np.float64)

    if not (
        len(count_h) == len(phase_h) == len(lin_acc_H) == len(ang_vel_H)
        == len(count_p) == len(phase_p) == len(left_foot_z) == len(right_foot_z)
    ):
        raise ValueError("heading 数据和 processed 数据长度不一致。")

    # 只保留第 6 个 phase 开始的数据：
    # 20s 总时长，周期 0.8s => 共 25 个 phase
    # 丢弃前 4s => 丢弃前 5 个完整 phase
    # 保留从第 6 个 phase 到第 25 个 phase，共 20 个 phase
    mask = count_h >= discard_time

    count = count_h[mask]
    phase = phase_h[mask]
    lin_acc_H = lin_acc_H[mask]
    ang_vel_H = ang_vel_H[mask]
    left_foot_z = left_foot_z[mask]
    right_foot_z = right_foot_z[mask]

    n = len(count)
    if n == 0:
        raise ValueError("丢弃启动段后没有剩余数据。")

    # phase 分桶
    # phase ∈ [0,1)，理论上不会等于 1，但这里稳妥起见做 clip
    bin_ids = np.floor(phase * num_bins).astype(int)
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)

    phase_centers = (np.arange(num_bins, dtype=np.float64) + 0.5) / num_bins
    dt_bin = period / num_bins

    # 初始化模板
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    lin_acc_mean = np.zeros((num_bins, 3), dtype=np.float64)
    lin_acc_std = np.zeros((num_bins, 3), dtype=np.float64)

    ang_vel_mean = np.zeros((num_bins, 3), dtype=np.float64)
    ang_vel_std = np.zeros((num_bins, 3), dtype=np.float64)

    left_foot_z_mean = np.zeros(num_bins, dtype=np.float64)
    left_foot_z_std = np.zeros(num_bins, dtype=np.float64)

    right_foot_z_mean = np.zeros(num_bins, dtype=np.float64)
    right_foot_z_std = np.zeros(num_bins, dtype=np.float64)

    valid_bins = np.zeros(num_bins, dtype=bool)

    for b in range(num_bins):
        idx = (bin_ids == b)
        c = np.count_nonzero(idx)
        bin_counts[b] = c

        if c == 0:
            continue

        valid_bins[b] = True

        lin_acc_mean[b] = lin_acc_H[idx].mean(axis=0)
        lin_acc_std[b] = lin_acc_H[idx].std(axis=0)

        ang_vel_mean[b] = ang_vel_H[idx].mean(axis=0)
        ang_vel_std[b] = ang_vel_H[idx].std(axis=0)

        left_foot_z_mean[b] = left_foot_z[idx].mean()
        left_foot_z_std[b] = left_foot_z[idx].std()

        right_foot_z_mean[b] = right_foot_z[idx].mean()
        right_foot_z_std[b] = right_foot_z[idx].std()

    if not np.all(valid_bins):
        missing = np.where(~valid_bins)[0]
        raise ValueError(
            f"存在空 bin，无法构建完整周期模板。空 bin 索引: {missing.tolist()}"
        )

    # 角加速度模板：
    # 不直接对原始 noisy 角加速度做平均，
    # 而是先得到 omega_H 的均值模板，再对模板做周期中心差分
    ang_acc_template = circular_central_difference(ang_vel_mean, dt_bin)

    template = {
        "period": np.array(period, dtype=np.float64),
        "discard_time": np.array(discard_time, dtype=np.float64),
        "num_bins": np.array(num_bins, dtype=np.int64),
        "dt_bin": np.array(dt_bin, dtype=np.float64),
        "phase_centers": phase_centers,
        "bin_counts": bin_counts,
        "valid_bins": valid_bins,

        # 最终给 MPC 用的主模板
        "torso_linear_acceleration_template": lin_acc_mean,
        "torso_angular_velocity_template": ang_vel_mean,
        "torso_angular_acceleration_template": ang_acc_template,

        # 保留标准差，便于人工检查模板稳定性
        "torso_linear_acceleration_std": lin_acc_std,
        "torso_angular_velocity_std": ang_vel_std,

        # 唯一性验证辅助信息
        "left_foot_z_mean": left_foot_z_mean,
        "left_foot_z_std": left_foot_z_std,
        "right_foot_z_mean": right_foot_z_mean,
        "right_foot_z_std": right_foot_z_std,
    }
    return template


def save_template_npz(template, npz_path):
    np.savez(npz_path, **template)


def save_template_csv(template, csv_path):
    phase_centers = template["phase_centers"]
    bin_counts = template["bin_counts"]

    lin_acc = template["torso_linear_acceleration_template"]
    lin_acc_std = template["torso_linear_acceleration_std"]

    ang_vel = template["torso_angular_velocity_template"]
    ang_vel_std = template["torso_angular_velocity_std"]

    ang_acc = template["torso_angular_acceleration_template"]

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

                "acc_H_x", "acc_H_y", "acc_H_z",
                "acc_H_std_x", "acc_H_std_y", "acc_H_std_z",

                "omega_H_x", "omega_H_y", "omega_H_z",
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

                    *lin_acc[i],
                    *lin_acc_std[i],

                    *ang_vel[i],
                    *ang_vel_std[i],

                    *ang_acc[i],

                    left_mean[i],
                    left_std[i],
                    right_mean[i],
                    right_std[i],
                ]
            )


def print_summary(template, npz_path, csv_path):
    num_bins = int(template["num_bins"])
    period = float(template["period"])
    dt_bin = float(template["dt_bin"])
    bin_counts = template["bin_counts"]

    lin_acc_std = template["torso_linear_acceleration_std"]
    ang_vel_std = template["torso_angular_velocity_std"]

    left_std = template["left_foot_z_std"]
    right_std = template["right_foot_z_std"]

    print("=" * 72)
    print("第 8 步：扰动模板构建完成")
    print("=" * 72)
    print(f"period                  : {period:.6f} s")
    print(f"num_bins                : {num_bins}")
    print(f"dt_bin                  : {dt_bin:.6f} s")
    print(f"输出 npz                : {npz_path}")
    print(f"输出 csv                : {csv_path}")

    print("\n每个 bin 的样本数统计：")
    print(f"  min bin count         : {bin_counts.min()}")
    print(f"  max bin count         : {bin_counts.max()}")
    print(f"  mean bin count        : {bin_counts.mean():.2f}")

    print("\n模板稳定性（标准差均值，仅供参考）：")
    print(f"  acc_H std mean        : {lin_acc_std.mean(axis=0)}")
    print(f"  omega_H std mean      : {ang_vel_std.mean(axis=0)}")

    print("\n唯一性验证辅助量（脚高度 std 的均值，仅供参考）：")
    print(f"  left_foot_z std mean  : {left_std.mean():.6f}")
    print(f"  right_foot_z std mean : {right_std.mean():.6f}")

    print("\n说明：")
    print("- 最终模板采用查表形式保存，而不是函数文件。")
    print("- 运行时加载 npz，根据当前 phase 对模板做线性插值。")
    print("- 这样最适合后续作为前馈项加入 MPC。")


def main():
    parser = argparse.ArgumentParser(description="第 8 步：基于 H 系数据构建 phase 扰动模板")
    parser.add_argument(
        "--heading",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_data_heading.npz",
        help="H 系扰动数据 npz",
    )
    parser.add_argument(
        "--processed",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_data_processed.npz",
        help="processed 数据 npz（用于左右脚高度唯一性验证）",
    )
    parser.add_argument(
        "--discard-time",
        type=float,
        default=4.0,
        help="丢弃启动段时长，默认 4.0 秒",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=0.8,
        help="步态周期，默认 0.8 秒",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=100,
        help="phase 分桶数，默认 100",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_template",
        help="输出文件前缀",
    )
    args = parser.parse_args()

    try:
        heading_data = load_npz(args.heading, HEADING_REQUIRED_KEYS)
        processed_data = load_npz(args.processed, PROCESSED_REQUIRED_KEYS)
        template = build_template(
            heading_data=heading_data,
            processed_data=processed_data,
            discard_time=args.discard_time,
            period=args.period,
            num_bins=args.num_bins,
        )
    except Exception as e:
        print(f"模板构建失败: {e}")
        sys.exit(1)

    npz_path = args.output_prefix + ".npz"
    csv_path = args.output_prefix + "_preview.csv"

    save_template_npz(template, npz_path)
    save_template_csv(template, csv_path)
    print_summary(template, npz_path, csv_path)


if __name__ == "__main__":
    main()