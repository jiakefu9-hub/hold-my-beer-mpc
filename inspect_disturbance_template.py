# 怎么用
# - 只打印统计信息：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/inspect_disturbance_template.py --no-plot
#
# - 打印统计信息并画图：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/inspect_disturbance_template.py
#
# - 指定别的模板文件：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/inspect_disturbance_template.py \
#       --npz /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_template.npz

import argparse
import os
import sys

import numpy as np


REQUIRED_KEYS = [
    "period",
    "num_bins",
    "dt_bin",
    "phase_centers",
    "bin_counts",
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
        raise FileNotFoundError(f"文件不存在: {npz_path}")

    data = np.load(npz_path)
    keys = list(data.keys())
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        raise KeyError(f"缺少必要字段: {missing}")

    return {k: data[k] for k in keys}


def print_basic_stats(name, arr):
    arr = np.asarray(arr)
    print(f"\n[{name}]")
    print(f"  shape : {arr.shape}")

    if arr.ndim == 1:
        print(f"  min   : {arr.min():.6f}")
        print(f"  max   : {arr.max():.6f}")
        print(f"  mean  : {arr.mean():.6f}")
        print(f"  std   : {arr.std():.6f}")
    elif arr.ndim == 2:
        for i in range(arr.shape[1]):
            col = arr[:, i]
            print(
                f"  dim{i}: min={col.min():.6f}, max={col.max():.6f}, "
                f"mean={col.mean():.6f}, std={col.std():.6f}"
            )


def format_std_box(std_arr, prefix):
    std_arr = np.asarray(std_arr)
    mean_std = std_arr.mean(axis=0)
    max_std = std_arr.max(axis=0)
    lines = [
        f"{prefix} std",
        f"x: mean={mean_std[0]:.4f}, max={max_std[0]:.4f}",
        f"y: mean={mean_std[1]:.4f}, max={max_std[1]:.4f}",
        f"z: mean={mean_std[2]:.4f}, max={max_std[2]:.4f}",
    ]
    return "\n".join(lines)


def print_summary(data):
    period = float(data["period"])
    num_bins = int(data["num_bins"])
    dt_bin = float(data["dt_bin"])
    phase_centers = np.asarray(data["phase_centers"])
    bin_counts = np.asarray(data["bin_counts"])

    acc = np.asarray(data["torso_linear_acceleration_template"])
    acc_std = np.asarray(data["torso_linear_acceleration_std"])

    omega = np.asarray(data["torso_angular_velocity_template"])
    omega_std = np.asarray(data["torso_angular_velocity_std"])

    alpha = np.asarray(data["torso_angular_acceleration_template"])

    left_std = np.asarray(data["left_foot_z_std"])
    right_std = np.asarray(data["right_foot_z_std"])

    print("=" * 72)
    print("最终扰动模板摘要")
    print("=" * 72)
    print(f"period                  : {period:.6f} s")
    print(f"num_bins                : {num_bins}")
    print(f"dt_bin                  : {dt_bin:.6f} s")
    print(f"phase range             : [{phase_centers.min():.6f}, {phase_centers.max():.6f}]")

    print("\nbin 统计：")
    print(f"  min bin count         : {bin_counts.min()}")
    print(f"  max bin count         : {bin_counts.max()}")
    print(f"  mean bin count        : {bin_counts.mean():.2f}")

    print_basic_stats("torso_linear_acceleration_template", acc)
    print_basic_stats("torso_linear_acceleration_std", acc_std)
    print_basic_stats("torso_angular_velocity_template", omega)
    print_basic_stats("torso_angular_velocity_std", omega_std)
    print_basic_stats("torso_angular_acceleration_template", alpha)

    print("\n唯一性验证辅助量：")
    print(f"  left_foot_z_std mean  : {left_std.mean():.6f}")
    print(f"  right_foot_z_std mean : {right_std.mean():.6f}")


def plot_template(data):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n未安装 matplotlib，无法画图。")
        print("安装命令: pip install matplotlib")
        return

    phase = np.asarray(data["phase_centers"])

    acc = np.asarray(data["torso_linear_acceleration_template"])
    acc_std = np.asarray(data["torso_linear_acceleration_std"])

    omega = np.asarray(data["torso_angular_velocity_template"])
    omega_std = np.asarray(data["torso_angular_velocity_std"])

    alpha = np.asarray(data["torso_angular_acceleration_template"])

    labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.canvas.manager.set_window_title("Disturbance Template")

    # 1) acc_H template
    for i in range(3):
        axes[0].plot(phase, acc[:, i], color=colors[i], label=f"acc_H_{labels[i]}")
        axes[0].fill_between(
            phase,
            acc[:, i] - acc_std[:, i],
            acc[:, i] + acc_std[:, i],
            color=colors[i],
            alpha=0.18,
        )
    axes[0].set_ylabel("m/s^2")
    axes[0].set_title("Template: Heading-Frame Linear Acceleration")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper left")
    axes[0].text(
        1.01,
        0.98,
        format_std_box(acc_std, "acc_H"),
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # 2) omega_H template
    for i in range(3):
        axes[1].plot(phase, omega[:, i], color=colors[i], label=f"omega_H_{labels[i]}")
        axes[1].fill_between(
            phase,
            omega[:, i] - omega_std[:, i],
            omega[:, i] + omega_std[:, i],
            color=colors[i],
            alpha=0.18,
        )
    axes[1].set_ylabel("rad/s")
    axes[1].set_title("Template: Heading-Frame Angular Velocity")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper left")
    axes[1].text(
        1.01,
        0.98,
        format_std_box(omega_std, "omega_H"),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # 3) alpha_H template
    for i in range(3):
        axes[2].plot(phase, alpha[:, i], color=colors[i], label=f"alpha_H_{labels[i]}")
    axes[2].set_ylabel("rad/s^2")
    axes[2].set_xlabel("phase")
    axes[2].set_title("Template: Heading-Frame Angular Acceleration")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].legend(loc="upper left")
    axes[2].text(
        1.01,
        0.98,
        "alpha_H is computed\nfrom omega_H template\nusing periodic central\ndifference\n(no separate std)",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    for ax in axes:
        ax.set_xlim(0.0, 1.0)

    fig.tight_layout(rect=[0, 0, 0.86, 1])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="检查 disturbance_template.npz 并画模板图")
    parser.add_argument(
        "--npz",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_template.npz",
        help="模板 npz 文件路径",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="只打印统计信息，不画图",
    )
    args = parser.parse_args()

    try:
        data = load_npz(args.npz)
    except Exception as e:
        print(f"读取失败: {e}")
        sys.exit(1)

    print_summary(data)

    if not args.no_plot:
        plot_template(data)


if __name__ == "__main__":
    main()