# 怎么用
# - 只打印统计信息：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/inspect_disturbance.py --no-plot
# - 打印统计信息并画图：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/inspect_disturbance.py
# - 指定别的 npz 文件：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/inspect_disturbance.py --npz /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_data.npz

import argparse
import os
import sys

import numpy as np


REQUIRED_KEYS = [
    "count",
    "phase",
    "torso_linear_acceleration",
    "torso_angular_velocity",
    "torso_angular_acceleration",
    "torso_quaternion",
    "left_foot_z",
    "right_foot_z",
]


def fmt_arr(x, precision=4):
    x = np.asarray(x)
    return np.array2string(x, precision=precision, suppress_small=False)


def print_basic_stats(name, arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        print(f"\n[{name}]")
        print(f"  shape : {arr.shape}")
        print(f"  min   : {arr.min():.6f}")
        print(f"  max   : {arr.max():.6f}")
        print(f"  mean  : {arr.mean():.6f}")
        print(f"  std   : {arr.std():.6f}")
    elif arr.ndim == 2:
        print(f"\n[{name}]")
        print(f"  shape : {arr.shape}")
        for i in range(arr.shape[1]):
            col = arr[:, i]
            print(
                f"  dim{i}: min={col.min():.6f}, max={col.max():.6f}, "
                f"mean={col.mean():.6f}, std={col.std():.6f}"
            )


def load_npz(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"文件不存在: {npz_path}")

    data = np.load(npz_path)
    keys = list(data.keys())

    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        raise KeyError(f"缺少必要字段: {missing}")

    result = {k: data[k] for k in keys}
    return result


def print_summary(data, head=5):
    count = np.asarray(data["count"])
    phase = np.asarray(data["phase"])
    lin_acc = np.asarray(data["torso_linear_acceleration"])
    ang_vel = np.asarray(data["torso_angular_velocity"])
    ang_acc = np.asarray(data["torso_angular_acceleration"])
    quat = np.asarray(data["torso_quaternion"])
    lz = np.asarray(data["left_foot_z"])
    rz = np.asarray(data["right_foot_z"])

    n = len(count)
    dt_est = np.median(np.diff(count)) if n > 1 else np.nan
    duration = count[-1] - count[0] if n > 1 else 0.0

    print("=" * 60)
    print("扰动数据摘要")
    print("=" * 60)
    print(f"样本数           : {n}")
    print(f"字段             : {list(data.keys())}")
    print(f"起始时间         : {count[0]:.6f} s")
    print(f"结束时间         : {count[-1]:.6f} s")
    print(f"总时长           : {duration:.6f} s")
    print(f"估计采样间隔 dt   : {dt_est:.6f} s")
    if not np.isnan(dt_est) and dt_est > 0:
        print(f"估计采样频率      : {1.0 / dt_est:.3f} Hz")
    print(f"phase 范围        : [{phase.min():.6f}, {phase.max():.6f}]")

    print_basic_stats("torso_linear_acceleration", lin_acc)
    print_basic_stats("torso_angular_velocity", ang_vel)
    print_basic_stats("torso_angular_acceleration", ang_acc)
    print_basic_stats("left_foot_z", lz)
    print_basic_stats("right_foot_z", rz)

    quat_norm = np.linalg.norm(quat, axis=1)
    print_basic_stats("quaternion_norm", quat_norm)

    print("\n" + "=" * 60)
    print(f"前 {min(head, n)} 行样本预览")
    print("=" * 60)
    for i in range(min(head, n)):
        print(
            f"[{i:04d}] "
            f"t={count[i]:.4f}, phase={phase[i]:.4f}, "
            f"lin_acc={fmt_arr(lin_acc[i])}, "
            f"ang_vel={fmt_arr(ang_vel[i])}, "
            f"ang_acc={fmt_arr(ang_acc[i])}, "
            f"quat={fmt_arr(quat[i])}, "
            f"Lz={lz[i]:.4f}, Rz={rz[i]:.4f}"
        )


def plot_data(data):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n未安装 matplotlib，无法画图。")
        print("安装命令: pip install matplotlib")
        return

    count = np.asarray(data["count"])
    phase = np.asarray(data["phase"])
    lin_acc = np.asarray(data["torso_linear_acceleration"])
    ang_vel = np.asarray(data["torso_angular_velocity"])
    ang_acc = np.asarray(data["torso_angular_acceleration"])
    quat = np.asarray(data["torso_quaternion"])
    lz = np.asarray(data["left_foot_z"])
    rz = np.asarray(data["right_foot_z"])

    # 用 phase 回卷位置来标记每个完整步态周期的边界
    wrap_indices = np.where(np.diff(phase) < -0.5)[0] + 1
    phase_boundary_times = count[wrap_indices]

    fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

    axes[0].plot(count, phase, label="phase", color="black")
    axes[0].set_ylabel("phase")
    axes[0].set_title("Phase vs Time")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    labels = ["x", "y", "z"]

    for i in range(3):
        axes[1].plot(count, lin_acc[:, i], label=f"acc_{labels[i]}")
    axes[1].set_ylabel("m/s^2")
    axes[1].set_title("Torso Linear Acceleration")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    for i in range(3):
        axes[2].plot(count, ang_vel[:, i], label=f"omega_{labels[i]}")
    axes[2].set_ylabel("rad/s")
    axes[2].set_title("Torso Angular Velocity")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].legend()

    for i in range(3):
        axes[3].plot(count, ang_acc[:, i], label=f"alpha_{labels[i]}")
    axes[3].set_ylabel("rad/s^2")
    axes[3].set_title("Torso Angular Acceleration")
    axes[3].grid(True, axis="y", alpha=0.3)
    axes[3].legend()

    axes[4].plot(count, lz, label="left_foot_z")
    axes[4].plot(count, rz, label="right_foot_z")
    axes[4].set_ylabel("m")
    axes[4].set_title("Foot Height")
    axes[4].grid(True, axis="y", alpha=0.3)
    axes[4].legend()

    quat_labels = ["w", "x", "y", "z"]
    for i in range(4):
        axes[5].plot(count, quat[:, i], label=f"quat_{quat_labels[i]}")
    axes[5].set_ylabel("quat")
    axes[5].set_xlabel("time [s]")
    axes[5].set_title("Torso Quaternion")
    axes[5].grid(True, axis="y", alpha=0.3)
    axes[5].legend()

    for ax in axes:
        for t in phase_boundary_times:
            ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="检查 disturbance_data.npz 的统计信息并画图")
    parser.add_argument(
        "--npz",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_data.npz",
        help="npz 文件路径",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="打印前几行样本",
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

    print_summary(data, head=args.head)

    if not args.no_plot:
        plot_data(data)


if __name__ == "__main__":
    main()