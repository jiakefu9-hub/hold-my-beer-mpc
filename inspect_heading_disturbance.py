# 怎么用
# - 只打印统计信息：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/inspect_heading_disturbance.py --no-plot
#
# - 打印统计信息并画图（单窗口）：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/inspect_heading_disturbance.py
#
# - 指定别的 heading npz 文件：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/inspect_heading_disturbance.py \
#       --npz /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_data_heading.npz

import argparse
import os
import sys

import numpy as np


REQUIRED_KEYS = [
    "count",
    "phase",
    "torso_quaternion",
    "yaw",
    "yaw_unwrapped",
    "yaw_ref",
    "R_heading_from_world",
    "R_heading_from_imu",
    "torso_linear_acceleration_heading",
    "torso_angular_velocity_heading",
]


def fmt_arr(x, precision=4):
    x = np.asarray(x)
    return np.array2string(x, precision=precision, suppress_small=False)


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
    elif arr.ndim == 3:
        flat = arr.reshape(arr.shape[0], -1)
        for i in range(flat.shape[1]):
            col = flat[:, i]
            print(
                f"  elem{i}: min={col.min():.6f}, max={col.max():.6f}, "
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

    return {k: data[k] for k in keys}


def rotmat_to_yaw(R):
    R = np.asarray(R)
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])


def print_summary(data, head=5):
    count = np.asarray(data["count"])
    phase = np.asarray(data["phase"])
    quat = np.asarray(data["torso_quaternion"])
    yaw = np.asarray(data["yaw"])
    yaw_unwrapped = np.asarray(data["yaw_unwrapped"])
    yaw_ref = np.asarray(data["yaw_ref"])
    H_R_W = np.asarray(data["R_heading_from_world"])
    H_R_IMU = np.asarray(data["R_heading_from_imu"])
    lin_acc_H = np.asarray(data["torso_linear_acceleration_heading"])
    ang_vel_H = np.asarray(data["torso_angular_velocity_heading"])

    n = len(count)
    dt_est = np.median(np.diff(count)) if n > 1 else np.nan
    duration = count[-1] - count[0] if n > 1 else 0.0
    quat_norm = np.linalg.norm(quat, axis=1)
    yaw_H_from_W = np.unwrap(rotmat_to_yaw(H_R_W))
    yaw_H_from_IMU = np.unwrap(rotmat_to_yaw(H_R_IMU))

    print("=" * 72)
    print("Heading 坐标系扰动数据摘要")
    print("=" * 72)
    print(f"样本数                  : {n}")
    print(f"字段                    : {list(data.keys())}")
    print(f"起始时间                : {count[0]:.6f} s")
    print(f"结束时间                : {count[-1]:.6f} s")
    print(f"总时长                  : {duration:.6f} s")
    print(f"估计采样间隔 dt          : {dt_est:.6f} s")
    if not np.isnan(dt_est) and dt_est > 0:
        print(f"估计采样频率             : {1.0 / dt_est:.3f} Hz")
    print(f"phase 范围               : [{phase.min():.6f}, {phase.max():.6f}]")

    print_basic_stats("yaw", yaw)
    print_basic_stats("yaw_unwrapped", yaw_unwrapped)
    print_basic_stats("yaw_ref", yaw_ref)
    print_basic_stats("yaw_H_from_W", yaw_H_from_W)
    print_basic_stats("yaw_H_from_IMU", yaw_H_from_IMU)
    print_basic_stats("torso_linear_acceleration_heading", lin_acc_H)
    print_basic_stats("torso_angular_velocity_heading", ang_vel_H)
    print_basic_stats("quaternion_norm", quat_norm)

    print("\n" + "=" * 72)
    print(f"前 {min(head, n)} 行样本预览")
    print("=" * 72)
    for i in range(min(head, n)):
        print(
            f"[{i:04d}] "
            f"t={count[i]:.4f}, phase={phase[i]:.4f}, "
            f"yaw={yaw[i]:.4f}, yaw_unwrapped={yaw_unwrapped[i]:.4f}, yaw_ref={yaw_ref[i]:.4f}, "
            f"lin_acc_H={fmt_arr(lin_acc_H[i])}, "
            f"ang_vel_H={fmt_arr(ang_vel_H[i])}"
        )

    print("\n" + "=" * 72)
    print("第一帧 ^H R_W")
    print("=" * 72)
    print(H_R_W[0])

    print("\n" + "=" * 72)
    print("第一帧 ^H R_IMU")
    print("=" * 72)
    print(H_R_IMU[0])


def plot_data(data):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n未安装 matplotlib，无法画图。")
        print("安装命令: pip install matplotlib")
        return

    count = np.asarray(data["count"])
    phase = np.asarray(data["phase"])
    yaw = np.asarray(data["yaw"])
    yaw_unwrapped = np.asarray(data["yaw_unwrapped"])
    yaw_ref = np.asarray(data["yaw_ref"])
    H_R_W = np.asarray(data["R_heading_from_world"])
    H_R_IMU = np.asarray(data["R_heading_from_imu"])
    lin_acc_H = np.asarray(data["torso_linear_acceleration_heading"])
    ang_vel_H = np.asarray(data["torso_angular_velocity_heading"])

    yaw_H_from_W = np.unwrap(rotmat_to_yaw(H_R_W))
    yaw_H_from_IMU = np.unwrap(rotmat_to_yaw(H_R_IMU))

    wrap_indices = np.where(np.diff(phase) < -0.5)[0] + 1
    phase_boundary_times = count[wrap_indices]
    labels = ["x", "y", "z"]

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    fig.canvas.manager.set_window_title("Heading Disturbance")

    axes[0].plot(count, phase, color="black", label="phase")
    axes[0].set_ylabel("phase")
    axes[0].set_title("Phase vs Time")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].plot(count, yaw, label="yaw", color="tab:blue", alpha=0.8)
    axes[1].plot(count, yaw_unwrapped, label="yaw_unwrapped", color="tab:orange", alpha=0.8)
    axes[1].set_ylabel("rad")
    axes[1].set_title("Yaw and Yaw Unwrapped")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    axes[2].plot(count, yaw_unwrapped, label="yaw_unwrapped", color="tab:gray", alpha=0.6)
    axes[2].plot(count, yaw_ref, label="yaw_ref (smoothed)", color="tab:red", linewidth=2.0)
    axes[2].set_ylabel("rad")
    axes[2].set_title("Yaw Reference")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].legend()

    axes[3].plot(count, yaw_H_from_W, label="yaw(^H R_W)", color="tab:green")
    axes[3].plot(count, yaw_H_from_IMU, label="yaw(^H R_IMU)", color="tab:purple")
    axes[3].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    axes[3].set_ylabel("rad")
    axes[3].set_title("Yaw in Heading Frame Diagnostics")
    axes[3].grid(True, axis="y", alpha=0.3)
    axes[3].legend()

    for i in range(3):
        axes[4].plot(count, lin_acc_H[:, i], label=f"acc_H_{labels[i]}")
    axes[4].set_ylabel("m/s^2")
    axes[4].set_title("Heading-Frame Linear Acceleration")
    axes[4].grid(True, axis="y", alpha=0.3)
    axes[4].legend()

    for i in range(3):
        axes[5].plot(count, ang_vel_H[:, i], label=f"omega_H_{labels[i]}")
    axes[5].set_ylabel("rad/s")
    axes[5].set_xlabel("time [s]")
    axes[5].set_title("Heading-Frame Angular Velocity")
    axes[5].grid(True, axis="y", alpha=0.3)
    axes[5].legend()

    for ax in axes:
        for t in phase_boundary_times:
            ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="检查 disturbance_data_heading.npz 的统计信息并画图（单窗口）")
    parser.add_argument(
        "--npz",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_data_heading.npz",
        help="heading npz 文件路径",
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