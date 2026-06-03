# 怎么用
# - 使用默认输入输出路径：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/process_heading_disturbance.py
#
# - 指定输入 processed npz：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/process_heading_disturbance.py \
#       --input /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_data_processed.npz
#
# - 指定平滑窗口（奇数，单位：样本点）：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/process_heading_disturbance.py \
#       --window-size 1001
#
# - 指定输出前缀：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/process_heading_disturbance.py \
#       --output-prefix /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_data_heading

import argparse
import csv
import os
import sys

import numpy as np


REQUIRED_KEYS = [
    "count",
    "phase",
    "torso_quaternion",
    "torso_linear_acceleration_world",
    "torso_angular_velocity_world",
]

OPTIONAL_KEYS = [
    "R_world_from_imu",
]


def load_npz(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"输入文件不存在: {npz_path}")

    data = np.load(npz_path)
    keys = list(data.keys())

    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        raise KeyError(f"缺少必要字段: {missing}")

    return {k: data[k] for k in keys}


def normalize_quaternion_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("遇到零范数四元数，无法转换。")
    return q / n


def quat_wxyz_to_rotmat_world_from_imu(q_wxyz: np.ndarray) -> np.ndarray:
    """
    输入四元数 q = [w, x, y, z]，输出 ^W R_IMU

    这个矩阵的物理含义是：
        v_W = ^W R_IMU @ v_IMU

    即：
    - 输入是 IMU 局部坐标系下表示的向量
    - 输出是世界坐标系下表示的同一个向量
    """
    w, x, y, z = normalize_quaternion_wxyz(q_wxyz)

    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def quat_to_yaw_wxyz(q: np.ndarray) -> np.ndarray:
    """
    从四元数 [w, x, y, z] 提取 yaw。
    """
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q.T
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return yaw


def centered_moving_average(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    对 yaw_unwrapped(t) 做中心滑动平均，得到慢变化的 yaw_ref(t)。

    这里采用的方法是：
    - 先对原始 yaw 做 unwrap，去掉 -pi / pi 跳变
    - 再做“中心滑动平均”（Moving Average）
    - 目标是保留“缓慢变化的中立朝向”，滤掉步态周期内的小幅左右摆动

    为什么用这个方法：
    - 它不需要额外依赖 scipy
    - 对当前项目第一版足够稳定、直观、可解释
    - 比直接用瞬时 yaw 更适合构造 heading-aligned frame

    注意：
    - window_size 必须是奇数
    - 窗口越大，yaw_ref 越平滑，但也越容易抹掉慢变化细节
    - 当前默认 1001 点；若 dt=0.002s，则约对应 2.002s 的平滑窗口
    """
    if window_size < 1:
        raise ValueError("window_size 必须 >= 1")
    if window_size % 2 == 0:
        raise ValueError("window_size 必须为奇数")

    x = np.asarray(x, dtype=np.float64)
    pad = window_size // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    y = np.convolve(x_pad, kernel, mode="valid")
    return y


def build_heading_rotation(yaw_ref: np.ndarray) -> np.ndarray:
    """
    根据慢变化的 yaw_ref(t) 构造 ^H R_W(t)。

    这里 H 坐标系定义为：
    - z 轴始终与世界坐标系竖直方向一致
    - x 轴始终指向机器人“慢变化的平均前进方向”
    - y 轴由右手系确定

    如果 yaw_ref 表示 H 相对 W 的平均朝向，则：
        ^H R_W
    的作用是把世界系向量转到 H 系，即：
        v_H = ^H R_W @ v_W

    对于绕 z 轴的旋转，有：
        ^H R_W = Rot_z(-yaw_ref)
    """
    yaw_ref = np.asarray(yaw_ref, dtype=np.float64)
    n = len(yaw_ref)
    H_R_W = np.zeros((n, 3, 3), dtype=np.float64)

    c = np.cos(yaw_ref)
    s = np.sin(yaw_ref)

    H_R_W[:, 0, 0] = c
    H_R_W[:, 0, 1] = s
    H_R_W[:, 0, 2] = 0.0

    H_R_W[:, 1, 0] = -s
    H_R_W[:, 1, 1] = c
    H_R_W[:, 1, 2] = 0.0

    H_R_W[:, 2, 0] = 0.0
    H_R_W[:, 2, 1] = 0.0
    H_R_W[:, 2, 2] = 1.0

    return H_R_W


def process_heading(data: dict, window_size: int) -> dict:
    count = np.asarray(data["count"], dtype=np.float64)
    phase = np.asarray(data["phase"], dtype=np.float64)
    quat = np.asarray(data["torso_quaternion"], dtype=np.float64)
    lin_acc_world = np.asarray(data["torso_linear_acceleration_world"], dtype=np.float64)
    ang_vel_world = np.asarray(data["torso_angular_velocity_world"], dtype=np.float64)

    n = len(count)
    if not (len(phase) == len(quat) == len(lin_acc_world) == len(ang_vel_world) == n):
        raise ValueError("输入数组长度不一致。")

    # 1) 由四元数提取 yaw
    yaw = quat_to_yaw_wxyz(quat)

    # 2) unwrap，得到连续的 yaw 曲线
    yaw_unwrapped = np.unwrap(yaw)

    # 3) 平滑，得到慢变化的中立朝向 yaw_ref(t)
    yaw_ref = centered_moving_average(yaw_unwrapped, window_size)

    # 4) 构造 ^H R_W(t)
    H_R_W = build_heading_rotation(yaw_ref)

    # 5) 准备 ^W R_IMU(t)
    if "R_world_from_imu" in data:
        W_R_IMU = np.asarray(data["R_world_from_imu"], dtype=np.float64)
    else:
        W_R_IMU = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            W_R_IMU[i] = quat_wxyz_to_rotmat_world_from_imu(quat[i])

    # 6) 计算 ^H R_IMU = ^H R_W @ ^W R_IMU
    H_R_IMU = np.einsum("nij,njk->nik", H_R_W, W_R_IMU)

    # 7) 把世界系向量转到 H 系
    #    a_H = ^H R_W @ a_W
    #    omega_H = ^H R_W @ omega_W
    lin_acc_heading = np.einsum("nij,nj->ni", H_R_W, lin_acc_world)
    ang_vel_heading = np.einsum("nij,nj->ni", H_R_W, ang_vel_world)

    processed = {
        "count": count,
        "phase": phase,
        "torso_quaternion": quat,
        "yaw": yaw,
        "yaw_unwrapped": yaw_unwrapped,
        "yaw_ref": yaw_ref,
        "R_heading_from_world": H_R_W,
        "R_heading_from_imu": H_R_IMU,
        "torso_linear_acceleration_heading": lin_acc_heading,
        "torso_angular_velocity_heading": ang_vel_heading,
    }
    return processed


def save_npz(processed: dict, npz_path: str):
    np.savez(npz_path, **processed)


def save_csv(processed: dict, csv_path: str):
    count = processed["count"]
    phase = processed["phase"]
    quat = processed["torso_quaternion"]
    yaw = processed["yaw"]
    yaw_unwrapped = processed["yaw_unwrapped"]
    yaw_ref = processed["yaw_ref"]
    H_R_W = processed["R_heading_from_world"]
    H_R_IMU = processed["R_heading_from_imu"]
    lin_acc_heading = processed["torso_linear_acceleration_heading"]
    ang_vel_heading = processed["torso_angular_velocity_heading"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "count",
                "phase",
                "quat_w", "quat_x", "quat_y", "quat_z",
                "yaw",
                "yaw_unwrapped",
                "yaw_ref",
                "H_R_W_00", "H_R_W_01", "H_R_W_02",
                "H_R_W_10", "H_R_W_11", "H_R_W_12",
                "H_R_W_20", "H_R_W_21", "H_R_W_22",
                "H_R_IMU_00", "H_R_IMU_01", "H_R_IMU_02",
                "H_R_IMU_10", "H_R_IMU_11", "H_R_IMU_12",
                "H_R_IMU_20", "H_R_IMU_21", "H_R_IMU_22",
                "lin_acc_H_x", "lin_acc_H_y", "lin_acc_H_z",
                "ang_vel_H_x", "ang_vel_H_y", "ang_vel_H_z",
            ]
        )

        for i in range(len(count)):
            writer.writerow(
                [
                    count[i],
                    phase[i],
                    *quat[i],
                    yaw[i],
                    yaw_unwrapped[i],
                    yaw_ref[i],
                    *H_R_W[i].reshape(-1),
                    *H_R_IMU[i].reshape(-1),
                    *lin_acc_heading[i],
                    *ang_vel_heading[i],
                ]
            )


def print_summary(processed: dict, npz_path: str, csv_path: str, window_size: int):
    count = processed["count"]
    yaw = processed["yaw"]
    yaw_unwrapped = processed["yaw_unwrapped"]
    yaw_ref = processed["yaw_ref"]

    print("=" * 70)
    print("Heading 坐标系数据处理完成")
    print("=" * 70)
    print(f"样本数                  : {len(count)}")
    print(f"起始时间                : {count[0]:.6f} s")
    print(f"结束时间                : {count[-1]:.6f} s")
    print(f"平滑方法                : 中心滑动平均 (Centered Moving Average)")
    print(f"平滑窗口                : {window_size} 点")
    if len(count) > 1:
        dt_est = np.median(np.diff(count))
        print(f"估计 dt                 : {dt_est:.6f} s")
        print(f"约对应平滑时长          : {window_size * dt_est:.6f} s")
    print(f"输出 npz                : {npz_path}")
    print(f"输出 csv                : {csv_path}")

    print("\n第一帧 yaw / yaw_unwrapped / yaw_ref：")
    print(f"yaw           = {yaw[0]:.6f}")
    print(f"yaw_unwrapped = {yaw_unwrapped[0]:.6f}")
    print(f"yaw_ref       = {yaw_ref[0]:.6f}")

    print("\n第一帧 ^H R_W：")
    print(processed["R_heading_from_world"][0])

    print("\n第一帧 ^H R_IMU：")
    print(processed["R_heading_from_imu"][0])


def main():
    parser = argparse.ArgumentParser(description="将世界系扰动数据转换到 heading-aligned H 坐标系")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_data_processed.npz",
        help="输入 processed npz 文件路径",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1001,
        help="yaw_ref 平滑窗口大小（奇数，单位：样本点）",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_data_heading",
        help="输出文件前缀",
    )
    args = parser.parse_args()

    if args.window_size % 2 == 0:
        print("错误：--window-size 必须为奇数。")
        sys.exit(1)

    try:
        data = load_npz(args.input)
        processed = process_heading(data, args.window_size)
    except Exception as e:
        print(f"处理失败: {e}")
        sys.exit(1)

    npz_path = args.output_prefix + ".npz"
    csv_path = args.output_prefix + "_preview.csv"

    save_npz(processed, npz_path)
    save_csv(processed, csv_path)
    print_summary(processed, npz_path, csv_path, args.window_size)


if __name__ == "__main__":
    main()