# 怎么用
# - 使用默认输入输出路径：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/process_disturbance.py
# - 指定原始 npz 文件：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/process_disturbance.py \
#       --input /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_data.npz
# - 指定输出文件前缀：
#   python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/process_disturbance.py \
#       --output-prefix /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_data_processed

import argparse
import csv
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


def load_raw_disturbance(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"输入文件不存在: {npz_path}")

    data = np.load(npz_path)
    keys = list(data.keys())

    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        raise KeyError(f"原始扰动文件缺少必要字段: {missing}")

    return {k: data[k] for k in keys}


def normalize_quaternion_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("遇到零范数四元数，无法转换为旋转矩阵。")
    return q / n


def quat_wxyz_to_rotmat_world_from_imu(q_wxyz: np.ndarray) -> np.ndarray:
    """
    将 MuJoCo 输出的四元数 [w, x, y, z] 转成旋转矩阵 R_W_IMU。

    这里最关键的约定是：
        v_W = R_W_IMU @ v_IMU

    也就是说，这个矩阵的作用是：
    - 把“在 IMU 局部坐标系下表示的向量”
    - 旋转到“世界坐标系下表示”

    这正是我们后面要做的事情：
    - {}^W omega_IMU = R_W_IMU @ {}^IMU omega_raw
    - {}^W a_IMU     = R_W_IMU @ {}^IMU a_raw - [0, 0, 9.81]^T

    注意：
    - MuJoCo 的 xquat 格式是 [w, x, y, z]
    - 我们这里显式使用这个顺序，避免把四元数顺序搞错
    """
    w, x, y, z = normalize_quaternion_wxyz(q_wxyz)

    # 该旋转矩阵满足：v_W = R_W_IMU @ v_IMU
    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def process_disturbance(raw: dict) -> dict:
    count = np.asarray(raw["count"], dtype=np.float64)
    phase = np.asarray(raw["phase"], dtype=np.float64)

    # 原始 IMU 数据：都还是局部坐标系下的
    lin_acc_local = np.asarray(raw["torso_linear_acceleration"], dtype=np.float64)
    ang_vel_local = np.asarray(raw["torso_angular_velocity"], dtype=np.float64)
    ang_acc_local = np.asarray(raw["torso_angular_acceleration"], dtype=np.float64)
    quat_wxyz = np.asarray(raw["torso_quaternion"], dtype=np.float64)

    left_foot_z = np.asarray(raw["left_foot_z"], dtype=np.float64)
    right_foot_z = np.asarray(raw["right_foot_z"], dtype=np.float64)

    n = len(count)
    if not (
        len(phase) == len(lin_acc_local) == len(ang_vel_local) == len(ang_acc_local)
        == len(quat_wxyz) == len(left_foot_z) == len(right_foot_z) == n
    ):
        raise ValueError("输入数组长度不一致，无法进行数据处理。")

    R_world_from_imu = np.zeros((n, 3, 3), dtype=np.float64)
    ang_vel_world = np.zeros((n, 3), dtype=np.float64)
    lin_acc_world = np.zeros((n, 3), dtype=np.float64)

    # 世界坐标系下的“向上的 9.81”
    # 原始 accelerometer 测的是 proper acceleration（带重力反作用力的读数）
    # 所以：
    #   1) 先把 IMU 局部读数旋转到世界系
    #   2) 再减去世界系中的 [0, 0, 9.81]
    # 这样得到的才是纯运动学线加速度
    gravity_reaction_world = np.array([0.0, 0.0, 9.81], dtype=np.float64)

    for i in range(n):
        q = quat_wxyz[i]
        R_W_IMU = quat_wxyz_to_rotmat_world_from_imu(q)
        R_world_from_imu[i] = R_W_IMU

        # 关键说明：
        # 下面这一步不是 R^T，也不是左乘错方向。
        # 因为我们定义的就是：
        #     v_W = R_W_IMU @ v_IMU
        # 所以局部角速度转世界系，直接这样做：
        ang_vel_world[i] = R_W_IMU @ ang_vel_local[i]

        # 同理，局部 accelerometer 原始读数先转到世界系
        # 再减去世界系中的重力反作用力 [0, 0, 9.81]
        # 得到纯运动学线加速度
        lin_acc_world[i] = R_W_IMU @ lin_acc_local[i] - gravity_reaction_world

    processed = {
        # 基本索引信息
        "count": count,
        "phase": phase,

        # 保留原始数据，便于回溯
        "torso_linear_acceleration_local": lin_acc_local,
        "torso_angular_velocity_local": ang_vel_local,
        "torso_angular_acceleration_local": ang_acc_local,
        "torso_quaternion": quat_wxyz,

        # 第 7 步处理后的结果
        "R_world_from_imu": R_world_from_imu,
        "torso_linear_acceleration_world": lin_acc_world,
        "torso_angular_velocity_world": ang_vel_world,

        # 先保留脚高度，供后面 phase 唯一性验证
        "left_foot_z": left_foot_z,
        "right_foot_z": right_foot_z,
    }
    return processed


def save_processed_npz(processed: dict, npz_path: str):
    np.savez(npz_path, **processed)


def save_processed_csv(processed: dict, csv_path: str):
    count = processed["count"]
    phase = processed["phase"]
    lin_acc_local = processed["torso_linear_acceleration_local"]
    ang_vel_local = processed["torso_angular_velocity_local"]
    ang_acc_local = processed["torso_angular_acceleration_local"]
    quat = processed["torso_quaternion"]
    R = processed["R_world_from_imu"]
    lin_acc_world = processed["torso_linear_acceleration_world"]
    ang_vel_world = processed["torso_angular_velocity_world"]
    left_foot_z = processed["left_foot_z"]
    right_foot_z = processed["right_foot_z"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "count",
                "phase",
                "lin_acc_local_x", "lin_acc_local_y", "lin_acc_local_z",
                "ang_vel_local_x", "ang_vel_local_y", "ang_vel_local_z",
                "ang_acc_local_x", "ang_acc_local_y", "ang_acc_local_z",
                "quat_w", "quat_x", "quat_y", "quat_z",
                "R00", "R01", "R02",
                "R10", "R11", "R12",
                "R20", "R21", "R22",
                "lin_acc_world_x", "lin_acc_world_y", "lin_acc_world_z",
                "ang_vel_world_x", "ang_vel_world_y", "ang_vel_world_z",
                "left_foot_z", "right_foot_z",
            ]
        )

        for i in range(len(count)):
            writer.writerow(
                [
                    count[i],
                    phase[i],
                    *lin_acc_local[i],
                    *ang_vel_local[i],
                    *ang_acc_local[i],
                    *quat[i],
                    *R[i].reshape(-1),
                    *lin_acc_world[i],
                    *ang_vel_world[i],
                    left_foot_z[i],
                    right_foot_z[i],
                ]
            )


def print_summary(processed: dict, npz_path: str, csv_path: str):
    count = processed["count"]
    n = len(count)
    duration = count[-1] - count[0] if n > 1 else 0.0

    print("=" * 60)
    print("第 7 步数据处理完成")
    print("=" * 60)
    print(f"样本数               : {n}")
    print(f"起始时间             : {count[0]:.6f} s")
    print(f"结束时间             : {count[-1]:.6f} s")
    print(f"总时长               : {duration:.6f} s")
    print(f"处理后 npz 保存路径   : {npz_path}")
    print(f"处理后 csv 保存路径   : {csv_path}")

    # 打印一个姿态矩阵示例，便于你核对方向
    q0 = processed["torso_quaternion"][0]
    R0 = processed["R_world_from_imu"][0]
    print("\n第一帧四元数 [w, x, y, z]：")
    print(q0)
    print("\n第一帧旋转矩阵 R_W_IMU（作用：v_W = R_W_IMU @ v_IMU）：")
    print(R0)

    print("\n第一帧局部角速度 -> 世界角速度：")
    print("omega_local =", processed["torso_angular_velocity_local"][0])
    print("omega_world =", processed["torso_angular_velocity_world"][0])

    print("\n第一帧局部线加速度 -> 世界线加速度（已去重力）：")
    print("acc_local   =", processed["torso_linear_acceleration_local"][0])
    print("acc_world   =", processed["torso_linear_acceleration_world"][0])


def main():
    parser = argparse.ArgumentParser(description="第 7 步：处理 disturbance_data.npz，完成坐标变换并导出新文件")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_data.npz",
        help="原始扰动数据 npz 路径",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="/home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model/disturbance_data_processed",
        help="输出文件前缀（会生成 .npz 和 _preview.csv）",
    )
    args = parser.parse_args()

    try:
        raw = load_raw_disturbance(args.input)
        processed = process_disturbance(raw)
    except Exception as e:
        print(f"处理失败: {e}")
        sys.exit(1)

    npz_path = args.output_prefix + ".npz"
    csv_path = args.output_prefix + "_preview.csv"

    save_processed_npz(processed, npz_path)
    save_processed_csv(processed, csv_path)
    print_summary(processed, npz_path, csv_path)


if __name__ == "__main__":
    main()