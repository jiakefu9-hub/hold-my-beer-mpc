"""把现有直线行走原始数据从 W 系表达转换为 H 系表达。

H 系采用重力对齐的 yaw-only 坐标系。每个样本所用的 heading yaw
来自“上一完整步态周期 torso yaw 的圆周平均”，因此不使用未来数据。
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from heading_template_utils import (
    normalize_quaternions_wxyz,
    quaternion_wxyz_to_rotmat,
    rotmat_to_quaternion_wxyz,
    rotation_to_rpy,
    rotation_z,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT = os.path.join(
    REPO_DIR,
    "disturbance_model_new",
    "torso_disturbance_straight.npz",
)
DEFAULT_OUTPUT_PREFIX = os.path.join(
    SCRIPT_DIR, "torso_disturbance_heading"
)

REQUIRED_KEYS = (
    "count",
    "phase",
    "torso_quaternion",
    "R_world_from_imu",
    "torso_linear_velocity_world",
    "torso_linear_acceleration_world",
    "torso_angular_velocity_world",
    "torso_angular_acceleration_world",
    "left_foot_z",
    "right_foot_z",
    "gait_period",
)


def circular_mean_angle(angles):
    angles = np.asarray(angles, dtype=np.float64)
    sine = float(np.mean(np.sin(angles)))
    cosine = float(np.mean(np.cos(angles)))
    concentration = float(np.hypot(sine, cosine))
    if concentration < 1e-6:
        raise ValueError("周期内 yaw 分布过散，无法定义可靠 heading。")
    return float(np.arctan2(sine, cosine)), concentration


def load_source(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到原始数据: {path}")
    with np.load(path, allow_pickle=False) as source:
        missing = [key for key in REQUIRED_KEYS if key not in source.files]
        if missing:
            raise KeyError(f"原始数据缺少字段: {missing}")
        return {key: source[key].copy() for key in source.files}


def build_heading_data(source, source_path):
    count = np.asarray(source["count"], dtype=np.float64)
    phase = np.asarray(source["phase"], dtype=np.float64)
    quaternion_W_B = normalize_quaternions_wxyz(
        source["torso_quaternion"]
    )
    rotation_W_B = np.asarray(
        source["R_world_from_imu"], dtype=np.float64
    )
    linear_velocity_W = np.asarray(
        source["torso_linear_velocity_world"], dtype=np.float64
    )
    acc_W = np.asarray(
        source["torso_linear_acceleration_world"], dtype=np.float64
    )
    omega_W = np.asarray(
        source["torso_angular_velocity_world"], dtype=np.float64
    )
    alpha_W = np.asarray(
        source["torso_angular_acceleration_world"], dtype=np.float64
    )
    period = float(np.asarray(source["gait_period"]).item())

    n = len(count)
    arrays = (
        phase,
        quaternion_W_B,
        rotation_W_B,
        linear_velocity_W,
        acc_W,
        omega_W,
        alpha_W,
        source["left_foot_z"],
        source["right_foot_z"],
    )
    if period <= 0.0 or any(len(value) != n for value in arrays):
        raise ValueError("原始数据长度不一致或 gait_period 无效。")
    rotation_from_quaternion = quaternion_wxyz_to_rotmat(
        quaternion_W_B
    )
    quaternion_rotation_error = np.max(
        np.linalg.norm(
            rotation_from_quaternion - rotation_W_B, axis=(1, 2)
        )
    )
    if quaternion_rotation_error > 1e-8:
        raise ValueError(
            "torso_quaternion 与 R_world_from_imu 坐标约定不一致。"
        )

    # 直接用 phase 回绕切周期，不依赖每周期恰好有固定采样点数。
    phase_reset = np.concatenate(
        [[False], np.diff(phase) < -0.5]
    )
    cycle_id = np.cumsum(phase_reset).astype(np.int64)
    yaw = rotation_to_rpy(rotation_W_B)[:, 2]
    cycle_heading = {}
    cycle_concentration = {}
    cycle_count = {}
    for cycle in np.unique(cycle_id):
        mask = cycle_id == cycle
        mean, concentration = circular_mean_angle(yaw[mask])
        cycle_heading[int(cycle)] = mean
        cycle_concentration[int(cycle)] = concentration
        cycle_count[int(cycle)] = int(np.count_nonzero(mask))

    heading_yaw = np.empty(n, dtype=np.float64)
    heading_source_cycle = np.empty(n, dtype=np.int64)
    heading_concentration = np.empty(n, dtype=np.float64)
    heading_fallback = np.zeros(n, dtype=bool)
    for i, cycle in enumerate(cycle_id):
        source_cycle = int(cycle) - 1
        if source_cycle not in cycle_heading:
            # 仅第一周期需要回退；模板会丢弃启动段，不会使用这些样本。
            source_cycle = int(cycle)
            heading_fallback[i] = True
        heading_yaw[i] = cycle_heading[source_cycle]
        heading_source_cycle[i] = source_cycle
        heading_concentration[i] = cycle_concentration[source_cycle]

    median_dt = float(np.median(np.diff(count)))
    expected_samples = int(round(period / median_dt))
    complete_cycles = {
        cycle
        for cycle, samples in cycle_count.items()
        if samples >= int(0.9 * expected_samples)
    }
    discard_time = float(
        np.asarray(source.get("discard_time", 4.0)).item()
    )
    used_cycles = np.unique(
        heading_source_cycle[count >= discard_time]
    )
    incomplete = [
        int(cycle)
        for cycle in used_cycles
        if int(cycle) not in complete_cycles
    ]
    if incomplete:
        raise ValueError(
            f"模板稳定段引用了不完整的上一周期: {incomplete}"
        )

    # 【核心代码】^H R_W = Rz(-heading_yaw)，只改变向量的表达基。
    rotation_H_W = rotation_z(-heading_yaw)
    rotation_H_B = np.einsum(
        "nij,njk->nik", rotation_H_W, rotation_W_B
    )
    quaternion_H_B = rotmat_to_quaternion_wxyz(rotation_H_B)
    linear_velocity_H = np.einsum(
        "nij,nj->ni", rotation_H_W, linear_velocity_W
    )
    acc_H = np.einsum("nij,nj->ni", rotation_H_W, acc_W)
    omega_H = np.einsum("nij,nj->ni", rotation_H_W, omega_W)
    alpha_H = np.einsum("nij,nj->ni", rotation_H_W, alpha_W)
    rotation_W_H = np.transpose(rotation_H_W, (0, 2, 1))
    roundtrip_errors = {
        "roundtrip_acceleration_max_error": np.max(
            np.linalg.norm(
                np.einsum("nij,nj->ni", rotation_W_H, acc_H)
                - acc_W,
                axis=1,
            )
        ),
        "roundtrip_linear_velocity_max_error": np.max(
            np.linalg.norm(
                np.einsum(
                    "nij,nj->ni", rotation_W_H, linear_velocity_H
                )
                - linear_velocity_W,
                axis=1,
            )
        ),
        "roundtrip_angular_velocity_max_error": np.max(
            np.linalg.norm(
                np.einsum("nij,nj->ni", rotation_W_H, omega_H)
                - omega_W,
                axis=1,
            )
        ),
        "roundtrip_angular_acceleration_max_error": np.max(
            np.linalg.norm(
                np.einsum("nij,nj->ni", rotation_W_H, alpha_H)
                - alpha_W,
                axis=1,
            )
        ),
        "roundtrip_orientation_max_error": np.max(
            np.linalg.norm(
                np.einsum(
                    "nij,njk->nik", rotation_W_H, rotation_H_B
                )
                - rotation_W_B,
                axis=(1, 2),
            )
        ),
    }
    if max(roundtrip_errors.values()) > 1e-10:
        raise ValueError("W→H→W 可逆性检查失败。")

    return {
        "count": count,
        "phase": phase,
        "cycle_id": cycle_id,
        "heading_source_cycle_id": heading_source_cycle,
        "heading_yaw": heading_yaw,
        "heading_concentration": heading_concentration,
        "heading_fallback": heading_fallback,
        "R_heading_from_world": rotation_H_W,
        "R_heading_from_torso": rotation_H_B,
        "torso_quaternion_heading": quaternion_H_B,
        "torso_linear_velocity_heading": linear_velocity_H,
        "torso_linear_acceleration_heading": acc_H,
        "torso_angular_velocity_heading": omega_H,
        "torso_angular_acceleration_heading": alpha_H,
        "left_foot_z": np.asarray(
            source["left_foot_z"], dtype=np.float64
        ),
        "right_foot_z": np.asarray(
            source["right_foot_z"], dtype=np.float64
        ),
        "gait_period": np.array(period, dtype=np.float64),
        "discard_time": np.array(discard_time, dtype=np.float64),
        "source_world_npz": np.array(
            os.path.abspath(source_path)
        ),
        "heading_definition": np.array(
            "previous_complete_gait_cycle_circular_mean_torso_yaw"
        ),
        "heading_z_definition": np.array("world_gravity_up"),
        "quaternion_order": np.array("wxyz"),
        **{
            key: np.array(value, dtype=np.float64)
            for key, value in roundtrip_errors.items()
        },
    }


def save_preview(data, path):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "count",
                "phase",
                "cycle_id",
                "heading_source_cycle_id",
                "heading_yaw",
                "heading_concentration",
                "heading_fallback",
                "linear_velocity_H_x",
                "linear_velocity_H_y",
                "linear_velocity_H_z",
                "acc_H_x",
                "acc_H_y",
                "acc_H_z",
                "omega_H_x",
                "omega_H_y",
                "omega_H_z",
                "alpha_H_x",
                "alpha_H_y",
                "alpha_H_z",
                "quat_H_B_w",
                "quat_H_B_x",
                "quat_H_B_y",
                "quat_H_B_z",
                "left_foot_z",
                "right_foot_z",
            ]
        )
        for i in range(len(data["count"])):
            writer.writerow(
                [
                    data["count"][i],
                    data["phase"][i],
                    data["cycle_id"][i],
                    data["heading_source_cycle_id"][i],
                    data["heading_yaw"][i],
                    data["heading_concentration"][i],
                    int(data["heading_fallback"][i]),
                    *data["torso_linear_velocity_heading"][i],
                    *data["torso_linear_acceleration_heading"][i],
                    *data["torso_angular_velocity_heading"][i],
                    *data["torso_angular_acceleration_heading"][i],
                    *data["torso_quaternion_heading"][i],
                    data["left_foot_z"][i],
                    data["right_foot_z"][i],
                ]
            )


def save_diagnostic_plot(data, path):
    count = data["count"]
    rpy_H_B = np.rad2deg(
        rotation_to_rpy(data["R_heading_from_torso"])
    )
    fig, axes = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
    axes[0].plot(
        count,
        np.rad2deg(data["heading_yaw"]),
        label="previous-cycle heading yaw",
    )
    axes[0].plot(
        count,
        rpy_H_B[:, 2] + np.rad2deg(data["heading_yaw"]),
        alpha=0.55,
        label="instantaneous torso yaw",
    )
    axes[0].set_ylabel("yaw [deg]")
    axes[0].legend()

    quantities = (
        ("torso_linear_acceleration_heading", "a_H [m/s²]"),
        ("torso_angular_velocity_heading", "omega_H [rad/s]"),
        ("torso_angular_acceleration_heading", "alpha_H [rad/s²]"),
    )
    labels = ("x", "y", "z")
    for axis, (key, title) in zip(axes[1:4], quantities):
        values = data[key]
        for component, label in enumerate(labels):
            axis.plot(count, values[:, component], label=label)
        axis.set_ylabel(title)
        axis.legend()
    for component, label in enumerate(("roll", "pitch", "yaw")):
        axes[4].plot(count, rpy_H_B[:, component], label=label)
    axes[4].set_ylabel("H-from-torso [deg]")
    axes[4].set_xlabel("time [s]")
    axes[4].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="将直线行走原始 W 数据转换为上一周期 heading 对齐的 H 数据"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-prefix", default=DEFAULT_OUTPUT_PREFIX
    )
    args = parser.parse_args()

    source = load_source(args.input)
    data = build_heading_data(source, args.input)
    output_dir = os.path.dirname(os.path.abspath(args.output_prefix))
    os.makedirs(output_dir, exist_ok=True)
    npz_path = args.output_prefix + ".npz"
    csv_path = args.output_prefix + "_preview.csv"
    png_path = args.output_prefix + ".png"
    np.savez(npz_path, **data)
    save_preview(data, csv_path)
    save_diagnostic_plot(data, png_path)

    stable = data["count"] >= float(data["discard_time"])
    print("W→H 转换完成")
    print(f"输入原始数据       : {os.path.abspath(args.input)}")
    print(f"输出 H 数据        : {npz_path}")
    print(f"预览 CSV           : {csv_path}")
    print(f"诊断图             : {png_path}")
    print(
        "稳定段 heading yaw : "
        f"mean={np.rad2deg(np.mean(data['heading_yaw'][stable])):.4f} deg, "
        f"range={np.rad2deg(np.ptp(data['heading_yaw'][stable])):.4f} deg"
    )
    print(
        "稳定段最小集中度   : "
        f"{np.min(data['heading_concentration'][stable]):.8f}"
    )
    print(
        "稳定段 fallback 数 : "
        f"{np.count_nonzero(data['heading_fallback'][stable])}"
    )
    print(
        "W→H→W 最大误差     : "
        f"{max(float(data[key]) for key in data if key.startswith('roundtrip_')):.3e}"
    )


if __name__ == "__main__":
    main()
