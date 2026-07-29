"""由 H-frame 中间数据生成 raw / half-smoothed / fully-smoothed 模板。"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from heading_template_utils import (
    align_quaternion_sequence_wxyz,
    circular_central_difference,
    circular_moving_average,
    circular_quaternion_moving_average,
    markley_quaternion_mean_wxyz,
    quaternion_wxyz_to_rotmat,
    rotation_to_rpy,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(
    SCRIPT_DIR, "torso_disturbance_heading.npz"
)
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "templates_heading")

REQUIRED_KEYS = (
    "count",
    "phase",
    "heading_yaw",
    "torso_quaternion_heading",
    "torso_linear_acceleration_heading",
    "torso_angular_velocity_heading",
    "left_foot_z",
    "right_foot_z",
    "gait_period",
    "heading_definition",
)


def load_source(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 H 数据: {path}")
    with np.load(path, allow_pickle=False) as source:
        missing = [key for key in REQUIRED_KEYS if key not in source.files]
        if missing:
            raise KeyError(f"H 数据缺少字段: {missing}")
        return {key: source[key].copy() for key in source.files}


def build_raw_template(source, discard_time, num_bins, source_path):
    count = np.asarray(source["count"], dtype=np.float64)
    phase = np.asarray(source["phase"], dtype=np.float64)
    heading_yaw = np.asarray(
        source["heading_yaw"], dtype=np.float64
    )
    quaternion_H_B = np.asarray(
        source["torso_quaternion_heading"], dtype=np.float64
    )
    acc_H = np.asarray(
        source["torso_linear_acceleration_heading"], dtype=np.float64
    )
    omega_H = np.asarray(
        source["torso_angular_velocity_heading"], dtype=np.float64
    )
    left_foot_z = np.asarray(source["left_foot_z"], dtype=np.float64)
    right_foot_z = np.asarray(
        source["right_foot_z"], dtype=np.float64
    )
    period = float(np.asarray(source["gait_period"]).item())

    lengths = (
        len(count),
        len(phase),
        len(heading_yaw),
        len(quaternion_H_B),
        len(acc_H),
        len(omega_H),
        len(left_foot_z),
        len(right_foot_z),
    )
    if len(set(lengths)) != 1 or period <= 0.0:
        raise ValueError("H 数据长度不一致或 gait_period 无效。")

    mask = count >= discard_time
    if not np.any(mask):
        raise ValueError("丢弃启动段后没有剩余数据。")
    phase = phase[mask]
    heading_yaw = heading_yaw[mask]
    quaternion_H_B = quaternion_H_B[mask]
    acc_H = acc_H[mask]
    omega_H = omega_H[mask]
    left_foot_z = left_foot_z[mask]
    right_foot_z = right_foot_z[mask]

    bin_ids = np.clip(
        np.floor(phase * num_bins).astype(int), 0, num_bins - 1
    )
    phase_centers = (
        np.arange(num_bins, dtype=np.float64) + 0.5
    ) / num_bins
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

    for bin_id in range(num_bins):
        selected = bin_ids == bin_id
        count_bin = int(np.count_nonzero(selected))
        bin_counts[bin_id] = count_bin
        if count_bin == 0:
            continue
        valid_bins[bin_id] = True
        acc_mean[bin_id] = np.mean(acc_H[selected], axis=0)
        acc_std[bin_id] = np.std(acc_H[selected], axis=0)
        omega_mean[bin_id] = np.mean(omega_H[selected], axis=0)
        omega_std[bin_id] = np.std(omega_H[selected], axis=0)
        orientation_mean[bin_id] = markley_quaternion_mean_wxyz(
            quaternion_H_B[selected]
        )
        dots = np.clip(
            np.abs(
                quaternion_H_B[selected] @ orientation_mean[bin_id]
            ),
            0.0,
            1.0,
        )
        orientation_dispersion[bin_id] = np.sqrt(
            np.mean((2.0 * np.arccos(dots)) ** 2)
        )
        left_mean[bin_id] = np.mean(left_foot_z[selected])
        left_std[bin_id] = np.std(left_foot_z[selected])
        right_mean[bin_id] = np.mean(right_foot_z[selected])
        right_std[bin_id] = np.std(right_foot_z[selected])

    if not np.all(valid_bins):
        missing = np.where(~valid_bins)[0].tolist()
        raise ValueError(f"存在空 phase bin: {missing}")

    orientation_mean = align_quaternion_sequence_wxyz(
        orientation_mean
    )
    orientation_rotation = quaternion_wxyz_to_rotmat(
        orientation_mean
    )
    alpha_template = circular_central_difference(omega_mean, dt_bin)

    return {
        "frame_name": np.array("heading"),
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
        "torso_orientation_rotation_matrix_template": (
            orientation_rotation
        ),
        "torso_orientation_dispersion_rad": orientation_dispersion,
        "orientation_average_method": np.array(
            "markley_quaternion_mean"
        ),
        "orientation_quaternion_order": np.array("wxyz"),
        "orientation_rotation_convention": np.array(
            "heading_from_torso_imu"
        ),
        "heading_definition": np.asarray(
            source["heading_definition"]
        ).copy(),
        "stable_heading_yaw_mean": np.array(
            np.mean(heading_yaw), dtype=np.float64
        ),
        "stable_heading_yaw_std": np.array(
            np.std(heading_yaw), dtype=np.float64
        ),
        "source_heading_npz": np.array(os.path.abspath(source_path)),
        "left_foot_z_mean": left_mean,
        "left_foot_z_std": left_std,
        "right_foot_z_mean": right_mean,
        "right_foot_z_std": right_std,
    }


def smooth_template(raw_template, window_size, fully_smoothed):
    out = {key: value for key, value in raw_template.items()}
    acc_raw = np.asarray(
        raw_template["torso_linear_acceleration_template"]
    )
    omega_raw = np.asarray(
        raw_template["torso_angular_velocity_template"]
    )
    orientation_raw = np.asarray(
        raw_template["torso_orientation_quaternion_template"]
    )

    if fully_smoothed:
        out["torso_linear_acceleration_template_raw"] = acc_raw
        out["torso_linear_acceleration_template"] = (
            circular_moving_average(acc_raw, window_size)
        )
    out["torso_angular_velocity_template_raw"] = omega_raw
    omega_smoothed = circular_moving_average(omega_raw, window_size)
    out["torso_angular_velocity_template"] = omega_smoothed
    out["torso_angular_acceleration_template"] = (
        circular_central_difference(
            omega_smoothed, float(raw_template["dt_bin"])
        )
    )
    out["torso_orientation_quaternion_template_raw"] = orientation_raw
    out["torso_orientation_rotation_matrix_template_raw"] = (
        quaternion_wxyz_to_rotmat(orientation_raw)
    )
    orientation_smoothed = circular_quaternion_moving_average(
        orientation_raw, window_size
    )
    out["torso_orientation_quaternion_template"] = (
        orientation_smoothed
    )
    out["torso_orientation_rotation_matrix_template"] = (
        quaternion_wxyz_to_rotmat(orientation_smoothed)
    )
    out["smoothing_window_size"] = np.array(
        window_size, dtype=np.int64
    )
    out["template_variant"] = np.array(
        "fully_smoothed" if fully_smoothed else "half_smoothed"
    )
    return out


def save_preview(template, path):
    phase = template["phase_centers"]
    counts = template["bin_counts"]
    acc = template["torso_linear_acceleration_template"]
    acc_std = template["torso_linear_acceleration_std"]
    omega = template["torso_angular_velocity_template"]
    omega_std = template["torso_angular_velocity_std"]
    alpha = template["torso_angular_acceleration_template"]
    quaternion = template["torso_orientation_quaternion_template"]
    rotation = template[
        "torso_orientation_rotation_matrix_template"
    ]
    dispersion = template["torso_orientation_dispersion_rad"]
    left_mean = template["left_foot_z_mean"]
    left_std = template["left_foot_z_std"]
    right_mean = template["right_foot_z_mean"]
    right_std = template["right_foot_z_std"]
    acc_raw = template.get(
        "torso_linear_acceleration_template_raw"
    )
    omega_raw = template.get(
        "torso_angular_velocity_template_raw"
    )

    with open(path, "w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        header = [
            "bin_id",
            "phase_center",
            "bin_count",
            "acc_H_x",
            "acc_H_y",
            "acc_H_z",
            "acc_H_std_x",
            "acc_H_std_y",
            "acc_H_std_z",
            "omega_H_x",
            "omega_H_y",
            "omega_H_z",
            "omega_H_std_x",
            "omega_H_std_y",
            "omega_H_std_z",
            "alpha_H_x",
            "alpha_H_y",
            "alpha_H_z",
            "orientation_quat_w",
            "orientation_quat_x",
            "orientation_quat_y",
            "orientation_quat_z",
            *[
                f"orientation_R_{row}{column}"
                for row in range(3)
                for column in range(3)
            ],
            "orientation_dispersion_rad",
            "left_foot_z_mean",
            "left_foot_z_std",
            "right_foot_z_mean",
            "right_foot_z_std",
        ]
        if acc_raw is not None:
            header += ["acc_H_raw_x", "acc_H_raw_y", "acc_H_raw_z"]
        if omega_raw is not None:
            header += [
                "omega_H_raw_x",
                "omega_H_raw_y",
                "omega_H_raw_z",
            ]
        writer.writerow(header)
        for i in range(len(phase)):
            row = [
                i,
                phase[i],
                counts[i],
                *acc[i],
                *acc_std[i],
                *omega[i],
                *omega_std[i],
                *alpha[i],
                *quaternion[i],
                *rotation[i].reshape(-1),
                dispersion[i],
                left_mean[i],
                left_std[i],
                right_mean[i],
                right_std[i],
            ]
            if acc_raw is not None:
                row += list(acc_raw[i])
            if omega_raw is not None:
                row += list(omega_raw[i])
            writer.writerow(row)


def plot_templates(raw, half, full, path):
    templates = (raw, half, full)
    titles = ("H Raw", "H Half Smoothed", "H Fully Smoothed")
    keys = (
        "torso_linear_acceleration_template",
        "torso_angular_velocity_template",
        "torso_angular_acceleration_template",
    )
    ylabels = (
        "a_H [m/s²]",
        "omega_H [rad/s]",
        "alpha_H [rad/s²]",
    )
    labels = ("x", "y", "z")
    colors = ("tab:blue", "tab:orange", "tab:green")
    fig, axes = plt.subplots(4, 3, figsize=(24, 15), sharex=True)
    for column, (template, title) in enumerate(
        zip(templates, titles)
    ):
        phase = template["phase_centers"]
        for row, (key, ylabel) in enumerate(zip(keys, ylabels)):
            values = template[key]
            for component, (label, color) in enumerate(
                zip(labels, colors)
            ):
                axes[row, column].plot(
                    phase,
                    values[:, component],
                    label=label,
                    color=color,
                )
            axes[row, column].set_ylabel(ylabel)
        rpy = np.rad2deg(
            rotation_to_rpy(
                template[
                    "torso_orientation_rotation_matrix_template"
                ]
            )
        )
        for component, (label, color) in enumerate(
            zip(("roll", "pitch", "yaw"), colors)
        ):
            axes[3, column].plot(
                phase, rpy[:, component], label=label, color=color
            )
        axes[3, column].set_ylabel("H RPY [deg]")
        axes[3, column].set_xlabel("phase")
        axes[0, column].set_title(title)
        for row in range(4):
            axes[row, column].grid(True, alpha=0.3)
            axes[row, column].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_summary(name, template, path):
    quaternion = template["torso_orientation_quaternion_template"]
    rotation = template[
        "torso_orientation_rotation_matrix_template"
    ]
    norm_error = np.max(
        np.abs(np.linalg.norm(quaternion, axis=1) - 1.0)
    )
    orth_error = np.max(
        np.linalg.norm(
            np.transpose(rotation, (0, 2, 1)) @ rotation - np.eye(3),
            axis=(1, 2),
        )
    )
    print(f"{name}: {path}")
    print(
        "  bin count min/max/mean = "
        f"{template['bin_counts'].min()}/"
        f"{template['bin_counts'].max()}/"
        f"{template['bin_counts'].mean():.2f}"
    )
    print(
        "  orientation dispersion mean = "
        f"{np.rad2deg(np.mean(template['torso_orientation_dispersion_rad'])):.4f} deg"
    )
    print(
        f"  quaternion/rotation error = {norm_error:.3e}/{orth_error:.3e}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="由 H-frame 中间数据生成三种 phase 扰动模板"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--discard-time", type=float, default=None)
    parser.add_argument("--num-bins", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=5)
    args = parser.parse_args()

    source = load_source(args.input)
    discard_time = (
        float(np.asarray(source["discard_time"]).item())
        if args.discard_time is None and "discard_time" in source
        else 4.0 if args.discard_time is None
        else args.discard_time
    )
    raw = build_raw_template(
        source,
        discard_time,
        args.num_bins,
        args.input,
    )
    raw["template_variant"] = np.array("raw")
    half = smooth_template(raw, args.window_size, False)
    full = smooth_template(raw, args.window_size, True)

    os.makedirs(args.output_dir, exist_ok=True)
    specifications = (
        ("heading_disturbance_template", "H Raw", raw),
        (
            "heading_disturbance_template_half_smoothed",
            "H Half Smoothed",
            half,
        ),
        (
            "heading_disturbance_template_fully_smoothed",
            "H Fully Smoothed",
            full,
        ),
    )
    for prefix, label, template in specifications:
        npz_path = os.path.join(args.output_dir, prefix + ".npz")
        csv_path = os.path.join(
            args.output_dir, prefix + "_preview.csv"
        )
        np.savez(npz_path, **template)
        save_preview(template, csv_path)
        print_summary(label, template, npz_path)
    plot_path = os.path.join(
        args.output_dir, "Heading_Disturbance_Template_Comparison.png"
    )
    plot_templates(raw, half, full, plot_path)
    print(f"H 模板对比图: {plot_path}")


if __name__ == "__main__":
    main()
