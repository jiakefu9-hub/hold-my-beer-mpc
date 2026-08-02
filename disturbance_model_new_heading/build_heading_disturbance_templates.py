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
    "torso_linear_velocity_heading",
    "torso_linear_acceleration_heading",
    "torso_angular_velocity_heading",
    "torso_angular_acceleration_heading",
    "R_heading_from_world",
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


def _phase_grid_ids(phase, num_bins):
    """把相位分到最近的 interval-start 网格点。"""
    return np.mod(
        np.floor(np.asarray(phase) * num_bins + 0.5).astype(np.int64),
        num_bins,
    )


def _vector_bin_statistics(values, bin_ids, num_bins):
    mean = np.zeros((num_bins, 3), dtype=np.float64)
    std = np.zeros_like(mean)
    counts = np.zeros(num_bins, dtype=np.int64)
    for bin_id in range(num_bins):
        selected = bin_ids == bin_id
        counts[bin_id] = int(np.count_nonzero(selected))
        if counts[bin_id]:
            mean[bin_id] = np.mean(values[selected], axis=0)
            std[bin_id] = np.std(values[selected], axis=0)
    return mean, std, counts


def _load_world_source(source):
    path = str(np.asarray(source["source_world_npz"]).item())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 H 数据对应的 W 原始文件: {path}")
    with np.load(path, allow_pickle=False) as world:
        required = (
            "count",
            "torso_linear_velocity_world",
            "torso_angular_velocity_world",
        )
        missing = [key for key in required if key not in world.files]
        if missing:
            raise KeyError(
                "严格控制区间模板需要重新采集的速度字段，缺少: "
                f"{missing}"
            )
        return path, {key: world[key].copy() for key in world.files}


def _build_interval_samples(source, control_dt):
    """由同一世界系下的速度端点构造未来 control_dt 区间平均量。"""
    world_path, world = _load_world_source(source)
    count = np.asarray(source["count"], dtype=np.float64)
    world_count = np.asarray(world["count"], dtype=np.float64)
    if count.shape != world_count.shape or not np.allclose(
        count, world_count, rtol=0.0, atol=1e-10
    ):
        raise ValueError("H/W 原始数据时间轴不一致。")

    source_dt = float(np.median(np.diff(count)))
    interval_steps = int(round(control_dt / source_dt))
    if (
        source_dt <= 0.0
        or interval_steps < 1
        or not np.isclose(
            interval_steps * source_dt, control_dt, rtol=0.0, atol=1e-10
        )
    ):
        raise ValueError("control_dt 必须是原始数据采样周期的整数倍。")

    start = np.arange(len(count) - interval_steps, dtype=np.int64)
    end = start + interval_steps
    valid = np.isclose(
        count[end] - count[start], control_dt, rtol=0.0, atol=1e-9
    )
    start = start[valid]
    end = end[valid]

    rotation_H_W = np.asarray(
        source["R_heading_from_world"], dtype=np.float64
    )[start]
    velocity_W = np.asarray(
        world["torso_linear_velocity_world"], dtype=np.float64
    )
    omega_W = np.asarray(
        world["torso_angular_velocity_world"], dtype=np.float64
    )
    interval_acc_H = np.einsum(
        "nij,nj->ni",
        rotation_H_W,
        (velocity_W[end] - velocity_W[start]) / control_dt,
    )
    interval_alpha_H = np.einsum(
        "nij,nj->ni",
        rotation_H_W,
        (omega_W[end] - omega_W[start]) / control_dt,
    )

    # 将控制区间拆成原始采样子区间；复合梯形积分的归一化权重如下。
    weights = np.ones(interval_steps + 1, dtype=np.float64)
    weights[[0, -1]] = 0.5
    weights /= np.sum(weights)
    omega_interval_W = np.zeros((len(start), 3), dtype=np.float64)
    for offset, weight in enumerate(weights):
        omega_interval_W += weight * omega_W[start + offset]
    interval_omega_H = np.einsum(
        "nij,nj->ni", rotation_H_W, omega_interval_W
    )
    return {
        "world_path": world_path,
        "source_dt": source_dt,
        "interval_steps": interval_steps,
        "phase": np.asarray(source["phase"], dtype=np.float64)[start],
        "count": count[start],
        "acc": interval_acc_H,
        "omega": interval_omega_H,
        "alpha": interval_alpha_H,
    }


def build_raw_template(
    source, discard_time, num_bins, source_path, control_dt
):
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
    alpha_H = np.asarray(
        source["torso_angular_acceleration_heading"], dtype=np.float64
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
        len(alpha_H),
        len(left_foot_z),
        len(right_foot_z),
    )
    if len(set(lengths)) != 1 or period <= 0.0:
        raise ValueError("H 数据长度不一致或 gait_period 无效。")

    node_mask = count >= discard_time
    if not np.any(node_mask):
        raise ValueError("丢弃启动段后没有剩余数据。")
    node_phase = phase[node_mask]
    node_bin_ids = _phase_grid_ids(node_phase, num_bins)
    phase_centers = np.arange(num_bins, dtype=np.float64) / num_bins
    dt_bin = period / num_bins
    acc_mean, acc_std, bin_counts = _vector_bin_statistics(
        acc_H[node_mask], node_bin_ids, num_bins
    )
    omega_mean, omega_std, _ = _vector_bin_statistics(
        omega_H[node_mask], node_bin_ids, num_bins
    )
    alpha_mean, alpha_std, _ = _vector_bin_statistics(
        alpha_H[node_mask], node_bin_ids, num_bins
    )
    valid_bins = bin_counts > 0

    orientation_mean = np.zeros((num_bins, 4), dtype=np.float64)
    orientation_dispersion = np.zeros(num_bins, dtype=np.float64)
    left_mean = np.zeros(num_bins, dtype=np.float64)
    left_std = np.zeros(num_bins, dtype=np.float64)
    right_mean = np.zeros(num_bins, dtype=np.float64)
    right_std = np.zeros(num_bins, dtype=np.float64)

    for bin_id in range(num_bins):
        selected = node_bin_ids == bin_id
        if not np.any(selected):
            continue
        orientation_mean[bin_id] = markley_quaternion_mean_wxyz(
            quaternion_H_B[node_mask][selected]
        )
        dots = np.clip(
            np.abs(
                quaternion_H_B[node_mask][selected]
                @ orientation_mean[bin_id]
            ),
            0.0,
            1.0,
        )
        orientation_dispersion[bin_id] = np.sqrt(
            np.mean((2.0 * np.arccos(dots)) ** 2)
        )
        left_mean[bin_id] = np.mean(left_foot_z[node_mask][selected])
        left_std[bin_id] = np.std(left_foot_z[node_mask][selected])
        right_mean[bin_id] = np.mean(right_foot_z[node_mask][selected])
        right_std[bin_id] = np.std(right_foot_z[node_mask][selected])

    if not np.all(valid_bins):
        missing = np.where(~valid_bins)[0].tolist()
        raise ValueError(f"存在空 phase bin: {missing}")

    orientation_mean = align_quaternion_sequence_wxyz(
        orientation_mean
    )
    orientation_rotation = quaternion_wxyz_to_rotmat(
        orientation_mean
    )

    interval = _build_interval_samples(source, control_dt)
    interval_mask = interval["count"] >= discard_time
    interval_bin_ids = _phase_grid_ids(
        interval["phase"][interval_mask], num_bins
    )
    interval_acc_mean, interval_acc_std, interval_counts = (
        _vector_bin_statistics(
            interval["acc"][interval_mask],
            interval_bin_ids,
            num_bins,
        )
    )
    interval_omega_mean, interval_omega_std, _ = _vector_bin_statistics(
        interval["omega"][interval_mask],
        interval_bin_ids,
        num_bins,
    )
    interval_alpha_mean, interval_alpha_std, _ = _vector_bin_statistics(
        interval["alpha"][interval_mask],
        interval_bin_ids,
        num_bins,
    )
    if np.any(interval_counts == 0):
        missing = np.where(interval_counts == 0)[0].tolist()
        raise ValueError(f"控制区间模板存在空 phase bin: {missing}")

    return {
        "template_schema_version": np.array(2, dtype=np.int64),
        "frame_name": np.array("heading"),
        "period": np.array(period, dtype=np.float64),
        "source_dt": np.array(interval["source_dt"], dtype=np.float64),
        "interval_dt": np.array(control_dt, dtype=np.float64),
        "interval_steps": np.array(
            interval["interval_steps"], dtype=np.int64
        ),
        "phase_reference": np.array("interval_start"),
        "phase_grid_convention": np.array("uniform_start_grid"),
        "discard_time": np.array(discard_time, dtype=np.float64),
        "num_bins": np.array(num_bins, dtype=np.int64),
        "dt_bin": np.array(dt_bin, dtype=np.float64),
        "phase_centers": phase_centers,
        "bin_counts": bin_counts,
        "valid_bins": valid_bins,
        "interval_bin_counts": interval_counts,
        "interval_valid_bins": interval_counts > 0,
        "torso_linear_acceleration_node_template": acc_mean,
        "torso_linear_acceleration_node_std": acc_std,
        "torso_angular_velocity_node_template": omega_mean,
        "torso_angular_velocity_node_std": omega_std,
        "torso_angular_acceleration_node_template": alpha_mean,
        "torso_angular_acceleration_node_std": alpha_std,
        "torso_linear_acceleration_interval_template": (
            interval_acc_mean
        ),
        "torso_linear_acceleration_interval_std": interval_acc_std,
        "torso_angular_velocity_interval_template": (
            interval_omega_mean
        ),
        "torso_angular_velocity_interval_std": interval_omega_std,
        "torso_angular_acceleration_interval_template": (
            interval_alpha_mean
        ),
        "torso_angular_acceleration_interval_std": interval_alpha_std,
        # 旧字段继续指向 node 量，便于已有绘图和对比脚本读取。
        "torso_linear_acceleration_template": acc_mean,
        "torso_linear_acceleration_std": acc_std,
        "torso_angular_velocity_template": omega_mean,
        "torso_angular_velocity_std": omega_std,
        "torso_angular_acceleration_template": alpha_mean,
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
            np.mean(heading_yaw[node_mask]), dtype=np.float64
        ),
        "stable_heading_yaw_std": np.array(
            np.std(heading_yaw[node_mask]), dtype=np.float64
        ),
        "source_heading_npz": np.array(os.path.abspath(source_path)),
        "source_world_npz": np.array(
            os.path.abspath(interval["world_path"])
        ),
        "left_foot_z_mean": left_mean,
        "left_foot_z_std": left_std,
        "right_foot_z_mean": right_mean,
        "right_foot_z_std": right_std,
    }


def smooth_template(
    raw_template,
    node_window_size,
    interval_window_size,
    fully_smoothed,
):
    out = {key: value for key, value in raw_template.items()}
    acc_raw = np.asarray(
        raw_template["torso_linear_acceleration_node_template"]
    )
    omega_raw = np.asarray(
        raw_template["torso_angular_velocity_node_template"]
    )
    alpha_raw = np.asarray(
        raw_template["torso_angular_acceleration_node_template"]
    )
    orientation_raw = np.asarray(
        raw_template["torso_orientation_quaternion_template"]
    )

    out["torso_linear_acceleration_node_template_raw"] = acc_raw
    out["torso_angular_velocity_node_template_raw"] = omega_raw
    out["torso_angular_acceleration_node_template_raw"] = alpha_raw
    if fully_smoothed:
        out["torso_linear_acceleration_node_template"] = (
            circular_moving_average(acc_raw, node_window_size)
        )
    out["torso_angular_velocity_node_template"] = (
        circular_moving_average(omega_raw, node_window_size)
    )
    out["torso_angular_acceleration_node_template"] = (
        circular_moving_average(alpha_raw, node_window_size)
    )

    interval_keys = (
        "torso_linear_acceleration_interval_template",
        "torso_angular_velocity_interval_template",
        "torso_angular_acceleration_interval_template",
    )
    for key in interval_keys:
        raw_value = np.asarray(raw_template[key])
        out[key + "_raw"] = raw_value
        # half-smoothed 有意保留区间冲击；fully-smoothed 只做一个控制区间
        # 宽度的小平滑，不再沿用旧模板约 40 ms 的削峰窗口。
        if fully_smoothed:
            out[key] = circular_moving_average(
                raw_value, interval_window_size
            )

    # 旧字段同步到 node 字段，保证已有预览/对比脚本的语义明确。
    out["torso_linear_acceleration_template_raw"] = acc_raw
    out["torso_angular_velocity_template_raw"] = omega_raw
    out["torso_linear_acceleration_template"] = out[
        "torso_linear_acceleration_node_template"
    ]
    out["torso_angular_velocity_template"] = out[
        "torso_angular_velocity_node_template"
    ]
    out["torso_angular_acceleration_template"] = out[
        "torso_angular_acceleration_node_template"
    ]
    out["torso_orientation_quaternion_template_raw"] = orientation_raw
    out["torso_orientation_rotation_matrix_template_raw"] = (
        quaternion_wxyz_to_rotmat(orientation_raw)
    )
    orientation_smoothed = circular_quaternion_moving_average(
        orientation_raw, node_window_size
    )
    out["torso_orientation_quaternion_template"] = (
        orientation_smoothed
    )
    out["torso_orientation_rotation_matrix_template"] = (
        quaternion_wxyz_to_rotmat(orientation_smoothed)
    )
    out["node_smoothing_window_size"] = np.array(
        node_window_size, dtype=np.int64
    )
    out["interval_smoothing_window_size"] = np.array(
        interval_window_size if fully_smoothed else 1,
        dtype=np.int64,
    )
    out["smoothing_window_size"] = np.array(
        node_window_size, dtype=np.int64
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
    parser.add_argument("--num-bins", type=int, default=400)
    parser.add_argument("--control-dt", type=float, default=0.006)
    parser.add_argument("--node-window-size", type=int, default=21)
    parser.add_argument("--interval-window-size", type=int, default=3)
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
        args.control_dt,
    )
    raw["template_variant"] = np.array("raw")
    half = smooth_template(
        raw,
        args.node_window_size,
        args.interval_window_size,
        False,
    )
    full = smooth_template(
        raw,
        args.node_window_size,
        args.interval_window_size,
        True,
    )

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
