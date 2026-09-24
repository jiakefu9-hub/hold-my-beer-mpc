#!/usr/bin/env python3
"""Offline H0 endpoint/control metrics for one ``g1_walk_pid.py`` run.

The primary window is fixed to task seconds [5,18): from the first walking
request through the end of the three-second stop-settle interval.  Startup,
steady walking and stopping subwindows are diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from endpoint_pose import EndpointModel, rotation
from hardware_pid_control import RELEASE_START_S, WALK_START_S, WALK_STOP_S

GRAVITY_H0 = np.array([0.0, 0.0, -9.81])
ANALYSIS_MOTOR_INDICES = (15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 12)


def read_records(path):
    rows = []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc
            schema = row.get("schema")
            event = row.get("event")
            if schema == "g1_lowstate_raw_v1":
                motors = {int(motor["index"]): motor for motor in row["motors"]}
                rows.append({
                    "schema": schema,
                    "received_monotonic_ns": row["received_monotonic_ns"],
                    "crc_valid": row.get("crc_valid", False),
                    "mapped_q_rad": [
                        motors[index]["q_rad"] for index in ANALYSIS_MOTOR_INDICES
                    ],
                })
            elif schema == "g1_torso_imu_raw_v1":
                rows.append({
                    "schema": schema,
                    "received_monotonic_ns": row["received_monotonic_ns"],
                    "quaternion_wxyz": row["quaternion_wxyz"],
                    "gyroscope_rad_s": row["gyroscope_rad_s"],
                    "accelerometer_raw_m_s2": row["accelerometer_raw_m_s2"],
                })
            elif schema in {"g1_pid_session_v1", "g1_hardware_pid_command_v1"} or event in {
                "task_epoch", "heading_reference_frozen",
            }:
                rows.append(row)
    return rows, digest.hexdigest()


def _single(rows, predicate, description):
    selected = [row for row in rows if predicate(row)]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one {description}, got {len(selected)}")
    return selected[0]


def _numeric(value):
    if isinstance(value, str):
        return float(value)
    return value


def _continuous_quaternions(quaternions):
    result = np.asarray(quaternions, dtype=float).copy()
    norms = np.linalg.norm(result, axis=1)
    if np.any(~np.isfinite(result)) or np.any((norms < 0.5) | (norms > 1.5)):
        raise ValueError("invalid torso quaternion")
    result /= norms[:, None]
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _interp(source_t, values, target_t):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.interp(target_t, source_t, values)
    return np.column_stack([
        np.interp(target_t, source_t, values[:, column])
        for column in range(values.shape[1])
    ])


def _rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rpy(rotations):
    return Rotation.from_matrix(rotations).as_euler("xyz", degrees=False)


def _quaternion_wxyz(rotations):
    xyzw = Rotation.from_matrix(rotations).as_quat()
    return xyzw[:, [3, 0, 1, 2]]


def _angular_velocity(rotations, dt):
    count = len(rotations)
    omega = np.zeros((count, 3), dtype=float)
    if count < 2:
        return omega
    increments_local = Rotation.from_matrix(
        np.einsum("nij,njk->nik", np.transpose(rotations[:-1], (0, 2, 1)), rotations[1:])
    ).as_rotvec() / dt
    increments_h0 = np.einsum("nij,nj->ni", rotations[:-1], increments_local)
    omega[:-1] = increments_h0
    omega[-1] = increments_h0[-1]
    return omega


def _norm_stats(values):
    values = np.linalg.norm(np.asarray(values, dtype=float), axis=1)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _scalar_stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _window_metrics(data, mask):
    if int(np.sum(mask)) < 2:
        raise ValueError("metric window has too few samples")
    result = {
        "samples": int(np.sum(mask)),
        "torso": {
            "linear_acceleration_h0_m_s2_norm": _norm_stats(
                data["torso_linear_acceleration_h0_m_s2"][mask]
            ),
            "horizontal_linear_acceleration_h0_m_s2_norm": _norm_stats(
                data["torso_linear_acceleration_h0_m_s2"][mask, :2]
            ),
            "angular_acceleration_h0_rad_s2_norm": _norm_stats(
                data["torso_angular_acceleration_h0_rad_s2"][mask]
            ),
        },
    }
    for side in ("left", "right"):
        orientation = data[f"{side}_orientation_h0_rpy_rad"][mask]
        circular_mean = np.arctan2(np.mean(np.sin(orientation), axis=0),
                                   np.mean(np.cos(orientation), axis=0))
        result[side] = {
            "endpoint_linear_acceleration_h0_m_s2_norm": _norm_stats(
                data[f"{side}_linear_acceleration_h0_m_s2"][mask]
            ),
            "endpoint_horizontal_linear_acceleration_h0_m_s2_norm": _norm_stats(
                data[f"{side}_linear_acceleration_h0_m_s2"][mask, :2]
            ),
            "endpoint_angular_acceleration_h0_rad_s2_norm": _norm_stats(
                data[f"{side}_angular_acceleration_h0_rad_s2"][mask]
            ),
            "bottle_z_tilt_from_h0_vertical_deg": _scalar_stats(
                data[f"{side}_tilt_deg"][mask]
            ),
            "upright_alignment": {
                "mean": float(np.mean(data[f"{side}_upright_alignment"][mask])),
                "min": float(np.min(data[f"{side}_upright_alignment"][mask])),
            },
            "orientation_h0_rpy_deg": {
                "circular_mean": np.rad2deg(circular_mean).tolist(),
                "unwrapped_std": np.rad2deg(
                    np.std(np.unwrap(orientation, axis=0), axis=0)
                ).tolist(),
            },
        }
    return result


def analyze(raw_path, sample_hz=200.0, filter_window_s=0.105):
    rows, digest = read_records(raw_path)
    epoch = int(_single(rows, lambda row: row.get("event") == "task_epoch",
                        "task_epoch")["task_epoch_monotonic_ns"])
    reference = _single(rows, lambda row: row.get("event") == "heading_reference_frozen",
                        "heading_reference_frozen")
    yaw0 = float(reference["yaw0_rad"])
    session = _single(rows, lambda row: row.get("schema") == "g1_pid_session_v1",
                      "PID session header")
    if session.get("primary_metric_window_s") != [WALK_START_S, RELEASE_START_S]:
        raise ValueError("raw session does not declare the required [5,18) primary window")
    low_all = sorted(
        [row for row in rows if row.get("schema") == "g1_lowstate_raw_v1"],
        key=lambda row: row["received_monotonic_ns"],
    )
    invalid_primary = [
        row for row in low_all
        if not row.get("crc_valid", False)
        and WALK_START_S <= (row["received_monotonic_ns"] - epoch) * 1e-9 < RELEASE_START_S
    ]
    if invalid_primary:
        raise ValueError("primary [5,18) window contains CRC-invalid LowState")
    low = [row for row in low_all if row.get("crc_valid", False)]
    imu = sorted(
        [row for row in rows if row.get("schema") == "g1_torso_imu_raw_v1"],
        key=lambda row: row["received_monotonic_ns"],
    )
    commands = sorted(
        [row for row in rows if row.get("schema") == "g1_hardware_pid_command_v1"
         and row.get("event") == "dds_write"],
        key=lambda row: row["task_elapsed_s"],
    )
    if len(low) < 100 or len(imu) < 100 or len(commands) < 100:
        raise ValueError("capture is incomplete: requires LowState, torso IMU and command streams")
    low_t = np.asarray([(row["received_monotonic_ns"] - epoch) * 1e-9 for row in low])
    imu_t = np.asarray([(row["received_monotonic_ns"] - epoch) * 1e-9 for row in imu])
    start = max(float(low_t[0]), float(imu_t[0]))
    stop = min(float(low_t[-1]), float(imu_t[-1]))
    dt = 1.0 / float(sample_hz)
    grid = np.arange(math.ceil(start / dt) * dt, stop, dt)
    if grid[0] > WALK_START_S or grid[-1] < RELEASE_START_S - dt:
        raise ValueError("raw streams do not cover the complete [5,18) primary window")

    low_q = np.asarray([
        [_numeric(value) for value in row["mapped_q_rad"]]
        for row in low
    ], dtype=float)
    if not np.isfinite(low_q).all():
        raise ValueError("mapped arm/waist state contains non-finite values")
    slots = np.zeros((len(grid), 13), dtype=float)
    slots[:, :11] = _interp(low_t, low_q, grid)
    imu_q = _continuous_quaternions([row["quaternion_wxyz"] for row in imu])
    q_grid = _continuous_quaternions(_interp(imu_t, imu_q, grid))
    accel_imu = _interp(imu_t, [row["accelerometer_raw_m_s2"] for row in imu], grid)
    gyro_imu = _interp(imu_t, [row["gyroscope_rad_s"] for row in imu], grid)

    h0_from_world = _rotation_z(-yaw0)
    world_from_body = np.asarray([rotation(quaternion) for quaternion in q_grid])
    h0_from_body = np.einsum("ij,njk->nik", h0_from_world, world_from_body)
    specific_force_h0 = np.einsum("nij,nj->ni", h0_from_body, accel_imu)
    body_acceleration_h0 = specific_force_h0 + GRAVITY_H0
    body_angular_velocity_h0 = np.einsum("nij,nj->ni", h0_from_body, gyro_imu)

    captured_config = raw_path.parent / "controller_config.yaml"
    model = EndpointModel(captured_config) if captured_config.is_file() else EndpointModel()
    torso_quaternion_wxyz = _quaternion_wxyz(h0_from_body)
    endpoint_positions = {side: [] for side in ("left", "right")}
    endpoint_rotations = {side: [] for side in ("left", "right")}
    for q_slots, h0_from_b in zip(slots, h0_from_body):
        relative = model.relative(q_slots)
        for side in ("left", "right"):
            position_b, body_from_endpoint = relative[side]
            endpoint_positions[side].append(h0_from_b @ position_b)
            endpoint_rotations[side].append(h0_from_b @ body_from_endpoint)

    window = max(5, int(round(filter_window_s * sample_hz)))
    if window % 2 == 0:
        window += 1
    if window >= len(grid):
        raise ValueError("capture is too short for the derivative filter")
    body_acceleration_filtered = savgol_filter(
        body_acceleration_h0, window, 3, axis=0, mode="interp"
    )
    body_angular_velocity_filtered = savgol_filter(
        body_angular_velocity_h0, window, 3, axis=0, mode="interp"
    )
    data = {
        "task_elapsed_s": grid,
        "arm_slots_rad": slots,
        "torso_orientation_h0_rpy_rad": _rpy(h0_from_body),
        "torso_orientation_h0_quaternion_wxyz": torso_quaternion_wxyz,
        "torso_specific_force_h0_m_s2": specific_force_h0,
        "torso_linear_acceleration_h0_m_s2": body_acceleration_filtered,
        "torso_angular_velocity_h0_rad_s": body_angular_velocity_filtered,
        "torso_angular_acceleration_h0_rad_s2": savgol_filter(
            body_angular_velocity_filtered, window, 3, deriv=1, delta=dt,
            axis=0, mode="interp"
        ),
    }
    for side in ("left", "right"):
        positions = np.asarray(endpoint_positions[side])
        rotations = np.asarray(endpoint_rotations[side])
        relative_acceleration = savgol_filter(
            positions, window, 3, deriv=2, delta=dt, axis=0, mode="interp"
        )
        linear_acceleration = body_acceleration_filtered + relative_acceleration
        angular_velocity = _angular_velocity(rotations, dt)
        angular_velocity = savgol_filter(
            angular_velocity, window, 3, axis=0, mode="interp"
        )
        angular_acceleration = savgol_filter(
            angular_velocity, window, 3, deriv=1, delta=dt, axis=0, mode="interp"
        )
        rpy = _rpy(rotations)
        upright = np.clip(rotations[:, 2, 2], -1.0, 1.0)
        data.update({
            f"{side}_position_from_torso_imu_h0_m": positions,
            f"{side}_orientation_h0_rpy_rad": rpy,
            f"{side}_orientation_h0_quaternion_wxyz": _quaternion_wxyz(rotations),
            f"{side}_linear_acceleration_h0_m_s2": linear_acceleration,
            f"{side}_angular_velocity_h0_rad_s": angular_velocity,
            f"{side}_angular_acceleration_h0_rad_s2": angular_acceleration,
            f"{side}_upright_alignment": upright,
            f"{side}_tilt_deg": np.rad2deg(np.arccos(upright)),
        })

    primary_mask = (grid >= WALK_START_S) & (grid < RELEASE_START_S)
    windows = {
        "primary_walk_start_through_stop_settle_end": {
            "interval_s": [WALK_START_S, RELEASE_START_S],
            "role": "headline",
            "metrics": _window_metrics(data, primary_mask),
        },
        "diagnostic_startup_walk": {
            "interval_s": [WALK_START_S, 7.0], "role": "diagnostic_only",
            "metrics": _window_metrics(data, (grid >= WALK_START_S) & (grid < 7.0)),
        },
        "diagnostic_later_walk": {
            "interval_s": [7.0, WALK_STOP_S], "role": "diagnostic_only",
            "metrics": _window_metrics(data, (grid >= 7.0) & (grid < WALK_STOP_S)),
        },
        "diagnostic_stop_settle": {
            "interval_s": [WALK_STOP_S, RELEASE_START_S], "role": "diagnostic_only",
            "metrics": _window_metrics(data, (grid >= WALK_STOP_S) & (grid < RELEASE_START_S)),
        },
    }

    command_primary = [
        row for row in commands
        if WALK_START_S <= float(row["task_elapsed_s"]) < RELEASE_START_S
    ]
    right_tracking = np.asarray([
        np.asarray(row["q_command_rad"], dtype=float)[5:10]
        - np.asarray(row["q_measured_rad"], dtype=float)[5:10]
        for row in command_primary
    ])
    gravity_errors = np.asarray([
        row["gravity_error_before_m_s2"] for row in command_primary
        if row.get("pid_active") and "gravity_error_before_m_s2" in row
    ], dtype=float)
    clipped = np.asarray([
        row.get("q_reference_clipped", [False] * 5) for row in command_primary
    ], dtype=bool)
    timing_compute = np.asarray([row["controller_compute_us"] for row in command_primary])
    timing_write = np.asarray([row["write_duration_us"] for row in command_primary])
    control = {
        "samples": len(command_primary),
        "right_joint_tracking_error_rad": {
            "rms_per_joint": np.sqrt(np.mean(right_tracking ** 2, axis=0)).tolist(),
            "max_abs_per_joint": np.max(np.abs(right_tracking), axis=0).tolist(),
        },
        "right_gravity_error_m_s2_norm": _norm_stats(gravity_errors),
        "q_reference_clip_fraction_per_joint": np.mean(clipped, axis=0).tolist(),
        "controller_compute_us": _scalar_stats(timing_compute),
        "dds_write_duration_us": _scalar_stats(timing_write),
    }
    governor_rows = [
        row for row in command_primary
        if row.get("pid_active")
        and "raw_pid_dq_ref_rad_s" in row
        and "governed_dq_ref_rad_s" in row
        and "governed_ddq_ref_rad_s2" in row
    ]
    if governor_rows:
        raw_pid_dq = np.asarray([
            row["raw_pid_dq_ref_rad_s"] for row in governor_rows
        ], dtype=float)
        governed_dq = np.asarray([
            row["governed_dq_ref_rad_s"] for row in governor_rows
        ], dtype=float)
        governed_ddq = np.asarray([
            row["governed_ddq_ref_rad_s2"] for row in governor_rows
        ], dtype=float)
        control["hardware_governor"] = {
            "samples": len(governor_rows),
            "raw_max_abs_dq_rad_s_per_joint": np.max(
                np.abs(raw_pid_dq), axis=0
            ).tolist(),
            "sent_max_abs_dq_rad_s_per_joint": np.max(
                np.abs(governed_dq), axis=0
            ).tolist(),
            "sent_max_abs_ddq_rad_s2_per_joint": np.max(
                np.abs(governed_ddq), axis=0
            ).tolist(),
            "raw_velocity_limited_fraction_per_joint": np.mean(np.asarray([
                row.get("raw_velocity_limited", [False] * 5)
                for row in governor_rows
            ], dtype=bool), axis=0).tolist(),
        }
    summary = {
        "schema": "g1_hardware_pid_analysis_v1",
        "source_raw_jsonl": str(raw_path.resolve()),
        "source_sha256": digest,
        "yaw0_rad": yaw0,
        "frame": "fixed H0: +X is pre-walk mean yaw, +Z is navigation-world vertical",
        "quaternion_order": "wxyz",
        "endpoint_model_xml_sha256": model.xml_hashes(),
        "endpoint_model_config": str(model.config),
        "captured_profile_sha256": session.get("profile_sha256"),
        "captured_controller_config_sha256": session.get("controller_config_sha256"),
        "capture_quality": {
            "lowstate_records_total": len(low_all),
            "lowstate_crc_invalid_total": len(low_all) - len(low),
            "lowstate_crc_invalid_primary_window": len(invalid_primary),
            "torso_imu_records_total": len(imu),
            "successful_command_records_total": len(commands),
        },
        "primary_metric_rule": "all samples from first walk command through end of stop-settle; [5,18)",
        "sample_hz": sample_hz,
        "derivative_filter": {
            "method": "uniform resampling plus cubic Savitzky-Golay",
            "requested_window_s": filter_window_s,
            "actual_window_samples": window,
            "actual_window_s": window / sample_hz,
        },
        "windows": windows,
        "control": control,
        "limitations": [
            "endpoint pose and relative acceleration are model-derived from measured joints",
            "torso IMU and LowState are paired by host receive-time interpolation, not hardware timestamps",
            "absolute world translation is unavailable",
            "zero-velocity RPC and a three-second settle interval do not independently prove physical standstill",
            "filtered hardware acceleration is not numerically identical to MuJoCo ground-truth acceleration",
            "the centered Savitzky-Golay derivative is offline/non-causal and is not an online controller input",
        ],
    }
    data["yaw0_rad"] = np.asarray(yaw0)
    return data, summary


def write_outputs(data, summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "metrics.npz", **data)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    fields = [
        "task_elapsed_s",
        *(f"torso_linear_acceleration_h0_{axis}_m_s2" for axis in "xyz"),
        *(f"torso_angular_velocity_h0_{axis}_rad_s" for axis in "xyz"),
        *(f"torso_angular_acceleration_h0_{axis}_rad_s2" for axis in "xyz"),
        *(f"torso_orientation_h0_{axis}_rad" for axis in ("roll", "pitch", "yaw")),
    ]
    for side in ("left", "right"):
        fields.extend([
            *(f"{side}_linear_acceleration_h0_{axis}_m_s2" for axis in "xyz"),
            *(f"{side}_angular_acceleration_h0_{axis}_rad_s2" for axis in "xyz"),
            *(f"{side}_orientation_h0_{axis}_rad" for axis in ("roll", "pitch", "yaw")),
            f"{side}_bottle_z_tilt_from_h0_vertical_deg",
            f"{side}_upright_alignment",
        ])
    with (output_dir / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, task_s in enumerate(data["task_elapsed_s"]):
            row = {"task_elapsed_s": task_s}
            for axis_index, axis in enumerate("xyz"):
                row[f"torso_linear_acceleration_h0_{axis}_m_s2"] = data[
                    "torso_linear_acceleration_h0_m_s2"
                ][index, axis_index]
                row[f"torso_angular_velocity_h0_{axis}_rad_s"] = data[
                    "torso_angular_velocity_h0_rad_s"
                ][index, axis_index]
                row[f"torso_angular_acceleration_h0_{axis}_rad_s2"] = data[
                    "torso_angular_acceleration_h0_rad_s2"
                ][index, axis_index]
            for axis_index, axis in enumerate(("roll", "pitch", "yaw")):
                row[f"torso_orientation_h0_{axis}_rad"] = data[
                    "torso_orientation_h0_rpy_rad"
                ][index, axis_index]
            for side in ("left", "right"):
                for axis_index, axis in enumerate("xyz"):
                    row[f"{side}_linear_acceleration_h0_{axis}_m_s2"] = data[
                        f"{side}_linear_acceleration_h0_m_s2"
                    ][index, axis_index]
                    row[f"{side}_angular_acceleration_h0_{axis}_rad_s2"] = data[
                        f"{side}_angular_acceleration_h0_rad_s2"
                    ][index, axis_index]
                for axis_index, axis in enumerate(("roll", "pitch", "yaw")):
                    row[f"{side}_orientation_h0_{axis}_rad"] = data[
                        f"{side}_orientation_h0_rpy_rad"
                    ][index, axis_index]
                row[f"{side}_bottle_z_tilt_from_h0_vertical_deg"] = data[
                    f"{side}_tilt_deg"
                ][index]
                row[f"{side}_upright_alignment"] = data[
                    f"{side}_upright_alignment"
                ][index]
            writer.writerow(row)

    time_axis = data["task_elapsed_s"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
        axes[0].plot(time_axis, np.linalg.norm(data[f"{side}_linear_acceleration_h0_m_s2"], axis=1),
                     label=side, color=color, linewidth=1.0)
        axes[1].plot(time_axis, np.linalg.norm(data[f"{side}_angular_acceleration_h0_rad_s2"], axis=1),
                     label=side, color=color, linewidth=1.0)
        axes[2].plot(time_axis, data[f"{side}_tilt_deg"], label=side, color=color, linewidth=1.0)
    labels = ("endpoint |a| in H0 [m/s^2]", "endpoint |alpha| in H0 [rad/s^2]",
              "bottle-Z tilt from H0 vertical [deg]")
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.axvspan(WALK_START_S, RELEASE_START_S, color="green", alpha=0.06,
                     label="primary [5,18)" if axis is axes[0] else None)
        axis.axvline(WALK_STOP_S, color="black", linestyle="--", linewidth=0.8)
    axes[0].legend()
    axes[-1].set_xlabel("task time [s]")
    fig.suptitle("Hardware PID bottle-center metrics in fixed H0")
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_metrics_h0.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=200.0)
    parser.add_argument("--filter-window-s", type=float, default=0.105)
    args = parser.parse_args()
    if not args.raw_jsonl.is_file():
        parser.error(f"input does not exist: {args.raw_jsonl}")
    if not 100.0 <= args.sample_hz <= 500.0:
        parser.error("sample-hz must be in [100,500]")
    if not 0.05 <= args.filter_window_s <= 0.25:
        parser.error("filter-window-s must be in [0.05,0.25]")
    data, summary = analyze(args.raw_jsonl, args.sample_hz, args.filter_window_s)
    write_outputs(data, summary, args.output_dir)
    primary = summary["windows"]["primary_walk_start_through_stop_settle_end"]
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "primary_interval_s": primary["interval_s"],
        "primary_samples": primary["metrics"]["samples"],
        "left": primary["metrics"]["left"],
        "right": primary["metrics"]["right"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
