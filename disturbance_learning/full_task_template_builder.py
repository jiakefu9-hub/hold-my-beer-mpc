"""Pure offline construction and validation for the T1 full-task template.

This module deliberately has no MPC or predictor dependency.  It consumes only
the strict pre-step episodes produced by ``full_task_fixed_pd_collector`` and
builds one absolute-task-time template on the frozen 6 ms anchor grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from disturbance_learning.full_task_protocol import (
    FullTaskCausalHeadingFrame,
    FullTaskContinuousHeadingFrame,
    FullTaskProtocol,
    is_valid_rotation_batch,
    recompute_continuous_heading_frames,
)
from disturbance_learning.full_task_recording import validate_full_task_raw
from disturbance_learning.full_task_template_asset import (
    TEMPLATE_SCHEMA_VERSION,
    TEMPLATE_SCHEMA_VERSION_V2,
    VECTOR_NAMES,
    sha256_file,
    validate_full_task_template,
)
from disturbance_model_new_heading.heading_template_utils import (
    markley_quaternion_mean_wxyz,
    quaternion_wxyz_to_rotmat,
    rotmat_to_quaternion_wxyz,
)


FIXED_PD_RAW_EXTENSION_VERSION = "full_task_fixed_pd_extension_v1"
FIXED_PD_RAW_EXTENSION_VERSION_V2 = "full_task_fixed_pd_extension_v2"


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def portable_asset(path: Path, repo_dir: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        portable = resolved.relative_to(repo_dir.resolve()).as_posix()
    except ValueError:
        portable = str(resolved)
    return {
        "path": portable,
        "absolute_path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def wrapped_angle_difference(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(np.asarray(lhs) - np.asarray(rhs)), np.cos(np.asarray(lhs) - np.asarray(rhs)))


def rotation_geodesic_error(reference: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """Return proper SO(3) geodesic angles for broadcastable rotations."""
    relative = np.swapaxes(reference, -1, -2) @ measured
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def _require_fixed_pd_episode(raw: dict[str, np.ndarray], protocol: FullTaskProtocol) -> None:
    validate_full_task_raw(raw, protocol, require_complete=True)
    required = (
        "torso_linear_acceleration_world_used",
        "torso_angular_velocity_world",
        "torso_angular_acceleration_world_used",
        "torso_rotation_world",
        "causal_h_rotation_world",
        "right_arm_pd_requested_tau",
        "right_arm_pd_saturated",
        "causal_h_concentration",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"fixed-PD raw episode lacks fields: {missing}")
    if not np.all(np.asarray(raw["mpc_diagnostics_valid"], dtype=bool) == False):
        raise ValueError("pure T1 raw must not contain MPC diagnostics")
    if not np.all(np.asarray(raw["runtime_mapping_safety_fallback_used"], dtype=bool) == False):
        raise ValueError("pure T1 raw must not call DDQ-to-torque mapping")
    if not np.all(is_valid_rotation_batch(raw["torso_rotation_world"])):
        raise ValueError("raw torso rotations are not valid SO(3)")
    heading_version = str(
        np.asarray(
            raw.get(
                "heading_frame_version",
                np.array(FullTaskCausalHeadingFrame.DEFINITION_VERSION),
            )
        ).item()
    )
    if heading_version == FullTaskContinuousHeadingFrame.DEFINITION_VERSION:
        anchors = np.asarray(raw["mpc_anchor"], dtype=bool)
        replay = recompute_continuous_heading_frames(
            np.asarray(raw["task_time"], dtype=np.float64)[anchors],
            np.asarray(raw["torso_rotation_world"], dtype=np.float64)[anchors],
            protocol,
        )
        expected = (
            np.asarray(raw["causal_h_yaw_world"], dtype=np.float64)[anchors],
            np.asarray(raw["causal_h_rotation_world"], dtype=np.float64)[anchors],
            np.asarray(raw["causal_h_concentration"], dtype=np.float64)[anchors],
            np.asarray(raw["causal_h_source_code"], dtype=np.int64)[anchors],
        )
        for name, actual, recorded in zip(
            ("yaw", "rotation", "concentration", "source_code"), replay, expected
        ):
            if not np.allclose(actual, recorded, atol=1e-12, rtol=0.0):
                raise ValueError(f"continuous-H raw replay mismatch in {name}")
    elif heading_version != FullTaskCausalHeadingFrame.DEFINITION_VERSION:
        raise ValueError(f"unsupported raw heading frame version: {heading_version}")


def causal_h_metrics(raw: dict[str, np.ndarray]) -> dict[str, Any]:
    """Report only causal-H continuity; this function never filters H."""
    anchors = np.flatnonzero(np.asarray(raw["mpc_anchor"], dtype=bool))
    yaw = np.asarray(raw["causal_h_yaw_world"], dtype=np.float64)[anchors]
    concentration = np.asarray(raw["causal_h_concentration"], dtype=np.float64)[anchors]
    cycles = np.asarray(raw["gait_cycle_index"], dtype=np.int64)[anchors]
    adjacent = np.abs(wrapped_angle_difference(yaw[1:], yaw[:-1]))
    boundaries = np.flatnonzero(cycles[1:] != cycles[:-1]) + 1
    boundary_jumps = adjacent[boundaries - 1] if len(boundaries) else np.empty(0)
    return {
        "anchor_count": int(len(anchors)),
        "max_adjacent_h_yaw_jump_rad": float(np.max(adjacent)) if len(adjacent) else 0.0,
        "min_circular_concentration": float(np.min(concentration)) if len(concentration) else np.nan,
        "cycle_boundary_anchor_indices": boundaries.astype(int).tolist(),
        "cycle_boundary_task_times": np.asarray(raw["task_time"])[anchors[boundaries]].astype(float).tolist(),
        "cycle_boundary_h_yaw_jump_rad": boundary_jumps.astype(float).tolist(),
        "max_cycle_boundary_h_yaw_jump_rad": float(np.max(boundary_jumps)) if len(boundary_jumps) else 0.0,
        "h_filtering": "none; values are the shared causal circular means",
    }


def _episode_heading_windows(raw: dict[str, np.ndarray], protocol: FullTaskProtocol) -> dict[str, np.ndarray]:
    """Extract anchor-frozen H-frame node/interval data from one raw episode."""
    _require_fixed_pd_episode(raw, protocol)
    anchor_count = protocol.headline_anchor_count
    anchors = np.arange(anchor_count, dtype=np.int64) * protocol.mpc_stride
    if not np.all(np.asarray(raw["mpc_anchor"], dtype=bool)[anchors]):
        raise ValueError("headline anchor grid is not present in raw episode")

    nodes = {name: np.empty((anchor_count, protocol.horizon + 1, 3), dtype=np.float64) for name in VECTOR_NAMES}
    intervals = {name: np.empty((anchor_count, protocol.horizon, 3), dtype=np.float64) for name in VECTOR_NAMES}
    node_rotation = np.empty((anchor_count, protocol.horizon + 1, 3, 3), dtype=np.float64)
    interval_rotation = np.empty((anchor_count, protocol.horizon, 3, 3), dtype=np.float64)
    source_vectors = {
        "acceleration": np.asarray(raw["torso_linear_acceleration_world_used"], dtype=np.float64),
        "angular_velocity": np.asarray(raw["torso_angular_velocity_world"], dtype=np.float64),
        "angular_acceleration": np.asarray(raw["torso_angular_acceleration_world_used"], dtype=np.float64),
    }
    source_rotation = np.asarray(raw["torso_rotation_world"], dtype=np.float64)
    h_rotation = np.asarray(raw["causal_h_rotation_world"], dtype=np.float64)

    for anchor_index, raw_anchor in enumerate(anchors):
        windows = protocol.future_window_sample_indices(anchor_index)
        rotation_heading_world = h_rotation[raw_anchor].T
        node_indices = windows["node"]
        for name, source in source_vectors.items():
            nodes[name][anchor_index] = source[node_indices] @ rotation_heading_world.T
        node_rotation[anchor_index] = rotation_heading_world @ source_rotation[node_indices]
        for interval_index, interval_start in enumerate(windows["interval_start"]):
            sample_indices = np.arange(interval_start, interval_start + protocol.mpc_stride)
            for name, source in source_vectors.items():
                intervals[name][anchor_index, interval_index] = np.mean(
                    source[sample_indices] @ rotation_heading_world.T, axis=0
                )
            sample_quaternion = rotmat_to_quaternion_wxyz(
                rotation_heading_world @ source_rotation[sample_indices]
            )
            interval_rotation[anchor_index, interval_index] = quaternion_wxyz_to_rotmat(
                markley_quaternion_mean_wxyz(sample_quaternion)
            )
    if not np.all(is_valid_rotation_batch(node_rotation)) or not np.all(is_valid_rotation_batch(interval_rotation)):
        raise ValueError("anchor-frozen H transform produced an invalid rotation")
    return {
        **{f"nodes_{name}": value for name, value in nodes.items()},
        **{f"intervals_{name}": value for name, value in intervals.items()},
        "nodes_rotation_heading": node_rotation,
        "intervals_rotation_heading": interval_rotation,
    }


def _mean_rotations(rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean [episode,...,3,3] rotations by Markley, retaining dispersion."""
    if rotations.ndim < 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotation stack must be [episode,...,3,3]")
    episode_count = rotations.shape[0]
    flat = rotations.reshape(episode_count, -1, 3, 3)
    quaternions = rotmat_to_quaternion_wxyz(flat.reshape(-1, 3, 3)).reshape(
        episode_count, -1, 4
    )
    mean_quaternion = np.empty((flat.shape[1], 4), dtype=np.float64)
    dispersion = np.empty(flat.shape[1], dtype=np.float64)
    for index in range(flat.shape[1]):
        mean_quaternion[index] = markley_quaternion_mean_wxyz(quaternions[:, index])
        dots = np.clip(np.abs(quaternions[:, index] @ mean_quaternion[index]), 0.0, 1.0)
        dispersion[index] = float(np.sqrt(np.mean((2.0 * np.arccos(dots)) ** 2)))
    shape = rotations.shape[1:-2]
    mean_rotation = quaternion_wxyz_to_rotmat(mean_quaternion).reshape(*shape, 3, 3)
    return mean_rotation, mean_quaternion.reshape(*shape, 4), dispersion.reshape(shape)


def build_full_task_template(
    raw_episodes: Iterable[dict[str, np.ndarray]],
    episode_ids: Iterable[str],
    protocol: FullTaskProtocol,
    *,
    template_schema_version: str = TEMPLATE_SCHEMA_VERSION,
) -> dict[str, np.ndarray]:
    """Average full future windows over build episodes on absolute task time."""
    raws = list(raw_episodes)
    identifiers = [str(value) for value in episode_ids]
    if len(raws) < 2 or len(raws) != len(identifiers):
        raise ValueError("template requires at least two named build episodes")
    supported_schemas = {
        TEMPLATE_SCHEMA_VERSION,
        TEMPLATE_SCHEMA_VERSION_V2,
    }
    if template_schema_version not in supported_schemas:
        raise ValueError(f"unsupported full-task template schema: {template_schema_version}")
    heading_versions = {
        str(
            np.asarray(
                raw.get(
                    "heading_frame_version",
                    np.array(FullTaskCausalHeadingFrame.DEFINITION_VERSION),
                )
            ).item()
        )
        for raw in raws
    }
    if len(heading_versions) != 1:
        raise ValueError("build episodes use different heading-frame definitions")
    heading_version = next(iter(heading_versions))
    expected_heading = (
        FullTaskContinuousHeadingFrame.DEFINITION_VERSION
        if template_schema_version == TEMPLATE_SCHEMA_VERSION_V2
        else FullTaskCausalHeadingFrame.DEFINITION_VERSION
    )
    if heading_version != expected_heading:
        raise ValueError(
            f"{template_schema_version} requires heading frame {expected_heading}, got {heading_version}"
        )
    windows = [_episode_heading_windows(raw, protocol) for raw in raws]
    template: dict[str, np.ndarray] = {
        "template_schema_version": np.array(template_schema_version),
        "protocol_version": np.array(protocol.protocol_version),
        "raw_schema_extension_version": np.array(
            FIXED_PD_RAW_EXTENSION_VERSION_V2
            if template_schema_version == TEMPLATE_SCHEMA_VERSION_V2
            else FIXED_PD_RAW_EXTENSION_VERSION
        ),
        "heading_frame_version": np.array(heading_version),
        "frame_name": np.array("anchor_frozen_causal_heading"),
        "physics_dt": np.array(protocol.physics_dt),
        "anchor_dt": np.array(protocol.mpc_dt),
        "horizon": np.array(protocol.horizon, dtype=np.int64),
        "anchor_task_time": protocol.headline_anchor_times.copy(),
        "build_episode_count": np.array(len(raws), dtype=np.int64),
        "build_episode_ids": np.asarray(identifiers),
        "node0_online_policy": np.array("always_replace_with_current_measurement"),
        "smoothing": np.array("none"),
        "orientation_average_method": np.array("markley_quaternion_mean"),
        "interval_orientation_definition": np.array("Markley_mean_of_three_2ms_samples_within_each_6ms_interval"),
    }
    for name in VECTOR_NAMES:
        node_stack = np.stack([value[f"nodes_{name}"] for value in windows])
        interval_stack = np.stack([value[f"intervals_{name}"] for value in windows])
        template[f"nodes_{name}_mean"] = np.mean(node_stack, axis=0)
        template[f"nodes_{name}_std"] = np.std(node_stack, axis=0)
        template[f"intervals_{name}_mean"] = np.mean(interval_stack, axis=0)
        template[f"intervals_{name}_std"] = np.std(interval_stack, axis=0)
    for prefix in ("nodes", "intervals"):
        stack = np.stack([value[f"{prefix}_rotation_heading"] for value in windows])
        mean_rotation, mean_quaternion, dispersion = _mean_rotations(stack)
        template[f"{prefix}_rotation_heading_mean"] = mean_rotation
        template[f"{prefix}_quaternion_heading_mean_wxyz"] = mean_quaternion
        template[f"{prefix}_orientation_dispersion_rad"] = dispersion
    validate_full_task_template(template, protocol)
    return template


def evaluate_heldout_template(
    template: dict[str, np.ndarray],
    raw_episodes: Iterable[dict[str, np.ndarray]],
    episode_ids: Iterable[str],
    protocol: FullTaskProtocol,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    """Evaluate only held-out episodes; node zero is excluded by online policy."""
    validate_full_task_template(template, protocol)
    records: list[dict[str, Any]] = []
    windows_out: list[dict[str, np.ndarray]] = []
    for identifier, raw in zip(episode_ids, raw_episodes):
        windows = _episode_heading_windows(raw, protocol)
        windows_out.append(windows)
        item: dict[str, Any] = {"episode_id": str(identifier)}
        for location, begin in (("nodes", 1), ("intervals", 0)):
            for name in VECTOR_NAMES:
                actual = windows[f"{location}_{name}"][:, begin:]
                predicted = template[f"{location}_{name}_mean"][:, begin:]
                error = predicted - actual
                item[f"{location}_{name}_rmse"] = float(np.sqrt(np.mean(error ** 2)))
                item[f"{location}_{name}_mae"] = float(np.mean(np.abs(error)))
            angle = rotation_geodesic_error(
                template[f"{location}_rotation_heading_mean"][:, begin:],
                windows[f"{location}_rotation_heading"][:, begin:],
            )
            item[f"{location}_orientation_rmse_rad"] = float(np.sqrt(np.mean(angle ** 2)))
            item[f"{location}_orientation_mae_rad"] = float(np.mean(angle))
        records.append(item)
    aggregate: dict[str, Any] = {"episode_count": len(records), "per_episode": records}
    if records:
        for key in records[0]:
            if key == "episode_id":
                continue
            values = np.asarray([item[key] for item in records], dtype=np.float64)
            aggregate[f"mean_{key}"] = float(np.mean(values))
            aggregate[f"max_{key}"] = float(np.max(values))
    return aggregate, windows_out


def episode_summary(raw: dict[str, np.ndarray], protocol: FullTaskProtocol) -> dict[str, Any]:
    """Safety and contract report for a pure fixed-PD episode."""
    validation = validate_full_task_raw(raw, protocol, require_complete=True)
    time = np.asarray(raw["task_time"], dtype=np.float64)
    rotation = np.asarray(raw["torso_rotation_world"], dtype=np.float64)
    roll = np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = np.arcsin(np.clip(-rotation[:, 2, 0], -1.0, 1.0))
    xy = np.asarray(raw["torso_position_world"], dtype=np.float64)[:, :2]
    headline_index = int(round(protocol.headline_end / protocol.physics_dt))
    xy_headline = xy[: headline_index + 1]
    displacement = float(np.linalg.norm(xy_headline[-1] - xy_headline[0]))
    arc_length = float(np.sum(np.linalg.norm(np.diff(xy_headline, axis=0), axis=1)))
    pd_requested = np.asarray(raw["right_arm_pd_requested_tau"], dtype=np.float64)
    pd_error = np.asarray(raw["right_arm_pd_position_error"], dtype=np.float64)
    h = causal_h_metrics(raw)
    pd_saturation = int(np.count_nonzero(raw["right_arm_pd_saturated"]))
    fallen = bool(np.min(np.asarray(raw["torso_position_world"])[:, 2]) < 0.45 or max(np.max(np.abs(roll)), np.max(np.abs(pitch))) > np.deg2rad(60.0))
    direct = (
        np.allclose(raw["planned_command"][int(round(6.398 / protocol.physics_dt)), :2], raw["nominal_command"][:2])
        and np.allclose(raw["planned_command"][int(round(6.4 / protocol.physics_dt)), :2], 0.0)
        and np.allclose(raw["planned_command"][int(round(6.402 / protocol.physics_dt)), :2], 0.0)
    )
    # The heading loop is demonstrably enabled once it has published a finite
    # reference.  Its correction may happen to be exactly zero in a symmetric
    # nominal run, so command difference is not a valid enabled/disabled test.
    heading_enabled = bool(np.any(np.isfinite(np.asarray(raw["heading_reference_world"], dtype=np.float64))))
    finite = all(np.all(np.isfinite(np.asarray(raw[name]))) for name in (
        "torso_position_world", "torso_linear_acceleration_world_used", "torso_angular_velocity_world", "torso_angular_acceleration_world_used", "right_arm_pd_requested_tau"
    ))
    h_ok = h["min_circular_concentration"] >= 0.95 and h["max_adjacent_h_yaw_jump_rad"] <= 0.15 and h["max_cycle_boundary_h_yaw_jump_rad"] <= 0.15
    passed = bool(
        validation["strict_pre_step"] and validation["tail_complete"] and direct and heading_enabled
        and finite and not fallen and 2.8 <= displacement <= 3.6 and pd_saturation == 0 and h_ok
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "smoke_passed": passed,
        "right_arm_mode": "fixed_posture_pd",
        "xy_start_world_m": xy_headline[0].tolist(),
        "xy_at_headline_end_world_m": xy_headline[-1].tolist(),
        "xy_displacement_m": displacement,
        "xy_arc_length_m": arc_length,
        "distance_gate_m": [2.8, 3.6],
        "fallen": fallen,
        "nan_inf_free": finite,
        "max_abs_roll_rad": float(np.max(np.abs(roll))),
        "max_abs_pitch_rad": float(np.max(np.abs(pitch))),
        "torso_height_min_m": float(np.min(np.asarray(raw["torso_position_world"])[:, 2])),
        "direct_step_effective": bool(direct),
        "heading_enabled": heading_enabled,
        "right_arm_pd_saturation_sample_count": pd_saturation,
        "right_arm_pd_requested_tau_abs_max": np.max(np.abs(pd_requested), axis=0).tolist(),
        "right_arm_pd_position_error_abs_max_rad": np.max(np.abs(pd_error), axis=0).tolist(),
        "first_frame_command_visible_time_s": float(time[0]),
        "first_policy_command_consumed_time_s": float(validation["first_policy_command_consumed_time"]),
        **validation,
        **h,
    }


def _time_plot_with_band(ax, time: np.ndarray, values: np.ndarray, start: float, end: float, label: str, color: str) -> None:
    mask = (time >= start - 1e-12) & (time < end + 1e-12)
    selected = values[:, mask]
    mean = np.mean(selected, axis=0)
    std = np.std(selected, axis=0)
    for episode in selected:
        ax.plot(time[mask], episode, color=color, alpha=0.16, lw=0.55)
    ax.plot(time[mask], mean, color=color, lw=1.7, label=label)
    ax.fill_between(time[mask], mean - std, mean + std, color=color, alpha=0.22)
    ax.grid(True, alpha=0.3)


def save_template_plots(
    *,
    output_dir: Path,
    build_raws: list[dict[str, np.ndarray]],
    build_windows: list[dict[str, np.ndarray]],
    heldout_windows: list[dict[str, np.ndarray]],
    heldout_raws: list[dict[str, np.ndarray]],
    template: dict[str, np.ndarray],
    protocol: FullTaskProtocol,
) -> list[Path]:
    """Create the required diagnostic plots; no plot changes the template."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    raw_time = np.asarray(build_raws[0]["task_time"], dtype=np.float64)
    headline = raw_time <= protocol.headline_end + 1e-12
    anchor_time = np.asarray(template["anchor_task_time"], dtype=np.float64)

    path = output_dir / "build_xy_trajectories.png"
    fig, ax = plt.subplots(figsize=(9, 6))
    xy_stack = np.stack([raw["torso_position_world"][headline, :2] for raw in build_raws])
    for xy in xy_stack:
        ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", alpha=0.22, lw=0.8)
    mean_xy = np.mean(xy_stack, axis=0)
    std_xy = np.std(xy_stack, axis=0)
    ax.plot(mean_xy[:, 0], mean_xy[:, 1], color="black", lw=2.0, label="build mean")
    ax.fill_between(mean_xy[:, 0], mean_xy[:, 1] - std_xy[:, 1], mean_xy[:, 1] + std_xy[:, 1], color="tab:blue", alpha=0.16, label="y ±1 std")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("world x [m]"); ax.set_ylabel("world y [m]"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    node_norms = {
        "acc norm [m/s²]": np.stack([np.linalg.norm(item["nodes_acceleration"][:, 0], axis=-1) for item in build_windows]),
        "omega norm [rad/s]": np.stack([np.linalg.norm(item["nodes_angular_velocity"][:, 0], axis=-1) for item in build_windows]),
        "alpha norm [rad/s²]": np.stack([np.linalg.norm(item["nodes_angular_acceleration"][:, 0], axis=-1) for item in build_windows]),
    }
    for filename, start, end, title in (
        ("startup_0_2p4s_build_band.png", 0.0, 2.4, "build startup, anchor node 0"),
        ("stop_6p2_8p0s_build_band.png", 6.2, 8.0, "build direct-stop response, anchor node 0"),
    ):
        path = output_dir / filename
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for axis, (label, values), color in zip(axes, node_norms.items(), ("tab:blue", "tab:orange", "tab:green")):
            _time_plot_with_band(axis, anchor_time, values, start, end, label, color)
            axis.legend(loc="upper right")
        axes[-1].set_xlabel("task time [s]")
        fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    path = output_dir / "heldout_truth_vs_template.png"
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    template_norms = [
        np.linalg.norm(template["nodes_acceleration_mean"][:, 1], axis=-1),
        np.linalg.norm(template["nodes_angular_velocity_mean"][:, 1], axis=-1),
        np.linalg.norm(template["nodes_angular_acceleration_mean"][:, 1], axis=-1),
    ]
    labels = ("acc norm [m/s²]", "omega norm [rad/s]", "alpha norm [rad/s²]")
    for axis, prediction, label in zip(axes[:3], template_norms, labels):
        axis.plot(anchor_time, prediction, color="black", lw=2.0, label="template node +1")
        for item in heldout_windows:
            field = "nodes_acceleration" if label.startswith("acc") else "nodes_angular_velocity" if label.startswith("omega") else "nodes_angular_acceleration"
            axis.plot(anchor_time, np.linalg.norm(item[field][:, 1], axis=-1), alpha=0.45, lw=0.8)
        axis.set_ylabel(label); axis.grid(True, alpha=0.3); axis.legend(loc="upper right")
    for item in heldout_windows:
        axes[3].plot(anchor_time, rotation_geodesic_error(template["nodes_rotation_heading_mean"][:, 1], item["nodes_rotation_heading"][:, 1]), alpha=0.6, lw=0.8)
    axes[3].set_ylabel("node +1 SO(3) error [rad]"); axes[3].set_xlabel("task time [s]"); axes[3].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    path = output_dir / "initial_perturbations_and_diversity.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    q_offsets = np.stack([raw["initial_lower_q_offset_rad"] for raw in build_raws])
    dq_offsets = np.stack([raw["initial_lower_dq_rad_s"] for raw in build_raws])
    axes[0, 0].imshow(q_offsets, aspect="auto", cmap="coolwarm"); axes[0, 0].set_title("build initial lower q offsets [rad]")
    axes[1, 0].imshow(dq_offsets, aspect="auto", cmap="coolwarm"); axes[1, 0].set_title("build initial lower dq [rad/s]")
    displacement = [np.linalg.norm(raw["torso_position_world"][int(round(protocol.headline_end / protocol.physics_dt)), :2] - raw["torso_position_world"][0, :2]) for raw in build_raws]
    arc = [np.sum(np.linalg.norm(np.diff(raw["torso_position_world"][:int(round(protocol.headline_end / protocol.physics_dt)) + 1, :2], axis=0), axis=1)) for raw in build_raws]
    axes[0, 1].bar(np.arange(len(displacement)), displacement); axes[0, 1].set_title("XY displacement [m]")
    axes[1, 1].bar(np.arange(len(arc)), arc); axes[1, 1].set_title("XY arc length [m]")
    for axis in axes.flat: axis.set_xlabel("build episode index")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    path = output_dir / "causal_h_metrics.png"
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    # The cycle transition is represented by the first *available 6 ms anchor*
    # after a 0.8 s boundary (e.g. 0.804 s), not by an invented off-grid tick.
    boundary_times = causal_h_metrics(build_raws[0])["cycle_boundary_task_times"]
    for raw in build_raws + heldout_raws:
        anchors = np.asarray(raw["mpc_anchor"], dtype=bool)
        time = np.asarray(raw["task_time"])[anchors]
        yaw = np.unwrap(np.asarray(raw["causal_h_yaw_world"])[anchors])
        concentration = np.asarray(raw["causal_h_concentration"])[anchors]
        jump = np.r_[0.0, np.abs(wrapped_angle_difference(yaw[1:], yaw[:-1]))]
        axes[0].plot(time, yaw, alpha=0.42, lw=0.7)
        axes[1].plot(time, concentration, alpha=0.42, lw=0.7)
        axes[2].plot(time, jump, alpha=0.42, lw=0.7)
    axes[0].set_ylabel("H yaw unwrapped [rad]"); axes[1].set_ylabel("concentration"); axes[2].set_ylabel("adjacent jump [rad]")
    axes[2].set_xlabel("task time [s]")
    for axis in axes:
        axis.axvline(protocol.stop_time, color="red", ls="--", lw=1.0)
        for boundary_time in boundary_times:
            axis.axvline(boundary_time, color="0.55", ls=":", lw=0.7)
        axis.grid(True, alpha=0.3)
    # Avoid Matplotlib's offset notation, which hides that concentration is
    # directly reported near one rather than after a hidden re-scaling.
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    return paths


def write_template_artifacts(
    *,
    output_dir: Path,
    repo_dir: Path,
    template: dict[str, np.ndarray],
    template_validation: dict[str, Any],
    build_episode_paths: list[Path],
    heldout_episode_paths: list[Path],
    collection_manifest: dict[str, Any],
    heldout_metrics: dict[str, Any],
    plot_paths: list[Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "full_task_template.npz"
    np.savez_compressed(template_path, **template)
    metrics_path = output_dir / "heldout_metrics.json"
    metrics_path.write_text(json.dumps(_json_value(heldout_metrics), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "template_schema_version": str(np.asarray(template["template_schema_version"]).item()),
        "template_validation": template_validation,
        "collection": collection_manifest,
        "build_raw_episodes": [portable_asset(path, repo_dir) for path in build_episode_paths],
        "heldout_raw_episodes": [portable_asset(path, repo_dir) for path in heldout_episode_paths],
        "template": portable_asset(template_path, repo_dir),
        "heldout_metrics": portable_asset(metrics_path, repo_dir),
        "plots": [portable_asset(path, repo_dir) for path in plot_paths],
        "scope": {
            "right_arm_mode": "fixed_posture_pd",
            "right_arm_mpc_called": False,
            "online_disturbance_predictor_called": False,
            "right_arm_process_called": False,
            "ddq_to_torque_mapping_called": False,
        },
    }
    manifest_path = output_dir / "full_task_template_manifest.json"
    manifest_path.write_text(json.dumps(_json_value(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"template": template_path, "manifest": manifest_path, "heldout_metrics": metrics_path}
