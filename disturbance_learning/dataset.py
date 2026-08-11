"""Build causal MPC-aligned disturbance-prediction training windows.

Raw signals are sampled immediately before each 2 ms ``mj_step``.  A training
sample is anchored on the 6 ms MPC grid.  Its history ends at the anchor, and
its nine labels use velocity differences across the following nine 6 ms
control intervals, exactly matching the existing template interval semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from disturbance_model_new_heading.heading_template_utils import rotation_z


DATASET_SCHEMA_VERSION = 1
HEADING_DEFINITION = (
    "previous_complete_gait_cycle_circular_mean_torso_yaw"
)
PRE_STEP_DEFINITION = "sampled_immediately_before_mj_step"
DEFAULT_HISTORY_STEPS = 34
DEFAULT_HORIZON = 9
DEFAULT_CONTROL_DT = 0.006

FEATURE_GROUPS = (
    ("torso_angular_velocity_heading", 3),
    ("torso_linear_acceleration_heading", 3),
    ("gravity_direction_torso", 3),
    ("lower_body_q", 12),
    ("lower_body_dq", 12),
    ("lower_body_policy_target", 12),
    ("runtime_command", 3),
    ("gait_phase_sin_cos", 2),
)

FEATURE_NAMES = tuple(
    [f"torso_angular_velocity_H_{axis}" for axis in "xyz"]
    + [f"torso_linear_acceleration_H_{axis}" for axis in "xyz"]
    + [f"gravity_direction_torso_{axis}" for axis in "xyz"]
    + [f"lower_body_q_{index}" for index in range(12)]
    + [f"lower_body_dq_{index}" for index in range(12)]
    + [f"lower_body_policy_target_{index}" for index in range(12)]
    + ["runtime_command_vx", "runtime_command_vy", "runtime_command_wz"]
    + ["gait_phase_sin", "gait_phase_cos"]
)

TARGET_NAMES = tuple(
    [f"torso_linear_acceleration_H_{axis}" for axis in "xyz"]
    + [f"torso_angular_acceleration_H_{axis}" for axis in "xyz"]
)

REQUIRED_RAW_ARRAYS = {
    "time": (),
    "physics_step_index": (),
    "torso_rotation_world": (3, 3),
    "torso_linear_velocity_world": (3,),
    "torso_linear_acceleration_world": (3,),
    "torso_angular_velocity_world": (3,),
    "torso_angular_acceleration_world": (3,),
    "gravity_direction_torso": (3,),
    "lower_body_q": (12,),
    "lower_body_dq": (12,),
    "lower_body_policy_target": (12,),
    "runtime_command": (3,),
    "gait_phase_sin_cos": (2,),
    "schedule_segment_id": (),
}


@dataclass(frozen=True)
class HeadingCycle:
    cycle_id: int
    yaw: float
    concentration: float
    sample_count: int
    end_time: float


def _scalar(data: Mapping[str, np.ndarray], name: str, cast):
    if name not in data:
        raise KeyError(f"raw data 缺少标量字段 {name!r}。")
    return cast(np.asarray(data[name]).item())


def _validate_raw(raw: Mapping[str, np.ndarray]) -> tuple[int, float, float]:
    missing = [name for name in REQUIRED_RAW_ARRAYS if name not in raw]
    if missing:
        raise KeyError(f"raw data 缺少字段: {missing}")

    time = np.asarray(raw["time"], dtype=np.float64)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("raw time 必须是一维且至少包含两个 pre-step 样本。")
    sample_count = len(time)
    for name, tail_shape in REQUIRED_RAW_ARRAYS.items():
        value = np.asarray(raw[name])
        expected_shape = (sample_count,) + tail_shape
        if value.shape != expected_shape:
            raise ValueError(
                f"raw {name} shape={value.shape}，期望 {expected_shape}。"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"raw {name} 包含 NaN 或 Inf。")

    raw_dt = _scalar(raw, "simulation_dt", float)
    gait_period = _scalar(raw, "gait_period", float)
    if raw_dt <= 0.0 or gait_period <= 0.0:
        raise ValueError("simulation_dt 和 gait_period 必须为正数。")
    expected_time = np.arange(sample_count, dtype=np.float64) * raw_dt
    if not np.allclose(time, expected_time, rtol=0.0, atol=1e-12):
        raise ValueError("raw pre-step time 必须从 0 开始并严格落在 2 ms 网格。")
    if not np.array_equal(
        np.asarray(raw["physics_step_index"], dtype=np.int64),
        np.arange(sample_count, dtype=np.int64),
    ):
        raise ValueError("raw physics_step_index 与 pre-step time 网格不一致。")
    expected_phase = np.mod(time / gait_period, 1.0)
    expected_phase_sin_cos = np.column_stack(
        (
            np.sin(2.0 * np.pi * expected_phase),
            np.cos(2.0 * np.pi * expected_phase),
        )
    )
    if not np.allclose(
        raw["gait_phase_sin_cos"],
        expected_phase_sin_cos,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("raw gait phase 与 pre-step time 不一致。")
    if _scalar(raw, "sample_timing", str) != PRE_STEP_DEFINITION:
        raise ValueError("raw data 不是声明的 mj_step 前采样语义。")

    rotations = np.asarray(raw["torso_rotation_world"], dtype=np.float64)
    orthogonality_error = np.max(
        np.linalg.norm(
            np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3),
            axis=(1, 2),
        )
    )
    determinant_error = np.max(np.abs(np.linalg.det(rotations) - 1.0))
    if orthogonality_error > 1e-6 or determinant_error > 1e-6:
        raise ValueError("raw torso_rotation_world 包含无效旋转矩阵。")
    return sample_count, raw_dt, gait_period


def _complete_heading_cycles(
    time: np.ndarray,
    rotations_world: np.ndarray,
    raw_dt: float,
    gait_period: float,
) -> dict[int, HeadingCycle]:
    cycle_ids = np.floor(time / gait_period + 1e-12).astype(np.int64)
    yaw = np.arctan2(rotations_world[:, 1, 0], rotations_world[:, 0, 0])
    expected_samples = int(round(gait_period / raw_dt))
    cycles: dict[int, HeadingCycle] = {}
    for cycle_id in np.unique(cycle_ids):
        mask = cycle_ids == cycle_id
        indices = np.flatnonzero(mask)
        start_time = float(cycle_id) * gait_period
        end_time = start_time + gait_period
        complete = (
            len(indices) >= int(np.floor(0.9 * expected_samples))
            and time[indices[0]] <= start_time + raw_dt + 1e-12
            and time[indices[-1]] >= end_time - raw_dt - 1e-12
        )
        if not complete:
            continue
        sine = float(np.sum(np.sin(yaw[mask])))
        cosine = float(np.sum(np.cos(yaw[mask])))
        concentration = float(np.hypot(sine, cosine) / len(indices))
        if concentration < 1e-6:
            continue
        cycles[int(cycle_id)] = HeadingCycle(
            cycle_id=int(cycle_id),
            yaw=float(np.arctan2(sine, cosine)),
            concentration=concentration,
            sample_count=len(indices),
            end_time=end_time,
        )
    return cycles


def _feature_slices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = []
    starts = []
    stops = []
    offset = 0
    for name, width in FEATURE_GROUPS:
        names.append(name)
        starts.append(offset)
        offset += width
        stops.append(offset)
    if offset != len(FEATURE_NAMES):
        raise AssertionError("feature group 与 feature name 数量不一致。")
    return (
        np.asarray(names),
        np.asarray(starts, dtype=np.int64),
        np.asarray(stops, dtype=np.int64),
    )


def build_supervised_windows(
    raw: Mapping[str, np.ndarray],
    *,
    history_steps: int = DEFAULT_HISTORY_STEPS,
    horizon: int = DEFAULT_HORIZON,
    control_dt: float = DEFAULT_CONTROL_DT,
) -> dict[str, np.ndarray]:
    """Convert one causal raw episode into fixed history/target windows."""
    sample_count, raw_dt, gait_period = _validate_raw(raw)
    history_steps = int(history_steps)
    horizon = int(horizon)
    control_dt = float(control_dt)
    if history_steps < 2 or horizon < 1 or control_dt <= 0.0:
        raise ValueError("history_steps>=2、horizon>=1、control_dt>0。")
    control_stride = int(round(control_dt / raw_dt))
    if control_stride < 1 or not np.isclose(
        control_stride * raw_dt, control_dt, rtol=0.0, atol=1e-12
    ):
        raise ValueError("control_dt 必须是 raw simulation_dt 的整数倍。")

    time = np.asarray(raw["time"], dtype=np.float64)
    rotations_world = np.asarray(
        raw["torso_rotation_world"], dtype=np.float64
    )
    heading_cycles = _complete_heading_cycles(
        time, rotations_world, raw_dt, gait_period
    )

    history_offsets = control_stride * np.arange(
        -(history_steps - 1), 1, dtype=np.int64
    )
    interval_offsets = (
        control_stride * np.arange(horizon, dtype=np.int64)[:, None]
        + np.asarray((0, control_stride), dtype=np.int64)[None, :]
    )

    anchors = []
    histories = []
    target_indices_all = []
    history_features = []
    history_omega_heading = []
    history_acc_heading = []
    target_values = []
    target_acc_heading = []
    target_alpha_heading = []
    heading_yaws = []
    heading_concentrations = []
    heading_source_cycles = []
    heading_source_end_times = []
    anchor_segments = []

    raw_acc_world = np.asarray(
        raw["torso_linear_acceleration_world"], dtype=np.float64
    )
    raw_velocity_world = np.asarray(
        raw["torso_linear_velocity_world"], dtype=np.float64
    )
    raw_omega_world = np.asarray(
        raw["torso_angular_velocity_world"], dtype=np.float64
    )
    anchor_candidates = np.arange(0, sample_count, control_stride)
    for anchor in anchor_candidates:
        history_indices = anchor + history_offsets
        target_indices = anchor + interval_offsets
        if history_indices[0] < 0 or target_indices[-1, -1] >= sample_count:
            continue

        anchor_cycle = int(
            np.floor(time[anchor] / gait_period + 1e-12)
        )
        source_cycle = anchor_cycle - 1
        heading = heading_cycles.get(source_cycle)
        if heading is None or heading.end_time > time[anchor] + 1e-12:
            continue

        rotation_heading_world = rotation_z(-heading.yaw)
        omega_heading = np.einsum(
            "ij,nj->ni",
            rotation_heading_world,
            raw_omega_world[history_indices],
        )
        acc_heading = np.einsum(
            "ij,nj->ni",
            rotation_heading_world,
            raw_acc_world[history_indices],
        )
        feature = np.concatenate(
            [
                omega_heading,
                acc_heading,
                np.asarray(raw["gravity_direction_torso"])[history_indices],
                np.asarray(raw["lower_body_q"])[history_indices],
                np.asarray(raw["lower_body_dq"])[history_indices],
                np.asarray(raw["lower_body_policy_target"])[history_indices],
                np.asarray(raw["runtime_command"])[history_indices],
                np.asarray(raw["gait_phase_sin_cos"])[history_indices],
            ],
            axis=1,
        )

        interval_acc_world = (
            raw_velocity_world[target_indices[:, 1]]
            - raw_velocity_world[target_indices[:, 0]]
        ) / control_dt
        interval_alpha_world = (
            raw_omega_world[target_indices[:, 1]]
            - raw_omega_world[target_indices[:, 0]]
        ) / control_dt
        interval_acc_heading = np.einsum(
            "ij,kj->ki",
            rotation_heading_world,
            interval_acc_world,
        )
        interval_alpha_heading = np.einsum(
            "ij,kj->ki",
            rotation_heading_world,
            interval_alpha_world,
        )
        target = np.concatenate(
            [interval_acc_heading, interval_alpha_heading], axis=1
        )

        anchors.append(anchor)
        histories.append(history_indices)
        target_indices_all.append(target_indices)
        history_features.append(feature)
        history_omega_heading.append(omega_heading)
        history_acc_heading.append(acc_heading)
        target_values.append(target)
        target_acc_heading.append(interval_acc_heading)
        target_alpha_heading.append(interval_alpha_heading)
        heading_yaws.append(heading.yaw)
        heading_concentrations.append(heading.concentration)
        heading_source_cycles.append(source_cycle)
        heading_source_end_times.append(heading.end_time)
        anchor_segments.append(int(raw["schedule_segment_id"][anchor]))

    if not anchors:
        raise ValueError(
            "没有可用窗口：至少需要一个完整 heading 周期、history 和 54 ms future。"
        )

    anchor_indices = np.asarray(anchors, dtype=np.int64)
    history_indices = np.asarray(histories, dtype=np.int64)
    target_raw_indices = np.asarray(target_indices_all, dtype=np.int64)
    history_group_names, feature_starts, feature_stops = _feature_slices()
    schedule_names = np.asarray(raw.get("schedule_segment_names", []))
    required_schedule_ids = np.asarray(
        raw.get("required_schedule_segment_ids", []), dtype=np.int64
    )
    source_episode_id = str(np.asarray(raw.get("episode_id", "episode")).item())

    return {
        "dataset_schema_version": np.array(DATASET_SCHEMA_VERSION),
        "source_episode_id": np.array(source_episode_id),
        "sample_episode_id": np.full(len(anchors), source_episode_id),
        "sample_timing": np.array(PRE_STEP_DEFINITION),
        "heading_definition": np.array(HEADING_DEFINITION),
        "heading_usage": np.array(
            "anchor_previous_cycle_heading_fixed_for_history_and_target"
        ),
        "history_definition": np.array(
            "34_causal_mpc_grid_pre_step_samples_ending_at_anchor"
        ),
        "target_definition": np.array(
            "nine_future_6ms_interval_accelerations_from_endpoint_velocity_differences"
        ),
        "simulation_dt": np.array(raw_dt, dtype=np.float64),
        "control_dt": np.array(control_dt, dtype=np.float64),
        "gait_period": np.array(gait_period, dtype=np.float64),
        "history_steps": np.array(history_steps, dtype=np.int64),
        "history_timestamp_span": np.array(
            (history_steps - 1) * control_dt, dtype=np.float64
        ),
        "history_nominal_window": np.array(
            history_steps * control_dt, dtype=np.float64
        ),
        "horizon": np.array(horizon, dtype=np.int64),
        "prediction_duration": np.array(
            horizon * control_dt, dtype=np.float64
        ),
        "feature_names": np.asarray(FEATURE_NAMES),
        "feature_group_names": history_group_names,
        "feature_group_start": feature_starts,
        "feature_group_stop": feature_stops,
        "target_names": np.asarray(TARGET_NAMES),
        "anchor_raw_index": anchor_indices,
        "anchor_time": time[anchor_indices],
        "history_raw_indices": history_indices,
        "history_time": time[history_indices],
        "target_raw_indices": target_raw_indices,
        "target_sample_time": time[target_raw_indices],
        "target_interval_start_time": (
            time[anchor_indices, None]
            + control_dt * np.arange(horizon, dtype=np.float64)[None, :]
        ),
        "target_interval_end_time": (
            time[anchor_indices, None]
            + control_dt
            * (np.arange(horizon, dtype=np.float64)[None, :] + 1.0)
        ),
        "history": np.asarray(history_features, dtype=np.float32),
        "history_torso_angular_velocity_heading": np.asarray(
            history_omega_heading, dtype=np.float32
        ),
        "history_torso_linear_acceleration_heading": np.asarray(
            history_acc_heading, dtype=np.float32
        ),
        "target": np.asarray(target_values, dtype=np.float32),
        "target_torso_linear_acceleration_heading": np.asarray(
            target_acc_heading, dtype=np.float32
        ),
        "target_torso_angular_acceleration_heading": np.asarray(
            target_alpha_heading, dtype=np.float32
        ),
        "heading_yaw_world": np.asarray(heading_yaws, dtype=np.float64),
        "heading_concentration": np.asarray(
            heading_concentrations, dtype=np.float64
        ),
        "heading_source_cycle_id": np.asarray(
            heading_source_cycles, dtype=np.int64
        ),
        "heading_source_cycle_end_time": np.asarray(
            heading_source_end_times, dtype=np.float64
        ),
        "anchor_schedule_segment_id": np.asarray(
            anchor_segments, dtype=np.int64
        ),
        "schedule_segment_names": schedule_names,
        "required_schedule_segment_ids": required_schedule_ids,
    }


def validate_supervised_windows(
    dataset: Mapping[str, np.ndarray],
    raw: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Run timing, causality, H-frame and target reconstruction checks."""
    _, raw_dt, gait_period = _validate_raw(raw)
    history = np.asarray(dataset["history"])
    target = np.asarray(dataset["target"])
    anchor_time = np.asarray(dataset["anchor_time"], dtype=np.float64)
    history_time = np.asarray(dataset["history_time"], dtype=np.float64)
    target_sample_time = np.asarray(
        dataset["target_sample_time"], dtype=np.float64
    )
    target_start = np.asarray(
        dataset["target_interval_start_time"], dtype=np.float64
    )
    target_end = np.asarray(
        dataset["target_interval_end_time"], dtype=np.float64
    )
    history_indices = np.asarray(dataset["history_raw_indices"])
    target_indices = np.asarray(dataset["target_raw_indices"])
    anchor_indices = np.asarray(dataset["anchor_raw_index"])
    heading_source_end = np.asarray(
        dataset["heading_source_cycle_end_time"], dtype=np.float64
    )
    control_dt = float(np.asarray(dataset["control_dt"]).item())
    horizon = int(np.asarray(dataset["horizon"]).item())

    expected_history_shape = (
        len(anchor_time),
        int(np.asarray(dataset["history_steps"]).item()),
        len(FEATURE_NAMES),
    )
    expected_target_shape = (len(anchor_time), horizon, len(TARGET_NAMES))
    if history.shape != expected_history_shape:
        raise ValueError(
            f"history shape={history.shape}，期望 {expected_history_shape}。"
        )
    if target.shape != expected_target_shape:
        raise ValueError(
            f"target shape={target.shape}，期望 {expected_target_shape}。"
        )
    if not np.all(np.isfinite(history)) or not np.all(np.isfinite(target)):
        raise ValueError("history/target 包含 NaN 或 Inf。")

    history_future_count = int(
        np.count_nonzero(history_time > anchor_time[:, None] + 1e-12)
    )
    if history_future_count:
        raise ValueError("history 包含 anchor 之后的 future leakage。")
    if not np.array_equal(history_indices[:, -1], anchor_indices):
        raise ValueError("history 最后一个样本不是当前 anchor。")
    if np.any(heading_source_end > anchor_time + 1e-12):
        raise ValueError("H heading 使用了 anchor 之后的周期数据。")

    control_stride = int(round(control_dt / raw_dt))
    expected_target_indices = (
        anchor_indices[:, None, None]
        + control_stride * np.arange(horizon, dtype=np.int64)[None, :, None]
        + np.asarray((0, control_stride), dtype=np.int64)[None, None, :]
    )
    if not np.array_equal(target_indices, expected_target_indices):
        raise ValueError("target raw indices 与 9 x 6 ms 区间定义不一致。")
    if not np.allclose(target_sample_time[:, :, 0], target_start, atol=1e-12):
        raise ValueError("每个 target interval 的首样本不是区间起点。")
    if not np.allclose(target_sample_time[:, :, -1], target_end, atol=1e-12):
        raise ValueError("每个 target interval 的末样本不是区间终点。")

    rotations_heading_world = rotation_z(
        -np.asarray(dataset["heading_yaw_world"], dtype=np.float64)
    )
    anchor_cycles = np.floor(
        anchor_time / gait_period + 1e-12
    ).astype(np.int64)
    expected_source_cycles = anchor_cycles - 1
    actual_source_cycles = np.asarray(
        dataset["heading_source_cycle_id"], dtype=np.int64
    )
    if not np.array_equal(actual_source_cycles, expected_source_cycles):
        raise ValueError("H heading source 不是 anchor 的上一完整步态周期。")
    heading_cycles = _complete_heading_cycles(
        np.asarray(raw["time"], dtype=np.float64),
        np.asarray(raw["torso_rotation_world"], dtype=np.float64),
        raw_dt,
        gait_period,
    )
    expected_heading_yaw = np.asarray(
        [heading_cycles[int(cycle)].yaw for cycle in expected_source_cycles]
    )
    heading_error = np.arctan2(
        np.sin(
            np.asarray(dataset["heading_yaw_world"], dtype=np.float64)
            - expected_heading_yaw
        ),
        np.cos(
            np.asarray(dataset["heading_yaw_world"], dtype=np.float64)
            - expected_heading_yaw
        ),
    )
    heading_yaw_error = float(np.max(np.abs(heading_error)))
    if heading_yaw_error > 1e-12:
        raise ValueError("H heading yaw 与上一完整周期圆周均值不一致。")

    reconstructed_history_omega = np.einsum(
        "nij,nhj->nhi",
        rotations_heading_world,
        np.asarray(raw["torso_angular_velocity_world"], dtype=np.float64)[
            history_indices
        ],
    )
    reconstructed_history_acc = np.einsum(
        "nij,nhj->nhi",
        rotations_heading_world,
        np.asarray(raw["torso_linear_acceleration_world"], dtype=np.float64)[
            history_indices
        ],
    )
    history_omega_error = float(
        np.max(
            np.abs(
                reconstructed_history_omega
                - np.asarray(
                    dataset["history_torso_angular_velocity_heading"]
                )
            )
        )
    )
    history_acc_error = float(
        np.max(
            np.abs(
                reconstructed_history_acc
                - np.asarray(
                    dataset["history_torso_linear_acceleration_heading"]
                )
            )
        )
    )
    if max(history_omega_error, history_acc_error) > 2e-5:
        raise ValueError("history 的 W→H 旋转重建误差过大。")

    raw_velocity = np.asarray(
        raw["torso_linear_velocity_world"], dtype=np.float64
    )
    raw_omega = np.asarray(
        raw["torso_angular_velocity_world"], dtype=np.float64
    )
    interval_acc_world = (
        raw_velocity[target_indices[:, :, 1]]
        - raw_velocity[target_indices[:, :, 0]]
    ) / control_dt
    interval_alpha_world = (
        raw_omega[target_indices[:, :, 1]]
        - raw_omega[target_indices[:, :, 0]]
    ) / control_dt
    reconstructed_acc = np.einsum(
        "nij,nkj->nki", rotations_heading_world, interval_acc_world
    )
    reconstructed_alpha = np.einsum(
        "nij,nkj->nki", rotations_heading_world, interval_alpha_world
    )
    acc_error = float(
        np.max(
            np.abs(
                reconstructed_acc
                - np.asarray(
                    dataset["target_torso_linear_acceleration_heading"]
                )
            )
        )
    )
    alpha_error = float(
        np.max(
            np.abs(
                reconstructed_alpha
                - np.asarray(
                    dataset["target_torso_angular_acceleration_heading"]
                )
            )
        )
    )
    if max(acc_error, alpha_error) > 2e-5:
        raise ValueError("H-frame interval target 重建误差过大。")

    segment_ids, segment_counts = np.unique(
        np.asarray(dataset["anchor_schedule_segment_id"], dtype=np.int64),
        return_counts=True,
    )
    segment_count_map = {
        str(int(segment_id)): int(count)
        for segment_id, count in zip(segment_ids, segment_counts)
    }
    required_ids: Iterable[int] = np.asarray(
        dataset.get("required_schedule_segment_ids", []), dtype=np.int64
    )
    missing_segments = [
        int(segment_id)
        for segment_id in required_ids
        if str(int(segment_id)) not in segment_count_map
    ]
    if missing_segments:
        raise ValueError(f"dataset 缺少 command schedule 段: {missing_segments}")

    heading_source_margin = float(
        np.min(anchor_time - heading_source_end)
    )
    if abs(heading_source_margin) <= 1e-12:
        heading_source_margin = 0.0

    return {
        "sample_count": int(len(anchor_time)),
        "raw_sample_count": int(len(raw["time"])),
        "history_shape": list(history.shape),
        "target_shape": list(target.shape),
        "history_timestamp_span_s": float(
            np.max(history_time[:, -1] - history_time[:, 0])
        ),
        "prediction_horizon_s": float(horizon * control_dt),
        "history_future_leak_count": history_future_count,
        "max_history_time_minus_anchor_s": float(
            np.max(history_time - anchor_time[:, None])
        ),
        "min_target_time_minus_anchor_s": float(
            np.min(target_sample_time - anchor_time[:, None, None])
        ),
        "max_target_interval_end_minus_anchor_s": float(
            np.max(target_end - anchor_time[:, None])
        ),
        "min_heading_source_margin_s": heading_source_margin,
        "heading_yaw_reconstruction_max_error": heading_yaw_error,
        "history_omega_reconstruction_max_error": history_omega_error,
        "history_acc_reconstruction_max_error": history_acc_error,
        "target_acc_reconstruction_max_error": acc_error,
        "target_alpha_reconstruction_max_error": alpha_error,
        "finite_history": True,
        "finite_target": True,
        "schedule_segment_sample_counts": segment_count_map,
    }
