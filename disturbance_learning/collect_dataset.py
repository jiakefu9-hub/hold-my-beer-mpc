#!/usr/bin/env python3
"""Collect one causal, pre-step-aligned learned-disturbance episode."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from disturbance_learning.dataset import (
    DEFAULT_CONTROL_DT,
    DEFAULT_HISTORY_STEPS,
    DEFAULT_HORIZON,
    HEADING_DEFINITION,
    PRE_STEP_DEFINITION,
    build_supervised_windows,
    validate_supervised_windows,
)
from disturbance_learning.command_schedule import (
    GAIT_PERIOD,
    REQUIRED_SCHEDULE_SEGMENT_IDS,
    SCHEDULE_DURATION,
    SCHEDULE_SEGMENT_NAMES,
    command_schedule,
)
from sim_support import get_gravity_orientation, get_site_vel, pd_control


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class EpisodeProfile:
    start_command: np.ndarray
    changed_command: np.ndarray
    initial_lower_q_offset: np.ndarray
    initial_lower_dq: np.ndarray


def build_episode_profile(
    nominal_command: np.ndarray, seed: int
) -> EpisodeProfile:
    """Create one reproducible, bounded command/state variation."""
    rng = np.random.default_rng(int(seed))
    nominal = np.asarray(nominal_command, dtype=np.float64)
    start_command = np.array(
        [
            np.clip(nominal[0] * rng.uniform(0.75, 1.15), 0.30, 0.60),
            rng.uniform(-0.055, 0.055),
            np.clip(nominal[2] + rng.uniform(-0.025, 0.025), -0.05, 0.05),
        ],
        dtype=np.float64,
    )
    changed_command = np.array(
        [
            np.clip(start_command[0] * rng.uniform(0.45, 0.80), 0.15, 0.45),
            np.clip(start_command[1] + rng.uniform(-0.10, 0.10), -0.12, 0.12),
            rng.uniform(-0.06, 0.06),
        ],
        dtype=np.float64,
    )
    return EpisodeProfile(
        start_command=start_command,
        changed_command=changed_command,
        initial_lower_q_offset=rng.normal(0.0, 0.006, size=12),
        initial_lower_dq=rng.normal(0.0, 0.01, size=12),
    )


def _repo_path(configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path if path.is_absolute() else REPO_DIR / path


def _load_config(config_file: str) -> tuple[dict, Path]:
    requested = Path(config_file)
    config_path = (
        requested
        if requested.is_absolute()
        else REPO_DIR / "configs" / requested
    )
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file), config_path.resolve()


def _sensor_address(model: mujoco.MjModel, name: str) -> int:
    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, name
    )
    if sensor_id < 0:
        raise ValueError(f"MuJoCo model 缺少 sensor {name!r}。")
    if int(model.sensor_dim[sensor_id]) != 3:
        raise ValueError(f"sensor {name!r} 必须是三维。")
    return int(model.sensor_adr[sensor_id])


def _site_id(model: mujoco.MjModel, name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"MuJoCo model 缺少 site {name!r}。")
    return site_id


def collect_raw_episode(
    config: dict,
    *,
    config_path: Path,
    episode_id: str,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run the lower-body policy and capture every 2 ms pre-step state."""
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    if not np.isclose(simulation_dt, 0.002, atol=1e-12):
        raise ValueError("B1 interval labels currently require simulation_dt=0.002 s.")
    if control_decimation < 1:
        raise ValueError("control_decimation 必须为正整数。")

    policy_path = _repo_path(str(config["policy_path"]))
    xml_path = _repo_path(str(config["xml_path"]))
    if not policy_path.is_file() or not xml_path.is_file():
        raise FileNotFoundError(
            f"独立仓库资源缺失: policy={policy_path}, xml={xml_path}"
        )

    kps = np.asarray(config["kps"], dtype=np.float64)
    kds = np.asarray(config["kds"], dtype=np.float64)
    default_angles = np.asarray(config["default_angles"], dtype=np.float64)
    arm_waist_kps = np.asarray(config["arm_waist_kps"], dtype=np.float64)
    arm_waist_kds = np.asarray(config["arm_waist_kds"], dtype=np.float64)
    arm_waist_target = np.asarray(
        config["arm_waist_target"], dtype=np.float64
    )
    cmd_scale = np.asarray(config["cmd_scale"], dtype=np.float32)
    cmd_nominal = np.asarray(config["cmd_init"], dtype=np.float64)
    num_actions = int(config["num_actions"])
    num_obs = int(config["num_obs"])
    if num_actions != 12:
        raise ValueError("B1 lower-body schema currently requires 12 policy actions.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt
    profile = build_episode_profile(cmd_nominal, seed)
    data.qpos[7:19] += profile.initial_lower_q_offset
    data.qvel[6:18] = profile.initial_lower_dq
    mujoco.mj_forward(model, data)

    acceleration_adr = _sensor_address(
        model, "imu-torso-linear-acceleration"
    )
    imu_site_id = _site_id(model, "imu_in_torso")
    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    observation = np.zeros(num_obs, dtype=np.float32)
    active_schedule = command_schedule(
        0.0, profile.start_command, profile.changed_command
    )
    previous_omega_world = np.zeros(3, dtype=np.float64)

    raw_lists: dict[str, list] = {
        "time": [],
        "physics_step_index": [],
        "torso_rotation_world": [],
        "torso_linear_velocity_world": [],
        "torso_linear_acceleration_world": [],
        "torso_angular_velocity_world": [],
        "torso_angular_acceleration_world": [],
        "gravity_direction_torso": [],
        "lower_body_q": [],
        "lower_body_dq": [],
        "lower_body_policy_target": [],
        "runtime_command": [],
        "gait_phase_sin_cos": [],
        "schedule_segment_id": [],
    }

    total_steps = int(round(SCHEDULE_DURATION / simulation_dt))
    for step_index in range(total_steps):
        time_s = step_index * simulation_dt
        lower_q = data.qpos[7:19].copy()
        lower_dq = data.qvel[6:18].copy()
        data.ctrl[:12] = pd_control(
            target_dof_pos,
            lower_q,
            kps,
            np.zeros_like(kds),
            lower_dq,
            kds,
        )

        arm_waist_q = data.qpos[19:30]
        arm_waist_dq = data.qvel[18:29]
        data.ctrl[12:23] = pd_control(
            arm_waist_target,
            arm_waist_q,
            arm_waist_kps,
            np.zeros_like(arm_waist_kds),
            arm_waist_dq,
            arm_waist_kds,
        )

        linear_velocity_world, omega_world = get_site_vel(
            model, data, imu_site_id
        )
        rotation_world_torso = data.site_xmat[imu_site_id].reshape(3, 3).copy()
        specific_force_torso = data.sensordata[
            acceleration_adr : acceleration_adr + 3
        ].copy()
        linear_acceleration_world = (
            np.zeros(3, dtype=np.float64)
            if step_index == 0
            else rotation_world_torso @ specific_force_torso
            + model.opt.gravity
        )
        angular_acceleration_world = (
            np.zeros(3, dtype=np.float64)
            if step_index == 0
            else (omega_world - previous_omega_world) / simulation_dt
        )
        previous_omega_world = omega_world.copy()
        gravity_direction_torso = (
            rotation_world_torso.T
            @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        )
        phase = (time_s / GAIT_PERIOD) % 1.0

        raw_lists["time"].append(time_s)
        raw_lists["physics_step_index"].append(step_index)
        raw_lists["torso_rotation_world"].append(rotation_world_torso)
        raw_lists["torso_linear_velocity_world"].append(
            linear_velocity_world.copy()
        )
        raw_lists["torso_linear_acceleration_world"].append(
            linear_acceleration_world
        )
        raw_lists["torso_angular_velocity_world"].append(omega_world.copy())
        raw_lists["torso_angular_acceleration_world"].append(
            angular_acceleration_world
        )
        raw_lists["gravity_direction_torso"].append(gravity_direction_torso)
        raw_lists["lower_body_q"].append(lower_q)
        raw_lists["lower_body_dq"].append(lower_dq)
        raw_lists["lower_body_policy_target"].append(
            target_dof_pos.copy()
        )
        raw_lists["runtime_command"].append(active_schedule.command.copy())
        raw_lists["gait_phase_sin_cos"].append(
            np.array(
                [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)]
            )
        )
        raw_lists["schedule_segment_id"].append(active_schedule.segment_id)

        mujoco.mj_step(model, data)
        next_step = step_index + 1
        if next_step % control_decimation == 0:
            policy_time = next_step * simulation_dt
            active_schedule = command_schedule(
                policy_time,
                profile.start_command,
                profile.changed_command,
            )
            policy_q = data.qpos[7:19].copy()
            policy_dq = data.qvel[6:18].copy()
            policy_quaternion = data.qpos[3:7].copy()
            policy_omega = data.qvel[3:6].copy()
            policy_phase = (policy_time / GAIT_PERIOD) % 1.0

            observation[:3] = policy_omega * float(config["ang_vel_scale"])
            observation[3:6] = get_gravity_orientation(policy_quaternion)
            observation[6:9] = active_schedule.command * cmd_scale
            observation[9 : 9 + num_actions] = (
                policy_q - default_angles
            ) * float(config["dof_pos_scale"])
            observation[
                9 + num_actions : 9 + 2 * num_actions
            ] = policy_dq * float(config["dof_vel_scale"])
            observation[
                9 + 2 * num_actions : 9 + 3 * num_actions
            ] = action
            observation[
                9 + 3 * num_actions : 9 + 3 * num_actions + 2
            ] = np.array(
                [
                    np.sin(2.0 * np.pi * policy_phase),
                    np.cos(2.0 * np.pi * policy_phase),
                ],
                dtype=np.float32,
            )
            with torch.inference_mode():
                action = (
                    policy(torch.from_numpy(observation).unsqueeze(0))
                    .cpu()
                    .numpy()
                    .squeeze()
                )
            target_dof_pos = (
                action * float(config["action_scale"]) + default_angles
            )

    raw = {
        name: np.asarray(values)
        for name, values in raw_lists.items()
    }
    raw.update(
        {
            "raw_schema_version": np.array(1, dtype=np.int64),
            "episode_id": np.array(episode_id),
            "seed": np.array(seed, dtype=np.int64),
            "sample_timing": np.array(PRE_STEP_DEFINITION),
            "simulation_dt": np.array(simulation_dt, dtype=np.float64),
            "gait_period": np.array(GAIT_PERIOD, dtype=np.float64),
            "heading_definition": np.array(HEADING_DEFINITION),
            "config_path": np.array(str(config_path)),
            "policy_path": np.array(str(policy_path.resolve())),
            "xml_path": np.array(str(xml_path.resolve())),
            "upper_body_mode": np.array("fixed_joint_space_pd"),
            "command_update_dt": np.array(
                simulation_dt * control_decimation, dtype=np.float64
            ),
            "start_command": profile.start_command.copy(),
            "changed_command": profile.changed_command.copy(),
            "initial_lower_q_offset": profile.initial_lower_q_offset.copy(),
            "initial_lower_dq": profile.initial_lower_dq.copy(),
            "schedule_segment_names": SCHEDULE_SEGMENT_NAMES.copy(),
            "required_schedule_segment_ids": (
                REQUIRED_SCHEDULE_SEGMENT_IDS.copy()
            ),
        }
    )
    return raw


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Collect one B1 causal pre-step disturbance dataset"
    )
    parser.add_argument(
        "config_file", nargs="?", default="g1.yaml", help="configs/ 下配置名"
    )
    parser.add_argument("--episode-id", default=f"b1_{timestamp}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-prefix",
        default=str(DEFAULT_DATA_DIR / f"b1_{timestamp}"),
        help="输出前缀；生成 _raw.npz、_windows.npz、_validation.json",
    )
    args = parser.parse_args()

    config, config_path = _load_config(args.config_file)
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = Path(f"{output_prefix}_raw.npz")
    windows_path = Path(f"{output_prefix}_windows.npz")
    validation_path = Path(f"{output_prefix}_validation.json")

    raw = collect_raw_episode(
        config,
        config_path=config_path,
        episode_id=args.episode_id,
        seed=args.seed,
    )
    dataset = build_supervised_windows(
        raw,
        history_steps=DEFAULT_HISTORY_STEPS,
        horizon=DEFAULT_HORIZON,
        control_dt=DEFAULT_CONTROL_DT,
    )
    report = validate_supervised_windows(dataset, raw)

    np.savez_compressed(raw_path, **raw)
    np.savez_compressed(windows_path, **dataset)
    validation_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"raw: {raw_path}")
    print(f"windows: {windows_path}")
    print(f"validation: {validation_path}")


if __name__ == "__main__":
    main()
