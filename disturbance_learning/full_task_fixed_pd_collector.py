#!/usr/bin/env python3
"""Collect T1 full-task raw episodes with a fixed upper-body PD only.

The collector intentionally does *not* import or instantiate ArmMPC,
disturbance predictors, the process runtime, RNEA, or DDQ-to-torque mapping.
It preserves the B1-proven fixed joint-space PD mechanism for waist/arms while
using the independent full-task direct-step schedule and strict pre-step raw
contract introduced for T1-A.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import torch
import yaml

from disturbance_learning.full_task_protocol import (
    DEFAULT_FULL_TASK_PROTOCOL,
    FullTaskClock,
    FullTaskContinuousHeadingFrame,
    FullTaskProtocol,
    direct_step_planned_command,
)
from disturbance_learning.full_task_recording import FullTaskRawRecorder, save_smoke_plots
from disturbance_learning.full_task_template_builder import (
    FIXED_PD_RAW_EXTENSION_VERSION,
    FIXED_PD_RAW_EXTENSION_VERSION_V2,
    TEMPLATE_SCHEMA_VERSION,
    TEMPLATE_SCHEMA_VERSION_V2,
    _json_value,
    causal_h_metrics,
    episode_summary,
    portable_asset,
    save_template_plots,
    sha256_file,
    build_full_task_template,
    evaluate_heldout_template,
    validate_full_task_template,
    write_template_artifacts,
)
from sim_support import (
    HeadingHoldController,
    TorsoAccelerationFilter,
    get_gravity_orientation,
    pd_control,
    quat_to_yaw_wxyz,
    resolve_scene_ids,
    update_torso_motion_state,
)


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_BASE = REPO_DIR / "disturbance_learning" / "data" / "full_task_template_v1"
DEFAULT_OUTPUT_BASE_V2 = REPO_DIR / "disturbance_learning" / "data" / "full_task_template_v2"
PD_UPPER_CTRL_SLICE = slice(12, 23)
PD_RIGHT_ARM_UPPER_INDEX = slice(6, 11)
PD_RIGHT_ARM_CTRL_SLICE = slice(18, 23)


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    role: str
    pair_id: str | None
    pair_seed: int | None
    sign: int
    initial_lower_q_offset_rad: np.ndarray
    initial_lower_dq_rad_s: np.ndarray


class FixedPdRawRecorder:
    """Compose the shared strict recorder with fixed-PD-only extension fields."""

    def __init__(self, *, protocol: FullTaskProtocol, clock: FullTaskClock,
                 nominal_command: np.ndarray, heading_frame_version: str) -> None:
        self.base = FullTaskRawRecorder(
            protocol=protocol, clock=clock, nominal_command=nominal_command,
            heading_frame_version=heading_frame_version,
        )
        self._extra: dict[str, list[Any]] = {
            "causal_h_concentration": [],
            "right_arm_pd_target": [],
            "right_arm_pd_position_error": [],
            "right_arm_pd_requested_tau": [],
            "right_arm_pd_commanded_tau": [],
            "right_arm_pd_saturated": [],
        }

    @property
    def protocol(self) -> FullTaskProtocol:
        return self.base.protocol

    def append(self, *, right_arm_pd_target: np.ndarray, right_arm_pd_position_error: np.ndarray,
               right_arm_pd_requested_tau: np.ndarray, right_arm_pd_commanded_tau: np.ndarray,
               right_arm_pd_saturated: np.ndarray, **kwargs: Any) -> None:
        self.base.append(**kwargs)
        h_state = self.base.causal_h.last_state
        if h_state is None:
            raise RuntimeError("causal H must exist after the first strict raw sample")
        values = {
            "causal_h_concentration": float(h_state.concentration),
            "right_arm_pd_target": np.asarray(right_arm_pd_target, dtype=np.float64).copy(),
            "right_arm_pd_position_error": np.asarray(right_arm_pd_position_error, dtype=np.float64).copy(),
            "right_arm_pd_requested_tau": np.asarray(right_arm_pd_requested_tau, dtype=np.float64).copy(),
            "right_arm_pd_commanded_tau": np.asarray(right_arm_pd_commanded_tau, dtype=np.float64).copy(),
            "right_arm_pd_saturated": np.asarray(right_arm_pd_saturated, dtype=bool).copy(),
        }
        for name, value in values.items():
            if np.asarray(value).shape not in {(), (5,)}:
                raise ValueError(f"fixed-PD field {name} has invalid shape")
            self._extra[name].append(value)

    def to_arrays(self) -> dict[str, np.ndarray]:
        raw = self.base.to_arrays()
        raw.update({name: np.asarray(values) for name, values in self._extra.items()})
        extension = (
            FIXED_PD_RAW_EXTENSION_VERSION_V2
            if self.base.heading_frame_version == FullTaskContinuousHeadingFrame.DEFINITION_VERSION
            else FIXED_PD_RAW_EXTENSION_VERSION
        )
        raw["collector_schema_extension_version"] = np.array(extension)
        raw["right_arm_mode"] = np.array("fixed_posture_pd")
        return raw


def _load_config(config_value: str) -> tuple[dict[str, Any], Path]:
    requested = Path(config_value).expanduser()
    path = requested if requested.is_absolute() else REPO_DIR / "configs" / requested
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream), path.resolve()


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_DIR / path


def _git_state() -> dict[str, str]:
    def read(*args: str) -> str:
        return subprocess.check_output(("git", *args), cwd=REPO_DIR, text=True).strip()
    return {
        "head": read("rev-parse", "HEAD"),
        "branch": read("rev-parse", "--abbrev-ref", "HEAD"),
        "status_short": read("status", "--short"),
    }


def paired_initial_perturbation(pair_seed: int, sign: int) -> tuple[np.ndarray, np.ndarray]:
    if sign not in (-1, 1):
        raise ValueError("paired perturbation sign must be -1 or +1")
    rng = np.random.default_rng(int(pair_seed))
    q_delta = np.clip(rng.normal(0.0, 0.006, size=12), -0.018, 0.018)
    dq_delta = np.clip(rng.normal(0.0, 0.01, size=12), -0.03, 0.03)
    return sign * q_delta, sign * dq_delta


def collection_specs() -> tuple[list[EpisodeSpec], list[EpisodeSpec]]:
    build = [EpisodeSpec("build_nominal", "build", None, None, 0, np.zeros(12), np.zeros(12))]
    for pair_index, pair_seed in enumerate(range(3101, 3106), start=1):
        for sign, suffix in ((1, "plus"), (-1, "minus")):
            q, dq = paired_initial_perturbation(pair_seed, sign)
            build.append(EpisodeSpec(f"build_pair_{pair_index:02d}_{suffix}", "build", f"build_pair_{pair_index:02d}", pair_seed, sign, q, dq))
    heldout: list[EpisodeSpec] = []
    for pair_index, pair_seed in enumerate(range(4101, 4103), start=1):
        for sign, suffix in ((1, "plus"), (-1, "minus")):
            q, dq = paired_initial_perturbation(pair_seed, sign)
            heldout.append(EpisodeSpec(f"heldout_pair_{pair_index:02d}_{suffix}", "heldout", f"heldout_pair_{pair_index:02d}", pair_seed, sign, q, dq))
    return build, heldout


def _fixed_pd_metadata(model: mujoco.MjModel, config: dict[str, Any]) -> dict[str, Any]:
    target = np.asarray(config["arm_waist_target"], dtype=np.float64)
    kps = np.asarray(config["arm_waist_kps"], dtype=np.float64)
    kds = np.asarray(config["arm_waist_kds"], dtype=np.float64)
    if target.shape != (11,) or kps.shape != (11,) or kds.shape != (11,):
        raise ValueError("fixed PD requires 11 waist/arm targets and gains")
    control_limited = np.asarray(model.actuator_ctrllimited[PD_RIGHT_ARM_CTRL_SLICE], dtype=bool)
    control_range = np.asarray(model.actuator_ctrlrange[PD_RIGHT_ARM_CTRL_SLICE], dtype=np.float64)
    return {
        "right_arm_mode": "fixed_posture_pd",
        "mechanism_source": "B1_fixed_joint_space_pd_semantics",
        "upper_body_target": target.tolist(),
        "upper_body_kp": kps.tolist(),
        "upper_body_kd": kds.tolist(),
        "right_arm_target": target[PD_RIGHT_ARM_UPPER_INDEX].tolist(),
        "right_arm_kp": kps[PD_RIGHT_ARM_UPPER_INDEX].tolist(),
        "right_arm_kd": kds[PD_RIGHT_ARM_UPPER_INDEX].tolist(),
        "right_arm_actuator_indices": list(range(PD_RIGHT_ARM_CTRL_SLICE.start, PD_RIGHT_ARM_CTRL_SLICE.stop)),
        "right_arm_ctrl_limited": control_limited.tolist(),
        "right_arm_ctrl_range": control_range.tolist(),
        "command_semantics": "B1 direct joint-space PD; requested torque is checked against ctrl range and no new clipping or mapping is inserted",
    }


def collect_fixed_pd_episode(config: dict[str, Any], *, config_path: Path, spec: EpisodeSpec,
                             protocol: FullTaskProtocol,
                             heading_frame_version: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    if not np.isclose(simulation_dt, protocol.physics_dt, atol=1e-12):
        raise ValueError("full-task collector requires 2 ms physics")
    if not np.isclose(simulation_dt * control_decimation, protocol.policy_dt, atol=1e-12):
        raise ValueError("full-task collector requires 20 ms lower-body policy updates")
    policy_path = _repo_path(str(config["policy_path"]))
    xml_path = _repo_path(str(config["xml_path"]))
    if not policy_path.is_file() or not xml_path.is_file():
        raise FileNotFoundError(f"missing policy or XML: {policy_path}, {xml_path}")

    kps = np.asarray(config["kps"], dtype=np.float64)
    kds = np.asarray(config["kds"], dtype=np.float64)
    default_angles = np.asarray(config["default_angles"], dtype=np.float64)
    arm_target = np.asarray(config["arm_waist_target"], dtype=np.float64)
    arm_kps = np.asarray(config["arm_waist_kps"], dtype=np.float64)
    arm_kds = np.asarray(config["arm_waist_kds"], dtype=np.float64)
    nominal_command = np.asarray(config["cmd_init"], dtype=np.float64)
    cmd_scale = np.asarray(config["cmd_scale"], dtype=np.float32)
    num_actions = int(config["num_actions"])
    num_obs = int(config["num_obs"])
    if num_actions != 12 or default_angles.shape != (12,) or nominal_command.shape != (3,):
        raise ValueError("fixed-PD full-task collector requires the frozen 12-action G1 policy schema")

    torch.manual_seed(0)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = simulation_dt
    data = mujoco.MjData(model)
    data.qpos[7:19] += np.asarray(spec.initial_lower_q_offset_rad, dtype=np.float64)
    data.qvel[6:18] = np.asarray(spec.initial_lower_dq_rad_s, dtype=np.float64)
    mujoco.mj_forward(model, data)
    scene_ids = resolve_scene_ids(model)
    if min(scene_ids.torso_id, scene_ids.imu_site_id, scene_ids.torso_acc_sensor_id) < 0:
        raise ValueError("model does not expose required torso sensors/sites")
    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()

    clock = FullTaskClock(protocol)
    clock.reset(float(data.time), epoch_label=spec.episode_id)
    recorder = FixedPdRawRecorder(
        protocol=protocol, clock=clock, nominal_command=nominal_command,
        heading_frame_version=heading_frame_version,
    )
    heading = HeadingHoldController(
        sample_dt=protocol.policy_dt,
        averaging_window=float(config["heading_filter_cycles"]) * protocol.gait_period,
        kp=float(config["heading_kp"]),
        kd=float(config["heading_kd"]),
        yaw_rate_feedforward=float(nominal_command[2]),
        max_abs_yaw_rate=float(config["heading_max_yaw_rate"]),
    )
    heading_state = heading.last_output
    torso_filter = TorsoAccelerationFilter(
        config,
        enabled=True,
        acc_alpha_key="mpc_torso_acc_filter_alpha",
        alpha_alpha_key="mpc_torso_alpha_filter_alpha",
    )
    motion_buffers = SimpleNamespace(prev_torso_lin_vel=np.zeros(3), prev_torso_ang_vel=np.zeros(3))
    action = np.zeros(num_actions, dtype=np.float32)
    policy_target = default_angles.copy()
    observation = np.zeros(num_obs, dtype=np.float32)
    cmd_planned = direct_step_planned_command(0.0, nominal_command, protocol).planned_command.astype(np.float32)
    cmd_runtime = cmd_planned.copy()
    last_policy_time = np.nan
    fixed_pd = _fixed_pd_metadata(model, config)
    upper_ctrl_limited = np.asarray(model.actuator_ctrllimited[PD_UPPER_CTRL_SLICE], dtype=bool)
    upper_ctrl_range = np.asarray(model.actuator_ctrlrange[PD_UPPER_CTRL_SLICE], dtype=np.float64)

    for sample_index in range(protocol.physics_steps):
        task_time = clock.observe(float(data.time))
        lower_q = data.qpos[7:19].copy()
        lower_dq = data.qvel[6:18].copy()
        data.ctrl[:12] = pd_control(policy_target, lower_q, kps, np.zeros_like(kds), lower_dq, kds)

        upper_q = data.qpos[19:30].copy()
        upper_dq = data.qvel[18:29].copy()
        upper_requested = pd_control(arm_target, upper_q, arm_kps, np.zeros_like(arm_kds), upper_dq, arm_kds)
        data.ctrl[PD_UPPER_CTRL_SLICE] = upper_requested
        right_requested = upper_requested[PD_RIGHT_ARM_UPPER_INDEX].copy()
        right_commanded = data.ctrl[PD_RIGHT_ARM_CTRL_SLICE].copy()
        right_range = upper_ctrl_range[PD_RIGHT_ARM_UPPER_INDEX]
        right_limited = upper_ctrl_limited[PD_RIGHT_ARM_UPPER_INDEX]
        right_saturated = right_limited & ((right_requested < right_range[:, 0] - 1e-12) | (right_requested > right_range[:, 1] + 1e-12))

        torso_state = update_torso_motion_state(model, data, scene_ids, motion_buffers, sample_index, simulation_dt)
        raw_acc, raw_alpha = torso_filter.update(torso_state)
        anchor = protocol.is_mpc_anchor_sample(sample_index)
        policy_update = bool(np.isfinite(last_policy_time) and np.isclose(last_policy_time, task_time, atol=1e-12))
        # The shared recorder appends after final ctrl, before mj_step: strict pre-step.
        recorder.append(
            simulation_time=float(data.time), sample_index=sample_index,
            planned_command=cmd_planned, runtime_command=cmd_runtime,
            policy_update_applied=policy_update, policy_command_consumed_time=last_policy_time,
            mpc_anchor=anchor, torso_position_world=data.xpos[scene_ids.torso_id].copy(),
            torso_rotation_world=torso_state.rotmat, torso_linear_velocity_world=torso_state.lin_vel,
            torso_angular_velocity_world=torso_state.ang_vel,
            torso_linear_acceleration_world_raw=raw_acc,
            torso_linear_acceleration_world_used=torso_state.lin_acc,
            torso_angular_acceleration_world_raw=raw_alpha,
            torso_angular_acceleration_world_used=torso_state.ang_acc,
            lower_body_q=lower_q, lower_body_dq=lower_dq, lower_body_policy_target=policy_target,
            right_arm_q=upper_q[PD_RIGHT_ARM_UPPER_INDEX], right_arm_dq=upper_dq[PD_RIGHT_ARM_UPPER_INDEX],
            right_arm_ddq_des=np.zeros(5), generalized_qpos=data.qpos, generalized_qvel=data.qvel,
            generalized_qacc=data.qacc, actuator_ctrl=data.ctrl, heading_state=heading_state,
            mpc_diagnostics=None, runtime_mapping_safety_fallback_used=False, runtime_executor_flags=0,
            right_arm_pd_target=arm_target[PD_RIGHT_ARM_UPPER_INDEX],
            right_arm_pd_position_error=arm_target[PD_RIGHT_ARM_UPPER_INDEX] - upper_q[PD_RIGHT_ARM_UPPER_INDEX],
            right_arm_pd_requested_tau=right_requested, right_arm_pd_commanded_tau=right_commanded,
            right_arm_pd_saturated=right_saturated,
        )
        mujoco.mj_step(model, data)

        next_index = sample_index + 1
        if next_index % control_decimation == 0:
            next_task_time = clock.observe(float(data.time))
            cmd_planned = direct_step_planned_command(next_task_time, nominal_command, protocol).planned_command.astype(np.float32)
            torso_yaw = quat_to_yaw_wxyz(data.xquat[scene_ids.torso_id].copy())
            heading_state = heading.update(torso_yaw, torso_state.ang_vel[2])
            cmd_runtime = cmd_planned.copy()
            cmd_runtime[2] = heading_state.yaw_rate_command
            qj = data.qpos[7:19].copy()
            dqj = data.qvel[6:18].copy()
            phase = (next_task_time / protocol.gait_period) % 1.0
            observation[:3] = data.qvel[3:6] * float(config["ang_vel_scale"])
            observation[3:6] = get_gravity_orientation(data.qpos[3:7])
            observation[6:9] = cmd_runtime * cmd_scale
            observation[9 : 9 + num_actions] = (qj - default_angles) * float(config["dof_pos_scale"])
            observation[9 + num_actions : 9 + 2 * num_actions] = dqj * float(config["dof_vel_scale"])
            observation[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            observation[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)], dtype=np.float32)
            with torch.inference_mode():
                action = policy(torch.from_numpy(observation).unsqueeze(0)).cpu().numpy().squeeze()
            policy_target = action * float(config["action_scale"]) + default_angles
            last_policy_time = next_task_time

    raw = recorder.to_arrays()
    raw.update({
        "episode_id": np.array(spec.episode_id), "episode_role": np.array(spec.role),
        "pair_id": np.array("" if spec.pair_id is None else spec.pair_id),
        "pair_seed": np.array(-1 if spec.pair_seed is None else spec.pair_seed, dtype=np.int64),
        "pair_sign": np.array(spec.sign, dtype=np.int64),
        "initial_lower_q_offset_rad": np.asarray(spec.initial_lower_q_offset_rad, dtype=np.float64),
        "initial_lower_dq_rad_s": np.asarray(spec.initial_lower_dq_rad_s, dtype=np.float64),
        "config_path": np.array(str(config_path)), "policy_path": np.array(str(policy_path.resolve())),
        "xml_path": np.array(str(xml_path.resolve())),
        "right_arm_pd_ctrl_limited": np.asarray(fixed_pd["right_arm_ctrl_limited"], dtype=bool),
        "right_arm_pd_ctrl_range": np.asarray(fixed_pd["right_arm_ctrl_range"], dtype=np.float64),
    })
    metadata = {
        "episode": _json_value(asdict(spec)), "fixed_pd": fixed_pd,
        "heading_frame_version": heading_frame_version,
        "config_path": config_path, "policy_path": policy_path, "xml_path": xml_path,
        "heading_control": {
            "enabled": True, "filter_cycles": float(config["heading_filter_cycles"]),
            "kp": float(config["heading_kp"]), "kd": float(config["heading_kd"]),
            "yaw_rate_feedforward": float(nominal_command[2]), "max_abs_yaw_rate": float(config["heading_max_yaw_rate"]),
        },
    }
    return raw, metadata


def save_episode(*, output_dir: Path, raw: dict[str, np.ndarray], metadata: dict[str, Any],
                 protocol: FullTaskProtocol, make_smoke_plots: bool) -> tuple[Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "full_task_fixed_pd_raw.npz"
    np.savez_compressed(raw_path, **raw)
    summary = episode_summary(raw, protocol)
    summary_path = output_dir / "episode_summary.json"
    summary_path.write_text(json.dumps(_json_value(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plots = save_smoke_plots(raw, protocol, output_dir) if make_smoke_plots else []
    if make_smoke_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        anchor_mask = np.asarray(raw["mpc_anchor"], dtype=bool)
        anchor_time = np.asarray(raw["task_time"], dtype=np.float64)[anchor_mask]
        h_yaw = np.unwrap(np.asarray(raw["causal_h_yaw_world"], dtype=np.float64)[anchor_mask])
        h_concentration = np.asarray(raw["causal_h_concentration"], dtype=np.float64)[anchor_mask]
        h_jump = np.r_[0.0, np.abs(np.arctan2(np.sin(np.diff(h_yaw)), np.cos(np.diff(h_yaw))))]
        h_path = output_dir / "fixed_pd_causal_h_metrics.png"
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(anchor_time, h_yaw); axes[0].set_ylabel("H yaw [rad]")
        axes[1].plot(anchor_time, h_concentration); axes[1].set_ylabel("concentration")
        axes[2].plot(anchor_time, h_jump); axes[2].set_ylabel("adjacent jump [rad]"); axes[2].set_xlabel("task time [s]")
        for axis in axes:
            axis.axvline(protocol.stop_time, color="red", ls="--", lw=1.0)
            axis.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(h_path, dpi=160); plt.close(fig)
        plots.append(h_path)

        arm_path = output_dir / "fixed_pd_right_arm_hold.png"
        time = np.asarray(raw["task_time"], dtype=np.float64)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(time, np.asarray(raw["right_arm_pd_position_error"], dtype=np.float64), lw=0.75)
        axes[0].set_ylabel("q target - q [rad]"); axes[0].grid(True, alpha=0.3)
        axes[1].plot(time, np.asarray(raw["right_arm_pd_requested_tau"], dtype=np.float64), lw=0.75)
        axes[1].set_ylabel("requested PD tau"); axes[1].set_xlabel("task time [s]"); axes[1].grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(arm_path, dpi=160); plt.close(fig)
        plots.append(arm_path)
    manifest = {
        "collector_schema_extension_version": str(
            np.asarray(raw["collector_schema_extension_version"]).item()
        ),
        "protocol": {
            "name": protocol.protocol_name, "version": protocol.protocol_version,
            "physics_dt": protocol.physics_dt, "anchor_dt": protocol.mpc_dt,
            "policy_dt": protocol.policy_dt, "gait_period": protocol.gait_period,
            "stop_time": protocol.stop_time, "headline_interval": "[0.0,8.0)",
            "record_end": protocol.record_end, "horizon": protocol.horizon,
            "planned_command": "nominal vx/vy for [0,6.4), direct zero vx/vy for [6.4,8.06]; planned wz is nominal feedforward",
            "task_t0": "planned command visible; no claim of physical motion; first lower-body policy consumption is 20 ms",
        },
        "time_semantics": {
            "strict_pre_step": "state/commands/final ctrl describe [t,t+2ms) and are saved before mj_step",
            "template_anchor": "6ms absolute task-time grid", "horizon_frame": "H frozen from anchor through its complete 54ms future window",
        },
        "episode": metadata["episode"], "right_arm": metadata["fixed_pd"],
        "heading_control": metadata["heading_control"],
        "control_chain": {
            "lower_body_policy": "torchscript locomotion policy", "right_arm_mode": "fixed_posture_pd",
            "mpc_called": False, "old_template_predictor_called": False, "neural_called": False,
            "hybrid_residual_called": False, "zoh_called": False, "process_called": False,
            "ddq_to_torque_mapping_called": False,
        },
        "causal_h": {**causal_h_metrics(raw), "additional_low_pass_filter": "none"},
        "heading_frame": {
            "version": metadata["heading_frame_version"],
            "definition": (
                "causal prefix before 0.8s; causal trailing 0.8s circular mean until 6.4s; "
                "then freeze the last pre-stop H; one anchor H is fixed across its 54ms window"
                if metadata["heading_frame_version"] == FullTaskContinuousHeadingFrame.DEFINITION_VERSION
                else "v1 causal prefix then previous complete gait-cycle circular mean"
            ),
            "additional_low_pass_filter": "none",
        },
        "assets": {
            "config": portable_asset(metadata["config_path"], REPO_DIR),
            "policy": portable_asset(metadata["policy_path"], REPO_DIR),
            "xml": portable_asset(metadata["xml_path"], REPO_DIR),
            "raw": portable_asset(raw_path, REPO_DIR),
            "summary": portable_asset(summary_path, REPO_DIR),
            "plots": [portable_asset(path, REPO_DIR) for path in plots],
        },
        "git": _git_state(), "validation": summary,
    }
    manifest_path = output_dir / "episode_manifest.json"
    manifest_path.write_text(json.dumps(_json_value(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return raw_path, summary


def _run_specs(*, root: Path, specs: list[EpisodeSpec], config: dict[str, Any], config_path: Path,
               protocol: FullTaskProtocol, heading_frame_version: str,
               smoke: bool = False) -> tuple[list[dict[str, np.ndarray]], list[Path], list[dict[str, Any]]]:
    raws: list[dict[str, np.ndarray]] = []
    paths: list[Path] = []
    summaries: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        print(f"[T1 fixed-PD] {index}/{len(specs)} {spec.episode_id}", flush=True)
        raw, metadata = collect_fixed_pd_episode(
            config, config_path=config_path, spec=spec, protocol=protocol,
            heading_frame_version=heading_frame_version,
        )
        raw_path, summary = save_episode(output_dir=root / "episodes" / spec.episode_id, raw=raw, metadata=metadata, protocol=protocol, make_smoke_plots=smoke)
        if summary["status"] != "PASS":
            raise RuntimeError(f"{spec.episode_id} failed fixed-PD safety/contract gate: {summary}")
        raws.append(raw); paths.append(raw_path); summaries.append(summary)
    return raws, paths, summaries


def run_nominal_smoke(root: Path, config: dict[str, Any], config_path: Path,
                      heading_frame_version: str) -> dict[str, Any]:
    protocol = DEFAULT_FULL_TASK_PROTOCOL
    spec = EpisodeSpec("nominal_fixed_pd_smoke", "nominal_smoke", None, None, 0, np.zeros(12), np.zeros(12))
    raw, metadata = collect_fixed_pd_episode(
        config, config_path=config_path, spec=spec, protocol=protocol,
        heading_frame_version=heading_frame_version,
    )
    raw_path, summary = save_episode(output_dir=root / "nominal_smoke", raw=raw, metadata=metadata, protocol=protocol, make_smoke_plots=True)
    result = {"raw_path": str(raw_path), "summary": summary, "output_dir": str(root / "nominal_smoke")}
    (root / "nominal_smoke_result.json").write_text(json.dumps(_json_value(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if summary["status"] != "PASS":
        raise RuntimeError("fixed-PD nominal smoke failed; do not start 11+4 collection")
    return result


def _build_existing_collection(root: Path, config: dict[str, Any], config_path: Path,
                               template_schema_version: str) -> dict[str, str]:
    protocol = DEFAULT_FULL_TASK_PROTOCOL
    build_specs, heldout_specs = collection_specs()
    def load(specs: list[EpisodeSpec]) -> tuple[list[dict[str, np.ndarray]], list[Path], list[dict[str, Any]]]:
        raws: list[dict[str, np.ndarray]] = []
        paths: list[Path] = []
        summaries: list[dict[str, Any]] = []
        for spec in specs:
            episode_dir = root / "episodes" / spec.episode_id
            raw_path = episode_dir / "full_task_fixed_pd_raw.npz"
            summary_path = episode_dir / "episode_summary.json"
            if not raw_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(f"cannot resume template build; missing {episode_dir}")
            with np.load(raw_path, allow_pickle=False) as source:
                raw = {name: source[name].copy() for name in source.files}
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "PASS":
                raise RuntimeError(f"cannot use failed episode {spec.episode_id}")
            raws.append(raw); paths.append(raw_path); summaries.append(summary)
        return raws, paths, summaries

    build_raws, build_paths, build_summary = load(build_specs)
    heldout_raws, heldout_paths, heldout_summary = load(heldout_specs)

    # This is an audit statistic, not a selection criterion: initial offsets
    # are paired and fixed before any template values are computed.
    headline_index = int(round(protocol.headline_end / protocol.physics_dt))
    build_xy = np.stack([raw["torso_position_world"][: headline_index + 1, :2] for raw in build_raws])
    pairwise_max_xy_separation = [
        float(np.max(np.linalg.norm(build_xy[left] - build_xy[right], axis=1)))
        for left in range(len(build_xy)) for right in range(left + 1, len(build_xy))
    ]
    displacement = np.linalg.norm(build_xy[:, -1] - build_xy[:, 0], axis=1)
    arc_length = np.array([
        np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)) for xy in build_xy
    ])
    diversity = {
        "pairwise_max_xy_separation_m": {
            "count": len(pairwise_max_xy_separation), "min": float(np.min(pairwise_max_xy_separation)),
            "mean": float(np.mean(pairwise_max_xy_separation)), "max": float(np.max(pairwise_max_xy_separation)),
        },
        "xy_displacement_m": {"min": float(np.min(displacement)), "mean": float(np.mean(displacement)),
                              "max": float(np.max(displacement)), "std": float(np.std(displacement))},
        "xy_arc_length_m": {"min": float(np.min(arc_length)), "mean": float(np.mean(arc_length)),
                              "max": float(np.max(arc_length)), "std": float(np.std(arc_length))},
    }
    h_metrics = [causal_h_metrics(raw) for raw in build_raws + heldout_raws]
    causal_h_summary = {
        "max_adjacent_h_yaw_jump_rad": max(item["max_adjacent_h_yaw_jump_rad"] for item in h_metrics),
        "min_circular_concentration": min(item["min_circular_concentration"] for item in h_metrics),
        "max_cycle_boundary_h_yaw_jump_rad": max(item["max_cycle_boundary_h_yaw_jump_rad"] for item in h_metrics),
        "additional_low_pass_filter": "none",
    }
    template = build_full_task_template(
        build_raws, [spec.episode_id for spec in build_specs], protocol,
        template_schema_version=template_schema_version,
    )
    validation = validate_full_task_template(
        template, protocol, expected_schema_version=template_schema_version
    )
    heldout_metrics, heldout_windows = evaluate_heldout_template(template, heldout_raws, [spec.episode_id for spec in heldout_specs], protocol)
    build_windows = []
    # Evaluation returns H-frozen windows without influencing the template.
    from disturbance_learning.full_task_template_builder import _episode_heading_windows
    for raw in build_raws:
        build_windows.append(_episode_heading_windows(raw, protocol))
    plot_paths = save_template_plots(output_dir=root / "plots", build_raws=build_raws, build_windows=build_windows,
                                     heldout_windows=heldout_windows, heldout_raws=heldout_raws, template=template, protocol=protocol)
    collection_manifest = {
        "protocol": {"name": protocol.protocol_name, "version": protocol.protocol_version, "headline": [0.0, protocol.headline_end], "record_end": protocol.record_end, "horizon": protocol.horizon},
        "right_arm_mode": "fixed_posture_pd", "perturbation_design": {
            "build": [_json_value(asdict(spec)) for spec in build_specs],
            "heldout": [_json_value(asdict(spec)) for spec in heldout_specs],
            "q_std_rad": 0.006, "q_clip_rad": [-0.018, 0.018], "dq_std_rad_s": 0.01, "dq_clip_rad_s": [-0.03, 0.03],
            "pairing": "each non-nominal pair shares one delta and uses +delta/-delta",
        },
        "frozen_conditions": {
            "planned_command": "nominal vx/vy for [0,6.4), direct zero vx/vy for [6.4,8.06]; planned wz remains feedforward",
            "heading_control": {"enabled": True, "filter_cycles": float(config["heading_filter_cycles"]),
                                "kp": float(config["heading_kp"]), "kd": float(config["heading_kd"]),
                                "max_abs_yaw_rate": float(config["heading_max_yaw_rate"])},
            "right_arm_fixed_pd": {
                "target": np.asarray(build_raws[0]["right_arm_pd_target"][0], dtype=float).tolist(),
                "kp": np.asarray(config["arm_waist_kps"], dtype=float)[PD_RIGHT_ARM_UPPER_INDEX].tolist(),
                "kd": np.asarray(config["arm_waist_kds"], dtype=float)[PD_RIGHT_ARM_UPPER_INDEX].tolist(),
                "ctrl_limited": np.asarray(build_raws[0]["right_arm_pd_ctrl_limited"], dtype=bool).tolist(),
                "ctrl_range_if_limited": np.asarray(build_raws[0]["right_arm_pd_ctrl_range"], dtype=float).tolist(),
                "saturation_policy": "record only for limited actuators; this XML reports all five right-arm actuators as unlimited, so no new clipping or mapping is inserted",
            },
            "payload": "base XML only; no runtime added payload mass or payload-model mutation",
            "physics_dt": protocol.physics_dt, "policy_dt": protocol.policy_dt, "gait_time_origin": "task time zero",
        },
        "trajectory_diversity": diversity, "causal_h_summary": causal_h_summary,
        "heading_frame_version": str(np.asarray(template["heading_frame_version"]).item()),
        "build_episode_summaries": build_summary, "heldout_episode_summaries": heldout_summary,
        "config": portable_asset(config_path, REPO_DIR), "policy": portable_asset(_repo_path(str(config["policy_path"])), REPO_DIR),
        "xml": portable_asset(_repo_path(str(config["xml_path"])), REPO_DIR), "git": _git_state(),
    }
    artifacts = write_template_artifacts(output_dir=root, repo_dir=REPO_DIR, template=template, template_validation=validation,
                                         build_episode_paths=build_paths, heldout_episode_paths=heldout_paths,
                                         collection_manifest=collection_manifest, heldout_metrics=heldout_metrics, plot_paths=plot_paths)
    result = {name: str(path) for name, path in artifacts.items()}
    (root / "batch_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_batch(root: Path, config: dict[str, Any], config_path: Path,
              template_schema_version: str, heading_frame_version: str) -> dict[str, str]:
    protocol = DEFAULT_FULL_TASK_PROTOCOL
    build_specs, heldout_specs = collection_specs()
    _run_specs(root=root, specs=build_specs, config=config, config_path=config_path,
               protocol=protocol, heading_frame_version=heading_frame_version)
    _run_specs(root=root, specs=heldout_specs, config=config, config_path=config_path,
               protocol=protocol, heading_frame_version=heading_frame_version)
    return _build_existing_collection(root, config, config_path, template_schema_version)


def main() -> None:
    parser = argparse.ArgumentParser(description="T1 full-task fixed-PD collector and absolute-time template builder")
    parser.add_argument("mode", choices=("nominal-smoke", "batch", "build-existing"))
    parser.add_argument("--config", default="g1.yaml")
    parser.add_argument("--template-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--output-dir", default="", help="empty: timestamped versioned full-task template directory")
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    is_v2 = args.template_version == "v2"
    output_base = DEFAULT_OUTPUT_BASE_V2 if is_v2 else DEFAULT_OUTPUT_BASE
    template_schema_version = TEMPLATE_SCHEMA_VERSION_V2 if is_v2 else TEMPLATE_SCHEMA_VERSION
    heading_frame_version = (
        FullTaskContinuousHeadingFrame.DEFINITION_VERSION
        if is_v2 else "full_task_cycle_held_heading_v1"
    )
    root = Path(args.output_dir).expanduser().resolve() if args.output_dir else output_base / timestamp
    if args.mode == "build-existing":
        if not args.output_dir or not root.is_dir():
            raise ValueError("build-existing requires an existing --output-dir containing all 11+4 episodes")
    else:
        root.mkdir(parents=True, exist_ok=False)
    config, config_path = _load_config(args.config)
    if args.mode == "nominal-smoke":
        result = run_nominal_smoke(root, config, config_path, heading_frame_version)
    elif args.mode == "batch":
        result = run_batch(
            root, config, config_path, template_schema_version, heading_frame_version
        )
    else:
        result = _build_existing_collection(root, config, config_path, template_schema_version)
    print(json.dumps(_json_value(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
