"""Fixed right-arm PD startup and exact full-task MPC handoff contract.

The formal task, gait, template, and continuous-H clocks all start at
simulation time zero.  This module only decides whether the right arm is still
executing the configured fixed-posture PD or whether the 24 ms legal MPC anchor
has been reached.  It does not reset or offset any
clock and is deliberately independent from MPC and DDQ-to-torque mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from disturbance_template.full_task_protocol import (
    DEFAULT_FULL_TASK_PROTOCOL,
    FullTaskProtocol,
)


STARTUP_PD_PROTOCOL_VERSION = "full_task_fixed_startup_pd_24ms_handoff_v2"
FORMAL_STARTUP_PD_DURATION_S = 0.024
RIGHT_ARM_MODE_FIXED_POSTURE_PD = 0
RIGHT_ARM_MODE_MPC_PROCESS = 1


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class StartupPdDecision:
    sample_index: int
    task_time: float
    mpc_anchor: bool
    mpc_control_enabled: bool
    first_mpc_anchor: bool
    right_arm_mode: int


@dataclass(frozen=True)
class FixedStartupPdHandoff:
    """Exact 2/6/20 ms startup handoff, shared by all runtime branches."""

    duration_s: float
    protocol: FullTaskProtocol = DEFAULT_FULL_TASK_PROTOCOL

    def __post_init__(self) -> None:
        duration = float(self.duration_s)
        if not np.isclose(
            duration, FORMAL_STARTUP_PD_DURATION_S, atol=1e-12, rtol=0.0
        ):
            raise ValueError("formal startup PD duration must be exactly 0.024 s")
        self.protocol.anchor_index(duration)

    @property
    def takeover_sample_index(self) -> int:
        return int(round(float(self.duration_s) / self.protocol.physics_dt))

    @property
    def takeover_anchor_index(self) -> int:
        return self.protocol.anchor_index(float(self.duration_s))

    def decision(self, sample_index: int) -> StartupPdDecision:
        index = int(sample_index)
        task_time = self.protocol.sample_time(index)
        mpc_anchor = self.protocol.is_mpc_anchor_sample(index)
        # Once the legal handoff anchor is reached, the process chain remains
        # active on the two cached-execution physics steps between MPC solves.
        control_enabled = bool(index >= self.takeover_sample_index)
        first = bool(index == self.takeover_sample_index)
        return StartupPdDecision(
            sample_index=index,
            task_time=task_time,
            mpc_anchor=mpc_anchor,
            mpc_control_enabled=control_enabled,
            first_mpc_anchor=first,
            right_arm_mode=(
                RIGHT_ARM_MODE_MPC_PROCESS
                if index >= self.takeover_sample_index
                else RIGHT_ARM_MODE_FIXED_POSTURE_PD
            ),
        )

    def validate_short_smoke_end(self, end_time: float) -> float:
        value = float(end_time)
        if not np.isfinite(value):
            raise ValueError("short smoke end must be finite")
        if value + 1e-12 < float(self.duration_s) + 0.2:
            raise ValueError("short smoke must cover at least 0.2 s after handoff")
        steps = value / self.protocol.physics_dt
        if not np.isclose(steps, round(steps), atol=1e-10, rtol=0.0):
            raise ValueError("short smoke end must be on the 2 ms grid")
        if value > self.protocol.headline_end + 1e-12:
            raise ValueError("short smoke end cannot exceed the headline")
        return value


@dataclass(frozen=True)
class FootContactIds:
    """MuJoCo ids used only for startup-state diagnostics."""

    floor_geom_id: int
    left_foot_geom_ids: frozenset[int]
    right_foot_geom_ids: frozenset[int]


def resolve_foot_contact_ids(model: mujoco.MjModel) -> FootContactIds:
    """Resolve bilateral foot collision geoms; never gates MPC handoff."""

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0:
        raise ValueError("MuJoCo model has no floor geom")
    sides: dict[str, set[int]] = {"left": set(), "right": set()}
    for geom_id in range(model.ngeom):
        if int(model.geom_contype[geom_id]) == 0:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        )
        for side in sides:
            if body_name.startswith(f"{side}_ankle_"):
                sides[side].add(geom_id)
    if not sides["left"] or not sides["right"]:
        raise ValueError("MuJoCo model does not expose bilateral ankle collision geoms")
    return FootContactIds(
        floor_geom_id=int(floor),
        left_foot_geom_ids=frozenset(sides["left"]),
        right_foot_geom_ids=frozenset(sides["right"]),
    )


def measure_foot_ground_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: FootContactIds,
) -> dict[str, float | int]:
    """Measure contacts for reporting only; output cannot affect takeover."""

    counts = {"left": 0, "right": 0}
    normal_force = {"left": 0.0, "right": 0.0}
    minimum_distance = 0.0
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if ids.floor_geom_id not in pair:
            continue
        other = (
            int(contact.geom2)
            if int(contact.geom1) == ids.floor_geom_id
            else int(contact.geom1)
        )
        side = (
            "left"
            if other in ids.left_foot_geom_ids
            else "right"
            if other in ids.right_foot_geom_ids
            else None
        )
        if side is None:
            continue
        counts[side] += 1
        mujoco.mj_contactForce(model, data, contact_index, contact_force)
        normal_force[side] += max(0.0, float(contact_force[0]))
        minimum_distance = min(minimum_distance, float(contact.dist))
    return {
        "left_count": counts["left"],
        "right_count": counts["right"],
        "left_normal_force_n": normal_force["left"],
        "right_normal_force_n": normal_force["right"],
        "max_penetration_m": max(0.0, -minimum_distance),
    }


def mapping_safety_snapshot(mapping_result: Any | None) -> dict[str, Any]:
    """Normalize sync/process mapper diagnostics after their common adapter."""

    nan5 = np.full(5, np.nan, dtype=np.float64)
    if mapping_result is None:
        return {
            "mapping_updated": False,
            "qacc_baseline": nan5,
            "first_pass_qacc_validated": nan5,
            "second_pass_qacc_validated": nan5,
            "final_qacc_validated": nan5,
            "second_pass_triggered": False,
            "safety_fallback_used": False,
            "safety_fallback_attempts": 0,
            "hold_last_safe_available": False,
            "hold_last_safe_used": False,
            "hold_last_safe_satisfied": False,
            "safe_hold_used": False,
            "safety_line_search_used": False,
            "safety_line_search_attempts": 0,
            "safety_line_search_time_s": 0.0,
            "final_output_certified": False,
            "no_safe_torque": False,
        }
    return {
        "mapping_updated": True,
        "qacc_baseline": np.asarray(mapping_result.qacc_baseline, dtype=np.float64),
        "first_pass_qacc_validated": np.asarray(
            mapping_result.first_pass_qacc_validated, dtype=np.float64
        ),
        "second_pass_qacc_validated": np.asarray(
            mapping_result.second_pass_qacc_validated, dtype=np.float64
        ),
        "final_qacc_validated": np.asarray(
            mapping_result.qacc_validated, dtype=np.float64
        ),
        "second_pass_triggered": bool(mapping_result.second_pass_triggered),
        "safety_fallback_used": bool(mapping_result.safety_fallback_used),
        "safety_fallback_attempts": int(mapping_result.safety_fallback_attempts),
        "hold_last_safe_available": bool(mapping_result.hold_last_safe_available),
        "hold_last_safe_used": bool(mapping_result.hold_last_safe_used),
        "hold_last_safe_satisfied": bool(mapping_result.hold_last_safe_satisfied),
        "safe_hold_used": bool(mapping_result.safe_hold_used),
        "safety_line_search_used": bool(mapping_result.safety_line_search_used),
        "safety_line_search_attempts": int(
            mapping_result.safety_line_search_attempts
        ),
        "safety_line_search_time_s": float(mapping_result.safety_line_search_time),
        "final_output_certified": bool(mapping_result.final_output_certified),
        "no_safe_torque": bool(mapping_result.no_safe_torque),
    }


@dataclass
class StartupPdTraceRecorder:
    handoff: FixedStartupPdHandoff
    runtime_mode: str
    _rows: list[dict[str, Any]] = field(default_factory=list)

    def append(self, **row: Any) -> None:
        index = int(row["sample_index"])
        decision = self.handoff.decision(index)
        simulation_time = float(row["simulation_time"])
        task_time = float(row["task_time"])
        expected = decision.task_time
        if not np.isclose(simulation_time, expected, atol=1e-10, rtol=0.0):
            raise ValueError("startup trace simulation time left the 2 ms task grid")
        if not np.isclose(task_time, expected, atol=1e-10, rtol=0.0):
            raise ValueError("startup trace task time must equal simulation time")
        if bool(row["mpc_control_enabled"]) != decision.mpc_control_enabled:
            raise ValueError("startup trace control mode disagrees with handoff contract")
        if bool(row["mpc_anchor"]) != decision.mpc_anchor:
            raise ValueError("startup trace MPC anchor disagrees with handoff contract")
        actual_tau = np.asarray(row["actual_right_arm_tau"], dtype=np.float64)
        pd_tau = np.asarray(row["fixed_posture_pd_tau"], dtype=np.float64)
        if actual_tau.shape != (5,) or pd_tau.shape != (5,):
            raise ValueError("startup trace right-arm torques must have shape (5,)")
        if decision.right_arm_mode == RIGHT_ARM_MODE_FIXED_POSTURE_PD:
            if not np.allclose(actual_tau, pd_tau, atol=1e-12, rtol=0.0):
                raise ValueError("startup prefix did not execute fixed-posture PD")
        stored = dict(row)
        stored["right_arm_mode"] = decision.right_arm_mode
        stored["actual_right_arm_tau"] = actual_tau.copy()
        stored["fixed_posture_pd_tau"] = pd_tau.copy()
        self._rows.append(stored)

    def to_arrays(self) -> dict[str, np.ndarray]:
        if not self._rows:
            raise ValueError("startup trace is empty")
        keys = tuple(self._rows[0]) + ("right_arm_mode",)
        arrays: dict[str, np.ndarray] = {}
        for key in keys:
            arrays[key] = np.asarray([row[key] for row in self._rows])
        return arrays

    def summary(self, *, dry_warmup: dict[str, Any] | None) -> dict[str, Any]:
        arrays = self.to_arrays()
        first_mpc_rows = np.flatnonzero(arrays["mpc_control_enabled"])
        if first_mpc_rows.size == 0:
            raise ValueError("startup trace ended before MPC handoff")
        handoff_row = int(first_mpc_rows[0])
        if handoff_row <= 0:
            raise ValueError("startup trace has no physically executed PD predecessor")
        previous_tau = np.asarray(arrays["previous_executed_tau"][handoff_row])
        first_mpc_tau = np.asarray(arrays["actual_right_arm_tau"][handoff_row])
        last_pd_tau = np.asarray(arrays["actual_right_arm_tau"][handoff_row - 1])
        if not np.allclose(previous_tau, last_pd_tau, atol=1e-12, rtol=0.0):
            raise ValueError("mapper previous torque is not the last physically executed PD")
        policy_rows = np.flatnonzero(arrays["policy_update_applied"])
        first_policy_time = (
            float(arrays["task_time"][policy_rows[0]])
            if policy_rows.size
            else None
        )
        anchor_prefix = arrays["mpc_anchor"] & (
            arrays["task_time"] < float(self.handoff.duration_s) - 1e-12
        )
        if np.any(arrays["mpc_control_enabled"][anchor_prefix]):
            raise ValueError("MPC output occurred during fixed-PD startup")
        if not np.all(arrays["predictor_updated"][arrays["mpc_anchor"]]):
            raise ValueError("template/H did not advance at every startup MPC anchor")
        predictor_anchor = int(arrays["predictor_template_anchor_index"][handoff_row])
        if predictor_anchor != self.handoff.takeover_anchor_index:
            raise ValueError("handoff template lookup restarted or used the wrong anchor")
        mapping = {
            name: _json_value(arrays[name][handoff_row])
            for name in (
                "desired_right_arm_ddq",
                "qacc_baseline",
                "first_pass_qacc_validated",
                "second_pass_qacc_validated",
                "final_qacc_validated",
                "second_pass_triggered",
                "safety_fallback_used",
                "safety_fallback_attempts",
                "hold_last_safe_available",
                "hold_last_safe_used",
                "hold_last_safe_satisfied",
                "safe_hold_used",
                "safety_line_search_used",
                "safety_line_search_attempts",
                "safety_line_search_time_s",
                "final_output_certified",
                "no_safe_torque",
            )
        }
        return {
            "protocol_version": STARTUP_PD_PROTOCOL_VERSION,
            "runtime_mode": self.runtime_mode,
            "startup_pd_duration_s": float(self.handoff.duration_s),
            "task_and_simulation_time_share_zero_origin": True,
            "first_lower_policy_update_simulation_time_s": first_policy_time,
            "first_lower_policy_update_task_time_s": first_policy_time,
            "handoff": {
                "simulation_time_s": float(arrays["simulation_time"][handoff_row]),
                "task_time_s": float(arrays["task_time"][handoff_row]),
                "template_absolute_task_time_s": float(
                    arrays["predictor_task_time"][handoff_row]
                ),
                "template_anchor_index": predictor_anchor,
                "gait_phase_cycles": float(arrays["gait_phase_cycles"][handoff_row]),
                "left_foot_contact_count": int(
                    arrays["left_foot_contact_count"][handoff_row]
                ),
                "right_foot_contact_count": int(
                    arrays["right_foot_contact_count"][handoff_row]
                ),
                "raw_torso_acceleration_norm_m_s2": float(
                    arrays["raw_torso_acceleration_norm_m_s2"][handoff_row]
                ),
                "base_vertical_velocity_m_s": float(
                    arrays["base_vertical_velocity_m_s"][handoff_row]
                ),
                "last_fixed_pd_tau_nm": last_pd_tau,
                "previous_executed_tau_input_nm": previous_tau,
                "first_mpc_tau_nm": first_mpc_tau,
                "tau_jump_nm": first_mpc_tau - last_pd_tau,
                "tau_jump_l2_nm": float(np.linalg.norm(first_mpc_tau - last_pd_tau)),
                "tau_jump_max_abs_nm": float(
                    np.max(np.abs(first_mpc_tau - last_pd_tau))
                ),
                "previous_tau_available": bool(
                    arrays["previous_executed_tau_available"][handoff_row]
                ),
                "mapping": mapping,
            },
            "prefix": {
                "included_in_headline": True,
                "mpc_output_count": int(
                    np.count_nonzero(arrays["mpc_control_enabled"][anchor_prefix])
                ),
                "fixed_pd_sample_count": int(
                    np.count_nonzero(
                        arrays["right_arm_mode"]
                        == RIGHT_ARM_MODE_FIXED_POSTURE_PD
                    )
                ),
                "predictor_anchor_count_before_handoff": int(
                    np.count_nonzero(anchor_prefix)
                ),
            },
            "whole_trace": {
                "sample_count": int(arrays["sample_index"].size),
                "last_simulation_time_s": float(arrays["simulation_time"][-1]),
                "last_task_time_s": float(arrays["task_time"][-1]),
                "predictor_fallback_count": int(
                    np.count_nonzero(arrays["predictor_fallback_used"])
                ),
                "final_unsafe_count": int(
                    np.count_nonzero(
                        arrays["mapping_updated"]
                        & ~arrays["final_output_certified"]
                    )
                ),
                "no_safe_torque_count": int(
                    np.count_nonzero(arrays["no_safe_torque"])
                ),
                "safe_hold_count": int(np.count_nonzero(arrays["safe_hold_used"])),
                "safety_line_search_count": int(
                    np.count_nonzero(arrays["safety_line_search_used"])
                ),
                "safety_line_search_attempts": int(
                    np.sum(arrays["safety_line_search_attempts"])
                ),
                "safety_line_search_extra_time_ms": float(
                    np.sum(arrays["safety_line_search_time_s"]) * 1e3
                ),
            },
            "dry_preflight": dry_warmup,
        }


def save_startup_pd_artifacts(
    recorder: StartupPdTraceRecorder,
    run_dir: Path,
    *,
    dry_warmup: dict[str, Any] | None,
) -> dict[str, Any]:
    output = Path(run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays = recorder.to_arrays()
    summary = recorder.summary(dry_warmup=dry_warmup)
    trace_path = output / "startup_pd_handoff_trace.npz"
    summary_path = output / "startup_pd_handoff_summary.json"
    plot_path = output / "startup_pd_handoff_transition.png"
    np.savez_compressed(trace_path, **arrays)
    summary_path.write_text(
        json.dumps(_json_value(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    handoff_time = float(recorder.handoff.duration_s)
    window = arrays["task_time"] <= min(
        float(arrays["task_time"][-1]), handoff_time + 0.2
    ) + 1e-12
    time = arrays["task_time"][window]
    linewidth = 0.8
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for joint in range(5):
        axes[0].plot(
            time,
            arrays["actual_right_arm_tau"][window, joint],
            lw=linewidth,
            label=f"joint {joint + 1}",
        )
    axes[0].set_ylabel("actual tau [Nm]")
    axes[1].plot(
        time,
        arrays["raw_torso_acceleration_norm_m_s2"][window],
        lw=linewidth,
        label="raw torso |acc|",
    )
    axes[1].plot(
        time,
        arrays["base_vertical_velocity_m_s"][window],
        lw=linewidth,
        label="base vz",
    )
    axes[1].set_ylabel("handoff state")
    axes[2].plot(
        time,
        arrays["predictor_task_time"][window],
        lw=linewidth,
        label="template lookup time",
    )
    axes[2].plot(
        time,
        arrays["gait_phase_cycles"][window],
        lw=linewidth,
        label="gait phase [cycles]",
    )
    axes[2].set_ylabel("clock/phase")
    axes[3].plot(
        time,
        arrays["mpc_control_enabled"][window].astype(float),
        lw=linewidth,
        label="MPC output enabled",
    )
    axes[3].plot(
        time,
        arrays["policy_update_applied"][window].astype(float),
        lw=linewidth,
        label="new lower policy action",
    )
    axes[3].set_ylabel("events")
    axes[3].set_xlabel("task time = simulation time [s]")
    for axis in axes:
        axis.axvline(handoff_time, color="0.35", ls="--", lw=linewidth)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    return {
        "trace_path": str(trace_path),
        "summary_path": str(summary_path),
        "plot_path": str(plot_path),
        "summary": summary,
    }
