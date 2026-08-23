"""Simulation-only scheduling and evidence for MPC-result latency experiments.

The packet contains an MPC result produced from one 6 ms source anchor.  A
fixed delay line makes that immutable packet visible on a later 2 ms physics
tick.  Torque mapping is deliberately absent from this module: activation
state mapping and certification stay in the existing runtime/process path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TRACE_SCHEMA_VERSION = "mpc_latency_trace_v1"
SUPPORTED_FIXED_DELAYS_MS = (0.0, 2.0, 4.0)


def _finite_vector(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    result = result.copy()
    result.setflags(write=False)
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_mapper_candidate_trace(error: str) -> dict[str, Any] | None:
    """Parse the compact mapper trace emitted only on NO_SAFE_TORQUE.

    ``predicted`` is the local linearization result used for candidate ordering;
    ``validated`` is the real current-state forward-dynamics result when that
    candidate was evaluated.  Parsing is diagnostic and cannot affect control.
    """
    marker = "NO_SAFE_TORQUE|CT1"
    start = str(error).find(marker)
    if start < 0:
        return None
    tokens = str(error)[start:].split(";")
    pass_names = {"F": "first_pass", "S": "second_pass", "R1": "rescue_1", "R2": "rescue_2"}
    result: dict[str, Any] = {
        "schema_version": "mapper_candidate_trace_v1",
        "predicted_candidates": [],
    }

    def number(text: str) -> float | None:
        try:
            value = float(text)
        except ValueError:
            return None
        return value if np.isfinite(value) else None

    scalar_names = {
        "B": "baseline_max_abs_qacc_rad_s2",
        "BEST": "best_validated_max_abs_qacc_rad_s2",
        "HL": "hold_last_validated_max_abs_qacc_rad_s2",
        "SH": "safe_hold_validated_max_abs_qacc_rad_s2",
    }
    for token in tokens[1:]:
        if token.startswith("MIN="):
            identity, value = token[4:].split(":", 1)
            pass_code, scale = identity.split("@", 1)
            result["minimum_predicted_candidate_type"] = (
                f"{pass_names.get(pass_code, pass_code)}_scale_{scale}"
            )
            result["minimum_predicted_max_abs_qacc_rad_s2"] = number(value)
            continue
        if "@" in token and "=" in token:
            identity, values = token.split("=", 1)
            pass_code, scale_text = identity.split("@", 1)
            predicted_text, validated_text = values.split("/", 1)
            result["predicted_candidates"].append(
                {
                    "candidate_type": f"{pass_names.get(pass_code, pass_code)}_scale_{scale_text}",
                    "pass": pass_names.get(pass_code, pass_code),
                    "scale": number(scale_text),
                    "predicted_max_abs_qacc_rad_s2": number(predicted_text),
                    "validated_max_abs_qacc_rad_s2": number(validated_text),
                }
            )
            continue
        if "=" in token:
            name, value = token.split("=", 1)
            if name in scalar_names:
                result[scalar_names[name]] = number(value)
    return result


@dataclass(frozen=True)
class MpcResultPacket:
    command_id: int
    source_time: float
    source_sample_index: int
    source_anchor_index: int
    ready_time: float
    q_ref: np.ndarray
    dq_ref: np.ndarray
    ddq_raw: np.ndarray
    ddq_des: np.ndarray
    source_right_arm_q: np.ndarray
    source_right_arm_dq: np.ndarray
    source_torso_acc: np.ndarray
    source_torso_omega: np.ndarray
    diagnostics: dict[str, Any]
    packet_ready_wall_ns: int


@dataclass(frozen=True)
class MpcResultActivation:
    packet: MpcResultPacket | None
    activated: bool
    activation_time: float
    activation_sample_index: int
    effective_delay: float | None
    dropped_count: int


class FixedMpcResultDelayLine:
    """Single-stream MPC result activation on the physics grid.

    There is intentionally no synthetic initial packet.  Until the first real
    packet is ready, the caller must keep executing startup fixed-posture PD.
    """

    def __init__(self, *, step_dt: float, requested_delay_s: float, mpc_dt: float):
        self.step_dt = float(step_dt)
        self.requested_delay_s = float(requested_delay_s)
        self.mpc_dt = float(mpc_dt)
        if not np.isfinite(self.step_dt) or self.step_dt <= 0.0:
            raise ValueError("step_dt must be finite and positive")
        if not np.isfinite(self.mpc_dt) or self.mpc_dt <= self.step_dt:
            raise ValueError("mpc_dt must be finite and greater than step_dt")
        if (
            not np.isfinite(self.requested_delay_s)
            or self.requested_delay_s < 0.0
            or self.requested_delay_s >= self.mpc_dt - 1e-12
        ):
            raise ValueError("experimental MPC-result latency must be in [0, mpc_dt)")
        self.delay_ticks = int(math.ceil(self.requested_delay_s / self.step_dt - 1e-12))
        self.quantized_delay_s = self.delay_ticks * self.step_dt
        self._pending: deque[MpcResultPacket] = deque()
        self._active: MpcResultPacket | None = None
        self._next_command_id = 1
        self._last_source_sample: int | None = None
        self._last_source_anchor: int | None = None
        self._last_observed_sample: int | None = None
        self._last_mapping_sample: int | None = None
        self._events: list[dict[str, Any]] = []

    @property
    def active_packet(self) -> MpcResultPacket | None:
        return self._active

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._events))

    def publish(
        self,
        *,
        source_time: float,
        source_sample_index: int,
        source_anchor_index: int,
        q_ref: Any,
        dq_ref: Any,
        ddq_raw: Any,
        ddq_des: Any,
        source_right_arm_q: Any,
        source_right_arm_dq: Any,
        source_torso_acc: Any,
        source_torso_omega: Any,
        diagnostics: dict[str, Any],
        packet_ready_wall_ns: int,
    ) -> MpcResultPacket:
        source_time = float(source_time)
        sample = int(source_sample_index)
        anchor = int(source_anchor_index)
        expected_time = sample * self.step_dt
        if not np.isfinite(source_time) or not np.isclose(
            source_time, expected_time, atol=1e-10, rtol=0.0
        ):
            raise ValueError("source_time must match source_sample_index on the 2 ms grid")
        if sample < 0 or anchor < 0 or sample * self.step_dt < -1e-12:
            raise ValueError("source indices must be non-negative")
        if not np.isclose(source_time / self.mpc_dt, anchor, atol=1e-10, rtol=0.0):
            raise ValueError("source_anchor_index does not match source_time")
        if self._last_source_sample is not None:
            expected_stride = int(round(self.mpc_dt / self.step_dt))
            if sample != self._last_source_sample + expected_stride or anchor != self._last_source_anchor + 1:
                raise ValueError("MPC source anchors must be strictly consecutive")
        wall_ns = int(packet_ready_wall_ns)
        if wall_ns <= 0:
            raise ValueError("packet_ready_wall_ns must be positive")
        packet = MpcResultPacket(
            command_id=self._next_command_id,
            source_time=source_time,
            source_sample_index=sample,
            source_anchor_index=anchor,
            ready_time=source_time + self.quantized_delay_s,
            q_ref=_finite_vector(q_ref, (5,), "q_ref"),
            dq_ref=_finite_vector(dq_ref, (5,), "dq_ref"),
            ddq_raw=_finite_vector(ddq_raw, (5,), "ddq_raw"),
            ddq_des=_finite_vector(ddq_des, (5,), "ddq_des"),
            source_right_arm_q=_finite_vector(source_right_arm_q, (5,), "source_right_arm_q"),
            source_right_arm_dq=_finite_vector(source_right_arm_dq, (5,), "source_right_arm_dq"),
            source_torso_acc=_finite_vector(source_torso_acc, (3,), "source_torso_acc"),
            source_torso_omega=_finite_vector(source_torso_omega, (3,), "source_torso_omega"),
            diagnostics=copy.deepcopy(diagnostics),
            packet_ready_wall_ns=wall_ns,
        )
        self._next_command_id += 1
        self._last_source_sample = sample
        self._last_source_anchor = anchor
        self._pending.append(packet)
        return packet

    def activate_ready(self, *, now: float, sample_index: int) -> MpcResultActivation:
        now = float(now)
        sample = int(sample_index)
        if not np.isfinite(now) or not np.isclose(now, sample * self.step_dt, atol=1e-10, rtol=0.0):
            raise ValueError("activation time must match sample_index on the physics grid")
        if self._last_observed_sample is not None and sample <= self._last_observed_sample:
            raise ValueError("latency scheduler time repeated or moved backwards")
        self._last_observed_sample = sample
        ready: list[MpcResultPacket] = []
        while self._pending and self._pending[0].ready_time <= now + 1e-12:
            ready.append(self._pending.popleft())
        activated = bool(ready)
        dropped = max(0, len(ready) - 1)
        if activated:
            self._active = ready[-1]
            effective = now - self._active.source_time
            self._events.append(
                {
                    "command_id": self._active.command_id,
                    "source_sample_index": self._active.source_sample_index,
                    "source_anchor_index": self._active.source_anchor_index,
                    "source_time_s": self._active.source_time,
                    "ready_time_s": self._active.ready_time,
                    "activation_sample_index": sample,
                    "activation_time_s": now,
                    "effective_delay_ms": effective * 1e3,
                    "dropped_count": dropped,
                }
            )
        else:
            effective = None if self._active is None else now - self._active.source_time
        return MpcResultActivation(
            packet=self._active,
            activated=activated,
            activation_time=now,
            activation_sample_index=sample,
            effective_delay=effective,
            dropped_count=dropped,
        )

    def mapping_update_due(self, *, sample_index: int, activated: bool, mode: str) -> bool:
        if self._active is None:
            return False
        sample = int(sample_index)
        if mode == "policy_update":
            due = bool(activated)
        elif mode == "twice_per_interval":
            due = bool(
                activated
                or self._last_mapping_sample is None
                or sample - self._last_mapping_sample >= 2
            )
        else:
            raise ValueError(f"unsupported mapper update mode: {mode}")
        if due:
            self._last_mapping_sample = sample
        return due

    def reset(self) -> None:
        """Start a new task epoch without retaining any executable packet."""
        self._pending.clear()
        self._active = None
        self._next_command_id = 1
        self._last_source_sample = None
        self._last_source_anchor = None
        self._last_observed_sample = None
        self._last_mapping_sample = None
        self._events.clear()

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "fixed",
            "definition": "MPC result availability latency with fresh-state execution",
            "step_dt_s": self.step_dt,
            "mpc_dt_s": self.mpc_dt,
            "requested_delay_ms": self.requested_delay_s * 1e3,
            "quantized_delay_ms": self.quantized_delay_s * 1e3,
            "delay_ticks": self.delay_ticks,
            "synthetic_initial_packet": False,
            "activation_state_mapper": True,
        }


class MpcLatencyTraceRecorder:
    """Versioned capture of source, ready, activation and certified timings."""

    def __init__(self, *, metadata: dict[str, Any]):
        self.metadata = copy.deepcopy(metadata)
        self._anchors: dict[int, dict[str, Any]] = {}
        self._executions: list[dict[str, Any]] = []
        self._failure: dict[str, Any] | None = None

    def begin_source(self, *, source_sample_index: int, source_anchor_index: int, source_time: float, source_sample_wall_ns: int) -> None:
        anchor = int(source_anchor_index)
        if anchor in self._anchors or (self._anchors and anchor != max(self._anchors) + 1):
            raise ValueError("trace source anchors must be unique and consecutive")
        wall_ns = int(source_sample_wall_ns)
        if wall_ns <= 0 or not np.isfinite(float(source_time)):
            raise ValueError("invalid source trace time")
        self._anchors[anchor] = {
            "source_anchor_index": anchor,
            "source_sample_index": int(source_sample_index),
            "source_time_s": float(source_time),
            "source_sample_wall_ns": wall_ns,
            "mpc_packet_ready_wall_ns": None,
            "first_certified_tau_ready_wall_ns": None,
        }

    def mark_packet_ready(self, *, source_anchor_index: int, wall_ns: int) -> None:
        row = self._anchors[int(source_anchor_index)]
        value = int(wall_ns)
        if value < row["source_sample_wall_ns"]:
            raise ValueError("packet-ready wall time precedes source sample")
        row["mpc_packet_ready_wall_ns"] = value

    def mark_first_certified(self, *, source_anchor_index: int, wall_ns: int) -> None:
        row = self._anchors[int(source_anchor_index)]
        value = int(wall_ns)
        ready = row["mpc_packet_ready_wall_ns"]
        if ready is None or value < ready:
            raise ValueError("certified torque time precedes packet-ready time")
        if row["first_certified_tau_ready_wall_ns"] is None:
            row["first_certified_tau_ready_wall_ns"] = value

    def record_execution(self, **row: Any) -> None:
        converted = _json_value(row)
        for key, value in converted.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"non-finite latency execution field: {key}")
        self._executions.append(converted)

    def record_failure(self, **row: Any) -> None:
        """Persist the one fail-closed pre-step that did not reach execution."""
        if self._failure is not None:
            raise ValueError("latency trace already contains a failure")
        converted = _json_value(row)
        for key, value in converted.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"non-finite latency failure field: {key}")
        self._failure = converted

    def payload(self) -> dict[str, Any]:
        anchors = [copy.deepcopy(self._anchors[index]) for index in sorted(self._anchors)]
        for row in anchors:
            ready = row["mpc_packet_ready_wall_ns"]
            certified = row["first_certified_tau_ready_wall_ns"]
            row["source_to_packet_ready_ms"] = (
                None if ready is None else (ready - row["source_sample_wall_ns"]) * 1e-6
            )
            row["source_to_first_certified_tau_ready_ms"] = (
                None if certified is None else (certified - row["source_sample_wall_ns"]) * 1e-6
            )
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "metadata": copy.deepcopy(self.metadata),
            "anchors": anchors,
            "executions": copy.deepcopy(self._executions),
            "failure": copy.deepcopy(self._failure),
        }

    def save(self, run_dir: Path, *, explicit_path: Path | None = None) -> dict[str, str]:
        output = Path(run_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        payload = self.payload()
        wrapper = {"payload": payload, "payload_sha256": _canonical_sha256(payload)}
        json_path = output / "mpc_latency_trace.json"
        csv_path = output / "mpc_latency_trace.csv"
        summary_path = output / "mpc_latency_summary.json"
        json_path.write_text(json.dumps(wrapper, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        anchors = payload["anchors"]
        if anchors:
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(anchors[0]))
                writer.writeheader()
                writer.writerows(anchors)
        ready = np.asarray([r["source_to_packet_ready_ms"] for r in anchors if r["source_to_packet_ready_ms"] is not None], dtype=np.float64)
        certified = np.asarray([r["source_to_first_certified_tau_ready_ms"] for r in anchors if r["source_to_first_certified_tau_ready_ms"] is not None], dtype=np.float64)
        def stats(values: np.ndarray) -> dict[str, float | int]:
            return {
                "count": int(values.size),
                "mean_ms": float(np.mean(values)) if values.size else 0.0,
                "p95_ms": float(np.percentile(values, 95)) if values.size else 0.0,
                "p99_ms": float(np.percentile(values, 99)) if values.size else 0.0,
                "max_ms": float(np.max(values)) if values.size else 0.0,
            }
        summary = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "payload_sha256": wrapper["payload_sha256"],
            "anchor_count": len(anchors),
            "execution_count": len(payload["executions"]),
            "failed_pre_step": payload["failure"] is not None,
            "failure": copy.deepcopy(payload["failure"]),
            "source_to_packet_ready": stats(ready),
            "source_to_first_certified_tau_ready": stats(certified),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if explicit_path is not None:
            target = Path(explicit_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(json_path.read_bytes())
        return {"json": str(json_path), "csv": str(csv_path), "summary": str(summary_path)}

    def save_failure_snapshot(
        self,
        run_dir: Path,
        *,
        arrays: dict[str, Any],
        explicit_path: Path | None = None,
    ) -> dict[str, str]:
        """Save trace plus lossless numeric inputs for deterministic diagnosis."""
        if self._failure is None:
            raise ValueError("failure metadata must be recorded before snapshot")
        output = Path(run_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        snapshot_path = output / "mpc_latency_failure_state.npz"
        converted = {
            str(name): np.asarray(values).copy() for name, values in arrays.items()
        }
        np.savez_compressed(snapshot_path, **converted)
        snapshot_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        self._failure["state_snapshot"] = {
            "path": str(snapshot_path),
            "sha256": snapshot_sha256,
            "array_names": sorted(converted),
            "nonfinite_count": {
                name: int(np.count_nonzero(~np.isfinite(values)))
                for name, values in converted.items()
            },
        }
        paths = self.save(run_dir, explicit_path=explicit_path)
        paths["failure_state"] = str(snapshot_path)
        return paths


def save_fail_closed_process_trace(
    recorder: MpcLatencyTraceRecorder,
    run_dir: Path,
    *,
    event: dict[str, Any],
    arrays: dict[str, Any],
    model: Any,
    data: Any,
    right_arm_qvel_indices: Any,
    right_arm_ctrl_indices: Any,
    torque_limits: Any,
    max_abs_qacc: float,
    explicit_path: Path | None = None,
) -> dict[str, str]:
    """Capture one process failure without changing the control decision.

    The independent ``mj_forward`` calls are diagnostic only.  The caller
    re-raises the original process error, so no candidate from this function is
    eligible for execution.
    """
    import mujoco

    snapshot = {
        str(name): np.asarray(values).copy() for name, values in arrays.items()
    }
    qvel_indices = np.asarray(right_arm_qvel_indices, dtype=np.int32)
    ctrl_indices = np.asarray(right_arm_ctrl_indices, dtype=np.int32)
    limits = np.asarray(torque_limits, dtype=np.float64)
    diagnostic_error = None
    audit_qacc: dict[str, np.ndarray] = {}
    try:
        scratch = mujoco.MjData(model)

        def current_state_qacc(tau: np.ndarray) -> np.ndarray:
            scratch.time = float(data.time)
            scratch.qpos[:] = snapshot["qpos"]
            scratch.qvel[:] = snapshot["qvel"]
            if model.na and "act" in snapshot:
                scratch.act[:] = snapshot["act"]
            scratch.qacc_warmstart[:] = snapshot["qacc_warmstart"]
            scratch.ctrl[:] = snapshot["fixed_ctrl"]
            scratch.qfrc_applied[:] = snapshot["qfrc_applied"]
            scratch.xfrc_applied[:] = snapshot["xfrc_applied"]
            clipped = np.clip(tau, limits[:, 0], limits[:, 1])
            scratch.ctrl[ctrl_indices] = clipped
            mujoco.mj_forward(model, scratch)
            return scratch.qacc[qvel_indices].copy()

        for name in (
            "mapper_safe_hold_tau",
            "fixed_posture_pd_tau",
            "previous_executed_tau",
        ):
            tau = snapshot[name]
            if tau.size:
                audit_qacc[name.removesuffix("_tau") + "_qacc"] = (
                    current_state_qacc(tau)
                )
    except Exception as error:
        diagnostic_error = f"{type(error).__name__}: {error}"

    snapshot.update(audit_qacc)

    def qacc_summary(name: str) -> tuple[Any, float | None, bool]:
        values = audit_qacc.get(name)
        if values is None:
            return None, None, False
        finite = bool(np.all(np.isfinite(values)))
        maximum = float(np.max(np.abs(values))) if finite else None
        return values if finite else None, maximum, bool(
            finite and maximum <= float(max_abs_qacc)
        )

    safe_values, safe_maximum, safe_certified = qacc_summary(
        "mapper_safe_hold_qacc"
    )
    fixed_values, fixed_maximum, fixed_certified = qacc_summary(
        "fixed_posture_pd_qacc"
    )
    previous_values, previous_maximum, previous_certified = qacc_summary(
        "previous_executed_qacc"
    )
    failure = copy.deepcopy(event)
    failure.update(
        {
            "mapper_safe_hold_definition": "active q_ref/dq_ref joint PD",
            "mapper_safe_hold_tau_nm": snapshot["mapper_safe_hold_tau"],
            "mapper_safe_hold_qacc_rad_s2": safe_values,
            "mapper_safe_hold_max_abs_qacc_rad_s2": safe_maximum,
            "mapper_safe_hold_limit_rad_s2": float(max_abs_qacc),
            "mapper_safe_hold_certified": safe_certified,
            "previous_tau_qacc_rad_s2": previous_values,
            "previous_tau_max_abs_qacc_rad_s2": previous_maximum,
            "previous_tau_certified": previous_certified,
            "fixed_posture_pd_diagnostic_tau_nm": snapshot[
                "fixed_posture_pd_tau"
            ],
            "fixed_posture_pd_diagnostic_qacc_rad_s2": fixed_values,
            "fixed_posture_pd_diagnostic_max_abs_qacc_rad_s2": (
                fixed_maximum
            ),
            "fixed_posture_pd_diagnostic_certified": fixed_certified,
            "diagnostic_forward_method": "independent current-state mj_forward",
            "diagnostic_error": diagnostic_error,
            "right_arm_d_ctrl_written": False,
            "mj_step_performed": False,
            "applied_tau_nm": None,
            "final_output_certified": False,
        }
    )
    recorder.record_failure(**failure)
    return recorder.save_failure_snapshot(
        run_dir, arrays=snapshot, explicit_path=explicit_path
    )


def load_and_validate_trace(path: Path) -> dict[str, Any]:
    wrapper = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(wrapper) != {"payload", "payload_sha256"}:
        raise ValueError("latency trace wrapper fields mismatch")
    payload = wrapper["payload"]
    if payload.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError("latency trace schema mismatch")
    if _canonical_sha256(payload) != wrapper["payload_sha256"]:
        raise ValueError("latency trace checksum mismatch")
    previous = None
    for row in payload.get("anchors", []):
        anchor = int(row["source_anchor_index"])
        if previous is not None and anchor != previous + 1:
            raise ValueError("latency trace has missing or reordered anchors")
        previous = anchor
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"latency trace contains non-finite {key}")
    return payload


def validate_experimental_latency_cli(*, mode: str, delay_ms: float | None, capture_path: str | None, full_task: bool, predictor: str | None, runtime_mode: str | None) -> None:
    if mode not in {"off", "fixed"}:
        raise ValueError("only off|fixed are implemented through L1-C")
    enabled = mode == "fixed"
    if enabled and delay_ms not in SUPPORTED_FIXED_DELAYS_MS:
        raise ValueError("fixed experimental latency must be exactly 0, 2, or 4 ms")
    if not enabled and delay_ms is not None:
        raise ValueError("--experimental-mpc-latency-ms requires fixed mode")
    if (enabled or capture_path is not None) and not full_task:
        raise ValueError("MPC latency experiment requires --full-task-smoke")
    if enabled and predictor not in (None, "full_task_template"):
        raise ValueError("MPC latency experiment requires full_task_template")
    if enabled and runtime_mode not in (None, "process"):
        raise ValueError("MPC latency experiment requires process runtime")
