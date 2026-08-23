"""Offline audit for state-only Unitree G1 inspection traces.

The audit is intentionally unable to modify the hardware configuration or turn
observations into site verification.  It checks persisted evidence for schema,
shape, finite values and stream monotonicity, then lists the facts that still
require a person and the target robot.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping


TRACE_AUDIT_SCHEMA = "unitree_hardware_state_trace_audit_v1"
MOTOR_COUNT = 35
RIGHT_ARM_MOTOR_INDICES = (22, 23, 24, 25, 26)


class HardwareStateTraceAuditError(RuntimeError):
    """The persisted trace is incomplete or internally inconsistent."""


_VECTOR_FIELDS = {
    "q_rad": MOTOR_COUNT,
    "dq_rad_s": MOTOR_COUNT,
    "ddq_rad_s2": MOTOR_COUNT,
    "tau_est_nm": MOTOR_COUNT,
    "imu_quaternion_wxyz": 4,
    "imu_gyroscope_rad_s": 3,
    "imu_accelerometer_raw_m_s2": 3,
    "imu_rpy_rad": 3,
}


def _finite_vector(record: Mapping, name: str, size: int) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in record[name])
    except (KeyError, TypeError, ValueError) as error:
        raise HardwareStateTraceAuditError(f"invalid {name}") from error
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise HardwareStateTraceAuditError(
            f"{name} must contain {size} finite values"
        )
    return result


def load_state_trace(path: str | Path) -> list[dict]:
    trace_path = Path(path).expanduser().resolve()
    records: list[dict] = []
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise HardwareStateTraceAuditError(
                    f"invalid JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise HardwareStateTraceAuditError(
                    f"line {line_number} is not a JSON object"
                )
            records.append(record)
    if not records:
        raise HardwareStateTraceAuditError("trace contains no state samples")
    return records


def audit_state_trace(
    records: Iterable[Mapping],
    *,
    source_kind: str,
    bridge_summary: Mapping | None = None,
) -> dict:
    """Audit recorded facts without approving model/frame/mode contracts."""

    if source_kind not in {"unverified_real_capture", "synthetic_test_fixture"}:
        raise ValueError("source_kind must explicitly label real or synthetic data")
    items = list(records)
    if not items:
        raise HardwareStateTraceAuditError("trace contains no state samples")

    sample_ids: list[int] = []
    timestamps: list[int] = []
    ticks: list[int] = []
    state_ages_ms: list[float] = []
    quaternion_norms: list[float] = []
    modes_pr: set[int] = set()
    modes_machine: set[int] = set()
    max_abs_dq = 0.0
    max_abs_tau = 0.0
    max_case_temp = -math.inf
    max_winding_temp = -math.inf

    for index, record in enumerate(items):
        vectors = {
            name: _finite_vector(record, name, size)
            for name, size in _VECTOR_FIELDS.items()
        }
        temperatures = record.get("motor_temperature_c")
        if not isinstance(temperatures, list) or len(temperatures) != MOTOR_COUNT:
            raise HardwareStateTraceAuditError(
                "motor_temperature_c must contain 35 pairs"
            )
        parsed_temperatures: list[tuple[float, float]] = []
        for pair in temperatures:
            if not isinstance(pair, list) or len(pair) != 2:
                raise HardwareStateTraceAuditError(
                    "motor_temperature_c must contain 35 pairs"
                )
            values = (float(pair[0]), float(pair[1]))
            if not all(math.isfinite(value) for value in values):
                raise HardwareStateTraceAuditError(
                    "motor_temperature_c contains NaN/Inf"
                )
            parsed_temperatures.append(values)

        try:
            sample_id = int(record["sample_id"])
            timestamp = int(record["source_monotonic_timestamp_ns"])
            read_timestamp = int(record["read_monotonic_ns"])
            robot_tick = int(record["robot_tick"])
            mode_pr = int(record["mode_pr"])
            mode_machine = int(record["mode_machine"])
        except (KeyError, TypeError, ValueError) as error:
            raise HardwareStateTraceAuditError(
                f"invalid scalar identity at record {index}"
            ) from error
        if sample_id <= 0 or timestamp <= 0 or read_timestamp < timestamp:
            raise HardwareStateTraceAuditError(
                f"invalid sample/time identity at record {index}"
            )
        if not 0 <= robot_tick <= 0xFFFFFFFF:
            raise HardwareStateTraceAuditError("robot_tick must fit uint32")
        if not 0 <= mode_pr <= 0xFF or not 0 <= mode_machine <= 0xFF:
            raise HardwareStateTraceAuditError("mode values must fit uint8")
        if sample_ids and sample_id <= sample_ids[-1]:
            raise HardwareStateTraceAuditError("sample_id repeated or regressed")
        if timestamps and timestamp <= timestamps[-1]:
            raise HardwareStateTraceAuditError("timestamp repeated or regressed")
        if ticks:
            delta = (robot_tick - ticks[-1]) & 0xFFFFFFFF
            if delta == 0 or delta >= (1 << 31):
                raise HardwareStateTraceAuditError(
                    "robot_tick repeated or regressed"
                )

        mapped = record.get("mapped_right_arm")
        if mapped is not None:
            mapped_values = tuple(float(value) for value in mapped)
            expected = tuple(vectors["q_rad"][i] for i in RIGHT_ARM_MOTOR_INDICES)
            if mapped_values != expected:
                raise HardwareStateTraceAuditError(
                    "mapped_right_arm does not match q slots 22..26"
                )

        quaternion_norm = math.sqrt(
            sum(value * value for value in vectors["imu_quaternion_wxyz"])
        )
        if quaternion_norm <= 1e-9:
            raise HardwareStateTraceAuditError("IMU quaternion has zero norm")
        sample_ids.append(sample_id)
        timestamps.append(timestamp)
        ticks.append(robot_tick)
        modes_pr.add(mode_pr)
        modes_machine.add(mode_machine)
        state_ages_ms.append((read_timestamp - timestamp) * 1e-6)
        quaternion_norms.append(quaternion_norm)
        max_abs_dq = max(max_abs_dq, max(abs(x) for x in vectors["dq_rad_s"]))
        max_abs_tau = max(max_abs_tau, max(abs(x) for x in vectors["tau_est_nm"]))
        max_case_temp = max(max_case_temp, max(x[0] for x in parsed_temperatures))
        max_winding_temp = max(
            max_winding_temp, max(x[1] for x in parsed_temperatures)
        )

    if bridge_summary is not None:
        if bridge_summary.get("output_capability") != "absent":
            raise HardwareStateTraceAuditError(
                "bridge summary does not prove output capability absent"
            )
        valid_count = int(bridge_summary.get("lowstate_crc_valid_count", -1))
        rejected_count = int(
            bridge_summary.get("lowstate_crc_rejected_count", -1)
        )
        paired_count = int(bridge_summary.get("paired_state_count", -1))
        if valid_count < len(items) or paired_count < len(items) or rejected_count < 0:
            raise HardwareStateTraceAuditError(
                "bridge counters do not cover the persisted trace"
            )

    dt_ms = [
        (current - previous) * 1e-6
        for previous, current in zip(timestamps[:-1], timestamps[1:])
    ]
    tick_deltas = [
        (current - previous) & 0xFFFFFFFF
        for previous, current in zip(ticks[:-1], ticks[1:])
    ]
    return {
        "schema": TRACE_AUDIT_SCHEMA,
        "status": "PASS",
        "source_kind": source_kind,
        "offline_trace_contract_passed": True,
        "hardware_session_verified": False,
        "verification_flags_modified": False,
        "command_output_observed": None,
        "output_capability_absent_from_bridge_summary": (
            bridge_summary is not None
        ),
        "sample_count": len(items),
        "first_sample_id": sample_ids[0],
        "last_sample_id": sample_ids[-1],
        "observed_mode_pr_candidates": sorted(modes_pr),
        "observed_mode_machine_candidates": sorted(modes_machine),
        "state_age_ms": {
            "min": min(state_ages_ms),
            "mean": fmean(state_ages_ms),
            "max": max(state_ages_ms),
        },
        "source_dt_ms": {
            "min": min(dt_ms) if dt_ms else None,
            "mean": fmean(dt_ms) if dt_ms else None,
            "max": max(dt_ms) if dt_ms else None,
        },
        "robot_tick_delta": {
            "min": min(tick_deltas) if tick_deltas else None,
            "max": max(tick_deltas) if tick_deltas else None,
        },
        "quaternion_norm": {
            "min": min(quaternion_norms),
            "max": max(quaternion_norms),
        },
        "dq_abs_max_rad_s": max_abs_dq,
        "tau_est_abs_max_nm": max_abs_tau,
        "motor_case_temperature_c_max": max_case_temp,
        "motor_winding_temperature_c_max": max_winding_temp,
        "site_gates_remaining": [
            "CONFIRM_TARGET_ROBOT_MODEL_AND_FIRMWARE",
            "CONFIRM_35_SLOT_INDEX_AND_MOTOR_SIGN",
            "CONFIRM_ROBOT_TICK_AND_MODE_SEMANTICS",
            "CONFIRM_TORSO_IMU_FRAME_GRAVITY_AND_LEVER_ARM",
            "MANUALLY_REVIEW_REAL_SESSION_BEFORE_FLAGS_CHANGE",
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_state_trace_files(
    trace_path: str | Path,
    *,
    source_kind: str,
    bridge_summary_path: str | Path | None = None,
) -> dict:
    resolved_trace = Path(trace_path).expanduser().resolve()
    bridge_summary = None
    resolved_bridge = None
    if bridge_summary_path is not None:
        resolved_bridge = Path(bridge_summary_path).expanduser().resolve()
        with resolved_bridge.open("r", encoding="utf-8") as stream:
            bridge_summary = json.load(stream)
        if not isinstance(bridge_summary, dict):
            raise HardwareStateTraceAuditError("bridge summary is not an object")
    report = audit_state_trace(
        load_state_trace(resolved_trace),
        source_kind=source_kind,
        bridge_summary=bridge_summary,
    )
    report["trace_path"] = str(resolved_trace)
    report["trace_sha256"] = _sha256(resolved_trace)
    report["bridge_summary_path"] = (
        str(resolved_bridge) if resolved_bridge is not None else None
    )
    report["bridge_summary_sha256"] = (
        _sha256(resolved_bridge) if resolved_bridge is not None else None
    )
    return report


__all__ = (
    "HardwareStateTraceAuditError",
    "TRACE_AUDIT_SCHEMA",
    "audit_state_trace",
    "audit_state_trace_files",
    "load_state_trace",
)
