"""Small deterministic command schedule shared by collection and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GAIT_PERIOD = 0.8
SCHEDULE_DURATION = 4.6
SCHEDULE_SEGMENT_NAMES = np.asarray(
    (
        "heading_warmup",
        "start_ramp",
        "steady_walking",
        "velocity_change",
        "stop_ramp",
        "stopped",
    )
)
REQUIRED_SCHEDULE_SEGMENT_IDS = np.asarray((1, 2, 3, 4, 5), dtype=np.int64)


@dataclass(frozen=True)
class CommandState:
    command: np.ndarray
    segment_id: int


def _lerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return (1.0 - fraction) * start + fraction * end


def command_schedule(
    time_s: float,
    nominal_command: np.ndarray,
    changed_command: np.ndarray | None = None,
) -> CommandState:
    """Cover heading warmup, start, steady, velocity change, stop and stopped."""
    nominal = np.asarray(nominal_command, dtype=np.float64)
    stopped = np.zeros(3, dtype=np.float64)
    changed = (
        np.array([0.55 * nominal[0], 0.10, -0.04], dtype=np.float64)
        if changed_command is None
        else np.asarray(changed_command, dtype=np.float64)
    )
    if time_s < 0.8:
        return CommandState(stopped.copy(), 0)
    if time_s < 1.4:
        return CommandState(
            _lerp(stopped, nominal, (time_s - 0.8) / 0.6), 1
        )
    if time_s < 2.2:
        return CommandState(nominal.copy(), 2)
    if time_s < 3.0:
        return CommandState(
            _lerp(nominal, changed, (time_s - 2.2) / 0.8), 3
        )
    if time_s < 3.6:
        return CommandState(
            _lerp(changed, stopped, (time_s - 3.0) / 0.6), 4
        )
    return CommandState(stopped.copy(), 5)
