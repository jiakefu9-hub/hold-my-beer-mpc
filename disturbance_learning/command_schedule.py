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
class CommandScheduleTiming:
    """Causal transition boundaries for one command schedule."""

    start_begin: float = 0.8
    start_end: float = 1.4
    change_begin: float = 2.2
    change_end: float = 3.0
    stop_begin: float = 3.0
    stop_end: float = 3.6
    run_end: float = 4.8

    def __post_init__(self) -> None:
        boundaries = (
            self.start_begin,
            self.start_end,
            self.change_begin,
            self.change_end,
            self.stop_begin,
            self.stop_end,
            self.run_end,
        )
        if not all(np.isfinite(boundaries)):
            raise ValueError("command schedule boundaries must be finite")
        if not (
            0.0 <= self.start_begin < self.start_end <= self.change_begin
            < self.change_end <= self.stop_begin < self.stop_end
            <= self.run_end
        ):
            raise ValueError("command schedule boundaries are not ordered")

    def stage_windows(self) -> dict[str, tuple[float, float]]:
        return {
            "start": (self.start_begin, self.start_end),
            "steady": (self.start_end, self.change_begin),
            "velocity_change": (self.change_begin, self.change_end),
            "changed_steady": (self.change_end, self.stop_begin),
            "stop": (self.stop_begin, self.stop_end),
            "stopped": (self.stop_end, self.run_end),
        }


DEFAULT_SCHEDULE_TIMING = CommandScheduleTiming()


@dataclass(frozen=True)
class CommandScheduleProfile:
    """A deliberately unseen command/timing combination for B2 validation."""

    name: str
    timing: CommandScheduleTiming
    start_command: tuple[float, float, float]
    changed_command: tuple[float, float, float]


GENERALIZATION_SCHEDULE_PROFILES = {
    profile.name: profile
    for profile in (
        CommandScheduleProfile(
            name="delayed_fast_lateral",
            timing=CommandScheduleTiming(
                start_begin=1.10,
                start_end=1.55,
                change_begin=2.45,
                change_end=2.90,
                stop_begin=3.65,
                stop_end=4.15,
                run_end=5.60,
            ),
            start_command=(0.58, -0.04, 0.03),
            changed_command=(0.34, 0.09, -0.05),
        ),
        CommandScheduleProfile(
            name="slow_start_speedup",
            timing=CommandScheduleTiming(
                start_begin=0.90,
                start_end=1.75,
                change_begin=2.25,
                change_end=3.10,
                stop_begin=3.55,
                stop_end=3.95,
                run_end=5.60,
            ),
            start_command=(0.34, 0.06, -0.02),
            changed_command=(0.50, -0.08, 0.045),
        ),
        CommandScheduleProfile(
            name="late_multiaxis",
            timing=CommandScheduleTiming(
                start_begin=1.30,
                start_end=1.85,
                change_begin=2.80,
                change_end=3.35,
                stop_begin=4.20,
                stop_end=4.85,
                run_end=5.60,
            ),
            start_command=(0.47, -0.07, 0.045),
            changed_command=(0.22, 0.11, 0.0),
        ),
    )
}


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
    timing: CommandScheduleTiming = DEFAULT_SCHEDULE_TIMING,
) -> CommandState:
    """Cover heading warmup, start, steady, velocity change, stop and stopped."""
    nominal = np.asarray(nominal_command, dtype=np.float64)
    stopped = np.zeros(3, dtype=np.float64)
    changed = (
        np.array([0.55 * nominal[0], 0.10, -0.04], dtype=np.float64)
        if changed_command is None
        else np.asarray(changed_command, dtype=np.float64)
    )
    if time_s < timing.start_begin:
        return CommandState(stopped.copy(), 0)
    if time_s < timing.start_end:
        return CommandState(
            _lerp(
                stopped,
                nominal,
                (time_s - timing.start_begin)
                / (timing.start_end - timing.start_begin),
            ),
            1,
        )
    if time_s < timing.change_begin:
        return CommandState(nominal.copy(), 2)
    if time_s < timing.change_end:
        return CommandState(
            _lerp(
                nominal,
                changed,
                (time_s - timing.change_begin)
                / (timing.change_end - timing.change_begin),
            ),
            3,
        )
    if time_s < timing.stop_begin:
        # Retain the existing six segment ids.  A changed-command hold belongs
        # to segment 3; evaluation code uses explicit timing windows.
        return CommandState(changed.copy(), 3)
    if time_s < timing.stop_end:
        return CommandState(
            _lerp(
                changed,
                stopped,
                (time_s - timing.stop_begin)
                / (timing.stop_end - timing.stop_begin),
            ),
            4,
        )
    return CommandState(stopped.copy(), 5)
