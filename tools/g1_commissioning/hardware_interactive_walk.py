"""M10 wall-clock walk/pause schedule and local terminal input; no SDK imports."""
from dataclasses import dataclass, asdict
import math
import os
import select
import sys
import termios
import threading
import tty

from hardware_pid_control import (
    WALK_START_S, WALK_STOP_S, RELEASE_START_S, FORWARD_SPEED_M_S, stage,
)

WALK_BUDGET_S = 20.0
PROTOCOL = dict(
    name='M10_space_pause_walk_v1', forward_command_budget_s=WALK_BUDGET_S,
    pause_key='SPACE', finish_key='Ctrl+C', pause_keeps_arm_control=True,
    timer='host commanded-walk intervals; excludes pauses, not measured foot motion',
    stop_settle_s=RELEASE_START_S-WALK_STOP_S,
    primary_metric_window_s=None,
)


@dataclass(frozen=True)
class WalkSnapshot:
    phase: str
    walking_elapsed_s: float
    plan_time_s: float
    release_start_s: float | None
    revision: int

    def evidence(self):
        return asdict(self)

    def motion(self, now_s, correction, inhibited=False):
        if not math.isfinite(float(correction)):
            raise ValueError('nonfinite heading correction')
        walking = self.phase == 'forward_walk' and not inhibited
        heading = self.phase in ('forward_walk', 'stop_settle') and not inhibited
        remaining = (WALK_BUDGET_S-self.walking_elapsed_s if walking else
                     self.release_start_s-now_s if heading else .2)
        return dict(vx_m_s=FORWARD_SPEED_M_S if walking else 0.,
                    yaw_rate_rad_s=float(correction) if heading else 0.,
                    duration_s=min(.2, max(.001, remaining)),
                    walking_active=walking, heading_hold_active=heading)


class PausableWalk:
    """Only the lifecycle clock pauses; sensor stamps and controller dt never do.

    Existing plan phase time is capped below WALK_STOP_S while this schedule is
    active, so its established handoff runs once and it keeps solving each tick.
    Stop-settle then uses the original 15..18 s plan phases at real-time speed.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._accumulated = 0.
        self._segment_start = WALK_START_S
        self._paused = False
        self._finished_at = None
        self._last_toggle = -math.inf
        self._revision = 0

    def _snapshot(self, now_s):
        now_s = float(now_s)
        if not math.isfinite(now_s) or now_s < 0:
            raise ValueError('invalid interactive walk time')
        elapsed = self._accumulated
        if not self._paused:
            elapsed += max(0., now_s-self._segment_start)
        if self._finished_at is None and elapsed >= WALK_BUDGET_S:
            self._finished_at = self._segment_start + WALK_BUDGET_S-self._accumulated
        if self._finished_at is not None:
            release = self._finished_at + RELEASE_START_S-WALK_STOP_S
            plan_s = WALK_STOP_S+max(0., now_s-self._finished_at)
            phase = 'stop_settle' if now_s < release else 'release_due'
            return WalkSnapshot(phase, WALK_BUDGET_S, plan_s, release, self._revision)
        phase = stage(now_s) if now_s < WALK_START_S else (
            'paused_hold' if self._paused else 'forward_walk')
        return WalkSnapshot(phase, elapsed, min(now_s, WALK_STOP_S-.001),
                            None, self._revision)

    def snapshot(self, now_s):
        with self._lock:
            return self._snapshot(now_s)

    def toggle(self, now_s):
        with self._lock:
            before = self._snapshot(now_s)
            if now_s < WALK_START_S or self._finished_at is not None:
                return False, before, 'outside_walk_phase'
            if now_s-self._last_toggle < .3:
                return False, before, 'key_debounce'
            self._accumulated = before.walking_elapsed_s
            self._segment_start = float(now_s)
            self._paused = not self._paused
            self._last_toggle = float(now_s)
            self._revision += 1
            return True, self._snapshot(now_s), 'space'


class SpacebarInput:
    """Cbreak keeps Ctrl+C signals; restoration also runs after software faults."""
    def __init__(self, fd=None):
        self.fd = sys.stdin.fileno() if fd is None else fd
        self.original = None

    @staticmethod
    def require_terminal():
        if not sys.stdin.isatty():
            raise ValueError('M10 requires an interactive terminal for SPACE/Ctrl+C')

    def open(self):
        if not os.isatty(self.fd):
            raise ValueError('M10 keyboard input is not a terminal')
        self.original = termios.tcgetattr(self.fd)
        try:
            # Discard keys typed before activation; confirmation already read.
            termios.tcflush(self.fd, termios.TCIFLUSH)
            tty.setcbreak(self.fd, termios.TCSANOW)
        except BaseException:
            self.close()
            raise

    def poll(self, timeout=.05):
        if not select.select([self.fd], [], [], timeout)[0]:
            return b''
        data = os.read(self.fd, 128)
        if not data:
            raise RuntimeError('M10 keyboard terminal disconnected')
        return data

    def close(self):
        if self.original is not None:
            original, self.original = self.original, None
            termios.tcsetattr(self.fd, termios.TCSANOW, original)
