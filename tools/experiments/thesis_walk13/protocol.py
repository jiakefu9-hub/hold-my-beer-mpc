"""Separate simulation scenario: 10 s forward, 3 s stop, 60 ms preview tail."""
from pathlib import Path
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from disturbance_template.full_task_protocol import FullTaskProtocol

class Walk13Protocol(FullTaskProtocol):
    def validate_task_time(self, task_time, *, allow_endpoint=True):
        # Accumulating 6,530 MuJoCo 2 ms steps creates picosecond drift.
        # Snap only values already within 1 ns of the exact physics grid;
        # this also preserves strict policy-event timestamps after 8 seconds.
        value = float(task_time)
        tick = round(value/self.physics_dt)*self.physics_dt
        if abs(value-tick) < 1e-9:
            value = tick
        return super().validate_task_time(value, allow_endpoint=allow_endpoint)

PROTOCOL = Walk13Protocol(stop_time=10., headline_end=13., record_end=13.06,
    protocol_version='full_task_direct_step_walk10_stop3_v1')
EVIDENCE = ROOT / 'docs/simulation/experiments/20261117_walk10_stop3'
STUDY = ROOT / 'evaluation/thesis_walk10_stop3_20261117'
variant = os.environ.get('THESIS_WALK13_VARIANT', '')
if variant:
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', variant):
        raise ValueError('invalid isolated scenario variant')
    EVIDENCE = EVIDENCE / 'variants' / variant
    STUDY = ROOT / 'evaluation' / ('thesis_walk10_stop3_' + variant)
