"""Isolated frequency and preview ablations; no hardware output."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'simple_controller_comparison'))
import study

study.ROOT = study.REPO/'evaluation/c_mpc_followup_20261012'
study.ENTRY = HERE/'simulation_entry.py'
study.PROVENANCE_EXTRA = dict(
    actual_control_period='per-case simulation_dt * arm_control_decimation; 2/4/6 ms',
    prediction_source='original checksummed historical 6 ms template, research resampling only',
    comparison_limit='same leg commands; actual closed-loop torso trajectories can differ')
original_sources = study.source_files
study.source_files = lambda: original_sources() + list(HERE.glob('*.py'))

if __name__ == '__main__':
    study.main()
