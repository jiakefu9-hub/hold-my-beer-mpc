"""Reproduce the three selected simulations. No robot SDK or output.

Reuse the validated comparison entry; the two MPC configs differ only by
research_hold_future_disturbance. Save compact physical trajectories, not video.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'simple_controller_comparison'))
import study

study.ROOT = study.REPO / 'evaluation/simulation_showcase_20261103'
study.ENTRY = HERE.parent / 'c_mpc_followup/simulation_entry.py'
study.PROVENANCE_EXTRA = dict(
    purpose='matched demonstration, not hardware evidence or best-family proof',
    prediction='frozen simulation template vs measured current disturbance held',
    extras='no attitude forecast correction, no interval lead, no higher frequency',
)
original_sources = study.source_files
study.source_files = lambda: original_sources() + list(HERE.glob('*.py')) + [
    study.ENTRY, HERE.parent / 'c_mpc_followup/resampled_preview.py',
    HERE.parent / 'c_mpc_followup/efficient_c.py',
    HERE.parent / 'mpc_comprehensive/strategy.py',
]

if __name__ == '__main__':
    study.main()
