"""Isolated follow-up C study; retains the original comparison and baselines."""
from pathlib import Path
import study

if __name__ == '__main__':
    study.ROOT = study.REPO / 'evaluation/acceleration_pid_refinement_20261012'
    study.main()
