"""Replay the saved 10 s walk + 3 s settle task; no robot interface."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('walk13_viewer',
    HERE / 'viewer.py')
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)
viewer.EVIDENCE = ROOT / 'docs/simulation/experiments/20261117_walk10_stop3'

if __name__ == '__main__':
    viewer.main()
