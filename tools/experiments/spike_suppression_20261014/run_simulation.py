"""Small single-factor closed-loop simulation study, no hardware output."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'tools/experiments/simple_controller_comparison'),str(ROOT/'tools/g1_commissioning')]
import study
from delta_acceleration import algebra_check
if __name__=='__main__':
    print(algebra_check(),flush=True)
    study.ROOT=ROOT/'evaluation/spike_suppression_20261014'
    study.ENTRY=HERE/'simulation_entry.py'
    original_sources=study.source_files
    study.source_files=lambda:original_sources()+list(HERE.glob('*.py'))
    study.main()
