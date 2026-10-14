"""Isolated offline simulation entry; production factories remain unchanged."""
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent),str(ROOT/'tools/g1_commissioning')]
from delta_acceleration import DeltaSparsePolicy
config=yaml.safe_load(Path(sys.argv[1]).read_text())
class ResearchPolicy(DeltaSparsePolicy):
    delta_weight=float(config.get('research_delta_acceleration_weight',0.))
import arm_mpc
arm_mpc.ArmMPCPolicy=ResearchPolicy
# Reuse established offline protocol; not the field learned predictor.
import importlib.util
path=ROOT/'tools/experiments/simple_controller_comparison/simulation_entry.py'
spec=importlib.util.spec_from_file_location('offline_simulation_entry',path)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
entry.run()
