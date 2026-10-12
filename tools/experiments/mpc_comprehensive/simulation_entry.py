"""Simulation-only batch entry. It never imports the robot SDK.

Batch CPU evidence is deliberately labelled research, not formal timing proof.
The simulator, gait, contacts, actuator limits and integration are unchanged.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import disturbance_template.full_task_runtime_preflight as preflight
cpu = int(os.environ['MPC_CONTROL_CPU'])
preflight.FORMAL_CONTROL_CPU = cpu
preflight.FORMAL_FULL_TASK_RUNTIME_VERSION = 'research_batch_single_thread_not_realtime_proof_v1'

# Suppress only presentation artifacts. Numeric diagnostics remain unchanged.
import sim_support
for name in (
    'save_mpc_end_effector_task_prediction_plot',
    'save_base_disturbance_node_template_tracking_plot',
    'save_base_disturbance_interval_template_prediction_plot',
    'save_mpc_diagnostics_preview', 'save_lqr_ddq_tracking_plot',
    'save_lqr_tracking_preview', 'save_control_preview',
):
    setattr(sim_support, name, lambda *args, **kwargs: None)

source = (ROOT / 'main_sim.py').read_text()
policy_anchor = '    arm_policy = controller_setup.policy'
assert source.count(policy_anchor) == 1
source = source.replace(policy_anchor, policy_anchor +
    '\n    from strategy import configure_policy\n'
    '    arm_policy = configure_policy(arm_policy, config)')
anchor = '        predictor_metadata = disturbance_predictor.metadata()'
assert source.count(anchor) == 1
source = source.replace(anchor,
    '        from preview import wrap_predictor\n'
    '        disturbance_predictor = wrap_predictor(disturbance_predictor, config)\n' + anchor)
exec(compile(source, str(ROOT / 'main_sim.py'), 'exec'),
     {'__name__': '__main__', '__file__': str(ROOT / 'main_sim.py')})
