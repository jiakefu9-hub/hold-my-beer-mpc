"""Isolated simulation adapter; the normal and hardware launchers are untouched.

The existing full-task protocol is gated by arm_controller='mpc'. Here that
protocol tag is reused, while actual_control_family records the real algorithm.
All position policies set acceleration_controller=False and execute PD only.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import disturbance_template.full_task_runtime_preflight as preflight
preflight.FORMAL_CONTROL_CPU = int(os.environ['MPC_CONTROL_CPU'])
preflight.FORMAL_FULL_TASK_RUNTIME_VERSION = 'simple_controller_research_not_realtime_proof_v1'
import sim_support
from controllers import make_setup

factory = sim_support.create_arm_controller
sim_support.create_arm_controller = lambda *args: make_setup(factory, *args)
# Keep the same causal IMU filter for all controller families.
filter_class = sim_support.TorsoAccelerationFilter
sim_support.TorsoAccelerationFilter = lambda config, enabled, **kw: filter_class(config, enabled=True, **kw)
for name in ('save_mpc_end_effector_task_prediction_plot',
             'save_base_disturbance_node_template_tracking_plot',
             'save_base_disturbance_interval_template_prediction_plot',
             'save_mpc_diagnostics_preview', 'save_lqr_ddq_tracking_plot',
             'save_lqr_tracking_preview', 'save_control_preview'):
    setattr(sim_support, name, lambda *args, **kwargs: None)

def run(extra_edits=None, extra_globals=None):
    source = (ROOT/'main_sim.py').read_text()
    # Position PD has no inverse-dynamics worker. Preserve parent/thread checks,
    # record absent worker affinity honestly, and never call this formal timing.
    edits = {
        '        if worker_snapshot is None:\n':
        '        if worker_snapshot is None and acceleration_controller:\n',
        '            worker_affinity=worker_snapshot["cpu_affinity"],':
        '            worker_affinity=(worker_snapshot["cpu_affinity"] if worker_snapshot else None),',
        '                and right_arm_dry_warmup is None\n':
        '                and right_arm_dry_warmup is None\n                and acceleration_controller\n',
        '        if startup_pd_handoff is not None:\n':
        '        if startup_pd_handoff is not None and acceleration_controller:\n',
        '        "mujoco_contact_validation_retained": True,':
        '        "mujoco_contact_validation_retained": bool(acceleration_controller),',
        '                        mpc_control_enabled and execution_update_due\n':
        '                        acceleration_controller and mpc_control_enabled and execution_update_due\n',
    }
    edits.update(extra_edits or {})
    for before, after in edits.items():
        assert source.count(before) == 1, 'simulation integration changed'
        source = source.replace(before, after)
    exec(compile(source, str(ROOT/'main_sim.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(ROOT/'main_sim.py'), **(extra_globals or {})})


if __name__ == '__main__':
    run()
