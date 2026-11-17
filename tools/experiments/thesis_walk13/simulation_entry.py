"""Isolated duration adapter. Reuses existing controllers without tuning them."""
from pathlib import Path
import importlib.util
import inspect
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import PROTOCOL, ROOT, STUDY, EVIDENCE
import mujoco
import numpy as np

recorded_frames=[]
original_step=mujoco.mj_step
def step_with_failure_record(model,data,*args,**kwargs):
    if not recorded_frames:
        recorded_frames.append((float(data.time),data.qpos.copy(),data.qvel.copy()))
    result=original_step(model,data,*args,**kwargs)
    recorded_frames.append((float(data.time),data.qpos.copy(),data.qvel.copy()))
    return result
mujoco.mj_step=step_with_failure_record
import disturbance_template.full_task_protocol as task_protocol
task_protocol.DEFAULT_FULL_TASK_PROTOCOL = PROTOCOL
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('thesis_existing_adapter',
    HERE / 'runtime_adapter.py')
existing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(existing)
import disturbance_predictor
import disturbance_template.full_task_recording as recording
from right_arm_runtime.sim_process import RightArmSimProcess

original_execute=RightArmSimProcess.execute
def execute_with_failure_context(self,*args,**kwargs):
    try:
        return original_execute(self,*args,**kwargs)
    except Exception as exc:
        print('SIMULATION_EXECUTION_FAILURE_TIME_S',kwargs.get('simulation_time'),flush=True)
        state={k:(v.tolist() if hasattr(v,'tolist') else v)
               for k,v in kwargs.items() if k in (
            'simulation_time','right_arm_q','right_arm_dq','q_ref','dq_ref',
            'ddq_des','qpos','qvel','reference_qacc')}
        state['error']=str(exc)
        (STUDY/'execution_failure_state.json').write_text(json.dumps(state,indent=2)+'\n')
        if recorded_frames:
            path=EVIDENCE/'failed_replays/D02_partial.npz'
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,time=np.array([v[0] for v in recorded_frames]),
                qpos=np.array([v[1] for v in recorded_frames]),
                qvel=np.array([v[2] for v in recorded_frames]))
        raise
RightArmSimProcess.execute=execute_with_failure_context

disturbance_predictor.DEFAULT_FULL_TASK_PROTOCOL = PROTOCOL
original_metadata = disturbance_predictor.FullTaskTemplatePredictor.metadata
def metadata(self):
    result = original_metadata(self)
    result.update(baseline_scope='fixed simulation task; stop at 10 s is known',
                  terminal_behavior='current measurement hold after 13 s',
                  scenario='thesis_walk10_stop3_v1')
    return result
disturbance_predictor.FullTaskTemplatePredictor.metadata = metadata

# The original smoke report has literal 6.4 s / 3.2 m checks. Reuse all
# fall, finite-data, QP and heading checks, with the new command boundary
# and nominal 5 m travel; retain the same +/-0.4 m distance tolerance.
report_source = inspect.getsource(recording.compute_smoke_summary)
for before, after in {
    't_6_398':'t_9_998', 't_6_400':'t_10_000',
    't_6_402':'t_10_002', 'anchor_t_6_396':'anchor_t_9_996',
    '6.398':'9.998', '6.400':'10.000', '6.402':'10.002', '6.396':'9.996',
    '2.8 <= xy_displacement <= 3.6':'4.6 <= xy_displacement <= 5.4',
    '"expected_displacement_m": 3.2':'"expected_displacement_m": 5.0',
    'xy_displacement - 3.2':'xy_displacement - 5.0',
    '"distance_gate_m": [2.8, 3.6]':'"distance_gate_m": [4.6, 5.4]',
}.items():
    assert before in report_source, before
    report_source = report_source.replace(before, after)
exec(compile(report_source,'<walk13 smoke checks>','exec'),recording.__dict__)
artifact_source = inspect.getsource(recording.save_full_task_smoke_artifacts)
artifact_source = artifact_source.replace('[0.0,8.0)', '[0.0,13.0)')
artifact_source = artifact_source.replace('6.4 s','10.0 s').replace('6.4s','10.0s')
exec(compile(artifact_source,'<walk13 recording metadata>','exec'),recording.__dict__)
# Old plot labels encode the 8-second scenario; omit those presentation plots.
recording.save_smoke_plots = lambda *args, **kwargs: []


# Same no-future-data baseline as the selected hardware/simulation comparison.
from types import SimpleNamespace
import yaml
from current_omega import current_omega_horizon, verify_adapter
verify_adapter()
config_path = Path(sys.argv[1]).resolve()
config = yaml.safe_load(config_path.read_text())
enabled = bool(config.get('research_current_omega_pose', False))
if enabled and config.get('research_hold_future_disturbance', False):
    raise ValueError('Two hold wrappers cannot be active together')
audit = dict(enabled=enabled, control_calls=0, future_template_values_consumed=False if enabled else None,
    vector_policy='hold current acc/omega/alpha',
    posture_policy='exp(omega_world * t) @ current_R; midpoint interval rotations')
import strategy
original_configure = strategy.configure_policy
def configure(policy, cfg):
    policy = original_configure(policy, cfg)
    if enabled:
        action = policy.compute_action
        def compute(observation, helpers):
            horizon = current_omega_horizon(helpers.disturbance_prediction[0],
                                            policy.control_dt, policy.horizon)
            proxy = SimpleNamespace(**vars(helpers))
            proxy.disturbance_prediction = horizon.nodes
            proxy.interval_disturbance_prediction = horizon.intervals
            audit['control_calls'] += 1
            return action(observation, proxy)
        policy.compute_action = compute
    return policy
strategy.configure_policy = configure
existing.configure_policy = configure
import atexit
@atexit.register
def save_adapter_audit():
    config_path.with_name(config_path.stem+'_adapter_audit.json').write_text(
        json.dumps(audit, indent=2)+'\n')

existing.base.run(extra_edits={
    '        protocol = DEFAULT_FULL_TASK_PROTOCOL\n':
        '        protocol = thesis_protocol\n',
    '        startup_pd_handoff = FixedStartupPdHandoff(startup_duration)\n':
        '        startup_pd_handoff = FixedStartupPdHandoff(startup_duration, thesis_protocol)\n',
}, extra_globals={'thesis_protocol': PROTOCOL})
