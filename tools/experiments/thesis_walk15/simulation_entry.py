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
sys.path.insert(0, str(HERE.parent / 'c_mpc_followup'))
spec = importlib.util.spec_from_file_location('thesis_existing_adapter',
    HERE.parent / 'c_mpc_followup/simulation_entry.py')
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
    result.update(baseline_scope='fixed simulation task; stop at 12 s is known',
                  terminal_behavior='current measurement hold after 15 s',
                  scenario='thesis_walk12_stop3_v1')
    return result
disturbance_predictor.FullTaskTemplatePredictor.metadata = metadata

# The original smoke report has literal 6.4 s / 3.2 m checks. Reuse all
# fall, finite-data, QP and heading checks, with the new command boundary
# and nominal 6 m travel; retain the same +/-0.4 m distance tolerance.
report_source = inspect.getsource(recording.compute_smoke_summary)
for before, after in {
    't_6_398':'t_11_998', 't_6_400':'t_12_000',
    't_6_402':'t_12_002', 'anchor_t_6_396':'anchor_t_11_994',
    '6.398':'11.998', '6.400':'12.000', '6.402':'12.002', '6.396':'11.994',
    '2.8 <= xy_displacement <= 3.6':'5.6 <= xy_displacement <= 6.4',
    '"expected_displacement_m": 3.2':'"expected_displacement_m": 6.0',
    'xy_displacement - 3.2':'xy_displacement - 6.0',
    '"distance_gate_m": [2.8, 3.6]':'"distance_gate_m": [5.6, 6.4]',
}.items():
    assert before in report_source, before
    report_source = report_source.replace(before, after)
exec(compile(report_source,'<walk15 smoke checks>','exec'),recording.__dict__)
artifact_source = inspect.getsource(recording.save_full_task_smoke_artifacts)
artifact_source = artifact_source.replace('[0.0,8.0)', '[0.0,15.0)')
artifact_source = artifact_source.replace('6.4 s','12.0 s').replace('6.4s','12.0s')
exec(compile(artifact_source,'<walk15 recording metadata>','exec'),recording.__dict__)
# Old plot labels encode the 8-second scenario; omit those presentation plots.
recording.save_smoke_plots = lambda *args, **kwargs: []

existing.base.run(extra_edits={
    '        protocol = DEFAULT_FULL_TASK_PROTOCOL\n':
        '        protocol = thesis_protocol\n',
    '        startup_pd_handoff = FixedStartupPdHandoff(startup_duration)\n':
        '        startup_pd_handoff = FixedStartupPdHandoff(startup_duration, thesis_protocol)\n',
}, extra_globals={'thesis_protocol': PROTOCOL})
