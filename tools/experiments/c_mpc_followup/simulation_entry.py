"""SIMULATION ONLY: optional 4/2 ms C and matched MPC preview ablations."""
from dataclasses import replace
from pathlib import Path
import sys
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent/'simple_controller_comparison'))
import simulation_entry as base
from efficient_c import EfficientAccelerationPID
sys.path.insert(0,str(HERE.parent/'mpc_comprehensive'))
from strategy import configure_policy
from disturbance_template.full_task_protocol import FullTaskProtocol
import disturbance_predictor
from resampled_preview import ResampledTemplate


factory=base.sim_support.create_arm_controller
def setup(config,*args):
    result=factory(config,*args)
    if config.get('simple_controller')=='acceleration_pid' and config.get('research_c_single_terms',False):
        policy=EfficientAccelerationPID(config,result.policy.neutral,result.policy.dt)
        return replace(result,policy=policy,metadata=dict(result.metadata,one_node_terms_optimization=True))
    if config.get('simple_controller','acceleration_mpc')=='acceleration_mpc':
        policy=configure_policy(result.policy,config)
        if config.get('research_hold_future_disturbance',False):
            original=policy.compute_action
            def current_only(obs,helpers):
                from types import SimpleNamespace
                proxy=SimpleNamespace(**vars(helpers))
                proxy.disturbance_prediction=(helpers.disturbance_prediction[0],)*10
                proxy.interval_disturbance_prediction=(helpers.disturbance_prediction[0],)*9
                return original(obs,proxy)
            policy.compute_action=current_only
        return replace(result,policy=policy)
    return result
base.sim_support.create_arm_controller=setup

if __name__=='__main__':
    c=yaml.safe_load(Path(sys.argv[1]).read_text())
    dt=float(c['simulation_dt'])*int(c['arm_control_decimation'])
    protocol=FullTaskProtocol(mpc_dt=dt,horizon=9)
    if dt!=.006 or c.get('research_resample_template',False):
        disturbance_predictor.FullTaskTemplatePredictor=ResampledTemplate
    base.run(extra_edits={
        '        protocol = DEFAULT_FULL_TASK_PROTOCOL\n':'        protocol = research_protocol\n',
        '        startup_pd_handoff = FixedStartupPdHandoff(startup_duration)\n':
        '        startup_pd_handoff = FixedStartupPdHandoff(startup_duration, research_protocol)\n',
    },extra_globals={'research_protocol':protocol})
