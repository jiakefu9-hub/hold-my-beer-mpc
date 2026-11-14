"""Record three new 15-second simulations and a separate task-matched template.

No robot SDK. Existing controllers, configurations and 8-second records stay
unchanged. Final acquisition restores the original 11-build/4-held-out design.
The earlier 2+1 acquisition is retained only for explicit legacy reproduction.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/thesis-walk15-matplotlib'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import PROTOCOL, ROOT, STUDY, EVIDENCE
import numpy as np
import mujoco
import yaml
from disturbance_template.full_task_fixed_pd_collector import (
    EpisodeSpec, collect_fixed_pd_episode)
from disturbance_template.full_task_template_builder import (
    build_full_task_template, evaluate_heldout_template, portable_asset)
from disturbance_template.full_task_template_asset import (
    TEMPLATE_SCHEMA_VERSION_V2, validate_full_task_template)
from disturbance_template.full_task_protocol import FullTaskContinuousHeadingFrame
from disturbance_template.full_task_recording import validate_full_task_raw

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def original_design_template(config, config_path):
    from full_template import build_original_design
    out=STUDY/'template'
    asset=out/'full_task_template.npz'
    asset_manifest=out/'full_task_template_manifest.json'
    if asset.is_file() and asset_manifest.is_file():
        saved=json.loads(asset_manifest.read_text())
        collection=saved['collection']
        assert collection['config']['sha256']==sha(config_path)
        assert collection['build_episode_count']==11
        assert collection['heldout_episode_count']==4
        assert len(saved['build_raw_episodes'])==11
        assert len(saved['heldout_raw_episodes'])==4
        for relative,digest in saved['source_sha256'].items():
            assert sha(ROOT/relative)==digest,relative
        for record in saved['build_raw_episodes']+saved['heldout_raw_episodes']+[saved['template']]:
            assert sha(ROOT/record['path'])==record['sha256'],record['path']
        with np.load(asset,allow_pickle=False) as archive:
            template={k:archive[k] for k in archive.files}
        validate_full_task_template(template,PROTOCOL)
        assert int(template['build_episode_count'])==11
        print('REUSE VERIFIED 11+4 TEMPLATE',flush=True)
    else:
        asset,asset_manifest=build_original_design(config,config_path,out)
        saved=json.loads(asset_manifest.read_text())
    # Runtime manifest contract uses name/version; the complete dataclass
    # fields are retained alongside these explicit aliases.
    contract=saved['collection']['protocol']
    contract.update(name=PROTOCOL.protocol_name,version=PROTOCOL.protocol_version)
    write_json(asset_manifest,saved)
    return asset,asset_manifest

def build_template(config, config_path):
    out = STUDY / 'template'
    out.mkdir(parents=True, exist_ok=True)
    raws, paths = [], []
    for name, sign, role in (('build_minus',-1,'build'),('build_plus',1,'build'),
                              ('heldout_nominal',0,'heldout')):
        path = out / (name+'.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                raw = {k:archive[k] for k in archive.files}
        else:
            # Small, symmetric initial leg perturbations; no controller tuning.
            offset = np.zeros(12); offset[[0,6]] = sign*0.001
            spec = EpisodeSpec(name, role, 'symmetric_01' if sign else None,
                0 if sign else None, sign, offset, np.zeros(12))
            raw, _ = collect_fixed_pd_episode(config, config_path=config_path,
                spec=spec, protocol=PROTOCOL,
                heading_frame_version=FullTaskContinuousHeadingFrame.DEFINITION_VERSION)
            np.savez_compressed(path, **raw)
        report = validate_full_task_raw(raw, PROTOCOL, require_complete=True)
        if np.min(raw['torso_position_world'][:,2]) < .45:
            raise RuntimeError(f'fixed-PD collection fell: {name}')
        write_json(out / (name+'_validation.json'), report)
        raws.append(raw); paths.append(path)
        print('TEMPLATE EPISODE', name, 'complete', flush=True)
    template = build_full_task_template(raws[:2], ['build_minus','build_plus'],
        PROTOCOL, template_schema_version=TEMPLATE_SCHEMA_VERSION_V2)
    validation = validate_full_task_template(template, PROTOCOL)
    heldout, _ = evaluate_heldout_template(template, raws[2:], ['heldout_nominal'], PROTOCOL)
    template_path = out / 'full_task_template.npz'
    np.savez_compressed(template_path, **template)
    write_json(out / 'heldout_metrics.json', heldout)
    manifest = dict(template_schema_version=TEMPLATE_SCHEMA_VERSION_V2,
        template_validation=validation,
        collection=dict(protocol=dict(name=PROTOCOL.protocol_name,
            version=PROTOCOL.protocol_version, horizon=9, stop_time=12.,
            headline_end=15., record_end=15.06),
            heading_frame_version=FullTaskContinuousHeadingFrame.DEFINITION_VERSION,
            build_episode_count=2, heldout_episode_count=1,
            initial_perturbation='paired hip-pitch offsets +/-0.001 rad'),
        build_raw_episodes=[portable_asset(p,ROOT) for p in paths[:2]],
        heldout_raw_episodes=[portable_asset(paths[2],ROOT)],
        template=portable_asset(template_path,ROOT),
        scope='new task-matched fixed-PD simulation template; not old 11+4 validation')
    manifest_path = out / 'full_task_template_manifest.json'
    write_json(manifest_path, manifest)
    return template_path, manifest_path

def package(case, run_dir, config, destination):
    summary = json.loads((run_dir/'full_task_smoke_summary.json').read_text())
    if not summary.get('tail_complete') or summary.get('fallen') or summary.get('nan_inf_count'):
        raise RuntimeError(f'{case}: no complete finite upright recording: {summary}')
    with np.load(run_dir/'trajectory.npz',allow_pickle=False) as archive:
        frames = {k:archive[k] for k in ('time','qpos','qvel')}
    mask = frames['time'] <= 15.+1e-9
    frames = {k:v[mask] for k,v in frames.items()}
    assert frames['time'][-1] >= 15.-1e-9
    assert np.all(np.diff(frames['time'])>0)
    assert all(np.isfinite(v).all() for v in frames.values())
    replay = destination/'replays'/(case+'.npz')
    replay.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(replay,**frames)
    model=mujoco.MjModel.from_xml_path(str(ROOT/config['xml_path']))
    data=mujoco.MjData(model)
    ids=[model.site(side+'_grasp_site').id for side in ('left','right')]
    axes=np.empty((len(mask.nonzero()[0]),2,3))
    velocities=np.empty((len(axes),2,6))
    for i,(q,dq) in enumerate(zip(frames['qpos'],frames['qvel'])):
        data.qpos[:]=q; data.qvel[:]=dq
        mujoco.mj_fwdPosition(model,data);mujoco.mj_fwdVelocity(model,data)
        axes[i]=data.site_xmat[ids].reshape(2,3,3)[:,:,2]
        for j,site in enumerate(ids):
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_SITE,
                                    site,velocities[i,j],0)
    with np.load(run_dir/'metrics.npz',allow_pickle=False) as archive:
        physical={k:archive[k][mask] for k in archive.files
                  if k.endswith(('_lin_acc_world','_ang_acc_world'))}
        assert np.allclose(archive['time'][mask],frames['time'],atol=1e-10,rtol=0)
    def vector_rms(value):
        return float(np.sqrt(np.mean(np.sum(np.asarray(value)**2,axis=-1))))
    results={}
    for label,begin,end in (('full',0.,15.),('walk',0.,12.),('stop',12.,15.)):
        selected=(frames['time']>=begin-1e-10)&(frames['time']<end-1e-10)
        results[label]={}
        for j,side in enumerate(('left','right')):
            z=axes[selected,j]; mean=z.mean(axis=0); mean/=np.linalg.norm(mean)
            tilt=np.degrees(np.arccos(np.clip(z[:,2],-1,1)))
            swing=np.degrees(np.arccos(np.clip(z@mean,-1,1)))
            results[label][side]=dict(tilt_mean_deg=float(tilt.mean()),
                swing_rms_deg=float(np.sqrt(np.mean(swing**2))),
                linear_acceleration_rms_m_s2=vector_rms(physical[side+'_ee_lin_acc_world'][selected]),
                angular_acceleration_rms_rad_s2=vector_rms(physical[side+'_ee_ang_acc_world'][selected]),
                linear_speed_rms_m_s=vector_rms(velocities[selected,j,3:]),
                angular_speed_rms_rad_s=vector_rms(velocities[selected,j,:3]))
    gates={k:summary.get(k) for k in ('status','fallen','nan_inf_count',
        'tail_complete','qp_failure_count','qp_fallback_count',
        'runtime_mapping_safety_fallback_count')}
    write_json(destination/(case+'_metrics.json'),dict(windows=results,gates=gates,
        scope='new 15-second simulation; old 8-second statistics are not reused'))
    print(case,'COMPLETE',gates,results['full']['right'],flush=True)
    return dict(replay=str(replay.relative_to(destination)),replay_sha256=sha(replay),
        xml_path=config['xml_path'],source_run=str(run_dir.relative_to(ROOT)),
        metrics=case+'_metrics.json',duration_s=15.,stop_time_s=12.)

def main():
    global STUDY, EVIDENCE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=('D01','D02','D03'),
                        default=['D01','D02','D03'])
    parser.add_argument('--variant',default='')
    parser.add_argument('--execution-max-qacc',type=float,default=10.)
    parser.add_argument('--template-design',choices=('original11x4','legacy2plus1'),
                        default='original11x4')
    args=parser.parse_args()
    if args.template_design=='original11x4' and not args.variant:
        parser.error('11+4 acquisition requires a separate --variant to preserve existing records')
    original_study=STUDY
    if args.variant:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',args.variant):
            parser.error('variant must use letters, numbers, underscores or hyphens')
        STUDY=STUDY/'variants'/args.variant
        EVIDENCE=EVIDENCE/'variants'/args.variant
        os.environ['THESIS_WALK15_VARIANT']=args.variant
    elif args.execution_max_qacc!=10.:
        parser.error('adjusted settings require a separate --variant')
    if not np.isfinite(args.execution_max_qacc) or args.execution_max_qacc<10.:
        parser.error('execution limit must be finite and at least 10')
    if (EVIDENCE/'manifest.json').exists():
        saved=json.loads((EVIDENCE/'manifest.json').read_text())
        if saved.get('complete'):
            raise SystemExit('Completed recording exists; use replay.py. Do not overwrite it.')
    STUDY.mkdir(parents=True,exist_ok=True)
    for name in ('configs','logs'):
        (STUDY/name).mkdir(exist_ok=True)
    old=json.loads((HERE.parent/'simulation_showcase/cases.json').read_text())
    base=yaml.safe_load((ROOT/old[2]['base_config']).read_text())
    base.update(old[2].get('overrides',{}))
    base_path=STUDY/'configs/template_collection.yaml'
    base_path.write_text(yaml.safe_dump(base,sort_keys=False))
    if args.template_design=='original11x4':
        asset,asset_manifest=original_design_template(base,base_path)
    elif args.variant:
        # Reuse the exact frozen 15-second task template from the original run.
        asset=original_study/'template/full_task_template.npz'
        asset_manifest=original_study/'template/full_task_template_manifest.json'
        if not asset.is_file() or not asset_manifest.is_file():
            parser.error('build the original task-matched template first')
    else:
        asset,asset_manifest=build_template(base,base_path)
    manifest=dict(simulation_only=True,task=dict(walk=[0.,12.],stop=[12.,15.],
        record_end=15.06,physics_dt=.002,control_dt=.006,horizon=9,
        no_hardware_arm_raise_or_takeover_sequence=True,
        numerical_startup_fixed_pd_s=.024),cases={},model_sha256={},
        template=str(asset.relative_to(ROOT)),template_sha256=sha(asset),
        template_manifest=str(asset_manifest.relative_to(ROOT)),
        template_manifest_sha256=sha(asset_manifest),
        mpc_pair_configuration_difference=['research_hold_future_disturbance'])
    manifest['variant']=args.variant or 'original'
    manifest['template_design']=args.template_design
    manifest['common_configuration_adjustments']=(dict(
        ddq_execution_max_abs_qacc=dict(original=10.,value=args.execution_max_qacc))
        if args.execution_max_qacc!=10. else {})
    manifest['adjustment_scope']='simulation execution acceptance only; MPC planned acceleration limit remains 8 rad/s^2'
    for name in ('scene_payload75.xml','g1_23dof_rev_1_0_payload75.xml'):
        p=ROOT/'resources/g1_description'/name
        manifest['model_sha256'][str(p.relative_to(ROOT))]=sha(p)
    configs=[]
    failures=[]
    # main_sim sanitizes the evaluation group to one path component; its
    # trajectories are separate from nested variant config/log directories.
    run_storage=ROOT/'evaluation'/STUDY.name
    for i,case in enumerate(old,1):
        ident=f'D{i:02}'; label='thesis15_'+ident.lower()
        c=yaml.safe_load((ROOT/case['base_config']).read_text());c.update(case.get('overrides',{}))
        c.update(full_task_template_path=str(asset.relative_to(ROOT)),
            full_task_template_manifest_path=str(asset_manifest.relative_to(ROOT)),
            full_task_template_sha256=sha(asset),full_task_template_manifest_sha256=sha(asset_manifest))
        c['ddq_execution_max_abs_qacc']=args.execution_max_qacc
        configs.append(c)
        config_path=STUDY/'configs'/(ident+'.yaml')
        config_path.write_text(yaml.safe_dump(c,sort_keys=False))
        completed=[]
        for path in sorted(run_storage.glob('20*_'+label)):
            report=path/'full_task_smoke_summary.json'
            if report.exists():
                summary=json.loads(report.read_text())
                if (summary.get('tail_complete') and not summary.get('fallen')
                        and not summary.get('nan_inf_count')
                        and summary.get('expected_displacement_m')==6.):
                    completed.append(path)
        if completed:
            entry=package(ident,completed[-1],c,EVIDENCE)
            entry.update(title=case['title'],controller_label=case['controller_label'],
                config=str(config_path.relative_to(ROOT)),config_sha256=sha(config_path))
            manifest['cases'][ident]=entry
            continue
        if ident not in args.cases:
            failures.append(dict(case=ident,status='not_completed',
                log=str((STUDY/'logs'/(ident+'.log')).relative_to(ROOT))))
            continue
        env=dict(os.environ,MPC_CONTROL_CPU='2',
            DISTURBANCE_LAB_FORMAL_LAUNCHER='disturbance_lab_run_sh',
            OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',PYTHONHASHSEED='0',
            PYTHONDONTWRITEBYTECODE='1')
        for key,directory,filename in (
            ('RIGHT_ARM_RNEA_LIBRARY','right-arm-rnea','libright_arm_rnea.so'),
            ('RIGHT_ARM_EXECUTOR_LIBRARY','right-arm-executor','libright_arm_executor.so'),
            ('DDQ_TORQUE_MAPPER_LIBRARY','ddq-torque-mapper','libddq_torque_mapper.so'),
            ('RIGHT_ARM_SIM_RUNTIME_WORKER','right-arm-sim-runtime','right_arm_sim_runtime_worker')):
            env[key]='/tmp/hold-my-beer-mpc-'+directory+'-build/'+filename
        command=['taskset','-c','2',sys.executable,str(HERE/'simulation_entry.py'),
            str(config_path),'--headless','--no-video','--full-task-smoke',
            '--run-label',label,'--evaluation-group',STUDY.name]
        start=time.monotonic(); print('RUN',ident,flush=True)
        with (STUDY/'logs'/(ident+'.log')).open('w') as log:
            proc=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if proc.returncode:
            failures.append(dict(case=ident,returncode=proc.returncode,
                log=str((STUDY/'logs'/(ident+'.log')).relative_to(ROOT))))
            print('FAILED',ident,'see log; continue independent cases',flush=True)
            continue
        run_dir=sorted(run_storage.glob('20*_'+label))[-1]
        entry=package(ident,run_dir,c,EVIDENCE)
        entry.update(title=case['title'],controller_label=case['controller_label'],
            config=str(config_path.relative_to(ROOT)),config_sha256=sha(config_path),
            wall_s=time.monotonic()-start)
        manifest['cases'][ident]=entry
    delta=[k for k in configs[1].keys()|configs[2].keys() if configs[1].get(k)!=configs[2].get(k)]
    assert delta==['research_hold_future_disturbance'],delta
    manifest['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}
    partial=EVIDENCE/'failed_replays/D02_partial.npz'
    if partial.exists():
        with np.load(partial,allow_pickle=False) as archive:
            duration=float(archive['time'][-1])
        manifest['cases']['D02_FAILED']=dict(
            title='No-Preview MPC — incomplete; execution exited at 13.98 s',
            controller_label='NO-PREVIEW MPC',
            replay=str(partial.relative_to(EVIDENCE)),replay_sha256=sha(partial),
            xml_path='resources/g1_description/scene_payload75.xml',
            duration_s=duration,stop_time_s=12.,complete=False,
            failure='NO_SAFE_TORQUE; execution-failure replay, not a 15 s success')
    manifest['failures']=failures
    manifest['complete']=all(k in manifest['cases'] for k in ('D01','D02','D03')) and not failures
    manifest['all_acceptance_gates_pass']=(manifest['complete'] and all(
        json.loads((EVIDENCE/manifest['cases'][ident]['metrics']).read_text())['gates']['status']=='PASS'
        for ident in ('D01','D02','D03')))
    write_json(EVIDENCE/'manifest.json',manifest)
    print('RECORDING STATUS',manifest['complete'],list(manifest['cases']),failures,flush=True)

if __name__=='__main__': main()
