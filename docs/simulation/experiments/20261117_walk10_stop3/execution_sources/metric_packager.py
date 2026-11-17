"""Same metric definitions as the previous comparison, on the new 13 s task."""
import hashlib
import json
import numpy as np
import mujoco
from protocol import ROOT
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def package(case, run_dir, config, destination):
    summary = json.loads((run_dir/'full_task_smoke_summary.json').read_text())
    if not summary.get('tail_complete') or summary.get('fallen') or summary.get('nan_inf_count'):
        raise RuntimeError(f'{case}: no complete finite upright recording: {summary}')
    with np.load(run_dir/'trajectory.npz',allow_pickle=False) as archive:
        frames = {k:archive[k] for k in ('time','qpos','qvel')}
    mask = frames['time'] <= 13.+1e-9
    frames = {k:v[mask] for k,v in frames.items()}
    assert frames['time'][-1] >= 13.-1e-9
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
    for label,begin,end in (('full',0.,13.),('walk',0.,10.),('stop',10.,13.)):
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
        scope='new 13-second simulation; old 8-second statistics are not reused'))
    print(case,'COMPLETE',gates,results['full']['right'],flush=True)
    return dict(replay=str(replay.relative_to(destination)),replay_sha256=sha(replay),
        xml_path=config['xml_path'],source_run=str(run_dir.relative_to(ROOT)),
        metrics=case+'_metrics.json',duration_s=13.,stop_time_s=10.)
