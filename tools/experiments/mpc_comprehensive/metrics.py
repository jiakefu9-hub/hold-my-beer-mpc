"""Matched 0 <= t < 8 s physical metrics; vector RMS, never component RMS mean."""
from pathlib import Path
import json
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]


def rms_vector(x):
    return float(np.sqrt(np.mean(np.sum(np.asarray(x)**2, axis=-1))))


def summarize(path, compact=True, xml_path='resources/g1_description/scene.xml'):
    path = Path(path)
    if not (path / 'metrics.npz').exists():
        return dict(complete=False, path=str(path))
    with np.load(path / 'metrics.npz') as archive:
        metrics = {k: archive[k] for k in archive.files}
    with np.load(path / 'trajectory.npz') as archive:
        tr = {k: archive[k] for k in archive.files if k in (
            'time', 'qpos', 'qvel', 'right_arm_q', 'right_arm_dq',
            'right_arm_qacc', 'right_arm_ddq_des', 'right_arm_tau',
            'right_arm_cpp_executor_actuator_tau', 'right_mpc_solve_time')}
    time = metrics['time']
    model = mujoco.MjModel.from_xml_path(str(ROOT / xml_path))
    data = mujoco.MjData(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s + '_grasp_site')
           for s in ('left', 'right')]
    assert all(i >= 0 for i in ids)
    axes = np.empty((len(time), 2, 3))
    velocities = np.empty((len(time), 2, 6))
    for i, (q, dq) in enumerate(zip(tr['qpos'], tr['qvel'])):
        data.qpos[:] = q
        data.qvel[:] = dq
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        for side, site in enumerate(ids):
            axes[i, side] = data.site_xmat[site].reshape(3, 3)[:, 2]
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE,
                                    site, velocities[i, side], 0)
    full = (time >= 0.) & (time < 8. - 1e-10)
    result = dict(complete=bool(time[-1] >= 8.), path=str(path), windows={})
    for label, mask in [('full', full), ('walk', full & (time < 6.4 - 1e-10)),
                        ('stop', full & (time >= 6.4 - 1e-10))]:
        result['windows'][label] = {}
        for j, side in enumerate(('left', 'right')):
            z = axes[mask, j]
            mean_z = np.mean(z, axis=0)
            mean_z /= np.linalg.norm(mean_z)
            tilt = np.degrees(np.arccos(np.clip(z[:, 2], -1, 1)))
            swing = np.degrees(np.arccos(np.clip(z @ mean_z, -1, 1)))
            nominal_velocity = np.zeros((int(mask.sum()), 3))
            nominal_velocity[time[mask] < 6.4 - 1e-10, 0] = .5
            result['windows'][label][side] = dict(
                tilt_mean=float(np.mean(tilt)), tilt_rms=float(np.sqrt(np.mean(tilt**2))),
                tilt_max=float(np.max(tilt)), swing_rms=float(np.sqrt(np.mean(swing**2))),
                acc=rms_vector(metrics[side + '_ee_lin_acc_world'][mask]),
                alpha=rms_vector(metrics[side + '_ee_ang_acc_world'][mask]),
                speed=rms_vector(velocities[mask, j, 3:]),
                speed_tracking_rms=rms_vector(velocities[mask, j, 3:] - nominal_velocity),
                omega=rms_vector(velocities[mask, j, :3]))
    gates = json.loads((path / 'full_task_smoke_summary.json').read_text())
    result['gates'] = {k: gates.get(k) for k in (
        'status', 'fallen', 'nan_inf_count', 'tail_complete', 'qp_failure_count',
        'qp_fallback_count', 'runtime_mapping_safety_fallback_count', 'xy_displacement_m')}
    for k in tr:
        if k.startswith('right_arm_'):
            result[k + '_absmax'] = np.max(np.abs(tr[k][full]), axis=0).tolist()
    if compact:
        np.savez_compressed(path / 'compact_metrics.npz', time=time, axes=axes,
            world_velocity=velocities, qpos=tr['qpos'], qvel=tr['qvel'],
            **{k: metrics[k] for k in metrics if k.endswith(('_lin_acc_world', '_ang_acc_world'))})
    return result
