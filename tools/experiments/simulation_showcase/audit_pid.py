"""Check the high-acceleration PID and record all small-sweep results.

The 105 ms speed smoothing below is a bandwidth diagnostic, NOT the hardware
metric pipeline and NOT substituted for the full-band comparison metrics.
"""
from pathlib import Path
import json
import sys

import mujoco
import numpy as np
from scipy.signal import savgol_filter, welch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = ROOT / 'evaluation/simulation_showcase_20261103'
DEST = ROOT / 'docs/simulation/experiments/20261103_showcase'
sys.path.insert(0, str(HERE.parent / 'mpc_comprehensive'))
from metrics import rms_vector


def diagnose(path, reconstruct=False):
    with np.load(path / 'compact_metrics.npz', allow_pickle=False) as data:
        t = data['time']
        acc = data['right_ee_lin_acc_world']
        velocity = data['world_velocity'][:, 1, 3:]
    full = (t >= 0) & (t < 8.-1e-10)
    result = dict(stored_acc_rms=rms_vector(acc[full]), windows={})
    for label, mask in [('full', full), ('middle', (t >= 1.) & (t < 6.4)),
                        ('stop', full & (t >= 6.4))]:
        freq, power = welch(acc[mask], fs=500., nperseg=min(1024, int(mask.sum())), axis=0)
        power = power.sum(axis=1)
        result['windows'][label] = dict(
            acc_rms=rms_vector(acc[mask]),
            dominant_frequency_hz=float(freq[np.argmax(power)]),
            power_fraction_above_30hz=float(power[freq > 30].sum()/power.sum()))
    smooth = savgol_filter(velocity, 53, 3, axis=0)
    filtered_acc = np.vstack([np.zeros((1, 3)), np.diff(smooth, axis=0)/.002])
    result['bandwidth_diagnostic_sg53_acc_rms'] = rms_vector(filtered_acc[full])
    trajectory = path / 'trajectory.npz'
    if trajectory.exists():
        with np.load(trajectory, allow_pickle=False) as tr:
            ref = tr['right_arm_target_dq'][full][::3]
            result['reference_speed_saturated_fraction'] = np.mean(np.abs(ref) >= .479, axis=0).tolist()
            result['reference_speed_sign_flip_fraction'] = np.mean(ref[1:]*ref[:-1] < 0., axis=0).tolist()
            if reconstruct:
                model = mujoco.MjModel.from_xml_path(str(ROOT / 'resources/g1_description/scene.xml'))
                data = mujoco.MjData(model)
                site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right_grasp_site')
                jv, jw = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                v, w = [], []
                for q, dq in zip(tr['qpos'], tr['qvel']):
                    data.qpos[:] = q
                    data.qvel[:] = dq
                    mujoco.mj_fwdPosition(model, data)
                    mujoco.mj_jacSite(model, data, jv, jw, site)
                    v.append(jv@dq)
                    w.append(jw@dq)
                fresh_acc = np.vstack([np.zeros((1, 3)), np.diff(v, axis=0)/.002])
                fresh_alpha = np.vstack([np.zeros((1, 3)), np.diff(w, axis=0)/.002])
                result['fresh_jacobian_reconstruction'] = dict(
                    acc_rms=rms_vector(fresh_acc[full]),
                    alpha_rms=rms_vector(fresh_alpha[full]),
                    acc_component_rms=np.sqrt(np.mean(fresh_acc[full]**2, axis=0)).tolist(),
                    max_sample_difference=float(np.max(np.abs(fresh_acc-acc))),
                    relative_rms_difference=float(rms_vector(fresh_acc[full])/rms_vector(acc[full])-1),
                    note='Fresh FK of saved qpos/qvel; small sample-stage difference from online recorder. No gravity added.')
    return result


def main():
    report = dict(simulation_only=True, physical_dt_s=.002, control_dt_s=.006,
                  evaluation_window_s=[0, 8], runs={})
    names = ['d01_position_pid', 'pid_damping012', 'pid_time20', 'pid_time20_d0036',
             'pid_kp012_d0012', 'pid_kp006_d0006',
             'pid_balanced_heldout_pair02minus', 'pid_soft_heldout_pair02minus',
             'd01_position_pid_balanced']
    for name in names:
        source = STUDY / 'results' / (name+'.json')
        result = json.loads(source.read_text())
        assert result['complete'] and result['gates']['status'] == 'PASS'
        path = Path(result['path'])
        report['runs'][name] = dict(result=result, diagnostics=diagnose(
            path, reconstruct=name == 'd01_position_pid'))
    reference = report['runs']['pid_kp012_d0012']['result']['windows']['full']['right']
    repeat = report['runs']['d01_position_pid_balanced']['result']['windows']['full']['right']
    report['selected_config_repeat_max_metric_difference'] = max(abs(reference[k]-repeat[k]) for k in reference)
    (DEST / 'pid_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for name, label in [('d01_position_pid', 'Original PID'),
                        ('d01_position_pid_balanced', 'Balanced PID')]:
        path = Path(report['runs'][name]['result']['path'])
        with np.load(path / 'compact_metrics.npz') as c, np.load(path / 'trajectory.npz') as tr:
            t = c['time']
            view = (t >= 2.) & (t <= 2.5)
            axes[0].plot(t[view], tr['right_arm_target_dq'][view, 0], label=label)
            acc = c['right_ee_lin_acc_world']
            axes[1].plot(t[view], np.linalg.norm(acc[view], axis=1), label=label)
            full = (t >= 0.) & (t < 8.-1e-10)
            f, power = welch(acc[full], fs=500., nperseg=1024, axis=0)
            axes[2].semilogy(f, power.sum(axis=1), label=label)
    axes[0].set(ylabel='Shoulder pitch dq_ref (rad/s)', xlabel='Simulation time (s)')
    axes[1].set(ylabel='Endpoint acceleration norm (m/s2)', xlabel='Simulation time (s)')
    axes[2].set(ylabel='Summed acceleration PSD', xlabel='Frequency (Hz)', xlim=(0, 80))
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Same PID equations and physical conditions; reduced feedback saturation')
    fig.tight_layout()
    fig.savefig(DEST / 'pid_reference_audit.png', dpi=150)
    plt.close(fig)
    print(json.dumps({n: d['diagnostics'] for n, d in report['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
