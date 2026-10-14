"""Offline phase comparison of preselected matched morning runs and final takes.

No DDS or controller changes. Derivatives are computed on continuous recordings
before selecting disjoint start/stop windows; samples are never concatenated
before differentiation. Raw logs and previous reports are read-only.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/g1_phase_matplotlib')
from pathlib import Path
import sys
import json
import socket
from unittest.mock import patch

import numpy as np
import orjson
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

ROOT = Path('/home/fjk/g1_ws/hold-my-beer-mpc')
sys.path.insert(0, str(ROOT / 'tools/g1_commissioning'))
from analyze_morning_20261013 import RUN_DIRS, GROUPS, BASE, sha, rms, endpoints
from analyze_hardware_pid import _angular_velocity

OUT = Path(__file__).resolve().parent
RUNS = dict(RUN_DIRS)
RUNS.update({
    'V01_PID': 'morning_M03_position_pid_zero_zero_payload75_walk_20261013_164407_686365_5c1c',
    'V02_HOLD': 'morning_M02_mpc_hold_zero_payload75_walk_20261013_164656_767428_c0d8',
    'V03_LEARNED': 'morning_M01_mpc_learned_zero_payload75_walk_20261013_164838_063152_fd65',
    'V04_PID_RELAXED': 'morning_M07_position_pid_zero_relaxed_zero_payload75_walk_20261013_165125_196608_32b3',
})
COHORTS = {
    'morning_matched': GROUPS,
    'final_video': {'position_pid': ['V01_PID'], 'hold_mpc': ['V02_HOLD'],
                    'learned_mpc': ['V03_LEARNED'], 'position_pid_relaxed': ['V04_PID_RELAXED']},
}
WINDOWS = {
    'start': [(5., 7.)],
    'middle': [(7., 15.)],
    'stop': [(15., 18.)],
    'start_and_stop': [(5., 7.), (15., 18.)],
    'early_stop': [(15., 16.)],
    'late_stop': [(16., 18.)],
    'late_walk': [(10., 15.)],
    'primary': [(5., 18.)],
    'start1_stop': [(5., 6.), (15., 18.)],
    'middle_after1': [(6., 15.)],
    'start3_stop': [(5., 8.), (15., 18.)],
    'middle_after3': [(8., 15.)],
}
METRICS = ['tilt_mean_deg', 'swing_rms_deg', 'acc_rms_m_s2', 'alpha_rms_rad_s2',
           'relative_speed_demeaned_rms_m_s', 'omega_rms_rad_s']
LABELS = {'position_pid': '位置 PID', 'hold_mpc': '无学习 MPC（当前扰动保持）',
          'learned_mpc': '学习 MPC', 'position_pid_relaxed': '放宽参考限速 PID'}


def summarize_axis(z):
    mean_axis = z.mean(axis=0)
    mean_axis /= np.linalg.norm(mean_axis)
    deviation = np.degrees(np.arccos(np.clip(z @ mean_axis, -1., 1.)))
    return float(np.mean(deviation**2))


def derive(f, window):
    body = savgol_filter(f['torso_specific_force_h0_m_s2'] + [0., 0., -9.81], window, 3, axis=0)
    out = {'t': f['task_elapsed_s'], 'torso_acc': body}
    for side in ('left', 'right'):
        rot = Rotation.from_quat(f[side + '_orientation_h0_quaternion_wxyz'][:, [1, 2, 3, 0]]).as_matrix()
        pos = f[side + '_position_from_torso_imu_h0_m']
        vel = savgol_filter(pos, window, 3, deriv=1, delta=.005, axis=0)
        omega = savgol_filter(_angular_velocity(rot, .005), window, 3, axis=0)
        alpha = savgol_filter(omega, window, 3, deriv=1, delta=.005, axis=0)
        acc = body + savgol_filter(pos, window, 3, deriv=2, delta=.005, axis=0)
        if window == 21:
            np.testing.assert_allclose(acc, f[side + '_linear_acceleration_h0_m_s2'], atol=1e-8)
            np.testing.assert_allclose(alpha, f[side + '_angular_acceleration_h0_rad_s2'], atol=1e-8)
        out[side] = {'z': rot[:, :, 2], 'tilt': f[side + '_tilt_deg'], 'velocity': vel,
                     'omega': omega, 'alpha': alpha, 'acc': acc}
    return out


def phase_metrics(arr, intervals):
    t = arr['t']
    masks = [(t >= lo) & (t < hi) for lo, hi in intervals]
    for mask, (lo, hi) in zip(masks, intervals):
        assert mask.sum() == round((hi-lo)*200), (lo, hi, mask.sum())
    mask = np.logical_or.reduce(masks)
    result = {'duration_s': float(mask.sum()*.005), 'sample_count': int(mask.sum()),
              'torso': {'acc_rms_m_s2': rms(arr['torso_acc'][mask])}}
    for side in ('left', 'right'):
        v = arr[side]
        tilt = v['tilt'][mask]
        velocity = v['velocity'][mask]
        # The main combined metric uses ONE mean direction across both phases.
        # Also retain within-phase-centered dispersion to expose mean shifts.
        within_swing_sq = sum(m.sum()*summarize_axis(v['z'][m]) for m in masks)/mask.sum()
        result[side] = dict(
            tilt_mean_deg=float(tilt.mean()), tilt_rms_deg=float(np.sqrt(np.mean(tilt**2))),
            tilt_max_deg=float(tilt.max()), swing_rms_deg=float(np.sqrt(summarize_axis(v['z'][mask]))),
            within_phase_swing_rms_deg=float(np.sqrt(within_swing_sq)),
            acc_rms_m_s2=rms(v['acc'][mask]), alpha_rms_rad_s2=rms(v['alpha'][mask]),
            relative_speed_demeaned_rms_m_s=rms(velocity-velocity.mean(axis=0)),
            relative_speed_raw_rms_m_s=rms(velocity), omega_rms_rad_s=rms(v['omega'][mask]))
    return result


def inspect_run(name):
    root = BASE/RUNS[name]
    summary = json.loads((root/'endpoint_analysis/summary.json').read_text())
    assert sha(root/'raw.jsonl') == summary['source_sha256']
    assert summary['session_status']['release_completed'] and not summary['session_status']['faults']
    assert summary['capture_quality']['status'] == 'recorded_checks_passed'
    assert summary['sample_hz'] == 200 and summary['derivative_filter']['actual_window_samples'] == 21
    session, stages = None, []
    commands = 0
    for line in (root/'raw.jsonl').open('rb'):
        r = orjson.loads(line)
        if r.get('event') == 'session_start': session = r
        if r.get('event') == 'task_stage':
            stages.append({'stage': r['stage'], 'task_elapsed_s': r['task_elapsed_s']})
        if r.get('event') == 'dds_write' and 5 <= (r.get('task_elapsed_s') or -1) < 18:
            commands += 1
            assert np.allclose(r['kp_command'][:5], 20.) and np.allclose(r['kd_command'][:5], 1.)
            assert np.allclose(r['q_command_rad'][:5], 0.)
    assert commands > 2000 and session['control_nominal_period_ms'] == 6
    assert session['zero_arm_neutral_experiment']['enabled']
    for stage, nominal in [('forward_walk',5.), ('stop_settle',15.), ('normal_stop_wait',18.)]:
        actual = next(s['task_elapsed_s'] for s in stages if s['stage'] == stage)
        assert abs(actual-nominal) < .010, (stage, actual)
    with np.load(root/'endpoint_analysis/metrics.npz') as f:
        arr = derive(f, 21)
        windows = {k: phase_metrics(arr, v) for k, v in WINDOWS.items()}
        # Check identical definitions to the previously reported contiguous windows.
        for key in ('primary', 'middle', 'stop'):
            previous = endpoints(f, WINDOWS[key][0])
            for side in ('left','right'):
                for metric in METRICS:
                    np.testing.assert_allclose(windows[key][side][metric], previous[side][metric], atol=1e-10)
        sensitivity = {str(ms): {key: phase_metrics(derive(f, round(ms*.2)), WINDOWS[key])
                               for key in ('start_and_stop', 'middle', 'stop')}
                       for ms in (75,135)}
    return dict(directory=root.name, stages=stages, windows=windows, sensitivity=sensitivity,
                source_sha256=summary['source_sha256'], arrays_sha256=sha(root/'endpoint_analysis/metrics.npz'),
                model_sha256=session['controller_config_sha256'],
                mpc_sha256=session.get('mpc_config_sha256'), torque_sha256=session.get('torque_config_sha256'),
                pid_sha256=session.get('position_pid_config_sha256')), arr


def aggregate(runs, groups):
    out = {}
    for group, ids in groups.items():
        out[group] = {'ids': ids, 'n': len(ids), 'windows': {}, 'sensitivity': {}}
        for name in WINDOWS:
            out[group]['windows'][name] = {}
            for side in ('left','right','torso'):
                out[group]['windows'][name][side] = {}
                for metric in runs[ids[0]]['windows'][name][side]:
                    values = np.asarray([runs[k]['windows'][name][side][metric] for k in ids])
                    out[group]['windows'][name][side][metric] = {
                        'mean': float(values.mean()), 'min': float(values.min()), 'max': float(values.max())}
        for ms in ('75','135'):
            out[group]['sensitivity'][ms] = {}
            for name in ('start_and_stop','middle','stop'):
                out[group]['sensitivity'][ms][name] = {}
                for side in ('left','right','torso'):
                    out[group]['sensitivity'][ms][name][side] = {
                        metric: float(np.mean([runs[k]['sensitivity'][ms][name][side][metric] for k in ids]))
                        for metric in runs[ids[0]]['sensitivity'][ms][name][side]}
    return out


def table(groups, name):
    lines = ['| 控制组 | 手臂 | 竖直倾角均值 ° | 平均轴摆动 RMS ° | 线加速度 m/s² | 角加速度 rad/s² | 相对线速度 m/s | 角速度 rad/s |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for group in ('position_pid','hold_mpc','learned_mpc','position_pid_relaxed'):
        if group not in groups: continue
        for side in ('left','right'):
            v = groups[group]['windows'][name][side]
            lines.append('| '+LABELS[group]+' | '+('左 PD' if side=='left' else '右')+' | '+
                         ' | '.join(f'{v[m]["mean"]:.4f}' for m in METRICS)+' |')
    return lines


def write_outputs(runs, cohorts, arrays):
    semantics = {
        'scope': 'Retrospective phase analysis. No robot execution, code/config change, or raw-log edits.',
        'cohort': 'Matched morning six runs, two per controller; final completed video takes independently reported. Camera retakes are not marked invalid.',
        'estimator': 'Joint+IMU reconstruction, 200 Hz, 105 ms third-order SG; same estimator across controllers. 75/135 ms sensitivity retained.',
        'phases': 'Nominal command schedule: start [5,7), middle [7,15), stop-settle [15,18). Stage entries checked within 10 ms. Stop is not guaranteed stationary throughout.',
        'union': 'Start+stop selects the union of time samples, duration weighted 2:3; derivatives computed before masking. One mean axis/mean relative velocity across union.',
        'swing': 'RMS angular distance from endpoint +Z to its normalized mean direction per run and selected window; union includes between-phase mean shifts.',
        'velocity': '3D endpoint-minus-torso-IMU velocity in fixed H0, demeaned per window; not absolute world speed.',
        'statistics': 'Each run computed independently, then equal-weight arithmetic mean across runs; no significance claim.',
        'hold': 'hold_current ablates learned prediction only; current IMU, gravity, and moving-base dynamics retained.'}
    result = {'windows': WINDOWS, 'semantics': semantics, 'runs': runs, 'cohorts': cohorts,
              'analysis_sha256': sha(Path(__file__))}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines = ['# 2026-10-14 起步、行走、停车分段比较', '', '## 结果解释', '',
             '- 主比较为 10 月 13 日上午六轮：位置 PID、当前扰动保持 MPC、学习 MPC 各两轮；最后录像四轮独立复核，不混合放宽限速 PID。',
             '- 学习 MPC 中途左/右摆动 RMS 为 0.6346/0.5178°，停车为 1.8519/0.6215°。相对左手的改善从 18.4% 增至 66.4%，但右手自身停车摆动并未比中途更小。',
             '- 最后学习 MPC 录像中途左/右摆动为 0.6524/0.5407°，停车为 1.9442/0.5496°；停车后半段 16～18 s 右手为 0.4355°，比该轮中途低 19.4%。这支持停车后半段可能更稳，不能推广为所有停车阶段都更稳。',
             '- 无学习 MPC 也有明显的停车姿态优势。上午学习与无学习 MPC 的停车摆动为 0.6215/0.6526°，差异远小于 MPC 与左 PD 或位置 PID 的差异。',
             '- 学习 MPC 中途右手线加速度比左手高 2.4%，停车高 9.0%；姿态优势并不等于冲击和线加速度优势。75/135 ms 滤波窗口下动态指标数值会变化，不能把 105 ms 平滑后的指标当作高频尖峰真值。',
             '- 原位置 PID 起步阶段正在纠偏；起步+停车的 3.9838° 包含两阶段平均方向的变化。分别以每阶段平均轴为中心再合并，仍为 2.5716°。即使只看 10～15 s，右 PID 摆动为 1.3988°，右学习 MPC 为 0.5096°。',
             '- 停车期身体线加速度 RMS 不同：PID 1.6327、无学习 MPC 2.0080、学习 MPC 2.0812 m/s²。跨运行的绝对加速度差不能全部归因为手臂控制器，需结合各轮左手和身体输入解释。',
             '- 本结果是小样本描述性比较。各阶段按预定任务时间划分；15～18 s 包含减速和稳定过程，不表示机器人这三秒一直静止。',
             '', '## 统计口径', '']
    lines += ['- '+v for v in semantics.values()]
    lines += ['', '## 实验清单', '', '| 标号 | 原始目录 |', '|---|---|']
    lines += [f'| {key} | {r["directory"]} |' for key,r in runs.items()]
    for cohort, groups in cohorts.items():
        lines += ['', '## '+cohort, '']
        for name in ('start_and_stop','middle','start','stop','early_stop','late_stop','late_walk','primary'):
            lines += ['', '### '+name+' '+str(WINDOWS[name]), ''] + table(groups,name)
    lines += ['', '## 每次运行的姿态指标（不只报告平均值）', '',
              '| 实验 | 阶段 | 左倾角 / 摆动 ° | 右倾角 / 摆动 ° | 身体线加速度 RMS m/s² |',
              '|---|---|---:|---:|---:|']
    for key,r in runs.items():
        for name in ('start','middle','stop'):
            w=r['windows'][name]
            lines.append(f'| {key} | {name} | {w["left"]["tilt_mean_deg"]:.4f} / {w["left"]["swing_rms_deg"]:.4f} | '
                         f'{w["right"]["tilt_mean_deg"]:.4f} / {w["right"]["swing_rms_deg"]:.4f} | {w["torso"]["acc_rms_m_s2"]:.4f} |')
    lines += ['', '合并阶段另存 `within_phase_swing_rms_deg`：先各阶段求平均轴，再按样本数合并平方摆幅；用于判断阶段间中心偏移的影响。',
              'summary.json 同时保留起步 1/3 秒边界敏感性、75/135 ms 滤波敏感性、组内 min/max 与源哈希。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3, figsize=(13,7.5))
    names = [('tilt_mean_deg','Mean tilt from vertical (deg)'),('swing_rms_deg','Swing about mean axis (deg)'),
             ('acc_rms_m_s2','Linear acceleration RMS (m/s²)'),('alpha_rms_rad_s2','Angular acceleration RMS (rad/s²)'),
             ('relative_speed_demeaned_rms_m_s','Demeaned relative velocity RMS (m/s)'),('omega_rms_rad_s','Angular velocity RMS (rad/s)')]
    groups=cohorts['morning_matched']; phases=['start','middle','stop'];x=np.arange(3)
    for ax,(metric,label) in zip(axes.flat,names):
        for gi,(g,color,short) in enumerate([('position_pid','tab:green','PID'),('hold_mpc','tab:orange','Hold MPC'),('learned_mpc','tab:blue','Learned MPC')]):
            for side,style in [('left','--'),('right','-')]:
                y=[groups[g]['windows'][p][side][metric]['mean'] for p in phases]
                lo=[groups[g]['windows'][p][side][metric]['min'] for p in phases]
                hi=[groups[g]['windows'][p][side][metric]['max'] for p in phases]
                ax.errorbar(x+(gi-1)*.055,y,yerr=[np.asarray(y)-lo,np.asarray(hi)-y],
                    fmt='o'+style,color=color,alpha=.55 if side=='left' else 1.,capsize=3,
                    label=short+(' left PD' if side=='left' else ' right'))
        ax.set_xticks(x, ['Start\n5–7 s','Middle\n7–15 s','Stop\n15–18 s'])
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Matched morning runs: means of two runs; bars = run range\nSame offline estimator; relative velocity is not absolute world speed', fontsize=11)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'phase_metrics.png',dpi=170);plt.close(fig)


with patch.object(socket, 'socket', side_effect=RuntimeError('offline analysis forbids network')):
    runs, arrays = {}, {}
    for key in RUNS:
        runs[key], arrays[key] = inspect_run(key)
    assert len({r['model_sha256'] for r in runs.values()}) == 1
    cohorts = {name: aggregate(runs, groups) for name, groups in COHORTS.items()}
    # Cross-check against the saved six-run full-task headline values.
    old = json.loads((BASE/'morning_comparison_20261013_final/summary.json').read_text())
    for group in GROUPS:
        for side in ('left','right'):
            for metric in METRICS:
                np.testing.assert_allclose(cohorts['morning_matched'][group]['windows']['primary'][side][metric]['mean'],
                                           old['groups'][group]['windows']['primary'][side][metric]['mean'],atol=1e-10)
    write_outputs(runs,cohorts,arrays)
    for name in ('start_and_stop','middle','start','stop','early_stop','late_stop'):
        print(name)
        print('\n'.join(table(cohorts['morning_matched'],name)))
    print(OUT/'REPORT.md')
