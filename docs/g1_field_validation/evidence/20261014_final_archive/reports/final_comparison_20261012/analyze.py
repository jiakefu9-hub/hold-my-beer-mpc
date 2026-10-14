"""Offline final T-series summary. No robot imports, network or control output."""
from pathlib import Path
import hashlib
import json
import numpy as np
import orjson
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RUNS = {
    'T01_162329': ('mpc_off', 'comparison_T01_mpc_original_timing_zero_payload75_walk_20261012_162329_534656_3347'),
    'T02_163351': ('mpc_on', 'comparison_T02_mpc_aligned_zero_payload75_walk_20261012_163351_662781_5116'),
    'T03_164534': ('c_on', 'comparison_T03_c_aligned_zero_payload75_walk_20261012_164534_985570_5b96'),
    'T04_165151': ('mpc_on', 'comparison_T04_mpc_aligned_repeat_zero_payload75_walk_20261012_165151_346352_46b8'),
    'T05_165900': ('c_on', 'comparison_T05_c_aligned_repeat_zero_payload75_walk_20261012_165900_620491_edf7'),
    'T07_172138': ('mpc_off', 'comparison_T07_mpc_original_timing_repeat_zero_payload75_walk_20261012_172138_027726_897e'),
    'T08_172328': ('c_high_gain', 'comparison_T08_c_attitude600_50_aligned_zero_payload75_walk_20261012_172328_285795_46fb'),
    'T10_173715': ('mpc_acc002_off', 'comparison_T10_mpc_acc002_original_timing_zero_payload75_walk_20261012_173715_779721_f3c5'),
    'T11_174034': ('c_off', 'comparison_T11_c_original_timing_zero_payload75_walk_20261012_174034_866230_b2fc'),
    'T12_174355': ('c_regularized', 'comparison_T12_c_regularization0024_aligned_zero_payload75_walk_20261012_174355_032082_c286'),
    'T01_174738': ('mpc_off', 'comparison_T01_mpc_original_timing_zero_payload75_walk_20261012_174738_286024_948b'),
    'T03_175013': ('c_on', 'comparison_T03_c_aligned_zero_payload75_walk_20261012_175013_388749_4bdd'),
}
EXCLUDED = {
    'comparison_T01_mpc_original_timing_zero_payload75_walk_20261012_160554_112639_685d': 'earlier narrowed bounds; stopped; not current configuration',
    'comparison_T03_c_aligned_zero_payload75_walk_20261012_163955_035771_4062': 'operator reported cable interference; entire run excluded',
    'comparison_T12_c_regularization0024_aligned_zero_payload75_walk_20261012_174249_381685_75cc': 'DDS initialization failure before any control output',
}
METRICS = ('tilt_mean_deg', 'tilt_rms_deg', 'swing_rms_deg', 'acc_rms_m_s2',
           'alpha_rms_rad_s2', 'relative_speed_demeaned_rms_m_s', 'omega_rms_rad_s')


def norm_rms(x):
    return float(np.sqrt(np.mean(np.sum(x*x, axis=1))))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def side_metrics(f, s, side, mask):
    q = f[side+'_orientation_h0_quaternion_wxyz'][mask].copy()
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    axes = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
    mean_axis = axes.mean(axis=0)
    mean_axis /= np.linalg.norm(mean_axis)
    swing = np.degrees(np.arccos(np.clip(axes@mean_axis, -1, 1)))
    tilt = f[side+'_tilt_deg'][mask]
    velocity = savgol_filter(f[side+'_position_from_torso_imu_h0_m'],
        s['derivative_filter']['actual_window_samples'], 3, deriv=1,
        delta=1/s['sample_hz'], axis=0)[mask]
    velocity -= velocity.mean(axis=0)
    result = dict(tilt_mean_deg=float(tilt.mean()), tilt_rms_deg=float(np.sqrt(np.mean(tilt**2))),
        tilt_max_deg=float(tilt.max()), swing_rms_deg=float(np.sqrt(np.mean(swing**2))),
        acc_rms_m_s2=norm_rms(f[side+'_linear_acceleration_h0_m_s2'][mask]),
        alpha_rms_rad_s2=norm_rms(f[side+'_angular_acceleration_h0_rad_s2'][mask]),
        relative_speed_demeaned_rms_m_s=norm_rms(velocity),
        omega_rms_rad_s=norm_rms(f[side+'_angular_velocity_h0_rad_s'][mask]))
    original = s['windows'][s['primary_window_key']]['metrics'][side]
    assert np.isclose(result['acc_rms_m_s2'], original['endpoint_linear_acceleration_h0_m_s2_norm']['rms'])
    assert np.isclose(result['tilt_rms_deg'], original['bottle_z_tilt_from_h0_vertical_deg']['rms'])
    return result


def run_metrics(name):
    root = BASE/name
    summary_path = root/'endpoint_analysis/summary.json'
    s = json.loads(summary_path.read_text())
    assert s['session_status']['release_completed'] and not s['session_status']['faults']
    assert s['capture_quality']['status'] == 'recorded_checks_passed'
    assert s['capture_quality']['journal_queue_dropped'] == 0
    assert s['sample_hz'] == 200 and s['derivative_filter']['actual_window_samples'] == 21
    with np.load(root/'endpoint_analysis/metrics.npz', allow_pickle=False) as f:
        mask = (f['task_elapsed_s'] >= 5) & (f['task_elapsed_s'] < 18)
        assert mask.sum() == 2600
        result = {side: side_metrics(f, s, side, mask) for side in ('left', 'right')}
        result['torso'] = dict(acc_rms_m_s2=norm_rms(f['torso_linear_acceleration_h0_m_s2'][mask]),
                              alpha_rms_rad_s2=norm_rms(f['torso_angular_acceleration_h0_rad_s2'][mask]))
    commands = []
    with (root/'raw.jsonl').open('rb') as stream:
        for line in stream:
            r = orjson.loads(line)
            if r.get('event') == 'session_start':
                session = r
            elif r.get('event') == 'dds_write' and 5 <= r.get('task_elapsed_s', -1) < 18 and r.get('mpc_active'):
                commands.append(r)
    assert session['control_nominal_period_ms'] == 6
    assert session['zero_arm_neutral_experiment']['enabled']
    assert session['left_pd_gain_experiment']['scale'] == 1
    q = np.rad2deg([r['q_measured_rad'][5:10] for r in commands])
    dq = np.array([r['dq_measured_rad_s'][5:10] for r in commands])
    ddq = np.abs([r['raw_mpc_ddq_rad_s2'] for r in commands])
    total_tau = np.abs([r['field_total_torque_estimate_at_latest_feedback_nm'] for r in commands])
    tau_ff = np.abs([r['tau_ff'][5:10] for r in commands])
    cfg = session['core']['torque_config']
    qlo, qhi = np.array(cfg['q_min_deg']), np.array(cfg['q_max_deg'])
    margin = np.minimum(q.min(axis=0)-qlo, qhi-q.max(axis=0))
    result['execution'] = dict(
        commands=len(commands), right_q_min_deg=q.min(axis=0).tolist(), right_q_max_deg=q.max(axis=0).tolist(),
        outer_q_margin_deg=margin.tolist(), max_abs_dq_rad_s=np.abs(dq).max(axis=0).tolist(),
        command_ddq_at_cap_percent=(100*(ddq >= cfg['max_ddq_rad_s2']-1e-3).mean(axis=0)).tolist(),
        integral_frozen_percent=100*np.mean([bool(r['mpc'].get('integral_frozen')) for r in commands]),
        brake_latched_percent=100*np.mean([bool(r.get('predictive_braking', {}).get('latched')) for r in commands]),
        brake_cost_active_percent=100*np.mean([np.any(np.array(r.get('predictive_braking', {}).get('activation', [])) > 0) for r in commands]),
        estimated_total_tau_max_abs_nm=total_tau.max(axis=0).tolist(),
        estimated_total_tau_margin_nm=(np.array(cfg['tau_abs_nm'])-total_tau.max(axis=0)).tolist(),
        packet_tau_ff_max_abs_nm=tau_ff.max(axis=0).tolist(),
        packet_tau_ff_margin_nm=(np.array(cfg['tau_ff_abs_nm'])-tau_ff.max(axis=0)).tolist(),
        fallback_count=sum(bool(r['mpc'].get('fallback_used')) for r in commands),
        solved_count=sum(bool(r['mpc'].get('solved')) for r in commands))
    result['timing'] = s['control']['complete_loop_timing']
    result['provenance'] = dict(directory=name, source_sha256=sha(root/'raw.jsonl'),
        endpoint_summary_sha256=sha(summary_path), endpoint_arrays_sha256=sha(root/'endpoint_analysis/metrics.npz'),
        alignment=session['align_accel_forecast'], controller_variant=session.get('controller_variant'),
        model_config_sha256=session['controller_config_sha256'], torque_config_sha256=session['torque_config_sha256'],
        mpc_config_sha256=session['mpc_config_sha256'], pid_config_sha256=session.get('acceleration_pid_config_sha256'),
        controller_source_sha256=session['control_source_sha256'],
        parameters=session['core']['cost'], session_status=s['session_status'], capture_quality=s['capture_quality'])
    assert result['provenance']['source_sha256'] == s['source_sha256']
    return result


def aggregate(runs):
    groups = {}
    for group in dict.fromkeys(g for g, _ in RUNS.values()):
        ids = [key for key, (g, _) in RUNS.items() if g == group]
        d = dict(run_ids=ids, n=len(ids))
        for side in ('left', 'right', 'torso'):
            d[side] = {}
            for metric in runs[ids[0]][side]:
                a = np.array([runs[k][side][metric] for k in ids])
                d[side][metric] = dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
                                      sample_std=float(a.std(ddof=1)) if len(a)>1 else None)
        groups[group] = d
    return groups


def main():
    runs = {key: run_metrics(name) for key, (_, name) in RUNS.items()}
    for key in ('model_config_sha256', 'torque_config_sha256'):
        assert len({r['provenance'][key] for r in runs.values()}) == 1
    groups = aggregate(runs)
    for g in groups.values():
        for key in ('alignment', 'controller_variant', 'mpc_config_sha256', 'pid_config_sha256'):
            assert len({runs[k]['provenance'][key] for k in g['run_ids']}) == 1
    out = dict(schema='g1_field_final_comparison_v1', window_s=[5,18], sample_hz=200, derivative_window_s=.105,
        aggregation='arithmetic mean of per-run metrics; std is sample standard deviation across runs',
        speed='derivative of endpoint position relative to torso IMU expressed in fixed H0, demeaned over each run window; not absolute world translation speed',
        swing='angular distance of each endpoint z axis to normalized mean z axis within each run; yaw about bottle axis excluded',
        torque='model-estimated total torque and transmitted feedforward limits; not measured physical torque margin',
        excluded=EXCLUDED, runs=runs, groups=groups,
        limitations=['Small nonrandomized sample; altered torso excitation across trials; no causal normalization by left arm',
                     'Endpoint metrics reconstructed from joint feedback and IMU with offline derivatives; not direct payload sensors',
                     'Selected MPC uses alignment off, selected C on; also retain equal-alignment comparisons',
                     'Single-run candidate results do not establish statistical significance',
                     'Deadline misses include wake lateness; work duration under 6 ms alone does not mean no deadline miss'])
    (HERE/'summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    lines = ['# 2026-10-12 final field comparison', '', 'Window [5,18); 200 Hz; 105 ms cubic SG derivatives.',
        'Mean of per-run metrics; velocity is demeaned endpoint-relative-to-torso velocity in H0.', '',
        '| Run | Side | Tilt mean deg | Tilt RMS deg | Swing RMS deg | Acc RMS m/s2 | Alpha RMS rad/s2 | Relative speed RMS m/s | Omega RMS rad/s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,r in runs.items():
        for side in ('left','right'):
            lines.append('| '+key+' | '+side+' | '+' | '.join(f'{r[side][m]:.5f}' for m in METRICS)+' |')
    lines += ['', '## Groups', '', '| Group | N | Side | Tilt mean | Tilt RMS | Swing RMS | Acc RMS | Alpha RMS | Relative speed RMS | Omega RMS |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,g in groups.items():
        for side in ('left','right'):
            lines.append(f'| {key} | {g["n"]} | {side} | '+' | '.join(f'{g[side][m]["mean"]:.5f}' for m in METRICS)+' |')
    lines += ['', '## Execution checks', '', '| Run | Shoulder pitch min deg | Shoulder yaw max deg | P99 complete work ms | Deadline misses / cycles | Brake latched % | Integral frozen % |',
              '|---|---:|---:|---:|---|---:|---:|']
    for key,r in runs.items():
        e,t=r['execution'],r['timing']
        lines.append(f'| {key} | {e["right_q_min_deg"][0]:.3f} | {e["right_q_max_deg"][2]:.3f} | {t["complete_work_ms"]["p99"]:.3f} | {t["deadline_misses"]}/{t["samples"]} | {e["brake_latched_percent"]:.2f} | {e["integral_frozen_percent"]:.2f} |')
    lines += ['', '## Exclusions', '']+[f'- {k}: {v}' for k,v in EXCLUDED.items()]
    lines += ['', '## Limitations', '']+['- '+v for v in out['limitations']]
    lines += ['', 'Reproduce: `/home/fjk/miniforge3/envs/g1_mpc/bin/python '+str(Path(__file__).resolve())+'`', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    for key in ('T03_175013',):
        print(key, json.dumps(runs[key]['right']))
    for key,g in groups.items():
        print(key, g['n'], json.dumps({m:round(g['right'][m]['mean'],6) for m in METRICS}))
    print('Saved',HERE/'REPORT.md')


if __name__ == '__main__':
    main()
