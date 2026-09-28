#!/usr/bin/env python3
"""Recompute offline torque-study comparisons from saved JSON/NPZ only.

No controller, simulator, SDK, network, or robot interface is imported. Failed
runs retain their observed prefix and are never scored as completed trials.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass
class SavedRun:
    label: str
    directory: Path
    study: dict
    summary: dict
    arrays: dict
    coverage_s: float
    input_sha256: dict


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close_metric(actual, expected, name):
    if not np.isfinite(actual) or not np.isclose(actual, expected, rtol=1e-9, atol=1e-11):
        raise ValueError(f"summary/NPZ mismatch for {name}: {expected!r} != {actual!r}")


def metrics(arrays, until_s):
    """Score intervals starting before end, and physical states through end."""
    interval = arrays['interval_t'] < until_s - 1e-10
    active = interval & (arrays['interval_active_seq'] >= 0)
    states = arrays['physics_t'] <= until_s + 1e-10
    controls = arrays['t'] < until_s - 1e-10
    error = arrays['interval_actual'] - arrays['interval_active_desired']
    core = arrays['core_ms'][controls]
    return dict(
        interval_start_s=0., interval_end_exclusive_s=float(until_s),
        active_command_rmse_rad_s2=(float(np.sqrt(np.mean(error[active]**2)))
                                   if np.any(active) else None),
        active_intervals=int(active.sum()),
        initial_hold_intervals=int((interval & ~active).sum()),
        min_outer_margin_deg=float(np.rad2deg(arrays['physics_q_outer_margin'][states].min())),
        max_abs_actual_acceleration_rad_s2=float(np.abs(arrays['interval_actual'][interval]).max()),
        controller_core_p99_ms=float(np.percentile(core, 99)),
        controller_core_max_ms=float(core.max()),
        controller_core_over_6ms=int(np.sum(core > 6.)),
        controller_samples=len(core),
    )


def load_run(label, directory, scenario):
    directory = Path(directory).resolve()
    summary_path, npz_path = directory/'summary.json', directory/f'{scenario}.npz'
    # Detect a concurrent modification of an unfinished study before reporting.
    initial_hashes = {str(p): sha256(p) for p in (summary_path, npz_path)}
    study = json.loads(summary_path.read_text())
    if study.get('schema') != 'g1_torque_robustness_study_v1':
        raise ValueError(f'{directory}: not a torque robustness study')
    summary = study['runs'][scenario]
    required = ('t', 'core_ms', 'tilt', 'physics_t', 'physics_q', 'physics_dq',
                'physics_q_outer_margin', 'interval_t', 'interval_actual',
                'interval_active_seq', 'interval_active_desired',
                'interval_actually_active_command_desired_vs_actual')
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in required}
    for prefix in ('', 'physics_', 'interval_'):
        times = arrays[prefix+'t']
        if (times.ndim != 1 or not len(times) or abs(times[0]) > 1e-10
                or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            raise ValueError(f'{label}: invalid {prefix}timestamps')
        expected_step = .006 if not prefix else .002
        if not np.allclose(np.diff(times), expected_step, rtol=0., atol=1e-10):
            raise ValueError(f'{label}: missing or irregular {prefix}samples')
    for key, value in arrays.items():
        times = arrays['physics_t' if key.startswith('physics_') else
                       'interval_t' if key.startswith('interval_') else 't']
        if len(value) != len(times) or not np.isfinite(value).all():
            raise ValueError(f'{label}: invalid trace {key}')
        if key in ('physics_q', 'physics_dq', 'physics_q_outer_margin',
                   'interval_actual', 'interval_active_desired',
                   'interval_actually_active_command_desired_vs_actual') and value.shape != (len(times), 5):
            raise ValueError(f'{label}: {key} must have five columns')
    coverage = float(arrays['physics_t'][-1])
    close_metric(float(arrays['interval_t'][-1]) + .002, coverage, '2ms coverage')
    if not np.allclose(arrays['interval_actually_active_command_desired_vs_actual'],
                       arrays['interval_actual']-arrays['interval_active_desired'],
                       rtol=1e-10, atol=1e-12):
        raise ValueError(f'{label}: active-command error identity does not hold')
    measured = metrics(arrays, coverage)
    active_rmse = summary['tracking_2ms']['actually_active_command_desired_vs_actual_rmse']
    if measured['active_command_rmse_rad_s2'] is None:
        if active_rmse is not None:
            raise ValueError(f'{label}: summary claims active-command samples without any')
    else:
        close_metric(measured['active_command_rmse_rad_s2'], active_rmse, 'active-command RMSE')
    close_metric(measured['max_abs_actual_acceleration_rad_s2'],
                 summary['max_abs_qacc_at_2ms'], 'maximum 2ms acceleration')
    close_metric(measured['min_outer_margin_deg'],
                 np.rad2deg(summary['minimum_margins_2ms']['q_outer_margin']), 'outer margin')
    close_metric(measured['controller_core_p99_ms'], summary['core_ms']['p99'], 'core p99')
    close_metric(coverage, summary['physical_state_scored_until_s'], 'physical coverage')
    if measured['active_intervals'] != summary['tracking_2ms']['active_command_samples']:
        raise ValueError(f'{label}: active-command sample count mismatch')
    if summary['status'] not in ('complete', 'failed'):
        raise ValueError(f'{label}: unsupported study status')
    complete = summary['status'] == 'complete'
    if bool(summary['metrics_cover_requested_duration']) != complete:
        raise ValueError(f'{label}: inconsistent completion semantics')
    if complete:
        close_metric(coverage, study['duration_s'], 'completed requested duration')
    for path, digest in initial_hashes.items():
        if sha256(path) != digest:
            raise ValueError(f'{label}: input changed during report generation: {path}')
    return SavedRun(label, directory, study, summary, arrays, coverage, initial_hashes)


def comparison(runs, until_s):
    included = [run for run in runs if run.coverage_s >= until_s-1e-10]
    labels = {run.label for run in included}
    return dict(until_s=float(until_s),
        excluded={run.label: f'observed only {run.coverage_s:g}s; shorter than requested window'
                  for run in runs if run.label not in labels},
        runs={run.label: metrics(run.arrays, until_s) for run in included})


def make_report(runs, scenario, compare_until=None):
    if not runs or len({run.label for run in runs}) != len(runs):
        raise ValueError('at least one run and unique labels are required')
    if compare_until is not None and (not np.isfinite(compare_until) or compare_until <= 0):
        raise ValueError('compare-until must be positive and finite')
    result = dict(schema='g1_torque_robustness_report_v1', scenario=scenario,
        dds=False, hardware_output=False, summary_npz_consistency_verified=True,
        semantics=dict(
            rmse='sqrt(mean(error**2)) over active 2ms intervals and all five joints; initial seq=-1 excluded',
            timing='whole controller step wall time saved by study; excludes plant stepping, transport and packet writes',
            windows='interval starts in [0,end); physical-state margins include end; failed prefix is not full duration',
            safety='offline model only; no hardware safety or real-time certification'),
        source_sha256={str(Path(__file__).resolve()): sha256(__file__)},
        inputs={run.label: dict(directory=str(run.directory), sha256=run.input_sha256,
            original_study_source_unchanged=run.study.get('source_unchanged_during_run')) for run in runs},
        runs={run.label: dict(status=run.summary['status'], failure=run.summary['failure'],
            requested_duration_s=run.study['duration_s'], observed_until_s=run.coverage_s,
            observed_prefix_metrics=metrics(run.arrays, run.coverage_s),
            full_duration_metrics=(metrics(run.arrays, run.coverage_s)
                                   if run.summary['status']=='complete' else None)) for run in runs},
        common_prefix_all=comparison(runs, min(run.coverage_s for run in runs)))
    if compare_until is not None:
        result['requested_window'] = comparison(runs, compare_until)
    return result


def markdown(report):
    lines = ['# Offline torque robustness comparison', '',
        f"Scenario: `{report['scenario']}`. Input summary/NPZ consistency checks passed.", '',
        'RMSE pools five joints over active-command 2ms intervals, excluding the initial hold.',
        'Timing is saved controller-step wall time, not end-to-end DDS latency or real-time proof.', '',
        '## Entire observed data (different durations; not a matched-window ranking)', '',
        '| Run | Outcome | Observed / requested s | Active RMSE rad/s² | Outer margin ° | Max acceleration rad/s² | Core p99 ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    def number(value):
        return 'n/a' if value is None else f'{value:.6g}'
    for name, run in report['runs'].items():
        m = run['observed_prefix_metrics']
        outcome = 'COMPLETE' if run['status']=='complete' else 'FAILED PREFIX ONLY'
        lines.append(f"| {name} | {outcome} | {run['observed_until_s']:g} / {run['requested_duration_s']:g} | "
            f"{number(m['active_command_rmse_rad_s2'])} | {m['min_outer_margin_deg']:.6g} | "
            f"{m['max_abs_actual_acceleration_rad_s2']:.6g} | {m['controller_core_p99_ms']:.6g} |")
    for key, title in [('common_prefix_all','Common prefix of all selected runs'),
                       ('requested_window','Explicit comparison window')]:
        if key not in report:
            continue
        group = report[key]
        lines += ['', '## '+title, '', f"Window: 0 ≤ t < {group['until_s']:g}s.", '',
            '| Run | Active RMSE rad/s² | Active samples | Outer margin ° | Core p99 ms |',
            '|---|---:|---:|---:|---:|']
        for name, m in group['runs'].items():
            lines.append(f"| {name} | {number(m['active_command_rmse_rad_s2'])} | {m['active_intervals']} | "
                f"{m['min_outer_margin_deg']:.6g} | {m['controller_core_p99_ms']:.6g} |")
        if group['excluded']:
            lines += ['', 'Excluded from this window (retained in the full report):', '']
            lines += [f'- {name}: {reason}.' for name, reason in group['excluded'].items()]
    lines += ['', '## Failure reasons', '']
    lines += [f"- {name}: {run['failure']}" for name, run in report['runs'].items()
              if run['status'] != 'complete'] or ['None.']
    lines += ['', 'Do not interpret a short surviving prefix as completion. The plotted ±5° shoulder bounds '
              'are this study’s original model bounds, not commissioned physical limits.', '',
              'Exact input hashes, counts, full-duration/null fields, and comparison values: `comparison.json`.', '']
    return '\n'.join(lines)


def plot(runs, output_dir, scenario):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for run in runs:
        a = run.arrays
        suffix = '' if run.summary['status']=='complete' else ' (FAILED PREFIX)'
        label = run.label + suffix
        axes[0].plot(a['physics_t'], np.rad2deg(a['physics_q'][:, 0]), label=label)
        axes[1].plot(a['physics_t'], a['physics_dq'][:, 0], label=label)
        active = a['interval_active_seq'] >= 0
        axes[2].plot(a['interval_t'][active], np.linalg.norm(
            a['interval_actual'][active]-a['interval_active_desired'][active], axis=1), label=label)
        axes[3].plot(a['t'], a['tilt'], label=label)
    for bound in (-5, 5):
        axes[0].axhline(bound, color='black', linestyle='--', linewidth=.8)
    axes[0].legend(fontsize=8, loc='best')
    for ax, label in zip(axes, ('shoulder pitch (deg)', 'shoulder velocity (rad/s)',
                               'active-command error norm (rad/s²)', 'bottle tilt (deg)')):
        ax.set_ylabel(label)
        ax.grid(True, alpha=.3)
    axes[-1].set_xlabel('physical time (s)')
    fig.suptitle(f'{scenario}: full observed traces; failed prefixes stop at failure')
    fig.tight_layout()
    fig.savefig(output_dir/'comparison.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', action='append', required=True, metavar='LABEL=DIRECTORY')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scenario', default='combined')
    parser.add_argument('--compare-until', type=float,
                        help='Additional common window; shorter runs are explicitly excluded')
    args = parser.parse_args(argv)
    runs = []
    for spec in args.study:
        label, separator, directory = spec.partition('=')
        if not separator or not label.strip() or not directory.strip():
            parser.error('--study requires LABEL=DIRECTORY')
        runs.append(load_run(label.strip(), directory, args.scenario))
    report = make_report(runs, args.scenario, args.compare_until)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'README.md').write_text(markdown(report))
    plot(runs, args.output_dir, args.scenario)
    print(args.output_dir/'README.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
