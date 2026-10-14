"""Export explicitly selected historical evidence; never connects to hardware.

Requires numpy and orjson from the analysis environment. Source files are read
only. The selection is retrospective and includes unfavorable/excluded records.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import csv
import hashlib
import json
import lzma
import shutil
import subprocess

import numpy as np
import orjson

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = Path('evaluation/hardware_shadow/commissioning')
FILES = []


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record(dst, src=None, kind='byte_copy', **extra):
    row = dict(path=str(dst.relative_to(HERE)), bytes=dst.stat().st_size,
               sha256=digest(dst), kind=kind, **extra)
    if src is not None:
        row.update(source_path=str(src.relative_to(ROOT)),
                   source_bytes=src.stat().st_size, source_sha256=digest(src))
    FILES.append(row)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    record(dst, src)


def write_json(dst, obj, src=None, kind='derived'):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    record(dst, src, kind)


def compress(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # A repeated export may reuse an already complete, verified XZ stream.
    if not dst.exists():
        temp = dst.with_suffix('.xz.partial')
        with src.open('rb') as inp, lzma.open(temp, 'wb', preset=3) as out:
            shutil.copyfileobj(inp, out, length=1024*1024)
        temp.replace(dst)
    h = hashlib.sha256()
    with lzma.open(dst, 'rb') as inp:
        for chunk in iter(lambda: inp.read(1024*1024), b''):
            h.update(chunk)
    assert h.hexdigest() == digest(src), f'Compression mismatch: {src}'
    record(dst, src, 'lossless_xz')
    print(f'raw archived {src.parent.name}: {dst.stat().st_size/1e6:.1f} MB', flush=True)


def csv_write(path, fields, rows, src, kind):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    record(path, src, kind)


def extract_run(src, dst):
    raw = src/'raw.jsonl'
    events, timings, commands, counts = [], [], [], {}
    event_names = {
        'session_start', 'session_end', 'task_stage', 'task_epoch', 'capture_drained',
        'sdk_shutdown', 'control_runtime', 'control_timing_summary',
        'heading_reference_frozen', 'normal_weight_release_started',
        'operator_arm_release', 'operator_stop_requested', 'exception', 'fault',
        'prepared_field_plan', 'unitree_sdk2_python_provenance',
        'velocity_final_zero', 'arm_release_failed', 'arm_release_completed'}
    for line in raw.open('rb'):
        r = orjson.loads(line)
        key = r.get('event', r.get('schema', 'unknown'))
        counts[key] = counts.get(key, 0)+1
        if r.get('event') in event_names:
            events.append(r)
        if 'timing' in r.get('schema', '') or 'cycle_complete' in r.get('schema', ''):
            timings.append({k:v for k,v in r.items() if isinstance(v, (int, float, str, bool)) or v is None})
        if r.get('event') == 'dds_write':
            c = {k:r.get(k) for k in ('sequence','task_elapsed_s','stage','weight',
                                      'controller_compute_us','control_actual_period_ms')}
            for field in ('q_measured_rad','dq_measured_rad_s','q_command_rad','dq_command_rad_s',
                          'kp_command','kd_command','tau_ff','tau_model_selected_nm',
                          'post_transition_ddq_rad_s2'):
                value = r.get(field)
                if isinstance(value, list):
                    for i,v in enumerate(value):
                        if isinstance(v, (int,float)):
                            c[f'{field}_{i}'] = v
            commands.append(c)
    write_json(dst/'events.json', dict(source_sha256=digest(raw), schema_counts=counts,
                                     selection='Selected lifecycle events; not the full log', events=events), raw)
    for name, rows in (('timing.csv',timings), ('commands.csv',commands)):
        if rows:
            fields=list(dict.fromkeys(k for row in rows for k in row))
            csv_write(dst/name, fields, rows, raw, 'selected_columns_all_recorded_rows')


def preview(npz, dst):
    with np.load(npz) as z:
        t=z['task_elapsed_s']; rows=[]
        # Decimation for mobile browsing only. Never use it to measure peaks.
        for i in np.flatnonzero((t >= 3) & (t < 21))[::4]:
            r={'task_elapsed_s':float(t[i])}
            for k in z.files:
                if k.endswith('_tilt_deg'):
                    r[k]=float(z[k][i])
                elif k.endswith(('orientation_h0_quaternion_wxyz','acceleration_h0_m_s2',
                                 'angular_velocity_h0_rad_s','angular_acceleration_h0_rad_s2',
                                 'position_from_torso_imu_h0_m')):
                    for j,v in enumerate(z[k][i]):r[f'{k}_{j}']=float(v)
            rows.append(r)
    csv_write(dst, list(rows[0]), rows, npz, '50hz_preview_not_for_recomputing_metrics')


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    args=parser.parse_args(); ROOT=args.source_root.resolve()
    base=ROOT/BASE
    phase=json.loads((base/'phase_comparison_20261014/summary.json').read_text())
    final=json.loads((base/'final_comparison_20261012/summary.json').read_text())
    runs=[]
    for key,r in phase['runs'].items():
        runs.append(dict(id=key, directory=r['directory'], raw=True,
                         cohort='morning' if key.startswith('M') else 'final_video',
                         status='included', expected_sha=r['source_sha256']))
    full_t={'T01_162329','T07_172138','T01_174738','T03_164534','T05_165900','T03_175013'}
    for key,r in final['runs'].items():
        p=r['provenance']
        runs.append(dict(id=key,directory=p['directory'],raw=key in full_t,cohort='T_series',
                         status='included',expected_sha=p['source_sha256']))
    for key,name,cohort in (
        ('HIST_MPC_164546','mpc_learned_acc001_alpha0005_omega1_walk_20261008_164546','historical'),
        ('HIST_PID_132052','g1_pid_6ms_20261005_132052','historical'),
        ('SPIKE_124321','morning_M01_mpc_learned_zero_payload75_walk_20261014_124321_974583_1119','spike'),
        ('SPIKE_125333','morning_M01_mpc_learned_zero_payload75_walk_20261014_125333_831147_ee9f','spike')):
        runs.append(dict(id=key,directory=name,raw=True,cohort=cohort,status='included'))
    for i,(name,reason) in enumerate(final['excluded'].items()):
        runs.append(dict(id=f'EXCLUDED_{i+1}',directory=name,raw=False,cohort='excluded',
                         status='excluded',reason=reason))
    for run in runs:
        if run['id']=='T02_163351':
            run['validity_note']='Legacy INVALID_RUN.md conflicts with final inclusion and user identifying T03 as cable-interfered. Both records retained; see README.'
        if run['id']=='T05_165900':
            run['validity_note']='Operator observed shoulder/torso contact; retained by agreed task-evaluation scope, not a no-contact trial.'

    raw_jobs=[]
    for run in runs:
        src=base/run['directory'];dst=HERE/'runs'/run['directory'];dst.mkdir(parents=True,exist_ok=True)
        raw=src/'raw.jsonl';run['raw_sha256']=digest(raw);run['raw_bytes']=raw.stat().st_size
        if run.get('expected_sha'):assert run.pop('expected_sha')==run['raw_sha256']
        for p in src.iterdir():
            if p.is_file() and p.suffix in ('.yaml','.conf','.json','.md'):
                copy(p,dst/p.name)
        for folder in ('endpoint_analysis','analysis','execution_analysis','field_review'):
            if not (src/folder).is_dir():continue
            for p in sorted((src/folder).rglob('*')):
                if not p.is_file() or 'execution' in p.relative_to(src/folder).parts:continue
                if p.suffix in ('.json','.npz','.png','.md'):
                    copy(p,dst/p.relative_to(src))
        extract_run(src,dst)
        for folder in ('endpoint_analysis','analysis'):
            npz=src/folder/'metrics.npz'
            if npz.exists():preview(npz,dst/'trajectory_preview_50hz.csv')
        if run['raw']:raw_jobs.append((raw,dst/'raw.jsonl.xz'))

    # Keep compact analyses verbatim; generated raw replay streams stay local.
    report_dirs={
        'final_comparison_20261012':base/'final_comparison_20261012',
        'morning_comparison_20261013_final':base/'morning_comparison_20261013_final',
        'phase_comparison_20261014':base/'phase_comparison_20261014',
        'spike_review_20261014':base/'spike_review_20261014',
        'impact_predictability_20261014':base/'impact_predictability_20261014',
        'posture_spike_causality_20261014':base/'posture_spike_causality_20261014',
        'engineering_readiness_20261013':base/'engineering_readiness_20261013',
        'timing_review_20261012_T01':base/'timing_review_20261012_T01',
        'forecast_timing_20261010':ROOT/'evaluation/forecast_timing_20261010',
    }
    for label,src in report_dirs.items():
        for p in sorted(src.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if p.suffix not in ('.md','.json','.csv','.png','.npz','.py','.txt'):continue
            if p.stat().st_size>12_000_000:continue
            copy(p,HERE/'reports'/label/p.relative_to(src))
    optional=[
        base/'spike_control_audit_20261014.md',
        base/'morning_preparation_20261013/baseline_parity.json',
        base/'morning_preparation_20261013/validation_v2/summary.json',
        base/'walk_h0_predictor_study_20260925/benchmark/metrics_aggregate.csv',
        base/'walk_h0_predictor_study_20260925/benchmark/metrics_per_trial.csv',
        base/'walk_h0_predictor_study_20260925/associations.json',
        base/'walk_h0_predictor_study_20260925/legacy_verification.json',
    ]
    for p in optional:
        assert p.exists(),p
        copy(p,HERE/'reports'/p.relative_to(base))
    for p in sorted((ROOT/'evaluation/spike_suppression_20261014/results').glob('*.json')):
        copy(p,HERE/'reports/spike_simulation_results'/p.name)
    p=ROOT/'evaluation/spike_suppression_20261014/bias_decomposition.json'
    if p.exists():copy(p,HERE/'reports/spike_simulation_results'/p.name)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda job:compress(*job), raw_jobs))
    write_json(HERE/'run_index.json',runs)
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ('manifest.json','run_index.json') and p.suffix in ('.md','.py','.csv'):
            record(p,kind='archive_documentation_or_tool')
    meta=dict(schema='g1_curated_field_archive_v1',created_date='2026-10-15',evidence_through='2026-10-14',
              source_repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              full_raw_runs=len(raw_jobs), all_run_records=len(runs),
              scope='Selected evidence only; not a backup of all local evaluation or videos',
              files=sorted(FILES,key=lambda x:x['path']))
    (HERE/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(f'Exported {len(FILES)} files, {sum(f["bytes"] for f in FILES)/1e6:.1f} MB',flush=True)


if __name__=='__main__':main()
