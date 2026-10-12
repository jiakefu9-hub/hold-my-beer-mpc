"""Rebuild C refinement tables from saved results; no physical output."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys

import yaml

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ROOT=REPO/'evaluation/acceleration_pid_refinement_20261012'
OLD=REPO/'evaluation/simple_controller_comparison_20261011'
OUT=REPO/'docs/simulation/experiments/20261012_c_refinement'
CONFIGS=REPO/'configs/simulation_acceleration_pid_20261012'
sys.path.insert(0,str(HERE.parent/'mpc_comprehensive'))
from contact_audit import audit

METRICS=['tilt_mean','swing_rms','acc','alpha','speed','omega']
SELECTED={
    'C_original':(ROOT,'val_c_original_nominal'),
    'C_original_with_constraints':(ROOT,'val_c_original_bounded_nominal'),
    'C_attitude':(ROOT,'val_c_attitude_nominal'),
    'C_acceleration_tradeoff':(ROOT,'val_c_acceleration_nominal'),
    'C_deployment_constraints_sim':(ROOT,'val_c_deploy_nominal'),
    'C_relaxed_clip_simulation_only':(ROOT,'r6_relaxed_post4'),
    'C_relaxed_not_field_selected':(ROOT,'val_c_field_relaxed_nominal'),
    'C_field_candidate':(ROOT,'serial_c_field_final_unloaded'),
    'MPC_original':(OLD,'bench_mpc_original_exact'),
    'MPC_conservative':(OLD,'final_mpc_conservative'),
    'MPC_acceleration':(OLD,'bench_mpc_acceleration'),
    'MPC_attitude':(OLD,'bench_mpc_attitude'),
}


def table(rows):
    lines=['| 方案 | 手臂 | 竖直倾角均值 ° | 相对平均轴摆动 RMS ° | 线加速度 m/s² | 角加速度 rad/s² | 线速度 m/s | 角速度 rad/s |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,side,m in rows:
        lines.append('| '+name+' | '+side+' | '+' | '.join(f'{m[k]:.5f}' for k in METRICS)+' |')
    return '\n'.join(lines)+'\n'


def main():
    OUT.mkdir(parents=True,exist_ok=True);CONFIGS.mkdir(parents=True,exist_ok=True)
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))]
    rows=[];selected={}
    for label,(root,name) in SELECTED.items():
        r=json.loads((root/'results'/f'{name}.json').read_text())
        assert r['complete'] and r['returncode']==0
        physical=audit(r)
        physical['max_abs_actuator_torque_nm']=max(r['right_arm_cpp_executor_actuator_tau_absmax'])
        physical['actuator_limit_nm']=25.
        selected[label]=dict(result=r,physical_audit=physical)
        if label.startswith('C_'):
            shutil.copy2(r['config'],CONFIGS/(label+'.yaml'))
        for side in ['left','right']:rows.append((label,side,r['windows']['full'][side]))
    (OUT/'METRICS.md').write_text('# 同一完整任务左右臂指标\n\n0 ≤ t < 8 s；除两种倾角外均为三维向量 RMS。主表为原 200 mm 负载。\n\n'+table(rows))
    valrows=[]
    for r in records:
        if r['name'].startswith('val_') and r['complete']:
            for side in ['left','right']:valrows.append((r['name'],side,r['windows']['full'][side]))
    (OUT/'VALIDATION_METRICS.md').write_text('# 独立初态、延迟、75 mm 负载复核\n\n'+table(valrows))
    counts=dict(attempts=len(records),completed=sum(r['complete'] and r['returncode']==0 for r in records),
        tuning_attempts=sum(r['name'].startswith('r') for r in records),
        tuning_completed=sum(r['name'].startswith('r') and r['complete'] for r in records),
        failures=[dict(name=r['name'],returncode=r['returncode'],log=str(ROOT/'logs'/(r['name']+'.log')))
                  for r in records if not r['complete'] or r['returncode']!=0])
    (OUT/'run_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    startup = ROOT/'startup_failures_missing_tmp_builds'
    (OUT/'startup_failures.json').write_text(json.dumps(dict(
        count=len(list(startup.glob('*.json'))), included_in_simulation_counts=False,
        reason='missing /tmp native library; no simulation initialized; rebuilt and reran same cases',
        retained_directory=str(startup)),indent=2)+'\n')
    (OUT/'all_results.json').write_text(json.dumps(records,indent=2)+'\n')
    (OUT/'selected_results.json').write_text(json.dumps(selected,indent=2)+'\n')
    with (OUT/'all_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['name','window','side',*METRICS]);writer.writeheader()
        for r in records:
            for window,values in r.get('windows',{}).items():
                for side,m in values.items():writer.writerow(dict(name=r['name'],window=window,side=side,**{k:m[k] for k in METRICS}))
    for p in HERE.glob('c_*.json'):shutil.copy2(p,OUT/p.name)
    for name in ('round6_contacts.json','field_candidates_contacts.json','final_candidate_contacts.json'):
        shutil.copy2(ROOT/name,OUT/name)
    serial={}
    for r in records:
        if r['name'].startswith('serial_') and r['complete']:
            perf=json.loads((Path(r['path'])/'perf_summary.json').read_text())
            serial[r['name']]=perf['total']['real_hardware_control']['right_arm_interval']
    (OUT/'serial_timing.json').write_text(json.dumps(serial,indent=2)+'\n')
    sources=list(HERE.glob('*.py'))+list((REPO/'tools/g1_commissioning').glob('*.py'))
    hashes={}
    for p in sources:
        dest=ROOT/'source_final'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest);hashes[str(p.relative_to(REPO))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(counts=counts,serial=serial),indent=2))


if __name__=='__main__':main()
