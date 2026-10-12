"""Export reviewable evidence for the C/MPC frequency and preview study."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ROOT=REPO/'evaluation/c_mpc_followup_20261012'
DOC=REPO/'docs/simulation/experiments/20261012_c_mpc_followup'
FIELDS=['tilt_mean','swing_rms','acc','alpha','speed','omega']


def table(rows):
    text=['| 方案 / 手臂 | 竖直倾角均值 ° | 平均轴摆动 RMS ° | 线加速度 RMS m/s² | 角加速度 RMS rad/s² | 线速度 RMS m/s | 角速度 RMS rad/s |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for name,values in rows:
        text.append('| '+name+' | '+' | '.join(f'{values[k]:.5f}' for k in FIELDS)+' |')
    return '\n'.join(text)+'\n'


def export():
    DOC.mkdir(parents=True,exist_ok=True)
    results={p.stem:json.loads(p.read_text()) for p in (ROOT/'results').glob('*.json')}
    counts=dict(attempts=len(results),complete=sum(r['complete'] for r in results.values()))
    counts['initialization_rejections']=[n for n,r in results.items() if not r['complete'] and
        any('experimental' in l and 'latency' in l for l in r.get('failure_details',[]))]
    counts['controller_failures']=[n for n,r in results.items() if not r['complete'] and n not in counts['initialization_rejections']]
    counts['physical_simulation_attempts']=counts['attempts']-len(counts['initialization_rejections'])
    counts['complete_but_smoke_gate_failed']=[n for n,r in results.items()
        if r['complete'] and r['gates']['status']!='PASS']
    counts['complete_and_smoke_gate_passed']=sum(r['complete'] and
        r.get('gates',{}).get('status')=='PASS' for r in results.values())
    rows=[];data=[]
    for n,r in sorted(results.items()):
        if r['complete']:
            for side in ['left','right']:
                v=r['windows']['full'][side]
                rows.append((n+' / '+side,v))
                data.append(dict(name=n,side=side,**v))
    (DOC/'ALL_METRICS.md').write_text('# 完整 0～8 s，逐轮左右臂\n\n'+table(rows))
    with (DOC/'all_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (DOC/'run_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    # Full result JSON includes independent source hashes and timing environments.
    (DOC/'evidence_results.json').write_text(json.dumps(results,indent=2)+'\n')
    for p in (ROOT/'configs').glob('*.yaml'):
        dest=DOC/'configs'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
    groups={}
    for n,r in results.items():
        if n.startswith('serial_'):
            key=n.rsplit('_',1)[0].removeprefix('serial_')
            groups.setdefault(key,[]).append(dict(name=n,**r['full_interval_ms']))
    (DOC/'timing_repetitions.json').write_text(json.dumps(groups,indent=2)+'\n')
    timing=['| 方案 | 周期 ms | 均值范围 ms | P99 范围 ms | 最大值 ms | 超期 / 样本 |',
            '|---|---:|---:|---:|---:|---:|']
    for n,rs in sorted(groups.items()):
        timing.append(f'| {n} | {rs[0]["budget_ms"]:.0f} | '+
            f'{min(r["mean"] for r in rs):.3f}～{max(r["mean"] for r in rs):.3f} | '+
            f'{min(r["p99"] for r in rs):.3f}～{max(r["p99"] for r in rs):.3f} | '+
            f'{max(r["max"] for r in rs):.3f} | '+
            f'{sum(r["overrun_count"] for r in rs)} / {sum(r["count"] for r in rs)} |')
    (DOC/'TIMING.md').write_text('# CPU 7 串行三次重复\n\n'+ '\n'.join(timing)+'\n\n'+
        'SCHED_OTHER、单线程 BLAS；包含预测、策略与区间内全部力矩映射，排除 MuJoCo 物理推进和绘图。'+
        '这不是 DDS、传感器到电机延迟或真机实时性证明。并行探索的耗时仍在 evidence_results.json 中保留。\n')
    selected=['old6','field6fast','field4_post36','mpc6_n9','mpc_attitude_alpha02',
        'mpc6_n1','mpc6_n3','mpc6_n6','mpc6_current_hold','old6_current_hold']
    (DOC/'SELECTED_METRICS.md').write_text('# 本次主要对照\n\n'+table([(n,results[n]['windows']['full']['right']) for n in selected]))
    parity={}
    for k in FIELDS:
        parity[k]=results['old6fast']['windows']['full']['right'][k]-results['old6']['windows']['full']['right'][k]
    (DOC/'efficiency_parity.json').write_text(json.dumps(dict(metric_difference=parity,
        mathematical_algorithm_unchanged=True,bitwise_equal_trajectory=False),indent=2)+'\n')
    # Keep the final research sources alongside the initial archive locally.
    for src in list(HERE.glob('*.py'))+list(HERE.glob('*.json')):
        target=ROOT/'source_after'/src.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    if Path('/tmp/g1-c-followup-tests.txt').exists():
        shutil.copy2('/tmp/g1-c-followup-tests.txt',DOC/'tests.txt')
    print(json.dumps(counts));print('\n'.join(timing))


if __name__=='__main__':export()
