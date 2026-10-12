"""Build the audit tables and standalone figure from preserved experiment data."""
from pathlib import Path
import json
import hashlib
import shutil
import numpy as np
import yaml
from analyze import read_results, eligible, METRICS, main as refresh_ledger
from study import ROOT, REPO

OUT = REPO/'docs/simulation/experiments/20261010_mpc_comprehensive'
LABELS = {'baseline':'原仿真基线','conservative':'保守范围方案',
          'acceleration_balanced':'加速度综合方案','attitude_balanced':'姿态综合方案'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    refresh_ledger()
    shutil.copy2(ROOT/'all_results.csv',OUT/'all_results.csv')
    results=read_results(); byname={r['name']:r for r in results}
    chosen=json.loads((ROOT/'selection.json').read_text())
    selected={k:byname[v] for k,v in chosen.items()}
    base=selected['baseline']['windows']['full']['right']
    complete=[r for r in results if r['complete'] and r['returncode']==0]
    validation=[r for r in results if r['name'][0] in 'cdefv']
    (OUT/'validation_results.json').write_text(json.dumps(validation,indent=2)+'\n')
    unique=set()
    for r in complete:
        c=yaml.safe_load(Path(r['config']).read_text())
        if c.get('research_rotation_innovation_gain'):c.setdefault('research_rotation_innovation_decay_s',.15)
        for a,b in [('research_rotation_innovation_gain','mpc_preview_rotation_correction_gain'),
                    ('research_rotation_innovation_decay_s','mpc_preview_rotation_decay_s'),
                    ('research_interval_lead_s','mpc_preview_interval_lead_s')]:
            if a in c:c[b]=c.pop(a)
        unique.add(json.dumps(c,sort_keys=True))
    summary=dict(attempts=len(results),complete=len(complete),eligible=sum(eligible(r) for r in results),
        distinct_complete_configurations=len(unique),selected=selected)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    metric_headers=['方案 / 手臂','平均竖直倾角 °','平均轴摆动 RMS °','线加速度 m/s²','角加速度 rad/s²','线速度 m/s','角速度 rad/s']
    rows=[]; gains=[]
    for label,r in selected.items():
        for side in ('left','right'):
            w=r['windows']['full'][side]
            rows.append([LABELS[label]+(' / 左 PD' if side=='left' else ' / 右 MPC')]+[f'{w[k]:.4f}' for k in METRICS])
        if label!='baseline':gains.append([LABELS[label]]+[f'{100*(1-r["windows"]["full"]["right"][k]/base[k]):+.2f}%' for k in METRICS])
    (OUT/'METRICS.md').write_text('# 完整 0–8 s 窗口\n\n'+table(metric_headers,rows)+'\n正值表示相对原右臂降低，负值表示恶化。\n\n'+table(metric_headers,gains))
    phys=[]
    for label,r in selected.items():
        phys.append([LABELS[label],f'{r["gates"]["xy_displacement_m"]:.4f}',
            ', '.join(f'{x:.2f}' for x in np.rad2deg(r['right_arm_q_absmax'])),
            ', '.join(f'{x:.2f}' for x in r['right_arm_cpp_executor_actuator_tau_absmax']),
            f'{max(r["right_arm_dq_absmax"]):.3f}',
            r['gates']['runtime_mapping_safety_fallback_count']])
    (OUT/'LIMITS.md').write_text('# 运动与执行复核\n\n'+table(['方案','行走距离 m','各关节最大绝对角 °','各电机力矩峰值 Nm','关节速度峰值 rad/s','映射安全回退'],phys)+'\n关节顺序：肩 pitch、roll、yaw、肘 pitch、腕 roll。实际瞬时 qacc 含约 0.024 s 的初始接触冲击，不能把 MPC 期望加速度上限当成实际加速度硬限。接触复核见 summary.json 中 contact_audit。\n')
    vrows=[]
    for r in validation:
        if r['name'].startswith('f'):continue
        g=r.get('gates',{});w=r.get('windows',{}).get('full',{}).get('right',{})
        vrows.append([r['name'],g.get('status','中止'),g.get('qp_failure_count','—'),g.get('runtime_mapping_safety_fallback_count','—')]+[f'{w[k]:.4f}' if k in w else '—' for k in METRICS])
    (OUT/'VALIDATION.md').write_text('# 全部复核结果（含失败）\n\n'+table(['实验','任务门槛','QP 回退','映射回退']+metric_headers[1:],vrows))
    timings=[];trows=[]
    for r in results:
        if not r.get('formal_entry') or not r.get('path'):continue
        perf=json.loads((Path(r['path'])/'perf_summary.json').read_text())
        section=perf['total']['real_hardware_control'];t=section['right_arm_interval'];qp=section['mpc_policy_update']
        timings.append(dict(name=r['name'],candidate=r['candidate'],interval=t,mpc_update=qp))
        trows.append([r['name']]+[f'{t[k]:.3f}' for k in ['mean','p95','p99','max']]+[f'{t["overrun_count"]}/{t["count"]}',f'{qp["p99"]:.3f}'])
    (OUT/'TIMING.json').write_text(json.dumps(timings,indent=2)+'\n')
    (OUT/'TIMING.md').write_text('# CPU 7 正式入口串行计时\n\n各配置重复三次；停止并行扫描后测量。数据为完整右臂区间累计计算时间，预算 6 ms；并非只统计 OSQP。\n\n'+table(['实验','平均 ms','P95 ms','P99 ms','最大 ms','超期 / 区间数','MPC update P99 ms'],trows)+'\n本机当前 SCHED_OTHER / powersave，结果不是硬实时准入证明。完整全身接触检查只用于仿真，真机执行路径不同；不能把仿真区间耗时直接视为真机耗时。\n')
    files=[REPO/'disturbance_template/preview_correction.py',REPO/'disturbance_predictor.py',REPO/'right_arm_runtime/sim_mpc_latency.py']+list((REPO/'tools/experiments/mpc_comprehensive').glob('*.py'))+list((REPO/'configs/simulation_optimization_20261010').glob('*.yaml'))
    (OUT/'source_sha256.json').write_text(json.dumps({str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Tilt to vertical','Swing about mean axis','Linear acceleration','Angular acceleration','Linear speed','Angular speed']
    fig,axs=plt.subplots(2,3,figsize=(11,6),constrained_layout=True)
    names=list(selected)
    for ax,k,title in zip(axs.flat,METRICS,labels):
        values=[selected[n]['windows']['full']['right'][k]/base[k] for n in names]
        ax.bar(range(4),values,color=['#888888','#46958b','#3378bc','#db8936'])
        ax.axhline(1,color='black',lw=.8,ls='--');ax.set_ylim(0,1.15)
        ax.set_xticks(range(4),['Baseline','Conserv.','Accel.','Attitude'],rotation=15);ax.set_title(title)
        for i,v in enumerate(values):ax.text(i,v+.02,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_ylabel('Ratio to baseline')
    fig.suptitle('Right endpoint, full 0–8 s task; identical gait and left PD\nLower ratios alone do not establish hardware performance')
    fig.savefig(OUT/'comparison.svg');fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='selected'}))


if __name__=='__main__':build()
