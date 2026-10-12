"""Export comparison metrics, final YAMLs and physical checks to tracked docs."""
from pathlib import Path
import csv
import json
import shutil
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = REPO/'evaluation/simple_controller_comparison_20261011'
OUT = REPO/'docs/simulation/experiments/20261011_simple_controllers'
CONFIGS = REPO/'configs/simulation_simple_controllers_20261011'
sys.path.insert(0, str(HERE.parent/'mpc_comprehensive'))
from contact_audit import audit

METRICS = ['tilt_mean', 'swing_rms', 'acc', 'alpha', 'speed', 'omega']
SELECTED = {
    'original_pid': 's06_original_pid',
    'ff_pid': 'final_ff_pid',
    'position_mpc': 'final_position_mpc',
    'acceleration_pid': 'final_acceleration_pid',
    'mpc_conservative': 'final_mpc_conservative',
    'mpc_original': 'bench_mpc_original_exact',
    'mpc_acceleration': 'bench_mpc_acceleration',
    'mpc_attitude': 'bench_mpc_attitude',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))]
    lookup = {r['name']: r for r in records}
    rows = []
    for r in records:
        if r.get('complete'):
            for window, values in r['windows'].items():
                for side, metrics in values.items():
                    rows.append(dict(name=r['name'], window=window, side=side, **metrics))
    with (OUT/'all_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT/'all_results.json').write_text(json.dumps(records, indent=2)+'\n')
    selected = {}
    table = ['# 完整任务实测物理指标', '',
             '相同任务 0 ≤ t < 8 s；两种角度单位 °，其余依次 m/s²、rad/s²、m/s、rad/s。', '',
             '| 控制器 | 手臂 | 竖直倾角均值 | 平均轴摆动 RMS | 线加速度 RMS | 角加速度 RMS | 线速度 RMS | 角速度 RMS |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for label, name in SELECTED.items():
        r = lookup[name]
        assert r['complete'] and r['returncode'] == 0
        cfg = yaml.safe_load(Path(r['config']).read_text())
        (CONFIGS/f'{label}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        physical = audit(r)
        torque_max = max(r['right_arm_cpp_executor_actuator_tau_absmax'])
        physical['max_abs_actuator_torque_nm'] = torque_max
        physical['torque_limit_nm'] = 25.
        path = Path(r['path'])
        perf = json.loads((path/'perf_summary.json').read_text())
        timing = perf['total']['real_hardware_control']['right_arm_interval']
        selected[label] = dict(result=r, physical_audit=physical, timing=timing)
        for side in ['left', 'right']:
            m = r['windows']['full'][side]
            table.append('| '+label+' | '+side+' | '+' | '.join(f'{m[k]:.5f}' for k in METRICS)+' |')
    (OUT/'METRICS.md').write_text('\n'.join(table)+'\n')
    (OUT/'selected_results.json').write_text(json.dumps(selected, indent=2)+'\n')
    (OUT/'run_counts.json').write_text(json.dumps(dict(
        attempts=len(records), completed_exports=sum(r['complete'] and r['returncode']==0 for r in records),
        note='Includes integration/debug attempts and repeated validation; not all successful exports pass quality gates.'), indent=2)+'\n')
    for name in ('validation_cases.json', 'validation_fix_cases.json', 'serial_cases.json'):
        shutil.copy2(ROOT/name, OUT/name)
    final_source = ROOT/'source_final'
    for p in list(HERE.glob('*.py')) + [REPO/'sim_support.py']:
        dest = final_source/p.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    print(json.dumps({label:dict(physical=s['physical_audit'], timing=s['timing']) for label,s in selected.items()},indent=2))


if __name__ == '__main__':
    main()
