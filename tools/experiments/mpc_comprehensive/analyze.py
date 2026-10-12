"""Summarize every attempted rollout, including failures, without hiding tradeoffs."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3] / 'evaluation/mpc_comprehensive_20261010'
METRICS = ['tilt_mean', 'swing_rms', 'acc', 'alpha', 'speed', 'omega']


def read_results():
    return sorted([json.loads(p.read_text()) for p in (ROOT / 'results').glob('*.json')],
                  key=lambda r: r['name'])


def eligible(r):
    if not r['complete'] or r['returncode']:
        return False
    g = r['gates']; w = r['windows']['full']
    return (g['status'] == 'PASS' and not g['fallen'] and g['tail_complete']
        and g['qp_failure_count'] == 0 and g['nan_inf_count'] == 0
        and g['runtime_mapping_safety_fallback_count'] == 0
        and 3.061 <= g['xy_displacement_m'] <= 3.384
        and w['right']['tilt_mean'] <= w['left']['tilt_mean']
        and w['right']['swing_rms'] <= w['left']['swing_rms'])


def score(r):
    if not eligible(r):
        return float('inf')
    w = r['windows']['full']['right']
    # Ranking aid only; final selection reports each metric and Pareto tradeoffs.
    # Absolute angular tolerance avoids rewarding tiny bias reductions over
    # acceleration performance; the left-arm limits remain hard eligibility gates.
    return (w['acc']/2.94151 + .45*w['alpha']/7.04514 + .25*w['omega']/.05958
            + .12*w['tilt_mean'] + .12*w['swing_rms'])


def main():
    rs = read_results()
    rows = []
    for r in rs:
        row = dict(name=r['name'], complete=r['complete'], returncode=r['returncode'],
                   eligible=eligible(r), scenario=r.get('scenario','nominal'), score=score(r))
        if r['complete']:
            for side in ('left', 'right'):
                row.update({side+'_'+k: r['windows']['full'][side][k] for k in METRICS})
            row.update(r['gates'])
        rows.append(row)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT/'all_results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    print('attempts',len(rs),'complete',sum(r['complete'] and not r['returncode'] for r in rs),
          'eligible',sum(eligible(r) for r in rs))
    for r in sorted(rs,key=score)[:20]:
        if not eligible(r): continue
        w=r['windows']['full']['right']
        print(r['name'], ' '.join(f'{k}={w[k]:.3f}' for k in METRICS),f"distance={r['gates']['xy_displacement_m']:.3f}")


if __name__ == '__main__':
    main()
