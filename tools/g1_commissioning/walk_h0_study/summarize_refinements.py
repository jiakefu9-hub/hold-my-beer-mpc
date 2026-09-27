#!/usr/bin/env python3
"""Read saved exploratory forecasts, independently verify RMSE, and plot.

No refitting, baseline edits or robot connection. No new-blind-data claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--refinement-dir', type=Path, required=True)
    args = parser.parse_args()
    roots = dict(baseline=args.baseline_dir/'benchmark',
                 **{name:args.refinement_dir/name for name in ('blend','local','nonlinear','innovation')})
    ledgers = {k:json.loads((v/'metrics_aggregate.json').read_text()) for k,v in roots.items()}
    picks = [('baseline',name) for name in ('zoh','legs_qdq_knn','hybrid_switch')]
    picks += [('blend','convex_blend')]
    picks += [('innovation','decayed_innovation')]
    for family in ('local','nonlinear'):
        choices = json.loads((roots[family]/'selection.json').read_text())
        picks += [(family,c['name']) for c in choices]
    def row(family,method,horizon,group='acc',region='full'):
        return next(r for r in ledgers[family] if r['method']==method and
                    r['horizon_ms']==horizon and r['group']==group and r['region']==region)
    inventory_checks = {}
    for family,root in roots.items():
        inv = json.loads((root/'artifact_sha256.json').read_text())
        for name, sha in inv.items():
            if digest(root/name) != sha:
                raise ValueError('modified calculation artifact: '+str(root/name))
        inventory_checks[family] = len(inv)
    records = []
    # Recompute from actual saved errors rather than trusting result tables.
    for family,name in picks:
        error,raw_error = [],[]
        for i in (9,10,11,12):
            path = roots[family]/(f'trial{i:02d}_predictions.npz' if family in ('blend','innovation') else f'predictions/trial{i:02d}.npz')
            d = dict(np.load(path))
            prediction = d['prediction'] if family in ('blend','innovation') else d[name]
            error.append(prediction[:,:,:3]-d['truth'][:,:,:3])
            raw_error.append(prediction[:,:,:3]-d['raw_acc_truth'])
        rmse = np.sqrt(np.mean(np.concatenate(error)**2,axis=(0,2)))
        raw_rmse = np.sqrt(np.mean(np.concatenate(raw_error)**2,axis=(0,2)))
        for j,h in enumerate(range(6,55,6)):
            np.testing.assert_allclose(rmse[j],row(family,name,h)['rmse'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(raw_rmse[j],row(family,name,h,'raw_acc_sensitivity')['rmse'],rtol=0,atol=1e-12)
            zoh = row('baseline','zoh',h)['rmse']
            old_knn = row('baseline','legs_qdq_knn',h)['rmse']
            old_hybrid = row('baseline','hybrid_switch',h)['rmse']
            records.append(dict(family=family,method=name,horizon_ms=h,acc_rmse=float(rmse[j]),
                raw_acc_sensitivity_rmse=float(raw_rmse[j]),
                reduction_vs_zoh_pct=float(100*(1-rmse[j]/zoh)),
                reduction_vs_old_knn_pct=float(100*(1-rmse[j]/old_knn)),
                reduction_vs_old_hybrid_pct=float(100*(1-rmse[j]/old_hybrid))))
    report=dict(status='Exploratory same-session comparison, not new blind validation.',
        definition='Full-window pooled three-scalar-component RMSE; same 15Hz H0 left-interval targets.',
        formula='reduction_percent=100*(1-new_RMSE/reference_RMSE); negative means worse.',
        artifact_hash_checks=inventory_checks,all_direct_rmse_checks_passed=True,
        script_sha256=digest(Path(__file__)),
        input_sha256={str(v/'metrics_aggregate.json'):digest(v/'metrics_aggregate.json') for v in roots.values()},
        rows=records)
    (args.refinement_dir/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    # Plot selected development candidate from each family; not the best test-score row.
    local=json.loads((roots['local']/'selected_candidate.json').read_text())
    nonlinear=json.loads((roots['nonlinear']/'selected_candidate.json').read_text())
    local_name=local.get('method',local.get('name'))
    selected=[('baseline','zoh','Hold current'),('baseline','hybrid_switch','Previous hybrid'),
              ('blend','convex_blend','Convex blend'),('local',local_name,'Leg+current IMU increment kNN'),
              ('nonlinear',nonlinear['name'],'Nonlinear history / RFF'),
              ('innovation','decayed_innovation','Current-IMU correction with learned decay')]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for family,name,label in selected:
        rows=[r for r in records if r['family']==family and r['method']==name]
        if len(rows)!=9:
            raise ValueError('missing candidate for plot: '+str((family,name)))
        axes[0].plot([r['horizon_ms'] for r in rows],[r['acc_rmse'] for r in rows],marker='o',ms=3,label=label)
        axes[1].plot([r['horizon_ms'] for r in rows],[r['raw_acc_sensitivity_rmse'] for r in rows],marker='o',ms=3,label=label)
    axes[0].set_title('Filtered target: same full-stage scoring window')
    axes[1].set_title('Same predictions scored against UNFILTERED target')
    axes[1].plot(list(range(6,55,6)),[row('baseline','zoh',h,'raw_acc_hold_baseline')['rmse'] for h in range(6,55,6)],
                 color='black',linestyle='--',label='Hold RAW current (separate baseline)')
    for ax in axes:
        ax.set_xlabel('Future interval endpoint (ms)');ax.set_ylabel('Acceleration component RMSE (m/s^2)')
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=6)
    fig.suptitle('Already-inspected runs 09-12: exploratory refinement, not blind validation')
    fig.savefig(args.refinement_dir/'comparison.png',dpi=140)
    fig.savefig(args.refinement_dir/'comparison.pdf')
    plt.close(fig)
    print('Verified',len(picks),'methods against saved forecasts and plotted comparison.')


if __name__=='__main__':
    main()
