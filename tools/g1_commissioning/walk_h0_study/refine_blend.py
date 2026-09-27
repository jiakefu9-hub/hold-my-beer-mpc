#!/usr/bin/env python3
"""Explore convex blending of existing OUT-OF-FOLD forecasts, offline only.

Baseline remains immutable. New weights see only development OOF predictions.
Comparison runs09--12 have already been inspected, so are NOT fresh blind data.
No pseudo-CV of these stacking weights: other OOF models can include a held-out
meta-fold in their base fits. We report weight-fit error only as training error.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import benchmark as b
import methods as m

BASE_NAMES = ('imu_legs_history_ridge', 'legs_qdq_knn')


def fit_weights(near, far, truth):
    """Least-squares w in [0,1] for w*near+(1-w)*far, per horizon/group."""
    w = np.zeros((9, 3))
    for j in range(9):
        for g in range(3):
            sl = slice(g*3, g*3+3)
            delta, error = near[:, j, sl]-far[:, j, sl], far[:, j, sl]-truth[:, j, sl]
            w[j, g] = np.clip(-np.sum(delta*error)/max(np.sum(delta*delta), 1e-15), 0., 1.)
    return w


def blend(near, far, weights):
    result = near.copy()  # Orientation stays on the existing history predictor.
    for g in range(3):
        sl = slice(g*3, g*3+3)
        result[:, :, sl] = (weights[None, :, g, None]*near[:, :, sl] +
                            (1-weights[None, :, g, None])*far[:, :, sl])
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = args.study_dir/'benchmark'
    paths = [base/f'development_oof/trial{i:02d}.npz' for i in b.DEVELOPMENT]
    protocol = dict(development=b.DEVELOPMENT, comparison=b.HELDOUT, excluded=6,
        status='Exploratory refinement; comparison09-12 previously inspected, not fresh blind evaluation.',
        candidate='Convex combination of frozen IMU+leg ridge and full-leg kNN, 27 weights: 9horizons x3physicalgroups.',
        orientation='Copied from history model; excluded from weight fitting.',
        weight_rule='clip(-sum((near-far)*(far-truth))/sum((near-far)^2),0,1), development OOF only.',
        cv_caveat='Weight fitting is not itself an independent CV score; base hyperparameters were already selected on development.',
        target='Unchanged H0/15Hz/LEFT 6ms interval and node targets; same anchors and all stages.',
        input_sha256={str(p):digest(p) for p in paths}, script_sha256=digest(Path(__file__)),
        dependency_sha256={str(Path(x.__file__)):digest(Path(x.__file__)) for x in (m,b)})
    b.dump(args.output_dir/'protocol.json', protocol)
    oof = [dict(np.load(path)) for path in paths]
    near, far, truth = [np.concatenate([d[key] for d in oof]) for key in (*BASE_NAMES, 'truth')]
    weights = fit_weights(near, far, truth)
    np.savez_compressed(args.output_dir/'model.npz', weights=weights, base_names=BASE_NAMES)
    b.dump(args.output_dir/'weights.json', dict(horizons_ms=m.HORIZONS_MS.tolist(),
        group_order=['acc','omega','alpha'], history_weights=weights.tolist(),
        complementary_leg_weights=(1-weights).tolist()))
    # Checks of the analytic solution and saved model round-trip, not a new score.
    rebuilt_weights = np.load(args.output_dir/'model.npz')['weights']
    np.testing.assert_array_equal(rebuilt_weights, weights)
    training = blend(near, far, weights)
    for j in range(9):
        for g in range(3):
            sl = slice(3*g, 3*g+3)
            loss = np.mean((training[:, j, sl]-truth[:, j, sl])**2)
            ends = [np.mean((x[:, j, sl]-truth[:, j, sl])**2) for x in (near, far)]
            if loss > min(ends)+1e-12:
                raise AssertionError('constrained solution worse than an endpoint on weight-fit data')
    threshold = json.loads((base/'protocol.json').read_text())['large_acc_threshold_m_s2']
    rows = []
    all_comparisons = []
    for i in b.HELDOUT:
        src = base/f'predictions/trial{i:02d}.npz'
        d = dict(np.load(src))
        pred = blend(d[BASE_NAMES[0]], d[BASE_NAMES[1]], weights)
        record = dict(np.load(args.study_dir/f'data/trial{i:02d}_prepared.npz'))
        rows += b.metrics(pred, d['truth'], d['raw_acc_truth'], d['raw_acc_hold'], record,
            d['anchors'], 'convex_blend', i, threshold)
        np.savez_compressed(args.output_dir/f'trial{i:02d}_predictions.npz', prediction=pred,
            truth=d['truth'], time=d['time'], anchors=d['anchors'],
            raw_acc_truth=d['raw_acc_truth'], raw_acc_hold=d['raw_acc_hold'])
        all_comparisons.append(dict(trial=i, input_sha256=digest(src)))
    b.dump(args.output_dir/'metrics_per_trial.json', rows)
    aggregate = []
    for region in (*b.REGIONS, 'large_filtered_acc'):
        for h in m.HORIZONS_MS:
            for group in (*m.GROUPS, 'raw_acc_sensitivity', 'raw_acc_hold_baseline', 'orientation_geodesic'):
                rr = [r for r in rows if r['region']==region and r['horizon_ms']==h and r['group']==group]
                if rr:
                    sse, count = sum(r['sse'] for r in rr), sum(r['scalar_count'] for r in rr)
                    aggregate.append(dict(method='convex_blend',region=region,horizon_ms=int(h),
                        group=group, rmse=float(np.sqrt(sse/count)),sse=sse,scalar_count=count))
    b.dump(args.output_dir/'metrics_aggregate.json', aggregate)
    b.dump(args.output_dir/'comparison_sources.json',all_comparisons)
    b.dump(args.output_dir/'artifact_sha256.json',{p.name:digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()})
    print('acc full:',[(r['horizon_ms'],r['rmse']) for r in aggregate if r['group']=='acc' and r['region']=='full'],flush=True)
    print('acc weights:',weights[:,0].tolist(),flush=True)


if __name__ == '__main__':
    main()
