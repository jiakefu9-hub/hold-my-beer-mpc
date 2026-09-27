#!/usr/bin/env python3
"""Small, predeclared offline local-predictor refinement; NEVER controls a robot.

Uses exactly benchmark.py's episodes, anchors, causal filtered H0 inputs, and
LEFT interval targets. The four comparison episodes have already been inspected;
results are exploratory reuse, NOT an independent new blind test.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

import benchmark as b
import methods as m

# Fixed before examining any refinement result. No post-result grid extensions.
CONFIGS = (
    dict(name="legs_current_imu_knn8", family="imu_lookup", k=8, residual=False),
    dict(name="legs_current_imu_knn24", family="imu_lookup", k=24, residual=False),
    dict(name="legs_current_imu_delta_knn8", family="imu_lookup", k=8, residual=True),
    dict(name="legs_current_imu_delta_knn24", family="imu_lookup", k=24, residual=True),
    dict(name="legs_local_linear64_ridge01", family="local_linear", k=64, ridge=.1, residual=False),
    dict(name="legs_local_linear64_ridge1", family="local_linear", k=64, ridge=1., residual=False),
)


def features(d, idx, config):
    legs = np.column_stack((d["qf"][idx], d["dqf"][idx]))
    if config["family"] == "imu_lookup":
        # Three physical IMU groups; orientation is deliberately not in distance.
        return np.column_stack((legs, d["y"][idx, :9]))
    return legs


def fit(x, y, current, config):
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std > 1e-6, std, 1.)
    scale = np.ones(x.shape[1])
    if config["family"] == "imu_lookup":
        scale[24:] = .5  # fixed, not tuned using comparison episodes
    z = (x-mean)/std*scale
    target = y-current if config["residual"] else y
    return dict(mean=mean, std=std, feature_scale=scale, train_z=z,
                train_y=target, tree=cKDTree(z), **config)


def predict(model, x, current, batch=64):
    z = (x-model["mean"])/model["std"]*model["feature_scale"]
    distance, indices = model["tree"].query(z, k=model["k"], workers=1)
    weights = 1./np.maximum(distance, .001)
    weights /= weights.sum(axis=1, keepdims=True)
    result = np.empty_like(current)
    for start in range(0, len(x), batch):
        stop = min(start+batch, len(x))
        w, ix = weights[start:stop], indices[start:stop]
        yy = model["train_y"][ix]
        ym = np.einsum("bk,bko->bo", w, yy)
        if model["family"] == "local_linear":
            xx = model["train_z"][ix]
            xm = np.einsum("bk,bkf->bf", w, xx)
            xc, yc = xx-xm[:, None, :], yy-ym[:, None, :]
            cov = np.einsum("bk,bkf,bkg->bfg", w, xc, xc)
            cov += model["ridge"]*np.eye(x.shape[1])[None, :, :]
            rhs = np.einsum("bk,bkf,bko->bfo", w, xc, yc)
            coef = np.linalg.solve(cov, rhs)
            ym += np.einsum("bf,bfo->bo", z[start:stop]-xm, coef)
        result[start:stop] = ym
    if model["residual"]:
        result += current
    return result


def smoke_tests():
    """Prefix feature equality plus exact affine and residual toy predictions."""
    rng = np.random.default_rng(349)
    d = dict(qf=rng.normal(size=(100, 12)), dqf=rng.normal(size=(100, 12)),
             y=rng.normal(size=(100, 12)))
    for c in CONFIGS:
        assert np.array_equal(features(d, np.array([50, 60]), c),
            features({k: v[:61] for k, v in d.items()}, np.array([50, 60]), c))
    xx = rng.normal(size=(100, 3))
    coef = rng.normal(size=(3, 5))
    yy = xx@coef+2.
    c = dict(name="toy", family="local_linear", k=64, ridge=1e-10, residual=False)
    fitted = fit(xx, yy, np.zeros_like(yy), c)
    pred = predict(fitted, xx[:10], np.zeros((10, 5)))
    assert np.max(np.abs(pred-yy[:10])) < 1e-7
    # A constant future increment remains that increment at any query baseline.
    c = dict(name="toy", family="imu_lookup", k=8, residual=True)
    fitted = fit(xx, yy+3., yy, c)
    assert np.max(np.abs(predict(fitted, xx[:10], yy[:10])-yy[:10]-3.)) < 1e-12
    return dict(causal_prefix=True, exact_affine=True, residual_reconstruction=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("models", "predictions", "development_oof"):
        (args.output_dir/name).mkdir()
    tests = smoke_tests()
    paths = {i: args.baseline_dir/"data"/f"trial{i:02d}_prepared.npz"
             for i in b.DEVELOPMENT+b.HELDOUT}
    ds = {i: dict(np.load(path)) for i, path in paths.items()}
    idx = {i: m.anchors(d) for i, d in ds.items()}
    ys = {i: m.targets(d, idx[i]) for i, d in ds.items()}
    zs = {i: m.hold(d, idx[i]) for i, d in ds.items()}
    source_paths = (Path(__file__), Path(m.__file__), Path(b.__file__))
    protocol = dict(configs=CONFIGS, development=b.DEVELOPMENT, comparison=b.HELDOUT,
        excluded=6, old_five_excluded=True, no_robot=True, deployable=False,
        caveat="Previously inspected comparison trials 9..12; exploratory refinement, not a new blind test.",
        selection="Equal mean episode LOOCV score from methods.score: acc,omega,alpha, all nine horizons. No comparison metrics used.",
        targets="Exactly baseline causal 15Hz H0 signals, LEFT samples h-6,h-4,h-2ms for acc/alpha, endpoint for omega/orientation.",
        inputs="Train-fold standardized qf,dqf; appended present acc,omega,alpha weighted0.5 after scaling for imu_lookup. No future inputs.",
        local_linear="64 nearest leg-state neighbors; inverse-distance normalized weights; weighted centered ridge slopes, unpenalized intercept; ridge0.1 or1.",
        residual="Fit future_label-current_filtered_y per horizon; add current query y back after neighbor averaging.",
        smoke_tests=tests, source_sha256={str(path): b.sha(path) for path in source_paths},
        input_sha256={str(path): b.sha(path) for path in paths.values()})
    b.dump(args.output_dir/"protocol.json", protocol)
    selection, oof = [], {}
    for config in CONFIGS:
        started = time.monotonic()
        name = config["name"]
        xs = {i: features(ds[i], idx[i], config) for i in b.DEVELOPMENT}
        folds, oof[name] = [], {}
        for val in b.DEVELOPMENT:
            train = [i for i in b.DEVELOPMENT if i != val]
            fitted = fit(np.vstack([xs[i] for i in train]),
                np.vstack([ys[i].reshape(len(idx[i]), -1) for i in train]),
                np.vstack([zs[i].reshape(len(idx[i]), -1) for i in train]), config)
            pred = predict(fitted, xs[val], zs[val].reshape(len(idx[val]), -1)).reshape(ys[val].shape)
            oof[name][val] = pred
            folds.append(dict(trial=val, score=m.score(pred, ys[val], zs[val])))
        row = dict(**config, folds=folds, mean_score=float(np.mean([r["score"] for r in folds])))
        selection.append(row)
        print(name, "CV", round(row["mean_score"], 5), "seconds", round(time.monotonic()-started, 1), flush=True)
        b.dump(args.output_dir/"selection_progress.json", selection)
    winner = min(selection, key=lambda row: row["mean_score"])["name"]
    # Selection persisted before evaluating ANY comparison forecast.
    b.dump(args.output_dir/"selection.json", selection)
    b.dump(args.output_dir/"selected_candidate.json", dict(method=winner,
        rule="Smallest development-only LOOCV methods.score; comparison unused in selection."))
    for i in b.DEVELOPMENT:
        np.savez_compressed(args.output_dir/"development_oof"/f"trial{i:02d}.npz",
            time=ds[i]["t"][idx[i]], anchors=idx[i], truth=ys[i],
            **{name: oof[name][i] for name in oof})
    del oof
    threshold = float(np.percentile(np.concatenate([
        np.linalg.norm(ys[i][:, :, :3], axis=2).ravel() for i in b.DEVELOPMENT]), 90))
    all_pred = {i: {} for i in b.HELDOUT}
    for config in CONFIGS:
        name = config["name"]
        fitted = fit(np.vstack([features(ds[i], idx[i], config) for i in b.DEVELOPMENT]),
            np.vstack([ys[i].reshape(len(idx[i]), -1) for i in b.DEVELOPMENT]),
            np.vstack([zs[i].reshape(len(idx[i]), -1) for i in b.DEVELOPMENT]), config)
        # Save EVERY candidate to allow checking unsuccessful variants as well.
        arrays = {key: val for key, val in fitted.items() if key != "tree"}
        arrays.update(train_trial=np.concatenate([np.full(len(idx[i]), i) for i in b.DEVELOPMENT]),
                      train_anchor=np.concatenate([idx[i] for i in b.DEVELOPMENT]))
        np.savez_compressed(args.output_dir/"models"/f"{name}.npz", **arrays)
        for i in b.HELDOUT:
            all_pred[i][name] = predict(fitted, features(ds[i], idx[i], config),
                zs[i].reshape(len(idx[i]), -1)).reshape(ys[i].shape)
    rows = []
    for i in b.HELDOUT:
        baseline_path = args.baseline_dir/"benchmark"/"predictions"/f"trial{i:02d}.npz"
        baseline = dict(np.load(baseline_path))
        assert np.array_equal(idx[i], baseline["anchors"])
        assert np.array_equal(ys[i], baseline["truth"])
        raw = m.targets(ds[i], idx[i], "acc_raw")
        assert np.array_equal(raw, baseline["raw_acc_truth"])
        raw_hold = ds[i]["acc_raw"][idx[i]]
        predictions = all_pred[i]
        for name in ("zoh", "legs_qdq_knn", "imu_legs_history_ridge", "hybrid_switch"):
            predictions[name] = baseline[name]
        np.savez_compressed(args.output_dir/"predictions"/f"trial{i:02d}.npz",
            anchors=idx[i], time=ds[i]["t"][idx[i]], truth=ys[i], raw_acc_truth=raw,
            raw_acc_hold=raw_hold, horizons_ms=m.HORIZONS_MS, **predictions)
        for name, pred in predictions.items():
            rows += b.metrics(pred, ys[i], raw, raw_hold, ds[i], idx[i], name, i, threshold)
    b.dump(args.output_dir/"metrics_per_trial.json", rows)
    aggregate = []
    keys = sorted({(r["method"], r["region"], r["horizon_ms"], r["group"]) for r in rows})
    for name, region, h, group in keys:
        selected = [r for r in rows if (r["method"], r["region"], r["horizon_ms"], r["group"]) == (name, region, h, group)]
        sse, count = sum(r["sse"] for r in selected), sum(r["scalar_count"] for r in selected)
        aggregate.append(dict(method=name, region=region, horizon_ms=h, group=group,
            rmse=float(np.sqrt(sse/count)), sse=sse, scalar_count=count,
            min_trial_rmse=min(r["rmse"] for r in selected), max_trial_rmse=max(r["rmse"] for r in selected)))
    b.dump(args.output_dir/"metrics_aggregate.json", aggregate)
    with (args.output_dir/"metrics_aggregate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    b.dump(args.output_dir/"artifact_sha256.json", {
        str(path.relative_to(args.output_dir)): b.sha(path)
        for path in sorted(args.output_dir.rglob("*")) if path.is_file()})
    print("Selected before comparison:", winner, flush=True)
    for name in ("legs_qdq_knn", "hybrid_switch", winner):
        print(name, [(r["horizon_ms"], round(r["rmse"], 6)) for r in aggregate
            if r["method"] == name and r["group"] == "acc" and r["region"] == "full"
            and r["horizon_ms"] in (6, 24, 54)], flush=True)


if __name__ == "__main__":
    main()
