#!/usr/bin/env python3
"""Replay NEW H0 runs offline; preserve the numerical path from data to metrics.

No robot access. Never mixes old tethered trials into training or evaluation.
Run with BLAS thread environment variables equal to 1 for predictable cost.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

import methods as m

DEVELOPMENT = (1, 2, 3, 4, 5, 7, 8)
HELDOUT = (9, 10, 11, 12)
REGIONS = {"full": (5., 18.), "startup": (5., 7.), "steady": (7., 15.), "stopping": (15., 18.)}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(pred, truth, raw_truth, raw_hold, d, idx, method, trial, peak_threshold):
    rows = []
    for name, (a, b) in REGIONS.items():
        mask = (d["t"][idx] >= a-1e-9) & (d["t"][idx] < b-1e-9)
        for j, ms in enumerate(m.HORIZONS_MS):
            diff = pred[mask, j]-truth[mask, j]
            # Only the scalar-angle diagnostics wrap; the selected physical vectors do not.
            diff[:, 9:12] = (diff[:, 9:12]+np.pi) % (2*np.pi)-np.pi
            eraw = pred[mask, j, :3]-raw_truth[mask, j]
            raw_zoh_error = raw_hold[mask]-raw_truth[mask, j]
            predicted_rot = Rotation.from_euler("xyz", pred[mask, j, 9:12])
            actual_rot = Rotation.from_euler("xyz", truth[mask, j, 9:12])
            angles = (predicted_rot.inv()*actual_rot).magnitude()
            base = dict(method=method, trial=trial, region=name, horizon_ms=int(ms), samples=int(mask.sum()))
            for group, sl in m.GROUPS.items():
                e = diff[:, sl]
                rows.append(dict(**base, group=group, rmse=float(np.sqrt(np.mean(e*e))),
                    sse=float(np.sum(e*e)), scalar_count=int(e.size),
                    p95_absolute_component_error=float(np.percentile(np.abs(e), 95)),
                    axis_rmse=np.sqrt(np.mean(e*e, axis=0)).tolist()))
            for group, e in (("raw_acc_sensitivity", eraw), ("raw_acc_hold_baseline", raw_zoh_error),
                             ("orientation_geodesic", angles[:, None])):
                rows.append(dict(**base, group=group, rmse=float(np.sqrt(np.mean(e*e))),
                    sse=float(np.sum(e*e)), scalar_count=int(e.size),
                    p95_absolute_component_error=float(np.percentile(np.abs(e), 95)),
                    axis_rmse=np.sqrt(np.mean(e*e, axis=0)).tolist()))
    for j, ms in enumerate(m.HORIZONS_MS):
        mask = np.linalg.norm(truth[:, j, :3], axis=1) > peak_threshold
        if mask.any():
            e = pred[mask, j, :3]-truth[mask, j, :3]
            rows.append(dict(method=method, trial=trial, region="large_filtered_acc", horizon_ms=int(ms),
                samples=int(mask.sum()), group="acc", rmse=float(np.sqrt(np.mean(e*e))),
                sse=float(np.sum(e*e)), scalar_count=int(e.size),
                p95_absolute_component_error=float(np.percentile(np.abs(e), 95)),
                axis_rmse=np.sqrt(np.mean(e*e, axis=0)).tolist()))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = args.output_dir/"models"
    model_dir.mkdir()
    pred_dir = args.output_dir/"predictions"
    pred_dir.mkdir()
    oof_dir = args.output_dir/"development_oof"
    oof_dir.mkdir()
    paths = {i: args.data_dir/f"trial{i:02d}_prepared.npz" for i in DEVELOPMENT+HELDOUT}
    ds = {i: dict(np.load(path)) for i, path in paths.items()}
    idx = {i: m.anchors(d) for i, d in ds.items()}
    if not all(np.array_equal(idx[i], idx[1]) for i in ds):
        raise ValueError("clock-average comparison requires an identical time grid")
    ys = {i: m.targets(d, idx[i]) for i, d in ds.items()}
    zs = {i: m.hold(d, idx[i]) for i, d in ds.items()}
    raw = {i: m.targets(d, idx[i], "acc_raw") for i, d in ds.items()}
    source_paths = (Path(__file__), Path(m.__file__))
    protocol = dict(
        development_trials=DEVELOPMENT, heldout_trials=HELDOUT, excluded_trial=6,
        exclusion="Trial 6 has no normal session_end/capture_drained; retained in audit, not primary model fitting/scoring.",
        old_tethered_five="Excluded from ALL fitting/scoring. Old numbers are historical only.",
        selection="Leave one whole development episode out; select parameters using ONLY these folds, then fit all development episodes.",
        heldout_caveat="Chronological heldout prediction comparison, not a claim of an unseen physical safety or control trial.",
        anchors="Every 6ms from 5.006s in [5,18); H0 reference must already be available; maximum 54ms future strictly before 18s; arm-release tail excluded.",
        target_definition="H0 fixed per run: acc/alpha LEFT average samples h-6,h-4,h-2 ms (simulation pre-step convention); omega/orientation endpoint h. Filtered15Hz, no future smoothing.",
        legacy_target_difference="Old five-run study used RIGHT interval samples h-4,h-2,h; new numbers must not be compared numerically as the same benchmark.",
        orientation_deployment="Research RPY is diagnostic; deployed MPC requires rotation matrices with 10 nodes including current plus9 future nodes, not 9 Euler rows.",
        raw_sensitivity="The SAME filtered-trained predictions additionally scored against unfiltered H0 acc; raw-current hold is separately reported. Not a raw-trained model comparison.",
        filter_caveat="Predicting a causal filtered signal is easier and does NOT remove the filter phase delay or prove raw impact prediction.",
        frames="H0 fixed before walking; no per-cycle dynamic yaw re-zeroing.",
        horizons_ms=m.HORIZONS_MS.tolist(), feature_history_ms=[int(l*2) for l in m.LAGS],
        scoring="sqrt(sum squared component errors / number of scalar components), not norm-RMSE; selection equal acc/omega/alpha MSE ratio against filtered ZOH, all nine horizons.",
        parameters=m.PARAMETERS, methods=m.METHODS, ridge_definition="standardize using train mean/std; solve (X'X/n + lambda I)B=X'(Y-meanY)/n",
        knn_definition="Euclidean distance in train-standardized feature space; k neighbors; w=1/max(distance,0.001), normalized; weighted future labels.",
        script_sha256={str(path): sha(path) for path in source_paths},
        input_sha256={str(path): sha(path) for path in paths.values()},
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        no_robot=True, deployable_model=False)
    dump(args.output_dir/"protocol.json", protocol)
    oof, selected = {}, []
    fitted = {}
    learned = [name for name in m.METHODS if name.endswith(("_ridge", "_knn"))]
    for name in learned:
        began = time.monotonic()
        kind = "ridge" if name.endswith("ridge") else "knn"
        # This is development-only. Heldout feature values are not used in scaling/selection.
        xs = {i: m.features(ds[i], idx[i], name) for i in DEVELOPMENT}
        scores, candidate_predictions = [], {}
        for param in m.PARAMETERS[kind]:
            folds, predictions = [], {}
            for val in DEVELOPMENT:
                train = [i for i in DEVELOPMENT if i != val]
                model = m.fit_model(np.vstack([xs[i] for i in train]),
                    np.vstack([ys[i].reshape(len(idx[i]), -1) for i in train]), kind, param)
                predictions[val] = m.predict(model, xs[val]).reshape(ys[val].shape)
                folds.append(dict(trial=val, score=m.score(predictions[val], ys[val], zs[val])))
            scores.append(dict(parameter=param, folds=folds, mean_score=float(np.mean([f["score"] for f in folds]))))
            candidate_predictions[param] = predictions
        best = min(scores, key=lambda x: x["mean_score"])
        selected.append(dict(method=name, feature_count=int(xs[1].shape[1]), chosen=best, candidates=scores))
        oof[name] = candidate_predictions[best["parameter"]]
        fitted[name] = m.fit_model(np.vstack([xs[i] for i in DEVELOPMENT]),
            np.vstack([ys[i].reshape(len(idx[i]), -1) for i in DEVELOPMENT]), kind, best["parameter"])
        model_arrays = m.serializable_model(fitted[name])
        model_arrays.update(train_trial=np.concatenate([np.full(len(idx[i]), i) for i in DEVELOPMENT]),
                            train_anchor=np.concatenate([idx[i] for i in DEVELOPMENT]))
        np.savez_compressed(model_dir/f"{name}.npz", **model_arrays)
        print(name, "parameter", best["parameter"], "CV", round(best["mean_score"], 4),
              "seconds", round(time.monotonic()-began, 2), flush=True)
    # Baseline CV is saved as well; phase baseline never reads heldout events in training.
    for name in ("zoh", "clock_average", "phase_average"):
        oof[name] = {}
        for val in DEVELOPMENT:
            train = [i for i in DEVELOPMENT if i != val]
            if name == "zoh":
                pred = zs[val]
            elif name == "clock_average":
                pred = np.mean([ys[i] for i in train], axis=0)
            else:
                template, _ = m.phase_template(ds, train)
                pred, _ = m.phase_predict(ds[val], idx[val], template)
            oof[name][val] = pred
        folds = [dict(trial=i, score=m.score(oof[name][i], ys[i], zs[i])) for i in DEVELOPMENT]
        selected.append(dict(method=name, chosen=dict(parameter=None, folds=folds,
            mean_score=float(np.mean([f["score"] for f in folds]))), candidates=[]))
    crossovers = []
    for threshold in (0, *m.HORIZONS_MS.tolist()):
        folds = [dict(trial=i, score=m.score(m.switched(oof["imu_legs_history_ridge"][i],
            oof["legs_qdq_knn"][i], threshold), ys[i], zs[i])) for i in DEVELOPMENT]
        crossovers.append(dict(parameter=threshold, folds=folds,
            mean_score=float(np.mean([f["score"] for f in folds]))))
    best_switch = min(crossovers, key=lambda x: x["mean_score"])
    oof["hybrid_switch"] = {i: m.switched(oof["imu_legs_history_ridge"][i], oof["legs_qdq_knn"][i],
                            best_switch["parameter"]) for i in DEVELOPMENT}
    selected.append(dict(method="hybrid_switch", chosen=best_switch, candidates=crossovers))
    # Freeze selection before constructing any heldout prediction metrics.
    dump(args.output_dir/"selection.json", selected)
    winner = min(selected, key=lambda row: row["chosen"]["mean_score"])["method"]
    dump(args.output_dir/"selected_candidate.json", dict(method=winner,
        rule="lowest mean leave-one-development-episode-out score; heldout not consulted",
        switch_threshold_ms=best_switch["parameter"]))
    for i in DEVELOPMENT:
        np.savez_compressed(oof_dir/f"trial{i:02d}.npz", time=ds[i]["t"][idx[i]], anchors=idx[i],
            truth=ys[i], **{name: oof[name][i] for name in m.METHODS})
    template, cycle_count = m.phase_template(ds, DEVELOPMENT)
    clock = np.mean([ys[i] for i in DEVELOPMENT], axis=0)
    np.savez_compressed(model_dir/"baselines.npz", phase_template=template, phase_cycles=cycle_count,
                        clock_average=clock, anchors=idx[1], development_trials=DEVELOPMENT)
    threshold = float(np.percentile(np.concatenate([
        np.linalg.norm(ys[i][:, :, :3], axis=2).ravel() for i in DEVELOPMENT]), 90))
    protocol["large_acc_threshold_m_s2"] = threshold
    protocol["selected_candidate_before_heldout"] = winner
    dump(args.output_dir/"protocol.json", protocol)
    rows = []
    for i in HELDOUT:
        predictions = {name: m.predict(fitted[name], m.features(ds[i], idx[i], name)).reshape(ys[i].shape)
                       for name in learned}
        phase, phase_usable = m.phase_predict(ds[i], idx[i], template)
        predictions.update(zoh=zs[i], clock_average=clock, phase_average=phase)
        predictions["hybrid_switch"] = m.switched(predictions["imu_legs_history_ridge"],
            predictions["legs_qdq_knn"], best_switch["parameter"])
        raw_hold = ds[i]["acc_raw"][idx[i]]
        np.savez_compressed(pred_dir/f"trial{i:02d}.npz", anchors=idx[i], time=ds[i]["t"][idx[i]],
            truth=ys[i], raw_acc_truth=raw[i], raw_acc_hold=raw_hold, phase_usable=phase_usable,
            horizons_ms=m.HORIZONS_MS, **predictions)
        for name, pred in predictions.items():
            rows += metrics(pred, ys[i], raw[i], raw_hold, ds[i], idx[i], name, i, threshold)
        print("Heldout", i, "scored", len(idx[i]), "anchors", flush=True)
    dump(args.output_dir/"metrics_per_trial.json", rows)
    with (args.output_dir/"metrics_per_trial.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    for name in m.METHODS:
        for region in (*REGIONS, "large_filtered_acc"):
            for h in m.HORIZONS_MS:
                for group in (*m.GROUPS, "raw_acc_sensitivity", "raw_acc_hold_baseline", "orientation_geodesic"):
                    rr = [r for r in rows if r["method"] == name and r["region"] == region
                          and r["horizon_ms"] == h and r["group"] == group]
                    if not rr:
                        continue
                    sse, count = sum(r["sse"] for r in rr), sum(r["scalar_count"] for r in rr)
                    aggregate.append(dict(method=name, region=region, horizon_ms=int(h), group=group,
                        rmse=float(np.sqrt(sse/count)), sse=sse, scalar_count=count,
                        min_trial_rmse=min(r["rmse"] for r in rr), max_trial_rmse=max(r["rmse"] for r in rr)))
    dump(args.output_dir/"metrics_aggregate.json", aggregate)
    with (args.output_dir/"metrics_aggregate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    # Inventory gives every persisted calculation artifact an independent checksum.
    inventory = {str(path.relative_to(args.output_dir)): sha(path)
                 for path in sorted(args.output_dir.rglob("*")) if path.is_file()}
    dump(args.output_dir/"artifact_sha256.json", inventory)
    print("Selected by development CV:", winner, "hybrid threshold", best_switch["parameter"], flush=True)


if __name__ == "__main__":
    main()
