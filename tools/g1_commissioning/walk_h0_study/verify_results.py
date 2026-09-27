#!/usr/bin/env python3
"""Rebuild saved forecasts, verify metric arithmetic, and show actual neighbors.

This does not train/select a new model. Trial 6 is a quarantined sensitivity
check with the already frozen development models; it never changes the ranking.
Outputs live OUTSIDE the original benchmark checksum inventory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import benchmark as b
import methods as m


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models(root):
    result = {}
    for path in sorted((root/"models").glob("*.npz")):
        if path.stem != "baselines":
            result[path.stem] = dict(np.load(path))
    return result


def reconstruct_model(model, x):
    # Independent arithmetic from saved scaler/coefficients/neighbor rows.
    standardized = (x-model["mean"])/model["std"]
    if str(model["kind"]) == "ridge":
        return standardized @ model["coef"] + model["ymean"]
    distance, rows = cKDTree(model["train_z"]).query(standardized, k=int(model["parameter"]), workers=1)
    unnormalized = 1. / np.maximum(distance, .001)
    normalized = unnormalized / np.sum(unnormalized, axis=1)[:, None]
    return np.sum(model["train_y"][rows] * normalized[:, :, None], axis=1)


def reconstruct_all(d, idx, models, baseline, threshold):
    result = {name: reconstruct_model(model, m.features(d, idx, name)).reshape(len(idx), 9, 12)
              for name, model in models.items()}
    result["zoh"] = np.tile(d["y"][idx, None, :], (1, 9, 1))
    result["clock_average"] = baseline["clock_average"]
    result["phase_average"], _ = m.phase_predict(d, idx, baseline["phase_template"])
    result["hybrid_switch"] = np.where((m.HORIZONS_MS <= threshold)[None, :, None],
        result["imu_legs_history_ridge"], result["legs_qdq_knn"])
    return result


def independent_truth(d, idx):
    result = np.stack([d["y"][idx+ms//2].copy() for ms in m.HORIZONS_MS], axis=1)
    raw = []
    for j, ms in enumerate(m.HORIZONS_MS):
        rows = idx[:, None] + ms//2 + np.array([-3, -2, -1])[None, :]
        means = d["y"][rows].mean(axis=1)
        result[:, j, :3], result[:, j, 6:9] = means[:, :3], means[:, 6:9]
        raw.append(d["acc_raw"][rows].mean(axis=1))
    return result, np.stack(raw, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.study_dir/"benchmark"
    protocol = json.loads((root/"protocol.json").read_text())
    inventory = json.loads((root/"artifact_sha256.json").read_text())
    for path, expected in inventory.items():
        if digest(root/path) != expected:
            raise ValueError("benchmark artifact changed: "+path)
    for field in ("script_sha256", "input_sha256"):
        for path, expected in protocol[field].items():
            if digest(Path(path)) != expected:
                raise ValueError(field+" mismatch: "+path)
    models = load_models(root)
    baseline = dict(np.load(root/"models/baselines.npz"))
    threshold = json.loads((root/"selected_candidate.json").read_text())["switch_threshold_ms"]
    allrows = json.loads((root/"metrics_per_trial.json").read_text())
    aggregate = json.loads((root/"metrics_aggregate.json").read_text())
    max_prediction_error, max_metric_error, max_label_error = 0., 0., 0.
    total_prediction_values = total_metrics = 0
    for trial in b.HELDOUT:
        d = dict(np.load(args.study_dir/f"data/trial{trial:02d}_prepared.npz"))
        saved = dict(np.load(root/f"predictions/trial{trial:02d}.npz"))
        idx = saved["anchors"]
        truth, raw_truth = independent_truth(d, idx)
        err = max(float(np.max(np.abs(truth-saved["truth"]))),
                  float(np.max(np.abs(raw_truth-saved["raw_acc_truth"]))))
        max_label_error = max(max_label_error, err)
        np.testing.assert_allclose(truth, saved["truth"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(raw_truth, saved["raw_acc_truth"], rtol=0, atol=1e-12)
        forecasts = reconstruct_all(d, idx, models, baseline, threshold)
        for name, value in forecasts.items():
            np.testing.assert_allclose(value, saved[name], rtol=0, atol=1e-12)
            max_prediction_error = max(max_prediction_error, float(np.max(np.abs(value-saved[name]))))
            total_prediction_values += value.size
        for row in (r for r in allrows if r["trial"] == trial):
            j = row["horizon_ms"]//6-1
            if row["region"] == "large_filtered_acc":
                mask = np.linalg.norm(truth[:, j, :3], axis=1) > protocol["large_acc_threshold_m_s2"]
            else:
                a, z = b.REGIONS[row["region"]]
                mask = (saved["time"] >= a-1e-9) & (saved["time"] < z-1e-9)
            pred, true = saved[row["method"]][mask, j], truth[mask, j]
            if row["group"] == "orientation_geodesic":
                error = (Rotation.from_euler("xyz", pred[:, 9:12]).inv()*
                         Rotation.from_euler("xyz", true[:, 9:12])).magnitude()[:, None]
            elif row["group"] == "raw_acc_sensitivity":
                error = pred[:, :3]-raw_truth[mask, j]
            elif row["group"] == "raw_acc_hold_baseline":
                error = d["acc_raw"][idx][mask]-raw_truth[mask, j]
            else:
                error = pred[:, m.GROUPS[row["group"]]]-true[:, m.GROUPS[row["group"]]]
                if row["group"] == "rpy_diagnostic":
                    error = (error+np.pi) % (2*np.pi)-np.pi
            rmse = float(np.sqrt(np.sum(error**2)/error.size))
            np.testing.assert_allclose(float(np.sum(error**2)), row["sse"], rtol=1e-12, atol=1e-12)
            if error.size != row["scalar_count"]:
                raise ValueError("recorded metric sample count differs from reconstructed errors")
            np.testing.assert_allclose(rmse, row["rmse"], rtol=0, atol=1e-12)
            max_metric_error = max(max_metric_error, abs(rmse-row["rmse"]))
            total_metrics += 1
        print("Verified exact saved calculations trial", trial, flush=True)
    for row in aggregate:
        matching = [r for r in allrows if all(r[k] == row[k] for k in ("method", "region", "horizon_ms", "group"))]
        expected = np.sqrt(sum(r["sse"] for r in matching)/sum(r["scalar_count"] for r in matching))
        np.testing.assert_allclose(expected, row["rmse"], rtol=0, atol=1e-12)
    # One concrete kNN calculation that a human can audit with a calculator.
    d = dict(np.load(args.study_dir/"data/trial09_prepared.npz"))
    saved = dict(np.load(root/"predictions/trial09.npz"))
    ai = int(np.argmin(np.abs(saved["time"]-10.)))
    model = models["legs_qdq_knn"]
    feature = m.features(d, saved["anchors"][ai:ai+1], "legs_qdq_knn")[0]
    z = (feature-model["mean"])/model["std"]
    distance, rows = cKDTree(model["train_z"]).query(z, k=int(model["parameter"]))
    unnormalized = 1./np.maximum(distance, .001)
    weight = unnormalized/unnormalized.sum()
    # Flat training labels are horizon-major, 12 channels per horizon.
    labels = model["train_y"][rows].reshape(len(rows), 9, 12)[:, 3, :3]
    weighted_sum = np.sum(labels*weight[:, None], axis=0)
    np.testing.assert_allclose(weighted_sum, saved["legs_qdq_knn"][ai, 3, :3], rtol=0, atol=1e-12)
    neighbors = [dict(row=int(r), source_trial=int(model["train_trial"][r]),
        source_grid_index=int(model["train_anchor"][r]),
        source_task_time_s=float(d["t"][model["train_anchor"][r]]),
        standardized_distance=float(di), unnormalized_inverse_distance=float(u), normalized_weight=float(w),
        future_24ms_acc_label=label.tolist())
        for r, di, u, w, label in zip(rows, distance, unnormalized, weight, labels)]
    pooled_row = next(r for r in aggregate if r["method"] == "legs_qdq_knn" and r["region"] == "full"
                      and r["group"] == "acc" and r["horizon_ms"] == 24)
    example = dict(trial=9, task_anchor_s=float(saved["time"][ai]), horizon_ms=24,
        left_interval_relative_ms=[18,20,22], units="m/s^2, H0, causal15Hz filtered",
        feature_order="q[0:12],dq[0:12] causal filtered; motor order left6 then right6",
        feature=feature.tolist(), standard_scaler_mean=model["mean"].tolist(),
        standard_scaler_std=model["std"].tolist(), neighbors=neighbors,
        sum_unnormalized_weights=float(unnormalized.sum()), prediction=weighted_sum.tolist(),
        actual_label=saved["truth"][ai,3,:3].tolist(),
        squared_errors=((weighted_sum-saved["truth"][ai,3,:3])**2).tolist(),
        pooled_full_window_heldout_rmse_calculation=pooled_row,
        formula="pooled RMSE=sqrt(sse/scalar_count), NOT the error of this one example alone")
    b.dump(args.study_dir/"calculation_example.json", example)
    # Orientation is diagnostic, excluded from primary selection. Check development
    # folds separately rather than choosing its predictor from heldout outcomes.
    orientation_cv = []
    for name in m.METHODS:
        ratios = []
        for trial in b.DEVELOPMENT:
            saved_oof = dict(np.load(root/f"development_oof/trial{trial:02d}.npz"))
            true = Rotation.from_euler("xyz", saved_oof["truth"][:, :, 9:12].reshape(-1, 3))
            pred = Rotation.from_euler("xyz", saved_oof[name][:, :, 9:12].reshape(-1, 3))
            hold = Rotation.from_euler("xyz", saved_oof["zoh"][:, :, 9:12].reshape(-1, 3))
            pe = (pred.inv()*true).magnitude().reshape(-1, 9)
            he = (hold.inv()*true).magnitude().reshape(-1, 9)
            ratios.append(dict(trial=trial, mse_ratio_by_horizon=(np.mean(pe*pe,axis=0)/np.mean(he*he,axis=0)).tolist()))
        orientation_cv.append(dict(method=name, folds=ratios,
            mean_mse_ratio=float(np.mean([r["mse_ratio_by_horizon"] for r in ratios]))))
    b.dump(args.study_dir/"orientation_development_cv.json", dict(
        note="Separate diagnostic computed from saved development OOF only; not a retroactive change of primary selection.",
        ranked=sorted(orientation_cv, key=lambda x: x["mean_mse_ratio"])))
    # Quarantined trial6 supplement, NOT added to ranking/training/primary aggregate.
    d6_path = args.study_dir/"data/trial06_prepared.npz"
    d6 = dict(np.load(d6_path))
    i6 = m.anchors(d6)
    truth6, raw6 = independent_truth(d6, i6)
    pred6 = reconstruct_all(d6, i6, models, baseline, threshold)
    supplement_rows = []
    for name, pred in pred6.items():
        supplement_rows += b.metrics(pred, truth6, raw6, d6["acc_raw"][i6], d6, i6, name, 6,
            protocol["large_acc_threshold_m_s2"])
    np.savez_compressed(args.study_dir/"supplemental_trial06.npz", time=d6["t"][i6], anchors=i6,
                        truth=truth6, raw_acc_truth=raw6, **pred6)
    b.dump(args.study_dir/"supplemental_trial06.json", dict(
        exclusion_unchanged="No normal session_end/capture_drained. Supplemental only; no fit, no parameter adjustment, no primary ranking change.",
        input_sha256=digest(d6_path), metrics=supplement_rows))
    b.dump(args.study_dir/"verification.json", dict(
        verified=True, script_sha256=digest(Path(__file__)), benchmark_inventory_sha256=digest(root/"artifact_sha256.json"),
        verified_benchmark_files=len(inventory), verified_prediction_scalars=total_prediction_values,
        verified_per_trial_metrics=total_metrics, verified_aggregate_metrics=len(aggregate),
        max_prediction_difference=max_prediction_error, max_label_difference=max_label_error,
        max_rmse_difference=max_metric_error,
        exact_methods="Ridge and kNN numerical arithmetic rebuilt from saved model arrays, labels independently indexed; phase reuses documented causal method; no fitting occurred.",
        supplementary_outputs={path.name: digest(path) for path in (
            args.study_dir/"calculation_example.json", args.study_dir/"orientation_development_cv.json",
            args.study_dir/"supplemental_trial06.json", args.study_dir/"supplemental_trial06.npz")}))
    print("All numerical checks passed; supplemental trial6 saved separately.", flush=True)


if __name__ == "__main__":
    main()
