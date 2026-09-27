#!/usr/bin/env python3
"""Audit withdrawn five-trial numbers; never include them in the new H0 model.

This intentionally does not import the original predictor implementation.  It
reconstructs the recorded 24-feature, eight-neighbour lookup independently from
the old extracted data and compares every output with the archived predictions.
The old inputs are read-only and remain scientifically withdrawn because of the
operator-reported tether loading.  Numerical reproducibility is not validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import lfilter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE = ROOT / "evaluation/hardware_shadow/commissioning/walk_predictor_study_20260918/benchmark"
DEFAULT_INPUT = ROOT / "evaluation/hardware_shadow/commissioning/walk_dataset_audit_20260917"
DEFAULT_OUTPUT = ROOT / "evaluation/hardware_shadow/commissioning/walk_h0_predictor_study_20260925/legacy_verification.json"
HORIZONS = (6, 24, 54)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_features_and_labels(path: Path):
    """Recreate fixed archival preprocessing and future labels, not H0 work."""
    with np.load(path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    grid = np.arange(-.5, 21.00001, .002)

    def past(times, values):
        indices = np.searchsorted(times, grid, side="right") - 1
        if np.any(indices < 0):
            raise ValueError("archive lacks causal prehistory")
        return values[indices]

    def filtered(values):
        gain = 1 - np.exp(-2 * np.pi * 15 * .002)
        return lfilter([gain], [1, -(1-gain)], values-values[0], axis=0)+values[0]

    acc = filtered(past(data["imu_t"], data["world_acc"]))
    omega = filtered(past(data["imu_t"], data["world_omega"]))
    alpha = filtered(np.vstack([np.zeros((1, 3)), np.diff(omega, axis=0)/.002]))
    rpy = past(data["imu_t"], np.unwrap(data["imu"][:, 6:9], axis=0))
    y = np.column_stack([acc, omega, alpha, rpy])
    q = filtered(past(data["low_t"], data["q"])[:, :12])
    dq = filtered(past(data["low_t"], data["dq"])[:, :12])
    anchors = np.where((grid >= 7.) & (grid < 14.7))[0][::3]
    labels = []
    for horizon in HORIZONS:
        end = anchors+horizon//2
        node = y[end].copy()
        interval = np.mean([y[end-j] for j in (0, 1, 2)], axis=0)
        node[:, :3] = interval[:, :3]
        node[:, 6:9] = interval[:, 6:9]
        labels.append(node)
    return np.column_stack([q, dq])[anchors], np.stack(labels, axis=1), grid[anchors]


def verify(archive: Path, inputs: Path):
    protocol = json.loads((archive / "protocol.json").read_text())
    selected = next(row for row in json.loads((archive / "selection.json").read_text())
                    if row["method"] == "knn_legs_qdq")
    if protocol["train"] != [1, 2, 3] or protocol["comparison"] != [4, 5]:
        raise ValueError("unexpected archival episode partition")
    if selected["chosen"]["param"] != 8 or selected["features"] != 24:
        raise ValueError("unexpected archival selected model")
    file_checks = {
        name: {"recorded_sha256": recorded, "actual_sha256": sha256(inputs/name),
               "matches": sha256(inputs/name) == recorded}
        for name, recorded in protocol["files"].items()
    }
    if not all(item["matches"] for item in file_checks.values()):
        raise ValueError("old extracted inputs no longer match recorded hashes")
    raw_checks = []
    for index in range(1, 6):
        recorded = json.loads((inputs/f"trial{index:02d}_audit.json").read_text())
        raw_path = ROOT / recorded["source"] / "raw.jsonl"
        actual = sha256(raw_path)
        raw_checks.append({"trial": index, "path": str(raw_path.relative_to(ROOT)),
                           "recorded_sha256": recorded["sha256"], "actual_sha256": actual,
                           "matches": actual == recorded["sha256"]})
    if not all(item["matches"] for item in raw_checks):
        raise ValueError("old raw logs no longer match extraction audit hashes")
    episodes = [independent_features_and_labels(inputs/f"trial{i:02d}.npz")
                for i in range(1, 6)]
    train_x = np.vstack([item[0] for item in episodes[:3]])
    train_y = np.vstack([item[1].reshape(len(item[0]), -1) for item in episodes[:3]])
    center, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.)
    tree = cKDTree((train_x-center)/scale)
    rows, errors, all_method_errors = [], [], {}
    metrics = json.loads((archive / "metrics.json").read_text())
    with np.load(archive / "steady_predictions.npz", allow_pickle=False) as predictions:
        for index in (4, 5):
            x, truth, t = episodes[index-1]
            distance, neighbors = tree.query((x-center)/scale, k=8)
            weights = 1/np.maximum(distance, 1e-3)
            weights /= weights.sum(axis=1, keepdims=True)
            rebuilt = np.sum(train_y[neighbors]*weights[:, :, None], axis=1).reshape(truth.shape)
            stored = predictions[f"trial{index}_knn_legs_qdq"]
            stored_truth = predictions[f"trial{index}_truth"]
            np.testing.assert_allclose(truth, stored_truth, atol=1e-12, rtol=0)
            np.testing.assert_allclose(rebuilt, stored, atol=1e-12, rtol=0)
            np.testing.assert_allclose(t, predictions[f"trial{index}_time"], atol=1e-12, rtol=0)
            error = stored[:, :, :3]-stored_truth[:, :, :3]
            errors.append(error)
            rmse = np.sqrt(np.mean(error**2, axis=(0, 2)))
            for horizon, value in zip(HORIZONS, rmse):
                metric = next(row for row in metrics if row["method"] == "knn_legs_qdq"
                              and row["trial"] == index and row["horizon_ms"] == horizon
                              and row["fit_scope"] == "steady" and row["region"] == "steady")
                np.testing.assert_allclose(value, metric["rmse"]["acc"], atol=1e-12, rtol=0)
            rows.append({"trial": index, "anchor_count": len(x),
                         "first_last_anchor_s": [float(t[0]), float(t[-1])],
                         "acc_rmse_m_s2": rmse.tolist(),
                         "rebuilt_truth_max_abs_difference": float(np.max(np.abs(truth-stored_truth))),
                         "rebuilt_prediction_max_abs_difference": float(np.max(np.abs(rebuilt-stored)))})
            for key in predictions.files:
                prefix = f"trial{index}_"
                if key.startswith(prefix) and key[len(prefix):] not in ("truth", "time"):
                    method = key[len(prefix):]
                    all_method_errors.setdefault(method, []).append(
                        predictions[key][:, :, :3]-stored_truth[:, :, :3])
    pooled = np.sqrt(np.mean(np.concatenate(errors)**2, axis=(0, 2)))
    current_script = ROOT / "tools/g1_commissioning/compare_walk_predictors.py"
    current_dependency = ROOT / "tools/g1_commissioning/analyze_walk_dataset.py"
    original_dependency = subprocess.check_output(
        ["git", "show", "3625712:tools/g1_commissioning/analyze_walk_dataset.py"], cwd=ROOT)
    original_hash = hashlib.sha256(original_dependency).hexdigest()
    if original_hash != protocol["dependency_sha256"]:
        raise ValueError("recorded legacy dependency not reproducible from Git")
    return {
        "purpose": "Numerical provenance audit of withdrawn tethered trials; NOT new H0 model/evidence",
        "source_archive": str(archive.relative_to(ROOT)),
        "archival_artifact_sha256": {name: sha256(archive/name) for name in
                                     ("protocol.json", "selection.json", "metrics.json", "steady_predictions.npz")},
        "verifier_sha256": sha256(Path(__file__)),
        "input_hash_checks": file_checks,
        "raw_log_hash_checks": raw_checks,
        "original_predictor_sha256_matches": sha256(current_script) == protocol["script_sha256"],
        "current_dependency_sha256_matches": sha256(current_dependency) == protocol["dependency_sha256"],
        "original_dependency_git_commit": "3625712",
        "original_dependency_sha256_matches": original_hash == protocol["dependency_sha256"],
        "dependency_difference": "Only zeros_after audit selection gained yaw_rate == 0; no preprocessing/labels/model change",
        "train_trials": [1, 2, 3], "evaluation_trials": [4, 5],
        "hyperparameter_provenance": selected,
        "model": "24 standardized leg q/dq features, Euclidean eight-neighbour inverse-distance lookup",
        "distance_weight": "w_i=(1/max(distance_i,0.001))/sum_j(1/max(distance_j,0.001))",
        "train_samples": len(train_x), "standardization_fit": "training trials only",
        "frame": "legacy IMU navigation world, NOT new fixed H0 frame",
        "preprocessing": "past-asof 2ms grid, causal15Hz first order filter; acceleration subtracts gravity",
        "alpha": "backward difference of filtered world omega, then causal15Hz filter",
        "horizons_ms": HORIZONS,
        "target_intervals_ms": [[0, 6], [18, 24], [48, 54]],
        "target_discrete_offsets_ms": [[2, 4, 6], [20, 22, 24], [50, 52, 54]],
        "scope": "steady anchors7 <= t <14.7s sampled every6ms; not startup/stop or closed-loop MPC",
        "rmse_formula": "sqrt(sum((prediction-truth)^2)/(number_of_anchors*3_axes)), pool complete trials4/5",
        "units": "m/s^2, three scalar components pooled; not vector norm RMSE",
        "per_trial": rows,
        "pooled_acc_rmse_m_s2": pooled.tolist(),
        "published_rounded": [round(float(value), 3) for value in pooled],
        "all_archived_methods_acc_rmse_m_s2": {
            key: np.sqrt(np.mean(np.concatenate(values)**2, axis=(0, 2))).tolist()
            for key, values in all_method_errors.items()},
        "conclusion": "Stored numbers and independent re-fit agree to1e-12; old trial physical validity remains withdrawn",
        "limitations": ["Raw-to-NPZ extraction is hash-traced, not re-executed in this verifier",
                        "Filtered targets, not raw impact truth", "Same-day exploratory trials, not fresh blind validation",
                        "Reproducibility does not establish model generalisation or hardware closed-loop benefit"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify(args.archive.resolve(), args.input_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"verified": True, "rmse": result["pooled_acc_rmse_m_s2"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
