#!/usr/bin/env python3
"""Export the already-fitted H0 innovation lookup, without fitting a new model.

Only compact numerical weights are published. Original recordings remain local.
This script has no SDK imports and never opens a robot connection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "evaluation/hardware_shadow/commissioning/walk_h0_refinement_20260925"
DEFAULT_OUTPUT = ROOT / "assets/g1_hardware_mpc_predictor"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    source, output = Path(source), Path(output)
    absolute_path = source / "local/models/legs_current_imu_knn8.npz"
    delta_path = source / "local/models/legs_current_imu_delta_knn8.npz"
    decay_path = source / "innovation/model.npz"
    absolute, delta, innovation = [dict(np.load(p, allow_pickle=False))
                                  for p in (absolute_path, delta_path, decay_path)]
    for key in ("mean", "std", "feature_scale", "train_z", "train_trial", "train_anchor"):
        np.testing.assert_array_equal(absolute[key], delta[key])
    if int(absolute["k"]) != 8 or int(delta["k"]) != 8:
        raise ValueError("expected the frozen k8 models")
    future = absolute["train_y"].reshape(-1, 9, 12)
    current = (absolute["train_y"] - delta["train_y"]).reshape(-1, 9, 12)
    np.testing.assert_allclose(current, np.repeat(current[:, :1], 9, axis=1),
                               rtol=0, atol=1e-11)
    output.mkdir(parents=True, exist_ok=True)
    bank_path = output / "bank.npz"
    if bank_path.exists() or (output / "manifest.json").exists():
        raise FileExistsError("refusing to overwrite a frozen published bank")
    np.savez_compressed(bank_path, mean=absolute["mean"], std=absolute["std"],
                        feature_scale=absolute["feature_scale"],
                        train_z=absolute["train_z"], future=future,
                        current=current[:, 0], decay=innovation["decay"],
                        train_trial=absolute["train_trial"],
                        train_anchor=absolute["train_anchor"])
    manifest = {
        "schema": "g1_hardware_mpc_frozen_predictor_v1",
        "bank_sha256": sha(bank_path), "bank_bytes": bank_path.stat().st_size,
        "fitting_performed": False, "source_study": "walk_h0_refinement_20260925",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p)
                          for p in (absolute_path, delta_path, decay_path)},
        "source_program_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "tools/g1_commissioning/walk_h0_study/refine_local.py",
            ROOT / "tools/g1_commissioning/walk_h0_study/refine_innovation.py",
            ROOT / "tools/g1_commissioning/walk_h0_study/methods.py")},
        "development_runs": [1, 2, 3, 4, 5, 7, 8],
        "excluded_run": 6, "comparison_runs_not_in_bank": [9, 10, 11, 12],
        "old_five_runs_excluded": True,
        "rows": len(future), "feature_count": 33, "neighbors": 8,
        "features": "qf[0:12], dqf[0:12], H0 filtered acc/omega/alpha; standardize then IMU scale0.5",
        "target_order": ["acc3", "omega3", "alpha3", "diagnostic_rpy3"],
        "horizons_ms": list(range(6, 55, 6)), "grid_ms": 2,
        "filter_hz": 15,
        "target_contract": "acc/alpha left interval samples h-6,h-4,h-2ms; omega/rpy endpoint h",
        "filter_caveat": "Future FILTERED estimates, not raw instantaneous acceleration. No fixed time advance or delay compensation.",
        "alpha_contract": "15Hz causal lowpass of backward difference of 15Hz filtered omega on 2ms grid",
        "innovation": "weighted historical future + decay[h,group]*(current - weighted historical current)",
        "rotation_adapter": "Measured node0 R; world-left SO3 exponential integration of trapezoidal predicted omega. RPY forecasts diagnostic only.",
        "frame": "fixed H0 yaw reference per run; torso IMU origin; gravity [0,0,-9.81]",
        "status": "Offline-fitted candidate; no claim of hardware MPC or raw-impact validation.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()
    print(json.dumps(export(a.source, a.output), indent=2))
