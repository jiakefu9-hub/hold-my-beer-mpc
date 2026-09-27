#!/usr/bin/env python3
"""Bounded offline nonlinear refinement; previous comparison runs are NOT blind.

Six preregistered random-Fourier + linear-history residual regressions. Each
scaler is fitted within its development fold. All predictions/model arrays are
saved; no SDK, robot connection, command publishing, or baseline artifact edits.
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

import benchmark as b
import methods as m

SEED = 20260925
RFF_COUNT = 384
CONFIGS = [dict(name=f"rff_leg_length{length}_ridge{ridge:g}", lengthscale=float(length), ridge=ridge)
           for length in (2, 4, 8) for ridge in (.001, .01)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def standardizer(x):
    center, scale = np.mean(x, axis=0), np.std(x, axis=0)
    return center, np.where(scale > 1e-6, scale, 1.)


def raw_features(d, idx):
    return m.features(d, idx, "imu_legs_history_ridge"), m.features(d, idx, "legs_qdq_knn")


def map_features(model, linear, legs):
    linear = (linear-model["linear_mean"])/model["linear_std"]
    legs = (legs-model["leg_mean"])/model["leg_std"]
    nonlinear = np.cos(legs@model["fourier_weights"]+model["fourier_bias"])
    return np.column_stack((linear, nonlinear))


def fit(linear, legs, truth, current, config):
    model = dict(lengthscale=config["lengthscale"], ridge=config["ridge"])
    model["linear_mean"], model["linear_std"] = standardizer(linear)
    model["leg_mean"], model["leg_std"] = standardizer(legs)
    rng = np.random.default_rng(SEED)
    model["fourier_weights"] = rng.normal(size=(legs.shape[1], RFF_COUNT))/config["lengthscale"]
    model["fourier_bias"] = rng.uniform(0., 2*np.pi, size=RFF_COUNT)
    phi = map_features(model, linear, legs)
    model["map_mean"], model["map_std"] = standardizer(phi)
    z = (phi-model["map_mean"])/model["map_std"]
    # Learn a correction to hold-current, rather than replacing measured state.
    delta = truth-current
    model["delta_mean"] = delta.mean(axis=0)
    model["coef"] = np.linalg.solve(z.T@z/len(z)+config["ridge"]*np.eye(z.shape[1]),
        z.T@(delta-model["delta_mean"])/len(z))
    return model


def predict(model, linear, legs, current):
    phi = map_features(model, linear, legs)
    z = (phi-model["map_mean"])/model["map_std"]
    return current+model["delta_mean"]+z@model["coef"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"models").mkdir()
    (args.output_dir/"predictions").mkdir()
    (args.output_dir/"development_oof").mkdir()
    trial_paths = {i: args.data_dir/f"trial{i:02d}_prepared.npz" for i in b.DEVELOPMENT+b.HELDOUT}
    ds = {i: dict(np.load(path)) for i, path in trial_paths.items()}
    idx = {i: m.anchors(d) for i, d in ds.items()}
    truths = {i: m.targets(d, idx[i]) for i, d in ds.items()}
    holds = {i: m.hold(d, idx[i]) for i, d in ds.items()}
    raw = {i: m.targets(d, idx[i], "acc_raw") for i, d in ds.items()}
    features = {i: raw_features(d, idx[i]) for i, d in ds.items()}
    script_paths = [Path(__file__), Path(m.__file__), Path(b.__file__)]
    protocol = dict(purpose="Exploratory nonlinear refinement on already-inspected comparison episodes; NOT a new blind validation.",
        development=b.DEVELOPMENT, comparison=b.HELDOUT, excluded=6,
        exact_splits_and_targets="Same frozen new12 benchmark: left h-6,h-4,h-2ms interval means; nodes omega/orientation; full [5.006,17.948) causal anchors.",
        selection="Mean seven-whole-episode leave-one-out score: equal acc/omega/alpha MSE ratio to filtered ZOH over9horizons; comparison not used in parameter selection.",
        configs=CONFIGS, random_seed=SEED, random_features=RFF_COUNT,
        input_definition="Linear: IMU12+leg24 history at0,12,30,60,120ms (180 features); nonlinear: current standardized leg angles+velocities24.",
        map_definition="cos(z_legs @ fixed Gaussian W/lengthscale + fixed uniform bias); concatenate linear history; TRAIN-only recenter/rescale; ridge to future-minus-current labels.",
        output_definition="Current filtered measurement + fitted future increment; 9 horizons x12channels; Euler orientation diagnostic only.",
        raw_sensitivity="Models trained on filtered targets; also score against raw H0 acceleration and raw-current baseline. This is not raw-trained impact prediction.",
        hashes={str(path): sha(path) for path in list(trial_paths.values())+script_paths},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        no_hardware=True, no_baseline_modification=True)
    # Frozen list written BEFORE any candidate scores exist.
    b.dump(args.output_dir/"protocol.json", protocol)
    choices, all_oof = [], {}
    for config in CONFIGS:
        began = time.monotonic()
        fold_scores, predictions = [], {}
        for val in b.DEVELOPMENT:
            train = [i for i in b.DEVELOPMENT if i != val]
            model = fit(np.vstack([features[i][0] for i in train]),
                np.vstack([features[i][1] for i in train]),
                np.vstack([truths[i].reshape(len(idx[i]), -1) for i in train]),
                np.vstack([holds[i].reshape(len(idx[i]), -1) for i in train]), config)
            predictions[val] = predict(model, *features[val], holds[val].reshape(len(idx[val]), -1)).reshape(truths[val].shape)
            fold_scores.append(dict(trial=val, score=m.score(predictions[val], truths[val], holds[val])))
        row = dict(**config, folds=fold_scores, mean_score=float(np.mean([r["score"] for r in fold_scores])),
                   elapsed_s=float(time.monotonic()-began))
        choices.append(row)
        all_oof[config["name"]] = predictions
        print(config["name"], "CV", round(row["mean_score"],5), "seconds", round(row["elapsed_s"],2), flush=True)
        b.dump(args.output_dir/"selection.json", choices)
    best = min(choices,key=lambda x:x["mean_score"])
    b.dump(args.output_dir/"selected_candidate.json", best)
    for trial in b.DEVELOPMENT:
        np.savez_compressed(args.output_dir/f"development_oof/trial{trial:02d}.npz", time=ds[trial]["t"][idx[trial]],
            anchors=idx[trial], truth=truths[trial], **{name: pp[trial] for name,pp in all_oof.items()})
    peak_threshold=float(np.percentile(np.concatenate([
        np.linalg.norm(truths[i][:,:,:3],axis=2).ravel() for i in b.DEVELOPMENT]),90))
    rows, reconstructed_max = [], 0.
    comparison = {i:{} for i in b.HELDOUT}
    # Save every candidate, not only the one selected by CV, so failed attempts remain visible.
    for config in CONFIGS:
        model = fit(np.vstack([features[i][0] for i in b.DEVELOPMENT]),
            np.vstack([features[i][1] for i in b.DEVELOPMENT]),
            np.vstack([truths[i].reshape(len(idx[i]),-1) for i in b.DEVELOPMENT]),
            np.vstack([holds[i].reshape(len(idx[i]),-1) for i in b.DEVELOPMENT]),config)
        model_path=args.output_dir/f"models/{config['name']}.npz"
        np.savez_compressed(model_path,**model,train_trials=b.DEVELOPMENT)
        reloaded=dict(np.load(model_path))
        for i in b.HELDOUT:
            prediction=predict(model,*features[i],holds[i].reshape(len(idx[i]),-1)).reshape(truths[i].shape)
            reconstructed=predict(reloaded,*features[i],holds[i].reshape(len(idx[i]),-1)).reshape(truths[i].shape)
            reconstructed_max=max(reconstructed_max,float(np.max(np.abs(prediction-reconstructed))))
            np.testing.assert_array_equal(prediction,reconstructed)
            comparison[i][config["name"]]=prediction
            rows+=b.metrics(prediction,truths[i],raw[i],ds[i]["acc_raw"][idx[i]],ds[i],idx[i],
                config["name"],i,peak_threshold)
    for i in b.HELDOUT:
        np.savez_compressed(args.output_dir/f"predictions/trial{i:02d}.npz",time=ds[i]["t"][idx[i]],anchors=idx[i],
            truth=truths[i],raw_acc_truth=raw[i],raw_acc_hold=ds[i]["acc_raw"][idx[i]],**comparison[i])
    b.dump(args.output_dir/"metrics_per_trial.json",rows)
    aggregate=[]
    for name in (c["name"] for c in CONFIGS):
        for region in (*b.REGIONS,"large_filtered_acc"):
            for h in m.HORIZONS_MS:
                for group in (*m.GROUPS,"raw_acc_sensitivity","raw_acc_hold_baseline","orientation_geodesic"):
                    rr=[r for r in rows if r['method']==name and r['region']==region and r['horizon_ms']==h and r['group']==group]
                    if rr:
                        sse,count=sum(r['sse'] for r in rr),sum(r['scalar_count'] for r in rr)
                        aggregate.append(dict(method=name,region=region,horizon_ms=int(h),group=group,
                            rmse=float(np.sqrt(sse/count)),sse=sse,scalar_count=count,
                            min_trial_rmse=min(r['rmse'] for r in rr),max_trial_rmse=max(r['rmse'] for r in rr)))
    b.dump(args.output_dir/"metrics_aggregate.json",aggregate)
    with (args.output_dir/"metrics_aggregate.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=tuple(aggregate[0]))
        writer.writeheader();writer.writerows(aggregate)
    b.dump(args.output_dir/"verification.json",dict(saved_model_prediction_max_difference=reconstructed_max,
        reconstructed_all6_candidates=True,comparison_trials=b.HELDOUT,protocol_frozen_before_cv=True))
    b.dump(args.output_dir/"artifact_sha256.json",{str(path.relative_to(args.output_dir)):sha(path)
        for path in sorted(args.output_dir.rglob('*')) if path.is_file()})
    print("Selected nonlinear candidate:",best['name'],"CV",best['mean_score'],flush=True)


if __name__ == "__main__":
    main()
