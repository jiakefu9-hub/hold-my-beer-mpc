#!/usr/bin/env python3
"""Exploratory offline forecasting ablations for the five recorded G1 runs.

No SDK/network/output. All episode split/scaling/parameter selection is explicit.
This is method screening, not a deployable predictor or a fresh blind evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.spatial import cKDTree

import analyze_walk_dataset as audit

HORIZONS = (6, 24, 54)
LAGS = (0, 6, 15, 30, 60)  # 2ms grid: 0, 12, 30, 60, 120 ms
RIDGE_GRID = (.001, .01, .1, 1.)
K_GRID = (8, 24, 64)
GROUPS = {"acc": slice(0, 3), "omega": slice(3, 6),
          "alpha": slice(6, 9), "rpy_diagnostic": slice(9, 12)}


def targets(d, anchors):
    """Nodes for omega/RPY; following-interval averages for acc/alpha."""
    result = []
    for ms in HORIZONS:
        h = ms // 2
        y = d["y"][anchors+h].copy()
        interval = np.mean([d["y"][anchors+h-j] for j in (0, 1, 2)], axis=0)
        y[:, :3], y[:, 6:9] = interval[:, :3], interval[:, 6:9]
        result.append(y)
    return np.stack(result, axis=1)


def history(x, anchors):
    if anchors.min() < max(LAGS):
        raise ValueError("insufficient past data")
    return np.column_stack([x[anchors-lag] for lag in LAGS])


def phase_info(d, anchors):
    ev = audit.leg_events(d["t"], d["hip"])
    e = np.array([d["t"][i] for i, sign in ev if sign == 1])
    last = np.searchsorted(e, d["t"][anchors], side="right")-1
    available = last >= 1
    # Missing phase is not repaired with future events or a heldout period.
    period = np.full(len(anchors), np.nan)
    phase = np.full(len(anchors), np.nan)
    for i in np.flatnonzero(available):
        j = last[i]
        period[i] = np.median(np.diff(e[max(0, j-3):j+1]))
        phase[i] = (d["t"][anchors[i]]-e[j])/period[i]
    return phase, period, available


def features(d, anchors, name):
    q = d["qf"][anchors]
    dq = d["dqf"][anchors]
    leg = np.column_stack([d["qf"], d["dqf"]])
    if name == "knees_q":
        return q[:, [3, 9]]
    if name == "knees_qdq":
        return np.column_stack([q[:, [3, 9]], dq[:, [3, 9]]])
    if name == "knees_clock":
        t = (d["t"][anchors]-5)/10
        return np.column_stack([q[:, [3, 9]], t, t*t, t*t*t])
    if name == "legs_qdq":
        return leg[anchors]
    if name == "legs_history":
        return history(leg, anchors)
    if name == "imu_history":
        return history(d["y"], anchors)
    if name == "imu_legs_history":
        return np.column_stack([history(d["y"], anchors), history(leg, anchors)])
    if name == "imu_phase":
        phase, _, available = phase_info(d, anchors)
        if not np.all(available):
            raise ValueError("phase not available; this method is steady-only")
        phi = 2*np.pi*phase
        harmonics = np.column_stack([f(n*phi) for n in range(1, 7)
                                     for f in (np.sin, np.cos)])
        return np.column_stack([history(d["y"], anchors), harmonics])
    raise ValueError(name)


def fit_scaler(x):
    mean, std = x.mean(axis=0), x.std(axis=0)
    return mean, np.where(std > 1e-6, std, 1.)


def fit_model(x, y, kind, param):
    mean, std = fit_scaler(x)
    z = (x-mean)/std
    ymean = y.mean(axis=0)
    if kind == "ridge":
        # Division by n keeps regularisation independent of episode length.
        coef = np.linalg.solve(z.T@z/len(z)+float(param)*np.eye(z.shape[1]),
                               z.T@(y-ymean)/len(z))
        return kind, mean, std, ymean, coef
    return kind, mean, std, cKDTree(z), y, int(param)


def predict(model, x):
    kind, mean, std, *state = model
    z = (x-mean)/std
    if kind == "ridge":
        ymean, coef = state
        return z@coef+ymean
    tree, y, k = state
    distance, idx = tree.query(z, k=k)
    weights = 1/np.maximum(distance, 1e-3)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(y[idx]*weights[..., None], axis=1)


def selection_score(pred, truth, zoh):
    p = pred.reshape(truth.shape)
    # Equal weight to three physical vector groups and three horizons, not yaw.
    return float(np.mean([np.mean((p[:, j, sl]-truth[:, j, sl])**2) /
                          max(np.mean((zoh[:, j, sl]-truth[:, j, sl])**2), 1e-10)
                          for j in range(3) for sl in list(GROUPS.values())[:3]]))


def phase_template(ds, train):
    bins = np.arange(128)/128
    cycles = []
    for k in train:
        d = ds[k]
        ev = audit.leg_events(d["t"], d["hip"])
        times = np.array([d["t"][i] for i, s in ev if s == 1])
        for a, b in zip(times[:-1], times[1:]):
            if a >= 7 and b < 14.8:
                cycles.append(audit.interp_columns(a+bins*(b-a), d["t"], d["y"]))
    return np.mean(cycles, axis=0)


def phase_predict(d, anchors, template, delta=False):
    phase, period, available = phase_info(d, anchors)
    if not available.all():
        raise ValueError("missing causal phase")
    current = audit.phase_sample(template, phase)
    out = []
    for h in HORIZONS:
        node = audit.phase_sample(template, phase+h/1000/period)
        interval = np.mean([audit.phase_sample(template, phase+(h/1000-j*.002)/period)
                            for j in (0, 1, 2)], axis=0)
        node[:, :3], node[:, 6:9] = interval[:, :3], interval[:, 6:9]
        out.append(node+d["y"][anchors]-current if delta else node)
    return np.stack(out, axis=1)


def metric_rows(pred, truth, trial, method, anchors, d, fit_scope):
    regions = {"steady": (7., 14.7), "startup": (5., 7.), "stopping": (15., 17.9)}
    rows = []
    for region, (a, b) in regions.items():
        mask = (d["t"][anchors] >= a) & (d["t"][anchors] < b)
        if not mask.any():
            continue
        for j, h in enumerate(HORIZONS):
            er = pred[mask, j]-truth[mask, j]
            row = dict(trial=trial, method=method, fit_scope=fit_scope,
                       region=region, horizon_ms=h, samples=int(mask.sum()),
                       rmse={name:float(np.sqrt(np.mean(er[:, sl]**2)))
                             for name, sl in GROUPS.items()},
                       per_axis_rmse={name:np.sqrt(np.mean(er[:, sl]**2,axis=0)).tolist()
                                      for name,sl in GROUPS.items()},
                       abs_error_p95={name:float(np.percentile(np.abs(er[:, sl]), 95))
                             for name, sl in GROUPS.items()})
            rows.append(row)
    return rows


def peak_rows(pred, truth, trial, method, threshold):
    rows=[]
    for j,h in enumerate(HORIZONS):
        # Future labels only select difficult scoring samples, never features.
        mask=np.linalg.norm(truth[:,j,:3],axis=1)>threshold
        if mask.any():
            error=pred[mask,j]-truth[mask,j]
            rows.append(dict(trial=trial,method=method,horizon_ms=h,
                samples=int(mask.sum()),acc_rmse_m_s2=float(np.sqrt(np.mean(error[:,:3]**2))),
                acc_abs_error_p95=float(np.percentile(np.abs(error[:,:3]),95))))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path(
        "evaluation/hardware_shadow/commissioning/walk_dataset_audit_20260917"))
    p.add_argument("--output-dir", type=Path, default=Path(
        "evaluation/hardware_shadow/commissioning/walk_predictor_study_20260918/benchmark"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ds = [audit.prepare(dict(np.load(args.input_dir/f"trial{i:02d}.npz"))) for i in range(1, 6)]
    for d in ds:
        d["qf"] = audit.lowpass(d["q"][:, :12], 15)
        d["dqf"] = audit.lowpass(d["dq"][:, :12], 15)
    protocol = dict(purpose="Exploratory comparison; trials4/5 have already been inspected, not new blind validation.",
        train=[1,2,3], comparison=[4,5], hyperparameter_selection="leave one complete episode out within trials1-3",
        sample_period_s=.006, grid_period_s=.002, feature_history_ms=[2*l for l in LAGS],
        horizons_ms=HORIZONS, ridge_grid=RIDGE_GRID, knn_grid=K_GRID,
        selection_score="equal-weight MSE/ZOH_MSE over acc, omega, alpha and 3 horizons",
        target_definition="acc/alpha 6ms interval ending at horizon; omega/RPY at node; RPY diagnostic only",
        preprocessing="causal15Hz; past-asof, no future interpolation; scaler fitted separately within each train fold",
        raw_source="unchanged NPZ derived from immutable JSONL; input SHA256 recorded",
        files={f"trial{i:02d}.npz":hashlib.sha256((args.input_dir/f"trial{i:02d}.npz").read_bytes()).hexdigest()
               for i in range(1,6)}, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256=hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest(),
        versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        whole_scope_definition="5--17.9s anchors, includes start/stop but not 0--21s full arm task",
        percentile_definition="pooled scalar component absolute errors, not vector-error norms",
        no_hardware=True, no_model_export=True)
    audit.write_json(args.output_dir/"protocol.json", protocol)
    rows, selection, predictions, peaks = [], [], {}, []
    steady = np.where((ds[0]["t"]>=7)&(ds[0]["t"]<14.7))[0][::3]
    whole = np.where((ds[0]["t"]>=5)&(ds[0]["t"]<17.9))[0][::3]
    threshold=float(np.percentile(np.concatenate([
        np.linalg.norm(targets(ds[j],steady)[:,:, :3],axis=2).ravel() for j in (0,1,2)]),90))
    protocol["peak_scoring"] = dict(training_percentile=90,acc_norm_threshold_m_s2=threshold,
        meaning="Large filtered future interval acceleration, not labelled foot contact; scoring only.")
    audit.write_json(args.output_dir/"protocol.json",protocol)
    baseline_template = phase_template(ds, [0,1,2])
    for k in (3,4):
        d = ds[k]; truth = targets(d, steady)
        clock = np.mean([targets(ds[j], steady) for j in (0,1,2)], axis=0)
        baselines = dict(zoh=np.repeat(d["y"][steady,None,:],3,axis=1), clock=clock,
                         phase=phase_predict(d,steady,baseline_template),
                         phase_delta=phase_predict(d,steady,baseline_template,True))
        for name,pred in baselines.items():
            rows += metric_rows(pred,truth,k+1,name,steady,d,"steady")
            peaks += peak_rows(pred,truth,k+1,name,threshold)
            predictions[f"trial{k+1}_{name}"] = pred
        predictions[f"trial{k+1}_truth"] = truth
        predictions[f"trial{k+1}_time"] = d["t"][steady]
        all_truth = targets(d,whole)
        rows += metric_rows(np.repeat(d["y"][whole,None,:],3,axis=1),
                            all_truth,k+1,"zoh",whole,d,"whole")
    experiments = [("ridge",name,"steady") for name in (
        "knees_q","knees_qdq","knees_clock","legs_qdq","legs_history",
        "imu_history","imu_legs_history","imu_phase")]
    experiments += [("knn",name,"steady") for name in ("knees_q","knees_qdq","legs_qdq","imu_legs_history")]
    experiments += [("ridge", "imu_legs_history", "whole"),
                    ("knn", "imu_legs_history", "whole")]
    for kind,name,scope in experiments:
        began=time.monotonic(); anchors = steady if scope=="steady" else whole
        xs=[features(d,anchors,name) for d in ds]
        ys=[targets(d,anchors) for d in ds]
        params=RIDGE_GRID if kind=="ridge" else K_GRID
        scores=[]
        for param in params:
            folds=[]
            for val in (0,1,2):
                train=[j for j in (0,1,2) if j!=val]
                x=np.vstack([xs[j] for j in train])
                y=np.vstack([ys[j].reshape(len(anchors),-1) for j in train])
                model=fit_model(x,y,kind,param)
                pred=predict(model,xs[val])
                zoh=np.repeat(ds[val]["y"][anchors,None,:],3,axis=1)
                folds.append(selection_score(pred,ys[val],zoh))
            scores.append(dict(param=param,fold_scores=folds,mean=float(np.mean(folds))))
        best=min(scores,key=lambda r:r["mean"])
        model=fit_model(np.vstack(xs[:3]),np.vstack([y.reshape(len(anchors),-1) for y in ys[:3]]),kind,best["param"])
        label=f"{kind}_{name}"+("_whole" if scope=="whole" else "")
        selection.append(dict(method=label,features=xs[0].shape[1],chosen=best,all_candidates=scores))
        for k in (3,4):
            pred=predict(model,xs[k]).reshape(ys[k].shape)
            rows+=metric_rows(pred,ys[k],k+1,label,anchors,ds[k],scope)
            if scope=="steady":
                predictions[f"trial{k+1}_{label}"]=pred
                peaks += peak_rows(pred,ys[k],k+1,label,threshold)
        print(label,"chosen",best["param"],"cv",round(best["mean"],3),
              "elapsed",round(time.monotonic()-began,2),flush=True)
    audit.write_json(args.output_dir/"selection.json",selection)
    audit.write_json(args.output_dir/"metrics.json",rows)
    audit.write_json(args.output_dir/"high_acceleration_metrics.json",peaks)
    np.savez_compressed(args.output_dir/"steady_predictions.npz",**predictions)
    plot_metrics(args.output_dir,rows,predictions)
    print("Wrote",args.output_dir,flush=True)


def plot_metrics(out,rows,predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods=list(dict.fromkeys(r["method"] for r in rows if r["fit_scope"]=="steady"))
    fig,axs=plt.subplots(1,3,figsize=(15,9),sharey=True)
    for ax,h in zip(axs,HORIZONS):
        vals=[]
        for name in methods:
            rr=[r for r in rows if r["region"]=="steady" and r["fit_scope"]=="steady" and r["horizon_ms"]==h and r["method"]==name]
            vals.append(np.sqrt(np.mean([r["rmse"]["acc"]**2 for r in rr])))
        ax.barh(methods,vals);ax.set_title(f"Interval ending +{h}ms");ax.set_xlabel("World acc RMSE (m/s2)");ax.grid(axis="x",alpha=.25)
    axs[0].invert_yaxis();fig.suptitle("Exploratory trials 4/5; model choices from complete-episode CV on 1/2/3")
    fig.tight_layout();fig.savefig(out/"method_comparison.png",dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(13,9),sharex=True)
    t=predictions["trial4_time"]+.024
    for ax,idx,label in zip(axs,[2,3,6],["az m/s2 (interval)","omega x rad/s (node)","alpha x rad/s2 (interval)"]):
        for name in ("truth","zoh","phase","ridge_imu_history","knn_legs_qdq","knn_imu_legs_history"):
            ax.plot(t,predictions[f"trial4_{name}"][:,1,idx],label=name,lw=1)
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axs[0].legend(ncol=3,fontsize=8);axs[-1].set_xlim(9,11.2);axs[-1].set_xlabel("Prediction target time (s), +24ms")
    fig.tight_layout();fig.savefig(out/"forecast_comparison.png",dpi=160);plt.close(fig)


if __name__=="__main__":
    main()
