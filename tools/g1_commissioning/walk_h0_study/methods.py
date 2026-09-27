"""Auditable, offline-only predictors for the September H0 walking study.

All features use data at or before the forecast anchor. Future values appear
only in supervised labels, training phase templates, and scoring. No SDK imports.
See benchmark.py for episode splits, fitted artifacts, and the calculation ledger.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

DT = .002
HORIZONS_MS = np.arange(6, 55, 6)
LAGS = (0, 6, 15, 30, 60)  # 0, 12, 30, 60, 120 milliseconds of PAST history.
GROUPS = {"acc": slice(0, 3), "omega": slice(3, 6), "alpha": slice(6, 9),
          "rpy_diagnostic": slice(9, 12)}
METHODS = {
    "zoh": "Hold the current filtered IMU measurement at every future horizon.",
    "clock_average": "Average development-trial future labels at the same task elapsed time.",
    "phase_average": "128-bin mean complete gait-landmark cycle; causal past-period estimate; ZOH until two landmarks.",
    "hips_clock_ridge": "Left/right hip-pitch angles plus task time, time squared/cubed; ridge regression.",
    "hips_qdq_knn": "Left/right hip-pitch angles and velocities; standardized inverse-distance nearest-neighbor lookup.",
    "knees_qdq_knn": "Left/right knee angles and velocities; same nearest-neighbor lookup (old study used knees, not hips).",
    "legs_qdq_knn": "All 12 leg joint angles and velocities (24 features); nearest-neighbor future-label lookup.",
    "imu_history_ridge": "Recent H0 acc, omega, alpha, orientation at five causal lags; ridge regression.",
    "imu_legs_history_ridge": "The same recent IMU history plus 12 leg angles/velocities at those lags; ridge regression.",
    "hybrid_switch": "Near-horizon IMU+leg ridge, farther-horizon full-leg lookup; crossover chosen only by development CV.",
}
PARAMETERS = {"ridge": (.01, .1, 1.), "knn": (8, 24, 64)}


def anchors(d):
    # Every future interval/node must remain strictly inside [5,18).
    first = max(5.006, float(d.get("reference_available_s", 5.006)))
    return np.flatnonzero((d["t"] >= first-1e-9) &
                          (d["t"] + HORIZONS_MS[-1]/1000 < 18.-1e-9))[::3]


def targets(d, idx, key="y"):
    """acc/alpha: LEFT samples h-6,h-4,h-2 ms; omega/orientation node h.

    Matches simulation's pre-step interval convention, NOT old-study right
    samples h-4,h-2,h. At h=6 the first interval uses t,t+2,t+4 ms.
    """
    x = d[key]
    out = []
    for h in HORIZONS_MS // 2:
        if key == "acc_raw":
            node = np.mean([x[idx+h-j] for j in (3, 2, 1)], axis=0)
        else:
            node = x[idx+h].copy()
            interval = np.mean([x[idx+h-j] for j in (3, 2, 1)], axis=0)
            node[:, :3], node[:, 6:9] = interval[:, :3], interval[:, 6:9]
        out.append(node)
    return np.stack(out, axis=1)


def hold(d, idx):
    return np.repeat(d["y"][idx, None, :], len(HORIZONS_MS), axis=1)


def history(x, idx):
    if np.min(idx) < max(LAGS):
        raise ValueError("insufficient causal history")
    return np.column_stack([x[idx-lag] for lag in LAGS])


def features(d, idx, method):
    q, dq = d["qf"][idx], d["dqf"][idx]
    if method == "hips_clock_ridge":
        t = (d["t"][idx]-5.)/10.
        return np.column_stack((q[:, (0, 6)], t, t*t, t*t*t))
    if method in ("hips_qdq_knn", "knees_qdq_knn"):
        joint = (0, 6) if method.startswith("hips") else (3, 9)
        return np.column_stack((q[:, joint], dq[:, joint]))
    legs = np.column_stack((d["qf"], d["dqf"]))
    if method == "legs_qdq_knn":
        return legs[idx]
    if method == "imu_history_ridge":
        return history(d["y"], idx)
    if method == "imu_legs_history_ridge":
        return np.column_stack((history(d["y"], idx), history(legs, idx)))
    raise ValueError(method)


def fit_model(x, y, kind, parameter):
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std > 1e-6, std, 1.)
    z = (x-mean)/std
    model = dict(kind=kind, mean=mean, std=std, parameter=parameter)
    if kind == "ridge":
        ym = y.mean(axis=0)
        model.update(ymean=ym, coef=np.linalg.solve(
            z.T@z/len(z)+float(parameter)*np.eye(z.shape[1]), z.T@(y-ym)/len(z)))
    elif kind == "knn":
        model.update(train_z=z, train_y=y, tree=cKDTree(z))
    else:
        raise ValueError(kind)
    return model


def predict(model, x):
    z = (x-model["mean"])/model["std"]
    if model["kind"] == "ridge":
        return z@model["coef"]+model["ymean"]
    distance, index = model["tree"].query(z, k=int(model["parameter"]), workers=1)
    weight = 1./np.maximum(distance, .001)
    weight /= weight.sum(axis=1, keepdims=True)
    return np.sum(model["train_y"][index]*weight[..., None], axis=1)


def serializable_model(model):
    return {k: v for k, v in model.items() if k != "tree"}


def score(pred, truth, baseline):
    """Dimensionless, equal weight to acc/omega/alpha and all nine horizons."""
    return float(np.mean([
        np.mean((pred[:, j, sl]-truth[:, j, sl])**2) /
        max(float(np.mean((baseline[:, j, sl]-truth[:, j, sl])**2)), 1e-10)
        for j in range(len(HORIZONS_MS)) for sl in tuple(GROUPS.values())[:3]]))


def positive_landmarks(d):
    # Left-minus-right hip pitch upward zero crossings with 0.03-rad hysteresis.
    # This is a kinematic landmark, never a claimed physical foot-contact label.
    armed, last, found = False, -np.inf, []
    for t, x in zip(d["t"], d["hip"]):
        if t < 5 or t >= 18:
            continue
        if x <= -.03:
            armed = True
        if armed and x >= 0 and t-last >= .3:
            found.append(t)
            last, armed = t, False
    return np.asarray(found)


def causal_phase(d, idx):
    events = positive_landmarks(d)
    last = np.searchsorted(events, d["t"][idx], side="right")-1
    available = last >= 1
    phase, period = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    for i in np.flatnonzero(available):
        j = last[i]
        period[i] = np.median(np.diff(events[max(0, j-3):j+1]))
        phase[i] = (d["t"][idx[i]]-events[j])/period[i]
    return phase, period, available


def phase_template(ds, train):
    bins, cycles = np.arange(128)/128., []
    for i in train:
        d = ds[i]
        events = positive_landmarks(d)
        for a, b in zip(events[:-1], events[1:]):
            if a >= 7 and b < 14.8:
                time = a+bins*(b-a)
                cycles.append(np.column_stack([np.interp(time, d["t"], d["y"][:, k])
                                               for k in range(12)]))
    if not cycles:
        raise ValueError("no complete development steady cycles")
    return np.mean(cycles, axis=0), len(cycles)


def periodic_sample(template, phase):
    xp = np.arange(len(template)+1)/len(template)
    yp = np.vstack((template, template[0]))
    return np.column_stack([np.interp(np.asarray(phase) % 1., xp, yp[:, k])
                            for k in range(12)])


def phase_predict(d, idx, template):
    phase, period, usable = causal_phase(d, idx)
    result = hold(d, idx)  # Explicit warmup fallback; no looking at next crossing.
    for j, ms in enumerate(HORIZONS_MS):
        node = periodic_sample(template, phase[usable]+ms/1000/period[usable])
        mean = np.mean([periodic_sample(template, phase[usable]+(ms/1000-k*DT)/period[usable])
                        for k in (3, 2, 1)], axis=0)
        node[:, :3], node[:, 6:9] = mean[:, :3], mean[:, 6:9]
        result[usable, j] = node
    return result, usable


def switched(near, far, threshold_ms):
    return np.where((HORIZONS_MS <= threshold_ms)[None, :, None], near, far)
