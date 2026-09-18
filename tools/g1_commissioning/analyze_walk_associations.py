#!/usr/bin/env python3
"""Offline leg-state / future-IMU association diagnostics; never robot I/O.

Selection uses whole episodes 1--3 only. Episodes 4--5 are the previously
inspected diagnostic holdout, not fresh independent validation episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_walk_dataset import DT, prepare, write_json

JOINTS = [f"{side}_{joint}" for side in ("L", "R")
          for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
TARGETS = [f"{kind}_{axis}" for kind in ("acc", "omega", "alpha", "rpy") for axis in "xyz"]
HORIZONS_MS = [0, 6, 12, 24, 54, 100]


def features(d):
    """Only the last received joint samples at t; no centred filtering."""
    q, v = d["q"][:, :12], d["dq"][:, :12]
    names = ["q_" + n for n in JOINTS] + ["dq_" + n for n in JOINTS]
    extras = []
    for label, x in (("q", q), ("dq", v)):
        for name, a, b in (("hip_pitch", 0, 6), ("knee", 3, 9)):
            for op, sign in (("sum", 1), ("diff", -1)):
                extras.append(x[:, a] + sign*x[:, b])
                names.append(f"{label}_{name}_{op}")
    return np.column_stack([q, v, *extras]), names


def correlation(x, y):
    x, y = x-x.mean(0), y-y.mean(0)
    denominator = np.sqrt(np.sum(x*x, axis=0)[:, None]*np.sum(y*y, axis=0)[None, :])
    return np.divide(x.T @ y, denominator, out=np.zeros_like(denominator), where=denominator > 1e-15)


def rank_train(rho):
    """Strongest minimum same-sign magnitude over the three train episodes."""
    train = rho[:3]
    stable_sign = (train.min(0) > 0) | (train.max(0) < 0)
    return np.where(stable_sign, np.abs(train).min(0), 0)


def top_records(rho, names, top=4):
    score = rank_train(rho)
    rows = []
    for target in range(len(TARGETS)):
        for f in np.argsort(score[:, target])[::-1][:top]:
            values = rho[:, f, target]
            rows.append(dict(feature=names[f], target=TARGETS[target],
                             train_rank_score=float(score[f, target]),
                             episode_rho=values.tolist(),
                             heldout_sign_agrees=bool(np.all(np.sign(values[3:]) == np.sign(values[:3].mean())))))
    return rows


def direction_lookup(trials, feats, names, horizon_ms=24):
    """Tiny explanatory lookup, not a selected control predictor.

    Fixed left-knee angle, fixed 12 quantile bins and velocity sign; trained
    only on episodes 1--3. No target-dependent choice of joint/bin count.
    """
    anchors = np.flatnonzero((trials[0]["t"] >= 7) & (trials[0]["t"] < 14.7))[::3]
    h = round(horizon_ms/1000/DT)
    a, v = names.index("q_L_knee"), names.index("dq_L_knee")
    train_q = np.concatenate([f[anchors, a] for f in feats[:3]])
    edges = np.quantile(train_q, np.linspace(0, 1, 13))[1:-1]
    train_id = np.searchsorted(edges, train_q, side="right")
    train_sign = np.concatenate([f[anchors, v] >= 0 for f in feats[:3]]).astype(int)
    train_y = np.concatenate([d["acc"][anchors+h, 2] for d in trials[:3]])
    average = np.array([train_y[train_id == j].mean() for j in range(12)])
    directional = np.array([[train_y[(train_id == j) & (train_sign == k)].mean()
                             if np.any((train_id == j) & (train_sign == k)) else average[j]
                             for k in range(2)] for j in range(12)])
    rows = []
    for i in (3, 4):
        ids = np.searchsorted(edges, feats[i][anchors, a], side="right")
        signs = (feats[i][anchors, v] >= 0).astype(int)
        truth = trials[i]["acc"][anchors+h, 2]
        rmse = lambda pred: float(np.sqrt(np.mean((pred-truth)**2)))
        rows.append(dict(trial=i+1, left_knee_angle_rmse=rmse(average[ids]),
                         left_knee_angle_and_velocity_sign_rmse=rmse(directional[ids, signs]),
                         current_acc_z_zoh_rmse=rmse(trials[i]["acc"][anchors, 2])))
    return dict(target="acc_z", horizon_ms=horizon_ms, units="m/s^2", train_trials=[1, 2, 3],
                bins=12, edges_rad=edges.tolist(), rows=rows,
                caveat="Illustrates angle's direction ambiguity only; ignores current IMU and other joints.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("evaluation/hardware_shadow/commissioning/walk_dataset_audit_20260917"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/hardware_shadow/commissioning/walk_predictor_study_20260918/associations"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    trials = [prepare(dict(np.load(args.input_dir/f"trial{i:02d}.npz"))) for i in range(1, 6)]
    feats = [features(d)[0] for d in trials]
    names = features(trials[0])[1]
    anchors = np.flatnonzero((trials[0]["t"] >= 7) & (trials[0]["t"] < 14.7))[::3]
    correlations = {}
    ranked = {}
    for ms in HORIZONS_MS:
        h = round(ms/1000/DT)
        rho = np.array([correlation(f[anchors], d["y"][anchors+h]) for f, d in zip(feats, trials)])
        correlations[str(ms)] = rho.tolist()
        ranked[str(ms)] = top_records(rho, names)
    write_json(out/"associations.json", dict(features=names, targets=TARGETS, horizons_ms=HORIZONS_MS,
               per_trial_rho=correlations, train_ranked_pairs=ranked,
               samples_per_episode=len(anchors), association_window_s=[7, 14.7],
               sampling_ms=6, source_imu_filter_hz=15,
               input="as-of received q/dq, without extra smoothing", target="same causal filtered world IMU as prior audit",
               selection="minimum absolute rho across train1--3, requiring matching signs; test4--5 never choose pairs",
               caveats=["Correlation is not incremental prediction accuracy or physical causation.",
                        "Both channels share periodic leg motion; autocorrelated samples are not independent trials.",
                        "Host receive timestamps; unknown source sampling and one-way delays.",
                        "No foot-contact ground truth; only five sling-protected episodes.",
                        "Attitude RPY is descriptive only; deployed attitude predictor must use rotation composition.",
                        "Episodes4--5 were already inspected previously, so only diagnostic holdout, not fresh validation."],
               source_files=[dict(path=str(args.input_dir/f"trial{i:02d}.npz"), sha256=hashlib.sha256((args.input_dir/f"trial{i:02d}.npz").read_bytes()).hexdigest()) for i in range(1, 6)]))
    lookup = direction_lookup(trials, feats, names)
    write_json(out/"knee_direction_lookup.json", lookup)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})

    # A single pair of heatmaps compares fitted associations to later episodes.
    rho24 = np.array(correlations["24"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 12), sharey=True)
    for ax, rho, title in ((axes[0], rho24[:3].mean(0), "Mean rho: development episodes 1--3"),
                           (axes[1], rho24[3:].mean(0), "Mean rho: diagnostic episodes 4--5")):
        im = ax.imshow(rho, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(np.arange(12), TARGETS, rotation=65, ha="right")
        ax.set_yticks(np.arange(len(names)), names)
        ax.set_title(title+"\nleg state at t versus world IMU at t + 24 ms")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, pad=.02, label="Pearson rho (not causality)")
    fig.subplots_adjust(left=.17, bottom=.14, right=.91, wspace=.08, top=.94)
    fig.savefig(out/"01_leg_future_imu_correlations.png", dpi=170)
    plt.close(fig)

    # Select on training only, keeping acceleration/omega/alpha all represented.
    selected = []
    for target in (2, 3, 6, 7):
        score = rank_train(rho24)
        fi = int(np.argmax(score[:, target]))
        selected.append((fi, target))
    leads = np.arange(0, 302, 6)
    lead_rho = np.empty((len(leads), 5, len(selected)))
    for hi, ms in enumerate(leads):
        h = round(ms/1000/DT)
        for k, (f, d) in enumerate(zip(feats, trials)):
            r = correlation(f[anchors], d["y"][anchors+h])
            lead_rho[hi, k] = [r[fi, ti] for fi, ti in selected]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for j, (ax, (fi, ti)) in enumerate(zip(axes.ravel(), selected)):
        for k in range(5):
            ax.plot(leads, lead_rho[:, k, j], label=f"episode{k+1}", ls="-" if k < 3 else "--")
        ax.axvline(54, color="gray", ls=":")
        ax.axhline(0, color="gray", lw=.5)
        ax.set_title(f"{names[fi]}(t) vs {TARGETS[ti]}(t + lead)")
        ax.set_ylabel("rho"); ax.set_xlabel("Future lead (ms)"); ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=3)
    fig.tight_layout(); fig.savefig(out/"02_train_selected_lead_curves.png", dpi=170); plt.close(fig)
    write_json(out/"lead_curves.json", dict(leads_ms=leads.tolist(), selected_pairs=[dict(feature=names[i], target=TARGETS[j]) for i, j in selected],
                values_axes="lead, episode, pair", rho=lead_rho.tolist(),
                selected_by="training1--3 stable 24ms correlations; lead curves themselves not used for choosing a lag"))

    # The same knee angle can occur on the flexing or extending branch.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, episode_ids, title in ((axes[0], (0, 1, 2), "Development episodes 1--3"),
                                    (axes[1], (3, 4), "Diagnostic episodes 4--5")):
        for positive, color, label in ((False, "tab:blue", "knee dq < 0"), (True, "tab:orange", "knee dq >= 0")):
            xx, yy = [], []
            for k in episode_ids:
                m = feats[k][anchors, names.index("dq_L_knee")] >= 0
                chosen = anchors[m == positive][::2]
                xx.extend(feats[k][chosen, names.index("q_L_knee")]*180/np.pi)
                yy.extend(trials[k]["acc"][chosen+12, 2])
            ax.scatter(xx, yy, s=4, alpha=.2, c=color, label=label, rasterized=True)
        ax.set_xlabel("Left knee q at t (deg)"); ax.set_title(title); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel("World az at t + 24 ms (m/s2, causal 15Hz)")
    fig.suptitle("One joint angle is not a unique phase: direction matters")
    fig.tight_layout(); fig.savefig(out/"03_knee_direction_ambiguity.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k, (d, f) in enumerate(zip(trials, feats)):
        axes[0].plot(f[anchors, names.index("q_hip_pitch_diff")], f[anchors, names.index("dq_hip_pitch_diff")], lw=.5, alpha=.65, label=f"episode{k+1}")
        axes[1].plot(d["t"][anchors], f[anchors, names.index("q_knee_sum")], lw=.8, label=f"episode{k+1}")
    axes[0].set_xlabel("L-R hip pitch q (rad)"); axes[0].set_ylabel("L-R hip pitch dq (rad/s)")
    axes[0].set_title("Joint angle + direction disambiguate phase")
    axes[1].set_xlabel("Task time (s)"); axes[1].set_ylabel("L+R knee q (rad)")
    axes[1].set_title("Bilateral combination has different symmetry")
    for ax in axes: ax.legend(ncol=3); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/"04_leg_phase_portrait.png", dpi=170); plt.close(fig)

    # Small checks target timestamp causality, not hardware behavior.
    for d, f in zip(trials, feats):
        assert np.isfinite(f).all() and f.shape == (len(d["t"]), len(names))
        assert np.all(d["age"] >= 0) and np.all(d["qage"] >= 0)
    synthetic_x = np.array([[0., 4.], [1., 3.], [2., 2.], [3., 1.]])
    assert np.allclose(correlation(synthetic_x, synthetic_x), [[1., -1.], [-1., 1.]])
    original = feats[0][:1000].copy()
    modified = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in trials[0].items()}
    modified["q"][1000:] = 1e6; modified["dq"][1000:] = -1e6
    assert np.array_equal(features(modified)[0][:1000], original)
    write_json(out/"checks.json", dict(finite_shape=True, asof_ages_nonnegative=True,
               correlation_sign_synthetic=True, features_unchanged_by_future_mutation=True))
    print(json.dumps(dict(output=str(out), top_24ms=ranked["24"][:12], knee_direction=lookup), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
