#!/usr/bin/env python3
"""Offline-only raw walking audit. No SDK, network, or command output.

Explicit chronological episode split: 1--3 development, 4--5 held out.
Extracted arrays are derived copies; source JSONL/profile/status remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from scipy.signal import lfilter

TRIALS = (
    "g1_walk_trial02_20260917_1443",
    "g1_walk_trial03_20260917_1455",
    "g1_walk_20260917_150150",
    "g1_walk_20260917_150634",
    "g1_walk_20260917_151055",
)
DT = .002


def lowpass(x, hz):
    gain = 1 - np.exp(-2 * np.pi * hz * DT)
    # Constant first sample initializes the filter without a startup impulse.
    x = np.asarray(x)
    return lfilter([gain], [1, -(1 - gain)], x - x[0], axis=0) + x[0]


def asof(times, values, grid):
    idx = np.searchsorted(times, grid, side="right") - 1
    if np.any(idx < 0):
        raise ValueError("no past sample; never backfill from the future")
    return values[idx], grid - times[idx]


def prepare(d, hz=15.):
    grid = np.arange(-.5, 21.00001, DT)
    a, age = asof(d["imu_t"], d["world_acc"], grid)
    w, _ = asof(d["imu_t"], d["world_omega"], grid)
    rpy, _ = asof(d["imu_t"], np.unwrap(d["imu"][:, 6:9], axis=0), grid)
    q, qage = asof(d["low_t"], d["q"], grid)
    dq, _ = asof(d["low_t"], d["dq"], grid)
    acc = lowpass(a, hz)
    omega = lowpass(w, hz)
    alpha = lowpass(np.vstack([np.zeros((1, 3)), np.diff(omega, axis=0)/DT]), hz)
    x = lowpass(q[:, 0] - q[:, 6], 10.)
    speed = lowpass(np.sqrt(np.mean(dq[:, :12]**2, axis=1)), 10.)
    return dict(t=grid, acc_raw=a, acc=acc, omega=omega, alpha=alpha,
                rpy=rpy, y=np.column_stack([acc, omega, alpha, rpy]),
                q=q, dq=dq, hip=x, speed=speed, age=age, qage=qage)


def leg_events(t, x):
    # Causal leg-configuration landmarks, NOT measured foot contacts.
    pos_ready = neg_ready = False
    last = {1: -np.inf, -1: -np.inf}
    result = []
    for i in range(len(t)):
        if t[i] < 5 or t[i] >= 17:
            continue
        if x[i] <= -.03:
            pos_ready = True
        if x[i] >= .03:
            neg_ready = True
        if pos_ready and x[i] >= 0 and t[i] - last[1] >= .3:
            result.append((i, 1)); last[1] = t[i]; pos_ready = False
        if neg_ready and x[i] <= 0 and t[i] - last[-1] >= .3:
            result.append((i, -1)); last[-1] = t[i]; neg_ready = False
    return result


def sustained(t, condition, start, seconds):
    count = 0
    for i in np.where(t >= start)[0]:
        count = count + 1 if condition[i] else 0
        if count >= int(round(seconds / DT)):
            return float(t[i-count+1]), float(t[i])
    return None, None


def event_predictions(times, training_period):
    rows = []
    for j in range(1, len(times)-1):
        prior = np.diff(times[max(0, j-3):j+1])
        estimate = float(np.median(prior))
        rows.append(dict(anchor_s=float(times[j]), target_s=float(times[j+1]),
                         adaptive_error_ms=float((times[j]+estimate-times[j+1])*1000),
                         fixed_training_error_ms=float((times[j]+training_period-times[j+1])*1000),
                         fixed_0p8_error_ms=float((times[j]+.8-times[j+1])*1000)))
    return rows


def interp_columns(x, xp, yp):
    return np.column_stack([np.interp(x, xp, yp[:, k]) for k in range(yp.shape[1])])


def phase_sample(template, phase):
    xp = np.linspace(0, 1, len(template)+1)
    yp = np.vstack([template, template[0]])
    return interp_columns(np.atleast_1d(phase) % 1, xp, yp)


def group_rmse(error):
    return {name: float(np.sqrt(np.mean(error[:, i:i+3]**2)))
            for i, name in ((0, "acc_m_s2"), (3, "omega_rad_s"),
                            (6, "alpha_rad_s2"), (9, "rpy_rad"))}


def analyze(out, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for s in summaries:
        q = s["quality"]
        if not (q["normal_release"] and q["queue_drained_zero_drops"] and
                not q["crc_errors"] and not q["tick_regressions"] and
                not q["failure_events"] and not q["nonfinite_values"] and
                s["operator_status"]["valid_for_disturbance_dataset"] and
                not q["velocity_missing_replies"] and
                set(q["velocity_return_codes"]) == {"0"} and
                q["fsm_values"] == [500] and
                not q["forward_requests_outside_window"] and
                all(not q[n]["callback_sequence_gaps"] and not q[n]["gaps_over_6ms"]
                    for n in ("imu", "low"))):
            raise ValueError(f"trial {s['index']}: failed quality checks; no pooled analysis")
    if any(s["profile"] != summaries[0]["profile"] for s in summaries):
        raise ValueError("control profiles differ; do not pool episodes silently")
    data = [dict(np.load(out / f"trial{i:02d}.npz")) for i in range(1, 6)]
    trials = [prepare(d) for d in data]
    events = [leg_events(d["t"], d["hip"]) for d in trials]
    positive = [np.array([d["t"][j] for j, sign in ev if sign == 1])
                for d, ev in zip(trials, events)]
    train_periods = np.concatenate([np.diff(e[(e >= 7) & (e < 14.8)]) for e in positive[:3]])
    train_period = float(np.median(train_periods))
    # All fit choices use development episodes only. Evaluation always retains
    # entire episodes 4/5; no cross-episode random sampling or future rephasing.
    bins = np.arange(128)/128
    cycles, raw_cycles, cycle_meta = [], [], []
    for k, (d, e) in enumerate(zip(trials, positive)):
        for a, b in zip(e[:-1], e[1:]):
            if a >= 7 and b < 14.8:
                # Retrospective time warping is permitted for TRAINING/shape
                # diagnostics, never for causal prediction on test episodes.
                cycles.append(interp_columns(a+bins*(b-a), d["t"], d["y"]))
                raw_cycles.append(interp_columns(a+bins*(b-a), d["t"], d["acc_raw"]))
                cycle_meta.append((k, a, b))
    cycles = np.array(cycles)
    train_ids = np.array([m[0] < 3 for m in cycle_meta])
    template = np.mean(cycles[train_ids], axis=0)
    raw_template = np.mean(np.array(raw_cycles)[train_ids], axis=0)
    task_mean = np.mean([d["y"] for d in trials[:3]], axis=0)
    period_rule = dict(train_trials=[1, 2, 3], heldout_trials=[4, 5], grid_dt_s=DT,
        imu_filter="causal 1-pole 15 Hz; angular acceleration=backward derivative of filtered world omega, then 15 Hz",
        leg_filter_hz=10, leg_landmark="q[0]-q[6] zero crossing; +/-0.03 rad re-arm, same-sign refractory 0.3s",
        period_estimator="median of last at most 3 completed same-sign cycles",
        steady_window_s=[7, 14.8], horizons_ms=[6, 12, 24, 36, 54],
        initial_training_period_s=train_period, official_phase_available=False,
        event_semantics="kinematic landmarks and accelerometer peaks; not ground-truth contact",
        template_scope="diagnostic baselines only, no deployable template exported",
        model_parameters_equal_excluding_confirmed_by=True,
        supplementary_checks="raw acceleration point forecast; 6ms interval-average acc/alpha; impact peak timing",
        world_conversion="R from torso quaternion wxyz; a_W=R*f_IMU+[0,0,-9.81]; omega_W=R*gyro_IMU; no bias subtraction",
        orientation_diagnostic_only="Unwrapped RPY used only for descriptive scoring, not a deployable rotation template.",
        software=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        source_timing="host receive timestamps; no one-way delay or true sample time known")
    write_json(out / "analysis_protocol.json", period_rule)
    trial_results = []
    for k, (d, e, raw, summary) in enumerate(zip(trials, positive, data, summaries)):
        t = d["t"]
        stable_e = e[(e >= 7) & (e < 14.8)]
        start, start_known = sustained(t, d["speed"] > .2, 5, .1)
        stop, stop_known = sustained(t, d["speed"] < .2, 15, .3)
        both = np.array([[d["t"][j], sign] for j, sign in events[k]])
        signed_start = both[(both[:, 0] >= 5) & (both[:, 0] < 6)]
        predictions = event_predictions(e[(e >= 5) & (e < 15)], train_period)
        steady_predictions = [r for r in predictions if r["anchor_s"] >= 7 and r["target_s"] < 14.8]
        # One accelerometer peak per half-cycle as retrospective impact proxy;
        # absent force sensors these labels do not establish touchdown time.
        impacts = []
        power = np.linalg.norm(d["acc_raw"]-lowpass(d["acc_raw"], 5), axis=1)
        for (a, sign), (b, _) in zip(both[:-1], both[1:]):
            if a < 7 or b >= 14.8:
                continue
            mask = np.flatnonzero((t >= a) & (t < b))
            j = mask[np.argmax(power[mask])]
            impacts.append(dict(leg_event_s=float(a), sign=int(sign), peak_s=float(t[j]),
                                peak_m_s2=float(power[j]), offset_s=float(t[j]-a)))
        ids = [i for i, m in enumerate(cycle_meta) if m[0] == k]
        local_cycles = cycles[ids]
        shape = {}
        for sl, name in ((slice(0, 3), "acc"), (slice(3, 6), "omega"), (slice(6, 9), "alpha")):
            normed = []
            for c in local_cycles:
                # Mean center each axis; flatten only AFTER centering.
                ca = c[:, sl]-c[:, sl].mean(axis=0)
                ta = template[:, sl]-template[:, sl].mean(axis=0)
                normed.append(np.corrcoef(ca.ravel(), ta.ravel())[0, 1])
            shape[name + "_phase_normalized_corr"] = stats(normed)
        trial_results.append(dict(index=k+1, partition="train" if k < 3 else "heldout",
            kinematic_onset_s=start, onset_confirmation_s=start_known,
            kinematic_stop_s=stop, stop_confirmation_s=stop_known,
            all_positive_landmarks_s=e.tolist(),
            first_landmarks=signed_start.tolist(),
            steady_full_cycle_s=stats(np.diff(stable_e)),
            steady_cycles=len(ids), event_prediction_rows=predictions,
            event_prediction_abs_error_ms={name:stats([abs(r[name]) for r in steady_predictions])
                for name in ("adaptive_error_ms", "fixed_training_error_ms", "fixed_0p8_error_ms")},
            impact_proxies=impacts, shape=shape,
            imu_age_at_2ms_grid_ms=stats(d["age"][(t >= 0)]*1000),
            lowstate_age_at_2ms_grid_ms=stats(d["qage"][(t >= 0)]*1000)))
    write_json(out / "event_analysis.json", trial_results)
    impact_offsets = {sign: float(np.median([p["offset_s"] for r in trial_results[:3]
        for p in r["impact_proxies"] if p["sign"] == sign])) for sign in (-1, 1)}
    impact_rows = [dict(trial=r["index"], **p,
        predicted_peak_s=p["leg_event_s"]+impact_offsets[p["sign"]],
        error_ms=(impact_offsets[p["sign"]]-p["offset_s"])*1000)
        for r in trial_results[3:] for p in r["impact_proxies"]]
    impact_summary = dict(training_offset_s=impact_offsets, heldout_predictions=impact_rows,
        heldout_absolute_error_ms=stats([abs(p["error_ms"]) for p in impact_rows]),
        caveat="Retrospective strongest acceleration peak per half-cycle; not independently measured foot contact.")
    write_json(out/"impact_prediction.json", impact_summary)
    # Predict entire held-out steady windows on 6ms query anchors.
    prediction_metrics = []
    interval_metrics, raw_metrics = [], []
    prediction_arrays = {}
    for k in (3, 4):
        d, e = trials[k], positive[k]
        t, y = d["t"], d["y"]
        anchors = np.where((t >= 7) & (t < 14.7))[0][::3]
        last = np.searchsorted(e, t[anchors], side="right")-1
        available = last >= 1
        anchors, last = anchors[available], last[available]
        periods = np.array([np.median(np.diff(e[max(0, j-3):j+1])) for j in last])
        phase_now = (t[anchors]-e[last])/periods
        phase_current = phase_sample(template, phase_now)
        for ms in (6, 12, 24, 36, 54):
            h = int(round(ms/1000/DT))
            truth = y[anchors+h]
            ph = phase_sample(template, phase_now + ms/1000/periods)
            # Last-cycle replay is also causal: every indexed observation <= t.
            past_time = t[anchors]+ms/1000-periods
            past_i = np.searchsorted(t, past_time, side="right")-1
            assert np.all(past_i <= anchors)
            methods = dict(zoh=y[anchors], task_time=task_mean[anchors+h],
                phase_template=ph, phase_delta=y[anchors]+ph-phase_current,
                previous_cycle=y[past_i])
            for name, pred in methods.items():
                prediction_metrics.append(dict(trial=k+1, horizon_ms=ms, method=name,
                    samples=len(anchors), rmse=group_rmse(pred-truth)))
            # A 6ms interval ending at the stated horizon, approximated by the
            # three 2ms right-endpoint samples. No future samples enter a forecast.
            interval_truth = np.mean([y[anchors+h-j] for j in (0, 1, 2)], axis=0)
            interval_phase = np.mean([phase_sample(template,
                phase_now+(h-j)*DT/periods) for j in (0, 1, 2)], axis=0)
            interval_methods = dict(zoh=y[anchors],
                task_time=np.mean([task_mean[anchors+h-j] for j in (0, 1, 2)], axis=0),
                phase_template=interval_phase,
                phase_delta=y[anchors]+interval_phase-phase_current)
            for name, pred in interval_methods.items():
                rmse = group_rmse(pred-interval_truth)
                interval_metrics.append(dict(trial=k+1, interval_end_ms=ms, method=name,
                    samples=len(anchors), rmse={key: rmse[key] for key in ("acc_m_s2", "alpha_rad_s2")}))
            raw_ph = phase_sample(raw_template, phase_now+ms/1000/periods)
            for name, pred in dict(zoh=d["acc_raw"][anchors], phase_template=raw_ph).items():
                raw_metrics.append(dict(trial=k+1, horizon_ms=ms, method=name,
                    samples=len(anchors), acc_rmse_m_s2=float(np.sqrt(np.mean((pred-d["acc_raw"][anchors+h])**2)))))
            if ms == 24:
                prediction_arrays[f"trial{k+1}_t"] = t[anchors]+ms/1000
                prediction_arrays[f"trial{k+1}_truth"] = truth
                for name, pred in methods.items():
                    prediction_arrays[f"trial{k+1}_{name}"] = pred
    write_json(out / "prediction_metrics.json", prediction_metrics)
    write_json(out / "interval_prediction_metrics.json", interval_metrics)
    write_json(out / "raw_acc_prediction_metrics.json", raw_metrics)
    np.savez_compressed(out / "heldout_predictions_24ms.npz", **prediction_arrays)
    # Pairwise task-time repeatability, no optimised shift; mean-centered axes.
    pairwise = []
    for a in range(5):
        for b in range(a+1, 5):
            mask = (trials[a]["t"] >= 7) & (trials[a]["t"] < 14.8)
            row = dict(a=a+1, b=b+1)
            for i, name in ((0, "acc"), (3, "omega"), (6, "alpha")):
                x, z = trials[a]["y"][mask, i:i+3], trials[b]["y"][mask, i:i+3]
                row[name+"_corr"] = float(np.corrcoef((x-x.mean(0)).ravel(), (z-z.mean(0)).ravel())[0,1])
            pairwise.append(row)
    write_json(out / "task_repeatability.json", pairwise)
    # Filtered overview. The unfiltered series remains in extracted arrays.
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for k,d in enumerate(trials):
        axes[0].plot(d["t"], d["hip"], label=f"trial {k+1}", lw=.8)
        axes[1].plot(d["t"], d["acc"][:,2], lw=.8)
        axes[2].plot(d["t"], d["omega"][:,0], lw=.8)
        axes[3].plot(d["t"], d["rpy"][:,2]*180/np.pi, lw=.8)
    for ax, title in zip(axes, ["Hip difference (rad)", "World az (m/s2)", "World omega x (rad/s)", "IMU world yaw (deg)"]):
        ax.set_ylabel(title); ax.axvline(5,color="black",ls="--"); ax.axvline(15,color="black",ls="--"); ax.grid(alpha=.25)
    axes[0].legend(ncol=5); axes[-1].set_xlim(3,18); axes[-1].set_xlabel("Task time (s)")
    fig.tight_layout(); fig.savefig(out/"task_alignment.png",dpi=160); plt.close(fig)
    fig, axes=plt.subplots(2,2,figsize=(12,7))
    for k,r in enumerate(trial_results):
        e=np.array(r["all_positive_landmarks_s"]); m=(e[:-1]>=5)&(e[1:]<15)
        axes[0,0].plot(e[1:][m],np.diff(e)[m],"o-",label=str(k+1))
        pred=r["event_prediction_rows"]
        axes[0,1].plot([p["target_s"] for p in pred],[p["adaptive_error_ms"] for p in pred],"o-",label=str(k+1))
    axes[0,0].axhline(.8,color="black",ls="--"); axes[0,0].set_title("Same-leg cycle duration (s), not fixed 0.8 s")
    axes[0,1].axhline(0,color="black",ls="--"); axes[0,1].set_title("Causal next-cycle landmark error (ms)")
    for ax,idx,title in ((axes[1,0],2,"Phase-normalized world az"),(axes[1,1],3,"Phase-normalized world omega x")):
        for k in range(5):
            ids=[i for i,m in enumerate(cycle_meta) if m[0]==k]
            ax.plot(bins,cycles[ids,:,idx].mean(axis=0),label=str(k+1))
        ax.set_title(title); ax.set_xlabel("Retrospective phase (shape diagnostic only)")
    for ax in axes.ravel():ax.grid(alpha=.25);ax.legend(ncol=5)
    fig.tight_layout();fig.savefig(out/"cycle_repeatability.png",dpi=160);plt.close(fig)
    fig, axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    xp=prediction_arrays["trial4_t"]
    for ax, idx, name in zip(axes,[2,3,6],["az (m/s2)","omega x (rad/s)","alpha x (rad/s2)"]):
        for method in ("truth","zoh","task_time","phase_delta"):
            ax.plot(xp,prediction_arrays[f"trial4_{method}"][:,idx],label=method,lw=1)
        ax.set_ylabel(name);ax.grid(alpha=.25)
    axes[0].legend(ncol=4);axes[-1].set_xlim(10,11.6);axes[-1].set_xlabel("Prediction target time (s), 24ms ahead, held-out trial4")
    fig.tight_layout();fig.savefig(out/"heldout_24ms.png",dpi=160);plt.close(fig)
    print("analysis written", flush=True)


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), p50=float(np.median(a)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
                max=float(a.max()))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def profile_dict(path):
    return dict(line.strip().split("=", 1) for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#"))


def imu_values(r):
    return (r["quaternion_wxyz"] + r["rpy_rad"] +
            r["gyroscope_rad_s"] + r["accelerometer_raw_m_s2"])


def audit_trial(directory, out, index):
    counts = Counter()
    imu, low, pelvis, q, dq, tau, commands, velocity, replies = ([] for _ in range(9))
    events, fsm, phase, ends, drained = ([] for _ in range(5))
    sha = hashlib.sha256()
    epoch = None
    invalid_crc = invalid_motor_layout = numeric_nonfinite = 0
    with (directory / "raw.jsonl").open("rb") as f:
        for line_no, line in enumerate(f, 1):
            sha.update(line)
            try:
                r = json.loads(line)
            except Exception as e:
                raise ValueError(f"{directory.name}:{line_no}: invalid JSON") from e
            schema, event = r.get("schema"), r.get("event")
            counts[schema if event is None else event] += 1
            if schema == "g1_torso_imu_raw_v1":
                imu.append([r["received_monotonic_ns"], r["host_callback_sequence"], *imu_values(r)])
            elif schema == "g1_lowstate_raw_v1":
                low.append([r["received_monotonic_ns"], r["host_callback_sequence"],
                            r["tick_raw"], r["mode_pr"], r["mode_machine"]])
                invalid_crc += not r["crc_valid"]
                motors = r["motors"]
                invalid_motor_layout += [m["index"] for m in motors] != list(range(35))
                q.append([m["q_rad"] for m in motors])
                dq.append([m["dq_rad_s"] for m in motors])
                tau.append([m["tau_est_nm"] for m in motors])
                pelvis.append(imu_values(r["pelvis_imu"]))
            elif schema == "g1_arm_static_command_record_v1":
                commands.append(r)
            elif event == "task_epoch":
                if epoch is not None:
                    raise ValueError("multiple task epochs")
                epoch = r["task_epoch_monotonic_ns"]
            elif event == "velocity_request":
                velocity.append(r)
            elif event == "velocity_reply":
                replies.append(r)
            elif event == "task_stage":
                events.append(r)
            elif event == "fsm_reply":
                fsm.append(r)
            elif event == "phase_reply":
                phase.append(r)
            elif event == "session_end":
                ends.append(r)
            elif event == "capture_drained":
                drained.append(r)
    if epoch is None or not imu or not low:
        raise ValueError(f"{directory.name}: missing required streams/epoch")
    arrays = dict(imu=np.asarray(imu, float), low=np.asarray(low, float),
                  pelvis=np.asarray(pelvis, float), q=np.asarray(q, float),
                  dq=np.asarray(dq, float), tau=np.asarray(tau, float))
    for a in arrays.values():
        numeric_nonfinite += int(np.sum(~np.isfinite(a)))
    if numeric_nonfinite or invalid_motor_layout:
        raise ValueError("invalid numeric data or motor layout; do not interpolate it")
    quality = {}
    for name in ("imu", "low"):
        a = arrays[name]
        t = (a[:, 0] - epoch) / 1e9
        arrays[name + "_t"] = t
        dt = np.diff(t)
        active = (t[1:] >= 0) & (t[1:] <= 21.05)
        quality[name] = dict(total=len(t), task_samples=int(((t >= 0) & (t <= 21)).sum()),
            coverage_s=[float(t.min()), float(t.max())],
            callback_sequence_gaps=int(np.sum(np.diff(a[:, 1]) != 1)),
            nonpositive_time_deltas=int(np.sum(dt <= 0)),
            interval_ms=stats(dt[active] * 1000),
            gaps_over_6ms=int(np.sum(dt[active] > .006)),
            gaps_over_20ms=int(np.sum(dt[active] > .020)))
        if np.any(dt <= 0):
            raise ValueError("non-monotonic per-topic callbacks; causal replay needs explicit repair")
    tick_delta = np.diff(arrays["low"][:, 2].astype(np.int64)) % (2**32)
    quality["tick_repeats"] = int(np.sum(tick_delta == 0))
    quality["tick_regressions"] = int(np.sum(tick_delta >= 2**31))
    quality["crc_errors"] = invalid_crc
    quality["nonfinite_values"] = numeric_nonfinite
    quality["quaternion_norm"] = stats(np.linalg.norm(arrays["imu"][:, 2:6], axis=1))
    rot = Rotation.from_quat(arrays["imu"][:, [3, 4, 5, 2]])
    rpy = rot.as_euler("xyz")
    rpy_error = (rpy - arrays["imu"][:, 6:9] + np.pi) % (2 * np.pi) - np.pi
    quality["quat_rpy_max_error_deg"] = float(np.max(np.abs(rpy_error)) * 180 / np.pi)
    arrays["world_specific_force"] = rot.apply(arrays["imu"][:, 12:15])
    arrays["world_acc"] = arrays["world_specific_force"] + [0, 0, -9.81]
    arrays["world_omega"] = rot.apply(arrays["imu"][:, 9:12])
    for key, columns in (("quaternion", slice(2, 6)), ("gyro", slice(9, 12)), ("acc", slice(12, 15))):
        vals = arrays["imu"][:, columns]
        same = np.all(np.diff(vals, axis=0) == 0, axis=1)
        quality[key + "_exact_adjacent_repeat_fraction"] = float(np.mean(same))
    v = np.array([[r["request_ns"], r["vx_m_s"], r["yaw_rate_rad_s"], r["duration_s"]] for r in velocity])
    v[:, 0] = (v[:, 0] - epoch) / 1e9
    arrays["velocity"] = v
    vr = {r["request_ns"]: r for r in replies}
    missing_reply = sum(r["request_ns"] not in vr for r in velocity)
    forward = v[v[:, 1] > 0]
    zeros_after = v[(v[:, 0] >= 15) & (v[:, 1] == 0)]
    arrays["command_t"] = np.array([(r["write_begin_monotonic_ns"] - epoch) / 1e9 for r in commands])
    arrays["command_weight"] = np.array([r["weight"] for r in commands])
    arrays["command_q"] = np.array([r["q_target"] for r in commands])
    arrays["command_q_measured"] = np.array([r["q_measured"] for r in commands])
    normal = (len(ends) == 1 and ends[0]["outcome"] == "normal_release_completed" and
              ends[0]["final_weight"] == 0 and ends[0]["velocity_final_rpc_ok"])
    queue_ok = len(drained) == 1 and drained[0]["queue_dropped"] == 0
    errors = [k for k in counts if k and ("fault" in k or "error" in k or "exception" in k or "failed" in k)]
    quality.update(normal_release=bool(normal), queue_drained_zero_drops=bool(queue_ok),
        failure_events=errors, fsm_return_codes=dict(Counter(str(r["return_code"]) for r in fsm)),
        fsm_values=sorted({r["fsm_id"] for r in fsm}),
        phase_return_codes=dict(Counter(str(r["return_code"]) for r in phase)),
        velocity_return_codes=dict(Counter(str(r["return_code"]) for r in replies)),
        velocity_missing_replies=missing_reply, nonzero_velocity_requests=len(forward),
        first_forward_request_s=float(forward[0, 0]),
        last_forward_request_s=float(forward[-1, 0]),
        first_zero_after_walk_s=float(zeros_after[0, 0]),
        forward_requests_outside_window=int(np.sum((forward[:, 0] < 5) | (forward[:, 0] >= 15))),
        velocity_rpc_ms=stats([(r["reply_ns"]-r["request_ns"])*1e-6 for r in replies]),
        arm_write_interval_ms=stats(np.diff(arrays["command_t"])*1000),
        arm_write_duration_ms=stats([(r["write_end_monotonic_ns"]-r["write_begin_monotonic_ns"])*1e-6 for r in commands]))
    baseline = (arrays["imu_t"] >= 4) & (arrays["imu_t"] < 5)
    walk = (arrays["imu_t"] >= 5) & (arrays["imu_t"] < 15)
    quality["baseline_world_acc_mean"] = arrays["world_acc"][baseline].mean(axis=0).tolist()
    quality["baseline_world_acc_std"] = arrays["world_acc"][baseline].std(axis=0).tolist()
    quality["baseline_rpy_deg"] = (arrays["imu"][baseline, 6:9].mean(axis=0)*180/np.pi).tolist()
    quality["walking_yaw_deg"] = stats(arrays["imu"][walk, 8]*180/np.pi)
    quality["walking_peak_acc_norm"] = float(np.linalg.norm(arrays["world_acc"][walk], axis=1).max())
    quality["walking_peak_gyro_norm"] = float(np.linalg.norm(arrays["world_omega"][walk], axis=1).max())
    arrays["epoch_ns"] = np.array(epoch, np.int64)
    np.savez_compressed(out / f"trial{index:02d}.npz", **arrays)
    profile = profile_dict(directory / "arm_profile.conf")
    dynamics_profile = {k: v for k, v in profile.items() if k != "confirmed_by"}
    summary = dict(index=index, source=str(directory), sha256=sha.hexdigest(),
        profile_sha256=hashlib.sha256((directory / "arm_profile.conf").read_bytes()).hexdigest(),
        profile=dynamics_profile, counters=dict(counts), quality=quality,
        stages=events, operator_status=json.loads((directory / "trial_status.json").read_text()))
    write_json(out / f"trial{index:02d}_audit.json", summary)
    print(f"extracted trial {index}: {len(imu)} torso, {len(low)} LowState", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("evaluation/hardware_shadow/commissioning"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reuse-extraction", action="store_true")
    args = p.parse_args()
    if args.reuse_extraction:
        summaries = [json.loads((args.output_dir / f"trial{i:02d}_audit.json").read_text()) for i in range(1, 6)]
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        summaries = [audit_trial(args.data_root / name, args.output_dir, i)
                     for i, name in enumerate(TRIALS, 1)]
    write_json(args.output_dir / "quality_summary.json", summaries)
    analyze(args.output_dir, summaries)


if __name__ == "__main__":
    main()
