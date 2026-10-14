"""Offline event timing study; reads logs only and never initializes DDS."""
import json
from pathlib import Path
import sys

import numpy as np
import orjson
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

ROOT = Path('/home/fjk/g1_ws/hold-my-beer-mpc')
sys.path.insert(0, str(ROOT / 'tools/g1_commissioning'))
from endpoint_pose import EndpointModel

RUNS = [
    'morning_M01_mpc_learned_zero_payload75_walk_20261013_150426_750630_999f',
    'morning_M06_mpc_learned_repeat_zero_payload75_walk_20261013_151617_849525_70f1',
    'morning_M01_mpc_learned_zero_payload75_walk_20261013_164838_063152_fd65',
    'morning_M01_mpc_learned_zero_payload75_walk_20261014_124321_974583_1119',
    'morning_M01_mpc_learned_zero_payload75_walk_20261014_125333_831147_ee9f',
]


def latest_index(times, queries):
    return np.maximum(0, np.searchsorted(times, queries, side='right') - 1)


def interp(t, x, at):
    return np.interp(at, t, x)


def analyze(folder, model):
    path = ROOT / 'evaluation/hardware_shadow/commissioning' / folder / 'raw.jsonl'
    commands, imus = [], []
    with path.open('rb') as stream:
        for line in stream:
            row = orjson.loads(line)
            if row.get('schema') == 'g1_hardware_mpc_command_v1' and 5 <= row['task_elapsed_s'] < 15:
                commands.append(row)
            elif row.get('schema') == 'g1_mpc_predictor_imu_v1':
                imus.append(row)
    imus.sort(key=lambda r: r['received_monotonic_ns'])
    it = np.asarray([r['received_monotonic_ns'] for r in imus], dtype=np.int64)
    ct = np.asarray([r['task_elapsed_s'] for r in commands])
    ci = latest_index(it, np.asarray([r['imu_received_monotonic_ns'] for r in commands], dtype=np.int64))
    tilt = np.zeros((len(commands), 2))
    for n, (row, i) in enumerate(zip(commands, ci)):
        q = np.asarray(row['q_measured_rad'])
        quat = np.asarray(imus[i]['quaternion_wxyz'])
        R_WB = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        rel = model.relative(q)
        for side_i, side in enumerate(('left', 'right')):
            z = R_WB @ rel[side][1][:, 2]
            tilt[n, side_i] = np.degrees(np.arccos(np.clip(z[2], -1, 1)))

    # Reconstruct raw torso translational acceleration in the same way as the
    # existing impact audit. Vertical-axis tilt is yaw invariant.
    qt = np.asarray([r['quaternion_wxyz'] for r in imus])[:, [1, 2, 3, 0]]
    specific = np.asarray([r['accelerometer_raw_m_s2'] for r in imus])
    raw = Rotation.from_quat(qt).apply(specific) + np.array([0., 0., -9.81])
    rn = np.linalg.norm(raw, axis=1)
    # Command task time tied to the actual selected IMU receive timestamp.
    imu_task = np.interp(it, np.asarray([r['imu_received_monotonic_ns'] for r in commands]), ct)
    walk = (imu_task >= 5) & (imu_task < 15)
    threshold = float(np.percentile(rn[walk], 90))
    peaks, _ = find_peaks(rn, height=threshold, prominence=1., distance=125)
    peaks = peaks[walk[peaks]]

    # The top-level raw field is a legacy 13-slot placeholder in these runs.
    # The five-joint acceleration actually selected by the torque mapper is
    # stored explicitly here and matches the MPC diagnostics/replay contract.
    ddq = np.asarray([r['post_transition_ddq_rad_s2'] for r in commands])
    tau = np.asarray([r['tau_total_estimated_at_feedback_nm'] for r in commands])
    ddq_norm = np.linalg.norm(ddq, axis=1)
    dtau_norm = np.r_[0., np.linalg.norm(np.diff(tau, axis=0), axis=1)]
    events = []
    for p in peaks:
        t0 = float(imu_task[p])
        pre = (ct >= t0 - .030) & (ct <= t0 - .012)
        post = (ct >= t0) & (ct <= t0 + .060)
        if pre.sum() < 2 or post.sum() < 3:
            continue
        base = np.median(tilt[pre], axis=0)
        dev = np.abs(tilt[post] - base)
        inds = np.flatnonzero(post)
        li = inds[np.argmax(dev[:, 0])]
        ri = inds[np.argmax(dev[:, 1])]
        ai = inds[np.argmax(ddq_norm[post])]
        ti = inds[np.argmax(dtau_norm[post])]
        events.append(dict(
            impact_task_s=t0, raw_torso_acc_norm=float(rn[p]),
            left_pre_tilt_deg=float(base[0]), right_pre_tilt_deg=float(base[1]),
            left_tilt_change_peak_deg=float(dev[:, 0].max()),
            right_tilt_change_peak_deg=float(dev[:, 1].max()),
            left_tilt_peak_lag_ms=float((ct[li] - t0) * 1000),
            right_tilt_peak_lag_ms=float((ct[ri] - t0) * 1000),
            ddq_norm_peak=float(ddq_norm[ai]), ddq_peak_lag_ms=float((ct[ai] - t0) * 1000),
            torque_step_norm_peak=float(dtau_norm[ti]), torque_step_peak_lag_ms=float((ct[ti] - t0) * 1000),
            right_tilt_change_at_ddq_peak_deg=float(abs(tilt[ai, 1] - base[1])),
            left_tilt_change_at_ddq_peak_deg=float(abs(tilt[ai, 0] - base[0])),
        ))
    return dict(source=str(path.relative_to(ROOT)), threshold=threshold, event_count=len(events), events=events)


def summarize(runs):
    ev = [e for r in runs for e in r['events']]
    arr = lambda key: np.asarray([e[key] for e in ev])
    summary = {'event_count': len(ev)}
    for key in ('raw_torso_acc_norm', 'left_pre_tilt_deg', 'right_pre_tilt_deg',
                'left_tilt_change_peak_deg', 'right_tilt_change_peak_deg',
                'left_tilt_peak_lag_ms', 'right_tilt_peak_lag_ms',
                'ddq_norm_peak', 'ddq_peak_lag_ms', 'torque_step_norm_peak',
                'torque_step_peak_lag_ms', 'right_tilt_change_at_ddq_peak_deg',
                'left_tilt_change_at_ddq_peak_deg'):
        x = arr(key)
        summary[key] = {'median': float(np.median(x)), 'p90': float(np.percentile(x, 90)), 'mean': float(x.mean())}
    left = arr('left_tilt_change_peak_deg'); right = arr('right_tilt_change_peak_deg')
    summary['right_peak_tilt_change_smaller_than_left_fraction'] = float(np.mean(right < left))
    summary['ddq_peak_before_right_tilt_peak_fraction'] = float(np.mean(arr('ddq_peak_lag_ms') < arr('right_tilt_peak_lag_ms')))
    summary['ddq_peak_before_left_tilt_peak_fraction'] = float(np.mean(arr('ddq_peak_lag_ms') < arr('left_tilt_peak_lag_ms')))
    return summary


out = Path(__file__).resolve().parent
model = EndpointModel(ROOT / 'configs/g1_payload75.yaml')
runs = [analyze(folder, model) for folder in RUNS]
result = {'method': 'Offline impact-proxy event alignment; impact is raw torso-IMU acceleration peak, not foot contact truth.',
          'window': 'pre median [-30,-12] ms; response [0,60] ms',
          'runs': runs, 'summary': summarize(runs)}
(out / 'event_timing.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['summary'], indent=2))
