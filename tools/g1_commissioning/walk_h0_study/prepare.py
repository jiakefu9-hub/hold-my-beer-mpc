#!/usr/bin/env python3
"""Offline extraction/audit of the twelve H0 runs; never opens a robot connection.

Source JSONL is immutable. Old W-frame runs are explicitly excluded. The old
auditor is reused for CRC, sequence, RPC and file-integrity checks, not its models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_walk_dataset as audit

TIMES = ('150839', '151234', '151420', '151604', '151801', '151919',
         '152027', '152219', '152321', '152428', '152826', '153034')
DEVELOPMENT = [1, 2, 3, 4, 5, 7, 8]
HELDOUT = [9, 10, 11, 12]
DEFAULT_OUT = Path('evaluation/hardware_shadow/commissioning/walk_h0_predictor_study_20260925')


def extra_records(raw, epoch):
    """Avoid reparsing large motor dictionaries on this metadata-only second pass."""
    remote, references, start = [], [], []
    byte_pattern = re.compile(rb'"wireless_remote_bytes":\[([^]]+)\]')
    time_pattern = re.compile(rb'"received_monotonic_ns":(\d+)')
    with raw.open('rb') as source:
        for line in source:
            if b'"event":"heading_reference_frozen"' in line:
                references.append(json.loads(line))
            elif b'"event":"session_start"' in line:
                start.append(json.loads(line))
            elif b'"schema":"g1_lowstate_raw_v1"' in line:
                match = byte_pattern.search(line)
                if match is None:
                    raise ValueError('LowState without remote record')
                b = bytes(int(v) for v in match[1].split(b','))
                if len(b) != 40:
                    raise ValueError('unexpected remote payload size')
                # SDK example/g1/low_level/gamepad.hpp: header2, buttons uint16,
                # then little-endian lx,rx,ry,L2,ly floats. Ignore reserved bytes.
                buttons = int.from_bytes(b[2:4], 'little')
                axes = struct.unpack('<5f', b[4:24])
                stamp = int(time_pattern.search(line)[1])
                remote.append([(stamp-epoch)*1e-9, buttons, *axes])
    if len(references) != 1 or len(start) != 1:
        raise ValueError('need exactly one frozen heading reference and session start')
    return references[0], start[0], np.asarray(remote)


def prepare_h0(d, reference):
    """Constant yaw rotation, causal last-received-value resampling and filters."""
    yaw0 = float(reference['yaw0_rad'])
    h0w = Rotation.from_euler('z', -yaw0)
    r_hi = h0w * Rotation.from_quat(d['imu'][:, [3, 4, 5, 2]])
    rpy = np.unwrap(r_hi.as_euler('xyz'), axis=0)
    acc = h0w.apply(d['world_acc'])
    omega = h0w.apply(d['world_omega'])
    # This adapter passes H0 quantities into the existing causal preprocessing;
    # 'world' here is a legacy helper key, not an additional W-frame conversion.
    adapted = dict(d, world_acc=acc, world_omega=omega, imu=d['imu'].copy())
    adapted['imu'][:, 6:9] = rpy
    p = audit.prepare(adapted)
    p['qf'] = audit.lowpass(p['q'][:, :12], 15.)
    p['dqf'] = audit.lowpass(p['dq'][:, :12], 15.)
    p['omega_raw'], _ = audit.asof(d['imu_t'], omega, p['t'])
    p['quaternion_h0_xyzw'], _ = audit.asof(d['imu_t'], r_hi.as_quat(), p['t'])
    p['yaw0_rad'] = np.asarray(yaw0)
    p['reference_available_s'] = np.asarray((reference['monotonic_ns']-int(d['epoch_ns']))*1e-9)
    return p


def extra_quality(d, p, ref, remote):
    primary = (p['t'] >= 5) & (p['t'] < 18)
    remote_active = (remote[:, 1] != 0) | (np.max(np.abs(remote[:, 2:]), axis=1) > .05)
    hit = remote[remote_active & (remote[:, 0] >= 0)]
    q = dict(reference=ref, imu_age_ms=audit.stats(p['age'][primary]*1000),
             lowstate_age_ms=audit.stats(p['qage'][primary]*1000),
             primary_imu_age_over_6ms=int(np.sum(p['age'][primary] > .006)),
             primary_lowstate_age_over_6ms=int(np.sum(p['qage'][primary] > .006)),
             remote_nonzero_primary_samples=int(np.sum(remote_active & (remote[:, 0] >= 5) & (remote[:, 0] < 18))),
             first_remote_active_task_s=float(hit[0, 0]) if len(hit) else None,
             remote_max_abs_axes=np.max(np.abs(remote[(remote[:, 0] >= 5) & (remote[:, 0] < 18), 2:]), axis=0).tolist(),
             imu_h0_yaw_deg=audit.stats(np.rad2deg(p['rpy'][primary, 2])),
             note='Remote threshold .05 is descriptive, not a certified safety gate.')
    for stage, (a, b) in dict(startup=(5, 7), steady=(7, 15), stopping=(15, 18)).items():
        mask = (p['t'] >= a) & (p['t'] < b)
        q[stage] = dict(yaw_deg=audit.stats(np.rad2deg(p['rpy'][mask, 2])),
                        acc_rms=np.sqrt(np.mean(p['acc'][mask]**2, axis=0)).tolist(),
                        acc_raw_rms=np.sqrt(np.mean(p['acc_raw'][mask]**2, axis=0)).tolist())
    q['final_leg_rms_speed_rad_s'] = float(np.sqrt(np.mean(p['dq'][(p['t'] >= 17.5) & (p['t'] < 18), :12]**2)))
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('evaluation/hardware_shadow/commissioning'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = args.out / 'data'
    data.mkdir(parents=True, exist_ok=True)
    protocol = dict(development=DEVELOPMENT, heldout=HELDOUT, supplemental=[6],
        excluded_old_sources=list(audit.TRIALS),
        old_exclusion_reason='Operator withdrew all five: tether exerted external pull; no new fit or evaluation uses them.',
        operator_report='New twelve runs were untethered independent walking. This is not a claim of zero fall risk.',
        primary_window_s=[5, 18], split_unit='complete run, chronological; never shuffled windows',
        selection='whole-episode CV within development only; heldout reserved for final comparison',
        frame='Fixed H0 per run: yaw0 from recorded pre-walk3-5s mean; Z retained from IMU navigation frame.',
        filters='Causal 15Hz one-pole on acc/omega/leg features; alpha = backward derivative of filtered omega then causal15Hz.',
        grid_s=.002, prediction_step_s=.006,
        no_source_timestamps='Host callback times define this replay; no motor/network roundtrip latency measurement.',
        preparer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        auditor_sha256=hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest())
    # Written before fitting/looking at heldout forecast errors.
    audit.write_json(args.out / 'dataset_protocol.json', protocol)
    summaries = []
    for i, suffix in enumerate(TIMES, 1):
        directory = args.root / ('g1_walk_h0_heading18_20260918_' + suffix)
        summary_path = data / f'trial{i:02d}_audit.json'
        # Reuse extraction only when path/stat matches; hash was produced in the
        # original full streaming audit. New prepared transformations always run.
        st = (directory / 'raw.jsonl').stat()
        signature = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get('source_stat') != signature or summary['source'] != str(directory):
                raise ValueError('cached source changed; use a new output directory')
        else:
            summary = audit.audit_trial(directory, data, i)
            summary['source_stat'] = signature
            audit.write_json(summary_path, summary)
        d = dict(np.load(data / f'trial{i:02d}.npz'))
        ref, session, remote = extra_records(directory / 'raw.jsonl', int(d['epoch_ns']))
        if session.get('heading_target') != 'fixed_run_h0_positive_x' or session.get('heading_hold_stop_s') != 18:
            raise ValueError('different frame/protocol; do not silently pool')
        p = prepare_h0(d, ref)
        np.savez_compressed(data / f'trial{i:02d}_prepared.npz', **p)
        np.savez_compressed(data / f'trial{i:02d}_remote.npz', remote=remote)
        summary['h0_quality'] = extra_quality(d, p, ref, remote)
        summary['split'] = 'development' if i in DEVELOPMENT else 'heldout' if i in HELDOUT else 'supplemental'
        summary['exclusion_note'] = 'Missing session_end/drain and final RPC reply: exclude primary benchmark despite covered [5,18).' if i == 6 else None
        audit.write_json(data / f'trial{i:02d}_h0_audit.json', summary)
        summaries.append(summary)
        print('prepared', i, 'yaw0_deg', round(np.rad2deg(ref['yaw0_rad']), 3),
              'primary_remote', summary['h0_quality']['remote_nonzero_primary_samples'], flush=True)
    audit.write_json(args.out / 'data_quality.json', summaries)
    print('Completed offline H0 extraction:', args.out, flush=True)


if __name__ == '__main__':
    main()
