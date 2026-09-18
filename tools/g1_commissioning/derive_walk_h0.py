#!/usr/bin/env python3
"""Derive fixed-H0 IMU arrays from one g1_walk_capture raw journal.

The input raw.jsonl remains the authoritative, unmodified record.  H0 has the
same vertical axis as the IMU navigation frame W; its +X axis is yaw0 recorded
by the heading_reference_frozen event.  Every torso and pelvis IMU sample,
including samples before yaw0 was frozen, is transformed in this offline pass.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

GRAVITY_H0_M_S2 = np.array([0.0, 0.0, -9.81])


def _normalise(q):
    q = np.asarray(q, dtype=float)
    norms = np.linalg.norm(q, axis=1)
    if np.any(~np.isfinite(q)) or np.any((norms < 0.5) | (norms > 1.5)):
        raise ValueError("invalid IMU quaternion")
    return q / norms[:, None]


def _multiply(left, right):
    """Hamilton product of wxyz quaternions; left may be one quaternion."""
    left = np.broadcast_to(np.asarray(left, dtype=float), np.asarray(right).shape)
    right = np.asarray(right, dtype=float)
    lw, lx, ly, lz = left.T
    rw, rx, ry, rz = right.T
    return np.column_stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ))


def _rotate(q, vector):
    """Rotate local vector into q's parent frame for wxyz q_parent_from_local."""
    q = _normalise(q)
    vector = np.asarray(vector, dtype=float)
    xyz = q[:, 1:]
    first = np.cross(xyz, vector)
    return vector + 2.0 * (q[:, :1] * first + np.cross(xyz, first))


def _rpy(q):
    q = _normalise(q)
    w, x, y, z = q.T
    return np.column_stack((
        np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
        np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ))


def _imu_arrays(rows, prefix, epoch_ns, yaw0):
    if not rows:
        raise ValueError(f"no {prefix} IMU samples")
    received = np.asarray([r["received_monotonic_ns"] for r in rows], dtype=np.int64)
    sequence = np.asarray([r["host_callback_sequence"] for r in rows], dtype=np.uint64)
    q_wi = _normalise([r["quaternion_wxyz"] for r in rows])
    gyro_i = np.asarray([r["gyroscope_rad_s"] for r in rows], dtype=float)
    accel_i = np.asarray([r["accelerometer_raw_m_s2"] for r in rows], dtype=float)
    temperature = np.asarray([r["temperature_raw"] for r in rows], dtype=float)
    half = -0.5 * yaw0
    q_h0w = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    q_h0i = _normalise(_multiply(q_h0w, q_wi))
    specific_force = _rotate(q_h0i, accel_i)
    result = {
        f"{prefix}_received_monotonic_ns": received,
        f"{prefix}_task_elapsed_s": (received - epoch_ns) * 1e-9,
        f"{prefix}_host_callback_sequence": sequence,
        f"{prefix}_quaternion_h0_from_imu_wxyz": q_h0i,
        f"{prefix}_rpy_h0_rad": _rpy(q_h0i),
        f"{prefix}_angular_velocity_h0_rad_s": _rotate(q_h0i, gyro_i),
        f"{prefix}_specific_force_h0_m_s2": specific_force,
        f"{prefix}_linear_acceleration_h0_m_s2": specific_force + GRAVITY_H0_M_S2,
        f"{prefix}_temperature_raw": temperature,
    }
    return result


def derive(raw_path):
    torso, pelvis, references, epochs = [], [], [], []
    digest = hashlib.sha256()
    with raw_path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            digest.update(line)
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc
            if row.get("schema") == "g1_torso_imu_raw_v1":
                torso.append(row)
            elif row.get("schema") == "g1_lowstate_raw_v1":
                pelvis.append(dict(received_monotonic_ns=row["received_monotonic_ns"],
                                   host_callback_sequence=row["host_callback_sequence"],
                                   **row["pelvis_imu"]))
            elif row.get("event") == "heading_reference_frozen":
                references.append(row)
            elif row.get("event") == "task_epoch":
                epochs.append(row["task_epoch_monotonic_ns"])
    if len(references) != 1:
        raise ValueError(f"expected exactly one heading_reference_frozen event, got {len(references)}")
    if len(epochs) != 1:
        raise ValueError(f"expected exactly one task_epoch event, got {len(epochs)}")
    yaw0 = float(references[0]["yaw0_rad"])
    if not np.isfinite(yaw0):
        raise ValueError("non-finite yaw0")
    arrays = {"yaw0_rad": np.asarray(yaw0),
              "rotation_h0_from_navigation_world_rad": np.asarray(-yaw0)}
    arrays.update(_imu_arrays(torso, "torso", epochs[0], yaw0))
    arrays.update(_imu_arrays(pelvis, "pelvis", epochs[0], yaw0))
    manifest = {
        "schema": "g1_walk_h0_derived_v1",
        "source_raw_jsonl": str(raw_path.resolve()),
        "source_sha256": digest.hexdigest(),
        "yaw0_rad": yaw0,
        "h0_definition": references[0]["h0_definition"],
        "reference_sample_count": references[0]["reference_sample_count"],
        "reference_observed_span_s": references[0]["reference_observed_span_s"],
        "reference_observed_interval_s": [references[0]["reference_first_sample_task_s"],
                                          references[0]["reference_last_sample_task_s"]],
        "reference_requested_interval_s": [references[0]["requested_reference_start_s"],
                                            references[0]["requested_reference_end_s"]],
        "torso_samples": len(torso),
        "pelvis_samples": len(pelvis),
        "transform": {
            "orientation": "q_H0_from_IMU = q_yaw(-yaw0) * q_W_from_IMU",
            "vectors": "v_H0 = R_H0_from_IMU * v_IMU",
            "specific_force": "rotated accelerometer_raw_m_s2; gravity retained",
            "linear_acceleration": "specific_force_h0 + [0,0,-9.81] m/s^2",
        },
        "raw_preserved": True,
        "note": "H0 is fixed for the run; it is not a continuously body-following frame.",
    }
    return arrays, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for h0_imu.npz and manifest.json")
    args = parser.parse_args()
    if not args.raw_jsonl.is_file():
        raise SystemExit(f"input does not exist: {args.raw_jsonl}")
    arrays, manifest = derive(args.raw_jsonl)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(f"output directory must be new: {args.output_dir}") from exc
    np.savez_compressed(args.output_dir / "h0_imu.npz", **arrays)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"wrote {args.output_dir}: {manifest['torso_samples']} torso and "
          f"{manifest['pelvis_samples']} pelvis samples, yaw0={manifest['yaw0_rad']:.9f} rad")


if __name__ == "__main__":
    main()
