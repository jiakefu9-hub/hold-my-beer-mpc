#!/usr/bin/env python3
"""Offline bottle-center FK from A3 joint logs and concurrent torso-IMU logs.

Uses the active simulation XML, not a copied arm transform. No DDS/network.
W is the torso IMU navigation frame; B is imu_in_torso; E is grasp_site.
Position is relative to B's origin (expressed in B or W), never absolute W.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
JOINTS = [
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
] + ["waist_yaw_joint"]


def rotation(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not 0.5 < np.linalg.norm(q) < 1.5:
        raise ValueError("invalid IMU quaternion")
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, q / np.linalg.norm(q))
    return matrix.reshape(3, 3)


def orientation(matrix):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(matrix).reshape(9))
    # R = Rz(yaw) Ry(pitch) Rx(roll), same wxyz/body-to-world convention.
    rpy = np.rad2deg([
        np.arctan2(matrix[2, 1], matrix[2, 2]),
        np.arctan2(-matrix[2, 0], np.hypot(matrix[0, 0], matrix[1, 0])),
        np.arctan2(matrix[1, 0], matrix[0, 0]),
    ])
    return {"quaternion_wxyz": quat.tolist(), "rpy_deg": rpy.tolist()}


class EndpointModel:
    def __init__(self, config=ROOT / "configs/g1.yaml"):
        self.config = Path(config).resolve()
        xml = Path(yaml.safe_load(self.config.read_text())["xml_path"])
        self.xml = xml if xml.is_absolute() else ROOT / xml
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.data = mujoco.MjData(self.model)
        self.joint_addresses = [self.model.joint(name).qposadr[0] for name in JOINTS]
        self.imu_id = self.model.site("imu_in_torso").id
        self.sites = {side: self.model.site(f"{side}_grasp_site").id
                      for side in ("left", "right")}
        self.right_dofs = [self.model.joint(name).dofadr[0] for name in JOINTS[5:10]]
        self._jac_rotation = np.zeros((3, self.model.nv))

    def right_gravity_error_and_jacobian(self, arm_slots, world_from_body):
        """One FK and analytic site Jacobian, with the measured torso fixed.

        e = (R_WE.T g_W)[:2]; de/dq = (g_E cross J_omega_E)[:2].
        A fixed H0 yaw rotation leaves gravity and this error unchanged.
        """
        q = np.asarray(arm_slots, dtype=float)
        R_WB = np.asarray(world_from_body, dtype=float)
        if q.shape != (13,) or not np.isfinite(q).all() or R_WB.shape != (3, 3) or not np.isfinite(R_WB).all():
            raise ValueError("invalid arm slots or torso rotation")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.joint_addresses] = q[:11]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, None, self._jac_rotation, self.sites["right"])
        R_GB = self.data.site_xmat[self.imu_id].reshape(3, 3)
        R_GE = self.data.site_xmat[self.sites["right"]].reshape(3, 3)
        R_BE = R_GB.T @ R_GE
        gravity_e = R_BE.T @ R_WB.T @ np.array([0.0, 0.0, -9.81])
        axes_e = R_GE.T @ self._jac_rotation[:, self.right_dofs]
        jacobian = np.cross(gravity_e, axes_e.T).T[:2]
        return gravity_e[:2].copy(), jacobian

    def relative(self, arm_slots):
        q = np.asarray(arm_slots, dtype=float)
        if q.shape != (13,) or not np.isfinite(q).all():
            raise ValueError("expected finite 13-slot Arm5 feedback")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.joint_addresses] = q[:11]
        mujoco.mj_kinematics(self.model, self.data)
        p_B = self.data.site_xpos[self.imu_id].copy()
        R_B = self.data.site_xmat[self.imu_id].reshape(3, 3).copy()
        return {
            side: (R_B.T @ (self.data.site_xpos[site] - p_B),
                   R_B.T @ self.data.site_xmat[site].reshape(3, 3))
            for side, site in self.sites.items()
        }

    def poses(self, arm_slots, imu_quaternion=None):
        R_WB = None if imu_quaternion is None else rotation(imu_quaternion)
        result = {}
        for side, (p_BE, R_BE) in self.relative(arm_slots).items():
            pose = {"position_from_torso_imu_B_m": p_BE.tolist(),
                    "orientation_B": orientation(R_BE),
                    "bottle_z_tilt_from_body_z_deg": float(np.rad2deg(
                        np.arccos(np.clip(R_BE[2, 2], -1, 1)))),
                    "orientation_W": None,
                    "position_from_torso_imu_W_m": None,
                    "bottle_z_tilt_from_vertical_deg": None}
            if R_WB is not None:
                R_WE = R_WB @ R_BE
                pose.update(
                    orientation_W=orientation(R_WE),
                    position_from_torso_imu_W_m=(R_WB @ p_BE).tolist(),
                    bottle_z_tilt_from_vertical_deg=float(np.rad2deg(
                        np.arccos(np.clip(R_WE[2, 2], -1, 1)))),
                )
            result[side] = pose
        return result

    def xml_hashes(self):
        hashes = {}
        def visit(path):
            path = path.resolve()
            if str(path) in hashes:
                return
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for node in ET.parse(path).iter("include"):
                visit(path.parent / node.attrib["file"])
        visit(self.xml)
        return hashes


def records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def nearest_imu(imus, times, timestamp_ns, max_skew_ms):
    index = int(np.searchsorted(times, timestamp_ns))
    candidates = [j for j in (index - 1, index) if 0 <= j < len(imus)]
    if not candidates:
        return None, None
    index = min(candidates, key=lambda j: abs(int(times[j]) - timestamp_ns))
    skew_ms = (int(times[index]) - timestamp_ns) / 1e6
    return (imus[index] if abs(skew_ms) <= max_skew_ms else None), skew_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True, type=Path)
    parser.add_argument("--imu", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/g1.yaml")
    parser.add_argument("--max-skew-ms", type=float, default=125.0)
    args = parser.parse_args()
    if not 0 < args.max_skew_ms <= 125.0:
        parser.error("max-skew-ms must be in (0,125]; this is static-pose analysis")
    model = EndpointModel(args.config)
    frames = [r for r in records(args.execute) if r.get("event") == "dds_write"
              and "q_measured" in r]
    imus = sorted([r for r in records(args.imu)
                   if r.get("schema") == "g1_imu_zero_observation_v1"],
                  key=lambda r: r["monotonic_ns"])
    if not frames or not imus:
        parser.error("requires real execute frames and concurrent IMU observations")
    times = np.array([r["monotonic_ns"] for r in imus], dtype=np.int64)
    rows = []
    for frame in frames:
        imu, skew_ms = nearest_imu(imus, times, frame["captured_monotonic_ns"], args.max_skew_ms)
        rows.append({"schema": "g1_endpoint_pose_v1",
                     "elapsed_s": frame["elapsed_s"], "phase": frame["phase"],
                     "weight": frame["weight"],
                     "state_monotonic_ns": frame["captured_monotonic_ns"],
                     "imu_monotonic_ns": None if imu is None else imu["monotonic_ns"],
                     "imu_minus_state_ms": skew_ms, "world_pose_valid": imu is not None,
                     "q_measured_rad": frame["q_measured"],
                     **model.poses(frame["q_measured"],
                                   None if imu is None else imu["quaternion_wxyz"])})
    meta = {"schema": "g1_endpoint_pose_metadata_v1", "read_only_offline": True,
            "base": "imu_in_torso", "endpoints": ["left_grasp_site", "right_grasp_site"],
            "xml_sha256": model.xml_hashes(), "max_skew_ms": args.max_skew_ms,
            "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (args.execute, args.imu)},
            "world_frame": "rt/secondary_imu local navigation frame",
            "body_frame": "imu_in_torso, rigidly attached to torso; body Z is not gravity vertical",
            "position_origin": "torso IMU; absolute world translation is unavailable",
            "limitations": "model-derived pose, not optical measurement; receive-time nearest match, not hardware synchronization; no acceleration estimate"}
    with args.output.open("x") as stream:
        for row in [meta, *rows]:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    hold = [r for r in rows if r["phase"] == "hold" and r["weight"] == 1.0]
    tail = [r for r in hold if r["elapsed_s"] >= hold[-1]["elapsed_s"] - 2.0] if hold else []
    world_tail = [r for r in tail if r["world_pose_valid"]]
    summary = {"frames": len(rows), "valid_world_frames": sum(r["world_pose_valid"] for r in rows),
               "hold_tail_samples": len(tail), "hold_tail_window_s": 2.0}
    for side in ("left", "right"):
        if tail:
            angles = np.deg2rad([r[side]["orientation_B"]["rpy_deg"] for r in tail])
            summary[side] = {
                "body_rpy_circular_mean_deg": np.rad2deg(np.arctan2(
                    np.sin(angles).mean(axis=0), np.cos(angles).mean(axis=0))).tolist(),
                "bottle_z_tilt_from_body_z_mean_deg": float(np.mean([
                    r[side]["bottle_z_tilt_from_body_z_deg"] for r in tail])),
                "position_from_torso_imu_B_mean_m": np.mean([
                    r[side]["position_from_torso_imu_B_m"] for r in tail], axis=0).tolist(),
            }
            if world_tail:
                angles = np.deg2rad([r[side]["orientation_W"]["rpy_deg"] for r in world_tail])
                summary[side].update(
                    world_rpy_circular_mean_deg=np.rad2deg(np.arctan2(
                        np.sin(angles).mean(axis=0), np.cos(angles).mean(axis=0))).tolist(),
                    bottle_z_tilt_from_world_vertical_mean_deg=float(np.mean([
                        r[side]["bottle_z_tilt_from_vertical_deg"] for r in world_tail])),
                )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
