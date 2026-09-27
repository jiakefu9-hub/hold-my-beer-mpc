"""Bounded causal H0 disturbance estimates for the opt-in hardware MPC tool.

No SDK or network imports. The lookup is the frozen September offline model,
not an online learner. All physical forecasts are *15 Hz filtered estimates*;
they are never labelled as latency-compensated instantaneous measurements.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from disturbance_types import DisturbanceHorizon, DisturbanceInput

DEFAULT_BANK = ROOT / "assets/g1_hardware_mpc_predictor/bank.npz"
GRID_NS = 2_000_000
CONTROL_DT = .006
GAIN = -math.expm1(-2 * math.pi * 15 * .002)
GRAVITY = np.array([0., 0., -9.81])


@dataclass(frozen=True)
class PredictionResult:
    horizon: DisturbanceHorizon
    diagnostics: dict


def _finite(value, shape, name):
    x = np.asarray(value, dtype=float)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return x.copy()


class FrozenInnovationBank:
    """One fixed tree, one search per control interval; no fitting or disk I/O."""

    def __init__(self, path=DEFAULT_BANK):
        path = Path(path)
        manifest = json.loads(path.with_name("manifest.json").read_text())
        if manifest.get("schema") != "g1_hardware_mpc_frozen_predictor_v1":
            raise ValueError("unsupported predictor manifest")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != manifest.get("bank_sha256"):
            raise ValueError("predictor bank SHA256 mismatch")
        self.manifest = manifest
        with np.load(path, allow_pickle=False) as archive:
            d = {key: archive[key] for key in archive.files}
        n = int(manifest["rows"])
        shapes = dict(mean=(33,), std=(33,), feature_scale=(33,),
                      train_z=(n, 33), future=(n, 9, 12), current=(n, 12),
                      decay=(9, 3))
        for key, shape in shapes.items():
            setattr(self, key, _finite(d[key], shape, key))
        if n < 8 or np.any(self.std <= 0) or np.any(self.feature_scale <= 0):
            raise ValueError("invalid predictor scale or row count")
        if np.any((self.decay < 0) | (self.decay > 1)) or np.any(np.diff(self.decay, axis=0) > 1e-12):
            raise ValueError("invalid innovation decay")
        self.train_trial = np.asarray(d["train_trial"], dtype=np.int64)
        self.train_anchor = np.asarray(d["train_anchor"], dtype=np.int64)
        if self.train_trial.shape != (n,) or self.train_anchor.shape != (n,):
            raise ValueError("invalid model provenance dimensions")
        self.tree = cKDTree(self.train_z)
        # Warm query avoids a first-command initialization surprise.
        self.tree.query(self.train_z[0], k=8, workers=1)

    def predict(self, feature, current_y):
        feature = _finite(feature, (33,), "feature")
        current_y = _finite(current_y, (12,), "current_y")
        z = (feature - self.mean) / self.std * self.feature_scale
        distances, indices = self.tree.query(z, k=8, workers=1)
        weights = 1. / np.maximum(distances, .001)
        weights /= weights.sum()
        absolute = np.einsum("k,kho->ho", weights, self.future[indices])
        neighbor_current = weights @ self.current[indices]
        correction = current_y - neighbor_current
        forecast = absolute.copy()
        forecast[:, :9] += np.repeat(self.decay, 3, axis=1) * correction[None, :9]
        # Exactly matches the research output, but NEVER used as a rotation.
        forecast[:, 9:] += correction[None, 9:]
        return forecast, {
            "neighbor_indices": indices.tolist(), "neighbor_distances": distances.tolist(),
            "neighbor_weights": weights.tolist(),
            "neighbor_training_trial": self.train_trial[indices].tolist(),
            "neighbor_training_anchor": self.train_anchor[indices].tolist(),
            "nearest_standardized_distance": float(distances[0]),
            "maximum_absolute_standardized_feature": float(np.max(np.abs(z))),
            "distance_is_diagnostic_not_validated_confidence": True,
        }


class HardwareMpcPredictor:
    """Thread-safe ingress; single control-thread query on a causal 2 ms grid.

    Observations are immutable copies in bounded past buffers. Filtering is in
    navigation W, then vectors are rotated to the fixed run H0 at query time.
    Linear filtering/differentiation commute with that constant rotation; this
    handles freezing yaw0 without fabricating history or resetting transients.

    ``query`` reports its grid anchor (at most 2 ms behind ``now_ns``), not a
    invented source timestamp. Raw DDS packets have only host receive stamps.
    """

    def __init__(self, mode="learned_filtered", bank_path=DEFAULT_BANK,
                 history_ns=750_000_000, max_stale_ns=100_000_000,
                 max_backlog_ns=50_000_000, max_samples=4096):
        if mode not in {"learned_filtered", "hold_current"}:
            raise ValueError("mode must be learned_filtered or hold_current")
        if not (GRID_NS <= max_backlog_ns <= history_ns <= 2_000_000_000):
            raise ValueError("invalid bounded predictor history/backlog")
        if max_stale_ns <= 0 or max_samples < 2:
            raise ValueError("invalid sample/staleness bounds")
        self.mode = mode
        self.bank = FrozenInnovationBank(bank_path) if mode == "learned_filtered" else None
        self.history_ns, self.max_stale_ns = int(history_ns), int(max_stale_ns)
        self.max_backlog_ns, self.max_samples = int(max_backlog_ns), int(max_samples)
        self._low, self._imu = deque(), deque()
        self._lock, self._query_lock = threading.Lock(), threading.Lock()
        self._origin_ns = 0
        self._anchor_ns = None
        self._filtered = None
        self._alpha = np.zeros(3)
        self._last_rotation = None
        self._low_stamp = self._imu_stamp = None
        self._sample_count = 0

    def set_grid_origin(self, epoch_ns):
        """Optional task epoch alignment; call before the first query only."""
        with self._query_lock:
            if self._anchor_ns is not None:
                raise RuntimeError("cannot shift predictor grid after filtering starts")
            self._origin_ns = int(epoch_ns)

    def _append(self, buffer, record):
        with self._lock:
            if buffer and record[0] < buffer[-1][0]:
                raise ValueError("receive timestamps must be nondecreasing")
            if buffer and record[0] == buffer[-1][0]:
                return False
            buffer.append(record)
            # Retain one sample before the cutoff for causal last-value lookup.
            cutoff = record[0] - self.history_ns
            while len(buffer) > 1 and buffer[1][0] < cutoff:
                buffer.popleft()
            while len(buffer) > self.max_samples:
                buffer.popleft()
        return True

    def observe_low(self, timestamp_ns, q35, dq35):
        q = _finite(q35, (35,), "q35")
        dq = _finite(dq35, (35,), "dq35")
        return self._append(self._low, (int(timestamp_ns), q[:12], dq[:12]))

    def observe_imu(self, timestamp_ns, quat_wxyz, gyro3, accel3):
        q = _finite(quat_wxyz, (4,), "quat_wxyz")
        norm = np.linalg.norm(q)
        if not .5 < norm < 1.5:
            raise ValueError("invalid torso IMU quaternion norm")
        q /= norm
        return self._append(self._imu, (int(timestamp_ns), q,
            _finite(gyro3, (3,), "gyro3"), _finite(accel3, (3,), "accel3")))

    def _grid_floor(self, now):
        return self._origin_ns + (now - self._origin_ns) // GRID_NS * GRID_NS

    def _advance(self, now):
        with self._lock:
            low, imu = list(self._low), list(self._imu)
        if not low or not imu:
            raise RuntimeError("predictor needs both LowState and torso IMU")
        target = self._grid_floor(now)
        if self._anchor_ns is not None and target < self._anchor_ns:
            raise RuntimeError("predictor query time moved backwards")
        if self._anchor_ns is None:
            first = max(low[0][0], imu[0][0], target - self.history_ns)
            first = self._grid_floor(first + GRID_NS - 1)
        else:
            if target - self._anchor_ns > self.max_backlog_ns:
                raise RuntimeError("predictor backlog exceeded; refusing unbounded catch-up")
            first = self._anchor_ns + GRID_NS
        low_times = np.fromiter((v[0] for v in low), dtype=np.int64)
        imu_times = np.fromiter((v[0] for v in imu), dtype=np.int64)
        for stamp in range(first, target + 1, GRID_NS):
            li = int(np.searchsorted(low_times, stamp, side="right")) - 1
            ii = int(np.searchsorted(imu_times, stamp, side="right")) - 1
            if li < 0 or ii < 0:
                raise RuntimeError("no past sample at grid time; refusing future backfill")
            l, m = low[li], imu[ii]
            if stamp - l[0] > self.max_stale_ns or stamp - m[0] > self.max_stale_ns:
                raise RuntimeError("predictor observation stale on causal grid")
            rotation = Rotation.from_quat(m[1][[1, 2, 3, 0]]).as_matrix()
            acc, omega = rotation @ m[3] + GRAVITY, rotation @ m[2]
            value = np.concatenate((acc, omega, l[1], l[2]))
            if self._filtered is None:
                self._filtered = value.copy()
            else:
                previous_omega = self._filtered[3:6].copy()
                self._filtered += GAIN * (value - self._filtered)
                derivative = (self._filtered[3:6] - previous_omega) / .002
                self._alpha += GAIN * (derivative - self._alpha)
            self._anchor_ns = stamp
            self._last_rotation = rotation
            self._low_stamp, self._imu_stamp = l[0], m[0]
            self._sample_count += 1
        if self._anchor_ns is None:
            raise RuntimeError("no common past data yet at the predictor grid")
        if now - self._low_stamp > self.max_stale_ns or now - self._imu_stamp > self.max_stale_ns:
            raise RuntimeError("predictor observation stale at query")

    def query(self, now_ns, yaw0_rad, use_learned=True):
        started = time.perf_counter_ns()
        now_ns, yaw0_rad = int(now_ns), float(yaw0_rad)
        if not math.isfinite(yaw0_rad):
            raise ValueError("H0 yaw reference must be finite")
        with self._query_lock:
            self._advance(now_ns)
            c, s = math.cos(yaw0_rad), math.sin(yaw0_rad)
            h0w = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
            rotation = h0w @ self._last_rotation
            physical = np.concatenate((h0w @ self._filtered[:3],
                                       h0w @ self._filtered[3:6], h0w @ self._alpha))
            rpy = Rotation.from_matrix(rotation).as_euler("xyz")
            current = np.concatenate((physical, rpy))
            feature = np.concatenate((self._filtered[6:18], self._filtered[18:30], physical))
            if self.bank is None or not use_learned:
                forecast, lookup_diagnostics = np.repeat(current[None, :], 9, axis=0), {}
            else:
                forecast, lookup_diagnostics = self.bank.predict(feature, current)
            if not np.isfinite(forecast).all():
                raise RuntimeError("nonfinite disturbance forecast")
            omega_nodes = np.vstack((physical[3:6], forecast[:, 3:6]))
            rotations = [rotation.copy()]
            intervals = []
            mean_omegas = .5*(omega_nodes[:-1] + omega_nodes[1:])
            deltas = Rotation.from_rotvec(mean_omegas*CONTROL_DT).as_matrix()
            half_deltas = Rotation.from_rotvec(mean_omegas*CONTROL_DT/2).as_matrix()
            for k in range(9):
                mean_omega = mean_omegas[k]
                midpoint = half_deltas[k] @ rotations[-1]
                rotations.append(deltas[k] @ rotations[-1])
                intervals.append(DisturbanceInput(forecast[k, :3].copy(), mean_omega.copy(),
                    forecast[k, 6:9].copy(), midpoint))
            nodes = [DisturbanceInput(physical[:3].copy(), physical[3:6].copy(),
                                      physical[6:9].copy(), rotations[0])]
            # MPC uses node omega/R and interval acc/alpha. Nonzero-node acc and
            # alpha below are representative interval estimates, NOT extra
            # independently trained instantaneous targets.
            nodes.extend(DisturbanceInput(forecast[k, :3].copy(), omega_nodes[k + 1].copy(),
                         forecast[k, 6:9].copy(), rotations[k + 1]) for k in range(9))
            diagnostics = dict(mode=self.mode if use_learned else "hold_current", frame="fixed_H0", origin="torso_IMU",
                anchor_monotonic_ns=self._anchor_ns, query_monotonic_ns=now_ns,
                query_minus_anchor_ms=(now_ns - self._anchor_ns) * 1e-6,
                lowstate_age_ms=(now_ns - self._low_stamp) * 1e-6,
                imu_age_ms=(now_ns - self._imu_stamp) * 1e-6,
                filtered_grid_samples=self._sample_count, filter_hz=15., grid_ms=2.,
                signal_contract="causal_filtered_estimates_not_raw_or_delay_compensated",
                features=feature.tolist(), current_y=current.tolist(),
                forecast_y=forecast.tolist(), horizons_ms=list(range(6, 55, 6)),
                future_node_monotonic_ns=[self._anchor_ns + int(h * 6_000_000) for h in range(10)],
                interval_left_monotonic_ns=[self._anchor_ns + int(h * 6_000_000) for h in range(9)],
                rotation_policy="measured_node0_plus_world_left_trapezoidal_omega_SO3_integration",
                diagnostic_rpy_not_used_by_controller=True,
                bank_sha256=None if self.bank is None else self.bank.manifest["bank_sha256"],
                **lookup_diagnostics)
            diagnostics["predictor_total_ms"] = (time.perf_counter_ns() - started) * 1e-6
            return PredictionResult(DisturbanceHorizon(tuple(nodes), tuple(intervals)), diagnostics)
