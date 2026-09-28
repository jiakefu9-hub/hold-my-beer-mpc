"""Bounded local DDQ->total-torque candidate search using a supplied model.

Matches the simulation's perturb/DLS/rank/forward-check structure. 'Accepted'
always means accepted by the supplied model, never certified physical safety.
No SDK, timers, thread, or publisher is created here.
"""
from __future__ import annotations

import time
import numpy as np

from hardware_arm_inverse_dynamics import finite_vector


class NoModelTorque(RuntimeError):
    def __init__(self, trace):
        super().__init__("no candidate satisfies the current forward-model envelope")
        self.trace = trace


class LocalTorqueMapper:
    def __init__(self, config):
        self.config = config
        self.limit = finite_vector(config["tau_abs_nm"], 5, "torque limits")
        self.epsilon = float(config["perturbation_nm"])
        self.reg = float(config["regularization"])
        self.scales = np.asarray(config["candidate_scales"], dtype=float)
        self.error_limit = float(config["max_joint_error_rad_s2"])
        self.acc_limit = float(config["max_abs_qacc_rad_s2"])
        self.second_threshold = float(config["second_pass_error_rad_s2"])
        self.rescue_passes = int(config["rescue_passes"])
        scalars = [self.epsilon, self.reg, self.error_limit, self.acc_limit, self.second_threshold]
        if (not np.isfinite(scalars).all() or min(scalars) <= 0
                or np.any(self.limit <= 0) or self.scales.ndim != 1
                or not 2 <= self.scales.size <= 8 or not np.isfinite(self.scales).all()
                or np.any((self.scales <= 0) | (self.scales > 1))
                or not 0 <= self.rescue_passes <= 2):
            raise ValueError("invalid bounded mapper configuration")

    def compute(self, forward, desired, nominal, safe_hold, previous=None, bounds=None):
        started = time.perf_counter_ns()
        desired = finite_vector(desired, 5, "desired acceleration")
        lower, upper = -self.limit, self.limit
        if bounds is not None:
            lower = np.maximum(lower, finite_vector(bounds[0], 5, "torque lower"))
            upper = np.minimum(upper, finite_vector(bounds[1], 5, "torque upper"))
        if np.any(lower > upper):
            raise ValueError("empty torque envelope")
        clip = lambda value: np.clip(finite_vector(value, 5, "torque"), lower, upper)
        trace = {"acceptance": "forward_model_only", "hardware_certified": False,
                 "passes": [], "fallback": None, "forward_calls": 0}

        def evaluate(torque):
            trace["forward_calls"] += 1
            acceleration = finite_vector(forward(torque), 5, "model acceleration")
            error = acceleration-desired
            return acceleration, float(np.linalg.norm(error)), float(np.max(np.abs(error)))

        def safe(acceleration):
            return bool(np.max(np.abs(acceleration)) <= self.acc_limit+1e-9)

        best_tau = clip(nominal)
        best_acc, best_error, best_max = evaluate(best_tau)
        trace["nominal_tau_nm"] = best_tau.tolist()
        trace["nominal_ddq_rad_s2"] = best_acc.tolist()
        trace["nominal_error_norm"] = best_error
        # First pass; optional second pass; bounded acceleration-limit rescue.
        for pass_index in range(2+self.rescue_passes):
            if pass_index == 1 and safe(best_acc) and best_max <= self.error_limit and best_error <= self.second_threshold:
                break
            if pass_index >= 2 and safe(best_acc):
                break
            base_tau, base_acc = best_tau.copy(), best_acc.copy()
            gain = np.zeros((5, 5))
            for j in range(5):
                delta = min(self.epsilon, upper[j]-base_tau[j])
                if delta <= 1e-12:
                    delta = -min(self.epsilon, base_tau[j]-lower[j])
                if abs(delta) > 1e-12:
                    perturbed = base_tau.copy(); perturbed[j] += delta
                    gain[:, j] = (evaluate(perturbed)[0]-base_acc)/delta
            u, singular, vt = np.linalg.svd(gain)
            # Same lambda convention as the simulation mapper: s/(s²+lambda).
            correction = vt.T @ ((singular/(singular**2+self.reg)) * (u.T @ (desired-base_acc)))
            candidates = []
            for scale in self.scales:
                torque = clip(base_tau+scale*correction)
                prediction = base_acc+gain@(torque-base_tau)
                error = prediction-desired
                candidates.append({"scale": float(scale), "tau_nm": torque,
                    "predicted_ddq_rad_s2": prediction,
                    "rank": (not (safe(prediction) and np.max(np.abs(error)) <= self.error_limit),
                             float(np.linalg.norm(error)))})
            candidates.sort(key=lambda row: row["rank"])
            record = {"gain_rad_s2_per_nm": gain.tolist(), "candidates": [], "selected_scale": 0.}
            # Include an already-good nominal in the selection; never worsen it
            # simply to claim that a nonzero correction was applied.
            evaluated = [(best_tau, best_acc, best_error, best_max, 0.)]
            strict_count = 0
            for candidate in candidates:
                acc, error, maximum = evaluate(candidate["tau_nm"])
                improves = error < best_error-1e-12
                strict = improves and safe(acc) and maximum <= self.error_limit
                strict_count += int(strict)
                evaluated.append((candidate["tau_nm"], acc, error, maximum, candidate["scale"]))
                record["candidates"].append({"scale": candidate["scale"],
                    "tau_nm": candidate["tau_nm"].tolist(),
                    "predicted_ddq_rad_s2": candidate["predicted_ddq_rad_s2"].tolist(),
                    "checked_ddq_rad_s2": acc.tolist(), "error_norm": error,
                    "max_joint_error": maximum, "model_acceleration_within_limit": safe(acc),
                    "improves": improves, "strict": strict})
                if len(record["candidates"]) >= 2 and strict_count:
                    break
            progress = [row for row in evaluated if row[2] <= best_error+1e-12]
            strict = [row for row in progress if safe(row[1]) and row[3] <= self.error_limit]
            bounded = [row for row in progress if safe(row[1])]
            selected = (min(strict, key=lambda row: row[2]) if strict else
                        min(bounded, key=lambda row: (row[3], row[2])) if bounded else
                        min(progress, key=lambda row: row[2]))
            best_tau, best_acc, best_error, best_max, record["selected_scale"] = selected
            trace["passes"].append(record)

        if not safe(best_acc):
            # Previous is re-evaluated at THIS state. Never assume last tick's
            # acceptance remains valid. Fallbacks need not improve tracking.
            if previous is not None:
                previous_tau = clip(previous)
                acceleration, error, maximum = evaluate(previous_tau)
                if safe(acceleration):
                    best_tau, best_acc, best_error, best_max = previous_tau, acceleration, error, maximum
                    trace["fallback"] = "previous_rechecked"
            if not safe(best_acc):
                hold_tau = clip(safe_hold)
                acceleration, error, maximum = evaluate(hold_tau)
                if not safe(acceleration):
                    trace["model_accepted"] = False
                    trace["elapsed_ms"] = (time.perf_counter_ns()-started)*1e-6
                    raise NoModelTorque(trace)
                selected = (hold_tau, acceleration, error, maximum)
                trace["fallback"] = "hold_rechecked"
                for scale in (.8, .6, .4, .2):
                    torque = clip(hold_tau+scale*(best_tau-hold_tau))
                    acceleration, error, maximum = evaluate(torque)
                    if safe(acceleration):
                        selected = torque, acceleration, error, maximum
                        trace["fallback"] = "line_search_to_hold"
                        break
                best_tau, best_acc, best_error, best_max = selected
        trace.update(tau_total_nm=best_tau.tolist(), checked_ddq_rad_s2=best_acc.tolist(),
                     error_norm=best_error, max_joint_error=best_max,
                     tracking_within_limit=best_max <= self.error_limit,
                     model_accepted=True, elapsed_ms=(time.perf_counter_ns()-started)*1e-6)
        return best_tau.copy(), trace
