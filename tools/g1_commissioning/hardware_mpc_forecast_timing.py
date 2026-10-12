"""Opt-in acceleration preview timing; no actuator/state delay model.

Keep measured node zero, joint feedback and all omega/attitude nodes unchanged.
Only resample future interval acc/alpha to the estimated host-write windows.
The estimate uses the current query/grid timestamps and a frozen 3.5 ms host
budget, measured on normal field captures. No future measurement is an input.
This is a feedforward approximation, NOT full command-time state prediction.
"""
from __future__ import annotations

import numpy as np

from hardware_mpc_predictor import PredictionResult
from disturbance_types import DisturbanceHorizon, DisturbanceInput

QUERY_TO_WRITE_MS = 3.5
INTERVAL_MS = 6.
HORIZON_INTERVALS = 9


def shifted_interval_means(values, lead_ms):
    """Integrate piecewise-constant 6 ms means; explicitly hold the final mean.

    Works on (..., 9, channels) arrays for runtime and batched offline checks.
    Fractional shifts mix overlapping intervals, never interpolate a mean as
    if it were an independently learned instantaneous measurement.
    """
    values = np.asarray(values, dtype=float)
    lead_ms = float(lead_ms)
    if (values.ndim < 2 or values.shape[-2] != HORIZON_INTERVALS
            or not np.isfinite(values).all()
            or not np.isfinite(lead_ms) or not 0. <= lead_ms <= 12.):
        raise ValueError('finite nine-interval forecast and lead within 0..12 ms required')
    if lead_ms == 0.:
        return values.copy()
    offset = lead_ms / INTERVAL_MS
    whole = int(np.floor(offset))
    fraction = offset - whole
    index = np.minimum(np.arange(HORIZON_INTERVALS) + whole, HORIZON_INTERVALS - 1)
    following = np.minimum(index + 1, HORIZON_INTERVALS - 1)
    return (1. - fraction) * values[..., index, :] + fraction * values[..., following, :]


def alignment_metadata():
    return dict(enabled=True, version='acceleration_interval_write_alignment_v1',
        query_to_write_budget_ms=QUERY_TO_WRITE_MS,
        budget_source='normal field host timestamps; not identified motor or DDS latency',
        fields=['interval_acc_world', 'interval_alpha_world'],
        unchanged=['measured_q_dq', 'node_zero', 'node_omega', 'node_attitude',
                   'interval_omega', 'interval_attitude', 'current_inverse_dynamics_bias'],
        tail_policy='hold final trained interval mean; no extra learned horizon',
        state_propagated=False, hardware_delay_identified=False,
        physical_performance_validated=False)


def align_acceleration_preview(result: PredictionResult) -> PredictionResult:
    """Advance only a learned horizon, after its causal query has completed."""
    diag = result.diagnostics
    if diag.get('mode') != 'learned_filtered':
        return result
    anchor = int(diag['anchor_monotonic_ns'])
    query = int(diag['query_monotonic_ns'])
    grid_age_ms = (query - anchor) * 1e-6
    if not 0. <= grid_age_ms < 2.:
        raise ValueError('forecast alignment requires a causal anchor within the 2 ms grid')
    horizon = result.horizon
    if len(horizon.nodes) != 10 or len(horizon.intervals) != HORIZON_INTERVALS:
        raise ValueError('forecast alignment requires the frozen nine-interval contract')
    lead_ms = grid_age_ms + QUERY_TO_WRITE_MS
    values = np.asarray([np.r_[d.acc_world, d.alpha_world] for d in horizon.intervals])
    shifted = shifted_interval_means(values, lead_ms)
    intervals = tuple(DisturbanceInput(v[:3].copy(), d.omega_world.copy(), v[3:].copy(),
                                      d.rot_world_body.copy())
                      for d, v in zip(horizon.intervals, shifted))
    write_ns = query + round(QUERY_TO_WRITE_MS * 1e6)
    evidence = dict(alignment_metadata(), applied=True, lead_from_anchor_ms=lead_ms,
        estimated_host_write_monotonic_ns=write_ns,
        acceleration_interval_left_monotonic_ns=[write_ns + k * 6_000_000 for k in range(9)],
        acc_interval_h0_m_s2=shifted[:, :3].tolist(),
        alpha_interval_h0_rad_s2=shifted[:, 3:].tolist(),
        extrapolated_tail_ms=lead_ms)
    # Preserve the original forecast and its timestamps verbatim for A/B audit.
    diagnostics = dict(diag, acceleration_forecast_alignment=evidence)
    return PredictionResult(DisturbanceHorizon(horizon.nodes, intervals), diagnostics)
