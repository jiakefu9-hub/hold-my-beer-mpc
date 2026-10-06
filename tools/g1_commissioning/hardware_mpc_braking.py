"""Soft, measured-state early braking. No new feasibility/abort condition.

The stopping estimate is a tuning model, NOT a certified physical stopping
distance. The real actuator gain, delay and friction remain uncalibrated.
"""
import numpy as np


def braking_cost(q, dq, lower, upper, *, deceleration, reaction_s, margin_rad, weight):
    q, dq, lower, upper = [np.asarray(x, dtype=float) for x in (q, dq, lower, upper)]
    if (any(x.shape != (5,) or not np.all(np.isfinite(x)) for x in (q,dq,lower,upper))
            or np.any(lower >= upper) or not np.isfinite(deceleration) or deceleration <= 0
            or not np.isfinite(reaction_s) or reaction_s < 0
            or not np.isfinite(margin_rad) or margin_rad <= 0
            or np.any(2*margin_rad >= upper-lower)
            or not np.isfinite(weight) or weight < 0):
        raise ValueError('invalid predictive braking inputs')
    speed = np.abs(dq)
    travel = speed*reaction_s + speed**2/(2*deceleration)
    gap = np.where(dq >= 0, upper-q, q-lower)
    # Cost starts when the estimated stop enters the soft margin, and reaches
    # full weight at the OUTER boundary. Smoothstep avoids an on/off switch.
    fraction = np.clip((travel-(gap-margin_rad))/margin_rad, 0., 1.)
    activation = fraction**2*(3-2*fraction)
    activation = np.where(speed > 1e-12, activation, 0.)
    return weight*activation, dict(activation=activation, velocity_cost=weight*activation,
        stopping_travel_rad=travel, distance_to_outer_rad=gap,
        estimated_stop_rad=q+np.sign(dq)*travel,
        assumed_deceleration_rad_s2=deceleration, reaction_allowance_s=reaction_s,
        soft_margin_rad=margin_rad, adds_hard_constraints=False,
        physical_stopping_distance_certified=False)
