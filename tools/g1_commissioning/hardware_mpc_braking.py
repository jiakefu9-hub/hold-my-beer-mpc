"""Measured-state early braking with a small, persistent recovery action.

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


class LatchedPredictiveBrake:
    """Keep braking until an outward joint velocity has actually stopped.

    The soft cost alone can disappear after one noisy/slower feedback sample,
    allowing the next MPC solve to accelerate outwards again.  This state
    machine adds no abort condition: once the estimated stop enters the
    operating box's early-warning band, it temporarily bounds only the
    *first* MPC acceleration.
    It releases after the joint is back inside that box and no longer moving
    outwards.  A separate outer box supplies recovery room.
    """

    def __init__(self, size=5):
        self.side = np.zeros(int(size), dtype=np.int8)

    def reset(self):
        self.side.fill(0)

    def update(self, q, dq, lower, upper, *, deceleration, reaction_s,
               margin_rad, weight, max_ddq, minimum_deceleration):
        weights, detail = braking_cost(q, dq, lower, upper,
            deceleration=deceleration, reaction_s=reaction_s,
            margin_rad=margin_rad, weight=weight)
        q, dq, lower, upper, max_ddq = [np.asarray(x, dtype=float) for x in
                                        (q, dq, lower, upper, max_ddq)]
        if (max_ddq.shape != self.side.shape or np.any(max_ddq <= 0)
                or not np.isfinite(minimum_deceleration)
                or minimum_deceleration <= 0
                or np.any(minimum_deceleration > max_ddq)):
            raise ValueError('invalid latched braking acceleration')

        # Release only after feedback proves that outward motion has stopped.
        release_upper = (self.side > 0) & (dq <= 0.) & (q <= upper)
        release_lower = (self.side < 0) & (dq >= 0.) & (q >= lower)
        self.side[release_upper | release_lower] = 0

        estimated_stop = np.asarray(detail['estimated_stop_rad'])
        # Latch when the stopping estimate enters the same early-warning band
        # used by the soft cost, rather than waiting until the inner boundary.
        self.side[(self.side == 0) & (dq > 0.)
                  & (estimated_stop >= upper-margin_rad)] = 1
        self.side[(self.side == 0) & (dq < 0.)
                  & (estimated_stop <= lower+margin_rad)] = -1

        acceleration_lower = -max_ddq.copy()
        acceleration_upper = max_ddq.copy()
        acceleration_upper[self.side > 0] = -minimum_deceleration
        acceleration_lower[self.side < 0] = minimum_deceleration
        # Preserve full braking cost while latched. This guides the remaining
        # horizon consistently with the first-action bound.
        activation = np.asarray(detail['activation']).copy()
        activation[self.side != 0] = 1.
        weights = weight*activation
        detail.update(activation=activation, velocity_cost=weights,
            latch_side=self.side.copy(), latched=bool(np.any(self.side)),
            first_acceleration_lower_rad_s2=acceleration_lower,
            first_acceleration_upper_rad_s2=acceleration_upper,
            minimum_braking_deceleration_rad_s2=float(minimum_deceleration),
            adds_hard_constraints=bool(np.any(self.side)),
            hard_constraint_semantics='first MPC action only; no abort gate')
        return weights, (acceleration_lower, acceleration_upper), detail
