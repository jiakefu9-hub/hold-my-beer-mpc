"""Hardware-preview-only position/velocity envelope; no output authority.

The original position, velocity and acceleration rows remain in the QP. Each
predicted node additionally obeys ``L + guard <= q + dq/rate <= U - guard``.
This limits outward speed before the original recovery logic reaches the outer
box. It is a nominal-model constraint, not a certified disturbance/delay bound.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from hardware_mpc_solver import CondensedArmMPCPolicy


class RecoveryEnvelopeMpcPolicy(CondensedArmMPCPolicy):
    """Append a fixed linear braking envelope without changing the shared MPC.

    ``recovery_rate_s_inv`` is scalar; ``recovery_guard_rad`` is scalar or five
    per-joint values. The default 0.15 degree reserve is an offline design value,
    not an identified hardware uncertainty bound. The envelope is anchored to
    the outer limits, so it permits recovery from outside the inner operating
    box without requiring an instantaneous reversal of measured velocity.

    For the sampled double integrator, ``a = -rate*v/(1 + rate*dt/2)`` keeps
    ``q + v/rate`` constant and decays velocity. Limit the rate so that this
    nominal braking witness uses at most 80% of the configured acceleration.
    This witness does not prove robustness to the actual torque mapper, delay
    or plant mismatch, nor feasibility from arbitrary initial states.
    """

    def __init__(self, *args, recovery_rate_s_inv=6.0,
                 recovery_guard_rad=np.deg2rad(0.15), **kwargs):
        self.recovery_rate_s_inv = float(recovery_rate_s_inv)
        guard = np.asarray(recovery_guard_rad, dtype=np.float64)
        if guard.ndim == 0:
            guard = np.full(5, float(guard))
        if (guard.shape != (5,) or not np.all(np.isfinite(guard))
                or np.any(guard < 0)):
            raise ValueError("recovery_guard_rad must be nonnegative scalar or five-vector")
        if (not np.isfinite(self.recovery_rate_s_inv)
                or self.recovery_rate_s_inv <= 0):
            raise ValueError("recovery_rate_s_inv must be finite and positive")
        self.recovery_guard_rad = guard.copy()
        super().__init__(*args, **kwargs)

    def _build_constraints(self):
        rate, dt = self.recovery_rate_s_inv, self.control_dt
        limits = self.safety_joint_limits
        if np.any(2*self.recovery_guard_rad >= limits[:, 1]-limits[:, 0]):
            raise ValueError("recovery guard leaves an empty position/velocity envelope")
        witness_acceleration = rate*self.max_dq/(1+0.5*rate*dt)
        if rate*dt >= 2 or np.any(witness_acceleration > 0.8*self.max_ddq+1e-12):
            raise ValueError("recovery rate exceeds the 80-percent nominal braking reserve")
        matrix, lower, upper = super()._build_constraints()
        self.recovery_row_start = matrix.shape[0]
        envelope = sparse.lil_matrix((self.horizon*self.n, self.num_variables))
        node_row = self.Sq + self.Sv/rate
        for k in range(1, self.horizon+1):
            envelope[(k-1)*self.n:k*self.n, self._cx(k):self._cx(k)+self.nx] = node_row
        return (
            sparse.vstack((matrix, envelope), format="csc"),
            np.r_[lower, np.tile(limits[:, 0]+self.recovery_guard_rad, self.horizon)],
            np.r_[upper, np.tile(limits[:, 1]-self.recovery_guard_rad, self.horizon)],
        )

    def _build_online_constraint_bounds(self, q, dq):
        lower, upper, states, inputs, active = super()._build_online_constraint_bounds(q, dq)
        # The shared recovery implementation can extend a zero-margin joint's
        # position box when its measurement is already outside the outer limit.
        # A measured-state hardware preview must retain the actual outer box.
        start = (self.horizon+1)*self.nx
        rows = slice(start, start+self.horizon*self.n)
        lower[rows] = np.maximum(lower[rows], np.tile(self.safety_joint_limits[:, 0], self.horizon))
        upper[rows] = np.minimum(upper[rows], np.tile(self.safety_joint_limits[:, 1], self.horizon))
        return lower, upper, states, inputs, active
