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

    def __init__(self, *args, recovery_rate_s_inv=6.0, recovery_reentry_enabled=False,
                 recovery_guard_rad=np.deg2rad(0.15), **kwargs):
        if not isinstance(recovery_reentry_enabled, bool):
            raise ValueError('recovery_reentry_enabled must be boolean')
        self.recovery_reentry_enabled = recovery_reentry_enabled
        self._reentry_time = None
        self._reentry_start = np.full(5, np.nan)
        self._reentry_lower = np.zeros(5)
        self._reentry_upper = np.zeros(5)
        self.reentry_diagnostics = dict(enabled=recovery_reentry_enabled, active=False)
        self._one_step = None
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

    def reset(self):
        super().reset()
        self._reentry_time = None
        self._reentry_start[:] = np.nan
        self._one_step = None

    def set_recovery_time(self, elapsed_s):
        """Actual elapsed control time, not a horizon reset every new solve."""
        if (not np.isfinite(elapsed_s) or elapsed_s < 0
                or self._reentry_time is not None and elapsed_s < self._reentry_time):
            raise ValueError('invalid/nonmonotonic recovery clock')
        self._reentry_time = float(elapsed_s)

    def first_step_acceleration_bounds(self, q, dq):
        """Require the FINAL forward-checked torque to obey QP node one too."""
        if self._one_step is None or not np.array_equal(np.r_[q,dq], self._one_step[0]):
            raise ValueError('missing/stale first-step state envelope')
        _, qlo, qhi, hlo, hhi = self._one_step
        dt, rate = self.control_dt, self.recovery_rate_s_inv
        offset = q+dt*dq+dq/rate
        coefficient = .5*dt**2+dt/rate
        lo = np.maximum.reduce((-self.max_ddq, (-self.max_dq-dq)/dt,
                                (qlo-q-dt*dq)/(.5*dt**2), (hlo-offset)/coefficient))
        hi = np.minimum.reduce((self.max_ddq, (self.max_dq-dq)/dt,
                                (qhi-q-dt*dq)/(.5*dt**2), (hhi-offset)/coefficient))
        if np.any(lo>hi+1e-8):
            raise ValueError('empty first-step acceleration envelope')
        return lo, hi

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
        self.reentry_diagnostics = dict(enabled=self.recovery_reentry_enabled, active=False)
        if self.recovery_reentry_enabled:
            if self._reentry_time is None:
                raise ValueError('reentry requires an elapsed-time clock')
            if (np.any(q<self.safety_joint_limits[:,0]-1e-10)
                    or np.any(q>self.safety_joint_limits[:,1]+1e-10)
                    or np.any(np.abs(dq)>self.max_dq+1e-10)):
                raise ValueError('reentry refuses a state outside the outer q/dq bounds')
            lo = self.safety_joint_limits[:,0]+self.recovery_guard_rad
            hi = self.safety_joint_limits[:,1]-self.recovery_guard_rad
            h = q+dq/self.recovery_rate_s_inv
            below, above = np.maximum(lo-h,0.), np.maximum(h-hi,0.)
            outside = np.maximum(below,above)>1e-10
            self._reentry_start[~outside] = np.nan
            begin = outside & np.isnan(self._reentry_start)
            self._reentry_start[begin] = self._reentry_time
            self._reentry_lower[begin], self._reentry_upper[begin] = below[begin], above[begin]
            duration = self.horizon*self.control_dt
            elapsed = np.where(outside,self._reentry_time-self._reentry_start,0.)
            self.reentry_diagnostics = dict(enabled=True, active=bool(np.any(outside)),
                axes=outside.tolist(), clock_s=self._reentry_time, duration_s=duration,
                initial_lower_excess_rad=np.where(outside,self._reentry_lower,0.).tolist(),
                initial_upper_excess_rad=np.where(outside,self._reentry_upper,0.).tolist(),
                elapsed_s=elapsed.tolist(), h_rad=h.tolist(), lower_rad=lo.tolist(), upper_rad=hi.tolist())
            if np.any(outside & (elapsed>=duration-1e-10)):
                raise ValueError('braking-envelope reentry deadline exceeded')
            # A fixed deadline per joint episode: no slack cost, no refreshed
            # grace period. Bounds contract to the ORIGINAL envelope within
            # at most N*dt (54 ms), while all original q/dq/ddq rows remain.
            fraction=np.maximum(0.,1.-(elapsed[None,:]+np.arange(1,self.horizon+1)[:,None]*self.control_dt)/duration)
            fraction*=outside[None,:]
            lower[self.recovery_row_start:] = (lo-fraction*self._reentry_lower).ravel()
            upper[self.recovery_row_start:] = (hi+fraction*self._reentry_upper).ravel()
        self._one_step = (np.r_[q,dq].copy(), lower[start:start+self.n].copy(),
            upper[start:start+self.n].copy(),
            lower[self.recovery_row_start:self.recovery_row_start+self.n].copy(),
            upper[self.recovery_row_start:self.recovery_row_start+self.n].copy())
        self.reentry_diagnostics.update(first_lower_rad=self._one_step[3].tolist(),
                                       first_upper_rad=self._one_step[4].tolist())
        return lower, upper, states, inputs, active
