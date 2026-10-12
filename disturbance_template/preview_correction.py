"""Optional causal correction of the SIMULATION full-task template preview.

No robot transport or hardware learned-predictor integration is provided here.
The frozen predictor remains the default. State node zero and the inverse
dynamics measurement are never shifted. Template validation remains in force.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from disturbance_types import DisturbanceHorizon, DisturbanceInput


class TemplatePreviewCorrection:
    """Correct future orientations with the preceding one-step prediction error.

    At time t, E = R_measured(t) R_raw_prediction(t|t-dt)^T. Future
    orientations become Exp(gain exp(-k dt/tau) Log(E)) R_raw(t+k dt).
    The stored prediction is always RAW, preventing recursive bias cancellation.
    This corrects the orientation forecast only; predicted omega is unchanged.

    Optional interval lead interpolates the already available acceleration and
    angular-acceleration interval means. It does not query future sensor data.
    The last available interval is held; no unavailable tail is extrapolated.
    """

    def __init__(self, inner, *, rotation_gain=0., rotation_decay_s=1.,
                 interval_lead_s=0.):
        self.inner = inner
        self.rotation_gain = float(rotation_gain)
        self.rotation_decay_s = float(rotation_decay_s)
        self.interval_lead_s = float(interval_lead_s)
        if not np.isfinite(self.rotation_gain) or not 0. <= self.rotation_gain <= 1.:
            raise ValueError('rotation correction gain must lie in [0, 1]')
        if not np.isfinite(self.rotation_decay_s) or self.rotation_decay_s <= 0.:
            raise ValueError('rotation correction decay must be positive and finite')
        if not np.isfinite(self.interval_lead_s) or not 0. <= self.interval_lead_s <= .009:
            raise ValueError('simulation interval lead must lie in [0, 0.009] s')
        self._previous_raw_next_rotation = None
        self._previous_time = None
        self._measurement = None
        self._time = None
        self._cached_time = None
        self._cached_preview = None

    def reset(self):
        self.inner.reset()
        self._previous_raw_next_rotation = None
        self._previous_time = None
        self._measurement = None
        self._time = None
        self._cached_time = None
        self._cached_preview = None

    def update(self, observation):
        self.inner.update(observation)
        self._measurement = observation.measured_disturbance
        self._time = float(observation.simulation_time)

    def predict(self, horizon, dt):
        # Preserve repeated predict calls at the same observation; do not treat
        # the current call's next node as the previous timestamp's prediction.
        if self._time is not None and self._cached_time == self._time:
            if len(self._cached_preview.intervals) != horizon or dt != self._cached_dt:
                raise ValueError('cached preview request changed grid')
            return self._cached_preview
        raw = self.inner.predict(horizon, dt)
        diagnostics = self.inner.get_last_diagnostics(copy_data=False)
        if diagnostics.get('terminal_hold_used', False):
            self._previous_raw_next_rotation = None
            self._previous_time = None
            return raw
        residual = np.zeros(3)
        if (self.rotation_gain and self._previous_raw_next_rotation is not None
                and np.isclose(self._time - self._previous_time, dt, rtol=0., atol=1e-9)):
            residual = Rotation.from_matrix(
                self._measurement.rot_world_body @ self._previous_raw_next_rotation.T
            ).as_rotvec()
        self._previous_raw_next_rotation = raw.nodes[1].rot_world_body.copy()
        self._previous_time = self._time

        def corrected_rotation(rotation, time_ahead):
            if self.rotation_gain == 0.:
                return rotation
            correction = self.rotation_gain * np.exp(-time_ahead/self.rotation_decay_s) * residual
            return Rotation.from_rotvec(correction).as_matrix() @ rotation

        nodes = [raw.nodes[0]]
        for i, d in enumerate(raw.nodes[1:], start=1):
            nodes.append(DisturbanceInput(acc_world=d.acc_world,
                omega_world=d.omega_world, alpha_world=d.alpha_world,
                rot_world_body=corrected_rotation(d.rot_world_body, i*dt)))
        intervals = []
        for i, d in enumerate(raw.intervals):
            index = min(i + self.interval_lead_s/dt, horizon-1.)
            lo = int(np.floor(index)); hi = min(lo+1, horizon-1)
            fraction = index-lo
            def shifted(key):
                return ((1.-fraction)*getattr(raw.intervals[lo], key)
                        + fraction*getattr(raw.intervals[hi], key))
            intervals.append(DisturbanceInput(acc_world=shifted('acc_world'),
                alpha_world=shifted('alpha_world'), omega_world=d.omega_world,
                rot_world_body=corrected_rotation(d.rot_world_body, (i+.5)*dt)))
        result = DisturbanceHorizon(nodes=tuple(nodes), intervals=tuple(intervals))
        self._cached_time = self._time
        self._cached_dt = dt
        self._cached_preview = result
        return result

    def metadata(self):
        return dict(self.inner.metadata(),
            future_source='validated template with causal orientation correction and optional interval lead',
            preview_correction=dict(
            simulation_only=True, rotation_gain=self.rotation_gain,
            rotation_decay_s=self.rotation_decay_s,
            interval_lead_s=self.interval_lead_s,
            node_zero='unchanged measured state',
            angular_velocity='unchanged template forecast',
            diagnostics='template residual diagnostics remain uncorrected; MPC preview logs record corrected inputs'))

    def get_last_diagnostics(self, copy_data=True):
        return self.inner.get_last_diagnostics(copy_data=copy_data)


def correct_template_preview(inner, config):
    gain = float(config.get('mpc_preview_rotation_correction_gain', 0.))
    lead = float(config.get('mpc_preview_interval_lead_s', 0.))
    if gain == 0. and lead == 0.:
        return inner
    return TemplatePreviewCorrection(inner, rotation_gain=gain,
        rotation_decay_s=config.get('mpc_preview_rotation_decay_s', 1.),
        interval_lead_s=lead)
