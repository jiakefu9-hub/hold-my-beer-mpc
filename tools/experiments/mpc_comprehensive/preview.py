"""Causal innovations added to the frozen simulation template (optional)."""
import numpy as np
from scipy.spatial.transform import Rotation
from disturbance_types import DisturbanceInput, DisturbanceHorizon


class InnovationPreview:
    def __init__(self, inner, config):
        self.inner = inner
        self.gain = float(config.get('research_innovation_gain', 0.0))
        self.decay = float(config.get('research_innovation_decay_s', .024))
        self.angular_gain = float(config.get('research_angular_innovation_gain', 0.0))
        self.lead_s = float(config.get('research_interval_lead_s', 0.0))
        self.rotation_gain = float(config.get('research_rotation_innovation_gain', 0.0))
        self.rotation_decay = float(config.get('research_rotation_innovation_decay_s', .15))
        self.previous_raw_next_rotation = None
        self.measurement = None

    def reset(self):
        self.inner.reset()
        self.measurement = None
        self.previous_raw_next_rotation = None

    def update(self, observation):
        self.inner.update(observation)
        self.measurement = observation.measured_disturbance

    def predict(self, horizon, dt):
        p = self.inner.predict(horizon, dt)
        m = self.measurement
        # Innovation is based only on the CURRENT measured sample. No rollout
        # truth, future sensor data or evaluation trajectory is available here.
        # Native node zero is ALREADY replaced by measurement. Its difference
        # would always be zero. The predictor diagnostics retain the template
        # residual before replacement, in the current world frame.
        diagnostics = self.inner.get_last_diagnostics(copy_data=False)
        def residual(key):
            value = np.asarray(diagnostics.get(key, np.zeros(3)))
            return np.where(np.isfinite(value), value, 0.)
        da = residual('anchor_acc_error')
        dw = residual('anchor_omega_error')
        dalpha = residual('anchor_alpha_error')
        rotation_residual = np.zeros(3)
        if self.rotation_gain and self.previous_raw_next_rotation is not None:
            rotation_residual = Rotation.from_matrix(
                m.rot_world_body @ self.previous_raw_next_rotation.T).as_rotvec()
        # Keep the UNCORRECTED one-step prediction: otherwise a recursively
        # corrected residual can artificially oscillate between bias and zero.
        self.previous_raw_next_rotation = p.nodes[1].rot_world_body.copy()
        def adjusted(d, t):
            f = np.exp(-t / self.decay)
            rotation = d.rot_world_body
            if self.rotation_gain:
                rv = self.rotation_gain * np.exp(-t / self.rotation_decay) * rotation_residual
                rotation = Rotation.from_rotvec(rv).as_matrix() @ rotation
            return DisturbanceInput(
                acc_world=d.acc_world + self.gain * f * da,
                omega_world=d.omega_world + self.angular_gain * f * dw,
                alpha_world=d.alpha_world + self.angular_gain * f * dalpha,
                rot_world_body=rotation)
        intervals = []
        for i, d in enumerate(p.intervals):
            index = np.clip(i + self.lead_s / dt, 0., len(p.intervals) - 1.)
            lo = int(np.floor(index)); hi = min(lo + 1, len(p.intervals) - 1)
            fraction = index - lo
            def shifted(key):
                return ((1. - fraction) * getattr(p.intervals[lo], key)
                        + fraction * getattr(p.intervals[hi], key))
            shifted_d = DisturbanceInput(acc_world=shifted('acc_world'),
                alpha_world=shifted('alpha_world'), omega_world=d.omega_world,
                rot_world_body=d.rot_world_body)
            intervals.append(adjusted(shifted_d, (i + .5) * dt))
        return DisturbanceHorizon(
            nodes=(p.nodes[0],) + tuple(adjusted(d, i * dt)
                for i, d in enumerate(p.nodes) if i > 0), intervals=tuple(intervals))

    def metadata(self):
        return dict(self.inner.metadata(), research_only=True,
                    innovation_gain=self.gain, innovation_decay_s=self.decay,
                    angular_innovation_gain=self.angular_gain,
                    interval_lead_s=self.lead_s,
                    rotation_innovation_gain=self.rotation_gain,
                    rotation_innovation_decay_s=self.rotation_decay,
                    future_source='frozen template plus causal current innovation')

    def get_last_diagnostics(self, copy_data=True):
        return self.inner.get_last_diagnostics(copy_data=copy_data)


def wrap_predictor(inner, config):
    if any(config.get(k, 0.) for k in ('research_innovation_gain',
        'research_angular_innovation_gain', 'research_interval_lead_s',
        'research_rotation_innovation_gain')):
        return InnovationPreview(inner, config)
    return inner
