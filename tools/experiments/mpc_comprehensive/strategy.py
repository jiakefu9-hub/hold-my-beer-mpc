"""Optional simulation research costs; the production policy is unchanged."""
from types import MethodType
import numpy as np


class TruncatedPreviewMPC:
    """Consume a nine-interval wire preview, optimize only its leading N steps.

    The simulation task/recording grid stays frozen. `horizon` here describes
    the preview interface; the actual QP length is explicitly recorded by
    `optimization_horizon` and the inner policy's cost definition.
    """
    def __init__(self, inner, wire_horizon):
        self.inner = inner
        self.horizon = wire_horizon
        self.optimization_horizon = inner.horizon

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def compute_action(self, observation, helpers):
        if isinstance(helpers, dict):
            reduced = dict(helpers)
        else:
            reduced = {name: getattr(helpers, name, None) for name in (
                'compute_mpc_terms', 'compute_mpc_terms_batch', 'disturbance',
                'disturbance_prediction', 'interval_disturbance_prediction')}
        n = self.optimization_horizon
        reduced['disturbance_prediction'] = reduced['disturbance_prediction'][:n+1]
        reduced['interval_disturbance_prediction'] = reduced['interval_disturbance_prediction'][:n]
        return self.inner.compute_action(observation, reduced)

    def get_cost_definition(self):
        result = self.inner.get_cost_definition()
        result.update(research_preview_horizon=self.horizon,
                      research_optimization_horizon=self.optimization_horizon)
        return result


def configure_policy(policy, config):
    horizon = int(config.get('research_optimization_horizon', policy.horizon))
    if not 1 <= horizon <= policy.horizon:
        raise ValueError('research QP horizon must fit the available template preview')
    if horizon != policy.horizon:
        from right_arm_control_setup import create_arm_controller
        shorter = dict(config, mpc_horizon=horizon)
        shorter.pop('research_optimization_horizon', None)
        inner = create_arm_controller(shorter, 'mpc', policy.default_q, policy.control_dt).policy
        return TruncatedPreviewMPC(configure_policy(inner, shorter), policy.horizon)
    weight = float(config.get('research_r_first_ddq_delta', 0.))
    if weight < 0. or not np.isfinite(weight):
        raise ValueError('first-step input-change weight must be finite and nonnegative')
    if weight == 0.:
        return policy
    original = policy._build_cost

    def build_cost(self, terms):
        hessians, linear = original(terms)
        previous = np.zeros(self.nu) if self._last_u_plan is None else self._last_u_plan[0]
        # w ||u_0 - u_previous||²; this is first-action smoothing, NOT a
        # full horizon jerk objective and NOT a physical actuator rate limit.
        hessians[0][self.nx:, self.nx:] += 2. * weight * np.eye(self.nu)
        linear[self.nx:self.stage_dim] -= 2. * weight * previous
        return hessians, linear

    policy._build_cost = MethodType(build_cost, policy)
    return policy
