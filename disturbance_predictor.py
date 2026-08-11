"""Small, predictor-neutral boundary for MPC disturbance previews.

The MPC-facing wire format intentionally remains ``DisturbanceInput`` and
``DisturbanceHorizon``.  This module only removes the main loop's knowledge of
the concrete predictor implementation; it does not alter preview semantics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from kinematics_helper import DisturbanceHorizon, DisturbanceInput


@dataclass(frozen=True)
class DisturbancePredictorObservation:
    """Causal information available at one MPC update instant.

    B0 intentionally contains only the quantities consumed by the legacy
    template.  Later neural work can extend this dataclass without changing
    the preview contract consumed by KinematicsHelper and ArmMPC.
    """

    simulation_time: float
    measured_disturbance: DisturbanceInput

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.simulation_time)
            or float(self.simulation_time) < 0.0
        ):
            raise ValueError("simulation_time 必须是有限非负数。")
        if not isinstance(self.measured_disturbance, DisturbanceInput):
            raise TypeError("measured_disturbance 必须是 DisturbanceInput。")


@runtime_checkable
class DisturbancePredictor(Protocol):
    """The only predictor behavior required by the MPC orchestration layer."""

    def reset(self) -> None:
        ...

    def update(self, observation: DisturbancePredictorObservation) -> None:
        ...

    def predict(self, horizon: int, dt: float) -> DisturbanceHorizon:
        ...

    def metadata(self) -> dict:
        ...

    def get_last_diagnostics(self) -> dict:
        ...


class _FixedPreviewPredictor:
    """Shared request validation for B0 predictors with fixed construction grid."""

    def __init__(self, control_dt: float, horizon: int) -> None:
        self._control_dt = float(control_dt)
        self._horizon = int(horizon)
        if not np.isfinite(self._control_dt) or self._control_dt <= 0.0:
            raise ValueError("control_dt 必须是有限正数。")
        if self._horizon < 1:
            raise ValueError("horizon 必须至少为 1。")
        self._observation: Optional[DisturbancePredictorObservation] = None

    def _validate_request(self, horizon: int, dt: float) -> None:
        if int(horizon) != self._horizon:
            raise ValueError(
                f"predict horizon={horizon} 与初始化 horizon={self._horizon} 不一致。"
            )
        if not np.isclose(
            float(dt), self._control_dt, rtol=1e-6, atol=1e-9
        ):
            raise ValueError(
                f"predict dt={dt} 与初始化 control_dt={self._control_dt} 不一致。"
            )

    def update(self, observation: DisturbancePredictorObservation) -> None:
        if not isinstance(observation, DisturbancePredictorObservation):
            raise TypeError("update() 需要 DisturbancePredictorObservation。")
        self._observation = observation

    def _require_observation(self) -> DisturbancePredictorObservation:
        if self._observation is None:
            raise RuntimeError("predict() 前必须先调用 update(observation)。")
        return self._observation


class TemplateDisturbancePredictor(_FixedPreviewPredictor):
    """Adapter preserving the mature phase-template implementation verbatim."""

    def __init__(
        self,
        *,
        template_dir: str,
        variant: str,
        control_dt: float,
        horizon: int,
        acc_limit: float = np.inf,
        alpha_limit: float = np.inf,
        slow_bias_enabled: bool = True,
        slow_bias_time_constant: float = 0.4,
    ) -> None:
        super().__init__(control_dt=control_dt, horizon=horizon)
        self._phase_kwargs = {
            "template_dir": template_dir,
            "variant": variant,
            "control_dt": self._control_dt,
            "horizon": self._horizon,
            "acc_limit": acc_limit,
            "alpha_limit": alpha_limit,
            "slow_bias_enabled": slow_bias_enabled,
            "slow_bias_time_constant": slow_bias_time_constant,
        }
        self._phase_predictor = self._create_phase_predictor()

    def _create_phase_predictor(self):
        # Lazy import prevents sim_support <-> interface import cycles while
        # retaining the original implementation as the numerical engine.
        from sim_support import PhaseDisturbancePredictor

        return PhaseDisturbancePredictor(**self._phase_kwargs)

    def reset(self) -> None:
        self._phase_predictor.reset()
        self._observation = None

    def predict(self, horizon: int, dt: float) -> DisturbanceHorizon:
        self._validate_request(horizon, dt)
        observation = self._require_observation()
        return self._phase_predictor.predict(
            observation.simulation_time, observation.measured_disturbance
        )

    def metadata(self) -> dict:
        return {
            **self._phase_predictor.metadata(),
            "predictor_type": "template",
            "interface": "update_then_predict_fixed_horizon",
        }

    def get_last_diagnostics(self) -> dict:
        return self._phase_predictor.get_last_diagnostics()


class ZeroOrderHoldPredictor(_FixedPreviewPredictor):
    """Explicit current-measurement hold with the same preview time contract."""

    def __init__(self, *, control_dt: float, horizon: int) -> None:
        super().__init__(control_dt=control_dt, horizon=horizon)
        self._last_diagnostics = self._empty_diagnostics()

    def reset(self) -> None:
        self._observation = None
        self._last_diagnostics = self._empty_diagnostics()

    def predict(self, horizon: int, dt: float) -> DisturbanceHorizon:
        self._validate_request(horizon, dt)
        observation = self._require_observation()
        measurement = observation.measured_disturbance
        nodes = tuple(
            self._copy_disturbance(measurement)
            for _ in range(self._horizon + 1)
        )
        intervals = tuple(
            self._copy_disturbance(measurement) for _ in range(self._horizon)
        )
        self._last_diagnostics = {
            "predictor_type": "zoh",
            "simulation_time": float(observation.simulation_time),
            "heading_ready": False,
            "one_step_prediction_valid": False,
        }
        return DisturbanceHorizon(nodes=nodes, intervals=intervals)

    def metadata(self) -> dict:
        return {
            "enabled": False,
            "predictor_type": "zoh",
            "interface": "update_then_predict_fixed_horizon",
            "prediction": "zero_order_hold_current_measurement",
            "node_definition": "instantaneous_at_t_k",
            "interval_definition": "average_over_[t_k,t_k+control_dt)",
        }

    def get_last_diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    @staticmethod
    def _copy_disturbance(source: DisturbanceInput) -> DisturbanceInput:
        def copied(name: str):
            value = getattr(source, name)
            return None if value is None else np.asarray(value).copy()

        return DisturbanceInput(
            acc_world=copied("acc_world"),
            omega_world=copied("omega_world"),
            alpha_world=copied("alpha_world"),
            rot_world_body=copied("rot_world_body"),
        )

    @staticmethod
    def _empty_diagnostics() -> dict:
        return {
            "predictor_type": "zoh",
            "simulation_time": np.nan,
            "heading_ready": False,
            "one_step_prediction_valid": False,
        }


def resolve_disturbance_predictor_name(config: dict) -> str:
    """Resolve the B0 selector while preserving the legacy boolean behavior."""
    configured = config.get("disturbance_predictor")
    if configured is None:
        return (
            "template"
            if bool(config.get("mpc_disturbance_feedforward_enabled", False))
            else "zoh"
        )
    name = str(configured).strip().lower()
    if name not in {"template", "zoh"}:
        raise ValueError(
            "disturbance_predictor 必须是 'template' 或 'zoh'。"
        )
    return name


def create_disturbance_predictor(
    config: dict,
    *,
    repo_dir: str,
    control_dt: float,
    horizon: int,
    acc_limit: float,
    alpha_limit: float,
) -> DisturbancePredictor:
    """Build the configured B0 predictor without exposing its type to main_sim."""
    name = resolve_disturbance_predictor_name(config)
    if name == "zoh":
        return ZeroOrderHoldPredictor(control_dt=control_dt, horizon=horizon)

    template_dir = str(
        config.get(
            "mpc_disturbance_template_dir",
            "disturbance_model_new_heading/templates_heading_interval",
        )
    )
    if not os.path.isabs(template_dir):
        template_dir = os.path.join(repo_dir, template_dir)
    return TemplateDisturbancePredictor(
        template_dir=template_dir,
        variant=config.get("mpc_disturbance_template", "fully_smoothed"),
        control_dt=control_dt,
        horizon=horizon,
        acc_limit=acc_limit,
        alpha_limit=alpha_limit,
        slow_bias_enabled=bool(
            config.get("mpc_disturbance_slow_bias_enabled", True)
        ),
        slow_bias_time_constant=float(
            config.get("mpc_disturbance_slow_bias_time_constant", 0.4)
        ),
    )
