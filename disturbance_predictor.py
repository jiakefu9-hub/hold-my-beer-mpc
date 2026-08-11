"""Small, predictor-neutral boundary for MPC disturbance previews.

The MPC-facing wire format intentionally remains ``DisturbanceInput`` and
``DisturbanceHorizon``.  This module only removes the main loop's knowledge of
the concrete predictor implementation; it does not alter preview semantics.
"""

from __future__ import annotations

import os
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch

from disturbance_learning.dataset import (
    FEATURE_NAMES,
    HEADING_DEFINITION,
    TARGET_NAMES,
)
from disturbance_learning.mlp_model import load_mlp_checkpoint, parameter_count
from disturbance_model_new_heading.heading_template_utils import rotation_z
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
    gravity_direction_torso: Optional[np.ndarray] = None
    lower_body_q: Optional[np.ndarray] = None
    lower_body_dq: Optional[np.ndarray] = None
    lower_body_policy_target: Optional[np.ndarray] = None
    runtime_command: Optional[np.ndarray] = None
    gait_phase_sin_cos: Optional[np.ndarray] = None
    # Causal feedback from the previous MPC update.  It is deliberately
    # optional so template/ZOH callers and offline tests remain unchanged.
    previous_mpc_success: Optional[bool] = None
    previous_control_interval_overrun: Optional[bool] = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.simulation_time)
            or float(self.simulation_time) < 0.0
        ):
            raise ValueError("simulation_time 必须是有限非负数。")
        if not isinstance(self.measured_disturbance, DisturbanceInput):
            raise TypeError("measured_disturbance 必须是 DisturbanceInput。")
        for name, expected_shape in (
            ("gravity_direction_torso", (3,)),
            ("lower_body_q", (12,)),
            ("lower_body_dq", (12,)),
            ("lower_body_policy_target", (12,)),
            ("runtime_command", (3,)),
            ("gait_phase_sin_cos", (2,)),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.shape != expected_shape or not np.all(np.isfinite(array)):
                raise ValueError(
                    f"{name} 必须是有限数组，shape={expected_shape}。"
                )
        for name in (
            "previous_mpc_success",
            "previous_control_interval_overrun",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} 必须是 bool 或 None。")


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

    def get_last_diagnostics(self, copy_data: bool = True) -> dict:
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

    def get_last_diagnostics(self, copy_data: bool = True) -> dict:
        return self._phase_predictor.get_last_diagnostics(copy_data=copy_data)


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

    def get_last_diagnostics(self, copy_data: bool = True) -> dict:
        return (
            dict(self._last_diagnostics)
            if copy_data
            else self._last_diagnostics
        )

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


class _MLPPreviewPredictor(_FixedPreviewPredictor):
    """Shared causal history, H-frame conversion and one-shot CPU inference."""

    _FALLBACK_NONE = 0
    _FALLBACK_HEADING = 1
    _FALLBACK_HISTORY = 2
    _FALLBACK_HISTORY_GAP = 3
    _FALLBACK_NONFINITE = 4
    _FALLBACK_INPUT_OUT_OF_RANGE = 5
    _FALLBACK_PREDICTION_OUT_OF_RANGE = 6
    _FALLBACK_SOLVER_QUALITY = 7
    _FALLBACK_CONTROL_OVERRUN = 8
    _FALLBACK_REASONS = {
        _FALLBACK_NONE: "none",
        _FALLBACK_HEADING: "heading_not_ready",
        _FALLBACK_HISTORY: "history_not_ready",
        _FALLBACK_HISTORY_GAP: "history_gap",
        _FALLBACK_NONFINITE: "nonfinite_prediction",
        _FALLBACK_INPUT_OUT_OF_RANGE: "normalized_input_out_of_range",
        _FALLBACK_PREDICTION_OUT_OF_RANGE: "residual_out_of_range",
        _FALLBACK_SOLVER_QUALITY: "previous_solver_failure_cooldown",
        _FALLBACK_CONTROL_OVERRUN: "previous_control_interval_overrun",
    }

    def __init__(
        self,
        *,
        checkpoint_path: str,
        prediction_mode: str,
        template_reference: TemplateDisturbancePredictor,
        control_dt: float,
        horizon: int,
        acc_limit: float,
        alpha_limit: float,
        safety_gate_enabled: bool = False,
        max_input_abs_z: float = np.inf,
        max_input_rms_z: float = np.inf,
        max_prediction_abs_z: float = np.inf,
        max_acc_correction_norm: float = np.inf,
        max_alpha_correction_norm: float = np.inf,
        solver_failure_streak_threshold: int = 2,
        solver_failure_cooldown_steps: int = 0,
        control_overrun_cooldown_steps: int = 0,
    ) -> None:
        super().__init__(control_dt=control_dt, horizon=horizon)
        self._checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self._checkpoint_path.is_file():
            raise FileNotFoundError(
                f"找不到 neural disturbance checkpoint: {self._checkpoint_path}"
            )
        self._checkpoint_sha256 = hashlib.sha256(
            self._checkpoint_path.read_bytes()
        ).hexdigest()
        eager_model, normalization, payload = load_mlp_checkpoint(
            self._checkpoint_path
        )
        stored_mode = str(payload.get("prediction_mode", "absolute"))
        if stored_mode != prediction_mode:
            raise ValueError(
                "MLP checkpoint 语义不匹配: "
                f"需要 {prediction_mode!r}，实际 {stored_mode!r}。"
            )
        if prediction_mode == "residual_template":
            reference_metadata = template_reference.metadata()
            residual_semantics_match = (
                payload.get("heading_definition") == HEADING_DEFINITION
                and np.isclose(
                    float(payload.get("control_dt", np.nan)),
                    self._control_dt,
                    rtol=1e-6,
                    atol=1e-9,
                )
                and payload.get("template_variant")
                == reference_metadata["variant"]
                and bool(payload.get("template_slow_bias_enabled"))
                == bool(reference_metadata["slow_bias_enabled"])
                and np.isclose(
                    float(
                        payload.get(
                            "template_slow_bias_time_constant", np.nan
                        )
                    ),
                    float(reference_metadata["slow_bias_time_constant"]),
                    rtol=1e-6,
                    atol=1e-9,
                )
            )
            if not residual_semantics_match:
                raise ValueError(
                    "residual checkpoint 的 H-frame/control_dt/template/"
                    "slow-bias 语义与在线 baseline 不一致。"
                )
        if (
            eager_model.horizon != self._horizon
            or eager_model.target_dim != len(TARGET_NAMES)
            or eager_model.feature_dim != len(FEATURE_NAMES)
            or list(payload.get("feature_names", [])) != list(FEATURE_NAMES)
            or list(payload.get("target_names", [])) != list(TARGET_NAMES)
        ):
            raise ValueError("MLP checkpoint 与当前 34x50 -> 9x6 schema 不匹配。")
        self._history_steps = int(eager_model.history_steps)
        if self._history_steps < 2:
            raise ValueError("MLP history_steps 必须至少为 2。")
        self._parameter_count = parameter_count(eager_model)
        self._model = eager_model
        with torch.inference_mode():
            self._model(
                torch.zeros(
                    (1, self._history_steps, len(FEATURE_NAMES)),
                    dtype=torch.float32,
                )
            )
        expected_normalization_shapes = {
            "feature_mean": (len(FEATURE_NAMES),),
            "feature_std": (len(FEATURE_NAMES),),
            "target_mean": (len(TARGET_NAMES),),
            "target_std": (len(TARGET_NAMES),),
        }
        for name, expected_shape in expected_normalization_shapes.items():
            value = normalization.get(name)
            if (
                value is None
                or value.shape != expected_shape
                or not np.all(np.isfinite(value))
                or (name.endswith("_std") and np.any(value <= 0.0))
            ):
                raise ValueError(f"checkpoint normalization {name} 无效。")
        self._feature_mean = normalization["feature_mean"].copy()
        self._feature_std = normalization["feature_std"].copy()
        self._target_mean = normalization["target_mean"].copy()
        self._target_std = normalization["target_std"].copy()
        self._history_buffer = np.empty(
            (self._history_steps, len(FEATURE_NAMES)), dtype=np.float32
        )
        self._history_features = np.empty_like(self._history_buffer)
        self._normalized_history = np.empty(
            (self._history_steps, len(FEATURE_NAMES)), dtype=np.float32
        )
        # The NumPy buffer stays alive for the predictor lifetime, so this
        # tensor view can be reused without rebuilding a Python/Torch wrapper
        # on every 6 ms update.
        self._normalized_input_tensor = torch.from_numpy(
            self._normalized_history
        ).unsqueeze(0)
        self._prediction_mode = prediction_mode
        self._template_reference = template_reference
        self._acc_limit = float(acc_limit)
        self._alpha_limit = float(alpha_limit)
        self._safety_gate_enabled = bool(safety_gate_enabled)
        safety_limits = {
            "max_input_abs_z": max_input_abs_z,
            "max_input_rms_z": max_input_rms_z,
            "max_prediction_abs_z": max_prediction_abs_z,
            "max_acc_correction_norm": max_acc_correction_norm,
            "max_alpha_correction_norm": max_alpha_correction_norm,
        }
        for name, value in safety_limits.items():
            if np.isnan(value) or float(value) <= 0.0:
                raise ValueError(f"{name} 必须是正数或 inf。")
        self._max_input_abs_z = float(max_input_abs_z)
        self._max_input_rms_z = float(max_input_rms_z)
        self._max_prediction_abs_z = float(max_prediction_abs_z)
        self._max_acc_correction_norm = float(max_acc_correction_norm)
        self._max_alpha_correction_norm = float(max_alpha_correction_norm)
        self._solver_failure_cooldown_steps = int(
            solver_failure_cooldown_steps
        )
        self._control_overrun_cooldown_steps = int(
            control_overrun_cooldown_steps
        )
        self._solver_failure_streak_threshold = int(
            solver_failure_streak_threshold
        )
        if self._solver_failure_streak_threshold < 1:
            raise ValueError("solver_failure_streak_threshold 必须至少为 1。")
        if self._solver_failure_cooldown_steps < 0:
            raise ValueError("solver_failure_cooldown_steps 不能为负数。")
        if self._control_overrun_cooldown_steps < 0:
            raise ValueError("control_overrun_cooldown_steps 不能为负数。")
        self._solver_failure_streak = 0
        self._solver_gate_remaining = 0
        self._timing_gate_remaining = 0
        self._last_correction_applied = False
        self._history_count = 0
        self._history_write_index = 0
        self._last_history_time: Optional[float] = None
        self._history_gap = False
        self._reset_safety_diagnostics()
        self._last_diagnostics = self._empty_neural_diagnostics()

    def reset(self) -> None:
        self._observation = None
        self._history_count = 0
        self._history_write_index = 0
        self._last_history_time = None
        self._history_gap = False
        self._solver_failure_streak = 0
        self._solver_gate_remaining = 0
        self._timing_gate_remaining = 0
        self._last_correction_applied = False
        self._reset_safety_diagnostics()
        self._template_reference.reset()
        self._last_diagnostics = self._empty_neural_diagnostics()

    def update(self, observation: DisturbancePredictorObservation) -> None:
        self._require_neural_fields(observation)
        if self._last_history_time is not None:
            elapsed = observation.simulation_time - self._last_history_time
            if not np.isclose(
                elapsed, self._control_dt, rtol=1e-6, atol=1e-9
            ):
                if elapsed <= 0.0:
                    raise ValueError("neural predictor 不支持重复或倒退时间戳。")
                self._history_count = 0
                self._history_write_index = 0
                self._history_gap = True
        super().update(observation)
        self._observation_feature_row(
            observation,
            out=self._history_buffer[self._history_write_index],
        )
        self._history_write_index = (
            self._history_write_index + 1
        ) % self._history_steps
        self._history_count = min(
            self._history_count + 1, self._history_steps
        )
        self._last_history_time = observation.simulation_time
        self._template_reference.update(observation)

    def metadata(self) -> dict:
        return {
            "enabled": True,
            "predictor_type": self._predictor_type,
            "interface": "update_then_predict_fixed_horizon",
            "checkpoint_path": str(self._checkpoint_path),
            "checkpoint_sha256": self._checkpoint_sha256,
            "prediction_mode": self._prediction_mode,
            "model": "flatten_mlp_one_shot",
            "parameter_count": self._parameter_count,
            "runtime_model": "pytorch_eager_cpu",
            "history_shape": [self._history_steps, len(FEATURE_NAMES)],
            "history_timestamp_span_s": (
                (self._history_steps - 1) * self._control_dt
            ),
            "output_shape": [self._horizon, len(TARGET_NAMES)],
            "inference_device": "cpu",
            "heading_definition": (
                "previous_complete_gait_cycle_circular_mean_torso_yaw"
            ),
            "node_definition": "instantaneous_at_t_k",
            "interval_definition": "average_over_[t_k,t_k+control_dt)",
            "template_reference": self._template_reference.metadata(),
            "safety_gate": {
                "enabled": self._safety_gate_enabled,
                "max_input_abs_z": self._max_input_abs_z,
                "max_input_rms_z": self._max_input_rms_z,
                "max_prediction_abs_z": self._max_prediction_abs_z,
                "max_acc_correction_norm": (
                    self._max_acc_correction_norm
                ),
                "max_alpha_correction_norm": (
                    self._max_alpha_correction_norm
                ),
                "solver_failure_cooldown_steps": (
                    self._solver_failure_cooldown_steps
                ),
                "solver_failure_streak_threshold": (
                    self._solver_failure_streak_threshold
                ),
                "control_overrun_cooldown_steps": (
                    self._control_overrun_cooldown_steps
                ),
                "fallback_codes": dict(self._FALLBACK_REASONS),
            },
        }

    def get_last_diagnostics(self, copy_data: bool = True) -> dict:
        if not copy_data:
            return self._last_diagnostics
        copied = {}
        for name, value in self._last_diagnostics.items():
            copied[name] = value.copy() if isinstance(value, np.ndarray) else value
        return copied

    def _reference_and_prediction(
        self,
    ) -> tuple[DisturbanceHorizon, Optional[np.ndarray], dict, int, float]:
        self._reset_safety_diagnostics()
        reference = self._template_reference.predict(
            self._horizon, self._control_dt
        )
        template_diagnostics = self._template_reference.get_last_diagnostics(
            copy_data=False
        )
        if not bool(template_diagnostics.get("heading_ready", False)):
            return (
                reference,
                None,
                template_diagnostics,
                self._FALLBACK_HEADING,
                0.0,
            )
        if self._history_count < self._history_steps:
            code = (
                self._FALLBACK_HISTORY_GAP
                if self._history_gap
                else self._FALLBACK_HISTORY
            )
            return reference, None, template_diagnostics, code, 0.0
        pre_inference_fallback = self._pre_inference_fallback_code()
        if pre_inference_fallback != self._FALLBACK_NONE:
            return (
                reference,
                None,
                template_diagnostics,
                pre_inference_fallback,
                0.0,
            )
        heading_yaw = float(template_diagnostics["heading_yaw_world"])
        features = self._build_history_features(heading_yaw)
        started = time.perf_counter()
        with torch.inference_mode():
            np.subtract(
                features,
                self._feature_mean[None, :],
                out=self._normalized_history,
            )
            np.divide(
                self._normalized_history,
                self._feature_std[None, :],
                out=self._normalized_history,
            )
            self._input_abs_z_max = max(
                float(np.max(self._normalized_history)),
                -float(np.min(self._normalized_history)),
            )
            self._input_rms_z = float(
                np.linalg.norm(self._normalized_history)
                / np.sqrt(self._normalized_history.size)
            )
            if not np.isfinite(
                self._input_abs_z_max + self._input_rms_z
            ):
                return (
                    reference,
                    None,
                    template_diagnostics,
                    self._FALLBACK_NONFINITE,
                    time.perf_counter() - started,
                )
            if self._safety_gate_enabled and (
                self._input_abs_z_max > self._max_input_abs_z
                or self._input_rms_z > self._max_input_rms_z
            ):
                return (
                    reference,
                    None,
                    template_diagnostics,
                    self._FALLBACK_INPUT_OUT_OF_RANGE,
                    time.perf_counter() - started,
                )
            normalized_prediction = self._model(
                self._normalized_input_tensor
            )[0].cpu().numpy()
            self._prediction_abs_z_max = max(
                float(np.max(normalized_prediction)),
                -float(np.min(normalized_prediction)),
            )
            prediction = (
                normalized_prediction * self._target_std[None, :]
                + self._target_mean[None, :]
            )
        inference_time = time.perf_counter() - started
        if not np.all(np.isfinite(prediction)):
            return (
                reference,
                None,
                template_diagnostics,
                self._FALLBACK_NONFINITE,
                inference_time,
            )
        self._acc_correction_norm_max = float(
            np.max(np.linalg.norm(prediction[:, :3], axis=1))
        )
        self._alpha_correction_norm_max = float(
            np.max(np.linalg.norm(prediction[:, 3:], axis=1))
        )
        if self._safety_gate_enabled and (
            self._prediction_abs_z_max > self._max_prediction_abs_z
            or self._acc_correction_norm_max
            > self._max_acc_correction_norm
            or self._alpha_correction_norm_max
            > self._max_alpha_correction_norm
        ):
            return (
                reference,
                None,
                template_diagnostics,
                self._FALLBACK_PREDICTION_OUT_OF_RANGE,
                inference_time,
            )
        self._history_gap = False
        return (
            reference,
            prediction.astype(np.float64),
            template_diagnostics,
            self._FALLBACK_NONE,
            inference_time,
        )

    def _finish_diagnostics(
        self,
        template_diagnostics: dict,
        fallback_code: int,
        inference_time: float,
    ) -> None:
        self._last_diagnostics = {
            **template_diagnostics,
            "predictor_type": self._predictor_type,
            "prediction_mode": self._prediction_mode,
            "history_count": self._history_count,
            "history_required": self._history_steps,
            "neural_inference_valid": fallback_code == self._FALLBACK_NONE,
            "neural_inference_time": float(inference_time),
            "fallback_used": fallback_code != self._FALLBACK_NONE,
            "fallback_code": int(fallback_code),
            "fallback_reason": self._FALLBACK_REASONS[fallback_code],
            "safety_gate_triggered": fallback_code
            in {
                self._FALLBACK_NONFINITE,
                self._FALLBACK_INPUT_OUT_OF_RANGE,
                self._FALLBACK_PREDICTION_OUT_OF_RANGE,
                self._FALLBACK_SOLVER_QUALITY,
                self._FALLBACK_CONTROL_OVERRUN,
            },
            "input_abs_z_max": self._input_abs_z_max,
            "input_rms_z": self._input_rms_z,
            "prediction_abs_z_max": self._prediction_abs_z_max,
            "acc_correction_norm_max": self._acc_correction_norm_max,
            "alpha_correction_norm_max": self._alpha_correction_norm_max,
            "solver_gate_remaining": self._solver_gate_remaining,
            "solver_failure_streak": self._solver_failure_streak,
            "timing_gate_remaining": self._timing_gate_remaining,
        }

    def _build_history_features(self, heading_yaw_world: float) -> np.ndarray:
        rotation_heading_world = rotation_z(-heading_yaw_world)
        if self._history_count != self._history_steps:
            raise AssertionError("online MLP history 尚未填满。")
        split = self._history_steps - self._history_write_index
        np.copyto(
            self._history_features[:split],
            self._history_buffer[self._history_write_index :],
        )
        if self._history_write_index:
            np.copyto(
                self._history_features[split:],
                self._history_buffer[: self._history_write_index],
            )
        features = self._history_features
        features[:, :3] = features[:, :3] @ rotation_heading_world.T
        features[:, 3:6] = features[:, 3:6] @ rotation_heading_world.T
        if features.shape != (self._history_steps, len(FEATURE_NAMES)):
            raise AssertionError("online MLP feature shape 与训练 schema 不一致。")
        return features

    @staticmethod
    def _observation_feature_row(
        observation: DisturbancePredictorObservation,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        measurement = observation.measured_disturbance
        row = (
            np.empty(len(FEATURE_NAMES), dtype=np.float32)
            if out is None
            else out
        )
        row[0:3] = measurement.omega_world
        row[3:6] = measurement.acc_world
        row[6:9] = observation.gravity_direction_torso
        row[9:21] = observation.lower_body_q
        row[21:33] = observation.lower_body_dq
        row[33:45] = observation.lower_body_policy_target
        row[45:48] = observation.runtime_command
        row[48:50] = observation.gait_phase_sin_cos
        return row

    @staticmethod
    def _require_neural_fields(
        observation: DisturbancePredictorObservation,
    ) -> None:
        if not isinstance(observation, DisturbancePredictorObservation):
            raise TypeError("update() 需要 DisturbancePredictorObservation。")
        missing = [
            name
            for name in (
                "gravity_direction_torso",
                "lower_body_q",
                "lower_body_dq",
                "lower_body_policy_target",
                "runtime_command",
                "gait_phase_sin_cos",
            )
            if getattr(observation, name) is None
        ]
        if missing:
            raise ValueError(f"neural predictor observation 缺少: {missing}")

    def _pre_inference_fallback_code(self) -> int:
        return self._FALLBACK_NONE

    def _reset_safety_diagnostics(self) -> None:
        self._input_abs_z_max = np.nan
        self._input_rms_z = np.nan
        self._prediction_abs_z_max = np.nan
        self._acc_correction_norm_max = np.nan
        self._alpha_correction_norm_max = np.nan

    def _zoh(self) -> DisturbanceHorizon:
        measurement = self._require_observation().measured_disturbance
        nodes = tuple(
            ZeroOrderHoldPredictor._copy_disturbance(measurement)
            for _ in range(self._horizon + 1)
        )
        intervals = tuple(
            ZeroOrderHoldPredictor._copy_disturbance(measurement)
            for _ in range(self._horizon)
        )
        return DisturbanceHorizon(nodes=nodes, intervals=intervals)

    def _prediction_world(
        self, prediction_heading: np.ndarray, heading_yaw_world: float
    ) -> np.ndarray:
        rotation_world_heading = rotation_z(heading_yaw_world)
        prediction_world = np.empty_like(prediction_heading)
        prediction_world[:, :3] = (
            prediction_heading[:, :3] @ rotation_world_heading.T
        )
        prediction_world[:, 3:] = (
            prediction_heading[:, 3:] @ rotation_world_heading.T
        )
        np.clip(
            prediction_world[:, :3],
            -self._acc_limit,
            self._acc_limit,
            out=prediction_world[:, :3],
        )
        np.clip(
            prediction_world[:, 3:],
            -self._alpha_limit,
            self._alpha_limit,
            out=prediction_world[:, 3:],
        )
        return prediction_world

    def _empty_neural_diagnostics(self) -> dict:
        return {
            "predictor_type": self._predictor_type,
            "prediction_mode": self._prediction_mode,
            "heading_ready": False,
            "heading_yaw_world": np.nan,
            "history_count": 0,
            "history_required": self._history_steps,
            "neural_inference_valid": False,
            "neural_inference_time": 0.0,
            "fallback_used": True,
            "fallback_code": self._FALLBACK_HISTORY,
            "fallback_reason": self._FALLBACK_REASONS[
                self._FALLBACK_HISTORY
            ],
            "safety_gate_triggered": False,
            "input_abs_z_max": np.nan,
            "input_rms_z": np.nan,
            "prediction_abs_z_max": np.nan,
            "acc_correction_norm_max": np.nan,
            "alpha_correction_norm_max": np.nan,
            "solver_gate_remaining": 0,
            "solver_failure_streak": 0,
            "timing_gate_remaining": 0,
            "one_step_prediction_valid": False,
        }


class NeuralDisturbancePredictor(_MLPPreviewPredictor):
    """Absolute interval acc/alpha MLP with measured omega/rotation ZOH."""

    _predictor_type = "neural"

    def __init__(self, **kwargs) -> None:
        super().__init__(prediction_mode="absolute", **kwargs)

    def predict(self, horizon: int, dt: float) -> DisturbanceHorizon:
        self._validate_request(horizon, dt)
        self._require_observation()
        (
            _,
            prediction_heading,
            template_diagnostics,
            fallback_code,
            inference_time,
        ) = self._reference_and_prediction()
        if prediction_heading is None:
            preview = self._zoh()
        else:
            prediction_world = self._prediction_world(
                prediction_heading,
                float(template_diagnostics["heading_yaw_world"]),
            )
            measurement = self._require_observation().measured_disturbance
            nodes = tuple(
                ZeroOrderHoldPredictor._copy_disturbance(measurement)
                for _ in range(self._horizon + 1)
            )
            intervals = tuple(
                DisturbanceInput(
                    acc_world=prediction_world[index, :3].copy(),
                    omega_world=np.asarray(measurement.omega_world).copy(),
                    alpha_world=prediction_world[index, 3:].copy(),
                    rot_world_body=np.asarray(
                        measurement.rot_world_body
                    ).copy(),
                )
                for index in range(self._horizon)
            )
            preview = DisturbanceHorizon(nodes=nodes, intervals=intervals)
        self._finish_diagnostics(
            template_diagnostics, fallback_code, inference_time
        )
        return preview

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "node_prediction": "measured_zero_order_hold",
            "interval_acc_alpha_prediction": "absolute_mlp",
            "interval_omega_rotation_prediction": "measured_zero_order_hold",
            "fallback": "measured_zero_order_hold",
        }


class ResidualHybridPredictor(_MLPPreviewPredictor):
    """Template preview plus a learned interval acc/alpha residual."""

    _predictor_type = "hybrid_residual"

    def __init__(self, **kwargs) -> None:
        super().__init__(prediction_mode="residual_template", **kwargs)

    def update(self, observation: DisturbancePredictorObservation) -> None:
        previous_correction_applied = self._last_correction_applied
        super().update(observation)
        if (
            self._safety_gate_enabled
            and observation.previous_control_interval_overrun is True
        ):
            self._timing_gate_remaining = max(
                self._timing_gate_remaining,
                self._control_overrun_cooldown_steps,
            )
        if observation.previous_mpc_success is True:
            self._solver_failure_streak = 0
        elif (
            observation.previous_mpc_success is False
            and previous_correction_applied
        ):
            self._solver_failure_streak += 1
        else:
            # Do not let a failed template probe create a template-only loop.
            self._solver_failure_streak = 0
        if self._safety_gate_enabled and (
            self._solver_failure_streak
            >= self._solver_failure_streak_threshold
        ):
            # One isolated QP failure is already handled by ArmMPC's braking
            # fallback.  Consecutive failures while residuals are active are
            # stronger causal evidence; perform a bounded template probe and
            # then allow the residual to be assessed again.
            self._solver_gate_remaining = max(
                self._solver_gate_remaining,
                self._solver_failure_cooldown_steps,
            )
            self._solver_failure_streak = 0

    def _pre_inference_fallback_code(self) -> int:
        if self._timing_gate_remaining > 0:
            self._timing_gate_remaining -= 1
            return self._FALLBACK_CONTROL_OVERRUN
        if self._solver_gate_remaining <= 0:
            return self._FALLBACK_NONE
        self._solver_gate_remaining -= 1
        return self._FALLBACK_SOLVER_QUALITY

    def predict(self, horizon: int, dt: float) -> DisturbanceHorizon:
        self._validate_request(horizon, dt)
        self._require_observation()
        (
            reference,
            residual_heading,
            template_diagnostics,
            fallback_code,
            inference_time,
        ) = self._reference_and_prediction()
        if residual_heading is None:
            preview = reference
        else:
            heading_yaw = float(template_diagnostics["heading_yaw_world"])
            rotation_world_heading = rotation_z(heading_yaw)
            # This prediction buffer is private to the current call.  Rotate
            # it in place, then apply it directly to the newly-created
            # template interval arrays.  No controller-visible preview is
            # mutated after return, and nodes/omega/rotation stay untouched.
            residual_heading[:, :3] = (
                residual_heading[:, :3] @ rotation_world_heading.T
            )
            residual_heading[:, 3:] = (
                residual_heading[:, 3:] @ rotation_world_heading.T
            )
            for index, baseline in enumerate(reference.intervals):
                np.add(
                    baseline.acc_world,
                    residual_heading[index, :3],
                    out=baseline.acc_world,
                )
                np.add(
                    baseline.alpha_world,
                    residual_heading[index, 3:],
                    out=baseline.alpha_world,
                )
                np.clip(
                    baseline.acc_world,
                    -self._acc_limit,
                    self._acc_limit,
                    out=baseline.acc_world,
                )
                np.clip(
                    baseline.alpha_world,
                    -self._alpha_limit,
                    self._alpha_limit,
                    out=baseline.alpha_world,
                )
            preview = reference
        self._finish_diagnostics(
            template_diagnostics, fallback_code, inference_time
        )
        self._last_correction_applied = fallback_code == self._FALLBACK_NONE
        return preview

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "node_prediction": "template_unchanged",
            "interval_acc_alpha_prediction": "template_plus_residual_mlp",
            "interval_omega_rotation_prediction": "template_unchanged",
            "fallback": "template_preview",
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
    if name not in {"template", "zoh", "neural", "hybrid_residual"}:
        raise ValueError(
            "disturbance_predictor 必须是 template、zoh、neural "
            "或 hybrid_residual。"
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
    template_kwargs = {
        "template_dir": template_dir,
        "variant": config.get(
            "mpc_disturbance_template", "fully_smoothed"
        ),
        "control_dt": control_dt,
        "horizon": horizon,
        "acc_limit": acc_limit,
        "alpha_limit": alpha_limit,
        "slow_bias_enabled": bool(
            config.get("mpc_disturbance_slow_bias_enabled", True)
        ),
        "slow_bias_time_constant": float(
            config.get("mpc_disturbance_slow_bias_time_constant", 0.4)
        ),
    }
    if name == "template":
        return TemplateDisturbancePredictor(**template_kwargs)

    checkpoint_key = (
        "neural_disturbance_model_path"
        if name == "neural"
        else "hybrid_residual_model_path"
    )
    checkpoint_default = (
        "disturbance_learning/artifacts/b2_mlp_baseline/mlp_checkpoint.pt"
        if name == "neural"
        else (
            "disturbance_learning/artifacts/hybrid_residual_mlp/"
            "residual_mlp_checkpoint.pt"
        )
    )
    checkpoint_path = str(config.get(checkpoint_key, checkpoint_default))
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(repo_dir, checkpoint_path)
    predictor_type = (
        NeuralDisturbancePredictor
        if name == "neural"
        else ResidualHybridPredictor
    )
    neural_kwargs = {
        "checkpoint_path": checkpoint_path,
        "template_reference": TemplateDisturbancePredictor(**template_kwargs),
        "control_dt": control_dt,
        "horizon": horizon,
        "acc_limit": acc_limit,
        "alpha_limit": alpha_limit,
    }
    if name == "hybrid_residual":
        neural_kwargs.update(
            {
                "safety_gate_enabled": bool(
                    config.get("hybrid_residual_safety_gate_enabled", True)
                ),
                "max_input_abs_z": float(
                    config.get("hybrid_residual_max_input_abs_z", 10.0)
                ),
                "max_input_rms_z": float(
                    config.get("hybrid_residual_max_input_rms_z", 2.0)
                ),
                "max_prediction_abs_z": float(
                    config.get("hybrid_residual_max_prediction_abs_z", 10.0)
                ),
                "max_acc_correction_norm": float(
                    config.get(
                        "hybrid_residual_max_acc_correction_norm", 15.0
                    )
                ),
                "max_alpha_correction_norm": float(
                    config.get(
                        "hybrid_residual_max_alpha_correction_norm", 55.0
                    )
                ),
                "solver_failure_cooldown_steps": int(
                    config.get(
                        "hybrid_residual_solver_failure_cooldown_steps", 1
                    )
                ),
                "solver_failure_streak_threshold": int(
                    config.get(
                        "hybrid_residual_solver_failure_streak_threshold", 2
                    )
                ),
                "control_overrun_cooldown_steps": int(
                    config.get(
                        "hybrid_residual_control_overrun_cooldown_steps", 1
                    )
                ),
            }
        )
    return predictor_type(
        **neural_kwargs,
    )
