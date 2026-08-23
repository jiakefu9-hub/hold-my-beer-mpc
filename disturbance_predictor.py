"""Small, predictor-neutral boundary for MPC disturbance previews.

The MPC-facing wire format intentionally remains ``DisturbanceInput`` and
``DisturbanceHorizon``.  This module only removes the main loop's knowledge of
the concrete predictor implementation; it does not alter preview semantics.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from disturbance_template.full_task_protocol import (
    DEFAULT_FULL_TASK_PROTOCOL,
    FullTaskCausalHeadingFrame,
    FullTaskContinuousHeadingFrame,
    is_valid_rotation_batch,
)
from disturbance_template.full_task_template_asset import (
    TEMPLATE_SCHEMA_VERSION,
    TEMPLATE_SCHEMA_VERSION_V2,
    load_npz_arrays,
    sha256_file,
    validate_full_task_template,
)
from disturbance_types import DisturbanceHorizon, DisturbanceInput


@dataclass(frozen=True)
class DisturbancePredictorObservation:
    """Causal information available at one MPC update instant.

    The final runtime predictors need only the absolute runtime timestamp and
    the measured disturbance.  Locomotion-policy features belong to the
    archived learned-predictor route and are deliberately not part of the
    shared online contract.
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


class FullTaskPredictorError(RuntimeError):
    """Fail-closed error with a stable machine-readable reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = str(reason_code)


class FullTaskTemplatePredictor(_FixedPreviewPredictor):
    """Exact absolute-task-time baseline built from the accepted T1 template."""

    _FALLBACK_NONE = 0
    _TERMINAL_HOLD_CODE = 100

    def __init__(
        self,
        *,
        template_path: str,
        manifest_path: str,
        expected_sha256: str,
        expected_manifest_sha256: str,
        repo_dir: str,
        control_dt: float,
        horizon: int,
        expected_schema_version: str = TEMPLATE_SCHEMA_VERSION,
        expected_heading_frame_version: str = FullTaskCausalHeadingFrame.DEFINITION_VERSION,
    ) -> None:
        super().__init__(control_dt=control_dt, horizon=horizon)
        self._protocol = DEFAULT_FULL_TASK_PROTOCOL
        if not np.isclose(
            self._control_dt,
            self._protocol.mpc_dt,
            atol=1e-12,
            rtol=0.0,
        ):
            raise FullTaskPredictorError(
                "control_dt_mismatch",
                "full-task template requires the frozen 6 ms anchor grid",
            )
        if self._horizon != self._protocol.horizon:
            raise FullTaskPredictorError(
                "horizon_mismatch",
                "full-task template requires the frozen 9-interval horizon",
            )
        self._repo_dir = Path(repo_dir).expanduser().resolve()
        if expected_schema_version not in {
            TEMPLATE_SCHEMA_VERSION,
            TEMPLATE_SCHEMA_VERSION_V2,
        }:
            raise FullTaskPredictorError(
                "configured_schema_invalid", "unsupported configured template schema"
            )
        expected_frame_for_schema = (
            FullTaskContinuousHeadingFrame.DEFINITION_VERSION
            if expected_schema_version == TEMPLATE_SCHEMA_VERSION_V2
            else FullTaskCausalHeadingFrame.DEFINITION_VERSION
        )
        if expected_heading_frame_version != expected_frame_for_schema:
            raise FullTaskPredictorError(
                "configured_heading_frame_invalid",
                "configured heading-frame version disagrees with template schema",
            )
        self._expected_schema_version = str(expected_schema_version)
        self._expected_heading_frame_version = str(expected_heading_frame_version)
        self._template_path = Path(template_path).expanduser().resolve()
        self._manifest_path = Path(manifest_path).expanduser().resolve()
        expected = self._validate_expected_sha256(
            expected_sha256, "expected_checksum_invalid"
        )
        expected_manifest = self._validate_expected_sha256(
            expected_manifest_sha256, "expected_manifest_checksum_invalid"
        )
        if expected == expected_manifest:
            raise FullTaskPredictorError(
                "configured_checksums_invalid",
                "template and manifest cannot have the same checksum",
            )
        self._expected_sha256 = expected
        self._expected_manifest_sha256 = expected_manifest
        self._template_sha256 = self._sha256_required(
            self._template_path, "template_missing"
        )
        if self._template_sha256 != self._expected_sha256:
            raise FullTaskPredictorError(
                "template_checksum_mismatch",
                f"configured {self._expected_sha256}, actual {self._template_sha256}",
            )
        self._manifest_sha256 = self._sha256_required(
            self._manifest_path, "manifest_missing"
        )
        if self._manifest_sha256 != self._expected_manifest_sha256:
            raise FullTaskPredictorError(
                "manifest_checksum_mismatch",
                "configured manifest SHA256 does not match the file",
            )
        try:
            manifest = json.loads(
                self._manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise FullTaskPredictorError("manifest_invalid", str(exc)) from exc
        try:
            self._validate_manifest(manifest)
        except FullTaskPredictorError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise FullTaskPredictorError(
                "manifest_invalid", str(exc)
            ) from exc
        try:
            template = load_npz_arrays(self._template_path)
        except (OSError, ValueError) as exc:
            raise FullTaskPredictorError(
                "template_load_failed", str(exc)
            ) from exc
        try:
            self._template_validation = validate_full_task_template(
                template,
                self._protocol,
                expected_schema_version=self._expected_schema_version,
            )
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            raise FullTaskPredictorError(
                "template_schema_or_shape_mismatch", str(exc)
            ) from exc
        if (
            str(np.asarray(template["protocol_version"]).item())
            != self._protocol.protocol_version
        ):
            raise FullTaskPredictorError(
                "template_protocol_mismatch",
                "NPZ protocol_version is not the frozen direct-step protocol",
            )
        self._anchor_task_time = np.asarray(
            template["anchor_task_time"], dtype=np.float64
        )
        self._nodes = {
            "acc": np.asarray(
                template["nodes_acceleration_mean"], dtype=np.float64
            ),
            "omega": np.asarray(
                template["nodes_angular_velocity_mean"], dtype=np.float64
            ),
            "alpha": np.asarray(
                template["nodes_angular_acceleration_mean"], dtype=np.float64
            ),
            "rotation": np.asarray(
                template["nodes_rotation_heading_mean"], dtype=np.float64
            ),
        }
        self._intervals = {
            "acc": np.asarray(
                template["intervals_acceleration_mean"], dtype=np.float64
            ),
            "omega": np.asarray(
                template["intervals_angular_velocity_mean"], dtype=np.float64
            ),
            "alpha": np.asarray(
                template["intervals_angular_acceleration_mean"],
                dtype=np.float64,
            ),
            "rotation": np.asarray(
                template["intervals_rotation_heading_mean"], dtype=np.float64
            ),
        }
        heading_type = (
            FullTaskContinuousHeadingFrame
            if self._expected_heading_frame_version
            == FullTaskContinuousHeadingFrame.DEFINITION_VERSION
            else FullTaskCausalHeadingFrame
        )
        self._heading = heading_type(self._protocol)
        self._epoch_origin_simulation_time: Optional[float] = None
        self._last_simulation_time: Optional[float] = None
        self._last_anchor_index: Optional[int] = None
        self._task_time = np.nan
        self._heading_state = None
        self._previous_one_step_prediction: Optional[
            dict[str, np.ndarray]
        ] = None
        self._pending_one_step = self._empty_one_step()
        self._last_diagnostics = self._empty_diagnostics()

    @staticmethod
    def _validate_expected_sha256(value: str, reason_code: str) -> str:
        expected = str(value).strip().lower()
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise FullTaskPredictorError(
                reason_code,
                "configured SHA256 must contain exactly 64 hex characters",
            )
        return expected

    @staticmethod
    def _sha256_required(path: Path, missing_code: str) -> str:
        if not path.is_file():
            raise FullTaskPredictorError(missing_code, f"missing {path}")
        return sha256_file(path)

    def _validate_manifest(self, manifest: dict) -> None:
        if manifest.get("template_schema_version") != self._expected_schema_version:
            raise FullTaskPredictorError(
                "manifest_schema_mismatch",
                "unexpected template_schema_version",
            )
        validation = manifest.get("template_validation", {})
        expected_validation = {
            "anchor_count": self._protocol.headline_anchor_count,
            "horizon": self._protocol.horizon,
            "node0_online_policy": (
                "always_replace_with_current_measurement"
            ),
            "rotation_valid": True,
            "smoothing": "none",
        }
        if any(
            validation.get(name) != value
            for name, value in expected_validation.items()
        ):
            raise FullTaskPredictorError(
                "manifest_validation_mismatch",
                "manifest template_validation is not the accepted T1 contract",
            )
        manifest_heading = validation.get(
            "heading_frame_version",
            FullTaskCausalHeadingFrame.DEFINITION_VERSION,
        )
        collection_heading = manifest.get("collection", {}).get(
            "heading_frame_version",
            FullTaskCausalHeadingFrame.DEFINITION_VERSION,
        )
        if (
            manifest_heading != self._expected_heading_frame_version
            or collection_heading != self._expected_heading_frame_version
        ):
            raise FullTaskPredictorError(
                "manifest_heading_frame_mismatch",
                "manifest heading-frame version differs from configured version",
            )
        protocol = manifest.get("collection", {}).get("protocol", {})
        if (
            protocol.get("name") != self._protocol.protocol_name
            or protocol.get("version") != self._protocol.protocol_version
            or int(protocol.get("horizon", -1)) != self._protocol.horizon
            or not np.isclose(
                float(protocol.get("record_end", np.nan)),
                self._protocol.record_end,
                atol=1e-12,
            )
        ):
            raise FullTaskPredictorError(
                "manifest_protocol_mismatch",
                "manifest protocol is not full_task_direct_step_v1",
            )
        asset = manifest.get("template", {})
        if str(asset.get("sha256", "")).lower() != self._expected_sha256:
            raise FullTaskPredictorError(
                "manifest_checksum_mismatch",
                "manifest template checksum differs from the pinned checksum",
            )
        portable = Path(str(asset.get("path", "")))
        portable_resolved = (
            (self._repo_dir / portable).resolve()
            if not portable.is_absolute()
            else portable.resolve()
        )
        absolute_value = asset.get("absolute_path")
        absolute_resolved = (
            None
            if not absolute_value
            else Path(str(absolute_value)).expanduser().resolve()
        )
        if self._template_path not in {
            portable_resolved,
            absolute_resolved,
        }:
            raise FullTaskPredictorError(
                "manifest_path_mismatch",
                "manifest does not identify the configured template path",
            )

    def reset(self) -> None:
        self._observation = None
        self._heading.reset()
        self._epoch_origin_simulation_time = None
        self._last_simulation_time = None
        self._last_anchor_index = None
        self._task_time = np.nan
        self._heading_state = None
        self._previous_one_step_prediction = None
        self._pending_one_step = self._empty_one_step()
        self._last_diagnostics = self._empty_diagnostics()

    def _fail(self, reason_code: str, message: str) -> None:
        diagnostics = self._empty_diagnostics()
        diagnostics.update(
            {
                "simulation_time": (
                    np.nan
                    if self._last_simulation_time is None
                    else self._last_simulation_time
                ),
                "task_time": self._task_time,
                "fail_closed": True,
                "fail_closed_reason_code": reason_code,
                "fallback_used": True,
                "fallback_code": -1,
                "fallback_reason": reason_code,
            }
        )
        self._last_diagnostics = diagnostics
        raise FullTaskPredictorError(reason_code, message)

    @staticmethod
    def _validate_measurement(measurement: DisturbanceInput) -> None:
        if not isinstance(measurement, DisturbanceInput):
            raise FullTaskPredictorError(
                "measurement_invalid",
                "measured_disturbance must be a DisturbanceInput",
            )
        for name, shape in (
            ("acc_world", (3,)),
            ("omega_world", (3,)),
            ("alpha_world", (3,)),
            ("rot_world_body", (3, 3)),
        ):
            value = np.asarray(getattr(measurement, name), dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise FullTaskPredictorError(
                    "measurement_invalid",
                    f"{name} must be finite with shape {shape}",
                )
        if not bool(
            is_valid_rotation_batch(
                np.asarray(measurement.rot_world_body, dtype=np.float64)
            )
        ):
            raise FullTaskPredictorError(
                "measurement_rotation_invalid",
                "measured rotation is not SO(3)",
            )

    def update(self, observation: DisturbancePredictorObservation) -> None:
        if not isinstance(observation, DisturbancePredictorObservation):
            raise TypeError(
                "update() requires DisturbancePredictorObservation"
            )
        try:
            self._validate_measurement(observation.measured_disturbance)
        except FullTaskPredictorError as exc:
            self._fail(exc.reason_code, str(exc))
        simulation_time = float(observation.simulation_time)
        if not np.isfinite(simulation_time):
            self._fail(
                "simulation_time_invalid",
                "simulation time must be finite",
            )
        if self._epoch_origin_simulation_time is None:
            self._epoch_origin_simulation_time = simulation_time
            task_time = 0.0
            anchor_index = 0
        else:
            if (
                self._last_simulation_time is not None
                and simulation_time <= self._last_simulation_time + 1e-12
            ):
                self._fail(
                    "task_time_backward_or_repeated",
                    "simulation time must advance after reset",
                )
            task_time = simulation_time - self._epoch_origin_simulation_time
            if task_time > self._protocol.record_end + 1e-12:
                self._fail(
                    "task_time_out_of_range",
                    "task time exceeds the frozen record tail",
                )
            anchor_float = task_time / self._protocol.mpc_dt
            anchor_index = int(round(anchor_float))
            if not np.isclose(
                anchor_float, anchor_index, atol=1e-9, rtol=0.0
            ):
                self._fail(
                    "task_time_not_on_anchor",
                    "queries require the exact 6 ms grid",
                )
            if (
                self._last_anchor_index is not None
                and anchor_index != self._last_anchor_index + 1
            ):
                self._fail(
                    "missing_anchor",
                    "every 6 ms anchor must be supplied exactly once",
                )
        self._task_time = (
            0.0 if abs(task_time) <= 1e-12 else float(task_time)
        )
        self._last_simulation_time = simulation_time
        self._last_anchor_index = anchor_index
        try:
            self._heading_state = self._heading.update(
                self._task_time,
                np.asarray(
                    observation.measured_disturbance.rot_world_body,
                    dtype=np.float64,
                ),
            )
        except ValueError as exc:
            self._fail("causal_h_update_failed", str(exc))
        self._pending_one_step = self._one_step_errors(
            observation.measured_disturbance
        )
        super().update(observation)

    def predict(self, horizon: int, dt: float) -> DisturbanceHorizon:
        if int(horizon) != self._horizon:
            self._fail(
                "request_horizon_mismatch",
                "requested horizon differs from the pinned template",
            )
        if not np.isclose(
            float(dt), self._control_dt, atol=1e-12, rtol=0.0
        ):
            self._fail(
                "request_dt_mismatch",
                "requested dt differs from the pinned 6 ms grid",
            )
        if self._observation is None:
            self._fail(
                "update_required",
                "update must establish task time and H before predict",
            )
        observation = self._observation
        if self._heading_state is None or self._last_anchor_index is None:
            self._fail(
                "update_required",
                "update must establish task time and H before predict",
            )
        if self._task_time >= self._protocol.headline_end - 1e-12:
            preview = self._terminal_hold(observation.measured_disturbance)
            self._last_diagnostics = self._build_diagnostics(
                template_node0=None, terminal_hold=True
            )
            self._previous_one_step_prediction = None
            return preview
        anchor_index = self._last_anchor_index
        if not 0 <= anchor_index < len(self._anchor_task_time):
            self._fail(
                "template_anchor_missing",
                "headline anchor is absent from the template",
            )
        if not np.isclose(
            self._anchor_task_time[anchor_index],
            self._task_time,
            atol=1e-12,
            rtol=0.0,
        ):
            self._fail(
                "template_anchor_time_mismatch",
                "template row does not exactly match task time",
            )
        rotation_world_heading = np.asarray(
            self._heading_state.rotation_world_heading, dtype=np.float64
        )
        node_acc = (
            self._nodes["acc"][anchor_index]
            @ rotation_world_heading.T
        )
        node_omega = (
            self._nodes["omega"][anchor_index]
            @ rotation_world_heading.T
        )
        node_alpha = (
            self._nodes["alpha"][anchor_index]
            @ rotation_world_heading.T
        )
        node_rotation = np.einsum(
            "ij,njk->nik",
            rotation_world_heading,
            self._nodes["rotation"][anchor_index],
        )
        interval_acc = (
            self._intervals["acc"][anchor_index]
            @ rotation_world_heading.T
        )
        interval_omega = (
            self._intervals["omega"][anchor_index]
            @ rotation_world_heading.T
        )
        interval_alpha = (
            self._intervals["alpha"][anchor_index]
            @ rotation_world_heading.T
        )
        interval_rotation = np.einsum(
            "ij,njk->nik",
            rotation_world_heading,
            self._intervals["rotation"][anchor_index],
        )
        if not np.all(
            is_valid_rotation_batch(node_rotation)
        ) or not np.all(is_valid_rotation_batch(interval_rotation)):
            self._fail(
                "world_rotation_invalid",
                "H-to-world transform produced a non-SO(3) rotation",
            )
        template_node0 = DisturbanceInput(
            acc_world=node_acc[0].copy(),
            omega_world=node_omega[0].copy(),
            alpha_world=node_alpha[0].copy(),
            rot_world_body=node_rotation[0].copy(),
        )
        measurement = observation.measured_disturbance
        nodes = [ZeroOrderHoldPredictor._copy_disturbance(measurement)]
        nodes.extend(
            DisturbanceInput(
                acc_world=node_acc[index].copy(),
                omega_world=node_omega[index].copy(),
                alpha_world=node_alpha[index].copy(),
                rot_world_body=node_rotation[index].copy(),
            )
            for index in range(1, self._horizon + 1)
        )
        intervals = tuple(
            DisturbanceInput(
                acc_world=interval_acc[index].copy(),
                omega_world=interval_omega[index].copy(),
                alpha_world=interval_alpha[index].copy(),
                rot_world_body=interval_rotation[index].copy(),
            )
            for index in range(self._horizon)
        )
        preview = DisturbanceHorizon(
            nodes=tuple(nodes), intervals=intervals
        )
        self._last_diagnostics = self._build_diagnostics(
            template_node0=template_node0, terminal_hold=False
        )
        self._remember_one_step(preview.nodes[1])
        return preview

    def _terminal_hold(
        self, measurement: DisturbanceInput
    ) -> DisturbanceHorizon:
        return DisturbanceHorizon(
            nodes=tuple(
                ZeroOrderHoldPredictor._copy_disturbance(measurement)
                for _ in range(self._horizon + 1)
            ),
            intervals=tuple(
                ZeroOrderHoldPredictor._copy_disturbance(measurement)
                for _ in range(self._horizon)
            ),
        )

    def _one_step_errors(
        self, measurement: DisturbanceInput
    ) -> dict:
        if self._previous_one_step_prediction is None:
            return self._empty_one_step()
        previous = self._previous_one_step_prediction
        return {
            "one_step_prediction_valid": True,
            "one_step_acc_error": (
                np.asarray(measurement.acc_world) - previous["acc_world"]
            ),
            "one_step_omega_error": (
                np.asarray(measurement.omega_world) - previous["omega_world"]
            ),
            "one_step_alpha_error": (
                np.asarray(measurement.alpha_world) - previous["alpha_world"]
            ),
            "one_step_rotation_error_angle": self._rotation_error_angle(
                np.asarray(measurement.rot_world_body),
                previous["rot_world_body"],
            ),
        }

    def _remember_one_step(self, prediction: DisturbanceInput) -> None:
        self._previous_one_step_prediction = {
            name: np.asarray(
                getattr(prediction, name), dtype=np.float64
            ).copy()
            for name in (
                "acc_world",
                "omega_world",
                "alpha_world",
                "rot_world_body",
            )
        }

    @staticmethod
    def _rotation_error_angle(
        measured: np.ndarray, predicted: np.ndarray
    ) -> float:
        relative = measured @ predicted.T
        cosine = np.clip(
            (np.trace(relative) - 1.0) * 0.5, -1.0, 1.0
        )
        return float(np.arccos(cosine))

    @staticmethod
    def _empty_one_step() -> dict:
        return {
            "one_step_prediction_valid": False,
            "one_step_acc_error": np.full(3, np.nan),
            "one_step_omega_error": np.full(3, np.nan),
            "one_step_alpha_error": np.full(3, np.nan),
            "one_step_rotation_error_angle": np.nan,
        }

    def _build_diagnostics(
        self,
        *,
        template_node0: Optional[DisturbanceInput],
        terminal_hold: bool,
    ) -> dict:
        measurement = self._require_observation().measured_disturbance
        nan_vector = np.full(3, np.nan)
        ready = template_node0 is not None
        return {
            "predictor_type": "full_task_template",
            "simulation_time": float(self._last_simulation_time),
            "task_time": float(self._task_time),
            "task_epoch_origin_simulation_time": float(
                self._epoch_origin_simulation_time
            ),
            "template_anchor_index": int(self._last_anchor_index),
            "template_query_mode": (
                "terminal_measurement_hold"
                if terminal_hold
                else "exact_absolute_task_time"
            ),
            "heading_ready": True,
            "heading_yaw_world": float(self._heading_state.yaw_world),
            "heading_concentration": float(
                self._heading_state.concentration
            ),
            "heading_source": self._heading_state.source,
            "phase": (
                float(self._task_time) / self._protocol.gait_period
            )
            % 1.0,
            "template_acc_world": (
                nan_vector.copy()
                if not ready
                else np.asarray(template_node0.acc_world).copy()
            ),
            "template_omega_world": (
                nan_vector.copy()
                if not ready
                else np.asarray(template_node0.omega_world).copy()
            ),
            "template_alpha_world": (
                nan_vector.copy()
                if not ready
                else np.asarray(template_node0.alpha_world).copy()
            ),
            "anchor_acc_error": (
                nan_vector.copy()
                if not ready
                else np.asarray(measurement.acc_world)
                - template_node0.acc_world
            ),
            "anchor_omega_error": (
                nan_vector.copy()
                if not ready
                else np.asarray(measurement.omega_world)
                - template_node0.omega_world
            ),
            "anchor_alpha_error": (
                nan_vector.copy()
                if not ready
                else np.asarray(measurement.alpha_world)
                - template_node0.alpha_world
            ),
            "anchor_rotation_error_angle": (
                np.nan
                if not ready
                else self._rotation_error_angle(
                    np.asarray(measurement.rot_world_body),
                    template_node0.rot_world_body,
                )
            ),
            "slow_bias_acc_world": np.zeros(3),
            "slow_bias_omega_world": np.zeros(3),
            "slow_bias_alpha_world": np.zeros(3),
            "fallback_used": False,
            "fallback_code": self._FALLBACK_NONE,
            "fallback_reason": "none",
            "terminal_hold_used": bool(terminal_hold),
            "terminal_hold_code": (
                self._TERMINAL_HOLD_CODE if terminal_hold else 0
            ),
            "terminal_hold_reason": (
                "after_headline_measurement_zoh"
                if terminal_hold
                else "none"
            ),
            "fail_closed": False,
            "fail_closed_reason_code": "none",
            "one_step_prediction_valid": (
                False
                if terminal_hold
                else self._pending_one_step["one_step_prediction_valid"]
            ),
            "one_step_acc_error": (
                nan_vector.copy()
                if terminal_hold
                else self._pending_one_step["one_step_acc_error"].copy()
            ),
            "one_step_omega_error": (
                nan_vector.copy()
                if terminal_hold
                else self._pending_one_step["one_step_omega_error"].copy()
            ),
            "one_step_alpha_error": (
                nan_vector.copy()
                if terminal_hold
                else self._pending_one_step["one_step_alpha_error"].copy()
            ),
            "one_step_rotation_error_angle": (
                np.nan
                if terminal_hold
                else self._pending_one_step[
                    "one_step_rotation_error_angle"
                ]
            ),
        }

    def _empty_diagnostics(self) -> dict:
        nan_vector = np.full(3, np.nan)
        return {
            "predictor_type": "full_task_template",
            "simulation_time": np.nan,
            "task_time": np.nan,
            "task_epoch_origin_simulation_time": np.nan,
            "template_anchor_index": -1,
            "template_query_mode": "not_initialized",
            "heading_ready": False,
            "heading_yaw_world": np.nan,
            "heading_concentration": np.nan,
            "heading_source": "none",
            "phase": np.nan,
            "template_acc_world": nan_vector.copy(),
            "template_omega_world": nan_vector.copy(),
            "template_alpha_world": nan_vector.copy(),
            "anchor_acc_error": nan_vector.copy(),
            "anchor_omega_error": nan_vector.copy(),
            "anchor_alpha_error": nan_vector.copy(),
            "anchor_rotation_error_angle": np.nan,
            "slow_bias_acc_world": np.zeros(3),
            "slow_bias_omega_world": np.zeros(3),
            "slow_bias_alpha_world": np.zeros(3),
            "fallback_used": False,
            "fallback_code": self._FALLBACK_NONE,
            "fallback_reason": "none",
            "terminal_hold_used": False,
            "terminal_hold_code": 0,
            "terminal_hold_reason": "none",
            "fail_closed": False,
            "fail_closed_reason_code": "none",
            **self._empty_one_step(),
        }

    def metadata(self) -> dict:
        return {
            "enabled": True,
            "predictor_type": "full_task_template",
            "baseline_scope": (
                "fixed direct-step task; stop time 6.4 s is known in advance"
            ),
            "path": str(self._template_path),
            "sha256": self._template_sha256,
            "manifest_path": str(self._manifest_path),
            "manifest_sha256": self._manifest_sha256,
            "template_schema_version": self._expected_schema_version,
            "protocol_version": self._protocol.protocol_version,
            "anchor_mode": (
                "exact_absolute_task_time_6ms_no_interpolation"
            ),
            "node0_policy": "always_current_measurement",
            "future_source": (
                "T1 template nodes 1..9 and all intervals"
            ),
            "heading_definition": self._expected_heading_frame_version,
            "heading_filter": "none",
            "template_smoothing": "none",
            "slow_bias_enabled": False,
            "terminal_behavior": (
                "measurement ZOH after [0,8.0) headline"
            ),
            "template_validation": dict(self._template_validation),
        }

    def get_last_diagnostics(self, copy_data: bool = True) -> dict:
        if not copy_data:
            return self._last_diagnostics
        return {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in self._last_diagnostics.items()
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
    if name not in {"template", "full_task_template", "zoh"}:
        raise ValueError(
            "disturbance_predictor 必须是 template、full_task_template 或 zoh。"
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
    if name == "full_task_template":
        required_keys = (
            "full_task_template_path",
            "full_task_template_manifest_path",
            "full_task_template_sha256",
            "full_task_template_manifest_sha256",
            "full_task_template_schema_version",
            "full_task_heading_frame_version",
        )
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise FullTaskPredictorError(
                "configuration_missing",
                "missing explicit full-task asset configuration: "
                + ", ".join(missing),
            )
        template_path = str(config["full_task_template_path"])
        manifest_path = str(config["full_task_template_manifest_path"])
        if not os.path.isabs(template_path):
            template_path = os.path.join(repo_dir, template_path)
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(repo_dir, manifest_path)
        return FullTaskTemplatePredictor(
            template_path=template_path,
            manifest_path=manifest_path,
            expected_sha256=str(config["full_task_template_sha256"]),
            expected_manifest_sha256=str(
                config["full_task_template_manifest_sha256"]
            ),
            repo_dir=repo_dir,
            control_dt=control_dt,
            horizon=horizon,
            expected_schema_version=str(
                config["full_task_template_schema_version"]
            ),
            expected_heading_frame_version=str(
                config["full_task_heading_frame_version"]
            ),
        )

    template_dir = str(
        config.get(
            "mpc_disturbance_template_dir",
            "phase_disturbance_template/templates_heading_interval",
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
    return TemplateDisturbancePredictor(**template_kwargs)
