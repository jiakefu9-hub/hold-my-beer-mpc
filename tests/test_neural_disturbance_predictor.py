"""Runtime semantic tests for absolute and residual MLP predictors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from disturbance_learning.dataset import (
    FEATURE_NAMES,
    HEADING_DEFINITION,
    TARGET_NAMES,
)
from disturbance_learning.mlp_model import MLPDisturbanceModel
from disturbance_model_new_heading.heading_template_utils import rotation_z
from disturbance_predictor import (
    DisturbancePredictorObservation,
    NeuralDisturbancePredictor,
    ResidualHybridPredictor,
    TemplateDisturbancePredictor,
    create_disturbance_predictor,
)
from kinematics_helper import DisturbanceInput


REPO_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_DIR / "disturbance_model_new_heading/templates_heading_interval"
CONTROL_DT = 0.006
HORIZON = 9


def _write_constant_checkpoint(
    path: Path, prediction_mode: str, target_mean: np.ndarray
) -> None:
    model = MLPDisturbanceModel(34, 50, (128, 128), 9, 6)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history_steps": 34,
            "feature_dim": 50,
            "hidden_sizes": [128, 128],
            "horizon": 9,
            "target_dim": 6,
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "prediction_mode": prediction_mode,
            "control_dt": CONTROL_DT,
            "heading_definition": HEADING_DEFINITION,
            "template_variant": "raw",
            "template_slow_bias_enabled": False,
            "template_slow_bias_time_constant": 0.4,
            "normalization": {
                "feature_mean": np.zeros(50, dtype=np.float32),
                "feature_std": np.ones(50, dtype=np.float32),
                "target_mean": np.asarray(target_mean, dtype=np.float32),
                "target_std": np.ones(6, dtype=np.float32),
            },
        },
        path,
    )


def _template() -> TemplateDisturbancePredictor:
    return TemplateDisturbancePredictor(
        template_dir=str(TEMPLATE_DIR),
        variant="raw",
        control_dt=CONTROL_DT,
        horizon=HORIZON,
        acc_limit=30.0,
        alpha_limit=40.0,
        slow_bias_enabled=False,
    )


def _observation(time_s: float, yaw: float = 0.25):
    index = int(round(time_s / CONTROL_DT))
    measurement = DisturbanceInput(
        acc_world=np.array([1.0, -2.0, 0.5]) + 0.001 * index,
        omega_world=np.array([0.2, -0.1, 0.3]) + 0.001 * index,
        alpha_world=np.array([-0.4, 0.6, 0.1]) + 0.001 * index,
        rot_world_body=rotation_z(yaw),
    )
    phase = (time_s / 0.8) % 1.0
    return DisturbancePredictorObservation(
        simulation_time=time_s,
        measured_disturbance=measurement,
        gravity_direction_torso=np.array([0.0, 0.0, -1.0]),
        lower_body_q=np.linspace(-0.2, 0.2, 12) + 0.001 * index,
        lower_body_dq=np.linspace(-0.5, 0.5, 12) - 0.001 * index,
        lower_body_policy_target=np.linspace(-0.1, 0.3, 12),
        runtime_command=np.array([0.4, 0.05, -0.02]),
        gait_phase_sin_cos=np.array(
            [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)]
        ),
    )


def _run_until_heading_ready(predictor):
    preview = None
    for index in range(135):
        predictor.update(_observation(index * CONTROL_DT))
        preview = predictor.predict(HORIZON, CONTROL_DT)
    return preview


class NeuralDisturbancePredictorTest(unittest.TestCase):
    def test_factory_selects_absolute_and_residual_checkpoint_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            absolute = Path(directory) / "absolute.pt"
            residual = Path(directory) / "residual.pt"
            _write_constant_checkpoint(absolute, "absolute", np.zeros(6))
            _write_constant_checkpoint(
                residual, "residual_template", np.zeros(6)
            )
            shared = {
                "mpc_disturbance_template_dir": str(TEMPLATE_DIR),
                "mpc_disturbance_template": "raw",
                "mpc_disturbance_slow_bias_enabled": False,
                "neural_disturbance_model_path": str(absolute),
                "hybrid_residual_model_path": str(residual),
            }
            neural = create_disturbance_predictor(
                {**shared, "disturbance_predictor": "neural"},
                repo_dir=str(REPO_DIR),
                control_dt=CONTROL_DT,
                horizon=HORIZON,
                acc_limit=30.0,
                alpha_limit=40.0,
            )
            hybrid = create_disturbance_predictor(
                {**shared, "disturbance_predictor": "hybrid_residual"},
                repo_dir=str(REPO_DIR),
                control_dt=CONTROL_DT,
                horizon=HORIZON,
                acc_limit=30.0,
                alpha_limit=40.0,
            )
            self.assertIsInstance(neural, NeuralDisturbancePredictor)
            self.assertIsInstance(hybrid, ResidualHybridPredictor)

    def test_absolute_mlp_replaces_only_interval_acc_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "absolute.pt"
            output_heading = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            _write_constant_checkpoint(checkpoint, "absolute", output_heading)
            predictor = NeuralDisturbancePredictor(
                checkpoint_path=str(checkpoint),
                template_reference=_template(),
                control_dt=CONTROL_DT,
                horizon=HORIZON,
                acc_limit=30.0,
                alpha_limit=40.0,
            )
            first = _observation(0.0)
            predictor.update(first)
            fallback = predictor.predict(HORIZON, CONTROL_DT)
            self.assertTrue(predictor.get_last_diagnostics()["fallback_used"])
            np.testing.assert_array_equal(
                fallback.intervals[0].acc_world,
                first.measured_disturbance.acc_world,
            )
            predictor.reset()
            preview = _run_until_heading_ready(predictor)
            diagnostics = predictor.get_last_diagnostics()
            self.assertTrue(diagnostics["neural_inference_valid"])
            self.assertFalse(diagnostics["fallback_used"])

            heading_yaw = float(diagnostics["heading_yaw_world"])
            rotation = rotation_z(heading_yaw)
            expected_acc = rotation @ output_heading[:3]
            expected_alpha = rotation @ output_heading[3:]
            final_observation = _observation(134 * CONTROL_DT)
            measured = final_observation.measured_disturbance
            final_features = predictor._build_history_features(heading_yaw)[-1]
            expected_features = np.concatenate(
                (
                    rotation_z(-heading_yaw) @ measured.omega_world,
                    rotation_z(-heading_yaw) @ measured.acc_world,
                    final_observation.gravity_direction_torso,
                    final_observation.lower_body_q,
                    final_observation.lower_body_dq,
                    final_observation.lower_body_policy_target,
                    final_observation.runtime_command,
                    final_observation.gait_phase_sin_cos,
                )
            )
            np.testing.assert_allclose(
                final_features, expected_features, rtol=1e-6, atol=1e-6
            )
            for interval in preview.intervals:
                np.testing.assert_allclose(interval.acc_world, expected_acc)
                np.testing.assert_allclose(interval.alpha_world, expected_alpha)
                np.testing.assert_array_equal(
                    interval.omega_world, measured.omega_world
                )
                np.testing.assert_array_equal(
                    interval.rot_world_body, measured.rot_world_body
                )
            for node in preview.nodes:
                np.testing.assert_array_equal(node.acc_world, measured.acc_world)
                np.testing.assert_array_equal(
                    node.alpha_world, measured.alpha_world
                )
                np.testing.assert_array_equal(node.omega_world, measured.omega_world)
                np.testing.assert_array_equal(
                    node.rot_world_body, measured.rot_world_body
                )

    def test_hybrid_preserves_template_nodes_omega_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "residual.pt"
            residual_heading = np.array(
                [0.01, -0.02, 0.03, -0.04, 0.05, -0.06]
            )
            _write_constant_checkpoint(
                checkpoint, "residual_template", residual_heading
            )
            hybrid = ResidualHybridPredictor(
                checkpoint_path=str(checkpoint),
                template_reference=_template(),
                control_dt=CONTROL_DT,
                horizon=HORIZON,
                acc_limit=30.0,
                alpha_limit=40.0,
            )
            baseline = _template()
            hybrid_preview = baseline_preview = None
            for index in range(135):
                observation = _observation(index * CONTROL_DT)
                hybrid.update(observation)
                baseline.update(observation)
                hybrid_preview = hybrid.predict(HORIZON, CONTROL_DT)
                baseline_preview = baseline.predict(HORIZON, CONTROL_DT)

            diagnostics = hybrid.get_last_diagnostics()
            self.assertTrue(diagnostics["neural_inference_valid"])
            rotation = rotation_z(float(diagnostics["heading_yaw_world"]))
            residual_acc = rotation @ residual_heading[:3]
            residual_alpha = rotation @ residual_heading[3:]
            for hybrid_node, baseline_node in zip(
                hybrid_preview.nodes, baseline_preview.nodes
            ):
                for name in (
                    "acc_world",
                    "omega_world",
                    "alpha_world",
                    "rot_world_body",
                ):
                    np.testing.assert_array_equal(
                        getattr(hybrid_node, name), getattr(baseline_node, name)
                    )
            for hybrid_interval, baseline_interval in zip(
                hybrid_preview.intervals, baseline_preview.intervals
            ):
                np.testing.assert_allclose(
                    hybrid_interval.acc_world,
                    baseline_interval.acc_world + residual_acc,
                )
                np.testing.assert_allclose(
                    hybrid_interval.alpha_world,
                    baseline_interval.alpha_world + residual_alpha,
                )
                np.testing.assert_array_equal(
                    hybrid_interval.omega_world,
                    baseline_interval.omega_world,
                )
                np.testing.assert_array_equal(
                    hybrid_interval.rot_world_body,
                    baseline_interval.rot_world_body,
                )

    def test_history_gap_forces_safe_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "absolute.pt"
            _write_constant_checkpoint(
                checkpoint, "absolute", np.zeros(6)
            )
            predictor = NeuralDisturbancePredictor(
                checkpoint_path=str(checkpoint),
                template_reference=_template(),
                control_dt=CONTROL_DT,
                horizon=HORIZON,
                acc_limit=30.0,
                alpha_limit=40.0,
            )
            _run_until_heading_ready(predictor)
            predictor.update(_observation(136 * CONTROL_DT))
            preview = predictor.predict(HORIZON, CONTROL_DT)
            diagnostics = predictor.get_last_diagnostics()
            self.assertEqual(diagnostics["fallback_code"], 3)
            self.assertTrue(diagnostics["fallback_used"])
            measured = _observation(136 * CONTROL_DT).measured_disturbance
            np.testing.assert_array_equal(
                preview.intervals[0].acc_world, measured.acc_world
            )


if __name__ == "__main__":
    unittest.main()
