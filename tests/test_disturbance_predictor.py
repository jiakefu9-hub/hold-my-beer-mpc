"""Regression tests for the B0 disturbance-predictor interface.

These tests deliberately exercise the production phase template.  They freeze
the timing and numerical behavior that the adapter must preserve before any
learned predictor is introduced.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from disturbance_predictor import (  # noqa: E402
    DisturbancePredictorObservation,
    TemplateDisturbancePredictor,
    ZeroOrderHoldPredictor,
    create_disturbance_predictor,
)
from kinematics_helper import DisturbanceInput  # noqa: E402
from sim_support import PhaseDisturbancePredictor  # noqa: E402


CONTROL_DT = 0.006
HORIZON = 9
TEMPLATE_DIR = (
    REPO_ROOT
    / "disturbance_model_new_heading"
    / "templates_heading_interval"
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _measured_disturbance(time_s: float) -> DisturbanceInput:
    """Deterministic, valid W-frame measurements distinct from the template."""
    phase = 2.0 * math.pi * time_s / 0.8
    return DisturbanceInput(
        acc_world=np.array(
            [0.7 + 0.2 * math.sin(phase), -0.4, 0.15 * math.cos(phase)],
            dtype=np.float64,
        ),
        omega_world=np.array(
            [0.05, -0.03 + 0.02 * math.cos(phase), 0.11],
            dtype=np.float64,
        ),
        alpha_world=np.array(
            [0.12 * math.sin(phase), -0.08, 0.04], dtype=np.float64
        ),
        rot_world_body=_rotation_z(0.18 + 0.04 * math.sin(phase)),
    )


def _legacy_predictor() -> PhaseDisturbancePredictor:
    return PhaseDisturbancePredictor(
        template_dir=str(TEMPLATE_DIR),
        variant="raw",
        control_dt=CONTROL_DT,
        horizon=HORIZON,
        acc_limit=30.0,
        alpha_limit=40.0,
        slow_bias_enabled=True,
        slow_bias_time_constant=0.4,
    )


def _adapter_predictor() -> TemplateDisturbancePredictor:
    return TemplateDisturbancePredictor(
        template_dir=str(TEMPLATE_DIR),
        variant="raw",
        control_dt=CONTROL_DT,
        horizon=HORIZON,
        acc_limit=30.0,
        alpha_limit=40.0,
        slow_bias_enabled=True,
        slow_bias_time_constant=0.4,
    )


class DisturbancePredictorRegressionTest(unittest.TestCase):
    def assert_disturbance_equal(self, actual, expected) -> None:
        for name in (
            "acc_world",
            "omega_world",
            "alpha_world",
            "rot_world_body",
        ):
            np.testing.assert_array_equal(
                np.asarray(getattr(actual, name)),
                np.asarray(getattr(expected, name)),
                err_msg=name,
            )

    def assert_horizon_equal(self, actual, expected) -> None:
        self.assertEqual(len(actual.nodes), HORIZON + 1)
        self.assertEqual(len(actual.intervals), HORIZON)
        self.assertEqual(len(expected.nodes), HORIZON + 1)
        self.assertEqual(len(expected.intervals), HORIZON)
        for actual_item, expected_item in zip(actual.nodes, expected.nodes):
            self.assert_disturbance_equal(actual_item, expected_item)
        for actual_item, expected_item in zip(
            actual.intervals, expected.intervals
        ):
            self.assert_disturbance_equal(actual_item, expected_item)

    def assert_diagnostics_equal(self, actual: dict, expected: dict) -> None:
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            actual_value = actual[name]
            expected_value = expected[name]
            if isinstance(actual_value, np.ndarray):
                np.testing.assert_array_equal(
                    actual_value, expected_value, err_msg=name
                )
            elif isinstance(actual_value, float) and math.isnan(actual_value):
                self.assertTrue(math.isnan(expected_value), msg=name)
            else:
                self.assertEqual(actual_value, expected_value, msg=name)

    def assert_node0_matches_measurement(self, actual, measurement) -> None:
        for name in ("acc_world", "omega_world", "alpha_world"):
            np.testing.assert_array_equal(
                np.asarray(getattr(actual, name)),
                np.asarray(getattr(measurement, name)),
                err_msg=name,
            )
        # The legacy template obtains node[0] orientation as R_meas @ I;
        # this is semantically the measured pose but has machine-epsilon
        # roundoff for non-identity rotations.
        np.testing.assert_allclose(
            actual.rot_world_body,
            measurement.rot_world_body,
            rtol=0.0,
            atol=1e-15,
            err_msg="rot_world_body",
        )

    def predict_adapter(self, predictor, time_s: float, disturbance):
        predictor.update(
            DisturbancePredictorObservation(
                simulation_time=time_s,
                measured_disturbance=disturbance,
            )
        )
        return predictor.predict(HORIZON, CONTROL_DT)

    def test_template_adapter_matches_legacy_through_zoh_activation_and_bias(
        self,
    ) -> None:
        legacy = _legacy_predictor()
        adapter = _adapter_predictor()

        times = np.arange(0.0, 0.804 + 1e-12, CONTROL_DT)
        self.assertAlmostEqual(float(times[-1]), 0.804, places=12)
        for time_s in times:
            measurement = _measured_disturbance(float(time_s))
            legacy_preview = legacy.predict(float(time_s), measurement)
            adapter_preview = self.predict_adapter(
                adapter, float(time_s), measurement
            )

            self.assert_horizon_equal(adapter_preview, legacy_preview)
            self.assert_diagnostics_equal(
                adapter.get_last_diagnostics(), legacy.get_last_diagnostics()
            )
            self.assert_node0_matches_measurement(
                adapter_preview.nodes[0], measurement
            )
            if time_s < 0.8:
                self.assertFalse(legacy.get_last_diagnostics()["heading_ready"])

        self.assertTrue(legacy.get_last_diagnostics()["heading_ready"])
        self.assertGreater(
            np.linalg.norm(legacy.runtime_state()["slow_bias_acc_world"]), 0.0
        )

    def test_template_adapter_matches_legacy_off_grid_interpolation(self) -> None:
        legacy = _legacy_predictor()
        adapter = _adapter_predictor()
        for time_s in np.arange(0.0, 0.804 + 1e-12, CONTROL_DT):
            measurement = _measured_disturbance(float(time_s))
            legacy.predict(float(time_s), measurement)
            self.predict_adapter(adapter, float(time_s), measurement)

        off_grid_time = 0.805
        self.assertIsNone(legacy._aligned_start_bin(off_grid_time))
        measurement = _measured_disturbance(off_grid_time)
        legacy_preview = legacy.predict(off_grid_time, measurement)
        adapter_preview = self.predict_adapter(adapter, off_grid_time, measurement)
        self.assert_horizon_equal(adapter_preview, legacy_preview)
        self.assert_diagnostics_equal(
            adapter.get_last_diagnostics(), legacy.get_last_diagnostics()
        )

    def test_template_adapter_reset_restarts_first_cycle_zoh(self) -> None:
        adapter = _adapter_predictor()
        for time_s in np.arange(0.0, 0.804 + 1e-12, CONTROL_DT):
            self.predict_adapter(
                adapter,
                float(time_s),
                _measured_disturbance(float(time_s)),
            )
        self.assertTrue(adapter.get_last_diagnostics()["heading_ready"])

        adapter.reset()
        measurement = _measured_disturbance(0.0)
        preview = self.predict_adapter(adapter, 0.0, measurement)
        self.assertFalse(adapter.get_last_diagnostics()["heading_ready"])
        for item in preview.nodes:
            self.assert_disturbance_equal(item, measurement)
        for item in preview.intervals:
            self.assert_disturbance_equal(item, measurement)

    def test_zero_order_hold_interface(self) -> None:
        predictor = ZeroOrderHoldPredictor(
            control_dt=CONTROL_DT, horizon=HORIZON
        )
        measurement = _measured_disturbance(0.123)
        predictor.update(
            DisturbancePredictorObservation(0.123, measurement)
        )
        preview = predictor.predict(HORIZON, CONTROL_DT)
        self.assertEqual(len(preview.nodes), HORIZON + 1)
        self.assertEqual(len(preview.intervals), HORIZON)
        for item in preview.nodes:
            self.assert_disturbance_equal(item, measurement)
        for item in preview.intervals:
            self.assert_disturbance_equal(item, measurement)

    def test_factory_supports_new_and_legacy_selector_configuration(self) -> None:
        template = create_disturbance_predictor(
            {
                "disturbance_predictor": "template",
                "mpc_disturbance_template_dir": (
                    "disturbance_model_new_heading/templates_heading_interval"
                ),
                "mpc_disturbance_template": "raw",
                "mpc_disturbance_slow_bias_enabled": True,
                "mpc_disturbance_slow_bias_time_constant": 0.4,
            },
            repo_dir=str(REPO_ROOT),
            control_dt=CONTROL_DT,
            horizon=HORIZON,
            acc_limit=30.0,
            alpha_limit=40.0,
        )
        legacy_zoh = create_disturbance_predictor(
            {"mpc_disturbance_feedforward_enabled": False},
            repo_dir=str(REPO_ROOT),
            control_dt=CONTROL_DT,
            horizon=HORIZON,
            acc_limit=30.0,
            alpha_limit=40.0,
        )
        explicit_zoh = create_disturbance_predictor(
            {"disturbance_predictor": "zoh"},
            repo_dir=str(REPO_ROOT),
            control_dt=CONTROL_DT,
            horizon=HORIZON,
            acc_limit=30.0,
            alpha_limit=40.0,
        )
        self.assertIsInstance(template, TemplateDisturbancePredictor)
        self.assertIsInstance(legacy_zoh, ZeroOrderHoldPredictor)
        self.assertIsInstance(explicit_zoh, ZeroOrderHoldPredictor)

    def test_archived_learned_modes_are_rejected(self) -> None:
        for name in ("neural", "hybrid_residual"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                create_disturbance_predictor(
                    {"disturbance_predictor": name},
                    repo_dir=str(REPO_ROOT),
                    control_dt=CONTROL_DT,
                    horizon=HORIZON,
                    acc_limit=30.0,
                    alpha_limit=40.0,
                )

    def test_runtime_observation_contract_is_minimal(self) -> None:
        self.assertEqual(
            set(DisturbancePredictorObservation.__dataclass_fields__),
            {"simulation_time", "measured_disturbance"},
        )


if __name__ == "__main__":
    unittest.main()
