import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from disturbance_learning.full_task_protocol import (
    DEFAULT_FULL_TASK_PROTOCOL,
    is_valid_rotation_batch,
)
from disturbance_model_new_heading.heading_template_utils import rotation_z
from disturbance_predictor import (
    DisturbancePredictorObservation,
    FullTaskPredictorError,
    FullTaskTemplatePredictor,
    create_disturbance_predictor,
    resolve_disturbance_predictor_name,
)
from kinematics_helper import DisturbanceInput


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = (
    ROOT
    / "disturbance_learning/data/full_task_template_v2/20260815_162850"
)
TEMPLATE_PATH = ASSET_DIR / "full_task_template.npz"
MANIFEST_PATH = ASSET_DIR / "full_task_template_manifest.json"
TEMPLATE_SHA256 = (
    "d4a0109adcff696936ef96160976161833ff9a7a7531e2e5d7ad9e50c10e17d4"
)
MANIFEST_SHA256 = (
    "7f313057a1ba3748da2b2322a39366b6553bff13f9dbba123534765ccfe9cd76"
)


def _measurement(task_time: float, yaw: float | None = None) -> DisturbanceInput:
    angle = 0.15 * np.sin(0.7 * task_time) if yaw is None else float(yaw)
    return DisturbanceInput(
        acc_world=np.array([101.0 + task_time, -2.0, 3.0]),
        omega_world=np.array([0.2, -0.1, 0.05 + task_time]),
        alpha_world=np.array([-0.3, 0.4, -0.5]),
        rot_world_body=rotation_z(angle),
    )


def _observation(
    simulation_time: float, task_time: float, yaw: float | None = None
) -> DisturbancePredictorObservation:
    return DisturbancePredictorObservation(
        simulation_time=simulation_time,
        measured_disturbance=_measurement(task_time, yaw),
    )


def _predictor(**overrides) -> FullTaskTemplatePredictor:
    kwargs = {
        "template_path": str(TEMPLATE_PATH),
        "manifest_path": str(MANIFEST_PATH),
        "expected_sha256": TEMPLATE_SHA256,
        "expected_manifest_sha256": MANIFEST_SHA256,
        "repo_dir": str(ROOT),
        "control_dt": 0.006,
        "horizon": 9,
        "expected_schema_version": "full_task_template_v2",
        "expected_heading_frame_version": "full_task_continuous_heading_v2",
    }
    kwargs.update(overrides)
    return FullTaskTemplatePredictor(**kwargs)


class FullTaskTemplatePredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with np.load(TEMPLATE_PATH, allow_pickle=False) as source:
            cls.template = {name: source[name].copy() for name in source.files}

    def test_pinned_assets_are_unchanged(self) -> None:
        self.assertEqual(
            hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest(),
            TEMPLATE_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            MANIFEST_SHA256,
        )

    def test_online_predictor_does_not_import_offline_builder(self) -> None:
        source = (ROOT / "disturbance_predictor.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("full_task_template_builder", source)
        self.assertIn("full_task_template_asset", source)

    def test_first_frame_is_valid_and_node0_is_measured(self) -> None:
        predictor = _predictor()
        observation = _observation(12.5, 0.0, yaw=0.42)
        predictor.update(observation)
        preview = predictor.predict(9, 0.006)
        diagnostics = predictor.get_last_diagnostics()
        self.assertEqual(len(preview.nodes), 10)
        self.assertEqual(len(preview.intervals), 9)
        self.assertEqual(diagnostics["task_time"], 0.0)
        self.assertTrue(diagnostics["heading_ready"])
        self.assertEqual(
            diagnostics["heading_source"], "causal_prefix"
        )
        self.assertFalse(diagnostics["fallback_used"])
        np.testing.assert_array_equal(
            preview.nodes[0].acc_world,
            observation.measured_disturbance.acc_world,
        )
        np.testing.assert_array_equal(
            preview.nodes[0].rot_world_body,
            observation.measured_disturbance.rot_world_body,
        )
        self.assertFalse(
            np.array_equal(
                preview.nodes[0].acc_world,
                diagnostics["template_acc_world"],
            )
        )
        self.assertTrue(
            np.all(is_valid_rotation_batch(
                np.stack([node.rot_world_body for node in preview.nodes])
            ))
        )

    def test_exact_boundary_rows_and_last_horizon(self) -> None:
        predictor = _predictor()
        wanted = {1066: 6.396, 1067: 6.402, 1333: 7.998}
        previews = {}
        diagnostics = {}
        epoch_simulation_time = 4.0
        for index in range(DEFAULT_FULL_TASK_PROTOCOL.headline_anchor_count):
            task_time = index * 0.006
            predictor.update(
                _observation(epoch_simulation_time + task_time, task_time)
            )
            if index in wanted:
                previews[index] = predictor.predict(9, 0.006)
                diagnostics[index] = predictor.get_last_diagnostics()
        for index, expected_time in wanted.items():
            self.assertEqual(diagnostics[index]["template_anchor_index"], index)
            self.assertAlmostEqual(
                diagnostics[index]["task_time"], expected_time, places=12
            )
            self.assertFalse(diagnostics[index]["fallback_used"])
            rotation_world_heading = rotation_z(
                diagnostics[index]["heading_yaw_world"]
            )
            expected_nodes = (
                self.template["nodes_acceleration_mean"][index]
                @ rotation_world_heading.T
            )
            expected_intervals = (
                self.template["intervals_acceleration_mean"][index]
                @ rotation_world_heading.T
            )
            np.testing.assert_allclose(
                np.stack(
                    [node.acc_world for node in previews[index].nodes[1:]]
                ),
                expected_nodes[1:],
                atol=1e-12,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                np.stack(
                    [item.acc_world for item in previews[index].intervals]
                ),
                expected_intervals,
                atol=1e-12,
                rtol=0.0,
            )
        self.assertLess(wanted[1066], DEFAULT_FULL_TASK_PROTOCOL.stop_time)
        self.assertGreater(wanted[1066] + 9 * 0.006, 6.4)
        self.assertGreater(wanted[1067], DEFAULT_FULL_TASK_PROTOCOL.stop_time)
        self.assertAlmostEqual(7.998 + 9 * 0.006, 8.052, places=12)

    def test_terminal_tail_is_explicit_hold_not_fallback(self) -> None:
        predictor = _predictor()
        for index in range(1335):
            task_time = index * 0.006
            predictor.update(_observation(task_time, task_time))
        preview = predictor.predict(9, 0.006)
        diagnostics = predictor.get_last_diagnostics()
        self.assertAlmostEqual(diagnostics["task_time"], 8.004, places=12)
        self.assertTrue(diagnostics["terminal_hold_used"])
        self.assertFalse(diagnostics["fallback_used"])
        for node in preview.nodes[1:]:
            np.testing.assert_array_equal(
                node.acc_world, preview.nodes[0].acc_world
            )

    def test_reset_backward_gap_off_grid_and_out_of_range_fail_closed(self) -> None:
        predictor = _predictor()
        with self.assertRaises(FullTaskPredictorError) as caught:
            predictor.predict(9, 0.006)
        self.assertEqual(caught.exception.reason_code, "update_required")

        predictor.reset()
        predictor.update(_observation(20.0, 0.0))
        with self.assertRaises(FullTaskPredictorError) as caught:
            predictor.update(_observation(20.0, 0.0))
        self.assertEqual(
            caught.exception.reason_code, "task_time_backward_or_repeated"
        )

        predictor.reset()
        predictor.update(_observation(30.0, 0.0))
        with self.assertRaises(FullTaskPredictorError) as caught:
            predictor.update(_observation(30.012, 0.012))
        self.assertEqual(caught.exception.reason_code, "missing_anchor")

        predictor.reset()
        predictor.update(_observation(40.0, 0.0))
        with self.assertRaises(FullTaskPredictorError) as caught:
            predictor.update(_observation(40.005, 0.005))
        self.assertEqual(caught.exception.reason_code, "task_time_not_on_anchor")

        predictor.reset()
        predictor.update(_observation(50.0, 0.0))
        predictor._last_simulation_time = 50.0 + 8.058
        predictor._last_anchor_index = 1343
        with self.assertRaises(FullTaskPredictorError) as caught:
            predictor.update(_observation(50.064 + 8.0, 8.064))
        self.assertEqual(caught.exception.reason_code, "task_time_out_of_range")

        predictor.reset()
        predictor.update(_observation(60.0, 0.0))
        predictor.predict(9, 0.006)
        self.assertEqual(predictor.get_last_diagnostics()["task_time"], 0.0)

    def test_invalid_manifest_and_checksum_fail_closed_at_load(self) -> None:
        with self.assertRaises(FullTaskPredictorError) as caught:
            _predictor(expected_sha256="0" * 64)
        self.assertEqual(caught.exception.reason_code, "template_checksum_mismatch")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["collection"]["protocol"]["version"] = "wrong_protocol"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(FullTaskPredictorError) as caught:
                _predictor(
                    manifest_path=str(path),
                    expected_manifest_sha256=digest,
                )
        self.assertEqual(caught.exception.reason_code, "manifest_protocol_mismatch")

    def test_factory_adds_only_the_new_mode(self) -> None:
        config = {
            "disturbance_predictor": "full_task_template",
            "full_task_template_path": str(TEMPLATE_PATH.relative_to(ROOT)),
            "full_task_template_manifest_path": str(
                MANIFEST_PATH.relative_to(ROOT)
            ),
            "full_task_template_sha256": TEMPLATE_SHA256,
            "full_task_template_manifest_sha256": MANIFEST_SHA256,
            "full_task_template_schema_version": "full_task_template_v2",
            "full_task_heading_frame_version": "full_task_continuous_heading_v2",
        }
        self.assertEqual(
            resolve_disturbance_predictor_name(config), "full_task_template"
        )
        predictor = create_disturbance_predictor(
            config,
            repo_dir=str(ROOT),
            control_dt=0.006,
            horizon=9,
            acc_limit=200.0,
            alpha_limit=1000.0,
        )
        self.assertIsInstance(predictor, FullTaskTemplatePredictor)
        with self.assertRaises(FullTaskPredictorError) as caught:
            create_disturbance_predictor(
                {"disturbance_predictor": "full_task_template"},
                repo_dir=str(ROOT),
                control_dt=0.006,
                horizon=9,
                acc_limit=200.0,
                alpha_limit=1000.0,
            )
        self.assertEqual(caught.exception.reason_code, "configuration_missing")


if __name__ == "__main__":
    unittest.main()
