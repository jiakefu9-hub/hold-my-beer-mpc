"""Regression tests for the shared right-arm controller assembly boundary."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import unittest

import numpy as np
import yaml

import right_arm_control_setup as shared_setup
import sim_support


REPO_ROOT = Path(__file__).resolve().parents[1]


def _top_level_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _definition_ast_hashes(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Assign):
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names = [node.name]
        for name in names:
            if name in {
                "RIGHT_ARM_JOINT_NAMES",
                "ArmControllerSetup",
                "create_arm_controller",
            }:
                payload = ast.dump(
                    node,
                    annotate_fields=True,
                    include_attributes=False,
                )
                result[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return result


class SharedRightArmControlSetupTest(unittest.TestCase):
    def test_extracted_definitions_match_frozen_pre_extraction_ast(self):
        self.assertEqual(
            _definition_ast_hashes(REPO_ROOT / "right_arm_control_setup.py"),
            {
                "RIGHT_ARM_JOINT_NAMES": (
                    "c568828bee7fc2e9b0333f0ada7e67fe8c304b0532eec36dc42a136c5473e23a"
                ),
                "ArmControllerSetup": (
                    "9c7d4467d08b862cb55061d5f9aab1b2da1aa773d78ef7c8d739f1d8be4b6c17"
                ),
                "create_arm_controller": (
                    "ecc54c558e531df14aeb67fffdb764f0931812c917208829915e50f1ac311a73"
                ),
            },
        )

    def test_simulation_reexports_the_single_shared_definitions(self):
        self.assertIs(
            sim_support.RIGHT_ARM_JOINT_NAMES,
            shared_setup.RIGHT_ARM_JOINT_NAMES,
        )
        self.assertIs(
            sim_support.ArmControllerSetup,
            shared_setup.ArmControllerSetup,
        )
        self.assertIs(
            sim_support.create_arm_controller,
            shared_setup.create_arm_controller,
        )

    def test_hardware_shadow_no_longer_imports_simulation_adapter(self):
        hardware_path = REPO_ROOT / "right_arm_runtime/hardware_shadow.py"
        self.assertNotIn("sim_support", _top_level_import_roots(hardware_path))
        self.assertIn(
            "right_arm_control_setup",
            _top_level_import_roots(hardware_path),
        )

    def test_shared_setup_has_no_platform_adapter_import(self):
        setup_path = REPO_ROOT / "right_arm_control_setup.py"
        import_roots = _top_level_import_roots(setup_path)
        self.assertNotIn("sim_support", import_roots)
        self.assertNotIn("right_arm_runtime", import_roots)
        self.assertNotIn("mujoco", import_roots)
        self.assertNotIn("matplotlib", import_roots)

    def test_default_mpc_setup_preserves_frozen_config_and_metadata(self):
        with (REPO_ROOT / "configs/g1.yaml").open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        default_q = np.asarray(config["arm_waist_target"], dtype=np.float64)[6:11]
        control_dt = float(config["simulation_dt"]) * int(
            config["arm_control_decimation"]
        )

        setup = shared_setup.create_arm_controller(
            config,
            "mpc",
            default_q,
            control_dt,
        )

        self.assertEqual(setup.policy_type, "ArmMPCPolicy")
        self.assertTrue(setup.acceleration_controller)
        self.assertEqual(setup.execution_max_abs_qacc, 10.0)
        self.assertEqual(
            setup.execution_safety_rescue_passes,
            int(config["mpc_execution_safety_rescue_passes"]),
        )
        self.assertEqual(
            setup.execution_hold_last_safe,
            bool(config["mpc_execution_hold_last_safe"]),
        )
        self.assertEqual(setup.ddq_saturation_limit, float(config["mpc_max_ddq"]))

        metadata = setup.metadata["mpc_config"]
        self.assertEqual(metadata["horizon"], int(config["mpc_horizon"]))
        self.assertEqual(metadata["control_dt"], control_dt)
        self.assertEqual(metadata["q_ee_acc"], config["mpc_q_ee_acc"])
        self.assertEqual(metadata["q_ee_alpha"], config["mpc_q_ee_alpha"])
        self.assertEqual(metadata["q_ee_omega"], config["mpc_q_ee_omega"])
        self.assertEqual(metadata["max_ddq"], float(config["mpc_max_ddq"]))
        self.assertEqual(metadata["solver"], "OSQP")
        self.assertEqual(
            metadata["base_prediction"],
            "continuous_heading_to_world_absolute_task_template_v2",
        )
        self.assertEqual(
            metadata["forward_dynamics_max_abs_qacc"],
            float(config["ddq_execution_max_abs_qacc"]),
        )


if __name__ == "__main__":
    unittest.main()
