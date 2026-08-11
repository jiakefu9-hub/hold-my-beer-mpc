"""Tests for the matched-payload runtime MJCF."""

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco

from payload_model import create_modeled_payload_mjcf


REPO_DIR = Path(__file__).resolve().parents[1]
SCENE_PATH = REPO_DIR / "resources/g1_description/scene.xml"


class ModeledPayloadMjcfTest(unittest.TestCase):
    def test_variant_loads_and_preserves_source(self):
        source_text = SCENE_PATH.read_text(encoding="utf-8")
        variant = create_modeled_payload_mjcf(
            SCENE_PATH, body_name="right_bottle", added_mass_kg=0.010
        )
        runtime_root = variant.scene_path.parent
        try:
            model = mujoco.MjModel.from_xml_path(str(variant.scene_path))
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "right_bottle"
            )
            self.assertGreaterEqual(body_id, 0)
            self.assertAlmostEqual(model.body_mass[body_id], 0.260, places=12)
            self.assertAlmostEqual(variant.nominal_mass_kg, 0.250, places=12)
            self.assertAlmostEqual(variant.modeled_mass_kg, 0.260, places=12)

            scene_root = ET.parse(variant.scene_path).getroot()
            robot_path = runtime_root / scene_root.find("include").get("file")
            compiler = ET.parse(robot_path).getroot().find("compiler")
            self.assertTrue(Path(compiler.get("meshdir")).is_absolute())
            self.assertEqual(
                SCENE_PATH.read_text(encoding="utf-8"), source_text
            )
        finally:
            variant.cleanup()
        self.assertFalse(runtime_root.exists())

    def test_missing_body_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected one payload body"):
            create_modeled_payload_mjcf(
                SCENE_PATH, body_name="missing_payload", added_mass_kg=0.005
            )


if __name__ == "__main__":
    unittest.main()
