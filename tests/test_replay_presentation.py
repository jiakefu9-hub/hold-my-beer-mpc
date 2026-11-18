"""Run with the g1_mpc Python; no simulator stepping or desktop is needed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np
from PIL import Image

VIEWER_DIR = Path(__file__).resolve().parents[1]/'tools/experiments/thesis_walk13'
sys.path.insert(0, str(VIEWER_DIR))
from presentation import CameraPreset, load_camera, save_camera
import viewer


class ReplayPresentationTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body>
          </worldbody></mujoco>''')
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()

    def test_camera_file_reproduces_framing_on_another_root_position(self):
        first = CameraPreset(azimuth=92., elevation=-6., distance=.85, fovy=38.,
                             offset_x=.12, offset_y=-.07, height=1.25,
                             width_px=1024, height_px=768)
        self.data.qpos[:2] = (2., 3.)
        first.apply(self.camera, self.model, self.data)
        saved = CameraPreset.capture(self.camera, self.model, self.data, (1024, 768))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'camera.json'
            save_camera(path, saved)
            reloaded = load_camera(path)
        self.data.qpos[:2] = (9., -4.)
        reloaded.apply(self.camera, self.model, self.data)
        np.testing.assert_allclose(self.camera.lookat, [9.12, -4.07, 1.25])
        self.assertEqual(self.camera.azimuth, 92.)
        self.assertEqual(self.model.vis.global_.fovy, 38.)
        self.assertEqual((reloaded.width_px, reloaded.height_px), (1024, 768))

    def test_follow_keeps_manual_pan_and_height(self):
        self.data.qpos[:2] = (4., 2.)
        viewer.camera_at(self.camera, self.data, offset=(.3, -.4, 1.4))
        np.testing.assert_allclose(self.camera.lookat, [4.3, 1.6, 1.4])
        self.data.qpos[:2] = (5., 3.)
        viewer.camera_at(self.camera, self.data, offset=(.3, -.4, 1.4))
        np.testing.assert_allclose(self.camera.lookat, [5.3, 2.6, 1.4])

    def test_default_capture_size_and_aspect_lock(self):
        preset = CameraPreset()
        self.assertEqual((preset.width_px, preset.height_px), (1000, 1000))
        self.assertEqual(preset.aspect_terms(), (1, 1))
        self.assertEqual(CameraPreset(width_px=1920, height_px=1080).aspect_terms(), (16, 9))
        self.assertIsNone(CameraPreset(lock_aspect=False).aspect_terms())

    def test_old_camera_files_keep_their_saved_window_dimensions(self):
        payload = dict(version=1, camera=dict(width_px=1280, height_px=720))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'camera.json'
            path.write_text(json.dumps(payload))
            preset = load_camera(path)
            self.assertEqual(preset.aspect_terms(), (16, 9))
            preset.lock_aspect = False
            save_camera(path, preset)
            self.assertFalse(load_camera(path).lock_aspect)

    def test_bad_camera_does_not_modify_current_view(self):
        original = self.camera.distance
        for changes in ({'distance': -.1}, {'azimuth': float('nan')},
                        {'fovy': 0.}, {'width_px': 500.5}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    CameraPreset(**changes).apply(self.camera, self.model, self.data)
                self.assertEqual(self.camera.distance, original)

    def test_typed_time_reaches_frame_despite_float_roundoff(self):
        frames = dict(time=np.array([9.220, np.nextafter(9.222, np.inf)]),
                      qpos=np.tile(self.data.qpos, (2, 1)),
                      qvel=np.zeros((2, self.model.nv)))
        frames['qpos'][1, 0] = 1.5
        viewer.set_frame(self.model, self.data, frames, 9.222)
        self.assertAlmostEqual(self.data.time, 9.222)
        self.assertEqual(self.data.qpos[0], 1.5)

    def test_invalid_preset_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'camera.json'
            for payload in ({'version': 1}, {'version': 9, 'camera': {}}, []):
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    load_camera(path)

    def test_png_orientation_and_case_time_metadata(self):
        # OpenGL rows start at the bottom; the PNG must put the red row on top.
        pixels = np.zeros((4, 6, 3), dtype=np.uint8)
        pixels[-1, :, 0] = 255
        case = dict(title='PID', replay='replays/D01.npz', replay_sha256='abc')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'capture.png'
            viewer.save_screenshot(path, pixels, 'D01', case, 5.252, CameraPreset())
            with Image.open(path) as im:
                self.assertEqual(im.size, (6, 4))
                self.assertEqual(im.getpixel((0, 0)), (255, 0, 0))
            metadata = json.loads(path.with_suffix('.json').read_text())
        self.assertEqual(metadata['case'], 'D01')
        self.assertEqual(metadata['task_time_s'], 5.252)
        self.assertEqual(metadata['image_size_px'], [6, 4])


if __name__ == '__main__':
    unittest.main()
