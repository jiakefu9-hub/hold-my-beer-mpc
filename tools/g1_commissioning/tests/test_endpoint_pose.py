"""Offline geometry checks; run with the existing g1_mpc environment."""
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from endpoint_pose import EndpointModel, nearest_imu, rotation


class EndpointPoseTest(unittest.TestCase):
    def test_relative_fk_and_imu_composition_match_full_model(self):
        model = EndpointModel()
        q = np.array([-.07, .02, -.03, -.17, .04,
                      -.06, -.01, .03, -.16, -.02, .4, 0, 0])
        relative = model.relative(q)
        # Pose the entire model arbitrarily, including nonzero waist yaw.
        # Relative FK must cancel the base/waist transform, not apply it twice.
        model.data.qpos[:3] = [1.0, -2.0, .8]
        model.data.qpos[3:7] = np.array([.9, .1, -.2, .3]) / np.linalg.norm([.9, .1, -.2, .3])
        mujoco.mj_kinematics(model.model, model.data)
        R_WB = model.data.site_xmat[model.imu_id].reshape(3, 3).copy()
        p_WB = model.data.site_xpos[model.imu_id].copy()
        for side, (p_BE, R_BE) in relative.items():
            site = model.sites[side]
            np.testing.assert_allclose(R_WB @ p_BE + p_WB, model.data.site_xpos[site], atol=1e-12)
            np.testing.assert_allclose(R_WB @ R_BE, model.data.site_xmat[site].reshape(3, 3), atol=1e-12)

    def test_zero_frame_and_freshness(self):
        model = EndpointModel()
        pose = model.poses(np.zeros(13), [1, 0, 0, 0])
        for side in ("left", "right"):
            # Wrist translation slope is NOT the bottle orientation.
            self.assertLess(pose[side]["bottle_z_tilt_from_vertical_deg"], .02)
            site = model.sites[side]
            np.testing.assert_allclose(model.model.site_pos[site], [0, 0, 0])
            np.testing.assert_allclose(rotation(pose[side]["orientation_W"]["quaternion_wxyz"]),
                                       model.relative(np.zeros(13))[side][1], atol=1e-12)
        samples = [{"monotonic_ns": 1000000000}]
        self.assertIsNone(nearest_imu(samples, [1000000000], 1200000000, 125)[0])
        self.assertIsNotNone(nearest_imu(samples, [1000000000], 1100000000, 125)[0])

    def test_body_tilt_does_not_follow_world_imu_rotation(self):
        model = EndpointModel()
        q = np.zeros(13)
        upright = model.poses(q, [1, 0, 0, 0])
        fallen = model.poses(q, [np.sqrt(.5), np.sqrt(.5), 0, 0])
        for side in ("left", "right"):
            self.assertEqual(upright[side]["orientation_B"], fallen[side]["orientation_B"])
            self.assertAlmostEqual(upright[side]["bottle_z_tilt_from_body_z_deg"],
                                   fallen[side]["bottle_z_tilt_from_body_z_deg"])
            self.assertGreater(fallen[side]["bottle_z_tilt_from_vertical_deg"], 89.9)



if __name__ == "__main__":
    unittest.main()
