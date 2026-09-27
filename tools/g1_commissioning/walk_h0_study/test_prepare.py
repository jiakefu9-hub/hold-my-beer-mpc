"""Synthetic frame/causality checks; no robot or experiment files needed."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from prepare import prepare_h0, audit


class PreparationTests(unittest.TestCase):
    def make_data(self):
        t = np.arange(-1., 21.1, .002)
        imu = np.zeros((len(t), 15))
        q = Rotation.from_euler('z', np.pi/2).as_quat()
        imu[:, 2:6] = q[[3, 0, 1, 2]]
        imu[:, 6:9] = [0, 0, np.pi/2]
        return dict(imu=imu, imu_t=t, low_t=t, epoch_ns=np.array(0),
                    world_acc=np.tile([0., 2., 0.], (len(t), 1)),
                    world_omega=np.tile([0., 1., 0.], (len(t), 1)),
                    q=np.zeros((len(t), 35)), dq=np.zeros((len(t), 35)))

    def test_h0_rotation_is_fixed_yaw_not_body_following(self):
        d = self.make_data()
        p = prepare_h0(d, dict(yaw0_rad=np.pi/2, monotonic_ns=5_000_001_000))
        np.testing.assert_allclose(p['acc_raw'], np.tile([2., 0., 0.], (len(p['t']), 1)), atol=1e-14)
        np.testing.assert_allclose(p['omega'], np.tile([1., 0., 0.], (len(p['t']), 1)), atol=1e-14)
        np.testing.assert_allclose(p['rpy'], 0, atol=1e-14)
        self.assertEqual(p['qf'].shape[1], 12)
        self.assertGreater(float(p['reference_available_s']), 5)

    def test_late_data_cannot_change_earlier_filtered_features(self):
        d = self.make_data()
        ref = dict(yaw0_rad=np.pi/2, monotonic_ns=5_000_001_000)
        before = prepare_h0(d, ref)
        d['world_acc'][d['imu_t'] > 10] = [100, 100, 100]
        after = prepare_h0(d, ref)
        mask = before['t'] < 10
        np.testing.assert_array_equal(before['acc'][mask], after['acc'][mask])

    def test_asof_missing_past_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.asof(np.array([1.]), np.array([2.]), np.array([0.]))


if __name__ == '__main__':
    unittest.main()
