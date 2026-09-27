"""Small offline integrity tests for the saved forecasting calculation."""
import unittest
import numpy as np
import methods as m


class ForecastIntegrity(unittest.TestCase):
    def dataset(self):
        t = np.arange(-.5, 21.002, .002)
        y = np.repeat(t[:, None], 12, axis=1)
        return dict(t=t, y=y, acc_raw=y[:, :3], qf=y.copy(), dqf=y.copy(),
                    hip=np.sin(2*np.pi*t))

    def test_future_labels_end_at_horizon(self):
        d = self.dataset()
        idx = np.array([3000])
        result = m.targets(d, idx)
        for j, ms in enumerate(m.HORIZONS_MS):
            self.assertAlmostEqual(result[0, j, 0], d["t"][idx[0]]+(ms-4)/1000)
            self.assertAlmostEqual(result[0, j, 3], d["t"][idx[0]]+ms/1000)

    def test_features_do_not_see_future(self):
        d = self.dataset()
        idx = np.array([3000])
        changed = {k: v.copy() for k, v in d.items()}
        for k in ("y", "qf", "dqf", "hip"):
            changed[k][3001:] = 12345.
        for name in m.METHODS:
            if name.endswith(("_ridge", "_knn")):
                np.testing.assert_array_equal(m.features(d, idx, name), m.features(changed, idx, name))
        for a, b in zip(m.causal_phase(d, idx), m.causal_phase(changed, idx)):
            np.testing.assert_array_equal(a, b)

    def test_all_targets_before_release(self):
        d = self.dataset()
        idx = m.anchors(d)
        self.assertTrue(np.all(d["t"][idx] >= 5.006-1e-9))
        self.assertTrue(np.all(d["t"][idx+27] < 18-1e-9))
        self.assertTrue(np.all(np.diff(idx) == 3))

    def test_knn_reconstruction_saved_arrays(self):
        rng = np.random.default_rng(11)
        x, y = rng.normal(size=(30, 4)), rng.normal(size=(30, 12))
        model = m.fit_model(x, y, "knn", 8)
        out = m.predict(model, x[:3])
        distance, index = model["tree"].query((x[:3]-model["mean"])/model["std"], k=8)
        w = 1/np.maximum(distance, .001)
        w /= w.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(out, np.sum(model["train_y"][index]*w[..., None], axis=1))

    def test_score_hold_is_one(self):
        rng = np.random.default_rng(22)
        truth, hold = rng.normal(size=(20, 9, 12)), rng.normal(size=(20, 9, 12))
        self.assertAlmostEqual(m.score(hold, truth, hold), 1.)
        self.assertEqual(m.score(truth, truth, hold), 0.)


if __name__ == "__main__":
    unittest.main()
