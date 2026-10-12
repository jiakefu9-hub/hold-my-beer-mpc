import unittest
import numpy as np
from strategy import configure_policy, TruncatedPreviewMPC


class Dummy:
    horizon = 9
    nx = 10
    nu = 5
    stage_dim = 15
    _last_u_plan = np.arange(45.).reshape(9, 5) / 10.
    def _build_cost(self, terms):
        return [np.zeros((15,15)) for _ in range(9)], np.zeros(145)


class TestStrategy(unittest.TestCase):
    def test_change_cost_matches_direct_quadratic_difference(self):
        p = configure_policy(Dummy(), dict(research_r_first_ddq_delta=.2))
        h, f = p._build_cost(None)
        def expanded(u):
            x = np.r_[np.zeros(10), u]
            return .5*x@h[0]@x + f[:15]@x
        a = np.array([1.,-2.,3.,.5,-.2]); b = np.array([.2,.4,.1,-.5,.1])
        previous = p._last_u_plan[0]
        expected = .2*(np.sum((a-previous)**2)-np.sum((b-previous)**2))
        self.assertAlmostEqual(expanded(a)-expanded(b), expected)
        self.assertGreaterEqual(np.linalg.eigvalsh(h[0]).min(), 0.)
        np.testing.assert_array_equal(h[1], np.zeros((15,15)))

    def test_shorter_qp_receives_only_leading_preview(self):
        class Inner:
            horizon = 3
            def compute_action(self, obs, helpers): return helpers
        p = TruncatedPreviewMPC(Inner(), 9)
        h = dict(disturbance_prediction=list(range(10)),
                 interval_disturbance_prediction=list(range(9)))
        got = p.compute_action({}, h)
        self.assertEqual(got['disturbance_prediction'], [0,1,2,3])
        self.assertEqual(got['interval_disturbance_prediction'], [0,1,2])
        self.assertEqual(len(h['disturbance_prediction']), 10)


if __name__ == '__main__':
    unittest.main()
