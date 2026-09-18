"""Small offline checks for causal replay; no hardware or recorded files needed."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "walk_audit", Path(__file__).parents[1] / "analyze_walk_dataset.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class WalkAuditTest(unittest.TestCase):
    def test_asof_uses_past_only(self):
        v, age = audit.asof(np.array([0., .003, .005]),
                            np.array([10., 20., 30.]), np.array([.002, .004]))
        np.testing.assert_array_equal(v, [10., 20.])
        np.testing.assert_allclose(age, [.002, .001])
        with self.assertRaises(ValueError):
            audit.asof(np.array([1.]), np.array([1.]), np.array([0.]))

    def test_filter_prefix_and_constant(self):
        x = np.random.default_rng(7).normal(size=(500, 3))
        np.testing.assert_array_equal(audit.lowpass(x, 15)[:251],
                                      audit.lowpass(x[:251], 15))
        np.testing.assert_allclose(audit.lowpass(np.ones((100, 3)), 15), 1)

    def test_events_prefix_and_full_not_half_cycle(self):
        t = np.arange(4., 16., audit.DT)
        x = .3 * np.sin(2*np.pi*(t-5.2))
        full = audit.leg_events(t, x)
        cut = int(np.searchsorted(t, 11.))
        self.assertEqual([e for e in full if e[0] < cut],
                         audit.leg_events(t[:cut], x[:cut]))
        times = np.array([t[i] for i, sign in full if sign == 1])
        np.testing.assert_allclose(np.diff(times), 1., atol=.00201)

    def test_next_event_forecast_does_not_use_target(self):
        old = audit.event_predictions(np.array([5., 6., 7., 8.]), 1.)[0]
        new = audit.event_predictions(np.array([5., 6., 7.2, 8.]), 1.)[0]
        # target is used only to score the already-issued prediction.
        pred = lambda r: r["target_s"] + r["adaptive_error_ms"]/1000
        self.assertAlmostEqual(pred(old), pred(new))
        self.assertAlmostEqual(pred(old), 7.)

    def test_phase_wrap_is_continuous(self):
        template = np.array([[0., 1.], [1., 2.], [0., 1.], [-1., 0.]])
        np.testing.assert_allclose(audit.phase_sample(template, [0., 1., 2.]),
                                  np.tile(template[0], (3, 1)))
        np.testing.assert_allclose(audit.phase_sample(template, [-.01]),
                                  audit.phase_sample(template, [.99]))


if __name__ == "__main__":
    unittest.main()
