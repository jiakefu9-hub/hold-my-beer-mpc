"""Checks for research-only template conversion, independent of robot SDK."""
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from disturbance_predictor import FullTaskTemplatePredictor, DisturbancePredictorObservation, FullTaskPredictorError
from disturbance_types import DisturbanceInput
from resampled_preview import ResampledTemplate, interval_weights


def predictor(cls, dt):
    c = yaml.safe_load((ROOT/'configs/simulation_optimization_20261010/conservative.yaml').read_text())
    return cls(template_path=str(ROOT/c['full_task_template_path']),
        manifest_path=str(ROOT/c['full_task_template_manifest_path']),
        expected_sha256=c['full_task_template_sha256'],
        expected_manifest_sha256=c['full_task_template_manifest_sha256'],
        expected_schema_version=c['full_task_template_schema_version'],
        expected_heading_frame_version=c['full_task_heading_frame_version'],
        repo_dir=str(ROOT), control_dt=dt, horizon=9)


class ResamplingTests(unittest.TestCase):
    def test_overlap_means_not_interpolated_point_samples(self):
        np.testing.assert_allclose(interval_weights(.004, .004), [.5,.5,0,0,0,0,0,0,0])
        np.testing.assert_allclose(interval_weights(.008, .004), [0,1,0,0,0,0,0,0,0], atol=1e-14)
        for dt in [.002,.004,.006]:
            self.assertAlmostEqual(interval_weights(.002,dt)@np.ones(9), 1.)
        with self.assertRaises(ValueError): interval_weights(.052, .004)

    def test_six_ms_conversion_is_identity(self):
        source = predictor(FullTaskTemplatePredictor,.006)
        converted = predictor(ResampledTemplate,.006)
        for attr in ['_nodes','_intervals']:
            for key in getattr(source,attr):
                np.testing.assert_array_equal(getattr(source,attr)[key],getattr(converted,attr)[key])
        for i in range(20):
            d=DisturbanceInput(acc_world=np.array([i,.2,.3]),omega_world=np.zeros(3),
                alpha_world=np.zeros(3),rot_world_body=np.eye(3))
            o=DisturbancePredictorObservation(simulation_time=i*.006,measured_disturbance=d)
            source.update(o);converted.update(o)
            a=source.predict(9,.006);b=converted.predict(9,.006)
            for left,right in zip(a.nodes+a.intervals,b.nodes+b.intervals):
                for key in ['acc_world','omega_world','alpha_world','rot_world_body']:
                    np.testing.assert_array_equal(getattr(left,key),getattr(right,key))

    def test_four_ms_uses_earlier_anchor_and_preserves_current_measurement(self):
        source = predictor(FullTaskTemplatePredictor,.006)
        p = predictor(ResampledTemplate,.004)
        # t=4 ms still uses the t=0 row: average its 0..6 and 6..12 means.
        np.testing.assert_allclose(p._intervals['acc'][1,0],
            .5*(source._intervals['acc'][0,0]+source._intervals['acc'][0,1]))
        for i in range(20):
            d=DisturbanceInput(acc_world=np.full(3,100.+i),omega_world=np.zeros(3),
                alpha_world=np.zeros(3),rot_world_body=np.eye(3))
            p.update(DisturbancePredictorObservation(i*.004,d))
            h=p.predict(9,.004)
            np.testing.assert_array_equal(h.nodes[0].acc_world,d.acc_world)
            self.assertEqual(len(h.intervals),9)
            for v in h.nodes+h.intervals:
                np.testing.assert_allclose(v.rot_world_body.T@v.rot_world_body,np.eye(3),atol=1e-12)
        # A repeated/skipped tick remains an error, not a permissive fallback.
        with self.assertRaises(FullTaskPredictorError):
            p.update(DisturbancePredictorObservation(21*.004,d))

    def test_two_ms_keeps_all_intervals_inside_original_horizon(self):
        p=predictor(ResampledTemplate,.002)
        self.assertEqual(p._nodes['acc'].shape[1:],(10,3))
        self.assertTrue(np.isfinite(p._intervals['acc']).all())
        self.assertAlmostEqual(p._anchor_task_time[1],.002)
        self.assertFalse(p.metadata()['hardware_predictor_changed'])


if __name__=='__main__': unittest.main()
