from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from disturbance_types import DisturbanceInput, DisturbanceHorizon
from disturbance_template.preview_correction import (
    TemplatePreviewCorrection, correct_template_preview,
)


def disturbance(angle=0., value=0.):
    return DisturbanceInput(acc_world=np.full(3,value), omega_world=np.full(3,value+1),
        alpha_world=np.full(3,2*value),
        rot_world_body=Rotation.from_rotvec([angle,0,0]).as_matrix())


class Predictor:
    def __init__(self): self.terminal=False
    def reset(self): self.terminal=False
    def update(self, observation): self.observation=observation
    def predict(self, horizon, dt):
        if self.terminal:
            d=self.observation.measured_disturbance
            return DisturbanceHorizon(nodes=(d,)*(horizon+1), intervals=(d,)*horizon)
        return DisturbanceHorizon(nodes=(self.observation.measured_disturbance,)+tuple(
            disturbance(.01*i,i) for i in range(1,horizon+1)), intervals=tuple(
            disturbance(.01*(i+.5),i+.5) for i in range(horizon)))
    def get_last_diagnostics(self,copy_data=True):return dict(terminal_hold_used=self.terminal)
    def metadata(self):return dict(predictor_type='full_task_template')


class TestCorrection(unittest.TestCase):
    def observation(self,time,angle):
        return SimpleNamespace(simulation_time=time,measured_disturbance=disturbance(angle,10.))

    def test_default_is_same_predictor_object(self):
        inner=Predictor();self.assertIs(correct_template_preview(inner,{}),inner)

    def test_correction_sign_current_node_and_decay(self):
        inner=Predictor();p=TemplatePreviewCorrection(inner,rotation_gain=1.,rotation_decay_s=1.)
        p.update(self.observation(0.,0.));p.predict(3,.006)
        obs=self.observation(.006,.03);p.update(obs);result=p.predict(3,.006)
        self.assertIs(result.nodes[0],obs.measured_disturbance)
        angle=Rotation.from_matrix(result.nodes[1].rot_world_body).as_rotvec()[0]
        self.assertAlmostEqual(angle,.01+(.03-.01)*np.exp(-.006))
        np.testing.assert_array_equal(result.nodes[1].omega_world,np.full(3,2.))

    def test_repeated_prediction_is_not_a_new_observation(self):
        p=TemplatePreviewCorrection(Predictor(),rotation_gain=1.)
        p.update(self.observation(0.,0.));p.predict(3,.006)
        p.update(self.observation(.006,.03));first=p.predict(3,.006)
        self.assertIs(first,p.predict(3,.006))
        with self.assertRaises(ValueError):p.predict(2,.006)

    def test_gap_and_reset_discard_stale_rotation_error(self):
        p=TemplatePreviewCorrection(Predictor(),rotation_gain=1.)
        p.update(self.observation(0.,0.));p.predict(3,.006)
        p.update(self.observation(.018,.03));gap=p.predict(3,.006)
        np.testing.assert_allclose(gap.nodes[1].rot_world_body,disturbance(.01).rot_world_body)
        p.reset();p.update(self.observation(.024,.03));reset=p.predict(3,.006)
        np.testing.assert_allclose(reset.nodes[1].rot_world_body,disturbance(.01).rot_world_body)

    def test_terminal_measurement_hold_is_preserved(self):
        inner=Predictor();p=TemplatePreviewCorrection(inner,rotation_gain=1.,interval_lead_s=.003)
        p.update(self.observation(0.,0.));p.predict(3,.006)
        inner.terminal=True;obs=self.observation(.006,.03);p.update(obs);result=p.predict(3,.006)
        self.assertTrue(all(d is obs.measured_disturbance for d in result.nodes+result.intervals))

    def test_lead_interpolates_only_interval_acceleration(self):
        p=TemplatePreviewCorrection(Predictor(),interval_lead_s=.003)
        p.update(self.observation(0.,0.));result=p.predict(3,.006)
        np.testing.assert_array_equal(result.intervals[0].acc_world,np.full(3,1.))
        np.testing.assert_array_equal(result.intervals[0].alpha_world,np.full(3,2.))
        np.testing.assert_array_equal(result.intervals[0].omega_world,np.full(3,1.5))
        np.testing.assert_array_equal(result.intervals[-1].acc_world,np.full(3,2.5))

    def test_invalid_configuration_rejected(self):
        for kwargs in [dict(rotation_gain=-.1),dict(rotation_gain=float('nan')),
                       dict(rotation_decay_s=0.),dict(interval_lead_s=-.001),dict(interval_lead_s=.1)]:
            with self.assertRaises(ValueError):TemplatePreviewCorrection(Predictor(),**kwargs)


if __name__=='__main__':unittest.main()
