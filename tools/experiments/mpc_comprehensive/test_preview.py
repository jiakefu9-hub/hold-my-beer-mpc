"""Checks that research preview corrections are causal and preserve node zero."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from disturbance_types import DisturbanceInput, DisturbanceHorizon
from preview import InnovationPreview
from scipy.spatial.transform import Rotation
from types import SimpleNamespace


def sample(value):
    return DisturbanceInput(acc_world=np.full(3, value),
        omega_world=np.full(3, value), alpha_world=np.full(3, 2 * value),
        rot_world_body=np.eye(3))


class Inner:
    def predict(self, horizon, dt):
        return DisturbanceHorizon(nodes=(sample(10.), sample(1.), sample(2.)),
                                 intervals=(sample(.5), sample(1.5)))
    def get_last_diagnostics(self, copy_data=True):
        return dict(anchor_acc_error=np.ones(3), anchor_omega_error=np.ones(3)*2,
                    anchor_alpha_error=np.ones(3)*3)


class TestPreview(unittest.TestCase):
    def test_zero_gain_preserves_values(self):
        p = InnovationPreview(Inner(), {}).predict(2, .006)
        np.testing.assert_array_equal(p.nodes[0].acc_world, np.full(3, 10.))
        np.testing.assert_array_equal(p.intervals[0].acc_world, np.full(3, .5))

    def test_innovation_uses_template_residual_not_replaced_node_zero(self):
        p = InnovationPreview(Inner(), dict(research_innovation_gain=.5,
            research_innovation_decay_s=.006)).predict(2, .006)
        np.testing.assert_array_equal(p.nodes[0].acc_world, np.full(3, 10.))
        np.testing.assert_allclose(p.nodes[1].acc_world, 1. + .5*np.exp(-1.))
        np.testing.assert_allclose(p.intervals[0].acc_world, .5 + .5*np.exp(-.5))

    def test_lead_changes_acceleration_only_and_holds_last_interval(self):
        p = InnovationPreview(Inner(), dict(research_interval_lead_s=.003)).predict(2, .006)
        np.testing.assert_allclose(p.intervals[0].acc_world, 1.)
        np.testing.assert_allclose(p.intervals[1].acc_world, 1.5)
        np.testing.assert_allclose(p.intervals[0].alpha_world, 2.)
        np.testing.assert_allclose(p.intervals[0].omega_world, .5)
        np.testing.assert_allclose(p.nodes[0].acc_world, 10.)

    def test_rotation_correction_uses_previous_raw_prediction(self):
        wrapper = InnovationPreview(Inner(), dict(research_rotation_innovation_gain=1.,
            research_rotation_innovation_decay_s=.15))
        wrapper.measurement = SimpleNamespace(rot_world_body=np.eye(3))
        wrapper.predict(2, .006)
        wrapper.measurement = SimpleNamespace(rot_world_body=Rotation.from_rotvec([.1,0,0]).as_matrix())
        p = wrapper.predict(2, .006)
        np.testing.assert_allclose(Rotation.from_matrix(p.nodes[1].rot_world_body).as_rotvec(),
                                   [.1*np.exp(-.006/.15),0,0],atol=1e-12)
        # Underlying uncorrected prediction stays identity between calls.
        again = wrapper.predict(2, .006)
        np.testing.assert_allclose(again.nodes[1].rot_world_body, p.nodes[1].rot_world_body)


if __name__ == '__main__':
    unittest.main()
