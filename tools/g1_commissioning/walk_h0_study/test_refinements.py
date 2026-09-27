import unittest
import numpy as np
from refine_blend import fit_weights, blend
import refine_nonlinear as n
from refine_innovation import decreasing_projection


class RefinementTests(unittest.TestCase):
    def test_decaying_correction_projection(self):
        np.testing.assert_allclose(decreasing_projection([.2,.8],[1.,1.]),[.5,.5])
        np.testing.assert_allclose(decreasing_projection([1.2,.8,-.1],[1.,1.,1.]),[1.,.8,0.])
        np.testing.assert_allclose(decreasing_projection([.2,.8],[3.,1.]),[.35,.35])

    def test_convex_weights_recover_known_mix(self):
        rng=np.random.default_rng(15)
        a,b=rng.normal(size=(30,9,12)),rng.normal(size=(30,9,12))
        truth=.3*a+.7*b
        w=fit_weights(a,b,truth)
        np.testing.assert_allclose(w,.3,atol=1e-12)
        np.testing.assert_allclose(blend(a,b,w)[:,:,:9],truth[:,:,:9],atol=1e-12)
        np.testing.assert_array_equal(blend(a,b,w)[:,:,9:],a[:,:,9:])

    def test_nonlinear_map_uses_only_query_and_frozen_parameters(self):
        rng=np.random.default_rng(16)
        model=dict(linear_mean=np.zeros(4),linear_std=np.ones(4),leg_mean=np.zeros(2),leg_std=np.ones(2),
                   fourier_weights=rng.normal(size=(2,8)),fourier_bias=rng.normal(size=8))
        x,z=rng.normal(size=(10,4)),rng.normal(size=(10,2))
        np.testing.assert_array_equal(n.map_features(model,x,z)[:4],n.map_features(model,x[:4],z[:4]))


if __name__=='__main__':
    unittest.main()
