"""Synthetic checks of method-study causality and numerical bookkeeping."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]))
import compare_walk_predictors as study


class PredictorStudyTest(unittest.TestCase):
    def test_targets_nodes_and_intervals(self):
        y=np.repeat(np.arange(200.)[:,None],12,axis=1)
        z=study.targets(dict(y=y),np.array([70]))
        np.testing.assert_allclose(z[0,:,0],[72,81,96])
        np.testing.assert_allclose(z[0,:,3],[73,82,97])

    def test_history_has_no_future(self):
        x=np.arange(200.)[:,None]; anchors=np.array([70,71])
        before=study.history(x,anchors)
        x[72:]=999
        np.testing.assert_array_equal(study.history(x,anchors),before)
        np.testing.assert_array_equal(before[0],70-np.array(study.LAGS))

    def test_model_scaler_uses_only_fit_rows(self):
        x=np.arange(20.)[:,None]; y=2*x+1
        model=study.fit_model(x,y,"ridge",.001)
        np.testing.assert_allclose(model[1],x.mean(0))
        before=model[1].copy()
        study.predict(model,np.array([[1000.]]))
        np.testing.assert_array_equal(model[1],before)

    def test_knn_lookup_has_correct_future_target(self):
        x=np.arange(20.)[:,None]; future=100+x
        model=study.fit_model(x,future,"knn",2)
        self.assertAlmostEqual(study.predict(model,np.array([[10.]]))[0,0],110.,places=1)

    def test_phase_does_not_use_future_events(self):
        t=np.arange(4.,17.,.002);hip=.3*np.sin(2*np.pi*(t-5.2))
        anchors=np.array([np.searchsorted(t,9.8)])
        a=study.phase_info(dict(t=t,hip=hip),anchors)
        hip[anchors[0]+1:]=0
        b=study.phase_info(dict(t=t,hip=hip),anchors)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


if __name__=="__main__":unittest.main()
