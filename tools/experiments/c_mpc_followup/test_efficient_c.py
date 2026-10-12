"""One-node evaluation must preserve C's bounded actions and feedforward signs."""
from types import SimpleNamespace
import unittest
import numpy as np
from efficient_c import EfficientAccelerationPID, AccelerationPID


class EfficientCTests(unittest.TestCase):
    def test_actions_match_batch_at_current_and_selected_preview_intervals(self):
        rng=np.random.default_rng(23)
        nodes=tuple(SimpleNamespace(value=i*.01) for i in range(10))
        intervals=tuple(SimpleNamespace(value=1+i*.1) for i in range(9))
        Jw=np.c_[np.eye(3),np.array([[.2,.1],[.1,.2],[.2,-.1]])]
        Jv=rng.normal(size=(3,5))*.12
        calls=[]
        def single(q,dq,node,interval,required):
            calls.append((node,interval))
            return dict(gravity_error=9.81*(q[:2]+node.value),
                G_g=np.c_[9.81*np.eye(2,5),np.zeros((2,5))],
                B_alpha=Jw,C_alpha=np.zeros((3,5)),D_alpha=np.full(3,interval.value),
                B_acc=Jv,C_acc=np.zeros((3,5)),D_acc=np.array([interval.value,2.,-.1]),
                C_omega=Jw,D_omega=np.full(3,node.value))
        def batch(qs,dqs,ns,ins,req):
            return tuple(single(q,dq,n,ins[min(i,len(ins)-1)],True)
                for i,(q,dq,n) in enumerate(zip(qs,dqs,ns)))
        h=SimpleNamespace(disturbance=nodes[0],disturbance_prediction=nodes,
            interval_disturbance_prediction=intervals,compute_mpc_terms=single,
            compute_mpc_terms_batch=batch)
        for ff in [False,True]:
            for k in [0,3]:
                for bounded in [False,True]:
                    c=dict(simple_controller='acceleration_pid',mpc_horizon=9,
                        mpc_q_min_deg=[-20]*5,mpc_q_max_deg=[20]*5,mpc_max_dq=2.,
                        simple_apid_full_rotation=True,simple_feedforward=ff,
                        simple_acc_preview_steps=k,simple_apid_bounded_allocation=bounded)
                    a=AccelerationPID(c,np.zeros(5),.006)
                    b=EfficientAccelerationPID(c,np.zeros(5),.006)
                    for i in range(10):
                        obs=dict(current_q=rng.normal(size=5)*.03,current_dq=rng.normal(size=5)*.1)
                        expected=a.compute_action(obs,h)
                        calls.clear();actual=b.compute_action(obs,h)
                        self.assertEqual(len(calls),1 if k==0 else 2)
                        for x,y in zip(expected,actual):
                            np.testing.assert_allclose(x,y,atol=1e-12,rtol=0)


if __name__=='__main__':unittest.main()
