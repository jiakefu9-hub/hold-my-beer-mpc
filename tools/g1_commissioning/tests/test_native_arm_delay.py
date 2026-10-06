"""Native propagation versus the original Python/MuJoCo loop, no DDS."""
from collections import deque
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from native_arm_delay import NativeArmDelay
from hardware_mpc_delay_preview import CommandHistory, IssuedCommand
from hardware_mpc_delay_plan import IntervalHorizonClock
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc
from g1_walk_pid import EXPECTED_TARGET_Q
from disturbance_types import DisturbanceInput, DisturbanceHorizon


class NativeDelayTest(unittest.TestCase):
    def test_native_mass_bias_matches_independent_python_path(self):
        c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10]);native=NativeArmDelay(c.inverse)
        rng=np.random.default_rng(51006)
        try:
            for _ in range(80):
                q=rng.uniform(-.3,.3,5);v=rng.uniform(-.8,.8,5)
                b=DisturbanceInput(rng.normal(size=3),rng.normal(size=3),rng.normal(size=3),
                                  Rotation.random(random_state=rng).as_matrix())
                expected=c.inverse.linear_dynamics(q,v,b)
                actual=native.linear_dynamics(q,v,b)
                for a,e in zip(actual,expected):
                    np.testing.assert_allclose(a,e,atol=2e-12,rtol=0.)
            native.close()
            with self.assertRaisesRegex(RuntimeError,'closed'):native.linear_dynamics(q,v,b)
        finally:native.close();c.close()

    def test_random_moving_base_partial_weight_and_command_switches(self):
        c=RightArmMeasuredTorqueMpc(EXPECTED_TARGET_Q[5:10]);native=NativeArmDelay(c.inverse)
        h=CommandHistory();h.reset_history();h.inverse=c.inverse;h.mapper=c.mapper
        h.torque_config=c.torque_config;h.assumed_command_delay_s=.006
        rng=np.random.default_rng(20261005)
        try:
            for n in range(80):
                q=rng.uniform(-.2,.2,5);v=rng.uniform(-.4,.4,5)
                def base():
                    return DisturbanceInput(rng.normal(size=3),rng.normal(size=3)*.3,
                        rng.normal(size=3),Rotation.from_rotvec(rng.normal(size=3)*.2).as_matrix())
                nodes=tuple(base() for _ in range(10))
                clock=IntervalHorizonClock(DisturbanceHorizon(nodes,nodes[:-1]))
                packets=[IssuedCommand(t,rng.normal(size=5),q+rng.normal(size=5)*.01,
                    v+rng.normal(size=5)*.01,float(rng.choice([0.,.1,.5,1.])))
                    for t in [-.005,.0033,.0075,.0113]] if n%3 else []
                age=float(rng.uniform(0,.02));h._issued=deque(packets);h.native=None
                expected=h._predict(q.copy(),v.copy(),clock,age,0.)
                h._issued=deque(packets);h.native=native
                actual=h._predict(q.copy(),v.copy(),clock,age,0.)
                np.testing.assert_allclose(actual[0],expected[0],atol=2e-12,rtol=0.)
                np.testing.assert_allclose(actual[1],expected[1],atol=2e-11,rtol=0.)
                self.assertEqual(actual[2]['prediction_steps'],expected[2]['prediction_steps'])
        finally:
            native.close();c.close()


if __name__=='__main__':unittest.main()
