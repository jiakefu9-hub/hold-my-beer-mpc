"""Current-omega posture extrapolation used by the retained no-preview baseline."""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from disturbance_types import DisturbanceHorizon, DisturbanceInput

def current_omega_horizon(current, dt, horizon=9):
    omega = np.asarray(current.omega_world, dtype=float)
    rotation = np.asarray(current.rot_world_body, dtype=float)
    def at(t):
        return replace(current,
            acc_world=np.asarray(current.acc_world).copy(),
            omega_world=omega.copy(),
            alpha_world=np.asarray(current.alpha_world).copy(),
            rot_world_body=Rotation.from_rotvec(omega*t).as_matrix() @ rotation)
    return DisturbanceHorizon(
        tuple(at(k*dt) for k in range(horizon+1)),
        tuple(at((k+.5)*dt) for k in range(horizon)))


def verify_adapter():
    """Check frame convention, midpoint timing, held vectors and zero omega."""
    r0 = Rotation.from_euler('xyz', [.3, -.2, .7]).as_matrix()
    current = DisturbanceInput(np.array([1., 2., 3.]),
        np.array([0., 0., 2.]), np.array([4., 5., 6.]), r0)
    result = current_omega_horizon(current, .006)
    assert len(result.nodes) == 10 and len(result.intervals) == 9
    np.testing.assert_allclose(result.nodes[0].rot_world_body, r0, atol=1e-14)
    for items, times in ((result.nodes, np.arange(10)*.006),
                         (result.intervals, (np.arange(9)+.5)*.006)):
        for item, t in zip(items, times):
            c, s = np.cos(2*t), np.sin(2*t)
            expected = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ r0
            np.testing.assert_allclose(item.rot_world_body, expected, atol=1e-14)
            for field in ('acc_world', 'omega_world', 'alpha_world'):
                np.testing.assert_array_equal(getattr(item, field), getattr(current, field))
    stopped = current_omega_horizon(replace(current, omega_world=np.zeros(3)), .006)
    for item in stopped.nodes + stopped.intervals:
        np.testing.assert_allclose(item.rot_world_body, r0, atol=1e-14)
