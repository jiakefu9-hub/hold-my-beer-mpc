import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from controllers import FeedforwardPID, PositionMPC, AccelerationPID


def config(**kw):
    return dict(simple_controller='ff_pid', mpc_horizon=9, mpc_max_dq=2.,
                mpc_q_min_deg=[-20]*5, mpc_q_max_deg=[20]*5, **kw)


def helpers(base_rate=.1):
    nodes = tuple(SimpleNamespace(t=.006*k) for k in range(10))
    def terms(qs, dqs, nodes, intervals, required):
        J = np.c_[np.eye(2), np.zeros((2, 3))]
        return tuple(dict(gravity_error=9.81*(J@q + np.array([base_rate*d.t, 0.])),
                          G_g=np.c_[9.81*J, np.zeros((2, 5))], C_omega=np.zeros((3, 5)),
                          D_omega=np.zeros(3), B_acc=np.zeros((3, 5)), C_acc=np.zeros((3, 5)),
                          D_acc=np.zeros(3), B_alpha=np.zeros((3, 5)), C_alpha=np.zeros((3, 5)),
                          D_alpha=np.zeros(3)) for q, d in zip(qs, nodes))
    return SimpleNamespace(disturbance=nodes[0], disturbance_prediction=nodes,
                           interval_disturbance_prediction=nodes[:-1], compute_mpc_terms_batch=terms)


class ControllersTest(unittest.TestCase):
    def test_bounded_acceleration_allocation_redistributes_instead_of_clipping(self):
        h=helpers(base_rate=0.)
        previous=h.compute_mpc_terms_batch
        def terms(*args):
            rows=previous(*args)
            for t in rows:
                t['gravity_error']=np.zeros(2)
                t['B_alpha']=np.array([[1.,1.,0.,0.,0.],[0.,0.,1.,0.,0.],[0.,0.,0.,1.,0.]])
                t['D_alpha']=np.array([-20.,0.,0.])
            return rows
        h.compute_mpc_terms_batch=terms
        obs=dict(current_q=np.zeros(5),current_dq=np.zeros(5))
        # First axis starts at its upper position bound, so the second axis
        # should supply the task acceleration lost to that active constraint.
        obs['current_q'][0]=np.deg2rad(20.)
        cfg=config(simple_apid_full_rotation=True)
        plain=AccelerationPID(cfg,np.zeros(5),.006)
        bounded=AccelerationPID(dict(cfg,simple_apid_bounded_allocation=True),np.zeros(5),.006)
        a=plain.compute_action(obs,h)[2];b=bounded.compute_action(obs,h)[2]
        self.assertLessEqual(b[0],1e-7)
        self.assertGreater(b[1],a[1])
        self.assertLessEqual(np.max(abs(b)),15.+1e-7)

    def test_position_only_run_exports_without_inverse_dynamics_samples(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from sim_support import compute_right_arm_trajectory_diagnostics
        tr = {key: np.zeros((3, 5)) for key in ('right_arm_tau_nominal',
              'right_arm_tau_mapping_correction', 'right_arm_ctrl', 'right_arm_qacc')}
        tr['ddq_execution_updated'] = np.zeros(3, dtype=bool)
        result = compute_right_arm_trajectory_diagnostics(tr)
        self.assertEqual(result['right_arm_torque_gap']['execution_call_count'], 0)

    def test_acceleration_pid_inertial_feedforward_sign_and_ablation(self):
        h = helpers(base_rate=0.)
        previous = h.compute_mpc_terms_batch
        def with_angular_acceleration(*args):
            values = previous(*args)
            for t in values:
                t['B_alpha'] = np.c_[np.eye(3), np.zeros((3, 2))]
                t['D_alpha'] = np.array([1., 0., 0.])
            return values
        h.compute_mpc_terms_batch = with_angular_acceleration
        obs = dict(current_q=np.zeros(5), current_dq=np.zeros(5))
        ff = AccelerationPID(config(), np.zeros(5), .006)
        noff = AccelerationPID(config(simple_feedforward=False), np.zeros(5), .006)
        self.assertLess(ff.compute_action(obs, h)[2][0], -.9)
        np.testing.assert_allclose(noff.compute_action(obs, h)[2], 0., atol=1e-12)
        self.assertFalse(ff.last_diagnostics['ddq_output_unused'])

    def test_feedforward_cancels_future_positive_base_rotation_before_error(self):
        obs = dict(current_q=np.zeros(5), current_dq=np.zeros(5))
        ff = FeedforwardPID(config(), np.zeros(5), .006)
        noff = FeedforwardPID(config(simple_feedforward=False), np.zeros(5), .006)
        qf, vf, unused = ff.compute_action(obs, helpers())
        qn, vn, _ = noff.compute_action(obs, helpers())
        self.assertLess(vf[0], 0.)
        np.testing.assert_allclose(vn, 0., atol=1e-12)
        np.testing.assert_array_equal(unused, 0.)

    def test_position_bounds_and_reference_rate(self):
        for cls in (FeedforwardPID, PositionMPC):
            p = cls(config(), np.zeros(5), .006)
            for _ in range(50):
                previous = np.zeros(5) if p.ref is None else p.ref.copy()
                q, v, _ = p.compute_action(dict(current_q=np.zeros(5), current_dq=np.zeros(5)), helpers(20.))
                self.assertLessEqual(np.max(np.abs(q)), np.deg2rad(20)+1e-10)
                self.assertLessEqual(np.max(np.abs(q-previous)), 2*.006+1e-10)

    def test_condensed_position_model_matches_first_order_recursion(self):
        p = PositionMPC(config(simple_position_beta=.3), np.zeros(5), .006)
        rng = np.random.default_rng(4)
        q0, commands = rng.normal(size=5), rng.normal(size=(9, 5))
        q, trajectory = q0.copy(), []
        for command in commands:
            q = .7*q+.3*command
            trajectory.append(q.copy())
        np.testing.assert_allclose(p.F@q0+p.L@commands.ravel(), np.asarray(trajectory).ravel())

    def test_position_mpc_preview_moves_before_current_error(self):
        p = PositionMPC(config(), np.zeros(5), .006)
        q, v, _ = p.compute_action(dict(current_q=np.zeros(5), current_dq=np.zeros(5)), helpers())
        self.assertLess(q[0], 0.)
        self.assertTrue(p.last_diagnostics['success'])


if __name__ == '__main__':
    unittest.main()
