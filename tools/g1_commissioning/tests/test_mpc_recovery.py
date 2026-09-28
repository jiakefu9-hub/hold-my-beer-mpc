"""Independent LP and structural checks for the offline braking envelope."""
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, OptimizeWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware_mpc_recovery import RecoveryEnvelopeMpcPolicy
from hardware_mpc_solver import CondensedArmMPCPolicy


def make_policy(kind=RecoveryEnvelopeMpcPolicy, **kwargs):
    values = dict(horizon=9, control_dt=.006, max_dq=1., max_ddq=8.,
                  joint_limit_margin=np.deg2rad([1., 1., 0., 0., 0.]),
                  solver_backend="osqp", solver_time_limit=.5)
    values.update(kwargs)
    return kind(np.zeros(5), **values)


def bounds_for(policy, q, dq):
    lower, upper, *_ = policy._build_online_constraint_bounds(q, dq)
    lower[:policy.nx] = upper[:policy.nx] = np.r_[q, dq]
    return lower, upper


def independent_feasibility(policy, q, dq):
    """HiGHS solves the full state/input constraints, without using condensing."""
    lower, upper = bounds_for(policy, q, dq)
    equality_end = (policy.horizon+1)*policy.nx
    matrix = policy._A_cons
    inequalities = sparse.vstack((matrix[equality_end:], -matrix[equality_end:]), format="csc")
    # HiGHS owns a process-global scheduler. Use the same explicit one-thread
    # setting as failure_evidence, otherwise test order can produce status 4
    # (scheduler conflict), which is NOT evidence of mathematical infeasibility.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=OptimizeWarning,
                                message='Unrecognized options detected.*')
        result = linprog(np.zeros(policy.num_variables), A_ub=inequalities,
                         b_ub=np.r_[upper[equality_end:], -lower[equality_end:]],
                         A_eq=matrix[:equality_end], b_eq=lower[:equality_end],
                         bounds=(None, None), method="highs", options={'threads': 1})
    if result.success:
        values = matrix@result.x
        if max(np.max(lower-values), np.max(values-upper)) > 1e-7:
            raise AssertionError("independent LP returned a violating trajectory")
    return result


class RecoveryEnvelopeTest(unittest.TestCase):
    def test_original_constraints_and_condensed_residuals_are_preserved(self):
        original, policy = make_policy(CondensedArmMPCPolicy), make_policy()
        count = original._A_cons.shape[0]
        self.assertEqual(policy.recovery_row_start, count)
        self.assertEqual(policy._A_cons.shape[0], count+45)
        np.testing.assert_array_equal(policy._A_cons[:count].toarray(), original._A_cons.toarray())
        np.testing.assert_array_equal(policy._l_template[:count], original._l_template)
        np.testing.assert_array_equal(policy._u_template[:count], original._u_template)
        rng = np.random.default_rng(2809)
        q, dq = np.zeros(5), rng.uniform(-.03, .03, 5)
        lower, upper = bounds_for(policy, q, dq)
        pc, qc, lc, uc, offset, _ = policy.condense(
            sparse.eye(policy.num_variables, format="csc"),
            np.zeros(policy.num_variables), lower, upper)
        u = rng.uniform(-1, 1, policy.horizon*policy.n)
        z = offset+policy._T@u
        full_values = policy._A_cons@z
        np.testing.assert_allclose((policy._ac@u-lc)/policy._row_scale,
                                  (full_values-lower)[policy._ineq_start:], atol=1e-13)
        np.testing.assert_allclose((uc-policy._ac@u)/policy._row_scale,
                                  (upper-full_values)[policy._ineq_start:], atol=1e-13)
        states, _ = policy._unpack_solution(z)
        np.testing.assert_allclose(full_values[count:].reshape(9, 5),
                                  states[1:, :5]+states[1:, 5:]/6., atol=1e-13)

    def test_boundary_state_grid_is_feasible_in_full_independent_lp(self):
        policy = make_policy()
        guarded = policy.safety_joint_limits.copy()
        guarded[:, 0] += policy.recovery_guard_rad
        guarded[:, 1] -= policy.recovery_guard_rad
        for joint in range(5):
            lo, hi = guarded[joint]
            positions = np.unique(np.r_[lo, lo+1e-7, policy.joint_limits[joint],
                                        (lo+hi)/2, hi-1e-7, hi])
            positions = positions[(positions >= lo) & (positions <= hi)]
            for position in positions:
                vmin = max(-1., -6*(position-lo))
                vmax = min(1., 6*(hi-position))
                for velocity in (vmin, 0., vmax):
                    q, dq = np.zeros(5), np.zeros(5)
                    q[joint], dq[joint] = position, velocity
                    with self.subTest(joint=joint, position=position, velocity=velocity):
                        self.assertTrue(independent_feasibility(policy, q, dq).success)

    def test_outer_boundary_at_rest_has_a_feasible_inward_recovery(self):
        policy = make_policy()
        for side in (0, 1):
            q = policy.safety_joint_limits[:, side].copy()
            self.assertTrue(independent_feasibility(policy, q, np.zeros(5)).success)

    def test_original_failed_state_still_rejected_by_independent_lp(self):
        policy = make_policy()
        q = np.deg2rad([-5.000875738158601, 1.0007702299410923,
                       .049881114647405814, 5.3605499332316455, -1.3051624011424736])
        dq = np.array([-.02278420294854764, .13940099573486905,
                       .001971124530040233, -.11846815712404526, .010252782951748713])
        best_next_q = q[0]+.006*dq[0]+.5*.006**2*8
        self.assertLess(best_next_q, policy.safety_joint_limits[0, 0])
        for candidate in (make_policy(CondensedArmMPCPolicy), policy):
            self.assertEqual(independent_feasibility(candidate, q, dq).status, 2)

    def test_zero_margin_joint_never_opens_outer_position_bounds(self):
        policy = make_policy()
        q, dq = np.zeros(5), np.zeros(5)
        q[2] = policy.safety_joint_limits[2, 0]-.001
        dq[2] = -.1
        lower, upper = bounds_for(policy, q, dq)
        start = (policy.horizon+1)*policy.nx
        rows = slice(start, start+policy.horizon*policy.n)
        self.assertTrue(np.all(lower[rows].reshape(9, 5) >= policy.safety_joint_limits[:, 0]))
        self.assertTrue(np.all(upper[rows].reshape(9, 5) <= policy.safety_joint_limits[:, 1]))
        self.assertEqual(independent_feasibility(policy, q, dq).status, 2)

    def test_rate_has_a_discrete_nominal_braking_witness(self):
        policy = make_policy()
        dt, rate = policy.control_dt, policy.recovery_rate_s_inv
        q = np.mean(policy.safety_joint_limits, axis=1)
        dq = np.array([-1., -.5, 0., .5, 1.])
        ddq = -rate*dq/(1+.5*rate*dt)
        self.assertTrue(np.all(np.abs(ddq) <= .8*policy.max_ddq))
        next_q, next_dq = q+dt*dq+.5*dt**2*ddq, dq+dt*ddq
        np.testing.assert_allclose(next_q+next_dq/rate, q+dq/rate, atol=1e-14)
        self.assertTrue(np.all(np.abs(next_dq) <= np.abs(dq)))

    def test_invalid_configuration_is_rejected(self):
        invalid = [dict(recovery_rate_s_inv=value) for value in (0., -1., np.nan, np.inf, 10.)]
        invalid += [dict(recovery_guard_rad=value) for value in
                    (-.1, np.nan, np.inf, [0., 0.], np.ones((5, 1)), np.deg2rad(4.))]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_policy(**kwargs)
        values = np.deg2rad([.15, .12, .1, .05, 0.])
        policy = make_policy(recovery_guard_rad=values)
        np.testing.assert_array_equal(policy.recovery_guard_rad, values)


if __name__ == "__main__":
    unittest.main()
