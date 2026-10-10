"""SDK-free velocity objective: independent geometry, quadratic cost and runtime."""
import unittest

import mujoco
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from arm_mpc import ArmMPCPolicy

from disturbance_types import DisturbanceInput
from g1_walk_mpc import (LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG,
    LEARNED_ACC_Y0015_MPC_CONFIG, LEARNED_VEL01_MPC_CONFIG,
    LEARNED_TORQUE_CONFIG, LEARNED_FIELD_MPC_CONFIGS)
from g1_walk_pid import EXPECTED_TARGET_Q
from hardware_mpc_control import load_mpc_config
from hardware_mpc_learned import RightArmLearnedTorqueMpc
from hardware_mpc_solver import CondensedArmMPCPolicy
from test_measured_torque_mpc import horizon


class LinearVelocityTests(unittest.TestCase):
    def controller(self, config=LEARNED_VEL01_MPC_CONFIG):
        c = RightArmLearnedTorqueMpc(EXPECTED_TARGET_Q[5:10], config,
                                    torque_config=LEARNED_TORQUE_CONFIG)
        self.addCleanup(c.close)
        c.policy.solver_time_limit = .1  # algebra test, not a timing certificate
        return c

    def test_candidates_change_one_objective_and_keep_baseline_velocity_zero(self):
        baseline = load_mpc_config(LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG)
        self.assertEqual(baseline['q_ee_vel'], 0.)
        for path, key in ((LEARNED_VEL01_MPC_CONFIG, 'q_ee_vel'),
                          (LEARNED_ACC_Y0015_MPC_CONFIG, 'q_ee_acc')):
            config = load_mpc_config(path)
            changed = {k for k in baseline.keys() | config.keys() if baseline.get(k) != config.get(k)}
            self.assertEqual(changed, {key})
            self.assertIn(path.resolve(), LEARNED_FIELD_MPC_CONFIGS)
        np.testing.assert_equal(load_mpc_config(LEARNED_ACC_Y0015_MPC_CONFIG)['q_ee_acc'], [.01,.015,.01])

    def test_invalid_velocity_weights_fail_before_model_creation(self):
        for weight in (-.1, [1, 2], [1, float('nan'), 1],
                       [[1, 2, 0], [0, 1, 0], [0, 0, 1]]):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                load_mpc_config({'q_ee_vel': weight})

    def test_velocity_matches_finite_difference_of_moving_base_geometry(self):
        c = self.controller(); h = c.helper; m = c.model.model
        rng = np.random.default_rng(109)
        data = mujoco.MjData(m)
        qpos = m.qpos0.copy(); qpos[c.model.joint_addresses] = EXPECTED_TARGET_Q[:11]
        eps = 1e-6
        for _ in range(8):
            q = c.nominal+rng.normal(0,.05,5); dq = rng.normal(0,.2,5)
            omega = rng.normal(0,.4,3)
            R = Rotation.from_rotvec(rng.normal(0,.3,3)).as_matrix()
            node = DisturbanceInput(np.zeros(3),omega,np.zeros(3),R)
            # Terminal/state-only path must still retain a nonzero J_v.
            terms = h.compute_mpc_terms(q,dq,qpos,node,None,False)
            positions = []
            for sign in (-1,1):
                data.qpos[:] = qpos
                data.qpos[h.joint_indices] = q+sign*eps*dq
                mujoco.mj_kinematics(m,data)
                Rimu = data.site_xmat[h.imu_site_id].reshape(3,3)
                r_local = Rimu.T @ (data.site_xpos[h.ee_site_id]-data.site_xpos[h.imu_site_id])
                positions.append(Rotation.from_rotvec(sign*eps*omega).as_matrix() @ R @ r_local)
            expected = (positions[1]-positions[0])/(2*eps)
            np.testing.assert_allclose(terms['D_vel']+terms['C_vel']@dq,expected,atol=2e-9)
            translated = qpos.copy(); translated[:3] += [1.2,-.4,.8]
            shifted = h.compute_mpc_terms(q,dq,translated,node,None,False)
            np.testing.assert_allclose(shifted['D_vel'],terms['D_vel'],atol=1e-12)
            np.testing.assert_allclose(shifted['C_vel'],terms['C_vel'],atol=1e-12)

    def test_batch_matches_scalar_with_node_not_interval_omega_and_terminal(self):
        c = self.controller(); h = c.helper; rng = np.random.default_rng(9)
        qpos = c.model.model.qpos0.copy()
        qs = c.nominal+rng.normal(0,.03,(10,5)); dqs = rng.normal(0,.15,(10,5))
        def node():
            return DisturbanceInput(rng.normal(size=3),rng.normal(size=3),rng.normal(size=3),
                                    Rotation.from_rotvec(rng.normal(0,.2,3)).as_matrix())
        nodes = tuple(node() for _ in range(10)); intervals = tuple(node() for _ in range(9))
        required = np.arange(10)%2 == 0
        batch = h.compute_mpc_terms_batch(qs,dqs,qpos,nodes,intervals,required)
        for k,terms in enumerate(batch):
            single = h.compute_mpc_terms(qs[k],dqs[k],qpos,nodes[k],
                                        intervals[k] if k<9 else None,bool(required[k]))
            for key in terms:
                np.testing.assert_allclose(terms[key],single[key],atol=2e-12,err_msg=f'{k}:{key}')
        self.assertGreater(np.linalg.norm(batch[-1]['C_vel']),.1)
        self.assertGreater(np.linalg.norm(batch[-1]['D_vel']),.01)

    def test_quadratic_increment_is_velocity_energy_including_terminal_and_cross_term(self):
        c = self.controller(); p = c.policy; rng = np.random.default_rng(14)
        terms = [{k:rng.normal(size=shape) for k,shape in p._step_term_shapes().items()}
                 for _ in range(p.horizon+1)]
        enabled_h, enabled_f = p._build_cost(terms)
        p._linear_velocity_cost_active = False
        disabled_h, disabled_f = p._build_cost(terms)
        dh = sparse.block_diag(enabled_h).toarray()-sparse.block_diag(disabled_h).toarray()
        df = enabled_f-disabled_f
        for _ in range(10):
            z = rng.normal(size=p.num_variables)
            expected = 0.
            for k,t in enumerate(terms):
                dq = z[p._cx(k)+p.nu:p._cx(k)+p.nx]
                v = t['D_vel']+t['C_vel']@dq
                scale = p.terminal_scale if k==p.horizon else 1.
                expected += scale*(v@p.Q_ee_vel@v-t['D_vel']@p.Q_ee_vel@t['D_vel'])
            self.assertAlmostEqual(.5*z@dh@z+df@z,expected,places=9)

    def test_zero_weight_preserves_existing_command(self):
        a = self.controller(LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG)
        config = load_mpc_config(LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG);config['q_ee_vel']=[0,0,0]
        b = self.controller(config)
        for c in (a,b):
            c.set_measured_dq(np.array([.1,0,.01,.03,0.]));c.set_disturbance_horizon(horizon())
        ra = a.step(EXPECTED_TARGET_Q,[1,0,0,0],0.,.006)
        rb = b.step(EXPECTED_TARGET_Q,[1,0,0,0],0.,.006)
        for key in ('raw_mpc_ddq_rad_s2','tau_total_estimated_at_feedback_nm'):
            np.testing.assert_array_equal(ra[2][key],rb[2][key])
        self.assertFalse(a.helper.include_linear_velocity_terms)

    def test_simulation_velocity_quadratic_matches_energy_and_condensed_form(self):
        rng = np.random.default_rng(1010)
        for weight in (.01, .1):
            config = load_mpc_config(LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG)
            config['q_ee_vel'] = weight
            p = self.controller(config).policy
            terms = [{k:rng.normal(size=shape) for k,shape in p._step_term_shapes().items()}
                     for _ in range(p.horizon+1)]
            sh, sf = ArmMPCPolicy._build_cost(p, terms)
            ch, cf = CondensedArmMPCPolicy._build_cost(p, terms)
            for a, b in zip(sh, ch):
                np.testing.assert_allclose(a, b, atol=2e-12)
            np.testing.assert_allclose(sf, cf, atol=2e-12)
            p._linear_velocity_cost_active = False
            zh, zf = ArmMPCPolicy._build_cost(p, terms)
            dh = sparse.block_diag(sh).toarray() - sparse.block_diag(zh).toarray()
            z = rng.normal(size=p.num_variables)
            expected = 0.
            for k, t in enumerate(terms):
                dq = z[p._cx(k)+p.nu:p._cx(k)+p.nx]
                v = t['D_vel'] + t['C_vel'] @ dq
                scale = p.terminal_scale if k == p.horizon else 1.
                expected += scale*weight*(v@v - t['D_vel']@t['D_vel'])
            self.assertAlmostEqual(.5*z@dh@z+(sf-zf)@z, expected, places=9)

    def test_velocity_objective_changes_action_and_logs_actual_weighted_quantity(self):
        baseline = self.controller(LEARNED_ACC_ALPHA_OMEGA1_MPC_CONFIG)
        trial = self.controller()
        records = []
        for c in (baseline,trial):
            c.set_measured_dq(np.array([.12,.04,.08,-.05,.02]));c.set_disturbance_horizon(horizon())
            records.append(c.step(EXPECTED_TARGET_Q,[1,0,0,0],0.,.006)[2])
        self.assertGreater(np.linalg.norm(np.array(records[0]['raw_mpc_ddq_rad_s2'])-
                                         records[1]['raw_mpc_ddq_rad_s2']),1e-5)
        step = records[1]['mpc']['one_step_prediction']
        v = np.asarray(step['ee_lin_vel_relative_imu_h0_m_s'])
        self.assertAlmostEqual(step['cost_terms']['linear_velocity_relative_imu'],.1*(v@v))

    def test_spawned_velocity_worker_matches_direct_runtime(self):
        from test_mpc_compute_process import ComputeProcessTests
        helper = ComputeProcessTests()
        try:
            helper.check_runtime_parity(dict(config=LEARNED_VEL01_MPC_CONFIG,
                predictor_mode='learned_filtered',stationary=False,field_trial=True,
                torque_config=LEARNED_TORQUE_CONFIG,assumed_command_delay_s=None))
        finally:
            helper.doCleanups()


if __name__ == '__main__':
    unittest.main()
