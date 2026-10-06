"""Offline geometry/QP/causality/envelope/output-boundary checks; no DDS."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_walk_pid import EXPECTED_TARGET_Q, load_profile, REQUIRED_CONFIRMATIONS
from g1_walk_mpc import main, MpcJournal
from hardware_mpc_control import RightArmHardwareMpc, HardwareMpcError
from hardware_mpc_predictor import HardwareMpcPredictor, FrozenInnovationBank
from hardware_mpc_solver import CondensedArmMPCPolicy
from arm_mpc import ArmMPCPolicy
from disturbance_types import DisturbanceInput, DisturbanceHorizon


def horizon(roll=0.):
    d = DisturbanceInput(np.zeros(3), np.zeros(3), np.zeros(3),
                        Rotation.from_euler("x", roll).as_matrix())
    return DisturbanceHorizon((d,)*10, (d,)*9)


class MpcContractTest(unittest.TestCase):
    def test_daqp_optimum_matches_independent_high_accuracy_qp(self):
        policy = CondensedArmMPCPolicy(EXPECTED_TARGET_Q[5:10], horizon=9,
                                      max_dq=.07, max_ddq=.2, solver_time_limit=.1)
        rng = np.random.default_rng(73)
        for _ in range(3):
            m = rng.normal(size=(policy.num_variables, 20))
            full_p = m@m.T + np.eye(policy.num_variables)
            linear = rng.normal(size=policy.num_variables)
            lower,upper = policy._l_template.copy(),policy._u_template.copy()
            lower[:10] = upper[:10] = np.r_[EXPECTED_TARGET_Q[5:10],np.zeros(5)]
            p = sparse.csc_matrix(np.triu(full_p))
            result,exc = policy._solve_qp(p,None,linear,lower,upper,np.zeros(policy.num_variables))
            self.assertIsNone(exc)
            self.assertTrue(policy._check_result(result,lower,upper)[0])
            pc,qc,lc,uc,offset,_ = policy.condense(p,linear,lower,upper)
            independent = osqp.OSQP()
            independent.setup(P=sparse.csc_matrix(pc),q=qc,A=policy._ac,l=lc,u=uc,
                              verbose=False,eps_abs=1e-9,eps_rel=1e-9,max_iter=50000,
                              polishing=True)
            expected = independent.solve(raise_error=False)
            self.assertEqual(expected.info.status,"solved")
            np.testing.assert_allclose(result.x,offset+policy._T@expected.x,atol=2e-7)

    def test_condensing_is_exact_dynamics_cost_and_constraints(self):
        policy = CondensedArmMPCPolicy(EXPECTED_TARGET_Q[5:10], horizon=9,
                                       max_dq=.07, max_ddq=.2, solver_time_limit=.0025)
        rng = np.random.default_rng(20260927)
        for _ in range(20):
            m = rng.normal(size=(policy.num_variables, 30))
            full_p = m @ m.T + np.eye(policy.num_variables)*.1
            linear = rng.normal(size=policy.num_variables)
            lower, upper = policy._l_template.copy(), policy._u_template.copy()
            lower[:10] = upper[:10] = np.r_[EXPECTED_TARGET_Q[5:10], rng.uniform(-.01,.01,5)]
            pc,qc,lc,uc,offset,_ = policy.condense(sparse.csc_matrix(np.triu(full_p)),linear,lower,upper)
            u = rng.uniform(-1,1,45)
            z = offset + policy._T @ u
            states, inputs = policy._unpack_solution(z)
            np.testing.assert_allclose(states, policy._rollout(lower[:10], inputs), atol=1e-14)
            constant = .5*offset@full_p@offset+linear@offset
            self.assertAlmostEqual(.5*z@full_p@z+linear@z, .5*u@pc@u+qc@u+constant, places=11)
            np.testing.assert_allclose((policy._ac@u-lc)/policy._row_scale,
                (policy._A_cons@z-lower)[policy._ineq_start:],atol=1e-12)

    def test_core_reference_bounds_fresh_horizon_and_tracking(self):
        c = RightArmHardwareMpc(EXPECTED_TARGET_Q[5:10])
        try:
            c.warmup(EXPECTED_TARGET_Q,[1,0,0,0],count=8)
            previous_q, previous_dq = c.nominal.copy(), np.zeros(5)
            for i in range(600):
                slots = EXPECTED_TARGET_Q.copy()
                slots[5:10] = previous_q + .01*np.sin(i*.013)
                c.set_measured_dq(np.ones(5)*.15)  # NOT a measured-velocity trip
                c.set_disturbance_horizon(horizon(.02*np.sin(i*.05)))
                q,dq,diag = c.step(slots,[1,0,0,0],0,.018 if i%40==0 else .006)
                self.assertTrue(diag["mpc"]["solved"])
                self.assertLessEqual(np.max(np.abs(q-c.nominal)),np.deg2rad(5)+1e-9)
                self.assertLessEqual(np.max(np.abs(dq)),.07+1e-9)
                self.assertLessEqual(np.max(np.abs(dq-previous_dq)),.2*.006+1e-9)
                np.testing.assert_allclose(q-previous_q,dq*.006,atol=1e-12)
                previous_q,previous_dq=q,dq
            with self.assertRaisesRegex(HardwareMpcError,"fresh"):
                c.step(EXPECTED_TARGET_Q,[1,0,0,0],0,.006)
            c.set_disturbance_horizon(horizon())
            with mock.patch.object(c.policy,"_solve_qp",return_value=(None,RuntimeError("injected"))):
                with self.assertRaisesRegex(HardwareMpcError,"rejected"):
                    c.step(EXPECTED_TARGET_Q,[1,0,0,0],0,.006)
        finally:
            c.close()

    def test_horizon_refuses_invalid_rotation(self):
        c=RightArmHardwareMpc(EXPECTED_TARGET_Q[5:10])
        try:
            h=horizon(); h.nodes[0].rot_world_body[0,0]=2
            with self.assertRaises(ValueError):c.set_disturbance_horizon(h)
        finally:c.close()

    def test_batched_cost_matches_simulation_objective(self):
        c=RightArmHardwareMpc(EXPECTED_TARGET_Q[5:10])
        try:
            rng=np.random.default_rng(42)
            shapes=dict(C_omega=(3,5),D_omega=(3,),G_g=(2,10),d_g=(2,),
                        C_acc=(3,5),B_acc=(3,5),D_acc=(3,),
                        C_alpha=(3,5),B_alpha=(3,5),D_alpha=(3,))
            for _ in range(30):
                terms=[{k:rng.normal(size=v) for k,v in shapes.items()} for _ in range(10)]
                actual=c.policy._build_cost(terms)
                expected=ArmMPCPolicy._build_cost(c.policy,terms)
                np.testing.assert_allclose(actual[1],expected[1],atol=1e-11)
                for a,b in zip(actual[0],expected[0]):
                    np.testing.assert_allclose(a,b,atol=1e-11)
                policy=c.policy
                full=sparse.block_diag(actual[0]).toarray()
                lower,upper=policy._l_template.copy(),policy._u_template.copy()
                lower[:10]=upper[:10]=np.r_[EXPECTED_TARGET_Q[5:10],np.zeros(5)]
                reference=policy.condense(sparse.csc_matrix(np.triu(full)),actual[1],lower,upper)
                fast=policy.condense(None,actual[1],lower,upper,cost_blocks=policy._cost_blocks)
                for a,b in zip(reference[:5],fast[:5]):
                    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-12)
                z=rng.normal(size=policy.num_variables)
                self.assertAlmostEqual(policy._objective(z,actual[1],None),
                                       .5*z@full@z+actual[1]@z,places=9)
        finally:c.close()

    def test_predictor_past_only_h0_and_so3(self):
        a,b=HardwareMpcPredictor("hold_current"),HardwareMpcPredictor("hold_current")
        for p in (a,b):
            for t in range(0,102_000_000,2_000_000):
                p.observe_low(t,np.zeros(35),np.zeros(35))
                p.observe_imu(t,[1,0,0,0],[0,0,.2],[1,0,9.81])
        b.observe_low(200_000_000,np.ones(35)*500,np.ones(35)*500)
        b.observe_imu(200_000_000,[0,1,0,0],[300,300,300],[50,50,50])
        x,y=a.query(100_000_000,np.pi/2),b.query(100_000_000,np.pi/2)
        np.testing.assert_allclose(x.diagnostics["features"],y.diagnostics["features"])
        np.testing.assert_allclose(x.horizon.nodes[0].acc_world,[0,-1,0],atol=1e-12)
        self.assertEqual(len(x.horizon.nodes),10);self.assertEqual(len(x.horizon.intervals),9)
        for node in x.horizon.nodes:
            np.testing.assert_allclose(node.rot_world_body.T@node.rot_world_body,np.eye(3),atol=1e-12)
        expected=Rotation.from_euler("z",.2*.054).as_matrix()@x.horizon.nodes[0].rot_world_body
        np.testing.assert_allclose(x.horizon.nodes[-1].rot_world_body,expected,atol=1e-12)
        with self.assertRaisesRegex(RuntimeError,"backlog"):
            a.query(200_000_000,0)

    def test_bank_matches_its_explicit_formula_and_provenance(self):
        b=FrozenInnovationBank()
        for index in (0,100,1000,15098):
            feature=b.train_z[index]/b.feature_scale*b.std+b.mean
            current=b.current[index]+.01
            forecast,d=b.predict(feature,current)
            rows,w=np.array(d["neighbor_indices"]),np.array(d["neighbor_weights"])
            expected=np.einsum("k,kho->ho",w,b.future[rows])
            correction=current-w@b.current[rows]
            expected[:,:9]+=np.repeat(b.decay,3,axis=1)*correction[:9]
            expected[:,9:]+=correction[9:]
            np.testing.assert_allclose(forecast,expected,atol=1e-12)
        self.assertEqual(set(b.train_trial),{1,2,3,4,5,7,8})

    def test_output_requires_separate_explicit_opt_in_and_profile(self):
        with mock.patch("g1_walk_mpc.run_device") as run:
            self.assertEqual(main(["fake0","--execute"]),1)
            run.assert_not_called()
        template=Path(__file__).resolve().parents[1]/"profiles/mpc_walk_capture.template"
        with self.assertRaisesRegex(ValueError,"FIELD_REVIEWED"):
            load_profile(template,"mpc")
        text=template.read_text().replace("=UNSET","=test-only").replace("=DRAFT","=FIELD_REVIEWED")
        for name in REQUIRED_CONFIRMATIONS:
            name=name.replace("pid","mpc")
            text=text.replace(name+"=false",name+"=true")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"profile";path.write_text(text)
            profile=load_profile(path,"mpc")
            self.assertEqual(profile["control_period_ms"],"6")
            journal=MpcJournal(Path(directory)/"log")
            journal.record({"schema":"g1_hardware_pid_command_v1","pid_active":True})
            journal.close()
            row=json.loads((Path(directory)/"log/raw.jsonl").read_text())
            self.assertEqual(row["schema"],"g1_hardware_mpc_command_v1")
            self.assertTrue(row["mpc_active"])


if __name__ == "__main__":
    unittest.main()
