"""Offline mathematical checks, not identification of the physical robot."""
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from endpoint_pose import EndpointModel, ROOT
from hardware_arm_inverse_dynamics import RightArmInverseDynamics
from robot_model_backend.cpp_rnea_backend import CppRightArmRneaBackend
from disturbance_types import DisturbanceInput


class InverseDynamicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = EndpointModel()
        cls.backend = CppRightArmRneaBackend(cls.model.xml,
            library_path=ROOT/'build/right_arm_rnea/libright_arm_rnea.so')
        cls.inverse = RightArmInverseDynamics(cls.model, cls.backend)

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()

    def test_independent_mujoco_mass_and_bias_and_prescribed_base_acceleration(self):
        rng = np.random.default_rng(280926)
        model, inverse = self.model.model, self.inverse
        data = mujoco.MjData(model)
        mass = np.empty((model.nv, model.nv))
        indices = inverse.v_indices
        for _ in range(50):
            q, dq, ddq = rng.uniform(-.3,.3,5), rng.uniform(-.4,.4,5), rng.uniform(-2.,2.,5)
            disturbance = DisturbanceInput(rng.normal(size=3),rng.normal(size=3),
                rng.normal(size=3),Rotation.random(random_state=rng).as_matrix())
            result, fast_mass, fast_bias = inverse.compute_with_linear_dynamics(q,dq,ddq,disturbance)
            data.qpos[:] = inverse.data.qpos
            data.qvel[:] = inverse.data.qvel
            mujoco.mj_fwdPosition(model,data)
            mujoco.mj_fwdVelocity(model,data)
            mujoco.mj_fullM(model,mass,data.qM)
            expected = (mass @ inverse.data.qacc + data.qfrc_bias)[indices]
            np.testing.assert_allclose(result['tau_model_nm'],expected,atol=1e-10,rtol=0.)
            np.testing.assert_allclose(result['tau_rigid_nm']+result['tau_armature_nm'],
                                       expected,atol=1e-10,rtol=0.)
            # Conditional arm forward dynamics with the SAME prescribed base:
            # this is model consistency, not measured hardware acceleration.
            nonarm = inverse.data.qacc.copy(); nonarm[indices] = 0.
            bias = (mass @ nonarm + data.qfrc_bias)[indices]
            np.testing.assert_allclose(fast_mass,mass[np.ix_(indices,indices)],atol=1e-12,rtol=0.)
            np.testing.assert_allclose(fast_bias,bias,atol=1e-12,rtol=0.)
            fresh_mass,fresh_bias = inverse.linear_dynamics(q,dq,disturbance)
            np.testing.assert_allclose(fresh_mass,fast_mass,atol=1e-12,rtol=0.)
            np.testing.assert_allclose(fresh_bias,fast_bias,atol=1e-12,rtol=0.)
            actual = np.linalg.solve(mass[np.ix_(indices,indices)],result['tau_model_nm']-bias)
            np.testing.assert_allclose(actual,ddq,atol=1e-9,rtol=0.)

    def test_imu_lever_arm_and_rotation_reconstruction(self):
        R = Rotation.from_euler('xyz',[.2,-.3,1.2]).as_matrix()
        a,w,alpha = np.array([1.,2.,3.]),np.array([.3,.7,-.2]),np.array([.8,-.4,.5])
        inv = self.inverse
        inv.compute(np.zeros(5),np.zeros(5),np.zeros(5),DisturbanceInput(a,w,alpha,R))
        mujoco.mj_kinematics(self.model.model,inv.data)
        np.testing.assert_allclose(inv.data.site_xmat[self.model.imu_id].reshape(3,3),R,atol=1e-12)
        root_R = R@inv.root_from_imu.T
        r = root_R@inv.root_to_imu
        np.testing.assert_allclose(inv.data.qacc[:3]+np.cross(alpha,r)+np.cross(w,np.cross(w,r)),a,atol=1e-12)
        np.testing.assert_allclose(root_R@inv.data.qvel[3:6],w,atol=1e-12)
        self.assertGreater(np.linalg.norm(inv.data.qacc[:3]-a),.01)

    def test_fixed_heading_yaw_does_not_change_physical_torque(self):
        R = Rotation.from_euler('xyz',[.1,-.2,.7]).as_matrix()
        yaw = Rotation.from_euler('z',1.8).as_matrix()
        a,w,alpha = np.array([1.,2.,0.]),np.array([.2,-.1,.4]),np.array([.1,.2,.3])
        args = (np.ones(5)*.12,np.ones(5)*.06,np.arange(5)*.02)
        t1=self.inverse.compute(*args,DisturbanceInput(a,w,alpha,R))['tau_model_nm']
        t2=self.inverse.compute(*args,DisturbanceInput(yaw@a,yaw@w,yaw@alpha,yaw@R))['tau_model_nm']
        np.testing.assert_allclose(t1,t2,atol=1e-12)

    def test_reject_bad_inputs_and_report_unidentified_payload(self):
        d=DisturbanceInput(np.zeros(3),np.zeros(3),np.zeros(3),np.eye(3))
        with self.assertRaises(ValueError):self.inverse.compute([np.nan]*5,np.zeros(5),np.zeros(5),d)
        d.rot_world_body=np.diag([-1.,1.,1.])
        with self.assertRaises(ValueError):self.inverse.compute(np.zeros(5),np.zeros(5),np.zeros(5),d)
        self.assertEqual(self.inverse.metadata['bottle_mass_kg_from_xml'],.25)
        self.assertFalse(self.inverse.metadata['hardware_parameters_calibrated'])


if __name__ == '__main__':
    unittest.main()
