"""SDK-free right-arm inverse dynamics with the measured torso as moving base.

This is an OFFLINE candidate, not calibrated hardware torque control. The
existing C++ model is reused; simulation contact/friction corrections are not.
Inputs are expressed in fixed H0, with acceleration at imu_in_torso excluding
gravity. Neither a DDS endpoint nor a hardware command is created here.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from disturbance_types import DisturbanceInput


def _cross3(a, b):
    """Fixed 3-vector cross product; avoids NumPy's general broadcasting path."""
    return np.array((a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                     a[0]*b[1]-a[1]*b[0]))


def finite_vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return value


class RightArmInverseDynamics:
    """Borrow EndpointModel/native backend; caller owns their lifetime.

    All upstream joints are frozen in a virtual model. The floating root is
    reconstructed so that its torso IMU has the measured motion. This is NOT
    an assertion that the real pelvis/waist is stationary, and requires no
    fabricated leg accelerations or contact forces. A serial arm's inverse
    dynamics depends on its base motion and own downstream inertias.
    """

    def __init__(self, model, backend):
        self.model, self.backend = model, backend
        m = model.model
        if (backend.nq, backend.nv) != (m.nq, m.nv):
            raise ValueError("inverse-dynamics model dimensions differ")
        if backend.scene_mjcf_path != model.xml.resolve():
            raise ValueError("inverse-dynamics XML must match native backend")
        free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or m.jnt_qposadr[free[0]] != 0 or m.jnt_dofadr[free[0]] != 0:
            raise ValueError("requires the reviewed single-root G1 floating model")
        self.data = mujoco.MjData(m)
        self.q_indices = np.array(model.joint_addresses[5:10])
        self.v_indices = np.array(model.right_dofs)
        self.data.qpos[:] = m.qpos0
        self.data.qpos[:7] = [0., 0., 0., 1., 0., 0., 0.]
        mujoco.mj_kinematics(m, self.data)
        self.root_to_imu = self.data.site_xpos[model.imu_id].copy()
        self.root_from_imu = self.data.site_xmat[model.imu_id].reshape(3, 3).copy()
        self.armature = m.dof_armature[self.v_indices].copy()
        self.zeros = np.zeros(5)
        self._mass = np.empty((m.nv, m.nv))
        self._arm_block = np.ix_(self.v_indices, self.v_indices)
        self.native_dynamics = None
        bottle = m.body("right_bottle")
        self.metadata = {
            "frame": "fixed_H0; acceleration_at_torso_IMU_excludes_gravity",
            "base": "virtual_frozen_upstream_joints_with_measured_torso_motion",
            "bottle_mass_kg_from_xml": float(m.body_mass[bottle.id]),
            "bottle_com_local_m": m.body_ipos[bottle.id].tolist(),
            "bottle_diagonal_inertia_kg_m2": m.body_inertia[bottle.id].tolist(),
            "model_joint_armature_kg_m2": self.armature.tolist(),
            "hardware_parameters_calibrated": False,
            "friction_compensation": "none; do not transfer simulation friction",
            "external_hand_wrench": "assumed_zero; no contact-force feedback",
            "upstream_reactions": "represented_by_observed_torso_motion_not_contact_solver",
        }

    def prepare_state(self, q, dq, disturbance):
        """Reconstruct the current moving base; no contacts are invented."""
        q = finite_vector(q, 5, "q")
        dq = finite_vector(dq, 5, "dq")
        if not isinstance(disturbance, DisturbanceInput):
            raise ValueError("expected a DisturbanceInput in fixed H0")
        acc = finite_vector(disturbance.acc_world, 3, "IMU point acceleration")
        omega = finite_vector(disturbance.omega_world, 3, "angular velocity")
        alpha = finite_vector(disturbance.alpha_world, 3, "angular acceleration")
        R = np.asarray(disturbance.rot_world_body, dtype=float)
        if (R.shape != (3, 3) or not np.isfinite(R).all()
                or np.max(np.abs(R.T @ R-np.eye(3))) > 1e-7
                or abs(np.linalg.det(R)-1.) > 1e-7):
            raise ValueError("IMU attitude must be a proper H0-from-IMU rotation")
        root_R = R @ self.root_from_imu.T
        r = root_R @ self.root_to_imu
        # Galilean translation velocity is arbitrary. Set IMU velocity zero
        # instantaneously; root velocity then follows the rigid lever arm.
        root_v = -_cross3(omega, r)
        root_a = acc - _cross3(alpha, r) + _cross3(omega, root_v)
        d = self.data
        d.qpos[:] = self.model.model.qpos0
        d.qpos[:3] = 0.
        mujoco.mju_mat2Quat(d.qpos[3:7], np.ascontiguousarray(root_R).reshape(9))
        d.qpos[self.q_indices] = q
        d.qvel[:] = 0.
        d.qvel[:3], d.qvel[3:6] = root_v, root_R.T @ omega
        d.qvel[self.v_indices] = dq
        d.qacc[:] = 0.
        d.qacc[:3], d.qacc[3:6] = root_a, root_R.T @ alpha
        return d

    def linear_dynamics(self, q, dq, disturbance):
        """Conditional M_arm, h_arm for tau=M_arm*ddq+h_arm.

        Uses independent MuJoCo mass/bias calculations, prescribing measured
        base acceleration. No friction/contact oracle or measured torque truth.
        The same-model inverse/forward identity alone proves no hardware gain.
        """
        self.prepare_state(q, dq, disturbance)
        if self.native_dynamics is not None:
            return self.native_dynamics.linear_dynamics(q,dq,disturbance)
        return self._prepared_linear_dynamics()

    def _prepared_linear_dynamics(self):
        d = self.data
        m = self.model.model
        if hasattr(mujoco, "mj_makeM"):
            # Only mass/bias are needed, not collision/constraint assembly.
            # Verified against full fwdPosition/fwdVelocity on random states.
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_crb(m, d)
            mujoco.mj_makeM(m, d)
            mujoco.mj_comVel(m, d)
            mujoco.mj_rne(m, d, 0, d.qfrc_bias)
        else:
            mujoco.mj_fwdPosition(m, d)
            mujoco.mj_fwdVelocity(m, d)
        mujoco.mj_fullM(m, self._mass, d.qM)
        mass = self._mass[self._arm_block].copy()
        nonarm_qacc = d.qacc.copy()
        nonarm_qacc[self.v_indices] = 0.
        bias = (self._mass @ nonarm_qacc + d.qfrc_bias)[self.v_indices].copy()
        if not np.isfinite(mass).all() or not np.isfinite(bias).all():
            raise ValueError("nonfinite arm dynamics")
        np.linalg.cholesky(mass)
        return mass, bias

    def compute_with_linear_dynamics(self, q, dq, ddq, disturbance):
        """One validated state preparation for native ID and MuJoCo M/bias."""
        inverse = self.compute(q, dq, ddq, disturbance)
        mass, bias = (self.native_dynamics.linear_dynamics(q,dq,disturbance)
                      if self.native_dynamics is not None else self._prepared_linear_dynamics())
        return inverse, mass, bias

    def compute(self, q, dq, ddq, disturbance):
        begin = time.perf_counter_ns()
        ddq = finite_vector(ddq, 5, "ddq")
        d = self.prepare_state(q, dq, disturbance)
        root_a = d.qacc[:3].copy()
        d.qacc[self.v_indices] = ddq
        result = self.backend.compute_feedforward(
            d.qpos, d.qvel, ddq, self.zeros, self.zeros, .006, 1.,
            reference_qacc=d.qacc,
        )
        # The native MJCF->Pinocchio model already includes joint armature.
        # Split it for audit; NEVER add the rotor term a second time.
        rotor = self.armature * ddq
        return {
            "tau_rigid_nm": result.tau_rnea - rotor,
            "tau_armature_nm": rotor,
            "tau_model_nm": result.tau_rnea.copy(),
            "root_linear_acceleration_h0_m_s2": root_a.copy(),
            "elapsed_ms": (time.perf_counter_ns()-begin)*1e-6,
            "native_rnea_ms": result.rnea_elapsed_time*1000.,
        }
