import unittest

import mujoco
import numpy as np
import yaml

from right_arm_runtime import (
    CppDdqTorqueMapper,
    CppNoSafeTorqueError,
    RightArmSimProcess,
)
from sim_support import (
    NoSafeTorqueError,
    local_forward_dynamics_torque_mapping,
)


class DdqSafetyFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("configs/g1.yaml", encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)
        cls.model = mujoco.MjModel.from_xml_path(cls.config["xml_path"])
        names = (
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        )
        joint_ids = np.array(
            [
                mujoco.mj_name2id(
                    cls.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                for name in names
            ],
            dtype=np.int32,
        )
        cls.qvel_indices = cls.model.jnt_dofadr[joint_ids].astype(
            np.int32
        )
        cls.qpos_indices = cls.model.jnt_qposadr[joint_ids].astype(
            np.int32
        )
        cls.ctrl_indices = np.array(
            [
                mujoco.mj_name2id(
                    cls.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                )
                for name in names
            ],
            dtype=np.int32,
        )
        cls.torque_limits = cls.model.jnt_actfrcrange[joint_ids].copy()
        cls.mapper_kwargs = {
            "perturbation": cls.config["ddq_execution_perturbation"],
            "regularization": cls.config["ddq_execution_regularization"],
            "second_pass_error_threshold": cls.config[
                "ddq_execution_second_pass_error_threshold"
            ],
            "max_joint_error": cls.config["ddq_execution_max_joint_error"],
            "max_abs_qacc": cls.config["ddq_execution_max_abs_qacc"],
            "enable_second_pass": cls.config[
                "mpc_execution_enable_second_pass"
            ],
            "max_safety_rescue_passes": cls.config[
                "mpc_execution_safety_rescue_passes"
            ],
            "previous_executed_tau": None,
        }

    def make_data(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        return data

    def python_compute(self, desired, safe_hold):
        data = self.make_data()
        return local_forward_dynamics_torque_mapping(
            self.model,
            data,
            mujoco.MjData(self.model),
            data.ctrl.copy(),
            np.full(5, desired, dtype=np.float64),
            np.zeros(5, dtype=np.float64),
            self.qvel_indices,
            self.ctrl_indices,
            self.torque_limits,
            safe_hold_tau=np.asarray(safe_hold, dtype=np.float64),
            **self.mapper_kwargs,
        )

    def cpp_compute(self, desired, safe_hold):
        data = self.make_data()
        mapper = CppDdqTorqueMapper(self.config["xml_path"])
        self.addCleanup(mapper.close)
        return mapper.compute(
            data=data,
            fixed_ctrl=data.ctrl.copy(),
            desired_qacc=np.full(5, desired, dtype=np.float64),
            tau_nominal=np.zeros(5, dtype=np.float64),
            safe_hold_tau=np.asarray(safe_hold, dtype=np.float64),
            **self.mapper_kwargs,
        )

    def test_normal_candidate_does_not_trigger_final_fallback(self):
        _, result = self.python_compute(0.0, np.zeros(5))
        self.assertTrue(result.final_output_certified)
        self.assertFalse(result.safe_hold_used)
        self.assertFalse(result.safety_line_search_used)
        self.assertEqual(result.safety_line_search_attempts, 0)

    def test_no_previous_uses_certified_safe_hold(self):
        tau, result = self.python_compute(80.0, np.zeros(5))
        self.assertTrue(result.final_output_certified)
        self.assertTrue(result.safe_hold_used)
        self.assertFalse(result.safety_line_search_used)
        self.assertEqual(result.safety_line_search_attempts, 4)
        np.testing.assert_array_equal(tau, np.zeros(5))
        self.assertLessEqual(np.max(np.abs(result.qacc_validated)), 10.0)

    def test_line_search_selects_certified_point_closer_than_safe_hold(self):
        tau, result = self.python_compute(20.0, np.zeros(5))
        self.assertTrue(result.final_output_certified)
        self.assertFalse(result.safe_hold_used)
        self.assertTrue(result.safety_line_search_used)
        self.assertGreater(result.safety_line_search_attempts, 0)
        self.assertLessEqual(result.safety_line_search_attempts, 4)
        self.assertGreater(np.linalg.norm(tau), 0.0)
        self.assertLessEqual(np.max(np.abs(result.qacc_validated)), 10.0)

    def test_unsafe_safe_hold_returns_no_safe_torque(self):
        unsafe_hold = np.full(5, 25.0)
        with self.assertRaisesRegex(NoSafeTorqueError, "NO_SAFE_TORQUE"):
            self.python_compute(80.0, unsafe_hold)
        with self.assertRaisesRegex(
            CppNoSafeTorqueError, "NO_SAFE_TORQUE"
        ):
            self.cpp_compute(80.0, unsafe_hold)

    def test_python_and_cpp_success_branches_match_and_are_bounded(self):
        for desired in (0.0, 20.0, 80.0):
            with self.subTest(desired=desired):
                python_tau, python_result = self.python_compute(
                    desired, np.zeros(5)
                )
                cpp_result = self.cpp_compute(desired, np.zeros(5))
                np.testing.assert_allclose(
                    cpp_result.values["tau_cmd"],
                    python_tau,
                    atol=1e-8,
                    rtol=1e-8,
                )
                np.testing.assert_allclose(
                    cpp_result.values["qacc_validated"],
                    python_result.qacc_validated,
                    atol=1e-8,
                    rtol=1e-8,
                )
                for name in (
                    "safe_hold_used",
                    "safety_line_search_used",
                    "safety_line_search_attempts",
                    "final_output_certified",
                    "no_safe_torque",
                ):
                    self.assertEqual(
                        int(cpp_result.values[name]),
                        int(getattr(python_result, name)),
                    )
                self.assertLessEqual(
                    int(cpp_result.values["safety_line_search_attempts"]), 4
                )

    def test_sync_and_process_mapper_outputs_match(self):
        data = self.make_data()
        arm_q = data.qpos[self.qpos_indices].copy()
        arm_dq = data.qvel[self.qvel_indices].copy()
        zero = np.zeros(5, dtype=np.float64)
        process = RightArmSimProcess(
            self.config["xml_path"],
            nq=self.model.nq,
            nv=self.model.nv,
            nu=self.model.nu,
            nbody=self.model.nbody,
            kp=np.asarray(self.config["arm_waist_kps"])[6:11],
            kd=np.asarray(self.config["arm_waist_kds"])[6:11],
            timeout_damping=self.config[
                "right_arm_executor_timeout_damping"
            ],
            q_ref_min=np.deg2rad(self.config["mpc_q_min_deg"]),
            q_ref_max=np.deg2rad(self.config["mpc_q_max_deg"]),
            dq_ref_abs_max=self.config[
                "right_arm_executor_dq_ref_abs_max"
            ],
            tau_min=self.torque_limits[:, 0],
            tau_max=self.torque_limits[:, 1],
            mapper_perturbation=self.mapper_kwargs["perturbation"],
            mapper_regularization=self.mapper_kwargs["regularization"],
            mapper_second_pass_error_threshold=self.mapper_kwargs[
                "second_pass_error_threshold"
            ],
            mapper_max_joint_error=self.mapper_kwargs["max_joint_error"],
            mapper_max_abs_qacc=self.mapper_kwargs["max_abs_qacc"],
            mapper_enable_second_pass=self.mapper_kwargs[
                "enable_second_pass"
            ],
            mapper_max_safety_rescue_passes=self.mapper_kwargs[
                "max_safety_rescue_passes"
            ],
        )
        self.addCleanup(process.close)
        process_result = process.execute(
            simulation_time=0.0,
            command_timestamp=0.0,
            command_id=1,
            command_source_state_id=1,
            execution_state_id=1,
            mapping_update_due=True,
            mujoco_timestep=self.model.opt.timestep,
            friction_breakaway_steps=self.config[
                "ddq_pinocchio_friction_breakaway_steps"
            ],
            qpos=data.qpos,
            qvel=data.qvel,
            reference_qacc=data.qacc,
            fixed_ctrl=data.ctrl,
            qacc_warmstart=data.qacc_warmstart,
            qfrc_applied=data.qfrc_applied,
            xfrc_applied=data.xfrc_applied,
            right_arm_q=arm_q,
            right_arm_dq=arm_dq,
            q_ref=arm_q,
            dq_ref=zero,
            ddq_des=np.full(5, 20.0),
            tau_passive=data.qfrc_passive[self.qvel_indices],
            friction_loss=self.model.dof_frictionloss[self.qvel_indices],
            tau_pd=zero,
            previous_executed_tau=None,
        )
        mapper = CppDdqTorqueMapper(self.config["xml_path"])
        self.addCleanup(mapper.close)
        sync = mapper.compute(
            data=data,
            fixed_ctrl=data.ctrl,
            desired_qacc=np.full(5, 20.0),
            tau_nominal=np.asarray(process_result.rnea_output.tau_ff),
            safe_hold_tau=zero,
            **self.mapper_kwargs,
        )
        np.testing.assert_allclose(
            process_result.mapper_output.tau_cmd,
            sync.values["tau_cmd"],
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            process_result.mapper_output.qacc_validated,
            sync.values["qacc_validated"],
            atol=1e-8,
            rtol=1e-8,
        )
        self.assertEqual(
            int(process_result.mapper_output.final_output_certified), 1
        )
        self.assertEqual(int(process_result.mapper_output.no_safe_torque), 0)
        self.assertLessEqual(
            int(process_result.mapper_output.safety_line_search_attempts), 4
        )


if __name__ == "__main__":
    unittest.main()
