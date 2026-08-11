import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

import mujoco
import numpy as np
import yaml

from right_arm_runtime.sim_process import (
    RightArmSimProcess,
    SimRuntimeError,
    python_layout_report,
)


class RightArmSimProcessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("configs/g1.yaml", encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)
        cls.model = mujoco.MjModel.from_xml_path(cls.config["xml_path"])
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)
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
            ]
        )
        cls.qpos_indices = cls.model.jnt_qposadr[joint_ids]
        cls.qvel_indices = cls.model.jnt_dofadr[joint_ids]

    def make_process(self, **overrides):
        config = self.config
        arguments = {
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "nbody": self.model.nbody,
            "kp": np.asarray(config["arm_waist_kps"])[6:11],
            "kd": np.asarray(config["arm_waist_kds"])[6:11],
            "timeout_damping": config[
                "right_arm_executor_timeout_damping"
            ],
            "q_ref_min": np.deg2rad(config["mpc_q_min_deg"]),
            "q_ref_max": np.deg2rad(config["mpc_q_max_deg"]),
            "dq_ref_abs_max": config[
                "right_arm_executor_dq_ref_abs_max"
            ],
            "tau_min": np.full(5, -25.0),
            "tau_max": np.full(5, 25.0),
            "mapper_perturbation": config["ddq_execution_perturbation"],
            "mapper_regularization": config["ddq_execution_regularization"],
            "mapper_second_pass_error_threshold": config[
                "ddq_execution_second_pass_error_threshold"
            ],
            "mapper_max_joint_error": config[
                "ddq_execution_max_joint_error"
            ],
            "mapper_max_abs_qacc": config[
                "ddq_execution_max_abs_qacc"
            ],
        }
        arguments.update(overrides)
        return RightArmSimProcess(
            config["xml_path"],
            **arguments,
        )

    def execute(
        self, process, *, mapping_update_due, state_id, **overrides
    ):
        data = self.data
        arguments = {
            "simulation_time": data.time,
            "command_timestamp": 0.0,
            "command_id": 1,
            "command_source_state_id": 1,
            "execution_state_id": state_id,
            "mapping_update_due": mapping_update_due,
            "mujoco_timestep": self.model.opt.timestep,
            "friction_breakaway_steps": 5.0,
            "qpos": data.qpos,
            "qvel": data.qvel,
            "reference_qacc": data.qacc,
            "fixed_ctrl": data.ctrl,
            "qacc_warmstart": data.qacc_warmstart,
            "qfrc_applied": data.qfrc_applied,
            "xfrc_applied": data.xfrc_applied,
            "right_arm_q": data.qpos[self.qpos_indices],
            "right_arm_dq": data.qvel[self.qvel_indices],
            "q_ref": data.qpos[self.qpos_indices],
            "dq_ref": np.zeros(5),
            "ddq_des": np.zeros(5),
            "tau_passive": data.qfrc_passive[self.qvel_indices],
            "friction_loss": self.model.dof_frictionloss[
                self.qvel_indices
            ],
            "tau_pd": np.zeros(5),
        }
        arguments.update(overrides)
        return process.execute(**arguments)

    @staticmethod
    def assert_worker_reaped(test_case, worker):
        test_case.assertIsNotNone(worker.poll())

    def test_start_failure_reaps_worker(self):
        # fake worker永远不创建共享内存；构造超时后必须同步终止并wait，
        # 不能把子进程留给不可靠的__del__。
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker_path = Path(temporary_directory) / "blocked_worker"
            worker_path.write_text(
                "#!/bin/sh\nexec sleep 30\n", encoding="utf-8"
            )
            worker_path.chmod(0o755)
            created_processes = []
            real_popen = subprocess.Popen

            def recording_popen(*args, **kwargs):
                worker = real_popen(*args, **kwargs)
                created_processes.append(worker)
                return worker

            with mock.patch(
                "right_arm_runtime.sim_process.subprocess.Popen",
                side_effect=recording_popen,
            ):
                with self.assertRaises(SimRuntimeError):
                    self.make_process(
                        worker_path=worker_path,
                        response_timeout_s=0.02,
                    )

        self.assertEqual(len(created_processes), 1)
        self.assert_worker_reaped(self, created_processes[0])

    def test_worker_scheduling_snapshot_matches_inherited_parent_state(self):
        process = self.make_process()
        self.addCleanup(process.close)
        snapshot = process.scheduling_snapshot()
        self.assertEqual(snapshot["policy"], os.sched_getscheduler(0))
        self.assertEqual(
            snapshot["priority"], os.sched_getparam(0).sched_priority
        )
        self.assertEqual(
            snapshot["cpu_affinity"], sorted(os.sched_getaffinity(0))
        )

    def test_invalid_request_poisons_process_and_forbids_reuse(self):
        process = self.make_process()
        worker = process._process
        invalid_qpos = self.data.qpos.copy()
        invalid_qpos[0] = np.nan
        with self.assertRaises(ValueError):
            self.execute(
                process,
                mapping_update_due=True,
                state_id=1,
                qpos=invalid_qpos,
            )
        self.assertTrue(process._failed)
        self.assertIsNone(process._process)
        self.assert_worker_reaped(self, worker)
        with self.assertRaisesRegex(SimRuntimeError, "上一次失败"):
            self.execute(
                process,
                mapping_update_due=True,
                state_id=2,
            )
        process.close()

    def test_response_timeout_poisons_process_and_forbids_reuse(self):
        process = self.make_process()
        process.response_timeout_s = 0.02
        worker = process._process
        os.kill(worker.pid, signal.SIGSTOP)
        with self.assertRaisesRegex(SimRuntimeError, "响应超时"):
            self.execute(
                process,
                mapping_update_due=True,
                state_id=1,
            )
        self.assertTrue(process._failed)
        self.assertIsNone(process._process)
        self.assert_worker_reaped(self, worker)
        with self.assertRaisesRegex(SimRuntimeError, "上一次失败"):
            self.execute(
                process,
                mapping_update_due=True,
                state_id=2,
            )
        process.close()

    def test_normal_close_is_idempotent_and_forbids_reuse(self):
        process = self.make_process()
        worker = process._process
        process.close()
        process.close()
        self.assert_worker_reaped(self, worker)
        with self.assertRaisesRegex(SimRuntimeError, "已经关闭"):
            self.execute(
                process,
                mapping_update_due=True,
                state_id=1,
            )

    def test_python_layout_matches_worker(self):
        # 临时进程只用于取得路径，立即关闭，避免单测遗留子进程。
        # 再由--print-layout检查所有协议偏移。
        temporary = self.make_process()
        worker = str(temporary.worker_path)
        temporary.close()
        output = subprocess.check_output(
            [worker, "--print-layout"], text=True
        )
        native = {
            key: int(value)
            for key, value in (
                line.split("=", 1) for line in output.splitlines()
            )
        }
        self.assertEqual(python_layout_report(), native)

    def test_external_step_and_cached_executor(self):
        with self.make_process() as process:
            first = self.execute(
                process, mapping_update_due=True, state_id=1
            )
            second = self.execute(
                process, mapping_update_due=False, state_id=2
            )
        self.assertTrue(first.mapping_updated)
        self.assertFalse(first.cached_feedforward_reused)
        self.assertFalse(second.mapping_updated)
        self.assertTrue(second.cached_feedforward_reused)
        self.assertEqual((first.request_id, second.request_id), (1, 2))
        np.testing.assert_array_equal(
            first.validated_tau_ff, second.validated_tau_ff
        )
        self.assertTrue(np.all(np.isfinite(second.final_tau)))

    def test_dead_worker_fails_closed(self):
        process = self.make_process()
        process._process.terminate()
        process._process.wait(timeout=2.0)
        with self.assertRaises(SimRuntimeError):
            self.execute(process, mapping_update_due=True, state_id=1)
        process.close()


if __name__ == "__main__":
    unittest.main()
