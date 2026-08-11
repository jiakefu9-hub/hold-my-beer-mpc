"""Fail-closed tests for the Unitree G1 hardware shadow boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
import unittest

import mujoco
import numpy as np
import yaml

from right_arm_runtime.hardware_shadow import (
    ARM_SDK_MOTOR_INDICES,
    G1_23DOF_MOTOR_TO_JOINT,
    HardwareContractError,
    HardwareFrameContract,
    HardwareShadowController,
    HardwareStateError,
    G1HardwareStateAdapter,
    ShadowCommandBuilder,
    load_hardware_shadow_config,
)
from right_arm_runtime.unitree_shm import RobotStateSnapshot


REPO_ROOT = Path(__file__).resolve().parents[2]


def _controller_config() -> dict:
    with (REPO_ROOT / "configs/g1.yaml").open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _contract(*, require_verified: bool = True) -> HardwareFrameContract:
    payload = load_hardware_shadow_config(
        REPO_ROOT / "configs/g1_hardware_shadow.yaml"
    )["hardware_shadow"]
    payload["joint_mapping_verified"] = True
    payload["robot_tick_monotonic_verified"] = True
    payload["imu"]["contract_verified"] = True
    payload["allowed_mode_pr"] = [0]
    payload["allowed_mode_machine"] = [0]
    return HardwareFrameContract.from_mapping(
        payload, require_verified=require_verified
    )


def _model() -> mujoco.MjModel:
    xml_path = REPO_ROOT / _controller_config()["xml_path"]
    return mujoco.MjModel.from_xml_path(str(xml_path))


def _state(
    timestamp_ns: int,
    sample_id: int,
    *,
    q: tuple[float, ...] | None = None,
    dq: tuple[float, ...] | None = None,
    gyro: tuple[float, ...] = (0.0, 0.0, 0.0),
    accelerometer: tuple[float, ...] = (0.0, 0.0, 9.81),
    temperature: tuple[tuple[int, int], ...] | None = None,
) -> RobotStateSnapshot:
    return RobotStateSnapshot(
        monotonic_timestamp_ns=timestamp_ns,
        sample_id=sample_id,
        robot_tick=sample_id,
        mode_pr=0,
        mode_machine=0,
        q=(0.0,) * 35 if q is None else q,
        dq=(0.0,) * 35 if dq is None else dq,
        ddq=(0.0,) * 35,
        tau_est=(0.0,) * 35,
        motor_temperature_c=(
            ((20, 25),) * 35 if temperature is None else temperature
        ),
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        imu_gyroscope=gyro,
        imu_accelerometer=accelerometer,
        imu_rpy=(0.0, 0.0, 0.0),
    )


class HardwareContractTest(unittest.TestCase):
    def test_state_bridge_source_has_no_command_transport(self):
        source = (
            REPO_ROOT
            / "cpp/unitree_arm_adapter/src/state_bridge_main.cpp"
        ).read_text(encoding="utf-8")
        self.assertNotIn("LowCmd_", source)
        self.assertNotIn("ChannelPublisher", source)
        self.assertNotIn("rt/arm_sdk", source)

    def test_repository_config_is_deliberately_not_armed(self):
        payload = load_hardware_shadow_config(
            REPO_ROOT / "configs/g1_hardware_shadow.yaml"
        )["hardware_shadow"]
        HardwareFrameContract.from_mapping(payload, require_verified=False)
        with self.assertRaises(HardwareContractError):
            HardwareFrameContract.from_mapping(payload, require_verified=True)

    def test_mapping_is_exactly_the_23_dof_mjcf_set(self):
        model = _model()
        self.assertEqual(len(G1_23DOF_MOTOR_TO_JOINT), 23)
        self.assertEqual(tuple(sorted(G1_23DOF_MOTOR_TO_JOINT)), (
            *range(13), *range(15, 20), *range(22, 27)
        ))
        for name in G1_23DOF_MOTOR_TO_JOINT.values():
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name),
                0,
            )


class HardwareStateAdapterTest(unittest.TestCase):
    def setUp(self):
        self.model = _model()
        self.adapter = G1HardwareStateAdapter(self.model, _contract())
        self.started_ns = time.monotonic_ns()

    def test_units_mapping_orientation_and_causal_alpha(self):
        q = np.zeros(35, dtype=np.float64)
        dq = np.zeros(35, dtype=np.float64)
        for motor, value in zip((22, 23, 24, 25, 26), (0.01, -0.02, 0.03, 0.04, -0.05)):
            q[motor] = value
            dq[motor] = 2.0 * value
        q[12] = 0.2
        first = self.adapter.convert(
            _state(
                self.started_ns,
                1,
                q=tuple(q),
                dq=tuple(dq),
                gyro=(0.0, 0.0, 0.02),
            ),
            now_ns=self.started_ns + 100_000,
        )
        self.assertFalse(first.derivative_ready)
        np.testing.assert_allclose(first.right_arm_q, q[22:27])
        np.testing.assert_allclose(first.right_arm_dq, dq[22:27])
        np.testing.assert_allclose(first.lower_body_q, q[:12])
        np.testing.assert_allclose(first.torso_linear_acceleration_world, 0.0)

        data = mujoco.MjData(self.model)
        data.qpos[:] = first.qpos_mujoco
        mujoco.mj_fwdPosition(self.model, data)
        imu_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso"
        )
        np.testing.assert_allclose(
            data.site_xmat[imu_id].reshape(3, 3), np.eye(3), atol=1e-12
        )

        second_timestamp = self.started_ns + 2_000_000
        second = self.adapter.convert(
            _state(
                second_timestamp,
                2,
                q=tuple(q),
                dq=tuple(dq),
                gyro=(0.0, 0.0, 0.04),
            ),
            now_ns=second_timestamp + 100_000,
        )
        self.assertTrue(second.derivative_ready)
        np.testing.assert_allclose(
            second.torso_angular_acceleration_world,
            (0.0, 0.0, 10.0),
            atol=1e-12,
        )

    def test_stale_future_duplicate_nonfinite_and_temperature_fail_closed(self):
        stale = _state(self.started_ns - 21_000_000, 1)
        with self.assertRaisesRegex(HardwareStateError, "stale"):
            self.adapter.convert(stale, now_ns=self.started_ns)

        future = _state(self.started_ns + 2_000_000, 1)
        with self.assertRaisesRegex(HardwareStateError, "future"):
            G1HardwareStateAdapter(self.model, _contract()).convert(
                future, now_ns=self.started_ns
            )

        valid = _state(self.started_ns, 1)
        duplicate_adapter = G1HardwareStateAdapter(self.model, _contract())
        duplicate_adapter.convert(valid, now_ns=self.started_ns + 100_000)
        with self.assertRaisesRegex(HardwareStateError, "sample_id"):
            duplicate_adapter.convert(
                valid, now_ns=self.started_ns + 200_000
            )

        tick_adapter = G1HardwareStateAdapter(self.model, _contract())
        tick_adapter.convert(valid, now_ns=self.started_ns + 100_000)
        with self.assertRaisesRegex(HardwareStateError, "robot tick"):
            tick_adapter.convert(
                replace(
                    valid,
                    monotonic_timestamp_ns=self.started_ns + 2_000_000,
                    sample_id=2,
                ),
                now_ns=self.started_ns + 2_100_000,
            )

        bad_q = list(valid.q)
        bad_q[22] = float("nan")
        with self.assertRaisesRegex(HardwareStateError, "nonfinite"):
            G1HardwareStateAdapter(self.model, _contract()).convert(
                replace(valid, q=tuple(bad_q)),
                now_ns=self.started_ns + 100_000,
            )

        temperatures = list(valid.motor_temperature_c)
        temperatures[22] = (86, 25)
        with self.assertRaisesRegex(HardwareStateError, "temperature"):
            G1HardwareStateAdapter(self.model, _contract()).convert(
                replace(valid, motor_temperature_c=tuple(temperatures)),
                now_ns=self.started_ns + 100_000,
            )


class ShadowCommandTest(unittest.TestCase):
    def test_command_is_protocol_shaped_but_cannot_request_output(self):
        model = _model()
        adapter = G1HardwareStateAdapter(model, _contract())
        timestamp = time.monotonic_ns()
        q = tuple(0.001 * index for index in range(35))
        state = _state(timestamp, 1, q=q)
        observation = adapter.convert(state, now_ns=timestamp + 100_000)
        command = ShadowCommandBuilder(_controller_config()).build(
            observation,
            np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.asarray([0.01, 0.02, 0.03, 0.04, 0.05]),
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
            state,
        )
        self.assertEqual(len(command.q_ref), 13)
        self.assertEqual(
            command.q_ref[:5], tuple(q[index] for index in ARM_SDK_MOTOR_INDICES[:5])
        )
        self.assertEqual(command.q_ref[5:10], (0.1, 0.2, 0.3, 0.4, 0.5))
        self.assertEqual(command.ddq_des[5:10], (1.0, 2.0, 3.0, 4.0, 5.0))
        self.assertEqual(command.tau_ff, (0.0,) * 13)
        self.assertEqual(command.arm_weight, 0.0)
        self.assertFalse(command.request_output)
        self.assertFalse(command.publish_performed)
        self.assertFalse(command.ready_for_output)


class CompleteShadowPathTest(unittest.TestCase):
    def test_real_state_shape_reaches_template_mpc_and_command_build(self):
        config = _controller_config()
        # Keep this integration smoke independent of the optional local C++
        # Pinocchio shared object; production keeps the configured backend.
        config["mpc_prediction_kinematics_backend"] = "mujoco"
        started_ns = time.monotonic_ns()
        with HardwareShadowController(
            repo_dir=REPO_ROOT,
            controller_config=config,
            contract=_contract(),
            predictor_name="template",
        ) as controller:
            self.assertIsNone(
                controller.process(
                    _state(started_ns, 1), now_ns=started_ns + 100_000
                )
            )
            timestamp = started_ns + 2_000_000
            result = controller.process(
                _state(timestamp, 2), now_ns=timestamp + 100_000
            )
            self.assertIsNotNone(result)
            self.assertTrue(result.mpc_success)
            self.assertEqual(result.predictor_requested, "template")
            self.assertEqual(result.predictor_used, "template")
            self.assertFalse(result.command.request_output)
            self.assertFalse(result.command.publish_performed)
            self.assertIn("complete_shadow_path", result.timing_s)
            summary = controller.summary()
            self.assertEqual(summary["timing"]["count"], 1)
            self.assertEqual(summary["output_capability"], "absent")


if __name__ == "__main__":
    unittest.main()
