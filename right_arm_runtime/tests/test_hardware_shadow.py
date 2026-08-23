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
from disturbance_template.full_task_protocol import (
    DEFAULT_FULL_TASK_PROTOCOL,
    direct_step_planned_command,
)
from right_arm_runtime.control_contracts import TaskClockEvent
from right_arm_runtime.full_task_control_core import FullTaskControlCoreError
from right_arm_runtime.hardware_output_contract import (
    FakeHardwareCommandSink,
    ValidatedStateIdentity,
    certify_for_offline_fake_sink,
)
from right_arm_runtime.unitree_shm import RobotStateSnapshot


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = DEFAULT_FULL_TASK_PROTOCOL
NOMINAL_COMMAND = np.asarray([0.5, 0.0, 0.0127], dtype=np.float64)


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


def _task_event(
    anchor: int,
    *,
    session_nonce: str,
    task_epoch_id: str,
    source_sample_id: int,
    source_timestamp_ns: int,
) -> TaskClockEvent:
    task_time_ns = anchor * 6_000_000
    planned = direct_step_planned_command(
        task_time_ns * 1e-9, NOMINAL_COMMAND, PROTOCOL
    ).planned_command
    runtime = planned.copy()
    runtime[2] = 0.02
    return TaskClockEvent(
        session_nonce=session_nonce,
        task_epoch_id=task_epoch_id,
        producer_sequence=anchor,
        event_monotonic_timestamp_ns=source_timestamp_ns + 300_000,
        source_sample_id=source_sample_id,
        task_time_ns=task_time_ns,
        full_task_anchor=anchor,
        planned_command_vx_vy_wz=tuple(planned),
        runtime_command_vx_vy_wz=tuple(runtime),
        heading_reference_rad=0.0,
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
        self.assertIn('kLowStateTopic = "rt/lowstate"', source)
        self.assertIn('kTorsoImuTopic = "rt/secondary_imu"', source)
        self.assertIn("max_source_skew_us{5000}", source)
        self.assertIn("LowStateCrcValid", source)
        self.assertIn("low_state_crc_rejected_count_", source)
        self.assertIn("crc32_core", source)

        cmake = (
            REPO_ROOT / "cpp/unitree_arm_adapter/CMakeLists.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("UNITREE_ARM_ADAPTER_BUILD_STATE_BRIDGE", cmake)
        launcher = (
            REPO_ROOT
            / "tools/realtime/run_hardware_state_inspection.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("-DUNITREE_ARM_ADAPTER_BUILD_DDS=OFF", launcher)
        self.assertIn(
            "-DUNITREE_ARM_ADAPTER_BUILD_STATE_BRIDGE=ON", launcher
        )
        self.assertNotIn("--enable-output", launcher)

    def test_repository_config_is_deliberately_not_armed(self):
        payload = load_hardware_shadow_config(
            REPO_ROOT / "configs/g1_hardware_shadow.yaml"
        )["hardware_shadow"]
        self.assertEqual(payload["imu"]["source_topic"], "rt/secondary_imu")
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
        self.assertEqual(first.validated_timestamp_ns, self.started_ns + 100_000)
        self.assertTrue(first.capabilities.right_arm_joint_state)
        self.assertTrue(first.capabilities.torso_rotation)
        self.assertFalse(first.capabilities.torso_angular_acceleration)
        self.assertFalse(first.capabilities.floating_base_translation)
        self.assertFalse(first.capabilities.floating_base_velocity)
        self.assertFalse(first.capabilities.foot_contacts)
        self.assertFalse(first.capabilities.external_forces)
        self.assertFalse(first.capabilities.mpc_observation_complete)
        self.assertFalse(first.capabilities.hardware_torque_state_complete)
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
        self.assertTrue(second.capabilities.mpc_observation_complete)
        self.assertFalse(second.capabilities.hardware_torque_state_complete)
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
    def test_learned_predictor_is_not_a_shadow_option(self):
        config = _controller_config()
        config["mpc_prediction_kinematics_backend"] = "mujoco"
        with self.assertRaisesRegex(
            HardwareContractError, "template or full_task_template"
        ):
            HardwareShadowController(
                repo_dir=REPO_ROOT,
                controller_config=config,
                contract=_contract(),
                predictor_name="hybrid_residual",
            )

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


class FullTaskOfflineShadowPathTest(unittest.TestCase):
    session_nonce = "h3-offline-session"
    task_epoch_id = "h3-offline-epoch"

    def _controller(self) -> HardwareShadowController:
        config = _controller_config()
        config["mpc_prediction_kinematics_backend"] = "mujoco"
        return HardwareShadowController(
            repo_dir=REPO_ROOT,
            controller_config=config,
            contract=_contract(),
            predictor_name="full_task_template",
        )

    def _prime_and_reset(
        self, controller: HardwareShadowController, base_ns: int
    ) -> None:
        controller.prime_full_task_state(
            _state(base_ns, 1), now_ns=base_ns + 100_000
        )
        controller.reset_full_task_epoch(
            session_nonce=self.session_nonce,
            task_epoch_id=self.task_epoch_id,
        )

    def _process_anchor(
        self,
        controller: HardwareShadowController,
        *,
        base_ns: int,
        anchor: int,
        sample_id: int | None = None,
        event_overrides: dict | None = None,
    ):
        source_sample_id = anchor + 2 if sample_id is None else sample_id
        source_timestamp_ns = base_ns + 2_000_000 + anchor * 6_000_000
        event = _task_event(
            anchor,
            session_nonce=self.session_nonce,
            task_epoch_id=self.task_epoch_id,
            source_sample_id=source_sample_id,
            source_timestamp_ns=source_timestamp_ns,
        )
        if event_overrides:
            event = replace(event, **event_overrides)
        raw = _state(source_timestamp_ns, source_sample_id)
        result = controller.process(
            raw,
            now_ns=source_timestamp_ns + 200_000,
            task_clock_event=event,
        )
        return raw, result

    def test_first_lowstate_cannot_implicitly_start_task_time(self):
        base_ns = time.monotonic_ns()
        with self._controller() as controller:
            controller.prime_full_task_state(
                _state(base_ns, 1), now_ns=base_ns + 100_000
            )
            source_ns = base_ns + 2_000_000
            with self.assertRaisesRegex(
                HardwareContractError, "explicit TaskClockEvent"
            ):
                controller.process(
                    _state(source_ns, 2), now_ns=source_ns + 100_000
                )

    def test_anchor_four_handoff_and_fake_sink_remain_output_absent(self):
        base_ns = time.monotonic_ns()
        with self._controller() as controller:
            self._prime_and_reset(controller, base_ns)
            sink = FakeHardwareCommandSink(
                session_nonce=self.session_nonce,
                watchdog_timeout_ns=10_000_000,
            )
            results = []
            for anchor in range(5):
                raw, result = self._process_anchor(
                    controller, base_ns=base_ns, anchor=anchor
                )
                results.append(result)
                identity = ValidatedStateIdentity(
                    session_nonce=self.session_nonce,
                    sample_id=result.source_sample_id,
                    source_timestamp_ns=result.intent.source_timestamp_ns,
                    validated_timestamp_ns=(
                        result.intent.source_timestamp_ns + 200_000
                    ),
                    arm_sdk_q=tuple(
                        raw.q[index] for index in ARM_SDK_MOTOR_INDICES
                    ),
                )
                certified = certify_for_offline_fake_sink(
                    result.proposal,
                    identity,
                    now_ns=result.proposal.generated_timestamp_ns + 1,
                )
                receipt = sink.submit(
                    certified,
                    now_ns=result.proposal.generated_timestamp_ns + 2,
                )
                self.assertTrue(receipt.accepted)
                self.assertFalse(receipt.dds_write_performed)
                self.assertFalse(receipt.hardware_output_performed)
            self.assertEqual(
                [item.intent.mpc_output_enabled for item in results],
                [False, False, False, False, True],
            )
            self.assertTrue(results[4].intent.first_mpc_anchor)
            self.assertEqual(results[4].task_time_s, 0.024)
            self.assertEqual(results[4].task_anchor, 4)
            self.assertEqual(
                results[4].intent.predictor_diagnostics[
                    "template_anchor_index"
                ],
                4,
            )
            self.assertFalse(
                results[4].intent.hardware_torque_state_complete
            )
            np.testing.assert_allclose(
                results[0].command.q_ref[5:10],
                np.asarray(_controller_config()["arm_waist_target"])[6:11],
            )
            np.testing.assert_allclose(
                results[4].command.q_ref[5:10],
                results[4].intent.generated_q_ref,
            )
            summary = controller.summary()
            self.assertEqual(summary["proposal_count"], 5)
            self.assertEqual(summary["command_write_count"], 0)
            self.assertEqual(summary["publish_count"], 0)
            self.assertFalse(summary["hardware_output_authorized"])
            self.assertEqual(summary["output_capability"], "absent")
            self.assertEqual(summary["timing"]["count"], 5)

    def test_anchor_gap_replay_and_future_state_fail_closed(self):
        base_ns = time.monotonic_ns()
        with self._controller() as controller:
            self._prime_and_reset(controller, base_ns)
            self._process_anchor(controller, base_ns=base_ns, anchor=0)
            with self.assertRaises(FullTaskControlCoreError) as caught:
                self._process_anchor(
                    controller,
                    base_ns=base_ns,
                    anchor=2,
                    sample_id=3,
                    event_overrides={"producer_sequence": 1},
                )
            self.assertEqual(
                caught.exception.reason_code, "task_anchor_gap_or_replay"
            )

        base_ns = time.monotonic_ns()
        with self._controller() as controller:
            self._prime_and_reset(controller, base_ns)
            source_ns = base_ns + 2_000_000
            with self.assertRaises(FullTaskControlCoreError) as caught:
                self._process_anchor(
                    controller,
                    base_ns=base_ns,
                    anchor=0,
                    event_overrides={
                        "event_monotonic_timestamp_ns": source_ns - 1
                    },
                )
            self.assertEqual(
                caught.exception.reason_code, "task_event_future_state"
            )

    def test_complete_recording_window_builds_only_offline_proposals(self):
        base_ns = time.monotonic_ns()
        with self._controller() as controller:
            self._prime_and_reset(controller, base_ns)
            sink = FakeHardwareCommandSink(
                session_nonce=self.session_nonce,
                watchdog_timeout_ns=10_000_000,
            )
            selected = {}
            accepted = 0
            for anchor in range(PROTOCOL.recorded_anchor_count):
                raw, result = self._process_anchor(
                    controller, base_ns=base_ns, anchor=anchor
                )
                if anchor in (1066, 1067, 1333, 1343):
                    selected[anchor] = result
                identity = ValidatedStateIdentity(
                    session_nonce=self.session_nonce,
                    sample_id=result.source_sample_id,
                    source_timestamp_ns=result.intent.source_timestamp_ns,
                    validated_timestamp_ns=(
                        result.intent.source_timestamp_ns + 200_000
                    ),
                    arm_sdk_q=tuple(
                        raw.q[index] for index in ARM_SDK_MOTOR_INDICES
                    ),
                )
                command = certify_for_offline_fake_sink(
                    result.proposal,
                    identity,
                    now_ns=result.proposal.generated_timestamp_ns + 1,
                )
                receipt = sink.submit(
                    command,
                    now_ns=result.proposal.generated_timestamp_ns + 2,
                )
                accepted += int(receipt.accepted)
                self.assertFalse(receipt.dds_write_performed)
                self.assertFalse(receipt.hardware_output_performed)

            self.assertEqual(accepted, PROTOCOL.recorded_anchor_count)
            self.assertAlmostEqual(selected[1066].task_time_s, 6.396, places=12)
            self.assertAlmostEqual(selected[1067].task_time_s, 6.402, places=12)
            self.assertAlmostEqual(selected[1333].task_time_s, 7.998, places=12)
            self.assertAlmostEqual(selected[1343].task_time_s, 8.058, places=12)
            self.assertEqual(
                selected[1066].proposal.q_ref[5:10],
                tuple(selected[1066].intent.generated_q_ref),
            )
            self.assertEqual(
                selected[1067].proposal.q_ref[5:10],
                tuple(selected[1067].intent.generated_q_ref),
            )
            self.assertEqual(
                len(selected[1333].intent.disturbance_horizon.nodes), 10
            )
            self.assertAlmostEqual(7.998 + 9 * 0.006, 8.052, places=12)
            self.assertFalse(
                selected[1343].intent.predictor_diagnostics["fallback_used"]
            )
            summary = controller.summary()
            self.assertEqual(
                summary["proposal_count"], PROTOCOL.recorded_anchor_count
            )
            self.assertEqual(summary["command_write_count"], 0)
            self.assertEqual(summary["publish_count"], 0)


if __name__ == "__main__":
    unittest.main()
