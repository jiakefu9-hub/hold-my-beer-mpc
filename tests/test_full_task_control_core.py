"""Shared full-task controller boundaries and fail-closed clock semantics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
import subprocess
import sys

import numpy as np

from disturbance_predictor import FullTaskTemplatePredictor
from disturbance_template.full_task_protocol import (
    DEFAULT_FULL_TASK_PROTOCOL,
    direct_step_planned_command,
    rotation_z,
)
from kinematics_helper import DisturbanceInput
from right_arm_runtime.control_contracts import ControlStateCapabilities
from right_arm_runtime.full_task_control_core import (
    FullTaskControlCoreError,
    FullTaskControlObservation,
    FullTaskRightArmControlCore,
    TaskClockEvent,
)
from right_arm_runtime.hardware_output_contract import (
    TaskClockEvent as HardwareTaskClockEvent,
)


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = (
    ROOT / "disturbance_template/data/full_task_template_v2/20260815_162850"
)
TEMPLATE_PATH = ASSET_DIR / "full_task_template.npz"
MANIFEST_PATH = ASSET_DIR / "full_task_template_manifest.json"
TEMPLATE_SHA256 = (
    "d4a0109adcff696936ef96160976161833ff9a7a7531e2e5d7ad9e50c10e17d4"
)
MANIFEST_SHA256 = (
    "6b48ee196d1f7d923dde057d3c0fb0e182f08512a65402c4c39c5e070a3243c6"
)
PROTOCOL = DEFAULT_FULL_TASK_PROTOCOL
NOMINAL_COMMAND = np.array([0.5, 0.0, 0.0127], dtype=np.float64)
MPC_CAPABILITIES = ControlStateCapabilities(
    right_arm_joint_state=True,
    torso_rotation=True,
    torso_angular_velocity=True,
    torso_linear_acceleration=True,
    torso_angular_acceleration=True,
)


class _FakeArmPolicy:
    horizon = PROTOCOL.horizon
    control_dt = PROTOCOL.mpc_dt

    def __init__(self) -> None:
        self.reset_count = 0
        self.compute_count = 0
        self.last_helpers = None
        self._diagnostics = {"ddq_raw": np.zeros(5, dtype=np.float64)}

    def reset(self) -> None:
        self.reset_count += 1
        self.compute_count = 0
        self.last_helpers = None
        self._diagnostics = {"ddq_raw": np.zeros(5, dtype=np.float64)}

    def compute_action(self, arm_obs, helpers):
        self.compute_count += 1
        self.last_helpers = helpers
        q = np.asarray(arm_obs["current_q"], dtype=np.float64)
        dq = np.asarray(arm_obs["current_dq"], dtype=np.float64)
        ddq = np.full(5, 0.01 * self.compute_count, dtype=np.float64)
        self._diagnostics = {"ddq_raw": ddq + 0.5, "success": True}
        return q + 0.001, dq + 0.002, ddq

    def get_last_diagnostics(self, copy_data=True):
        return dict(self._diagnostics)


def _predictor() -> FullTaskTemplatePredictor:
    return FullTaskTemplatePredictor(
        template_path=str(TEMPLATE_PATH),
        manifest_path=str(MANIFEST_PATH),
        expected_sha256=TEMPLATE_SHA256,
        expected_manifest_sha256=MANIFEST_SHA256,
        repo_dir=str(ROOT),
        control_dt=PROTOCOL.mpc_dt,
        horizon=PROTOCOL.horizon,
        expected_schema_version="full_task_template_v2",
        expected_heading_frame_version="full_task_continuous_heading_v2",
    )


def _measurement(task_time: float) -> DisturbanceInput:
    return DisturbanceInput(
        acc_world=np.array([0.2 + task_time, -0.1, 0.3]),
        omega_world=np.array([0.01, 0.02, -0.03]),
        alpha_world=np.array([-0.2, 0.1, 0.05]),
        rot_world_body=rotation_z(0.02 * task_time),
    )


def _observation(anchor: int, *, session: str = "session-a", valid: bool = True):
    task_time_ns = anchor * 6_000_000
    task_time = task_time_ns * 1e-9
    return FullTaskControlObservation(
        session_nonce=session,
        source_sample_id=anchor * PROTOCOL.mpc_stride,
        source_timestamp_ns=task_time_ns,
        validated_timestamp_ns=task_time_ns + 100,
        state_source="mujoco_strict_pre_step",
        state_valid=valid,
        capabilities=MPC_CAPABILITIES,
        current_q=np.linspace(-0.02, 0.02, 5),
        current_dq=np.linspace(0.01, -0.01, 5),
        measured_disturbance=_measurement(task_time),
    )


def _event(
    anchor: int,
    *,
    session: str = "session-a",
    epoch: str = "epoch-a",
    planned_override=None,
) -> TaskClockEvent:
    task_time_ns = anchor * 6_000_000
    task_time = task_time_ns * 1e-9
    planned = direct_step_planned_command(
        task_time, NOMINAL_COMMAND, PROTOCOL
    ).planned_command
    if planned_override is not None:
        planned = np.asarray(planned_override, dtype=np.float64)
    runtime = planned.copy()
    runtime[2] = 0.02
    return TaskClockEvent(
        session_nonce=session,
        task_epoch_id=epoch,
        producer_sequence=anchor,
        event_monotonic_timestamp_ns=task_time_ns + 200,
        source_sample_id=anchor * PROTOCOL.mpc_stride,
        task_time_ns=task_time_ns,
        full_task_anchor=anchor,
        planned_command_vx_vy_wz=tuple(planned),
        runtime_command_vx_vy_wz=tuple(runtime),
        heading_reference_rad=0.0,
    )


def _helper_factory(_measurement, horizon):
    return SimpleNamespace(
        horizon=horizon,
        torso_relative_position_reference=np.array([0.1, 0.2, 0.3]),
    )


class FullTaskRightArmControlCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = _FakeArmPolicy()
        self.core = FullTaskRightArmControlCore(
            predictor=_predictor(),
            arm_policy=self.policy,
            nominal_command=NOMINAL_COMMAND,
        )

    def test_task_event_is_one_shared_type(self) -> None:
        self.assertIs(TaskClockEvent, HardwareTaskClockEvent)

    def test_lightweight_contract_import_does_not_load_mujoco(self) -> None:
        script = (
            "import sys; "
            "import right_arm_runtime.control_contracts; "
            "import right_arm_runtime.hardware_output_contract; "
            "assert 'mujoco' not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_explicit_reset_and_epoch_zero_are_required(self) -> None:
        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(_observation(0), _event(0), _helper_factory)
        self.assertEqual(caught.exception.reason_code, "reset_required")

        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(_observation(1), _event(1), _helper_factory)
        self.assertEqual(
            caught.exception.reason_code, "task_epoch_did_not_start_at_zero"
        )

    def test_startup_runs_predictor_and_mpc_but_forbids_output_until_anchor_four(self):
        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        intents = [
            self.core.step(_observation(anchor), _event(anchor), _helper_factory)
            for anchor in range(5)
        ]
        self.assertEqual(self.policy.compute_count, 5)
        self.assertEqual(
            [intent.mpc_output_enabled for intent in intents],
            [False, False, False, False, True],
        )
        self.assertFalse(any(intent.first_mpc_anchor for intent in intents[:4]))
        self.assertTrue(intents[4].first_mpc_anchor)
        self.assertEqual(intents[4].task_time_ns, 24_000_000)
        self.assertEqual(intents[4].full_task_anchor, 4)
        self.assertTrue(intents[4].state_capabilities.mpc_observation_complete)
        self.assertFalse(intents[4].hardware_torque_state_complete)
        self.assertEqual(
            intents[4].predictor_diagnostics["template_anchor_index"], 4
        )
        self.assertAlmostEqual(
            intents[4].predictor_diagnostics["task_time"], 0.024, places=12
        )
        self.assertEqual(
            intents[4].predictor_diagnostics["heading_source"], "causal_prefix"
        )
        np.testing.assert_array_equal(
            intents[4].disturbance_horizon.nodes[0].acc_world,
            _observation(4).measured_disturbance.acc_world,
        )
        np.testing.assert_array_equal(
            intents[4].torso_relative_position_reference,
            np.array([0.1, 0.2, 0.3]),
        )

    def test_stop_boundary_last_anchor_and_horizon_are_not_rebased(self) -> None:
        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        selected = {}
        for anchor in range(PROTOCOL.headline_anchor_count):
            intent = self.core.step(
                _observation(anchor), _event(anchor), _helper_factory
            )
            if anchor in (1066, 1067, 1333):
                selected[anchor] = intent
        self.assertEqual(selected[1066].task_time_ns, 6_396_000_000)
        self.assertEqual(selected[1067].task_time_ns, 6_402_000_000)
        self.assertEqual(
            selected[1066].predictor_diagnostics["template_anchor_index"], 1066
        )
        self.assertEqual(
            selected[1067].predictor_diagnostics["template_anchor_index"], 1067
        )
        self.assertEqual(selected[1333].task_time_ns, 7_998_000_000)
        self.assertEqual(len(selected[1333].disturbance_horizon.nodes), 10)
        self.assertAlmostEqual(7.998 + 9 * 0.006, 8.052, places=12)

    def test_command_schedule_gap_replay_and_off_grid_fail_closed(self) -> None:
        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        self.core.step(_observation(0), _event(0), _helper_factory)
        anchor_gap = TaskClockEvent(
            **{**_event(2).__dict__, "producer_sequence": 1}
        )
        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(_observation(2), anchor_gap, _helper_factory)
        self.assertEqual(caught.exception.reason_code, "task_anchor_gap_or_replay")

        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        bad_event = TaskClockEvent(
            **{**_event(0).__dict__, "task_time_ns": 1_000_000}
        )
        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(_observation(0), bad_event, _helper_factory)
        self.assertEqual(caught.exception.reason_code, "task_time_not_on_anchor")

        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(
                _observation(0),
                _event(0, planned_override=np.array([0.0, 0.0, 0.0127])),
                _helper_factory,
            )
        self.assertEqual(
            caught.exception.reason_code, "planned_command_protocol_mismatch"
        )

    def test_invalid_state_identity_and_session_fail_closed(self) -> None:
        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(_observation(0, valid=False), _event(0), _helper_factory)
        self.assertEqual(caught.exception.reason_code, "state_not_validated")

        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(
                _observation(0, session="session-b"),
                _event(0),
                _helper_factory,
            )
        self.assertEqual(caught.exception.reason_code, "session_mismatch")

    def test_source_binding_capabilities_and_future_state_fail_closed(self) -> None:
        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        bad_binding = TaskClockEvent(
            **{**_event(0).__dict__, "source_sample_id": 99}
        )
        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(_observation(0), bad_binding, _helper_factory)
        self.assertEqual(caught.exception.reason_code, "task_event_source_mismatch")

        self.core.step(_observation(0), _event(0), _helper_factory)
        future_state = TaskClockEvent(
            **{**_event(1).__dict__, "event_monotonic_timestamp_ns": 5_000_000}
        )
        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(_observation(1), future_state, _helper_factory)
        self.assertEqual(caught.exception.reason_code, "task_event_future_state")

        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        not_yet_validated = TaskClockEvent(
            **{**_event(0).__dict__, "event_monotonic_timestamp_ns": 50}
        )
        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(
                _observation(0), not_yet_validated, _helper_factory
            )
        self.assertEqual(caught.exception.reason_code, "task_event_future_state")

        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        incomplete = ControlStateCapabilities(
            right_arm_joint_state=True,
            torso_rotation=True,
            torso_angular_velocity=True,
            torso_linear_acceleration=False,
            torso_angular_acceleration=True,
        )
        observation = _observation(0)
        observation = FullTaskControlObservation(
            **{**observation.__dict__, "capabilities": incomplete}
        )
        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(observation, _event(0), _helper_factory)
        self.assertEqual(caught.exception.reason_code, "mpc_state_incomplete")

    def test_task_event_sequence_gap_is_not_silently_compressed(self) -> None:
        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        self.core.step(_observation(0), _event(0), _helper_factory)
        skipped_sequence = TaskClockEvent(
            **{**_event(1).__dict__, "producer_sequence": 2}
        )
        with self.assertRaises(FullTaskControlCoreError) as caught:
            self.core.step(_observation(1), skipped_sequence, _helper_factory)
        self.assertEqual(
            caught.exception.reason_code, "task_event_sequence_gap_or_replay"
        )

    def test_reset_restarts_task_time_without_inheriting_predictor_epoch(self) -> None:
        self.core.reset(session_nonce="session-a", task_epoch_id="epoch-a")
        self.core.step(_observation(0), _event(0), _helper_factory)
        self.core.step(_observation(1), _event(1), _helper_factory)
        self.core.reset(session_nonce="session-b", task_epoch_id="epoch-b")
        restarted = self.core.step(
            _observation(0, session="session-b"),
            _event(0, session="session-b", epoch="epoch-b"),
            _helper_factory,
        )
        self.assertEqual(restarted.task_time_ns, 0)
        self.assertEqual(restarted.full_task_anchor, 0)
        self.assertEqual(restarted.predictor_diagnostics["template_anchor_index"], 0)


if __name__ == "__main__":
    unittest.main()
