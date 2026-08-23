"""Offline future-output contract and fake-sink tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from right_arm_runtime.hardware_output_contract import (
    FakeHardwareCommandSink,
    FutureCommandMode,
    HardwareControlProposal,
    HardwareOutputContractError,
    SafetyClass,
    ValidatedStateIdentity,
    certify_for_offline_fake_sink,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _state() -> ValidatedStateIdentity:
    return ValidatedStateIdentity(
        session_nonce="offline-session-a",
        sample_id=42,
        source_timestamp_ns=1_000_000,
        validated_timestamp_ns=1_100_000,
        arm_sdk_q=tuple(0.01 * index for index in range(13)),
    )


def _proposal(
    *, mode: FutureCommandMode = FutureCommandMode.ROBOT_PD_PLUS_FEEDFORWARD
) -> HardwareControlProposal:
    state = _state()
    active_mask = tuple(5 <= index < 10 for index in range(13))
    q_ref = list(state.arm_sdk_q)
    for index in range(5, 10):
        q_ref[index] += 0.02
    kp = tuple(20.0 if active else 0.0 for active in active_mask)
    kd = tuple(1.0 if active else 0.0 for active in active_mask)
    if mode is FutureCommandMode.DIRECT_TORQUE:
        kp = (0.0,) * 13
        kd = (0.0,) * 13
    return HardwareControlProposal(
        session_nonce=state.session_nonce,
        proposal_id=7,
        source_sample_id=state.sample_id,
        source_timestamp_ns=state.source_timestamp_ns,
        task_epoch_id="full-task-epoch-a",
        task_time_ns=24_000_000,
        full_task_anchor=4,
        generated_timestamp_ns=1_200_000,
        expires_timestamp_ns=2_000_000,
        mode=mode,
        arm_weight=0.1,
        active_mask=active_mask,
        q_ref=tuple(q_ref),
        dq_ref=(0.0,) * 13,
        ddq_des=tuple(0.1 if active else 0.0 for active in active_mask),
        kp=kp,
        kd=kd,
        tau=tuple(0.2 if active else 0.0 for active in active_mask),
        diagnostics={"predicted_max_abs_qacc": 10.293},
    )


class HardwareOutputContractTest(unittest.TestCase):
    def test_valid_contract_reaches_fake_sink_without_output_capability(self):
        command = certify_for_offline_fake_sink(
            _proposal(), _state(), now_ns=1_300_000
        )
        self.assertFalse(command.hardware_safety_certified)
        self.assertFalse(command.hardware_output_authorized)
        self.assertIn(
            (SafetyClass.DIAGNOSTIC, "PREDICTED_QACC_RECORDED"),
            command.safety_events,
        )
        receipt = FakeHardwareCommandSink(
            session_nonce="offline-session-a", watchdog_timeout_ns=500_000
        ).submit(command, now_ns=1_400_000)
        self.assertTrue(receipt.accepted)
        self.assertFalse(receipt.dds_write_performed)
        self.assertFalse(receipt.hardware_output_performed)
        self.assertEqual(receipt.sink, "offline_in_memory_fake_sink")

    def test_expiry_and_state_binding_fail_closed(self):
        with self.assertRaisesRegex(HardwareOutputContractError, "expired"):
            certify_for_offline_fake_sink(
                _proposal(), _state(), now_ns=2_000_001
            )
        with self.assertRaisesRegex(HardwareOutputContractError, "session"):
            certify_for_offline_fake_sink(
                replace(_proposal(), session_nonce="wrong"),
                _state(),
                now_ns=1_300_000,
            )
        with self.assertRaisesRegex(HardwareOutputContractError, "sample"):
            certify_for_offline_fake_sink(
                replace(_proposal(), source_sample_id=41),
                _state(),
                now_ns=1_300_000,
            )

    def test_task_time_binding_requires_exact_absolute_anchor(self):
        with self.assertRaisesRegex(HardwareOutputContractError, "6 ms anchor"):
            certify_for_offline_fake_sink(
                replace(_proposal(), task_time_ns=0),
                _state(),
                now_ns=1_300_000,
            )
        command = certify_for_offline_fake_sink(
            _proposal(), _state(), now_ns=1_300_000
        )
        self.assertEqual(command.task_time_ns, 24_000_000)
        self.assertEqual(command.full_task_anchor, 4)
        self.assertEqual(command.task_epoch_id, "full-task-epoch-a")

    def test_direct_torque_rejects_duplicate_robot_side_pd(self):
        bad = replace(
            _proposal(mode=FutureCommandMode.DIRECT_TORQUE),
            kp=tuple(10.0 if 5 <= i < 10 else 0.0 for i in range(13)),
        )
        with self.assertRaisesRegex(HardwareOutputContractError, "kp/kd"):
            certify_for_offline_fake_sink(bad, _state(), now_ns=1_300_000)

    def test_inactive_slots_are_zero_action_holds(self):
        tau = list(_proposal().tau)
        tau[0] = 0.1
        with self.assertRaisesRegex(HardwareOutputContractError, "inactive slot"):
            certify_for_offline_fake_sink(
                replace(_proposal(), tau=tuple(tau)),
                _state(),
                now_ns=1_300_000,
            )

    def test_nonfinite_and_fake_sink_replay_fail_closed(self):
        tau = list(_proposal().tau)
        tau[5] = float("nan")
        with self.assertRaisesRegex(HardwareOutputContractError, "NaN/Inf"):
            certify_for_offline_fake_sink(
                replace(_proposal(), tau=tuple(tau)),
                _state(),
                now_ns=1_300_000,
            )
        command = certify_for_offline_fake_sink(
            _proposal(), _state(), now_ns=1_300_000
        )
        sink = FakeHardwareCommandSink(
            session_nonce="offline-session-a", watchdog_timeout_ns=500_000
        )
        self.assertTrue(sink.submit(command, now_ns=1_400_000).accepted)
        replay = sink.submit(command, now_ns=1_500_000)
        self.assertFalse(replay.accepted)
        self.assertEqual(replay.reason, "COMMAND_REPLAY_OR_REGRESSION")

    def test_fake_sink_rejects_restart_nonce_expiry_and_forged_authority(self):
        command = certify_for_offline_fake_sink(
            _proposal(), _state(), now_ns=1_300_000
        )
        restarted = FakeHardwareCommandSink(
            session_nonce="offline-session-b", watchdog_timeout_ns=500_000
        )
        self.assertEqual(
            restarted.submit(command, now_ns=1_400_000).reason,
            "SESSION_MISMATCH",
        )
        same_session = FakeHardwareCommandSink(
            session_nonce="offline-session-a", watchdog_timeout_ns=500_000
        )
        self.assertEqual(
            same_session.submit(command, now_ns=2_000_001).reason,
            "COMMAND_EXPIRED",
        )
        self.assertEqual(
            same_session.submit(
                replace(command, hardware_output_authorized=True),
                now_ns=1_400_000,
            ).reason,
            "REAL_OUTPUT_AUTHORIZATION_FORBIDDEN",
        )

    def test_watchdog_is_bounded_and_never_writes(self):
        command = certify_for_offline_fake_sink(
            _proposal(), _state(), now_ns=1_300_000
        )
        sink = FakeHardwareCommandSink(
            session_nonce="offline-session-a", watchdog_timeout_ns=500_000
        )
        sink.submit(command, now_ns=1_400_000)
        healthy = sink.watchdog_receipt(now_ns=1_900_000)
        expired = sink.watchdog_receipt(now_ns=1_900_001)
        self.assertTrue(healthy.accepted)
        self.assertFalse(expired.accepted)
        self.assertEqual(expired.reason, "WATCHDOG_EXPIRED")
        self.assertFalse(expired.dds_write_performed)

    def test_live_launchers_do_not_import_offline_output_contract(self):
        for relative in (
            "run_hardware_shadow.py",
            "tools/realtime/run_hardware_shadow.sh",
            "tools/realtime/run_hardware_state_inspection.sh",
        ):
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("hardware_output_contract", source)


if __name__ == "__main__":
    unittest.main()
