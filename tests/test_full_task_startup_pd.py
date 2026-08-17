"""Tests for exact fixed-PD startup and full-task MPC handoff semantics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from disturbance_template.full_task_protocol import DEFAULT_FULL_TASK_PROTOCOL
from disturbance_template.full_task_runtime_preflight import (
    FormalFullTaskEnvironmentError,
    THREAD_ENVIRONMENT_NAMES,
    validate_formal_full_task_runtime,
)
from disturbance_template.full_task_startup_pd import (
    FixedStartupPdHandoff,
    RIGHT_ARM_MODE_FIXED_POSTURE_PD,
    StartupPdTraceRecorder,
    mapping_safety_snapshot,
)


def _mapping_result() -> SimpleNamespace:
    vector = np.arange(5, dtype=np.float64)
    return SimpleNamespace(
        qacc_baseline=vector,
        first_pass_qacc_validated=vector + 1.0,
        second_pass_qacc_validated=vector + 2.0,
        qacc_validated=vector + 3.0,
        second_pass_triggered=True,
        safety_fallback_used=False,
        safety_fallback_attempts=0,
        hold_last_safe_available=False,
        hold_last_safe_used=False,
        hold_last_safe_satisfied=False,
        safe_hold_used=False,
        safety_line_search_used=False,
        safety_line_search_attempts=0,
        safety_line_search_time=0.0,
        final_output_certified=True,
        no_safe_torque=False,
    )


class FixedStartupPdHandoffTest(unittest.TestCase):
    def test_formal_24ms_takes_over_only_on_anchor_four(self) -> None:
        mode24 = FixedStartupPdHandoff(0.024)
        self.assertEqual(mode24.takeover_anchor_index, 4)
        for time_ms in (0, 6, 12, 18):
            decision = mode24.decision(time_ms // 2)
            self.assertFalse(decision.mpc_control_enabled)
        self.assertTrue(mode24.decision(12).first_mpc_anchor)
        self.assertTrue(mode24.decision(12).mpc_control_enabled)

        with self.assertRaises(ValueError):
            FixedStartupPdHandoff(0.054)

    def test_first_lower_policy_update_remains_20ms(self) -> None:
        protocol = DEFAULT_FULL_TASK_PROTOCOL
        self.assertEqual(protocol.policy_stride, 10)
        self.assertAlmostEqual(protocol.policy_stride * protocol.physics_dt, 0.020)
        self.assertFalse(10 % protocol.policy_stride)

    def test_short_smoke_must_cover_handoff_plus_200ms(self) -> None:
        handoff = FixedStartupPdHandoff(0.024)
        self.assertEqual(handoff.validate_short_smoke_end(0.300), 0.300)
        with self.assertRaises(ValueError):
            handoff.validate_short_smoke_end(0.222)
        with self.assertRaises(ValueError):
            handoff.validate_short_smoke_end(0.225)

    def test_contact_diagnostics_cannot_gate_formal_handoff(self) -> None:
        decision_fields = FixedStartupPdHandoff(0.024).decision(12).__dataclass_fields__
        self.assertFalse(any("contact" in name for name in decision_fields))
        main_source = (Path(__file__).parents[1] / "main_sim.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("full_task_arming", main_source)
        self.assertNotIn("FullTaskArmingGate", main_source)

    def _build_trace(self, runtime_mode: str) -> StartupPdTraceRecorder:
        handoff = FixedStartupPdHandoff(0.024)
        trace = StartupPdTraceRecorder(handoff, runtime_mode)
        last_tau = None
        for index in range(30):
            decision = handoff.decision(index)
            is_anchor = decision.mpc_anchor
            fixed_tau = np.full(5, 0.5 + 0.01 * index)
            actual_tau = (
                fixed_tau.copy()
                if decision.right_arm_mode == RIGHT_ARM_MODE_FIXED_POSTURE_PD
                else np.full(5, 1.0 + 0.01 * index)
            )
            mapping = (
                mapping_safety_snapshot(_mapping_result())
                if decision.mpc_control_enabled
                else mapping_safety_snapshot(None)
            )
            task_time = decision.task_time
            trace.append(
                sample_index=index,
                simulation_time=task_time,
                task_time=task_time,
                mpc_anchor=is_anchor,
                mpc_control_enabled=decision.mpc_control_enabled,
                policy_update_applied=index == 10,
                predictor_updated=is_anchor,
                predictor_task_time=task_time if is_anchor else np.nan,
                predictor_template_anchor_index=index // 3 if is_anchor else -1,
                predictor_fallback_used=False,
                gait_phase_cycles=(task_time % 0.8) / 0.8,
                left_foot_contact_count=1,
                right_foot_contact_count=1,
                raw_torso_acceleration_norm_m_s2=2.0,
                base_vertical_velocity_m_s=0.0,
                planned_command=np.array([0.5, 0.0, 0.0]),
                runtime_command=np.array([0.5, 0.0, 0.01]),
                fixed_posture_pd_tau=fixed_tau,
                actual_right_arm_tau=actual_tau,
                desired_right_arm_ddq=np.zeros(5),
                previous_executed_tau_available=last_tau is not None,
                previous_executed_tau=(
                    np.full(5, np.nan) if last_tau is None else last_tau.copy()
                ),
                **mapping,
            )
            last_tau = actual_tau.copy()
        return trace

    def test_trace_proves_fixed_pd_template_progress_and_previous_tau(self) -> None:
        trace = self._build_trace("process")
        summary = trace.summary(dry_warmup={"performed": True})
        self.assertEqual(summary["first_lower_policy_update_task_time_s"], 0.020)
        self.assertEqual(summary["handoff"]["template_anchor_index"], 4)
        self.assertEqual(summary["handoff"]["template_absolute_task_time_s"], 0.024)
        self.assertTrue(summary["handoff"]["previous_tau_available"])
        np.testing.assert_allclose(
            summary["handoff"]["previous_executed_tau_input_nm"],
            summary["handoff"]["last_fixed_pd_tau_nm"],
            atol=0.0,
            rtol=0.0,
        )
        self.assertTrue(summary["prefix"]["included_in_headline"])
        self.assertEqual(summary["prefix"]["mpc_output_count"], 0)
        self.assertEqual(summary["prefix"]["predictor_anchor_count_before_handoff"], 4)

    def test_wrong_startup_torque_is_rejected(self) -> None:
        handoff = FixedStartupPdHandoff(0.024)
        trace = StartupPdTraceRecorder(handoff, "process")
        with self.assertRaises(ValueError):
            trace.append(
                sample_index=0,
                simulation_time=0.0,
                task_time=0.0,
                mpc_anchor=True,
                mpc_control_enabled=False,
                actual_right_arm_tau=np.ones(5),
                fixed_posture_pd_tau=np.zeros(5),
            )

    def test_sync_and_process_share_timing_and_normalized_diagnostics(self) -> None:
        sync = self._build_trace("sync").summary(dry_warmup=None)
        process = self._build_trace("process").summary(dry_warmup=None)
        encode = lambda value: json.dumps(
            value,
            sort_keys=True,
            default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item,
        )
        self.assertEqual(encode(sync["handoff"]), encode(process["handoff"]))
        self.assertEqual(sync["prefix"], process["prefix"])
        self.assertEqual(sync["whole_trace"], process["whole_trace"])
        self.assertNotEqual(sync["runtime_mode"], process["runtime_mode"])


class FormalRuntimePreflightTest(unittest.TestCase):
    @staticmethod
    def _environment() -> dict[str, str]:
        environment = {
            "DISTURBANCE_LAB_FORMAL_LAUNCHER": "disturbance_lab_run_sh",
            "MPC_CONTROL_CPU": "7",
        }
        environment.update({name: "1" for name in THREAD_ENVIRONMENT_NAMES})
        return environment

    def test_parent_and_worker_cpu7_single_thread_pass(self) -> None:
        evidence = validate_formal_full_task_runtime(
            parent_affinity=[7],
            worker_affinity=[7],
            environment=self._environment(),
            torch_num_threads=1,
            torch_num_interop_threads=1,
            gc_disabled_during_control=True,
        )
        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.parent_cpu_affinity, [7])
        self.assertEqual(evidence.worker_cpu_affinity, [7])
        self.assertFalse(evidence.dynamic_arming_enabled)
        self.assertEqual(evidence.mpc_handoff_anchor_index, 4)

    def test_bad_parent_affinity_fails_before_formal_runtime(self) -> None:
        with self.assertRaises(FormalFullTaskEnvironmentError):
            validate_formal_full_task_runtime(
                parent_affinity=[0, 7],
                environment=self._environment(),
                torch_num_threads=1,
                torch_num_interop_threads=1,
            )

    def test_empty_thread_environment_fails(self) -> None:
        environment = self._environment()
        environment["OPENBLAS_NUM_THREADS"] = ""
        with self.assertRaises(FormalFullTaskEnvironmentError):
            validate_formal_full_task_runtime(
                parent_affinity=[7],
                environment=environment,
                torch_num_threads=1,
                torch_num_interop_threads=1,
            )

    def test_parent_worker_mismatch_and_enabled_gc_fail(self) -> None:
        with self.assertRaises(FormalFullTaskEnvironmentError):
            validate_formal_full_task_runtime(
                parent_affinity=[7],
                worker_affinity=[6],
                environment=self._environment(),
                torch_num_threads=1,
                torch_num_interop_threads=1,
                gc_disabled_during_control=False,
            )


if __name__ == "__main__":
    unittest.main()
