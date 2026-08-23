"""Integration-level contracts for L1 experimental latency wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from disturbance_template.full_task_startup_pd import (
    FixedStartupPdHandoff,
    StartupPdTraceRecorder,
    mapping_safety_snapshot,
)


def _safe_mapping() -> SimpleNamespace:
    vector = np.zeros(5, dtype=np.float64)
    return SimpleNamespace(
        qacc_baseline=vector,
        first_pass_qacc_validated=vector,
        second_pass_qacc_validated=vector,
        qacc_validated=vector,
        second_pass_triggered=False,
        safety_fallback_used=False,
        safety_fallback_attempts=0,
        hold_last_safe_available=True,
        hold_last_safe_used=False,
        hold_last_safe_satisfied=True,
        safe_hold_used=False,
        safety_line_search_used=False,
        safety_line_search_attempts=0,
        safety_line_search_time=0.0,
        final_output_certified=True,
        no_safe_torque=False,
    )


class ExperimentalLatencyIntegrationContractTest(unittest.TestCase):
    def _delayed_trace(self, actual_handoff_sample: int) -> dict:
        handoff = FixedStartupPdHandoff(0.024)
        trace = StartupPdTraceRecorder(
            handoff=handoff,
            runtime_mode="process",
            allow_delayed_actuation=True,
        )
        previous = None
        for index in range(actual_handoff_sample + 5):
            task_time = index * 0.002
            anchor = index % 3 == 0
            mpc_enabled = index >= actual_handoff_sample
            fixed_tau = np.full(5, 0.2 + index * 0.01)
            actual_tau = (
                np.full(5, 1.0 + index * 0.01)
                if mpc_enabled
                else fixed_tau.copy()
            )
            active_source_time = (
                0.024 if mpc_enabled else task_time if anchor else np.nan
            )
            active_anchor = 4 if mpc_enabled else index // 3 if anchor else -1
            trace.append(
                sample_index=index,
                simulation_time=task_time,
                task_time=task_time,
                mpc_anchor=anchor,
                mpc_control_enabled=mpc_enabled,
                policy_update_applied=index == 10,
                predictor_updated=anchor or mpc_enabled,
                predictor_task_time=active_source_time,
                predictor_template_anchor_index=active_anchor,
                predictor_fallback_used=False,
                gait_phase_cycles=task_time / 0.8,
                left_foot_contact_count=1,
                right_foot_contact_count=1,
                raw_torso_acceleration_norm_m_s2=1.0,
                base_vertical_velocity_m_s=0.0,
                planned_command=np.array([0.5, 0.0, 0.0]),
                runtime_command=np.array([0.5, 0.0, 0.0]),
                fixed_posture_pd_tau=fixed_tau,
                actual_right_arm_tau=actual_tau,
                desired_right_arm_ddq=np.zeros(5),
                previous_executed_tau_available=previous is not None,
                previous_executed_tau=(
                    np.full(5, np.nan) if previous is None else previous.copy()
                ),
                **(
                    mapping_safety_snapshot(_safe_mapping())
                    if mpc_enabled
                    else mapping_safety_snapshot(None)
                ),
            )
            previous = actual_tau.copy()
        return trace.summary(dry_warmup={"performed": True})

    def test_2_and_4ms_delays_keep_logical_anchor_four_and_pd_until_ready(self):
        for sample, expected_time in ((13, 0.026), (14, 0.028)):
            summary = self._delayed_trace(sample)
            self.assertEqual(
                summary["logical_source_handoff"]["template_anchor_index"], 4
            )
            self.assertAlmostEqual(
                summary["handoff"]["simulation_time_s"], expected_time
            )
            self.assertEqual(summary["handoff"]["template_anchor_index"], 4)
            self.assertTrue(summary["handoff"]["previous_tau_available"])
            np.testing.assert_array_equal(
                summary["handoff"]["previous_executed_tau_input_nm"],
                summary["handoff"]["last_fixed_pd_tau_nm"],
            )

    def test_main_keeps_fail_closed_guard_and_fresh_state_process(self):
        source = (Path(__file__).parents[1] / "main_sim.py").read_text()
        guard = source[source.index("if acceleration_controller and mpc_control_enabled"):]
        self.assertIn("not bool(mapping_result.final_output_certified)", guard)
        self.assertIn("bool(mapping_result.no_safe_torque)", guard)
        self.assertIn("NO_SAFE_TORQUE", guard)
        self.assertIn("qpos=d.qpos", source)
        self.assertIn("qvel=d.qvel", source)
        self.assertIn("previous_executed_tau=previous_executed_tau", source)
        self.assertIn("experimental_mpc_delay_line.active_packet is not None", source)
        self.assertIn('"mapper_candidate_trace":', source)
        self.assertIn("right_arm_q_rad=right_arm_q.copy()", source)
        self.assertIn("right_arm_dq_rad_s=right_arm_dq.copy()", source)
        self.assertIn("previous_executed_tau_nm=", source)
        failure_capture = source.index("except SimRuntimeError as process_error")
        trace_save = source.index("save_fail_closed_process_trace", failure_capture)
        right_arm_write = source.index("d.ctrl[12:23] = tau_arm_waist", trace_save)
        physics_step = source.index("mujoco.mj_step(m, d)", right_arm_write)
        self.assertLess(failure_capture, trace_save)
        self.assertLess(trace_save, right_arm_write)
        self.assertLess(right_arm_write, physics_step)
        helper_source = (
            Path(__file__).parents[1]
            / "right_arm_runtime"
            / "sim_mpc_latency.py"
        ).read_text()
        self.assertIn('"right_arm_d_ctrl_written": False', helper_source)
        self.assertIn('"mj_step_performed": False', helper_source)


if __name__ == "__main__":
    unittest.main()
