"""T1-A regressions for full-task time, pre-step schema, and causal H."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from disturbance_learning.full_task_protocol import (
    DEFAULT_FULL_TASK_PROTOCOL,
    FullTaskCausalHeadingFrame,
    FullTaskContinuousHeadingFrame,
    FullTaskClock,
    direct_step_planned_command,
    is_valid_rotation_batch,
    rotation_z,
)
from disturbance_learning.full_task_recording import (
    FullTaskRawRecorder,
    compute_smoke_summary,
    save_full_task_smoke_artifacts,
    validate_full_task_raw,
)


REPO_DIR = Path(__file__).resolve().parents[1]
PROTOCOL = DEFAULT_FULL_TASK_PROTOCOL
NOMINAL = np.array([0.5, 0.0, 0.0127], dtype=np.float64)


def _heading_state(initialized: bool) -> SimpleNamespace:
    value = 0.0 if initialized else np.nan
    return SimpleNamespace(
        reference_world=value,
        yaw_filtered=value,
        yaw_error=value,
        yaw_rate_correction=0.0,
        command_saturated=False,
    )


def _synthetic_complete_recorder() -> FullTaskRawRecorder:
    clock = FullTaskClock(PROTOCOL)
    clock.reset(0.0, epoch_label="synthetic_nominal")
    recorder = FullTaskRawRecorder(
        protocol=PROTOCOL,
        clock=clock,
        nominal_command=NOMINAL,
    )
    last_policy_time = np.nan
    for sample_index in range(PROTOCOL.physics_steps):
        task_time = sample_index * PROTOCOL.physics_dt
        if sample_index > 0 and sample_index % PROTOCOL.policy_stride == 0:
            last_policy_time = task_time
        policy_update = bool(
            np.isfinite(last_policy_time)
            and np.isclose(last_policy_time, task_time, atol=1e-12)
        )
        planned = direct_step_planned_command(task_time, NOMINAL, PROTOCOL).planned_command
        runtime = planned.copy()
        x = min(0.4 * task_time, 3.2)
        recorder.append(
            simulation_time=task_time,
            sample_index=sample_index,
            planned_command=planned,
            runtime_command=runtime,
            policy_update_applied=policy_update,
            policy_command_consumed_time=last_policy_time,
            mpc_anchor=PROTOCOL.is_mpc_anchor_sample(sample_index),
            torso_position_world=np.array([x, 0.0, 0.8]),
            torso_rotation_world=np.eye(3),
            torso_linear_velocity_world=np.array([0.4, 0.0, 0.0]),
            torso_angular_velocity_world=np.zeros(3),
            torso_linear_acceleration_world_raw=np.zeros(3),
            torso_linear_acceleration_world_used=np.zeros(3),
            torso_angular_acceleration_world_raw=np.zeros(3),
            torso_angular_acceleration_world_used=np.zeros(3),
            lower_body_q=np.zeros(12),
            lower_body_dq=np.zeros(12),
            lower_body_policy_target=np.zeros(12),
            right_arm_q=np.zeros(5),
            right_arm_dq=np.zeros(5),
            right_arm_ddq_des=np.zeros(5),
            generalized_qpos=np.zeros(30),
            generalized_qvel=np.zeros(29),
            generalized_qacc=np.zeros(29),
            actuator_ctrl=np.zeros(23),
            heading_state=_heading_state(task_time >= PROTOCOL.policy_dt - 1e-12),
            mpc_diagnostics={
                "success": True,
                "fallback_used": False,
                "solver_status_val": 1,
            },
            runtime_mapping_safety_fallback_used=False,
            runtime_executor_flags=0,
        )
    return recorder


class FullTaskProtocolTest(unittest.TestCase):
    def _synthetic_smoke_inputs(self):
        raw = _synthetic_complete_recorder().to_arrays()
        validation = validate_full_task_raw(raw, PROTOCOL, require_complete=True)
        return raw, validation

    def test_direct_step_and_anchor_boundaries_are_exact(self) -> None:
        np.testing.assert_array_equal(
            direct_step_planned_command(0.0, NOMINAL).planned_command,
            NOMINAL,
        )
        np.testing.assert_array_equal(
            direct_step_planned_command(6.398, NOMINAL).planned_command,
            NOMINAL,
        )
        stopped = direct_step_planned_command(6.4, NOMINAL).planned_command
        np.testing.assert_array_equal(stopped, np.array([0.0, 0.0, NOMINAL[2]]))
        self.assertEqual(PROTOCOL.anchor_index(6.396), 1066)
        self.assertEqual(PROTOCOL.anchor_index(6.402), 1067)
        with self.assertRaises(ValueError):
            PROTOCOL.anchor_index(6.4)
        with self.assertRaises(ValueError):
            direct_step_planned_command(PROTOCOL.record_end + 0.002, NOMINAL)

    def test_headline_anchor_count_and_tail_indices(self) -> None:
        self.assertEqual(PROTOCOL.headline_anchor_count, 1334)
        self.assertAlmostEqual(PROTOCOL.last_headline_anchor_time, 7.998)
        self.assertAlmostEqual(PROTOCOL.last_horizon_node_time, 8.052)
        self.assertEqual(PROTOCOL.physics_steps, 4030)
        self.assertEqual(PROTOCOL.recorded_anchor_count, 1344)
        window = PROTOCOL.future_window_sample_indices(1333)
        np.testing.assert_array_equal(
            window["node"], 3999 + 3 * np.arange(10)
        )
        self.assertEqual(int(window["node"][-1]), 4026)

    def test_task_clock_reset_backward_and_out_of_range(self) -> None:
        clock = FullTaskClock(PROTOCOL)
        with self.assertRaises(RuntimeError):
            clock.observe(0.0)
        clock.reset(10.0, epoch_label="first")
        self.assertEqual(clock.observe(10.0), 0.0)
        self.assertAlmostEqual(clock.observe(10.006), 0.006)
        with self.assertRaises(ValueError):
            clock.observe(10.004)
        clock.reset(20.0, epoch_label="second")
        self.assertEqual(clock.epoch_index, 1)
        self.assertEqual(clock.epoch_label, "second")
        self.assertEqual(clock.observe(20.0), 0.0)
        with self.assertRaises(ValueError):
            clock.observe(20.0 + PROTOCOL.record_end + PROTOCOL.physics_dt)

    def test_first_cycle_causal_heading_and_wraparound(self) -> None:
        helper = FullTaskCausalHeadingFrame(PROTOCOL)
        first = helper.update(0.0, rotation_z(np.pi - 0.02))
        second = helper.update(0.006, rotation_z(-np.pi + 0.02))
        self.assertEqual(first.source, "first_cycle_causal_prefix")
        self.assertEqual(first.source_sample_count, 1)
        self.assertGreater(abs(second.yaw_world), np.pi - 0.03)
        self.assertTrue(bool(is_valid_rotation_batch(second.rotation_world_heading)))
        self.assertTrue(
            bool(is_valid_rotation_batch(helper.rotation_heading_body(rotation_z(0.3))))
        )

    def test_cycle_switch_uses_only_previous_complete_cycle(self) -> None:
        helper = FullTaskCausalHeadingFrame(PROTOCOL)
        state = None
        for anchor in range(PROTOCOL.anchor_index(0.804) + 1):
            task_time = anchor * PROTOCOL.mpc_dt
            yaw = 0.1 if task_time < PROTOCOL.gait_period else 1.2
            state = helper.update(task_time, rotation_z(yaw))
        self.assertIsNotNone(state)
        self.assertEqual(state.source, "previous_complete_cycle")
        self.assertEqual(state.source_cycle_index, 0)
        self.assertEqual(state.source_sample_count, 134)
        self.assertAlmostEqual(state.yaw_world, 0.1, places=12)
        self.assertNotAlmostEqual(state.yaw_world, 1.2)

    def test_causal_heading_reset_backward_and_invalid_rotation(self) -> None:
        helper = FullTaskCausalHeadingFrame(PROTOCOL)
        helper.update(0.0, np.eye(3))
        helper.update(0.006, np.eye(3))
        with self.assertRaises(ValueError):
            helper.update(0.0, np.eye(3))
        helper.reset()
        reset_state = helper.update(0.0, rotation_z(0.4))
        self.assertAlmostEqual(reset_state.yaw_world, 0.4)
        invalid = np.eye(3)
        invalid[0, 0] = 2.0
        helper.reset()
        with self.assertRaises(ValueError):
            helper.update(0.0, invalid)

    def test_continuous_heading_wrap_prefix_and_rolling_window(self) -> None:
        helper = FullTaskContinuousHeadingFrame(PROTOCOL)
        first = helper.update(0.0, rotation_z(np.pi - 0.02))
        second = helper.update(0.006, rotation_z(-np.pi + 0.02))
        self.assertEqual(first.source, "causal_prefix")
        self.assertEqual(first.source_sample_count, 1)
        self.assertGreater(abs(second.yaw_world), np.pi - 0.03)

        helper.reset()
        states = []
        measured = []
        last_anchor = PROTOCOL.anchor_index(0.804)
        for anchor in range(last_anchor + 1):
            time = anchor * PROTOCOL.mpc_dt
            yaw = 0.001 * anchor
            measured.append(yaw)
            states.append(helper.update(time, rotation_z(yaw)))
        self.assertEqual(states[133].source, "causal_prefix")  # t=0.798 s
        self.assertEqual(states[134].source, "rolling_0p8s")  # t=0.804 s
        expected = np.arctan2(
            np.sum(np.sin(measured[1:135])), np.sum(np.cos(measured[1:135]))
        )
        self.assertAlmostEqual(states[134].yaw_world, expected, places=12)
        self.assertEqual(states[134].source_sample_count, 134)
        boundary_jump = abs(
            np.arctan2(
                np.sin(states[134].yaw_world - states[133].yaw_world),
                np.cos(states[134].yaw_world - states[133].yaw_world),
            )
        )
        self.assertLess(boundary_jump, 0.01)

    def test_continuous_heading_stop_freeze_no_future_and_reset(self) -> None:
        def replay(future_yaw: float) -> list[float]:
            helper = FullTaskContinuousHeadingFrame(PROTOCOL)
            output = []
            stop_after = PROTOCOL.anchor_index(6.414)
            for anchor in range(stop_after + 1):
                time = anchor * PROTOCOL.mpc_dt
                yaw = 0.1 + 0.0001 * anchor if time <= 0.3 else future_yaw
                output.append(helper.update(time, rotation_z(yaw)).yaw_world)
            return output

        baseline = replay(0.2)
        changed_future = replay(-1.0)
        prefix_last = PROTOCOL.anchor_index(0.3)
        np.testing.assert_allclose(
            baseline[: prefix_last + 1], changed_future[: prefix_last + 1], atol=0.0, rtol=0.0
        )

        helper = FullTaskContinuousHeadingFrame(PROTOCOL)
        states = []
        for anchor in range(PROTOCOL.anchor_index(6.414) + 1):
            time = anchor * PROTOCOL.mpc_dt
            states.append(helper.update(time, rotation_z(0.0002 * anchor)))
        pre_stop = states[PROTOCOL.anchor_index(6.396)]
        post_stop = states[PROTOCOL.anchor_index(6.402)]
        later = states[PROTOCOL.anchor_index(6.414)]
        self.assertEqual(post_stop.source, "frozen_pre_stop")
        self.assertEqual(post_stop.yaw_world, pre_stop.yaw_world)
        self.assertEqual(later.yaw_world, pre_stop.yaw_world)
        with self.assertRaises(ValueError):
            helper.update(6.414, np.eye(3))
        helper.reset()
        reset = helper.update(0.0, rotation_z(-0.4))
        self.assertAlmostEqual(reset.yaw_world, -0.4)

    def test_complete_raw_and_manifest_round_trip(self) -> None:
        recorder = _synthetic_complete_recorder()
        raw = recorder.to_arrays()
        validation = validate_full_task_raw(raw, PROTOCOL, require_complete=True)
        self.assertEqual(validation["raw_sample_count"], 4030)
        self.assertEqual(validation["headline_anchor_count"], 1334)
        self.assertAlmostEqual(validation["last_raw_time"], 8.058)
        self.assertTrue(validation["tail_complete"])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = save_full_task_smoke_artifacts(
                recorder=recorder,
                run_dir=output,
                repo_dir=REPO_DIR,
                config_path=REPO_DIR / "configs/g1.yaml",
                policy_path=REPO_DIR / "policy/motion.pt",
                xml_path=REPO_DIR / "resources/g1_description/scene.xml",
                legacy_template_path=(
                    REPO_DIR
                    / "disturbance_model_new_heading/templates_heading_interval/heading_disturbance_template.npz"
                ),
                predictor_metadata={"predictor_type": "template"},
                control_chain={
                    "arm_controller": "mpc",
                    "right_arm_execution_runtime": "process",
                },
                initial_lower_q_offset=np.zeros(12),
                initial_lower_dq=np.zeros(12),
                heading_enabled=True,
            )
            self.assertEqual(result["summary"]["status"], "PASS")
            with np.load(result["raw_path"], allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["task_time"], raw["task_time"])
                self.assertEqual(str(loaded["raw_schema_version"]), "full_task_raw_v1")
            with result["manifest_path"].open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["schema_version"], "full_task_raw_v1")
            self.assertEqual(manifest["protocol"]["version"], "full_task_direct_step_v1")
            self.assertFalse(manifest["scope"]["final_full_task_template_generated"])
            self.assertEqual(len(result["plot_paths"]), 5)

    def test_smoke_summary_passes_without_mapping_fallback(self) -> None:
        raw, validation = self._synthetic_smoke_inputs()
        summary = compute_smoke_summary(
            raw, PROTOCOL, validation, heading_enabled=True
        )
        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(summary["smoke_passed"])
        self.assertTrue(summary["nominal_mapping_path_passed"])
        self.assertEqual(summary["warnings"], [])
        self.assertEqual(summary["runtime_mapping_safety_fallback_count"], 0)

    def test_certified_mapping_fallback_is_a_warning_not_smoke_failure(self) -> None:
        raw, validation = self._synthetic_smoke_inputs()
        raw["runtime_mapping_safety_fallback_used"][[12, 24]] = True
        summary = compute_smoke_summary(
            raw, PROTOCOL, validation, heading_enabled=True
        )
        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(summary["smoke_passed"])
        self.assertFalse(summary["nominal_mapping_path_passed"])
        self.assertEqual(summary["warnings"], ["MAPPING_SAFETY_FALLBACK_USED"])
        self.assertEqual(summary["runtime_mapping_safety_fallback_count"], 2)

    def test_real_task_condition_failure_still_fails_smoke(self) -> None:
        raw, validation = self._synthetic_smoke_inputs()
        validation["strict_pre_step"] = False
        summary = compute_smoke_summary(
            raw, PROTOCOL, validation, heading_enabled=True
        )
        self.assertEqual(summary["status"], "FAIL")
        self.assertFalse(summary["smoke_passed"])



if __name__ == "__main__":
    unittest.main()
