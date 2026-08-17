"""Fast T1 tests for fixed-PD episode/template semantics without simulation."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from disturbance_template.full_task_fixed_pd_collector import collection_specs
from disturbance_template.full_task_protocol import (
    DEFAULT_FULL_TASK_PROTOCOL,
    FullTaskClock,
    FullTaskContinuousHeadingFrame,
    direct_step_planned_command,
    rotation_z,
)
from disturbance_template.full_task_recording import FullTaskRawRecorder
from disturbance_template.full_task_template_builder import (
    build_full_task_template,
    causal_h_metrics,
    evaluate_heldout_template,
    validate_full_task_template,
    TEMPLATE_SCHEMA_VERSION_V2,
    _episode_heading_windows,
)


PROTOCOL = DEFAULT_FULL_TASK_PROTOCOL
NOMINAL = np.array([0.5, 0.0, 0.0127])


def _heading_state(initialized: bool) -> SimpleNamespace:
    value = 0.0 if initialized else np.nan
    return SimpleNamespace(
        reference_world=value,
        yaw_filtered=value,
        yaw_error=value,
        yaw_rate_correction=0.0,
        command_saturated=False,
    )


def _raw(offset: float, *, continuous: bool = False, yaw_rate: float = 0.0) -> dict[str, np.ndarray]:
    clock = FullTaskClock(PROTOCOL)
    clock.reset(0.0, epoch_label=f"synthetic_{offset}")
    recorder = FullTaskRawRecorder(
        protocol=PROTOCOL,
        clock=clock,
        nominal_command=NOMINAL,
        heading_frame_version=(
            FullTaskContinuousHeadingFrame.DEFINITION_VERSION
            if continuous else "full_task_cycle_held_heading_v1"
        ),
    )
    last_policy_time = np.nan
    concentration = []
    for sample_index in range(PROTOCOL.physics_steps):
        time = sample_index * PROTOCOL.physics_dt
        if sample_index > 0 and sample_index % PROTOCOL.policy_stride == 0:
            last_policy_time = time
        planned = direct_step_planned_command(time, NOMINAL, PROTOCOL).planned_command
        acceleration = np.array([offset + time, 2.0 * time, -time])
        omega = np.array([time, -offset, 0.1])
        alpha = np.array([0.0, time + offset, -time])
        recorder.append(
            simulation_time=time,
            sample_index=sample_index,
            planned_command=planned,
            runtime_command=planned,
            policy_update_applied=bool(np.isfinite(last_policy_time) and np.isclose(last_policy_time, time)),
            policy_command_consumed_time=last_policy_time,
            mpc_anchor=PROTOCOL.is_mpc_anchor_sample(sample_index),
            torso_position_world=np.array([0.4 * time, offset, 0.8]),
            torso_rotation_world=rotation_z(yaw_rate * time),
            torso_linear_velocity_world=np.array([0.4, 0.0, 0.0]),
            torso_angular_velocity_world=omega,
            torso_linear_acceleration_world_raw=acceleration,
            torso_linear_acceleration_world_used=acceleration,
            torso_angular_acceleration_world_raw=alpha,
            torso_angular_acceleration_world_used=alpha,
            lower_body_q=np.zeros(12), lower_body_dq=np.zeros(12), lower_body_policy_target=np.zeros(12),
            right_arm_q=np.zeros(5), right_arm_dq=np.zeros(5), right_arm_ddq_des=np.zeros(5),
            generalized_qpos=np.zeros(30), generalized_qvel=np.zeros(29), generalized_qacc=np.zeros(29), actuator_ctrl=np.zeros(23),
            heading_state=_heading_state(time >= PROTOCOL.policy_dt), mpc_diagnostics=None,
            runtime_mapping_safety_fallback_used=False, runtime_executor_flags=0,
        )
        concentration.append(recorder.causal_h.last_state.concentration)
    raw = recorder.to_arrays()
    raw.update({
        "right_arm_pd_requested_tau": np.zeros((PROTOCOL.physics_steps, 5)),
        "right_arm_pd_saturated": np.zeros((PROTOCOL.physics_steps, 5), dtype=bool),
        "right_arm_pd_position_error": np.zeros((PROTOCOL.physics_steps, 5)),
        "causal_h_concentration": np.asarray(concentration),
        "initial_lower_q_offset_rad": np.full(12, offset),
        "initial_lower_dq_rad_s": np.full(12, -offset),
    })
    return raw


class FullTaskTemplateBuilderTest(unittest.TestCase):
    def test_pair_specs_are_zero_mean_and_split_isolated(self) -> None:
        build, heldout = collection_specs()
        self.assertEqual(len(build), 11)
        self.assertEqual(len(heldout), 4)
        self.assertTrue(np.allclose(build[0].initial_lower_q_offset_rad, 0.0))
        self.assertFalse({item.pair_seed for item in build if item.pair_seed}.intersection({item.pair_seed for item in heldout if item.pair_seed}))
        for index in range(1, len(build), 2):
            self.assertTrue(np.allclose(build[index].initial_lower_q_offset_rad + build[index + 1].initial_lower_q_offset_rad, 0.0))
            self.assertTrue(np.allclose(build[index].initial_lower_dq_rad_s + build[index + 1].initial_lower_dq_rad_s, 0.0))

    def test_anchor_frozen_windows_template_and_heldout_metrics(self) -> None:
        first, second, heldout = _raw(-0.01), _raw(0.01), _raw(0.02)
        template = build_full_task_template([first, second], ["first", "second"], PROTOCOL)
        report = validate_full_task_template(template, PROTOCOL)
        self.assertEqual(report["anchor_count"], 1334)
        self.assertEqual(template["nodes_acceleration_mean"].shape, (1334, 10, 3))
        self.assertEqual(template["intervals_angular_acceleration_mean"].shape, (1334, 9, 3))
        self.assertTrue(np.allclose(template["nodes_acceleration_mean"][0, 0], np.array([0.0, 0.0, 0.0])))
        metrics, _ = evaluate_heldout_template(template, [heldout], ["heldout"], PROTOCOL)
        self.assertEqual(metrics["episode_count"], 1)
        self.assertGreater(metrics["mean_nodes_acceleration_rmse"], 0.0)
        self.assertGreaterEqual(metrics["max_nodes_orientation_rmse_rad"], 0.0)

    def test_h_metrics_are_unfiltered_and_finite(self) -> None:
        raw = _raw(0.0)
        metrics = causal_h_metrics(raw)
        self.assertEqual(metrics["anchor_count"], PROTOCOL.recorded_anchor_count)
        self.assertEqual(metrics["h_filtering"], "none; values are the shared causal circular means")
        self.assertGreaterEqual(metrics["min_circular_concentration"], 0.999999)
        self.assertAlmostEqual(metrics["max_adjacent_h_yaw_jump_rad"], 0.0)

    def test_v2_builder_replays_shared_continuous_h_and_freezes_window_frame(self) -> None:
        first = _raw(-0.01, continuous=True, yaw_rate=0.08)
        second = _raw(0.01, continuous=True, yaw_rate=0.08)
        template = build_full_task_template(
            [first, second], ["first", "second"], PROTOCOL,
            template_schema_version=TEMPLATE_SCHEMA_VERSION_V2,
        )
        report = validate_full_task_template(
            template, PROTOCOL, expected_schema_version=TEMPLATE_SCHEMA_VERSION_V2
        )
        self.assertEqual(
            report["heading_frame_version"],
            FullTaskContinuousHeadingFrame.DEFINITION_VERSION,
        )
        windows = _episode_heading_windows(first, PROTOCOL)
        anchor_index = 200
        raw_anchor = anchor_index * PROTOCOL.mpc_stride
        future_raw = raw_anchor + PROTOCOL.horizon * PROTOCOL.mpc_stride
        expected = (
            first["causal_h_rotation_world"][raw_anchor].T
            @ first["torso_rotation_world"][future_raw]
        )
        np.testing.assert_allclose(
            windows["nodes_rotation_heading"][anchor_index, -1],
            expected,
            atol=1e-12,
            rtol=0.0,
        )
        self.assertFalse(
            np.allclose(
                first["causal_h_rotation_world"][raw_anchor],
                first["causal_h_rotation_world"][future_raw],
            )
        )

if __name__ == "__main__":
    unittest.main()
