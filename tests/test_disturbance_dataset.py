"""Regression tests for B1 causal history/target construction."""

from __future__ import annotations

import unittest

import numpy as np

from disturbance_learning.collect_dataset import (
    build_episode_profile,
    command_schedule,
)
from disturbance_learning.dataset import (
    FEATURE_NAMES,
    HEADING_DEFINITION,
    PRE_STEP_DEFINITION,
    build_supervised_windows,
    validate_supervised_windows,
)
from disturbance_model_new_heading.heading_template_utils import rotation_z


RAW_DT = 0.002
CONTROL_DT = 0.006
PERIOD = 0.8


def _synthetic_raw() -> dict[str, np.ndarray]:
    sample_count = int(round(2.4 / RAW_DT))
    time = np.arange(sample_count, dtype=np.float64) * RAW_DT
    cycle = np.floor(time / PERIOD + 1e-12).astype(np.int64)
    yaw = np.choose(np.minimum(cycle, 2), (0.1, 0.2, 0.3))
    rotation = rotation_z(yaw)
    q = np.column_stack(
        [time + 0.01 * index for index in range(12)]
    )
    dq = np.column_stack(
        [0.5 * time - 0.01 * index for index in range(12)]
    )
    target = np.column_stack(
        [0.25 * time + 0.02 * index for index in range(12)]
    )
    phase = np.mod(time / PERIOD, 1.0)
    return {
        "time": time,
        "physics_step_index": np.arange(sample_count, dtype=np.int64),
        "torso_rotation_world": rotation,
        "torso_linear_velocity_world": np.column_stack(
            (0.5 * time * time, time * time, -0.25 * time * time)
        ),
        "torso_linear_acceleration_world": np.column_stack(
            (time, 2.0 * time, -time)
        ),
        "torso_angular_velocity_world": np.column_stack(
            (0.5 * time, -time, 0.2 + 0.1 * time)
        ),
        "torso_angular_acceleration_world": np.column_stack(
            (3.0 * time, -2.0 * time, 1.0 + time)
        ),
        "gravity_direction_torso": np.tile(
            np.array([0.0, 0.0, -1.0]), (sample_count, 1)
        ),
        "lower_body_q": q,
        "lower_body_dq": dq,
        "lower_body_policy_target": target,
        "runtime_command": np.column_stack(
            (0.2 + 0.1 * time, np.zeros_like(time), 0.01 * time)
        ),
        "gait_phase_sin_cos": np.column_stack(
            (np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase))
        ),
        "schedule_segment_id": np.floor(time / 0.4).astype(np.int64),
        "simulation_dt": np.array(RAW_DT),
        "gait_period": np.array(PERIOD),
        "sample_timing": np.array(PRE_STEP_DEFINITION),
        "heading_definition": np.array(HEADING_DEFINITION),
        "episode_id": np.array("synthetic"),
        "schedule_segment_names": np.asarray(
            ("s0", "s1", "s2", "s3", "s4", "s5")
        ),
        "required_schedule_segment_ids": np.asarray([], dtype=np.int64),
    }


class DisturbanceDatasetTest(unittest.TestCase):
    def test_alignment_and_heading_match_template_semantics(self) -> None:
        raw = _synthetic_raw()
        dataset = build_supervised_windows(raw)
        report = validate_supervised_windows(dataset, raw)

        self.assertEqual(dataset["history"].shape[1:], (34, len(FEATURE_NAMES)))
        self.assertEqual(dataset["target"].shape[1:], (9, 6))
        self.assertEqual(int(dataset["anchor_raw_index"][0]), 402)
        np.testing.assert_array_equal(
            dataset["history_raw_indices"][0], np.arange(303, 403, 3)
        )
        np.testing.assert_array_equal(
            dataset["target_raw_indices"][0, 0], np.array([402, 405])
        )
        np.testing.assert_array_equal(
            dataset["target_raw_indices"][0, -1], np.array([426, 429])
        )
        self.assertAlmostEqual(dataset["heading_yaw_world"][0], 0.1)
        self.assertEqual(dataset["heading_source_cycle_id"][0], 0)

        expected_acc = (
            rotation_z(-0.1)
            @ (
                raw["torso_linear_velocity_world"][405]
                - raw["torso_linear_velocity_world"][402]
            )
            / CONTROL_DT
        )
        np.testing.assert_allclose(
            dataset["target_torso_linear_acceleration_heading"][0, 0],
            expected_acc,
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertEqual(report["history_future_leak_count"], 0)
        self.assertAlmostEqual(report["max_history_time_minus_anchor_s"], 0.0)
        self.assertAlmostEqual(report["min_target_time_minus_anchor_s"], 0.0)
        self.assertAlmostEqual(
            report["max_target_interval_end_minus_anchor_s"], 0.054
        )
        self.assertGreaterEqual(report["min_heading_source_margin_s"], 0.0)

    def test_future_mutation_cannot_change_first_history(self) -> None:
        raw = _synthetic_raw()
        original = build_supervised_windows(raw)
        anchor = int(original["anchor_raw_index"][0])

        mutated = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in raw.items()
        }
        future = np.arange(len(raw["time"])) > anchor
        for name in (
            "torso_linear_acceleration_world",
            "torso_linear_velocity_world",
            "torso_angular_velocity_world",
            "gravity_direction_torso",
            "lower_body_q",
            "lower_body_dq",
            "lower_body_policy_target",
            "runtime_command",
        ):
            mutated[name][future] += 1000.0
        mutated["torso_rotation_world"][future] = rotation_z(1.0)
        changed = build_supervised_windows(mutated)

        np.testing.assert_array_equal(
            original["history"][0], changed["history"][0]
        )
        self.assertEqual(
            original["heading_yaw_world"][0], changed["heading_yaw_world"][0]
        )
        self.assertFalse(
            np.array_equal(original["target"][0], changed["target"][0])
        )

    def test_command_schedule_covers_requested_transitions(self) -> None:
        nominal = np.array([0.5, 0.0, 0.0127])
        states = [
            command_schedule(time_s, nominal)
            for time_s in (0.4, 1.0, 1.6, 2.6, 3.3, 4.0)
        ]
        self.assertEqual([state.segment_id for state in states], list(range(6)))
        np.testing.assert_array_equal(states[0].command, np.zeros(3))
        np.testing.assert_array_equal(states[2].command, nominal)
        np.testing.assert_array_equal(states[-1].command, np.zeros(3))
        self.assertGreater(states[1].command[0], 0.0)
        self.assertGreater(states[3].command[1], 0.0)
        self.assertGreater(states[4].command[0], 0.0)

    def test_episode_seed_produces_bounded_reproducible_variation(self) -> None:
        nominal = np.array([0.5, 0.0, 0.0127])
        first = build_episode_profile(nominal, 1000)
        repeated = build_episode_profile(nominal, 1000)
        second = build_episode_profile(nominal, 1001)
        np.testing.assert_array_equal(first.start_command, repeated.start_command)
        np.testing.assert_array_equal(
            first.initial_lower_q_offset, repeated.initial_lower_q_offset
        )
        self.assertFalse(np.array_equal(first.start_command, second.start_command))
        self.assertTrue(0.30 <= first.start_command[0] <= 0.60)
        self.assertTrue(-0.12 <= first.changed_command[1] <= 0.12)
        self.assertLess(np.max(np.abs(first.initial_lower_q_offset)), 0.03)


if __name__ == "__main__":
    unittest.main()
