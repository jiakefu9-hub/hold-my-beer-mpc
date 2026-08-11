"""Fast structural tests for the B2 one-shot MLP baseline."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from disturbance_learning.train_mlp import (
    MLPDisturbanceModel,
    compute_normalization,
    parameter_count,
    split_episode_ids,
)


class MLPBaselineTest(unittest.TestCase):
    def test_model_outputs_whole_horizon_in_one_call(self) -> None:
        model = MLPDisturbanceModel(
            history_steps=34,
            feature_dim=50,
            hidden_sizes=(128, 128),
            horizon=9,
            target_dim=6,
        )
        output = model(torch.zeros(4, 34, 50))
        self.assertEqual(tuple(output.shape), (4, 9, 6))
        self.assertEqual(parameter_count(model), 241206)

    def test_episode_split_is_disjoint_and_deterministic(self) -> None:
        episode_ids = [f"episode_{index}" for index in range(18)]
        first = split_episode_ids(
            episode_ids,
            train_count=12,
            validation_count=3,
            test_count=3,
            seed=42,
        )
        repeated = split_episode_ids(
            episode_ids,
            train_count=12,
            validation_count=3,
            test_count=3,
            seed=42,
        )
        self.assertEqual(first, repeated)
        self.assertEqual(len(first["train"]), 12)
        self.assertEqual(len(first["validation"]), 3)
        self.assertEqual(len(first["test"]), 3)
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["validation"]) & set(first["test"]))

    def test_normalization_uses_feature_and_target_channels(self) -> None:
        history = np.arange(5 * 34 * 50, dtype=np.float32).reshape(5, 34, 50)
        target = np.arange(5 * 9 * 6, dtype=np.float32).reshape(5, 9, 6)
        normalization = compute_normalization(history, target)
        self.assertEqual(normalization["feature_mean"].shape, (50,))
        self.assertEqual(normalization["feature_std"].shape, (50,))
        self.assertEqual(normalization["target_mean"].shape, (6,))
        self.assertEqual(normalization["target_std"].shape, (6,))
        self.assertTrue(np.all(normalization["feature_std"] > 0.0))
        self.assertTrue(np.all(normalization["target_std"] > 0.0))


if __name__ == "__main__":
    unittest.main()
