"""Small shared MLP definition and checkpoint loader for offline/online use."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


class MLPDisturbanceModel(nn.Module):
    """One-shot causal history to a complete interval disturbance horizon."""

    def __init__(
        self,
        history_steps: int,
        feature_dim: int,
        hidden_sizes: Iterable[int],
        horizon: int,
        target_dim: int,
    ) -> None:
        super().__init__()
        hidden_sizes = tuple(int(value) for value in hidden_sizes)
        if not hidden_sizes or any(value < 1 for value in hidden_sizes):
            raise ValueError("MLP 至少需要一个正数 hidden size。")
        self.history_steps = int(history_steps)
        self.feature_dim = int(feature_dim)
        self.horizon = int(horizon)
        self.target_dim = int(target_dim)
        widths = (
            self.history_steps * self.feature_dim,
            *hidden_sizes,
            self.horizon * self.target_dim,
        )
        layers: list[nn.Module] = [nn.Flatten()]
        for index, (input_width, output_width) in enumerate(
            zip(widths[:-1], widths[1:])
        ):
            layers.append(nn.Linear(input_width, output_width))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        prediction = self.network(history)
        return prediction.reshape(-1, self.horizon, self.target_dim)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def load_mlp_checkpoint(
    checkpoint_path: Path,
) -> tuple[MLPDisturbanceModel, dict[str, np.ndarray], dict]:
    """Load a CPU inference checkpoint and preserve its semantic metadata."""
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = MLPDisturbanceModel(
        history_steps=int(payload["history_steps"]),
        feature_dim=int(payload["feature_dim"]),
        hidden_sizes=payload["hidden_sizes"],
        horizon=int(payload["horizon"]),
        target_dim=int(payload["target_dim"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    normalization = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in payload["normalization"].items()
    }
    return model, normalization, payload
