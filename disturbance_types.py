"""MuJoCo-free disturbance wire types shared by predictor and control core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DisturbanceInput:
    acc_world: Optional[np.ndarray] = None
    omega_world: Optional[np.ndarray] = None
    alpha_world: Optional[np.ndarray] = None
    # Current node may be measured; full-task future nodes/intervals use the
    # current anchor's W_R_H multiplied by the v2 H-frame template rotation.
    rot_world_body: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DisturbanceHorizon:
    """MPC node and following-interval disturbances remain distinct."""

    nodes: tuple
    intervals: tuple


__all__ = ("DisturbanceHorizon", "DisturbanceInput")
