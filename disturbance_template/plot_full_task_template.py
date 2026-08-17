#!/usr/bin/env python3
"""Plot the frozen Full-Task Template v2 in its runtime interval semantics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = (
    PACKAGE_DIR
    / "data"
    / "full_task_template_v2"
    / "20260815_162850"
    / "full_task_template.npz"
)
DEFAULT_OUTPUT = PACKAGE_DIR / "assets" / "full_task_template_v2_overview.png"


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Convert a batch of rotation matrices to roll/pitch/yaw in radians."""
    rotation = np.asarray(rotation, dtype=np.float64)
    return np.column_stack(
        [
            np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2]),
            np.arcsin(np.clip(-rotation[:, 2, 0], -1.0, 1.0)),
            np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]),
        ]
    )


def plot_template(template_path: Path, output_path: Path) -> None:
    with np.load(template_path, allow_pickle=False) as loaded:
        template = {name: np.asarray(loaded[name]) for name in loaded.files}

    time = np.asarray(template["anchor_task_time"], dtype=np.float64)
    if time.shape != (1334,) or not np.isclose(float(template["anchor_dt"]), 0.006):
        raise ValueError("unexpected Full-Task Template anchor grid")
    if int(template["horizon"]) != 9 or str(template["smoothing"]) != "none":
        raise ValueError("unexpected Full-Task Template horizon or smoothing")

    rows = (
        (
            "H-frame linear acceleration",
            "m/s²",
            template["intervals_acceleration_mean"][:, 0],
            template["intervals_acceleration_std"][:, 0],
        ),
        (
            "H-frame angular velocity",
            "rad/s",
            template["intervals_angular_velocity_mean"][:, 0],
            template["intervals_angular_velocity_std"][:, 0],
        ),
        (
            "H-frame angular acceleration",
            "rad/s²",
            template["intervals_angular_acceleration_mean"][:, 0],
            template["intervals_angular_acceleration_std"][:, 0],
        ),
        (
            "H-frame torso orientation",
            "deg",
            np.rad2deg(
                rotation_to_rpy(template["intervals_rotation_heading_mean"][:, 0])
            ),
            None,
        ),
    )
    component_names = (("x", "y", "z"), ("x", "y", "z"), ("x", "y", "z"), ("roll", "pitch", "yaw"))
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    for row_index, (title, unit, mean, std) in enumerate(rows):
        for component in range(3):
            axis = axes[row_index, component]
            axis.plot(time, mean[:, component], color=colors[component], lw=1.0)
            if std is not None:
                axis.fill_between(
                    time,
                    mean[:, component] - std[:, component],
                    mean[:, component] + std[:, component],
                    color=colors[component],
                    alpha=0.18,
                    linewidth=0.0,
                )
            axis.axvline(6.4, color="black", ls="--", lw=0.8, alpha=0.7)
            axis.grid(True, alpha=0.25)
            axis.set_title(f"{title}: {component_names[row_index][component]}")
            axis.set_ylabel(unit)
            if row_index == 3:
                axis.set_xlabel("anchor task time (s)")

    figure.suptitle(
        "Full-Task Template v2 — first future 6 ms interval at every 6 ms anchor\n"
        "11-episode mean; shaded band = ±1 std; dashed line = direct stop at 6.4 s",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    plot_template(arguments.template.resolve(), arguments.output.resolve())
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
