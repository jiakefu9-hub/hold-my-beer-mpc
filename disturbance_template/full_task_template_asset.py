"""Runtime-safe schema and integrity helpers for full-task template assets.

This module intentionally has no collector, raw-recorder, plotting, training,
or cross-episode averaging dependency.  Both the offline builder and the
online predictor import this one schema validator so a runtime import cannot
pull the offline construction stack into the control process.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from disturbance_template.full_task_protocol import (
    FullTaskCausalHeadingFrame,
    FullTaskContinuousHeadingFrame,
    FullTaskProtocol,
    is_valid_rotation_batch,
)


TEMPLATE_SCHEMA_VERSION = "full_task_template_v1"
TEMPLATE_SCHEMA_VERSION_V2 = "full_task_template_v2"
VECTOR_NAMES = ("acceleration", "angular_velocity", "angular_acceleration")


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 of one required asset without loading it in memory."""

    asset_path = Path(path)
    digest = hashlib.sha256()
    with asset_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load a template NPZ with pickle explicitly disabled."""

    with np.load(Path(path), allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def validate_full_task_template(
    template: dict[str, np.ndarray],
    protocol: FullTaskProtocol,
    *,
    expected_schema_version: str | None = None,
) -> dict[str, Any]:
    """Validate the frozen absolute-time template schema and SO(3) fields."""

    anchor_count = protocol.headline_anchor_count
    schema_version = str(
        np.asarray(template["template_schema_version"]).item()
    )
    if schema_version not in {
        TEMPLATE_SCHEMA_VERSION,
        TEMPLATE_SCHEMA_VERSION_V2,
    }:
        raise ValueError("unexpected full-task template schema")
    if (
        expected_schema_version is not None
        and schema_version != expected_schema_version
    ):
        raise ValueError(
            "full-task template schema does not match configured version"
        )
    expected_heading = (
        FullTaskContinuousHeadingFrame.DEFINITION_VERSION
        if schema_version == TEMPLATE_SCHEMA_VERSION_V2
        else FullTaskCausalHeadingFrame.DEFINITION_VERSION
    )
    heading_version = str(
        np.asarray(
            template.get(
                "heading_frame_version",
                np.array(FullTaskCausalHeadingFrame.DEFINITION_VERSION),
            )
        ).item()
    )
    if heading_version != expected_heading:
        raise ValueError(
            "full-task template heading-frame version disagrees with schema"
        )
    if not np.array_equal(
        template["anchor_task_time"], protocol.headline_anchor_times
    ):
        raise ValueError(
            "template anchor task-time grid disagrees with protocol"
        )

    expected_vector_shapes = {
        f"nodes_{name}_{stat}": (
            anchor_count,
            protocol.horizon + 1,
            3,
        )
        for name in VECTOR_NAMES
        for stat in ("mean", "std")
    }
    expected_vector_shapes.update(
        {
            f"intervals_{name}_{stat}": (
                anchor_count,
                protocol.horizon,
                3,
            )
            for name in VECTOR_NAMES
            for stat in ("mean", "std")
        }
    )
    for name, shape in expected_vector_shapes.items():
        value = np.asarray(template[name])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(
                f"template field {name} has invalid shape or values"
            )

    for prefix, horizon_size in (
        ("nodes", protocol.horizon + 1),
        ("intervals", protocol.horizon),
    ):
        rotations = np.asarray(
            template[f"{prefix}_rotation_heading_mean"]
        )
        quaternion = np.asarray(
            template[f"{prefix}_quaternion_heading_mean_wxyz"]
        )
        if rotations.shape != (
            anchor_count,
            horizon_size,
            3,
            3,
        ) or quaternion.shape != (anchor_count, horizon_size, 4):
            raise ValueError(f"template {prefix} orientation shape is invalid")
        if not np.all(is_valid_rotation_batch(rotations)):
            raise ValueError(f"template {prefix} orientation is not SO(3)")
        if not np.allclose(
            np.linalg.norm(quaternion, axis=-1), 1.0, atol=1e-8
        ):
            raise ValueError(
                f"template {prefix} quaternion is not normalized"
            )

    return {
        "anchor_count": anchor_count,
        "horizon": protocol.horizon,
        "node0_online_policy": str(
            np.asarray(template["node0_online_policy"]).item()
        ),
        "rotation_valid": True,
        "smoothing": str(np.asarray(template["smoothing"]).item()),
        "template_schema_version": schema_version,
        "heading_frame_version": heading_version,
    }
