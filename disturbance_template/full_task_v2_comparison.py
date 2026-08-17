#!/usr/bin/env python3
"""Create auditable v1/v2 H and held-out comparisons for one explicit v2 asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from disturbance_template.full_task_template_builder import (
    _json_value,
    causal_h_metrics,
    portable_asset,
    wrapped_angle_difference,
)


REPO_DIR = Path(__file__).resolve().parents[1]
PHYSICAL_PARITY_FIELDS = (
    "task_time",
    "planned_command",
    "runtime_command",
    "torso_position_world",
    "torso_rotation_world",
    "torso_angular_velocity_world",
    "torso_linear_acceleration_world_raw",
    "torso_linear_acceleration_world_used",
    "torso_angular_acceleration_world_raw",
    "torso_angular_acceleration_world_used",
    "lower_body_q",
    "lower_body_dq",
    "right_arm_q",
    "right_arm_dq",
    "right_arm_pd_requested_tau",
)
METRIC_KEYS = (
    "mean_nodes_acceleration_rmse",
    "mean_nodes_angular_velocity_rmse",
    "mean_nodes_angular_acceleration_rmse",
    "mean_nodes_orientation_rmse_rad",
    "mean_intervals_acceleration_rmse",
    "mean_intervals_angular_velocity_rmse",
    "mean_intervals_angular_acceleration_rmse",
    "mean_intervals_orientation_rmse_rad",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _episode_assets(manifest: dict[str, Any]) -> dict[str, Path]:
    assets = manifest["build_raw_episodes"] + manifest["heldout_raw_episodes"]
    return {
        Path(item["path"]).parent.name: (REPO_DIR / item["path"]).resolve()
        for item in assets
    }


def run(v1_manifest_path: Path, v2_manifest_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v1_manifest = _load_json(v1_manifest_path)
    v2_manifest = _load_json(v2_manifest_path)
    if v1_manifest.get("template_schema_version") != "full_task_template_v1":
        raise ValueError("the explicit v1 manifest is not template v1")
    if v2_manifest.get("template_schema_version") != "full_task_template_v2":
        raise ValueError("the explicit v2 manifest is not template v2")
    v1_assets = _episode_assets(v1_manifest)
    v2_assets = _episode_assets(v2_manifest)
    if set(v1_assets) != set(v2_assets):
        raise ValueError("v1/v2 episode identities differ")

    plots_dir = v2_manifest_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    per_episode: list[dict[str, Any]] = []
    physical_max: dict[str, float] = {name: 0.0 for name in PHYSICAL_PARITY_FIELDS}
    v2_series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for episode_id in sorted(v1_assets):
        v1_raw = _load_npz(v1_assets[episode_id])
        v2_raw = _load_npz(v2_assets[episode_id])
        if str(np.asarray(v2_raw["heading_frame_version"]).item()) != "full_task_continuous_heading_v2":
            raise ValueError(f"{episode_id} is not continuous-H v2 raw")
        for field in PHYSICAL_PARITY_FIELDS:
            difference = float(
                np.max(
                    np.abs(
                        np.asarray(v2_raw[field], dtype=np.float64)
                        - np.asarray(v1_raw[field], dtype=np.float64)
                    )
                )
            )
            physical_max[field] = max(physical_max[field], difference)
        v1_h = causal_h_metrics(v1_raw)
        v2_h = causal_h_metrics(v2_raw)
        per_episode.append({"episode_id": episode_id, "v1": v1_h, "v2": v2_h})
        anchors = np.asarray(v2_raw["mpc_anchor"], dtype=bool)
        time = np.asarray(v2_raw["task_time"], dtype=np.float64)[anchors]
        yaw = np.asarray(v2_raw["causal_h_yaw_world"], dtype=np.float64)[anchors]
        jump = np.r_[0.0, np.abs(wrapped_angle_difference(yaw[1:], yaw[:-1]))]
        v2_series.append((episode_id, time, yaw, jump))

    if any(value != 0.0 for value in physical_max.values()):
        raise ValueError(f"v1/v2 physical fixed-PD raw changed: {physical_max}")

    path_h = plots_dir / "v2_all_h_yaw_trajectories.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    for episode_id, time, yaw, _ in v2_series:
        ax.plot(time, np.unwrap(yaw), lw=0.7, alpha=0.55, label=episode_id)
    ax.axvline(6.4, color="red", ls="--", lw=1.0, label="direct stop / H freeze")
    ax.set_xlabel("task time [s]")
    ax.set_ylabel("continuous H yaw [rad, unwrapped]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=6)
    fig.tight_layout()
    fig.savefig(path_h, dpi=170)
    plt.close(fig)

    path_jump = plots_dir / "v2_adjacent_h_yaw_jump.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    for episode_id, time, _, jump in v2_series:
        ax.plot(time, jump, lw=0.7, alpha=0.55, label=episode_id)
    ax.axvline(6.4, color="red", ls="--", lw=1.0)
    ax.set_xlabel("task time [s]")
    ax.set_ylabel("wrapped adjacent H yaw jump [rad]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_jump, dpi=170)
    plt.close(fig)

    v1_jump = max(item["v1"]["max_adjacent_h_yaw_jump_rad"] for item in per_episode)
    v2_jump = max(item["v2"]["max_adjacent_h_yaw_jump_rad"] for item in per_episode)
    path_jump_compare = plots_dir / "v1_v2_max_h_yaw_jump.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(("v1 cycle-held", "v2 continuous"), (v1_jump, v2_jump), color=("tab:orange", "tab:blue"))
    ax.set_ylabel("max wrapped adjacent jump [rad]")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_jump_compare, dpi=170)
    plt.close(fig)

    v1_metrics_path = (REPO_DIR / v1_manifest["heldout_metrics"]["path"]).resolve()
    v2_metrics_path = (REPO_DIR / v2_manifest["heldout_metrics"]["path"]).resolve()
    v1_metrics = _load_json(v1_metrics_path)
    v2_metrics = _load_json(v2_metrics_path)
    metric_comparison = {
        key: {
            "v1": float(v1_metrics[key]),
            "v2": float(v2_metrics[key]),
            "v2_minus_v1": float(v2_metrics[key] - v1_metrics[key]),
        }
        for key in METRIC_KEYS
    }
    path_metrics = plots_dir / "v1_v2_heldout_error_comparison.png"
    labels = [key.removeprefix("mean_").replace("_rmse", "") for key in METRIC_KEYS]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for axis, indices in zip(axes, (range(3), range(3, 8))):
        selected = list(indices)
        axis.bar(x[selected] - 0.18, [v1_metrics[METRIC_KEYS[i]] for i in selected], 0.36, label="v1")
        axis.bar(x[selected] + 0.18, [v2_metrics[METRIC_KEYS[i]] for i in selected], 0.36, label="v2")
        axis.set_xticks(x[selected], [labels[i] for i in selected], rotation=20, ha="right")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("RMSE (native units)")
    axes[1].set_ylabel("RMSE (native units / rad)")
    fig.tight_layout()
    fig.savefig(path_metrics, dpi=170)
    plt.close(fig)

    comparison = {
        "v1_manifest": portable_asset(v1_manifest_path, REPO_DIR),
        "v2_manifest": str(v2_manifest_path.resolve()),
        "episode_identity_and_initial_conditions_equal": True,
        "physical_raw_max_abs_difference": physical_max,
        "v1_max_adjacent_h_yaw_jump_rad": v1_jump,
        "v2_max_adjacent_h_yaw_jump_rad": v2_jump,
        "v2_min_circular_concentration": min(
            item["v2"]["min_circular_concentration"] for item in per_episode
        ),
        "per_episode_h": per_episode,
        "heldout_error_comparison": metric_comparison,
        "additional_h_low_pass_filter": "none",
    }
    comparison_path = v2_manifest_path.parent / "v1_v2_comparison.json"
    comparison_path.write_text(
        json.dumps(_json_value(comparison), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    comparison_assets = {
        "report": portable_asset(comparison_path, REPO_DIR),
        "plots": [
            portable_asset(path, REPO_DIR)
            for path in (path_h, path_jump, path_jump_compare, path_metrics)
        ],
    }
    v2_manifest["v1_v2_comparison"] = comparison_assets
    v2_manifest_path.write_text(
        json.dumps(_json_value(v2_manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"comparison": comparison, "assets": comparison_assets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-manifest", required=True)
    parser.add_argument("--v2-manifest", required=True)
    args = parser.parse_args()
    result = run(
        Path(args.v1_manifest).expanduser().resolve(),
        Path(args.v2_manifest).expanduser().resolve(),
    )
    print(json.dumps(_json_value(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
