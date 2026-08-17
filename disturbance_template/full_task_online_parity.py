"""T2 offline-online replay parity for the one explicitly accepted T1 asset.

The replay is deliberately prefix-causal: at anchor k the online predictor has
received exactly anchors 0..k from one held-out raw episode.  It never scans for
templates, interpolates task time, or supplies future yaw observations.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from disturbance_template.full_task_protocol import (
    DEFAULT_FULL_TASK_PROTOCOL,
    is_valid_rotation_batch,
)
from disturbance_template.rotation_utils import rotation_z
from disturbance_predictor import (
    DisturbancePredictorObservation,
    FullTaskTemplatePredictor,
)
from kinematics_helper import DisturbanceInput


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = (
    ROOT
    / "disturbance_template/data/full_task_template_v2/20260815_162850"
)
TEMPLATE_PATH = ASSET_DIR / "full_task_template.npz"
MANIFEST_PATH = ASSET_DIR / "full_task_template_manifest.json"
TEMPLATE_SHA256 = (
    "d4a0109adcff696936ef96160976161833ff9a7a7531e2e5d7ad9e50c10e17d4"
)
MANIFEST_SHA256 = (
    "6b48ee196d1f7d923dde057d3c0fb0e182f08512a65402c4c39c5e070a3243c6"
)
HELDOUT_EPISODE_IDS = (
    "heldout_pair_01_plus",
    "heldout_pair_01_minus",
    "heldout_pair_02_plus",
    "heldout_pair_02_minus",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _rotation_geodesic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ij,...kj->...ik", left, right)
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.arccos(cosine)


def _max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _wrap(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _predictor() -> FullTaskTemplatePredictor:
    return FullTaskTemplatePredictor(
        template_path=str(TEMPLATE_PATH),
        manifest_path=str(MANIFEST_PATH),
        expected_sha256=TEMPLATE_SHA256,
        expected_manifest_sha256=MANIFEST_SHA256,
        repo_dir=str(ROOT),
        control_dt=DEFAULT_FULL_TASK_PROTOCOL.mpc_dt,
        horizon=DEFAULT_FULL_TASK_PROTOCOL.horizon,
        expected_schema_version="full_task_template_v2",
        expected_heading_frame_version="full_task_continuous_heading_v2",
    )


def _measurement(raw: dict[str, np.ndarray], index: int) -> DisturbanceInput:
    return DisturbanceInput(
        acc_world=np.asarray(
            raw["torso_linear_acceleration_world_used"][index],
            dtype=np.float64,
        ),
        omega_world=np.asarray(
            raw["torso_angular_velocity_world"][index], dtype=np.float64
        ),
        alpha_world=np.asarray(
            raw["torso_angular_acceleration_world_used"][index],
            dtype=np.float64,
        ),
        rot_world_body=np.asarray(
            raw["torso_rotation_world"][index], dtype=np.float64
        ),
    )


def replay_episode(
    raw_path: Path, template: dict[str, np.ndarray]
) -> dict[str, Any]:
    protocol = DEFAULT_FULL_TASK_PROTOCOL
    with np.load(raw_path, allow_pickle=False) as source:
        raw = {name: source[name].copy() for name in source.files}
    if str(np.asarray(raw["protocol_version"]).item()) != protocol.protocol_version:
        raise ValueError(f"protocol mismatch in {raw_path}")
    if str(np.asarray(raw["episode_role"]).item()) != "heldout":
        raise ValueError(f"non-held-out replay input: {raw_path}")
    anchor_raw_indices = np.flatnonzero(raw["mpc_anchor"])[
        : protocol.headline_anchor_count
    ]
    expected_raw_indices = (
        np.arange(protocol.headline_anchor_count, dtype=np.int64)
        * protocol.mpc_stride
    )
    if not np.array_equal(anchor_raw_indices, expected_raw_indices):
        raise ValueError("held-out raw does not have the exact headline anchor grid")

    predictor = _predictor()
    maxima = {
        "task_time_s": 0.0,
        "h_yaw_wrapped_rad": 0.0,
        "h_rotation_matrix_abs": 0.0,
        "node0_measurement_abs": 0.0,
        "nodes_1_9_acc_abs": 0.0,
        "nodes_1_9_omega_abs": 0.0,
        "nodes_1_9_alpha_abs": 0.0,
        "nodes_1_9_rotation_matrix_abs": 0.0,
        "nodes_1_9_rotation_geodesic_rad": 0.0,
        "intervals_acc_abs": 0.0,
        "intervals_omega_abs": 0.0,
        "intervals_alpha_abs": 0.0,
        "intervals_rotation_matrix_abs": 0.0,
        "intervals_rotation_geodesic_rad": 0.0,
    }
    fallback_count = 0
    invalid_rotation_count = 0
    terminal_hold_count = 0
    query_indices = []
    for anchor_index, raw_index in enumerate(anchor_raw_indices):
        measurement = _measurement(raw, int(raw_index))
        predictor.update(
            DisturbancePredictorObservation(
                simulation_time=float(raw["simulation_time"][raw_index]),
                measured_disturbance=measurement,
            )
        )
        preview = predictor.predict(protocol.horizon, protocol.mpc_dt)
        diagnostics = predictor.get_last_diagnostics()
        expected_time = float(template["anchor_task_time"][anchor_index])
        maxima["task_time_s"] = max(
            maxima["task_time_s"],
            abs(float(diagnostics["task_time"]) - expected_time),
        )
        if int(diagnostics["template_anchor_index"]) != anchor_index:
            raise ValueError("online template anchor index diverged from replay index")
        query_indices.append(int(diagnostics["template_anchor_index"]))
        raw_h_yaw = float(raw["causal_h_yaw_world"][raw_index])
        online_h_yaw = float(diagnostics["heading_yaw_world"])
        maxima["h_yaw_wrapped_rad"] = max(
            maxima["h_yaw_wrapped_rad"],
            abs(float(_wrap(online_h_yaw - raw_h_yaw))),
        )
        online_h_rotation = rotation_z(online_h_yaw)
        raw_h_rotation = np.asarray(
            raw["causal_h_rotation_world"][raw_index], dtype=np.float64
        )
        maxima["h_rotation_matrix_abs"] = max(
            maxima["h_rotation_matrix_abs"],
            _max_abs(online_h_rotation, raw_h_rotation),
        )

        actual_node_acc = np.stack([item.acc_world for item in preview.nodes])
        actual_node_omega = np.stack([item.omega_world for item in preview.nodes])
        actual_node_alpha = np.stack([item.alpha_world for item in preview.nodes])
        actual_node_rotation = np.stack(
            [item.rot_world_body for item in preview.nodes]
        )
        actual_interval_acc = np.stack(
            [item.acc_world for item in preview.intervals]
        )
        actual_interval_omega = np.stack(
            [item.omega_world for item in preview.intervals]
        )
        actual_interval_alpha = np.stack(
            [item.alpha_world for item in preview.intervals]
        )
        actual_interval_rotation = np.stack(
            [item.rot_world_body for item in preview.intervals]
        )
        expected_node_acc = (
            template["nodes_acceleration_mean"][anchor_index]
            @ online_h_rotation.T
        )
        expected_node_omega = (
            template["nodes_angular_velocity_mean"][anchor_index]
            @ online_h_rotation.T
        )
        expected_node_alpha = (
            template["nodes_angular_acceleration_mean"][anchor_index]
            @ online_h_rotation.T
        )
        expected_node_rotation = np.einsum(
            "ij,njk->nik",
            online_h_rotation,
            template["nodes_rotation_heading_mean"][anchor_index],
        )
        expected_interval_acc = (
            template["intervals_acceleration_mean"][anchor_index]
            @ online_h_rotation.T
        )
        expected_interval_omega = (
            template["intervals_angular_velocity_mean"][anchor_index]
            @ online_h_rotation.T
        )
        expected_interval_alpha = (
            template["intervals_angular_acceleration_mean"][anchor_index]
            @ online_h_rotation.T
        )
        expected_interval_rotation = np.einsum(
            "ij,njk->nik",
            online_h_rotation,
            template["intervals_rotation_heading_mean"][anchor_index],
        )
        maxima["node0_measurement_abs"] = max(
            maxima["node0_measurement_abs"],
            _max_abs(actual_node_acc[0], measurement.acc_world),
            _max_abs(actual_node_omega[0], measurement.omega_world),
            _max_abs(actual_node_alpha[0], measurement.alpha_world),
            _max_abs(actual_node_rotation[0], measurement.rot_world_body),
        )
        for key, actual, expected in (
            ("nodes_1_9_acc_abs", actual_node_acc[1:], expected_node_acc[1:]),
            ("nodes_1_9_omega_abs", actual_node_omega[1:], expected_node_omega[1:]),
            ("nodes_1_9_alpha_abs", actual_node_alpha[1:], expected_node_alpha[1:]),
            ("intervals_acc_abs", actual_interval_acc, expected_interval_acc),
            ("intervals_omega_abs", actual_interval_omega, expected_interval_omega),
            ("intervals_alpha_abs", actual_interval_alpha, expected_interval_alpha),
        ):
            maxima[key] = max(maxima[key], _max_abs(actual, expected))
        maxima["nodes_1_9_rotation_matrix_abs"] = max(
            maxima["nodes_1_9_rotation_matrix_abs"],
            _max_abs(actual_node_rotation[1:], expected_node_rotation[1:]),
        )
        maxima["nodes_1_9_rotation_geodesic_rad"] = max(
            maxima["nodes_1_9_rotation_geodesic_rad"],
            float(
                np.max(
                    _rotation_geodesic(
                        actual_node_rotation[1:], expected_node_rotation[1:]
                    )
                )
            ),
        )
        maxima["intervals_rotation_matrix_abs"] = max(
            maxima["intervals_rotation_matrix_abs"],
            _max_abs(actual_interval_rotation, expected_interval_rotation),
        )
        maxima["intervals_rotation_geodesic_rad"] = max(
            maxima["intervals_rotation_geodesic_rad"],
            float(
                np.max(
                    _rotation_geodesic(
                        actual_interval_rotation, expected_interval_rotation
                    )
                )
            ),
        )
        invalid_rotation_count += int(
            not np.all(is_valid_rotation_batch(actual_node_rotation))
        )
        invalid_rotation_count += int(
            not np.all(is_valid_rotation_batch(actual_interval_rotation))
        )
        fallback_count += int(bool(diagnostics["fallback_used"]))
        terminal_hold_count += int(bool(diagnostics["terminal_hold_used"]))

    tolerances = {
        "task_time_s": 2e-12,
        "h_yaw_wrapped_rad": 2e-12,
        "h_rotation_matrix_abs": 2e-12,
        "node0_measurement_abs": 0.0,
        "nodes_1_9_acc_abs": 2e-12,
        "nodes_1_9_omega_abs": 2e-12,
        "nodes_1_9_alpha_abs": 2e-12,
        "nodes_1_9_rotation_matrix_abs": 2e-12,
        "nodes_1_9_rotation_geodesic_rad": 5e-8,
        "intervals_acc_abs": 2e-12,
        "intervals_omega_abs": 2e-12,
        "intervals_alpha_abs": 2e-12,
        "intervals_rotation_matrix_abs": 2e-12,
        "intervals_rotation_geodesic_rad": 5e-8,
    }
    passed = (
        all(maxima[name] <= tolerance for name, tolerance in tolerances.items())
        and fallback_count == 0
        and terminal_hold_count == 0
        and invalid_rotation_count == 0
        and query_indices == list(range(protocol.headline_anchor_count))
    )
    return {
        "episode_id": str(np.asarray(raw["episode_id"]).item()),
        "raw_path": str(raw_path.resolve()),
        "headline_anchor_count": len(anchor_raw_indices),
        "last_anchor_task_time": float(raw["task_time"][anchor_raw_indices[-1]]),
        "max_abs_errors": maxima,
        "tolerances": tolerances,
        "predictor_fallback_count": fallback_count,
        "terminal_hold_count_in_headline": terminal_hold_count,
        "invalid_rotation_batch_count": invalid_rotation_count,
        "query_indices_exact_0_to_1333": query_indices
        == list(range(protocol.headline_anchor_count)),
        "causal_prefix_contract": (
            "at anchor k only raw anchor yaws 0..k were passed to update; "
            "future yaw samples were not supplied"
        ),
        "implicit_interpolation_used": False,
        "status": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=None,
        help="T2 result directory; the accepted T1 asset is always read-only.",
    )
    args = parser.parse_args()
    output_dir = (
        ROOT
        / "evaluation/t2_full_task_template_online"
        / datetime.now().strftime("%Y%m%d_%H%M%S_offline_online_parity")
        if args.output_dir is None
        else Path(args.output_dir).expanduser()
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(TEMPLATE_PATH, allow_pickle=False) as source:
        template = {name: source[name].copy() for name in source.files}
    episodes = [
        replay_episode(
            ASSET_DIR / "episodes" / episode_id / "full_task_fixed_pd_raw.npz",
            template,
        )
        for episode_id in HELDOUT_EPISODE_IDS
    ]
    global_maxima = {
        name: max(episode["max_abs_errors"][name] for episode in episodes)
        for name in episodes[0]["max_abs_errors"]
    }
    passed = all(episode["status"] == "PASS" for episode in episodes)
    report = {
        "stage": "T2 offline-online full-task template replay parity",
        "status": "PASS" if passed else "FAIL",
        "template": {
            "path": str(TEMPLATE_PATH.resolve()),
            "sha256": TEMPLATE_SHA256,
            "manifest_path": str(MANIFEST_PATH.resolve()),
            "manifest_sha256": MANIFEST_SHA256,
            "selection": "explicit pinned path; no directory scan",
        },
        "protocol": {
            "version": DEFAULT_FULL_TASK_PROTOCOL.protocol_version,
            "anchor_dt": DEFAULT_FULL_TASK_PROTOCOL.mpc_dt,
            "headline_interval": "[0.0,8.0)",
            "anchor_count": DEFAULT_FULL_TASK_PROTOCOL.headline_anchor_count,
            "last_anchor": DEFAULT_FULL_TASK_PROTOCOL.last_headline_anchor_time,
            "last_horizon_node": DEFAULT_FULL_TASK_PROTOCOL.last_horizon_node_time,
        },
        "global_max_abs_errors": global_maxima,
        "episodes": episodes,
    }
    path = output_dir / "offline_online_parity.json"
    path.write_text(
        json.dumps(_json_value(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_value(report), indent=2, ensure_ascii=False))
    print(f"report={path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
