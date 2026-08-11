#!/usr/bin/env python3
"""Collect the reproducible multi-episode dataset used by B2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from disturbance_learning.collect_dataset import (
    REPO_DIR,
    _load_config,
    collect_raw_episode,
)
from disturbance_learning.dataset import (
    DEFAULT_CONTROL_DT,
    DEFAULT_HISTORY_STEPS,
    DEFAULT_HORIZON,
    build_supervised_windows,
    validate_supervised_windows,
)


DEFAULT_EXPERIMENT_CONFIG = Path(__file__).with_name("mlp_baseline.yaml")


def _repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else REPO_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect B2 MLP episodes")
    parser.add_argument("--robot-config", default="g1.yaml")
    parser.add_argument(
        "--experiment-config", default=str(DEFAULT_EXPERIMENT_CONFIG)
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="validate and keep already completed episode files",
    )
    args = parser.parse_args()

    experiment_path = Path(args.experiment_config).expanduser().resolve()
    with experiment_path.open("r", encoding="utf-8") as file:
        experiment = yaml.safe_load(file)
    robot_config, robot_config_path = _load_config(args.robot_config)
    collection = experiment["collection"]
    output_dir = _repo_path(collection["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_count = int(collection["episode_count"])
    seed_start = int(collection["seed_start"])

    manifest_episodes = []
    for offset in range(episode_count):
        seed = seed_start + offset
        episode_id = f"b2_episode_{offset:02d}_seed_{seed}"
        prefix = output_dir / episode_id
        raw_path = Path(f"{prefix}_raw.npz")
        windows_path = Path(f"{prefix}_windows.npz")
        validation_path = Path(f"{prefix}_validation.json")

        if args.reuse_existing and raw_path.is_file() and windows_path.is_file():
            with np.load(raw_path, allow_pickle=False) as source:
                raw = {name: source[name].copy() for name in source.files}
            with np.load(windows_path, allow_pickle=False) as source:
                dataset = {name: source[name].copy() for name in source.files}
            report = validate_supervised_windows(dataset, raw)
        else:
            raw = collect_raw_episode(
                robot_config,
                config_path=robot_config_path,
                episode_id=episode_id,
                seed=seed,
            )
            dataset = build_supervised_windows(
                raw,
                history_steps=DEFAULT_HISTORY_STEPS,
                horizon=DEFAULT_HORIZON,
                control_dt=DEFAULT_CONTROL_DT,
            )
            report = validate_supervised_windows(dataset, raw)
            np.savez_compressed(raw_path, **raw)
            np.savez_compressed(windows_path, **dataset)
            validation_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        item = {
            "episode_id": episode_id,
            "seed": seed,
            "raw_path": str(raw_path),
            "windows_path": str(windows_path),
            "sample_count": int(report["sample_count"]),
            "history_shape": report["history_shape"],
            "target_shape": report["target_shape"],
            "start_command": np.asarray(raw["start_command"]).tolist(),
            "changed_command": np.asarray(raw["changed_command"]).tolist(),
        }
        manifest_episodes.append(item)
        print(
            f"[{offset + 1:02d}/{episode_count:02d}] {episode_id}: "
            f"{item['sample_count']} windows"
        )

    manifest = {
        "experiment_config": str(experiment_path),
        "robot_config": str(robot_config_path),
        "episode_count": episode_count,
        "total_sample_count": int(
            sum(item["sample_count"] for item in manifest_episodes)
        ),
        "episodes": manifest_episodes,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
