#!/usr/bin/env python3
"""Train an MLP correction to the exact online phase-template intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from disturbance_learning.collect_dataset import REPO_DIR
from disturbance_learning.dataset import (
    FEATURE_NAMES,
    HEADING_DEFINITION,
    TARGET_NAMES,
)
from disturbance_learning.mlp_model import load_mlp_checkpoint, parameter_count
from disturbance_learning.train_mlp import (
    _normalized_arrays,
    _repo_path,
    _select_episodes,
    _stack_windows,
    benchmark_inference,
    compute_normalization,
    load_episodes,
    overfit_sanity_check,
    predict_mlp,
    prediction_metrics,
    split_episode_ids,
    template_predictions,
    train_model,
    zoh_predictions,
)


DEFAULT_CONFIG = Path(__file__).with_name("residual_mlp.yaml")


def _template_for_split(episodes, episode_ids, robot_config) -> np.ndarray:
    return np.concatenate(
        [
            template_predictions(episode, robot_config)
            for episode in _select_episodes(episodes, episode_ids)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual-template MLP")
    parser.add_argument("--experiment-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--robot-config", default="g1.yaml")
    args = parser.parse_args()

    experiment_path = Path(args.experiment_config).expanduser().resolve()
    with experiment_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    robot_config_path = REPO_DIR / "configs" / args.robot_config
    with robot_config_path.open("r", encoding="utf-8") as file:
        robot_config = yaml.safe_load(file)
    if config["model"].get("prediction_mode") != "residual_template":
        raise ValueError("residual checkpoint 必须声明 prediction_mode=residual_template。")

    episodes = load_episodes(_repo_path(config["collection"]["output_dir"]))
    split_config = config["split"]
    split = split_episode_ids(
        [episode.episode_id for episode in episodes],
        train_count=int(split_config["train_episodes"]),
        validation_count=int(split_config["validation_episodes"]),
        test_count=int(split_config["test_episodes"]),
        seed=int(split_config["shuffle_seed"]),
    )
    arrays = {
        name: _stack_windows(episodes, episode_ids)
        for name, episode_ids in split.items()
    }
    template_by_split = {
        name: _template_for_split(episodes, episode_ids, robot_config)
        for name, episode_ids in split.items()
    }
    residual_targets = {
        name: arrays[name][1] - template_by_split[name]
        for name in arrays
    }
    normalization = compute_normalization(
        arrays["train"][0], residual_targets["train"]
    )
    normalized = {
        name: _normalized_arrays(
            values[0], residual_targets[name], normalization
        )
        for name, values in arrays.items()
    }
    overfit = overfit_sanity_check(
        config, normalized["train"][0], normalized["train"][1]
    )
    model, training = train_model(
        config,
        normalized["train"][0],
        normalized["train"][1],
        normalized["validation"][0],
        normalized["validation"][1],
    )

    hybrid_split_metrics = {}
    for name in ("train", "validation", "test"):
        predicted_residual = predict_mlp(
            model, arrays[name][0], normalization
        )
        hybrid_split_metrics[name] = prediction_metrics(
            template_by_split[name] + predicted_residual,
            arrays[name][1],
            arrays[name][2],
        )

    test_episodes = _select_episodes(episodes, split["test"])
    test_target = arrays["test"][1]
    test_segments = arrays["test"][2]
    zoh = np.concatenate(
        [zoh_predictions(episode) for episode in test_episodes]
    )
    predicted_residual = predict_mlp(
        model, arrays["test"][0], normalization
    )
    hybrid_prediction = template_by_split["test"] + predicted_residual
    comparison = {
        "zoh": prediction_metrics(zoh, test_target, test_segments),
        "template": prediction_metrics(
            template_by_split["test"], test_target, test_segments
        ),
        "hybrid_residual": prediction_metrics(
            hybrid_prediction, test_target, test_segments
        ),
    }
    timing = benchmark_inference(
        model,
        arrays["test"][0][0],
        normalization,
        config["timing"],
    )

    artifact_dir = _repo_path(config["outputs"]["artifact_dir"])
    summary_dir = _repo_path(config["outputs"]["summary_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "residual_mlp_checkpoint.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "history_steps": int(arrays["train"][0].shape[1]),
        "feature_dim": int(arrays["train"][0].shape[2]),
        "hidden_sizes": list(config["model"]["hidden_sizes"]),
        "horizon": int(arrays["train"][1].shape[1]),
        "target_dim": int(arrays["train"][1].shape[2]),
        "feature_names": list(FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "prediction_mode": "residual_template",
        "control_dt": float(
            np.asarray(
                _select_episodes(episodes, split["train"])[0].windows[
                    "control_dt"
                ]
            ).item()
        ),
        "heading_definition": HEADING_DEFINITION,
        "template_variant": str(robot_config["mpc_disturbance_template"]),
        "template_slow_bias_enabled": bool(
            robot_config["mpc_disturbance_slow_bias_enabled"]
        ),
        "template_slow_bias_time_constant": float(
            robot_config["mpc_disturbance_slow_bias_time_constant"]
        ),
        "normalization": normalization,
        "episode_split": split,
    }
    torch.save(checkpoint, checkpoint_path)
    reloaded_model, reloaded_normalization, reloaded_payload = (
        load_mlp_checkpoint(checkpoint_path)
    )
    normalized_sample, _ = _normalized_arrays(
        arrays["test"][0][:1],
        residual_targets["test"][:1],
        reloaded_normalization,
    )
    with torch.inference_mode():
        reference = model(torch.from_numpy(normalized_sample)).numpy()
        reloaded = reloaded_model(torch.from_numpy(normalized_sample)).numpy()
    roundtrip_error = float(np.max(np.abs(reference - reloaded)))
    if roundtrip_error > 1e-7:
        raise RuntimeError("residual checkpoint reload parity 失败。")
    if reloaded_payload.get("prediction_mode") != "residual_template":
        raise RuntimeError("residual checkpoint 语义 metadata 丢失。")

    checkpoint_relative = str(checkpoint_path.relative_to(REPO_DIR))
    summary = {
        "stage": "neural_closed_loop_residual_training",
        "data": {
            "episode_count": len(episodes),
            "total_sample_count": int(
                sum(len(episode.windows["target"]) for episode in episodes)
            ),
            "split_episode_ids": split,
            "split_sample_counts": {
                name: int(len(values[0])) for name, values in arrays.items()
            },
            "history_shape_per_sample": list(arrays["train"][0].shape[1:]),
            "absolute_target_shape_per_sample": list(
                arrays["train"][1].shape[1:]
            ),
            "learning_target": (
                "absolute_future_interval_acc_alpha_H_minus_"
                "sequential_template_interval_acc_alpha_H_with_slow_bias"
            ),
        },
        "model": {
            "type": "flatten_mlp_one_shot",
            "prediction_mode": "residual_template",
            "hidden_sizes": list(config["model"]["hidden_sizes"]),
            "parameter_count": parameter_count(model),
            "output_shape": [9, 6],
        },
        "normalization": {
            "fit_split": "train_episodes_only",
            "feature_channel_count": len(FEATURE_NAMES),
            "residual_target_channel_count": len(TARGET_NAMES),
            "stored_with_local_checkpoint": True,
        },
        "overfit_sanity": overfit,
        "training": {
            name: value
            for name, value in training.items()
            if name != "history"
        },
        "hybrid_split_metrics": hybrid_split_metrics,
        "test_comparison": comparison,
        "cpu_residual_mlp_inference": timing,
        "checkpoint_path_local_ignored": checkpoint_relative,
        "checkpoint_reload_max_error": roundtrip_error,
    }
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = summary_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"checkpoint: {checkpoint_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
