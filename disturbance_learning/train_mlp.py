#!/usr/bin/env python3
"""Train and evaluate the B2 feed-forward disturbance baseline."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from disturbance_learning.collect_dataset import REPO_DIR
from disturbance_learning.dataset import FEATURE_NAMES, TARGET_NAMES
from disturbance_model_new_heading.heading_template_utils import rotation_z
from disturbance_predictor import (
    DisturbancePredictorObservation,
    TemplateDisturbancePredictor,
)
from kinematics_helper import DisturbanceInput


DEFAULT_EXPERIMENT_CONFIG = Path(__file__).with_name("mlp_baseline.yaml")
STAGE_NAMES = {
    1: "start",
    2: "steady",
    3: "velocity_change",
    4: "stop",
    5: "stopped",
}


@dataclass(frozen=True)
class EpisodeData:
    episode_id: str
    raw: dict[str, np.ndarray]
    windows: dict[str, np.ndarray]


class MLPDisturbanceModel(nn.Module):
    """One-shot 34x50 history to 9x6 interval prediction."""

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
) -> tuple[MLPDisturbanceModel, dict[str, np.ndarray]]:
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
    return model, normalization


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_DIR / path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def load_episodes(data_dir: Path) -> list[EpisodeData]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 B2 episode manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = []
    for item in manifest["episodes"]:
        raw_path = Path(item["raw_path"])
        windows_path = Path(item["windows_path"])
        raw = _load_npz(raw_path)
        windows = _load_npz(windows_path)
        episode_id = str(np.asarray(raw["episode_id"]).item())
        if episode_id != item["episode_id"]:
            raise ValueError("manifest 与 raw episode_id 不一致。")
        if not np.all(windows["sample_episode_id"] == episode_id):
            raise ValueError("window sample_episode_id 跨 episode。")
        episodes.append(EpisodeData(episode_id, raw, windows))
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("manifest 包含重复 episode_id。")
    return episodes


def split_episode_ids(
    episode_ids: Iterable[str],
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[str]]:
    episode_ids = np.asarray(tuple(episode_ids))
    counts = (int(train_count), int(validation_count), int(test_count))
    if any(count < 1 for count in counts) or sum(counts) != len(episode_ids):
        raise ValueError("train/validation/test episode 数量必须为正且覆盖全部 episode。")
    permutation = np.random.default_rng(int(seed)).permutation(len(episode_ids))
    shuffled = episode_ids[permutation].tolist()
    train_end = counts[0]
    validation_end = train_end + counts[1]
    split = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    if len(set().union(*(set(values) for values in split.values()))) != len(
        episode_ids
    ):
        raise AssertionError("episode split 发生重叠或遗漏。")
    return split


def _select_episodes(
    episodes: list[EpisodeData], episode_ids: Iterable[str]
) -> list[EpisodeData]:
    by_id = {episode.episode_id: episode for episode in episodes}
    return [by_id[episode_id] for episode_id in episode_ids]


def _stack_windows(
    episodes: list[EpisodeData], episode_ids: Iterable[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = _select_episodes(episodes, episode_ids)
    return (
        np.concatenate([episode.windows["history"] for episode in selected]),
        np.concatenate([episode.windows["target"] for episode in selected]),
        np.concatenate(
            [
                episode.windows["anchor_schedule_segment_id"]
                for episode in selected
            ]
        ),
        np.concatenate(
            [episode.windows["sample_episode_id"] for episode in selected]
        ),
    )


def compute_normalization(
    train_history: np.ndarray, train_target: np.ndarray
) -> dict[str, np.ndarray]:
    feature_values = train_history.reshape(-1, train_history.shape[-1]).astype(
        np.float64
    )
    target_values = train_target.reshape(-1, train_target.shape[-1]).astype(
        np.float64
    )
    feature_mean = feature_values.mean(axis=0)
    feature_std = feature_values.std(axis=0)
    target_mean = target_values.mean(axis=0)
    target_std = target_values.std(axis=0)
    feature_std = np.maximum(feature_std, 1e-6)
    target_std = np.maximum(target_std, 1e-6)
    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
    }


def _normalized_arrays(
    history: np.ndarray,
    target: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    normalized_history = (
        history - normalization["feature_mean"][None, None, :]
    ) / normalization["feature_std"][None, None, :]
    normalized_target = (
        target - normalization["target_mean"][None, None, :]
    ) / normalization["target_std"][None, None, :]
    return normalized_history.astype(np.float32), normalized_target.astype(
        np.float32
    )


def _new_model(config: dict, history_shape: tuple[int, ...]) -> MLPDisturbanceModel:
    return MLPDisturbanceModel(
        history_steps=history_shape[1],
        feature_dim=history_shape[2],
        hidden_sizes=config["model"]["hidden_sizes"],
        horizon=9,
        target_dim=6,
    )


def overfit_sanity_check(
    config: dict,
    history: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int | bool]:
    sanity = config["overfit_sanity"]
    sample_count = min(int(sanity["sample_count"]), len(history))
    torch.manual_seed(int(config["training"]["seed"]))
    model = _new_model(config, history.shape)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(sanity["learning_rate"])
    )
    inputs = torch.from_numpy(history[:sample_count])
    labels = torch.from_numpy(target[:sample_count])
    loss_function = nn.MSELoss()
    model.train()
    with torch.no_grad():
        initial_loss = float(loss_function(model(inputs), labels).item())
    for _ in range(int(sanity["steps"])):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(inputs), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = float(loss_function(model(inputs), labels).item())
    ratio = final_loss / max(initial_loss, 1e-12)
    required_ratio = float(sanity["required_loss_ratio"])
    passed = ratio <= required_ratio
    if not passed:
        raise RuntimeError(
            f"MLP overfit sanity 失败: loss ratio={ratio:.6f} > {required_ratio:.6f}"
        )
    return {
        "sample_count": sample_count,
        "steps": int(sanity["steps"]),
        "initial_normalized_mse": initial_loss,
        "final_normalized_mse": final_loss,
        "loss_ratio": ratio,
        "required_loss_ratio": required_ratio,
        "passed": passed,
    }


def _dataset(history: np.ndarray, target: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(history), torch.from_numpy(target))


def _evaluate_normalized_mse(
    model: nn.Module, history: np.ndarray, target: np.ndarray
) -> float:
    model.eval()
    total_squared_error = 0.0
    value_count = 0
    loader = DataLoader(_dataset(history, target), batch_size=512)
    with torch.inference_mode():
        for inputs, labels in loader:
            error = model(inputs) - labels
            total_squared_error += float(torch.sum(error * error).item())
            value_count += error.numel()
    return total_squared_error / value_count


def train_model(
    config: dict,
    train_history: np.ndarray,
    train_target: np.ndarray,
    validation_history: np.ndarray,
    validation_target: np.ndarray,
) -> tuple[MLPDisturbanceModel, dict[str, object]]:
    training = config["training"]
    torch.manual_seed(int(training["seed"]))
    model = _new_model(config, train_history.shape)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loss_function = nn.MSELoss()
    generator = torch.Generator().manual_seed(int(training["seed"]))
    loader = DataLoader(
        _dataset(train_history, train_target),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    best_state = None
    best_epoch = -1
    best_validation_loss = np.inf
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()
    for epoch in range(int(training["max_epochs"])):
        model.train()
        squared_error = 0.0
        value_count = 0
        for inputs, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = loss_function(prediction, labels)
            loss.backward()
            optimizer.step()
            squared_error += float(
                torch.sum((prediction.detach() - labels) ** 2).item()
            )
            value_count += prediction.numel()
        train_loss = squared_error / value_count
        validation_loss = _evaluate_normalized_mse(
            model, validation_history, validation_target
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_normalized_mse": train_loss,
                "validation_normalized_mse": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(training["patience"]):
            break
    if best_state is None:
        raise RuntimeError("正式训练没有产生可用 checkpoint。")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_normalized_mse": float(best_validation_loss),
        "train_seconds": float(time.perf_counter() - started),
        "history": history,
    }


def predict_mlp(
    model: nn.Module,
    history: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> np.ndarray:
    normalized = (
        history - normalization["feature_mean"][None, None, :]
    ) / normalization["feature_std"][None, None, :]
    predictions = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(normalized.astype(np.float32))),
        batch_size=512,
    )
    with torch.inference_mode():
        for (inputs,) in loader:
            predictions.append(model(inputs).cpu().numpy())
    normalized_prediction = np.concatenate(predictions)
    return (
        normalized_prediction * normalization["target_std"][None, None, :]
        + normalization["target_mean"][None, None, :]
    )


def zoh_predictions(episode: EpisodeData) -> np.ndarray:
    windows = episode.windows
    raw = episode.raw
    anchor_indices = windows["anchor_raw_index"].astype(np.int64)
    rotation_heading_world = rotation_z(-windows["heading_yaw_world"])
    current_acc = np.einsum(
        "nij,nj->ni",
        rotation_heading_world,
        raw["torso_linear_acceleration_world"][anchor_indices],
    )
    current_alpha = np.einsum(
        "nij,nj->ni",
        rotation_heading_world,
        raw["torso_angular_acceleration_world"][anchor_indices],
    )
    current = np.concatenate((current_acc, current_alpha), axis=1)
    return np.repeat(current[:, None, :], 9, axis=1)


def template_predictions(episode: EpisodeData, robot_config: dict) -> np.ndarray:
    raw = episode.raw
    windows = episode.windows
    control_dt = float(np.asarray(windows["control_dt"]).item())
    simulation_dt = float(np.asarray(raw["simulation_dt"]).item())
    stride = int(round(control_dt / simulation_dt))
    predictor = TemplateDisturbancePredictor(
        template_dir=str(
            REPO_DIR
            / robot_config.get(
                "mpc_disturbance_template_dir",
                "disturbance_model_new_heading/templates_heading_interval",
            )
        ),
        variant=str(robot_config.get("mpc_disturbance_template", "raw")),
        control_dt=control_dt,
        horizon=9,
        acc_limit=float(robot_config.get("ddq_torso_acc_limit", 30.0)),
        alpha_limit=float(robot_config.get("ddq_torso_alpha_limit", 40.0)),
        slow_bias_enabled=bool(
            robot_config.get("mpc_disturbance_slow_bias_enabled", True)
        ),
        slow_bias_time_constant=float(
            robot_config.get("mpc_disturbance_slow_bias_time_constant", 0.4)
        ),
    )
    positions = {
        int(anchor): position
        for position, anchor in enumerate(windows["anchor_raw_index"])
    }
    result = np.empty_like(windows["target"], dtype=np.float64)
    filled = np.zeros(len(result), dtype=bool)
    for anchor in range(0, len(raw["time"]), stride):
        measured = DisturbanceInput(
            acc_world=raw["torso_linear_acceleration_world"][anchor],
            omega_world=raw["torso_angular_velocity_world"][anchor],
            alpha_world=raw["torso_angular_acceleration_world"][anchor],
            rot_world_body=raw["torso_rotation_world"][anchor],
        )
        predictor.update(
            DisturbancePredictorObservation(
                simulation_time=float(raw["time"][anchor]),
                measured_disturbance=measured,
            )
        )
        preview = predictor.predict(9, control_dt)
        position = positions.get(anchor)
        if position is None:
            continue
        rotation_heading_world = rotation_z(
            -float(windows["heading_yaw_world"][position])
        )
        acc_world = np.stack(
            [interval.acc_world for interval in preview.intervals]
        )
        alpha_world = np.stack(
            [interval.alpha_world for interval in preview.intervals]
        )
        result[position, :, :3] = np.einsum(
            "ij,nj->ni", rotation_heading_world, acc_world
        )
        result[position, :, 3:] = np.einsum(
            "ij,nj->ni", rotation_heading_world, alpha_world
        )
        filled[position] = True
    if not np.all(filled):
        raise RuntimeError("template baseline 未覆盖全部 dataset anchors。")
    return result.astype(np.float32)


def prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    segment_ids: np.ndarray,
) -> dict[str, object]:
    def values(mask: np.ndarray) -> dict[str, float | int]:
        error = prediction[mask] - target[mask]
        acc_rmse = float(np.sqrt(np.mean(error[:, :, :3] ** 2)))
        alpha_rmse = float(np.sqrt(np.mean(error[:, :, 3:] ** 2)))
        acc_target_rms = float(np.sqrt(np.mean(target[mask, :, :3] ** 2)))
        alpha_target_rms = float(np.sqrt(np.mean(target[mask, :, 3:] ** 2)))
        return {
            "sample_count": int(np.count_nonzero(mask)),
            "acc_rmse": acc_rmse,
            "alpha_rmse": alpha_rmse,
            "acc_target_rms": acc_target_rms,
            "alpha_target_rms": alpha_target_rms,
            "acc_relative_rmse": acc_rmse / max(acc_target_rms, 1e-12),
            "alpha_relative_rmse": alpha_rmse
            / max(alpha_target_rms, 1e-12),
        }

    all_samples = np.ones(len(target), dtype=bool)
    by_stage = {
        stage_name: values(segment_ids == segment_id)
        for segment_id, stage_name in STAGE_NAMES.items()
    }
    return {"overall": values(all_samples), "by_stage": by_stage}


def benchmark_inference(
    model: nn.Module,
    sample_history: np.ndarray,
    normalization: dict[str, np.ndarray],
    timing_config: dict,
) -> dict[str, object]:
    torch.set_num_threads(int(timing_config["torch_num_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model.eval()
    sample = torch.from_numpy(sample_history[None].astype(np.float32))
    feature_mean = torch.from_numpy(normalization["feature_mean"])
    feature_std = torch.from_numpy(normalization["feature_std"])
    target_mean = torch.from_numpy(normalization["target_mean"])
    target_std = torch.from_numpy(normalization["target_std"])

    def inference() -> torch.Tensor:
        normalized = (sample - feature_mean) / feature_std
        return model(normalized) * target_std + target_mean

    warmup = int(timing_config["warmup_iterations"])
    iterations = int(timing_config["benchmark_iterations"])
    repeats = int(timing_config.get("benchmark_repeats", 1))
    with torch.inference_mode():
        for _ in range(warmup):
            inference()
        timings = np.empty((repeats, iterations), dtype=np.float64)
        for repeat in range(repeats):
            for index in range(iterations):
                started = time.perf_counter_ns()
                inference()
                timings[repeat, index] = (
                    time.perf_counter_ns() - started
                ) * 1e-6
    flat_timings = timings.reshape(-1)
    return {
        "scope": "batch1_feature_normalization_mlp_target_denormalization",
        "warmup_iterations": warmup,
        "benchmark_iterations": iterations,
        "benchmark_repeats": repeats,
        "total_measured_inferences": int(flat_timings.size),
        "torch_num_threads": int(timing_config["torch_num_threads"]),
        "mean_ms": float(np.mean(flat_timings)),
        "p95_ms": float(np.percentile(flat_timings, 95)),
        "p99_ms": float(np.percentile(flat_timings, 99)),
        "max_ms": float(np.max(flat_timings)),
        "repeat_max_ms": np.max(timings, axis=1).tolist(),
    }


def _jsonable_normalization(
    normalization: dict[str, np.ndarray]
) -> dict[str, list[float]]:
    return {name: value.tolist() for name, value in normalization.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train B2 MLP baseline")
    parser.add_argument(
        "--experiment-config", default=str(DEFAULT_EXPERIMENT_CONFIG)
    )
    parser.add_argument("--robot-config", default="g1.yaml")
    args = parser.parse_args()

    experiment_path = Path(args.experiment_config).expanduser().resolve()
    with experiment_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    robot_config_path = REPO_DIR / "configs" / args.robot_config
    with robot_config_path.open("r", encoding="utf-8") as file:
        robot_config = yaml.safe_load(file)
    data_dir = _repo_path(config["collection"]["output_dir"])
    episodes = load_episodes(data_dir)
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
    normalization = compute_normalization(arrays["train"][0], arrays["train"][1])
    normalized = {
        name: _normalized_arrays(values[0], values[1], normalization)
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

    mlp_split_metrics = {}
    for name in ("train", "validation", "test"):
        prediction = predict_mlp(model, arrays[name][0], normalization)
        mlp_split_metrics[name] = prediction_metrics(
            prediction, arrays[name][1], arrays[name][2]
        )

    test_episodes = _select_episodes(episodes, split["test"])
    test_target = np.concatenate(
        [episode.windows["target"] for episode in test_episodes]
    )
    test_segments = np.concatenate(
        [episode.windows["anchor_schedule_segment_id"] for episode in test_episodes]
    )
    zoh = np.concatenate([zoh_predictions(episode) for episode in test_episodes])
    template = np.concatenate(
        [template_predictions(episode, robot_config) for episode in test_episodes]
    )
    mlp_test = predict_mlp(model, arrays["test"][0], normalization)
    comparison = {
        "zoh": prediction_metrics(zoh, test_target, test_segments),
        "template": prediction_metrics(template, test_target, test_segments),
        "mlp": prediction_metrics(mlp_test, test_target, test_segments),
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
    checkpoint_path = artifact_dir / "mlp_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history_steps": int(arrays["train"][0].shape[1]),
            "feature_dim": int(arrays["train"][0].shape[2]),
            "hidden_sizes": list(config["model"]["hidden_sizes"]),
            "horizon": 9,
            "target_dim": 6,
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "normalization": normalization,
            "episode_split": split,
        },
        checkpoint_path,
    )
    reloaded_model, reloaded_normalization = load_mlp_checkpoint(checkpoint_path)
    roundtrip_history, _ = _normalized_arrays(
        arrays["test"][0][:1], arrays["test"][1][:1], reloaded_normalization
    )
    with torch.inference_mode():
        reference = model(torch.from_numpy(roundtrip_history)).numpy()
        reloaded = reloaded_model(torch.from_numpy(roundtrip_history)).numpy()
    checkpoint_roundtrip_error = float(np.max(np.abs(reference - reloaded)))
    if checkpoint_roundtrip_error > 1e-7:
        raise RuntimeError("saved MLP checkpoint reload parity 失败。")

    checkpoint_relative = str(checkpoint_path.relative_to(REPO_DIR))
    full_summary = {
        "stage": "B2_mlp_baseline",
        "data": {
            "episode_count": len(episodes),
            "total_sample_count": int(
                sum(len(episode.windows["target"]) for episode in episodes)
            ),
            "history_shape_per_sample": list(arrays["train"][0].shape[1:]),
            "target_shape_per_sample": list(arrays["train"][1].shape[1:]),
            "split_episode_ids": split,
            "split_sample_counts": {
                name: int(len(values[0])) for name, values in arrays.items()
            },
        },
        "model": {
            "type": "flatten_mlp_one_shot",
            "hidden_sizes": list(config["model"]["hidden_sizes"]),
            "parameter_count": parameter_count(model),
            "output_shape": [9, 6],
        },
        "normalization": _jsonable_normalization(normalization),
        "overfit_sanity": overfit,
        "training": training,
        "mlp_split_metrics": mlp_split_metrics,
        "test_comparison": comparison,
        "cpu_inference": timing,
        "checkpoint_path_local_ignored": checkpoint_relative,
        "checkpoint_reload_max_error": checkpoint_roundtrip_error,
    }
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(full_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    light_summary = {
        "stage": full_summary["stage"],
        "data": full_summary["data"],
        "model": full_summary["model"],
        "normalization": {
            "fit_split": "train_episodes_only",
            "feature_channel_count": len(FEATURE_NAMES),
            "target_channel_count": len(TARGET_NAMES),
            "stored_with_local_checkpoint": True,
        },
        "overfit_sanity": full_summary["overfit_sanity"],
        "training": {
            name: value
            for name, value in full_summary["training"].items()
            if name != "history"
        },
        "mlp_split_metrics": full_summary["mlp_split_metrics"],
        "test_comparison": full_summary["test_comparison"],
        "cpu_inference": full_summary["cpu_inference"],
        "checkpoint_path_local_ignored": checkpoint_relative,
        "checkpoint_reload_max_error": checkpoint_roundtrip_error,
    }
    summary_path = summary_dir / "summary.json"
    summary_path.write_text(
        json.dumps(light_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(light_summary, indent=2, sort_keys=True))
    print(f"checkpoint: {checkpoint_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
