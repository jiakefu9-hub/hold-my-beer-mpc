"""MuJoCo/Pinocchio 预测运动学一致性与速度基准。"""

import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

# 允许直接执行 ``python robot_model_backend/validate_and_benchmark.py``。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_model_backend import (  # noqa: E402
    MujocoPredictionBackend,
    PinocchioPredictionBackend,
)


RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


def _rotation_error_angle(rotation_a, rotation_b):
    relative = rotation_a.T @ rotation_b
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def _max_errors(reference, candidate):
    return {
        "ee_position": float(
            np.max(
                np.abs(
                    reference.ee_position_world
                    - candidate.ee_position_world
                )
            )
        ),
        "ee_rotation_angle": _rotation_error_angle(
            reference.ee_rotation_world, candidate.ee_rotation_world
        ),
        "imu_position": float(
            np.max(
                np.abs(
                    reference.imu_position_world
                    - candidate.imu_position_world
                )
            )
        ),
        "imu_rotation_angle": _rotation_error_angle(
            reference.imu_rotation_world, candidate.imu_rotation_world
        ),
        "J_v": float(np.max(np.abs(reference.J_v_world - candidate.J_v_world))),
        "J_w": float(np.max(np.abs(reference.J_w_world - candidate.J_w_world))),
        "dJ_v": float(
            np.max(np.abs(reference.dJ_v_world - candidate.dJ_v_world))
        ),
        "dJ_w": float(
            np.max(np.abs(reference.dJ_w_world - candidate.dJ_w_world))
        ),
    }


def _benchmark(backend, cases, repeats):
    # 先热身，避免首次调用污染统计。
    for qpos, q, dq in cases:
        backend.evaluate(qpos, q, dq, acceleration_required=True)
    samples = []
    for index in range(repeats):
        qpos, q, dq = cases[index % len(cases)]
        start = time.perf_counter()
        backend.evaluate(qpos, q, dq, acceleration_required=True)
        samples.append((time.perf_counter() - start) * 1e6)
    values = np.asarray(samples)
    return {
        "mean_us": float(np.mean(values)),
        "p50_us": float(np.percentile(values, 50)),
        "p95_us": float(np.percentile(values, 95)),
        "p99_us": float(np.percentile(values, 99)),
        "max_us": float(np.max(values)),
    }


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        default=str(repo / "resources/g1_description/scene.xml"),
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    # arccos 对接近 1 的旋转余弦会放大浮点舍入；
    # 位置/Jacobian 实际仍保持在 1e-14 量级。
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    mujoco_backend = MujocoPredictionBackend(
        model, RIGHT_ARM_JOINT_NAMES
    )
    pinocchio_backend = PinocchioPredictionBackend(
        model, args.scene, RIGHT_ARM_JOINT_NAMES
    )
    rng = np.random.default_rng(args.seed)
    joint_ranges = model.jnt_range[mujoco_backend.joint_ids]
    cases = []
    max_errors = {
        name: 0.0
        for name in (
            "ee_position",
            "ee_rotation_angle",
            "imu_position",
            "imu_rotation_angle",
            "J_v",
            "J_w",
            "dJ_v",
            "dJ_w",
        )
    }
    for _ in range(args.samples):
        qpos = model.qpos0.copy()
        # 额外随机化 floating base，专门检查 wxyz -> xyzw 转换。
        yaw = rng.uniform(-np.pi, np.pi)
        qpos[:3] += rng.uniform(-0.2, 0.2, size=3)
        qpos[3:7] = np.array(
            [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)]
        )
        q = rng.uniform(joint_ranges[:, 0], joint_ranges[:, 1])
        dq = rng.uniform(-1.0, 1.0, size=len(RIGHT_ARM_JOINT_NAMES))
        reference = mujoco_backend.evaluate(qpos, q, dq)
        candidate = pinocchio_backend.evaluate(qpos, q, dq)
        errors = _max_errors(reference, candidate)
        for name, value in errors.items():
            max_errors[name] = max(max_errors[name], value)
        cases.append((qpos, q, dq))

    print("一致性最大误差:")
    for name, value in max_errors.items():
        print(f"  {name:20s}: {value:.3e}")
    if any(value > args.tolerance for value in max_errors.values()):
        raise SystemExit(
            f"一致性检查失败：误差超过 tolerance={args.tolerance:.1e}"
        )

    print(f"\n单预测节点基准（{args.repeats} 次，单位 us）:")
    for backend in (mujoco_backend, pinocchio_backend):
        stats = _benchmark(backend, cases, args.repeats)
        print(
            f"  {backend.backend_name:10s}: mean={stats['mean_us']:.1f}, "
            f"p50={stats['p50_us']:.1f}, p95={stats['p95_us']:.1f}, "
            f"p99={stats['p99_us']:.1f}, max={stats['max_us']:.1f}"
        )


if __name__ == "__main__":
    main()
