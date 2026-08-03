"""MuJoCo/Pinocchio 整机 RNEA 一致性与核心耗时基准。"""

import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

# 允许直接执行 ``python robot_model_backend/validate_inverse_dynamics.py``。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_model_backend import PinocchioPredictionBackend  # noqa: E402


RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


def _percentiles(samples):
    values = np.asarray(samples, dtype=np.float64) * 1e6
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
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
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    backend = PinocchioPredictionBackend(
        model,
        args.scene,
        RIGHT_ARM_JOINT_NAMES,
    )
    arm_v = backend.mj_arm_v_indices
    arm_q = backend.mj_arm_q_indices
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in RIGHT_ARM_JOINT_NAMES
        ],
        dtype=np.int32,
    )
    joint_ranges = model.jnt_range[joint_ids]
    rng = np.random.default_rng(args.seed)
    cases = []
    max_abs_error = 0.0
    max_norm_error = 0.0

    # 【核心验证】MuJoCo 逆动力学的 inverse + passive +
    # constraint 还原 M*qacc+h，可与 Pinocchio 整机 RNEA 直接对比。
    for _ in range(args.samples):
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        data.qpos[arm_q] = rng.uniform(
            joint_ranges[:, 0], joint_ranges[:, 1]
        )
        yaw = rng.uniform(-np.pi, np.pi)
        data.qpos[3:7] = np.asarray(
            [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)]
        )
        data.qvel[:] = 0.0
        data.qvel[:6] = rng.uniform(-0.5, 0.5, size=6)
        data.qvel[arm_v] = rng.uniform(-1.0, 1.0, size=5)
        desired = rng.uniform(-8.0, 8.0, size=5)
        data.qacc[:] = 0.0
        data.qacc[arm_v] = desired
        mujoco.mj_inverse(model, data)
        reference = (
            data.qfrc_inverse[arm_v]
            + data.qfrc_passive[arm_v]
            + data.qfrc_constraint[arm_v]
        )
        candidate = backend.compute_right_arm_rnea(
            data.qpos, data.qvel, desired
        )
        error = candidate - reference
        max_abs_error = max(max_abs_error, float(np.max(np.abs(error))))
        max_norm_error = max(max_norm_error, float(np.linalg.norm(error)))
        cases.append((data, desired))

    print("整机 RNEA 一致性：")
    print(f"  单关节最大绝对误差: {max_abs_error:.3e} N m")
    print(f"  5 维最大误差范数:   {max_norm_error:.3e} N m")
    if max_abs_error > args.tolerance:
        raise SystemExit(
            f"RNEA 一致性失败：误差超过 {args.tolerance:.1e} N m"
        )

    # 分别测量库的核心调用；完整 DDQ 执行链仍以仿真统计为准。
    mujoco_times = []
    pinocchio_times = []
    for index in range(args.repeats):
        data, desired = cases[index % len(cases)]
        start = time.perf_counter()
        mujoco.mj_inverse(model, data)
        mujoco_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        backend.compute_right_arm_rnea(data.qpos, data.qvel, desired)
        pinocchio_times.append(time.perf_counter() - start)

    print(f"\n核心调用基准（{args.repeats} 次，单位 us）：")
    for name, samples in (
        ("MuJoCo mj_inverse", mujoco_times),
        ("Pinocchio rnea", pinocchio_times),
    ):
        stats = _percentiles(samples)
        print(
            f"  {name:20s}: mean={stats['mean']:.1f}, "
            f"p95={stats['p95']:.1f}, p99={stats['p99']:.1f}, "
            f"max={stats['max']:.1f}"
        )


if __name__ == "__main__":
    main()
