#!/usr/bin/env python3
"""C++ C ABI 与 Python Pinocchio 右臂 RNEA 的逐拍一致性和耗时测试。"""

import argparse
import ctypes
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from robot_model_backend import PinocchioPredictionBackend  # noqa: E402


RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)
JOINT_COUNT = len(RIGHT_ARM_JOINT_NAMES)
KINEMATICS_MAX_STATES = 32
DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
UINT8_PTR = ctypes.POINTER(ctypes.c_uint8)


class COutput(ctypes.Structure):
    _fields_ = [
        ("tau_rnea", ctypes.c_double * JOINT_COUNT),
        ("tau_constraint_friction", ctypes.c_double * JOINT_COUNT),
        ("tau_ff", ctypes.c_double * JOINT_COUNT),
        ("core_elapsed_ns", ctypes.c_uint64),
        ("rnea_elapsed_ns", ctypes.c_uint64),
    ]


class CKinematicsBatchOutput(ctypes.Structure):
    _fields_ = [
        ("state_count", ctypes.c_int32),
        ("ee_position_world", ctypes.c_double * (KINEMATICS_MAX_STATES * 3)),
        ("ee_rotation_world", ctypes.c_double * (KINEMATICS_MAX_STATES * 9)),
        ("imu_position_world", ctypes.c_double * (KINEMATICS_MAX_STATES * 3)),
        ("imu_rotation_world", ctypes.c_double * (KINEMATICS_MAX_STATES * 9)),
        (
            "J_v_world",
            ctypes.c_double * (KINEMATICS_MAX_STATES * 3 * JOINT_COUNT),
        ),
        (
            "J_w_world",
            ctypes.c_double * (KINEMATICS_MAX_STATES * 3 * JOINT_COUNT),
        ),
        (
            "dJ_v_world",
            ctypes.c_double * (KINEMATICS_MAX_STATES * 3 * JOINT_COUNT),
        ),
        (
            "dJ_w_world",
            ctypes.c_double * (KINEMATICS_MAX_STATES * 3 * JOINT_COUNT),
        ),
        ("core_elapsed_ns", ctypes.c_uint64),
    ]


class NativeRnea:
    """【半核心】只负责 ctypes 类型边界和 handle 生命周期。"""

    def __init__(self, library_path, scene_path):
        self.handle = None
        self.library = ctypes.CDLL(str(library_path))
        self._configure_signatures()
        if int(self.library.right_arm_rnea_abi_version()) != 3:
            raise RuntimeError("right_arm_rnea C ABI 版本不兼容")
        error = ctypes.create_string_buffer(1024)
        self.handle = self.library.right_arm_rnea_create(
            str(scene_path).encode(), error, len(error)
        )
        if not self.handle:
            raise RuntimeError(error.value.decode(errors="replace"))
        self.nq = int(self.library.right_arm_rnea_mujoco_nq(self.handle))
        self.nv = int(self.library.right_arm_rnea_mujoco_nv(self.handle))

    def assert_c_abi_rejects_bad_qpos_count(
        self, qpos, qvel, reference_qacc, ddq, passive, friction, timestep, breakaway
    ):
        """直接越过 Python shape 检查，确认 C ABI 自己拒绝错误维度。"""
        qpos = np.ascontiguousarray(qpos, dtype=np.float64)
        qvel = np.ascontiguousarray(qvel, dtype=np.float64)
        reference_qacc = np.ascontiguousarray(reference_qacc, dtype=np.float64)
        ddq = np.ascontiguousarray(ddq, dtype=np.float64)
        passive = np.ascontiguousarray(passive, dtype=np.float64)
        friction = np.ascontiguousarray(friction, dtype=np.float64)
        output = COutput()
        error = ctypes.create_string_buffer(1024)
        status = self.library.right_arm_rnea_compute(
            self.handle,
            qpos.ctypes.data_as(DOUBLE_PTR),
            qpos.size - 1,
            qvel.ctypes.data_as(DOUBLE_PTR),
            qvel.size,
            reference_qacc.ctypes.data_as(DOUBLE_PTR),
            reference_qacc.size,
            ddq.ctypes.data_as(DOUBLE_PTR),
            ddq.size,
            passive.ctypes.data_as(DOUBLE_PTR),
            passive.size,
            friction.ctypes.data_as(DOUBLE_PTR),
            friction.size,
            float(timestep),
            float(breakaway),
            ctypes.byref(output),
            error,
            len(error),
        )
        if status != 2:
            raise AssertionError(
                f"C ABI 应返回 dimension_mismatch=2，实际 {status}: "
                f"{error.value.decode(errors='replace')}"
            )

    def _configure_signatures(self):
        lib = self.library
        lib.right_arm_rnea_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.right_arm_rnea_create.restype = ctypes.c_void_p
        lib.right_arm_rnea_destroy.argtypes = [ctypes.c_void_p]
        lib.right_arm_rnea_destroy.restype = None
        lib.right_arm_rnea_mujoco_nq.argtypes = [ctypes.c_void_p]
        lib.right_arm_rnea_mujoco_nq.restype = ctypes.c_size_t
        lib.right_arm_rnea_mujoco_nv.argtypes = [ctypes.c_void_p]
        lib.right_arm_rnea_mujoco_nv.restype = ctypes.c_size_t
        lib.right_arm_rnea_compute.argtypes = [
            ctypes.c_void_p,
            DOUBLE_PTR,
            ctypes.c_size_t,
            DOUBLE_PTR,
            ctypes.c_size_t,
            DOUBLE_PTR,
            ctypes.c_size_t,
            DOUBLE_PTR,
            ctypes.c_size_t,
            DOUBLE_PTR,
            ctypes.c_size_t,
            DOUBLE_PTR,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(COutput),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.right_arm_rnea_compute.restype = ctypes.c_int
        lib.right_arm_kinematics_batch_compute.argtypes = [
            ctypes.c_void_p,
            DOUBLE_PTR,
            ctypes.c_size_t,
            DOUBLE_PTR,
            DOUBLE_PTR,
            UINT8_PTR,
            ctypes.c_size_t,
            ctypes.POINTER(CKinematicsBatchOutput),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.right_arm_kinematics_batch_compute.restype = ctypes.c_int
        lib.right_arm_rnea_abi_version.argtypes = []
        lib.right_arm_rnea_abi_version.restype = ctypes.c_uint32

    @staticmethod
    def _array(values, expected_shape):
        result = np.ascontiguousarray(values, dtype=np.float64)
        if result.shape != expected_shape:
            raise ValueError(f"输入维度应为 {expected_shape}，实际 {result.shape}")
        return result

    def compute(
        self, qpos, qvel, reference_qacc, ddq, passive, friction, timestep, breakaway
    ):
        qpos = self._array(qpos, (self.nq,))
        qvel = self._array(qvel, (self.nv,))
        reference_qacc = self._array(reference_qacc, (self.nv,))
        ddq = self._array(ddq, (JOINT_COUNT,))
        passive = self._array(passive, (JOINT_COUNT,))
        friction = self._array(friction, (JOINT_COUNT,))
        output = COutput()
        error = ctypes.create_string_buffer(1024)
        status = self.library.right_arm_rnea_compute(
            self.handle,
            qpos.ctypes.data_as(DOUBLE_PTR),
            qpos.size,
            qvel.ctypes.data_as(DOUBLE_PTR),
            qvel.size,
            reference_qacc.ctypes.data_as(DOUBLE_PTR),
            reference_qacc.size,
            ddq.ctypes.data_as(DOUBLE_PTR),
            ddq.size,
            passive.ctypes.data_as(DOUBLE_PTR),
            passive.size,
            friction.ctypes.data_as(DOUBLE_PTR),
            friction.size,
            float(timestep),
            float(breakaway),
            ctypes.byref(output),
            error,
            len(error),
        )
        if status != 0:
            raise RuntimeError(
                f"C++ RNEA status={status}: "
                f"{error.value.decode(errors='replace')}"
            )
        return {
            "tau_rnea": np.asarray(
                output.tau_rnea, dtype=np.float64
            ).copy(),
            "tau_constraint_friction": np.asarray(
                output.tau_constraint_friction, dtype=np.float64
            ).copy(),
            "tau_ff": np.asarray(output.tau_ff, dtype=np.float64).copy(),
            "core_elapsed_ns": int(output.core_elapsed_ns),
            "rnea_elapsed_ns": int(output.rnea_elapsed_ns),
        }

    def compute_kinematics_batch(
        self, qpos_reference, q_arm, dq_arm, acceleration_required
    ):
        qpos = self._array(qpos_reference, (self.nq,))
        q_arm = np.ascontiguousarray(q_arm, dtype=np.float64)
        dq_arm = np.ascontiguousarray(dq_arm, dtype=np.float64)
        acceleration_required = np.ascontiguousarray(
            acceleration_required, dtype=np.uint8
        )
        if q_arm.ndim != 2 or q_arm.shape[1:] != (JOINT_COUNT,):
            raise ValueError("q_arm 必须为 state_count x 5")
        if dq_arm.shape != q_arm.shape:
            raise ValueError("dq_arm 维度必须与 q_arm 相同")
        if acceleration_required.shape != (q_arm.shape[0],):
            raise ValueError("acceleration_required 维度必须为 state_count")

        output = CKinematicsBatchOutput()
        error = ctypes.create_string_buffer(1024)
        status = self.library.right_arm_kinematics_batch_compute(
            self.handle,
            qpos.ctypes.data_as(DOUBLE_PTR),
            qpos.size,
            q_arm.ctypes.data_as(DOUBLE_PTR),
            dq_arm.ctypes.data_as(DOUBLE_PTR),
            acceleration_required.ctypes.data_as(UINT8_PTR),
            q_arm.shape[0],
            ctypes.byref(output),
            error,
            len(error),
        )
        if status != 0:
            raise RuntimeError(
                f"C++ batch kinematics status={status}: "
                f"{error.value.decode(errors='replace')}"
            )
        count = int(output.state_count)

        def array(name, tail_shape):
            source = np.ctypeslib.as_array(getattr(output, name))
            return source[: count * int(np.prod(tail_shape))].reshape(
                (count, *tail_shape)
            ).copy()

        return {
            "ee_position_world": array("ee_position_world", (3,)),
            "ee_rotation_world": array("ee_rotation_world", (3, 3)),
            "imu_position_world": array("imu_position_world", (3,)),
            "imu_rotation_world": array("imu_rotation_world", (3, 3)),
            "J_v_world": array("J_v_world", (3, JOINT_COUNT)),
            "J_w_world": array("J_w_world", (3, JOINT_COUNT)),
            "dJ_v_world": array("dJ_v_world", (3, JOINT_COUNT)),
            "dJ_w_world": array("dJ_w_world", (3, JOINT_COUNT)),
            "core_elapsed_ns": int(output.core_elapsed_ns),
        }

    def assert_batch_c_abi_rejects_invalid_inputs(
        self, qpos_reference, q_arm, dq_arm, acceleration_required
    ):
        """确认批接口自己拒绝越界节点数和非 0/1 标志。"""
        qpos = np.ascontiguousarray(qpos_reference, dtype=np.float64)
        q_arm = np.ascontiguousarray(q_arm, dtype=np.float64)
        dq_arm = np.ascontiguousarray(dq_arm, dtype=np.float64)
        flags = np.ascontiguousarray(acceleration_required, dtype=np.uint8)
        output = CKinematicsBatchOutput()
        error = ctypes.create_string_buffer(1024)

        status = self.library.right_arm_kinematics_batch_compute(
            self.handle,
            qpos.ctypes.data_as(DOUBLE_PTR),
            qpos.size,
            q_arm.ctypes.data_as(DOUBLE_PTR),
            dq_arm.ctypes.data_as(DOUBLE_PTR),
            flags.ctypes.data_as(UINT8_PTR),
            0,
            ctypes.byref(output),
            error,
            len(error),
        )
        if status != 2:
            raise AssertionError(
                f"零节点应返回 dimension_mismatch=2，实际 {status}: "
                f"{error.value.decode(errors='replace')}"
            )

        bad_flags = flags.copy()
        bad_flags[0] = 2
        status = self.library.right_arm_kinematics_batch_compute(
            self.handle,
            qpos.ctypes.data_as(DOUBLE_PTR),
            qpos.size,
            q_arm.ctypes.data_as(DOUBLE_PTR),
            dq_arm.ctypes.data_as(DOUBLE_PTR),
            bad_flags.ctypes.data_as(UINT8_PTR),
            q_arm.shape[0],
            ctypes.byref(output),
            error,
            len(error),
        )
        if status != 1:
            raise AssertionError(
                f"非法 acceleration flag 应返回 invalid_argument=1，实际 "
                f"{status}: {error.value.decode(errors='replace')}"
            )

    def close(self):
        if self.handle:
            self.library.right_arm_rnea_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def _random_unit_quaternion_wxyz(rng):
    value = rng.normal(size=4)
    value /= np.linalg.norm(value)
    return value


def _stats_us(samples):
    values = np.asarray(samples, dtype=np.float64) * 1e6
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def _print_stats(name, values):
    stats = _stats_us(values)
    print(
        f"  {name:28s}: mean={stats['mean']:.2f}, "
        f"p95={stats['p95']:.2f}, p99={stats['p99']:.2f}, "
        f"max={stats['max']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--scene",
        default=str(REPO_DIR / "resources/g1_description/scene.xml"),
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--friction-breakaway-steps", type=float, default=5.0)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        raise SystemExit("samples/repeats 必须大于零")

    scene = Path(args.scene).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    backend = PinocchioPredictionBackend(
        model, str(scene), RIGHT_ARM_JOINT_NAMES
    )
    native = NativeRnea(Path(args.library).resolve(), scene)
    if native.nq != model.nq or native.nv != model.nv:
        raise SystemExit(
            f"C++ 模型 nq/nv={native.nq}/{native.nv}，"
            f"MuJoCo={model.nq}/{model.nv}"
        )

    arm_v = backend.mj_arm_v_indices
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in RIGHT_ARM_JOINT_NAMES
        ],
        dtype=np.int32,
    )
    rng = np.random.default_rng(args.seed)
    cases = []
    max_rnea_error = 0.0
    max_friction_error = 0.0
    max_feedforward_error = 0.0

    # 【核心一致性测试】随机化整机标量关节、浮动基姿态/速度和右臂 DDQ。
    for sample_index in range(args.samples):
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        for joint_id in range(model.njnt):
            joint_type = int(model.jnt_type[joint_id])
            if joint_type not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                continue
            q_index = int(model.jnt_qposadr[joint_id])
            if model.jnt_limited[joint_id]:
                lower, upper = model.jnt_range[joint_id]
                data.qpos[q_index] = rng.uniform(lower, upper)
            else:
                data.qpos[q_index] = rng.uniform(-0.5, 0.5)
        data.qpos[:3] += rng.uniform(-0.5, 0.5, size=3)
        data.qpos[3:7] = _random_unit_quaternion_wxyz(rng)
        data.qvel[:] = rng.uniform(-1.0, 1.0, size=model.nv)
        reference_qacc = rng.uniform(-20.0, 20.0, size=model.nv)
        desired = rng.uniform(-8.0, 8.0, size=JOINT_COUNT)
        if sample_index == 0:
            # 显式覆盖零速 breakaway 分支，不能只依赖随机样本碰巧接近零。
            data.qvel[arm_v] = 0.0

        # qfrc_passive 是生成名义 tau_ff 所需的当前 MuJoCo 被动力输入。
        mujoco.mj_forward(model, data)
        passive = data.qfrc_passive[arm_v].copy()
        friction_loss = model.dof_frictionloss[arm_v].copy()
        python_tau = backend.compute_right_arm_rnea(
            data.qpos,
            data.qvel,
            desired,
            reference_qacc=reference_qacc,
        )
        breakaway_velocity = (
            args.friction_breakaway_steps
            * float(model.opt.timestep)
            * np.abs(desired)
        )
        friction_direction = np.where(
            np.abs(data.qvel[arm_v]) < breakaway_velocity,
            np.sign(desired),
            np.sign(data.qvel[arm_v]),
        )
        python_friction = -friction_loss * friction_direction
        python_feedforward = python_tau - passive - python_friction
        candidate = native.compute(
            data.qpos,
            data.qvel,
            reference_qacc,
            desired,
            passive,
            friction_loss,
            model.opt.timestep,
            args.friction_breakaway_steps,
        )
        max_rnea_error = max(
            max_rnea_error,
            float(np.max(np.abs(candidate["tau_rnea"] - python_tau))),
        )
        max_friction_error = max(
            max_friction_error,
            float(
                np.max(
                    np.abs(
                        candidate["tau_constraint_friction"]
                        - python_friction
                    )
                )
            ),
        )
        max_feedforward_error = max(
            max_feedforward_error,
            float(
                np.max(
                    np.abs(candidate["tau_ff"] - python_feedforward)
                )
            ),
        )
        cases.append(
            (
                data.qpos.copy(),
                data.qvel.copy(),
                reference_qacc.copy(),
                desired.copy(),
                passive,
                friction_loss,
            )
        )

    print("Python/C++ 随机状态一致性：")
    print(f"  tau_rnea 最大绝对误差: {max_rnea_error:.3e} N m")
    print(f"  摩擦项最大绝对误差:   {max_friction_error:.3e} N m")
    print(f"  tau_ff 最大绝对误差:   {max_feedforward_error:.3e} N m")
    if max(max_rnea_error, max_friction_error, max_feedforward_error) > args.tolerance:
        raise SystemExit(
            f"C++ parity 失败：误差超过 {args.tolerance:.1e} N m"
        )

    # 【核心批处理一致性】同一个冻结整机状态下随机生成完整预测窗口，
    # 同时覆盖需要/不需要 Jacobian 导数的节点。
    batch_count = min(KINEMATICS_MAX_STATES, max(2, min(args.samples, 13)))
    qpos_reference = cases[0][0]
    q_arm = np.empty((batch_count, JOINT_COUNT), dtype=np.float64)
    for joint, joint_id in enumerate(joint_ids):
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            q_arm[:, joint] = rng.uniform(lower, upper, size=batch_count)
        else:
            q_arm[:, joint] = rng.uniform(-0.5, 0.5, size=batch_count)
    dq_arm = rng.uniform(-1.0, 1.0, size=(batch_count, JOINT_COUNT))
    acceleration_required = (np.arange(batch_count) % 2 == 0).astype(np.uint8)
    native_batch = native.compute_kinematics_batch(
        qpos_reference, q_arm, dq_arm, acceleration_required
    )
    reference_batch = {
        name: []
        for name in (
            "ee_position_world",
            "ee_rotation_world",
            "imu_position_world",
            "imu_rotation_world",
            "J_v_world",
            "J_w_world",
            "dJ_v_world",
            "dJ_w_world",
        )
    }
    for state in range(batch_count):
        reference = backend.evaluate(
            qpos_reference,
            q_arm[state],
            dq_arm[state],
            acceleration_required=bool(acceleration_required[state]),
        )
        for name in reference_batch:
            reference_batch[name].append(np.asarray(getattr(reference, name)))

    batch_errors = {}
    for name, values in reference_batch.items():
        expected = np.stack(values, axis=0)
        batch_errors[name] = float(
            np.max(np.abs(native_batch[name] - expected))
        )
    max_batch_error = max(batch_errors.values())
    print("\nPython/C++ 批运动学一致性：")
    print(f"  预测节点: {batch_count}")
    for name, error in batch_errors.items():
        print(f"  {name:22s}: max_abs={error:.3e}")
    if max_batch_error > args.tolerance:
        raise SystemExit(
            f"C++ batch kinematics parity 失败：最大误差 "
            f"{max_batch_error:.3e} > {args.tolerance:.1e}"
        )
    native.assert_batch_c_abi_rejects_invalid_inputs(
        qpos_reference, q_arm, dq_arm, acceleration_required
    )

    # 维度错误必须由 C ABI 自身拒绝，而不是越界读取。
    native.assert_c_abi_rejects_bad_qpos_count(
        *cases[0], model.opt.timestep, args.friction_breakaway_steps
    )

    python_wall = []
    native_wall = []
    native_core = []
    native_rnea = []
    for index in range(args.repeats):
        (
            qpos,
            qvel,
            reference_qacc,
            desired,
            passive,
            friction_loss,
        ) = cases[index % len(cases)]
        start = time.perf_counter_ns()
        backend.compute_right_arm_rnea(
            qpos,
            qvel,
            desired,
            reference_qacc=reference_qacc,
        )
        python_wall.append((time.perf_counter_ns() - start) * 1e-9)
        start = time.perf_counter_ns()
        result = native.compute(
            qpos,
            qvel,
            reference_qacc,
            desired,
            passive,
            friction_loss,
            model.opt.timestep,
            args.friction_breakaway_steps,
        )
        native_wall.append((time.perf_counter_ns() - start) * 1e-9)
        native_core.append(result["core_elapsed_ns"] * 1e-9)
        native_rnea.append(result["rnea_elapsed_ns"] * 1e-9)

    print(f"\nRNEA 基准（{args.repeats} 次，单位 us）：")
    _print_stats("Python Pinocchio backend wall", python_wall)
    _print_stats("C ABI + ctypes wall", native_wall)
    _print_stats("C++ mapping + RNEA core", native_core)
    _print_stats("C++ pinocchio::rnea only", native_rnea)
    native.close()


if __name__ == "__main__":
    main()
