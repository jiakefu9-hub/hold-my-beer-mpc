#!/usr/bin/env python3
"""C++ DDQ→力矩映射与 Python 参考实现的一致性和耗时基准。"""

from __future__ import annotations

import argparse
import ctypes
import pathlib
import sys
import time
from dataclasses import dataclass

import mujoco
import numpy as np


ARM_DOF = 5
MAX_SCALES = 8
JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


class NativeState(ctypes.Structure):
    _fields_ = [
        ("time", ctypes.c_double),
        ("qpos", ctypes.POINTER(ctypes.c_double)),
        ("qpos_count", ctypes.c_int32),
        ("qvel", ctypes.POINTER(ctypes.c_double)),
        ("qvel_count", ctypes.c_int32),
        ("ctrl", ctypes.POINTER(ctypes.c_double)),
        ("ctrl_count", ctypes.c_int32),
        ("qacc_warmstart", ctypes.POINTER(ctypes.c_double)),
        ("qacc_warmstart_count", ctypes.c_int32),
        ("qfrc_applied", ctypes.POINTER(ctypes.c_double)),
        ("qfrc_applied_count", ctypes.c_int32),
        ("xfrc_applied", ctypes.POINTER(ctypes.c_double)),
        ("xfrc_applied_count", ctypes.c_int32),
    ]


class NativeRequest(ctypes.Structure):
    _fields_ = [
        ("desired_qacc", ctypes.c_double * ARM_DOF),
        ("tau_nominal", ctypes.c_double * ARM_DOF),
        ("has_previous_executed_tau", ctypes.c_int32),
        ("previous_executed_tau", ctypes.c_double * ARM_DOF),
    ]


class NativeParams(ctypes.Structure):
    _fields_ = [
        ("perturbation", ctypes.c_double),
        ("regularization", ctypes.c_double),
        ("validation_scales", ctypes.c_double * MAX_SCALES),
        ("validation_scale_count", ctypes.c_int32),
        ("second_pass_error_threshold", ctypes.c_double),
        ("max_joint_error", ctypes.c_double),
        ("max_abs_qacc", ctypes.c_double),
        ("enable_second_pass", ctypes.c_int32),
        ("max_safety_rescue_passes", ctypes.c_int32),
    ]


class NativeOutput(ctypes.Structure):
    _fields_ = [
        ("tau_cmd", ctypes.c_double * ARM_DOF),
        ("tau_nominal", ctypes.c_double * ARM_DOF),
        ("tau_correction_raw", ctypes.c_double * ARM_DOF),
        ("tau_correction", ctypes.c_double * ARM_DOF),
        ("tau_cmd_raw", ctypes.c_double * ARM_DOF),
        ("qacc_baseline", ctypes.c_double * ARM_DOF),
        ("qacc_predicted", ctypes.c_double * ARM_DOF),
        ("qacc_prediction_error", ctypes.c_double * ARM_DOF),
        ("qacc_validated", ctypes.c_double * ARM_DOF),
        ("qacc_validation_error", ctypes.c_double * ARM_DOF),
        ("qacc_linearization_error", ctypes.c_double * ARM_DOF),
        ("gain_matrix", ctypes.c_double * (ARM_DOF * ARM_DOF)),
        ("singular_values", ctypes.c_double * ARM_DOF),
        ("condition_number", ctypes.c_double),
        ("validation_scale", ctypes.c_double),
        ("validation_attempts", ctypes.c_int32),
        ("validation_improved", ctypes.c_int32),
        ("validation_tracking_safety_satisfied", ctypes.c_int32),
        ("validation_qacc_safety_satisfied", ctypes.c_int32),
        ("validation_safe_candidate_count", ctypes.c_int32),
        ("validation_total_error_rejections", ctypes.c_int32),
        ("validation_joint_error_rejections", ctypes.c_int32),
        ("validation_qacc_limit_rejections", ctypes.c_int32),
        ("first_pass_qacc_validated", ctypes.c_double * ARM_DOF),
        ("first_pass_qacc_validation_error", ctypes.c_double * ARM_DOF),
        ("second_pass_triggered", ctypes.c_int32),
        ("second_pass_accepted", ctypes.c_int32),
        ("second_pass_tau_correction_raw", ctypes.c_double * ARM_DOF),
        ("second_pass_tau_correction", ctypes.c_double * ARM_DOF),
        ("second_pass_qacc_predicted", ctypes.c_double * ARM_DOF),
        ("second_pass_qacc_validated", ctypes.c_double * ARM_DOF),
        ("second_pass_qacc_validation_error", ctypes.c_double * ARM_DOF),
        ("second_pass_qacc_linearization_error", ctypes.c_double * ARM_DOF),
        ("second_pass_gain_matrix", ctypes.c_double * (ARM_DOF * ARM_DOF)),
        ("second_pass_singular_values", ctypes.c_double * ARM_DOF),
        ("second_pass_condition_number", ctypes.c_double),
        ("second_pass_validation_scale", ctypes.c_double),
        ("second_pass_validation_attempts", ctypes.c_int32),
        ("second_pass_tracking_safety_satisfied", ctypes.c_int32),
        ("second_pass_qacc_safety_satisfied", ctypes.c_int32),
        ("second_pass_safe_candidate_count", ctypes.c_int32),
        ("second_pass_total_error_rejections", ctypes.c_int32),
        ("second_pass_joint_error_rejections", ctypes.c_int32),
        ("second_pass_qacc_limit_rejections", ctypes.c_int32),
        ("safety_fallback_used", ctypes.c_int32),
        ("safety_fallback_satisfied", ctypes.c_int32),
        ("safety_fallback_attempts", ctypes.c_int32),
        ("hold_last_safe_available", ctypes.c_int32),
        ("hold_last_safe_used", ctypes.c_int32),
        ("hold_last_safe_satisfied", ctypes.c_int32),
        ("hold_last_safe_qacc", ctypes.c_double * ARM_DOF),
        ("full_forward_calls", ctypes.c_int32),
        ("forward_skip_calls", ctypes.c_int32),
        ("validated_pass_count", ctypes.c_int32),
        ("baseline_elapsed_ns", ctypes.c_uint64),
        ("first_pass_elapsed_ns", ctypes.c_uint64),
        ("second_pass_elapsed_ns", ctypes.c_uint64),
        ("rescue_elapsed_ns", ctypes.c_uint64),
        ("hold_last_elapsed_ns", ctypes.c_uint64),
        ("total_elapsed_ns", ctypes.c_uint64),
    ]


def pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def native_array(values) -> np.ndarray:
    return np.ctypeslib.as_array(values).copy()


@dataclass
class Case:
    data: mujoco.MjData
    desired_qacc: np.ndarray
    tau_nominal: np.ndarray
    previous_tau: np.ndarray | None
    params: dict


class NativeMapper:
    def __init__(self, library_path: pathlib.Path, scene_path: pathlib.Path):
        self.library = ctypes.CDLL(str(library_path))
        self.library.ddq_torque_mapper_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        self.library.ddq_torque_mapper_create.restype = ctypes.c_void_p
        self.library.ddq_torque_mapper_destroy.argtypes = [ctypes.c_void_p]
        self.library.ddq_torque_mapper_default_params.argtypes = [
            ctypes.POINTER(NativeParams)
        ]
        self.library.ddq_torque_mapper_compute.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NativeState),
            ctypes.POINTER(NativeRequest),
            ctypes.POINTER(NativeParams),
            ctypes.POINTER(NativeOutput),
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        self.library.ddq_torque_mapper_compute.restype = ctypes.c_int32
        error = ctypes.create_string_buffer(2048)
        self.handle = self.library.ddq_torque_mapper_create(
            str(scene_path).encode(), error, len(error)
        )
        if not self.handle:
            raise RuntimeError(error.value.decode())

    def close(self):
        if self.handle:
            self.library.ddq_torque_mapper_destroy(self.handle)
            self.handle = None

    def compute(self, case: Case) -> NativeOutput:
        # 【非核心桥接】保证 C ABI 看到连续 double 数组，并在调用期间保活。
        qpos = np.ascontiguousarray(case.data.qpos, dtype=np.float64)
        qvel = np.ascontiguousarray(case.data.qvel, dtype=np.float64)
        ctrl = np.ascontiguousarray(case.data.ctrl, dtype=np.float64)
        warmstart = np.ascontiguousarray(case.data.qacc_warmstart, dtype=np.float64)
        qfrc = np.ascontiguousarray(case.data.qfrc_applied, dtype=np.float64)
        xfrc = np.ascontiguousarray(case.data.xfrc_applied, dtype=np.float64).reshape(-1)
        state = NativeState(
            float(case.data.time),
            pointer(qpos),
            qpos.size,
            pointer(qvel),
            qvel.size,
            pointer(ctrl),
            ctrl.size,
            pointer(warmstart),
            warmstart.size,
            pointer(qfrc),
            qfrc.size,
            pointer(xfrc),
            xfrc.size,
        )
        previous = (
            np.zeros(ARM_DOF, dtype=np.float64)
            if case.previous_tau is None
            else case.previous_tau
        )
        request = NativeRequest(
            (ctypes.c_double * ARM_DOF)(*case.desired_qacc),
            (ctypes.c_double * ARM_DOF)(*case.tau_nominal),
            int(case.previous_tau is not None),
            (ctypes.c_double * ARM_DOF)(*previous),
        )
        params = NativeParams()
        self.library.ddq_torque_mapper_default_params(ctypes.byref(params))
        params.perturbation = case.params["perturbation"]
        params.regularization = case.params["regularization"]
        scales = case.params["validation_scales"]
        params.validation_scale_count = len(scales)
        for index, scale in enumerate(scales):
            params.validation_scales[index] = scale
        params.second_pass_error_threshold = case.params[
            "second_pass_error_threshold"
        ]
        params.max_joint_error = case.params["max_joint_error"]
        params.max_abs_qacc = case.params["max_abs_qacc"]
        params.enable_second_pass = int(case.params["enable_second_pass"])
        params.max_safety_rescue_passes = case.params["max_safety_rescue_passes"]
        output = NativeOutput()
        error = ctypes.create_string_buffer(2048)
        status = self.library.ddq_torque_mapper_compute(
            self.handle,
            ctypes.byref(state),
            ctypes.byref(request),
            ctypes.byref(params),
            ctypes.byref(output),
            error,
            len(error),
        )
        if status != 0:
            raise RuntimeError(f"C++ compute status={status}: {error.value.decode()}")
        return output


def make_case(model: mujoco.MjModel, rng: np.random.Generator, index: int) -> Case:
    data = mujoco.MjData(model)
    data.time = 1.0 + index * model.opt.timestep
    data.qpos[:] = model.qpos0
    # 在可动关节附近随机取合法构型；浮动基保持单位四元数和正常高度。
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qpos_index = int(model.jnt_qposadr[joint_id])
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            center = 0.5 * (lower + upper)
            radius = min(0.15, 0.2 * (upper - lower))
            data.qpos[qpos_index] = np.clip(
                center + rng.uniform(-radius, radius), lower, upper
            )
        else:
            data.qpos[qpos_index] = rng.uniform(-0.1, 0.1)
    data.qvel[:] = rng.normal(0.0, 0.25, model.nv)
    data.ctrl[:] = rng.uniform(-4.0, 4.0, model.nu)
    data.qacc_warmstart[:] = rng.normal(0.0, 0.15, model.nv)
    data.qfrc_applied[:] = rng.normal(0.0, 0.08, model.nv)
    data.xfrc_applied[:] = rng.normal(0.0, 0.03, (model.nbody, 6))

    params = {
        "perturbation": 0.1,
        "regularization": 5.0,
        "validation_scales": (1.0, 0.5, 0.25, 0.125),
        "second_pass_error_threshold": 5.0,
        "max_joint_error": 4.0,
        "max_abs_qacc": 8.0,
        "enable_second_pass": True,
        "max_safety_rescue_passes": 2,
    }
    # 交替覆盖：普通严格验收、强制第二轮、救援/hold-last 尝试。
    if index % 3 == 1:
        params.update(
            second_pass_error_threshold=0.0,
            max_joint_error=100.0,
            max_abs_qacc=100.0,
        )
    elif index % 3 == 2:
        params.update(max_joint_error=1.0, max_abs_qacc=2.0)
    desired_qacc = rng.uniform(-7.0, 7.0, ARM_DOF)
    tau_nominal = rng.uniform(-20.0, 20.0, ARM_DOF)
    previous_tau = rng.uniform(-12.0, 12.0, ARM_DOF) if index % 2 else None
    return Case(data, desired_qacc, tau_nominal, previous_tau, params)


def python_compute(model, scratch, qvel_indices, ctrl_indices, limits, case, function):
    return function(
        model,
        case.data,
        scratch,
        case.data.ctrl.copy(),
        case.desired_qacc,
        case.tau_nominal,
        qvel_indices,
        ctrl_indices,
        limits,
        previous_executed_tau=case.previous_tau,
        **case.params,
    )


def check_close(label, actual, expected, max_errors, atol=3e-8, rtol=3e-8):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    finite = np.isfinite(actual) & np.isfinite(expected)
    error = (
        float(np.max(np.abs(actual[finite] - expected[finite])))
        if np.any(finite)
        else 0.0
    )
    max_errors[label] = max(max_errors.get(label, 0.0), error)
    if not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True):
        raise AssertionError(
            f"{label} 不一致: max_abs={error:.3e}\nC++={actual}\nPython={expected}"
        )


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64) * 1e6
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=pathlib.Path, required=True)
    parser.add_argument("--scene", type=pathlib.Path, required=True)
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    repo_dir = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_dir))
    from right_arm_runtime.cpp_ddq_mapper import CppDdqTorqueMapper
    from sim_support import local_forward_dynamics_torque_mapping

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    scratch = mujoco.MjData(model)
    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
    )
    qvel_indices = model.jnt_dofadr[joint_ids].astype(np.int32)
    ctrl_indices = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES],
        dtype=np.int32,
    )
    limits = model.jnt_actfrcrange[joint_ids].copy()
    rng = np.random.default_rng(args.seed)
    cases = [make_case(model, rng, index) for index in range(args.samples)]
    native = NativeMapper(args.library, args.scene)
    runtime = CppDdqTorqueMapper(args.scene, args.library)
    max_errors = {}
    runtime_max_errors = {}
    branch_fields = (
        "validation_attempts",
        "validation_improved",
        "validation_tracking_safety_satisfied",
        "validation_qacc_safety_satisfied",
        "validation_safe_candidate_count",
        "validation_total_error_rejections",
        "validation_joint_error_rejections",
        "validation_qacc_limit_rejections",
        "second_pass_triggered",
        "second_pass_accepted",
        "second_pass_validation_attempts",
        "second_pass_tracking_safety_satisfied",
        "second_pass_qacc_safety_satisfied",
        "second_pass_safe_candidate_count",
        "second_pass_total_error_rejections",
        "second_pass_joint_error_rejections",
        "second_pass_qacc_limit_rejections",
        "safety_fallback_used",
        "safety_fallback_satisfied",
        "safety_fallback_attempts",
        "hold_last_safe_available",
        "hold_last_safe_used",
        "hold_last_safe_satisfied",
    )
    coverage = {"second": 0, "rescue": 0, "hold_attempt": 0, "hold_used": 0}
    try:
        for index, case in enumerate(cases):
            python_tau, python_result = python_compute(
                model,
                scratch,
                qvel_indices,
                ctrl_indices,
                limits,
                case,
                local_forward_dynamics_torque_mapping,
            )
            cpp = native.compute(case)
            runtime_result = runtime.compute(
                data=case.data,
                fixed_ctrl=case.data.ctrl,
                desired_qacc=case.desired_qacc,
                tau_nominal=case.tau_nominal,
                previous_executed_tau=case.previous_tau,
                **case.params,
            )
            array_pairs = {
                "tau_cmd": (native_array(cpp.tau_cmd), python_tau),
                "tau_nominal": (native_array(cpp.tau_nominal), python_result.tau_nominal),
                "tau_correction_raw": (
                    native_array(cpp.tau_correction_raw),
                    python_result.tau_correction_raw,
                ),
                "tau_correction": (
                    native_array(cpp.tau_correction),
                    python_result.tau_correction,
                ),
                "tau_cmd_raw": (native_array(cpp.tau_cmd_raw), python_result.tau_cmd_raw),
                "qacc_baseline": (
                    native_array(cpp.qacc_baseline),
                    python_result.qacc_baseline,
                ),
                "qacc_predicted": (
                    native_array(cpp.qacc_predicted),
                    python_result.qacc_predicted,
                ),
                "qacc_prediction_error": (
                    native_array(cpp.qacc_prediction_error),
                    python_result.qacc_prediction_error,
                ),
                "qacc_validated": (
                    native_array(cpp.qacc_validated),
                    python_result.qacc_validated,
                ),
                "qacc_validation_error": (
                    native_array(cpp.qacc_validation_error),
                    python_result.qacc_validation_error,
                ),
                "qacc_linearization_error": (
                    native_array(cpp.qacc_linearization_error),
                    python_result.qacc_linearization_error,
                ),
                "gain_matrix": (
                    native_array(cpp.gain_matrix).reshape(ARM_DOF, ARM_DOF),
                    python_result.gain_matrix,
                ),
                "singular_values": (
                    native_array(cpp.singular_values),
                    python_result.singular_values,
                ),
                "hold_last_safe_qacc": (
                    native_array(cpp.hold_last_safe_qacc),
                    python_result.hold_last_safe_qacc,
                ),
                "first_pass_qacc_validated": (
                    native_array(cpp.first_pass_qacc_validated),
                    python_result.first_pass_qacc_validated,
                ),
                "first_pass_qacc_validation_error": (
                    native_array(cpp.first_pass_qacc_validation_error),
                    python_result.first_pass_qacc_validation_error,
                ),
                "second_pass_tau_correction_raw": (
                    native_array(cpp.second_pass_tau_correction_raw),
                    python_result.second_pass_tau_correction_raw,
                ),
                "second_pass_tau_correction": (
                    native_array(cpp.second_pass_tau_correction),
                    python_result.second_pass_tau_correction,
                ),
                "second_pass_qacc_predicted": (
                    native_array(cpp.second_pass_qacc_predicted),
                    python_result.second_pass_qacc_predicted,
                ),
                "second_pass_qacc_validated": (
                    native_array(cpp.second_pass_qacc_validated),
                    python_result.second_pass_qacc_validated,
                ),
                "second_pass_qacc_validation_error": (
                    native_array(cpp.second_pass_qacc_validation_error),
                    python_result.second_pass_qacc_validation_error,
                ),
                "second_pass_qacc_linearization_error": (
                    native_array(cpp.second_pass_qacc_linearization_error),
                    python_result.second_pass_qacc_linearization_error,
                ),
                "second_pass_gain_matrix": (
                    native_array(cpp.second_pass_gain_matrix).reshape(
                        ARM_DOF, ARM_DOF
                    ),
                    python_result.second_pass_gain_matrix,
                ),
                "second_pass_singular_values": (
                    native_array(cpp.second_pass_singular_values),
                    python_result.second_pass_singular_values,
                ),
            }
            for label, pair in array_pairs.items():
                check_close(label, *pair, max_errors)
                check_close(
                    label,
                    runtime_result.values[label],
                    pair[0],
                    runtime_max_errors,
                    atol=0.0,
                    rtol=0.0,
                )
            check_close(
                "condition_number",
                cpp.condition_number,
                python_result.condition_number,
                max_errors,
            )
            check_close(
                "condition_number",
                runtime_result.values["condition_number"],
                cpp.condition_number,
                runtime_max_errors,
                atol=0.0,
                rtol=0.0,
            )
            check_close(
                "validation_scale",
                cpp.validation_scale,
                python_result.validation_scale,
                max_errors,
            )
            check_close(
                "validation_scale",
                runtime_result.values["validation_scale"],
                cpp.validation_scale,
                runtime_max_errors,
                atol=0.0,
                rtol=0.0,
            )
            check_close(
                "second_pass_validation_scale",
                cpp.second_pass_validation_scale,
                python_result.second_pass_validation_scale,
                max_errors,
            )
            check_close(
                "second_pass_validation_scale",
                runtime_result.values["second_pass_validation_scale"],
                cpp.second_pass_validation_scale,
                runtime_max_errors,
                atol=0.0,
                rtol=0.0,
            )
            check_close(
                "second_pass_condition_number",
                cpp.second_pass_condition_number,
                python_result.second_pass_condition_number,
                max_errors,
            )
            check_close(
                "second_pass_condition_number",
                runtime_result.values["second_pass_condition_number"],
                cpp.second_pass_condition_number,
                runtime_max_errors,
                atol=0.0,
                rtol=0.0,
            )
            for field in branch_fields:
                cpp_value = int(getattr(cpp, field))
                python_value = int(bool(getattr(python_result, field))) if isinstance(
                    getattr(python_result, field), (bool, np.bool_)
                ) else int(getattr(python_result, field))
                if cpp_value != python_value:
                    raise AssertionError(
                        f"sample={index} 分支字段 {field}: C++={cpp_value}, Python={python_value}"
                    )
                runtime_value = int(runtime_result.values[field])
                if runtime_value != cpp_value:
                    raise AssertionError(
                        f"sample={index} 生产适配层分支 {field}: "
                        f"runtime={runtime_value}, C ABI={cpp_value}"
                    )
            if cpp.full_forward_calls != 1 or cpp.forward_skip_calls < 5:
                raise AssertionError("MuJoCo 调用计数不符合一次基线加五次扰动的下界。")
            if (
                runtime_result.full_forward_calls != cpp.full_forward_calls
                or runtime_result.forward_skip_calls != cpp.forward_skip_calls
                or runtime_result.validated_pass_count != cpp.validated_pass_count
            ):
                raise AssertionError(
                    f"sample={index} 生产适配层 MuJoCo 调用计数与 C ABI 不一致。"
                )
            coverage["second"] += int(bool(cpp.second_pass_triggered))
            coverage["rescue"] += int(bool(cpp.safety_fallback_used))
            coverage["hold_attempt"] += int(bool(cpp.hold_last_safe_available))
            coverage["hold_used"] += int(bool(cpp.hold_last_safe_used))

        print("数值一致性通过")
        print(f"  随机状态: {args.samples}")
        for label, error in sorted(max_errors.items()):
            print(f"  {label}: max_abs={error:.3e}")
        print(
            "  分支覆盖: "
            f"second={coverage['second']}, rescue={coverage['rescue']}, "
            f"hold_attempt={coverage['hold_attempt']}, hold_used={coverage['hold_used']}"
        )
        print(
            "  生产 ctypes 适配层与原始 C ABI: "
            f"max_abs={max(runtime_max_errors.values(), default=0.0):.3e}"
        )

        benchmark_case = cases[1 if len(cases) > 1 else 0]
        python_times = []
        native_times = []
        native_core_times = []
        runtime_times = []
        runtime_core_times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            python_compute(
                model,
                scratch,
                qvel_indices,
                ctrl_indices,
                limits,
                benchmark_case,
                local_forward_dynamics_torque_mapping,
            )
            python_times.append(time.perf_counter() - start)
        for _ in range(args.repeats):
            start = time.perf_counter()
            output = native.compute(benchmark_case)
            native_times.append(time.perf_counter() - start)
            native_core_times.append(output.total_elapsed_ns * 1e-9)
        runtime_kwargs = {
            "data": benchmark_case.data,
            "fixed_ctrl": benchmark_case.data.ctrl,
            "desired_qacc": benchmark_case.desired_qacc,
            "tau_nominal": benchmark_case.tau_nominal,
            "previous_executed_tau": benchmark_case.previous_tau,
            **benchmark_case.params,
        }
        # 先完成一次稳定 data pointer 和参数缓存绑定，再测生产热路径。
        runtime.compute(**runtime_kwargs)
        for _ in range(args.repeats):
            start = time.perf_counter()
            output = runtime.compute(**runtime_kwargs)
            runtime_times.append(time.perf_counter() - start)
            runtime_core_times.append(output.core_elapsed_time)
        print("耗时基准 [us]")
        for name, values in (
            ("Python reference wall", python_times),
            ("C ABI + ctypes wall", native_times),
            ("C++ internal total", native_core_times),
            ("Production adapter wall", runtime_times),
            ("Production adapter C++ core", runtime_core_times),
        ):
            summary = percentile_summary(values)
            print(
                f"  {name}: mean={summary['mean']:.2f}, p95={summary['p95']:.2f}, "
                f"p99={summary['p99']:.2f}, max={summary['max']:.2f}"
            )
    finally:
        runtime.close()
        native.close()


if __name__ == "__main__":
    main()
