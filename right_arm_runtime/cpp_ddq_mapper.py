"""C++ MuJoCo DDQ→力矩局部映射的 ctypes 适配层。"""

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import time

import numpy as np


ARM_DOF = 5
MAX_SCALES = 8
DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)
DEFAULT_LIBRARY_PATH = Path(
    "/tmp/hold-my-beer-mpc-ddq-torque-mapper-build/"
    "libddq_torque_mapper.so"
)


class _State(ctypes.Structure):
    _fields_ = [
        ("time", ctypes.c_double),
        ("qpos", DOUBLE_POINTER),
        ("qpos_count", ctypes.c_int32),
        ("qvel", DOUBLE_POINTER),
        ("qvel_count", ctypes.c_int32),
        ("ctrl", DOUBLE_POINTER),
        ("ctrl_count", ctypes.c_int32),
        ("qacc_warmstart", DOUBLE_POINTER),
        ("qacc_warmstart_count", ctypes.c_int32),
        ("qfrc_applied", DOUBLE_POINTER),
        ("qfrc_applied_count", ctypes.c_int32),
        ("xfrc_applied", DOUBLE_POINTER),
        ("xfrc_applied_count", ctypes.c_int32),
    ]


class _Request(ctypes.Structure):
    _fields_ = [
        ("desired_qacc", ctypes.c_double * ARM_DOF),
        ("tau_nominal", ctypes.c_double * ARM_DOF),
        ("safe_hold_tau", ctypes.c_double * ARM_DOF),
        ("has_previous_executed_tau", ctypes.c_int32),
        ("previous_executed_tau", ctypes.c_double * ARM_DOF),
    ]


class _Params(ctypes.Structure):
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


class _Output(ctypes.Structure):
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
        ("safe_hold_used", ctypes.c_int32),
        ("safety_line_search_used", ctypes.c_int32),
        ("safety_line_search_attempts", ctypes.c_int32),
        ("final_output_certified", ctypes.c_int32),
        ("no_safe_torque", ctypes.c_int32),
        ("full_forward_calls", ctypes.c_int32),
        ("forward_skip_calls", ctypes.c_int32),
        ("validated_pass_count", ctypes.c_int32),
        ("baseline_elapsed_ns", ctypes.c_uint64),
        ("first_pass_elapsed_ns", ctypes.c_uint64),
        ("second_pass_elapsed_ns", ctypes.c_uint64),
        ("rescue_elapsed_ns", ctypes.c_uint64),
        ("hold_last_elapsed_ns", ctypes.c_uint64),
        ("safety_line_search_elapsed_ns", ctypes.c_uint64),
        ("total_elapsed_ns", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class CppDdqMapperResult:
    values: dict
    wall_elapsed_time: float
    core_elapsed_time: float
    full_forward_calls: int
    forward_skip_calls: int
    validated_pass_count: int


class CppNoSafeTorqueError(RuntimeError):
    """原生 mapper 明确报告当前状态不存在认证安全力矩。"""


class CppDdqTorqueMapper:
    """【半核心】持有原生 MuJoCo model/data 和固定 ABI 缓冲区。"""

    def __init__(self, scene_mjcf_path, library_path=None):
        configured_path = library_path or os.environ.get(
            "DDQ_TORQUE_MAPPER_LIBRARY", str(DEFAULT_LIBRARY_PATH)
        )
        self.library_path = Path(configured_path).expanduser().resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(
                "C++ DDQ→力矩共享库不存在，请先运行 "
                "./cpp/ddq_torque_mapper/build_and_test.sh："
                f"{self.library_path}"
            )
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_signatures()
        if int(self._library.ddq_torque_mapper_abi_version()) != 2:
            raise RuntimeError("C++ DDQ→力矩 ABI 版本不匹配。")
        self._error = ctypes.create_string_buffer(2048)
        self._handle = self._library.ddq_torque_mapper_create(
            str(Path(scene_mjcf_path).expanduser().resolve()).encode("utf-8"),
            self._error,
            len(self._error),
        )
        if not self._handle:
            raise RuntimeError(self._error_text("创建 C++ DDQ→力矩 mapper 失败"))
        self.nq = int(self._library.ddq_torque_mapper_nq(self._handle))
        self.nv = int(self._library.ddq_torque_mapper_nv(self._handle))
        self.nu = int(self._library.ddq_torque_mapper_nu(self._handle))
        self.nbody = int(
            self._library.ddq_torque_mapper_nbody(self._handle)
        )
        self._request = _Request()
        self._params = _Params()
        self._output = _Output()
        self._library.ddq_torque_mapper_default_params(
            ctypes.byref(self._params)
        )
        # 【非核心热路径缓存】MuJoCo MjData 的底层数组在 data 生命周期内
        # 地址稳定；保留 ndarray owner 和 ctypes 指针，可避免每拍重复创建
        # 六组 view/pointer。若调用方换了 MjData，则在 compute 中自动重绑。
        self._bound_data = None
        self._bound_state_arrays = None
        self._ctrl_owner = None
        self._state = _State()
        self._state_pointer = ctypes.pointer(self._state)
        self._request_pointer = ctypes.pointer(self._request)
        self._params_pointer = ctypes.pointer(self._params)
        self._output_pointer = ctypes.pointer(self._output)
        self._params_key = None
        self._request_views = {
            name: np.ctypeslib.as_array(getattr(self._request, name))
            for name in (
                "desired_qacc",
                "tau_nominal",
                "safe_hold_tau",
                "previous_executed_tau",
            )
        }
        self._output_vector_views = {
            name: np.ctypeslib.as_array(getattr(self._output, name))
            for name in (
                "tau_cmd",
                "tau_nominal",
                "tau_correction_raw",
                "tau_correction",
                "tau_cmd_raw",
                "qacc_baseline",
                "qacc_predicted",
                "qacc_prediction_error",
                "qacc_validated",
                "qacc_validation_error",
                "qacc_linearization_error",
                "singular_values",
                "first_pass_qacc_validated",
                "first_pass_qacc_validation_error",
                "second_pass_tau_correction_raw",
                "second_pass_tau_correction",
                "second_pass_qacc_predicted",
                "second_pass_qacc_validated",
                "second_pass_qacc_validation_error",
                "second_pass_qacc_linearization_error",
                "second_pass_singular_values",
                "hold_last_safe_qacc",
            )
        }
        self._output_matrix_views = {
            name: np.ctypeslib.as_array(getattr(self._output, name)).reshape(
                ARM_DOF, ARM_DOF
            )
            for name in ("gain_matrix", "second_pass_gain_matrix")
        }

    def _configure_signatures(self):
        library = self._library
        library.ddq_torque_mapper_abi_version.argtypes = []
        library.ddq_torque_mapper_abi_version.restype = ctypes.c_int32
        library.ddq_torque_mapper_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        library.ddq_torque_mapper_create.restype = ctypes.c_void_p
        library.ddq_torque_mapper_destroy.argtypes = [ctypes.c_void_p]
        for name in (
            "ddq_torque_mapper_nq",
            "ddq_torque_mapper_nv",
            "ddq_torque_mapper_nu",
            "ddq_torque_mapper_nbody",
        ):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_int32
        library.ddq_torque_mapper_default_params.argtypes = [
            ctypes.POINTER(_Params)
        ]
        library.ddq_torque_mapper_compute.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_State),
            ctypes.POINTER(_Request),
            ctypes.POINTER(_Params),
            ctypes.POINTER(_Output),
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        library.ddq_torque_mapper_compute.restype = ctypes.c_int32
        library.ddq_torque_mapper_status_string.argtypes = [ctypes.c_int32]
        library.ddq_torque_mapper_status_string.restype = ctypes.c_char_p

    @staticmethod
    def _array(values, shape, name):
        result = np.ascontiguousarray(values, dtype=np.float64)
        if result.shape != shape:
            raise ValueError(f"{name} 维度应为 {shape}，当前 {result.shape}。")
        return result

    @staticmethod
    def _pointer(values):
        return values.ctypes.data_as(DOUBLE_POINTER)

    def _error_text(self, prefix):
        detail = self._error.value.decode("utf-8", errors="replace")
        return f"{prefix}: {detail}" if detail else prefix

    def _bind_state_arrays(self, data):
        """把稳定的 MjData 数组绑定到 C ABI；借用仅持续到 data 被替换。"""
        qpos = self._array(data.qpos, (self.nq,), "qpos")
        qvel = self._array(data.qvel, (self.nv,), "qvel")
        warmstart = self._array(
            data.qacc_warmstart, (self.nv,), "qacc_warmstart"
        )
        qfrc = self._array(data.qfrc_applied, (self.nv,), "qfrc_applied")
        xfrc = self._array(
            data.xfrc_applied, (self.nbody, 6), "xfrc_applied"
        ).reshape(-1)
        self._bound_data = data
        # ndarray owner 必须与 ctypes pointer 一起保活；不能只缓存裸指针。
        self._bound_state_arrays = (qpos, qvel, warmstart, qfrc, xfrc)
        state = self._state
        state.qpos = self._pointer(qpos)
        state.qpos_count = qpos.size
        state.qvel = self._pointer(qvel)
        state.qvel_count = qvel.size
        state.qacc_warmstart = self._pointer(warmstart)
        state.qacc_warmstart_count = warmstart.size
        state.qfrc_applied = self._pointer(qfrc)
        state.qfrc_applied_count = qfrc.size
        state.xfrc_applied = self._pointer(xfrc)
        state.xfrc_applied_count = xfrc.size

    def _update_params(
        self,
        *,
        perturbation,
        regularization,
        validation_scales,
        second_pass_error_threshold,
        max_joint_error,
        max_abs_qacc,
        enable_second_pass,
        max_safety_rescue_passes,
    ):
        """配置通常整场不变，只在数值真正变化时写 ctypes 结构。"""
        scales = tuple(float(value) for value in validation_scales)
        if not 0 < len(scales) <= MAX_SCALES:
            raise ValueError("validation_scales 数量必须在 1 到 8 之间。")
        key = (
            float(perturbation),
            float(regularization),
            scales,
            float(second_pass_error_threshold),
            float(max_joint_error),
            float(max_abs_qacc),
            bool(enable_second_pass),
            int(max_safety_rescue_passes),
        )
        if key == self._params_key:
            return
        self._params.perturbation = key[0]
        self._params.regularization = key[1]
        self._params.second_pass_error_threshold = key[3]
        self._params.max_joint_error = key[4]
        self._params.max_abs_qacc = key[5]
        self._params.enable_second_pass = int(key[6])
        self._params.max_safety_rescue_passes = key[7]
        self._params.validation_scale_count = len(scales)
        for index in range(MAX_SCALES):
            self._params.validation_scales[index] = (
                scales[index] if index < len(scales) else 0.0
            )
        self._params_key = key

    def compute(
        self,
        *,
        data,
        fixed_ctrl,
        desired_qacc,
        tau_nominal,
        previous_executed_tau,
        safe_hold_tau,
        perturbation,
        regularization,
        validation_scales=(1.0, 0.5, 0.25, 0.125),
        second_pass_error_threshold=5.0,
        max_joint_error=4.0,
        max_abs_qacc=8.0,
        enable_second_pass=True,
        max_safety_rescue_passes=2,
    ):
        """【核心桥接】一次完成完整局部映射、验收与安全分支。"""
        wall_start = time.perf_counter_ns()
        if data is not self._bound_data:
            self._bind_state_arrays(data)
        ctrl = self._array(fixed_ctrl, (self.nu,), "fixed_ctrl")
        # fixed_ctrl 在主循环中每拍新建，因此只借用到本次同步 C 调用结束；
        # 不把输出改为借用 view，避免下一拍覆盖历史诊断。
        self._ctrl_owner = ctrl
        self._state.time = float(data.time)
        self._state.ctrl = self._pointer(ctrl)
        self._state.ctrl_count = ctrl.size
        desired = self._array(desired_qacc, (ARM_DOF,), "desired_qacc")
        nominal = self._array(tau_nominal, (ARM_DOF,), "tau_nominal")
        safe_hold = self._array(safe_hold_tau, (ARM_DOF,), "safe_hold_tau")
        np.copyto(self._request_views["desired_qacc"], desired)
        np.copyto(self._request_views["tau_nominal"], nominal)
        np.copyto(self._request_views["safe_hold_tau"], safe_hold)
        has_previous = previous_executed_tau is not None
        self._request.has_previous_executed_tau = int(has_previous)
        if has_previous:
            previous = self._array(
                previous_executed_tau,
                (ARM_DOF,),
                "previous_executed_tau",
            )
            np.copyto(self._request_views["previous_executed_tau"], previous)
        self._update_params(
            perturbation=perturbation,
            regularization=regularization,
            validation_scales=validation_scales,
            second_pass_error_threshold=second_pass_error_threshold,
            max_joint_error=max_joint_error,
            max_abs_qacc=max_abs_qacc,
            enable_second_pass=enable_second_pass,
            max_safety_rescue_passes=max_safety_rescue_passes,
        )
        self._error[0] = 0
        status = self._library.ddq_torque_mapper_compute(
            self._handle,
            self._state_pointer,
            self._request_pointer,
            self._params_pointer,
            self._output_pointer,
            self._error,
            len(self._error),
        )
        if status != 0:
            status_text = self._library.ddq_torque_mapper_status_string(status)
            name = (
                status_text.decode("ascii", errors="replace")
                if status_text
                else str(status)
            )
            error = self._error_text(f"C++ DDQ→力矩 {name}")
            if status == 6:
                raise CppNoSafeTorqueError(error)
            raise RuntimeError(error)

        output = self._output
        values = {
            name: view.copy()
            for name, view in self._output_vector_views.items()
        }
        values.update(
            (name, view.copy())
            for name, view in self._output_matrix_views.items()
        )
        for name in (
            "condition_number",
            "validation_scale",
            "second_pass_condition_number",
            "second_pass_validation_scale",
        ):
            values[name] = float(getattr(output, name))
        for name in (
            "validation_attempts",
            "validation_safe_candidate_count",
            "validation_total_error_rejections",
            "validation_joint_error_rejections",
            "validation_qacc_limit_rejections",
            "second_pass_validation_attempts",
            "second_pass_safe_candidate_count",
            "second_pass_total_error_rejections",
            "second_pass_joint_error_rejections",
            "second_pass_qacc_limit_rejections",
            "safety_fallback_attempts",
            "safety_line_search_attempts",
        ):
            values[name] = int(getattr(output, name))
        for name in (
            "validation_improved",
            "validation_tracking_safety_satisfied",
            "validation_qacc_safety_satisfied",
            "second_pass_triggered",
            "second_pass_accepted",
            "second_pass_tracking_safety_satisfied",
            "second_pass_qacc_safety_satisfied",
            "safety_fallback_used",
            "safety_fallback_satisfied",
            "hold_last_safe_available",
            "hold_last_safe_used",
            "hold_last_safe_satisfied",
            "safe_hold_used",
            "safety_line_search_used",
            "final_output_certified",
            "no_safe_torque",
        ):
            values[name] = bool(getattr(output, name))
        for field, native_name in (
            ("baseline_time", "baseline_elapsed_ns"),
            ("first_pass_time", "first_pass_elapsed_ns"),
            ("second_pass_time", "second_pass_elapsed_ns"),
            ("rescue_time", "rescue_elapsed_ns"),
            ("hold_last_time", "hold_last_elapsed_ns"),
            (
                "safety_line_search_time",
                "safety_line_search_elapsed_ns",
            ),
        ):
            values[field] = float(getattr(output, native_name)) * 1e-9
        wall_elapsed_time = (time.perf_counter_ns() - wall_start) * 1e-9
        return CppDdqMapperResult(
            values=values,
            wall_elapsed_time=wall_elapsed_time,
            core_elapsed_time=float(output.total_elapsed_ns) * 1e-9,
            full_forward_calls=int(output.full_forward_calls),
            forward_skip_calls=int(output.forward_skip_calls),
            validated_pass_count=int(output.validated_pass_count),
        )

    def close(self):
        if getattr(self, "_handle", None):
            self._library.ddq_torque_mapper_destroy(self._handle)
            self._handle = None
        self._bound_data = None
        self._bound_state_arrays = None
        self._ctrl_owner = None

    def __del__(self):
        self.close()
