#include "right_arm_executor/right_arm_executor_c.h"

#include "right_arm_executor/right_arm_executor.hpp"

#include <chrono>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

using right_arm_executor::ExecutorConfig;
using right_arm_executor::ExecutorInput;
using right_arm_executor::ExecutorMode;
using right_arm_executor::ExecutorOutput;
using right_arm_executor::JointVector;
using right_arm_executor::OutputSemantics;
using right_arm_executor::RightArmExecutor;
using right_arm_executor::kJointCount;

static_assert(kJointCount == RAE_JOINT_COUNT, "C/C++ joint count mismatch");
static_assert(
    static_cast<std::uint32_t>(ExecutorMode::kActive) == RAE_MODE_ACTIVE &&
        static_cast<std::uint32_t>(ExecutorMode::kCommandTimedOut) ==
            RAE_MODE_COMMAND_TIMED_OUT &&
        static_cast<std::uint32_t>(ExecutorMode::kStateTimedOut) ==
            RAE_MODE_STATE_TIMED_OUT &&
        static_cast<std::uint32_t>(ExecutorMode::kInvalidCommand) ==
            RAE_MODE_INVALID_COMMAND &&
        static_cast<std::uint32_t>(ExecutorMode::kInvalidState) ==
            RAE_MODE_INVALID_STATE,
    "C/C++ mode values mismatch");

struct rae_executor_handle {
    explicit rae_executor_handle(ExecutorConfig config)
        : executor(std::move(config)) {}

    RightArmExecutor executor;
};

namespace {

bool IsValidSemantics(std::uint32_t value) noexcept {
    return value == RAE_OUTPUT_HOST_FULL_TORQUE ||
           value == RAE_OUTPUT_DEVICE_PD;
}

OutputSemantics ToCppSemantics(std::uint32_t value) noexcept {
    return value == RAE_OUTPUT_DEVICE_PD ? OutputSemantics::kDevicePd
                                         : OutputSemantics::kHostFullTorque;
}

template <typename Source>
JointVector FromCArray(const Source& source) noexcept {
    JointVector result{};
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        result[joint] = source[joint];
    }
    return result;
}

void ToCArray(const JointVector& source, double* destination) noexcept {
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        destination[joint] = source[joint];
    }
}

bool ConfigHeaderIsValid(const rae_config_v1& config) noexcept {
    return config.struct_size == sizeof(rae_config_v1) &&
           config.abi_version == RAE_ABI_VERSION_V1;
}

bool InputHeaderIsValid(const rae_input_v1& input) noexcept {
    return input.struct_size == sizeof(rae_input_v1) &&
           input.abi_version == RAE_ABI_VERSION_V1;
}

ExecutorConfig ToCppConfig(const rae_config_v1& config) {
    ExecutorConfig result;
    result.output_semantics = ToCppSemantics(config.output_semantics);
    result.command_timeout_ns = config.command_timeout_ns;
    result.state_timeout_ns = config.state_timeout_ns;
    result.kp = FromCArray(config.kp);
    result.kd = FromCArray(config.kd);
    result.timeout_damping = FromCArray(config.timeout_damping);
    result.q_ref_min = FromCArray(config.q_ref_min);
    result.q_ref_max = FromCArray(config.q_ref_max);
    result.dq_ref_abs_max = FromCArray(config.dq_ref_abs_max);
    result.tau_min = FromCArray(config.tau_min);
    result.tau_max = FromCArray(config.tau_max);
    return result;
}

void FillCConfig(const ExecutorConfig& source, rae_config_v1& destination) noexcept {
    std::memset(&destination, 0, sizeof(destination));
    destination.struct_size = sizeof(destination);
    destination.abi_version = RAE_ABI_VERSION_V1;
    destination.output_semantics =
        static_cast<std::uint32_t>(source.output_semantics);
    destination.command_timeout_ns = source.command_timeout_ns;
    destination.state_timeout_ns = source.state_timeout_ns;
    ToCArray(source.kp, destination.kp);
    ToCArray(source.kd, destination.kd);
    ToCArray(source.timeout_damping, destination.timeout_damping);
    ToCArray(source.q_ref_min, destination.q_ref_min);
    ToCArray(source.q_ref_max, destination.q_ref_max);
    ToCArray(source.dq_ref_abs_max, destination.dq_ref_abs_max);
    ToCArray(source.tau_min, destination.tau_min);
    ToCArray(source.tau_max, destination.tau_max);
}

ExecutorInput ToCppInput(const rae_input_v1& source) noexcept {
    ExecutorInput result;
    result.command_timestamp_ns = source.command_timestamp_ns;
    result.state_timestamp_ns = source.state_timestamp_ns;
    result.q = FromCArray(source.q);
    result.dq = FromCArray(source.dq);
    result.q_ref = FromCArray(source.q_ref);
    result.dq_ref = FromCArray(source.dq_ref);
    result.tau_ff = FromCArray(source.tau_ff);
    return result;
}

std::uint32_t OutputFlags(const ExecutorOutput& output) noexcept {
    std::uint32_t flags = 0;
    if (output.position_reference_clamped) {
        flags |= RAE_FLAG_POSITION_REFERENCE_CLAMPED;
    }
    if (output.velocity_reference_clamped) {
        flags |= RAE_FLAG_VELOCITY_REFERENCE_CLAMPED;
    }
    if (output.torque_clamped) {
        flags |= RAE_FLAG_PREDICTED_TOTAL_TORQUE_CLAMPED;
    }
    if (output.feedforward_clamped) {
        flags |= RAE_FLAG_FEEDFORWARD_CLAMPED;
    }
    if (output.damping_fallback_active) {
        flags |= RAE_FLAG_DAMPING_FALLBACK_ACTIVE;
    }
    if (output.device_total_torque_limit_required) {
        flags |= RAE_FLAG_DEVICE_TOTAL_TORQUE_LIMIT_REQUIRED;
    }
    return flags;
}

void FillCOutput(
    const ExecutorOutput& source,
    std::uint64_t core_elapsed_ns,
    rae_output_v1& destination) noexcept {
    // 【核心代码】固定布局就地写出；Step 路径不构造字符串或动态容器。
    std::memset(&destination, 0, sizeof(destination));
    destination.struct_size = sizeof(destination);
    destination.abi_version = RAE_ABI_VERSION_V1;
    destination.executor_mode = static_cast<std::uint32_t>(source.mode);
    destination.output_semantics =
        static_cast<std::uint32_t>(source.output_semantics);
    destination.flags = OutputFlags(source);
    destination.command_age_ns = source.command_age_ns;
    destination.state_age_ns = source.state_age_ns;
    destination.core_elapsed_ns = core_elapsed_ns;
    ToCArray(source.effective_q_ref, destination.effective_q_ref);
    ToCArray(source.effective_dq_ref, destination.effective_dq_ref);
    ToCArray(source.pd_torque, destination.predicted_pd_tau);
    ToCArray(source.tau_raw, destination.predicted_total_tau_raw);
    ToCArray(source.tau_command, destination.predicted_total_tau_limited);
    ToCArray(source.actuator_q_ref, destination.actuator_q_ref);
    ToCArray(source.actuator_dq_ref, destination.actuator_dq_ref);
    ToCArray(source.actuator_kp, destination.actuator_kp);
    ToCArray(source.actuator_kd, destination.actuator_kd);
    ToCArray(source.actuator_tau_ff, destination.actuator_tau_ff);
}

}  // namespace

extern "C" {

std::uint32_t rae_abi_version(void) {
    return RAE_ABI_VERSION_V1;
}

std::int32_t rae_get_default_config_v1(
    std::uint32_t output_semantics,
    rae_config_v1* out_config) {
    if (out_config == nullptr || !IsValidSemantics(output_semantics)) {
        return RAE_STATUS_INVALID_ARGUMENT;
    }
    ExecutorConfig config = right_arm_executor::MakeProjectDefaultConfig();
    config.output_semantics = ToCppSemantics(output_semantics);
    FillCConfig(config, *out_config);
    return RAE_STATUS_OK;
}

std::int32_t rae_create_v1(
    const rae_config_v1* config,
    rae_executor_handle** out_handle) {
    if (out_handle == nullptr) {
        return RAE_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = nullptr;
    if (config == nullptr) {
        return RAE_STATUS_INVALID_ARGUMENT;
    }
    if (!ConfigHeaderIsValid(*config)) {
        return RAE_STATUS_INCOMPATIBLE_ABI;
    }
    if (!IsValidSemantics(config->output_semantics)) {
        return RAE_STATUS_INVALID_CONFIG;
    }
    try {
        *out_handle = new rae_executor_handle(ToCppConfig(*config));
    } catch (const std::bad_alloc&) {
        return RAE_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
        return RAE_STATUS_INVALID_CONFIG;
    } catch (...) {
        return RAE_STATUS_INTERNAL_ERROR;
    }
    return RAE_STATUS_OK;
}

void rae_destroy(rae_executor_handle* handle) {
    delete handle;
}

std::int32_t rae_step_v1(
    const rae_executor_handle* handle,
    const rae_input_v1* input,
    std::int64_t now_ns,
    rae_output_v1* output) {
    if (handle == nullptr || input == nullptr || output == nullptr) {
        return RAE_STATUS_INVALID_ARGUMENT;
    }
    if (!InputHeaderIsValid(*input)) {
        return RAE_STATUS_INCOMPATIBLE_ABI;
    }
    const ExecutorInput cpp_input = ToCppInput(*input);
    // 【半核心诊断】两次 steady_clock 只测 C++ 核心，不含 C 数组复制、
    // ctypes/IPC；RightArmExecutor::Step 本身不分配内存。
    const auto core_start = std::chrono::steady_clock::now();
    const ExecutorOutput result = handle->executor.Step(cpp_input, now_ns);
    const auto core_end = std::chrono::steady_clock::now();
    const auto core_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        core_end - core_start);
    FillCOutput(
        result,
        static_cast<std::uint64_t>(core_elapsed.count()),
        *output);
    return RAE_STATUS_OK;
}

const char* rae_status_string(std::int32_t status) {
    switch (status) {
        case RAE_STATUS_OK:
            return "ok";
        case RAE_STATUS_INVALID_ARGUMENT:
            return "invalid_argument";
        case RAE_STATUS_INCOMPATIBLE_ABI:
            return "incompatible_abi";
        case RAE_STATUS_INVALID_CONFIG:
            return "invalid_config";
        case RAE_STATUS_OUT_OF_MEMORY:
            return "out_of_memory";
        case RAE_STATUS_INTERNAL_ERROR:
            return "internal_error";
        default:
            return "unknown";
    }
}

const char* rae_executor_mode_string(std::uint32_t executor_mode) {
    if (executor_mode > RAE_MODE_INVALID_STATE) {
        return "unknown";
    }
    return right_arm_executor::ToString(
        static_cast<ExecutorMode>(executor_mode));
}

const char* rae_output_semantics_string(std::uint32_t output_semantics) {
    if (!IsValidSemantics(output_semantics)) {
        return "unknown";
    }
    return right_arm_executor::ToString(ToCppSemantics(output_semantics));
}

}  // extern "C"
