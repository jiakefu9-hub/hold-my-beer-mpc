#include "right_arm_executor/right_arm_executor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace right_arm_executor {
namespace {

constexpr double kDegreesToRadians = 0.017453292519943295;

bool IsFinite(const JointVector& values) noexcept {
    for (const double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

void ValidateConfig(const ExecutorConfig& config) {
    if (config.output_semantics != OutputSemantics::kHostFullTorque &&
        config.output_semantics != OutputSemantics::kDevicePd) {
        throw std::invalid_argument("output_semantics is invalid");
    }
    if (config.command_timeout_ns <= 0 || config.state_timeout_ns <= 0) {
        throw std::invalid_argument("command/state timeout must be positive");
    }

    if (!IsFinite(config.kp) || !IsFinite(config.kd) ||
        !IsFinite(config.timeout_damping) || !IsFinite(config.q_ref_min) ||
        !IsFinite(config.q_ref_max) || !IsFinite(config.dq_ref_abs_max) ||
        !IsFinite(config.tau_min) || !IsFinite(config.tau_max)) {
        throw std::invalid_argument("executor config contains NaN or Inf");
    }

    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        if (config.kp[joint] < 0.0 || config.kd[joint] < 0.0 ||
            config.timeout_damping[joint] < 0.0 ||
            config.dq_ref_abs_max[joint] < 0.0) {
            throw std::invalid_argument("executor gains and velocity limits must be non-negative");
        }
        if (!(config.q_ref_min[joint] < config.q_ref_max[joint]) ||
            !(config.tau_min[joint] < config.tau_max[joint])) {
            throw std::invalid_argument("executor lower limits must be smaller than upper limits");
        }
    }
}

double ClampAndReport(double value, double lower, double upper, bool& clipped) noexcept {
    const double result = std::clamp(value, lower, upper);
    clipped = clipped || result != value;
    return result;
}

void ApplyDampingFallback(
    const ExecutorConfig& config,
    const JointVector& dq,
    bool state_is_usable,
    ExecutorOutput& output) noexcept {
    // 【核心代码】超时或命令非法时不继续执行旧前馈，只保留耗散能量的阻尼项。
    output.damping_fallback_active = true;
    output.device_total_torque_limit_required =
        config.output_semantics == OutputSemantics::kDevicePd;
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        if (state_is_usable) {
            output.tau_raw[joint] = -config.timeout_damping[joint] * dq[joint];
            output.tau_command[joint] = ClampAndReport(
                output.tau_raw[joint],
                config.tau_min[joint],
                config.tau_max[joint],
                output.torque_clamped);
        }

        if (config.output_semantics == OutputSemantics::kHostFullTorque) {
            // 主机力矩模式不能依靠设备 PD；状态无效/陈旧时只能输出零力矩。
            output.actuator_tau_ff[joint] = output.tau_command[joint];
        } else {
            // 设备 PD 模式让设备基于自己的最新编码器速度执行阻尼。
            // kp=0，因此 q_ref 的数值不会产生位置力矩。
            output.actuator_kd[joint] = config.timeout_damping[joint];
        }
    }
}

void FillActiveActuatorCommand(
    const ExecutorConfig& config,
    const ExecutorInput& input,
    ExecutorOutput& output) noexcept {
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        if (config.output_semantics == OutputSemantics::kHostFullTorque) {
            // 【核心代码】完整力矩已由主机算好；设备端增益严格置零，避免重复 PD。
            output.actuator_tau_ff[joint] = output.tau_command[joint];
            continue;
        }

        // 【核心代码】设备 PD 模式只发送一次 PD 参数；tau_ff 不含 PD。
        output.actuator_q_ref[joint] = output.effective_q_ref[joint];
        output.actuator_dq_ref[joint] = output.effective_dq_ref[joint];
        output.actuator_kp[joint] = config.kp[joint];
        output.actuator_kd[joint] = config.kd[joint];
        output.actuator_tau_ff[joint] = ClampAndReport(
            input.tau_ff[joint],
            config.tau_min[joint],
            config.tau_max[joint],
            output.feedforward_clamped);
    }
    // 设备使用发送后的前馈重新形成“当前状态下预计总力矩”，用于诊断。
    if (config.output_semantics == OutputSemantics::kDevicePd) {
        output.torque_clamped = false;
        for (std::size_t joint = 0; joint < kJointCount; ++joint) {
            output.tau_raw[joint] =
                output.actuator_tau_ff[joint] + output.pd_torque[joint];
            output.tau_command[joint] = ClampAndReport(
                output.tau_raw[joint],
                config.tau_min[joint],
                config.tau_max[joint],
                output.torque_clamped);
        }
        // C ABI 只负责字段语义；真实设备必须在 PD 汇总后执行最终硬限幅。
        output.device_total_torque_limit_required = true;
    }
}

}  // namespace

RightArmExecutor::RightArmExecutor(ExecutorConfig config) : config_(std::move(config)) {
    ValidateConfig(config_);
}

ExecutorOutput RightArmExecutor::Step(const ExecutorInput& input, std::int64_t now_ns) const noexcept {
    ExecutorOutput output;
    output.output_semantics = config_.output_semantics;

    // 【核心代码】先独立判断状态时间戳，避免新命令配上陈旧状态。
    const bool state_values_valid = IsFinite(input.q) && IsFinite(input.dq);
    const bool state_timestamp_valid =
        input.state_timestamp_ns >= 0 && now_ns >= input.state_timestamp_ns;
    if (!state_values_valid || !state_timestamp_valid) {
        output.mode = ExecutorMode::kInvalidState;
        ApplyDampingFallback(config_, input.dq, false, output);
        return output;
    }
    output.state_age_ns = now_ns - input.state_timestamp_ns;
    if (output.state_age_ns > config_.state_timeout_ns) {
        output.mode = ExecutorMode::kStateTimedOut;
        ApplyDampingFallback(config_, input.dq, false, output);
        return output;
    }

    const bool timestamp_valid =
        input.command_timestamp_ns >= 0 && now_ns >= input.command_timestamp_ns;
    const bool command_values_valid =
        IsFinite(input.q_ref) && IsFinite(input.dq_ref) && IsFinite(input.tau_ff);

    if (!timestamp_valid || !command_values_valid) {
        output.mode = ExecutorMode::kInvalidCommand;
        ApplyDampingFallback(config_, input.dq, true, output);
        return output;
    }

    output.command_age_ns = now_ns - input.command_timestamp_ns;
    if (output.command_age_ns > config_.command_timeout_ns) {
        output.mode = ExecutorMode::kCommandTimedOut;
        ApplyDampingFallback(config_, input.dq, true, output);
        return output;
    }

    output.mode = ExecutorMode::kActive;
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        output.effective_q_ref[joint] = ClampAndReport(
            input.q_ref[joint],
            config_.q_ref_min[joint],
            config_.q_ref_max[joint],
            output.position_reference_clamped);
        output.effective_dq_ref[joint] = ClampAndReport(
            input.dq_ref[joint],
            -config_.dq_ref_abs_max[joint],
            config_.dq_ref_abs_max[joint],
            output.velocity_reference_clamped);

        output.pd_torque[joint] =
            config_.kp[joint] * (output.effective_q_ref[joint] - input.q[joint]) +
            config_.kd[joint] * (output.effective_dq_ref[joint] - input.dq[joint]);

        // host 模式会真正发送这个总力矩；device 模式先计算它用于诊断。
        output.tau_raw[joint] =
            input.tau_ff[joint] + output.pd_torque[joint];

        if (!std::isfinite(output.tau_raw[joint])) {
            output.mode = ExecutorMode::kInvalidCommand;
            output.tau_raw.fill(0.0);
            output.tau_command.fill(0.0);
            output.torque_clamped = false;
            ApplyDampingFallback(config_, input.dq, true, output);
            return output;
        }

        output.tau_command[joint] = ClampAndReport(
            output.tau_raw[joint],
            config_.tau_min[joint],
            config_.tau_max[joint],
            output.torque_clamped);
    }
    FillActiveActuatorCommand(config_, input, output);
    return output;
}

ExecutorConfig MakeProjectDefaultConfig() {
    ExecutorConfig config;
    config.kp = {80.0, 80.0, 60.0, 80.0, 30.0};
    config.kd = {5.0, 5.0, 3.0, 2.0, 1.0};
    config.timeout_damping = {5.0, 5.0, 3.0, 2.0, 1.0};
    config.q_ref_min = {
        -5.0 * kDegreesToRadians,
        -5.0 * kDegreesToRadians,
        -20.0 * kDegreesToRadians,
        -40.0 * kDegreesToRadians,
        -40.0 * kDegreesToRadians,
    };
    config.q_ref_max = {
        5.0 * kDegreesToRadians,
        3.0 * kDegreesToRadians,
        5.0 * kDegreesToRadians,
        40.0 * kDegreesToRadians,
        40.0 * kDegreesToRadians,
    };
    config.dq_ref_abs_max.fill(1.0);
    config.tau_min.fill(-25.0);
    config.tau_max.fill(25.0);
    config.command_timeout_ns = 30'000'000;
    config.state_timeout_ns = 10'000'000;
    return config;
}

const char* ToString(ExecutorMode mode) noexcept {
    switch (mode) {
        case ExecutorMode::kActive:
            return "active";
        case ExecutorMode::kCommandTimedOut:
            return "command_timed_out";
        case ExecutorMode::kStateTimedOut:
            return "state_timed_out";
        case ExecutorMode::kInvalidCommand:
            return "invalid_command";
        case ExecutorMode::kInvalidState:
            return "invalid_state";
    }
    return "unknown";
}

const char* ToString(OutputSemantics semantics) noexcept {
    switch (semantics) {
        case OutputSemantics::kHostFullTorque:
            return "host_full_torque";
        case OutputSemantics::kDevicePd:
            return "device_pd";
    }
    return "unknown";
}

}  // namespace right_arm_executor
