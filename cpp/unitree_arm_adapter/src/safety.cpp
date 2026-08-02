#include "unitree_arm_adapter/safety.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace unitree_arm_adapter {
namespace {

constexpr double kDegreesToRadians = 0.017453292519943295;

template <std::size_t Size>
bool ArrayFinite(const std::array<double, Size>& values) noexcept {
    for (const double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

double Clamp(
    double value, double lower, double upper, bool& clamped) noexcept {
    const double output = std::clamp(value, lower, upper);
    clamped = clamped || output != value;
    return output;
}

CommandPlan SafeRelease(
    AdapterMode mode,
    const RobotStatePayload* state,
    std::uint64_t command_age,
    std::uint64_t state_age) noexcept {
    CommandPlan plan;
    plan.mode = mode;
    plan.command_age_ns = command_age;
    plan.state_age_ns = state_age;
    plan.arm_weight = 0.0;

    // weight=0表示把控制权交回机器人内部控制器。q仅填当前值，避免
    // 固件在权重过渡期间看到突变参考；kp/kd/tau始终为零。
    if (state != nullptr && IsFinite(*state)) {
        for (std::size_t local = 0; local < kArmSdkJointCount; ++local) {
            plan.q[local] = state->q[kArmSdkMotorIndices[local]];
        }
    }
    return plan;
}

bool TimestampAge(
    std::uint64_t timestamp,
    std::uint64_t now,
    std::uint64_t& age) noexcept {
    if (timestamp == 0U || timestamp > now) {
        age = 0U;
        return false;
    }
    age = now - timestamp;
    return true;
}

bool IsArmOvertemperature(
    const SafetyConfig& config,
    const RobotStatePayload& state) noexcept {
    // 只检查arm_sdk真正接管的13个关节，不把腿部温度混入右臂释放条件。
    for (const std::size_t motor_index : kArmSdkMotorIndices) {
        const auto& temperature = state.motor_temperature_c[motor_index];
        if (temperature[0] > config.motor_casing_temperature_max_c ||
            temperature[1] > config.motor_winding_temperature_max_c) {
            return true;
        }
    }
    return false;
}

}  // namespace

SafetyConfig MakeDefaultSafetyConfig() {
    SafetyConfig config;
    config.q_min.fill(-3.141592653589793);
    config.q_max.fill(3.141592653589793);
    config.dq_abs_max.fill(1.0);
    config.kp_max.fill(200.0);
    config.kd_max.fill(50.0);
    config.tau_abs_max.fill(25.0);

    // 本项目局部右臂顺序在13维arm_sdk数组中的位置为5..9。
    config.q_min[5] = -5.0 * kDegreesToRadians;
    config.q_max[5] = 5.0 * kDegreesToRadians;
    config.q_min[6] = -5.0 * kDegreesToRadians;
    config.q_max[6] = 3.0 * kDegreesToRadians;
    config.q_min[7] = -20.0 * kDegreesToRadians;
    config.q_max[7] = 5.0 * kDegreesToRadians;
    config.q_min[8] = -40.0 * kDegreesToRadians;
    config.q_max[8] = 40.0 * kDegreesToRadians;
    config.q_min[9] = -40.0 * kDegreesToRadians;
    config.q_max[9] = 40.0 * kDegreesToRadians;
    return config;
}

bool IsFinite(const ArmCommandPayload& command) noexcept {
    return std::isfinite(command.arm_weight) &&
           ArrayFinite(command.q_ref) && ArrayFinite(command.dq_ref) &&
           ArrayFinite(command.ddq_des) &&
           ArrayFinite(command.kp) && ArrayFinite(command.kd) &&
           ArrayFinite(command.tau);
}

bool IsFinite(const RobotStatePayload& state) noexcept {
    if (!ArrayFinite(state.q) || !ArrayFinite(state.dq) ||
        !ArrayFinite(state.ddq) || !ArrayFinite(state.tau_est) ||
        !ArrayFinite(state.imu_quaternion_wxyz) ||
        !ArrayFinite(state.imu_gyroscope) ||
        !ArrayFinite(state.imu_accelerometer) ||
        !ArrayFinite(state.imu_rpy)) {
        return false;
    }
    double quaternion_norm_squared = 0.0;
    for (const double value : state.imu_quaternion_wxyz) {
        quaternion_norm_squared += value * value;
    }
    return quaternion_norm_squared > 0.25 && quaternion_norm_squared < 2.25;
}

CommandPlan BuildCommandPlan(
    const SafetyConfig& config,
    const ArmCommandPayload* command,
    const RobotStatePayload* state,
    std::uint64_t now_ns,
    bool deadline_healthy) noexcept {
    std::uint64_t command_age = 0U;
    std::uint64_t state_age = 0U;

    if (!deadline_healthy) {
        return SafeRelease(
            AdapterMode::kSafeReleaseDeadline,
            state,
            command_age,
            state_age);
    }
    if (state == nullptr || !IsFinite(*state)) {
        return SafeRelease(
            AdapterMode::kSafeReleaseInvalidState,
            nullptr,
            command_age,
            state_age);
    }
    if (!TimestampAge(state->monotonic_timestamp_ns, now_ns, state_age)) {
        return SafeRelease(
            AdapterMode::kSafeReleaseInvalidState,
            state,
            command_age,
            state_age);
    }
    if (state_age > config.state_timeout_ns) {
        return SafeRelease(
            AdapterMode::kSafeReleaseStateStale,
            state,
            command_age,
            state_age);
    }
    if (IsArmOvertemperature(config, *state)) {
        // 【核心过热保护】温度检查优先于上游命令；即使命令新鲜也必须释放。
        return SafeRelease(
            AdapterMode::kSafeReleaseOvertemperature,
            state,
            command_age,
            state_age);
    }
    if (command == nullptr ||
        (command->flags & kCommandRequestOutput) == 0U) {
        return SafeRelease(
            AdapterMode::kSafeReleaseNoCommand,
            state,
            command_age,
            state_age);
    }
    if (!IsFinite(*command) ||
        !TimestampAge(command->monotonic_timestamp_ns, now_ns, command_age)) {
        return SafeRelease(
            AdapterMode::kSafeReleaseInvalidCommand,
            state,
            command_age,
            state_age);
    }
    if (command_age > config.command_timeout_ns) {
        return SafeRelease(
            AdapterMode::kSafeReleaseCommandStale,
            state,
            command_age,
            state_age);
    }

    const auto mode = static_cast<CommandMode>(command->mode);
    if (mode != CommandMode::kRobotPdPlusFeedforward &&
        mode != CommandMode::kDirectTorque) {
        return SafeRelease(
            AdapterMode::kSafeReleaseInvalidCommand,
            state,
            command_age,
            state_age);
    }

    CommandPlan plan;
    plan.mode = mode == CommandMode::kRobotPdPlusFeedforward
                    ? AdapterMode::kActiveRobotPd
                    : AdapterMode::kActiveDirectTorque;
    plan.command_age_ns = command_age;
    plan.state_age_ns = state_age;
    plan.arm_weight = Clamp(command->arm_weight, 0.0, 1.0, plan.clamped);
    plan.active = true;

    for (std::size_t joint = 0; joint < kArmSdkJointCount; ++joint) {
        plan.tau[joint] = Clamp(
            command->tau[joint],
            -config.tau_abs_max[joint],
            config.tau_abs_max[joint],
            plan.clamped);
        if (mode == CommandMode::kRobotPdPlusFeedforward) {
            plan.q[joint] = Clamp(
                command->q_ref[joint],
                config.q_min[joint],
                config.q_max[joint],
                plan.clamped);
            plan.dq[joint] = Clamp(
                command->dq_ref[joint],
                -config.dq_abs_max[joint],
                config.dq_abs_max[joint],
                plan.clamped);
            plan.kp[joint] = Clamp(
                command->kp[joint], 0.0, config.kp_max[joint], plan.clamped);
            plan.kd[joint] = Clamp(
                command->kd[joint], 0.0, config.kd_max[joint], plan.clamped);
        } else {
            // 【核心安全语义】tau已经包含PD时，发送给机器人底层的kp/kd
            // 必须严格为零，杜绝同一误差被计算两遍。
            plan.q[joint] = state->q[kArmSdkMotorIndices[joint]];
            plan.dq[joint] = 0.0;
            plan.kp[joint] = 0.0;
            plan.kd[joint] = 0.0;
        }
    }
    return plan;
}

}  // namespace unitree_arm_adapter
