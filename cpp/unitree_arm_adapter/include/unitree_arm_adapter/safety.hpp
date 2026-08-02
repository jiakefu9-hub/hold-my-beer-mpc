#pragma once

#include <array>
#include <cstdint>

#include "unitree_arm_adapter/protocol.hpp"

namespace unitree_arm_adapter {

struct SafetyConfig {
    std::uint64_t command_timeout_ns{30'000'000ULL};
    std::uint64_t state_timeout_ns{20'000'000ULL};
    std::array<double, kArmSdkJointCount> q_min{};
    std::array<double, kArmSdkJointCount> q_max{};
    std::array<double, kArmSdkJointCount> dq_abs_max{};
    std::array<double, kArmSdkJointCount> kp_max{};
    std::array<double, kArmSdkJointCount> kd_max{};
    std::array<double, kArmSdkJointCount> tau_abs_max{};
    // 与Unitree SDK2 g1/common/terminations.hpp默认硬上限一致。
    std::int16_t motor_casing_temperature_max_c{85};
    std::int16_t motor_winding_temperature_max_c{120};
};

struct CommandPlan {
    AdapterMode mode{AdapterMode::kStartup};
    std::uint64_t command_age_ns{0};
    std::uint64_t state_age_ns{0};
    double arm_weight{0.0};
    std::array<double, kArmSdkJointCount> q{};
    std::array<double, kArmSdkJointCount> dq{};
    std::array<double, kArmSdkJointCount> kp{};
    std::array<double, kArmSdkJointCount> kd{};
    std::array<double, kArmSdkJointCount> tau{};
    bool active{false};
    bool clamped{false};
};

[[nodiscard]] SafetyConfig MakeDefaultSafetyConfig();

// 【核心代码】所有DDS发布前都必须经过这一唯一入口。
[[nodiscard]] CommandPlan BuildCommandPlan(
    const SafetyConfig& config,
    const ArmCommandPayload* command,
    const RobotStatePayload* state,
    std::uint64_t now_ns,
    bool deadline_healthy) noexcept;

[[nodiscard]] bool IsFinite(const ArmCommandPayload& command) noexcept;
[[nodiscard]] bool IsFinite(const RobotStatePayload& state) noexcept;

}  // namespace unitree_arm_adapter
