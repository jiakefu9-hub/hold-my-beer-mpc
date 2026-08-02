#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace unitree_arm_adapter {

constexpr std::uint64_t kSharedMemoryMagic = 0x473141524d504331ULL;
constexpr std::uint32_t kProtocolVersion = 2;
constexpr std::size_t kMotorCount = 35;
constexpr std::size_t kArmSdkJointCount = 13;
constexpr std::size_t kRightArmJointCount = 5;

// 顺序严格沿用 Unitree 官方 G1 arm5 DDS 示例：左臂5、右臂5、腰3。
constexpr std::array<std::size_t, kArmSdkJointCount> kArmSdkMotorIndices{
    15, 16, 17, 18, 19,
    22, 23, 24, 25, 26,
    12, 13, 14,
};
constexpr std::array<std::size_t, kRightArmJointCount> kRightArmMotorIndices{
    22, 23, 24, 25, 26,
};
constexpr std::size_t kArmWeightMotorIndex = 29;

enum class CommandMode : std::uint32_t {
    kInvalid = 0,
    // 机器人底层计算 PD；tau 字段只能放不含 PD 的前馈力矩。
    kRobotPdPlusFeedforward = 1,
    // 机器人底层 kp/kd 强制为零；tau 字段是已经合成好的最终力矩。
    kDirectTorque = 2,
};

enum CommandFlags : std::uint32_t {
    // CLI 的 --enable-output 之外，Python 还必须逐拍显式请求输出。
    kCommandRequestOutput = 1U << 0U,
};

enum class AdapterMode : std::uint32_t {
    kStartup = 0,
    kActiveRobotPd = 1,
    kActiveDirectTorque = 2,
    kDryRun = 3,
    kSafeReleaseNoCommand = 4,
    kSafeReleaseCommandStale = 5,
    kSafeReleaseStateStale = 6,
    kSafeReleaseInvalidCommand = 7,
    kSafeReleaseInvalidState = 8,
    kSafeReleaseDeadline = 9,
    kSafeReleaseOvertemperature = 10,
};

enum AdapterStatusFlags : std::uint32_t {
    kStatusOutputEnabled = 1U << 0U,
    kStatusDdsWritePerformed = 1U << 1U,
    kStatusCommandSnapshotValid = 1U << 2U,
    kStatusStateSnapshotValid = 1U << 3U,
    kStatusCommandClamped = 1U << 4U,
    kStatusDeadlineHealthy = 1U << 5U,
};

struct ArmCommandPayload {
    // 必须来自本机 CLOCK_MONOTONIC；Python 对应 time.monotonic_ns()。
    std::uint64_t monotonic_timestamp_ns{0};
    std::uint64_t command_id{0};
    std::uint32_t mode{static_cast<std::uint32_t>(CommandMode::kInvalid)};
    std::uint32_t flags{0};
    double arm_weight{0.0};
    std::array<double, kArmSdkJointCount> q_ref{};
    std::array<double, kArmSdkJointCount> dq_ref{};
    // 预留给未来C++ RNEA；当前适配器不在缺少floating-base状态时擅自使用。
    std::array<double, kArmSdkJointCount> ddq_des{};
    std::array<double, kArmSdkJointCount> kp{};
    std::array<double, kArmSdkJointCount> kd{};
    // Robot-PD模式：纯tau_ff；DirectTorque模式：最终tau_cmd。
    std::array<double, kArmSdkJointCount> tau{};
};

struct RobotStatePayload {
    // DDS回调接收本条LowState时的本机CLOCK_MONOTONIC时间。
    std::uint64_t monotonic_timestamp_ns{0};
    std::uint64_t sample_id{0};
    std::uint32_t robot_tick{0};
    std::uint8_t mode_pr{0};
    std::uint8_t mode_machine{0};
    std::array<std::uint8_t, 2> reserved{};
    std::array<double, kMotorCount> q{};
    std::array<double, kMotorCount> dq{};
    std::array<double, kMotorCount> ddq{};
    std::array<double, kMotorCount> tau_est{};
    // Unitree定义：temperature[][0]为机壳，[][1]为绕组，单位为摄氏度。
    std::array<std::array<std::int16_t, 2>, kMotorCount> motor_temperature_c{};
    std::array<double, 4> imu_quaternion_wxyz{};
    std::array<double, 3> imu_gyroscope{};
    std::array<double, 3> imu_accelerometer{};
    std::array<double, 3> imu_rpy{};
};

struct AdapterStatusPayload {
    std::uint64_t monotonic_timestamp_ns{0};
    std::uint64_t loop_count{0};
    std::uint64_t command_id{0};
    std::uint64_t command_age_ns{0};
    std::uint64_t state_age_ns{0};
    std::uint64_t wake_lateness_ns{0};
    std::uint64_t execution_time_ns{0};
    std::uint64_t deadline_miss_count{0};
    std::uint64_t command_stale_count{0};
    std::uint64_t state_stale_count{0};
    // 以2 ms控制拍计数，不是独立热事件的次数。
    std::uint64_t overtemperature_count{0};
    std::uint32_t mode{static_cast<std::uint32_t>(AdapterMode::kStartup)};
    std::uint32_t flags{0};
};

template <typename Payload>
struct alignas(64) SeqlockSlot {
    // Linux进程间seqlock。偶数表示稳定，奇数表示写入中。
    alignas(8) std::uint64_t sequence{0};
    Payload payload{};
};

struct SharedMemoryLayout {
    std::uint64_t magic{kSharedMemoryMagic};
    std::uint32_t version{kProtocolVersion};
    // 不能在类型尚未定义完成时使用sizeof；首次映射时再写入真实大小。
    std::uint32_t layout_size{0};
    SeqlockSlot<ArmCommandPayload> command{};
    SeqlockSlot<RobotStatePayload> state{};
    SeqlockSlot<AdapterStatusPayload> status{};
};

static_assert(std::is_standard_layout_v<ArmCommandPayload>);
static_assert(std::is_trivially_copyable_v<ArmCommandPayload>);
static_assert(std::is_standard_layout_v<RobotStatePayload>);
static_assert(std::is_trivially_copyable_v<RobotStatePayload>);
static_assert(std::is_standard_layout_v<AdapterStatusPayload>);
static_assert(std::is_trivially_copyable_v<AdapterStatusPayload>);
static_assert(alignof(SeqlockSlot<ArmCommandPayload>) >= 64);
static_assert(__atomic_always_lock_free(sizeof(std::uint64_t), nullptr));

}  // namespace unitree_arm_adapter
