#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace unitree_arm_adapter {

constexpr std::uint64_t kSharedMemoryMagic = 0x473141524d504331ULL;
constexpr std::uint32_t kProtocolVersion = 3;
constexpr std::size_t kMotorCount = 35;
constexpr std::size_t kArmSdkJointCount = 13;
constexpr std::size_t kRightArmJointCount = 5;
constexpr std::size_t kSha256Bytes = 32;
using Sha256Digest = std::array<std::uint8_t, kSha256Bytes>;

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
    // 预留给未来site-verified output writer。Stage 2 的公开Python writer
    // 始终清除此位，仓库也没有真实命令publisher target。
    kCommandRequestOutput = 1U << 0U,
    // 生命周期请求互斥；future-output supervisor会进一步校验状态转换。
    kCommandRequestArmingPd = 1U << 1U,
    kCommandRequestActive = 1U << 2U,
    kCommandRequestRelease = 1U << 3U,
};

enum StateIngressFlags : std::uint32_t {
    kStateLowStateCrcValid = 1U << 0U,
    kStatePairedIngressValidated = 1U << 1U,
    kStateTorsoImuPresent = 1U << 2U,
    // Test-only provenance; never represents a real hardware sample.
    kStateSyntheticFixture = 1U << 31U,
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
    // 命令通过当前C++安全检查；不代表已经写DDS或完成真机安全认证。
    kStatusCommandAcceptedBySafety = 1U << 6U,
    // receipt完整回显了本拍command identity。
    kStatusReceiptIdentityValid = 1U << 7U,
    kStatusPreSinkDeadlineHealthy = 1U << 8U,
    kStatusPreSinkExpiryHealthy = 1U << 9U,
    // recording/HIL sink写入和DDS写入是两个独立事实。
    kStatusSinkWritePerformed = 1U << 10U,
};

enum class ReceiptReason : std::uint32_t {
    kNone = 0,
    kAcceptedOutputDisabled = 1,
    kDdsWritePerformed = 2,
    kSafeReleaseNoCommand = 10,
    kSafeReleaseCommandStale = 11,
    kSafeReleaseStateStale = 12,
    kSafeReleaseInvalidCommand = 13,
    kSafeReleaseInvalidState = 14,
    kSafeReleaseDeadline = 15,
    kSafeReleaseOvertemperature = 16,
    kOutputEnabledButNotWritten = 17,
};

struct ArmCommandPayload {
    // 必须来自本机 CLOCK_MONOTONIC；Python 对应 time.monotonic_ns()。
    std::uint64_t monotonic_timestamp_ns{0};
    // producer_sequence 与 command_id 分离：前者检查上游任务事件连续性，
    // 后者标识下游命令/receipt。二者都禁止依赖共享内存seqlock序号。
    std::uint64_t producer_sequence{0};
    std::uint64_t command_id{0};
    std::uint64_t source_sample_id{0};
    std::uint64_t source_timestamp_ns{0};
    std::uint64_t task_time_ns{0};
    std::uint64_t full_task_anchor{0};
    std::uint64_t expires_timestamp_ns{0};
    std::uint64_t session_nonce{0};
    std::uint64_t task_epoch_id{0};
    std::uint64_t safety_policy_id{0};
    std::uint32_t mode{static_cast<std::uint32_t>(CommandMode::kInvalid)};
    std::uint32_t flags{0};
    // 低13位对应arm_sdk 13 slots；其余位必须为零。
    std::uint32_t active_mask{0};
    std::uint32_t reserved{0};
    double arm_weight{0.0};
    Sha256Digest safety_policy_sha256{};
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
    // 两个source中较早的本机CLOCK_MONOTONIC接收时间。
    std::uint64_t monotonic_timestamp_ns{0};
    // 完成CRC/skew/pairing检查并构造本payload的时间；HIL不得自行伪造。
    std::uint64_t validated_timestamp_ns{0};
    // 由state bridge启动时显式绑定；HIL只能核对，不能从CLI覆盖state证据。
    std::uint64_t ingress_session_nonce{0};
    std::uint64_t low_state_timestamp_ns{0};
    std::uint64_t torso_imu_timestamp_ns{0};
    std::uint64_t source_skew_ns{0};
    std::uint64_t sample_id{0};
    std::uint32_t robot_tick{0};
    std::uint32_t ingress_flags{0};
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
    // protocol-v3 receipt。它完整回显command/source/task/policy identity；
    // OUTPUT_ENABLED和DDS_WRITE_PERFORMED仍是两个独立事实。
    std::uint64_t monotonic_timestamp_ns{0};
    std::uint64_t loop_count{0};
    std::uint64_t receipt_id{0};
    std::uint64_t producer_sequence{0};
    std::uint64_t command_id{0};
    std::uint64_t source_sample_id{0};
    std::uint64_t source_timestamp_ns{0};
    std::uint64_t observed_state_sample_id{0};
    std::uint64_t observed_state_timestamp_ns{0};
    std::uint64_t task_time_ns{0};
    std::uint64_t full_task_anchor{0};
    std::uint64_t command_timestamp_ns{0};
    std::uint64_t expires_timestamp_ns{0};
    std::uint64_t dds_write_timestamp_ns{0};
    std::uint64_t sink_write_timestamp_ns{0};
    std::uint64_t pre_sink_check_timestamp_ns{0};
    std::uint64_t pre_sink_deadline_ns{0};
    std::uint64_t session_nonce{0};
    std::uint64_t task_epoch_id{0};
    std::uint64_t safety_policy_id{0};
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
    std::uint32_t requested_command_mode{
        static_cast<std::uint32_t>(CommandMode::kInvalid)};
    std::uint32_t flags{0};
    std::uint32_t receipt_reason{
        static_cast<std::uint32_t>(ReceiptReason::kNone)};
    std::uint32_t guard_reason{0};
    std::uint32_t requested_active_mask{0};
    std::uint32_t executed_active_mask{0};
    double requested_arm_weight{0.0};
    double executed_arm_weight{0.0};
    Sha256Digest safety_policy_sha256{};
    std::array<double, kArmSdkJointCount> selected_q{};
    std::array<double, kArmSdkJointCount> selected_dq{};
    std::array<double, kArmSdkJointCount> selected_ddq_des{};
    std::array<double, kArmSdkJointCount> selected_kp{};
    std::array<double, kArmSdkJointCount> selected_kd{};
    std::array<double, kArmSdkJointCount> selected_tau{};
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

// protocol-v3 ABI freeze.  Python ctypes mirrors these exact sizes/offsets;
// adding/reordering a field must bump the protocol version and update parity
// tests rather than silently preserving only the top-level byte count.
static_assert(sizeof(ArmCommandPayload) == 768);
static_assert(offsetof(ArmCommandPayload, producer_sequence) == 8);
static_assert(offsetof(ArmCommandPayload, source_sample_id) == 24);
static_assert(offsetof(ArmCommandPayload, session_nonce) == 64);
static_assert(offsetof(ArmCommandPayload, mode) == 88);
static_assert(offsetof(ArmCommandPayload, safety_policy_sha256) == 112);
static_assert(offsetof(ArmCommandPayload, q_ref) == 144);
static_assert(offsetof(ArmCommandPayload, tau) == 664);
static_assert(sizeof(RobotStatePayload) == 1440);
static_assert(offsetof(RobotStatePayload, validated_timestamp_ns) == 8);
static_assert(offsetof(RobotStatePayload, ingress_session_nonce) == 16);
static_assert(offsetof(RobotStatePayload, sample_id) == 48);
static_assert(offsetof(RobotStatePayload, ingress_flags) == 60);
static_assert(offsetof(RobotStatePayload, q) == 72);
static_assert(offsetof(RobotStatePayload, imu_rpy) == 1416);
static_assert(sizeof(AdapterStatusPayload) == 928);
static_assert(offsetof(AdapterStatusPayload, receipt_id) == 16);
static_assert(offsetof(AdapterStatusPayload, source_sample_id) == 40);
static_assert(offsetof(AdapterStatusPayload, session_nonce) == 136);
static_assert(offsetof(AdapterStatusPayload, receipt_reason) == 236);
static_assert(offsetof(AdapterStatusPayload, requested_arm_weight) == 256);
static_assert(offsetof(AdapterStatusPayload, selected_q) == 304);
static_assert(offsetof(AdapterStatusPayload, selected_tau) == 824);
static_assert(sizeof(SharedMemoryLayout) == 3328);
static_assert(offsetof(SharedMemoryLayout, command) == 64);
static_assert(offsetof(SharedMemoryLayout, state) == 896);
static_assert(offsetof(SharedMemoryLayout, status) == 2368);

}  // namespace unitree_arm_adapter
