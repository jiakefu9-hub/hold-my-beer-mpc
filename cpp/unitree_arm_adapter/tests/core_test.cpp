#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "unitree_arm_adapter/safety.hpp"
#include "unitree_arm_adapter/protocol_supervisor_adapter.hpp"
#include "unitree_arm_adapter/receipt.hpp"
#include "unitree_arm_adapter/seqlock.hpp"
#include "unitree_arm_adapter/shared_memory.hpp"

namespace ua = unitree_arm_adapter;

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __FILE__ << ':' << __LINE__                            \
                      << " CHECK失败: " #condition << '\n';                    \
            ++failures;                                                        \
        }                                                                       \
    } while (false)

constexpr std::uint64_t kNow = 1'000'000'000ULL;
constexpr double kPi = 3.14159265358979323846;

ua::RobotStatePayload ValidState() {
    ua::RobotStatePayload state;
    state.monotonic_timestamp_ns = kNow - 1'000'000ULL;
    state.validated_timestamp_ns = kNow - 900'000ULL;
    state.ingress_session_nonce = 17U;
    state.low_state_timestamp_ns = kNow - 1'000'000ULL;
    state.torso_imu_timestamp_ns = kNow - 950'000ULL;
    state.source_skew_ns = 50'000ULL;
    state.sample_id = 7;
    state.ingress_flags = ua::kStateLowStateCrcValid |
                          ua::kStatePairedIngressValidated |
                          ua::kStateTorsoImuPresent;
    state.imu_quaternion_wxyz[0] = 1.0;
    for (std::size_t index = 0; index < ua::kMotorCount; ++index) {
        state.q[index] = 0.01 * static_cast<double>(index);
    }
    return state;
}

ua::ArmCommandPayload ValidCommand(ua::CommandMode mode) {
    ua::ArmCommandPayload command;
    command.monotonic_timestamp_ns = kNow - 2'000'000ULL;
    command.producer_sequence = 0;
    command.command_id = 11;
    command.source_sample_id = 7;
    command.source_timestamp_ns = kNow - 1'000'000ULL;
    command.task_epoch_id = 23;
    command.task_time_ns = 0;
    command.full_task_anchor = 0;
    command.expires_timestamp_ns = kNow + 4'000'000ULL;
    command.session_nonce = 17;
    command.safety_policy_id = 29;
    command.safety_policy_sha256.fill(0xa5U);
    command.mode = static_cast<std::uint32_t>(mode);
    command.flags = ua::kCommandRequestOutput |
                    ua::kCommandRequestArmingPd;
    command.active_mask = (1U << ua::kArmSdkJointCount) - 1U;
    command.arm_weight = 0.5;
    command.kp.fill(20.0);
    command.kd.fill(1.0);
    command.tau.fill(3.0);
    return command;
}

void TestSeqlock() {
    ua::SeqlockSlot<ua::ArmCommandPayload> slot;
    auto command = ValidCommand(ua::CommandMode::kRobotPdPlusFeedforward);
    ua::WriteSeqlock(slot, command);
    ua::ArmCommandPayload read;
    std::uint64_t published_sequence = 0U;
    CHECK(ua::ReadSeqlockWithSequence(slot, read, published_sequence));
    CHECK(published_sequence == 2U);
    CHECK(read.command_id == command.command_id);
    CHECK(read.tau[4] == 3.0);
    command.command_id = 12U;
    ua::WriteSeqlock(slot, command);
    CHECK(ua::ReadSeqlockWithSequence(slot, read, published_sequence));
    CHECK(published_sequence == 4U);
    CHECK(read.command_id == 12U);

    __atomic_store_n(&slot.sequence, 3ULL, __ATOMIC_RELEASE);
    CHECK(!ua::ReadSeqlock(slot, read, 3));
}

void TestSafetyModes() {
    const auto config = ua::MakeDefaultSafetyConfig();
    auto state = ValidState();
    auto command = ValidCommand(ua::CommandMode::kRobotPdPlusFeedforward);

    // Robot-PD模式不在适配器重复计算PD；tau必须原样保持纯前馈值。
    command.q_ref[5] = 0.02;
    state.q[22] = -0.02;
    auto plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(plan.active);
    CHECK(plan.mode == ua::AdapterMode::kActiveRobotPd);
    CHECK(plan.tau[5] == 3.0);
    CHECK(plan.kp[5] == 20.0);

    command = ValidCommand(ua::CommandMode::kDirectTorque);
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(plan.mode == ua::AdapterMode::kActiveDirectTorque);
    CHECK(plan.kp[5] == 0.0);
    CHECK(plan.kd[5] == 0.0);
    CHECK(plan.q[5] == state.q[22]);

    command = ValidCommand(ua::CommandMode::kRobotPdPlusFeedforward);
    command.flags = 0;
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(!plan.active);
    CHECK(plan.mode == ua::AdapterMode::kSafeReleaseNoCommand);
    CHECK(plan.arm_weight == 0.0);

    command = ValidCommand(ua::CommandMode::kRobotPdPlusFeedforward);
    command.monotonic_timestamp_ns = kNow - config.command_timeout_ns - 1;
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(plan.mode == ua::AdapterMode::kSafeReleaseCommandStale);

    command = ValidCommand(ua::CommandMode::kRobotPdPlusFeedforward);
    state.monotonic_timestamp_ns = kNow - config.state_timeout_ns - 1;
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(plan.mode == ua::AdapterMode::kSafeReleaseStateStale);

    state = ValidState();
    command.monotonic_timestamp_ns = kNow + 1;
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(plan.mode == ua::AdapterMode::kSafeReleaseInvalidCommand);

    command = ValidCommand(ua::CommandMode::kRobotPdPlusFeedforward);
    command.tau[3] = std::numeric_limits<double>::quiet_NaN();
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(plan.mode == ua::AdapterMode::kSafeReleaseInvalidCommand);

    command = ValidCommand(ua::CommandMode::kRobotPdPlusFeedforward);
    state.imu_quaternion_wxyz.fill(0.0);
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(plan.mode == ua::AdapterMode::kSafeReleaseInvalidState);

    state = ValidState();
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, false);
    CHECK(plan.mode == ua::AdapterMode::kSafeReleaseDeadline);

    // 腿部过温不属于本适配器的13关节接管范围。
    state = ValidState();
    command = ValidCommand(ua::CommandMode::kRobotPdPlusFeedforward);
    state.motor_temperature_c[0][0] = 200;
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(plan.mode == ua::AdapterMode::kActiveRobotPd);

    // 右肩机壳温度超过85°C时，无条件进入独立过温释放模式。
    state.motor_temperature_c[22][0] = 86;
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(plan.mode == ua::AdapterMode::kSafeReleaseOvertemperature);
    CHECK(!plan.active);
    CHECK(plan.arm_weight == 0.0);
    CHECK(plan.kp[5] == 0.0);
    CHECK(plan.kd[5] == 0.0);
    CHECK(plan.tau[5] == 0.0);

    // 等于硬上限仍允许；任一受控关节绕组超过120°C才触发。
    state = ValidState();
    state.motor_temperature_c[26][0] = 85;
    state.motor_temperature_c[26][1] = 120;
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(plan.mode == ua::AdapterMode::kActiveRobotPd);
    state.motor_temperature_c[26][1] = 121;
    plan = ua::BuildCommandPlan(config, &command, &state, kNow, true);
    CHECK(plan.mode == ua::AdapterMode::kSafeReleaseOvertemperature);
}

void TestLimits() {
    const auto config = ua::MakeDefaultSafetyConfig();
    auto state = ValidState();
    auto command = ValidCommand(ua::CommandMode::kRobotPdPlusFeedforward);
    command.arm_weight = 2.0;
    command.q_ref[5] = 1.0;
    command.dq_ref[5] = 10.0;
    command.kp[5] = 300.0;
    command.kd[5] = 60.0;
    command.tau[5] = 100.0;
    const auto plan = ua::BuildCommandPlan(
        config, &command, &state, kNow, true);
    CHECK(plan.clamped);
    CHECK(plan.arm_weight == 1.0);
    CHECK(std::abs(plan.q[5] - 5.0 * kPi / 180.0) < 1e-12);
    CHECK(plan.dq[5] == 1.0);
    CHECK(plan.kp[5] == 200.0);
    CHECK(plan.kd[5] == 50.0);
    CHECK(plan.tau[5] == 25.0);
}

void TestSharedMemory() {
    const std::string name =
        "/unitree_arm_adapter_test_" + std::to_string(::getpid());
    ua::SharedMemoryRegion::Unlink(name);
    {
        auto writer = ua::SharedMemoryRegion::Open(name, true);
        CHECK(writer.get()->magic == ua::kSharedMemoryMagic);
        CHECK(writer.get()->version == ua::kProtocolVersion);
        CHECK(writer.get()->layout_size == sizeof(ua::SharedMemoryLayout));
        auto command = ValidCommand(ua::CommandMode::kDirectTorque);
        ua::WriteSeqlock(writer.get()->command, command);

        auto reader = ua::SharedMemoryRegion::Open(name, false);
        ua::ArmCommandPayload read;
        CHECK(ua::ReadSeqlock(reader.get()->command, read));
        CHECK(read.command_id == command.command_id);
        CHECK(read.mode == command.mode);
    }
    ua::SharedMemoryRegion::Unlink(name);
}

void TestProtocolV3ConversionAndReceipt() {
    const auto state = ValidState();
    const auto command = ValidCommand(ua::CommandMode::kDirectTorque);
    const auto supervisor_state = ua::ToSupervisorState(
        state, ua::StateConversionContext{17U, true});
    CHECK(supervisor_state.validated);
    CHECK(supervisor_state.session_nonce == 17U);
    CHECK(supervisor_state.validated_timestamp_ns ==
          state.validated_timestamp_ns);
    CHECK(supervisor_state.q[5] == state.q[22]);

    const auto proposal = ua::ToSupervisorProposal(command);
    CHECK(proposal.session_nonce == command.session_nonce);
    CHECK(proposal.producer_sequence == command.producer_sequence);
    CHECK(proposal.proposal_id == command.command_id);
    CHECK(proposal.task_epoch_id == command.task_epoch_id);
    CHECK(proposal.safety_policy_id == command.safety_policy_id);
    CHECK(proposal.safety_policy_sha256 == command.safety_policy_sha256);
    CHECK(proposal.active_mask[0]);
    CHECK(proposal.active_mask[12]);
    CHECK(proposal.requested_lifecycle ==
          ua::hardware_supervisor::RequestedLifecycle::kArmingPd);

    auto release = command;
    release.flags = ua::kCommandRequestRelease;
    release.active_mask = 0U;
    const auto release_proposal = ua::ToSupervisorProposal(release);
    CHECK(release_proposal.semantics ==
          ua::hardware_supervisor::CommandSemantics::kRelease);

    auto invalid_mask = command;
    invalid_mask.active_mask |= 1U << ua::kArmSdkJointCount;
    CHECK(ua::ToSupervisorProposal(invalid_mask).semantics ==
          ua::hardware_supervisor::CommandSemantics::kInvalid);

    const auto safety = ua::MakeDefaultSafetyConfig();
    const auto plan = ua::BuildCommandPlan(
        safety, &command, &state, kNow, true);
    ua::ReceiptContext context;
    context.receipt_timestamp_ns = kNow;
    context.loop_count = 4;
    context.receipt_id = 4;
    context.command_snapshot_valid = true;
    context.state_snapshot_valid = true;
    context.deadline_healthy = true;
    context.pre_sink_check_timestamp_ns = kNow;
    context.pre_sink_deadline_ns = kNow + 1'000'000ULL;
    context.pre_sink_deadline_healthy = true;
    context.pre_sink_expiry_healthy = true;
    context.sink_write_performed = true;
    context.sink_write_timestamp_ns = kNow;
    const auto receipt = ua::BuildAdapterReceipt(
        &command, &state, plan, context);
    CHECK(receipt.command_id == command.command_id);
    CHECK(receipt.source_sample_id == command.source_sample_id);
    CHECK(receipt.session_nonce == command.session_nonce);
    CHECK(receipt.executed_active_mask == command.active_mask);
    CHECK(receipt.selected_tau[4] == plan.tau[4]);
    CHECK((receipt.flags & ua::kStatusSinkWritePerformed) != 0U);
    CHECK((receipt.flags & ua::kStatusDdsWritePerformed) == 0U);
    CHECK((receipt.flags & ua::kStatusPreSinkDeadlineHealthy) != 0U);
    CHECK((receipt.flags & ua::kStatusPreSinkExpiryHealthy) != 0U);
}

}  // namespace

int main() {
    try {
        TestSeqlock();
        TestSafetyModes();
        TestLimits();
        TestSharedMemory();
        TestProtocolV3ConversionAndReceipt();
    } catch (const std::exception& error) {
        std::cerr << "未捕获异常: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << "个测试失败。\n";
        return 1;
    }
    std::cout << "unitree_arm_adapter核心测试全部通过。\n";
    return 0;
}
