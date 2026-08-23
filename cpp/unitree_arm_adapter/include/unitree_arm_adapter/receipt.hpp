#pragma once

#include <cstdint>
#include <limits>

#include "unitree_arm_adapter/hardware_command_supervisor.hpp"
#include "unitree_arm_adapter/safety.hpp"

namespace unitree_arm_adapter {

[[nodiscard]] inline bool ProtocolCommandIdentityValid(
    const ArmCommandPayload& command) noexcept {
    const std::uint32_t lifecycle = command.flags & (
        kCommandRequestArmingPd |
        kCommandRequestActive |
        kCommandRequestRelease);
    const bool one_lifecycle = lifecycle == kCommandRequestArmingPd ||
                               lifecycle == kCommandRequestActive ||
                               lifecycle == kCommandRequestRelease;
    bool digest_nonzero = false;
    for (const auto value : command.safety_policy_sha256) {
        digest_nonzero = digest_nonzero || value != 0U;
    }
    return command.command_id != 0U && command.source_sample_id != 0U &&
           command.source_timestamp_ns != 0U &&
           command.session_nonce != 0U && command.task_epoch_id != 0U &&
           command.safety_policy_id != 0U &&
           command.full_task_anchor <=
               std::numeric_limits<std::uint64_t>::max() / 6'000'000ULL &&
           command.task_time_ns ==
               command.full_task_anchor * 6'000'000ULL &&
           command.expires_timestamp_ns > command.monotonic_timestamp_ns &&
           (command.active_mask >> kArmSdkJointCount) == 0U &&
           one_lifecycle && digest_nonzero;
}

// 所有adapter/HIL路径共用的receipt运行时字段。command identity不得在各
// executable中手写一套近似复制逻辑。
struct ReceiptContext {
    std::uint64_t receipt_timestamp_ns{0};
    std::uint64_t loop_count{0};
    std::uint64_t receipt_id{0};
    std::uint64_t dds_write_timestamp_ns{0};
    std::uint64_t sink_write_timestamp_ns{0};
    std::uint64_t pre_sink_check_timestamp_ns{0};
    std::uint64_t pre_sink_deadline_ns{0};
    std::uint64_t wake_lateness_ns{0};
    std::uint64_t execution_time_ns{0};
    std::uint64_t command_age_ns{0};
    std::uint64_t state_age_ns{0};
    std::uint64_t deadline_miss_count{0};
    std::uint64_t command_stale_count{0};
    std::uint64_t state_stale_count{0};
    std::uint64_t overtemperature_count{0};
    bool command_snapshot_valid{false};
    bool state_snapshot_valid{false};
    bool deadline_healthy{false};
    bool output_enabled{false};
    bool dds_write_performed{false};
    bool sink_write_performed{false};
    bool pre_sink_deadline_healthy{false};
    bool pre_sink_expiry_healthy{false};
    std::uint32_t guard_reason{0};
};

[[nodiscard]] inline std::uint32_t PackActiveMask(
    const std::array<bool, kArmSdkJointCount>& mask) noexcept {
    std::uint32_t bits = 0U;
    for (std::size_t slot = 0; slot < kArmSdkJointCount; ++slot) {
        if (mask[slot]) {
            bits |= 1U << slot;
        }
    }
    return bits;
}

[[nodiscard]] inline ReceiptReason ReasonForReceipt(
    const CommandPlan& plan,
    const ReceiptContext& context) noexcept {
    if (context.dds_write_performed) {
        return ReceiptReason::kDdsWritePerformed;
    }
    if (plan.active) {
        return context.output_enabled
            ? ReceiptReason::kOutputEnabledButNotWritten
            : ReceiptReason::kAcceptedOutputDisabled;
    }
    switch (plan.mode) {
        case AdapterMode::kSafeReleaseNoCommand:
            return ReceiptReason::kSafeReleaseNoCommand;
        case AdapterMode::kSafeReleaseCommandStale:
            return ReceiptReason::kSafeReleaseCommandStale;
        case AdapterMode::kSafeReleaseStateStale:
            return ReceiptReason::kSafeReleaseStateStale;
        case AdapterMode::kSafeReleaseInvalidCommand:
            return ReceiptReason::kSafeReleaseInvalidCommand;
        case AdapterMode::kSafeReleaseInvalidState:
            return ReceiptReason::kSafeReleaseInvalidState;
        case AdapterMode::kSafeReleaseDeadline:
            return ReceiptReason::kSafeReleaseDeadline;
        case AdapterMode::kSafeReleaseOvertemperature:
            return ReceiptReason::kSafeReleaseOvertemperature;
        default:
            return ReceiptReason::kNone;
    }
}

[[nodiscard]] inline AdapterStatusPayload BuildAdapterReceipt(
    const ArmCommandPayload* command,
    const RobotStatePayload* state,
    const CommandPlan& plan,
    const ReceiptContext& context) noexcept {
    AdapterStatusPayload receipt;
    receipt.monotonic_timestamp_ns = context.receipt_timestamp_ns;
    receipt.loop_count = context.loop_count;
    receipt.receipt_id = context.receipt_id;
    receipt.command_age_ns = plan.command_age_ns;
    receipt.state_age_ns = plan.state_age_ns;
    receipt.wake_lateness_ns = context.wake_lateness_ns;
    receipt.execution_time_ns = context.execution_time_ns;
    receipt.deadline_miss_count = context.deadline_miss_count;
    receipt.command_stale_count = context.command_stale_count;
    receipt.state_stale_count = context.state_stale_count;
    receipt.overtemperature_count = context.overtemperature_count;
    receipt.mode = static_cast<std::uint32_t>(plan.mode);
    receipt.receipt_reason = static_cast<std::uint32_t>(
        ReasonForReceipt(plan, context));
    receipt.guard_reason = context.guard_reason;
    const bool sink_accepted = context.sink_write_performed ||
                               context.dds_write_performed;
    receipt.executed_arm_weight = sink_accepted ? plan.arm_weight : 0.0;
    receipt.sink_write_timestamp_ns = context.sink_write_performed
        ? context.sink_write_timestamp_ns
        : 0U;
    receipt.pre_sink_check_timestamp_ns =
        context.pre_sink_check_timestamp_ns;
    receipt.pre_sink_deadline_ns = context.pre_sink_deadline_ns;
    if (context.output_enabled) {
        receipt.flags |= kStatusOutputEnabled;
    }
    if (context.dds_write_performed) {
        receipt.flags |= kStatusDdsWritePerformed;
        receipt.dds_write_timestamp_ns = context.dds_write_timestamp_ns;
    }
    if (context.sink_write_performed) {
        receipt.flags |= kStatusSinkWritePerformed;
    }
    if (context.pre_sink_deadline_healthy) {
        receipt.flags |= kStatusPreSinkDeadlineHealthy;
    }
    if (context.pre_sink_expiry_healthy) {
        receipt.flags |= kStatusPreSinkExpiryHealthy;
    }
    if (context.command_snapshot_valid) {
        receipt.flags |= kStatusCommandSnapshotValid;
    }
    if (context.state_snapshot_valid) {
        receipt.flags |= kStatusStateSnapshotValid;
    }
    if (plan.clamped) {
        receipt.flags |= kStatusCommandClamped;
    }
    if (context.deadline_healthy) {
        receipt.flags |= kStatusDeadlineHealthy;
    }
    if (plan.active) {
        receipt.flags |= kStatusCommandAcceptedBySafety;
    }
    if (state != nullptr && context.state_snapshot_valid) {
        receipt.observed_state_sample_id = state->sample_id;
        receipt.observed_state_timestamp_ns = state->monotonic_timestamp_ns;
    }
    if (command != nullptr && context.command_snapshot_valid) {
        if (ProtocolCommandIdentityValid(*command)) {
            receipt.flags |= kStatusReceiptIdentityValid;
        }
        receipt.producer_sequence = command->producer_sequence;
        receipt.command_id = command->command_id;
        receipt.source_sample_id = command->source_sample_id;
        receipt.source_timestamp_ns = command->source_timestamp_ns;
        receipt.task_time_ns = command->task_time_ns;
        receipt.full_task_anchor = command->full_task_anchor;
        receipt.command_timestamp_ns = command->monotonic_timestamp_ns;
        receipt.expires_timestamp_ns = command->expires_timestamp_ns;
        receipt.requested_command_mode = command->mode;
        receipt.requested_active_mask = command->active_mask;
        receipt.executed_active_mask = plan.active && sink_accepted
            ? command->active_mask
            : 0U;
        receipt.requested_arm_weight = command->arm_weight;
        receipt.session_nonce = command->session_nonce;
        receipt.task_epoch_id = command->task_epoch_id;
        receipt.safety_policy_id = command->safety_policy_id;
        receipt.safety_policy_sha256 = command->safety_policy_sha256;
        receipt.selected_ddq_des = plan.active
            ? command->ddq_des
            : std::array<double, kArmSdkJointCount>{};
    }
    receipt.selected_q = plan.q;
    receipt.selected_dq = plan.dq;
    receipt.selected_kp = plan.kp;
    receipt.selected_kd = plan.kd;
    receipt.selected_tau = plan.tau;
    return receipt;
}

// Publisher-absent HIL path: record the exact supervisor-selected 13-slot
// plan without equating a recording-sink write with a DDS/hardware write.
[[nodiscard]] inline AdapterStatusPayload BuildAdapterReceipt(
    const ArmCommandPayload* command,
    const RobotStatePayload* state,
    const hardware_supervisor::SupervisorResult& result,
    const ReceiptContext& context) noexcept {
    AdapterStatusPayload receipt;
    receipt.monotonic_timestamp_ns = context.receipt_timestamp_ns;
    receipt.loop_count = context.loop_count;
    receipt.receipt_id = context.receipt_id;
    receipt.command_age_ns = context.command_age_ns;
    receipt.state_age_ns = context.state_age_ns;
    receipt.wake_lateness_ns = context.wake_lateness_ns;
    receipt.execution_time_ns = context.execution_time_ns;
    receipt.deadline_miss_count = context.deadline_miss_count;
    receipt.command_stale_count = context.command_stale_count;
    receipt.state_stale_count = context.state_stale_count;
    receipt.overtemperature_count = context.overtemperature_count;
    receipt.mode = static_cast<std::uint32_t>(AdapterMode::kDryRun);
    receipt.guard_reason = context.guard_reason != 0U
        ? context.guard_reason
        : static_cast<std::uint32_t>(result.reason);
    const bool sink_accepted = context.sink_write_performed ||
                               context.dds_write_performed;
    receipt.executed_arm_weight = sink_accepted
        ? result.plan.arm_weight
        : 0.0;
    receipt.executed_active_mask = sink_accepted
        ? PackActiveMask(result.plan.active_mask)
        : 0U;
    receipt.selected_q = result.plan.q;
    receipt.selected_dq = result.plan.dq;
    receipt.selected_ddq_des = result.plan.ddq_des;
    receipt.selected_kp = result.plan.kp;
    receipt.selected_kd = result.plan.kd;
    receipt.selected_tau = result.plan.tau;
    receipt.sink_write_timestamp_ns = context.sink_write_performed
        ? context.sink_write_timestamp_ns
        : 0U;
    receipt.dds_write_timestamp_ns = context.dds_write_performed
        ? context.dds_write_timestamp_ns
        : 0U;
    receipt.pre_sink_check_timestamp_ns =
        context.pre_sink_check_timestamp_ns;
    receipt.pre_sink_deadline_ns = context.pre_sink_deadline_ns;
    if (context.command_snapshot_valid) {
        receipt.flags |= kStatusCommandSnapshotValid;
    }
    if (context.state_snapshot_valid) {
        receipt.flags |= kStatusStateSnapshotValid;
    }
    if (context.deadline_healthy) {
        receipt.flags |= kStatusDeadlineHealthy;
    }
    if (context.pre_sink_deadline_healthy) {
        receipt.flags |= kStatusPreSinkDeadlineHealthy;
    }
    if (context.pre_sink_expiry_healthy) {
        receipt.flags |= kStatusPreSinkExpiryHealthy;
    }
    if (context.output_enabled) {
        receipt.flags |= kStatusOutputEnabled;
    }
    if (context.sink_write_performed) {
        receipt.flags |= kStatusSinkWritePerformed;
    }
    if (context.dds_write_performed) {
        receipt.flags |= kStatusDdsWritePerformed;
    }
    if (result.plan.ready_for_sink) {
        receipt.flags |= kStatusCommandAcceptedBySafety;
    }
    receipt.receipt_reason = static_cast<std::uint32_t>(
        context.dds_write_performed
            ? ReceiptReason::kDdsWritePerformed
            : context.sink_write_performed
                ? ReceiptReason::kAcceptedOutputDisabled
                : ReceiptReason::kNone);
    if (state != nullptr && context.state_snapshot_valid) {
        receipt.observed_state_sample_id = state->sample_id;
        receipt.observed_state_timestamp_ns = state->monotonic_timestamp_ns;
    }
    if (command != nullptr && context.command_snapshot_valid) {
        if (ProtocolCommandIdentityValid(*command)) {
            receipt.flags |= kStatusReceiptIdentityValid;
        }
        receipt.producer_sequence = command->producer_sequence;
        receipt.command_id = command->command_id;
        receipt.source_sample_id = command->source_sample_id;
        receipt.source_timestamp_ns = command->source_timestamp_ns;
        receipt.task_time_ns = command->task_time_ns;
        receipt.full_task_anchor = command->full_task_anchor;
        receipt.command_timestamp_ns = command->monotonic_timestamp_ns;
        receipt.expires_timestamp_ns = command->expires_timestamp_ns;
        receipt.session_nonce = command->session_nonce;
        receipt.task_epoch_id = command->task_epoch_id;
        receipt.safety_policy_id = command->safety_policy_id;
        receipt.requested_command_mode = command->mode;
        receipt.requested_active_mask = command->active_mask;
        receipt.requested_arm_weight = command->arm_weight;
        receipt.safety_policy_sha256 = command->safety_policy_sha256;
    }
    return receipt;
}

}  // namespace unitree_arm_adapter
