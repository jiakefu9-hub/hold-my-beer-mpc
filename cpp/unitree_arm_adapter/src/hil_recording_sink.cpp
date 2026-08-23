#include "unitree_arm_adapter/hil_recording_sink.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "unitree_arm_adapter/periodic_loop.hpp"

namespace unitree_arm_adapter::hil {
namespace {

template <std::size_t Size>
void WriteArray(
    std::ostream& stream, const std::array<double, Size>& values) {
    stream << '[';
    for (std::size_t index = 0; index < Size; ++index) {
        if (index != 0U) {
            stream << ',';
        }
        stream << values[index];
    }
    stream << ']';
}

void WriteSha256(std::ostream& stream, const Sha256Digest& digest) {
    const auto original_flags = stream.flags();
    const auto original_fill = stream.fill();
    stream << '"' << std::hex << std::setfill('0');
    for (const std::uint8_t byte : digest) {
        stream << std::setw(2) << static_cast<unsigned int>(byte);
    }
    stream << '"';
    stream.flags(original_flags);
    stream.fill(original_fill);
}

bool Finite(const ReceiptRecord& receipt) noexcept {
    if (!std::isfinite(receipt.requested_arm_weight) ||
        !std::isfinite(receipt.executed_arm_weight)) {
        return false;
    }
    for (const auto* values : {
             &receipt.q,
             &receipt.dq,
             &receipt.ddq_des,
             &receipt.kp,
             &receipt.kd,
             &receipt.tau}) {
        for (const double value : *values) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

std::uint32_t EncodeActiveMask(
    const std::array<bool, hardware_supervisor::kCommandSlotCount>& mask) noexcept {
    std::uint32_t encoded = 0U;
    for (std::size_t index = 0; index < mask.size(); ++index) {
        if (mask[index]) {
            encoded |= std::uint32_t{1U} << index;
        }
    }
    return encoded;
}

ReceiptRecord MakeReceiptRecord(
    const AdapterStatusPayload& status,
    const hardware_supervisor::SupervisorResult& result,
    std::uint64_t scheduled_time_ns,
    std::uint64_t work_before_sink_ns,
    std::uint32_t requested_flags,
    bool command_snapshot_valid,
    bool state_snapshot_valid,
    bool final_deadline_healthy,
    bool final_expiry_healthy,
    bool paired_ingress_attested,
    bool synthetic_fixture_input,
    bool offline_fixture_policy,
    bool offline_ownership_attested,
    bool would_write_command_sink_performed) noexcept {
    ReceiptRecord record;
    record.receipt_id = status.receipt_id;
    record.receipt_timestamp_ns = status.monotonic_timestamp_ns;
    record.scheduled_time_ns = scheduled_time_ns;
    record.command_timestamp_ns = status.command_timestamp_ns;
    record.expires_timestamp_ns = status.expires_timestamp_ns;
    record.command_id = status.command_id;
    record.producer_sequence = status.producer_sequence;
    record.session_nonce = status.session_nonce;
    record.source_sample_id = status.source_sample_id;
    record.source_timestamp_ns = status.source_timestamp_ns;
    record.observed_state_sample_id = status.observed_state_sample_id;
    record.observed_state_timestamp_ns =
        status.observed_state_timestamp_ns;
    record.task_epoch_id = status.task_epoch_id;
    record.task_time_ns = status.task_time_ns;
    record.full_task_anchor = status.full_task_anchor;
    record.safety_policy_id = status.safety_policy_id;
    record.safety_policy_sha256 = status.safety_policy_sha256;
    record.wake_lateness_ns = status.wake_lateness_ns;
    record.work_before_sink_ns = work_before_sink_ns;
    record.pre_sink_check_timestamp_ns =
        status.pre_sink_check_timestamp_ns;
    record.pre_sink_deadline_ns = status.pre_sink_deadline_ns;
    record.requested_command_mode = status.requested_command_mode;
    record.requested_flags = requested_flags;
    record.requested_semantics = status.requested_command_mode;
    record.executed_semantics =
        static_cast<std::uint32_t>(result.plan.semantics);
    record.lifecycle_state = static_cast<std::uint32_t>(result.state);
    record.reason = static_cast<std::uint32_t>(result.reason);
    record.guard_reason = status.guard_reason;
    record.requested_active_mask = status.requested_active_mask;
    record.executed_active_mask = status.executed_active_mask;
    record.requested_arm_weight = status.requested_arm_weight;
    record.executed_arm_weight = status.executed_arm_weight;
    record.q = status.selected_q;
    record.dq = status.selected_dq;
    record.ddq_des = status.selected_ddq_des;
    record.kp = status.selected_kp;
    record.kd = status.selected_kd;
    record.tau = status.selected_tau;
    record.command_snapshot_valid = command_snapshot_valid;
    record.state_snapshot_valid = state_snapshot_valid;
    record.final_deadline_healthy = final_deadline_healthy;
    record.final_expiry_healthy = final_expiry_healthy;
    record.ready_for_sink = result.plan.ready_for_sink &&
        final_deadline_healthy && final_expiry_healthy;
    record.supervisor_write_permitted = result.plan.write_permitted &&
        final_deadline_healthy && final_expiry_healthy;
    record.release_plan = result.plan.release_plan;
    record.command_clamped =
        (status.flags & kStatusCommandClamped) != 0U;
    record.paired_ingress_attested = paired_ingress_attested;
    record.synthetic_fixture_input = synthetic_fixture_input;
    record.offline_fixture_policy = offline_fixture_policy;
    record.offline_ownership_attested = offline_ownership_attested;
    record.would_write_command_sink_performed =
        would_write_command_sink_performed;
    return record;
}

void RecordingSink::Record(const ReceiptRecord& receipt) {
    if (receipt.receipt_id == 0U || receipt.receipt_timestamp_ns == 0U) {
        throw std::invalid_argument("HIL receipt identity must be positive");
    }
    if (!Finite(receipt)) {
        throw std::invalid_argument("HIL receipt contains nonfinite command data");
    }
    if (receipt.transport_write_performed || receipt.hardware_output_performed) {
        throw std::invalid_argument(
            "publisher-absent HIL receipt cannot claim an external write");
    }
    if (!records_.empty() && receipt.receipt_id <= records_.back().receipt_id) {
        throw std::invalid_argument("HIL receipt id repeated or regressed");
    }
    records_.push_back(receipt);
    records_.back().receipt_sink_write_performed = true;
}

void RecordingSink::WriteJsonLines(const std::string& path) const {
    if (path.empty()) {
        throw std::invalid_argument("HIL JSONL path must be nonempty");
    }
    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open HIL JSONL output: " + path);
    }
    stream.precision(17);
    for (const auto& item : records_) {
        stream
            << "{\"schema\":\"unitree_arm_publisher_absent_hil_receipt_v1\""
            << ",\"receipt_id\":" << item.receipt_id
            << ",\"receipt_timestamp_ns\":" << item.receipt_timestamp_ns
            << ",\"scheduled_time_ns\":" << item.scheduled_time_ns
            << ",\"command_timestamp_ns\":" << item.command_timestamp_ns
            << ",\"expires_timestamp_ns\":" << item.expires_timestamp_ns
            << ",\"command_id\":" << item.command_id
            << ",\"producer_sequence\":" << item.producer_sequence
            << ",\"session_nonce\":" << item.session_nonce
            << ",\"source_sample_id\":" << item.source_sample_id
            << ",\"source_timestamp_ns\":" << item.source_timestamp_ns
            << ",\"observed_state_sample_id\":"
            << item.observed_state_sample_id
            << ",\"observed_state_timestamp_ns\":"
            << item.observed_state_timestamp_ns
            << ",\"task_epoch_id\":" << item.task_epoch_id
            << ",\"task_time_ns\":" << item.task_time_ns
            << ",\"full_task_anchor\":" << item.full_task_anchor
            << ",\"safety_policy_id\":" << item.safety_policy_id
            << ",\"safety_policy_sha256\":";
        WriteSha256(stream, item.safety_policy_sha256);
        stream
            << ",\"wake_lateness_ns\":" << item.wake_lateness_ns
            << ",\"work_before_sink_ns\":" << item.work_before_sink_ns
            << ",\"pre_sink_check_timestamp_ns\":"
            << item.pre_sink_check_timestamp_ns
            << ",\"pre_sink_deadline_ns\":" << item.pre_sink_deadline_ns
            << ",\"requested_command_mode\":"
            << item.requested_command_mode
            << ",\"requested_flags\":" << item.requested_flags
            << ",\"requested_semantics\":" << item.requested_semantics
            << ",\"executed_semantics\":" << item.executed_semantics
            << ",\"lifecycle_state\":" << item.lifecycle_state
            << ",\"reason\":" << item.reason
            << ",\"guard_reason\":" << item.guard_reason
            << ",\"requested_active_mask\":" << item.requested_active_mask
            << ",\"executed_active_mask\":" << item.executed_active_mask
            << ",\"requested_arm_weight\":" << item.requested_arm_weight
            << ",\"executed_arm_weight\":" << item.executed_arm_weight
            << ",\"q\":";
        WriteArray(stream, item.q);
        stream << ",\"dq\":";
        WriteArray(stream, item.dq);
        stream << ",\"ddq_des\":";
        WriteArray(stream, item.ddq_des);
        stream << ",\"kp\":";
        WriteArray(stream, item.kp);
        stream << ",\"kd\":";
        WriteArray(stream, item.kd);
        stream << ",\"tau\":";
        WriteArray(stream, item.tau);
        stream
            << ",\"command_snapshot_valid\":"
            << (item.command_snapshot_valid ? "true" : "false")
            << ",\"state_snapshot_valid\":"
            << (item.state_snapshot_valid ? "true" : "false")
            << ",\"final_deadline_healthy\":"
            << (item.final_deadline_healthy ? "true" : "false")
            << ",\"final_expiry_healthy\":"
            << (item.final_expiry_healthy ? "true" : "false")
            << ",\"ready_for_sink\":"
            << (item.ready_for_sink ? "true" : "false")
            << ",\"supervisor_write_permitted\":"
            << (item.supervisor_write_permitted ? "true" : "false")
            << ",\"release_plan\":"
            << (item.release_plan ? "true" : "false")
            << ",\"command_clamped\":"
            << (item.command_clamped ? "true" : "false")
            << ",\"paired_ingress_attested\":"
            << (item.paired_ingress_attested ? "true" : "false")
            << ",\"synthetic_fixture_input\":"
            << (item.synthetic_fixture_input ? "true" : "false")
            << ",\"offline_fixture_policy\":"
            << (item.offline_fixture_policy ? "true" : "false")
            << ",\"offline_ownership_attested\":"
            << (item.offline_ownership_attested ? "true" : "false")
            << ",\"receipt_sink_write_performed\":true"
            << ",\"would_write_command_sink_performed\":"
            << (item.would_write_command_sink_performed ? "true" : "false")
            << ",\"transport_write_performed\":false"
            << ",\"hardware_output_performed\":false}\n";
    }
    if (!stream) {
        throw std::runtime_error("failed while writing HIL JSONL output: " + path);
    }
}

RecordingCommandSink::Attempt RecordingCommandSink::SubmitIfCertified(
    const hardware_supervisor::HardwareCommandPlan& plan,
    const ArmCommandPayload* command,
    std::uint64_t deadline_ns) {
    Attempt attempt;
    attempt.check_timestamp_ns = MonotonicNowNs();
    attempt.deadline_healthy = deadline_ns != 0U &&
        attempt.check_timestamp_ns <= deadline_ns;
    attempt.expiry_healthy = command != nullptr &&
        command->monotonic_timestamp_ns != 0U &&
        command->monotonic_timestamp_ns <= attempt.check_timestamp_ns &&
        command->expires_timestamp_ns >= attempt.check_timestamp_ns;
    if (command == nullptr || !plan.ready_for_sink ||
        !plan.write_permitted || !attempt.deadline_healthy ||
        !attempt.expiry_healthy) {
        return attempt;
    }
    const bool identity_matches =
        plan.producer_sequence == command->producer_sequence &&
        plan.proposal_id == command->command_id &&
        plan.source_sample_id == command->source_sample_id &&
        plan.task_epoch_id == command->task_epoch_id &&
        plan.task_time_ns == command->task_time_ns &&
        plan.full_task_anchor == command->full_task_anchor &&
        plan.safety_policy_id == command->safety_policy_id &&
        plan.safety_policy_sha256 == command->safety_policy_sha256 &&
        plan.arm_weight == command->arm_weight &&
        EncodeActiveMask(plan.active_mask) == command->active_mask;
    if (!identity_matches) {
        return attempt;
    }
    Record record;
    record.check_timestamp_ns = attempt.check_timestamp_ns;
    record.deadline_ns = deadline_ns;
    record.command_id = command->command_id;
    record.producer_sequence = command->producer_sequence;
    record.source_sample_id = command->source_sample_id;
    record.source_timestamp_ns = command->source_timestamp_ns;
    record.task_epoch_id = command->task_epoch_id;
    record.task_time_ns = command->task_time_ns;
    record.full_task_anchor = command->full_task_anchor;
    record.safety_policy_id = command->safety_policy_id;
    record.plan = plan;
    records_.push_back(record);
    attempt.performed = true;
    return attempt;
}

}  // namespace unitree_arm_adapter::hil
