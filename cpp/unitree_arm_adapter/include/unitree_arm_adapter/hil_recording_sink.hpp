#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "unitree_arm_adapter/hardware_command_supervisor.hpp"
#include "unitree_arm_adapter/protocol.hpp"

namespace unitree_arm_adapter::hil {

struct ReceiptRecord {
    std::uint64_t receipt_id{0};
    std::uint64_t receipt_timestamp_ns{0};
    std::uint64_t scheduled_time_ns{0};
    std::uint64_t command_timestamp_ns{0};
    std::uint64_t expires_timestamp_ns{0};
    std::uint64_t command_id{0};
    std::uint64_t producer_sequence{0};
    std::uint64_t session_nonce{0};
    std::uint64_t source_sample_id{0};
    std::uint64_t source_timestamp_ns{0};
    std::uint64_t observed_state_sample_id{0};
    std::uint64_t observed_state_timestamp_ns{0};
    std::uint64_t task_epoch_id{0};
    std::uint64_t task_time_ns{0};
    std::uint64_t full_task_anchor{0};
    std::uint64_t safety_policy_id{0};
    Sha256Digest safety_policy_sha256{};
    std::uint64_t wake_lateness_ns{0};
    std::uint64_t work_before_sink_ns{0};
    std::uint64_t pre_sink_check_timestamp_ns{0};
    std::uint64_t pre_sink_deadline_ns{0};
    std::uint32_t requested_command_mode{0};
    std::uint32_t requested_flags{0};
    std::uint32_t requested_semantics{0};
    std::uint32_t executed_semantics{0};
    std::uint32_t lifecycle_state{0};
    std::uint32_t reason{0};
    std::uint32_t guard_reason{0};
    std::uint32_t requested_active_mask{0};
    std::uint32_t executed_active_mask{0};
    double requested_arm_weight{0.0};
    double executed_arm_weight{0.0};
    std::array<double, hardware_supervisor::kCommandSlotCount> q{};
    std::array<double, hardware_supervisor::kCommandSlotCount> dq{};
    std::array<double, hardware_supervisor::kCommandSlotCount> ddq_des{};
    std::array<double, hardware_supervisor::kCommandSlotCount> kp{};
    std::array<double, hardware_supervisor::kCommandSlotCount> kd{};
    std::array<double, hardware_supervisor::kCommandSlotCount> tau{};
    bool command_snapshot_valid{false};
    bool state_snapshot_valid{false};
    bool final_deadline_healthy{false};
    bool final_expiry_healthy{false};
    bool ready_for_sink{false};
    bool supervisor_write_permitted{false};
    bool release_plan{true};
    bool command_clamped{false};
    bool paired_ingress_attested{false};
    bool synthetic_fixture_input{false};
    bool offline_fixture_policy{false};
    bool offline_ownership_attested{false};
    bool receipt_sink_write_performed{false};
    bool would_write_command_sink_performed{false};
    bool transport_write_performed{false};
    bool hardware_output_performed{false};
};

[[nodiscard]] std::uint32_t EncodeActiveMask(
    const std::array<bool, hardware_supervisor::kCommandSlotCount>& mask) noexcept;

// Converts the shared protocol-v3 receipt into the durable HIL record while
// retaining the supervisor-only lifecycle and would-write decision.
[[nodiscard]] ReceiptRecord MakeReceiptRecord(
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
    bool would_write_command_sink_performed) noexcept;

class RecordingSink {
public:
    void Record(const ReceiptRecord& receipt);
    void WriteJsonLines(const std::string& path) const;

    [[nodiscard]] const std::vector<ReceiptRecord>& records() const noexcept {
        return records_;
    }

private:
    std::vector<ReceiptRecord> records_{};
};

// Publisher-absent stand-in for the future command sink. Receipt recording is
// unconditional; this sink is invoked only after the final deadline/expiry
// gates and the supervisor's write_permitted decision all pass.
class RecordingCommandSink {
public:
    struct Attempt {
        std::uint64_t check_timestamp_ns{0};
        bool deadline_healthy{false};
        bool expiry_healthy{false};
        bool performed{false};
    };

    struct Record {
        std::uint64_t check_timestamp_ns{0};
        std::uint64_t deadline_ns{0};
        std::uint64_t command_id{0};
        std::uint64_t producer_sequence{0};
        std::uint64_t source_sample_id{0};
        std::uint64_t source_timestamp_ns{0};
        std::uint64_t task_epoch_id{0};
        std::uint64_t task_time_ns{0};
        std::uint64_t full_task_anchor{0};
        std::uint64_t safety_policy_id{0};
        hardware_supervisor::HardwareCommandPlan plan{};
    };

    [[nodiscard]] Attempt SubmitIfCertified(
        const hardware_supervisor::HardwareCommandPlan& plan,
        const ArmCommandPayload* command,
        std::uint64_t deadline_ns);

    [[nodiscard]] const std::vector<Record>& records() const noexcept {
        return records_;
    }

private:
    std::vector<Record> records_{};
};

}  // namespace unitree_arm_adapter::hil
