#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "unitree_arm_adapter/hil_recording_sink.hpp"
#include "unitree_arm_adapter/periodic_loop.hpp"
#include "unitree_arm_adapter/receipt.hpp"

namespace hil = unitree_arm_adapter::hil;

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __FILE__ << ':' << __LINE__                            \
                      << " CHECK failed: " #condition << '\n';                 \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

hil::ReceiptRecord ValidReceipt() {
    hil::ReceiptRecord item;
    item.receipt_id = 1;
    item.receipt_timestamp_ns = 2'000'000;
    item.command_id = 7;
    item.source_sample_id = 11;
    item.source_timestamp_ns = 1'000'000;
    item.observed_state_sample_id = 11;
    item.observed_state_timestamp_ns = 1'000'000;
    item.task_epoch_id = 3;
    item.task_time_ns = 24'000'000;
    item.full_task_anchor = 4;
    item.safety_policy_id = 19;
    item.safety_policy_sha256.fill(0xabU);
    item.pre_sink_check_timestamp_ns = 1'900'000;
    item.pre_sink_deadline_ns = 2'100'000;
    item.requested_command_mode = 2;
    item.guard_reason = 24;
    item.requested_flags = 5;
    item.requested_active_mask = 0x3E0U;
    item.executed_active_mask = 0x3E0U;
    item.requested_arm_weight = 0.1;
    item.executed_arm_weight = 0.1;
    item.q.fill(0.2);
    item.tau.fill(0.3);
    item.command_snapshot_valid = true;
    item.state_snapshot_valid = true;
    item.final_deadline_healthy = true;
    item.final_expiry_healthy = true;
    item.ready_for_sink = true;
    item.supervisor_write_permitted = true;
    item.release_plan = false;
    return item;
}

void TestRecordingAndJson() {
    hil::RecordingSink sink;
    sink.Record(ValidReceipt());
    CHECK(sink.records().size() == 1U);
    CHECK(sink.records().front().receipt_sink_write_performed);
    CHECK(!sink.records().front().transport_write_performed);
    CHECK(!sink.records().front().hardware_output_performed);
    const std::string path =
        "/tmp/unitree_arm_hil_receipt_" + std::to_string(::getpid()) + ".jsonl";
    sink.WriteJsonLines(path);
    std::ifstream stream(path);
    const std::string text{
        std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    CHECK(text.find("\"source_sample_id\":11") != std::string::npos);
    CHECK(text.find("\"observed_state_sample_id\":11") != std::string::npos);
    CHECK(text.find("\"full_task_anchor\":4") != std::string::npos);
    CHECK(text.find("\"safety_policy_id\":19") != std::string::npos);
    CHECK(text.find(
              "\"safety_policy_sha256\":\"abababababababababababababababab"
              "abababababababababababababababab\"") != std::string::npos);
    CHECK(text.find("\"executed_active_mask\":992") != std::string::npos);
    CHECK(text.find("\"guard_reason\":24") != std::string::npos);
    CHECK(text.find("\"command_clamped\":false") != std::string::npos);
    CHECK(text.find("\"transport_write_performed\":false") != std::string::npos);
    CHECK(text.find("\"hardware_output_performed\":false") != std::string::npos);
    CHECK(text.find("\"receipt_sink_write_performed\":true") !=
          std::string::npos);
    std::remove(path.c_str());
}

void TestFailClosedReceiptClaims() {
    hil::RecordingSink sink;
    auto item = ValidReceipt();
    item.transport_write_performed = true;
    bool rejected = false;
    try {
        sink.Record(item);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);

    item = ValidReceipt();
    sink.Record(item);
    item.receipt_id = 1;
    rejected = false;
    try {
        sink.Record(item);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
}

void TestMaskEncoding() {
    std::array<bool, unitree_arm_adapter::hardware_supervisor::kCommandSlotCount>
        mask{};
    for (std::size_t index = 5; index < 10; ++index) {
        mask[index] = true;
    }
    CHECK(hil::EncodeActiveMask(mask) == 0x3E0U);
}

void TestWouldWriteCommandSinkIsSeparateAndFailClosed() {
    hil::RecordingCommandSink sink;
    unitree_arm_adapter::hardware_supervisor::HardwareCommandPlan rejected;
    rejected.ready_for_sink = false;
    rejected.write_permitted = false;
    unitree_arm_adapter::ArmCommandPayload command;
    const std::uint64_t now_ns = unitree_arm_adapter::MonotonicNowNs();
    command.monotonic_timestamp_ns = now_ns - 1U;
    command.expires_timestamp_ns = now_ns + 1'000'000'000U;
    CHECK(!sink.SubmitIfCertified(
                   rejected, &command, now_ns + 1'000'000'000U)
               .performed);
    CHECK(sink.records().empty());

    auto accepted = rejected;
    accepted.ready_for_sink = true;
    accepted.write_permitted = true;
    accepted.release_plan = false;
    accepted.arm_weight = 0.2;
    accepted.producer_sequence = 1U;
    accepted.proposal_id = 2U;
    accepted.source_sample_id = 3U;
    accepted.task_epoch_id = 4U;
    accepted.task_time_ns = 24'000'000U;
    accepted.full_task_anchor = 4U;
    accepted.safety_policy_id = 5U;
    accepted.safety_policy_sha256.fill(0xa5U);
    accepted.active_mask[5] = true;
    accepted.tau[5] = 1.25;
    command.producer_sequence = accepted.producer_sequence;
    command.command_id = accepted.proposal_id;
    command.source_sample_id = accepted.source_sample_id;
    command.task_epoch_id = accepted.task_epoch_id;
    command.task_time_ns = accepted.task_time_ns;
    command.full_task_anchor = accepted.full_task_anchor;
    command.safety_policy_id = accepted.safety_policy_id;
    command.safety_policy_sha256 = accepted.safety_policy_sha256;
    command.arm_weight = accepted.arm_weight;
    command.active_mask = 1U << 5U;
    CHECK(!sink.SubmitIfCertified(accepted, &command, 1U).performed);
    auto expired = command;
    expired.expires_timestamp_ns = now_ns - 1U;
    CHECK(!sink.SubmitIfCertified(
                   accepted, &expired, now_ns + 1'000'000'000U)
               .performed);
    CHECK(sink.records().empty());
    CHECK(sink.SubmitIfCertified(
                  accepted, &command, now_ns + 1'000'000'000U)
              .performed);
    CHECK(sink.records().size() == 1U);
    CHECK(sink.records().front().plan.active_mask[5]);
    CHECK(sink.records().front().plan.arm_weight == 0.2);
    CHECK(sink.records().front().plan.tau[5] == 1.25);
    CHECK(sink.records().front().command_id == 2U);
    CHECK(sink.records().front().source_sample_id == 3U);
}

void TestAcceptedReceiptMatchesWouldWriteRecord() {
    const std::uint64_t now_ns = unitree_arm_adapter::MonotonicNowNs();
    unitree_arm_adapter::ArmCommandPayload command;
    command.monotonic_timestamp_ns = now_ns - 100U;
    command.expires_timestamp_ns = now_ns + 1'000'000'000U;
    command.producer_sequence = 7U;
    command.command_id = 8U;
    command.source_sample_id = 9U;
    command.source_timestamp_ns = now_ns - 500U;
    command.task_epoch_id = 10U;
    command.task_time_ns = 24'000'000U;
    command.full_task_anchor = 4U;
    command.safety_policy_id = 11U;
    command.safety_policy_sha256.fill(0x5aU);
    command.active_mask = 0x3e0U;
    command.arm_weight = 0.25;

    unitree_arm_adapter::hardware_supervisor::SupervisorResult result;
    result.state = unitree_arm_adapter::hardware_supervisor::LifecycleState::
        kActive;
    result.reason = unitree_arm_adapter::hardware_supervisor::SupervisorReason::
        kAcceptedActive;
    auto& plan = result.plan;
    plan.ready_for_sink = true;
    plan.write_permitted = true;
    plan.release_plan = false;
    plan.producer_sequence = command.producer_sequence;
    plan.proposal_id = command.command_id;
    plan.source_sample_id = command.source_sample_id;
    plan.task_epoch_id = command.task_epoch_id;
    plan.task_time_ns = command.task_time_ns;
    plan.full_task_anchor = command.full_task_anchor;
    plan.safety_policy_id = command.safety_policy_id;
    plan.safety_policy_sha256 = command.safety_policy_sha256;
    plan.arm_weight = command.arm_weight;
    for (std::size_t slot = 0U; slot < plan.active_mask.size(); ++slot) {
        plan.active_mask[slot] = (command.active_mask & (1U << slot)) != 0U;
        plan.q[slot] = 0.1 + static_cast<double>(slot);
        plan.dq[slot] = 0.2 + static_cast<double>(slot);
        plan.ddq_des[slot] = 0.3 + static_cast<double>(slot);
        plan.kp[slot] = 0.4 + static_cast<double>(slot);
        plan.kd[slot] = 0.5 + static_cast<double>(slot);
        plan.tau[slot] = 0.6 + static_cast<double>(slot);
    }

    hil::RecordingCommandSink command_sink;
    const auto attempt = command_sink.SubmitIfCertified(
        plan, &command, now_ns + 1'000'000'000U);
    CHECK(attempt.performed);
    CHECK(command_sink.records().size() == 1U);

    unitree_arm_adapter::AdapterStatusPayload status;
    status.receipt_id = 12U;
    status.monotonic_timestamp_ns = attempt.check_timestamp_ns;
    status.command_timestamp_ns = command.monotonic_timestamp_ns;
    status.expires_timestamp_ns = command.expires_timestamp_ns;
    status.command_id = command.command_id;
    status.producer_sequence = command.producer_sequence;
    status.session_nonce = 13U;
    status.source_sample_id = command.source_sample_id;
    status.source_timestamp_ns = command.source_timestamp_ns;
    status.task_epoch_id = command.task_epoch_id;
    status.task_time_ns = command.task_time_ns;
    status.full_task_anchor = command.full_task_anchor;
    status.safety_policy_id = command.safety_policy_id;
    status.safety_policy_sha256 = command.safety_policy_sha256;
    status.requested_active_mask = command.active_mask;
    status.executed_active_mask = hil::EncodeActiveMask(plan.active_mask);
    status.requested_arm_weight = command.arm_weight;
    status.executed_arm_weight = plan.arm_weight;
    status.selected_q = plan.q;
    status.selected_dq = plan.dq;
    status.selected_ddq_des = plan.ddq_des;
    status.selected_kp = plan.kp;
    status.selected_kd = plan.kd;
    status.selected_tau = plan.tau;
    const auto receipt = hil::MakeReceiptRecord(
        status,
        result,
        now_ns,
        100U,
        0U,
        true,
        true,
        true,
        true,
        true,
        false,
        true,
        true,
        attempt.performed);
    const auto& written = command_sink.records().front();
    CHECK(receipt.command_id == written.command_id);
    CHECK(receipt.producer_sequence == written.producer_sequence);
    CHECK(receipt.source_sample_id == written.source_sample_id);
    CHECK(receipt.source_timestamp_ns == written.source_timestamp_ns);
    CHECK(receipt.task_epoch_id == written.task_epoch_id);
    CHECK(receipt.task_time_ns == written.task_time_ns);
    CHECK(receipt.full_task_anchor == written.full_task_anchor);
    CHECK(receipt.safety_policy_id == written.safety_policy_id);
    CHECK(receipt.executed_active_mask ==
          hil::EncodeActiveMask(written.plan.active_mask));
    CHECK(receipt.executed_arm_weight == written.plan.arm_weight);
    CHECK(receipt.q == written.plan.q);
    CHECK(receipt.dq == written.plan.dq);
    CHECK(receipt.ddq_des == written.plan.ddq_des);
    CHECK(receipt.kp == written.plan.kp);
    CHECK(receipt.kd == written.plan.kd);
    CHECK(receipt.tau == written.plan.tau);
}

void TestRejectedFinalGateIsNotAnAcceptedAbiSinkWrite() {
    unitree_arm_adapter::ArmCommandPayload command;
    command.command_id = 8U;
    command.active_mask = 0x3e0U;
    command.arm_weight = 0.25;
    unitree_arm_adapter::hardware_supervisor::SupervisorResult result;
    result.reason = unitree_arm_adapter::hardware_supervisor::SupervisorReason::
        kAcceptedActive;
    result.plan.ready_for_sink = true;
    result.plan.write_permitted = true;
    result.plan.arm_weight = command.arm_weight;
    result.plan.active_mask[5] = true;
    unitree_arm_adapter::ReceiptContext context;
    context.receipt_timestamp_ns = 100U;
    context.receipt_id = 1U;
    context.command_snapshot_valid = true;
    context.guard_reason = static_cast<std::uint32_t>(
        unitree_arm_adapter::hardware_supervisor::SupervisorReason::
            kDeadlineMiss);
    context.sink_write_performed = false;
    context.sink_write_timestamp_ns = 99U;
    const auto receipt = unitree_arm_adapter::BuildAdapterReceipt(
        &command, nullptr, result, context);
    CHECK((receipt.flags & unitree_arm_adapter::kStatusSinkWritePerformed) ==
          0U);
    CHECK(receipt.sink_write_timestamp_ns == 0U);
    CHECK(receipt.receipt_reason == static_cast<std::uint32_t>(
              unitree_arm_adapter::ReceiptReason::kNone));
    CHECK(receipt.guard_reason == static_cast<std::uint32_t>(
              unitree_arm_adapter::hardware_supervisor::SupervisorReason::
                  kDeadlineMiss));
    CHECK(receipt.executed_active_mask == 0U);
    CHECK(receipt.executed_arm_weight == 0.0);
}

}  // namespace

int main() {
    TestRecordingAndJson();
    TestFailClosedReceiptClaims();
    TestMaskEncoding();
    TestWouldWriteCommandSinkIsSeparateAndFailClosed();
    TestAcceptedReceiptMatchesWouldWriteRecord();
    TestRejectedFinalGateIsNotAnAcceptedAbiSinkWrite();
    if (failures != 0) {
        return 1;
    }
    std::cout << "publisher-absent HIL recording sink tests passed.\n";
    return 0;
}
