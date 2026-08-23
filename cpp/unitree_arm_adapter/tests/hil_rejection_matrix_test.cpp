#include <cstdint>
#include <iostream>

#include "unitree_arm_adapter/hil_recording_sink.hpp"
#include "unitree_arm_adapter/periodic_loop.hpp"
#include "unitree_arm_adapter/protocol_supervisor_adapter.hpp"

namespace ua = unitree_arm_adapter;
namespace hs = unitree_arm_adapter::hardware_supervisor;
namespace hil = unitree_arm_adapter::hil;

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __FILE__ << ':' << __LINE__                            \
                      << " CHECK failed: " #condition << '\n';                 \
            ++failures;                                                        \
        }                                                                       \
    } while (false)

constexpr std::uint64_t kSession = 77U;
constexpr std::uint64_t kPolicyId = 91U;

ua::Sha256Digest Digest() {
    ua::Sha256Digest digest{};
    digest.fill(0xa5U);
    return digest;
}

hs::SupervisorPolicy VerifiedPolicy() {
    hs::SupervisorPolicy policy;
    policy.site_policy_verified = true;
    policy.ownership_policy_verified = true;
    policy.startup_pd_verified = true;
    policy.active_control_verified = true;
    policy.release_behavior_verified = true;
    policy.output_authorized = true;
    policy.state_timeout_ns = 20'000'000U;
    policy.proposal_timeout_ns = 20'000'000U;
    policy.maximum_command_ticks = 3U;
    policy.maximum_arm_weight = 1.0;
    policy.maximum_weight_step_per_tick = 1.0;
    policy.release_weight_step_per_tick = 0.05;
    policy.safety_policy_id = kPolicyId;
    policy.safety_policy_sha256 = Digest();
    policy.limits.verified = true;
    policy.limits.q_min.fill(-3.0);
    policy.limits.q_max.fill(3.0);
    policy.limits.dq_abs_max.fill(10.0);
    policy.limits.kp_max.fill(100.0);
    policy.limits.kd_max.fill(20.0);
    policy.limits.tau_abs_max.fill(50.0);
    return policy;
}

hs::StateSample State(
    std::uint64_t sample_id,
    std::uint64_t now_ns,
    std::uint64_t session_nonce = kSession) {
    hs::StateSample state;
    state.validated = true;
    state.session_nonce = session_nonce;
    state.sample_id = sample_id;
    state.source_timestamp_ns = now_ns - 1'000'000U;
    state.validated_timestamp_ns = now_ns - 900'000U;
    for (std::size_t slot = 0U; slot < hs::kCommandSlotCount; ++slot) {
        state.q[slot] = 0.01 * static_cast<double>(slot);
    }
    return state;
}

ua::ArmCommandPayload Command(
    const hs::StateSample& source,
    std::uint64_t now_ns,
    std::uint64_t command_id = 1U,
    std::uint64_t anchor = 0U) {
    ua::ArmCommandPayload command;
    command.monotonic_timestamp_ns = now_ns - 100'000U;
    command.producer_sequence = anchor;
    command.command_id = command_id;
    command.source_sample_id = source.sample_id;
    command.source_timestamp_ns = source.source_timestamp_ns;
    command.task_time_ns = anchor * hs::kMpcAnchorPeriodNs;
    command.full_task_anchor = anchor;
    command.expires_timestamp_ns = now_ns + 100'000'000U;
    command.session_nonce = source.session_nonce;
    command.task_epoch_id = 5U;
    command.safety_policy_id = kPolicyId;
    command.safety_policy_sha256 = Digest();
    command.mode = static_cast<std::uint32_t>(
        ua::CommandMode::kRobotPdPlusFeedforward);
    command.flags = ua::kCommandRequestArmingPd;
    command.active_mask = 0x3e0U;
    command.arm_weight = 0.1;
    command.q_ref = source.q;
    for (std::size_t slot = 5U; slot < 10U; ++slot) {
        command.q_ref[slot] = 0.02;
        command.kp[slot] = 20.0;
        command.kd[slot] = 1.0;
    }
    return command;
}

hs::SupervisorSignals HealthySignals() {
    hs::SupervisorSignals signals;
    signals.deadline_healthy = true;
    signals.ownership_confirmed = true;
    return signals;
}

hs::SupervisorResult Evaluate(
    hs::HardwareCommandSupervisor& supervisor,
    const ua::ArmCommandPayload& command,
    const hs::StateSample& source,
    const hs::StateSample& actuation,
    std::uint64_t now_ns,
    hs::SupervisorSignals signals = HealthySignals()) {
    const hs::ControlProposal proposal = ua::ToSupervisorProposal(command);
    return supervisor.EvaluateNew(
        &proposal, &source, &actuation, now_ns, signals);
}

void CheckRejectedBeforeSink(
    const hs::SupervisorResult& result,
    const ua::ArmCommandPayload& command) {
    hil::RecordingCommandSink sink;
    const auto attempt = sink.SubmitIfCertified(
        result.plan,
        &command,
        ua::MonotonicNowNs() + 1'000'000'000U);
    CHECK(!attempt.performed);
    CHECK(sink.records().empty());
}

void TestAcceptedControlReachesOnlyTheFakeSink() {
    const std::uint64_t now_ns = ua::MonotonicNowNs();
    const auto state = State(1U, now_ns);
    const auto command = Command(state, now_ns);
    hs::HardwareCommandSupervisor supervisor(VerifiedPolicy(), kSession);
    const auto result = Evaluate(
        supervisor, command, state, state, now_ns);
    CHECK(result.reason == hs::SupervisorReason::kAcceptedArmingPd);
    hil::RecordingCommandSink sink;
    const auto attempt = sink.SubmitIfCertified(
        result.plan, &command, now_ns + 1'000'000'000U);
    CHECK(attempt.performed);
    CHECK(sink.records().size() == 1U);
}

void TestNegativeAcceptanceMatrixNeverReachesTheFakeSink() {
    const std::uint64_t now_ns = ua::MonotonicNowNs();
    const auto state = State(1U, now_ns);
    const auto base = Command(state, now_ns);

    {
        auto stale = state;
        stale.source_timestamp_ns =
            now_ns - VerifiedPolicy().state_timeout_ns - 1U;
        stale.validated_timestamp_ns = stale.source_timestamp_ns + 1U;
        auto command = Command(stale, now_ns);
        hs::HardwareCommandSupervisor supervisor(VerifiedPolicy(), kSession);
        const auto result = Evaluate(
            supervisor, command, stale, stale, now_ns);
        CHECK(result.reason == hs::SupervisorReason::kStateStaleOrFuture);
        CheckRejectedBeforeSink(result, command);
    }
    {
        hs::HardwareCommandSupervisor supervisor(VerifiedPolicy(), kSession);
        CHECK(Evaluate(supervisor, base, state, state, now_ns).plan.write_permitted);
        const auto state2 = State(2U, now_ns);
        auto replay = Command(state2, now_ns, 1U, 1U);
        const auto result = Evaluate(
            supervisor, replay, state2, state2, now_ns);
        CHECK(
            result.reason ==
            hs::SupervisorReason::kProposalReplayOrRegression);
        CheckRejectedBeforeSink(result, replay);
    }
    {
        const auto restarted_state = State(1U, now_ns, kSession + 1U);
        const auto restarted = Command(restarted_state, now_ns);
        hs::HardwareCommandSupervisor supervisor(VerifiedPolicy(), kSession);
        const auto result = Evaluate(
            supervisor,
            restarted,
            restarted_state,
            restarted_state,
            now_ns);
        CHECK(result.reason == hs::SupervisorReason::kStateSessionMismatch);
        CheckRejectedBeforeSink(result, restarted);
    }
    {
        auto wrong_anchor = base;
        wrong_anchor.task_time_ns = 1U;
        hs::HardwareCommandSupervisor supervisor(VerifiedPolicy(), kSession);
        const auto result = Evaluate(
            supervisor, wrong_anchor, state, state, now_ns);
        CHECK(result.reason == hs::SupervisorReason::kTaskAnchorMismatch);
        CheckRejectedBeforeSink(result, wrong_anchor);
    }
    {
        auto wrong_source = base;
        ++wrong_source.source_sample_id;
        hs::HardwareCommandSupervisor supervisor(VerifiedPolicy(), kSession);
        const auto result = Evaluate(
            supervisor, wrong_source, state, state, now_ns);
        CHECK(result.reason == hs::SupervisorReason::kSourceBindingMismatch);
        CheckRejectedBeforeSink(result, wrong_source);
    }
    {
        auto inactive_action = base;
        inactive_action.tau[0] = 1.0;
        hs::HardwareCommandSupervisor supervisor(VerifiedPolicy(), kSession);
        const auto result = Evaluate(
            supervisor, inactive_action, state, state, now_ns);
        CHECK(result.reason == hs::SupervisorReason::kInactiveSlotAction);
        CheckRejectedBeforeSink(result, inactive_action);
    }
    {
        hs::HardwareCommandSupervisor supervisor(VerifiedPolicy(), kSession);
        CHECK(Evaluate(supervisor, base, state, state, now_ns).plan.write_permitted);
        const auto state2 = State(2U, now_ns);
        auto double_pd = Command(state2, now_ns, 2U, 1U);
        double_pd.mode = static_cast<std::uint32_t>(
            ua::CommandMode::kDirectTorque);
        double_pd.flags = ua::kCommandRequestActive;
        const auto result = Evaluate(
            supervisor, double_pd, state2, state2, now_ns);
        CHECK(result.reason == hs::SupervisorReason::kDuplicateRobotPd);
        CheckRejectedBeforeSink(result, double_pd);
    }
    {
        auto expired = base;
        expired.monotonic_timestamp_ns = now_ns - 2'000'000U;
        expired.expires_timestamp_ns = now_ns - 1U;
        hs::HardwareCommandSupervisor supervisor(VerifiedPolicy(), kSession);
        const auto result = Evaluate(
            supervisor, expired, state, state, now_ns);
        CHECK(
            result.reason ==
            hs::SupervisorReason::kProposalExpiredOrFuture);
        CheckRejectedBeforeSink(result, expired);
    }
    {
        auto missed = HealthySignals();
        missed.deadline_healthy = false;
        hs::HardwareCommandSupervisor supervisor(VerifiedPolicy(), kSession);
        const auto result = Evaluate(
            supervisor, base, state, state, now_ns, missed);
        CHECK(result.reason == hs::SupervisorReason::kDeadlineMiss);
        CheckRejectedBeforeSink(result, base);
    }
    {
        hs::HardwareCommandSupervisor supervisor(
            hs::SupervisorPolicy{}, kSession);
        const auto result = Evaluate(
            supervisor, base, state, state, now_ns);
        CHECK(result.reason == hs::SupervisorReason::kSitePolicyUnverified);
        CheckRejectedBeforeSink(result, base);
    }
}

}  // namespace

int main() {
    TestAcceptedControlReachesOnlyTheFakeSink();
    TestNegativeAcceptanceMatrixNeverReachesTheFakeSink();
    if (failures != 0) {
        return 1;
    }
    std::cout << "HIL rejection matrix tests passed.\n";
    return 0;
}
