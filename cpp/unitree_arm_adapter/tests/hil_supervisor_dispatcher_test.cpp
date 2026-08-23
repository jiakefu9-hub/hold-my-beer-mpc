#include <cstdint>
#include <iostream>

#include "unitree_arm_adapter/hil_supervisor_dispatcher.hpp"

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
            ++failures;                                                         \
        }                                                                       \
    } while (false)

constexpr std::uint64_t kNow = 1'000'000'000U;
constexpr std::uint64_t kSession = 77U;
constexpr std::uint64_t kPolicy = 91U;

ua::Sha256Digest Digest() {
    ua::Sha256Digest digest{};
    digest.fill(0xa5U);
    return digest;
}

hs::SupervisorPolicy Policy() {
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
    policy.safety_policy_id = kPolicy;
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

struct StatePair {
    ua::RobotStatePayload payload{};
    hs::StateSample supervisor{};
};

StatePair State(std::uint64_t sample_id, std::uint64_t age_ns) {
    StatePair pair;
    pair.supervisor.validated = true;
    pair.supervisor.session_nonce = kSession;
    pair.supervisor.sample_id = sample_id;
    pair.supervisor.source_timestamp_ns = kNow - age_ns;
    pair.supervisor.validated_timestamp_ns =
        pair.supervisor.source_timestamp_ns + 100U;
    for (std::size_t slot = 0U; slot < hs::kCommandSlotCount; ++slot) {
        pair.supervisor.q[slot] =
            0.01 * static_cast<double>(sample_id + slot);
        pair.supervisor.dq[slot] = 0.0;
    }
    pair.payload.monotonic_timestamp_ns =
        pair.supervisor.source_timestamp_ns;
    pair.payload.validated_timestamp_ns =
        pair.supervisor.validated_timestamp_ns;
    pair.payload.ingress_session_nonce = kSession;
    pair.payload.sample_id = sample_id;
    return pair;
}

ua::ArmCommandPayload Command(
    const StatePair& source,
    std::uint64_t command_id,
    std::uint64_t anchor) {
    ua::ArmCommandPayload command;
    command.monotonic_timestamp_ns = kNow - 500U;
    command.producer_sequence = anchor;
    command.command_id = command_id;
    command.source_sample_id = source.supervisor.sample_id;
    command.source_timestamp_ns = source.supervisor.source_timestamp_ns;
    command.task_time_ns = anchor * hs::kMpcAnchorPeriodNs;
    command.full_task_anchor = anchor;
    command.expires_timestamp_ns = kNow + 15'000'000U;
    command.session_nonce = kSession;
    command.task_epoch_id = 5U;
    command.safety_policy_id = kPolicy;
    command.safety_policy_sha256 = Digest();
    command.mode = static_cast<std::uint32_t>(
        ua::CommandMode::kRobotPdPlusFeedforward);
    command.flags = ua::kCommandRequestArmingPd;
    command.active_mask = 0x3e0U;
    command.arm_weight = 0.1;
    command.q_ref = source.supervisor.q;
    for (std::size_t slot = 5U; slot < 10U; ++slot) {
        command.q_ref[slot] = 0.02;
        command.kp[slot] = 20.0;
        command.kd[slot] = 1.0;
    }
    return command;
}

hs::SupervisorSignals Signals() {
    hs::SupervisorSignals signals;
    signals.deadline_healthy = true;
    signals.ownership_confirmed = true;
    return signals;
}

void Observe(
    hil::HilSupervisorDispatcher& dispatcher,
    std::uint64_t sequence,
    const StatePair& state) {
    CHECK(dispatcher.ObserveState(
              sequence, state.payload, state.supervisor) ==
          hil::StateCacheObservation::kAdded);
}

void TestLaggedSourceTwoHoldsNewAnchorAndReplay() {
    hs::HardwareCommandSupervisor supervisor(Policy(), kSession);
    hil::HilSupervisorDispatcher dispatcher(supervisor, 8U);
    const auto source0 = State(1U, 5'000'000U);
    const auto latest0 = State(2U, 2'000'000U);
    Observe(dispatcher, 2U, source0);
    Observe(dispatcher, 4U, latest0);
    const auto command0 = Command(source0, 1U, 0U);

    const auto first = dispatcher.Dispatch(
        &command0, 2U, kNow, Signals());
    CHECK(first.new_command);
    CHECK(first.source_state_found);
    CHECK(first.latest_state_found);
    CHECK(first.supervisor.reason == hs::SupervisorReason::kAcceptedArmingPd);
    CHECK(first.supervisor.plan.q[0] == latest0.supervisor.q[0]);
    CHECK(first.supervisor.plan.q[5] == 0.02);

    const auto hold_at_2ms = dispatcher.Dispatch(
        &command0, 2U, kNow + 2'000'000U, Signals());
    CHECK(!hold_at_2ms.new_command);
    CHECK(hold_at_2ms.supervisor.reason ==
          hs::SupervisorReason::kAcceptedHeldCommand);
    const auto hold_at_4ms = dispatcher.Dispatch(
        &command0, 2U, kNow + 4'000'000U, Signals());
    CHECK(hold_at_4ms.supervisor.reason ==
          hs::SupervisorReason::kAcceptedHeldCommand);
    const auto fourth_tick = dispatcher.Dispatch(
        &command0, 2U, kNow + 6'000'000U, Signals());
    CHECK(fourth_tick.supervisor.reason ==
          hs::SupervisorReason::kCommandHoldExceeded);
    CHECK(fourth_tick.supervisor.plan.release_plan);
    CHECK(fourth_tick.supervisor.plan.arm_weight <
          first.supervisor.plan.arm_weight);

    hs::HardwareCommandSupervisor next_supervisor(Policy(), kSession);
    hil::HilSupervisorDispatcher next(next_supervisor, 8U);
    Observe(next, 2U, source0);
    Observe(next, 4U, latest0);
    CHECK(next.Dispatch(&command0, 2U, kNow, Signals()).supervisor.reason ==
          hs::SupervisorReason::kAcceptedArmingPd);
    CHECK(next.Dispatch(
                  &command0, 2U, kNow + 2'000'000U, Signals())
              .supervisor.reason == hs::SupervisorReason::kAcceptedHeldCommand);
    CHECK(next.Dispatch(
                  &command0, 2U, kNow + 4'000'000U, Signals())
              .supervisor.reason == hs::SupervisorReason::kAcceptedHeldCommand);
    const auto source1 = State(3U, 1'000'000U);
    Observe(next, 6U, source1);
    const auto command1 = Command(source1, 2U, 1U);
    const auto next_anchor = next.Dispatch(
        &command1, 4U, kNow + 6'000'000U, Signals());
    CHECK(next_anchor.new_command);
    CHECK(next_anchor.source_state_found);
    CHECK(next_anchor.supervisor.reason ==
          hs::SupervisorReason::kAcceptedArmingPd);

    const auto rewritten_old = next.Dispatch(
        &command0, 6U, kNow + 8'000'000U, Signals());
    CHECK(rewritten_old.new_command);
    CHECK(rewritten_old.supervisor.reason ==
          hs::SupervisorReason::kProposalReplayOrRegression);
    CHECK(rewritten_old.supervisor.state == hs::LifecycleState::kLatchedFault);
}

void TestMissingOrWrongSourceFailsClosed() {
    const auto available = State(2U, 2'000'000U);
    const auto missing = State(1U, 5'000'000U);
    const auto command = Command(missing, 1U, 0U);

    hs::HardwareCommandSupervisor missing_supervisor(Policy(), kSession);
    hil::HilSupervisorDispatcher missing_dispatcher(missing_supervisor, 8U);
    Observe(missing_dispatcher, 2U, available);
    const auto missing_result = missing_dispatcher.Dispatch(
        &command, 2U, kNow, Signals());
    CHECK(!missing_result.source_state_found);
    CHECK(missing_result.supervisor.reason ==
          hs::SupervisorReason::kSourceBindingMismatch);
    CHECK(!missing_result.supervisor.plan.write_permitted);

    hs::HardwareCommandSupervisor wrong_supervisor(Policy(), kSession);
    hil::HilSupervisorDispatcher wrong_dispatcher(wrong_supervisor, 8U);
    Observe(wrong_dispatcher, 2U, missing);
    auto wrong = command;
    wrong.source_timestamp_ns += 1U;
    const auto wrong_result = wrong_dispatcher.Dispatch(
        &wrong, 2U, kNow, Signals());
    CHECK(!wrong_result.source_state_found);
    CHECK(wrong_result.supervisor.reason ==
          hs::SupervisorReason::kSourceBindingMismatch);
    CHECK(!wrong_result.supervisor.plan.write_permitted);
}

}  // namespace

int main() {
    TestLaggedSourceTwoHoldsNewAnchorAndReplay();
    TestMissingOrWrongSourceFailsClosed();
    if (failures != 0) {
        return 1;
    }
    std::cout << "HIL supervisor dispatcher tests passed.\n";
    return 0;
}
