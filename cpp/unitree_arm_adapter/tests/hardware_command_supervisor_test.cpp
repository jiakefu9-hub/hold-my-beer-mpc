#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

#include "unitree_arm_adapter/hardware_command_supervisor.hpp"

namespace hs = unitree_arm_adapter::hardware_supervisor;

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

constexpr std::uint64_t kNow = 1'000'000'000ULL;
constexpr std::uint64_t kSession = 77ULL;
constexpr std::uint64_t kTaskEpoch = 9ULL;
constexpr std::uint64_t kPolicyId = 1234ULL;

hs::Sha256Digest PolicySha256() {
    hs::Sha256Digest digest{};
    for (std::size_t index = 0; index < digest.size(); ++index) {
        digest[index] = static_cast<std::uint8_t>(index + 1U);
    }
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
    policy.state_timeout_ns = 20'000'000ULL;
    policy.proposal_timeout_ns = 20'000'000ULL;
    policy.maximum_command_ticks = 3U;
    policy.maximum_arm_weight = 1.0;
    policy.maximum_weight_step_per_tick = 0.25;
    policy.release_weight_step_per_tick = 0.05;
    policy.safety_policy_id = kPolicyId;
    policy.safety_policy_sha256 = PolicySha256();
    policy.limits.verified = true;
    policy.limits.q_min.fill(-2.0);
    policy.limits.q_max.fill(2.0);
    policy.limits.dq_abs_max.fill(2.0);
    policy.limits.kp_max.fill(100.0);
    policy.limits.kd_max.fill(20.0);
    policy.limits.tau_abs_max.fill(50.0);
    return policy;
}

hs::StateSample ValidState(std::uint64_t sample_id = 1ULL) {
    hs::StateSample state;
    state.validated = true;
    state.session_nonce = kSession;
    state.sample_id = sample_id;
    state.source_timestamp_ns = kNow - 2'000'000ULL + sample_id;
    state.validated_timestamp_ns =
        state.source_timestamp_ns + 100'000ULL;
    for (std::size_t slot = 0; slot < hs::kCommandSlotCount; ++slot) {
        state.q[slot] = 0.01 * static_cast<double>(slot);
        state.dq[slot] = 0.001 * static_cast<double>(slot);
    }
    return state;
}

hs::ControlProposal ValidProposal(
    const hs::StateSample& state,
    std::uint64_t proposal_id = 1ULL,
    std::uint64_t anchor = 0ULL,
    hs::RequestedLifecycle lifecycle = hs::RequestedLifecycle::kArmingPd,
    hs::CommandSemantics semantics =
        hs::CommandSemantics::kRobotPdPlusFeedforward,
    double weight = 0.1) {
    hs::ControlProposal proposal;
    proposal.session_nonce = kSession;
    proposal.producer_sequence = anchor;
    proposal.proposal_id = proposal_id;
    proposal.source_sample_id = state.sample_id;
    proposal.source_timestamp_ns = state.source_timestamp_ns;
    proposal.task_epoch_id = kTaskEpoch;
    proposal.task_time_ns = anchor * hs::kMpcAnchorPeriodNs;
    proposal.full_task_anchor = anchor;
    proposal.generated_timestamp_ns = state.validated_timestamp_ns + 100'000ULL;
    proposal.expires_timestamp_ns = kNow + 5'000'000ULL;
    proposal.semantics = semantics;
    proposal.requested_lifecycle = lifecycle;
    proposal.arm_weight = weight;
    proposal.safety_policy_id = kPolicyId;
    proposal.safety_policy_sha256 = PolicySha256();
    proposal.q_ref = state.q;
    for (std::size_t slot = 5; slot < 10; ++slot) {
        proposal.active_mask[slot] = true;
        proposal.q_ref[slot] = 0.02;
        proposal.dq_ref[slot] = 0.01;
        proposal.ddq_des[slot] = 0.2;
        proposal.kp[slot] = semantics == hs::CommandSemantics::kDirectTorque
                                ? 0.0
                                : 20.0;
        proposal.kd[slot] = semantics == hs::CommandSemantics::kDirectTorque
                                ? 0.0
                                : 1.0;
        proposal.tau[slot] = 2.0;
    }
    return proposal;
}

hs::SupervisorSignals HealthySignals() {
    hs::SupervisorSignals signals;
    signals.deadline_healthy = true;
    signals.ownership_confirmed = true;
    return signals;
}

void TestUnverifiedPolicyCannotArm() {
    hs::SupervisorPolicy policy;
    hs::HardwareCommandSupervisor supervisor(policy, kSession);
    const auto state = ValidState();
    const auto proposal = ValidProposal(state);
    const auto result = supervisor.Evaluate(
        &proposal, &state, kNow, HealthySignals());
    CHECK(result.state == hs::LifecycleState::kDisarmed);
    CHECK(result.reason == hs::SupervisorReason::kSitePolicyUnverified);
    CHECK(!result.plan.ready_for_sink);
    CHECK(!result.plan.write_permitted);
    CHECK(result.plan.arm_weight == 0.0);

    policy = VerifiedPolicy();
    policy.output_authorized = false;
    hs::HardwareCommandSupervisor no_authorization(policy, kSession);
    const auto denied = no_authorization.Evaluate(
        &proposal, &state, kNow, HealthySignals());
    CHECK(denied.state == hs::LifecycleState::kDisarmed);
    CHECK(denied.reason == hs::SupervisorReason::kOutputAuthorizationMissing);
    CHECK(!denied.plan.write_permitted);

    policy = VerifiedPolicy();
    policy.ownership_policy_verified = false;
    hs::HardwareCommandSupervisor no_ownership(policy, kSession);
    const auto ownership_denied = no_ownership.Evaluate(
        &proposal, &state, kNow, HealthySignals());
    CHECK(
        ownership_denied.reason ==
        hs::SupervisorReason::kOwnershipUnverified);
    CHECK(ownership_denied.state == hs::LifecycleState::kDisarmed);
}

void TestArmingActiveAndThirteenSlotFormatting() {
    hs::HardwareCommandSupervisor supervisor(
        VerifiedPolicy(), kSession);
    const auto state1 = ValidState(1ULL);
    const auto arming = ValidProposal(state1);
    const auto arming_result = supervisor.Evaluate(
        &arming, &state1, kNow, HealthySignals());
    CHECK(arming_result.state == hs::LifecycleState::kArmingPd);
    CHECK(arming_result.reason == hs::SupervisorReason::kAcceptedArmingPd);
    CHECK(arming_result.plan.producer_sequence == 0ULL);
    CHECK(arming_result.plan.ready_for_sink);
    CHECK(arming_result.plan.write_permitted);
    CHECK(!arming_result.plan.release_plan);
    CHECK(arming_result.plan.q[4] == state1.q[4]);
    CHECK(arming_result.plan.kp[4] == 0.0);
    CHECK(arming_result.plan.tau[4] == 0.0);
    CHECK(arming_result.plan.q[5] == 0.02);
    CHECK(arming_result.plan.kp[5] == 20.0);

    for (std::uint64_t anchor = 1ULL; anchor < 4ULL; ++anchor) {
        const auto warm_state = ValidState(anchor + 1ULL);
        const auto warm = ValidProposal(
            warm_state,
            anchor + 1ULL,
            anchor,
            hs::RequestedLifecycle::kArmingPd,
            hs::CommandSemantics::kRobotPdPlusFeedforward,
            0.1);
        CHECK(supervisor.Evaluate(
                  &warm, &warm_state, kNow, HealthySignals())
                  .state == hs::LifecycleState::kArmingPd);
    }
    const auto state2 = ValidState(5ULL);
    const auto active = ValidProposal(
        state2,
        5ULL,
        4ULL,
        hs::RequestedLifecycle::kActive,
        hs::CommandSemantics::kDirectTorque,
        0.2);
    const auto active_result = supervisor.Evaluate(
        &active, &state2, kNow, HealthySignals());
    CHECK(active_result.state == hs::LifecycleState::kActive);
    CHECK(active_result.reason == hs::SupervisorReason::kAcceptedActive);
    CHECK(active_result.plan.semantics == hs::CommandSemantics::kDirectTorque);
    CHECK(active_result.plan.producer_sequence == 4ULL);
    CHECK(active_result.plan.safety_policy_id == kPolicyId);
    CHECK(active_result.plan.safety_policy_sha256 == PolicySha256());
    CHECK(active_result.plan.q[5] == state2.q[5]);
    CHECK(active_result.plan.dq[5] == 0.0);
    CHECK(active_result.plan.kp[5] == 0.0);
    CHECK(active_result.plan.kd[5] == 0.0);
    CHECK(active_result.plan.tau[5] == 2.0);

    const auto state3 = ValidState(6ULL);
    hs::SupervisorSignals ownership_lost = HealthySignals();
    ownership_lost.ownership_confirmed = false;
    const auto guarded = supervisor.Evaluate(
        nullptr, &state3, kNow, ownership_lost);
    CHECK(guarded.state == hs::LifecycleState::kSoftGuardReleasing);
    CHECK(guarded.reason == hs::SupervisorReason::kOwnershipLost);
    CHECK(std::abs(guarded.plan.arm_weight - 0.15) < 1e-12);
}

void TestIdentityReplaySourceAnchorAndExpiry() {
    const auto exercise = [](const hs::ControlProposal& proposal,
                             const hs::StateSample& state,
                             hs::SupervisorReason reason,
                             bool hard_fault) {
        hs::HardwareCommandSupervisor supervisor(
            VerifiedPolicy(), kSession);
        const auto result = supervisor.Evaluate(
            &proposal, &state, kNow, HealthySignals());
        CHECK(result.reason == reason);
        CHECK(result.state == (hard_fault
                                   ? hs::LifecycleState::kLatchedFault
                                   : hs::LifecycleState::kDisarmed));
        CHECK(!result.plan.write_permitted);
    };

    auto state = ValidState();
    auto proposal = ValidProposal(state);
    proposal.session_nonce = kSession + 1ULL;
    exercise(
        proposal,
        state,
        hs::SupervisorReason::kProposalSessionMismatch,
        true);

    proposal = ValidProposal(state);
    proposal.safety_policy_id = kPolicyId + 1ULL;
    exercise(
        proposal,
        state,
        hs::SupervisorReason::kSafetyPolicyIdentityMismatch,
        true);

    proposal = ValidProposal(state);
    proposal.safety_policy_sha256[7] ^= 0xffU;
    exercise(
        proposal,
        state,
        hs::SupervisorReason::kSafetyPolicyIdentityMismatch,
        true);

    proposal = ValidProposal(state);
    proposal.source_sample_id += 1ULL;
    exercise(
        proposal,
        state,
        hs::SupervisorReason::kSourceBindingMismatch,
        true);

    proposal = ValidProposal(state);
    proposal.task_time_ns += 1ULL;
    exercise(
        proposal,
        state,
        hs::SupervisorReason::kTaskAnchorMismatch,
        true);

    proposal = ValidProposal(state, 1ULL, 2ULL);
    exercise(
        proposal,
        state,
        hs::SupervisorReason::kTaskAnchorGapOrReplay,
        true);

    proposal = ValidProposal(state);
    proposal.expires_timestamp_ns = kNow - 1ULL;
    exercise(
        proposal,
        state,
        hs::SupervisorReason::kProposalExpiredOrFuture,
        false);

    hs::HardwareCommandSupervisor replay_supervisor(
        VerifiedPolicy(), kSession);
    const auto first = ValidProposal(state);
    const auto accepted = replay_supervisor.Evaluate(
        &first, &state, kNow, HealthySignals());
    CHECK(accepted.state == hs::LifecycleState::kArmingPd);
    const auto state2 = ValidState(2ULL);
    auto replay = ValidProposal(state2, 1ULL, 1ULL);
    const auto replay_result = replay_supervisor.Evaluate(
        &replay, &state2, kNow, HealthySignals());
    CHECK(
        replay_result.reason ==
        hs::SupervisorReason::kProposalReplayOrRegression);
    CHECK(replay_result.state == hs::LifecycleState::kLatchedFault);

    hs::HardwareCommandSupervisor epoch_supervisor(
        VerifiedPolicy(), kSession);
    CHECK(epoch_supervisor.Evaluate(
              &first, &state, kNow, HealthySignals())
              .state == hs::LifecycleState::kArmingPd);
    auto epoch_state = ValidState(2ULL);
    auto wrong_epoch = ValidProposal(epoch_state, 2ULL, 1ULL);
    wrong_epoch.task_epoch_id = kTaskEpoch + 1ULL;
    const auto epoch_result = epoch_supervisor.Evaluate(
        &wrong_epoch, &epoch_state, kNow, HealthySignals());
    CHECK(epoch_result.reason == hs::SupervisorReason::kTaskEpochMismatch);
    CHECK(epoch_result.state == hs::LifecycleState::kLatchedFault);

    hs::HardwareCommandSupervisor sequence_supervisor(
        VerifiedPolicy(), kSession);
    CHECK(sequence_supervisor.Evaluate(
              &first, &state, kNow, HealthySignals())
              .state == hs::LifecycleState::kArmingPd);
    auto sequence_state = ValidState(2ULL);
    auto sequence_gap = ValidProposal(sequence_state, 2ULL, 1ULL);
    sequence_gap.producer_sequence = 3ULL;
    const auto sequence_result = sequence_supervisor.Evaluate(
        &sequence_gap, &sequence_state, kNow, HealthySignals());
    CHECK(
        sequence_result.reason ==
        hs::SupervisorReason::kProposalReplayOrRegression);

    hs::HardwareCommandSupervisor restart_supervisor(
        VerifiedPolicy(), kSession);
    auto restart_state = ValidState();
    restart_state.session_nonce = kSession + 1ULL;
    const auto restart_proposal = ValidProposal(restart_state);
    const auto restart_result = restart_supervisor.Evaluate(
        &restart_proposal, &restart_state, kNow, HealthySignals());
    CHECK(
        restart_result.reason ==
        hs::SupervisorReason::kStateSessionMismatch);
    CHECK(restart_result.state == hs::LifecycleState::kLatchedFault);
}

void TestMaskFiniteDoublePdAndLimits() {
    const auto exercise = [](hs::ControlProposal proposal,
                             hs::SupervisorReason reason,
                             bool hard_fault) {
        const auto state = ValidState();
        hs::HardwareCommandSupervisor supervisor(
            VerifiedPolicy(), kSession);
        const auto result = supervisor.Evaluate(
            &proposal, &state, kNow, HealthySignals());
        CHECK(result.reason == reason);
        CHECK(result.state == (hard_fault
                                   ? hs::LifecycleState::kLatchedFault
                                   : hs::LifecycleState::kDisarmed));
        CHECK(!result.plan.write_permitted);
    };

    const auto state = ValidState();
    auto proposal = ValidProposal(state);
    proposal.tau[0] = 0.1;
    exercise(proposal, hs::SupervisorReason::kInactiveSlotAction, true);

    proposal = ValidProposal(
        state,
        1ULL,
        0ULL,
        hs::RequestedLifecycle::kArmingPd,
        hs::CommandSemantics::kDirectTorque);
    proposal.kp[5] = 1.0;
    const auto duplicate_pd = hs::FormatCommandPlan(
        VerifiedPolicy(), proposal, state);
    CHECK(!duplicate_pd.valid);
    CHECK(duplicate_pd.reason == hs::SupervisorReason::kDuplicateRobotPd);

    proposal = ValidProposal(state);
    proposal.tau[5] = 51.0;
    exercise(proposal, hs::SupervisorReason::kSiteLimitViolation, false);

    proposal = ValidProposal(state);
    proposal.tau[5] = std::numeric_limits<double>::quiet_NaN();
    exercise(proposal, hs::SupervisorReason::kInvalidProposal, true);

    proposal = ValidProposal(state);
    proposal.semantics = hs::CommandSemantics::kInvalid;
    exercise(proposal, hs::SupervisorReason::kInvalidModeOrLifecycle, true);

    proposal = ValidProposal(state);
    proposal.active_mask.fill(false);
    proposal.q_ref = state.q;
    proposal.dq_ref.fill(0.0);
    proposal.ddq_des.fill(0.0);
    proposal.kp.fill(0.0);
    proposal.kd.fill(0.0);
    proposal.tau.fill(0.0);
    exercise(proposal, hs::SupervisorReason::kInvalidActiveMask, true);

    auto invalid_state = state;
    invalid_state.q[0] = std::numeric_limits<double>::quiet_NaN();
    const auto invalid_state_format = hs::FormatCommandPlan(
        VerifiedPolicy(), ValidProposal(state), invalid_state);
    CHECK(!invalid_state_format.valid);
    CHECK(
        invalid_state_format.reason == hs::SupervisorReason::kInvalidState);
}

void TestDeadlineSoftReleaseAndLatchedReset() {
    hs::HardwareCommandSupervisor supervisor(
        VerifiedPolicy(), kSession);
    const auto state1 = ValidState(1ULL);
    const auto arming = ValidProposal(state1, 1ULL, 0ULL);
    CHECK(supervisor.Evaluate(
              &arming, &state1, kNow, HealthySignals())
              .state == hs::LifecycleState::kArmingPd);

    const auto state2 = ValidState(2ULL);
    hs::SupervisorSignals deadline = HealthySignals();
    deadline.deadline_healthy = false;
    const auto releasing = supervisor.Evaluate(
        nullptr, &state2, kNow, deadline);
    CHECK(releasing.state == hs::LifecycleState::kSoftGuardReleasing);
    CHECK(releasing.reason == hs::SupervisorReason::kDeadlineMiss);
    CHECK(releasing.plan.ready_for_sink);
    CHECK(releasing.plan.write_permitted);
    CHECK(std::abs(releasing.plan.arm_weight - 0.05) < 1e-12);

    const auto state3 = ValidState(3ULL);
    const auto released = supervisor.Evaluate(
        nullptr, &state3, kNow, deadline);
    CHECK(released.state == hs::LifecycleState::kDisarmed);
    CHECK(released.reason == hs::SupervisorReason::kSoftGuardComplete);
    CHECK(released.plan.arm_weight == 0.0);

    hs::HardwareCommandSupervisor hard_fault(
        VerifiedPolicy(), kSession);
    hs::SupervisorSignals hard = HealthySignals();
    hard.request_latched_fault = true;
    const auto latched = hard_fault.Evaluate(nullptr, &state1, kNow, hard);
    CHECK(latched.state == hs::LifecycleState::kLatchedFault);
    CHECK(!latched.plan.write_permitted);
    const auto still_latched = hard_fault.Evaluate(
        &arming, &state1, kNow, HealthySignals());
    CHECK(still_latched.state == hs::LifecycleState::kLatchedFault);
    hs::SupervisorSignals reset = HealthySignals();
    reset.reset_latched_fault = true;
    const auto reset_result = hard_fault.Evaluate(
        nullptr, &state1, kNow, reset);
    CHECK(reset_result.state == hs::LifecycleState::kDisarmed);
    CHECK(reset_result.reason == hs::SupervisorReason::kLatchedFaultReset);
    CHECK(!reset_result.plan.write_permitted);

    hs::HardwareCommandSupervisor requested_release(
        VerifiedPolicy(), kSession);
    CHECK(requested_release.Evaluate(
              &arming, &state1, kNow, HealthySignals())
              .state == hs::LifecycleState::kArmingPd);
    auto release_state = ValidState(2ULL);
    auto release = ValidProposal(
        release_state,
        2ULL,
        1ULL,
        hs::RequestedLifecycle::kDisarmed,
        hs::CommandSemantics::kRelease,
        0.0);
    release.active_mask.fill(false);
    release.q_ref = release_state.q;
    release.dq_ref.fill(0.0);
    release.ddq_des.fill(0.0);
    release.kp.fill(0.0);
    release.kd.fill(0.0);
    release.tau.fill(0.0);
    const auto requested = requested_release.Evaluate(
        &release, &release_state, kNow, HealthySignals());
    CHECK(requested.state == hs::LifecycleState::kSoftGuardReleasing);
    CHECK(requested.reason == hs::SupervisorReason::kSoftGuardRequested);
    CHECK(std::abs(requested.plan.arm_weight - 0.05) < 1e-12);
    CHECK(requested.plan.write_permitted);
}

void TestWeightStepAndStateFreshness() {
    auto policy = VerifiedPolicy();
    policy.maximum_weight_step_per_tick = 0.05;
    const auto state = ValidState();
    auto proposal = ValidProposal(state);
    proposal.arm_weight = 0.1;
    hs::HardwareCommandSupervisor supervisor(policy, kSession);
    const auto weight = supervisor.Evaluate(
        &proposal, &state, kNow, HealthySignals());
    CHECK(weight.reason == hs::SupervisorReason::kWeightStepViolation);
    CHECK(weight.state == hs::LifecycleState::kDisarmed);
    CHECK(!weight.plan.write_permitted);

    auto stale = ValidState();
    stale.source_timestamp_ns = kNow - policy.state_timeout_ns - 1ULL;
    stale.validated_timestamp_ns = stale.source_timestamp_ns + 1ULL;
    hs::HardwareCommandSupervisor stale_supervisor(
        policy, kSession);
    const auto stale_result = stale_supervisor.Evaluate(
        &proposal, &stale, kNow, HealthySignals());
    CHECK(stale_result.reason == hs::SupervisorReason::kStateStaleOrFuture);
    CHECK(stale_result.state == hs::LifecycleState::kDisarmed);

    auto future_validated = ValidState();
    future_validated.validated_timestamp_ns = kNow + 1ULL;
    auto future_proposal = ValidProposal(future_validated);
    future_proposal.generated_timestamp_ns = kNow + 2ULL;
    hs::HardwareCommandSupervisor future_supervisor(
        policy, kSession);
    const auto future_result = future_supervisor.Evaluate(
        &future_proposal,
        &future_validated,
        kNow,
        HealthySignals());
    CHECK(
        future_result.reason ==
        hs::SupervisorReason::kStateStaleOrFuture);
    CHECK(future_result.state == hs::LifecycleState::kDisarmed);
}

void TestLaggedSourceStateAndTwoMillisecondCommandHold() {
    hs::HardwareCommandSupervisor supervisor(
        VerifiedPolicy(), kSession);
    const auto source0 = ValidState(1ULL);
    auto actuation0 = ValidState(2ULL);
    actuation0.q[0] = 0.75;
    auto anchor0 = ValidProposal(source0, 1ULL, 0ULL);
    anchor0.expires_timestamp_ns = kNow + 20'000'000ULL;

    const auto accepted0 = supervisor.EvaluateNew(
        &anchor0, &source0, &actuation0, kNow, HealthySignals());
    CHECK(accepted0.reason == hs::SupervisorReason::kAcceptedArmingPd);
    CHECK(accepted0.plan.q[0] == actuation0.q[0]);
    CHECK(supervisor.command_tick_count() == 1U);

    auto actuation2ms = ValidState(3ULL);
    actuation2ms.q[0] = 0.80;
    const auto held2ms = supervisor.ContinueLast(
        &actuation2ms, kNow + 2'000'000ULL, HealthySignals());
    CHECK(held2ms.reason == hs::SupervisorReason::kAcceptedHeldCommand);
    CHECK(held2ms.plan.proposal_id == anchor0.proposal_id);
    CHECK(held2ms.plan.q[0] == actuation2ms.q[0]);
    CHECK(supervisor.command_tick_count() == 2U);

    auto actuation4ms = ValidState(4ULL);
    actuation4ms.q[0] = 0.85;
    const auto held4ms = supervisor.ContinueLast(
        &actuation4ms, kNow + 4'000'000ULL, HealthySignals());
    CHECK(held4ms.reason == hs::SupervisorReason::kAcceptedHeldCommand);
    CHECK(held4ms.plan.q[0] == actuation4ms.q[0]);
    CHECK(supervisor.command_tick_count() == 3U);

    const auto source6ms = ValidState(4ULL);
    auto actuation6ms = ValidState(5ULL);
    actuation6ms.q[0] = 0.90;
    auto anchor1 = ValidProposal(source6ms, 2ULL, 1ULL);
    anchor1.expires_timestamp_ns = kNow + 30'000'000ULL;
    const auto accepted6ms = supervisor.EvaluateNew(
        &anchor1,
        &source6ms,
        &actuation6ms,
        kNow + 6'000'000ULL,
        HealthySignals());
    CHECK(accepted6ms.reason == hs::SupervisorReason::kAcceptedArmingPd);
    CHECK(accepted6ms.plan.full_task_anchor == 1ULL);
    CHECK(accepted6ms.plan.q[0] == actuation6ms.q[0]);
    CHECK(supervisor.command_tick_count() == 1U);

    const auto rewritten_old = supervisor.EvaluateNew(
        &anchor0,
        &source0,
        &actuation6ms,
        kNow + 8'000'000ULL,
        HealthySignals());
    CHECK(
        rewritten_old.reason ==
        hs::SupervisorReason::kProposalReplayOrRegression);
    CHECK(rewritten_old.state == hs::LifecycleState::kLatchedFault);
    CHECK(!rewritten_old.plan.write_permitted);
}

void TestHoldLimitExpiryDeadlineAndOwnershipFailClosed() {
    const auto source = ValidState(1ULL);
    auto proposal = ValidProposal(source, 1ULL, 0ULL);
    proposal.expires_timestamp_ns = kNow + 20'000'000ULL;

    hs::HardwareCommandSupervisor limit_supervisor(
        VerifiedPolicy(), kSession);
    CHECK(limit_supervisor.EvaluateNew(
              &proposal, &source, &source, kNow, HealthySignals())
              .reason == hs::SupervisorReason::kAcceptedArmingPd);
    const auto limit_state2 = ValidState(2ULL);
    CHECK(limit_supervisor.ContinueLast(
              &limit_state2,
              kNow + 2'000'000ULL,
              HealthySignals())
              .reason == hs::SupervisorReason::kAcceptedHeldCommand);
    const auto limit_state3 = ValidState(3ULL);
    CHECK(limit_supervisor.ContinueLast(
              &limit_state3,
              kNow + 4'000'000ULL,
              HealthySignals())
              .reason == hs::SupervisorReason::kAcceptedHeldCommand);
    const auto limit_state4 = ValidState(4ULL);
    const auto limit = limit_supervisor.ContinueLast(
        &limit_state4, kNow + 6'000'000ULL, HealthySignals());
    CHECK(limit.reason == hs::SupervisorReason::kCommandHoldExceeded);
    CHECK(limit.plan.semantics == hs::CommandSemantics::kRelease);

    hs::HardwareCommandSupervisor expiry_supervisor(
        VerifiedPolicy(), kSession);
    auto short_lived = proposal;
    short_lived.expires_timestamp_ns = kNow + 3'000'000ULL;
    CHECK(expiry_supervisor.EvaluateNew(
              &short_lived, &source, &source, kNow, HealthySignals())
              .reason == hs::SupervisorReason::kAcceptedArmingPd);
    const auto expiry_state2 = ValidState(2ULL);
    const auto expired = expiry_supervisor.ContinueLast(
        &expiry_state2, kNow + 4'000'000ULL, HealthySignals());
    CHECK(
        expired.reason ==
        hs::SupervisorReason::kProposalExpiredOrFuture);

    hs::HardwareCommandSupervisor deadline_supervisor(
        VerifiedPolicy(), kSession);
    CHECK(deadline_supervisor.EvaluateNew(
              &proposal, &source, &source, kNow, HealthySignals())
              .reason == hs::SupervisorReason::kAcceptedArmingPd);
    auto missed = HealthySignals();
    missed.deadline_healthy = false;
    const auto deadline_state2 = ValidState(2ULL);
    const auto deadline = deadline_supervisor.ContinueLast(
        &deadline_state2, kNow + 2'000'000ULL, missed);
    CHECK(deadline.reason == hs::SupervisorReason::kDeadlineMiss);

    hs::HardwareCommandSupervisor ownership_supervisor(
        VerifiedPolicy(), kSession);
    CHECK(ownership_supervisor.EvaluateNew(
              &proposal, &source, &source, kNow, HealthySignals())
              .reason == hs::SupervisorReason::kAcceptedArmingPd);
    auto lost = HealthySignals();
    lost.ownership_confirmed = false;
    const auto ownership_state2 = ValidState(2ULL);
    const auto ownership = ownership_supervisor.ContinueLast(
        &ownership_state2, kNow + 2'000'000ULL, lost);
    CHECK(ownership.reason == hs::SupervisorReason::kOwnershipLost);
}

}  // namespace

int main() {
    TestUnverifiedPolicyCannotArm();
    TestArmingActiveAndThirteenSlotFormatting();
    TestIdentityReplaySourceAnchorAndExpiry();
    TestMaskFiniteDoublePdAndLimits();
    TestDeadlineSoftReleaseAndLatchedReset();
    TestWeightStepAndStateFreshness();
    TestLaggedSourceStateAndTwoMillisecondCommandHold();
    TestHoldLimitExpiryDeadlineAndOwnershipFailClosed();
    if (failures != 0) {
        std::cerr << failures << " hardware supervisor checks failed.\n";
        return 1;
    }
    std::cout << "hardware command supervisor tests passed.\n";
    return 0;
}
