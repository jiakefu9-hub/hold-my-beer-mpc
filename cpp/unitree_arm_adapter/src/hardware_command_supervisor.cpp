#include "unitree_arm_adapter/hardware_command_supervisor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace unitree_arm_adapter::hardware_supervisor {
namespace {

template <typename T, std::size_t Size>
bool ArrayFinite(const std::array<T, Size>& values) noexcept {
    for (const T value : values) {
        if (!std::isfinite(static_cast<double>(value))) {
            return false;
        }
    }
    return true;
}

bool TimestampAge(
    std::uint64_t timestamp,
    std::uint64_t now,
    std::uint64_t timeout) noexcept {
    return timestamp != 0U && timestamp <= now &&
           now - timestamp <= timeout;
}

bool PolicyLimitsValid(const SupervisorPolicy& policy) noexcept {
    if (!policy.limits.verified ||
        !ArrayFinite(policy.limits.q_min) ||
        !ArrayFinite(policy.limits.q_max) ||
        !ArrayFinite(policy.limits.dq_abs_max) ||
        !ArrayFinite(policy.limits.kp_max) ||
        !ArrayFinite(policy.limits.kd_max) ||
        !ArrayFinite(policy.limits.tau_abs_max)) {
        return false;
    }
    for (std::size_t slot = 0; slot < kCommandSlotCount; ++slot) {
        if (!(policy.limits.q_min[slot] < policy.limits.q_max[slot]) ||
            policy.limits.dq_abs_max[slot] <= 0.0 ||
            policy.limits.kp_max[slot] < 0.0 ||
            policy.limits.kd_max[slot] < 0.0 ||
            policy.limits.tau_abs_max[slot] <= 0.0) {
            return false;
        }
    }
    return true;
}

bool ProposalFinite(const ControlProposal& proposal) noexcept {
    return std::isfinite(proposal.arm_weight) &&
           ArrayFinite(proposal.q_ref) && ArrayFinite(proposal.dq_ref) &&
           ArrayFinite(proposal.ddq_des) && ArrayFinite(proposal.kp) &&
           ArrayFinite(proposal.kd) && ArrayFinite(proposal.tau);
}

bool StateFinite(const StateSample& state) noexcept {
    return ArrayFinite(state.q) && ArrayFinite(state.dq);
}

bool IsZero(double value) noexcept { return value == 0.0; }

template <typename T, std::size_t Size>
bool AnyNonzero(const std::array<T, Size>& values) noexcept {
    for (const T value : values) {
        if (value != T{}) {
            return true;
        }
    }
    return false;
}

bool SitePolicyConfigurationValid(
    const SupervisorPolicy& policy,
    std::uint64_t session_nonce) noexcept {
    return session_nonce != 0U && policy.safety_policy_id != 0U &&
           AnyNonzero(policy.safety_policy_sha256) &&
           policy.site_policy_verified && policy.startup_pd_verified &&
           policy.release_behavior_verified &&
           policy.state_timeout_ns > 0U &&
           policy.proposal_timeout_ns > 0U &&
           std::isfinite(policy.maximum_arm_weight) &&
           policy.maximum_arm_weight > 0.0 &&
           std::isfinite(policy.maximum_weight_step_per_tick) &&
           policy.maximum_weight_step_per_tick > 0.0 &&
           std::isfinite(policy.release_weight_step_per_tick) &&
           policy.release_weight_step_per_tick > 0.0 &&
           policy.maximum_command_ticks >= 1U &&
           PolicyLimitsValid(policy);
}

}  // namespace

FormatResult FormatCommandPlan(
    const SupervisorPolicy& policy,
    const ControlProposal& proposal,
    const StateSample& state) noexcept {
    FormatResult result;
    if (!state.validated || !StateFinite(state)) {
        result.reason = SupervisorReason::kInvalidState;
        return result;
    }
    if (!ProposalFinite(proposal)) {
        result.reason = SupervisorReason::kInvalidProposal;
        return result;
    }
    if (!PolicyLimitsValid(policy)) {
        result.reason = SupervisorReason::kSitePolicyUnverified;
        return result;
    }
    if (proposal.semantics != CommandSemantics::kRobotPdPlusFeedforward &&
        proposal.semantics != CommandSemantics::kDirectTorque) {
        result.reason = SupervisorReason::kInvalidModeOrLifecycle;
        return result;
    }
    if (!(proposal.arm_weight >= 0.0 &&
          proposal.arm_weight <= policy.maximum_arm_weight)) {
        result.reason = SupervisorReason::kSiteLimitViolation;
        return result;
    }

    bool any_active = false;
    HardwareCommandPlan plan;
    plan.semantics = proposal.semantics;
    plan.producer_sequence = proposal.producer_sequence;
    plan.proposal_id = proposal.proposal_id;
    plan.source_sample_id = proposal.source_sample_id;
    plan.task_epoch_id = proposal.task_epoch_id;
    plan.safety_policy_id = proposal.safety_policy_id;
    plan.safety_policy_sha256 = proposal.safety_policy_sha256;
    plan.task_time_ns = proposal.task_time_ns;
    plan.full_task_anchor = proposal.full_task_anchor;
    plan.arm_weight = proposal.arm_weight;
    plan.active_mask = proposal.active_mask;
    plan.release_plan = false;

    for (std::size_t slot = 0; slot < kCommandSlotCount; ++slot) {
        if (!proposal.active_mask[slot]) {
            if (proposal.q_ref[slot] != state.q[slot] ||
                !IsZero(proposal.dq_ref[slot]) ||
                !IsZero(proposal.ddq_des[slot]) ||
                !IsZero(proposal.kp[slot]) ||
                !IsZero(proposal.kd[slot]) ||
                !IsZero(proposal.tau[slot])) {
                result.reason = SupervisorReason::kInactiveSlotAction;
                return result;
            }
            plan.q[slot] = state.q[slot];
            continue;
        }
        any_active = true;
        if (proposal.semantics == CommandSemantics::kDirectTorque &&
            (!IsZero(proposal.kp[slot]) || !IsZero(proposal.kd[slot]))) {
            result.reason = SupervisorReason::kDuplicateRobotPd;
            return result;
        }
        if (proposal.q_ref[slot] < policy.limits.q_min[slot] ||
            proposal.q_ref[slot] > policy.limits.q_max[slot] ||
            std::abs(proposal.dq_ref[slot]) >
                policy.limits.dq_abs_max[slot] ||
            proposal.kp[slot] < 0.0 ||
            proposal.kp[slot] > policy.limits.kp_max[slot] ||
            proposal.kd[slot] < 0.0 ||
            proposal.kd[slot] > policy.limits.kd_max[slot] ||
            std::abs(proposal.tau[slot]) >
                policy.limits.tau_abs_max[slot]) {
            result.reason = SupervisorReason::kSiteLimitViolation;
            return result;
        }
        plan.ddq_des[slot] = proposal.ddq_des[slot];
        plan.tau[slot] = proposal.tau[slot];
        if (proposal.semantics == CommandSemantics::kDirectTorque) {
            // The final torque already contains controller-side PD.  Keep the
            // robot-side reference inert even if an upstream diagnostic
            // proposal carried q/dq values.
            plan.q[slot] = state.q[slot];
            plan.dq[slot] = 0.0;
            plan.kp[slot] = 0.0;
            plan.kd[slot] = 0.0;
        } else {
            plan.q[slot] = proposal.q_ref[slot];
            plan.dq[slot] = proposal.dq_ref[slot];
            plan.kp[slot] = proposal.kp[slot];
            plan.kd[slot] = proposal.kd[slot];
        }
    }
    if (!any_active) {
        result.reason = SupervisorReason::kInvalidActiveMask;
        return result;
    }
    result.valid = true;
    result.reason = SupervisorReason::kDisarmed;
    result.plan = plan;
    return result;
}

HardwareCommandSupervisor::HardwareCommandSupervisor(
    SupervisorPolicy policy,
    std::uint64_t session_nonce) noexcept
    : policy_(policy), session_nonce_(session_nonce) {}

bool HardwareCommandSupervisor::PolicyCanArm() const noexcept {
    return SitePolicyConfigurationValid(policy_, session_nonce_) &&
           policy_.ownership_policy_verified && policy_.output_authorized;
}

HardwareCommandPlan HardwareCommandSupervisor::ReleasePlan(
    const StateSample* state,
    double weight,
    bool ready_for_sink,
    bool write_permitted) const noexcept {
    HardwareCommandPlan plan;
    plan.semantics = CommandSemantics::kRelease;
    plan.arm_weight = std::max(0.0, weight);
    plan.release_plan = true;
    plan.ready_for_sink = ready_for_sink;
    plan.write_permitted = write_permitted;
    if (state != nullptr && StateFinite(*state)) {
        plan.q = state->q;
    }
    return plan;
}

SupervisorResult HardwareCommandSupervisor::Reject(
    SupervisorReason reason,
    const StateSample* state,
    bool hard_fault) noexcept {
    if (hard_fault) {
        state_ = LifecycleState::kLatchedFault;
        last_arm_weight_ = 0.0;
        // A hardware-specific emergency release policy is a site gate.  A
        // structural hard fault never authorizes a speculative write.
        return SupervisorResult{
            state_, reason, ReleasePlan(state, 0.0, false, false)};
    }
    return EnterSoftGuard(reason, state);
}

SupervisorResult HardwareCommandSupervisor::EnterSoftGuard(
    SupervisorReason reason,
    const StateSample* state) noexcept {
    if (state_ == LifecycleState::kDisarmed || last_arm_weight_ <= 0.0) {
        state_ = LifecycleState::kDisarmed;
        last_arm_weight_ = 0.0;
        return SupervisorResult{
            state_, reason, ReleasePlan(state, 0.0, false, false)};
    }
    state_ = LifecycleState::kSoftGuardReleasing;
    last_arm_weight_ = std::max(
        0.0, last_arm_weight_ - policy_.release_weight_step_per_tick);
    const bool policy_allows_release = PolicyCanArm();
    const bool completed = last_arm_weight_ <= 0.0;
    const auto output_state = completed ? LifecycleState::kDisarmed : state_;
    HardwareCommandPlan plan = ReleasePlan(
        state,
        last_arm_weight_,
        policy_allows_release,
        policy_allows_release);
    state_ = output_state;
    return SupervisorResult{
        state_, completed ? SupervisorReason::kSoftGuardComplete : reason, plan};
}

bool HardwareCommandSupervisor::RuntimeStateValid(
    const StateSample* state, std::uint64_t now_ns) const noexcept {
    return state != nullptr && state->validated && StateFinite(*state) &&
           state->session_nonce == session_nonce_ && state->sample_id != 0U &&
           state->source_timestamp_ns != 0U &&
           state->validated_timestamp_ns >= state->source_timestamp_ns &&
           TimestampAge(
               state->source_timestamp_ns,
               now_ns,
               policy_.state_timeout_ns) &&
           TimestampAge(
               state->validated_timestamp_ns,
               now_ns,
               policy_.state_timeout_ns);
}

void HardwareCommandSupervisor::ResetCommandHistory() noexcept {
    last_proposal_id_ = 0U;
    last_producer_sequence_ = 0U;
    last_source_sample_id_ = 0U;
    task_epoch_id_ = 0U;
    last_full_task_anchor_ = 0U;
    last_arm_weight_ = 0.0;
    has_last_proposal_ = false;
    command_tick_count_ = 0U;
    last_proposal_ = ControlProposal{};
    last_plan_ = HardwareCommandPlan{};
}

void HardwareCommandSupervisor::CommitProposal(
    const ControlProposal& proposal,
    const HardwareCommandPlan& plan) noexcept {
    last_proposal_id_ = proposal.proposal_id;
    last_producer_sequence_ = proposal.producer_sequence;
    last_source_sample_id_ = proposal.source_sample_id;
    task_epoch_id_ = proposal.task_epoch_id;
    last_full_task_anchor_ = proposal.full_task_anchor;
    last_arm_weight_ = plan.arm_weight;
    has_last_proposal_ = true;
    command_tick_count_ = 1U;
    last_proposal_ = proposal;
    last_plan_ = plan;
}

HardwareCommandPlan HardwareCommandSupervisor::PlanForActuationState(
    HardwareCommandPlan plan,
    const StateSample& actuation_state) const noexcept {
    for (std::size_t slot = 0; slot < kCommandSlotCount; ++slot) {
        if (!plan.active_mask[slot] ||
            plan.semantics == CommandSemantics::kDirectTorque) {
            plan.q[slot] = actuation_state.q[slot];
        }
        if (plan.semantics == CommandSemantics::kDirectTorque) {
            plan.dq[slot] = 0.0;
            plan.kp[slot] = 0.0;
            plan.kd[slot] = 0.0;
        }
    }
    return plan;
}

SupervisorResult HardwareCommandSupervisor::Evaluate(
    const ControlProposal* proposal,
    const StateSample* state,
    std::uint64_t now_ns,
    const SupervisorSignals& signals) noexcept {
    return EvaluateNew(proposal, state, state, now_ns, signals);
}

SupervisorResult HardwareCommandSupervisor::EvaluateNew(
    const ControlProposal* proposal,
    const StateSample* source_state,
    const StateSample* actuation_state,
    std::uint64_t now_ns,
    const SupervisorSignals& signals) noexcept {
    if (state_ == LifecycleState::kLatchedFault) {
        if (signals.reset_latched_fault && !signals.request_latched_fault) {
            state_ = LifecycleState::kDisarmed;
            ResetCommandHistory();
            return SupervisorResult{
                state_, SupervisorReason::kLatchedFaultReset,
                ReleasePlan(actuation_state, 0.0, false, false)};
        }
        return SupervisorResult{
            state_, SupervisorReason::kLatchedFault,
            ReleasePlan(actuation_state, 0.0, false, false)};
    }
    if (actuation_state == nullptr || !actuation_state->validated ||
        !StateFinite(*actuation_state) ||
        actuation_state->session_nonce == 0U ||
        actuation_state->sample_id == 0U ||
        actuation_state->source_timestamp_ns == 0U ||
        actuation_state->validated_timestamp_ns <
            actuation_state->source_timestamp_ns) {
        return Reject(
            SupervisorReason::kInvalidState, actuation_state, true);
    }
    if (actuation_state->session_nonce != session_nonce_) {
        return Reject(
            SupervisorReason::kStateSessionMismatch,
            actuation_state, true);
    }
    if (!PolicyCanArm()) {
        state_ = LifecycleState::kDisarmed;
        last_arm_weight_ = 0.0;
        const auto reason = !SitePolicyConfigurationValid(
                                    policy_, session_nonce_)
                                ? SupervisorReason::kSitePolicyUnverified
                            : !policy_.ownership_policy_verified
                                ? SupervisorReason::kOwnershipUnverified
                                : SupervisorReason::kOutputAuthorizationMissing;
        return SupervisorResult{
            state_, reason,
            ReleasePlan(actuation_state, 0.0, false, false)};
    }
    if (!RuntimeStateValid(actuation_state, now_ns)) {
        return Reject(
            SupervisorReason::kStateStaleOrFuture,
            actuation_state, false);
    }
    if (signals.request_latched_fault) {
        return Reject(
            SupervisorReason::kHardFaultRequested,
            actuation_state, true);
    }
    if (!signals.deadline_healthy) {
        return Reject(
            SupervisorReason::kDeadlineMiss, actuation_state, false);
    }
    if (signals.request_soft_guard) {
        return EnterSoftGuard(
            SupervisorReason::kSoftGuardRequested, actuation_state);
    }
    if (!signals.ownership_confirmed && state_ != LifecycleState::kDisarmed) {
        return EnterSoftGuard(
            SupervisorReason::kOwnershipLost, actuation_state);
    }
    if (proposal == nullptr) {
        return Reject(
            SupervisorReason::kMissingProposal, actuation_state, false);
    }
    if (!signals.ownership_confirmed) {
        return SupervisorResult{
            LifecycleState::kDisarmed,
            SupervisorReason::kOwnershipUnverified,
            ReleasePlan(actuation_state, 0.0, false, false)};
    }
    if (proposal->session_nonce != session_nonce_) {
        return Reject(
            SupervisorReason::kProposalSessionMismatch,
            actuation_state, true);
    }
    if (proposal->safety_policy_id != policy_.safety_policy_id ||
        proposal->safety_policy_sha256 != policy_.safety_policy_sha256) {
        return Reject(
            SupervisorReason::kSafetyPolicyIdentityMismatch,
            actuation_state, true);
    }
    if (proposal->proposal_id == 0U || proposal->task_epoch_id == 0U ||
        proposal->proposal_id <= last_proposal_id_) {
        return Reject(
            SupervisorReason::kProposalReplayOrRegression,
            actuation_state, true);
    }
    if (source_state == nullptr || !source_state->validated ||
        !StateFinite(*source_state) ||
        source_state->session_nonce != session_nonce_ ||
        proposal->source_sample_id != source_state->sample_id ||
        proposal->source_timestamp_ns != source_state->source_timestamp_ns ||
        proposal->source_sample_id <= last_source_sample_id_ ||
        source_state->validated_timestamp_ns <
            source_state->source_timestamp_ns ||
        source_state->validated_timestamp_ns > now_ns ||
        !TimestampAge(
            source_state->source_timestamp_ns,
            now_ns,
            policy_.state_timeout_ns) ||
        !TimestampAge(
            source_state->validated_timestamp_ns,
            now_ns,
            policy_.state_timeout_ns) ||
        actuation_state->sample_id < source_state->sample_id ||
        actuation_state->source_timestamp_ns <
            source_state->source_timestamp_ns) {
        return Reject(
            SupervisorReason::kSourceBindingMismatch,
            actuation_state, true);
    }
    if (proposal->full_task_anchor >
            std::numeric_limits<std::uint64_t>::max() /
                kMpcAnchorPeriodNs ||
        proposal->task_time_ns !=
            proposal->full_task_anchor * kMpcAnchorPeriodNs) {
        return Reject(
            SupervisorReason::kTaskAnchorMismatch,
            actuation_state, true);
    }
    if (last_proposal_id_ == 0U) {
        if (proposal->full_task_anchor != 0U ||
            proposal->producer_sequence != 0U) {
            return Reject(
                SupervisorReason::kTaskAnchorGapOrReplay,
                actuation_state, true);
        }
    } else {
        if (proposal->producer_sequence != last_producer_sequence_ + 1U) {
            return Reject(
                SupervisorReason::kProposalReplayOrRegression,
                actuation_state, true);
        }
        if (proposal->task_epoch_id != task_epoch_id_) {
            return Reject(
                SupervisorReason::kTaskEpochMismatch,
                actuation_state, true);
        }
        if (proposal->full_task_anchor != last_full_task_anchor_ + 1U) {
            return Reject(
                SupervisorReason::kTaskAnchorGapOrReplay,
                actuation_state, true);
        }
    }
    if (proposal->generated_timestamp_ns <
            source_state->validated_timestamp_ns ||
        proposal->generated_timestamp_ns > now_ns ||
        proposal->expires_timestamp_ns <= proposal->generated_timestamp_ns ||
        proposal->expires_timestamp_ns < now_ns ||
        now_ns - proposal->generated_timestamp_ns >
            policy_.proposal_timeout_ns) {
        return Reject(
            SupervisorReason::kProposalExpiredOrFuture,
            actuation_state, false);
    }
    if (!ProposalFinite(*proposal)) {
        return Reject(
            SupervisorReason::kInvalidProposal, actuation_state, true);
    }
    if (proposal->requested_lifecycle == RequestedLifecycle::kDisarmed) {
        if (proposal->semantics != CommandSemantics::kRelease) {
            return Reject(
                SupervisorReason::kInvalidModeOrLifecycle,
                actuation_state, true);
        }
        const HardwareCommandPlan release = ReleasePlan(
            actuation_state, last_arm_weight_, false, false);
        CommitProposal(*proposal, release);
        return EnterSoftGuard(
            SupervisorReason::kSoftGuardRequested, actuation_state);
    }
    if (proposal->requested_lifecycle == RequestedLifecycle::kArmingPd) {
        if (proposal->semantics !=
                CommandSemantics::kRobotPdPlusFeedforward ||
            (state_ != LifecycleState::kDisarmed &&
             state_ != LifecycleState::kArmingPd)) {
            return Reject(
                SupervisorReason::kInvalidModeOrLifecycle,
                actuation_state, true);
        }
    } else if (proposal->requested_lifecycle == RequestedLifecycle::kActive) {
        if (!policy_.active_control_verified ||
            (state_ != LifecycleState::kArmingPd &&
             state_ != LifecycleState::kActive)) {
            return Reject(
                SupervisorReason::kInvalidModeOrLifecycle,
                actuation_state, true);
        }
    } else {
        return Reject(
            SupervisorReason::kInvalidModeOrLifecycle,
            actuation_state, true);
    }
    if (std::abs(proposal->arm_weight - last_arm_weight_) >
        policy_.maximum_weight_step_per_tick) {
        return Reject(
            SupervisorReason::kWeightStepViolation,
            actuation_state, false);
    }
    FormatResult formatted = FormatCommandPlan(
        policy_, *proposal, *source_state);
    if (!formatted.valid) {
        const bool hard = formatted.reason !=
                              SupervisorReason::kWeightStepViolation &&
                          formatted.reason !=
                              SupervisorReason::kSiteLimitViolation;
        return Reject(formatted.reason, actuation_state, hard);
    }
    formatted.plan = PlanForActuationState(
        formatted.plan, *actuation_state);
    state_ = proposal->requested_lifecycle == RequestedLifecycle::kArmingPd
                 ? LifecycleState::kArmingPd
                 : LifecycleState::kActive;
    formatted.plan.ready_for_sink = true;
    formatted.plan.write_permitted = true;
    CommitProposal(*proposal, formatted.plan);
    return SupervisorResult{
        state_,
        state_ == LifecycleState::kArmingPd
            ? SupervisorReason::kAcceptedArmingPd
            : SupervisorReason::kAcceptedActive,
        formatted.plan};
}

SupervisorResult HardwareCommandSupervisor::ContinueLast(
    const StateSample* actuation_state,
    std::uint64_t now_ns,
    const SupervisorSignals& signals) noexcept {
    if (state_ == LifecycleState::kLatchedFault) {
        if (signals.reset_latched_fault && !signals.request_latched_fault) {
            state_ = LifecycleState::kDisarmed;
            ResetCommandHistory();
            return SupervisorResult{
                state_, SupervisorReason::kLatchedFaultReset,
                ReleasePlan(actuation_state, 0.0, false, false)};
        }
        return SupervisorResult{
            state_, SupervisorReason::kLatchedFault,
            ReleasePlan(actuation_state, 0.0, false, false)};
    }
    if (actuation_state == nullptr || !actuation_state->validated ||
        !StateFinite(*actuation_state)) {
        return Reject(
            SupervisorReason::kInvalidState, actuation_state, true);
    }
    if (actuation_state->session_nonce != session_nonce_) {
        return Reject(
            SupervisorReason::kStateSessionMismatch,
            actuation_state, true);
    }
    if (!PolicyCanArm()) {
        state_ = LifecycleState::kDisarmed;
        last_arm_weight_ = 0.0;
        const auto reason = !SitePolicyConfigurationValid(
                                    policy_, session_nonce_)
                                ? SupervisorReason::kSitePolicyUnverified
                            : !policy_.ownership_policy_verified
                                ? SupervisorReason::kOwnershipUnverified
                                : SupervisorReason::kOutputAuthorizationMissing;
        return SupervisorResult{
            state_, reason,
            ReleasePlan(actuation_state, 0.0, false, false)};
    }
    if (!RuntimeStateValid(actuation_state, now_ns)) {
        return Reject(
            SupervisorReason::kStateStaleOrFuture,
            actuation_state, false);
    }
    if (signals.request_latched_fault) {
        return Reject(
            SupervisorReason::kHardFaultRequested,
            actuation_state, true);
    }
    if (!signals.deadline_healthy) {
        return Reject(
            SupervisorReason::kDeadlineMiss, actuation_state, false);
    }
    if (signals.request_soft_guard) {
        return EnterSoftGuard(
            SupervisorReason::kSoftGuardRequested, actuation_state);
    }
    if (!signals.ownership_confirmed) {
        return EnterSoftGuard(
            SupervisorReason::kOwnershipLost, actuation_state);
    }
    if (!has_last_proposal_) {
        return Reject(
            SupervisorReason::kMissingProposal, actuation_state, false);
    }
    if (state_ == LifecycleState::kSoftGuardReleasing ||
        state_ == LifecycleState::kDisarmed) {
        return EnterSoftGuard(
            SupervisorReason::kSoftGuardRequested, actuation_state);
    }
    if (command_tick_count_ >= policy_.maximum_command_ticks) {
        return Reject(
            SupervisorReason::kCommandHoldExceeded,
            actuation_state, false);
    }
    if (last_proposal_.expires_timestamp_ns < now_ns ||
        last_proposal_.generated_timestamp_ns > now_ns ||
        now_ns - last_proposal_.generated_timestamp_ns >
            policy_.proposal_timeout_ns ||
        actuation_state->sample_id < last_source_sample_id_ ||
        actuation_state->source_timestamp_ns <
            last_proposal_.source_timestamp_ns) {
        return Reject(
            SupervisorReason::kProposalExpiredOrFuture,
            actuation_state, false);
    }
    ++command_tick_count_;
    HardwareCommandPlan plan = PlanForActuationState(
        last_plan_, *actuation_state);
    plan.ready_for_sink = true;
    plan.write_permitted = true;
    last_plan_ = plan;
    return SupervisorResult{
        state_, SupervisorReason::kAcceptedHeldCommand, plan};
}

}  // namespace unitree_arm_adapter::hardware_supervisor
