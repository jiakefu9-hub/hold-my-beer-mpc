#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace unitree_arm_adapter::hardware_supervisor {

// This contract is intentionally independent of protocol-v2/v3 and DDS.  It
// describes the last C++ boundary immediately before a future output sink.
constexpr std::size_t kCommandSlotCount = 13;
constexpr std::uint64_t kMpcAnchorPeriodNs = 6'000'000ULL;
constexpr std::size_t kSha256Bytes = 32;
using Sha256Digest = std::array<std::uint8_t, kSha256Bytes>;

enum class CommandSemantics : std::uint32_t {
    kInvalid = 0,
    kRobotPdPlusFeedforward = 1,
    kDirectTorque = 2,
    kRelease = 3,
};

enum class LifecycleState : std::uint32_t {
    kDisarmed = 0,
    kArmingPd = 1,
    kActive = 2,
    kSoftGuardReleasing = 3,
    kLatchedFault = 4,
};

enum class RequestedLifecycle : std::uint32_t {
    kDisarmed = 0,
    kArmingPd = 1,
    kActive = 2,
};

enum class SupervisorReason : std::uint32_t {
    kDisarmed = 0,
    kAcceptedArmingPd = 1,
    kAcceptedActive = 2,
    kSoftGuardRequested = 3,
    kSoftGuardComplete = 4,
    kSitePolicyUnverified = 5,
    kOwnershipUnverified = 6,
    kOutputAuthorizationMissing = 7,
    kInvalidState = 8,
    kStateSessionMismatch = 9,
    kStateStaleOrFuture = 10,
    kMissingProposal = 11,
    kInvalidProposal = 12,
    kProposalSessionMismatch = 13,
    kProposalReplayOrRegression = 14,
    kSourceBindingMismatch = 15,
    kProposalExpiredOrFuture = 16,
    kTaskAnchorMismatch = 17,
    kInvalidModeOrLifecycle = 18,
    kInvalidActiveMask = 19,
    kInactiveSlotAction = 20,
    kDuplicateRobotPd = 21,
    kSiteLimitViolation = 22,
    kWeightStepViolation = 23,
    kDeadlineMiss = 24,
    kOwnershipLost = 25,
    kHardFaultRequested = 26,
    kLatchedFault = 27,
    kLatchedFaultReset = 28,
    kSafetyPolicyIdentityMismatch = 29,
    kTaskEpochMismatch = 30,
    kTaskAnchorGapOrReplay = 31,
    kAcceptedHeldCommand = 32,
    kCommandHoldExceeded = 33,
};

struct SiteLimits {
    bool verified{false};
    std::array<double, kCommandSlotCount> q_min{};
    std::array<double, kCommandSlotCount> q_max{};
    std::array<double, kCommandSlotCount> dq_abs_max{};
    std::array<double, kCommandSlotCount> kp_max{};
    std::array<double, kCommandSlotCount> kd_max{};
    std::array<double, kCommandSlotCount> tau_abs_max{};
};

struct SupervisorPolicy {
    // All authorization fields default false.  Offline construction can never
    // accidentally arm merely because numeric arrays happen to be populated.
    bool site_policy_verified{false};
    bool ownership_policy_verified{false};
    bool startup_pd_verified{false};
    bool active_control_verified{false};
    bool release_behavior_verified{false};
    bool output_authorized{false};
    std::uint64_t state_timeout_ns{0};
    std::uint64_t proposal_timeout_ns{0};
    // One newly accepted 6 ms proposal plus at most two 2 ms hold ticks.
    // This remains zero until an explicit site/offline policy supplies it.
    std::uint32_t maximum_command_ticks{0};
    double maximum_arm_weight{0.0};
    double maximum_weight_step_per_tick{0.0};
    double release_weight_step_per_tick{0.0};
    std::uint64_t safety_policy_id{0};
    Sha256Digest safety_policy_sha256{};
    SiteLimits limits{};
};

struct StateSample {
    bool validated{false};
    std::uint64_t session_nonce{0};
    std::uint64_t sample_id{0};
    std::uint64_t source_timestamp_ns{0};
    std::uint64_t validated_timestamp_ns{0};
    std::array<double, kCommandSlotCount> q{};
    std::array<double, kCommandSlotCount> dq{};
};

struct ControlProposal {
    std::uint64_t session_nonce{0};
    std::uint64_t producer_sequence{0};
    std::uint64_t proposal_id{0};
    std::uint64_t source_sample_id{0};
    std::uint64_t source_timestamp_ns{0};
    std::uint64_t task_epoch_id{0};
    std::uint64_t task_time_ns{0};
    std::uint64_t full_task_anchor{0};
    std::uint64_t generated_timestamp_ns{0};
    std::uint64_t expires_timestamp_ns{0};
    CommandSemantics semantics{CommandSemantics::kInvalid};
    RequestedLifecycle requested_lifecycle{RequestedLifecycle::kDisarmed};
    double arm_weight{0.0};
    std::uint64_t safety_policy_id{0};
    Sha256Digest safety_policy_sha256{};
    std::array<bool, kCommandSlotCount> active_mask{};
    std::array<double, kCommandSlotCount> q_ref{};
    std::array<double, kCommandSlotCount> dq_ref{};
    std::array<double, kCommandSlotCount> ddq_des{};
    std::array<double, kCommandSlotCount> kp{};
    std::array<double, kCommandSlotCount> kd{};
    std::array<double, kCommandSlotCount> tau{};
};

struct SupervisorSignals {
    bool deadline_healthy{true};
    bool ownership_confirmed{false};
    bool request_soft_guard{false};
    bool request_latched_fault{false};
    bool reset_latched_fault{false};
};

struct HardwareCommandPlan {
    CommandSemantics semantics{CommandSemantics::kRelease};
    std::uint64_t producer_sequence{0};
    std::uint64_t proposal_id{0};
    std::uint64_t source_sample_id{0};
    std::uint64_t task_epoch_id{0};
    std::uint64_t safety_policy_id{0};
    Sha256Digest safety_policy_sha256{};
    std::uint64_t task_time_ns{0};
    std::uint64_t full_task_anchor{0};
    double arm_weight{0.0};
    std::array<bool, kCommandSlotCount> active_mask{};
    std::array<double, kCommandSlotCount> q{};
    std::array<double, kCommandSlotCount> dq{};
    std::array<double, kCommandSlotCount> ddq_des{};
    std::array<double, kCommandSlotCount> kp{};
    std::array<double, kCommandSlotCount> kd{};
    std::array<double, kCommandSlotCount> tau{};
    bool ready_for_sink{false};
    bool write_permitted{false};
    bool release_plan{true};
};

struct FormatResult {
    bool valid{false};
    SupervisorReason reason{SupervisorReason::kInvalidProposal};
    HardwareCommandPlan plan{};
};

struct SupervisorResult {
    LifecycleState state{LifecycleState::kDisarmed};
    SupervisorReason reason{SupervisorReason::kDisarmed};
    HardwareCommandPlan plan{};
};

// Pure 13-slot formatter.  Identity/time/lifecycle checks remain the
// supervisor's responsibility; this function validates finite values, active
// mask semantics, duplicate-PD prevention, and explicitly verified limits.
[[nodiscard]] FormatResult FormatCommandPlan(
    const SupervisorPolicy& policy,
    const ControlProposal& proposal,
    const StateSample& state) noexcept;

class HardwareCommandSupervisor {
public:
    HardwareCommandSupervisor(
        SupervisorPolicy policy,
        std::uint64_t session_nonce) noexcept;

    [[nodiscard]] SupervisorResult Evaluate(
        const ControlProposal* proposal,
        const StateSample* state,
        std::uint64_t now_ns,
        const SupervisorSignals& signals) noexcept;

    // A new 6 ms proposal is bound to the exact historical source state, but
    // formatted against the latest 2 ms actuation state.  This distinction is
    // required once control computation takes nonzero wall-clock time.
    [[nodiscard]] SupervisorResult EvaluateNew(
        const ControlProposal* proposal,
        const StateSample* source_state,
        const StateSample* actuation_state,
        std::uint64_t now_ns,
        const SupervisorSignals& signals) noexcept;

    // Only the caller that proved the command seqlock sequence is unchanged
    // may use this path.  It never accepts a rewritten command identity.
    [[nodiscard]] SupervisorResult ContinueLast(
        const StateSample* actuation_state,
        std::uint64_t now_ns,
        const SupervisorSignals& signals) noexcept;

    [[nodiscard]] LifecycleState state() const noexcept { return state_; }
    [[nodiscard]] std::uint64_t last_proposal_id() const noexcept {
        return last_proposal_id_;
    }
    [[nodiscard]] std::uint64_t last_producer_sequence() const noexcept {
        return last_producer_sequence_;
    }
    [[nodiscard]] std::uint64_t last_source_sample_id() const noexcept {
        return last_source_sample_id_;
    }
    [[nodiscard]] double last_arm_weight() const noexcept {
        return last_arm_weight_;
    }
    [[nodiscard]] std::uint32_t command_tick_count() const noexcept {
        return command_tick_count_;
    }

private:
    [[nodiscard]] bool PolicyCanArm() const noexcept;
    [[nodiscard]] HardwareCommandPlan ReleasePlan(
        const StateSample* state,
        double weight,
        bool ready_for_sink,
        bool write_permitted) const noexcept;
    [[nodiscard]] SupervisorResult Reject(
        SupervisorReason reason,
        const StateSample* state,
        bool hard_fault) noexcept;
    [[nodiscard]] SupervisorResult EnterSoftGuard(
        SupervisorReason reason,
        const StateSample* state) noexcept;
    [[nodiscard]] bool RuntimeStateValid(
        const StateSample* state, std::uint64_t now_ns) const noexcept;
    void ResetCommandHistory() noexcept;
    void CommitProposal(
        const ControlProposal& proposal,
        const HardwareCommandPlan& plan) noexcept;
    [[nodiscard]] HardwareCommandPlan PlanForActuationState(
        HardwareCommandPlan plan,
        const StateSample& actuation_state) const noexcept;

    SupervisorPolicy policy_{};
    std::uint64_t session_nonce_{0};
    LifecycleState state_{LifecycleState::kDisarmed};
    std::uint64_t last_proposal_id_{0};
    std::uint64_t last_producer_sequence_{0};
    std::uint64_t last_source_sample_id_{0};
    std::uint64_t task_epoch_id_{0};
    std::uint64_t last_full_task_anchor_{0};
    double last_arm_weight_{0.0};
    bool has_last_proposal_{false};
    std::uint32_t command_tick_count_{0};
    ControlProposal last_proposal_{};
    HardwareCommandPlan last_plan_{};
};

}  // namespace unitree_arm_adapter::hardware_supervisor
