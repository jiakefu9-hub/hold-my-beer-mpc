#pragma once

#include <cstdint>

#include "unitree_arm_adapter/hardware_command_supervisor.hpp"
#include "unitree_arm_adapter/protocol.hpp"

namespace unitree_arm_adapter {

// Ingress validation is deliberately explicit.  Reading a state seqlock does
// not by itself prove pairing/model/frame validity and must not silently mint a
// hardware_supervisor::StateSample.
struct StateConversionContext {
    std::uint64_t expected_session_nonce{0};
    // Replay/HIL caller must additionally attest that this shared-memory
    // ingress came from the expected bridge/session.  The validation timestamp
    // itself is always read from the bridge payload, never supplied here.
    bool ingress_validated{false};
};

[[nodiscard]] inline hardware_supervisor::StateSample ToSupervisorState(
    const RobotStatePayload& source,
    const StateConversionContext& context) noexcept {
    hardware_supervisor::StateSample output;
    output.validated = context.ingress_validated;
    const std::uint32_t required_flags =
        kStateLowStateCrcValid |
        kStatePairedIngressValidated |
        kStateTorsoImuPresent;
    output.validated = output.validated &&
        (source.ingress_flags & required_flags) == required_flags;
    output.validated = output.validated &&
        source.ingress_session_nonce != 0U &&
        source.ingress_session_nonce == context.expected_session_nonce;
    output.session_nonce = source.ingress_session_nonce;
    output.sample_id = source.sample_id;
    output.source_timestamp_ns = source.monotonic_timestamp_ns;
    output.validated_timestamp_ns = source.validated_timestamp_ns;
    for (std::size_t slot = 0; slot < kArmSdkJointCount; ++slot) {
        const std::size_t motor = kArmSdkMotorIndices[slot];
        output.q[slot] = source.q[motor];
        output.dq[slot] = source.dq[motor];
    }
    return output;
}

[[nodiscard]] inline hardware_supervisor::RequestedLifecycle
RequestedLifecycleFromProtocol(std::uint32_t flags) noexcept {
    const std::uint32_t lifecycle = flags & (
        kCommandRequestArmingPd |
        kCommandRequestActive |
        kCommandRequestRelease);
    if (lifecycle == kCommandRequestArmingPd) {
        return hardware_supervisor::RequestedLifecycle::kArmingPd;
    }
    if (lifecycle == kCommandRequestActive) {
        return hardware_supervisor::RequestedLifecycle::kActive;
    }
    if (lifecycle == kCommandRequestRelease) {
        return hardware_supervisor::RequestedLifecycle::kDisarmed;
    }
    // Multiple or absent lifecycle bits are intentionally invalid.
    return static_cast<hardware_supervisor::RequestedLifecycle>(
        0xffffffffU);
}

[[nodiscard]] inline hardware_supervisor::CommandSemantics
CommandSemanticsFromProtocol(std::uint32_t mode) noexcept {
    if (mode == static_cast<std::uint32_t>(
                    CommandMode::kRobotPdPlusFeedforward)) {
        return hardware_supervisor::CommandSemantics::
            kRobotPdPlusFeedforward;
    }
    if (mode == static_cast<std::uint32_t>(CommandMode::kDirectTorque)) {
        return hardware_supervisor::CommandSemantics::kDirectTorque;
    }
    return hardware_supervisor::CommandSemantics::kInvalid;
}

[[nodiscard]] inline hardware_supervisor::ControlProposal
ToSupervisorProposal(const ArmCommandPayload& source) noexcept {
    hardware_supervisor::ControlProposal output;
    output.session_nonce = source.session_nonce;
    output.producer_sequence = source.producer_sequence;
    output.proposal_id = source.command_id;
    output.source_sample_id = source.source_sample_id;
    output.source_timestamp_ns = source.source_timestamp_ns;
    output.task_epoch_id = source.task_epoch_id;
    output.task_time_ns = source.task_time_ns;
    output.full_task_anchor = source.full_task_anchor;
    output.generated_timestamp_ns = source.monotonic_timestamp_ns;
    output.expires_timestamp_ns = source.expires_timestamp_ns;
    output.requested_lifecycle = RequestedLifecycleFromProtocol(source.flags);
    output.semantics = output.requested_lifecycle ==
            hardware_supervisor::RequestedLifecycle::kDisarmed
        ? hardware_supervisor::CommandSemantics::kRelease
        : CommandSemanticsFromProtocol(source.mode);
    output.arm_weight = source.arm_weight;
    output.safety_policy_id = source.safety_policy_id;
    output.safety_policy_sha256 = source.safety_policy_sha256;
    for (std::size_t slot = 0; slot < kArmSdkJointCount; ++slot) {
        output.active_mask[slot] =
            (source.active_mask & (1U << slot)) != 0U;
    }
    // High mask bits cannot be represented by the supervisor and therefore
    // invalidate the proposal instead of being silently discarded.
    if ((source.active_mask >> kArmSdkJointCount) != 0U) {
        output.semantics = hardware_supervisor::CommandSemantics::kInvalid;
    }
    output.q_ref = source.q_ref;
    output.dq_ref = source.dq_ref;
    output.ddq_des = source.ddq_des;
    output.kp = source.kp;
    output.kd = source.kd;
    output.tau = source.tau;
    return output;
}

}  // namespace unitree_arm_adapter
