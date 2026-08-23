#include "unitree_arm_adapter/hil_supervisor_dispatcher.hpp"

#include "unitree_arm_adapter/protocol_supervisor_adapter.hpp"

namespace unitree_arm_adapter::hil {

HilSupervisorDispatcher::HilSupervisorDispatcher(
    hardware_supervisor::HardwareCommandSupervisor& supervisor,
    std::size_t state_cache_capacity)
    : supervisor_(supervisor), state_cache_(state_cache_capacity) {}

StateCacheObservation HilSupervisorDispatcher::ObserveState(
    std::uint64_t published_sequence,
    const RobotStatePayload& payload,
    const hardware_supervisor::StateSample& state) {
    const StateCacheObservation observation = state_cache_.Observe(
        published_sequence, payload, state);
    if (observation == StateCacheObservation::kInvalidState ||
        observation == StateCacheObservation::kSequenceRegression ||
        observation == StateCacheObservation::kSampleRegression) {
        state_ingress_fault_latched_ = true;
    }
    return observation;
}

const CachedState* HilSupervisorDispatcher::latest_state() const noexcept {
    return state_ingress_fault_latched_ ? nullptr : state_cache_.latest();
}

DispatchResult HilSupervisorDispatcher::Dispatch(
    const ArmCommandPayload* command,
    std::uint64_t command_published_sequence,
    std::uint64_t now_ns,
    hardware_supervisor::SupervisorSignals signals) {
    DispatchResult output;
    const CachedState* const latest = latest_state();
    output.latest_state_found = latest != nullptr;
    if (command == nullptr || command_published_sequence == 0U) {
        output.supervisor = supervisor_.EvaluateNew(
            nullptr,
            nullptr,
            latest != nullptr ? &latest->supervisor_state : nullptr,
            now_ns,
            signals);
        return output;
    }

    output.new_command = !have_command_sequence_ ||
        command_published_sequence != last_command_sequence_;
    output.command_sequence_regressed = have_command_sequence_ &&
        command_published_sequence < last_command_sequence_;
    signals.request_latched_fault = signals.request_latched_fault ||
        output.command_sequence_regressed;
    if (!output.new_command) {
        output.supervisor = supervisor_.ContinueLast(
            latest != nullptr ? &latest->supervisor_state : nullptr,
            now_ns,
            signals);
        return output;
    }

    const CachedState* const source = state_cache_.FindSource(
        command->source_sample_id, command->source_timestamp_ns);
    output.source_state_found = source != nullptr;
    const hardware_supervisor::ControlProposal proposal =
        ToSupervisorProposal(*command);
    output.supervisor = supervisor_.EvaluateNew(
        &proposal,
        source != nullptr ? &source->supervisor_state : nullptr,
        latest != nullptr ? &latest->supervisor_state : nullptr,
        now_ns,
        signals);
    last_command_sequence_ = command_published_sequence;
    have_command_sequence_ = true;
    return output;
}

}  // namespace unitree_arm_adapter::hil
