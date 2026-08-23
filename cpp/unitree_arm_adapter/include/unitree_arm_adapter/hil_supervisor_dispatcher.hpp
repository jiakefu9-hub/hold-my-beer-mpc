#pragma once

#include <cstddef>
#include <cstdint>

#include "unitree_arm_adapter/hardware_command_supervisor.hpp"
#include "unitree_arm_adapter/hil_state_cache.hpp"
#include "unitree_arm_adapter/protocol.hpp"

namespace unitree_arm_adapter::hil {

struct DispatchResult {
    hardware_supervisor::SupervisorResult supervisor{};
    bool new_command{false};
    bool source_state_found{false};
    bool latest_state_found{false};
    bool command_sequence_regressed{false};
};

// Owns the 2 ms/6 ms protocol distinction. A stable command slot sequence is
// an explicit hold; a changed sequence is always re-evaluated as new input.
class HilSupervisorDispatcher {
public:
    HilSupervisorDispatcher(
        hardware_supervisor::HardwareCommandSupervisor& supervisor,
        std::size_t state_cache_capacity);

    [[nodiscard]] StateCacheObservation ObserveState(
        std::uint64_t published_sequence,
        const RobotStatePayload& payload,
        const hardware_supervisor::StateSample& state);

    [[nodiscard]] DispatchResult Dispatch(
        const ArmCommandPayload* command,
        std::uint64_t command_published_sequence,
        std::uint64_t now_ns,
        hardware_supervisor::SupervisorSignals signals);

    [[nodiscard]] const CachedState* latest_state() const noexcept;
    [[nodiscard]] bool state_ingress_fault_latched() const noexcept {
        return state_ingress_fault_latched_;
    }

private:
    hardware_supervisor::HardwareCommandSupervisor& supervisor_;
    ValidatedStateCache state_cache_;
    std::uint64_t last_command_sequence_{0};
    bool have_command_sequence_{false};
    bool state_ingress_fault_latched_{false};
};

}  // namespace unitree_arm_adapter::hil
