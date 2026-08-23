#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>

#include "unitree_arm_adapter/hardware_command_supervisor.hpp"
#include "unitree_arm_adapter/protocol.hpp"

namespace unitree_arm_adapter::hil {

enum class StateCacheObservation : std::uint32_t {
    kAdded = 0,
    kUnchanged = 1,
    kInvalidState = 2,
    kSequenceRegression = 3,
    kSampleRegression = 4,
};

struct CachedState {
    std::uint64_t published_sequence{0};
    RobotStatePayload payload{};
    hardware_supervisor::StateSample supervisor_state{};
};

// Bounded cache needed because a 6 ms proposal can bind to a state that is no
// longer the newest sample when the 2 ms sink loop observes the command.
class ValidatedStateCache {
public:
    explicit ValidatedStateCache(std::size_t capacity);

    [[nodiscard]] StateCacheObservation Observe(
        std::uint64_t published_sequence,
        const RobotStatePayload& payload,
        const hardware_supervisor::StateSample& supervisor_state);

    [[nodiscard]] const CachedState* FindSource(
        std::uint64_t sample_id,
        std::uint64_t source_timestamp_ns) const noexcept;
    [[nodiscard]] const CachedState* latest() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] std::uint64_t last_published_sequence() const noexcept {
        return last_published_sequence_;
    }

private:
    std::size_t capacity_{0};
    std::uint64_t last_published_sequence_{0};
    std::deque<CachedState> entries_{};
};

}  // namespace unitree_arm_adapter::hil
