#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "right_arm_sim_runtime/protocol.hpp"

namespace right_arm_sim_runtime {

struct ModelDimensions {
    std::uint32_t nq{0};
    std::uint32_t nv{0};
    std::uint32_t nu{0};
    std::uint32_t nbody{0};
};

class SimulationRuntime {
public:
    explicit SimulationRuntime(const std::string& scene_path);
    ~SimulationRuntime();

    SimulationRuntime(const SimulationRuntime&) = delete;
    SimulationRuntime& operator=(const SimulationRuntime&) = delete;

    [[nodiscard]] const ModelDimensions& dimensions() const noexcept;
    [[nodiscard]] bool Process(
        const SimulationRequestPayload& request,
        SimulationResponsePayload& response) noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::uint64_t MonotonicNowNs() noexcept;
[[nodiscard]] const char* RuntimeStatusString(RuntimeStatus status) noexcept;

}  // namespace right_arm_sim_runtime
