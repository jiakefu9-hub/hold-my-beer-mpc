#pragma once

#include <cstdint>
#include <mutex>
#include <optional>

#include "g1_commissioning/core.hpp"

namespace g1_commissioning {

class LowStateInbox {
public:
    void Handle(const void* message);
    [[nodiscard]] std::optional<StateSample> Latest() const;
    [[nodiscard]] std::uint64_t received_count() const;
    [[nodiscard]] std::uint64_t crc_rejected_count() const;

private:
    mutable std::mutex mutex_;
    std::optional<StateSample> latest_;
    std::uint64_t received_count_{0};
    std::uint64_t crc_rejected_count_{0};
    std::uint64_t capture_sequence_{0};
    std::uint32_t previous_tick_{0};
    bool have_previous_tick_{false};
    bool crc_error_latched_{false};
    bool tick_regression_latched_{false};
};

[[nodiscard]] std::uint64_t MonotonicNowNs();

}  // namespace g1_commissioning
