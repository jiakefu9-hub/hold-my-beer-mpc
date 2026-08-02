#pragma once

#include <cstdint>

namespace unitree_arm_adapter {

[[nodiscard]] std::uint64_t MonotonicNowNs();

struct PeriodicTick {
    std::uint64_t scheduled_time_ns{0};
    std::uint64_t start_time_ns{0};
    std::uint64_t wake_lateness_ns{0};
    std::uint64_t missed_periods{0};
    bool deadline_healthy{true};
};

// 【核心代码】使用CLOCK_MONOTONIC绝对时间睡眠，避免sleep_for逐拍累积漂移。
class AbsolutePeriodicTimer {
public:
    explicit AbsolutePeriodicTimer(std::uint64_t period_ns);

    [[nodiscard]] PeriodicTick WaitNext();
    [[nodiscard]] std::uint64_t period_ns() const noexcept { return period_ns_; }

private:
    std::uint64_t period_ns_{0};
    std::uint64_t next_deadline_ns_{0};
};

}  // namespace unitree_arm_adapter
