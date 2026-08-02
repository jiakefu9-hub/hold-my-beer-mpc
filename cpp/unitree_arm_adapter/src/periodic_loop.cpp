#include "unitree_arm_adapter/periodic_loop.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <time.h>

namespace unitree_arm_adapter {
namespace {

timespec ToTimespec(std::uint64_t nanoseconds) noexcept {
    timespec value{};
    value.tv_sec = static_cast<time_t>(nanoseconds / 1'000'000'000ULL);
    value.tv_nsec = static_cast<long>(nanoseconds % 1'000'000'000ULL);
    return value;
}

}  // namespace

std::uint64_t MonotonicNowNs() {
    timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        throw std::runtime_error(
            std::string("clock_gettime failed: ") + std::strerror(errno));
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
}

AbsolutePeriodicTimer::AbsolutePeriodicTimer(std::uint64_t period_ns)
    : period_ns_(period_ns), next_deadline_ns_(MonotonicNowNs()) {
    if (period_ns_ == 0U) {
        throw std::invalid_argument("period_ns must be positive");
    }
}

PeriodicTick AbsolutePeriodicTimer::WaitNext() {
    next_deadline_ns_ += period_ns_;
    const std::uint64_t scheduled = next_deadline_ns_;
    const timespec deadline = ToTimespec(scheduled);

    int result = 0;
    do {
        result = ::clock_nanosleep(
            CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr);
    } while (result == EINTR);
    if (result != 0) {
        throw std::runtime_error(
            std::string("clock_nanosleep failed: ") + std::strerror(result));
    }

    PeriodicTick tick;
    tick.scheduled_time_ns = scheduled;
    tick.start_time_ns = MonotonicNowNs();
    if (tick.start_time_ns > scheduled) {
        tick.wake_lateness_ns = tick.start_time_ns - scheduled;
        tick.missed_periods = tick.wake_lateness_ns / period_ns_;
    }
    tick.deadline_healthy = tick.missed_periods == 0U;

    // 如果本轮已经跨过完整周期，跳过过期节点，但继续保持绝对时间网格。
    next_deadline_ns_ += tick.missed_periods * period_ns_;
    return tick;
}

}  // namespace unitree_arm_adapter
