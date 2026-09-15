#include "g1_commissioning/device_state.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>

#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/hg/LowState_.hpp>

namespace g1_commissioning {
namespace {

bool LowStateCrcValid(const unitree_hg::msg::dds_::LowState_& message) {
    static_assert(
        sizeof(unitree_hg::msg::dds_::LowState_) % sizeof(std::uint32_t) == 0U,
        "LowState must contain complete uint32 words");
    auto copy = message;
    const auto words = static_cast<std::uint32_t>(
        sizeof(copy) / sizeof(std::uint32_t) - 1U);
    return message.crc() == crc32_core(
        reinterpret_cast<std::uint32_t*>(&copy), words);
}

}  // namespace

std::uint64_t MonotonicNowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

void LowStateInbox::Handle(const void* raw_message) {
    if (raw_message == nullptr) {
        return;
    }
    const auto message =
        *static_cast<const unitree_hg::msg::dds_::LowState_*>(raw_message);
    const std::uint64_t received_ns = MonotonicNowNs();
    const bool crc_valid = LowStateCrcValid(message);

    std::lock_guard<std::mutex> lock(mutex_);
    ++received_count_;
    if (!crc_valid) {
        ++crc_rejected_count_;
        crc_error_latched_ = true;
        if (latest_) {
            latest_->crc_valid = false;
        }
        return;
    }

    if (have_previous_tick_ && TickRegressed(previous_tick_, message.tick())) {
        tick_regression_latched_ = true;
    }
    previous_tick_ = message.tick();
    have_previous_tick_ = true;

    StateSample sample;
    sample.synthetic_fixture = false;
    sample.crc_valid = !crc_error_latched_;
    sample.tick_regression = tick_regression_latched_;
    sample.capture_sequence = ++capture_sequence_;
    sample.captured_monotonic_ns = received_ns;
    sample.version = message.version();
    sample.tick = message.tick();
    sample.mode_pr = message.mode_pr();
    sample.mode_machine = message.mode_machine();
    for (std::size_t motor = 0; motor < kMotorCount; ++motor) {
        sample.q[motor] = static_cast<double>(
            message.motor_state().at(motor).q());
        sample.dq[motor] = static_cast<double>(
            message.motor_state().at(motor).dq());
    }
    latest_ = sample;
}

std::optional<StateSample> LowStateInbox::Latest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
}

std::uint64_t LowStateInbox::received_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_count_;
}

std::uint64_t LowStateInbox::crc_rejected_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return crc_rejected_count_;
}

}  // namespace g1_commissioning
