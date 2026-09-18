#pragma once

#include "g1_commissioning/core.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace g1_commissioning {

// Host monotonic task time, NOT a claimed robot gait phase or distance.
struct TimedWalkPlan {
    static constexpr double kWalkStart = 5.0;
    static constexpr double kWalkStop = 15.0;
    static constexpr double kReleaseStart = 18.0;
    static constexpr double kEnd = 21.0;
    static constexpr double kForwardSpeed = 0.5;
    static constexpr double kVelocityLease = 0.2;

    static bool Walking(double time) {
        return std::isfinite(time) && time >= kWalkStart && time < kWalkStop;
    }
    static const char* Stage(double time) {
        if (time < 3.0) return "arm_ramp_in";
        if (time < kWalkStart) return "stationary_baseline";
        if (time < kWalkStop) return "forward_walk";
        if (time < kReleaseStart) return "stop_settle";
        if (time < kEnd) return "arm_ramp_out";
        return "complete";
    }
    static double Lease(double time) {
        return Walking(time) ? std::min(kVelocityLease, kWalkStop - time)
                             : kVelocityLease;
    }
    static void Validate(const SiteProfile& profile) {
        if (!IsA3BalanceHold(profile) || profile.required_fsm != 500 ||
            std::abs(profile.max_weight - 1.0) > 1e-9 ||
            std::abs(profile.weight_rate_per_s - 1.0 / 3.0) > 1e-9 ||
            std::abs(profile.hold_s - 15.0) > 1e-9) {
            throw std::runtime_error(
                "timed walk requires A3 FSM-500 arm profile: weight 1, 3/15/3 seconds");
        }
    }
};

}  // namespace g1_commissioning
