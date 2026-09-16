#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <stdexcept>

namespace g1_commissioning {

// Fixed IMU-navigation-world +X target (yaw=0), not startup-heading capture.
// Same hold-heading idea as simulation: causal yaw/rate average, PD correction,
// bounded yaw-rate. 0.8 s is a FILTER window, not a claim about hardware gait.
// Uses measured torso yaw; gyro is projected onto world vertical for the PD
// damping term only. Saved gyro/accel/quaternion/RPY remain entirely raw.
class CaptureHeading {
public:
    static constexpr double kWindowS = 0.8;
    static constexpr double kKp = 0.6;
    static constexpr double kKd = 0.1;
    static constexpr double kMaxRate = 0.25;
    struct Output {
        double reference{0}, filtered_yaw{0}, filtered_rate{0}, correction{0};
    };
    void Observe(std::uint64_t ns, double yaw, double vertical_rate) {
        if (!std::isfinite(yaw) || !std::isfinite(vertical_rate))
            throw std::runtime_error("heading input must be finite");
        if (!samples_.empty() && ns <= samples_.back().ns) return;
        samples_.push_back({ns, yaw, vertical_rate});
        while (samples_.size() > 1 && ns - samples_.front().ns > 800000000ULL)
            samples_.pop_front();
    }
    Output Current() const {
        if (samples_.empty()) throw std::runtime_error("heading needs torso samples");
        double sine = 0, cosine = 0, rate = 0;
        for (const auto& s : samples_) {
            sine += std::sin(s.yaw); cosine += std::cos(s.yaw); rate += s.rate;
        }
        const double yaw = std::atan2(sine, cosine);
        rate /= static_cast<double>(samples_.size());
        const double error = std::atan2(std::sin(-yaw), std::cos(-yaw));
        return {0.0, yaw, rate, std::clamp(kKp * error - kKd * rate, -kMaxRate, kMaxRate)};
    }
private:
    struct Sample { std::uint64_t ns; double yaw, rate; };
    std::deque<Sample> samples_;
};

}  // namespace g1_commissioning
