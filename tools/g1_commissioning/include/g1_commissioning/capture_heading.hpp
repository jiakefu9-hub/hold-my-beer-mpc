#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <deque>
#include <stdexcept>

namespace g1_commissioning {

// One fixed run frame, H0: its +X direction is the circular mean of torso yaw
// during the final two seconds before walking. The reference is frozen before
// the first non-zero velocity request; it never follows the robot afterwards.
// A minimum one-second observed span permits fail-closed rejection of an
// incomplete baseline while accepting the requested 1--2 second mean.
//
// The one-second control window below is only a causal feedback filter. It is
// not a gait-period assumption. The gyro input is the measured angular-rate
// component about navigation-world vertical. Raw capture remains unmodified.
class CaptureHeading {
public:
    static constexpr double kReferenceStartS = 3.0;
    static constexpr double kReferenceEndS = 5.0;
    static constexpr double kMinimumReferenceSpanS = 1.0;
    static constexpr double kControlWindowS = 1.0;
    static constexpr double kKp = 1.0;
    static constexpr double kKd = 0.1;
    static constexpr double kMaxRate = 0.25;
    struct Output {
        bool reference_frozen{false};
        std::size_t reference_samples{0};
        double reference_first_s{0}, reference_last_s{0}, reference_span_s{0};
        double reference{0}, filtered_yaw{0}, relative_yaw{0}, error{0};
        double filtered_rate{0}, correction{0};
    };
    void Observe(std::uint64_t ns, double task_elapsed_s, double yaw, double vertical_rate) {
        if (!std::isfinite(task_elapsed_s) || !std::isfinite(yaw) || !std::isfinite(vertical_rate))
            throw std::runtime_error("heading input must be finite");
        if (!samples_.empty() && ns <= samples_.back().ns) return;
        samples_.push_back({ns, yaw, vertical_rate});
        constexpr auto window_ns = static_cast<std::uint64_t>(kControlWindowS * 1e9);
        while (samples_.size() > 1 && ns - samples_.front().ns > window_ns)
            samples_.pop_front();
        if (!reference_frozen_ && task_elapsed_s >= kReferenceStartS &&
            task_elapsed_s < kReferenceEndS) {
            reference_sine_ += std::sin(yaw);
            reference_cosine_ += std::cos(yaw);
            if (reference_count_ == 0) reference_first_s_ = task_elapsed_s;
            reference_last_s_ = task_elapsed_s;
            ++reference_count_;
        }
    }
    void FreezeReference() {
        if (reference_frozen_) return;
        const double span = ReferenceSpan();
        if (reference_count_ < 2 || span < kMinimumReferenceSpanS)
            throw std::runtime_error("heading H0 needs at least one second of pre-walk yaw samples");
        reference_ = std::atan2(reference_sine_, reference_cosine_);
        reference_frozen_ = true;
    }
    Output Current() const {
        if (samples_.empty()) throw std::runtime_error("heading needs torso samples");
        double sine = 0, cosine = 0, rate = 0;
        for (const auto& s : samples_) {
            sine += std::sin(s.yaw); cosine += std::cos(s.yaw); rate += s.rate;
        }
        const double yaw = std::atan2(sine, cosine);
        rate /= static_cast<double>(samples_.size());
        const double relative = reference_frozen_ ? Wrap(yaw - reference_) : 0.0;
        const double error = reference_frozen_ ? -relative : 0.0;
        Output out;
        out.reference_frozen = reference_frozen_;
        out.reference_samples = reference_count_;
        out.reference_first_s = reference_first_s_;
        out.reference_last_s = reference_last_s_;
        out.reference_span_s = ReferenceSpan();
        out.reference = reference_frozen_ ? reference_ : 0.0;
        out.filtered_yaw = yaw;
        out.relative_yaw = relative;
        out.error = error;
        out.filtered_rate = rate;
        out.correction = reference_frozen_
            ? std::clamp(kKp * error - kKd * rate, -kMaxRate, kMaxRate) : 0.0;
        return out;
    }
    bool ReferenceFrozen() const { return reference_frozen_; }
private:
    static double Wrap(double value) {
        return std::atan2(std::sin(value), std::cos(value));
    }
    double ReferenceSpan() const {
        return reference_count_ > 1 ? reference_last_s_ - reference_first_s_ : 0.0;
    }
    struct Sample { std::uint64_t ns; double yaw, rate; };
    std::deque<Sample> samples_;
    bool reference_frozen_{false};
    std::size_t reference_count_{0};
    double reference_sine_{0}, reference_cosine_{0};
    double reference_first_s_{0}, reference_last_s_{0}, reference_{0};
};

}  // namespace g1_commissioning
