#include "g1_commissioning/timed_walk.hpp"
#include "g1_commissioning/capture_heading.hpp"
#include <iostream>
#include <limits>
#include <string>

namespace gc = g1_commissioning;
int failures = 0;
#define CHECK(x) do { if (!(x)) { std::cerr << __LINE__ << ": " << #x << '\n'; ++failures; } } while(false)

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    auto profile = gc::LoadSiteProfile(argv[1]);
    profile.hold_s = 15;
    profile.total_timeout_s = 22;
    gc::TimedWalkPlan::Validate(profile);
    CHECK(gc::ValidateProfile(profile, gc::ValidationUse::kPreview).ok());
    CHECK(!gc::ValidateProfile(profile, gc::ValidationUse::kRealOutput).ok());
    auto state = gc::LoadOfflineSnapshot(argv[2]).state;
    gc::TrajectoryPlanner planner(profile, state);
    CHECK(std::abs(planner.total_duration_s() - 21) < 1e-9);
    CHECK(planner.Sample(0).weight == 0);
    CHECK(std::abs(planner.Sample(1.5).weight - 0.5) < 1e-9);
    for (double t : {3., 5., 14.999, 15., 17.999, 18.}) CHECK(planner.Sample(t).weight == 1);
    CHECK(std::abs(planner.Sample(19.5).weight - 0.5) < 1e-9);
    CHECK(planner.Sample(21).terminal && planner.Sample(21).weight == 0);
    CHECK(!gc::TimedWalkPlan::Walking(4.999));
    CHECK(gc::TimedWalkPlan::Walking(5));
    CHECK(gc::TimedWalkPlan::Walking(14.999));
    CHECK(!gc::TimedWalkPlan::Walking(15));
    CHECK(!gc::TimedWalkPlan::Walking(std::numeric_limits<double>::quiet_NaN()));
    CHECK(!gc::TimedWalkPlan::HeadingHold(4.999));
    CHECK(gc::TimedWalkPlan::HeadingHold(5));
    CHECK(gc::TimedWalkPlan::HeadingHold(15));
    CHECK(gc::TimedWalkPlan::HeadingHold(17.999));
    CHECK(!gc::TimedWalkPlan::HeadingHold(18));
    CHECK(!gc::TimedWalkPlan::HeadingHold(std::numeric_limits<double>::quiet_NaN()));
    CHECK(std::abs(gc::TimedWalkPlan::Lease(14.99) - 0.01) < 1e-9);
    CHECK(std::abs(gc::TimedWalkPlan::Lease(17.99) - 0.01) < 1e-9);
    int walk_frames = 0;
    for (int frame = 0; frame <= 1050; ++frame) {
        const double t = static_cast<double>(frame) / 50.0;
        const auto command = planner.Sample(t);
        if (gc::TimedWalkPlan::Walking(t)) {
            ++walk_frames;
            CHECK(command.weight == 1);
            CHECK(t + gc::TimedWalkPlan::Lease(t) <= 15.00000001);
        }
        for (std::size_t slot = 0; slot < gc::kArmSlotCount; ++slot) {
            if (t >= 3 && profile.valid_slots[slot])
                CHECK(std::abs(command.q[slot] - profile.target_q[slot]) < 1e-9);
        }
    }
    CHECK(walk_frames == 500);
    profile.hold_s = 20;
    bool rejected = false;
    try { gc::TimedWalkPlan::Validate(profile); } catch (...) { rejected = true; }
    CHECK(rejected);
    profile.hold_s = 15;
    profile.max_weight = 0.5;
    rejected = false;
    try { gc::TimedWalkPlan::Validate(profile); } catch (...) { rejected = true; }
    CHECK(rejected);
    gc::CaptureHeading heading;
    constexpr double pi = 3.14159265358979323846;
    heading.Observe(1000000000, 3.0, pi - 0.01, 0);
    heading.Observe(2000000000, 4.0, -pi + 0.01, 0);
    CHECK(!heading.Current().reference_frozen);
    CHECK(heading.Current().correction == 0);
    CHECK(std::abs(std::abs(heading.Current().filtered_yaw) - pi) < 1e-9);
    heading.FreezeReference();
    CHECK(heading.Current().reference_frozen);
    CHECK(std::abs(std::abs(heading.Current().reference) - pi) < 1e-9);
    heading.Observe(2100000000, 5.1, -pi + 0.11, 0);
    const auto correction = heading.Current();
    CHECK(std::abs(correction.correction + 0.06) < 1e-9);
    heading.Observe(3200000000, 6.2, pi, 0);
    CHECK(std::abs(heading.Current().correction) < 1e-9);
    heading.Observe(4300000000, 7.3, -pi + 1.0, 0);
    CHECK(heading.Current().correction == -gc::CaptureHeading::kMaxRate);
    CHECK(std::abs(std::abs(heading.Current().reference) - pi) < 1e-9);
    gc::CaptureHeading incomplete;
    incomplete.Observe(1, 4.0, 0.2, 0);
    incomplete.Observe(2, 4.5, 0.2, 0);
    rejected = false;
    try { incomplete.FreezeReference(); } catch (...) { rejected = true; }
    CHECK(rejected);
    std::cout << "offline 21-second schedule, fixed-pose arms, leases and H0 heading checks: " << failures << " failures\n";
    return failures ? 1 : 0;
}
