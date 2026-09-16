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
    profile.hold_s = 13;
    profile.total_timeout_s = 20;
    gc::TimedWalkPlan::Validate(profile);
    CHECK(gc::ValidateProfile(profile, gc::ValidationUse::kPreview).ok());
    CHECK(!gc::ValidateProfile(profile, gc::ValidationUse::kRealOutput).ok());
    auto state = gc::LoadOfflineSnapshot(argv[2]).state;
    gc::TrajectoryPlanner planner(profile, state);
    CHECK(std::abs(planner.total_duration_s() - 19) < 1e-9);
    CHECK(planner.Sample(0).weight == 0);
    CHECK(std::abs(planner.Sample(1.5).weight - 0.5) < 1e-9);
    for (double t : {3., 5., 12.999, 13., 15.999, 16.}) CHECK(planner.Sample(t).weight == 1);
    CHECK(std::abs(planner.Sample(17.5).weight - 0.5) < 1e-9);
    CHECK(planner.Sample(19).terminal && planner.Sample(19).weight == 0);
    CHECK(!gc::TimedWalkPlan::Walking(4.999));
    CHECK(gc::TimedWalkPlan::Walking(5));
    CHECK(gc::TimedWalkPlan::Walking(12.999));
    CHECK(!gc::TimedWalkPlan::Walking(13));
    CHECK(!gc::TimedWalkPlan::Walking(std::numeric_limits<double>::quiet_NaN()));
    CHECK(std::abs(gc::TimedWalkPlan::Lease(12.99) - 0.01) < 1e-9);
    int walk_frames = 0;
    for (int frame = 0; frame <= 950; ++frame) {
        const double t = static_cast<double>(frame) / 50.0;
        const auto command = planner.Sample(t);
        if (gc::TimedWalkPlan::Walking(t)) {
            ++walk_frames;
            CHECK(command.weight == 1);
            CHECK(t + gc::TimedWalkPlan::Lease(t) <= 13.00000001);
        }
        for (std::size_t slot = 0; slot < gc::kArmSlotCount; ++slot) {
            if (t >= 3 && profile.valid_slots[slot])
                CHECK(std::abs(command.q[slot] - profile.target_q[slot]) < 1e-9);
        }
    }
    CHECK(walk_frames == 400);
    profile.hold_s = 20;
    bool rejected = false;
    try { gc::TimedWalkPlan::Validate(profile); } catch (...) { rejected = true; }
    CHECK(rejected);
    profile.hold_s = 13;
    profile.max_weight = 0.5;
    rejected = false;
    try { gc::TimedWalkPlan::Validate(profile); } catch (...) { rejected = true; }
    CHECK(rejected);
    gc::CaptureHeading heading;
    constexpr double pi = 3.14159265358979323846;
    heading.Observe(1000000000, pi - 0.01, 0);
    heading.Observe(1100000000, -pi + 0.01, 0);
    CHECK(heading.Current().reference == 0);
    CHECK(std::abs(std::abs(heading.Current().filtered_yaw) - pi) < 1e-9);
    heading.Observe(2100000000, 0.10, 0);
    const auto correction = heading.Current();
    CHECK(std::abs(correction.correction + 0.06) < 1e-9);
    heading.Observe(3100000000, 0, 0);
    CHECK(std::abs(heading.Current().correction) < 1e-9);
    heading.Observe(4100000000, 1.0, 0);
    CHECK(heading.Current().correction == -gc::CaptureHeading::kMaxRate);
    CHECK(heading.Current().reference == 0);
    std::cout << "offline 19-second schedule, fixed-pose arms, leases and heading checks: " << failures << " failures\n";
    return failures ? 1 : 0;
}
