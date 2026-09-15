#include "g1_commissioning/arm_stop_interlock.hpp"

#include <iostream>

using g1_commissioning::ArmStopInterlock;
namespace {
int failures = 0;
#define CHECK(x) do { if (!(x)) { std::cerr << __LINE__ << ": " #x "\n"; ++failures; } } while (false)
constexpr std::uint64_t t = 1000000000ULL;
constexpr std::uint64_t ms = 1000000ULL;

void Healthy(ArmStopInterlock& gate) { gate.ObserveMode(0, 4, t, t + ms); }
}

int main() {
    ArmStopInterlock startup;
    CHECK(!startup.Check(t).empty());
    Healthy(startup);
    CHECK(startup.Check(t + 2 * ms).empty());
    startup.ObserveMode(0, 4, t + 50 * ms, t + 51 * ms);
    CHECK(startup.Check(t + 52 * ms).empty());

    for (int mode : {0, 1, 2, 3, 500, -1}) {
        ArmStopInterlock gate;
        Healthy(gate);
        gate.ObserveMode(0, mode, t + 10 * ms, t + 11 * ms);
        CHECK(!gate.Check(t + 12 * ms).empty());
        gate.ObserveMode(0, 4, t + 20 * ms, t + 21 * ms);
        CHECK(!gate.Check(t + 22 * ms).empty());
    }
    for (int rc : {-1, 3102, 7301}) {
        ArmStopInterlock gate;
        Healthy(gate);
        gate.ObserveMode(rc, 4, t + 10 * ms, t + 11 * ms);
        CHECK(!gate.Check(t + 12 * ms).empty());
    }
    ArmStopInterlock stale;
    Healthy(stale);
    CHECK(stale.Check(t + 200 * ms).empty());
    CHECK(!stale.Check(t + 200 * ms + 1).empty());
    stale.ObserveMode(0, 4, t + 201 * ms, t + 202 * ms);
    CHECK(!stale.Check(t + 203 * ms).empty());

    // An observation gap cannot be hidden by a healthy late reply, even if
    // the consumer did not call Check during the gap.
    ArmStopInterlock gap;
    Healthy(gap);
    gap.ObserveMode(0, 4, t + 190 * ms, t + 201 * ms);
    CHECK(!gap.Check(t + 202 * ms).empty());
    ArmStopInterlock late;
    late.ObserveMode(0, 4, t, t + 201 * ms);
    CHECK(!late.Check(t + 202 * ms).empty());
    ArmStopInterlock future;
    Healthy(future);
    CHECK(!future.Check(t - 1).empty());
    ArmStopInterlock malformed;
    malformed.ObserveMode(0, 4, t, t - 1);
    CHECK(!malformed.Check(t).empty());
    ArmStopInterlock reordered;
    Healthy(reordered);
    reordered.ObserveMode(0, 4, t - 1, t + ms);
    CHECK(!reordered.Check(t + ms).empty());

    // All 16-bit button patterns: extra pressed keys must not mask L2+B.
    for (unsigned keys = 0; keys <= 65535U; ++keys) {
        ArmStopInterlock gate;
        Healthy(gate);
        gate.ObserveRemote(static_cast<std::uint8_t>(keys & 255U),
                           static_cast<std::uint8_t>(keys >> 8U));
        CHECK(gate.Check(t + ms).empty() ==
              ((keys & ArmStopInterlock::kL2B) != ArmStopInterlock::kL2B));
    }
    ArmStopInterlock pulse;
    Healthy(pulse);
    pulse.ObserveRemote(0x20, 0x02);
    pulse.ObserveRemote(0, 0);  // button release is not a reset
    pulse.ObserveMode(0, 4, t + 10 * ms, t + 11 * ms);
    CHECK(!pulse.Check(t + 12 * ms).empty());
    ArmStopInterlock before_start;
    before_start.ObserveRemote(0x20, 0x02);
    Healthy(before_start);
    CHECK(!before_start.Check(t + ms).empty());

    // Model a pre-write consumer; actual call sites have source checks as well.
    for (bool use_remote : {false, true}) {
        ArmStopInterlock gate;
        Healthy(gate);
        int writes = 0;
        for (unsigned i = 0; i < 5; ++i) {
            if (i == 2) {
                if (use_remote) gate.ObserveRemote(0x20, 0x02);
                else gate.ObserveMode(0, 1, t + i * ms, t + i * ms);
            }
            if (!gate.Check(t + i * ms).empty()) break;
            ++writes;
        }
        CHECK(writes == 2);
    }
    std::cout << "interlock_test_failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
