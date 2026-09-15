#include "g1_commissioning/mode_step.hpp"

#include <iostream>
#include <stdexcept>

void Check(bool value) {
    if (!value) throw std::runtime_error("mode step test failed");
}

int main() {
    for (int target : {1, 4}) {
        int gets = 0, sets = 0, observations = 0;
        const bool sent = g1_commissioning::RunModeStep(target,
            [&] { return gets++ == 0 ? (target == 1 ? 0 : 1) : target; },
            [&](int value) { Check(value == target); ++sets; return 0; },
            [&] { ++observations; });
        Check(sent && gets == 2 && sets == 1 && observations == 1);
        sets = 0;
        Check(!g1_commissioning::RunModeStep(target, [&] { return target; },
            [&](int) { ++sets; return 0; }, [] {}));
        Check(sets == 0);
    }
    for (int target : {0, 2, 3, 500, 501, 801, -1}) {
        int calls = 0;
        bool failed = false;
        try {
            g1_commissioning::RunModeStep(target,
                [&] { ++calls; return 0; },
                [&](int) { ++calls; return 0; }, [&] { ++calls; });
        } catch (const std::runtime_error&) { failed = true; }
        Check(failed && calls == 0);
    }
    for (int failure : {0, 1, 2, 3, 4}) {
        int gets = 0, sets = 0, observations = 0;
        bool failed = false;
        try {
            g1_commissioning::RunModeStep(4,
                [&] {
                    ++gets;
                    if (failure == 0) throw std::runtime_error("getter failed");
                    if (failure == 1) return 0;  // May not skip damping.
                    return gets == 1 ? 1 : 0;  // Wrong final mode.
                },
                [&](int) { ++sets; return failure == 2 ? 7302 : 0; },
                [&] {
                    ++observations;
                    if (failure == 3) throw std::runtime_error("state lost");
                });
        } catch (const std::runtime_error&) { failed = true; }
        Check(failed && sets <= 1);
        if (failure <= 1) Check(sets == 0);
        if (failure == 2) Check(observations == 0 && gets == 1);
        if (failure == 3) Check(gets == 1);
    }
    std::cout << "mode-step whitelist, sequence, one-shot and fail-stop: PASS\n";
}
