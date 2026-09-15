#pragma once

#include <stdexcept>

namespace g1_commissioning {

// One explicit transition only. Never retries a setter or sends a fallback.
// Get must throw on a failed RPC; Observe only samples state, never commands.
template <typename Get, typename Set, typename Observe>
bool RunModeStep(int target, Get get, Set set, Observe observe) {
    if (target != 1 && target != 4) {
        throw std::runtime_error("only damping=1 or locked-standing=4 is allowed");
    }
    const int before = get();
    if (before == target) {
        return false;
    }
    const int expected = target == 1 ? 0 : 1;
    if (before != expected) {
        throw std::runtime_error("current FSM is not the permitted source mode");
    }
    if (set(target) != 0) {
        throw std::runtime_error("mode RPC returned an error; no retry or fallback");
    }
    observe();
    if (get() != target) {
        throw std::runtime_error("FSM readback did not match requested mode");
    }
    return true;
}

}  // namespace g1_commissioning
