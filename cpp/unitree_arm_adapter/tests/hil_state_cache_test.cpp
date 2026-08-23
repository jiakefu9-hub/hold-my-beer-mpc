#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "unitree_arm_adapter/hil_state_cache.hpp"

namespace hil = unitree_arm_adapter::hil;
namespace hs = unitree_arm_adapter::hardware_supervisor;

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __FILE__ << ':' << __LINE__                            \
                      << " CHECK failed: " #condition << '\n';                 \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

hs::StateSample State(std::uint64_t sample_id) {
    hs::StateSample state;
    state.validated = true;
    state.session_nonce = 77U;
    state.sample_id = sample_id;
    state.source_timestamp_ns = 1'000U + sample_id * 100U;
    state.validated_timestamp_ns = state.source_timestamp_ns + 10U;
    state.q.fill(static_cast<double>(sample_id));
    return state;
}

unitree_arm_adapter::RobotStatePayload Payload(
    const hs::StateSample& state) {
    unitree_arm_adapter::RobotStatePayload payload;
    payload.monotonic_timestamp_ns = state.source_timestamp_ns;
    payload.validated_timestamp_ns = state.validated_timestamp_ns;
    payload.sample_id = state.sample_id;
    payload.ingress_session_nonce = state.session_nonce;
    return payload;
}

void TestLaggedExactLookupAndEviction() {
    hil::ValidatedStateCache cache(3U);
    for (std::uint64_t sample = 1U; sample <= 3U; ++sample) {
        const auto state = State(sample);
        CHECK(cache.Observe(sample * 2U, Payload(state), state) ==
              hil::StateCacheObservation::kAdded);
    }
    const auto* lagged = cache.FindSource(
        1U, State(1U).source_timestamp_ns);
    CHECK(lagged != nullptr);
    CHECK(lagged != nullptr && lagged->published_sequence == 2U);
    CHECK(cache.latest() != nullptr);
    CHECK(cache.latest() != nullptr &&
          cache.latest()->supervisor_state.sample_id == 3U);
    CHECK(cache.FindSource(1U, State(1U).source_timestamp_ns + 1U) == nullptr);

    const auto fourth = State(4U);
    CHECK(cache.Observe(8U, Payload(fourth), fourth) ==
          hil::StateCacheObservation::kAdded);
    CHECK(cache.size() == 3U);
    CHECK(cache.FindSource(1U, State(1U).source_timestamp_ns) == nullptr);
}

void TestUnchangedAndRegressionsFailClosed() {
    hil::ValidatedStateCache cache(4U);
    const auto first = State(1U);
    CHECK(cache.Observe(2U, Payload(first), first) ==
          hil::StateCacheObservation::kAdded);
    CHECK(cache.Observe(2U, Payload(first), first) ==
          hil::StateCacheObservation::kUnchanged);
    CHECK(cache.Observe(1U, Payload(first), first) ==
          hil::StateCacheObservation::kInvalidState);

    const auto second = State(2U);
    CHECK(cache.Observe(4U, Payload(second), second) ==
          hil::StateCacheObservation::kAdded);
    CHECK(cache.Observe(2U, Payload(first), first) ==
          hil::StateCacheObservation::kSequenceRegression);

    auto duplicate_sample = second;
    duplicate_sample.validated_timestamp_ns += 1U;
    CHECK(cache.Observe(6U, Payload(duplicate_sample), duplicate_sample) ==
          hil::StateCacheObservation::kSampleRegression);
}

void TestInvalidStateIsNotCached() {
    hil::ValidatedStateCache cache(2U);
    auto invalid = State(1U);
    invalid.validated = false;
    CHECK(cache.Observe(2U, Payload(invalid), invalid) ==
          hil::StateCacheObservation::kInvalidState);
    CHECK(cache.size() == 0U);
    CHECK(cache.latest() == nullptr);
}

void TestZeroCapacityRejected() {
    bool rejected = false;
    try {
        hil::ValidatedStateCache cache(0U);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
}

}  // namespace

int main() {
    TestLaggedExactLookupAndEviction();
    TestUnchangedAndRegressionsFailClosed();
    TestInvalidStateIsNotCached();
    TestZeroCapacityRejected();
    if (failures != 0) {
        return 1;
    }
    std::cout << "HIL validated state cache tests passed.\n";
    return 0;
}
