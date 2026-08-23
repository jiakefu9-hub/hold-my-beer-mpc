#include "unitree_arm_adapter/hil_state_cache.hpp"

#include <stdexcept>

namespace unitree_arm_adapter::hil {

ValidatedStateCache::ValidatedStateCache(std::size_t capacity)
    : capacity_(capacity) {
    if (capacity_ == 0U) {
        throw std::invalid_argument("HIL state cache capacity must be positive");
    }
}

StateCacheObservation ValidatedStateCache::Observe(
    std::uint64_t published_sequence,
    const RobotStatePayload& payload,
    const hardware_supervisor::StateSample& supervisor_state) {
    if (published_sequence == 0U || (published_sequence & 1U) != 0U) {
        return StateCacheObservation::kInvalidState;
    }
    if (published_sequence == last_published_sequence_) {
        return StateCacheObservation::kUnchanged;
    }
    if (published_sequence < last_published_sequence_) {
        return StateCacheObservation::kSequenceRegression;
    }
    last_published_sequence_ = published_sequence;
    if (!supervisor_state.validated || supervisor_state.sample_id == 0U ||
        supervisor_state.source_timestamp_ns == 0U ||
        supervisor_state.validated_timestamp_ns <
            supervisor_state.source_timestamp_ns) {
        return StateCacheObservation::kInvalidState;
    }
    if (!entries_.empty() &&
        (supervisor_state.sample_id <=
             entries_.back().supervisor_state.sample_id ||
         supervisor_state.source_timestamp_ns <
             entries_.back().supervisor_state.source_timestamp_ns)) {
        return StateCacheObservation::kSampleRegression;
    }
    entries_.push_back(CachedState{
        published_sequence, payload, supervisor_state});
    if (entries_.size() > capacity_) {
        entries_.pop_front();
    }
    return StateCacheObservation::kAdded;
}

const CachedState* ValidatedStateCache::FindSource(
    std::uint64_t sample_id,
    std::uint64_t source_timestamp_ns) const noexcept {
    for (auto iterator = entries_.rbegin(); iterator != entries_.rend();
         ++iterator) {
        if (iterator->supervisor_state.sample_id == sample_id &&
            iterator->supervisor_state.source_timestamp_ns ==
                source_timestamp_ns) {
            return &*iterator;
        }
    }
    return nullptr;
}

const CachedState* ValidatedStateCache::latest() const noexcept {
    return entries_.empty() ? nullptr : &entries_.back();
}

}  // namespace unitree_arm_adapter::hil
