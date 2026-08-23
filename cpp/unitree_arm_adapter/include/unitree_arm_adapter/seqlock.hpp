#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "unitree_arm_adapter/protocol.hpp"

namespace unitree_arm_adapter {

inline std::uint64_t AtomicLoadAcquire(const std::uint64_t* value) noexcept {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

inline void AtomicStoreRelease(
    std::uint64_t* value, std::uint64_t desired) noexcept {
    __atomic_store_n(value, desired, __ATOMIC_RELEASE);
}

template <typename Payload>
void WriteSeqlock(SeqlockSlot<Payload>& slot, const Payload& payload) noexcept {
    static_assert(std::is_trivially_copyable_v<Payload>);
    std::uint64_t sequence = AtomicLoadAcquire(&slot.sequence);
    if ((sequence & 1U) != 0U) {
        ++sequence;
    }

    // 【核心代码】奇数封住读者，payload写完后再发布下一个偶数版本。
    AtomicStoreRelease(&slot.sequence, sequence + 1U);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    std::memcpy(&slot.payload, &payload, sizeof(Payload));
    __atomic_thread_fence(__ATOMIC_RELEASE);
    AtomicStoreRelease(&slot.sequence, sequence + 2U);
}

template <typename Payload>
bool ReadSeqlockWithSequence(
    const SeqlockSlot<Payload>& slot,
    Payload& output,
    std::uint64_t& published_sequence,
    std::uint32_t max_attempts = 100U) noexcept {
    static_assert(std::is_trivially_copyable_v<Payload>);
    for (std::uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
        const std::uint64_t before = AtomicLoadAcquire(&slot.sequence);
        if ((before & 1U) != 0U) {
            continue;
        }
        std::memcpy(&output, &slot.payload, sizeof(Payload));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        const std::uint64_t after = AtomicLoadAcquire(&slot.sequence);
        if (before == after && (after & 1U) == 0U) {
            published_sequence = after;
            return true;
        }
    }
    return false;
}

template <typename Payload>
bool ReadSeqlock(
    const SeqlockSlot<Payload>& slot,
    Payload& output,
    std::uint32_t max_attempts = 100U) noexcept {
    std::uint64_t ignored_sequence = 0U;
    return ReadSeqlockWithSequence(
        slot, output, ignored_sequence, max_attempts);
}

}  // namespace unitree_arm_adapter
