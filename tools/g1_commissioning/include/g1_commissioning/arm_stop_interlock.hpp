#pragma once

#include <cstdint>
#include <mutex>
#include <string>

namespace g1_commissioning {

// Software interlock only; not an independent or hard real-time emergency stop.
// No reset: a trip requires a new process and the full startup procedure.
class ArmStopInterlock {
public:
    static constexpr std::uint64_t kMaxModeAgeNs = 200000000ULL;
    static constexpr std::uint16_t kL2B = (1U << 5U) | (1U << 9U);

    // Pinned SDK example/g1/low_level/gamepad.hpp: button word at bytes 2,3,
    // little-endian, L2=bit5, B=bit9. Caller must first validate LowState CRC.
    void ObserveRemote(std::uint8_t byte2, std::uint8_t byte3) {
        const auto keys = static_cast<std::uint16_t>(
            static_cast<unsigned>(byte2) | (static_cast<unsigned>(byte3) << 8U));
        if ((keys & kL2B) == kL2B) Trip("remote L2+B observed");
    }

    void ObserveMode(int rc, int fsm, std::uint64_t request_ns,
                     std::uint64_t reply_ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!fault_.empty()) return;
        if (rc != 0) fault_ = "FSM getter failed: rc=" + std::to_string(rc);
        else if (fsm != 4) fault_ = "left locked-standing: FSM=" + std::to_string(fsm);
        else if (request_ns == 0 || reply_ns < request_ns ||
                 reply_ns - request_ns > kMaxModeAgeNs)
            fault_ = "invalid or late FSM reply";
        else if (have_mode_ && (request_ns <= mode_request_ns_ ||
                 reply_ns < mode_request_ns_ ||
                 reply_ns - mode_request_ns_ > kMaxModeAgeNs))
            fault_ = "FSM observation gap";
        else {
            mode_request_ns_ = request_ns;
            have_mode_ = true;
        }
    }

    void Trip(const std::string& reason) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fault_.empty()) fault_ = reason;
    }

    // Before the first reply, refuse output but allow startup to wait.
    // Once monitoring starts, expired data is a permanent trip, even if a late
    // healthy reply subsequently arrives. Timestamp is request start, not receipt.
    std::string Check(std::uint64_t now_ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!fault_.empty()) return fault_;
        if (!have_mode_) return "FSM not yet observed";
        if (now_ns < mode_request_ns_ || now_ns - mode_request_ns_ > kMaxModeAgeNs)
            fault_ = "FSM observation stale or future";
        return fault_;
    }

private:
    std::mutex mutex_;
    bool have_mode_{false};
    std::uint64_t mode_request_ns_{0};
    std::string fault_;
};

}  // namespace g1_commissioning
