#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "unitree_arm_adapter/periodic_loop.hpp"
#include "unitree_arm_adapter/seqlock.hpp"
#include "unitree_arm_adapter/shared_memory.hpp"

namespace ua = unitree_arm_adapter;

namespace {

ua::RobotStatePayload State(
    std::uint64_t sample_id,
    std::uint64_t session_nonce) {
    const std::uint64_t now_ns = ua::MonotonicNowNs();
    ua::RobotStatePayload state;
    state.monotonic_timestamp_ns = now_ns;
    state.validated_timestamp_ns = now_ns;
    state.low_state_timestamp_ns = now_ns;
    state.torso_imu_timestamp_ns = now_ns;
    state.ingress_session_nonce = session_nonce;
    state.sample_id = sample_id;
    state.ingress_flags = ua::kStateLowStateCrcValid |
                          ua::kStatePairedIngressValidated |
                          ua::kStateTorsoImuPresent |
                          ua::kStateSyntheticFixture;
    state.imu_quaternion_wxyz[0] = 1.0;
    return state;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "expected shared-memory name and session nonce");
        }
        const std::string name = argv[1];
        const std::uint64_t session_nonce = std::stoull(argv[2]);
        if (session_nonce == 0U) {
            throw std::invalid_argument("session nonce must be nonzero");
        }
        ua::SharedMemoryRegion::Unlink(name);
        auto region = ua::SharedMemoryRegion::Open(name, true);
        auto* const layout = region.get();
        const ua::RobotStatePayload source = State(1U, session_nonce);
        ua::WriteSeqlock(layout->state, source);

        bool command_seen = false;
        for (std::size_t attempt = 0U; attempt < 2'000U; ++attempt) {
            if (ua::AtomicLoadAcquire(&layout->command.sequence) != 0U) {
                command_seen = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!command_seen) {
            throw std::runtime_error("Python did not publish a v3 command");
        }

        bool consumer_seen = false;
        for (std::size_t attempt = 0U; attempt < 2'000U; ++attempt) {
            if (ua::AtomicLoadAcquire(&layout->status.sequence) != 0U) {
                consumer_seen = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!consumer_seen) {
            throw std::runtime_error("HIL consumer did not publish a receipt");
        }

        for (std::uint64_t sample = 2U; sample <= 12U; ++sample) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            ua::WriteSeqlock(
                layout->state, State(sample, session_nonce));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        ua::SharedMemoryRegion::Unlink(name);
        std::cout << "fixture_writer_completed=true\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HIL fixture writer failed: " << error.what() << '\n';
        return 1;
    }
}
