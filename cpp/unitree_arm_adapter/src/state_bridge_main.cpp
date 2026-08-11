#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "unitree_arm_adapter/periodic_loop.hpp"
#include "unitree_arm_adapter/seqlock.hpp"
#include "unitree_arm_adapter/shared_memory.hpp"

namespace ua = unitree_arm_adapter;

namespace {

constexpr const char* kLowStateTopic = "rt/lowstate";
volatile std::sig_atomic_t stop_requested = 0;

void HandleSignal(int) { stop_requested = 1; }

struct Options {
    std::string network_interface;
    std::string shared_memory_name{"/g1_arm_mpc_shadow"};
    std::uint64_t duration_s{0};
    bool unlink_on_exit{false};
};

std::uint64_t ParseUnsigned(const std::string& value, const char* name) {
    std::size_t parsed = 0;
    const auto result = std::stoull(value, &parsed);
    if (parsed != value.size()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    return result;
}

void PrintUsage(const char* executable) {
    std::cout
        << "Usage: " << executable << " NETWORK_INTERFACE [options]\n"
        << "  --shm-name NAME   POSIX shared-memory name\n"
        << "  --duration-s N    0 means run until SIGINT/SIGTERM\n\n"
        << "  --unlink-on-exit  remove this bridge's shm name on exit\n\n"
        << "This binary contains no LowCmd type and creates no publisher.\n";
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto require_value = [&](const char* name) -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return argv[index];
        };
        if (argument == "--shm-name") {
            options.shared_memory_name = require_value("--shm-name");
        } else if (argument == "--duration-s") {
            options.duration_s = ParseUnsigned(
                require_value("--duration-s"), "--duration-s");
        } else if (argument == "--unlink-on-exit") {
            options.unlink_on_exit = true;
        } else if (argument == "--help" || argument == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (!argument.empty() && argument.front() != '-' &&
                   options.network_interface.empty()) {
            options.network_interface = argument;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.network_interface.empty()) {
        throw std::invalid_argument("NETWORK_INTERFACE is required");
    }
    return options;
}

ua::RobotStatePayload ConvertState(
    const unitree_hg::msg::dds_::LowState_& message,
    std::uint64_t sample_id) {
    ua::RobotStatePayload state;
    state.monotonic_timestamp_ns = ua::MonotonicNowNs();
    state.sample_id = sample_id;
    state.robot_tick = message.tick();
    state.mode_pr = message.mode_pr();
    state.mode_machine = message.mode_machine();
    for (std::size_t index = 0; index < ua::kMotorCount; ++index) {
        const auto& motor = message.motor_state().at(index);
        state.q[index] = static_cast<double>(motor.q());
        state.dq[index] = static_cast<double>(motor.dq());
        state.ddq[index] = static_cast<double>(motor.ddq());
        state.tau_est[index] = static_cast<double>(motor.tau_est());
        state.motor_temperature_c[index][0] = motor.temperature().at(0);
        state.motor_temperature_c[index][1] = motor.temperature().at(1);
    }
    const auto& imu = message.imu_state();
    for (std::size_t index = 0; index < 4; ++index) {
        state.imu_quaternion_wxyz[index] =
            static_cast<double>(imu.quaternion().at(index));
    }
    for (std::size_t index = 0; index < 3; ++index) {
        state.imu_gyroscope[index] =
            static_cast<double>(imu.gyroscope().at(index));
        state.imu_accelerometer[index] =
            static_cast<double>(imu.accelerometer().at(index));
        state.imu_rpy[index] = static_cast<double>(imu.rpy().at(index));
    }
    return state;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        std::signal(SIGINT, HandleSignal);
        std::signal(SIGTERM, HandleSignal);
        auto region = ua::SharedMemoryRegion::Open(
            options.shared_memory_name, true);
        auto* layout = region.get();
        std::atomic<std::uint64_t> sample_id{0};

        unitree::robot::ChannelFactory::Instance()->Init(
            0, options.network_interface);
        auto subscriber = std::make_unique<unitree::robot::ChannelSubscriber<
            unitree_hg::msg::dds_::LowState_>>(kLowStateTopic);
        subscriber->InitChannel(
            [layout, &sample_id](const void* raw_message) {
                if (raw_message == nullptr) {
                    return;
                }
                const auto& message = *static_cast<const
                    unitree_hg::msg::dds_::LowState_*>(raw_message);
                const auto state = ConvertState(
                    message,
                    sample_id.fetch_add(1, std::memory_order_relaxed) + 1U);
                ua::WriteSeqlock(layout->state, state);
            },
            1);

        std::cout
            << "Unitree LowState read-only bridge: no LowCmd publisher is "
               "compiled into this executable.\n";
        const auto started = std::chrono::steady_clock::now();
        while (!stop_requested) {
            if (options.duration_s != 0U &&
                std::chrono::steady_clock::now() - started >=
                    std::chrono::seconds(options.duration_s)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "state samples=" << sample_id.load() << "\n";
        if (options.unlink_on_exit) {
            region = ua::SharedMemoryRegion{};
            ua::SharedMemoryRegion::Unlink(options.shared_memory_name);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "unitree_arm_state_bridge error: " << error.what() << '\n';
        return 1;
    }
}
