#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "unitree_arm_adapter/periodic_loop.hpp"
#include "unitree_arm_adapter/seqlock.hpp"
#include "unitree_arm_adapter/shared_memory.hpp"

namespace ua = unitree_arm_adapter;

namespace {

constexpr const char* kLowStateTopic = "rt/lowstate";
constexpr const char* kTorsoImuTopic = "rt/secondary_imu";
volatile std::sig_atomic_t stop_requested = 0;

void HandleSignal(int) { stop_requested = 1; }

struct Options {
    std::string network_interface;
    std::string shared_memory_name{"/g1_arm_mpc_shadow"};
    std::uint64_t session_nonce{0};
    std::uint64_t duration_s{0};
    std::uint64_t max_source_skew_us{5000};
    std::string summary_json;
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
        << "  --session-nonce N  explicit nonzero ingress session identity\n"
        << "  --duration-s N    0 means run until SIGINT/SIGTERM\n\n"
        << "  --max-source-skew-us N  LowState/torso-IMU pairing limit\n"
        << "  --summary-json PATH  write receive/CRC/pairing counters\n"
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
        } else if (argument == "--session-nonce") {
            options.session_nonce = ParseUnsigned(
                require_value("--session-nonce"), "--session-nonce");
        } else if (argument == "--duration-s") {
            options.duration_s = ParseUnsigned(
                require_value("--duration-s"), "--duration-s");
        } else if (argument == "--max-source-skew-us") {
            options.max_source_skew_us = ParseUnsigned(
                require_value("--max-source-skew-us"),
                "--max-source-skew-us");
        } else if (argument == "--summary-json") {
            options.summary_json = require_value("--summary-json");
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
    if (options.max_source_skew_us == 0U) {
        throw std::invalid_argument("--max-source-skew-us must be positive");
    }
    if (options.session_nonce == 0U) {
        throw std::invalid_argument("--session-nonce must be nonzero");
    }
    return options;
}

bool LowStateCrcValid(
    const unitree_hg::msg::dds_::LowState_& message) {
    static_assert(
        sizeof(unitree_hg::msg::dds_::LowState_) % sizeof(std::uint32_t) == 0U,
        "LowState wire object must contain complete uint32 words");
    auto copy = message;
    const auto word_count = static_cast<std::uint32_t>(
        sizeof(copy) / sizeof(std::uint32_t) - 1U);
    const auto computed = crc32_core(
        reinterpret_cast<std::uint32_t*>(&copy), word_count);
    return message.crc() == computed;
}

std::uint64_t AbsoluteDifference(
    std::uint64_t first, std::uint64_t second);

ua::RobotStatePayload ConvertState(
    const unitree_hg::msg::dds_::LowState_& message,
    const unitree_hg::msg::dds_::IMUState_& torso_imu,
    std::uint64_t low_state_timestamp_ns,
    std::uint64_t torso_imu_timestamp_ns,
    std::uint64_t validated_timestamp_ns,
    std::uint64_t session_nonce,
    std::uint64_t sample_id) {
    ua::RobotStatePayload state;
    state.monotonic_timestamp_ns = std::min(
        low_state_timestamp_ns, torso_imu_timestamp_ns);
    state.validated_timestamp_ns = validated_timestamp_ns;
    state.ingress_session_nonce = session_nonce;
    state.low_state_timestamp_ns = low_state_timestamp_ns;
    state.torso_imu_timestamp_ns = torso_imu_timestamp_ns;
    state.source_skew_ns = AbsoluteDifference(
        low_state_timestamp_ns, torso_imu_timestamp_ns);
    state.sample_id = sample_id;
    state.robot_tick = message.tick();
    state.ingress_flags = ua::kStateLowStateCrcValid |
                          ua::kStatePairedIngressValidated |
                          ua::kStateTorsoImuPresent;
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
    for (std::size_t index = 0; index < 4; ++index) {
        state.imu_quaternion_wxyz[index] =
            static_cast<double>(torso_imu.quaternion().at(index));
    }
    for (std::size_t index = 0; index < 3; ++index) {
        state.imu_gyroscope[index] =
            static_cast<double>(torso_imu.gyroscope().at(index));
        state.imu_accelerometer[index] =
            static_cast<double>(torso_imu.accelerometer().at(index));
        state.imu_rpy[index] =
            static_cast<double>(torso_imu.rpy().at(index));
    }
    return state;
}

std::uint64_t AbsoluteDifference(
    std::uint64_t first, std::uint64_t second) {
    return first >= second ? first - second : second - first;
}

class PairedStateWriter {
public:
    PairedStateWriter(
        ua::SharedMemoryLayout* layout,
        std::uint64_t max_source_skew_ns,
        std::uint64_t session_nonce)
        : layout_(layout),
          max_source_skew_ns_(max_source_skew_ns),
          session_nonce_(session_nonce) {}

    void OnLowState(const void* raw_message) {
        if (raw_message == nullptr) {
            return;
        }
        ++low_state_received_count_;
        const auto message = *static_cast<const
            unitree_hg::msg::dds_::LowState_*>(raw_message);
        if (!LowStateCrcValid(message)) {
            ++low_state_crc_rejected_count_;
            return;
        }
        ua::RobotStatePayload state;
        std::lock_guard<std::mutex> lock(mutex_);
        low_state_ = message;
        low_state_timestamp_ns_ = ua::MonotonicNowNs();
        ++low_state_sequence_;
        ++low_state_valid_count_;
        if (TryBuildStateLocked(state)) {
            ua::WriteSeqlock(layout_->state, state);
        }
    }

    void OnTorsoImu(const void* raw_message) {
        if (raw_message == nullptr) {
            return;
        }
        ua::RobotStatePayload state;
        std::lock_guard<std::mutex> lock(mutex_);
        torso_imu_ = *static_cast<const
            unitree_hg::msg::dds_::IMUState_*>(raw_message);
        torso_imu_timestamp_ns_ = ua::MonotonicNowNs();
        ++torso_imu_sequence_;
        ++torso_imu_count_;
        if (TryBuildStateLocked(state)) {
            ua::WriteSeqlock(layout_->state, state);
        }
    }

    [[nodiscard]] std::uint64_t low_state_received_count() const {
        return low_state_received_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t low_state_valid_count() const {
        return low_state_valid_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t low_state_crc_rejected_count() const {
        return low_state_crc_rejected_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t torso_imu_count() const {
        return torso_imu_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t paired_state_count() const {
        return paired_state_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t rejected_skew_count() const {
        return rejected_skew_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t max_accepted_skew_ns() const {
        return max_accepted_skew_ns_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::array<std::uint32_t, 2> last_version() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return low_state_.version();
    }
    [[nodiscard]] std::uint8_t last_mode_pr() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return low_state_.mode_pr();
    }
    [[nodiscard]] std::uint8_t last_mode_machine() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return low_state_.mode_machine();
    }

private:
    bool TryBuildStateLocked(ua::RobotStatePayload& state) {
        if (
            low_state_sequence_ == 0U || torso_imu_sequence_ == 0U ||
            low_state_sequence_ == published_low_state_sequence_ ||
            torso_imu_sequence_ == published_torso_imu_sequence_) {
            return false;
        }
        const std::uint64_t skew_ns = AbsoluteDifference(
            low_state_timestamp_ns_, torso_imu_timestamp_ns_);
        if (skew_ns > max_source_skew_ns_) {
            ++rejected_skew_count_;
            return false;
        }
        const std::uint64_t sample_id =
            paired_state_count_.load(std::memory_order_relaxed) + 1U;
        state = ConvertState(
            low_state_,
            torso_imu_,
            low_state_timestamp_ns_,
            torso_imu_timestamp_ns_,
            ua::MonotonicNowNs(),
            session_nonce_,
            sample_id);
        published_low_state_sequence_ = low_state_sequence_;
        published_torso_imu_sequence_ = torso_imu_sequence_;
        paired_state_count_.store(sample_id, std::memory_order_relaxed);
        const std::uint64_t prior_max =
            max_accepted_skew_ns_.load(std::memory_order_relaxed);
        if (skew_ns > prior_max) {
            max_accepted_skew_ns_.store(skew_ns, std::memory_order_relaxed);
        }
        return true;
    }

    ua::SharedMemoryLayout* layout_;
    std::uint64_t max_source_skew_ns_;
    std::uint64_t session_nonce_;
    mutable std::mutex mutex_;
    unitree_hg::msg::dds_::LowState_ low_state_;
    unitree_hg::msg::dds_::IMUState_ torso_imu_;
    std::uint64_t low_state_timestamp_ns_{0};
    std::uint64_t torso_imu_timestamp_ns_{0};
    std::uint64_t low_state_sequence_{0};
    std::uint64_t torso_imu_sequence_{0};
    std::uint64_t published_low_state_sequence_{0};
    std::uint64_t published_torso_imu_sequence_{0};
    std::atomic<std::uint64_t> low_state_received_count_{0};
    std::atomic<std::uint64_t> low_state_valid_count_{0};
    std::atomic<std::uint64_t> low_state_crc_rejected_count_{0};
    std::atomic<std::uint64_t> torso_imu_count_{0};
    std::atomic<std::uint64_t> paired_state_count_{0};
    std::atomic<std::uint64_t> rejected_skew_count_{0};
    std::atomic<std::uint64_t> max_accepted_skew_ns_{0};
};

void WriteSummaryJson(
    const std::string& path,
    const Options& options,
    const PairedStateWriter& writer) {
    if (path.empty()) {
        return;
    }
    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open bridge summary: " + path);
    }
    const auto version = writer.last_version();
    stream
        << "{\n"
        << "  \"schema\": \"unitree_state_bridge_summary_v1\",\n"
        << "  \"network_interface\": \"" << options.network_interface
        << "\",\n"
        << "  \"lowstate_topic\": \"" << kLowStateTopic << "\",\n"
        << "  \"torso_imu_topic\": \"" << kTorsoImuTopic << "\",\n"
        << "  \"output_capability\": \"absent\",\n"
        << "  \"ingress_session_nonce\": " << options.session_nonce
        << ",\n"
        << "  \"lowstate_received_count\": "
        << writer.low_state_received_count() << ",\n"
        << "  \"lowstate_crc_valid_count\": "
        << writer.low_state_valid_count() << ",\n"
        << "  \"lowstate_crc_rejected_count\": "
        << writer.low_state_crc_rejected_count() << ",\n"
        << "  \"torso_imu_received_count\": "
        << writer.torso_imu_count() << ",\n"
        << "  \"paired_state_count\": "
        << writer.paired_state_count() << ",\n"
        << "  \"rejected_source_skew_count\": "
        << writer.rejected_skew_count() << ",\n"
        << "  \"max_accepted_source_skew_us\": "
        << writer.max_accepted_skew_ns() / 1000U << ",\n"
        << "  \"last_lowstate_version\": [" << version[0] << ", "
        << version[1] << "],\n"
        << "  \"last_mode_pr\": "
        << static_cast<unsigned int>(writer.last_mode_pr()) << ",\n"
        << "  \"last_mode_machine\": "
        << static_cast<unsigned int>(writer.last_mode_machine()) << "\n"
        << "}\n";
    if (!stream) {
        throw std::runtime_error("failed to write bridge summary: " + path);
    }
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
        PairedStateWriter writer(
            layout,
            options.max_source_skew_us * 1000U,
            options.session_nonce);

        unitree::robot::ChannelFactory::Instance()->Init(
            0, options.network_interface);
        auto subscriber = std::make_unique<unitree::robot::ChannelSubscriber<
            unitree_hg::msg::dds_::LowState_>>(kLowStateTopic);
        subscriber->InitChannel(
            [&writer](const void* raw_message) {
                writer.OnLowState(raw_message);
            },
            1);
        auto torso_imu_subscriber =
            std::make_unique<unitree::robot::ChannelSubscriber<
                unitree_hg::msg::dds_::IMUState_>>(kTorsoImuTopic);
        torso_imu_subscriber->InitChannel(
            [&writer](const void* raw_message) {
                writer.OnTorsoImu(raw_message);
            },
            1);

        std::cout
            << "Unitree paired LowState + secondary torso-IMU read-only "
               "bridge: no LowCmd publisher is compiled into this "
               "executable.\n";
        const auto started = std::chrono::steady_clock::now();
        while (!stop_requested) {
            if (options.duration_s != 0U &&
                std::chrono::steady_clock::now() - started >=
                    std::chrono::seconds(options.duration_s)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout
            << "lowstate received=" << writer.low_state_received_count()
            << " crc valid=" << writer.low_state_valid_count()
            << " crc rejected=" << writer.low_state_crc_rejected_count()
            << " torso imu samples=" << writer.torso_imu_count()
            << " paired states=" << writer.paired_state_count()
            << " rejected skew=" << writer.rejected_skew_count()
            << " max accepted skew us="
            << writer.max_accepted_skew_ns() / 1000U << "\n";
        WriteSummaryJson(options.summary_json, options, writer);
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
