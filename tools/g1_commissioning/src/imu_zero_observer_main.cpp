#include "g1_commissioning/core.hpp"
#include "g1_commissioning/device_fsm_monitor.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace gc = g1_commissioning;

namespace {

constexpr const char* kTorsoImuTopic = "rt/secondary_imu";
constexpr const char* kPermit = "IMU_ZERO_READ_ONLY";
constexpr double kPi = 3.14159265358979323846;
volatile std::sig_atomic_t stop_requested = 0;

void HandleSignal(int) { stop_requested = 1; }

struct Options {
    std::string network_interface;
    std::string log_path;
    std::string permit;
    std::uint64_t duration_s{0};
};

struct ImuSample {
    std::array<double, 4> quaternion{};
    std::array<double, 3> rpy{};
    std::array<double, 3> gyroscope{};
    std::array<double, 3> accelerometer{};
    std::uint64_t received_monotonic_ns{0};
    std::uint64_t sequence{0};
};

void Usage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " NETWORK_INTERFACE --log NEW.jsonl"
        << " --permit-read-only-observer " << kPermit
        << " [--duration-s N]\n\n"
        << "Continuously observes rt/secondary_imu and getter-only FSM state.\n"
        << "It has no command publisher and no mode setter. duration 0 runs until Ctrl-C.\n";
}

std::uint64_t ParseUnsigned(const std::string& value, const char* name) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    return parsed;
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&](const char* name) {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return std::string(argv[index]);
        };
        if (argument == "--log") {
            options.log_path = value("--log");
        } else if (argument == "--permit-read-only-observer") {
            options.permit = value("--permit-read-only-observer");
        } else if (argument == "--duration-s") {
            options.duration_s = ParseUnsigned(
                value("--duration-s"), "--duration-s");
        } else if (argument == "--help" || argument == "-h") {
            Usage(argv[0]);
            std::exit(0);
        } else if (!argument.empty() && argument.front() != '-' &&
                   options.network_interface.empty()) {
            options.network_interface = argument;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.network_interface.empty() || options.log_path.empty()) {
        throw std::invalid_argument("NETWORK_INTERFACE and --log are required");
    }
    if (options.permit != kPermit) {
        throw std::invalid_argument(
            std::string("--permit-read-only-observer must equal ") + kPermit);
    }
    if (options.duration_s > 3600U) {
        throw std::invalid_argument("--duration-s must be in [0, 3600]");
    }
    return options;
}

class ImuInbox {
public:
    void Handle(const void* raw_message) {
        if (raw_message == nullptr) return;
        const auto message = *static_cast<const
            unitree_hg::msg::dds_::IMUState_*>(raw_message);
        ImuSample sample;
        for (std::size_t index = 0; index < 4; ++index) {
            sample.quaternion[index] =
                static_cast<double>(message.quaternion().at(index));
        }
        for (std::size_t index = 0; index < 3; ++index) {
            sample.rpy[index] = static_cast<double>(message.rpy().at(index));
            sample.gyroscope[index] =
                static_cast<double>(message.gyroscope().at(index));
            sample.accelerometer[index] =
                static_cast<double>(message.accelerometer().at(index));
        }
        sample.received_monotonic_ns = gc::MonotonicNowNs();
        std::lock_guard<std::mutex> lock(mutex_);
        sample.sequence = ++sequence_;
        latest_ = sample;
    }

    [[nodiscard]] std::optional<ImuSample> Latest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return latest_;
    }

private:
    mutable std::mutex mutex_;
    std::optional<ImuSample> latest_;
    std::uint64_t sequence_{0};
};

double WrapPi(double angle) {
    return std::remainder(angle, 2.0 * kPi);
}

double Degrees(double radians) {
    return radians * 180.0 / kPi;
}

bool Finite(const ImuSample& sample) {
    for (const double value : sample.quaternion) {
        if (!std::isfinite(value)) return false;
    }
    for (const auto& values :
         {sample.rpy, sample.gyroscope, sample.accelerometer}) {
        for (const double value : values) {
            if (!std::isfinite(value)) return false;
        }
    }
    double norm_squared = 0.0;
    for (const double value : sample.quaternion) {
        norm_squared += value * value;
    }
    return norm_squared > 0.25 && norm_squared < 2.25;
}

template <std::size_t Size>
void JsonArray(std::ostream& output, const std::array<double, Size>& values) {
    output << '[';
    for (std::size_t index = 0; index < Size; ++index) {
        if (index != 0U) output << ',';
        output << values[index];
    }
    output << ']';
}

}  // namespace

int main(int argc, char** argv) {
    std::ofstream log;
    try {
        const Options options = ParseOptions(argc, argv);
        if (std::filesystem::exists(options.log_path)) {
            throw std::runtime_error("refusing to overwrite log: " + options.log_path);
        }
        log.open(options.log_path);
        if (!log) throw std::runtime_error("cannot create log: " + options.log_path);
        log << std::setprecision(17);

        std::signal(SIGINT, HandleSignal);
        std::signal(SIGTERM, HandleSignal);
        unitree::robot::ChannelFactory::Instance()->Init(
            0, options.network_interface);

        auto inbox = std::make_shared<ImuInbox>();
        auto subscriber = std::make_shared<unitree::robot::ChannelSubscriber<
            unitree_hg::msg::dds_::IMUState_>>(kTorsoImuTopic);
        subscriber->InitChannel(
            [inbox](const void* message) { inbox->Handle(message); }, 1);

        gc::FsmGetter fsm_getter;
        fsm_getter.SetTimeout(0.2F);
        fsm_getter.Init();

        log << "{\"schema\":\"g1_imu_zero_observer_start_v1\""
            << ",\"read_only\":true,\"publisher_created\":false"
            << ",\"mode_setter_registered\":false"
            << ",\"network_interface\":\""
            << gc::JsonEscape(options.network_interface) << "\""
            << ",\"imu_topic\":\"" << kTorsoImuTopic << "\"}\n";
        log.flush();

        std::cout << "READ ONLY: no publisher, no mode setter. Ctrl-C stops.\n"
                  << "Hold physical heading fixed while changing modes.\n";

        const auto wait_deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(5);
        std::optional<ImuSample> sample;
        while (!stop_requested && std::chrono::steady_clock::now() < wait_deadline) {
            sample = inbox->Latest();
            if (sample && Finite(*sample)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!sample || !Finite(*sample)) {
            throw std::runtime_error("no finite torso IMU sample within 5 seconds");
        }

        const auto started = std::chrono::steady_clock::now();
        const std::uint64_t started_ns = gc::MonotonicNowNs();
        double reference_yaw = sample->rpy[2];
        double previous_yaw = reference_yaw;
        std::uint64_t previous_sequence = 0;
        int fsm_id = -1;
        int fsm_rc = -1;
        int previous_fsm = -9999;
        auto next_fsm_query = started;
        auto next_print = started;

        while (!stop_requested) {
            const auto now = std::chrono::steady_clock::now();
            if (options.duration_s != 0U &&
                now - started >= std::chrono::seconds(options.duration_s)) {
                break;
            }
            if (now >= next_fsm_query) {
                fsm_rc = fsm_getter.Get(fsm_id);
                next_fsm_query = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(500);
            }
            if (now < next_print) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            next_print = now + std::chrono::milliseconds(200);
            sample = inbox->Latest();
            if (!sample || !Finite(*sample) ||
                sample->sequence == previous_sequence) {
                continue;
            }
            previous_sequence = sample->sequence;
            const double elapsed_s = static_cast<double>(
                gc::MonotonicNowNs() - started_ns) / 1.0e9;
            const double delta_yaw_deg = Degrees(WrapPi(
                sample->rpy[2] - reference_yaw));
            const double step_yaw_deg = Degrees(WrapPi(
                sample->rpy[2] - previous_yaw));
            previous_yaw = sample->rpy[2];
            const bool fsm_changed = fsm_rc == 0 && fsm_id != previous_fsm;
            if (fsm_changed) previous_fsm = fsm_id;
            const bool yaw_jump_candidate = std::abs(step_yaw_deg) >= 5.0;

            std::array<double, 3> rpy_deg{};
            for (std::size_t index = 0; index < 3; ++index) {
                rpy_deg[index] = Degrees(sample->rpy[index]);
            }
            std::cout << std::fixed << std::setprecision(3)
                      << '+' << elapsed_s << "s fsm=";
            if (fsm_rc == 0) std::cout << fsm_id;
            else std::cout << "ERR(" << fsm_rc << ')';
            std::cout << " q=[" << std::setprecision(5)
                      << sample->quaternion[0] << ',' << sample->quaternion[1]
                      << ',' << sample->quaternion[2] << ','
                      << sample->quaternion[3] << "] rpy_deg=["
                      << std::setprecision(2) << rpy_deg[0] << ',' << rpy_deg[1]
                      << ',' << rpy_deg[2] << "] dyaw_deg=" << delta_yaw_deg;
            if (fsm_changed) std::cout << "  <FSM_CHANGE>";
            if (yaw_jump_candidate) std::cout << "  <YAW_JUMP_CANDIDATE>";
            std::cout << '\n';

            log << "{\"schema\":\"g1_imu_zero_observation_v1\""
                << ",\"elapsed_s\":" << elapsed_s
                << ",\"monotonic_ns\":" << sample->received_monotonic_ns
                << ",\"imu_sequence\":" << sample->sequence
                << ",\"fsm_return_code\":" << fsm_rc
                << ",\"fsm_id\":" << fsm_id
                << ",\"fsm_changed\":" << (fsm_changed ? "true" : "false")
                << ",\"yaw_jump_candidate\":"
                << (yaw_jump_candidate ? "true" : "false")
                << ",\"delta_yaw_deg\":" << delta_yaw_deg
                << ",\"step_yaw_deg\":" << step_yaw_deg
                << ",\"quaternion_wxyz\":";
            JsonArray(log, sample->quaternion);
            log << ",\"rpy_rad\":";
            JsonArray(log, sample->rpy);
            log << ",\"rpy_deg\":";
            JsonArray(log, rpy_deg);
            log << ",\"gyroscope_rad_s\":";
            JsonArray(log, sample->gyroscope);
            log << ",\"accelerometer_raw_m_s2\":";
            JsonArray(log, sample->accelerometer);
            log << "}\n";
            log.flush();
        }
        log << "{\"schema\":\"g1_imu_zero_observer_end_v1\""
            << ",\"outcome\":\"stopped\",\"publisher_created\":false}\n";
        log.flush();
        return 0;
    } catch (const std::exception& error) {
        if (log.is_open()) {
            log << "{\"schema\":\"g1_imu_zero_observer_end_v1\""
                << ",\"outcome\":\"failed\",\"publisher_created\":false"
                << ",\"reason\":\"" << gc::JsonEscape(error.what())
                << "\"}\n";
            log.flush();
        }
        std::cerr << "read-only IMU observer failed: " << error.what() << '\n';
        return 1;
    }
}
