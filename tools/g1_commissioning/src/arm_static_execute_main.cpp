#include "g1_commissioning/core.hpp"
#include "g1_commissioning/device_state.hpp"
#include "g1_commissioning/device_fsm_monitor.hpp"

#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace gc = g1_commissioning;

namespace {

constexpr const char* kStateTopic = "rt/lowstate";
constexpr const char* kArmSdkTopic = "rt/arm_sdk";
constexpr const char* kOutputPermit = "A2_HOISTED_STATIC_ONLY";
volatile std::sig_atomic_t stop_requested = 0;

void HandleSignal(int) { stop_requested = 1; }

struct Options {
    std::string network_interface;
    std::string profile_path;
    std::string log_path;
    std::string session_label;
    std::string output_permit;
};

void Usage(const char* executable) {
    std::cout
        << "DANGER: explicit A2 hoisted-static rt/arm_sdk output executable.\n\n"
        << "Usage: " << executable << " NETWORK_INTERFACE --profile FILE"
        << " --log JSONL --session-label LABEL"
        << " --permit-real-output " << kOutputPermit << "\n\n"
        << "It never switches modes and never releases the factory motion service.\n"
        << "It still requires an interactive robot-ID confirmation before publisher creation.\n";
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
        if (argument == "--profile") {
            options.profile_path = value("--profile");
        } else if (argument == "--log") {
            options.log_path = value("--log");
        } else if (argument == "--session-label") {
            options.session_label = value("--session-label");
        } else if (argument == "--permit-real-output") {
            options.output_permit = value("--permit-real-output");
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
    if (options.network_interface.empty() || options.profile_path.empty() ||
        options.log_path.empty() || options.session_label.empty()) {
        throw std::invalid_argument(
            "NETWORK_INTERFACE, --profile, --log and --session-label are required");
    }
    if (options.output_permit != kOutputPermit) {
        throw std::invalid_argument(
            std::string("--permit-real-output must equal ") + kOutputPermit);
    }
    for (const unsigned char character : options.session_label) {
        if (!(std::isalnum(character) || character == '_' || character == '-')) {
            throw std::invalid_argument(
                "session label may contain only letters, digits, _ and -");
        }
    }
    return options;
}

std::string JoinErrors(const gc::ValidationResult& validation) {
    std::ostringstream output;
    for (std::size_t index = 0; index < validation.errors.size(); ++index) {
        if (index != 0U) {
            output << "; ";
        }
        output << validation.errors[index];
    }
    return output.str();
}

std::optional<gc::StateSample> WaitForStableStartupState(
    const gc::LowStateInbox& inbox,
    gc::ArmStopInterlock& interlock,
    const gc::SiteProfile& profile,
    std::uint64_t minimum_capture_sequence,
    std::string& failure) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::duration<double>(profile.startup_wait_s);
    std::uint64_t previous_sequence = 0;
    std::size_t consecutive = 0;
    std::optional<gc::StateSample> last;
    while (std::chrono::steady_clock::now() < deadline) {
        if (stop_requested != 0) {
            failure = "operator requested exit during startup";
            return std::nullopt;
        }
        const auto stop = interlock.Check(gc::MonotonicNowNs());
        if (!stop.empty()) {
            if (stop != "FSM not yet observed") {
                failure = stop;
                return std::nullopt;
            }
            consecutive = 0;
            failure = stop;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        const auto state = inbox.Latest();
        if (!state || state->capture_sequence <= minimum_capture_sequence ||
            state->capture_sequence == previous_sequence) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        previous_sequence = state->capture_sequence;
        const auto validation = gc::ValidateState(
            *state, profile, gc::MonotonicNowNs(), true, true);
        if (!validation.ok()) {
            consecutive = 0;
            failure = JoinErrors(validation);
        } else {
            ++consecutive;
            last = state;
            if (consecutive >= profile.startup_valid_samples) {
                return last;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (failure.empty()) {
        failure = "timed out before enough new LowState samples arrived";
    }
    return std::nullopt;
}

unitree_hg::msg::dds_::LowCmd_ MakeDdsMessage(
    const gc::SiteProfile& profile,
    const gc::CommandFrame& frame,
    const gc::StateSample* state) {
    unitree_hg::msg::dds_::LowCmd_ message;
    message.mode_pr(state != nullptr ? state->mode_pr : profile.expected_mode_pr);
    message.mode_machine(
        state != nullptr ? state->mode_machine : profile.expected_mode_machine);
    for (std::size_t slot = 0; slot < gc::kArmSlotCount; ++slot) {
        if (!profile.valid_slots[slot]) {
            continue;
        }
        auto& motor = message.motor_cmd().at(gc::kArmMotorIndices[slot]);
        motor.mode(1);
        motor.q(static_cast<float>(frame.q[slot]));
        motor.dq(static_cast<float>(frame.dq[slot]));
        motor.kp(static_cast<float>(frame.kp[slot]));
        motor.kd(static_cast<float>(frame.kd[slot]));
        motor.tau(static_cast<float>(frame.tau[slot]));
    }
    message.motor_cmd().at(gc::kWeightMotorIndex).q(
        static_cast<float>(frame.weight));
    message.crc(0U);
    const auto words = static_cast<std::uint32_t>(
        sizeof(message) / sizeof(std::uint32_t) - 1U);
    message.crc(crc32_core(
        reinterpret_cast<std::uint32_t*>(&message), words));
    return message;
}

bool PublishAndLog(
    unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>& publisher,
    std::ofstream& log,
    const gc::SiteProfile& profile,
    const gc::CommandFrame& frame,
    const gc::StateSample* state,
    const std::string& reason) {
    const auto message = MakeDdsMessage(profile, frame, state);
    const bool written = publisher.Write(message);
    log << gc::FrameJson(
               frame, state, written ? "dds_write" : "dds_write_failed", reason)
        << '\n';
    log.flush();
    return written;
}

int FaultStop(
    unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>& publisher,
    std::ofstream& log,
    const gc::SiteProfile& profile,
    const gc::StateSample* state,
    const std::string& reason,
    std::uint64_t sequence) {
    const auto frame = gc::MakeFaultZeroWeightFrame(profile, sequence);
    const bool written = PublishAndLog(
        publisher, log, profile, frame, state, reason);
    log << "{\"schema\":\"g1_arm_static_session_end_v1\","
        << "\"outcome\":\"fault\",\"reason\":\""
        << gc::JsonEscape(reason) << "\",\"fault_zero_weight_write\":"
        << (written ? "true" : "false") << "}\n";
    log.flush();
    return 3;
}

// On mode/remote-monitor trip, do NOT send even a final position/weight frame:
// leave the operator's damping request alone. No claim about firmware watchdogs.
int InterlockStop(std::ofstream& log, const std::string& reason) {
    log << "{\"schema\":\"g1_arm_static_session_end_v1\","
        << "\"outcome\":\"interlock_stop\",\"normal_output_stopped\":true,"
        << "\"final_frame_attempted\":false,\"reason\":\""
        << gc::JsonEscape(reason) << "\"}\n";
    log.flush();
    std::cerr << "A2 interlock stopped output: " << reason << '\n';
    return 4;
}

}  // namespace

int main(int argc, char** argv) {
    std::ofstream log;
    bool publisher_created = false;
    try {
        const Options options = ParseOptions(argc, argv);
        const gc::SiteProfile profile = gc::LoadSiteProfile(options.profile_path);
        const auto profile_validation = gc::ValidateProfile(
            profile, gc::ValidationUse::kRealOutput);
        if (!profile_validation.ok()) {
            throw std::runtime_error(
                "real-output profile rejected: " + JoinErrors(profile_validation));
        }
        if (std::filesystem::exists(options.log_path)) {
            throw std::runtime_error("refusing to overwrite log: " + options.log_path);
        }
        log.open(options.log_path);
        if (!log) {
            throw std::runtime_error("cannot create log: " + options.log_path);
        }
        log << gc::ProfileSummaryJson(profile, profile_validation) << '\n'
            << "{\"schema\":\"g1_arm_static_session_start_v1\","
            << "\"session_label\":\"" << gc::JsonEscape(options.session_label)
            << "\",\"network_interface\":\""
            << gc::JsonEscape(options.network_interface)
            << "\",\"publisher_created\":false}\n";
        log.flush();

        std::signal(SIGINT, HandleSignal);
        std::signal(SIGTERM, HandleSignal);
        unitree::robot::ChannelFactory::Instance()->Init(
            0, options.network_interface);
        auto inbox = std::make_shared<gc::LowStateInbox>();
        auto interlock = std::make_shared<gc::ArmStopInterlock>();
        auto subscriber = std::make_shared<unitree::robot::ChannelSubscriber<
            unitree_hg::msg::dds_::LowState_>>(kStateTopic);
        subscriber->InitChannel(
            [inbox, interlock](const void* message) {
                gc::HandleArmStopState(*inbox, *interlock, message);
            }, 1);
        gc::FsmMonitor fsm_monitor(interlock);
        log << "{\"schema\":\"g1_arm_stop_interlock_config_v1\","
            << "\"required_fsm\":4,\"mode_max_age_ms\":200,"
            << "\"rpc_timeout_ms\":100,\"poll_wait_ms\":50,"
            << "\"remote_stop_mask\":544,\"auto_reset\":false}\n";

        std::string startup_failure;
        auto initial = WaitForStableStartupState(
            *inbox, *interlock, profile, 0U, startup_failure);
        if (!initial) {
            throw std::runtime_error(
                "startup state gate failed before publisher creation: " +
                startup_failure);
        }
        log << gc::FrameJson(
                   gc::MakeFaultZeroWeightFrame(profile, 0U), &*initial,
                   "startup_state_validated", "publisher_not_created")
            << '\n';
        log.flush();

        const std::string expected_confirmation = "EXECUTE " + profile.robot_id;
        std::cout
            << "Publisher has NOT been created. Confirm hoist, clear workspace, "
            << "field approver, and robot ID.\nType exactly: "
            << expected_confirmation << "\n> " << std::flush;
        std::string confirmation;
        if (!std::getline(std::cin, confirmation) ||
            confirmation != expected_confirmation) {
            throw std::runtime_error(
                "interactive confirmation rejected; publisher was not created");
        }
        if (stop_requested != 0) {
            throw std::runtime_error(
                "signal received before publisher creation; no output attempted");
        }

        const std::uint64_t preconfirm_sequence = initial->capture_sequence;
        startup_failure.clear();
        initial = WaitForStableStartupState(
            *inbox, *interlock, profile, preconfirm_sequence, startup_failure);
        if (!initial) {
            throw std::runtime_error(
                "post-confirmation state gate failed before publisher creation: " +
                startup_failure);
        }

        const auto before_publisher = interlock->Check(gc::MonotonicNowNs());
        if (!before_publisher.empty()) {
            throw std::runtime_error("pre-publisher interlock: " + before_publisher);
        }
        // This is intentionally the first construction point of a command
        // publisher. All profile, state and human confirmation gates precede it.
        unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>
            publisher(kArmSdkTopic);
        publisher.InitChannel();
        publisher_created = true;
        log << "{\"schema\":\"g1_arm_static_publisher_event_v1\","
            << "\"publisher_created\":true,\"topic\":\"rt/arm_sdk\"}\n";
        log.flush();

        const gc::TrajectoryPlanner planner(profile, *initial);
        const auto period = std::chrono::duration_cast<
            std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(profile.control_period_ms / 1000.0));
        const auto start = std::chrono::steady_clock::now();
        auto scheduled = start;
        std::uint64_t sequence = 1;
        double last_weight = 0.0;

        while (true) {
            std::this_thread::sleep_until(scheduled);
            const auto actual = std::chrono::steady_clock::now();
            const std::uint64_t scheduled_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    scheduled.time_since_epoch()).count());
            const std::uint64_t actual_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    actual.time_since_epoch()).count());
            const auto state = inbox->Latest();
            const auto stop = interlock->Check(gc::MonotonicNowNs());
            if (!stop.empty()) return InterlockStop(log, stop);
            if (!gc::DeadlineHealthy(
                    scheduled_ns, actual_ns, profile.deadline_tolerance_ms)) {
                return FaultStop(
                    publisher, log, profile, state ? &*state : nullptr,
                    "control deadline missed", sequence);
            }
            if (!state) {
                return FaultStop(
                    publisher, log, profile, nullptr,
                    "state unavailable", sequence);
            }
            const auto state_validation = gc::ValidateState(
                *state, profile, actual_ns, false, true);
            if (!state_validation.ok()) {
                return FaultStop(
                    publisher, log, profile, &*state,
                    "runtime state gate: " + JoinErrors(state_validation), sequence);
            }

            if (stop_requested != 0) {
                const auto release_start = actual;
                const double release_initial_weight = last_weight;
                while (true) {
                    const auto release_now = std::chrono::steady_clock::now();
                    const double release_elapsed =
                        std::chrono::duration<double>(
                            release_now - release_start).count();
                    const auto release_state = inbox->Latest();
                    const auto release_stop = interlock->Check(gc::MonotonicNowNs());
                    if (!release_stop.empty()) return InterlockStop(log, release_stop);
                    if (!release_state) {
                        return FaultStop(
                            publisher, log, profile, nullptr,
                            "state unavailable during SIGINT release", sequence);
                    }
                    const auto release_validation = gc::ValidateState(
                        *release_state, profile, gc::MonotonicNowNs(), false, true);
                    if (!release_validation.ok()) {
                        return FaultStop(
                            publisher, log, profile, &*release_state,
                            "invalid state during SIGINT release: " +
                                JoinErrors(release_validation), sequence);
                    }
                    auto release = gc::MakeSigintReleaseFrame(
                        profile, *release_state, release_initial_weight,
                        release_elapsed, sequence++);
                    const auto before_release = interlock->Check(gc::MonotonicNowNs());
                    if (!before_release.empty()) return InterlockStop(log, before_release);
                    if (!PublishAndLog(
                            publisher, log, profile, release, &*release_state,
                            "operator_requested_sigint_release")) {
                        return FaultStop(
                            publisher, log, profile, &*release_state,
                            "DDS write failed during SIGINT release", sequence);
                    }
                    if (release.terminal) {
                        log << "{\"schema\":\"g1_arm_static_session_end_v1\","
                            << "\"outcome\":\"sigint_release_completed\"}\n";
                        log.flush();
                        return 130;
                    }
                    std::this_thread::sleep_for(period);
                }
            }

            const double elapsed = std::chrono::duration<double>(actual - start).count();
            if (elapsed > profile.total_timeout_s) {
                return FaultStop(
                    publisher, log, profile, &*state,
                    "finite session timeout exceeded", sequence);
            }
            auto frame = planner.Sample(elapsed);
            frame.sequence = sequence++;
            const auto tracking = gc::ValidateRuntimeTracking(
                *state, profile, *initial, frame);
            if (!tracking.ok()) {
                return FaultStop(
                    publisher, log, profile, &*state,
                    "tracking gate: " + JoinErrors(tracking), sequence);
            }
            const auto before_write = interlock->Check(gc::MonotonicNowNs());
            if (!before_write.empty()) return InterlockStop(log, before_write);
            if (!PublishAndLog(
                    publisher, log, profile, frame, &*state,
                    frame.terminal ? "normal_release_completed" : "planned_command")) {
                return FaultStop(
                    publisher, log, profile, &*state,
                    "DDS write failed", sequence);
            }
            last_weight = frame.weight;
            if (frame.terminal) {
                log << "{\"schema\":\"g1_arm_static_session_end_v1\","
                    << "\"outcome\":\"normal_release_completed\"}\n";
                log.flush();
                return 0;
            }
            scheduled += period;
        }
    } catch (const std::exception& error) {
        if (log.is_open()) {
            log << "{\"schema\":\"g1_arm_static_session_end_v1\","
                << "\"outcome\":\"refused_or_startup_failure\","
                << "\"publisher_may_have_been_created\":"
                << (publisher_created ? "true" : "false") << ','
                << "\"reason\":\"" << gc::JsonEscape(error.what()) << "\"}\n";
            log.flush();
        }
        std::cerr << "A2 static commissioning refused/failed: "
                  << error.what() << '\n';
        return 1;
    }
}
