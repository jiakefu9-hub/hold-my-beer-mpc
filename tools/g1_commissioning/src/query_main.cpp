#include "g1_commissioning/core.hpp"
#include "g1_commissioning/device_state.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <unitree/common/json/jsonize.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_api.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/robot/g1/loco/g1_loco_api.hpp>
#include <unitree/robot/go2/public/jsonize_type.hpp>

namespace gc = g1_commissioning;

namespace {

constexpr const char* kLowStateTopic = "rt/lowstate";
constexpr const char* kQueryPermit = "A1_READ_ONLY_GETTERS";

struct Options {
    std::string network_interface;
    std::string output_path;
    std::string snapshot_path;
    std::string service_name;
    std::string permit;
    double state_wait_s{5.0};
    double rpc_timeout_s{2.0};
};

struct IntResult {
    int return_code{-1};
    int value{0};
};

void Usage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " NETWORK_INTERFACE --output JSON --snapshot-output FILE\n"
        << "       --service-name sport --permit-read-only-query "
        << kQueryPermit << " [--state-wait-s N] [--rpc-timeout-s N]\n\n"
        << "This sends getter RPC requests and subscribes to rt/lowstate.\n"
        << "It has no motor-command message or command-topic publisher.\n";
}

double ParseDouble(const std::string& value, const char* name) {
    std::size_t consumed = 0;
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
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
        if (argument == "--output") {
            options.output_path = value("--output");
        } else if (argument == "--snapshot-output") {
            options.snapshot_path = value("--snapshot-output");
        } else if (argument == "--service-name") {
            options.service_name = value("--service-name");
        } else if (argument == "--permit-read-only-query") {
            options.permit = value("--permit-read-only-query");
        } else if (argument == "--state-wait-s") {
            options.state_wait_s = ParseDouble(value("--state-wait-s"), argument.c_str());
        } else if (argument == "--rpc-timeout-s") {
            options.rpc_timeout_s = ParseDouble(
                value("--rpc-timeout-s"), argument.c_str());
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
    if (options.network_interface.empty() || options.output_path.empty() ||
        options.snapshot_path.empty() || options.service_name.empty()) {
        throw std::invalid_argument(
            "NETWORK_INTERFACE, --output, --snapshot-output and --service-name are required");
    }
    if (options.service_name != "sport") {
        throw std::invalid_argument(
            "this pinned SDK query supports only the sport service; do not guess loco");
    }
    if (options.permit != kQueryPermit) {
        throw std::invalid_argument(
            std::string("--permit-read-only-query must equal ") + kQueryPermit);
    }
    if (!(options.state_wait_s > 0.0 && options.state_wait_s <= 10.0) ||
        !(options.rpc_timeout_s > 0.0 && options.rpc_timeout_s <= 5.0)) {
        throw std::invalid_argument("query timeouts exceed bounded limits");
    }
    return options;
}

// These wrappers deliberately register only getter API IDs. The official
// LocoClient::Init registers setters too, so it is not used by this executable.
class ReadOnlyLocoClient final : public unitree::robot::Client {
public:
    ReadOnlyLocoClient() : Client("sport", false) {}

    void Init() override {
        SetApiVersion("1.0.0.0");
        RegistApi(unitree::robot::g1::ROBOT_API_ID_LOCO_GET_FSM_ID);
        RegistApi(unitree::robot::g1::ROBOT_API_ID_LOCO_GET_FSM_MODE);
        RegistApi(unitree::robot::g1::ROBOT_API_ID_LOCO_GET_BALANCE_MODE);
    }

    IntResult GetInt(int api_id) {
        std::string data;
        IntResult result;
        result.return_code = Call(api_id, std::string{}, data);
        if (result.return_code == 0) {
            unitree::robot::go2::JsonizeDataInt decoded;
            unitree::common::FromJsonString(data, decoded);
            result.value = decoded.data;
        }
        return result;
    }
};

class ReadOnlyMotionModeClient final : public unitree::robot::Client {
public:
    ReadOnlyMotionModeClient() : Client("motion_switcher", false) {}

    void Init() override {
        SetApiVersion("1.0.0.1");
        RegistApi(unitree::robot::b2::MOTION_SWITCHER_API_ID_CHECK_MODE);
    }

    int Check(std::string& form, std::string& name) {
        std::string data;
        const int result = Call(
            unitree::robot::b2::MOTION_SWITCHER_API_ID_CHECK_MODE,
            std::string{}, data);
        if (result == 0) {
            unitree::robot::b2::JsonizeModeName decoded;
            unitree::common::FromJsonString(data, decoded);
            form = decoded.form;
            name = decoded.name;
        }
        return result;
    }
};

void WriteIntResult(
    std::ostream& output, const char* name, const IntResult& result) {
    output << ",\"" << name << "\":{\"return_code\":"
           << result.return_code << ",\"raw_value\":" << result.value << '}';
}

template <std::size_t Size>
void WriteCsv(std::ostream& output, const std::array<double, Size>& values) {
    for (std::size_t index = 0; index < Size; ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << values[index];
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::ofstream output;
    try {
        const Options options = ParseOptions(argc, argv);
        if (std::filesystem::exists(options.output_path)) {
            throw std::runtime_error("refusing to overwrite output: " +
                                     options.output_path);
        }
        if (std::filesystem::exists(options.snapshot_path)) {
            throw std::runtime_error("refusing to overwrite snapshot: " +
                                     options.snapshot_path);
        }
        output.open(options.output_path);
        if (!output) {
            throw std::runtime_error("cannot create output: " + options.output_path);
        }

        unitree::robot::ChannelFactory::Instance()->Init(
            0, options.network_interface);
        auto inbox = std::make_shared<gc::LowStateInbox>();
        auto subscriber = std::make_shared<unitree::robot::ChannelSubscriber<
            unitree_hg::msg::dds_::LowState_>>(kLowStateTopic);
        subscriber->InitChannel(
            [inbox](const void* message) { inbox->Handle(message); }, 1);

        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::duration<double>(options.state_wait_s);
        std::optional<gc::StateSample> state;
        while (std::chrono::steady_clock::now() < deadline) {
            state = inbox->Latest();
            if (state && state->crc_valid && !state->tick_regression) {
                const std::uint64_t now = gc::MonotonicNowNs();
                if (now >= state->captured_monotonic_ns &&
                    now - state->captured_monotonic_ns <= 20000000ULL) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!state || !state->crc_valid || state->tick_regression ||
            gc::MonotonicNowNs() < state->captured_monotonic_ns ||
            gc::MonotonicNowNs() - state->captured_monotonic_ns > 20000000ULL) {
            throw std::runtime_error(
                "no fresh CRC-valid non-regressing LowState; no RPC was sent");
        }

        ReadOnlyLocoClient loco;
        loco.SetTimeout(static_cast<float>(options.rpc_timeout_s));
        loco.Init();
        const IntResult fsm_id = loco.GetInt(
            unitree::robot::g1::ROBOT_API_ID_LOCO_GET_FSM_ID);
        const IntResult fsm_mode = loco.GetInt(
            unitree::robot::g1::ROBOT_API_ID_LOCO_GET_FSM_MODE);
        const IntResult balance_mode = loco.GetInt(
            unitree::robot::g1::ROBOT_API_ID_LOCO_GET_BALANCE_MODE);

        ReadOnlyMotionModeClient motion;
        motion.SetTimeout(static_cast<float>(options.rpc_timeout_s));
        motion.Init();
        std::string motion_form;
        std::string motion_name;
        const int motion_return = motion.Check(motion_form, motion_name);

        const auto final_state = inbox->Latest();
        const std::uint64_t snapshot_now = gc::MonotonicNowNs();
        if (!final_state || !final_state->crc_valid ||
            final_state->tick_regression ||
            snapshot_now < final_state->captured_monotonic_ns ||
            snapshot_now - final_state->captured_monotonic_ns > 20000000ULL) {
            throw std::runtime_error(
                "LowState became invalid/stale while getter RPCs ran");
        }
        state = final_state;

        std::ofstream snapshot_output(options.snapshot_path);
        if (!snapshot_output) {
            throw std::runtime_error(
                "cannot create snapshot: " + options.snapshot_path);
        }
        snapshot_output.precision(17);
        snapshot_output
            << "schema=g1_arm_state_snapshot_v1\n"
            << "synthetic_fixture=false\n"
            << "capture_sequence=" << state->capture_sequence << '\n'
            << "captured_monotonic_ns=" << state->captured_monotonic_ns << '\n'
            << "validation_now_monotonic_ns=" << snapshot_now << '\n'
            << "version=" << state->version[0] << ',' << state->version[1] << '\n'
            << "crc_valid=true\n"
            << "tick_regression=false\n"
            << "tick=" << state->tick << '\n'
            << "mode_pr=" << static_cast<unsigned>(state->mode_pr) << '\n'
            << "mode_machine=" << static_cast<unsigned>(state->mode_machine) << '\n'
            << "q=";
        WriteCsv(snapshot_output, state->q);
        snapshot_output << "\ndq=";
        WriteCsv(snapshot_output, state->dq);
        snapshot_output << '\n';
        snapshot_output.flush();
        if (!snapshot_output) {
            throw std::runtime_error("failed while writing state snapshot");
        }

        output << "{\"schema\":\"g1_a1_read_only_query_v1\""
               << ",\"query_only\":true,\"command_publisher_created\":false"
               << ",\"network_interface\":\""
               << gc::JsonEscape(options.network_interface) << "\""
               << ",\"loco_service\":\"sport\""
               << ",\"model_certified\":false"
               << ",\"mode_semantics_certified\":false"
               << ",\"operator_identity_required\":true"
               << ",\"lowstate\":{\"capture_sequence\":"
               << state->capture_sequence << ",\"tick\":" << state->tick
               << ",\"version\":[" << state->version[0] << ','
               << state->version[1] << ']'
               << ",\"mode_pr\":" << static_cast<unsigned>(state->mode_pr)
               << ",\"mode_machine\":"
               << static_cast<unsigned>(state->mode_machine)
               << ",\"received_count\":" << inbox->received_count()
               << ",\"crc_rejected_count\":"
               << inbox->crc_rejected_count() << '}';
        WriteIntResult(output, "fsm_id", fsm_id);
        WriteIntResult(output, "fsm_mode", fsm_mode);
        WriteIntResult(output, "balance_mode", balance_mode);
        output << ",\"motion_switcher\":{\"return_code\":"
               << motion_return << ",\"form\":\""
               << gc::JsonEscape(motion_form) << "\",\"name\":\""
               << gc::JsonEscape(motion_name) << "\"}}\n";
        std::cout << "a1_query_completed=true\n"
                  << "command_publisher_created=false\n"
                  << "output=" << options.output_path << '\n'
                  << "snapshot_output=" << options.snapshot_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        if (output.is_open()) {
            output << "{\"schema\":\"g1_a1_read_only_query_v1\","
                   << "\"query_only\":true,\"outcome\":\"failed\","
                   << "\"command_publisher_created\":false,\"reason\":\""
                   << gc::JsonEscape(error.what()) << "\"}\n";
            output.flush();
        }
        std::cerr << "read-only query failed: " << error.what() << '\n';
        return 1;
    }
}
