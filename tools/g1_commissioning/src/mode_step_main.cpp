#include "g1_commissioning/device_state.hpp"
#include "g1_commissioning/mode_step.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <unitree/common/json/jsonize.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/robot/g1/loco/g1_loco_api.hpp>
#include <unitree/robot/go2/public/jsonize_type.hpp>

namespace gc = g1_commissioning;
namespace {
volatile std::sig_atomic_t interrupted = 0;
void OnSignal(int) { interrupted = 1; }

class ModeClient final : public unitree::robot::Client {
public:
    ModeClient() : Client("sport", false) {}
    void Init() override {
        SetApiVersion("1.0.0.0");
        RegistApi(unitree::robot::g1::ROBOT_API_ID_LOCO_GET_FSM_ID);
        RegistApi(unitree::robot::g1::ROBOT_API_ID_LOCO_SET_FSM_ID);
    }
    int Get(int& value) {
        std::string data;
        const int ret = Call(unitree::robot::g1::ROBOT_API_ID_LOCO_GET_FSM_ID,
                             std::string{}, data);
        if (ret == 0) {
            unitree::robot::go2::JsonizeDataInt json;
            unitree::common::FromJsonString(data, json);
            value = json.data;
        }
        return ret;
    }
    int SetOnce(int target) {
        if (sent_ || (target != 1 && target != 4)) {
            throw std::runtime_error("setter whitelist/one-shot guard");
        }
        sent_ = true;
        unitree::robot::go2::JsonizeDataInt json;
        json.data = target;
        std::string data;
        return Call(unitree::robot::g1::ROBOT_API_ID_LOCO_SET_FSM_ID,
                    unitree::common::ToJsonString(json), data);
    }
private:
    bool sent_{false};
};

gc::StateSample Fresh(const gc::LowStateInbox& inbox) {
    if (interrupted) throw std::runtime_error("interrupted; no further requests");
    const auto state = inbox.Latest();
    const auto now = gc::MonotonicNowNs();
    if (!state || !state->crc_valid || state->tick_regression ||
        now < state->captured_monotonic_ns ||
        now - state->captured_monotonic_ns > 20000000ULL) {
        throw std::runtime_error("state absent/stale/CRC-invalid/tick-regressed");
    }
    if (state->mode_pr != 0 || state->mode_machine != 4) {
        throw std::runtime_error("raw state fields differ from A1a target");
    }
    for (std::size_t i = 0; i < gc::kMotorCount; ++i) {
        if (!std::isfinite(state->q[i]) || !std::isfinite(state->dq[i])) {
            throw std::runtime_error("nonfinite joint state");
        }
    }
    return *state;
}

void WriteState(std::ostream& log, const gc::StateSample& state) {
    log << "{\"event\":\"state\",\"monotonic_ns\":" << gc::MonotonicNowNs()
        << ",\"tick\":" << state.tick << ",\"mode_pr\":" << unsigned(state.mode_pr)
        << ",\"mode_machine\":" << unsigned(state.mode_machine) << ",\"q\":[";
    for (std::size_t i = 0; i < gc::kMotorCount; ++i) {
        if (i) log << ',';
        log << state.q[i];
    }
    log << "],\"dq\":[";
    for (std::size_t i = 0; i < gc::kMotorCount; ++i) {
        if (i) log << ',';
        log << state.dq[i];
    }
    log << "]}\n" << std::flush;
}
}  // namespace

int main(int argc, char** argv) {
    std::ofstream log;
    bool attempted = false;
    try {
        if (argc != 8 || std::string(argv[2]) != "--target" ||
            std::string(argv[4]) != "--log" || std::string(argv[6]) != "--permit" ||
            std::string(argv[7]) != "A1B_HOISTED_MODE_ONLY") {
            throw std::runtime_error("usage: NIC --target damp|locked-stand --log NEW_JSONL --permit A1B_HOISTED_MODE_ONLY");
        }
        const std::string target_name = argv[3];
        const int target = target_name == "damp" ? 1 : target_name == "locked-stand" ? 4 : -1;
        if (target == -1 || std::string(argv[1]).empty()) {
            throw std::runtime_error("invalid target or network interface");
        }
        if (std::filesystem::exists(argv[5])) throw std::runtime_error("log already exists");
        log.open(argv[5]);
        if (!log) throw std::runtime_error("cannot open log");
        log.exceptions(std::ios::badbit | std::ios::failbit);
        log.precision(17);
        log << "{\"event\":\"start\",\"schema\":\"g1_a1b_mode_step_v1\",\"target_fsm\":"
            << target << ",\"query_only\":false,\"joint_command_publisher_created\":false,\"network_interface\":\""
            << gc::JsonEscape(argv[1]) << "\"}\n" << std::flush;
        std::signal(SIGINT, OnSignal);
        std::signal(SIGTERM, OnSignal);
        unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
        auto inbox = std::make_shared<gc::LowStateInbox>();
        auto subscriber = std::make_shared<unitree::robot::ChannelSubscriber<
            unitree_hg::msg::dds_::LowState_>>("rt/lowstate");
        subscriber->InitChannel([inbox](const void* msg) { inbox->Handle(msg); }, 1);
        const auto wait_end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!inbox->Latest() && !interrupted && std::chrono::steady_clock::now() < wait_end) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        WriteState(log, Fresh(*inbox));
        ModeClient client;
        client.SetTimeout(2.0F);
        client.Init();
        auto get = [&] {
            Fresh(*inbox);
            int value = -1;
            const int ret = client.Get(value);
            log << "{\"event\":\"get_fsm\",\"return_code\":" << ret
                << ",\"raw_value\":" << value << ",\"monotonic_ns\":"
                << gc::MonotonicNowNs() << "}\n" << std::flush;
            if (ret != 0) throw std::runtime_error("FSM getter failed");
            Fresh(*inbox);
            return value;
        };
        auto set = [&](int value) {
            Fresh(*inbox);
            log << "{\"event\":\"mode_request_intent\",\"target_fsm\":" << value
                << ",\"monotonic_ns\":" << gc::MonotonicNowNs() << "}\n" << std::flush;
            attempted = true;
            const int ret = client.SetOnce(value);
            log << "{\"event\":\"mode_request_result\",\"return_code\":" << ret
                << ",\"monotonic_ns\":" << gc::MonotonicNowNs() << "}\n" << std::flush;
            Fresh(*inbox);
            return ret;
        };
        auto observe = [&] {
            // Only observe after the single RPC. No implicit mode rollback on failure.
            const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(6);
            while (std::chrono::steady_clock::now() < end) {
                WriteState(log, Fresh(*inbox));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        };
        const bool changed = gc::RunModeStep(target, get, set, observe);
        WriteState(log, Fresh(*inbox));
        log << "{\"event\":\"complete\",\"fsm_readback_verified\":true,\"mode_request_attempted\":"
            << (attempted ? "true" : "false") << ",\"already_in_target\":"
            << (changed ? "false" : "true") << ",\"physical_pose_verified\":false}\n" << std::flush;
        std::cout << "A1b FSM readback verified: " << target
                  << "; mode_request_attempted=" << attempted << '\n';
        return 0;
    } catch (const std::exception& error) {
        if (log.is_open()) {
            try {
                log << "{\"event\":\"failed\",\"mode_request_attempted\":"
                    << (attempted ? "true" : "false") << ",\"reason\":\""
                    << gc::JsonEscape(error.what()) << "\"}\n" << std::flush;
            } catch (...) {}
        }
        std::cerr << "A1b stopped: " << error.what() << '\n';
        return 1;
    }
}
