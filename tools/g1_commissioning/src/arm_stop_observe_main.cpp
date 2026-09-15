// Publisher-free, three-second live check of the exact A2 stop monitoring path.
#include "g1_commissioning/device_fsm_monitor.hpp"
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace gc = g1_commissioning;

int main(int argc, char** argv) {
    if (argc != 4 || std::string(argv[3]) != "A2_READ_ONLY_STOP_OBSERVE") {
        std::cerr << "Usage: " << argv[0]
                  << " NIC NEW_JSONL A2_READ_ONLY_STOP_OBSERVE\n";
        return 1;
    }
    try {
        if (std::filesystem::exists(argv[2])) throw std::runtime_error("log exists");
        std::ofstream log(argv[2]);
        if (!log) throw std::runtime_error("cannot create log");
        log << "{\"query_only\":true,\"command_publisher_created\":false}\n";
        unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
        auto inbox = std::make_shared<gc::LowStateInbox>();
        auto gate = std::make_shared<gc::ArmStopInterlock>();
        unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>
            subscriber("rt/lowstate");
        subscriber.InitChannel([inbox, gate](const void* message) {
            gc::HandleArmStopState(*inbox, *gate, message);
        }, 1);
        gc::FsmMonitor monitor(gate);
        const auto start = gc::MonotonicNowNs();
        bool ready = false;
        while (gc::MonotonicNowNs() - start < 3000000000ULL) {
            const auto state = inbox->Latest();
            const auto now = gc::MonotonicNowNs();
            const auto reason = gate->Check(now);
            const bool state_ok = state && state->crc_valid && !state->tick_regression &&
                now >= state->captured_monotonic_ns &&
                now - state->captured_monotonic_ns <= 20000000ULL;
            log << "{\"monotonic_ns\":" << now
                << ",\"state_ok\":" << (state_ok ? "true" : "false")
                << ",\"interlock_reason\":\"" << gc::JsonEscape(reason) << "\"}\n";
            const bool startup_wait = !ready && now - start < 500000000ULL;
            if (!reason.empty() && !(startup_wait && reason == "FSM not yet observed"))
                throw std::runtime_error(reason);
            if (!state_ok && !startup_wait) throw std::runtime_error("LowState gate failed");
            if (reason.empty() && state_ok) ready = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!ready) throw std::runtime_error("monitor never became ready");
        log << "{\"outcome\":\"read_only_stop_observation_passed\"}\n";
        std::cout << "read_only_stop_observation_passed=true\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Read-only interlock observation failed: " << error.what() << '\n';
        return 2;
    }
}
