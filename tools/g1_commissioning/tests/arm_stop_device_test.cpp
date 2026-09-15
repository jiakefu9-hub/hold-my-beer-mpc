#include "g1_commissioning/device_fsm_monitor.hpp"
#include <unitree/dds_wrapper/common/crc.h>
#include <iostream>

namespace gc = g1_commissioning;

int main() {
    // Local messages only: no factory, client, subscriber or publisher is started.
    unitree_hg::msg::dds_::LowState_ message;
    message.tick(100);
    auto stamp = [&] {
        message.crc(crc32_core(reinterpret_cast<std::uint32_t*>(&message),
            static_cast<std::uint32_t>(sizeof(message) / sizeof(std::uint32_t) - 1)));
    };
    gc::LowStateInbox inbox;
    gc::ArmStopInterlock gate;
    const auto t = gc::MonotonicNowNs();
    gate.ObserveMode(0, 4, t, t);
    stamp();
    gc::HandleArmStopState(inbox, gate, &message);
    if (!gate.Check(t).empty()) return 1;
    message.wireless_remote()[2] = 0x20;
    message.wireless_remote()[3] = 0x02;
    stamp();
    gc::HandleArmStopState(inbox, gate, &message);
    message.wireless_remote()[2] = 0;
    message.wireless_remote()[3] = 0;
    stamp();
    gc::HandleArmStopState(inbox, gate, &message);
    if (gate.Check(t) != "remote L2+B observed") return 2;
    gc::LowStateInbox invalid_inbox;
    gc::ArmStopInterlock invalid_gate;
    invalid_gate.ObserveMode(0, 4, t, t);
    message.wireless_remote()[2] = 0x20; // deliberately do not update CRC
    message.wireless_remote()[3] = 0x02;
    gc::HandleArmStopState(invalid_inbox, invalid_gate, &message);
    if (invalid_inbox.crc_rejected_count() != 1) return 3;
    if (!invalid_gate.Check(t).empty()) return 4;
    gc::HandleArmStopState(invalid_inbox, invalid_gate, nullptr);
    std::cout << "offline_device_stop_decode_passed=true\n";
    return 0;
}
