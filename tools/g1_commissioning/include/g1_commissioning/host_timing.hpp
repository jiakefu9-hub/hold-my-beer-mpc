#pragma once

#include <cstdint>
#include <string>

namespace g1_commissioning {

constexpr std::uint64_t kTimingPeriodNs = 6000000ULL;
std::uint64_t HostNowNs();
// Only this process's calling thread is changed. No scheduler, governor, IRQ,
// kernel or robot settings are changed. New threads inherit the support mask.
void PrepareTimingSupport(int control_cpu);
void PinTimingControl(int control_cpu);
std::string TimingEnvironmentJson(int control_cpu, const std::string& nic,
                                  const std::string& stage);

struct HostTimingRecord {
    std::uint64_t scheduled_ns{0}, started_ns{0}, snapshot_done_ns{0}, finished_ns{0};
    std::uint64_t state_received_ns{0}, imu_received_ns{0};
    std::uint64_t state_sequence{0}, imu_sequence{0}, missed_slots_after{0};
    bool state_valid{false}, imu_fresh{false};
    int cpu{-1};
};
std::string HostTimingJson(const HostTimingRecord& record);
// Advance the original absolute grid. Skip expired slots, never catch up in a
// burst or redefine the phase to now+period after each late wakeup.
std::uint64_t NextTimingSlot(std::uint64_t scheduled, std::uint64_t finished,
                             std::uint64_t& missed);

}  // namespace g1_commissioning
