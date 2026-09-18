#include "g1_commissioning/host_timing.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

namespace gc = g1_commissioning;
int main(int argc, char** argv) {
    std::uint64_t missed = 99;
    if (gc::NextTimingSlot(10000000, 10000001, missed) != 16000000 || missed != 0) return 1;
    if (gc::NextTimingSlot(10000000, 16000000, missed) != 16000000 || missed != 0) return 2;
    if (gc::NextTimingSlot(10000000, 16000001, missed) != 22000000 || missed != 1) return 3;
    if (gc::NextTimingSlot(10000000, 28000001, missed) != 34000000 || missed != 3) return 4;
    gc::HostTimingRecord r;
    r.scheduled_ns = 100; r.started_ns = 101; r.finished_ns = 102; r.cpu = 7;
    const auto json = gc::HostTimingJson(r);
    if (json.find("\"scheduled_ns\":100") == std::string::npos ||
        json.find("\"cpu\":7") == std::string::npos) return 5;
    bool rejected = false;
    try { gc::PinTimingControl(-1); } catch (...) { rejected = true; }
    if (!rejected) return 6;
    if (argc == 3 && std::string(argv[1]) == "--host-sample") {
        const int cpu = std::stoi(argv[2]);
        gc::PrepareTimingSupport(cpu);
        gc::PinTimingControl(cpu);
        std::cout << gc::TimingEnvironmentJson(cpu, "", "offline_host_only") << '\n';
        // Optional local smoke only: no SDK/DDS, no state processing or MPC.
        std::vector<double> lateness;
        auto scheduled = gc::HostNowNs() + gc::kTimingPeriodNs;
        std::uint64_t skipped = 0;
        for (int i = 0; i < 200; ++i) {
            std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::nanoseconds(scheduled)));
            const auto now = gc::HostNowNs();
            lateness.push_back(static_cast<double>(now-scheduled)*1e-6);
            scheduled = gc::NextTimingSlot(scheduled, gc::HostNowNs(), missed);
            skipped += missed;
        }
        std::sort(lateness.begin(), lateness.end());
        std::cout << "{\"scope\":\"offline_host_timer_only_not_dds\",\"samples\":200,\"mean_ms\":"
            << std::accumulate(lateness.begin(), lateness.end(), 0.0)/200.0
            << ",\"p99_nearest_rank_ms\":" << lateness[197] << ",\"max_ms\":" << lateness.back()
            << ",\"skipped_slots\":" << skipped << "}\n";
    } else if (argc != 1) return 7;
    return 0;
}
