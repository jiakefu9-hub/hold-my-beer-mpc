#include "g1_commissioning/host_timing.hpp"
#include "g1_commissioning/core.hpp"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <sys/utsname.h>

namespace g1_commissioning {
namespace {
std::string Read(const std::string& path) {
    std::ifstream input(path);
    std::string result;
    std::getline(input, result);
    return input || !result.empty() ? result : "unavailable";
}
std::string Quote(const std::string& s) { return "\"" + JsonEscape(s) + "\""; }
void SetAffinity(const cpu_set_t& mask) {
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
        throw std::runtime_error(std::string("timing affinity failed: ") + std::strerror(errno));
}
cpu_set_t Affinity() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
        throw std::runtime_error("cannot read timing thread affinity");
    return mask;
}
void CheckOther() {
    if (sched_getscheduler(0) != SCHED_OTHER)
        throw std::runtime_error("ordinary-scheduler timing requires SCHED_OTHER; do not use chrt");
}
std::string MaskJson(const cpu_set_t& mask) {
    std::ostringstream out;
    out << '[';
    bool first = true;
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &mask)) {
        if (!first) out << ',';
        out << i;
        first = false;
    }
    out << ']';
    return out.str();
}
}

std::uint64_t HostNowNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
void PinTimingControl(int cpu) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) throw std::runtime_error("invalid timing CPU");
    CheckOther();
    cpu_set_t mask;
    CPU_ZERO(&mask); CPU_SET(cpu, &mask);
    SetAffinity(mask);
    const auto actual = Affinity();
    if (CPU_COUNT(&actual) != 1 || !CPU_ISSET(cpu, &actual))
        throw std::runtime_error("timing control CPU readback mismatch");
}
void PrepareTimingSupport(int cpu) {
    CheckOther();
    const auto original = Affinity();
    // Probe the explicitly requested CPU even if inherited housekeeping affinity
    // excludes it. The kernel still enforces cpuset/online permission boundaries.
    try { PinTimingControl(cpu); } catch (...) { SetAffinity(original); throw; }
    SetAffinity(original);
    auto support = original;
    const auto base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    const auto core = Read(base + "core_id"), package = Read(base + "physical_package_id");
    for (int other = 0; other < CPU_SETSIZE; ++other) if (CPU_ISSET(other, &support)) {
        const auto topology = "/sys/devices/system/cpu/cpu" + std::to_string(other) + "/topology/";
        if (other == cpu || (core != "unavailable" && package != "unavailable" &&
            Read(topology + "core_id") == core && Read(topology + "physical_package_id") == package))
            CPU_CLR(other, &support);
    }
    if (CPU_COUNT(&support) == 0)
        throw std::runtime_error("no support CPUs left; do not taskset the whole probe to its control CPU");
    SetAffinity(support);
}
std::string TimingEnvironmentJson(int cpu, const std::string& nic, const std::string& stage) {
    struct utsname kernel{};
    const bool have_kernel = uname(&kernel) == 0;
    const auto base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/";
    std::ostringstream out;
    out << "{\"schema\":\"g1_host_timing_environment_v1\",\"stage\":" << Quote(stage)
        << ",\"monotonic_ns\":" << HostNowNs() << ",\"control_cpu\":" << cpu
        << ",\"main_affinity\":" << MaskJson(Affinity())
        << ",\"main_scheduler\":" << sched_getscheduler(0)
        << ",\"kernel_release\":" << Quote(have_kernel ? kernel.release : "unavailable")
        << ",\"kernel_realtime\":" << Quote(Read("/sys/kernel/realtime"))
        << ",\"governor\":" << Quote(Read(base + "cpufreq/scaling_governor"))
        << ",\"frequency_khz_snapshot\":" << Quote(Read(base + "cpufreq/scaling_cur_freq"))
        << ",\"smt_siblings\":" << Quote(Read(base + "topology/thread_siblings_list"))
        << ",\"probe_period_ns\":" << kTimingPeriodNs
        << ",\"system_settings_changed\":false,\"threads\":[";
    bool first = true;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task")) {
        const auto tid_text = entry.path().filename().string();
        const auto tid = static_cast<pid_t>(std::stoi(tid_text));
        cpu_set_t mask; CPU_ZERO(&mask);
        if (sched_getaffinity(tid, sizeof(mask), &mask) != 0) continue;
        if (!first) out << ',';
        first = false;
        out << "{\"tid\":" << tid << ",\"name\":" << Quote(Read(entry.path().string() + "/comm"))
            << ",\"scheduler\":" << sched_getscheduler(tid) << ",\"affinity\":" << MaskJson(mask) << '}';
    }
    out << "],\"nic\":" << Quote(nic);
    if (!nic.empty() && nic.find('/') == std::string::npos && nic != "." && nic != "..") {
        for (const auto& name : {"carrier", "speed", "mtu", "statistics/rx_errors", "statistics/rx_dropped"})
            out << ',' << Quote(name) << ':' << Quote(Read("/sys/class/net/" + nic + "/" + name));
    }
    out << '}';
    return out.str();
}
std::string HostTimingJson(const HostTimingRecord& r) {
    std::ostringstream out;
    out << "{\"schema\":\"g1_host_probe_tick_v1\",\"scheduled_ns\":" << r.scheduled_ns
        << ",\"started_ns\":" << r.started_ns << ",\"snapshot_done_ns\":" << r.snapshot_done_ns
        << ",\"finished_ns\":" << r.finished_ns << ",\"state_received_ns\":" << r.state_received_ns
        << ",\"imu_received_ns\":" << r.imu_received_ns << ",\"state_sequence\":" << r.state_sequence
        << ",\"imu_sequence\":" << r.imu_sequence << ",\"missed_slots_after\":" << r.missed_slots_after
        << ",\"state_valid\":" << (r.state_valid ? "true" : "false")
        << ",\"imu_fresh\":" << (r.imu_fresh ? "true" : "false") << ",\"cpu\":" << r.cpu << '}';
    return out.str();
}
std::uint64_t NextTimingSlot(std::uint64_t scheduled, std::uint64_t finished, std::uint64_t& missed) {
    const auto next = scheduled + kTimingPeriodNs;
    missed = finished > next ? (finished - next + kTimingPeriodNs - 1) / kTimingPeriodNs : 0;
    return next + missed * kTimingPeriodNs;
}
}  // namespace g1_commissioning
