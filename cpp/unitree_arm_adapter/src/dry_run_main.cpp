#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "unitree_arm_adapter/periodic_loop.hpp"
#include "unitree_arm_adapter/safety.hpp"
#include "unitree_arm_adapter/seqlock.hpp"
#include "unitree_arm_adapter/shared_memory.hpp"

namespace ua = unitree_arm_adapter;

namespace {

struct Options {
    std::string shared_memory_name{"/g1_arm_mpc"};
    std::uint64_t period_us{2'000};
    std::uint64_t iterations{1'000};
    bool synthetic_input{false};
    bool print_layout{false};
    bool reset_shared_memory{false};
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
        << "用法: " << executable << " [选项]\n"
        << "  --shm-name NAME       POSIX共享内存名，默认/g1_arm_mpc\n"
        << "  --period-us N         周期，默认2000 us\n"
        << "  --iterations N        运行次数，0表示持续运行\n"
        << "  --synthetic-input     每拍写入有效的合成状态与命令\n"
        << "  --print-layout        输出共享内存ABI布局后退出\n"
        << "  --reset-shm           启动前显式删除同名共享内存\n"
        << "  --unlink-on-exit      正常退出时删除共享内存名字\n"
        << "\n该程序永远不会访问DDS，也没有--enable-output选项。\n";
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
        } else if (argument == "--period-us") {
            options.period_us = ParseUnsigned(
                require_value("--period-us"), "--period-us");
        } else if (argument == "--iterations") {
            options.iterations = ParseUnsigned(
                require_value("--iterations"), "--iterations");
        } else if (argument == "--synthetic-input") {
            options.synthetic_input = true;
        } else if (argument == "--print-layout") {
            options.print_layout = true;
        } else if (argument == "--reset-shm") {
            options.reset_shared_memory = true;
        } else if (argument == "--unlink-on-exit") {
            options.unlink_on_exit = true;
        } else if (argument == "--help" || argument == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.period_us == 0U) {
        throw std::invalid_argument("--period-us must be positive");
    }
    return options;
}

void PrintLayout() {
    std::cout
        << "protocol_version=" << ua::kProtocolVersion << '\n'
        << "layout_size=" << sizeof(ua::SharedMemoryLayout) << '\n'
        << "command_offset=" << offsetof(ua::SharedMemoryLayout, command) << '\n'
        << "command_payload_size=" << sizeof(ua::ArmCommandPayload) << '\n'
        << "state_offset=" << offsetof(ua::SharedMemoryLayout, state) << '\n'
        << "state_payload_size=" << sizeof(ua::RobotStatePayload) << '\n'
        << "status_offset=" << offsetof(ua::SharedMemoryLayout, status) << '\n'
        << "status_payload_size=" << sizeof(ua::AdapterStatusPayload) << '\n';
}

void WriteSyntheticInput(
    ua::SharedMemoryLayout& layout,
    std::uint64_t now_ns,
    std::uint64_t sample_id) {
    ua::RobotStatePayload state;
    state.monotonic_timestamp_ns = now_ns;
    state.sample_id = sample_id;
    state.robot_tick = static_cast<std::uint32_t>(sample_id);
    state.imu_quaternion_wxyz[0] = 1.0;
    ua::WriteSeqlock(layout.state, state);

    ua::ArmCommandPayload command;
    command.monotonic_timestamp_ns = now_ns;
    command.command_id = sample_id;
    command.mode = static_cast<std::uint32_t>(
        ua::CommandMode::kRobotPdPlusFeedforward);
    command.flags = ua::kCommandRequestOutput;
    command.arm_weight = 0.2;
    command.kp.fill(20.0);
    command.kd.fill(1.0);
    ua::WriteSeqlock(layout.command, command);
}

struct TimingSummary {
    double mean_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double max_us{0.0};
};

TimingSummary Summarize(std::vector<std::uint64_t> values) {
    TimingSummary summary;
    if (values.empty()) {
        return summary;
    }
    long double sum = 0.0L;
    for (const auto value : values) {
        sum += value;
    }
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double probability) {
        const auto rank = static_cast<std::size_t>(
            std::ceil(probability * static_cast<double>(values.size())));
        return values[std::min(values.size() - 1U, std::max<std::size_t>(1U, rank) - 1U)];
    };
    summary.mean_us = static_cast<double>(sum / values.size() / 1'000.0L);
    summary.p95_us = static_cast<double>(percentile(0.95)) / 1'000.0;
    summary.p99_us = static_cast<double>(percentile(0.99)) / 1'000.0;
    summary.max_us = static_cast<double>(values.back()) / 1'000.0;
    return summary;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        if (options.print_layout) {
            PrintLayout();
            return 0;
        }
        if (options.reset_shared_memory) {
            ua::SharedMemoryRegion::Unlink(options.shared_memory_name);
        }
        auto region = ua::SharedMemoryRegion::Open(
            options.shared_memory_name, true);
        auto& layout = *region.get();
        const auto safety = ua::MakeDefaultSafetyConfig();
        const std::uint64_t period_ns = options.period_us * 1'000ULL;
        ua::AbsolutePeriodicTimer timer(period_ns);

        std::uint64_t deadline_misses = 0;
        std::uint64_t stale_commands = 0;
        std::uint64_t stale_states = 0;
        std::uint64_t overtemperature = 0;
        std::vector<std::uint64_t> execution_samples;
        std::vector<std::uint64_t> wake_lateness_samples;
        if (options.iterations != 0U) {
            execution_samples.reserve(options.iterations);
            wake_lateness_samples.reserve(options.iterations);
        }
        bool prior_execution_healthy = true;
        std::uint64_t loop = 0;

        std::cout << "C++适配器干运行：不会创建DDS发布器，不会发送机器人命令。\n";
        while (options.iterations == 0U || loop < options.iterations) {
            const ua::PeriodicTick tick = timer.WaitNext();
            ++loop;
            if (options.synthetic_input) {
                WriteSyntheticInput(layout, tick.start_time_ns, loop);
            }

            ua::ArmCommandPayload command;
            ua::RobotStatePayload state;
            const bool command_valid = ua::ReadSeqlock(layout.command, command);
            const bool state_valid = ua::ReadSeqlock(layout.state, state);
            const bool deadline_healthy =
                tick.deadline_healthy && prior_execution_healthy;
            const ua::CommandPlan plan = ua::BuildCommandPlan(
                safety,
                command_valid ? &command : nullptr,
                state_valid ? &state : nullptr,
                tick.start_time_ns,
                deadline_healthy);

            if (plan.mode == ua::AdapterMode::kSafeReleaseCommandStale) {
                ++stale_commands;
            }
            if (plan.mode == ua::AdapterMode::kSafeReleaseStateStale) {
                ++stale_states;
            }
            if (plan.mode == ua::AdapterMode::kSafeReleaseOvertemperature) {
                ++overtemperature;
            }

            ua::AdapterStatusPayload status;
            status.monotonic_timestamp_ns = ua::MonotonicNowNs();
            status.loop_count = loop;
            status.command_id = command_valid ? command.command_id : 0U;
            status.command_age_ns = plan.command_age_ns;
            status.state_age_ns = plan.state_age_ns;
            status.wake_lateness_ns = tick.wake_lateness_ns;
            // mode保留实际安全判定；未设置OutputEnabled即表示仅干运行。
            status.mode = static_cast<std::uint32_t>(plan.mode);
            if (command_valid) {
                status.flags |= ua::kStatusCommandSnapshotValid;
            }
            if (state_valid) {
                status.flags |= ua::kStatusStateSnapshotValid;
            }
            if (plan.clamped) {
                status.flags |= ua::kStatusCommandClamped;
            }
            if (deadline_healthy) {
                status.flags |= ua::kStatusDeadlineHealthy;
            }

            const std::uint64_t finish_ns = ua::MonotonicNowNs();
            const std::uint64_t execution_ns = finish_ns - tick.start_time_ns;
            const bool execution_missed =
                finish_ns > tick.scheduled_time_ns + period_ns;
            if (!tick.deadline_healthy || execution_missed) {
                ++deadline_misses;
            }
            prior_execution_healthy = !execution_missed;
            status.execution_time_ns = execution_ns;
            status.deadline_miss_count = deadline_misses;
            status.command_stale_count = stale_commands;
            status.state_stale_count = stale_states;
            status.overtemperature_count = overtemperature;
            ua::WriteSeqlock(layout.status, status);

            if (options.iterations != 0U) {
                execution_samples.push_back(execution_ns);
                wake_lateness_samples.push_back(tick.wake_lateness_ns);
            }
        }

        const TimingSummary execution = Summarize(std::move(execution_samples));
        const TimingSummary wake = Summarize(std::move(wake_lateness_samples));
        std::cout << std::fixed << std::setprecision(3)
                  << "完成 " << loop << " 拍。\n"
                  << "核心执行[us] mean/p95/p99/max="
                  << execution.mean_us << '/' << execution.p95_us << '/'
                  << execution.p99_us << '/' << execution.max_us << "\n"
                  << "唤醒迟到[us] mean/p95/p99/max="
                  << wake.mean_us << '/' << wake.p95_us << '/'
                  << wake.p99_us << '/' << wake.max_us << "\n"
                  << "deadline_miss=" << deadline_misses
                  << ", command_stale=" << stale_commands
                  << ", state_stale=" << stale_states
                  << ", overtemperature=" << overtemperature << "。\n";
        if (options.unlink_on_exit) {
            ua::SharedMemoryRegion::Unlink(options.shared_memory_name);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "unitree_arm_adapter_dry_run错误: " << error.what() << '\n';
        return 1;
    }
}
