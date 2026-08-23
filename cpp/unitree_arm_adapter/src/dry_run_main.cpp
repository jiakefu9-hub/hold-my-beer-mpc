#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "unitree_arm_adapter/periodic_loop.hpp"
#include "unitree_arm_adapter/receipt.hpp"
#include "unitree_arm_adapter/safety.hpp"
#include "unitree_arm_adapter/seqlock.hpp"
#include "unitree_arm_adapter/shared_memory.hpp"

namespace ua = unitree_arm_adapter;

namespace {

struct Options {
    std::string shared_memory_name{"/g1_arm_mpc"};
    std::string csv_path;
    std::uint64_t period_us{2'000};
    std::uint64_t iterations{1'000};
    std::uint64_t warmup_iterations{0};
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
        << "  --warmup-iterations N 统计前预热拍数，默认0\n"
        << "  --csv PATH            结束后写逐拍CSV（热路径不写盘）\n"
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
        } else if (argument == "--warmup-iterations") {
            options.warmup_iterations = ParseUnsigned(
                require_value("--warmup-iterations"),
                "--warmup-iterations");
        } else if (argument == "--csv") {
            options.csv_path = require_value("--csv");
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
    if (options.iterations == 0U && !options.csv_path.empty()) {
        throw std::invalid_argument(
            "--csv requires finite --iterations to avoid unbounded memory");
    }
    if (options.iterations != 0U &&
        options.warmup_iterations >
            std::numeric_limits<std::uint64_t>::max() - options.iterations) {
        throw std::invalid_argument(
            "--warmup-iterations + --iterations overflows uint64");
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
    state.validated_timestamp_ns = now_ns;
    state.ingress_session_nonce = 1U;
    state.low_state_timestamp_ns = now_ns;
    state.torso_imu_timestamp_ns = now_ns;
    state.source_skew_ns = 0U;
    state.sample_id = sample_id;
    state.robot_tick = static_cast<std::uint32_t>(sample_id);
    state.ingress_flags = ua::kStateLowStateCrcValid |
                          ua::kStatePairedIngressValidated |
                          ua::kStateTorsoImuPresent |
                          ua::kStateSyntheticFixture;
    state.imu_quaternion_wxyz[0] = 1.0;
    ua::WriteSeqlock(layout.state, state);

    ua::ArmCommandPayload command;
    command.monotonic_timestamp_ns = now_ns;
    command.producer_sequence = sample_id - 1U;
    command.command_id = sample_id;
    command.source_sample_id = sample_id;
    command.source_timestamp_ns = now_ns;
    command.task_time_ns = (sample_id - 1U) * 6'000'000ULL;
    command.full_task_anchor = sample_id - 1U;
    command.expires_timestamp_ns = now_ns + 30'000'000ULL;
    command.session_nonce = 1U;
    command.task_epoch_id = 1U;
    command.safety_policy_id = 1U;
    command.mode = static_cast<std::uint32_t>(
        ua::CommandMode::kRobotPdPlusFeedforward);
    command.flags = ua::kCommandRequestOutput |
                    ua::kCommandRequestActive;
    command.active_mask = (1U << ua::kArmSdkJointCount) - 1U;
    command.safety_policy_sha256.fill(0xa5U);
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

struct SignedTimingSummary {
    double mean_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double max_us{0.0};
};

// 【非核心诊断】统计窗口先保存在内存，CSV 在控制循环结束后统一写盘。
struct TimingSample {
    std::uint64_t measurement_index{0};
    std::uint64_t loop_index{0};
    std::uint64_t scheduled_time_ns{0};
    std::uint64_t start_time_ns{0};
    std::uint64_t finish_time_ns{0};
    std::uint64_t wake_lateness_ns{0};
    std::uint64_t work_time_ns{0};
    std::int64_t completion_lateness_ns{0};
    std::uint64_t period_jitter_ns{0};
    std::uint64_t deadline_miss_event{0};
    std::uint64_t skipped_periods{0};
    std::uint64_t command_age_at_start_ns{0};
    std::uint64_t state_age_at_start_ns{0};
    std::uint64_t command_age_at_finish_ns{0};
    std::uint64_t state_age_at_finish_ns{0};
    std::uint32_t mode{0};
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

SignedTimingSummary SummarizeSigned(std::vector<std::int64_t> values) {
    SignedTimingSummary summary;
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
        return values[
            std::min(values.size() - 1U,
                     std::max<std::size_t>(1U, rank) - 1U)];
    };
    summary.mean_us = static_cast<double>(sum / values.size() / 1'000.0L);
    summary.p95_us = static_cast<double>(percentile(0.95)) / 1'000.0;
    summary.p99_us = static_cast<double>(percentile(0.99)) / 1'000.0;
    summary.max_us = static_cast<double>(values.back()) / 1'000.0;
    return summary;
}

std::int64_t SignedDifference(
    std::uint64_t left, std::uint64_t right) noexcept {
    constexpr std::uint64_t kMax =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (left >= right) {
        return static_cast<std::int64_t>(std::min(left - right, kMax));
    }
    return -static_cast<std::int64_t>(std::min(right - left, kMax));
}

std::uint64_t AbsolutePeriodJitter(
    std::uint64_t current_start,
    std::uint64_t previous_start,
    std::uint64_t period_ns) noexcept {
    if (previous_start == 0U) {
        return 0U;
    }
    const std::uint64_t actual_period = current_start - previous_start;
    return actual_period >= period_ns
        ? actual_period - period_ns
        : period_ns - actual_period;
}

std::uint64_t TimestampAgeAt(
    std::uint64_t timestamp, std::uint64_t now) noexcept {
    return timestamp != 0U && timestamp <= now ? now - timestamp : 0U;
}

void WriteCsv(
    const std::string& path, const std::vector<TimingSample>& samples) {
    if (path.empty()) {
        return;
    }
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open timing CSV: " + path);
    }
    stream
        << "measurement_index,loop_index,scheduled_time_ns,start_time_ns,"
        << "finish_time_ns,wake_lateness_ns,work_time_ns,"
        << "completion_lateness_ns,period_jitter_ns,deadline_miss_event,"
        << "skipped_periods,command_age_at_start_ns,state_age_at_start_ns,"
        << "command_age_at_finish_ns,state_age_at_finish_ns,mode\n";
    for (const auto& sample : samples) {
        stream
            << sample.measurement_index << ',' << sample.loop_index << ','
            << sample.scheduled_time_ns << ',' << sample.start_time_ns << ','
            << sample.finish_time_ns << ',' << sample.wake_lateness_ns << ','
            << sample.work_time_ns << ',' << sample.completion_lateness_ns
            << ',' << sample.period_jitter_ns << ','
            << sample.deadline_miss_event << ',' << sample.skipped_periods
            << ',' << sample.command_age_at_start_ns << ','
            << sample.state_age_at_start_ns << ','
            << sample.command_age_at_finish_ns << ','
            << sample.state_age_at_finish_ns << ',' << sample.mode << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing timing CSV: " + path);
    }
}

template <typename Value>
std::vector<Value> ExtractSamples(
    const std::vector<TimingSample>& samples,
    Value TimingSample::* field) {
    std::vector<Value> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) {
        values.push_back(sample.*field);
    }
    return values;
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
        const bool finite_run = options.iterations != 0U;
        const std::uint64_t total_iterations = finite_run
            ? options.warmup_iterations + options.iterations
            : 0U;

        std::uint64_t deadline_misses = 0;
        std::uint64_t skipped_periods = 0;
        std::uint64_t stale_commands = 0;
        std::uint64_t stale_states = 0;
        std::uint64_t overtemperature = 0;
        std::vector<TimingSample> timing_samples;
        if (finite_run) {
            timing_samples.reserve(options.iterations);
        }
        bool prior_execution_healthy = true;
        std::uint64_t loop = 0;
        std::uint64_t measured_loops = 0;
        std::uint64_t previous_measured_start_ns = 0;

        std::cout << "C++适配器干运行：不会创建DDS发布器，不会发送机器人命令。\n";
        while (!finite_run || loop < total_iterations) {
            // 预热不进入统计，并在第一拍正式样本前清空预热阶段计数。
            if (loop == options.warmup_iterations) {
                deadline_misses = 0;
                skipped_periods = 0;
                stale_commands = 0;
                stale_states = 0;
                overtemperature = 0;
                prior_execution_healthy = true;
            }
            const ua::PeriodicTick tick = timer.WaitNext();
            ++loop;
            const bool measuring = loop > options.warmup_iterations;
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

            const std::uint64_t deadline_ns =
                tick.scheduled_time_ns + period_ns;
            // status中的本拍耗时是写入前下界；正式benchmark以写入后的
            // finish_ns为准，因此work_time完整包含一次status seqlock写。
            const std::uint64_t before_status_ns = ua::MonotonicNowNs();
            const bool provisional_miss = before_status_ns > deadline_ns;
            ua::ReceiptContext receipt_context;
            receipt_context.receipt_timestamp_ns = before_status_ns;
            receipt_context.loop_count = loop;
            receipt_context.receipt_id = loop;
            receipt_context.wake_lateness_ns = tick.wake_lateness_ns;
            receipt_context.execution_time_ns =
                before_status_ns - tick.start_time_ns;
            receipt_context.deadline_miss_count =
                deadline_misses + (provisional_miss ? 1U : 0U);
            receipt_context.command_stale_count = stale_commands;
            receipt_context.state_stale_count = stale_states;
            receipt_context.overtemperature_count = overtemperature;
            receipt_context.command_snapshot_valid = command_valid;
            receipt_context.state_snapshot_valid = state_valid;
            receipt_context.deadline_healthy = deadline_healthy;
            receipt_context.pre_sink_check_timestamp_ns = before_status_ns;
            receipt_context.pre_sink_deadline_ns = deadline_ns;
            receipt_context.pre_sink_deadline_healthy = !provisional_miss;
            receipt_context.pre_sink_expiry_healthy = command_valid &&
                command.expires_timestamp_ns != 0U &&
                before_status_ns <= command.expires_timestamp_ns;
            const ua::AdapterStatusPayload status = ua::BuildAdapterReceipt(
                command_valid ? &command : nullptr,
                state_valid ? &state : nullptr,
                plan,
                receipt_context);
            ua::WriteSeqlock(layout.status, status);

            const std::uint64_t finish_ns = ua::MonotonicNowNs();
            const std::uint64_t work_ns = finish_ns - tick.start_time_ns;
            const bool deadline_missed = finish_ns > deadline_ns;
            if (deadline_missed) {
                ++deadline_misses;
            }
            skipped_periods += tick.missed_periods;
            prior_execution_healthy = !deadline_missed;

            if (measuring && finite_run) {
                ++measured_loops;
                const std::uint64_t period_jitter_ns = AbsolutePeriodJitter(
                    tick.start_time_ns,
                    previous_measured_start_ns,
                    period_ns);
                previous_measured_start_ns = tick.start_time_ns;
                const std::uint64_t command_age_finish_ns = command_valid
                    ? TimestampAgeAt(command.monotonic_timestamp_ns, finish_ns)
                    : 0U;
                const std::uint64_t state_age_finish_ns = state_valid
                    ? TimestampAgeAt(state.monotonic_timestamp_ns, finish_ns)
                    : 0U;
                const std::int64_t completion_lateness_ns =
                    SignedDifference(finish_ns, deadline_ns);

                TimingSample sample;
                sample.measurement_index = measured_loops;
                sample.loop_index = loop;
                sample.scheduled_time_ns = tick.scheduled_time_ns;
                sample.start_time_ns = tick.start_time_ns;
                sample.finish_time_ns = finish_ns;
                sample.wake_lateness_ns = tick.wake_lateness_ns;
                sample.work_time_ns = work_ns;
                sample.completion_lateness_ns = completion_lateness_ns;
                sample.period_jitter_ns = period_jitter_ns;
                sample.deadline_miss_event = deadline_missed ? 1U : 0U;
                sample.skipped_periods = tick.missed_periods;
                sample.command_age_at_start_ns = plan.command_age_ns;
                sample.state_age_at_start_ns = plan.state_age_ns;
                sample.command_age_at_finish_ns = command_age_finish_ns;
                sample.state_age_at_finish_ns = state_age_finish_ns;
                sample.mode = static_cast<std::uint32_t>(plan.mode);
                timing_samples.push_back(sample);
            }
        }

        WriteCsv(options.csv_path, timing_samples);
        const TimingSummary work = Summarize(ExtractSamples(
            timing_samples, &TimingSample::work_time_ns));
        const TimingSummary wake = Summarize(ExtractSamples(
            timing_samples, &TimingSample::wake_lateness_ns));
        const TimingSummary jitter = Summarize(ExtractSamples(
            timing_samples, &TimingSample::period_jitter_ns));
        const TimingSummary command_age = Summarize(ExtractSamples(
            timing_samples, &TimingSample::command_age_at_finish_ns));
        const TimingSummary state_age = Summarize(ExtractSamples(
            timing_samples, &TimingSample::state_age_at_finish_ns));
        const SignedTimingSummary completion = SummarizeSigned(ExtractSamples(
            timing_samples, &TimingSample::completion_lateness_ns));
        std::cout << std::fixed << std::setprecision(3)
                  << "完成 " << loop << " 拍（预热 "
                  << options.warmup_iterations << "，统计 "
                  << measured_loops << "）。\n"
                  << "完整工作[us] mean/p95/p99/max="
                  << work.mean_us << '/' << work.p95_us << '/'
                  << work.p99_us << '/' << work.max_us << "\n"
                  << "唤醒迟到[us] mean/p95/p99/max="
                  << wake.mean_us << '/' << wake.p95_us << '/'
                  << wake.p99_us << '/' << wake.max_us << "\n"
                  << "周期抖动绝对值[us] mean/p95/p99/max="
                  << jitter.mean_us << '/' << jitter.p95_us << '/'
                  << jitter.p99_us << '/' << jitter.max_us << "\n"
                  << "完成相对deadline[us] mean/p95/p99/max="
                  << completion.mean_us << '/' << completion.p95_us << '/'
                  << completion.p99_us << '/' << completion.max_us
                  << "（负值表示余量）\n"
                  << "命令年龄@finish[us] mean/p95/p99/max="
                  << command_age.mean_us << '/' << command_age.p95_us << '/'
                  << command_age.p99_us << '/' << command_age.max_us << "\n"
                  << "状态年龄@finish[us] mean/p95/p99/max="
                  << state_age.mean_us << '/' << state_age.p95_us << '/'
                  << state_age.p99_us << '/' << state_age.max_us << "\n"
                  << "deadline_miss=" << deadline_misses
                  << ", skipped_periods=" << skipped_periods
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
