#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "unitree_arm_adapter/periodic_loop.hpp"
#include "unitree_arm_adapter/safety.hpp"
#include "unitree_arm_adapter/seqlock.hpp"
#include "unitree_arm_adapter/shared_memory.hpp"

namespace ua = unitree_arm_adapter;

namespace {

constexpr const char* kArmSdkTopic = "rt/arm_sdk";
constexpr const char* kLowStateTopic = "rt/lowstate";
volatile std::sig_atomic_t stop_requested = 0;

void HandleSignal(int) { stop_requested = 1; }

struct Options {
    std::string network_interface;
    std::string shared_memory_name{"/g1_arm_mpc"};
    std::uint64_t period_us{2'000};
    std::uint64_t iterations{0};
    bool enable_output{false};
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
        << "用法: " << executable << " NETWORK_INTERFACE [选项]\n"
        << "  --shm-name NAME       POSIX共享内存名，默认/g1_arm_mpc\n"
        << "  --period-us N         高频适配周期，默认2000 us\n"
        << "  --iterations N        0表示持续运行\n"
        << "  --enable-output       真正向rt/arm_sdk发布（高风险）\n"
        << "\n默认只订阅LowState并写共享内存，绝不会发布命令。\n"
        << "即使启用输出，每条上游命令仍必须带request-output标志。\n";
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
        } else if (argument == "--enable-output") {
            options.enable_output = true;
        } else if (argument == "--help" || argument == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (!argument.empty() && argument.front() != '-' &&
                   options.network_interface.empty()) {
            options.network_interface = argument;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.network_interface.empty()) {
        throw std::invalid_argument("NETWORK_INTERFACE is required");
    }
    if (options.period_us == 0U) {
        throw std::invalid_argument("--period-us must be positive");
    }
    return options;
}

ua::RobotStatePayload ConvertState(
    const unitree_hg::msg::dds_::LowState_& message,
    std::uint64_t sample_id) {
    ua::RobotStatePayload state;
    state.monotonic_timestamp_ns = ua::MonotonicNowNs();
    state.sample_id = sample_id;
    state.robot_tick = message.tick();
    state.mode_pr = message.mode_pr();
    state.mode_machine = message.mode_machine();
    for (std::size_t index = 0; index < ua::kMotorCount; ++index) {
        const auto& motor = message.motor_state().at(index);
        state.q[index] = static_cast<double>(motor.q());
        state.dq[index] = static_cast<double>(motor.dq());
        state.ddq[index] = static_cast<double>(motor.ddq());
        state.tau_est[index] = static_cast<double>(motor.tau_est());
        state.motor_temperature_c[index][0] = motor.temperature().at(0);
        state.motor_temperature_c[index][1] = motor.temperature().at(1);
    }
    const auto& imu = message.imu_state();
    for (std::size_t index = 0; index < 4; ++index) {
        state.imu_quaternion_wxyz[index] =
            static_cast<double>(imu.quaternion().at(index));
    }
    for (std::size_t index = 0; index < 3; ++index) {
        state.imu_gyroscope[index] =
            static_cast<double>(imu.gyroscope().at(index));
        state.imu_accelerometer[index] =
            static_cast<double>(imu.accelerometer().at(index));
        state.imu_rpy[index] = static_cast<double>(imu.rpy().at(index));
    }
    return state;
}

void FillLowCommand(
    const ua::CommandPlan& plan,
    unitree_hg::msg::dds_::LowCmd_& message) {
    // 【核心代码】13维数组与官方arm5示例使用完全相同的关节顺序。
    for (std::size_t local = 0; local < ua::kArmSdkJointCount; ++local) {
        auto& motor = message.motor_cmd().at(ua::kArmSdkMotorIndices[local]);
        motor.q(static_cast<float>(plan.q[local]));
        motor.dq(static_cast<float>(plan.dq[local]));
        motor.kp(static_cast<float>(plan.kp[local]));
        motor.kd(static_cast<float>(plan.kd[local]));
        motor.tau(static_cast<float>(plan.tau[local]));
    }
    message.motor_cmd().at(ua::kArmWeightMotorIndex)
        .q(static_cast<float>(plan.arm_weight));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        std::signal(SIGINT, HandleSignal);
        std::signal(SIGTERM, HandleSignal);

        auto region = ua::SharedMemoryRegion::Open(
            options.shared_memory_name, true);
        auto* layout = region.get();
        std::atomic<std::uint64_t> state_sample_id{0};

        unitree::robot::ChannelFactory::Instance()->Init(
            0, options.network_interface);
        unitree::robot::ChannelSubscriberPtr<
            unitree_hg::msg::dds_::LowState_> subscriber;
        subscriber.reset(new unitree::robot::ChannelSubscriber<
                         unitree_hg::msg::dds_::LowState_>(kLowStateTopic));
        subscriber->InitChannel(
            [layout, &state_sample_id](const void* raw_message) {
                if (raw_message == nullptr) {
                    return;
                }
                const auto& message = *static_cast<const
                    unitree_hg::msg::dds_::LowState_*>(raw_message);
                const auto state = ConvertState(
                    message,
                    state_sample_id.fetch_add(1, std::memory_order_relaxed) + 1U);
                ua::WriteSeqlock(layout->state, state);
            },
            1);

        unitree::robot::ChannelPublisherPtr<
            unitree_hg::msg::dds_::LowCmd_> publisher;
        if (options.enable_output) {
            publisher.reset(new unitree::robot::ChannelPublisher<
                            unitree_hg::msg::dds_::LowCmd_>(kArmSdkTopic));
            publisher->InitChannel();
            std::cerr
                << "警告：--enable-output已启用，满足双重许可和安全检查后会发布命令。\n";
        } else {
            std::cout
                << "DDS只读干运行：订阅rt/lowstate，但不创建rt/arm_sdk发布器。\n";
        }

        const auto safety = ua::MakeDefaultSafetyConfig();
        const std::uint64_t period_ns = options.period_us * 1'000ULL;
        ua::AbsolutePeriodicTimer timer(period_ns);
        unitree_hg::msg::dds_::LowCmd_ low_command;
        std::uint64_t loop = 0;
        std::uint64_t deadline_misses = 0;
        std::uint64_t command_stale = 0;
        std::uint64_t state_stale = 0;
        std::uint64_t overtemperature = 0;
        std::uint64_t write_count = 0;
        std::uint64_t max_execution_ns = 0;
        long double execution_sum_ns = 0.0L;
        bool prior_execution_healthy = true;

        while (!stop_requested &&
               (options.iterations == 0U || loop < options.iterations)) {
            const ua::PeriodicTick tick = timer.WaitNext();
            ++loop;
            ua::ArmCommandPayload command;
            ua::RobotStatePayload state;
            const bool command_valid = ua::ReadSeqlock(layout->command, command);
            const bool state_valid = ua::ReadSeqlock(layout->state, state);
            const bool deadline_healthy =
                tick.deadline_healthy && prior_execution_healthy;
            const ua::CommandPlan plan = ua::BuildCommandPlan(
                safety,
                command_valid ? &command : nullptr,
                state_valid ? &state : nullptr,
                tick.start_time_ns,
                deadline_healthy);

            bool wrote = false;
            if (options.enable_output) {
                FillLowCommand(plan, low_command);
                publisher->Write(low_command);
                wrote = true;
                ++write_count;
            }
            if (plan.mode == ua::AdapterMode::kSafeReleaseCommandStale) {
                ++command_stale;
            }
            if (plan.mode == ua::AdapterMode::kSafeReleaseStateStale) {
                ++state_stale;
            }
            if (plan.mode == ua::AdapterMode::kSafeReleaseOvertemperature) {
                ++overtemperature;
            }

            const std::uint64_t finish_ns = ua::MonotonicNowNs();
            const std::uint64_t execution_ns = finish_ns - tick.start_time_ns;
            const bool execution_missed =
                finish_ns > tick.scheduled_time_ns + period_ns;
            if (!tick.deadline_healthy || execution_missed) {
                ++deadline_misses;
            }
            prior_execution_healthy = !execution_missed;

            ua::AdapterStatusPayload status;
            status.monotonic_timestamp_ns = finish_ns;
            status.loop_count = loop;
            status.command_id = command_valid ? command.command_id : 0U;
            status.command_age_ns = plan.command_age_ns;
            status.state_age_ns = plan.state_age_ns;
            status.wake_lateness_ns = tick.wake_lateness_ns;
            status.execution_time_ns = execution_ns;
            status.deadline_miss_count = deadline_misses;
            status.command_stale_count = command_stale;
            status.state_stale_count = state_stale;
            status.overtemperature_count = overtemperature;
            status.mode = static_cast<std::uint32_t>(plan.mode);
            if (options.enable_output) {
                status.flags |= ua::kStatusOutputEnabled;
            }
            if (wrote) {
                status.flags |= ua::kStatusDdsWritePerformed;
            }
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
            ua::WriteSeqlock(layout->status, status);
            max_execution_ns = std::max(max_execution_ns, execution_ns);
            execution_sum_ns += static_cast<long double>(execution_ns);
        }

        // 退出前仅在用户明确启用输出时发送一次weight=0释放帧。
        if (options.enable_output) {
            ua::RobotStatePayload state;
            const bool state_valid = ua::ReadSeqlock(layout->state, state);
            const auto release = ua::BuildCommandPlan(
                safety,
                nullptr,
                state_valid ? &state : nullptr,
                ua::MonotonicNowNs(),
                true);
            FillLowCommand(release, low_command);
            publisher->Write(low_command);
        }

        const double mean_us = loop == 0U
                                   ? 0.0
                                   : static_cast<double>(
                                         execution_sum_ns / loop / 1'000.0L);
        std::cout << std::fixed << std::setprecision(3)
                  << "退出：loop=" << loop << ", DDS写入=" << write_count
                  << ", core mean/max=" << mean_us << '/'
                  << static_cast<double>(max_execution_ns) / 1'000.0
                  << " us, deadline_miss=" << deadline_misses
                  << ", command_stale=" << command_stale
                  << ", state_stale=" << state_stale
                  << ", overtemperature=" << overtemperature << "。\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "unitree_arm_adapter_dds错误: " << error.what() << '\n';
        return 1;
    }
}
