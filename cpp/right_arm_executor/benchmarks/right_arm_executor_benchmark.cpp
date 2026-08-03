#include "right_arm_executor/right_arm_executor_c.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t ParseIterations(int argc, char** argv) {
    if (argc < 2) {
        return 1'000'000;
    }
    char* end = nullptr;
    const auto value = std::strtoull(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || value == 0) {
        throw std::invalid_argument("iterations must be a positive integer");
    }
    return value;
}

void Check(std::int32_t status, const char* operation) {
    if (status != RAE_STATUS_OK) {
        throw std::runtime_error(
            std::string(operation) + ": " + rae_status_string(status));
    }
}

void RunOne(std::uint32_t semantics, std::uint64_t iterations) {
    rae_config_v1 config{};
    Check(rae_get_default_config_v1(semantics, &config), "default config");
    rae_executor_handle* handle = nullptr;
    Check(rae_create_v1(&config, &handle), "create");

    rae_input_v1 input{};
    input.struct_size = sizeof(input);
    input.abi_version = RAE_ABI_VERSION_V1;
    for (std::size_t joint = 0; joint < RAE_JOINT_COUNT; ++joint) {
        input.q[joint] = 0.01 * static_cast<double>(joint);
        input.dq[joint] = -0.02 * static_cast<double>(joint);
        input.q_ref[joint] = 0.02 * static_cast<double>(joint);
        input.dq_ref[joint] = 0.01;
        input.tau_ff[joint] = 0.1;
    }

    rae_output_v1 output{};
    std::int64_t now_ns = 1'000'000'000;
    constexpr std::uint64_t kWarmup = 10'000;
    for (std::uint64_t index = 0; index < kWarmup; ++index) {
        input.command_timestamp_ns = now_ns;
        input.state_timestamp_ns = now_ns;
        Check(rae_step_v1(handle, &input, now_ns, &output), "warmup step");
        now_ns += 2'000'000;
    }

    // 【非核心微基准】固定结构体复用，循环内不分配内存；sink 防止结果被优化掉。
    double sink = 0.0;
    std::uint64_t measured_core_ns = 0;
    const auto start = Clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
        input.command_timestamp_ns = now_ns;
        input.state_timestamp_ns = now_ns;
        Check(rae_step_v1(handle, &input, now_ns, &output), "benchmark step");
        sink += output.actuator_tau_ff[index % RAE_JOINT_COUNT];
        measured_core_ns += output.core_elapsed_ns;
        now_ns += 2'000'000;
    }
    const auto end = Clock::now();
    const double wall_ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    const double wall_ns_per_step = wall_ns / static_cast<double>(iterations);
    const double core_ns_per_step =
        static_cast<double>(measured_core_ns) / static_cast<double>(iterations);

    std::cout << std::fixed << std::setprecision(1)
              << rae_output_semantics_string(semantics)
              << ": iterations=" << iterations
              << " C_ABI_wall=" << wall_ns_per_step << " ns/step"
              << " core=" << core_ns_per_step << " ns/step"
              << " rate=" << (1.0e3 / wall_ns_per_step) << " Mstep/s"
              << " sink=" << sink << '\n';
    rae_destroy(handle);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint64_t iterations = ParseIterations(argc, argv);
        RunOne(RAE_OUTPUT_HOST_FULL_TORQUE, iterations);
        RunOne(RAE_OUTPUT_DEVICE_PD, iterations);
    } catch (const std::exception& error) {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
