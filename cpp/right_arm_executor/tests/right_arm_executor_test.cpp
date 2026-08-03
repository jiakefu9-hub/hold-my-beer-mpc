#include "right_arm_executor/right_arm_executor.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using right_arm_executor::ExecutorConfig;
using right_arm_executor::ExecutorInput;
using right_arm_executor::ExecutorMode;
using right_arm_executor::JointVector;
using right_arm_executor::OutputSemantics;
using right_arm_executor::RightArmExecutor;
using right_arm_executor::kJointCount;

int failures = 0;

void Expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void ExpectNear(double actual, double expected, const std::string& message) {
    if (std::abs(actual - expected) > 1e-12) {
        ++failures;
        std::cerr << "FAIL: " << message << ": expected " << expected
                  << ", got " << actual << '\n';
    }
}

ExecutorConfig MakeTestConfig() {
    ExecutorConfig config;
    config.kp.fill(10.0);
    config.kd.fill(2.0);
    config.timeout_damping.fill(3.0);
    config.q_ref_min.fill(-0.5);
    config.q_ref_max.fill(0.5);
    config.dq_ref_abs_max.fill(1.0);
    config.tau_min.fill(-5.0);
    config.tau_max.fill(5.0);
    config.command_timeout_ns = 30'000'000;
    config.state_timeout_ns = 10'000'000;
    return config;
}

void TestNominalPdAndFeedforward() {
    RightArmExecutor executor(MakeTestConfig());
    ExecutorInput input;
    input.command_timestamp_ns = 100;
    input.state_timestamp_ns = 100;
    input.q.fill(0.1);
    input.dq.fill(0.2);
    input.q_ref.fill(0.3);
    input.dq_ref.fill(0.1);
    input.tau_ff.fill(0.5);

    const auto output = executor.Step(input, 200);
    Expect(output.mode == ExecutorMode::kActive, "nominal command must be active");
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        // 0.5 + 10*(0.3-0.1) + 2*(0.1-0.2) = 2.3
        ExpectNear(output.tau_raw[joint], 2.3, "nominal raw torque");
        ExpectNear(output.tau_command[joint], 2.3, "nominal command torque");
        ExpectNear(output.actuator_kp[joint], 0.0, "host mode device kp must be zero");
        ExpectNear(output.actuator_kd[joint], 0.0, "host mode device kd must be zero");
        ExpectNear(output.actuator_tau_ff[joint], 2.3, "host mode sends full torque once");
    }
    Expect(!output.position_reference_clamped, "nominal position must not clamp");
    Expect(!output.velocity_reference_clamped, "nominal velocity must not clamp");
    Expect(!output.torque_clamped, "nominal torque must not clamp");
}

void TestAllActiveLimits() {
    RightArmExecutor executor(MakeTestConfig());
    ExecutorInput input;
    input.command_timestamp_ns = 0;
    input.state_timestamp_ns = 0;
    input.q.fill(0.0);
    input.q_ref.fill(4.0);
    input.dq_ref.fill(3.0);
    input.tau_ff.fill(20.0);

    const auto output = executor.Step(input, 0);
    Expect(output.mode == ExecutorMode::kActive, "limited command must remain active");
    Expect(output.position_reference_clamped, "position reference clamp must be reported");
    Expect(output.velocity_reference_clamped, "velocity reference clamp must be reported");
    Expect(output.torque_clamped, "torque clamp must be reported");
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        ExpectNear(output.effective_q_ref[joint], 0.5, "clamped position reference");
        ExpectNear(output.effective_dq_ref[joint], 1.0, "clamped velocity reference");
        ExpectNear(output.tau_raw[joint], 27.0, "raw torque before clamp");
        ExpectNear(output.tau_command[joint], 5.0, "clamped torque");
        ExpectNear(output.actuator_tau_ff[joint], 5.0, "host actuator torque clamp");
    }
}

void TestTimeoutUsesDampingOnly() {
    RightArmExecutor executor(MakeTestConfig());
    ExecutorInput input;
    input.command_timestamp_ns = 10;
    input.state_timestamp_ns = 30'000'011;
    input.dq = {1.0, -2.0, 0.5, -0.25, 3.0};
    input.q_ref.fill(0.4);
    input.tau_ff.fill(100.0);

    const auto output = executor.Step(input, 30'000'011);
    Expect(output.mode == ExecutorMode::kCommandTimedOut, "stale command must time out");
    const JointVector expected{-3.0, 5.0, -1.5, 0.75, -5.0};
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        ExpectNear(output.tau_command[joint], expected[joint], "timeout damping torque");
    }
    Expect(output.torque_clamped, "timeout damping torque clamp must be reported");
}

void TestInvalidCommandUsesDamping() {
    RightArmExecutor executor(MakeTestConfig());
    ExecutorInput input;
    input.command_timestamp_ns = 0;
    input.state_timestamp_ns = 0;
    input.dq.fill(0.5);
    input.q_ref[2] = std::numeric_limits<double>::quiet_NaN();

    const auto output = executor.Step(input, 0);
    Expect(output.mode == ExecutorMode::kInvalidCommand, "NaN command must be rejected");
    for (const double torque : output.tau_command) {
        ExpectNear(torque, -1.5, "invalid command damping torque");
    }
}

void TestInvalidStateOutputsZero() {
    RightArmExecutor executor(MakeTestConfig());
    ExecutorInput input;
    input.command_timestamp_ns = 0;
    input.state_timestamp_ns = 0;
    input.dq[1] = std::numeric_limits<double>::infinity();

    const auto output = executor.Step(input, 0);
    Expect(output.mode == ExecutorMode::kInvalidState, "invalid state must be reported");
    for (const double torque : output.tau_command) {
        ExpectNear(torque, 0.0, "invalid state zero torque");
    }
}

void TestFutureTimestampIsRejected() {
    RightArmExecutor executor(MakeTestConfig());
    ExecutorInput input;
    input.command_timestamp_ns = 101;
    input.state_timestamp_ns = 100;
    input.dq.fill(-0.25);

    const auto output = executor.Step(input, 100);
    Expect(output.mode == ExecutorMode::kInvalidCommand, "future timestamp must be rejected");
    for (const double torque : output.tau_command) {
        ExpectNear(torque, 0.75, "future timestamp damping torque");
    }
}

void TestDevicePdDoesNotRepeatPd() {
    ExecutorConfig config = MakeTestConfig();
    config.output_semantics = OutputSemantics::kDevicePd;
    RightArmExecutor executor(config);
    ExecutorInput input;
    input.command_timestamp_ns = 100;
    input.state_timestamp_ns = 100;
    input.q.fill(0.1);
    input.dq.fill(0.2);
    input.q_ref.fill(0.3);
    input.dq_ref.fill(0.1);
    input.tau_ff.fill(0.5);

    const auto output = executor.Step(input, 200);
    Expect(output.mode == ExecutorMode::kActive, "device-PD command must be active");
    Expect(output.device_total_torque_limit_required,
           "device-PD must require final device torque limit");
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        ExpectNear(output.pd_torque[joint], 1.8, "device-PD predicted PD torque");
        ExpectNear(output.tau_raw[joint], 2.3, "device-PD predicted total torque");
        ExpectNear(output.actuator_q_ref[joint], 0.3, "device-PD q reference");
        ExpectNear(output.actuator_dq_ref[joint], 0.1, "device-PD dq reference");
        ExpectNear(output.actuator_kp[joint], 10.0, "device-PD kp");
        ExpectNear(output.actuator_kd[joint], 2.0, "device-PD kd");
        // 关键断言：发给设备的 tau_ff 仍为 0.5，而不是已加 PD 的 2.3。
        ExpectNear(output.actuator_tau_ff[joint], 0.5,
                   "device-PD feedforward must exclude PD");
    }
}

void TestStateTimeoutHasIndependentFallback() {
    ExecutorConfig host_config = MakeTestConfig();
    RightArmExecutor host_executor(host_config);
    ExecutorInput input;
    input.command_timestamp_ns = 20'000'000;
    input.state_timestamp_ns = 0;
    input.dq.fill(1.0);

    const auto host_output = host_executor.Step(input, 20'000'000);
    Expect(host_output.mode == ExecutorMode::kStateTimedOut,
           "stale state must be distinguished from stale command");
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        ExpectNear(host_output.actuator_tau_ff[joint], 0.0,
                   "host mode cannot compute damping from stale state");
    }

    ExecutorConfig device_config = MakeTestConfig();
    device_config.output_semantics = OutputSemantics::kDevicePd;
    RightArmExecutor device_executor(device_config);
    const auto device_output = device_executor.Step(input, 20'000'000);
    Expect(device_output.mode == ExecutorMode::kStateTimedOut,
           "device-PD stale state mode");
    Expect(device_output.damping_fallback_active,
           "device-PD stale state must activate local damping");
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        ExpectNear(device_output.actuator_kp[joint], 0.0,
                   "fallback device kp must be zero");
        ExpectNear(device_output.actuator_kd[joint], 3.0,
                   "fallback device kd must use timeout damping");
        ExpectNear(device_output.actuator_tau_ff[joint], 0.0,
                   "fallback device feedforward must be zero");
    }
}

void TestBadConfigIsRejected() {
    ExecutorConfig config = MakeTestConfig();
    config.tau_min[0] = config.tau_max[0];
    bool threw = false;
    try {
        const RightArmExecutor executor(config);
        (void)executor;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Expect(threw, "invalid config must throw during construction");
}

}  // namespace

int main() {
    TestNominalPdAndFeedforward();
    TestAllActiveLimits();
    TestTimeoutUsesDampingOnly();
    TestInvalidCommandUsesDamping();
    TestInvalidStateOutputsZero();
    TestFutureTimestampIsRejected();
    TestDevicePdDoesNotRepeatPd();
    TestStateTimeoutHasIndependentFallback();
    TestBadConfigIsRejected();

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "All right_arm_executor tests passed\n";
    return 0;
}
