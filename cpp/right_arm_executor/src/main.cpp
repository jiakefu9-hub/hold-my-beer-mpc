#include "right_arm_executor/right_arm_executor.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

int main() {
    using namespace right_arm_executor;
    using Clock = std::chrono::steady_clock;

    RightArmExecutor executor(MakeProjectDefaultConfig());
    const auto now = Clock::now().time_since_epoch();
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

    ExecutorInput input;
    input.command_timestamp_ns = now_ns;
    input.state_timestamp_ns = now_ns;
    input.q_ref = {0.02, -0.01, 0.03, 0.10, -0.05};
    input.tau_ff = {0.5, -0.2, 0.1, 0.3, -0.1};

    const ExecutorOutput output = executor.Step(input, now_ns);
    std::cout << "mode=" << ToString(output.mode)
              << " semantics=" << ToString(output.output_semantics)
              << " actuator_tau_ff=[";
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        if (joint != 0) {
            std::cout << ", ";
        }
        std::cout << std::fixed << std::setprecision(3)
                  << output.actuator_tau_ff[joint];
    }
    std::cout << "]\n";
    return output.mode == ExecutorMode::kActive ? 0 : 1;
}
