#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace right_arm_executor {

// 【核心代码】执行器固定控制右臂 5 个关节，顺序与当前 MPC 保持一致。
constexpr std::size_t kJointCount = 5;
using JointVector = std::array<double, kJointCount>;

enum class OutputSemantics : std::uint32_t {
    // 主机已经算出完整力矩；发送给设备的 kp/kd 必须为零。
    kHostFullTorque = 0,
    // 主机只发送 tau_ff，最终 PD 由设备计算，禁止主机重复加入 PD。
    kDevicePd = 1,
};

enum class ExecutorMode : std::uint32_t {
    kActive = 0,
    kCommandTimedOut = 1,
    kStateTimedOut = 2,
    kInvalidCommand = 3,
    kInvalidState = 4,
};

struct ExecutorConfig {
    OutputSemantics output_semantics{OutputSemantics::kHostFullTorque};
    JointVector kp{};
    JointVector kd{};
    JointVector timeout_damping{};
    JointVector q_ref_min{};
    JointVector q_ref_max{};
    JointVector dq_ref_abs_max{};
    JointVector tau_min{};
    JointVector tau_max{};
    std::int64_t command_timeout_ns{30'000'000};
    std::int64_t state_timeout_ns{10'000'000};
};

struct ExecutorInput {
    // 【核心代码】两个时间戳和 now_ns 必须来自同一个单调时钟。
    // command_timestamp_ns 表示上层 MPC 命令的生成时刻；
    // state_timestamp_ns 表示 q/dq 对应的传感器采样时刻。
    std::int64_t command_timestamp_ns{0};
    std::int64_t state_timestamp_ns{0};
    JointVector q{};
    JointVector dq{};
    JointVector q_ref{};
    JointVector dq_ref{};
    // tau_ff 只包含前馈力矩，不应重复包含下面的 PD 项。
    JointVector tau_ff{};
};

struct ExecutorOutput {
    ExecutorMode mode{ExecutorMode::kInvalidState};
    OutputSemantics output_semantics{OutputSemantics::kHostFullTorque};
    std::int64_t command_age_ns{0};
    std::int64_t state_age_ns{0};
    JointVector effective_q_ref{};
    JointVector effective_dq_ref{};
    JointVector pd_torque{};
    // tau_raw/tau_command 是按当前状态估计的总力矩及其限幅结果。
    // host-full-torque 模式直接发送 tau_command；device-PD 模式只把它
    // 用作诊断，最终总力矩必须由设备端再次限幅。
    JointVector tau_raw{};
    JointVector tau_command{};

    // 【核心代码】以下五组量就是适配器应该发送给设备的字段。
    // host-full-torque: actuator_kp/kd=0，actuator_tau_ff=tau_command。
    // device-PD: actuator_tau_ff 不含 PD，PD 仅由设备根据其最新状态计算。
    JointVector actuator_q_ref{};
    JointVector actuator_dq_ref{};
    JointVector actuator_kp{};
    JointVector actuator_kd{};
    JointVector actuator_tau_ff{};
    bool position_reference_clamped{false};
    bool velocity_reference_clamped{false};
    bool torque_clamped{false};
    bool feedforward_clamped{false};
    bool damping_fallback_active{false};
    bool device_total_torque_limit_required{false};
};

class RightArmExecutor {
public:
    explicit RightArmExecutor(ExecutorConfig config);

    // 【核心代码】无动态分配、无系统调用；调用者传入单调时钟 now_ns。
    [[nodiscard]] ExecutorOutput Step(const ExecutorInput& input, std::int64_t now_ns) const noexcept;

    [[nodiscard]] const ExecutorConfig& config() const noexcept { return config_; }

private:
    ExecutorConfig config_;
};

// 当前项目参数的独立快照，仅供仿真接线和示例使用；真机前必须重新确认。
[[nodiscard]] ExecutorConfig MakeProjectDefaultConfig();
[[nodiscard]] const char* ToString(ExecutorMode mode) noexcept;
[[nodiscard]] const char* ToString(OutputSemantics semantics) noexcept;

}  // namespace right_arm_executor
