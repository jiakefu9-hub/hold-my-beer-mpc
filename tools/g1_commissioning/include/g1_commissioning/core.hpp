#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace g1_commissioning {

constexpr std::size_t kMotorCount = 35;
constexpr std::size_t kArmSlotCount = 13;
constexpr std::size_t kWeightMotorIndex = 29;

// Unitree G1 Arm5 order: left arm 5, right arm 5, waist 3.
constexpr std::array<std::size_t, kArmSlotCount> kArmMotorIndices{
    15, 16, 17, 18, 19,
    22, 23, 24, 25, 26,
    12, 13, 14,
};

constexpr std::array<const char*, kArmSlotCount> kArmSlotNames{
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow_pitch", "left_elbow_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow_pitch", "right_wrist_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
};

struct SiteProfile {
    std::string schema;
    bool synthetic_fixture{false};
    std::string robot_id;
    std::string model_name;
    std::string joint_layout;
    std::string confirmed_by;
    std::string loco_service;
    std::string profile_status;

    bool hardware_identity_confirmed{false};
    bool mapping_confirmed{false};
    bool weight_scope_confirmed{false};
    bool invalid_slots_confirmed{false};
    bool locked_stand_mode_confirmed{false};
    // Pre-test procedure reviews, NOT claims of previous hardware test success.
    // Unknown firmware loss behavior must be treated as potentially persistent
    // output; review an independent operator response and prohibit auto-resume.
    bool loss_response_plan_reviewed{false};
    bool normal_release_plan_reviewed{false};
    bool emergency_procedure_confirmed{false};
    bool no_competing_user_publishers_confirmed{false};
    bool safety_parameters_confirmed{false};

    std::uint8_t expected_mode_pr{0};
    std::uint8_t expected_mode_machine{0};
    std::array<bool, kArmSlotCount> valid_slots{};
    std::string invalid_slot_policy;
    std::size_t selected_slot{0};

    double offset_rad{0.0};
    double max_velocity_rad_s{0.0};
    std::array<double, kArmSlotCount> kp{};
    std::array<double, kArmSlotCount> kd{};
    std::array<double, kArmSlotCount> q_min{};
    std::array<double, kArmSlotCount> q_max{};
    double max_weight{0.0};
    double weight_rate_per_s{0.0};
    double hold_s{0.0};
    double control_period_ms{0.0};
    double state_timeout_ms{0.0};
    double startup_wait_s{0.0};
    std::size_t startup_valid_samples{0};
    double startup_max_abs_dq_rad_s{0.0};
    double runtime_max_abs_dq_rad_s{0.0};
    double max_selected_tracking_error_rad{0.0};
    double max_unselected_drift_rad{0.0};
    double deadline_tolerance_ms{0.0};
    double total_timeout_s{0.0};
};

struct StateSample {
    bool synthetic_fixture{false};
    bool crc_valid{false};
    bool tick_regression{false};
    std::uint64_t capture_sequence{0};
    std::uint64_t captured_monotonic_ns{0};
    std::array<std::uint32_t, 2> version{};
    std::uint32_t tick{0};
    std::uint8_t mode_pr{0};
    std::uint8_t mode_machine{0};
    std::array<double, kMotorCount> q{};
    std::array<double, kMotorCount> dq{};
};

struct OfflineSnapshot {
    StateSample state;
    std::uint64_t validation_now_monotonic_ns{0};
};

enum class ValidationUse {
    kPreview,
    kRealOutput,
};

struct ValidationResult {
    std::vector<std::string> errors;
    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

enum class Phase {
    kRampIn,
    kMoveOut,
    kHold,
    kMoveBack,
    kRampOut,
    kComplete,
    kSigintRelease,
    kFaultZeroWeight,
};

struct CommandFrame {
    std::uint64_t sequence{0};
    double elapsed_s{0.0};
    Phase phase{Phase::kRampIn};
    double weight{0.0};
    std::array<double, kArmSlotCount> q{};
    std::array<double, kArmSlotCount> dq{};
    std::array<double, kArmSlotCount> kp{};
    std::array<double, kArmSlotCount> kd{};
    std::array<double, kArmSlotCount> tau{};
    bool terminal{false};
};

SiteProfile LoadSiteProfile(const std::string& path);
OfflineSnapshot LoadOfflineSnapshot(const std::string& path);

ValidationResult ValidateProfile(
    const SiteProfile& profile, ValidationUse use);
ValidationResult ValidateState(
    const StateSample& state,
    const SiteProfile& profile,
    std::uint64_t now_monotonic_ns,
    bool startup_check,
    bool require_real_state);
ValidationResult ValidateRuntimeTracking(
    const StateSample& state,
    const SiteProfile& profile,
    const StateSample& initial_state,
    const CommandFrame& command);

class TrajectoryPlanner {
public:
    TrajectoryPlanner(SiteProfile profile, StateSample initial_state);

    [[nodiscard]] CommandFrame Sample(double elapsed_s) const;
    [[nodiscard]] double total_duration_s() const noexcept;

private:
    SiteProfile profile_;
    StateSample initial_state_;
    double ramp_duration_s_{0.0};
    double move_duration_s_{0.0};
};

CommandFrame MakeSigintReleaseFrame(
    const SiteProfile& profile,
    const StateSample& current_state,
    double previous_weight,
    double release_elapsed_s,
    std::uint64_t sequence);
CommandFrame MakeFaultZeroWeightFrame(
    const SiteProfile& profile,
    std::uint64_t sequence);

[[nodiscard]] bool TickRegressed(
    std::uint32_t previous, std::uint32_t current) noexcept;
[[nodiscard]] bool DeadlineHealthy(
    std::uint64_t scheduled_ns,
    std::uint64_t actual_ns,
    double tolerance_ms) noexcept;
[[nodiscard]] const char* PhaseName(Phase phase) noexcept;
[[nodiscard]] std::string JsonEscape(const std::string& value);
[[nodiscard]] std::string ProfileSummaryJson(
    const SiteProfile& profile,
    const ValidationResult& output_validation);
[[nodiscard]] std::string FrameJson(
    const CommandFrame& frame,
    const StateSample* measured,
    const std::string& event,
    const std::string& reason);

}  // namespace g1_commissioning
