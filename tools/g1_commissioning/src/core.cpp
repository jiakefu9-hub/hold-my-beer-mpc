#include "g1_commissioning/core.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace g1_commissioning {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHardOffsetLimitRad = 5.0 * kPi / 180.0;
constexpr double kHardVelocityLimitRadS = 0.1;
constexpr double kHardWeightLimit = 0.5;
constexpr double kHardWeightRateLimitPerS = 0.2;
constexpr double kHardKpLimit = 80.0;
constexpr double kHardKdLimit = 5.0;
constexpr double kHardTotalTimeoutS = 30.0;
constexpr std::uint16_t kArm5ValidMask = 0x07ffU;  // arms 10 + waist yaw

std::string Trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1U);
}

using KeyValues = std::map<std::string, std::string>;

KeyValues LoadKeyValues(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open file: " + path);
    }
    KeyValues values;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        line = Trim(line);
        if (line.empty()) {
            continue;
        }
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error(
                path + ":" + std::to_string(line_number) +
                ": expected key=value");
        }
        const std::string key = Trim(line.substr(0, equals));
        const std::string value = Trim(line.substr(equals + 1U));
        if (key.empty() || value.empty()) {
            throw std::runtime_error(
                path + ":" + std::to_string(line_number) +
                ": empty key or value");
        }
        if (!values.emplace(key, value).second) {
            throw std::runtime_error("duplicate key in " + path + ": " + key);
        }
    }
    return values;
}

std::string Take(KeyValues& values, const std::string& key) {
    const auto found = values.find(key);
    if (found == values.end()) {
        throw std::runtime_error("missing required key: " + key);
    }
    const std::string value = found->second;
    values.erase(found);
    return value;
}

bool ParseBool(const std::string& value, const std::string& key) {
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw std::runtime_error(key + " must be true or false");
}

double ParseDouble(const std::string& value, const std::string& key) {
    std::size_t consumed = 0;
    const double result = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(result)) {
        throw std::runtime_error(key + " must be a finite number");
    }
    return result;
}

std::uint64_t ParseUint64(const std::string& value, const std::string& key) {
    std::size_t consumed = 0;
    const auto result = std::stoull(value, &consumed, 0);
    if (consumed != value.size()) {
        throw std::runtime_error(key + " must be an unsigned integer");
    }
    return result;
}

std::size_t ParseSize(const std::string& value, const std::string& key) {
    const auto parsed = ParseUint64(value, key);
    if (parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(key + " is too large");
    }
    return static_cast<std::size_t>(parsed);
}

std::uint8_t ParseByte(const std::string& value, const std::string& key) {
    const auto parsed = ParseUint64(value, key);
    if (parsed > std::numeric_limits<std::uint8_t>::max()) {
        throw std::runtime_error(key + " must fit uint8");
    }
    return static_cast<std::uint8_t>(parsed);
}

template <std::size_t Size>
std::array<double, Size> ParseDoubleArray(
    const std::string& value, const std::string& key) {
    std::array<double, Size> output{};
    std::stringstream stream(value);
    std::string item;
    std::size_t index = 0;
    while (std::getline(stream, item, ',')) {
        if (index >= Size) {
            throw std::runtime_error(key + " has too many values");
        }
        output[index++] = ParseDouble(Trim(item), key);
    }
    if (index != Size) {
        throw std::runtime_error(
            key + " must have exactly " + std::to_string(Size) + " values");
    }
    return output;
}

std::array<bool, kArmSlotCount> ParseValidSlots(
    const std::string& value) {
    std::array<bool, kArmSlotCount> valid{};
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto slot = ParseSize(Trim(item), "valid_slots");
        if (slot >= kArmSlotCount || valid[slot]) {
            throw std::runtime_error("valid_slots contains invalid/duplicate slot");
        }
        valid[slot] = true;
    }
    return valid;
}

template <std::size_t Size>
bool Finite(const std::array<double, Size>& values) {
    return std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value);
    });
}

void AddIf(
    ValidationResult& result, bool condition, const std::string& message) {
    if (condition) {
        result.errors.push_back(message);
    }
}

template <std::size_t Size>
void WriteArray(std::ostream& output, const std::array<double, Size>& values) {
    output << '[';
    for (std::size_t index = 0; index < Size; ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

std::uint16_t SlotMask(const SiteProfile& profile) {
    std::uint16_t mask = 0U;
    for (std::size_t slot = 0; slot < kArmSlotCount; ++slot) {
        if (profile.valid_slots[slot]) {
            mask |= static_cast<std::uint16_t>(1U << slot);
        }
    }
    return mask;
}

bool Placeholder(const std::string& value) {
    return value.empty() || value == "UNSET" || value == "TODO";
}

}  // namespace

SiteProfile LoadSiteProfile(const std::string& path) {
    auto values = LoadKeyValues(path);
    SiteProfile profile;
    profile.schema = Take(values, "schema");
    if (profile.schema != "g1_arm_static_site_v2") {
        throw std::runtime_error(
            "unsupported profile schema: use g1_arm_static_site_v2 and review "
            "loss_response_plan_reviewed/normal_release_plan_reviewed; "
            "legacy confirmation flags are not migrated automatically");
    }
    profile.synthetic_fixture = ParseBool(
        Take(values, "synthetic_fixture"), "synthetic_fixture");
    profile.robot_id = Take(values, "robot_id");
    profile.model_name = Take(values, "model_name");
    profile.joint_layout = Take(values, "joint_layout");
    profile.confirmed_by = Take(values, "confirmed_by");
    profile.loco_service = Take(values, "loco_service");
    profile.profile_status = Take(values, "profile_status");

    profile.hardware_identity_confirmed = ParseBool(
        Take(values, "hardware_identity_confirmed"),
        "hardware_identity_confirmed");
    profile.mapping_confirmed = ParseBool(
        Take(values, "mapping_confirmed"), "mapping_confirmed");
    profile.weight_scope_confirmed = ParseBool(
        Take(values, "weight_scope_confirmed"), "weight_scope_confirmed");
    profile.invalid_slots_confirmed = ParseBool(
        Take(values, "invalid_slots_confirmed"), "invalid_slots_confirmed");
    profile.locked_stand_mode_confirmed = ParseBool(
        Take(values, "locked_stand_mode_confirmed"),
        "locked_stand_mode_confirmed");
    profile.loss_response_plan_reviewed = ParseBool(
        Take(values, "loss_response_plan_reviewed"), "loss_response_plan_reviewed");
    profile.normal_release_plan_reviewed = ParseBool(
        Take(values, "normal_release_plan_reviewed"), "normal_release_plan_reviewed");
    profile.emergency_procedure_confirmed = ParseBool(
        Take(values, "emergency_procedure_confirmed"),
        "emergency_procedure_confirmed");
    profile.no_competing_user_publishers_confirmed = ParseBool(
        Take(values, "no_competing_user_publishers_confirmed"),
        "no_competing_user_publishers_confirmed");
    profile.safety_parameters_confirmed = ParseBool(
        Take(values, "safety_parameters_confirmed"),
        "safety_parameters_confirmed");

    profile.expected_mode_pr = ParseByte(
        Take(values, "expected_mode_pr"), "expected_mode_pr");
    profile.expected_mode_machine = ParseByte(
        Take(values, "expected_mode_machine"), "expected_mode_machine");
    profile.valid_slots = ParseValidSlots(Take(values, "valid_slots"));
    profile.invalid_slot_policy = Take(values, "invalid_slot_policy");
    profile.selected_slot = ParseSize(
        Take(values, "selected_slot"), "selected_slot");

    profile.offset_rad = ParseDouble(Take(values, "offset_rad"), "offset_rad");
    profile.max_velocity_rad_s = ParseDouble(
        Take(values, "max_velocity_rad_s"), "max_velocity_rad_s");
    profile.kp = ParseDoubleArray<kArmSlotCount>(Take(values, "kp"), "kp");
    profile.kd = ParseDoubleArray<kArmSlotCount>(Take(values, "kd"), "kd");
    profile.q_min = ParseDoubleArray<kArmSlotCount>(
        Take(values, "q_min"), "q_min");
    profile.q_max = ParseDoubleArray<kArmSlotCount>(
        Take(values, "q_max"), "q_max");
    profile.max_weight = ParseDouble(
        Take(values, "max_weight"), "max_weight");
    profile.weight_rate_per_s = ParseDouble(
        Take(values, "weight_rate_per_s"), "weight_rate_per_s");
    profile.hold_s = ParseDouble(Take(values, "hold_s"), "hold_s");
    profile.control_period_ms = ParseDouble(
        Take(values, "control_period_ms"), "control_period_ms");
    profile.state_timeout_ms = ParseDouble(
        Take(values, "state_timeout_ms"), "state_timeout_ms");
    profile.startup_wait_s = ParseDouble(
        Take(values, "startup_wait_s"), "startup_wait_s");
    profile.startup_valid_samples = ParseSize(
        Take(values, "startup_valid_samples"), "startup_valid_samples");
    profile.startup_max_abs_dq_rad_s = ParseDouble(
        Take(values, "startup_max_abs_dq_rad_s"),
        "startup_max_abs_dq_rad_s");
    profile.runtime_max_abs_dq_rad_s = ParseDouble(
        Take(values, "runtime_max_abs_dq_rad_s"),
        "runtime_max_abs_dq_rad_s");
    profile.max_selected_tracking_error_rad = ParseDouble(
        Take(values, "max_selected_tracking_error_rad"),
        "max_selected_tracking_error_rad");
    profile.max_unselected_drift_rad = ParseDouble(
        Take(values, "max_unselected_drift_rad"),
        "max_unselected_drift_rad");
    profile.deadline_tolerance_ms = ParseDouble(
        Take(values, "deadline_tolerance_ms"), "deadline_tolerance_ms");
    profile.total_timeout_s = ParseDouble(
        Take(values, "total_timeout_s"), "total_timeout_s");

    if (!values.empty()) {
        throw std::runtime_error("unknown profile key: " + values.begin()->first);
    }
    return profile;
}

OfflineSnapshot LoadOfflineSnapshot(const std::string& path) {
    auto values = LoadKeyValues(path);
    if (Take(values, "schema") != "g1_arm_state_snapshot_v1") {
        throw std::runtime_error("unsupported state snapshot schema");
    }
    OfflineSnapshot snapshot;
    snapshot.state.synthetic_fixture = ParseBool(
        Take(values, "synthetic_fixture"), "synthetic_fixture");
    snapshot.state.capture_sequence = ParseUint64(
        Take(values, "capture_sequence"), "capture_sequence");
    snapshot.state.captured_monotonic_ns = ParseUint64(
        Take(values, "captured_monotonic_ns"), "captured_monotonic_ns");
    const auto version = ParseDoubleArray<2>(Take(values, "version"), "version");
    for (std::size_t index = 0; index < version.size(); ++index) {
        if (version[index] < 0.0 ||
            version[index] > std::numeric_limits<std::uint32_t>::max() ||
            std::floor(version[index]) != version[index]) {
            throw std::runtime_error("version must contain two uint32 values");
        }
        snapshot.state.version[index] =
            static_cast<std::uint32_t>(version[index]);
    }
    snapshot.validation_now_monotonic_ns = ParseUint64(
        Take(values, "validation_now_monotonic_ns"),
        "validation_now_monotonic_ns");
    snapshot.state.crc_valid = ParseBool(
        Take(values, "crc_valid"), "crc_valid");
    snapshot.state.tick_regression = ParseBool(
        Take(values, "tick_regression"), "tick_regression");
    const auto tick = ParseUint64(Take(values, "tick"), "tick");
    if (tick > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("tick must fit uint32");
    }
    snapshot.state.tick = static_cast<std::uint32_t>(tick);
    snapshot.state.mode_pr = ParseByte(Take(values, "mode_pr"), "mode_pr");
    snapshot.state.mode_machine = ParseByte(
        Take(values, "mode_machine"), "mode_machine");
    snapshot.state.q = ParseDoubleArray<kMotorCount>(Take(values, "q"), "q");
    snapshot.state.dq = ParseDoubleArray<kMotorCount>(Take(values, "dq"), "dq");
    if (!values.empty()) {
        throw std::runtime_error("unknown state key: " + values.begin()->first);
    }
    return snapshot;
}

ValidationResult ValidateProfile(
    const SiteProfile& profile, ValidationUse use) {
    ValidationResult result;
    AddIf(result, profile.schema != "g1_arm_static_site_v2",
          "unsupported profile schema");
    AddIf(result, profile.joint_layout != "g1_23_arm5",
          "only the explicitly confirmed g1_23_arm5 layout is supported");
    AddIf(result, profile.loco_service != "sport",
          "this pinned SDK build only supports the sport Loco service");
    AddIf(result, SlotMask(profile) != kArm5ValidMask,
          "g1_23_arm5 valid_slots must be exactly 0..10");
    AddIf(result, profile.invalid_slot_policy != "zero",
          "invalid_slot_policy must be zero");
    AddIf(result, profile.selected_slot < 5U || profile.selected_slot > 9U,
          "first commissioning motion must select one right Arm5 slot (5..9)");
    AddIf(result, !(std::abs(profile.offset_rad) > 0.0 &&
                    std::abs(profile.offset_rad) <= kHardOffsetLimitRad),
          "offset_rad must be nonzero and no more than 5 degrees");
    AddIf(result, !(profile.max_velocity_rad_s > 0.0 &&
                    profile.max_velocity_rad_s <= kHardVelocityLimitRadS),
          "max_velocity_rad_s must be in (0, 0.1]");
    AddIf(result, !(profile.max_weight > 0.0 &&
                    profile.max_weight <= kHardWeightLimit),
          "max_weight must be in (0, 0.5]");
    AddIf(result, !(profile.weight_rate_per_s > 0.0 &&
                    profile.weight_rate_per_s <= kHardWeightRateLimitPerS),
          "weight_rate_per_s must be in (0, 0.2]");
    AddIf(result, !(profile.hold_s >= 0.0 && profile.hold_s <= 2.0),
          "hold_s must be in [0, 2]");
    AddIf(result, !(profile.control_period_ms >= 4.0 &&
                    profile.control_period_ms <= 20.0),
          "control_period_ms must be in [4, 20]");
    AddIf(result, !(profile.state_timeout_ms > 0.0 &&
                    profile.state_timeout_ms <= 20.0),
          "state_timeout_ms must be in (0, 20]");
    AddIf(result, !(profile.startup_wait_s > 0.0 &&
                    profile.startup_wait_s <= 10.0),
          "startup_wait_s must be in (0, 10]");
    AddIf(result, profile.startup_valid_samples < 10U ||
                      profile.startup_valid_samples > 500U,
          "startup_valid_samples must be in [10, 500]");
    AddIf(result, !(profile.startup_max_abs_dq_rad_s > 0.0 &&
                    profile.startup_max_abs_dq_rad_s <= 0.2),
          "startup_max_abs_dq_rad_s must be in (0, 0.2]");
    AddIf(result, !(profile.runtime_max_abs_dq_rad_s > 0.0 &&
                    profile.runtime_max_abs_dq_rad_s <= 0.5),
          "runtime_max_abs_dq_rad_s must be in (0, 0.5]");
    AddIf(result, !(profile.max_selected_tracking_error_rad > 0.0 &&
                    profile.max_selected_tracking_error_rad <= 0.2),
          "max_selected_tracking_error_rad must be in (0, 0.2]");
    AddIf(result, !(profile.max_unselected_drift_rad > 0.0 &&
                    profile.max_unselected_drift_rad <= 0.2),
          "max_unselected_drift_rad must be in (0, 0.2]");
    AddIf(result, !(profile.deadline_tolerance_ms >= 0.0 &&
                    profile.deadline_tolerance_ms <=
                        profile.control_period_ms),
          "deadline_tolerance_ms must be in [0, control_period_ms]");
    AddIf(result, !(profile.total_timeout_s > 0.0 &&
                    profile.total_timeout_s <= kHardTotalTimeoutS),
          "total_timeout_s must be in (0, 30]");
    AddIf(result, !Finite(profile.kp) || !Finite(profile.kd) ||
                      !Finite(profile.q_min) || !Finite(profile.q_max),
          "profile arrays must be finite");

    for (std::size_t slot = 0; slot < kArmSlotCount; ++slot) {
        const std::string prefix = std::string("slot ") +
                                   std::to_string(slot) + " (" +
                                   kArmSlotNames[slot] + "): ";
        if (profile.valid_slots[slot]) {
            AddIf(result, !(profile.kp[slot] > 0.0 &&
                            profile.kp[slot] <= kHardKpLimit),
                  prefix + "kp must be in (0, 80]");
            AddIf(result, !(profile.kd[slot] > 0.0 &&
                            profile.kd[slot] <= kHardKdLimit),
                  prefix + "kd must be in (0, 5]");
            AddIf(result, !(profile.q_min[slot] < profile.q_max[slot] &&
                            profile.q_min[slot] >= -2.0 * kPi &&
                            profile.q_max[slot] <= 2.0 * kPi),
                  prefix + "q limits are invalid");
        } else {
            AddIf(result, profile.kp[slot] != 0.0 || profile.kd[slot] != 0.0 ||
                              profile.q_min[slot] != 0.0 ||
                              profile.q_max[slot] != 0.0,
                  prefix + "invalid slots must have zero kp/kd/q limits");
        }
    }

    const double ramp = profile.weight_rate_per_s > 0.0
                            ? profile.max_weight / profile.weight_rate_per_s
                            : std::numeric_limits<double>::infinity();
    const double move = profile.max_velocity_rad_s > 0.0
                            ? std::abs(profile.offset_rad) /
                                  profile.max_velocity_rad_s
                            : std::numeric_limits<double>::infinity();
    const double planned = 2.0 * ramp + 2.0 * move + profile.hold_s;
    AddIf(result, !std::isfinite(planned) ||
                      profile.total_timeout_s < planned + 0.5,
          "total_timeout_s must exceed the planned trajectory by 0.5 s");

    if (use == ValidationUse::kRealOutput) {
        AddIf(result, profile.synthetic_fixture,
              "synthetic_fixture profiles are forbidden for real output");
        AddIf(result, Placeholder(profile.robot_id),
              "robot_id must be filled from the target robot");
        AddIf(result, Placeholder(profile.model_name),
              "model_name must be filled from target documentation");
        AddIf(result, Placeholder(profile.confirmed_by),
              "confirmed_by must identify the field approver");
        AddIf(result, profile.profile_status != "FIELD_REVIEWED",
              "profile_status must be FIELD_REVIEWED");
        AddIf(result, !profile.hardware_identity_confirmed,
              "hardware identity is not confirmed");
        AddIf(result, !profile.mapping_confirmed,
              "Arm5 motor mapping is not confirmed");
        AddIf(result, !profile.weight_scope_confirmed,
              "arm weight scope is not confirmed");
        AddIf(result, !profile.invalid_slots_confirmed,
              "invalid slot policy is not confirmed");
        AddIf(result, !profile.locked_stand_mode_confirmed,
              "locked-stand raw mode contract is not confirmed");
        AddIf(result, !profile.loss_response_plan_reviewed,
              "state-loss operator response/no-auto-resume plan is not reviewed");
        AddIf(result, !profile.normal_release_plan_reviewed,
              "normal arm-weight release procedure is not reviewed (prior hardware success is not required)");
        AddIf(result, !profile.emergency_procedure_confirmed,
              "field emergency procedure is not confirmed");
        AddIf(result, !profile.no_competing_user_publishers_confirmed,
              "absence of competing user publishers is not confirmed");
        AddIf(result, !profile.safety_parameters_confirmed,
              "numeric safety parameters are not confirmed");
    }
    return result;
}

ValidationResult ValidateState(
    const StateSample& state,
    const SiteProfile& profile,
    std::uint64_t now_monotonic_ns,
    bool startup_check,
    bool require_real_state) {
    ValidationResult result;
    AddIf(result, require_real_state && state.synthetic_fixture,
          "synthetic state is forbidden for real output");
    AddIf(result, !state.crc_valid, "LowState CRC is invalid");
    AddIf(result, state.tick_regression, "LowState tick regressed");
    AddIf(result, state.capture_sequence == 0U,
          "state capture sequence is zero");
    AddIf(result, state.captured_monotonic_ns == 0U,
          "state timestamp is zero");
    AddIf(result, state.captured_monotonic_ns > now_monotonic_ns,
          "state timestamp is in the future");
    if (state.captured_monotonic_ns <= now_monotonic_ns) {
        const double age_ms = static_cast<double>(
            now_monotonic_ns - state.captured_monotonic_ns) / 1.0e6;
        AddIf(result, age_ms > profile.state_timeout_ms,
              "state is stale");
    }
    AddIf(result, state.mode_pr != profile.expected_mode_pr,
          "mode_pr does not match the confirmed raw value");
    AddIf(result, state.mode_machine != profile.expected_mode_machine,
          "mode_machine does not match the confirmed raw value");
    AddIf(result, !Finite(state.q) || !Finite(state.dq),
          "state q/dq contains non-finite values");

    for (std::size_t slot = 0; slot < kArmSlotCount; ++slot) {
        if (!profile.valid_slots[slot]) {
            continue;
        }
        const std::size_t motor = kArmMotorIndices[slot];
        AddIf(result,
              state.q[motor] < profile.q_min[slot] ||
                  state.q[motor] > profile.q_max[slot],
              std::string(kArmSlotNames[slot]) + " is outside q limits");
        const double dq_limit = startup_check
                                    ? profile.startup_max_abs_dq_rad_s
                                    : profile.runtime_max_abs_dq_rad_s;
        AddIf(result, std::abs(state.dq[motor]) > dq_limit,
              std::string(kArmSlotNames[slot]) + " exceeds dq limit");
    }
    if (startup_check && profile.selected_slot < kArmSlotCount &&
        profile.valid_slots[profile.selected_slot]) {
        const std::size_t motor = kArmMotorIndices[profile.selected_slot];
        const double endpoint = state.q[motor] + profile.offset_rad;
        AddIf(result,
              endpoint < profile.q_min[profile.selected_slot] ||
                  endpoint > profile.q_max[profile.selected_slot],
              "requested selected-joint endpoint exceeds q limits");
    }
    return result;
}

ValidationResult ValidateRuntimeTracking(
    const StateSample& state,
    const SiteProfile& profile,
    const StateSample& initial_state,
    const CommandFrame& command) {
    ValidationResult result;
    for (std::size_t slot = 0; slot < kArmSlotCount; ++slot) {
        if (!profile.valid_slots[slot]) {
            continue;
        }
        const std::size_t motor = kArmMotorIndices[slot];
        if (slot == profile.selected_slot) {
            AddIf(result,
                  std::abs(state.q[motor] - command.q[slot]) >
                      profile.max_selected_tracking_error_rad,
                  "selected joint exceeded tracking error limit");
        } else {
            AddIf(result,
                  std::abs(state.q[motor] - initial_state.q[motor]) >
                      profile.max_unselected_drift_rad,
                  std::string(kArmSlotNames[slot]) +
                      " exceeded unselected drift limit");
        }
    }
    return result;
}

TrajectoryPlanner::TrajectoryPlanner(
    SiteProfile profile, StateSample initial_state)
    : profile_(std::move(profile)), initial_state_(std::move(initial_state)) {
    ramp_duration_s_ = profile_.max_weight / profile_.weight_rate_per_s;
    move_duration_s_ =
        std::abs(profile_.offset_rad) / profile_.max_velocity_rad_s;
}

double TrajectoryPlanner::total_duration_s() const noexcept {
    return 2.0 * ramp_duration_s_ + 2.0 * move_duration_s_ + profile_.hold_s;
}

CommandFrame TrajectoryPlanner::Sample(double elapsed_s) const {
    CommandFrame frame;
    frame.elapsed_s = std::max(0.0, elapsed_s);
    frame.sequence = static_cast<std::uint64_t>(
        std::floor(frame.elapsed_s * 1000.0)) + 1U;
    for (std::size_t slot = 0; slot < kArmSlotCount; ++slot) {
        if (!profile_.valid_slots[slot]) {
            continue;
        }
        frame.q[slot] = initial_state_.q[kArmMotorIndices[slot]];
        frame.kp[slot] = profile_.kp[slot];
        frame.kd[slot] = profile_.kd[slot];
    }

    const double ramp_in_end = ramp_duration_s_;
    const double move_out_end = ramp_in_end + move_duration_s_;
    const double hold_end = move_out_end + profile_.hold_s;
    const double move_back_end = hold_end + move_duration_s_;
    const double ramp_out_end = move_back_end + ramp_duration_s_;
    double selected_offset = 0.0;

    if (frame.elapsed_s < ramp_in_end) {
        frame.phase = Phase::kRampIn;
        frame.weight = profile_.weight_rate_per_s * frame.elapsed_s;
    } else if (frame.elapsed_s < move_out_end) {
        frame.phase = Phase::kMoveOut;
        frame.weight = profile_.max_weight;
        const double ratio = (frame.elapsed_s - ramp_in_end) / move_duration_s_;
        selected_offset = profile_.offset_rad * std::clamp(ratio, 0.0, 1.0);
    } else if (frame.elapsed_s < hold_end) {
        frame.phase = Phase::kHold;
        frame.weight = profile_.max_weight;
        selected_offset = profile_.offset_rad;
    } else if (frame.elapsed_s < move_back_end) {
        frame.phase = Phase::kMoveBack;
        frame.weight = profile_.max_weight;
        const double ratio = (frame.elapsed_s - hold_end) / move_duration_s_;
        selected_offset = profile_.offset_rad *
                          (1.0 - std::clamp(ratio, 0.0, 1.0));
    } else if (frame.elapsed_s < ramp_out_end) {
        frame.phase = Phase::kRampOut;
        const double release_elapsed = frame.elapsed_s - move_back_end;
        frame.weight = std::max(
            0.0, profile_.max_weight -
                     profile_.weight_rate_per_s * release_elapsed);
    } else {
        frame.phase = Phase::kComplete;
        frame.weight = 0.0;
        frame.terminal = true;
    }
    frame.weight = std::clamp(frame.weight, 0.0, profile_.max_weight);
    frame.q[profile_.selected_slot] += selected_offset;
    return frame;
}

CommandFrame MakeSigintReleaseFrame(
    const SiteProfile& profile,
    const StateSample& current_state,
    double previous_weight,
    double release_elapsed_s,
    std::uint64_t sequence) {
    CommandFrame frame;
    frame.sequence = sequence;
    frame.elapsed_s = release_elapsed_s;
    frame.phase = Phase::kSigintRelease;
    frame.weight = std::max(
        0.0, previous_weight -
                 profile.weight_rate_per_s * std::max(0.0, release_elapsed_s));
    frame.terminal = frame.weight <= 0.0;
    for (std::size_t slot = 0; slot < kArmSlotCount; ++slot) {
        if (!profile.valid_slots[slot]) {
            continue;
        }
        frame.q[slot] = current_state.q[kArmMotorIndices[slot]];
        frame.kp[slot] = profile.kp[slot];
        frame.kd[slot] = profile.kd[slot];
    }
    return frame;
}

CommandFrame MakeFaultZeroWeightFrame(
    const SiteProfile&, std::uint64_t sequence) {
    CommandFrame frame;
    frame.sequence = sequence;
    frame.phase = Phase::kFaultZeroWeight;
    frame.weight = 0.0;
    frame.terminal = true;
    return frame;
}

bool TickRegressed(
    std::uint32_t previous, std::uint32_t current) noexcept {
    const std::uint32_t delta = current - previous;
    return delta >= (std::uint32_t{1} << 31U);
}

bool DeadlineHealthy(
    std::uint64_t scheduled_ns,
    std::uint64_t actual_ns,
    double tolerance_ms) noexcept {
    if (actual_ns <= scheduled_ns) {
        return true;
    }
    const double lateness_ms = static_cast<double>(actual_ns - scheduled_ns) / 1.0e6;
    return lateness_ms <= tolerance_ms;
}

const char* PhaseName(Phase phase) noexcept {
    switch (phase) {
        case Phase::kRampIn: return "ramp_in";
        case Phase::kMoveOut: return "move_out";
        case Phase::kHold: return "hold";
        case Phase::kMoveBack: return "move_back";
        case Phase::kRampOut: return "ramp_out";
        case Phase::kComplete: return "complete";
        case Phase::kSigintRelease: return "sigint_release";
        case Phase::kFaultZeroWeight: return "fault_zero_weight";
    }
    return "unknown";
}

std::string JsonEscape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(character)
                           << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

std::string ProfileSummaryJson(
    const SiteProfile& profile,
    const ValidationResult& output_validation) {
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"schema\":\"g1_arm_static_profile_summary_v1\""
           << ",\"robot_id\":\"" << JsonEscape(profile.robot_id) << "\""
           << ",\"model_name\":\"" << JsonEscape(profile.model_name) << "\""
           << ",\"joint_layout\":\"" << JsonEscape(profile.joint_layout) << "\""
           << ",\"selected_slot\":" << profile.selected_slot
           << ",\"selected_joint\":\""
           << (profile.selected_slot < kArmSlotCount
                   ? kArmSlotNames[profile.selected_slot]
                   : "invalid") << "\""
           << ",\"offset_rad\":" << profile.offset_rad
           << ",\"max_velocity_rad_s\":" << profile.max_velocity_rad_s
           << ",\"max_weight\":" << profile.max_weight
           << ",\"profile_schema\":\"" << JsonEscape(profile.schema) << "\""
           << ",\"loss_response_plan_reviewed\":"
           << (profile.loss_response_plan_reviewed ? "true" : "false")
           << ",\"normal_release_plan_reviewed\":"
           << (profile.normal_release_plan_reviewed ? "true" : "false")
           << ",\"plan_review_is_not_hardware_validation\":true"
           << ",\"real_output_profile_gate_passed\":"
           << (output_validation.ok() ? "true" : "false")
           << ",\"readiness_errors\":[";
    for (std::size_t index = 0; index < output_validation.errors.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << '"' << JsonEscape(output_validation.errors[index]) << '"';
    }
    output << "]}";
    return output.str();
}

std::string FrameJson(
    const CommandFrame& frame,
    const StateSample* measured,
    const std::string& event,
    const std::string& reason) {
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"schema\":\"g1_arm_static_command_record_v1\""
           << ",\"event\":\"" << JsonEscape(event) << "\""
           << ",\"reason\":\"" << JsonEscape(reason) << "\""
           << ",\"sequence\":" << frame.sequence
           << ",\"elapsed_s\":" << frame.elapsed_s
           << ",\"phase\":\"" << PhaseName(frame.phase) << "\""
           << ",\"weight\":" << frame.weight
           << ",\"terminal\":" << (frame.terminal ? "true" : "false")
           << ",\"q_target\":";
    WriteArray(output, frame.q);
    output << ",\"dq_target\":";
    WriteArray(output, frame.dq);
    output << ",\"kp\":";
    WriteArray(output, frame.kp);
    output << ",\"kd\":";
    WriteArray(output, frame.kd);
    output << ",\"tau_ff\":";
    WriteArray(output, frame.tau);
    if (measured != nullptr) {
        std::array<double, kArmSlotCount> measured_q{};
        std::array<double, kArmSlotCount> measured_dq{};
        for (std::size_t slot = 0; slot < kArmSlotCount; ++slot) {
            measured_q[slot] = measured->q[kArmMotorIndices[slot]];
            measured_dq[slot] = measured->dq[kArmMotorIndices[slot]];
        }
        output << ",\"capture_sequence\":" << measured->capture_sequence
               << ",\"captured_monotonic_ns\":"
               << measured->captured_monotonic_ns
               << ",\"robot_tick\":" << measured->tick
               << ",\"mode_pr\":" << static_cast<unsigned>(measured->mode_pr)
               << ",\"mode_machine\":"
               << static_cast<unsigned>(measured->mode_machine)
               << ",\"q_measured\":";
        WriteArray(output, measured_q);
        output << ",\"dq_measured\":";
        WriteArray(output, measured_dq);
    }
    output << '}';
    return output.str();
}

}  // namespace g1_commissioning
