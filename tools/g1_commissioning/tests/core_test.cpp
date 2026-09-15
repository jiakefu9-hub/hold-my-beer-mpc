#include "g1_commissioning/core.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace gc = g1_commissioning;

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __FILE__ << ':' << __LINE__                            \
                      << " CHECK failed: " #condition << '\n';                  \
            ++failures;                                                        \
        }                                                                       \
    } while (false)

bool Contains(const gc::ValidationResult& result, const std::string& needle) {
    return std::any_of(
        result.errors.begin(), result.errors.end(),
        [&](const std::string& error) {
            return error.find(needle) != std::string::npos;
        });
}

void TestProfile(const std::string& profile_path) {
    const auto synthetic = gc::LoadSiteProfile(profile_path);
    CHECK(gc::ValidateProfile(synthetic, gc::ValidationUse::kPreview).ok());
    const auto synthetic_real = gc::ValidateProfile(
        synthetic, gc::ValidationUse::kRealOutput);
    CHECK(!synthetic_real.ok());
    CHECK(Contains(synthetic_real, "synthetic_fixture"));

    auto real = synthetic;
    real.synthetic_fixture = false;
    CHECK(gc::ValidateProfile(real, gc::ValidationUse::kRealOutput).ok());

    auto wrong_layout = real;
    wrong_layout.joint_layout = "g1_29_arm7";
    CHECK(Contains(gc::ValidateProfile(
        wrong_layout, gc::ValidationUse::kRealOutput), "g1_23_arm5"));

    auto missing_confirmation = real;
    missing_confirmation.loss_response_plan_reviewed = false;
    CHECK(Contains(gc::ValidateProfile(
        missing_confirmation, gc::ValidationUse::kRealOutput),
        "state-loss"));

    auto missing_release_review = real;
    missing_release_review.normal_release_plan_reviewed = false;
    CHECK(Contains(gc::ValidateProfile(missing_release_review,
        gc::ValidationUse::kRealOutput), "release procedure"));
    auto missing_stop = real;
    missing_stop.emergency_procedure_confirmed = false;
    CHECK(Contains(gc::ValidateProfile(missing_stop,
        gc::ValidationUse::kRealOutput), "emergency"));
    auto draft = real;
    draft.profile_status = "DRAFT";
    CHECK(Contains(gc::ValidateProfile(draft,
        gc::ValidationUse::kRealOutput), "FIELD_REVIEWED"));
    auto old_schema = real;
    old_schema.schema = "g1_arm_static_site_v1";
    CHECK(Contains(gc::ValidateProfile(old_schema,
        gc::ValidationUse::kRealOutput), "schema"));
    const auto summary = gc::ProfileSummaryJson(real,
        gc::ValidateProfile(real, gc::ValidationUse::kRealOutput));
    CHECK(summary.find("\"plan_review_is_not_hardware_validation\":true") != std::string::npos);
    CHECK(summary.find("\"loss_response_plan_reviewed\":true") != std::string::npos);
    CHECK(summary.find("loss_recovery_confirmed") == std::string::npos);

    auto unsafe_offset = real;
    unsafe_offset.offset_rad = 0.2;
    CHECK(Contains(gc::ValidateProfile(
        unsafe_offset, gc::ValidationUse::kPreview), "5 degrees"));

    auto weight_boundary = real;
    weight_boundary.max_weight = 0.5;
    weight_boundary.offset_rad = 5.0 * 3.14159265358979323846 / 180.0;
    weight_boundary.total_timeout_s = 30.0;
    CHECK(gc::ValidateProfile(
        weight_boundary, gc::ValidationUse::kRealOutput).ok());
    auto unsafe_weight = weight_boundary;
    unsafe_weight.max_weight = 0.500001;
    CHECK(Contains(gc::ValidateProfile(
        unsafe_weight, gc::ValidationUse::kPreview), "max_weight"));

    auto invalid_slot_nonzero = real;
    invalid_slot_nonzero.kp[12] = 1.0;
    CHECK(Contains(gc::ValidateProfile(
        invalid_slot_nonzero, gc::ValidationUse::kPreview),
        "invalid slots"));
}

void TestState(
    const std::string& profile_path,
    const std::string& state_path) {
    auto profile = gc::LoadSiteProfile(profile_path);
    const auto loaded = gc::LoadOfflineSnapshot(state_path);
    CHECK(gc::ValidateState(
        loaded.state, profile, loaded.validation_now_monotonic_ns,
        true, false).ok());

    CHECK(Contains(gc::ValidateState(
        loaded.state, profile, loaded.validation_now_monotonic_ns,
        true, true), "synthetic state"));

    auto real = loaded.state;
    real.synthetic_fixture = false;
    CHECK(gc::ValidateState(
        real, profile, loaded.validation_now_monotonic_ns,
        true, true).ok());

    auto crc = real;
    crc.crc_valid = false;
    CHECK(Contains(gc::ValidateState(
        crc, profile, loaded.validation_now_monotonic_ns, true, true), "CRC"));

    auto stale = real;
    stale.captured_monotonic_ns = loaded.validation_now_monotonic_ns - 21000000ULL;
    CHECK(Contains(gc::ValidateState(
        stale, profile, loaded.validation_now_monotonic_ns, true, true), "stale"));

    auto future = real;
    future.captured_monotonic_ns = loaded.validation_now_monotonic_ns + 1U;
    CHECK(Contains(gc::ValidateState(
        future, profile, loaded.validation_now_monotonic_ns, true, true),
        "future"));

    auto wrong_mode = real;
    wrong_mode.mode_machine = 5;
    CHECK(Contains(gc::ValidateState(
        wrong_mode, profile, loaded.validation_now_monotonic_ns, true, true),
        "mode_machine"));

    auto tick = real;
    tick.tick_regression = true;
    CHECK(Contains(gc::ValidateState(
        tick, profile, loaded.validation_now_monotonic_ns, true, true),
        "tick regressed"));

    auto nonfinite = real;
    nonfinite.q[22] = std::numeric_limits<double>::quiet_NaN();
    CHECK(Contains(gc::ValidateState(
        nonfinite, profile, loaded.validation_now_monotonic_ns, true, true),
        "non-finite"));

    auto moving = real;
    moving.dq[22] = profile.startup_max_abs_dq_rad_s + 0.01;
    CHECK(Contains(gc::ValidateState(
        moving, profile, loaded.validation_now_monotonic_ns, true, true),
        "dq limit"));

    auto endpoint = real;
    endpoint.q[22] = 0.99;
    CHECK(Contains(gc::ValidateState(
        endpoint, profile, loaded.validation_now_monotonic_ns, true, true),
        "endpoint"));
}

void TestPlanner(
    const std::string& profile_path,
    const std::string& state_path) {
    const auto profile = gc::LoadSiteProfile(profile_path);
    const auto state = gc::LoadOfflineSnapshot(state_path).state;
    const gc::TrajectoryPlanner planner(profile, state);
    CHECK(std::abs(planner.total_duration_s() - 6.2) < 1e-12);

    const auto first = planner.Sample(0.0);
    CHECK(first.phase == gc::Phase::kRampIn);
    CHECK(first.weight == 0.0);
    CHECK(first.q[profile.selected_slot] == state.q[22]);

    const auto ramp = planner.Sample(1.0);
    CHECK(std::abs(ramp.weight - 0.05) < 1e-12);
    CHECK(ramp.q[profile.selected_slot] == state.q[22]);

    const auto move_a = planner.Sample(2.5);
    const auto move_b = planner.Sample(2.52);
    CHECK(move_a.phase == gc::Phase::kMoveOut);
    CHECK(std::abs(move_b.q[profile.selected_slot] -
                   move_a.q[profile.selected_slot]) <=
          profile.max_velocity_rad_s * 0.0200001);
    for (std::size_t slot = 0; slot < gc::kArmSlotCount; ++slot) {
        if (slot != profile.selected_slot && profile.valid_slots[slot]) {
            CHECK(move_a.q[slot] == state.q[gc::kArmMotorIndices[slot]]);
        }
    }
    CHECK(move_a.q[11] == 0.0 && move_a.kp[11] == 0.0 &&
          move_a.kd[11] == 0.0 && move_a.tau[11] == 0.0);
    CHECK(move_a.q[12] == 0.0 && move_a.kp[12] == 0.0 &&
          move_a.kd[12] == 0.0 && move_a.tau[12] == 0.0);

    const auto terminal = planner.Sample(planner.total_duration_s());
    CHECK(terminal.phase == gc::Phase::kComplete);
    CHECK(terminal.terminal);
    CHECK(terminal.weight == 0.0);
    CHECK(terminal.q[profile.selected_slot] == state.q[22]);

    auto current = state;
    current.q[22] = 0.015;
    const auto sigint = gc::MakeSigintReleaseFrame(
        profile, current, 0.1, 1.0, 77);
    CHECK(sigint.phase == gc::Phase::kSigintRelease);
    CHECK(std::abs(sigint.weight - 0.05) < 1e-12);
    CHECK(sigint.q[profile.selected_slot] == current.q[22]);
    CHECK(sigint.q[11] == 0.0 && sigint.kp[11] == 0.0);

    const auto fault = gc::MakeFaultZeroWeightFrame(profile, 78);
    CHECK(fault.phase == gc::Phase::kFaultZeroWeight);
    CHECK(fault.weight == 0.0 && fault.terminal);
    CHECK(std::all_of(fault.q.begin(), fault.q.end(), [](double x) {
        return x == 0.0;
    }));

    auto tracking_state = state;
    tracking_state.q[22] = move_a.q[profile.selected_slot] + 0.11;
    CHECK(Contains(gc::ValidateRuntimeTracking(
        tracking_state, profile, state, move_a), "tracking"));
    tracking_state = state;
    tracking_state.q[15] = 0.11;
    CHECK(Contains(gc::ValidateRuntimeTracking(
        tracking_state, profile, state, move_a), "drift"));
}

void TestA3(
    const std::string& profile_path,
    const std::string& state_path) {
    const auto profile = gc::LoadSiteProfile(profile_path);
    auto state = gc::LoadOfflineSnapshot(state_path).state;
    for (std::size_t slot = 0; slot <= 10U; ++slot) {
        state.q[gc::kArmMotorIndices[slot]] = 0.3;
    }
    CHECK(gc::IsA3BalanceHold(profile));
    CHECK(gc::ValidateProfile(profile, gc::ValidationUse::kPreview).ok());
    CHECK(Contains(gc::ValidateProfile(
        profile, gc::ValidationUse::kRealOutput), "synthetic_fixture"));

    auto real = profile;
    real.synthetic_fixture = false;
    CHECK(gc::ValidateProfile(real, gc::ValidationUse::kRealOutput).ok());
    auto wrong_fsm = real;
    wrong_fsm.required_fsm = 4;
    CHECK(Contains(gc::ValidateProfile(
        wrong_fsm, gc::ValidationUse::kPreview), "FSM 500"));
    auto nonzero_target = real;
    nonzero_target.target_q[5] = 0.01;
    CHECK(Contains(gc::ValidateProfile(
        nonzero_target, gc::ValidationUse::kPreview), "must be zero"));
    auto excessive_weight = real;
    excessive_weight.max_weight = 1.001;
    CHECK(Contains(gc::ValidateProfile(
        excessive_weight, gc::ValidationUse::kPreview), "max_weight"));

    gc::TrajectoryPlanner planner(profile, state);
    CHECK(std::abs(planner.total_duration_s() - 11.0) < 1e-12);
    const auto start = planner.Sample(0.0);
    CHECK(start.phase == gc::Phase::kRampIn && start.weight == 0.0);
    CHECK(std::abs(start.q[5] - 0.3) < 1e-12);
    const auto halfway = planner.Sample(1.5);
    CHECK(halfway.phase == gc::Phase::kRampIn);
    CHECK(std::abs(halfway.weight - 0.5) < 1e-12);
    CHECK(std::abs(halfway.q[5] - 0.15) < 1e-12);
    const auto hold = planner.Sample(3.0);
    CHECK(hold.phase == gc::Phase::kHold);
    CHECK(std::abs(hold.weight - 1.0) < 1e-12);
    CHECK(std::all_of(hold.q.begin(), hold.q.end(), [](double q) {
        return q == 0.0;
    }));
    const auto release = planner.Sample(9.5);
    CHECK(release.phase == gc::Phase::kRampOut);
    CHECK(std::abs(release.weight - 0.5) < 1e-12);
    const auto done = planner.Sample(11.0);
    CHECK(done.phase == gc::Phase::kComplete && done.terminal);
    CHECK(done.weight == 0.0);

    auto tracking = state;
    tracking.q[22] = 0.51;
    CHECK(Contains(gc::ValidateRuntimeTracking(
        tracking, profile, state, hold), "A3 tracking"));
}

void TestTimeAndTick() {
    CHECK(!gc::TickRegressed(10U, 10U));
    CHECK(!gc::TickRegressed(10U, 11U));
    CHECK(!gc::TickRegressed(0xffffffffU, 0U));
    CHECK(gc::TickRegressed(11U, 10U));
    CHECK(gc::TickRegressed(0U, 0x80000000U));

    CHECK(gc::DeadlineHealthy(1000000U, 1000000U, 1.0));
    CHECK(gc::DeadlineHealthy(1000000U, 1500000U, 1.0));
    CHECK(!gc::DeadlineHealthy(1000000U, 2500000U, 1.0));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: core_test A2_PROFILE A3_PROFILE STATE\n";
        return 2;
    }
    TestProfile(argv[1]);
    TestState(argv[1], argv[3]);
    TestPlanner(argv[1], argv[3]);
    TestA3(argv[2], argv[3]);
    TestTimeAndTick();
    if (failures != 0) {
        std::cerr << failures << " commissioning core checks failed\n";
        return 1;
    }
    std::cout << "commissioning core checks passed\n";
    return 0;
}
