#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "unitree_arm_adapter/hardware_command_supervisor.hpp"
#include "unitree_arm_adapter/hil_recording_sink.hpp"
#include "unitree_arm_adapter/hil_supervisor_dispatcher.hpp"
#include "unitree_arm_adapter/periodic_loop.hpp"
#include "unitree_arm_adapter/protocol_supervisor_adapter.hpp"
#include "unitree_arm_adapter/receipt.hpp"
#include "unitree_arm_adapter/seqlock.hpp"
#include "unitree_arm_adapter/shared_memory.hpp"

namespace ua = unitree_arm_adapter;
namespace hs = unitree_arm_adapter::hardware_supervisor;
namespace hil = unitree_arm_adapter::hil;

namespace {

constexpr std::uint64_t kHilPeriodUs = 2'000U;

struct Options {
    std::string shared_memory_name{"/g1_arm_mpc_shadow"};
    std::string record_jsonl;
    std::uint64_t iterations{1'000U};
    std::uint64_t period_us{kHilPeriodUs};
    std::uint64_t session_nonce{0U};
    std::uint64_t safety_policy_id{0U};
    ua::Sha256Digest safety_policy_sha256{};
    bool paired_ingress_validated{false};
    bool allow_synthetic_fixture{false};
    bool offline_fixture_policy{false};
    bool offline_ownership_confirmed{false};
};

std::uint64_t ParseUnsigned(const std::string& value, const char* name) {
    std::size_t parsed = 0U;
    const std::uint64_t result = std::stoull(value, &parsed);
    if (parsed != value.size()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    return result;
}

std::uint8_t ParseHexByte(char high, char low) {
    const auto nibble = [](char value) -> std::uint8_t {
        const unsigned char byte = static_cast<unsigned char>(value);
        if (value >= '0' && value <= '9') {
            return static_cast<std::uint8_t>(value - '0');
        }
        const char lower = static_cast<char>(std::tolower(byte));
        if (lower >= 'a' && lower <= 'f') {
            return static_cast<std::uint8_t>(lower - 'a' + 10);
        }
        throw std::invalid_argument("SHA256 must contain hexadecimal digits");
    };
    return static_cast<std::uint8_t>((nibble(high) << 4U) | nibble(low));
}

ua::Sha256Digest ParseSha256(const std::string& value) {
    if (value.size() != ua::kSha256Bytes * 2U) {
        throw std::invalid_argument("--safety-policy-sha256 needs 64 hex digits");
    }
    ua::Sha256Digest digest{};
    for (std::size_t index = 0U; index < digest.size(); ++index) {
        digest[index] = ParseHexByte(value[index * 2U], value[index * 2U + 1U]);
    }
    return digest;
}

bool DigestIsZero(const ua::Sha256Digest& digest) noexcept {
    return std::all_of(
        digest.begin(), digest.end(), [](std::uint8_t value) {
            return value == 0U;
        });
}

void PrintUsage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --shm-name NAME                 existing protocol-v3 shared memory\n"
        << "  --iterations N                  finite 2 ms ticks (default 1000)\n"
        << "  --period-us N                   fixed tick period; must be 2000\n"
        << "  --record-jsonl PATH             required durable receipt output\n"
        << "  --session-nonce N               required expected session identity\n"
        << "  --paired-ingress-validated      attest expected paired bridge/session\n"
        << "  --allow-synthetic-fixture       explicitly allow test-only state input\n"
        << "  --offline-fixture-policy        enable non-hardware test limits\n"
        << "  --offline-ownership-confirmed   test-only ownership assertion\n"
        << "  --safety-policy-id N            fixture policy identity\n"
        << "  --safety-policy-sha256 HEX      fixture policy checksum\n\n"
        << "This executable only records would-write plans and receipts. It has no\n"
        << "device command transport and no output-enable option.\n";
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
        } else if (argument == "--iterations") {
            options.iterations = ParseUnsigned(
                require_value("--iterations"), "--iterations");
        } else if (argument == "--period-us") {
            options.period_us = ParseUnsigned(
                require_value("--period-us"), "--period-us");
        } else if (argument == "--record-jsonl") {
            options.record_jsonl = require_value("--record-jsonl");
        } else if (argument == "--session-nonce") {
            options.session_nonce = ParseUnsigned(
                require_value("--session-nonce"), "--session-nonce");
        } else if (argument == "--safety-policy-id") {
            options.safety_policy_id = ParseUnsigned(
                require_value("--safety-policy-id"), "--safety-policy-id");
        } else if (argument == "--safety-policy-sha256") {
            options.safety_policy_sha256 = ParseSha256(
                require_value("--safety-policy-sha256"));
        } else if (argument == "--paired-ingress-validated") {
            options.paired_ingress_validated = true;
        } else if (argument == "--allow-synthetic-fixture") {
            options.allow_synthetic_fixture = true;
        } else if (argument == "--offline-fixture-policy") {
            options.offline_fixture_policy = true;
        } else if (argument == "--offline-ownership-confirmed") {
            options.offline_ownership_confirmed = true;
        } else if (argument == "--help" || argument == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.iterations == 0U) {
        throw std::invalid_argument("--iterations must be positive and finite");
    }
    if (options.period_us != kHilPeriodUs) {
        throw std::invalid_argument("--period-us must be exactly 2000");
    }
    if (options.record_jsonl.empty()) {
        throw std::invalid_argument("--record-jsonl is required");
    }
    if (options.session_nonce == 0U) {
        throw std::invalid_argument("--session-nonce must be nonzero");
    }
    if (options.offline_fixture_policy &&
        (options.safety_policy_id == 0U ||
         DigestIsZero(options.safety_policy_sha256))) {
        throw std::invalid_argument(
            "offline fixture policy requires nonzero policy id and checksum");
    }
    if (options.allow_synthetic_fixture &&
        !options.offline_fixture_policy) {
        throw std::invalid_argument(
            "synthetic input is only available with offline fixture policy");
    }
    return options;
}

hs::SupervisorPolicy MakePolicy(const Options& options) {
    hs::SupervisorPolicy policy;
    policy.safety_policy_id = options.safety_policy_id;
    policy.safety_policy_sha256 = options.safety_policy_sha256;
    if (!options.offline_fixture_policy) {
        return policy;
    }

    // These deliberately broad limits only exercise the publisher-absent
    // contract. They are never hardware verification or a deployable policy.
    policy.site_policy_verified = true;
    policy.ownership_policy_verified = true;
    policy.startup_pd_verified = true;
    policy.active_control_verified = true;
    policy.release_behavior_verified = true;
    policy.output_authorized = true;
    policy.state_timeout_ns = 20'000'000U;
    policy.proposal_timeout_ns = 20'000'000U;
    policy.maximum_command_ticks = 3U;
    policy.maximum_arm_weight = 1.0;
    policy.maximum_weight_step_per_tick = 1.0;
    policy.release_weight_step_per_tick = 0.05;
    policy.limits.verified = true;
    policy.limits.q_min.fill(-3.2);
    policy.limits.q_max.fill(3.2);
    policy.limits.dq_abs_max.fill(30.0);
    policy.limits.kp_max.fill(500.0);
    policy.limits.kd_max.fill(100.0);
    policy.limits.tau_abs_max.fill(100.0);
    return policy;
}

std::uint64_t SaturatingAdd(
    std::uint64_t left, std::uint64_t right) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left
        ? std::numeric_limits<std::uint64_t>::max()
        : left + right;
}

std::uint64_t AgeAt(std::uint64_t timestamp, std::uint64_t now) noexcept {
    return timestamp != 0U && timestamp <= now ? now - timestamp : 0U;
}

bool IsSynthetic(const ua::RobotStatePayload& state) noexcept {
    return (state.ingress_flags & ua::kStateSyntheticFixture) != 0U;
}

int Run(const Options& options) {
    const std::uint64_t period_ns = options.period_us * 1'000U;
    ua::SharedMemoryRegion region = ua::SharedMemoryRegion::Open(
        options.shared_memory_name, false);
    ua::SharedMemoryLayout* const layout = region.get();
    hs::HardwareCommandSupervisor supervisor(
        MakePolicy(options), options.session_nonce);
    ua::AbsolutePeriodicTimer timer(period_ns);
    hil::RecordingSink sink;
    hil::RecordingCommandSink command_sink;
    hil::HilSupervisorDispatcher dispatcher(supervisor, 64U);

    std::uint64_t deadline_miss_count = 0U;
    std::uint64_t command_stale_count = 0U;
    std::uint64_t state_stale_count = 0U;

    for (std::uint64_t loop = 0U; loop < options.iterations; ++loop) {
        const ua::PeriodicTick tick = timer.WaitNext();
        ua::ArmCommandPayload command;
        ua::RobotStatePayload state;
        std::uint64_t command_sequence = 0U;
        std::uint64_t state_sequence = 0U;
        const bool command_read = ua::ReadSeqlockWithSequence(
            layout->command, command, command_sequence) &&
            command_sequence != 0U;
        const bool state_read = ua::ReadSeqlockWithSequence(
            layout->state, state, state_sequence) && state_sequence != 0U;
        const bool synthetic_input = state_read && IsSynthetic(state);
        const bool ingress_attested = options.paired_ingress_validated &&
            (!synthetic_input || options.allow_synthetic_fixture);

        if (state_read) {
            const hs::StateSample converted_state = ua::ToSupervisorState(
                state,
                ua::StateConversionContext{
                    options.session_nonce, ingress_attested});
            (void)dispatcher.ObserveState(
                state_sequence, state, converted_state);
        }
        const hil::CachedState* const latest_state = dispatcher.latest_state();

        // The supervisor uses a current-time gate. RecordingCommandSink then
        // takes its own clock reading and repeats both checks immediately at
        // the would-write boundary; these booleans are not its sole evidence.
        const std::uint64_t pre_sink_check_ns = ua::MonotonicNowNs();
        const std::uint64_t pre_sink_deadline_ns = SaturatingAdd(
            tick.scheduled_time_ns, period_ns);
        const bool supervisor_deadline_healthy = tick.deadline_healthy &&
            pre_sink_check_ns <= pre_sink_deadline_ns;
        const bool supervisor_expiry_healthy = command_read &&
            command.monotonic_timestamp_ns != 0U &&
            command.monotonic_timestamp_ns <= pre_sink_check_ns &&
            command.expires_timestamp_ns >= pre_sink_check_ns;

        hs::SupervisorSignals signals;
        signals.deadline_healthy = supervisor_deadline_healthy;
        signals.ownership_confirmed =
            options.offline_ownership_confirmed;
        const hil::DispatchResult dispatch = dispatcher.Dispatch(
            command_read ? &command : nullptr,
            command_read ? command_sequence : 0U,
            pre_sink_check_ns,
            signals);
        const hs::SupervisorResult& result = dispatch.supervisor;
        const hil::RecordingCommandSink::Attempt command_sink_attempt =
            command_sink.SubmitIfCertified(
                result.plan,
                command_read ? &command : nullptr,
                pre_sink_deadline_ns);
        const bool final_deadline_healthy = supervisor_deadline_healthy &&
            command_sink_attempt.deadline_healthy;
        const bool final_expiry_healthy = supervisor_expiry_healthy &&
            command_sink_attempt.expiry_healthy;
        if (!final_deadline_healthy) {
            ++deadline_miss_count;
        }
        if (command_read && !final_expiry_healthy) {
            ++command_stale_count;
        }
        if (result.reason == hs::SupervisorReason::kStateStaleOrFuture ||
            result.reason == hs::SupervisorReason::kInvalidState) {
            ++state_stale_count;
        }

        ua::ReceiptContext receipt_context;
        receipt_context.receipt_timestamp_ns =
            command_sink_attempt.check_timestamp_ns;
        receipt_context.loop_count = loop + 1U;
        receipt_context.receipt_id = loop + 1U;
        receipt_context.pre_sink_check_timestamp_ns =
            command_sink_attempt.check_timestamp_ns;
        receipt_context.pre_sink_deadline_ns = pre_sink_deadline_ns;
        receipt_context.wake_lateness_ns = tick.wake_lateness_ns;
        receipt_context.execution_time_ns =
            command_sink_attempt.check_timestamp_ns - tick.start_time_ns;
        receipt_context.command_age_ns = command_read
            ? AgeAt(
                  command.monotonic_timestamp_ns,
                  command_sink_attempt.check_timestamp_ns)
            : 0U;
        receipt_context.state_age_ns = latest_state != nullptr
            ? AgeAt(
                  latest_state->payload.monotonic_timestamp_ns,
                  command_sink_attempt.check_timestamp_ns)
            : 0U;
        receipt_context.deadline_miss_count = deadline_miss_count;
        receipt_context.command_stale_count = command_stale_count;
        receipt_context.state_stale_count = state_stale_count;
        receipt_context.command_snapshot_valid = command_read;
        receipt_context.state_snapshot_valid = latest_state != nullptr;
        receipt_context.deadline_healthy = final_deadline_healthy;
        receipt_context.pre_sink_deadline_healthy = final_deadline_healthy;
        receipt_context.pre_sink_expiry_healthy = final_expiry_healthy;
        receipt_context.output_enabled = false;
        receipt_context.dds_write_performed = false;
        receipt_context.sink_write_performed =
            command_sink_attempt.performed;
        receipt_context.sink_write_timestamp_ns =
            command_sink_attempt.performed
                ? command_sink_attempt.check_timestamp_ns
                : 0U;
        if (!final_deadline_healthy) {
            receipt_context.guard_reason = static_cast<std::uint32_t>(
                hs::SupervisorReason::kDeadlineMiss);
        } else if (command_read && !final_expiry_healthy) {
            receipt_context.guard_reason = static_cast<std::uint32_t>(
                hs::SupervisorReason::kProposalExpiredOrFuture);
        } else if (result.plan.write_permitted &&
                   !result.plan.release_plan &&
                   !command_sink_attempt.performed) {
            receipt_context.guard_reason = static_cast<std::uint32_t>(
                hs::SupervisorReason::kSourceBindingMismatch);
        } else {
            receipt_context.guard_reason =
                static_cast<std::uint32_t>(result.reason);
        }

        const ua::AdapterStatusPayload final_status = ua::BuildAdapterReceipt(
            command_read ? &command : nullptr,
            latest_state != nullptr ? &latest_state->payload : nullptr,
            result,
            receipt_context);
        hil::ReceiptRecord record = hil::MakeReceiptRecord(
            final_status,
            result,
            tick.scheduled_time_ns,
            command_sink_attempt.check_timestamp_ns - tick.start_time_ns,
            command_read ? command.flags : 0U,
            command_read,
            latest_state != nullptr,
            final_deadline_healthy,
            final_expiry_healthy,
            ingress_attested,
            synthetic_input,
            options.offline_fixture_policy,
            options.offline_ownership_confirmed,
            command_sink_attempt.performed);
        sink.Record(record);
        ua::WriteSeqlock(layout->status, final_status);
    }

    sink.WriteJsonLines(options.record_jsonl);
    std::cout
        << "publisher_absent_hil_completed=true\n"
        << "records=" << sink.records().size() << '\n'
        << "would_write_command_sink_count="
        << command_sink.records().size() << '\n'
        << "deadline_miss_count=" << deadline_miss_count << '\n'
        << "command_stale_count=" << command_stale_count << '\n'
        << "state_stale_count=" << state_stale_count << '\n'
        << "device_command_transport_present=false\n"
        << "hardware_output_performed=false\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Run(ParseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "publisher-absent HIL failed: " << error.what() << '\n';
        return 1;
    }
}
