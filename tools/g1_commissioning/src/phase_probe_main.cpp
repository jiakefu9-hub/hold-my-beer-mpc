#include "g1_commissioning/raw_capture.hpp"
#include <chrono>
#include <csignal>
#include <iostream>
#include <sched.h>
#include <unitree/robot/channel/channel_factory.hpp>

namespace gc = g1_commissioning;
namespace {
volatile std::sig_atomic_t stopped = 0;
void Signal(int) { stopped = 1; }
constexpr const char* kPermit = "PHASE_30S_READ_ONLY";
}

int main(int argc, char** argv) {
    std::shared_ptr<gc::RawJournal> journal;
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: g1_phase_probe NIC --output-dir NEW_DIR --permit-read-only "
                << kPermit << "\n30 s observer after FSM 500 and data are available."
                << " Remote walking is performed by the operator. No motion output.\n"
                << "Optional: --timing-cpu N (default 7). 6 ms read-only timing probe, SCHED_OTHER."
                << " Do not taskset the whole process to one CPU; support threads use other cores.\n";
            return 0;
        }
        if ((argc != 6 && argc != 8) || std::string(argv[2]) != "--output-dir" ||
            std::string(argv[4]) != "--permit-read-only" || std::string(argv[5]) != kPermit)
            throw std::runtime_error("invalid arguments; use --help (no network initialized)");
        int timing_cpu = 7;
        if (argc == 8) {
            if (std::string(argv[6]) != "--timing-cpu") throw std::runtime_error("expected --timing-cpu N");
            std::size_t used = 0;
            timing_cpu = std::stoi(argv[7], &used);
            if (used != std::string(argv[7]).size()) throw std::runtime_error("invalid timing CPU");
        }
        gc::PrepareTimingSupport(timing_cpu);  // before creating DDS, RPC or logger threads
        journal = std::make_shared<gc::RawJournal>(argv[3], 8192, true);
        journal->Text(gc::TimingEnvironmentJson(timing_cpu, argv[1], "before_dds"));
        journal->Text(gc::CaptureEvent("session_start",
            ",\"program\":\"g1_phase_probe\",\"read_only\":true,\"publisher_created\":false"
            ",\"duration_s\":30,\"phase_units_and_leg_mapping\":\"unverified\""
            ",\"network_interface\":\"" + gc::JsonEscape(argv[1]) + "\""));
        std::signal(SIGINT, Signal); std::signal(SIGTERM, Signal);
        unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
        auto inbox = std::make_shared<gc::LowStateInbox>();
        bool interrupted = false;
        std::uint64_t imu_count = 0, low_count = 0, phase_ok = 0, phase_failed = 0;
        bool imu_healthy = true, lowstate_healthy = true;
        {
            gc::CaptureStreams streams(journal, inbox);
            gc::CaptureGetters getters(journal);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            bool ready = false;
            while (!stopped && std::chrono::steady_clock::now() < deadline) {
                const auto state = inbox->Latest();
                const auto now = gc::MonotonicNowNs();
                if (getters.latest_fsm() == 500 && streams.ImuFresh(now) && state &&
                    state->crc_valid && now >= state->captured_monotonic_ns &&
                    now - state->captured_monotonic_ns < 100000000ULL) { ready = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (!ready) throw std::runtime_error("FSM 500 and fresh torso/LowState not ready in 5 s");
            gc::PinTimingControl(timing_cpu);
            journal->Text(gc::TimingEnvironmentJson(timing_cpu, argv[1], "observation_ready"));
            const auto epoch = gc::HostNowNs();
            const auto end = epoch + 30000000000ULL;
            journal->Text(gc::CaptureEvent("observation_start", ",\"planned_duration_s\":30,\"epoch_ns\":" +
                std::to_string(epoch) + ",\"planned_end_ns\":" + std::to_string(end)));
            std::cout << "READY: 30-second READ-ONLY recording; timing CPU=" << timing_cpu
                << ", period=6 ms, SCHED_OTHER. You may use the remote.\n";
            auto scheduled = epoch, next_print = epoch;
            std::optional<gc::HostTimingRecord> pending;
            while (!stopped && scheduled < end) {
                std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::nanoseconds(scheduled)));
                gc::HostTimingRecord sample;
                sample.started_ns = gc::HostNowNs();
                if (sample.started_ns >= end) break;
                sample.scheduled_ns = scheduled;
                sample.cpu = sched_getcpu();
                // Publish the PREVIOUS complete iteration. Its finished_ns thus
                // includes console/log enqueue work without serializing on this core.
                if (pending) journal->Push(*pending);
                if (!journal->Healthy()) throw std::runtime_error("raw recording failed/overflowed");
                const auto state = inbox->Latest();
                const auto imu = streams.LatestImu();
                const auto received_now = gc::MonotonicNowNs();
                sample.snapshot_done_ns = received_now;
                if (state) {
                    sample.state_received_ns = state->captured_monotonic_ns;
                    sample.state_sequence = state->capture_sequence;
                }
                if (imu) { sample.imu_received_ns = imu->received_ns; sample.imu_sequence = imu->sequence; }
                sample.state_valid = state && state->crc_valid && !state->tick_regression &&
                    received_now >= state->captured_monotonic_ns &&
                    received_now - state->captured_monotonic_ns <= 100000000ULL;
                sample.imu_fresh = streams.ImuFresh(gc::MonotonicNowNs());
                if (!sample.state_valid) lowstate_healthy = false;
                if (!sample.imu_fresh) imu_healthy = false;
                if (received_now >= next_print) {
                    std::cout << "t=" << static_cast<double>(received_now-epoch)*1e-9
                        << " fsm=" << getters.latest_fsm() << " phase " << getters.PhaseStatus()
                        << " imu=" << streams.imu_count() << " lowstate=" << streams.low_count() << '\n';
                    next_print = received_now + 500000000ULL;
                }
                sample.finished_ns = gc::HostNowNs();
                scheduled = gc::NextTimingSlot(scheduled, sample.finished_ns, sample.missed_slots_after);
                pending = sample;
            }
            if (pending) journal->Push(*pending);
            // The last sample is nominally at 29.994 s; keep capturing until
            // the full 30 s window ends, rather than silently shortening it.
            if (!stopped && gc::HostNowNs() < end)
                std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::nanoseconds(end)));
            journal->Text(gc::CaptureEvent("observation_end", ",\"end_ns\":" + std::to_string(gc::HostNowNs())));
            journal->Text(gc::TimingEnvironmentJson(timing_cpu, argv[1], "observation_end"));
            interrupted = stopped != 0;
            getters.Stop();
            streams.Stop();
            imu_count = streams.imu_count(); low_count = streams.low_count();
            phase_ok = getters.phase_success_count(); phase_failed = getters.phase_failure_count();
        }
        journal->Text(gc::CaptureEvent("session_end", ",\"outcome\":\"" +
            std::string(interrupted ? "interrupted" : "observation_completed") +
            "\",\"imu_callbacks\":" + std::to_string(imu_count) +
            ",\"lowstate_callbacks\":" + std::to_string(low_count) +
            ",\"phase_parse_success\":" + std::to_string(phase_ok) +
            ",\"phase_failure\":" + std::to_string(phase_failed) +
            ",\"imu_continuously_fresh\":" + (imu_healthy ? "true" : "false") +
            ",\"lowstate_continuously_healthy\":" + (lowstate_healthy ? "true" : "false") +
            ",\"crc_rejected\":" + std::to_string(inbox->crc_rejected_count()) +
            ",\"queue_dropped\":" + std::to_string(journal->dropped()) +
            ",\"periodicity_verified\":false,\"publisher_created\":false"));
        journal->Finish();
        std::cout << "Saved " << journal->written() << " records to " << argv[3]
            << "/raw.jsonl; phase success/failure=" << phase_ok << '/' << phase_failed
            << ". Valid replies alone do NOT establish a gait period.\n";
        if (!journal->Healthy() || !imu_healthy || !lowstate_healthy || inbox->crc_rejected_count() != 0) return 3;
        return interrupted ? 130 : 0;
    } catch (const std::exception& e) {
        if (journal) {
            journal->Text(gc::CaptureEvent("session_error", ",\"reason\":\"" + gc::JsonEscape(e.what()) + "\""));
            journal->Finish();
        }
        std::cerr << "Phase observation: " << e.what() << '\n';
        return 1;
    }
}
