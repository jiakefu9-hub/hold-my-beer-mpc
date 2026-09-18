#include "g1_commissioning/raw_capture.hpp"
#include "g1_commissioning/capture_heading.hpp"
#include "g1_commissioning/timed_walk.hpp"
#include <chrono>
#include <csignal>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>

namespace gc = g1_commissioning;
namespace {
volatile std::sig_atomic_t stopped = 0;
void Signal(int) { stopped = 1; }
constexpr const char* kPermit = "TIMED_WALK_RAW_CAPTURE";

std::string Errors(const gc::ValidationResult& v) {
    std::string result;
    for (const auto& s : v.errors) result += s + "; ";
    return result;
}

// Only the velocity setter is registered. No mode/start/ownership RPC exists in
// this executable. The operator establishes FSM 500 before starting it.
class VelocityClient final : public unitree::robot::Client {
public:
    VelocityClient() : Client("sport", false) { Init(); }
    void Init() override {
        SetTimeout(0.1F);
        SetApiVersion("1.0.0.0");
        RegistApi(unitree::robot::g1::ROBOT_API_ID_LOCO_SET_VELOCITY);
    }
    int Send(float vx, float wz, float lease) {
        unitree::robot::g1::JsonizeVelocityCommand command;
        command.velocity = {vx, 0.0F, wz};
        command.duration = lease;
        std::string reply;
        return Call(unitree::robot::g1::ROBOT_API_ID_LOCO_SET_VELOCITY,
                    unitree::common::ToJsonString(command), reply);
    }
};

class WalkWorker {
public:
    WalkWorker(std::shared_ptr<gc::RawJournal> journal, gc::CaptureStreams& streams,
               std::function<std::string()> health)
        : journal_(std::move(journal)), streams_(streams), health_(std::move(health)) {}
    ~WalkWorker() { Stop(); }
    void Start(std::uint64_t epoch) {
        epoch_ = epoch;
        worker_ = std::thread([this] { Run(); });
    }
    void RequestZero() { zero_requested_ = true; wake_.notify_all(); }
    void Stop() {
        stop_worker_ = true; zero_requested_ = true; wake_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    bool Failed() const { return failed_.load(); }
    std::uint64_t LastZeroReply() const { return zero_reply_.load(); }
private:
    int Send(double vx, double wz, double lease, double t, const gc::CaptureHeading::Output& h) {
        std::ostringstream fields;
        fields << std::setprecision(17) << ",\"task_elapsed_s\":" << t
            << ",\"vx_m_s\":" << vx << ",\"vy_m_s\":0,\"yaw_rate_rad_s\":" << wz
            << ",\"duration_s\":" << lease << ",\"heading_reference_rad\":" << h.reference
            << ",\"heading_reference_frozen\":" << (h.reference_frozen ? "true" : "false")
            << ",\"heading_relative_yaw_rad\":" << h.relative_yaw
            << ",\"heading_error_rad\":" << h.error
            << ",\"heading_filtered_yaw_rad\":" << h.filtered_yaw
            << ",\"heading_filtered_vertical_rate_rad_s\":" << h.filtered_rate;
        const auto begin = gc::MonotonicNowNs();
        journal_->Text(gc::CaptureEvent("velocity_request", fields.str() +
            ",\"request_ns\":" + std::to_string(begin)));
        const int rc = client_.Send(static_cast<float>(vx), static_cast<float>(wz), static_cast<float>(lease));
        const auto end = gc::MonotonicNowNs();
        journal_->Text(gc::CaptureEvent("velocity_reply", ",\"request_ns\":" +
            std::to_string(begin) + ",\"reply_ns\":" + std::to_string(end) +
            ",\"return_code\":" + std::to_string(rc) +
            ",\"zero_command\":" + (vx == 0 && wz == 0 ? "true" : "false")));
        if (rc == 0 && vx == 0 && wz == 0) zero_reply_ = end;
        return rc;
    }
    void Run() noexcept {
        gc::CaptureHeading heading;
        gc::CaptureHeading::Output h;
        try {
            while (!stop_worker_) {
                const auto iteration = std::chrono::steady_clock::now();
                const auto health = health_();
                if (!health.empty()) {
                    failed_ = true; zero_requested_ = true;
                    journal_->Text(gc::CaptureEvent("velocity_health_stop", ",\"reason\":\"" + gc::JsonEscape(health) + "\""));
                }
                const auto imu = streams_.LatestImu();
                if (imu && health.empty()) {
                    const auto& rpy = imu->message.rpy();
                    const auto& gyro = imu->message.gyroscope();
                    const double roll = rpy[0], pitch = rpy[1];
                    const double vertical_rate = -std::sin(pitch) * gyro[0] +
                        std::cos(pitch) * std::sin(roll) * gyro[1] +
                        std::cos(pitch) * std::cos(roll) * gyro[2];
                    const double observation_t =
                        static_cast<double>(gc::MonotonicNowNs() - epoch_) * 1e-9;
                    heading.Observe(imu->received_ns, observation_t, rpy[2], vertical_rate);
                }
                // Re-evaluate time immediately before each call; never queue an
                // old nonzero command to be sent after the stop boundary.
                const double t = static_cast<double>(gc::MonotonicNowNs() - epoch_) * 1e-9;
                if (t >= gc::TimedWalkPlan::kWalkStart && !heading.ReferenceFrozen()) {
                    heading.FreezeReference();
                    h = heading.Current();
                    std::ostringstream fields;
                    fields << std::setprecision(17)
                        << ",\"task_elapsed_s\":" << t
                        << ",\"yaw0_rad\":" << h.reference
                        << ",\"h0_from_navigation_world_yaw_rad\":" << -h.reference
                        << ",\"requested_reference_start_s\":" << gc::CaptureHeading::kReferenceStartS
                        << ",\"requested_reference_end_s\":" << gc::CaptureHeading::kReferenceEndS
                        << ",\"reference_sample_count\":" << h.reference_samples
                        << ",\"reference_first_sample_task_s\":" << h.reference_first_s
                        << ",\"reference_last_sample_task_s\":" << h.reference_last_s
                        << ",\"reference_observed_span_s\":" << h.reference_span_s
                        << ",\"h0_definition\":\"fixed_run_frame_x_along_pre_walk_mean_yaw_z_vertical\"";
                    journal_->Text(gc::CaptureEvent("heading_reference_frozen", fields.str()));
                }
                if (imu && health.empty()) h = heading.Current();
                const bool walking = !zero_requested_ && !stopped && gc::TimedWalkPlan::Walking(t);
                if (Send(walking ? gc::TimedWalkPlan::kForwardSpeed : 0.0,
                         walking ? h.correction : 0.0, gc::TimedWalkPlan::Lease(t), t, h) != 0) {
                    failed_ = true; zero_requested_ = true;
                    break;
                }
                if (failed_) break;
                auto next = iteration + std::chrono::milliseconds(50);
                // Align nominal RPC dispatch to 5 s and 13 s even when the
                // periodic worker has acquired a small scheduling offset.
                for (double boundary : {gc::TimedWalkPlan::kWalkStart, gc::TimedWalkPlan::kWalkStop}) {
                    if (t < boundary) {
                        const auto at = std::chrono::steady_clock::time_point(
                            std::chrono::nanoseconds(epoch_ + static_cast<std::uint64_t>(boundary * 1e9)));
                        next = std::min(next, at);
                    }
                }
                // No catch-up bursts after a slow reply.
                next = std::max(next, std::chrono::steady_clock::now() + std::chrono::milliseconds(1));
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait_until(lock, next, [this] { return stop_worker_.load(); });
            }
        } catch (const std::exception& e) {
            failed_ = true;
            journal_->Text(gc::CaptureEvent("velocity_exception", ",\"reason\":\"" + gc::JsonEscape(e.what()) + "\""));
        }
        // Best effort only: a zero-velocity RPC cannot certify a physical stop.
        try {
            if (Send(0, 0, gc::TimedWalkPlan::kVelocityLease,
                     static_cast<double>(gc::MonotonicNowNs() - epoch_) * 1e-9, h) != 0) failed_ = true;
        } catch (...) { failed_ = true; }
    }
    std::shared_ptr<gc::RawJournal> journal_;
    gc::CaptureStreams& streams_;
    std::function<std::string()> health_;
    VelocityClient client_;
    std::atomic<bool> stop_worker_{false}, zero_requested_{false}, failed_{false};
    std::atomic<std::uint64_t> zero_reply_{0};
    std::uint64_t epoch_{0};
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread worker_;
};

unitree_hg::msg::dds_::LowCmd_ ArmMessage(const gc::SiteProfile& profile,
    const gc::CommandFrame& frame, const gc::StateSample& state) {
    unitree_hg::msg::dds_::LowCmd_ message;
    message.mode_pr(state.mode_pr); message.mode_machine(state.mode_machine);
    for (std::size_t slot = 0; slot < gc::kArmSlotCount; ++slot) {
        if (!profile.valid_slots[slot]) continue;
        auto& m = message.motor_cmd()[gc::kArmMotorIndices[slot]];
        m.mode(1); m.q(static_cast<float>(frame.q[slot]));
        m.dq(static_cast<float>(frame.dq[slot])); m.tau(static_cast<float>(frame.tau[slot]));
        m.kp(static_cast<float>(frame.kp[slot])); m.kd(static_cast<float>(frame.kd[slot]));
    }
    message.motor_cmd()[gc::kWeightMotorIndex].q(static_cast<float>(frame.weight));
    message.crc(crc32_core(reinterpret_cast<std::uint32_t*>(&message),
        static_cast<std::uint32_t>(sizeof(message) / sizeof(std::uint32_t) - 1U)));
    return message;
}

gc::StateSample Startup(gc::LowStateInbox& inbox, gc::CaptureStreams& streams,
    gc::ArmStopInterlock& interlock, gc::RawJournal& journal, const gc::SiteProfile& profile,
    std::uint64_t after_sequence) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(profile.startup_wait_s);
    std::uint64_t previous = after_sequence;
    std::size_t consecutive = 0;
    while (!stopped && std::chrono::steady_clock::now() < deadline) {
        if (!journal.Healthy()) throw std::runtime_error("recorder unavailable");
        const auto stop = interlock.Check(gc::MonotonicNowNs());
        if (!stop.empty() && stop != "FSM not yet observed") throw std::runtime_error(stop);
        const auto s = inbox.Latest();
        if (s && s->capture_sequence > previous) {
            previous = s->capture_sequence;
            if (stop.empty() && streams.ImuFresh(gc::MonotonicNowNs()) &&
                gc::ValidateState(*s, profile, gc::MonotonicNowNs(), true, true).ok()) {
                if (++consecutive >= profile.startup_valid_samples) return *s;
            } else consecutive = 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("startup interrupted or fresh FSM-500/LowState/torso samples unavailable");
}

int Execute(const std::string& nic, const gc::SiteProfile& profile,
            const std::string& profile_path, std::shared_ptr<gc::RawJournal> journal) {
    std::filesystem::copy_file(profile_path, std::filesystem::path(journal->directory()) / "arm_profile.conf");
    journal->Text(gc::CaptureEvent("session_start",
        ",\"program\":\"g1_walk_capture\",\"publisher_created\":false,\"mode_setter_registered\":false"
        ",\"walk_start_s\":5,\"walk_stop_s\":15,\"release_start_s\":18,\"end_s\":21"
        ",\"speed_m_s\":0.5,\"heading_hold\":true,\"heading_filter_s\":1"
        ",\"heading_target\":\"fixed_run_h0_positive_x\""
        ",\"heading_reference_start_s\":3,\"heading_reference_end_s\":5"
        ",\"heading_kp\":1,\"heading_kd\":0.1,\"heading_max_rate_rad_s\":0.25"
        ",\"raw_data_transformed\":false,\"h0_derived_data_requires_postprocess\":true"
        ",\"phase_controls_timing\":false"
        ",\"network_interface\":\"" + gc::JsonEscape(nic) + "\""));
    journal->Text(gc::ProfileSummaryJson(profile, gc::ValidateProfile(profile, gc::ValidationUse::kRealOutput)));
    unitree::robot::ChannelFactory::Instance()->Init(0, nic);
    auto inbox = std::make_shared<gc::LowStateInbox>();
    auto interlock = std::make_shared<gc::ArmStopInterlock>(500);
    gc::CaptureStreams streams(journal, inbox, interlock);
    gc::CaptureGetters getters(journal, interlock);
    auto initial = Startup(*inbox, streams, *interlock, *journal, profile, 0);
    std::cout << "REAL WALK OUTPUT: 3 s arm entry, 2 s wait, 10 s at 0.5 m/s with heading hold,"
        << " 3 s stop hold, 3 s release. Robot must already balance in FSM 500.\n"
        << "Type exactly: EXECUTE " << profile.robot_id << "\n> " << std::flush;
    std::string reply;
    if (!std::getline(std::cin, reply) || reply != "EXECUTE " + profile.robot_id || stopped)
        throw std::runtime_error("confirmation rejected before command output");
    // Exclude every pre-confirmation sample, including samples received while
    // waiting for console input, from the second consecutive-fresh-state gate.
    const auto before = inbox->Latest();
    initial = Startup(*inbox, streams, *interlock, *journal, profile,
                      before ? before->capture_sequence : initial.capture_sequence);
    const auto health = [&]() -> std::string {
        const auto stop = interlock->Check(gc::MonotonicNowNs());
        if (!stop.empty()) return stop;
        if (!journal->Healthy()) return "raw recorder failure/queue overflow";
        if (!streams.ImuFresh(gc::MonotonicNowNs())) return "torso IMU unavailable/stale/nonfinite";
        const auto s = inbox->Latest();
        if (!s) return "no LowState";
        return Errors(gc::ValidateState(*s, profile, gc::MonotonicNowNs(), false, true));
    };
    WalkWorker velocity(journal, streams, health);  // client only, no requests yet
    const auto gate = health();
    if (!gate.empty()) throw std::runtime_error("pre-publisher: " + gate);
    unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_> publisher("rt/arm_sdk");
    publisher.InitChannel();
    journal->Text(gc::CaptureEvent("arm_publisher_created", ",\"topic\":\"rt/arm_sdk\""));
    gc::TrajectoryPlanner planner(profile, initial);
    const auto epoch = gc::MonotonicNowNs();
    journal->Text(gc::CaptureEvent("task_epoch", ",\"task_epoch_monotonic_ns\":" + std::to_string(epoch)));
    gc::CommandFrame last = planner.Sample(0);
    std::uint64_t sequence = 0;
    double abort_start = -1;
    gc::CommandFrame abort_hold;
    std::string last_stage;
    const auto publish = [&](gc::CommandFrame frame, const gc::StateSample& state, const std::string& reason) {
        const auto before_write = health();
        if (!before_write.empty()) throw std::runtime_error(before_write);
        frame.sequence = ++sequence;
        const auto write_begin = gc::MonotonicNowNs();
        const bool ok = publisher.Write(ArmMessage(profile, frame, state));
        const auto write_end = gc::MonotonicNowNs();
        auto record = gc::FrameJson(frame, &state, ok ? "dds_write" : "dds_write_failed", reason);
        record.pop_back();
        record += ",\"write_begin_monotonic_ns\":" + std::to_string(write_begin) +
            ",\"write_end_monotonic_ns\":" + std::to_string(write_end) + "}";
        journal->Text(std::move(record));
        if (!ok) throw std::runtime_error("arm DDS write failed");
        last = frame;
    };
    try {
        publish(last, initial, "initial_zero_weight");
        velocity.Start(epoch);
        std::cout << "RUNNING: task t=0; raw capture active. Ctrl-C requests walking stop then arm release.\n";
        while (true) {
            const auto iteration = std::chrono::steady_clock::now();
            const double t = static_cast<double>(gc::MonotonicNowNs() - epoch) * 1e-9;
            const auto failure = health();
            if (!failure.empty()) throw std::runtime_error(failure);
            if (velocity.Failed()) throw std::runtime_error("velocity RPC worker failed");
            if (t > profile.total_timeout_s + (abort_start >= 0 ? 6.0 : 0.0))
                throw std::runtime_error("finite session timeout");
            auto frame = planner.Sample(t);
            std::string stage = gc::TimedWalkPlan::Stage(t);
            if (stopped && abort_start < 0) {
                abort_start = t;
                abort_hold = last;
                velocity.RequestZero();
                journal->Text(gc::CaptureEvent("operator_stop_requested"));
            }
            if (abort_start >= 0) {
                frame = abort_hold;
                frame.elapsed_s = t;
                const double since_abort = t - abort_start;
                frame.weight = abort_hold.weight * std::clamp(1.0 - (since_abort - 3.0) / 3.0, 0.0, 1.0);
                frame.phase = since_abort < 3 ? gc::Phase::kHold : gc::Phase::kSigintRelease;
                frame.terminal = since_abort >= 6.0;
                stage = since_abort < 3 ? "operator_stop_settle" : "operator_arm_release";
            }
            if (stage != last_stage) {
                journal->Text(gc::CaptureEvent("task_stage", ",\"stage\":\"" + stage +
                    "\",\"task_elapsed_s\":" + std::to_string(t)));
                std::cout << "t=" << t << " " << stage << '\n';
                last_stage = stage;
            }
            // A successful zero-speed RPC is required before normal arm release.
            const double required_zero_after = abort_start >= 0 ? abort_start : gc::TimedWalkPlan::kWalkStop;
            const bool releasing = abort_start >= 0 ? t >= abort_start + 3.0 : t >= gc::TimedWalkPlan::kReleaseStart;
            if (releasing && velocity.LastZeroReply() < epoch + static_cast<std::uint64_t>(required_zero_after * 1e9))
                throw std::runtime_error("no successful post-stop zero-speed reply before arm release");
            const auto state = inbox->Latest();
            if (!state) throw std::runtime_error("LowState vanished");
            publish(frame, *state, stage);
            if (frame.terminal) {
                velocity.Stop();
                getters.Stop();
                journal->Text(gc::CaptureEvent("session_end", ",\"outcome\":\"" +
                    std::string(abort_start >= 0 ? "operator_stop_release_completed" : "normal_release_completed") +
                    "\",\"final_weight\":0,\"velocity_final_rpc_ok\":" + (velocity.Failed() ? "false" : "true") +
                    ",\"physical_stop_verified\":false"));
                publisher.CloseChannel();
                return velocity.Failed() ? 3 : (abort_start >= 0 ? 130 : 0);
            }
            std::this_thread::sleep_until(iteration + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(profile.control_period_ms / 1000.0)));
        }
    } catch (const std::exception& e) {
        velocity.Stop();
        bool final_arm_attempted = false, final_arm_written = false;
        // Respect a mode/remote stop: no final arm command after its latch.
        if (interlock->Check(gc::MonotonicNowNs()).empty()) {
            const auto state = inbox->Latest();
            if (state) {
                final_arm_attempted = true;
                try {
                    final_arm_written = publisher.Write(ArmMessage(profile,
                        gc::MakeFaultZeroWeightFrame(profile, ++sequence), *state));
                } catch (...) {}
            }
        }
        journal->Text(gc::CaptureEvent("session_fault", ",\"reason\":\"" + gc::JsonEscape(e.what()) +
            "\",\"final_arm_attempted\":" + (final_arm_attempted ? "true" : "false") +
            ",\"final_arm_write\":" + (final_arm_written ? "true" : "false") +
            ",\"physical_stop_verified\":false"));
        publisher.CloseChannel();
        std::cerr << "Capture stopped: " << e.what() << '\n';
        return 3;
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::shared_ptr<gc::RawJournal> journal;
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: g1_walk_capture NIC --profile FIELD_PROFILE --output-dir NEW_DIR"
                << " --permit-real-output " << kPermit << "\n"
                << "Real arm AND walking output. 21 s timed plan; H0 heading is frozen from"
                << " mean torso yaw during task seconds [3,5)."
                << " Operator must establish FSM 500; program never switches modes.\n";
            return 0;
        }
        if (argc != 8 || std::string(argv[2]) != "--profile" ||
            std::string(argv[4]) != "--output-dir" || std::string(argv[6]) != "--permit-real-output" ||
            std::string(argv[7]) != kPermit)
            throw std::runtime_error("invalid arguments; use --help (no network initialized)");
        const auto profile = gc::LoadSiteProfile(argv[3]);
        gc::TimedWalkPlan::Validate(profile);
        const auto validation = gc::ValidateProfile(profile, gc::ValidationUse::kRealOutput);
        if (!validation.ok()) throw std::runtime_error("profile refused: " + Errors(validation));
        journal = std::make_shared<gc::RawJournal>(argv[5]);
        std::signal(SIGINT, Signal); std::signal(SIGTERM, Signal);
        const int result = Execute(argv[1], profile, argv[3], journal);
        journal->Text(gc::CaptureEvent("capture_drained", ",\"queue_dropped\":" + std::to_string(journal->dropped())));
        journal->Finish();
        std::cout << "Saved raw capture: " << argv[5] << "/raw.jsonl; records=" << journal->written()
            << " dropped=" << journal->dropped() << '\n';
        return journal->Healthy() ? result : 3;
    } catch (const std::exception& e) {
        if (journal) {
            journal->Text(gc::CaptureEvent("startup_error", ",\"reason\":\"" + gc::JsonEscape(e.what()) + "\""));
            journal->Finish();
        }
        std::cerr << "Walk capture refused/failed: " << e.what() << '\n';
        return 1;
    }
}
