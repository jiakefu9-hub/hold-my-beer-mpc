#include "g1_commissioning/raw_capture.hpp"
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <unitree/dds_wrapper/common/crc.h>

namespace g1_commissioning {
namespace {
template <typename T> void Number(std::ostream& out, T value) {
    if constexpr (std::is_floating_point_v<T>) {
        if (!std::isfinite(value)) {
            out << (std::isnan(value) ? "\"NaN\"" :
                    (value > 0 ? "\"Infinity\"" : "\"-Infinity\""));
            return;
        }
    }
    out << +value;
}
template <typename Values> void Array(std::ostream& out, const Values& values) {
    out << '[';
    bool first = true;
    for (auto value : values) {
        if (!first) out << ',';
        first = false;
        Number(out, value);
    }
    out << ']';
}
void ImuFields(std::ostream& out, const unitree_hg::msg::dds_::IMUState_& m) {
    out << "\"quaternion_wxyz\":"; Array(out, m.quaternion());
    out << ",\"rpy_rad\":"; Array(out, m.rpy());
    out << ",\"gyroscope_rad_s\":"; Array(out, m.gyroscope());
    out << ",\"accelerometer_raw_m_s2\":"; Array(out, m.accelerometer());
    out << ",\"temperature_raw\":" << m.temperature();
}
}

std::string CaptureEvent(const std::string& event, const std::string& fields) {
    return "{\"schema\":\"g1_capture_event_v1\",\"event\":\"" +
        JsonEscape(event) + "\",\"monotonic_ns\":" +
        std::to_string(MonotonicNowNs()) + fields + "}";
}

std::string RawImuJson(const RawImuRecord& r) {
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\"schema\":\"g1_torso_imu_raw_v1\",\"topic\":\"rt/secondary_imu\""
        << ",\"received_monotonic_ns\":" << r.received_ns
        << ",\"host_callback_sequence\":" << r.sequence
        << ",\"source_timestamp_available\":false,";
    ImuFields(out, r.message);
    out << '}';
    return out.str();
}

std::string RawLowStateJson(const RawLowStateRecord& r) {
    auto m = r.message;
    const bool crc_valid = m.crc() == crc32_core(
        reinterpret_cast<std::uint32_t*>(&m),
        static_cast<std::uint32_t>(sizeof(m) / sizeof(std::uint32_t) - 1U));
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\"schema\":\"g1_lowstate_raw_v1\",\"topic\":\"rt/lowstate\""
        << ",\"received_monotonic_ns\":" << r.received_ns
        << ",\"host_callback_sequence\":" << r.sequence
        << ",\"tick_raw\":" << m.tick()
        << ",\"mode_pr\":" << +m.mode_pr()
        << ",\"mode_machine\":" << +m.mode_machine()
        << ",\"crc_raw\":" << m.crc()
        << ",\"crc_valid\":" << (crc_valid ? "true" : "false")
        << ",\"version\":"; Array(out, m.version());
    out << ",\"wireless_remote_bytes\":"; Array(out, m.wireless_remote());
    out << ",\"reserve\":"; Array(out, m.reserve());
    out << ",\"pelvis_imu\":{"; ImuFields(out, m.imu_state()); out << '}';
    out << ",\"motors\":[";
    for (std::size_t i = 0; i < m.motor_state().size(); ++i) {
        const auto& motor = m.motor_state()[i];
        if (i != 0) out << ',';
        out << "{\"index\":" << i << ",\"mode\":" << +motor.mode()
            << ",\"q_rad\":"; Number(out, motor.q());
        out << ",\"dq_rad_s\":"; Number(out, motor.dq());
        out << ",\"ddq_raw_rad_s2\":"; Number(out, motor.ddq());
        out << ",\"tau_est_nm\":"; Number(out, motor.tau_est());
        out << ",\"temperature_raw\":"; Array(out, motor.temperature());
        out << ",\"vol_raw\":"; Number(out, motor.vol());
        out << ",\"sensor_raw\":"; Array(out, motor.sensor());
        out << ",\"motorstate_raw\":" << motor.motorstate() << '}';
    }
    out << "]}";
    return out.str();
}

RawJournal::RawJournal(const std::string& directory, std::size_t capacity, bool record_timing)
    : directory_(directory), capacity_(capacity), record_timing_(record_timing) {
    if (capacity_ == 0 || !std::filesystem::create_directory(directory_))
        throw std::runtime_error("capture directory must be new; parent must exist: " + directory_);
    output_.open(std::filesystem::path(directory_) / "raw.jsonl");
    if (!output_) throw std::runtime_error("cannot create raw.jsonl");
    worker_ = std::thread([this] { Run(); });
}
RawJournal::~RawJournal() { Finish(); }
void RawJournal::Push(RawImuRecord r) { Enqueue(std::move(r)); }
void RawJournal::Push(RawLowStateRecord r) { Enqueue(std::move(r)); }
void RawJournal::Push(HostTimingRecord r) { Enqueue(std::move(r)); }
void RawJournal::Text(std::string json) { Enqueue(std::move(json)); }
void RawJournal::Enqueue(Record r) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || failed_ || queue_.size() >= capacity_) {
        ++dropped_;
        return;
    }
    queue_.push_back({std::move(r), record_timing_ ? MonotonicNowNs() : 0});
    wake_.notify_one();
}
void RawJournal::Finish() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
}
void RawJournal::Run() noexcept {
    try {
        auto last_flush = std::chrono::steady_clock::now();
        while (true) {
            QueuedRecord r;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait_for(lock, std::chrono::milliseconds(100),
                               [this] { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) {
                    output_.flush();
                    if (!output_) { failed_ = true; return; }
                    if (stopping_) break;
                    continue;
                }
                r = std::move(queue_.front());
                queue_.pop_front();
            }
            const auto dequeued = record_timing_ ? MonotonicNowNs() : 0;
            auto json = std::visit([](const auto& value) -> std::string {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, RawImuRecord>) return RawImuJson(value);
                else if constexpr (std::is_same_v<T, RawLowStateRecord>) return RawLowStateJson(value);
                else if constexpr (std::is_same_v<T, HostTimingRecord>) return HostTimingJson(value);
                else return value;
            }, r.data);
            if (record_timing_) {
                const auto serialized = MonotonicNowNs();
                if (json.empty() || json.back() != '}') throw std::runtime_error("expected JSON object");
                json.pop_back();
                json += ",\"journal_enqueued_ns\":" + std::to_string(r.enqueued_ns) +
                    ",\"journal_dequeued_ns\":" + std::to_string(dequeued) +
                    ",\"journal_serialized_ns\":" + std::to_string(serialized) + "}";
            }
            output_ << json << '\n';
            ++written_;
            const auto now = std::chrono::steady_clock::now();
            if (now - last_flush >= std::chrono::milliseconds(100)) {
                output_.flush();
                last_flush = now;
            }
            if (!output_) { failed_ = true; return; }
        }
        output_.flush();
        if (!output_) failed_ = true;
    } catch (...) { failed_ = true; }
}

CaptureStreams::CaptureStreams(std::shared_ptr<RawJournal> journal,
    std::shared_ptr<LowStateInbox> inbox, std::shared_ptr<ArmStopInterlock> interlock)
    : journal_(std::move(journal)), inbox_(std::move(inbox)), interlock_(std::move(interlock)),
      imu_sub_("rt/secondary_imu"), low_sub_("rt/lowstate") {
    // Queue length 0 invokes callbacks directly, avoiding an SDK latest-only
    // queue. Callbacks copy to our bounded journal; disk work is asynchronous.
    imu_sub_.InitChannel([this](const void* raw) { ImuCallback(raw); }, 0);
    low_sub_.InitChannel([this](const void* raw) { LowCallback(raw); }, 0);
}
CaptureStreams::~CaptureStreams() { Stop(); }
void CaptureStreams::Stop() { low_sub_.CloseChannel(); imu_sub_.CloseChannel(); }
void CaptureStreams::ImuCallback(const void* raw) {
    if (!raw) return;
    RawImuRecord r{MonotonicNowNs(), ++imu_count_,
        *static_cast<const unitree_hg::msg::dds_::IMUState_*>(raw)};
    { std::lock_guard<std::mutex> lock(mutex_); latest_ = r; }
    journal_->Push(std::move(r));
}
void CaptureStreams::LowCallback(const void* raw) {
    if (!raw) return;
    RawLowStateRecord r{MonotonicNowNs(), ++low_count_,
        *static_cast<const unitree_hg::msg::dds_::LowState_*>(raw)};
    if (interlock_) HandleArmStopState(*inbox_, *interlock_, raw);
    else inbox_->Handle(raw);
    journal_->Push(std::move(r));
}
std::optional<RawImuRecord> CaptureStreams::LatestImu() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
}
bool CaptureStreams::ImuFresh(std::uint64_t now) const {
    const auto r = LatestImu();
    now = std::max(now, MonotonicNowNs());
    if (!r || now < r->received_ns || now - r->received_ns > 100000000ULL) return false;
    for (const auto& a : {r->message.rpy(), r->message.gyroscope(), r->message.accelerometer()})
        for (float v : a) if (!std::isfinite(v)) return false;
    double norm = 0;
    for (float v : r->message.quaternion()) {
        if (!std::isfinite(v)) return false;
        norm += static_cast<double>(v) * v;
    }
    return norm > 0.25 && norm < 2.25;
}

void PhaseGetter::Init() {
    SetApiVersion("1.0.0.0");
    RegistApi(unitree::robot::g1::ROBOT_API_ID_LOCO_GET_PHASE);
}
int PhaseGetter::Query(std::string& raw) {
    return Call(unitree::robot::g1::ROBOT_API_ID_LOCO_GET_PHASE, std::string{}, raw);
}
CaptureGetters::CaptureGetters(std::shared_ptr<RawJournal> journal,
    std::shared_ptr<ArmStopInterlock> interlock)
    : journal_(std::move(journal)), interlock_(std::move(interlock)) {
    fsm_.SetTimeout(0.1F); fsm_.Init();
    phase_.SetTimeout(0.1F); phase_.Init();
    fsm_thread_ = std::thread([this] { FsmLoop(); });
    try { phase_thread_ = std::thread([this] { PhaseLoop(); }); }
    catch (...) { Stop(); throw; }
}
CaptureGetters::~CaptureGetters() { Stop(); }
void CaptureGetters::Stop() {
    stopping_ = true;
    wake_.notify_all();
    if (phase_thread_.joinable()) phase_thread_.join();
    if (fsm_thread_.joinable()) fsm_thread_.join();
}
bool CaptureGetters::Wait(int ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    return wake_.wait_for(lock, std::chrono::milliseconds(ms), [this] { return stopping_.load(); });
}
std::string CaptureGetters::PhaseStatus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return phase_status_;
}
void CaptureGetters::FsmLoop() noexcept {
    try {
        while (!stopping_) {
            const auto begin = MonotonicNowNs();
            int value = -1;
            const int rc = fsm_.Get(value);
            const auto end = MonotonicNowNs();
            fsm_value_ = rc == 0 ? value : -1;
            journal_->Text(CaptureEvent("fsm_reply", ",\"request_ns\":" +
                std::to_string(begin) + ",\"reply_ns\":" + std::to_string(end) +
                ",\"return_code\":" + std::to_string(rc) + ",\"fsm_id\":" + std::to_string(value)));
            if (interlock_) interlock_->ObserveMode(rc, value, begin, end);
            if (Wait(50)) return;
        }
    } catch (const std::exception& e) {
        if (interlock_) interlock_->Trip(std::string("FSM worker: ") + e.what());
        journal_->Text(CaptureEvent("fsm_exception", ",\"reason\":\"" + JsonEscape(e.what()) + "\""));
    }
}
void CaptureGetters::PhaseLoop() noexcept {
    try {
        while (!stopping_) {
            const auto begin = MonotonicNowNs();
            std::string raw;
            const int rc = phase_.Query(raw);
            const auto end = MonotonicNowNs();
            bool parsed = false;
            std::vector<float> values;
            std::string error;
            if (rc == 0) {
                try {
                    unitree::robot::g1::JsonizeDataVecFloat decoded;
                    unitree::common::FromJsonString(raw, decoded);
                    values = decoded.data;
                    parsed = !values.empty();
                    for (float v : values) parsed = parsed && std::isfinite(v);
                } catch (const std::exception& e) { error = e.what(); }
            }
            if (rc == 0 && parsed) ++phase_success_; else ++phase_failure_;
            std::ostringstream fields;
            fields << std::setprecision(17) << ",\"api_id\":7006,\"deprecated\":true"
                   << ",\"request_ns\":" << begin << ",\"reply_ns\":" << end
                   << ",\"return_code\":" << rc << ",\"parse_ok\":" << (parsed ? "true" : "false")
                   << ",\"phase_values_raw\":"; Array(fields, values);
            fields << ",\"raw_reply\":\"" << JsonEscape(raw) << "\""
                   << ",\"parse_error\":\"" << JsonEscape(error) << "\"";
            journal_->Text(CaptureEvent("phase_reply", fields.str()));
            {
                std::ostringstream status;
                status << "rc=" << rc << " values="; Array(status, values);
                std::lock_guard<std::mutex> lock(mutex_);
                phase_status_ = status.str();
            }
            // Upper bound 50 requests/s; 500 ms backoff when unavailable.
            const auto spent_ms = static_cast<int>((end - begin) / 1000000ULL);
            if (Wait(parsed && rc == 0 ? std::max(1, 20 - spent_ms) : 500)) return;
        }
    } catch (const std::exception& e) {
        ++phase_failure_;
        journal_->Text(CaptureEvent("phase_exception", ",\"reason\":\"" + JsonEscape(e.what()) + "\""));
    }
}
}  // namespace g1_commissioning
