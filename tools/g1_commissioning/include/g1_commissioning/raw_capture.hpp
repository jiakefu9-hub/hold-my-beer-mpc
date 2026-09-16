#pragma once

#include "g1_commissioning/device_fsm_monitor.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace g1_commissioning {

struct RawImuRecord {
    std::uint64_t received_ns{0}, sequence{0};
    unitree_hg::msg::dds_::IMUState_ message;
};
struct RawLowStateRecord {
    std::uint64_t received_ns{0}, sequence{0};
    unitree_hg::msg::dds_::LowState_ message;
};

std::string RawImuJson(const RawImuRecord& record);
std::string RawLowStateJson(const RawLowStateRecord& record);
std::string CaptureEvent(const std::string& event, const std::string& fields = "");

// Copies every delivered callback to a bounded queue. Disk I/O/JSON formatting
// happen on the writer, never on the motor control thread. Overflow is counted
// and latched, never silently overwritten; this does not prove DDS delivered all
// robot samples (secondary_imu has no source timestamp/sequence in this SDK).
class RawJournal {
public:
    explicit RawJournal(const std::string& new_directory,
                        std::size_t queue_capacity = 8192);
    ~RawJournal();
    void Push(RawImuRecord record);
    void Push(RawLowStateRecord record);
    void Text(std::string json);
    bool Healthy() const { return !failed_.load() && dropped_.load() == 0; }
    void Finish();  // call after stopping all producers; drains the queue
    std::uint64_t written() const { return written_.load(); }
    std::uint64_t dropped() const { return dropped_.load(); }
    const std::string& directory() const { return directory_; }
private:
    using Record = std::variant<RawImuRecord, RawLowStateRecord, std::string>;
    void Enqueue(Record record);
    void Run() noexcept;
    const std::string directory_;
    const std::size_t capacity_;
    std::ofstream output_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Record> queue_;
    bool stopping_{false};
    std::atomic<bool> failed_{false};
    std::atomic<std::uint64_t> dropped_{0}, written_{0};
    std::thread worker_;
};

// No publishers. Read-only observer leaves interlock null so mode changes remain
// observable; output executor attaches its existing FSM/L2+B interlock.
class CaptureStreams {
public:
    CaptureStreams(std::shared_ptr<RawJournal> journal,
                   std::shared_ptr<LowStateInbox> inbox,
                   std::shared_ptr<ArmStopInterlock> interlock = nullptr);
    ~CaptureStreams();
    void Stop();
    std::optional<RawImuRecord> LatestImu() const;
    bool ImuFresh(std::uint64_t now_ns) const;
    std::uint64_t imu_count() const { return imu_count_.load(); }
    std::uint64_t low_count() const { return low_count_.load(); }
private:
    void ImuCallback(const void* raw);
    void LowCallback(const void* raw);
    std::shared_ptr<RawJournal> journal_;
    std::shared_ptr<LowStateInbox> inbox_;
    std::shared_ptr<ArmStopInterlock> interlock_;
    mutable std::mutex mutex_;
    std::optional<RawImuRecord> latest_;
    std::atomic<std::uint64_t> imu_count_{0}, low_count_{0};
    unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::IMUState_> imu_sub_;
    unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_> low_sub_;
};

class PhaseGetter final : public unitree::robot::Client {
public:
    PhaseGetter() : Client("sport", false) {}
    void Init() override;
    int Query(std::string& raw_reply);
};

// FSM and deprecated phase getter have separate workers and separate clients.
// Phase failure is data, not a gait estimate and never a control transition.
class CaptureGetters {
public:
    CaptureGetters(std::shared_ptr<RawJournal> journal,
                   std::shared_ptr<ArmStopInterlock> interlock = nullptr);
    ~CaptureGetters();
    void Stop();
    int latest_fsm() const { return fsm_value_.load(); }
    std::uint64_t phase_success_count() const { return phase_success_.load(); }
    std::uint64_t phase_failure_count() const { return phase_failure_.load(); }
    std::string PhaseStatus() const;
private:
    bool Wait(int milliseconds);
    void FsmLoop() noexcept;
    void PhaseLoop() noexcept;
    std::shared_ptr<RawJournal> journal_;
    std::shared_ptr<ArmStopInterlock> interlock_;
    FsmGetter fsm_;
    PhaseGetter phase_;
    std::atomic<bool> stopping_{false};
    std::atomic<int> fsm_value_{-1};
    std::atomic<std::uint64_t> phase_success_{0}, phase_failure_{0};
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::string phase_status_{"not queried"};
    std::thread fsm_thread_, phase_thread_;
};

}  // namespace g1_commissioning
