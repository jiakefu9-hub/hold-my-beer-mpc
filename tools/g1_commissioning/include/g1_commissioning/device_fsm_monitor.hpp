#pragma once

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include <unitree/robot/client/client.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/g1/loco/g1_loco_api.hpp>
#include <unitree/robot/go2/public/jsonize_type.hpp>

#include "g1_commissioning/arm_stop_interlock.hpp"
#include "g1_commissioning/device_state.hpp"

namespace g1_commissioning {

// Shared by executor and publisher-free observer. Bad CRC latches the inbox
// gate; never interpret remote bytes from a rejected sample.
inline void HandleArmStopState(LowStateInbox& inbox, ArmStopInterlock& interlock,
                               const void* message) {
    inbox.Handle(message);
    const auto state = inbox.Latest();
    if (message != nullptr && state && state->crc_valid) {
        const auto& remote = static_cast<const
            unitree_hg::msg::dds_::LowState_*>(message)->wireless_remote();
        interlock.ObserveRemote(remote[2], remote[3]);
    }
}

// Getter-only client: never registers or calls any mode setter.
class FsmGetter final : public unitree::robot::Client {
public:
    FsmGetter() : Client("sport", false) {}
    void Init() override {
        SetApiVersion("1.0.0.0");
        RegistApi(unitree::robot::g1::ROBOT_API_ID_LOCO_GET_FSM_ID);
    }
    int Get(int& fsm) {
        std::string data;
        const int rc = Call(unitree::robot::g1::ROBOT_API_ID_LOCO_GET_FSM_ID,
                            std::string{}, data);
        if (rc == 0) {
            unitree::robot::go2::JsonizeDataInt decoded;
            unitree::common::FromJsonString(data, decoded);
            fsm = decoded.data;
        }
        return rc;
    }
};

// RPC cannot block the command thread. SDK RPC timeout is 100 ms; wait 50 ms
// between replies and subsequent requests. A stalled worker also fails closed
// through the interlock's independent 200 ms age check in the command thread.
class FsmMonitor {
public:
    explicit FsmMonitor(std::shared_ptr<ArmStopInterlock> interlock)
        : interlock_(std::move(interlock)) {
        getter_.SetTimeout(0.1F);
        getter_.Init();
        worker_ = std::thread([this] { Run(); });
    }
    ~FsmMonitor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    FsmMonitor(const FsmMonitor&) = delete;
    FsmMonitor& operator=(const FsmMonitor&) = delete;

private:
    void Run() noexcept {
        try {
            while (true) {
                const auto requested = MonotonicNowNs();
                int fsm = -1;
                const int rc = getter_.Get(fsm);
                interlock_->ObserveMode(rc, fsm, requested, MonotonicNowNs());
                if (!interlock_->Check(MonotonicNowNs()).empty()) return;
                std::unique_lock<std::mutex> lock(mutex_);
                if (wake_.wait_for(lock, std::chrono::milliseconds(50),
                                   [this] { return stopping_; })) return;
            }
        } catch (const std::exception& error) {
            interlock_->Trip(std::string("FSM monitor exception: ") + error.what());
        } catch (...) {
            interlock_->Trip("FSM monitor unknown exception");
        }
    }
    std::shared_ptr<ArmStopInterlock> interlock_;
    FsmGetter getter_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_{false};
    std::thread worker_;
};

}  // namespace g1_commissioning
