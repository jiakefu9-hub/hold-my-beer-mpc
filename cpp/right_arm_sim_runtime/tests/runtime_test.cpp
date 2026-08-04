#include <mujoco/mujoco.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include "right_arm_sim_runtime/protocol.hpp"
#include "right_arm_sim_runtime/runtime.hpp"
#include "right_arm_sim_runtime/shared_memory.hpp"

namespace rsr = right_arm_sim_runtime;

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __FILE__ << ':' << __LINE__                            \
                      << " CHECK failed: " #condition << '\n';                 \
            ++failures;                                                        \
        }                                                                       \
    } while (false)

struct ModelDeleter {
    void operator()(mjModel* value) const noexcept {
        if (value != nullptr) mj_deleteModel(value);
    }
};
struct DataDeleter {
    void operator()(mjData* value) const noexcept {
        if (value != nullptr) mj_deleteData(value);
    }
};

using ModelPtr = std::unique_ptr<mjModel, ModelDeleter>;
using DataPtr = std::unique_ptr<mjData, DataDeleter>;

struct Fixture {
    ModelPtr model;
    DataPtr data;
    rsr::SimulationRequestPayload request;
};

Fixture MakeFixture(const std::string& scene_path) {
    char error[1024]{};
    ModelPtr model(mj_loadXML(scene_path.c_str(), nullptr, error, sizeof(error)));
    if (!model) {
        throw std::runtime_error(std::string("mj_loadXML: ") + error);
    }
    DataPtr data(mj_makeData(model.get()));
    if (!data) {
        throw std::runtime_error("mj_makeData failed");
    }
    mj_forward(model.get(), data.get());

    rsr::SimulationRequestPayload request;
    request.session_id = 42;
    request.request_id = 1;
    request.command_id = 11;
    request.command_source_state_id = 100;
    request.execution_state_id = 101;
    const std::uint64_t now = rsr::MonotonicNowNs();
    request.publish_monotonic_ns = now;
    // Executor超时使用MuJoCo虚拟时间；第0拍的0 ns必须合法。
    request.command_timestamp_ns = 0;
    request.state_timestamp_ns = 0;
    request.flags = rsr::kRequestMappingUpdateDue;
    request.nq = static_cast<std::uint32_t>(model->nq);
    request.nv = static_cast<std::uint32_t>(model->nv);
    request.nu = static_cast<std::uint32_t>(model->nu);
    request.nbody = static_cast<std::uint32_t>(model->nbody);
    request.simulation_time = data->time;
    request.mujoco_timestep = model->opt.timestep;
    request.friction_breakaway_steps = 5.0;
    std::copy(data->qpos, data->qpos + model->nq, request.qpos);
    std::copy(data->qvel, data->qvel + model->nv, request.qvel);
    std::copy(data->qacc, data->qacc + model->nv, request.reference_qacc);
    std::copy(data->ctrl, data->ctrl + model->nu, request.fixed_ctrl);
    std::copy(
        data->qacc_warmstart,
        data->qacc_warmstart + model->nv,
        request.qacc_warmstart);
    std::copy(
        data->qfrc_applied,
        data->qfrc_applied + model->nv,
        request.qfrc_applied);
    std::copy(
        data->xfrc_applied,
        data->xfrc_applied + 6 * model->nbody,
        request.xfrc_applied);

    constexpr const char* names[rsr::kArmDof] = {
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    };
    for (std::size_t joint = 0; joint < rsr::kArmDof; ++joint) {
        const int id = mj_name2id(model.get(), mjOBJ_JOINT, names[joint]);
        if (id < 0) {
            throw std::runtime_error(std::string("missing joint: ") + names[joint]);
        }
        const int qpos_index = model->jnt_qposadr[id];
        const int qvel_index = model->jnt_dofadr[id];
        request.right_arm_q[joint] = data->qpos[qpos_index];
        request.right_arm_dq[joint] = data->qvel[qvel_index];
        request.q_ref[joint] = request.right_arm_q[joint];
        request.dq_ref[joint] = 0.0;
        request.ddq_des[joint] = 0.1 * static_cast<double>(joint + 1U);
        request.tau_passive[joint] = data->qfrc_passive[qvel_index];
        request.friction_loss[joint] = model->dof_frictionloss[qvel_index];
        request.tau_pd[joint] = 0.0;
    }
    request.mapper_config = {};
    const int config_status = rae_get_default_config_v1(
        RAE_OUTPUT_HOST_FULL_TORQUE, &request.executor_config);
    if (config_status != RAE_STATUS_OK) {
        throw std::runtime_error("rae_get_default_config_v1 failed");
    }
    return {std::move(model), std::move(data), request};
}

void TestSeqlock() {
    rsr::SeqlockSlot<rsr::SimulationRequestPayload> slot;
    rsr::SimulationRequestPayload request;
    request.session_id = 123;
    request.request_id = 456;
    request.qpos[17] = 8.5;
    rsr::WriteSeqlock(slot, request);
    rsr::SimulationRequestPayload copy;
    CHECK(rsr::ReadSeqlock(slot, copy));
    CHECK(copy.session_id == 123);
    CHECK(copy.request_id == 456);
    CHECK(copy.qpos[17] == 8.5);
    __atomic_store_n(&slot.sequence, 3ULL, __ATOMIC_RELEASE);
    CHECK(!rsr::ReadSeqlock(slot, copy, 3));
}

void TestCore(const std::string& scene_path) {
    rsr::SimulationRuntime runtime(scene_path);
    auto fixture = MakeFixture(scene_path);
    const auto& dimensions = runtime.dimensions();
    CHECK(dimensions.nq == fixture.request.nq);
    CHECK(dimensions.nv == fixture.request.nv);
    CHECK(dimensions.nu == fixture.request.nu);
    CHECK(dimensions.nbody == fixture.request.nbody);

    rsr::SimulationResponsePayload response;
    CHECK(runtime.Process(fixture.request, response));
    CHECK(response.status == static_cast<std::uint32_t>(rsr::RuntimeStatus::kOk));
    CHECK(response.request_id == fixture.request.request_id);
    CHECK((response.flags & rsr::kResponseMappingUpdated) != 0U);
    CHECK((response.flags & rsr::kResponseFinalTorqueFinite) != 0U);
    for (std::size_t joint = 0; joint < rsr::kArmDof; ++joint) {
        CHECK(std::isfinite(response.final_tau[joint]));
        CHECK(std::abs(
            response.validated_tau_ff[joint] -
            (response.mapper_output.tau_cmd[joint] -
             fixture.request.tau_pd[joint])) < 1e-12);
    }

    auto reuse = fixture.request;
    reuse.request_id = 2;
    reuse.command_id = 12;
    reuse.execution_state_id = 102;
    reuse.flags = 0;
    reuse.publish_monotonic_ns = rsr::MonotonicNowNs();
    reuse.command_timestamp_ns = 0;
    reuse.state_timestamp_ns = 0;
    CHECK(runtime.Process(reuse, response));
    CHECK((response.flags & rsr::kResponseCachedFeedforwardReused) != 0U);
    CHECK((response.flags & rsr::kResponseMappingUpdated) == 0U);

    auto changed = reuse;
    changed.request_id = 3;
    changed.command_id = 13;
    changed.execution_state_id = 103;
    changed.executor_config.kp[0] += 1.0;
    changed.publish_monotonic_ns = rsr::MonotonicNowNs();
    changed.command_timestamp_ns = 0;
    changed.state_timestamp_ns = 0;
    CHECK(!runtime.Process(changed, response));
    CHECK(response.status == static_cast<std::uint32_t>(
        rsr::RuntimeStatus::kNoCachedFeedforward));

    changed.request_id = 4;
    changed.command_id = 14;
    changed.execution_state_id = 104;
    changed.flags = rsr::kRequestMappingUpdateDue;
    changed.publish_monotonic_ns = rsr::MonotonicNowNs();
    changed.command_timestamp_ns = 0;
    changed.state_timestamp_ns = 0;
    CHECK(runtime.Process(changed, response));

    auto mismatch = changed;
    mismatch.request_id = 5;
    mismatch.nq -= 1;
    mismatch.publish_monotonic_ns = rsr::MonotonicNowNs();
    mismatch.command_timestamp_ns = 0;
    mismatch.state_timestamp_ns = 0;
    CHECK(!runtime.Process(mismatch, response));
    CHECK(response.status == static_cast<std::uint32_t>(
        rsr::RuntimeStatus::kModelDimensionMismatch));

    auto invalid = changed;
    invalid.request_id = 6;
    invalid.qpos[0] = std::numeric_limits<double>::quiet_NaN();
    invalid.publish_monotonic_ns = rsr::MonotonicNowNs();
    invalid.command_timestamp_ns = 0;
    invalid.state_timestamp_ns = 0;
    CHECK(!runtime.Process(invalid, response));
    CHECK(response.status == static_cast<std::uint32_t>(
        rsr::RuntimeStatus::kInvalidRequest));
}

void Notify(int descriptor) {
    const unsigned char byte = 1;
    if (::write(descriptor, &byte, 1) != 1) {
        throw std::runtime_error("pipe write failed");
    }
}

void WaitNotification(int descriptor) {
    unsigned char byte = 0;
    if (::read(descriptor, &byte, 1) != 1) {
        throw std::runtime_error("pipe read failed");
    }
}

void TestExternalStep(
    const std::string& worker_path, const std::string& scene_path) {
    int request_pipe[2]{};
    int response_pipe[2]{};
    if (::pipe(request_pipe) != 0 || ::pipe(response_pipe) != 0) {
        throw std::runtime_error("pipe failed");
    }
    const std::string name =
        "/right_arm_sim_runtime_test_" + std::to_string(::getpid());
    rsr::SharedMemoryRegion::Unlink(name);
    const pid_t child = ::fork();
    if (child < 0) {
        throw std::runtime_error("fork failed");
    }
    if (child == 0) {
        ::close(request_pipe[1]);
        ::close(response_pipe[0]);
        const std::string request_fd = std::to_string(request_pipe[0]);
        const std::string response_fd = std::to_string(response_pipe[1]);
        ::execl(
            worker_path.c_str(), worker_path.c_str(),
            "--scene", scene_path.c_str(),
            "--shm-name", name.c_str(),
            "--request-fd", request_fd.c_str(),
            "--response-fd", response_fd.c_str(),
            "--unlink-on-exit",
            static_cast<char*>(nullptr));
        _exit(127);
    }
    ::close(request_pipe[0]);
    ::close(response_pipe[1]);

    std::unique_ptr<rsr::SharedMemoryRegion> region;
    for (int attempt = 0; attempt < 500 && !region; ++attempt) {
        try {
            region = std::make_unique<rsr::SharedMemoryRegion>(
                rsr::SharedMemoryRegion::Open(name, false));
        } catch (const std::exception&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    if (!region) {
        ::kill(child, SIGKILL);
        throw std::runtime_error("worker did not create shared memory");
    }

    auto fixture = MakeFixture(scene_path);
    fixture.request.session_id = 99;
    fixture.request.request_id = 1001;
    fixture.request.command_id = 2001;
    fixture.request.publish_monotonic_ns = rsr::MonotonicNowNs();
    fixture.request.command_timestamp_ns = 0;
    fixture.request.state_timestamp_ns = 0;
    rsr::WriteSeqlock(region->get()->request, fixture.request);
    Notify(request_pipe[1]);
    WaitNotification(response_pipe[0]);
    rsr::SimulationResponsePayload response;
    CHECK(rsr::ReadSeqlock(region->get()->response, response));
    CHECK(response.session_id == 99);
    CHECK(response.request_id == 1001);
    CHECK(response.status == static_cast<std::uint32_t>(rsr::RuntimeStatus::kOk));
    CHECK((response.flags & rsr::kResponseFinalTorqueFinite) != 0U);

    rsr::SimulationRequestPayload shutdown;
    shutdown.session_id = 99;
    shutdown.request_id = 1002;
    shutdown.flags = rsr::kRequestShutdown;
    rsr::WriteSeqlock(region->get()->request, shutdown);
    Notify(request_pipe[1]);
    WaitNotification(response_pipe[0]);
    CHECK(rsr::ReadSeqlock(region->get()->response, response));
    CHECK(response.status == static_cast<std::uint32_t>(
        rsr::RuntimeStatus::kShutdown));
    ::close(request_pipe[1]);
    ::close(response_pipe[0]);
    int status = 0;
    CHECK(::waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);
    region.reset();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: runtime_test WORKER SCENE\n";
        return 2;
    }
    try {
        TestSeqlock();
        TestCore(argv[2]);
        TestExternalStep(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "uncaught test error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " checks failed\n";
        return 1;
    }
    std::cout << "right_arm_sim_runtime tests passed\n";
    return 0;
}
