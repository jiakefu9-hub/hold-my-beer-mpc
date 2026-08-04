#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ddq_torque_mapper/ddq_torque_mapper_c.h"
#include "right_arm_executor/right_arm_executor_c.h"
#include "right_arm_rnea/right_arm_rnea_c.h"

namespace right_arm_sim_runtime {

constexpr std::uint64_t kSharedMemoryMagic = 0x475253494d525431ULL;
constexpr std::uint32_t kProtocolVersion = 1;
constexpr std::size_t kArmDof = 5;

// 固定上限只服务进程间POD布局。启动后仍必须与scene.xml的真实维度相等。
constexpr std::size_t kMaxNq = 64;
constexpr std::size_t kMaxNv = 64;
constexpr std::size_t kMaxNu = 64;
constexpr std::size_t kMaxNbody = 64;
constexpr std::size_t kMaxXfrc = 6 * kMaxNbody;
constexpr std::size_t kErrorCapacity = 512;

enum RequestFlags : std::uint32_t {
    kRequestMappingUpdateDue = 1U << 0U,
    kRequestShutdown = 1U << 1U,
    kRequestHasPreviousExecutedTau = 1U << 2U,
};

enum ResponseFlags : std::uint32_t {
    kResponseMappingUpdated = 1U << 0U,
    kResponseCachedFeedforwardReused = 1U << 1U,
    kResponseExecutorFallbackActive = 1U << 2U,
    kResponseFinalTorqueFinite = 1U << 3U,
};

enum class RuntimeStatus : std::uint32_t {
    kOk = 0,
    kShutdown = 1,
    kInvalidRequest = 2,
    kModelDimensionMismatch = 3,
    kExecutorConfigError = 4,
    kRneaError = 5,
    kMapperError = 6,
    kExecutorError = 7,
    kNoCachedFeedforward = 8,
    kInternalError = 9,
};

struct MapperConfigPayload {
    double perturbation{0.1};
    double regularization{5.0};
    double validation_scales[DDQ_TORQUE_MAPPER_MAX_VALIDATION_SCALES]{
        1.0, 0.5, 0.25, 0.125};
    std::int32_t validation_scale_count{4};
    std::int32_t enable_second_pass{1};
    std::int32_t max_safety_rescue_passes{2};
    std::int32_t reserved{0};
    double second_pass_error_threshold{5.0};
    double max_joint_error{4.0};
    double max_abs_qacc{10.0};
};

struct SimulationRequestPayload {
    // external-step下同一个request包含完整命令和完整状态，只有Python写。
    std::uint64_t session_id{0};
    std::uint64_t request_id{0};
    std::uint64_t command_id{0};
    std::uint64_t command_source_state_id{0};
    std::uint64_t execution_state_id{0};
    std::uint64_t publish_monotonic_ns{0};
    std::uint64_t command_timestamp_ns{0};
    std::uint64_t state_timestamp_ns{0};
    std::uint32_t flags{0};
    std::uint32_t nq{0};
    std::uint32_t nv{0};
    std::uint32_t nu{0};
    std::uint32_t nbody{0};
    double simulation_time{0.0};
    double mujoco_timestep{0.002};
    double friction_breakaway_steps{5.0};

    double qpos[kMaxNq]{};
    double qvel[kMaxNv]{};
    double reference_qacc[kMaxNv]{};
    double fixed_ctrl[kMaxNu]{};
    double qacc_warmstart[kMaxNv]{};
    double qfrc_applied[kMaxNv]{};
    double xfrc_applied[kMaxXfrc]{};

    double right_arm_q[kArmDof]{};
    double right_arm_dq[kArmDof]{};
    double q_ref[kArmDof]{};
    double dq_ref[kArmDof]{};
    double ddq_des[kArmDof]{};
    double tau_passive[kArmDof]{};
    double friction_loss[kArmDof]{};
    double tau_pd[kArmDof]{};
    double previous_executed_tau[kArmDof]{};

    MapperConfigPayload mapper_config{};
    rae_config_v1 executor_config{};
};

struct SimulationResponsePayload {
    std::uint64_t session_id{0};
    std::uint64_t request_id{0};
    std::uint64_t command_id{0};
    std::uint64_t command_source_state_id{0};
    std::uint64_t execution_state_id{0};
    std::uint64_t request_publish_monotonic_ns{0};
    std::uint64_t worker_start_monotonic_ns{0};
    std::uint64_t worker_finish_monotonic_ns{0};
    std::uint64_t total_elapsed_ns{0};
    std::uint32_t status{static_cast<std::uint32_t>(RuntimeStatus::kInternalError)};
    std::uint32_t flags{0};

    RightArmRneaOutput rnea_output{};
    DdqTorqueMapperOutput mapper_output{};
    rae_output_v1 executor_output{};
    double validated_tau_ff[kArmDof]{};
    double final_tau[kArmDof]{};
    char error[kErrorCapacity]{};
};

template <typename Payload>
struct alignas(64) SeqlockSlot {
    alignas(8) std::uint64_t sequence{0};
    Payload payload{};
};

struct SharedMemoryLayout {
    std::uint64_t magic{kSharedMemoryMagic};
    std::uint32_t version{kProtocolVersion};
    std::uint32_t layout_size{0};
    SeqlockSlot<SimulationRequestPayload> request{};
    SeqlockSlot<SimulationResponsePayload> response{};
};

template <typename Payload>
inline void WriteSeqlock(SeqlockSlot<Payload>& slot, const Payload& value) noexcept {
    std::uint64_t sequence = __atomic_load_n(&slot.sequence, __ATOMIC_RELAXED);
    if ((sequence & 1U) != 0U) {
        ++sequence;
    }
    __atomic_store_n(&slot.sequence, sequence + 1U, __ATOMIC_RELEASE);
    slot.payload = value;
    __atomic_store_n(&slot.sequence, sequence + 2U, __ATOMIC_RELEASE);
}

template <typename Payload>
inline bool ReadSeqlock(
    const SeqlockSlot<Payload>& slot,
    Payload& output,
    std::size_t max_attempts = 100U) noexcept {
    for (std::size_t attempt = 0; attempt < max_attempts; ++attempt) {
        const std::uint64_t before =
            __atomic_load_n(&slot.sequence, __ATOMIC_ACQUIRE);
        if ((before & 1U) != 0U) {
            continue;
        }
        output = slot.payload;
        const std::uint64_t after =
            __atomic_load_n(&slot.sequence, __ATOMIC_ACQUIRE);
        if (before == after && (after & 1U) == 0U) {
            return true;
        }
    }
    return false;
}

static_assert(std::is_standard_layout_v<SimulationRequestPayload>);
static_assert(std::is_trivially_copyable_v<SimulationRequestPayload>);
static_assert(std::is_standard_layout_v<SimulationResponsePayload>);
static_assert(std::is_trivially_copyable_v<SimulationResponsePayload>);
static_assert(alignof(SeqlockSlot<SimulationRequestPayload>) >= 64);
static_assert(__atomic_always_lock_free(sizeof(std::uint64_t), nullptr));

}  // namespace right_arm_sim_runtime
