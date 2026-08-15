#include "right_arm_sim_runtime/runtime.hpp"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace right_arm_sim_runtime {
namespace {

using Clock = std::chrono::steady_clock;

template <std::size_t Size>
bool Finite(const double (&values)[Size], std::size_t count = Size) noexcept {
    if (count > Size) {
        return false;
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

void CopyError(char* destination, const std::string& message) noexcept {
    const std::size_t count = std::min(message.size(), kErrorCapacity - 1U);
    std::memcpy(destination, message.data(), count);
    destination[count] = '\0';
}

bool ExecutorConfigEqual(
    const rae_config_v1& left, const rae_config_v1& right) noexcept {
    if (left.struct_size != right.struct_size ||
        left.abi_version != right.abi_version ||
        left.output_semantics != right.output_semantics ||
        left.command_timeout_ns != right.command_timeout_ns ||
        left.state_timeout_ns != right.state_timeout_ns) {
        return false;
    }
    const auto equal_array = [](const double* a, const double* b) {
        for (std::size_t index = 0; index < kArmDof; ++index) {
            if (a[index] != b[index]) {
                return false;
            }
        }
        return true;
    };
    return equal_array(left.kp, right.kp) &&
           equal_array(left.kd, right.kd) &&
           equal_array(left.timeout_damping, right.timeout_damping) &&
           equal_array(left.q_ref_min, right.q_ref_min) &&
           equal_array(left.q_ref_max, right.q_ref_max) &&
           equal_array(left.dq_ref_abs_max, right.dq_ref_abs_max) &&
           equal_array(left.tau_min, right.tau_min) &&
           equal_array(left.tau_max, right.tau_max);
}

}  // namespace

class SimulationRuntime::Impl {
public:
    explicit Impl(const std::string& scene_path) {
#if defined(RIGHT_ARM_SIM_RUNTIME_MUJOCO_PLUGIN_DIR)
        // Python import会自动加载mesh decoder；独立C++进程必须显式完成。
        mj_loadAllPluginLibraries(
            RIGHT_ARM_SIM_RUNTIME_MUJOCO_PLUGIN_DIR, nullptr);
#endif
        std::array<char, kErrorCapacity> error{};
        rnea_ = right_arm_rnea_create(
            scene_path.c_str(), error.data(), error.size());
        if (rnea_ == nullptr) {
            throw std::runtime_error(
                std::string("right_arm_rnea_create failed: ") + error.data());
        }
        mapper_ = ddq_torque_mapper_create(
            scene_path.c_str(), error.data(),
            static_cast<std::int32_t>(error.size()));
        if (mapper_ == nullptr) {
            right_arm_rnea_destroy(rnea_);
            rnea_ = nullptr;
            throw std::runtime_error(
                std::string("ddq_torque_mapper_create failed: ") + error.data());
        }

        dimensions_.nq = static_cast<std::uint32_t>(
            right_arm_rnea_mujoco_nq(rnea_));
        dimensions_.nv = static_cast<std::uint32_t>(
            right_arm_rnea_mujoco_nv(rnea_));
        const int mapper_nq = ddq_torque_mapper_nq(mapper_);
        const int mapper_nv = ddq_torque_mapper_nv(mapper_);
        const int mapper_nu = ddq_torque_mapper_nu(mapper_);
        const int mapper_nbody = ddq_torque_mapper_nbody(mapper_);
        if (mapper_nq <= 0 || mapper_nv <= 0 || mapper_nu <= 0 ||
            mapper_nbody <= 0 ||
            static_cast<std::uint32_t>(mapper_nq) != dimensions_.nq ||
            static_cast<std::uint32_t>(mapper_nv) != dimensions_.nv) {
            throw std::runtime_error("RNEA/mapper model dimensions disagree");
        }
        dimensions_.nu = static_cast<std::uint32_t>(mapper_nu);
        dimensions_.nbody = static_cast<std::uint32_t>(mapper_nbody);
        if (dimensions_.nq > kMaxNq || dimensions_.nv > kMaxNv ||
            dimensions_.nu > kMaxNu || dimensions_.nbody > kMaxNbody) {
            throw std::runtime_error("scene dimensions exceed protocol bounds");
        }
    }

    ~Impl() {
        if (executor_ != nullptr) {
            rae_destroy(executor_);
        }
        if (mapper_ != nullptr) {
            ddq_torque_mapper_destroy(mapper_);
        }
        if (rnea_ != nullptr) {
            right_arm_rnea_destroy(rnea_);
        }
    }

    const ModelDimensions& dimensions() const noexcept { return dimensions_; }

    bool Process(
        const SimulationRequestPayload& request,
        SimulationResponsePayload& response) noexcept {
        response = {};
        response.session_id = request.session_id;
        response.request_id = request.request_id;
        response.command_id = request.command_id;
        response.command_source_state_id = request.command_source_state_id;
        response.execution_state_id = request.execution_state_id;
        response.request_publish_monotonic_ns = request.publish_monotonic_ns;
        response.worker_start_monotonic_ns = MonotonicNowNs();

        const auto fail = [&](RuntimeStatus status, const std::string& message) {
            response.status = static_cast<std::uint32_t>(status);
            CopyError(response.error, message);
            Finish(response);
            return false;
        };

        try {
            if ((request.flags & kRequestShutdown) != 0U) {
                response.status = static_cast<std::uint32_t>(
                    RuntimeStatus::kShutdown);
                Finish(response);
                return true;
            }
            if (request.session_id == 0U || request.request_id == 0U ||
                request.command_id == 0U || request.execution_state_id == 0U) {
                return fail(RuntimeStatus::kInvalidRequest,
                            "session/request/command/state ids must be non-zero");
            }
            if (request.session_id != active_session_id_) {
                active_session_id_ = request.session_id;
                last_request_id_ = 0U;
                cache_valid_ = false;
            }
            if (request.request_id <= last_request_id_) {
                return fail(RuntimeStatus::kInvalidRequest,
                            "request_id is not strictly increasing");
            }
            last_request_id_ = request.request_id;
            if (!DimensionsMatch(request)) {
                return fail(RuntimeStatus::kModelDimensionMismatch,
                            "request dimensions do not match loaded scene");
            }
            const std::uint64_t wall_now_ns =
                response.worker_start_monotonic_ns;
            if (!std::isfinite(request.simulation_time) ||
                request.simulation_time < 0.0 ||
                request.simulation_time >
                    static_cast<double>(
                        std::numeric_limits<std::int64_t>::max()) * 1e-9) {
                return fail(RuntimeStatus::kInvalidRequest,
                            "simulation_time cannot be represented in int64 ns");
            }
            const auto simulation_now_ns = static_cast<std::int64_t>(
                std::llround(request.simulation_time * 1e9));
            if (request.publish_monotonic_ns == 0U ||
                request.publish_monotonic_ns > wall_now_ns ||
                request.command_timestamp_ns >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()) ||
                request.state_timestamp_ns >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()) ||
                request.command_timestamp_ns >
                    static_cast<std::uint64_t>(simulation_now_ns) ||
                request.state_timestamp_ns >
                    static_cast<std::uint64_t>(simulation_now_ns)) {
                return fail(RuntimeStatus::kInvalidRequest,
                            "publish wall time or virtual timestamps are invalid");
            }
            if (!RequestFinite(request)) {
                return fail(RuntimeStatus::kInvalidRequest,
                            "request contains NaN or Inf");
            }

            bool config_changed = false;
            std::string executor_error;
            if (!EnsureExecutor(
                    request.executor_config, config_changed, executor_error)) {
                return fail(RuntimeStatus::kExecutorConfigError, executor_error);
            }
            if (config_changed) {
                cache_valid_ = false;
            }

            const bool mapping_update =
                (request.flags & kRequestMappingUpdateDue) != 0U;
            if (mapping_update) {
                cache_valid_ = false;
                if (!UpdateMapping(request, response)) {
                    Finish(response);
                    return false;
                }
                response.flags |= kResponseMappingUpdated;
            } else {
                if (!cache_valid_) {
                    return fail(RuntimeStatus::kNoCachedFeedforward,
                                "mapping update is false and cache is empty");
                }
                response.rnea_output = cached_rnea_output_;
                response.mapper_output = cached_mapper_output_;
                response.flags |= kResponseCachedFeedforwardReused;
            }

            std::copy(
                cached_validated_tau_ff_.begin(),
                cached_validated_tau_ff_.end(),
                response.validated_tau_ff);

            rae_input_v1 executor_input{};
            executor_input.struct_size = sizeof(executor_input);
            executor_input.abi_version = RAE_ABI_VERSION_V1;
            executor_input.command_timestamp_ns =
                static_cast<std::int64_t>(request.command_timestamp_ns);
            executor_input.state_timestamp_ns =
                static_cast<std::int64_t>(request.state_timestamp_ns);
            for (std::size_t joint = 0; joint < kArmDof; ++joint) {
                executor_input.q[joint] = request.right_arm_q[joint];
                executor_input.dq[joint] = request.right_arm_dq[joint];
                executor_input.q_ref[joint] = request.q_ref[joint];
                executor_input.dq_ref[joint] = request.dq_ref[joint];
                executor_input.tau_ff[joint] =
                    cached_validated_tau_ff_[joint];
            }
            response.executor_output.struct_size =
                sizeof(response.executor_output);
            response.executor_output.abi_version = RAE_ABI_VERSION_V1;
            const int executor_status = rae_step_v1(
                executor_, &executor_input,
                simulation_now_ns,
                &response.executor_output);
            if (executor_status != RAE_STATUS_OK) {
                return fail(
                    RuntimeStatus::kExecutorError,
                    std::string("rae_step_v1: ") +
                        rae_status_string(executor_status));
            }
            const bool device_pd =
                request.executor_config.output_semantics ==
                RAE_OUTPUT_DEVICE_PD;
            for (std::size_t joint = 0; joint < kArmDof; ++joint) {
                response.final_tau[joint] = device_pd
                    ? response.executor_output.predicted_total_tau_limited[joint]
                    : response.executor_output.actuator_tau_ff[joint];
            }
            if (response.executor_output.executor_mode != RAE_MODE_ACTIVE) {
                response.flags |= kResponseExecutorFallbackActive;
            }
            if (Finite(response.final_tau)) {
                response.flags |= kResponseFinalTorqueFinite;
            } else {
                return fail(RuntimeStatus::kExecutorError,
                            "executor returned non-finite final torque");
            }
            response.status = static_cast<std::uint32_t>(RuntimeStatus::kOk);
            Finish(response);
            return true;
        } catch (const std::exception& error) {
            return fail(RuntimeStatus::kInternalError, error.what());
        } catch (...) {
            return fail(RuntimeStatus::kInternalError, "unknown exception");
        }
    }

private:
    static void Finish(SimulationResponsePayload& response) noexcept {
        response.worker_finish_monotonic_ns = MonotonicNowNs();
        response.total_elapsed_ns =
            response.worker_finish_monotonic_ns -
            response.worker_start_monotonic_ns;
    }

    bool DimensionsMatch(const SimulationRequestPayload& request) const noexcept {
        return request.nq == dimensions_.nq &&
               request.nv == dimensions_.nv &&
               request.nu == dimensions_.nu &&
               request.nbody == dimensions_.nbody;
    }

    bool RequestFinite(const SimulationRequestPayload& request) const noexcept {
        const auto& mapper = request.mapper_config;
        if (!std::isfinite(request.simulation_time) ||
            !std::isfinite(request.mujoco_timestep) ||
            request.mujoco_timestep <= 0.0 ||
            !std::isfinite(request.friction_breakaway_steps) ||
            request.friction_breakaway_steps < 0.0 ||
            !Finite(request.qpos, request.nq) ||
            !Finite(request.qvel, request.nv) ||
            !Finite(request.reference_qacc, request.nv) ||
            !Finite(request.fixed_ctrl, request.nu) ||
            !Finite(request.qacc_warmstart, request.nv) ||
            !Finite(request.qfrc_applied, request.nv) ||
            !Finite(request.xfrc_applied, 6U * request.nbody) ||
            !Finite(request.right_arm_q) || !Finite(request.right_arm_dq) ||
            !Finite(request.q_ref) || !Finite(request.dq_ref) ||
            !Finite(request.ddq_des) || !Finite(request.tau_passive) ||
            !Finite(request.friction_loss) || !Finite(request.tau_pd) ||
            !Finite(request.previous_executed_tau) ||
            !std::isfinite(mapper.perturbation) ||
            !std::isfinite(mapper.regularization) ||
            !Finite(mapper.validation_scales) ||
            !std::isfinite(mapper.second_pass_error_threshold) ||
            !std::isfinite(mapper.max_joint_error) ||
            !std::isfinite(mapper.max_abs_qacc)) {
            return false;
        }
        for (double friction : request.friction_loss) {
            if (friction < 0.0) {
                return false;
            }
        }
        return true;
    }

    bool EnsureExecutor(
        const rae_config_v1& requested,
        bool& changed,
        std::string& error) noexcept {
        changed = executor_ == nullptr ||
                  !ExecutorConfigEqual(requested, executor_config_);
        if (!changed) {
            return true;
        }
        if (requested.struct_size != sizeof(rae_config_v1) ||
            requested.abi_version != RAE_ABI_VERSION_V1) {
            error = "executor config ABI mismatch";
            return false;
        }
        rae_executor_handle* replacement = nullptr;
        const int status = rae_create_v1(&requested, &replacement);
        if (status != RAE_STATUS_OK || replacement == nullptr) {
            error = std::string("rae_create_v1: ") + rae_status_string(status);
            return false;
        }
        if (executor_ != nullptr) {
            rae_destroy(executor_);
        }
        executor_ = replacement;
        executor_config_ = requested;
        return true;
    }

    bool UpdateMapping(
        const SimulationRequestPayload& request,
        SimulationResponsePayload& response) noexcept {
        std::array<char, kErrorCapacity> error{};
        const RightArmRneaStatus rnea_status = right_arm_rnea_compute(
            rnea_,
            request.qpos, request.nq,
            request.qvel, request.nv,
            request.reference_qacc, request.nv,
            request.ddq_des, kArmDof,
            request.tau_passive, kArmDof,
            request.friction_loss, kArmDof,
            request.mujoco_timestep,
            request.friction_breakaway_steps,
            &response.rnea_output,
            error.data(), error.size());
        if (rnea_status != RIGHT_ARM_RNEA_OK) {
            response.status = static_cast<std::uint32_t>(
                RuntimeStatus::kRneaError);
            CopyError(
                response.error,
                std::string("right_arm_rnea_compute: ") + error.data());
            return false;
        }

        DdqTorqueMapperState mapper_state{};
        mapper_state.time = request.simulation_time;
        mapper_state.qpos = request.qpos;
        mapper_state.qpos_count = static_cast<std::int32_t>(request.nq);
        mapper_state.qvel = request.qvel;
        mapper_state.qvel_count = static_cast<std::int32_t>(request.nv);
        mapper_state.ctrl = request.fixed_ctrl;
        mapper_state.ctrl_count = static_cast<std::int32_t>(request.nu);
        mapper_state.qacc_warmstart = request.qacc_warmstart;
        mapper_state.qacc_warmstart_count =
            static_cast<std::int32_t>(request.nv);
        mapper_state.qfrc_applied = request.qfrc_applied;
        mapper_state.qfrc_applied_count =
            static_cast<std::int32_t>(request.nv);
        mapper_state.xfrc_applied = request.xfrc_applied;
        mapper_state.xfrc_applied_count =
            static_cast<std::int32_t>(6U * request.nbody);

        DdqTorqueMapperRequest mapper_request{};
        for (std::size_t joint = 0; joint < kArmDof; ++joint) {
            mapper_request.desired_qacc[joint] = request.ddq_des[joint];
            mapper_request.tau_nominal[joint] = std::clamp(
                response.rnea_output.tau_ff[joint] + request.tau_pd[joint],
                request.executor_config.tau_min[joint],
                request.executor_config.tau_max[joint]);
            mapper_request.safe_hold_tau[joint] = request.tau_pd[joint];
            mapper_request.previous_executed_tau[joint] =
                request.previous_executed_tau[joint];
        }
        mapper_request.has_previous_executed_tau =
            (request.flags & kRequestHasPreviousExecutedTau) != 0U;

        DdqTorqueMapperParams mapper_params{};
        mapper_params.perturbation = request.mapper_config.perturbation;
        mapper_params.regularization = request.mapper_config.regularization;
        std::copy(
            std::begin(request.mapper_config.validation_scales),
            std::end(request.mapper_config.validation_scales),
            mapper_params.validation_scales);
        mapper_params.validation_scale_count =
            request.mapper_config.validation_scale_count;
        mapper_params.second_pass_error_threshold =
            request.mapper_config.second_pass_error_threshold;
        mapper_params.max_joint_error =
            request.mapper_config.max_joint_error;
        mapper_params.max_abs_qacc = request.mapper_config.max_abs_qacc;
        mapper_params.enable_second_pass =
            request.mapper_config.enable_second_pass;
        mapper_params.max_safety_rescue_passes =
            request.mapper_config.max_safety_rescue_passes;

        error.fill(0);
        const int mapper_status = ddq_torque_mapper_compute(
            mapper_, &mapper_state, &mapper_request, &mapper_params,
            &response.mapper_output, error.data(),
            static_cast<std::int32_t>(error.size()));
        if (mapper_status != DDQ_TORQUE_MAPPER_OK) {
            response.status = static_cast<std::uint32_t>(
                mapper_status == DDQ_TORQUE_MAPPER_NO_SAFE_TORQUE
                    ? RuntimeStatus::kNoSafeTorque
                    : RuntimeStatus::kMapperError);
            CopyError(
                response.error,
                std::string("ddq_torque_mapper_compute: ") + error.data());
            return false;
        }
        if (response.mapper_output.final_output_certified == 0 ||
            response.mapper_output.no_safe_torque != 0) {
            response.status = static_cast<std::uint32_t>(
                RuntimeStatus::kNoSafeTorque);
            CopyError(response.error,
                      "mapper returned no certified final output");
            return false;
        }
        for (std::size_t joint = 0; joint < kArmDof; ++joint) {
            cached_validated_tau_ff_[joint] =
                response.mapper_output.tau_cmd[joint] - request.tau_pd[joint];
        }
        cached_rnea_output_ = response.rnea_output;
        cached_mapper_output_ = response.mapper_output;
        cache_valid_ = true;
        return true;
    }

    ModelDimensions dimensions_{};
    RightArmRneaHandle* rnea_{nullptr};
    DdqTorqueMapperHandle* mapper_{nullptr};
    rae_executor_handle* executor_{nullptr};
    rae_config_v1 executor_config_{};
    std::uint64_t active_session_id_{0};
    std::uint64_t last_request_id_{0};
    bool cache_valid_{false};
    std::array<double, kArmDof> cached_validated_tau_ff_{};
    RightArmRneaOutput cached_rnea_output_{};
    DdqTorqueMapperOutput cached_mapper_output_{};
};

SimulationRuntime::SimulationRuntime(const std::string& scene_path)
    : impl_(std::make_unique<Impl>(scene_path)) {}

SimulationRuntime::~SimulationRuntime() = default;

const ModelDimensions& SimulationRuntime::dimensions() const noexcept {
    return impl_->dimensions();
}

bool SimulationRuntime::Process(
    const SimulationRequestPayload& request,
    SimulationResponsePayload& response) noexcept {
    return impl_->Process(request, response);
}

std::uint64_t MonotonicNowNs() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
}

const char* RuntimeStatusString(RuntimeStatus status) noexcept {
    switch (status) {
        case RuntimeStatus::kOk: return "ok";
        case RuntimeStatus::kShutdown: return "shutdown";
        case RuntimeStatus::kInvalidRequest: return "invalid_request";
        case RuntimeStatus::kModelDimensionMismatch:
            return "model_dimension_mismatch";
        case RuntimeStatus::kExecutorConfigError:
            return "executor_config_error";
        case RuntimeStatus::kRneaError: return "rnea_error";
        case RuntimeStatus::kMapperError: return "mapper_error";
        case RuntimeStatus::kExecutorError: return "executor_error";
        case RuntimeStatus::kNoCachedFeedforward:
            return "no_cached_feedforward";
        case RuntimeStatus::kInternalError: return "internal_error";
        case RuntimeStatus::kNoSafeTorque: return "NO_SAFE_TORQUE";
    }
    return "unknown";
}

}  // namespace right_arm_sim_runtime
