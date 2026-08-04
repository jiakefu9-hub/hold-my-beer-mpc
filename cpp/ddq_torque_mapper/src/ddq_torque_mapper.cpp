#include "ddq_torque_mapper/ddq_torque_mapper_c.h"

#include <mujoco/mujoco.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kArmDof = DDQ_TORQUE_MAPPER_ARM_DOF;
constexpr int kAbiVersion = 1;

using Clock = std::chrono::steady_clock;
using Vector5 = Eigen::Matrix<double, kArmDof, 1>;
using Matrix5 = Eigen::Matrix<double, kArmDof, kArmDof, Eigen::RowMajor>;

constexpr std::array<const char*, kArmDof> kRightArmJointNames = {
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
};

uint64_t elapsed_ns(const Clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

void set_error(char* buffer, const int32_t capacity, const std::string& message) {
  if (buffer == nullptr || capacity <= 0) {
    return;
  }
  if (message.empty()) {
    buffer[0] = '\0';
    return;
  }
  std::snprintf(buffer, static_cast<std::size_t>(capacity), "%s", message.c_str());
}

bool all_finite(const double* values, const int count) {
  if (values == nullptr || count < 0) {
    return false;
  }
  for (int index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

double max_abs(const Vector5& values) {
  return values.cwiseAbs().maxCoeff();
}

Vector5 load_vector5(const double values[kArmDof]) {
  return Eigen::Map<const Vector5>(values);
}

void store_vector5(const Vector5& values, double output[kArmDof]) {
  Eigen::Map<Vector5> output_map(output);
  output_map = values;
}

void store_matrix5(const Matrix5& values, double output[kArmDof * kArmDof]) {
  Eigen::Map<Matrix5> output_map(output);
  output_map = values;
}

Vector5 clip_vector(const Vector5& values,
                    const Vector5& lower,
                    const Vector5& upper) {
  return values.cwiseMax(lower).cwiseMin(upper);
}

struct MjModelDeleter {
  void operator()(mjModel* model) const {
    if (model != nullptr) {
      mj_deleteModel(model);
    }
  }
};

struct MjDataDeleter {
  void operator()(mjData* data) const {
    if (data != nullptr) {
      mj_deleteData(data);
    }
  }
};

struct PassResult {
  Vector5 tau_cmd = Vector5::Zero();
  Vector5 tau_cmd_raw = Vector5::Zero();
  Vector5 correction_raw = Vector5::Zero();
  Vector5 correction = Vector5::Zero();
  Vector5 qacc_predicted = Vector5::Zero();
  Vector5 qacc_validated = Vector5::Zero();
  Vector5 qacc_validation_error = Vector5::Zero();
  Vector5 qacc_linearization_error = Vector5::Zero();
  Matrix5 gain_matrix = Matrix5::Zero();
  Vector5 singular_values = Vector5::Zero();
  double condition_number = std::numeric_limits<double>::infinity();
  double validation_scale = 0.0;
  int validation_attempts = 0;
  bool improved = false;
  bool tracking_safety_satisfied = false;
  bool qacc_safety_satisfied = false;
  int safe_candidate_count = 0;
  int total_error_rejections = 0;
  int joint_error_rejections = 0;
  int qacc_limit_rejections = 0;
  // 下一轮 forwardSkip 使用本轮已验收工作点的完整 qacc 作为约束求解
  // warm-start；右臂 5 维输出仍由 qacc_validated 单独保存。
  std::vector<double> qacc_validated_full;
};

struct Candidate {
  double scale = 0.0;
  Vector5 tau_raw = Vector5::Zero();
  Vector5 tau = Vector5::Zero();
  bool predicted_safe = false;
  double predicted_error_norm = 0.0;
};

}  // namespace

struct DdqTorqueMapperHandle {
  std::unique_ptr<mjModel, MjModelDeleter> model;
  std::unique_ptr<mjData, MjDataDeleter> scratch;
  std::array<int, kArmDof> qvel_indices{};
  std::array<int, kArmDof> ctrl_indices{};
  Vector5 torque_lower = Vector5::Zero();
  Vector5 torque_upper = Vector5::Zero();

  // 【非核心缓存】避免每拍为 warm-start 重新分配内存。
  std::vector<double> warmstart;

  DdqTorqueMapperHandle(mjModel* raw_model, mjData* raw_data)
      : model(raw_model),
        scratch(raw_data),
        warmstart(static_cast<std::size_t>(raw_model->nv), 0.0) {}
};

namespace {

void validate_current_model(DdqTorqueMapperHandle& handle) {
  const mjModel* model = handle.model.get();
  // 当前项目 scene 的这些维度均为 0。若以后模型增加激活态、mocap、
  // equality 开关或插件状态，应先扩展 C ABI，不能静默沿用陈旧 scratch。
  if (model->na != 0 || model->nmocap != 0 || model->neq != 0 ||
      model->nuserdata != 0 || model->npluginstate != 0) {
    throw std::runtime_error(
        "当前窄 C ABI 只支持本项目 na=nmocap=neq=nuserdata=npluginstate=0 "
        "的 scene；模型已变化，请先扩展状态输入。");
  }

  for (int arm_index = 0; arm_index < kArmDof; ++arm_index) {
    const char* name = kRightArmJointNames[arm_index];
    const int joint_id = mj_name2id(model, mjOBJ_JOINT, name);
    const int actuator_id = mj_name2id(model, mjOBJ_ACTUATOR, name);
    if (joint_id < 0 || actuator_id < 0) {
      throw std::runtime_error(std::string("找不到右臂 joint/actuator: ") + name);
    }
    if (model->jnt_type[joint_id] != mjJNT_HINGE) {
      throw std::runtime_error(std::string("右臂关节不是单自由度 hinge: ") + name);
    }
    if (model->actuator_trnid[2 * actuator_id] != joint_id ||
        std::abs(model->actuator_gear[6 * actuator_id] - 1.0) > 1e-12) {
      throw std::runtime_error(std::string("右臂执行器不是 gear=1 direct-drive: ") +
                               name);
    }
    const double lower = model->jnt_actfrcrange[2 * joint_id];
    const double upper = model->jnt_actfrcrange[2 * joint_id + 1];
    if (!(std::isfinite(lower) && std::isfinite(upper) && lower < upper)) {
      throw std::runtime_error(std::string("右臂关节力矩范围无效: ") + name);
    }
    handle.qvel_indices[arm_index] = model->jnt_dofadr[joint_id];
    handle.ctrl_indices[arm_index] = actuator_id;
    handle.torque_lower[arm_index] = lower;
    handle.torque_upper[arm_index] = upper;
  }
}

int validate_inputs(const DdqTorqueMapperHandle& handle,
                    const DdqTorqueMapperState* state,
                    const DdqTorqueMapperRequest* request,
                    const DdqTorqueMapperParams* params,
                    const DdqTorqueMapperOutput* output,
                    std::string& error) {
  if (state == nullptr || request == nullptr || params == nullptr || output == nullptr) {
    error = "state/request/params/output 不能为空。";
    return DDQ_TORQUE_MAPPER_INVALID_ARGUMENT;
  }
  const mjModel* model = handle.model.get();
  if (state->qpos_count != model->nq || state->qvel_count != model->nv ||
      state->ctrl_count != model->nu ||
      state->qacc_warmstart_count != model->nv ||
      state->qfrc_applied_count != model->nv ||
      state->xfrc_applied_count != 6 * model->nbody) {
    error = "状态数组维度与加载的 MuJoCo scene 不一致。";
    return DDQ_TORQUE_MAPPER_DIMENSION_MISMATCH;
  }
  if (!std::isfinite(state->time) || !all_finite(state->qpos, model->nq) ||
      !all_finite(state->qvel, model->nv) ||
      !all_finite(state->ctrl, model->nu) ||
      !all_finite(state->qacc_warmstart, model->nv) ||
      !all_finite(state->qfrc_applied, model->nv) ||
      !all_finite(state->xfrc_applied, 6 * model->nbody) ||
      !all_finite(request->desired_qacc, kArmDof) ||
      !all_finite(request->tau_nominal, kArmDof) ||
      (request->has_previous_executed_tau != 0 &&
       !all_finite(request->previous_executed_tau, kArmDof))) {
    error = "输入包含空指针、NaN 或 Inf。";
    return DDQ_TORQUE_MAPPER_INVALID_ARGUMENT;
  }
  if (!(std::isfinite(params->perturbation) && params->perturbation > 0.0) ||
      !(std::isfinite(params->regularization) && params->regularization >= 0.0) ||
      !(std::isfinite(params->second_pass_error_threshold) &&
        params->second_pass_error_threshold >= 0.0) ||
      !(std::isfinite(params->max_joint_error) && params->max_joint_error > 0.0) ||
      !(std::isfinite(params->max_abs_qacc) && params->max_abs_qacc > 0.0) ||
      params->validation_scale_count <= 0 ||
      params->validation_scale_count > DDQ_TORQUE_MAPPER_MAX_VALIDATION_SCALES ||
      params->max_safety_rescue_passes < 0) {
    error = "映射参数范围无效。";
    return DDQ_TORQUE_MAPPER_INVALID_ARGUMENT;
  }
  for (int index = 0; index < params->validation_scale_count; ++index) {
    const double scale = params->validation_scales[index];
    if (!(std::isfinite(scale) && scale > 0.0 && scale <= 1.0)) {
      error = "validation_scales 必须全部位于 (0, 1]。";
      return DDQ_TORQUE_MAPPER_INVALID_ARGUMENT;
    }
  }
  return DDQ_TORQUE_MAPPER_OK;
}

void copy_state_inputs(DdqTorqueMapperHandle& handle,
                       const DdqTorqueMapperState& state) {
  mjModel* model = handle.model.get();
  mjData* data = handle.scratch.get();
  data->time = state.time;
  mju_copy(data->qpos, state.qpos, model->nq);
  mju_copy(data->qvel, state.qvel, model->nv);
  mju_copy(data->qacc_warmstart, state.qacc_warmstart, model->nv);
  mju_copy(data->ctrl, state.ctrl, model->nu);
  mju_copy(data->qfrc_applied, state.qfrc_applied, model->nv);
  mju_copy(data->xfrc_applied, state.xfrc_applied, 6 * model->nbody);
  std::copy(state.qacc_warmstart,
            state.qacc_warmstart + model->nv,
            handle.warmstart.begin());
}

void prepare_ctrl(DdqTorqueMapperHandle& handle,
                  const Vector5& arm_tau,
                  const std::vector<double>& warmstart) {
  mjModel* model = handle.model.get();
  mjData* data = handle.scratch.get();
  if (warmstart.size() != static_cast<std::size_t>(model->nv)) {
    throw std::runtime_error("forwardSkip warm-start 维度与模型不一致");
  }
  // copy_state_inputs 已安装本拍完整 ctrl，MuJoCo forward/forwardSkip 不会
  // 改写 ctrl；各次试算只覆盖五个右臂执行器，避免重复复制完整 nu 向量。
  for (int index = 0; index < kArmDof; ++index) {
    data->ctrl[handle.ctrl_indices[index]] = arm_tau[index];
  }
  mju_copy(data->qacc_warmstart, warmstart.data(), model->nv);
}

void copy_full_qacc(const DdqTorqueMapperHandle& handle,
                    std::vector<double>& output) {
  const mjModel* model = handle.model.get();
  output.resize(static_cast<std::size_t>(model->nv));
  mju_copy(output.data(), handle.scratch->qacc, model->nv);
}

Vector5 read_right_arm_qacc(const DdqTorqueMapperHandle& handle) {
  Vector5 qacc;
  for (int index = 0; index < kArmDof; ++index) {
    qacc[index] = handle.scratch->qacc[handle.qvel_indices[index]];
  }
  return qacc;
}

PassResult solve_validated_pass(DdqTorqueMapperHandle& handle,
                                const Vector5& desired_qacc,
                                const Vector5& base_tau,
                                const Vector5& base_qacc,
                                const std::vector<double>& base_warmstart,
                                const DdqTorqueMapperParams& params,
                                int& forward_skip_calls,
                                int& validated_pass_count) {
  ++validated_pass_count;
  PassResult result;
  result.qacc_validated_full = base_warmstart;
  mjModel* model = handle.model.get();
  mjData* data = handle.scratch.get();

  // 【核心代码】在同一个 qpos/qvel 工作点对五个右臂力矩分别扰动。
  // 基线完整 forward 已生成位置和速度阶段缓存，因此这里只重算加速度阶段。
  for (int column = 0; column < kArmDof; ++column) {
    double signed_perturbation = params.perturbation;
    if (base_tau[column] + signed_perturbation > handle.torque_upper[column]) {
      signed_perturbation = -signed_perturbation;
    }
    Vector5 perturbed_tau = base_tau;
    perturbed_tau[column] += signed_perturbation;
    prepare_ctrl(handle, perturbed_tau, base_warmstart);
    mj_forwardSkip(model, data, mjSTAGE_VEL, 1);
    ++forward_skip_calls;
    result.gain_matrix.col(column) =
        (read_right_arm_qacc(handle) - base_qacc) / signed_perturbation;
  }

  // 【核心代码】与 numpy.linalg.svd 的阻尼伪逆公式相同。
  Eigen::JacobiSVD<Matrix5> svd(result.gain_matrix,
                                Eigen::ComputeFullU | Eigen::ComputeFullV);
  result.singular_values = svd.singularValues();
  const Vector5 acceleration_error = desired_qacc - base_qacc;
  Vector5 damped_inverse;
  for (int index = 0; index < kArmDof; ++index) {
    const double singular = result.singular_values[index];
    damped_inverse[index] =
        singular / (singular * singular + params.regularization);
  }
  result.correction_raw =
      svd.matrixV() *
      (damped_inverse.array() * (svd.matrixU().transpose() * acceleration_error).array())
          .matrix();

  const double base_error_norm = (base_qacc - desired_qacc).norm();
  result.tau_cmd = base_tau;
  result.tau_cmd_raw = base_tau;
  result.qacc_validated = base_qacc;
  double best_error_norm = std::numeric_limits<double>::infinity();
  bool has_best_qacc_safe = false;
  double best_qacc_safe_joint_error = std::numeric_limits<double>::infinity();
  double best_qacc_safe_error_norm = std::numeric_limits<double>::infinity();
  Candidate best_qacc_safe;
  Vector5 best_qacc_safe_qacc = Vector5::Zero();
  std::vector<double> best_qacc_safe_warmstart;
  bool has_best_progress = false;
  double best_progress_error_norm = std::numeric_limits<double>::infinity();
  Candidate best_progress;
  Vector5 best_progress_qacc = Vector5::Zero();
  std::vector<double> best_progress_warmstart;

  std::array<Candidate, DDQ_TORQUE_MAPPER_MAX_VALIDATION_SCALES> candidates{};
  for (int index = 0; index < params.validation_scale_count; ++index) {
    Candidate& candidate = candidates[static_cast<std::size_t>(index)];
    candidate.scale = params.validation_scales[index];
    candidate.tau_raw = base_tau + candidate.scale * result.correction_raw;
    candidate.tau =
        clip_vector(candidate.tau_raw, handle.torque_lower, handle.torque_upper);
    const Vector5 predicted_qacc =
        base_qacc + result.gain_matrix * (candidate.tau - base_tau);
    const Vector5 predicted_error = predicted_qacc - desired_qacc;
    candidate.predicted_safe =
        max_abs(predicted_error) <= params.max_joint_error &&
        max_abs(predicted_qacc) <= params.max_abs_qacc;
    candidate.predicted_error_norm = predicted_error.norm();
  }
  const auto candidate_less = [](const Candidate& left, const Candidate& right) {
    if (left.predicted_safe != right.predicted_safe) {
      return left.predicted_safe && !right.predicted_safe;
    }
    if (left.predicted_error_norm != right.predicted_error_norm) {
      return left.predicted_error_norm < right.predicted_error_norm;
    }
    return left.scale > right.scale;
  };
  // 最多 8 个元素，用稳定插入排序避免 stable_sort 的临时堆缓冲。
  for (int index = 1; index < params.validation_scale_count; ++index) {
    Candidate value = candidates[static_cast<std::size_t>(index)];
    int insertion = index;
    while (insertion > 0 &&
           candidate_less(value,
                          candidates[static_cast<std::size_t>(insertion - 1)])) {
      candidates[static_cast<std::size_t>(insertion)] =
          candidates[static_cast<std::size_t>(insertion - 1)];
      --insertion;
    }
    candidates[static_cast<std::size_t>(insertion)] = value;
  }

  // 【核心代码】正常至少真实验收两个候选；已有安全候选便停止，
  // 否则继续较保守比例。该顺序与 Python candidate_specs 排序一致。
  const int minimum_validations =
      std::min(2, params.validation_scale_count);
  for (int candidate_index = 0;
       candidate_index < params.validation_scale_count;
       ++candidate_index) {
    const Candidate& candidate =
        candidates[static_cast<std::size_t>(candidate_index)];
    ++result.validation_attempts;
    prepare_ctrl(handle, candidate.tau, base_warmstart);
    mj_forwardSkip(model, data, mjSTAGE_VEL, 1);
    ++forward_skip_calls;
    const Vector5 candidate_qacc = read_right_arm_qacc(handle);
    const Vector5 candidate_error = candidate_qacc - desired_qacc;
    const double candidate_error_norm = candidate_error.norm();
    const bool total_error_improved = candidate_error_norm < base_error_norm;
    const bool joint_error_safe = max_abs(candidate_error) <= params.max_joint_error;
    const bool qacc_safe = max_abs(candidate_qacc) <= params.max_abs_qacc;
    result.total_error_rejections += static_cast<int>(!total_error_improved);
    result.joint_error_rejections += static_cast<int>(!joint_error_safe);
    result.qacc_limit_rejections += static_cast<int>(!qacc_safe);

    if (total_error_improved && joint_error_safe && qacc_safe) {
      ++result.safe_candidate_count;
      if (candidate_error_norm < best_error_norm) {
        result.validation_scale = candidate.scale;
        result.tau_cmd = candidate.tau;
        result.tau_cmd_raw = candidate.tau_raw;
        result.qacc_validated = candidate_qacc;
        copy_full_qacc(handle, result.qacc_validated_full);
        best_error_norm = candidate_error_norm;
      }
    }
    // 一旦存在完整安全候选，两个降级候选不会再参与最终选择。
    if (result.safe_candidate_count == 0 && total_error_improved && qacc_safe) {
      const double joint_error = max_abs(candidate_error);
      if (!has_best_qacc_safe || joint_error < best_qacc_safe_joint_error ||
          (joint_error == best_qacc_safe_joint_error &&
           candidate_error_norm < best_qacc_safe_error_norm)) {
        has_best_qacc_safe = true;
        best_qacc_safe_joint_error = joint_error;
        best_qacc_safe_error_norm = candidate_error_norm;
        best_qacc_safe = candidate;
        best_qacc_safe_qacc = candidate_qacc;
        copy_full_qacc(handle, best_qacc_safe_warmstart);
      }
    }
    // qacc-safe 降级候选的优先级高于 progress 候选；存在前者后无需再
    // 复制后者的完整 nv warm-start。
    if (result.safe_candidate_count == 0 && !has_best_qacc_safe &&
        total_error_improved &&
        (!has_best_progress || candidate_error_norm < best_progress_error_norm)) {
      has_best_progress = true;
      best_progress_error_norm = candidate_error_norm;
      best_progress = candidate;
      best_progress_qacc = candidate_qacc;
      copy_full_qacc(handle, best_progress_warmstart);
    }
    if (result.safe_candidate_count > 0 &&
        result.validation_attempts >= minimum_validations) {
      break;
    }
  }

  result.tracking_safety_satisfied = result.safe_candidate_count > 0;
  if (!result.tracking_safety_satisfied && has_best_qacc_safe) {
    result.validation_scale = best_qacc_safe.scale;
    result.tau_cmd = best_qacc_safe.tau;
    result.tau_cmd_raw = best_qacc_safe.tau_raw;
    result.qacc_validated = best_qacc_safe_qacc;
    result.qacc_validated_full = std::move(best_qacc_safe_warmstart);
  } else if (!result.tracking_safety_satisfied && has_best_progress) {
    result.validation_scale = best_progress.scale;
    result.tau_cmd = best_progress.tau;
    result.tau_cmd_raw = best_progress.tau_raw;
    result.qacc_validated = best_progress_qacc;
    result.qacc_validated_full = std::move(best_progress_warmstart);
  }
  result.qacc_safety_satisfied =
      max_abs(result.qacc_validated) <= params.max_abs_qacc;
  result.correction = result.tau_cmd - base_tau;
  result.qacc_predicted = base_qacc + result.gain_matrix * result.correction;
  result.qacc_validation_error = result.qacc_validated - desired_qacc;
  result.qacc_linearization_error =
      result.qacc_validated - result.qacc_predicted;
  result.improved = result.validation_scale > 0.0;
  if (result.singular_values[kArmDof - 1] >
      std::numeric_limits<double>::epsilon()) {
    result.condition_number =
        result.singular_values[0] / result.singular_values[kArmDof - 1];
  }
  return result;
}

}  // namespace

extern "C" {

int32_t ddq_torque_mapper_abi_version(void) { return kAbiVersion; }

const char* ddq_torque_mapper_status_string(const int32_t status) {
  switch (status) {
    case DDQ_TORQUE_MAPPER_OK:
      return "ok";
    case DDQ_TORQUE_MAPPER_INVALID_ARGUMENT:
      return "invalid argument";
    case DDQ_TORQUE_MAPPER_DIMENSION_MISMATCH:
      return "dimension mismatch";
    case DDQ_TORQUE_MAPPER_MODEL_ERROR:
      return "model error";
    case DDQ_TORQUE_MAPPER_NUMERICAL_ERROR:
      return "numerical error";
    case DDQ_TORQUE_MAPPER_INTERNAL_ERROR:
      return "internal error";
    default:
      return "unknown status";
  }
}

DdqTorqueMapperHandle* ddq_torque_mapper_create(
    const char* scene_mjcf_path,
    char* error_message,
    const int32_t error_message_capacity) {
  try {
    if (scene_mjcf_path == nullptr || scene_mjcf_path[0] == '\0') {
      set_error(error_message, error_message_capacity, "scene_mjcf_path 不能为空。");
      return nullptr;
    }
    char load_error[2048] = {};
    mjModel* raw_model =
        mj_loadXML(scene_mjcf_path, nullptr, load_error, sizeof(load_error));
    if (raw_model == nullptr) {
      set_error(error_message, error_message_capacity,
                std::string("MuJoCo MJCF 加载失败: ") + load_error);
      return nullptr;
    }
    std::unique_ptr<mjModel, MjModelDeleter> model_guard(raw_model);
    mjData* raw_data = mj_makeData(raw_model);
    if (raw_data == nullptr) {
      set_error(error_message, error_message_capacity, "mj_makeData 失败。");
      return nullptr;
    }
    auto handle = std::make_unique<DdqTorqueMapperHandle>(raw_model, raw_data);
    model_guard.release();
    validate_current_model(*handle);
    set_error(error_message, error_message_capacity, "");
    return handle.release();
  } catch (const std::exception& exception) {
    set_error(error_message, error_message_capacity, exception.what());
    return nullptr;
  } catch (...) {
    set_error(error_message, error_message_capacity, "创建 mapper 时发生未知异常。");
    return nullptr;
  }
}

void ddq_torque_mapper_destroy(DdqTorqueMapperHandle* handle) { delete handle; }

int32_t ddq_torque_mapper_nq(const DdqTorqueMapperHandle* handle) {
  return handle == nullptr ? -1 : handle->model->nq;
}

int32_t ddq_torque_mapper_nv(const DdqTorqueMapperHandle* handle) {
  return handle == nullptr ? -1 : handle->model->nv;
}

int32_t ddq_torque_mapper_nu(const DdqTorqueMapperHandle* handle) {
  return handle == nullptr ? -1 : handle->model->nu;
}

int32_t ddq_torque_mapper_nbody(const DdqTorqueMapperHandle* handle) {
  return handle == nullptr ? -1 : handle->model->nbody;
}

void ddq_torque_mapper_default_params(DdqTorqueMapperParams* params) {
  if (params == nullptr) {
    return;
  }
  std::memset(params, 0, sizeof(*params));
  params->perturbation = 0.1;
  params->regularization = 5.0;
  params->validation_scales[0] = 1.0;
  params->validation_scales[1] = 0.5;
  params->validation_scales[2] = 0.25;
  params->validation_scales[3] = 0.125;
  params->validation_scale_count = 4;
  params->second_pass_error_threshold = 5.0;
  params->max_joint_error = 4.0;
  params->max_abs_qacc = 8.0;
  params->enable_second_pass = 1;
  params->max_safety_rescue_passes = 2;
}

int32_t ddq_torque_mapper_compute(
    DdqTorqueMapperHandle* handle,
    const DdqTorqueMapperState* state,
    const DdqTorqueMapperRequest* request,
    const DdqTorqueMapperParams* params,
    DdqTorqueMapperOutput* output,
    char* error_message,
    const int32_t error_message_capacity) {
  const auto total_start = Clock::now();
  try {
    if (handle == nullptr) {
      set_error(error_message, error_message_capacity, "handle 不能为空。");
      return DDQ_TORQUE_MAPPER_INVALID_ARGUMENT;
    }
    std::string validation_error;
    const int validation_status =
        validate_inputs(*handle, state, request, params, output, validation_error);
    if (validation_status != DDQ_TORQUE_MAPPER_OK) {
      set_error(error_message, error_message_capacity, validation_error);
      return validation_status;
    }
    std::memset(output, 0, sizeof(*output));
    copy_state_inputs(*handle, *state);

    const Vector5 desired_qacc = load_vector5(request->desired_qacc);
    const Vector5 tau_nominal =
        clip_vector(load_vector5(request->tau_nominal),
                    handle->torque_lower,
                    handle->torque_upper);

    // 【核心代码】唯一一次完整 forward：复制当前物理状态后，
    // 以名义力矩建立约束、位置/速度缓存和右臂基线加速度。
    const auto baseline_start = Clock::now();
    prepare_ctrl(*handle, tau_nominal, handle->warmstart);
    mj_forward(handle->model.get(), handle->scratch.get());
    output->full_forward_calls = 1;
    const Vector5 qacc_baseline = read_right_arm_qacc(*handle);
    std::vector<double> baseline_warmstart;
    copy_full_qacc(*handle, baseline_warmstart);
    output->baseline_elapsed_ns = elapsed_ns(baseline_start);

    int forward_skip_calls = 0;
    int validated_pass_count = 0;
    const auto first_pass_start = Clock::now();
    const PassResult first_pass =
        solve_validated_pass(*handle,
                             desired_qacc,
                             tau_nominal,
                             qacc_baseline,
                             baseline_warmstart,
                             *params,
                             forward_skip_calls,
                             validated_pass_count);
    output->first_pass_elapsed_ns = elapsed_ns(first_pass_start);

    const double first_pass_residual_norm =
        first_pass.qacc_validation_error.norm();
    const bool second_pass_triggered =
        params->enable_second_pass != 0 && first_pass.improved &&
        (!first_pass.tracking_safety_satisfied ||
         first_pass_residual_norm > params->second_pass_error_threshold);
    PassResult second_pass;
    bool has_second_pass = false;
    if (second_pass_triggered) {
      const auto second_pass_start = Clock::now();
      second_pass = solve_validated_pass(*handle,
                                         desired_qacc,
                                         first_pass.tau_cmd,
                                         first_pass.qacc_validated,
                                         first_pass.qacc_validated_full,
                                         *params,
                                         forward_skip_calls,
                                         validated_pass_count);
      output->second_pass_elapsed_ns = elapsed_ns(second_pass_start);
      has_second_pass = true;
    }

    PassResult final_pass = first_pass;
    bool second_pass_accepted = false;
    if (has_second_pass && second_pass.tracking_safety_satisfied) {
      final_pass = second_pass;
      second_pass_accepted = true;
    } else if (first_pass.tracking_safety_satisfied) {
      final_pass = first_pass;
    } else if (has_second_pass && second_pass.improved &&
               second_pass.qacc_safety_satisfied) {
      final_pass = second_pass;
      second_pass_accepted = true;
    }

    bool safety_fallback_used = false;
    bool safety_fallback_satisfied = final_pass.qacc_safety_satisfied;
    int safety_fallback_attempts = 0;
    if (!safety_fallback_satisfied && params->max_safety_rescue_passes > 0) {
      // 【半核心代码】必要时在已改善工作点继续重线性化；次数受配置限制。
      safety_fallback_used = true;
      const auto rescue_start = Clock::now();
      for (int pass = 0; pass < params->max_safety_rescue_passes; ++pass) {
        ++safety_fallback_attempts;
        const PassResult rescue_pass =
            solve_validated_pass(*handle,
                                 desired_qacc,
                                 final_pass.tau_cmd,
                                 final_pass.qacc_validated,
                                 final_pass.qacc_validated_full,
                                 *params,
                                 forward_skip_calls,
                                 validated_pass_count);
        if (!rescue_pass.improved) {
          break;
        }
        final_pass = rescue_pass;
        if (final_pass.qacc_safety_satisfied) {
          break;
        }
      }
      safety_fallback_satisfied = final_pass.qacc_safety_satisfied;
      output->rescue_elapsed_ns = elapsed_ns(rescue_start);
    }

    // 【核心安全代码】救援仍失败时重新验收上一拍力矩，绝不未经本拍
    // 接触/浮动基动力学检查就直接复用。
    const bool hold_last_safe_available =
        !safety_fallback_satisfied && request->has_previous_executed_tau != 0;
    bool hold_last_safe_used = false;
    bool hold_last_safe_satisfied = false;
    Vector5 hold_last_safe_qacc = Vector5::Zero();
    Vector5 hold_tau = Vector5::Zero();
    if (hold_last_safe_available) {
      const auto hold_start = Clock::now();
      hold_tau = clip_vector(load_vector5(request->previous_executed_tau),
                             handle->torque_lower,
                             handle->torque_upper);
      prepare_ctrl(*handle, hold_tau, final_pass.qacc_validated_full);
      mj_forwardSkip(handle->model.get(), handle->scratch.get(), mjSTAGE_VEL, 1);
      ++forward_skip_calls;
      hold_last_safe_qacc = read_right_arm_qacc(*handle);
      hold_last_safe_satisfied = max_abs(hold_last_safe_qacc) <= params->max_abs_qacc;
      if (hold_last_safe_satisfied) {
        hold_last_safe_used = true;
        safety_fallback_satisfied = true;
      }
      output->hold_last_elapsed_ns = elapsed_ns(hold_start);
    }

    const Vector5 tau_cmd = hold_last_safe_used ? hold_tau : final_pass.tau_cmd;
    const Vector5 qacc_predicted =
        hold_last_safe_used ? hold_last_safe_qacc : final_pass.qacc_predicted;
    const Vector5 qacc_validated =
        hold_last_safe_used ? hold_last_safe_qacc : final_pass.qacc_validated;
    const Vector5 qacc_prediction_error = qacc_predicted - desired_qacc;
    const Vector5 qacc_validation_error = qacc_validated - desired_qacc;
    const Vector5 qacc_linearization_error =
        hold_last_safe_used ? Vector5::Zero() : final_pass.qacc_linearization_error;

    store_vector5(tau_cmd, output->tau_cmd);
    store_vector5(tau_nominal, output->tau_nominal);
    store_vector5(first_pass.correction_raw, output->tau_correction_raw);
    store_vector5(tau_cmd - tau_nominal, output->tau_correction);
    store_vector5(hold_last_safe_used ? tau_cmd : final_pass.tau_cmd_raw,
                  output->tau_cmd_raw);
    store_vector5(qacc_baseline, output->qacc_baseline);
    store_vector5(qacc_predicted, output->qacc_predicted);
    store_vector5(qacc_prediction_error, output->qacc_prediction_error);
    store_vector5(qacc_validated, output->qacc_validated);
    store_vector5(qacc_validation_error, output->qacc_validation_error);
    store_vector5(qacc_linearization_error, output->qacc_linearization_error);
    store_matrix5(final_pass.gain_matrix, output->gain_matrix);
    store_vector5(final_pass.singular_values, output->singular_values);
    output->condition_number = final_pass.condition_number;

    output->validation_scale = first_pass.validation_scale;
    output->validation_attempts = first_pass.validation_attempts;
    output->validation_improved = static_cast<int32_t>(first_pass.improved);
    output->validation_tracking_safety_satisfied =
        static_cast<int32_t>(first_pass.tracking_safety_satisfied);
    output->validation_qacc_safety_satisfied =
        static_cast<int32_t>(first_pass.qacc_safety_satisfied);
    output->validation_safe_candidate_count = first_pass.safe_candidate_count;
    output->validation_total_error_rejections = first_pass.total_error_rejections;
    output->validation_joint_error_rejections = first_pass.joint_error_rejections;
    output->validation_qacc_limit_rejections = first_pass.qacc_limit_rejections;
    store_vector5(
        first_pass.qacc_validated, output->first_pass_qacc_validated);
    store_vector5(
        first_pass.qacc_validation_error,
        output->first_pass_qacc_validation_error);
    output->second_pass_triggered = static_cast<int32_t>(second_pass_triggered);
    output->second_pass_accepted = static_cast<int32_t>(second_pass_accepted);
    if (has_second_pass) {
      store_vector5(
          second_pass.correction_raw,
          output->second_pass_tau_correction_raw);
      store_vector5(
          second_pass.correction, output->second_pass_tau_correction);
      store_vector5(
          second_pass.qacc_predicted, output->second_pass_qacc_predicted);
      store_vector5(
          second_pass.qacc_validated, output->second_pass_qacc_validated);
      store_vector5(
          second_pass.qacc_validation_error,
          output->second_pass_qacc_validation_error);
      store_vector5(
          second_pass.qacc_linearization_error,
          output->second_pass_qacc_linearization_error);
      store_matrix5(
          second_pass.gain_matrix, output->second_pass_gain_matrix);
      store_vector5(
          second_pass.singular_values,
          output->second_pass_singular_values);
      output->second_pass_condition_number = second_pass.condition_number;
      output->second_pass_validation_scale = second_pass.validation_scale;
      output->second_pass_validation_attempts = second_pass.validation_attempts;
      output->second_pass_tracking_safety_satisfied =
          static_cast<int32_t>(second_pass.tracking_safety_satisfied);
      output->second_pass_qacc_safety_satisfied =
          static_cast<int32_t>(second_pass.qacc_safety_satisfied);
      output->second_pass_safe_candidate_count =
          second_pass.safe_candidate_count;
      output->second_pass_total_error_rejections =
          second_pass.total_error_rejections;
      output->second_pass_joint_error_rejections =
          second_pass.joint_error_rejections;
      output->second_pass_qacc_limit_rejections =
          second_pass.qacc_limit_rejections;
    } else {
      output->second_pass_condition_number =
          std::numeric_limits<double>::infinity();
      output->second_pass_validation_scale = 0.0;
      output->second_pass_validation_attempts = 0;
      output->second_pass_tracking_safety_satisfied = 0;
      output->second_pass_qacc_safety_satisfied = 0;
    }
    output->safety_fallback_used = static_cast<int32_t>(safety_fallback_used);
    output->safety_fallback_satisfied =
        static_cast<int32_t>(safety_fallback_satisfied);
    output->safety_fallback_attempts = safety_fallback_attempts;
    output->hold_last_safe_available =
        static_cast<int32_t>(hold_last_safe_available);
    output->hold_last_safe_used = static_cast<int32_t>(hold_last_safe_used);
    output->hold_last_safe_satisfied =
        static_cast<int32_t>(hold_last_safe_satisfied);
    store_vector5(hold_last_safe_qacc, output->hold_last_safe_qacc);
    output->forward_skip_calls = forward_skip_calls;
    output->validated_pass_count = validated_pass_count;
    output->total_elapsed_ns = elapsed_ns(total_start);

    if (!all_finite(output->tau_cmd, kArmDof) ||
        !all_finite(output->qacc_validated, kArmDof)) {
      set_error(error_message, error_message_capacity,
                "映射结果包含 NaN 或 Inf。\n");
      return DDQ_TORQUE_MAPPER_NUMERICAL_ERROR;
    }
    set_error(error_message, error_message_capacity, "");
    return DDQ_TORQUE_MAPPER_OK;
  } catch (const std::exception& exception) {
    set_error(error_message, error_message_capacity, exception.what());
    return DDQ_TORQUE_MAPPER_INTERNAL_ERROR;
  } catch (...) {
    set_error(error_message, error_message_capacity, "compute 中发生未知异常。");
    return DDQ_TORQUE_MAPPER_INTERNAL_ERROR;
  }
}

}  // extern "C"
