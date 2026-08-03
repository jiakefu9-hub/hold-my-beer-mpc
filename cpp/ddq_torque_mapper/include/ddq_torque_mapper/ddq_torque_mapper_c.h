#ifndef DDQ_TORQUE_MAPPER_DDQ_TORQUE_MAPPER_C_H_
#define DDQ_TORQUE_MAPPER_DDQ_TORQUE_MAPPER_C_H_

#include <stdint.h>

#if defined(_WIN32)
#if defined(DDQ_TORQUE_MAPPER_BUILDING_LIBRARY)
#define DDQ_TORQUE_MAPPER_API __declspec(dllexport)
#else
#define DDQ_TORQUE_MAPPER_API __declspec(dllimport)
#endif
#else
#define DDQ_TORQUE_MAPPER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
  DDQ_TORQUE_MAPPER_ARM_DOF = 5,
  DDQ_TORQUE_MAPPER_MAX_VALIDATION_SCALES = 8,
};

typedef enum DdqTorqueMapperStatus {
  DDQ_TORQUE_MAPPER_OK = 0,
  DDQ_TORQUE_MAPPER_INVALID_ARGUMENT = 1,
  DDQ_TORQUE_MAPPER_DIMENSION_MISMATCH = 2,
  DDQ_TORQUE_MAPPER_MODEL_ERROR = 3,
  DDQ_TORQUE_MAPPER_NUMERICAL_ERROR = 4,
  DDQ_TORQUE_MAPPER_INTERNAL_ERROR = 5,
} DdqTorqueMapperStatus;

typedef struct DdqTorqueMapperHandle DdqTorqueMapperHandle;

/*
 * 【核心输入】当前物理拍的完整动力学输入。
 *
 * qacc_warmstart 虽然不是机器人测量量，但 MuJoCo 约束求解会读取它；
 * 显式传入才能与 Python 版本的候选验收严格使用同一个求解初值。
 */
typedef struct DdqTorqueMapperState {
  double time;
  const double* qpos;
  int32_t qpos_count;
  const double* qvel;
  int32_t qvel_count;
  const double* ctrl;
  int32_t ctrl_count;
  const double* qacc_warmstart;
  int32_t qacc_warmstart_count;
  const double* qfrc_applied;
  int32_t qfrc_applied_count;
  const double* xfrc_applied;
  int32_t xfrc_applied_count;
} DdqTorqueMapperState;

/* 【核心输入】本拍右臂期望加速度、名义力矩及可选上一拍安全力矩。 */
typedef struct DdqTorqueMapperRequest {
  double desired_qacc[DDQ_TORQUE_MAPPER_ARM_DOF];
  double tau_nominal[DDQ_TORQUE_MAPPER_ARM_DOF];
  int32_t has_previous_executed_tau;
  double previous_executed_tau[DDQ_TORQUE_MAPPER_ARM_DOF];
} DdqTorqueMapperRequest;

/* 【半核心配置】字段语义与 sim_support.py 的 Python 实现保持一致。 */
typedef struct DdqTorqueMapperParams {
  double perturbation;
  double regularization;
  double validation_scales[DDQ_TORQUE_MAPPER_MAX_VALIDATION_SCALES];
  int32_t validation_scale_count;
  double second_pass_error_threshold;
  double max_joint_error;
  double max_abs_qacc;
  int32_t enable_second_pass;
  int32_t max_safety_rescue_passes;
} DdqTorqueMapperParams;

/*
 * 【核心输出】最终执行力矩和足以复核选择过程的诊断。
 * 所有 5x5 矩阵都按 C row-major 展平。
 */
typedef struct DdqTorqueMapperOutput {
  double tau_cmd[DDQ_TORQUE_MAPPER_ARM_DOF];
  double tau_nominal[DDQ_TORQUE_MAPPER_ARM_DOF];
  double tau_correction_raw[DDQ_TORQUE_MAPPER_ARM_DOF];
  double tau_correction[DDQ_TORQUE_MAPPER_ARM_DOF];
  double tau_cmd_raw[DDQ_TORQUE_MAPPER_ARM_DOF];

  double qacc_baseline[DDQ_TORQUE_MAPPER_ARM_DOF];
  double qacc_predicted[DDQ_TORQUE_MAPPER_ARM_DOF];
  double qacc_prediction_error[DDQ_TORQUE_MAPPER_ARM_DOF];
  double qacc_validated[DDQ_TORQUE_MAPPER_ARM_DOF];
  double qacc_validation_error[DDQ_TORQUE_MAPPER_ARM_DOF];
  double qacc_linearization_error[DDQ_TORQUE_MAPPER_ARM_DOF];
  double gain_matrix[DDQ_TORQUE_MAPPER_ARM_DOF * DDQ_TORQUE_MAPPER_ARM_DOF];
  double singular_values[DDQ_TORQUE_MAPPER_ARM_DOF];
  double condition_number;

  double validation_scale;
  int32_t validation_attempts;
  int32_t validation_improved;
  int32_t validation_tracking_safety_satisfied;
  int32_t validation_qacc_safety_satisfied;
  int32_t validation_safe_candidate_count;
  int32_t validation_total_error_rejections;
  int32_t validation_joint_error_rejections;
  int32_t validation_qacc_limit_rejections;
  double first_pass_qacc_validated[DDQ_TORQUE_MAPPER_ARM_DOF];
  double first_pass_qacc_validation_error[DDQ_TORQUE_MAPPER_ARM_DOF];

  int32_t second_pass_triggered;
  int32_t second_pass_accepted;
  double second_pass_tau_correction_raw[DDQ_TORQUE_MAPPER_ARM_DOF];
  double second_pass_tau_correction[DDQ_TORQUE_MAPPER_ARM_DOF];
  double second_pass_qacc_predicted[DDQ_TORQUE_MAPPER_ARM_DOF];
  double second_pass_qacc_validated[DDQ_TORQUE_MAPPER_ARM_DOF];
  double second_pass_qacc_validation_error[DDQ_TORQUE_MAPPER_ARM_DOF];
  double second_pass_qacc_linearization_error[DDQ_TORQUE_MAPPER_ARM_DOF];
  double second_pass_gain_matrix[
      DDQ_TORQUE_MAPPER_ARM_DOF * DDQ_TORQUE_MAPPER_ARM_DOF];
  double second_pass_singular_values[DDQ_TORQUE_MAPPER_ARM_DOF];
  double second_pass_condition_number;
  double second_pass_validation_scale;
  int32_t second_pass_validation_attempts;
  int32_t second_pass_tracking_safety_satisfied;
  int32_t second_pass_qacc_safety_satisfied;
  int32_t second_pass_safe_candidate_count;
  int32_t second_pass_total_error_rejections;
  int32_t second_pass_joint_error_rejections;
  int32_t second_pass_qacc_limit_rejections;

  int32_t safety_fallback_used;
  int32_t safety_fallback_satisfied;
  int32_t safety_fallback_attempts;
  int32_t hold_last_safe_available;
  int32_t hold_last_safe_used;
  int32_t hold_last_safe_satisfied;
  double hold_last_safe_qacc[DDQ_TORQUE_MAPPER_ARM_DOF];

  /* 【非核心诊断】真实 MuJoCo 调用数及 C++ 内部耗时，单位 ns。 */
  int32_t full_forward_calls;
  int32_t forward_skip_calls;
  int32_t validated_pass_count;
  uint64_t baseline_elapsed_ns;
  uint64_t first_pass_elapsed_ns;
  uint64_t second_pass_elapsed_ns;
  uint64_t rescue_elapsed_ns;
  uint64_t hold_last_elapsed_ns;
  uint64_t total_elapsed_ns;
} DdqTorqueMapperOutput;

DDQ_TORQUE_MAPPER_API int32_t ddq_torque_mapper_abi_version(void);
DDQ_TORQUE_MAPPER_API const char* ddq_torque_mapper_status_string(int32_t status);

DDQ_TORQUE_MAPPER_API DdqTorqueMapperHandle* ddq_torque_mapper_create(
    const char* scene_mjcf_path,
    char* error_message,
    int32_t error_message_capacity);

DDQ_TORQUE_MAPPER_API void ddq_torque_mapper_destroy(
    DdqTorqueMapperHandle* handle);

DDQ_TORQUE_MAPPER_API int32_t ddq_torque_mapper_nq(
    const DdqTorqueMapperHandle* handle);
DDQ_TORQUE_MAPPER_API int32_t ddq_torque_mapper_nv(
    const DdqTorqueMapperHandle* handle);
DDQ_TORQUE_MAPPER_API int32_t ddq_torque_mapper_nu(
    const DdqTorqueMapperHandle* handle);
DDQ_TORQUE_MAPPER_API int32_t ddq_torque_mapper_nbody(
    const DdqTorqueMapperHandle* handle);

DDQ_TORQUE_MAPPER_API void ddq_torque_mapper_default_params(
    DdqTorqueMapperParams* params);

DDQ_TORQUE_MAPPER_API int32_t ddq_torque_mapper_compute(
    DdqTorqueMapperHandle* handle,
    const DdqTorqueMapperState* state,
    const DdqTorqueMapperRequest* request,
    const DdqTorqueMapperParams* params,
    DdqTorqueMapperOutput* output,
    char* error_message,
    int32_t error_message_capacity);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DDQ_TORQUE_MAPPER_DDQ_TORQUE_MAPPER_C_H_
