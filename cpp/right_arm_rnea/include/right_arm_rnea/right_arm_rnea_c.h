#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(RIGHT_ARM_RNEA_BUILDING_LIBRARY)
#    define RIGHT_ARM_RNEA_API __declspec(dllexport)
#  else
#    define RIGHT_ARM_RNEA_API __declspec(dllimport)
#  endif
#else
#  define RIGHT_ARM_RNEA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { RIGHT_ARM_RNEA_JOINT_COUNT = 5 };

typedef struct RightArmRneaHandle RightArmRneaHandle;

typedef enum RightArmRneaStatus {
    RIGHT_ARM_RNEA_OK = 0,
    RIGHT_ARM_RNEA_INVALID_ARGUMENT = 1,
    RIGHT_ARM_RNEA_DIMENSION_MISMATCH = 2,
    RIGHT_ARM_RNEA_NONFINITE_INPUT = 3,
    RIGHT_ARM_RNEA_MODEL_ERROR = 4,
    RIGHT_ARM_RNEA_INTERNAL_ERROR = 5,
} RightArmRneaStatus;

typedef struct RightArmRneaOutput {
    // 【核心输出】Pinocchio 刚体动力学力矩，不含 MuJoCo passive/friction。
    double tau_rnea[RIGHT_ARM_RNEA_JOINT_COUNT];
    // 与当前 Python 执行链一致的滑动摩擦广义力。
    double tau_constraint_friction[RIGHT_ARM_RNEA_JOINT_COUNT];
    // tau_ff = tau_rnea - tau_passive - tau_constraint_friction。
    double tau_ff[RIGHT_ARM_RNEA_JOINT_COUNT];
    // 从 MuJoCo 状态映射开始，到取出 5 维 RNEA 结果为止。
    uint64_t core_elapsed_ns;
    // 仅 Pinocchio rnea() 调用本身，便于区分映射和库核心耗时。
    uint64_t rnea_elapsed_ns;
} RightArmRneaOutput;

// 从仿真使用的 scene.xml 建模。函数会让 MuJoCo 解析完整 scene，并让
// Pinocchio 解析 scene 中唯一 include 的机器人 MJCF，以保持关节顺序一致。
RIGHT_ARM_RNEA_API RightArmRneaHandle* right_arm_rnea_create(
    const char* scene_mjcf_path,
    char* error_message,
    size_t error_capacity);

RIGHT_ARM_RNEA_API void right_arm_rnea_destroy(RightArmRneaHandle* handle);

RIGHT_ARM_RNEA_API size_t right_arm_rnea_mujoco_nq(
    const RightArmRneaHandle* handle);
RIGHT_ARM_RNEA_API size_t right_arm_rnea_mujoco_nv(
    const RightArmRneaHandle* handle);

// 【核心 C ABI】输入与当前 Python Pinocchio 后端完全相同的 MuJoCo
// qpos/qvel 和右臂 ddq。tau_passive、friction_loss 均为右臂 5 维。
// handle 内部复用 Pinocchio Data，因此同一 handle 不可被多个线程并发调用。
RIGHT_ARM_RNEA_API RightArmRneaStatus right_arm_rnea_compute(
    RightArmRneaHandle* handle,
    const double* mujoco_qpos,
    size_t qpos_count,
    const double* mujoco_qvel,
    size_t qvel_count,
    const double* desired_right_arm_ddq,
    size_t ddq_count,
    const double* tau_passive,
    size_t passive_count,
    const double* friction_loss,
    size_t friction_count,
    double mujoco_timestep,
    double friction_breakaway_steps,
    RightArmRneaOutput* output,
    char* error_message,
    size_t error_capacity);

RIGHT_ARM_RNEA_API const char* right_arm_rnea_status_string(
    RightArmRneaStatus status);
RIGHT_ARM_RNEA_API uint32_t right_arm_rnea_abi_version(void);

#ifdef __cplusplus
}
#endif
