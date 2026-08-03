#ifndef RIGHT_ARM_EXECUTOR_RIGHT_ARM_EXECUTOR_C_H_
#define RIGHT_ARM_EXECUTOR_RIGHT_ARM_EXECUTOR_C_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(RAE_BUILDING_LIBRARY)
#define RAE_API __declspec(dllexport)
#else
#define RAE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define RAE_API __attribute__((visibility("default")))
#else
#define RAE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 【核心代码】C ABI 的固定版本和固定 5 维布局。新增不兼容字段时必须升版本，
// 不能静默改变 v1 结构体含义。
#define RAE_ABI_VERSION_V1 1u
#define RAE_JOINT_COUNT 5u

enum {
    RAE_STATUS_OK = 0,
    RAE_STATUS_INVALID_ARGUMENT = 1,
    RAE_STATUS_INCOMPATIBLE_ABI = 2,
    RAE_STATUS_INVALID_CONFIG = 3,
    RAE_STATUS_OUT_OF_MEMORY = 4,
    RAE_STATUS_INTERNAL_ERROR = 5,
};

enum {
    RAE_OUTPUT_HOST_FULL_TORQUE = 0,
    RAE_OUTPUT_DEVICE_PD = 1,
};

enum {
    RAE_MODE_ACTIVE = 0,
    RAE_MODE_COMMAND_TIMED_OUT = 1,
    RAE_MODE_STATE_TIMED_OUT = 2,
    RAE_MODE_INVALID_COMMAND = 3,
    RAE_MODE_INVALID_STATE = 4,
};

enum {
    RAE_FLAG_POSITION_REFERENCE_CLAMPED = 1u << 0,
    RAE_FLAG_VELOCITY_REFERENCE_CLAMPED = 1u << 1,
    RAE_FLAG_PREDICTED_TOTAL_TORQUE_CLAMPED = 1u << 2,
    RAE_FLAG_FEEDFORWARD_CLAMPED = 1u << 3,
    RAE_FLAG_DAMPING_FALLBACK_ACTIVE = 1u << 4,
    RAE_FLAG_DEVICE_TOTAL_TORQUE_LIMIT_REQUIRED = 1u << 5,
};

typedef struct rae_config_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t output_semantics;
    uint32_t reserved0;
    int64_t command_timeout_ns;
    int64_t state_timeout_ns;
    double kp[RAE_JOINT_COUNT];
    double kd[RAE_JOINT_COUNT];
    double timeout_damping[RAE_JOINT_COUNT];
    double q_ref_min[RAE_JOINT_COUNT];
    double q_ref_max[RAE_JOINT_COUNT];
    double dq_ref_abs_max[RAE_JOINT_COUNT];
    double tau_min[RAE_JOINT_COUNT];
    double tau_max[RAE_JOINT_COUNT];
} rae_config_v1;

typedef struct rae_input_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int64_t command_timestamp_ns;
    int64_t state_timestamp_ns;
    double q[RAE_JOINT_COUNT];
    double dq[RAE_JOINT_COUNT];
    double q_ref[RAE_JOINT_COUNT];
    double dq_ref[RAE_JOINT_COUNT];
    // 始终只表示前馈，不允许预先包含 PD。
    double tau_ff[RAE_JOINT_COUNT];
} rae_input_v1;

typedef struct rae_output_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t executor_mode;
    uint32_t output_semantics;
    uint32_t flags;
    uint32_t reserved0;
    int64_t command_age_ns;
    int64_t state_age_ns;
    // 仅计 C++ RightArmExecutor::Step；不包含 ctypes/IPC/调用者开销。
    uint64_t core_elapsed_ns;
    double effective_q_ref[RAE_JOINT_COUNT];
    double effective_dq_ref[RAE_JOINT_COUNT];
    double predicted_pd_tau[RAE_JOINT_COUNT];
    double predicted_total_tau_raw[RAE_JOINT_COUNT];
    double predicted_total_tau_limited[RAE_JOINT_COUNT];

    // 【核心代码】适配器只发送以下五组 actuator_* 字段。
    // host-full-torque: kp/kd 为零，tau_ff 已是完整限幅后力矩。
    // device-PD: tau_ff 不含 PD，设备用 q/dq/kp/kd 计算且负责最终总限幅。
    double actuator_q_ref[RAE_JOINT_COUNT];
    double actuator_dq_ref[RAE_JOINT_COUNT];
    double actuator_kp[RAE_JOINT_COUNT];
    double actuator_kd[RAE_JOINT_COUNT];
    double actuator_tau_ff[RAE_JOINT_COUNT];
} rae_output_v1;

typedef struct rae_executor_handle rae_executor_handle;

RAE_API uint32_t rae_abi_version(void);
RAE_API int32_t rae_get_default_config_v1(
    uint32_t output_semantics,
    rae_config_v1* out_config);
RAE_API int32_t rae_create_v1(
    const rae_config_v1* config,
    rae_executor_handle** out_handle);
RAE_API void rae_destroy(rae_executor_handle* handle);

// 【核心代码】Step 内部不分配内存、不读时钟、不做 I/O；调用者提供 now_ns。
// 运行期超时或数据非法仍返回 RAE_STATUS_OK，具体状态写入 executor_mode；
// 非零返回值只表示 ABI/指针/配置等调用层错误。
RAE_API int32_t rae_step_v1(
    const rae_executor_handle* handle,
    const rae_input_v1* input,
    int64_t now_ns,
    rae_output_v1* output);

RAE_API const char* rae_status_string(int32_t status);
RAE_API const char* rae_executor_mode_string(uint32_t executor_mode);
RAE_API const char* rae_output_semantics_string(uint32_t output_semantics);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // RIGHT_ARM_EXECUTOR_RIGHT_ARM_EXECUTOR_C_H_
