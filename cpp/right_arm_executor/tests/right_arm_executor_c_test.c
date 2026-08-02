#include "right_arm_executor/right_arm_executor_c.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void expect_true(int condition, const char* message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void expect_near(double actual, double expected, const char* message) {
    if (fabs(actual - expected) > 1e-12) {
        ++failures;
        fprintf(stderr, "FAIL: %s: expected %.12f, got %.12f\n",
                message, expected, actual);
    }
}

static void configure_test_values(rae_config_v1* config) {
    size_t joint;
    config->command_timeout_ns = 30000000;
    config->state_timeout_ns = 10000000;
    for (joint = 0; joint < RAE_JOINT_COUNT; ++joint) {
        config->kp[joint] = 10.0;
        config->kd[joint] = 2.0;
        config->timeout_damping[joint] = 3.0;
        config->q_ref_min[joint] = -0.5;
        config->q_ref_max[joint] = 0.5;
        config->dq_ref_abs_max[joint] = 1.0;
        config->tau_min[joint] = -5.0;
        config->tau_max[joint] = 5.0;
    }
}

static rae_input_v1 nominal_input(void) {
    rae_input_v1 input;
    size_t joint;
    memset(&input, 0, sizeof(input));
    input.struct_size = (uint32_t)sizeof(input);
    input.abi_version = RAE_ABI_VERSION_V1;
    input.command_timestamp_ns = 100;
    input.state_timestamp_ns = 100;
    for (joint = 0; joint < RAE_JOINT_COUNT; ++joint) {
        input.q[joint] = 0.1;
        input.dq[joint] = 0.2;
        input.q_ref[joint] = 0.3;
        input.dq_ref[joint] = 0.1;
        input.tau_ff[joint] = 0.5;
    }
    return input;
}

static void test_mode(uint32_t semantics) {
    rae_config_v1 config;
    rae_executor_handle* handle = NULL;
    rae_input_v1 input = nominal_input();
    rae_output_v1 output;
    size_t joint;
    int32_t status = rae_get_default_config_v1(semantics, &config);
    expect_true(status == RAE_STATUS_OK, "default C config");
    configure_test_values(&config);
    status = rae_create_v1(&config, &handle);
    expect_true(status == RAE_STATUS_OK && handle != NULL, "create C handle");
    if (handle == NULL) {
        return;
    }

    memset(&output, 0, sizeof(output));
    status = rae_step_v1(handle, &input, 200, &output);
    expect_true(status == RAE_STATUS_OK, "C ABI step status");
    expect_true(output.struct_size == sizeof(output), "C output struct size");
    expect_true(output.executor_mode == RAE_MODE_ACTIVE, "C active mode");
    expect_true(output.output_semantics == semantics, "C output semantics");
    expect_true(output.core_elapsed_ns < 1000000000ULL,
                "C core elapsed diagnostic must be plausible");

    for (joint = 0; joint < RAE_JOINT_COUNT; ++joint) {
        expect_near(output.predicted_pd_tau[joint], 1.8,
                    "C predicted PD torque");
        expect_near(output.predicted_total_tau_raw[joint], 2.3,
                    "C predicted total torque");
        if (semantics == RAE_OUTPUT_HOST_FULL_TORQUE) {
            expect_near(output.actuator_kp[joint], 0.0,
                        "C host mode actuator kp");
            expect_near(output.actuator_kd[joint], 0.0,
                        "C host mode actuator kd");
            expect_near(output.actuator_tau_ff[joint], 2.3,
                        "C host mode full torque field");
        } else {
            expect_near(output.actuator_kp[joint], 10.0,
                        "C device mode actuator kp");
            expect_near(output.actuator_kd[joint], 2.0,
                        "C device mode actuator kd");
            // 【核心测试】设备 PD 模式不得把已经含 PD 的 2.3 再当 tau_ff。
            expect_near(output.actuator_tau_ff[joint], 0.5,
                        "C device mode feedforward excludes PD");
        }
    }
    if (semantics == RAE_OUTPUT_DEVICE_PD) {
        expect_true(
            (output.flags & RAE_FLAG_DEVICE_TOTAL_TORQUE_LIMIT_REQUIRED) != 0,
            "device-PD final torque limit flag");
    }
    rae_destroy(handle);
}

static void test_abi_rejection(void) {
    rae_config_v1 config;
    rae_executor_handle* handle = NULL;
    int32_t status = rae_get_default_config_v1(
        RAE_OUTPUT_HOST_FULL_TORQUE, &config);
    expect_true(status == RAE_STATUS_OK, "default config before ABI rejection");
    config.abi_version = 999;
    status = rae_create_v1(&config, &handle);
    expect_true(status == RAE_STATUS_INCOMPATIBLE_ABI,
                "unknown ABI version must be rejected");
    expect_true(handle == NULL, "rejected ABI must not return a handle");
}

static void test_independent_state_timeout(void) {
    rae_config_v1 config;
    rae_executor_handle* handle = NULL;
    rae_input_v1 input = nominal_input();
    rae_output_v1 output;
    int32_t status = rae_get_default_config_v1(
        RAE_OUTPUT_HOST_FULL_TORQUE, &config);
    expect_true(status == RAE_STATUS_OK, "state timeout default config");
    configure_test_values(&config);
    status = rae_create_v1(&config, &handle);
    expect_true(status == RAE_STATUS_OK && handle != NULL,
                "state timeout handle");
    if (handle == NULL) {
        return;
    }
    input.command_timestamp_ns = 20000000;
    input.state_timestamp_ns = 0;
    status = rae_step_v1(handle, &input, 20000000, &output);
    expect_true(status == RAE_STATUS_OK, "state timeout step status");
    expect_true(output.executor_mode == RAE_MODE_STATE_TIMED_OUT,
                "state timeout runtime mode");
    rae_destroy(handle);
}

int main(void) {
    expect_true(rae_abi_version() == RAE_ABI_VERSION_V1, "ABI version function");
    test_mode(RAE_OUTPUT_HOST_FULL_TORQUE);
    test_mode(RAE_OUTPUT_DEVICE_PD);
    test_abi_rejection();
    test_independent_state_timeout();

    if (failures != 0) {
        fprintf(stderr, "%d C ABI assertion(s) failed\n", failures);
        return 1;
    }
    puts("All right_arm_executor C ABI tests passed");
    return 0;
}
