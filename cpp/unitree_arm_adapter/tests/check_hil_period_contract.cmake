if(NOT DEFINED HIL_BINARY)
    message(FATAL_ERROR "HIL_BINARY is required")
endif()

execute_process(
    COMMAND "${HIL_BINARY}"
        --period-us 1000
        --record-jsonl /tmp/unitree_arm_adapter_hil_period_contract.jsonl
        --session-nonce 1
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)

if(result EQUAL 0)
    message(FATAL_ERROR "HIL accepted a non-2ms period")
endif()

set(output "${stdout}${stderr}")
if(NOT output MATCHES "--period-us must be exactly 2000")
    message(FATAL_ERROR
        "HIL rejected the invocation for an unexpected reason: ${output}")
endif()
