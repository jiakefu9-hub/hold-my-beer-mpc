if(NOT DEFINED PREVIEW OR NOT DEFINED PROFILE OR NOT DEFINED STATE OR
   NOT DEFINED OUTPUT)
    message(FATAL_ERROR "preview test inputs are required")
endif()

file(REMOVE "${OUTPUT}")
execute_process(
    COMMAND "${PREVIEW}" --profile "${PROFILE}" --state "${STATE}"
            --output "${OUTPUT}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "preview failed (${result}): ${stdout}\n${stderr}")
endif()
file(READ "${OUTPUT}" records)
if(NOT records MATCHES "offline_would_write")
    message(FATAL_ERROR "preview did not retain would-write records")
endif()
if(NOT records MATCHES "real_output_profile_gate_passed.*false")
    message(FATAL_ERROR "synthetic preview incorrectly passed real-output profile gate")
endif()
if(NOT stdout MATCHES "offline_preview_completed=true")
    message(FATAL_ERROR "preview completion marker missing")
endif()
