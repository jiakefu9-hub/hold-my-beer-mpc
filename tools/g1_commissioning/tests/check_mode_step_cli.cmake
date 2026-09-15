# Rejected arguments must exit before opening evidence files or initializing DDS.
execute_process(COMMAND "${EXECUTABLE}" RESULT_VARIABLE no_args
    OUTPUT_QUIET ERROR_QUIET TIMEOUT 3)
if(NOT no_args STREQUAL "1")
    message(FATAL_ERROR "missing arguments were not rejected")
endif()
execute_process(COMMAND "${EXECUTABLE}" lo --target 500 --log "${OUTPUT}"
    --permit A1B_HOISTED_MODE_ONLY RESULT_VARIABLE invalid_target
    OUTPUT_QUIET ERROR_QUIET TIMEOUT 3)
if(NOT invalid_target STREQUAL "1" OR EXISTS "${OUTPUT}")
    message(FATAL_ERROR "forbidden target was not rejected before initialization")
endif()
execute_process(COMMAND "${EXECUTABLE}" lo --target damp --log "${OUTPUT}"
    --permit WRONG RESULT_VARIABLE invalid_permit
    OUTPUT_QUIET ERROR_QUIET TIMEOUT 3)
if(NOT invalid_permit STREQUAL "1" OR EXISTS "${OUTPUT}")
    message(FATAL_ERROR "missing permission was not rejected before initialization")
endif()
