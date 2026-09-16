foreach(arguments IN ITEMS "--permit-real-output;WRONG"
    "fake_nic;--profile;${PROFILE};--output-dir;${OUTPUT};--permit-real-output;TIMED_WALK_RAW_CAPTURE"
    "fake_nic;--profile;${SYNTHETIC};--output-dir;${OUTPUT};--permit-real-output;TIMED_WALK_RAW_CAPTURE")
    execute_process(COMMAND "${EXECUTABLE}" ${arguments}
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 3)
    if(NOT rc EQUAL 1)
        message(FATAL_ERROR "invalid CLI/profile must reject immediately: ${rc} ${out} ${err}")
    endif()
    if(EXISTS "${OUTPUT}")
        message(FATAL_ERROR "invalid profile reached session creation")
    endif()
endforeach()
