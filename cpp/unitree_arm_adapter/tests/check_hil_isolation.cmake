if(NOT DEFINED HIL_BINARY OR NOT EXISTS "${HIL_BINARY}")
    message(FATAL_ERROR "HIL_BINARY is missing")
endif()

set(HIL_SOURCES
    "${HIL_MAIN_SOURCE}"
    "${HIL_SINK_SOURCE}"
    "${HIL_CACHE_SOURCE}"
    "${HIL_DISPATCH_SOURCE}")
set(FORBIDDEN_CAPABILITIES
    "LowCmd_"
    "ChannelPublisher"
    "rt/arm_sdk")

foreach(SOURCE_PATH IN LISTS HIL_SOURCES)
    if(NOT EXISTS "${SOURCE_PATH}")
        message(FATAL_ERROR "HIL source is missing: ${SOURCE_PATH}")
    endif()
    file(READ "${SOURCE_PATH}" SOURCE_TEXT)
    foreach(CAPABILITY IN LISTS FORBIDDEN_CAPABILITIES)
        string(FIND "${SOURCE_TEXT}" "${CAPABILITY}" LOCATION)
        if(NOT LOCATION EQUAL -1)
            message(FATAL_ERROR
                "HIL source contains forbidden output capability: ${CAPABILITY}")
        endif()
    endforeach()
endforeach()

find_program(STRINGS_TOOL strings)
if(NOT STRINGS_TOOL)
    message(FATAL_ERROR "strings tool is unavailable")
endif()
execute_process(
    COMMAND "${STRINGS_TOOL}" "${HIL_BINARY}"
    RESULT_VARIABLE STRINGS_RESULT
    OUTPUT_VARIABLE BINARY_STRINGS)
if(NOT STRINGS_RESULT EQUAL 0)
    message(FATAL_ERROR "could not inspect HIL binary strings")
endif()
foreach(CAPABILITY IN LISTS FORBIDDEN_CAPABILITIES)
    string(FIND "${BINARY_STRINGS}" "${CAPABILITY}" LOCATION)
    if(NOT LOCATION EQUAL -1)
        message(FATAL_ERROR
            "HIL binary contains forbidden output capability: ${CAPABILITY}")
    endif()
endforeach()

find_program(LDD_TOOL ldd)
if(NOT LDD_TOOL)
    message(FATAL_ERROR "ldd tool is unavailable")
endif()
execute_process(
    COMMAND "${LDD_TOOL}" "${HIL_BINARY}"
    RESULT_VARIABLE LDD_RESULT
    OUTPUT_VARIABLE LINKED_LIBRARIES
    ERROR_VARIABLE LDD_ERROR)
if(NOT LDD_RESULT EQUAL 0)
    message(FATAL_ERROR "could not inspect HIL dependencies: ${LDD_ERROR}")
endif()
string(TOLOWER "${LINKED_LIBRARIES}" LINKED_LIBRARIES_LOWER)
string(FIND "${LINKED_LIBRARIES_LOWER}" "unitree_sdk" SDK_LOCATION)
if(NOT SDK_LOCATION EQUAL -1)
    message(FATAL_ERROR "HIL binary links a device SDK library")
endif()

message(STATUS "publisher-absent HIL capability isolation verified")
