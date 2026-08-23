cmake_minimum_required(VERSION 3.16)
project(unitree_arm_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(UNITREE_ARM_ADAPTER_BUILD_DDS
    "保留名；Stage 2 明确禁止构建真实命令publisher" OFF)
option(UNITREE_ARM_ADAPTER_BUILD_STATE_BRIDGE
    "构建只订阅状态、没有命令publisher的Unitree bridge" OFF)
set(UNITREE_SDK2_DIR "/home/fjk/g1_ws/unitree_sdk2" CACHE PATH
    "Unitree SDK2源码目录")

find_package(Threads REQUIRED)

add_library(unitree_arm_adapter_core
    src/hardware_command_supervisor.cpp
    src/periodic_loop.cpp
    src/safety.cpp
    src/shared_memory.cpp)
target_include_directories(unitree_arm_adapter_core PUBLIC include)
target_link_libraries(unitree_arm_adapter_core PUBLIC Threads::Threads)
target_compile_options(unitree_arm_adapter_core PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_library(unitree_arm_adapter_hil_recording
    src/hil_recording_sink.cpp
    src/hil_state_cache.cpp
    src/hil_supervisor_dispatcher.cpp)
target_include_directories(unitree_arm_adapter_hil_recording PUBLIC include)
target_link_libraries(unitree_arm_adapter_hil_recording PUBLIC
    unitree_arm_adapter_core)
target_compile_options(unitree_arm_adapter_hil_recording PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

# This executable deliberately links only the local core and recording sink.
# The optional device SDK subtree below is not part of its dependency graph.
add_executable(unitree_arm_adapter_hil src/hil_main.cpp)
target_link_libraries(unitree_arm_adapter_hil PRIVATE
    unitree_arm_adapter_hil_recording)
target_compile_options(unitree_arm_adapter_hil PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(unitree_arm_adapter_dry_run src/dry_run_main.cpp)
target_link_libraries(unitree_arm_adapter_dry_run PRIVATE unitree_arm_adapter_core)
target_compile_options(unitree_arm_adapter_dry_run PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(UNITREE_ARM_ADAPTER_BUILD_DDS)
    message(FATAL_ERROR
        "真实hardware output尚未获授权；本阶段不存在DDS命令publisher target")
endif()

if(UNITREE_ARM_ADAPTER_BUILD_STATE_BRIDGE)
    if(NOT EXISTS "${UNITREE_SDK2_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "UNITREE_SDK2_DIR无效: ${UNITREE_SDK2_DIR}")
    endif()
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${UNITREE_SDK2_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/unitree_sdk2" EXCLUDE_FROM_ALL)
endif()

if(UNITREE_ARM_ADAPTER_BUILD_STATE_BRIDGE)
    # Shadow integration uses a separate state-only executable.  It does not
    # include LowCmd or instantiate any publisher, so no CLI flag can enable
    # actuation in this process.
    add_executable(unitree_arm_state_bridge src/state_bridge_main.cpp)
    target_link_libraries(unitree_arm_state_bridge PRIVATE
        unitree_arm_adapter_core unitree_sdk2)
    target_compile_options(unitree_arm_state_bridge PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_executable(unitree_arm_adapter_core_test tests/core_test.cpp)
    target_link_libraries(unitree_arm_adapter_core_test PRIVATE
        unitree_arm_adapter_core)
    target_compile_options(unitree_arm_adapter_core_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_test(NAME unitree_arm_adapter_core_test
        COMMAND unitree_arm_adapter_core_test)

    add_executable(unitree_arm_adapter_hardware_supervisor_test
        tests/hardware_command_supervisor_test.cpp)
    target_link_libraries(unitree_arm_adapter_hardware_supervisor_test PRIVATE
        unitree_arm_adapter_core)
    target_compile_options(unitree_arm_adapter_hardware_supervisor_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_test(NAME unitree_arm_adapter_hardware_supervisor_test
        COMMAND unitree_arm_adapter_hardware_supervisor_test)

    add_executable(unitree_arm_adapter_hil_recording_test
        tests/hil_recording_sink_test.cpp)
    target_link_libraries(unitree_arm_adapter_hil_recording_test PRIVATE
        unitree_arm_adapter_hil_recording)
    target_compile_options(unitree_arm_adapter_hil_recording_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_test(NAME unitree_arm_adapter_hil_recording_test
        COMMAND unitree_arm_adapter_hil_recording_test)

    add_executable(unitree_arm_adapter_hil_state_cache_test
        tests/hil_state_cache_test.cpp)
    target_link_libraries(unitree_arm_adapter_hil_state_cache_test PRIVATE
        unitree_arm_adapter_hil_recording)
    target_compile_options(unitree_arm_adapter_hil_state_cache_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_test(NAME unitree_arm_adapter_hil_state_cache_test
        COMMAND unitree_arm_adapter_hil_state_cache_test)

    add_executable(unitree_arm_adapter_hil_supervisor_dispatcher_test
        tests/hil_supervisor_dispatcher_test.cpp)
    target_link_libraries(
        unitree_arm_adapter_hil_supervisor_dispatcher_test PRIVATE
        unitree_arm_adapter_hil_recording)
    target_compile_options(
        unitree_arm_adapter_hil_supervisor_dispatcher_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_test(NAME unitree_arm_adapter_hil_supervisor_dispatcher_test
        COMMAND unitree_arm_adapter_hil_supervisor_dispatcher_test)

    add_executable(unitree_arm_adapter_hil_rejection_matrix_test
        tests/hil_rejection_matrix_test.cpp)
    target_link_libraries(
        unitree_arm_adapter_hil_rejection_matrix_test PRIVATE
        unitree_arm_adapter_hil_recording)
    target_compile_options(
        unitree_arm_adapter_hil_rejection_matrix_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_test(NAME unitree_arm_adapter_hil_rejection_matrix_test
        COMMAND unitree_arm_adapter_hil_rejection_matrix_test)

    add_executable(unitree_arm_adapter_hil_fixture_writer
        tests/hil_fixture_writer.cpp)
    target_link_libraries(unitree_arm_adapter_hil_fixture_writer PRIVATE
        unitree_arm_adapter_core)
    target_compile_options(unitree_arm_adapter_hil_fixture_writer PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_test(NAME unitree_arm_adapter_hil_end_to_end_test
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_hil_end_to_end.py
            $<TARGET_FILE:unitree_arm_adapter_hil>
            $<TARGET_FILE:unitree_arm_adapter_hil_fixture_writer>)

    add_test(NAME unitree_arm_adapter_hil_capability_isolation
        COMMAND ${CMAKE_COMMAND}
            -DHIL_BINARY=$<TARGET_FILE:unitree_arm_adapter_hil>
            -DHIL_MAIN_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/hil_main.cpp
            -DHIL_SINK_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/hil_recording_sink.cpp
            -DHIL_CACHE_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/hil_state_cache.cpp
            -DHIL_DISPATCH_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/hil_supervisor_dispatcher.cpp
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_hil_isolation.cmake)

    add_test(NAME unitree_arm_adapter_hil_period_contract
        COMMAND ${CMAKE_COMMAND}
            -DHIL_BINARY=$<TARGET_FILE:unitree_arm_adapter_hil>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_hil_period_contract.cmake)
endif()
