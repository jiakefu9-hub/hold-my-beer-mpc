cmake_minimum_required(VERSION 3.16)
project(g1_commissioning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(G1_COMMISSIONING_BUILD_DEVICE_QUERY
    "Build the networked A1 getter-only query executable" OFF)
option(G1_COMMISSIONING_BUILD_REAL_OUTPUT
    "Build the explicit A2/A3 rt/arm_sdk output executables" OFF)
option(G1_COMMISSIONING_BUILD_MODE_STEP
    "Build explicit A1b one-shot damping/locked-standing RPC tool" OFF)
set(UNITREE_SDK2_DIR "/home/fjk/g1_ws/unitree_sdk2" CACHE PATH
    "Pinned external Unitree SDK2 source directory")

add_library(g1_commissioning_core src/core.cpp)
target_include_directories(g1_commissioning_core PUBLIC include)
target_compile_options(g1_commissioning_core PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(g1_arm_static_preview src/preview_main.cpp)
target_link_libraries(g1_arm_static_preview PRIVATE g1_commissioning_core)
target_compile_options(g1_arm_static_preview PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(G1_COMMISSIONING_BUILD_DEVICE_QUERY OR G1_COMMISSIONING_BUILD_REAL_OUTPUT
   OR G1_COMMISSIONING_BUILD_MODE_STEP)
    if(NOT EXISTS "${UNITREE_SDK2_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "UNITREE_SDK2_DIR is invalid: ${UNITREE_SDK2_DIR}")
    endif()
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${UNITREE_SDK2_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/unitree_sdk2"
        EXCLUDE_FROM_ALL)

    add_library(g1_commissioning_device_state src/device_state.cpp)
    target_include_directories(g1_commissioning_device_state PUBLIC include)
    target_link_libraries(g1_commissioning_device_state PUBLIC
        g1_commissioning_core unitree_sdk2)
    target_compile_options(g1_commissioning_device_state PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(G1_COMMISSIONING_BUILD_DEVICE_QUERY)
    add_executable(g1_commissioning_query src/query_main.cpp)
    target_link_libraries(g1_commissioning_query PRIVATE
        g1_commissioning_device_state unitree_sdk2)
    target_compile_options(g1_commissioning_query PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_executable(g1_imu_zero_observer src/imu_zero_observer_main.cpp)
    target_link_libraries(g1_imu_zero_observer PRIVATE
        g1_commissioning_device_state unitree_sdk2)
    target_compile_options(g1_imu_zero_observer PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(G1_COMMISSIONING_BUILD_MODE_STEP)
    add_executable(g1_commissioning_mode_step src/mode_step_main.cpp)
    target_link_libraries(g1_commissioning_mode_step PRIVATE
        g1_commissioning_device_state unitree_sdk2)
    target_compile_options(g1_commissioning_mode_step PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(G1_COMMISSIONING_BUILD_REAL_OUTPUT)
    add_executable(g1_arm_stop_observe src/arm_stop_observe_main.cpp)
    target_link_libraries(g1_arm_stop_observe PRIVATE
        g1_commissioning_device_state unitree_sdk2)
    add_executable(g1_arm_static_execute src/arm_static_execute_main.cpp)
    target_link_libraries(g1_arm_static_execute PRIVATE
        g1_commissioning_device_state unitree_sdk2)
    target_compile_options(g1_arm_static_execute PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_executable(g1_arm_balance_hold_execute src/arm_static_execute_main.cpp)
    target_compile_definitions(g1_arm_balance_hold_execute PRIVATE
        G1_A3_BALANCE_HOLD_EXECUTABLE=1)
    target_link_libraries(g1_arm_balance_hold_execute PRIVATE
        g1_commissioning_device_state unitree_sdk2)
    target_compile_options(g1_arm_balance_hold_execute PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

include(CTest)
if(BUILD_TESTING)
    if(G1_COMMISSIONING_BUILD_REAL_OUTPUT)
        add_executable(g1_arm_stop_device_test tests/arm_stop_device_test.cpp)
        target_link_libraries(g1_arm_stop_device_test PRIVATE
            g1_commissioning_device_state unitree_sdk2)
        add_test(NAME g1_arm_stop_device_decode COMMAND g1_arm_stop_device_test)
    endif()
    add_executable(g1_arm_stop_interlock_test tests/arm_stop_interlock_test.cpp)
    target_include_directories(g1_arm_stop_interlock_test PRIVATE include)
    add_test(NAME g1_arm_stop_interlock COMMAND g1_arm_stop_interlock_test)
    add_executable(g1_commissioning_mode_step_test tests/mode_step_test.cpp)
    target_include_directories(g1_commissioning_mode_step_test PRIVATE include)
    add_test(NAME g1_commissioning_mode_step
        COMMAND g1_commissioning_mode_step_test)

    add_executable(g1_commissioning_core_test tests/core_test.cpp)
    target_link_libraries(g1_commissioning_core_test PRIVATE
        g1_commissioning_core)
    target_compile_options(g1_commissioning_core_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

    set(test_profile "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/synthetic_profile.conf")
    set(test_state "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/synthetic_state.conf")
    set(test_a3_profile "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/synthetic_a3_profile.conf")
    add_test(NAME g1_profile_migration_rejection
        COMMAND "${CMAKE_COMMAND}"
            -DPREVIEW=$<TARGET_FILE:g1_arm_static_preview>
            -DPROFILE=${test_profile}
            -DSTATE=${test_state}
            -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_profile_migration.cmake)
    add_test(NAME g1_commissioning_core
        COMMAND g1_commissioning_core_test "${test_profile}" "${test_a3_profile}" "${test_state}")
    add_test(NAME g1_commissioning_preview
        COMMAND "${CMAKE_COMMAND}"
            -DPREVIEW=$<TARGET_FILE:g1_arm_static_preview>
            -DPROFILE=${test_profile}
            -DSTATE=${test_state}
            -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/synthetic_preview.jsonl
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_preview_test.cmake)
    add_test(NAME g1_commissioning_source_isolation
        COMMAND "${CMAKE_COMMAND}"
            -DCORE=${CMAKE_CURRENT_SOURCE_DIR}/src/core.cpp
            -DPREVIEW=${CMAKE_CURRENT_SOURCE_DIR}/src/preview_main.cpp
            -DQUERY=${CMAKE_CURRENT_SOURCE_DIR}/src/query_main.cpp
            -DIMU_OBSERVER=${CMAKE_CURRENT_SOURCE_DIR}/src/imu_zero_observer_main.cpp
            -DMODE_STEP=${CMAKE_CURRENT_SOURCE_DIR}/src/mode_step_main.cpp
            -DEXECUTE=${CMAKE_CURRENT_SOURCE_DIR}/src/arm_static_execute_main.cpp
            -DFSM_MONITOR=${CMAKE_CURRENT_SOURCE_DIR}/include/g1_commissioning/device_fsm_monitor.hpp
            -DSTOP_OBSERVER=${CMAKE_CURRENT_SOURCE_DIR}/src/arm_stop_observe_main.cpp
            -DCMAKE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
            -DADAPTER_CMAKE=${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/unitree_arm_adapter/CMakeLists.txt
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_source_isolation.cmake)
    if(G1_COMMISSIONING_BUILD_MODE_STEP)
        add_test(NAME g1_commissioning_mode_step_cli_rejection
            COMMAND "${CMAKE_COMMAND}"
                -DEXECUTABLE=$<TARGET_FILE:g1_commissioning_mode_step>
                -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/forbidden_mode.jsonl
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_mode_step_cli.cmake)
    endif()
endif()
