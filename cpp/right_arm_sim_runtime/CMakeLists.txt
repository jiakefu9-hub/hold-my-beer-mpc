cmake_minimum_required(VERSION 3.16)
project(right_arm_sim_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

# Pinocchio/Boost来自conda并可能由比系统GCC更新的libstdc++构建。显式把
# 同一环境的运行库放在链接搜索首位，避免可执行文件误连Ubuntu旧版本。
set(RIGHT_ARM_SIM_RUNTIME_TOOLCHAIN_LIB_DIR "" CACHE PATH
    "Directory containing the libstdc++ used by Pinocchio")
if(RIGHT_ARM_SIM_RUNTIME_TOOLCHAIN_LIB_DIR)
    link_directories(BEFORE "${RIGHT_ARM_SIM_RUNTIME_TOOLCHAIN_LIB_DIR}")
    list(PREPEND CMAKE_BUILD_RPATH
        "${RIGHT_ARM_SIM_RUNTIME_TOOLCHAIN_LIB_DIR}")
endif()

set(MUJOCO_ROOT "" CACHE PATH "MuJoCo Python package directory")
find_path(
    MUJOCO_INCLUDE_DIR mujoco/mujoco.h
    HINTS "${MUJOCO_ROOT}/include" REQUIRED)
find_library(
    MUJOCO_LIBRARY NAMES mujoco libmujoco.so.3.6.0
    HINTS "${MUJOCO_ROOT}" NO_DEFAULT_PATH REQUIRED)

# 三个既有原生模块保持各自C ABI；runtime不复制其算法实现。
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../right_arm_rnea"
    "${CMAKE_CURRENT_BINARY_DIR}/right_arm_rnea")
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../ddq_torque_mapper"
    "${CMAKE_CURRENT_BINARY_DIR}/ddq_torque_mapper")
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../right_arm_executor"
    "${CMAKE_CURRENT_BINARY_DIR}/right_arm_executor")

add_library(right_arm_sim_runtime_core
    src/runtime.cpp
    src/shared_memory.cpp)
target_include_directories(right_arm_sim_runtime_core
    PUBLIC include
    PRIVATE "${MUJOCO_INCLUDE_DIR}")
target_link_libraries(right_arm_sim_runtime_core
    PUBLIC
        right_arm_rnea
        ddq_torque_mapper
        right_arm_executor_shared
        Threads::Threads
        "${MUJOCO_LIBRARY}")
target_compile_definitions(right_arm_sim_runtime_core PRIVATE
    RIGHT_ARM_SIM_RUNTIME_MUJOCO_PLUGIN_DIR="${MUJOCO_ROOT}/plugin")
target_compile_options(right_arm_sim_runtime_core PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(right_arm_sim_runtime_worker src/worker_main.cpp)
target_link_libraries(
    right_arm_sim_runtime_worker PRIVATE right_arm_sim_runtime_core)
target_compile_options(right_arm_sim_runtime_worker PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

include(CTest)
if(BUILD_TESTING)
    add_executable(right_arm_sim_runtime_test tests/runtime_test.cpp)
    target_include_directories(right_arm_sim_runtime_test PRIVATE
        "${MUJOCO_INCLUDE_DIR}")
    target_link_libraries(right_arm_sim_runtime_test PRIVATE
        right_arm_sim_runtime_core "${MUJOCO_LIBRARY}")
    target_compile_options(right_arm_sim_runtime_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_test(
        NAME right_arm_sim_runtime_test
        COMMAND right_arm_sim_runtime_test
            $<TARGET_FILE:right_arm_sim_runtime_worker>
            "${CMAKE_CURRENT_SOURCE_DIR}/../../resources/g1_description/scene.xml")
endif()
