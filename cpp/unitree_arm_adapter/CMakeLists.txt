cmake_minimum_required(VERSION 3.16)
project(unitree_arm_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(UNITREE_ARM_ADAPTER_BUILD_DDS "构建Unitree SDK2 DDS适配器" OFF)
set(UNITREE_SDK2_DIR "/home/fjk/g1_ws/unitree_sdk2" CACHE PATH
    "Unitree SDK2源码目录")

find_package(Threads REQUIRED)

add_library(unitree_arm_adapter_core
    src/periodic_loop.cpp
    src/safety.cpp
    src/shared_memory.cpp)
target_include_directories(unitree_arm_adapter_core PUBLIC include)
target_link_libraries(unitree_arm_adapter_core PUBLIC Threads::Threads)
target_compile_options(unitree_arm_adapter_core PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(unitree_arm_adapter_dry_run src/dry_run_main.cpp)
target_link_libraries(unitree_arm_adapter_dry_run PRIVATE unitree_arm_adapter_core)
target_compile_options(unitree_arm_adapter_dry_run PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(UNITREE_ARM_ADAPTER_BUILD_DDS)
    if(NOT EXISTS "${UNITREE_SDK2_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "UNITREE_SDK2_DIR无效: ${UNITREE_SDK2_DIR}")
    endif()
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${UNITREE_SDK2_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/unitree_sdk2" EXCLUDE_FROM_ALL)
    add_executable(unitree_arm_adapter_dds src/dds_main.cpp)
    target_link_libraries(unitree_arm_adapter_dds PRIVATE
        unitree_arm_adapter_core unitree_sdk2)
    target_compile_options(unitree_arm_adapter_dds PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(unitree_arm_adapter_core_test tests/core_test.cpp)
    target_link_libraries(unitree_arm_adapter_core_test PRIVATE
        unitree_arm_adapter_core)
    target_compile_options(unitree_arm_adapter_core_test PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_test(NAME unitree_arm_adapter_core_test
        COMMAND unitree_arm_adapter_core_test)
endif()
