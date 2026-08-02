cmake_minimum_required(VERSION 3.16)

project(ddq_torque_mapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(MUJOCO_ROOT "" CACHE PATH "MuJoCo Python package directory")
find_path(
    MUJOCO_INCLUDE_DIR
    mujoco/mujoco.h
    HINTS "${MUJOCO_ROOT}/include"
    REQUIRED)
find_library(
    MUJOCO_LIBRARY
    NAMES mujoco libmujoco.so.3.6.0
    HINTS "${MUJOCO_ROOT}"
    NO_DEFAULT_PATH
    REQUIRED)

add_library(ddq_torque_mapper SHARED
    src/ddq_torque_mapper.cpp
)
target_include_directories(ddq_torque_mapper
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${MUJOCO_INCLUDE_DIR}
)
target_link_libraries(ddq_torque_mapper
    PRIVATE
        Eigen3::Eigen
        ${MUJOCO_LIBRARY}
)
target_compile_definitions(
    ddq_torque_mapper PRIVATE DDQ_TORQUE_MAPPER_BUILDING_LIBRARY)
target_compile_options(ddq_torque_mapper
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(ddq_torque_mapper PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    BUILD_RPATH "${MUJOCO_ROOT}"
)
