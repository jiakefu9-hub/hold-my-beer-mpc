cmake_minimum_required(VERSION 3.16)

project(right_arm_rnea LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pinocchio 4.1 REQUIRED)

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

add_library(right_arm_rnea SHARED
    src/right_arm_rnea.cpp
)
target_include_directories(right_arm_rnea
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${MUJOCO_INCLUDE_DIR}
)
target_link_libraries(right_arm_rnea
    PRIVATE
        # RNEA 在 default 中，MJCF 解析器在 parsers 中；不链接并行/可视化
        # 等无关组件，避免给 2 ms 路径引入额外运行时依赖。
        pinocchio::pinocchio_parsers
        ${MUJOCO_LIBRARY}
)
target_compile_definitions(right_arm_rnea PRIVATE RIGHT_ARM_RNEA_BUILDING_LIBRARY)
target_compile_options(right_arm_rnea
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(right_arm_rnea PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    BUILD_RPATH "${MUJOCO_ROOT};${CMAKE_PREFIX_PATH}/lib"
)
