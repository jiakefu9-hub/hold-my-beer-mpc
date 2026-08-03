cmake_minimum_required(VERSION 3.16)

project(right_arm_executor VERSION 1.0.0 LANGUAGES C CXX)

include(GNUInstallDirs)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

function(rae_enable_warnings target_name)
    target_compile_options(${target_name}
        PRIVATE
            $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    )
endfunction()

add_library(right_arm_executor_core
    src/right_arm_executor.cpp
)
target_include_directories(right_arm_executor_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(right_arm_executor_core PUBLIC cxx_std_17)
set_target_properties(right_arm_executor_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
rae_enable_warnings(right_arm_executor_core)

# 【核心代码】稳定 C ABI 的共享库；最终文件名为 libright_arm_executor.so.1。
add_library(right_arm_executor_shared SHARED
    src/right_arm_executor_c.cpp
)
target_link_libraries(right_arm_executor_shared
    PRIVATE right_arm_executor_core
)
target_include_directories(right_arm_executor_shared
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(right_arm_executor_shared PUBLIC cxx_std_17)
target_compile_definitions(right_arm_executor_shared PRIVATE RAE_BUILDING_LIBRARY)
set_target_properties(right_arm_executor_shared PROPERTIES
    OUTPUT_NAME right_arm_executor
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
rae_enable_warnings(right_arm_executor_shared)

add_executable(right_arm_executor_example
    src/main.cpp
)
target_link_libraries(right_arm_executor_example PRIVATE right_arm_executor_core)
rae_enable_warnings(right_arm_executor_example)

option(RAE_BUILD_BENCHMARK "Build the fixed-input C ABI microbenchmark" ON)
if(RAE_BUILD_BENCHMARK)
    add_executable(right_arm_executor_benchmark
        benchmarks/right_arm_executor_benchmark.cpp
    )
    target_link_libraries(
        right_arm_executor_benchmark PRIVATE right_arm_executor_shared
    )
    target_compile_features(right_arm_executor_benchmark PRIVATE cxx_std_17)
    rae_enable_warnings(right_arm_executor_benchmark)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(right_arm_executor_test
        tests/right_arm_executor_test.cpp
    )
    target_link_libraries(right_arm_executor_test PRIVATE right_arm_executor_core)
    rae_enable_warnings(right_arm_executor_test)
    add_test(NAME right_arm_executor_test COMMAND right_arm_executor_test)

    # 这个测试文件由 C 编译器编译，直接验证头文件和符号不是伪 C ABI。
    add_executable(right_arm_executor_c_test
        tests/right_arm_executor_c_test.c
    )
    target_link_libraries(
        right_arm_executor_c_test PRIVATE right_arm_executor_shared
    )
    set_target_properties(
        right_arm_executor_c_test PROPERTIES LINKER_LANGUAGE CXX
    )
    rae_enable_warnings(right_arm_executor_c_test)
    add_test(NAME right_arm_executor_c_test COMMAND right_arm_executor_c_test)
endif()

install(
    TARGETS right_arm_executor_shared
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    FILES
        include/right_arm_executor/right_arm_executor.hpp
        include/right_arm_executor/right_arm_executor_c.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/right_arm_executor
)
