#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${RIGHT_ARM_EXECUTOR_BUILD_DIR:-/tmp/hold-my-beer-mpc-right-arm-executor-build}"

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DRAE_BUILD_BENCHMARK=ON
cmake --build "$BUILD_DIR" --parallel
ctest --test-dir "$BUILD_DIR" --output-on-failure
"$BUILD_DIR/right_arm_executor_example"
"$BUILD_DIR/right_arm_executor_benchmark" \
    "${RAE_BENCHMARK_ITERATIONS:-500000}"

printf 'shared library: %s\n' "$BUILD_DIR/libright_arm_executor.so"
