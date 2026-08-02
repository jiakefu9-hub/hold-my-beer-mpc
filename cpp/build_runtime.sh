#!/usr/bin/env bash
set -euo pipefail

# 【非核心代码】只构建仿真在线调用的三个共享库；单元测试和基准由各
# 子目录 build_and_test.sh 负责，避免每次 run.sh 重跑长测试。
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/fjk/miniforge3/bin/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-g1_mpc}"
RNEA_BUILD_DIR="${RIGHT_ARM_RNEA_BUILD_DIR:-/tmp/hold-my-beer-mpc-right-arm-rnea-build}"
EXECUTOR_BUILD_DIR="${RIGHT_ARM_EXECUTOR_BUILD_DIR:-/tmp/hold-my-beer-mpc-right-arm-executor-build}"
MAPPER_BUILD_DIR="${DDQ_TORQUE_MAPPER_BUILD_DIR:-/tmp/hold-my-beer-mpc-ddq-torque-mapper-build}"

ENV_PREFIX="$($CONDA_BIN run -n "$CONDA_ENV_NAME" python -c 'import sys; print(sys.prefix)')"
MUJOCO_ROOT="$($CONDA_BIN run -n "$CONDA_ENV_NAME" python -c 'import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)')"

if [[ ! -f "$RNEA_BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S "$REPO_DIR/cpp/right_arm_rnea" -B "$RNEA_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$ENV_PREFIX" \
        -DMUJOCO_ROOT="$MUJOCO_ROOT"
fi
cmake --build "$RNEA_BUILD_DIR" --parallel

if [[ ! -f "$EXECUTOR_BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S "$REPO_DIR/cpp/right_arm_executor" -B "$EXECUTOR_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DRAE_BUILD_BENCHMARK=OFF
fi
cmake --build "$EXECUTOR_BUILD_DIR" --parallel

if [[ ! -f "$MAPPER_BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S "$REPO_DIR/cpp/ddq_torque_mapper" -B "$MAPPER_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$ENV_PREFIX" \
        -DMUJOCO_ROOT="$MUJOCO_ROOT"
fi
cmake --build "$MAPPER_BUILD_DIR" --parallel
