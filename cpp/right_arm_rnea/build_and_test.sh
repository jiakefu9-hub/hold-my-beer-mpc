#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${RIGHT_ARM_RNEA_BUILD_DIR:-/tmp/hold-my-beer-mpc-right-arm-rnea-build}"
CONDA_BIN="${CONDA_BIN:-/home/fjk/miniforge3/bin/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-g1_mpc}"

ENV_PREFIX="$($CONDA_BIN run -n "$CONDA_ENV_NAME" python -c 'import sys; print(sys.prefix)')"
MUJOCO_ROOT="$($CONDA_BIN run -n "$CONDA_ENV_NAME" python -c 'import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)')"

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$ENV_PREFIX" \
    -DMUJOCO_ROOT="$MUJOCO_ROOT"
cmake --build "$BUILD_DIR" --parallel

# 一键测试同时覆盖随机状态数值一致性、passive/friction 后处理和耗时。
LD_LIBRARY_PATH="$ENV_PREFIX/lib:$MUJOCO_ROOT:${LD_LIBRARY_PATH:-}" \
    "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV_NAME" \
    python "$SCRIPT_DIR/parity_benchmark.py" \
    --library "$BUILD_DIR/libright_arm_rnea.so" \
    --scene "$REPO_DIR/resources/g1_description/scene.xml" \
    "$@"
