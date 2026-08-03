#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${DDQ_TORQUE_MAPPER_BUILD_DIR:-/tmp/hold-my-beer-mpc-ddq-torque-mapper-build}"
CONDA_BIN="${CONDA_BIN:-/home/fjk/miniforge3/bin/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-g1_mpc}"

ENV_PREFIX="$($CONDA_BIN run -n "$CONDA_ENV_NAME" python -c 'import sys; print(sys.prefix)')"
MUJOCO_ROOT="$($CONDA_BIN run -n "$CONDA_ENV_NAME" python -c 'import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)')"

# 【非核心构建】Release 与实际在线路径使用同一优化级别；产物放 /tmp，
# 不污染源码目录，也不会把本机二进制误提交到 Git。
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$ENV_PREFIX" \
    -DMUJOCO_ROOT="$MUJOCO_ROOT"
cmake --build "$BUILD_DIR" --parallel

# 【核心验证】同一状态分别运行 Python 参考链与 C++ 全链，再报告 wall/core 耗时。
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hold-my-beer-mpc-matplotlib}" \
PYTHONDONTWRITEBYTECODE=1 \
LD_LIBRARY_PATH="$ENV_PREFIX/lib:$MUJOCO_ROOT:${LD_LIBRARY_PATH:-}" \
    "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV_NAME" \
    python "$SCRIPT_DIR/parity_benchmark.py" \
    --library "$BUILD_DIR/libddq_torque_mapper.so" \
    --scene "$REPO_DIR/resources/g1_description/scene.xml" \
    "$@"
