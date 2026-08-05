#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${RIGHT_ARM_SIM_RUNTIME_BUILD_DIR:-/tmp/hold-my-beer-mpc-right-arm-sim-runtime-build}"
conda_bin="${CONDA_BIN:-/home/fjk/miniforge3/bin/conda}"
conda_env_name="${CONDA_ENV_NAME:-g1_mpc}"

env_prefix="$($conda_bin run -n "$conda_env_name" python -c 'import sys; print(sys.prefix)')"
mujoco_root="$($conda_bin run -n "$conda_env_name" python -c 'import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)')"

cmake -S "$script_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$env_prefix" \
    -DRIGHT_ARM_SIM_RUNTIME_TOOLCHAIN_LIB_DIR="$env_prefix/lib" \
    -DMUJOCO_ROOT="$mujoco_root"
cmake --build "$build_dir" --parallel
LD_LIBRARY_PATH="$env_prefix/lib:$mujoco_root:${LD_LIBRARY_PATH:-}" \
    ctest --test-dir "$build_dir" --output-on-failure
LD_LIBRARY_PATH="$env_prefix/lib:$mujoco_root:${LD_LIBRARY_PATH:-}" \
    "$build_dir/right_arm_sim_runtime_worker" --print-layout
