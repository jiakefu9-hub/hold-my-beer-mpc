#!/usr/bin/env bash
set -euo pipefail

build_dir="${UNITREE_ARM_ADAPTER_BUILD_DIR:-/tmp/hold-my-beer-mpc-unitree-arm-adapter-build}"
build_dds="${UNITREE_ARM_BUILD_DDS:-OFF}"
sdk_dir="${UNITREE_SDK2_DIR:-/home/fjk/g1_ws/unitree_sdk2}"

cmake -S "$(dirname "$0")" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUNITREE_ARM_ADAPTER_BUILD_DDS="$build_dds" \
  -DUNITREE_SDK2_DIR="$sdk_dir"
cmake --build "$build_dir" --parallel
ctest --test-dir "$build_dir" --output-on-failure
"$build_dir/unitree_arm_adapter_dry_run" \
  --shm-name "/g1_arm_mpc_build_test_$$" \
  --iterations 20 --synthetic-input --reset-shm --unlink-on-exit
