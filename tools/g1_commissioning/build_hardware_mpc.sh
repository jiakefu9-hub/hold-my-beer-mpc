#!/usr/bin/env bash
# Local C++ kinematics only. No Unitree executable, network or robot command.
set -euo pipefail
MPC_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MPC_PYTHON="${G1_MPC_PYTHON:-/home/fjk/miniforge3/envs/g1_mpc/bin/python}"
MPC_PREFIX="$($MPC_PYTHON -c 'import sys; print(sys.prefix)')"
MPC_MUJOCO="$($MPC_PYTHON -c 'import pathlib,mujoco; print(pathlib.Path(mujoco.__file__).parent)')"
cmake -S "$MPC_REPO_DIR/cpp/right_arm_rnea" -B "$MPC_REPO_DIR/build/right_arm_rnea" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$MPC_PREFIX" -DMUJOCO_ROOT="$MPC_MUJOCO"
cmake --build "$MPC_REPO_DIR/build/right_arm_rnea" --parallel 4
"$MPC_PYTHON" "$MPC_REPO_DIR/tools/g1_commissioning/g1_walk_mpc.py" --preflight
