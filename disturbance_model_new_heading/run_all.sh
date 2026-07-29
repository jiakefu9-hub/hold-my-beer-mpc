#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_BIN="/home/fjk/miniforge3/bin/conda"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hold-my-beer-mpc-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

echo "[1/3] 将现有 W 原始数据转换到 H 系"
"$CONDA_BIN" run --no-capture-output -n g1_mpc \
    python "$SCRIPT_DIR/convert_world_to_heading.py" \
    --input "$REPO_DIR/disturbance_model_new/torso_disturbance_straight.npz"

echo "[2/3] 生成 raw / half-smoothed / fully-smoothed H 模板"
"$CONDA_BIN" run --no-capture-output -n g1_mpc \
    python "$SCRIPT_DIR/build_heading_disturbance_templates.py"

echo "[3/3] 对比新 H 模板与原有 W 模板"
"$CONDA_BIN" run --no-capture-output -n g1_mpc \
    python "$SCRIPT_DIR/compare_heading_world_templates.py" \
    --world-template-dir "$REPO_DIR/disturbance_model_new/templates_world"

echo "H-frame 模板生成与对比全部完成。"
