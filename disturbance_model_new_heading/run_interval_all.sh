#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_BIN="/home/fjk/miniforge3/bin/conda"
WORLD_PREFIX="$REPO_DIR/disturbance_model_new/torso_disturbance_straight_interval"
HEADING_PREFIX="$SCRIPT_DIR/torso_disturbance_heading_interval"
TEMPLATE_DIR="$SCRIPT_DIR/templates_heading_interval"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hold-my-beer-mpc-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

echo "[1/3] 前台重新采集带 IMU 世界系线速度的 step 前原始数据"
"$CONDA_BIN" run --no-capture-output -n g1_mpc \
    python "$REPO_DIR/disturbance_model_new/collect_torso_disturbance_and_check_yaw.py" \
    g1.yaml \
    --output-prefix "$WORLD_PREFIX"

echo "[2/3] 将新原始数据转换到 H 系"
"$CONDA_BIN" run --no-capture-output -n g1_mpc \
    python "$SCRIPT_DIR/convert_world_to_heading.py" \
    --input "$WORLD_PREFIX.npz" \
    --output-prefix "$HEADING_PREFIX"

echo "[3/3] 生成 2 ms 相位网格、未来 6 ms 滑动区间模板"
"$CONDA_BIN" run --no-capture-output -n g1_mpc \
    python "$SCRIPT_DIR/build_heading_disturbance_templates.py" \
    --input "$HEADING_PREFIX.npz" \
    --output-dir "$TEMPLATE_DIR" \
    --num-bins 400 \
    --control-dt 0.006 \
    --node-window-size 21 \
    --interval-window-size 3

echo "6 ms H-frame 区间模板生成完成: $TEMPLATE_DIR"
