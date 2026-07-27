#!/usr/bin/env bash
set -euo pipefail

# 【非核心代码】固定使用已经安装 MuJoCo/Torch/OSQP 的项目环境。
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hold-my-beer-mpc-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

RUN_ARGS=("$@")
# 服务器没有可用 X11 时自动退化为 headless；桌面环境仍会正常打开 viewer。
if ! xdpyinfo -display "${DISPLAY:-}" >/dev/null 2>&1; then
    echo "[run.sh] 未检测到可用显示器，自动使用 headless 模式并关闭视频。"
    RUN_ARGS=(--headless --no-video "${RUN_ARGS[@]}")
fi

exec /home/fjk/miniforge3/bin/conda run --no-capture-output -n g1_mpc \
    python "$REPO_DIR/main_sim.py" g1.yaml "${RUN_ARGS[@]}"
