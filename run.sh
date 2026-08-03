#!/usr/bin/env bash
set -euo pipefail

# 【非核心代码】固定使用已经安装 MuJoCo/Torch/OSQP 的项目环境。
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hold-my-beer-mpc-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

RUN_ARGS=("$@")

# 构建目录可配置时，加载路径必须与它们同步，避免误加载 /tmp
# 中上一次的共享库。
export RIGHT_ARM_RNEA_LIBRARY="${RIGHT_ARM_RNEA_LIBRARY:-${RIGHT_ARM_RNEA_BUILD_DIR:-/tmp/hold-my-beer-mpc-right-arm-rnea-build}/libright_arm_rnea.so}"
export RIGHT_ARM_EXECUTOR_LIBRARY="${RIGHT_ARM_EXECUTOR_LIBRARY:-${RIGHT_ARM_EXECUTOR_BUILD_DIR:-/tmp/hold-my-beer-mpc-right-arm-executor-build}/libright_arm_executor.so}"
export DDQ_TORQUE_MAPPER_LIBRARY="${DDQ_TORQUE_MAPPER_LIBRARY:-${DDQ_TORQUE_MAPPER_BUILD_DIR:-/tmp/hold-my-beer-mpc-ddq-torque-mapper-build}/libddq_torque_mapper.so}"

# C++ RNEA 和 2 ms 安全执行器是当前默认在线后端；增量构建确保从全新
# checkout 直接运行时不会因为缺少 /tmp 中的共享库失败。
"$REPO_DIR/cpp/build_runtime.sh"
# 服务器没有可用 X11 时自动退化为 headless；桌面环境仍会正常打开 viewer。
if ! xdpyinfo -display "${DISPLAY:-}" >/dev/null 2>&1; then
    echo "[run.sh] 未检测到可用显示器，自动使用 headless 模式并关闭视频。"
    RUN_ARGS=(--headless --no-video "${RUN_ARGS[@]}")
fi

exec /home/fjk/miniforge3/bin/conda run --no-capture-output -n g1_mpc \
    python "$REPO_DIR/main_sim.py" g1.yaml "${RUN_ARGS[@]}"
