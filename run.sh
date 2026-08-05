#!/usr/bin/env bash
set -euo pipefail

# 【非核心代码】固定使用已经安装 MuJoCo/Torch/OSQP 的项目环境。
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hold-my-beer-mpc-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

RUN_ARGS=("$@")

# 【非核心代码】实时性能实验固定为单线程数值库，避免 BLAS/OpenMP
# 在很小的矩阵上临时创建线程并引入不可重复的调度长尾。
CONTROL_NUM_THREADS="${MPC_CONTROL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="$CONTROL_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$CONTROL_NUM_THREADS"
export MKL_NUM_THREADS="$CONTROL_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$CONTROL_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$CONTROL_NUM_THREADS"
export BLIS_NUM_THREADS="$CONTROL_NUM_THREADS"
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# 构建目录可配置时，加载路径必须与它们同步，避免误加载 /tmp
# 中上一次的共享库。
export RIGHT_ARM_RNEA_LIBRARY="${RIGHT_ARM_RNEA_LIBRARY:-${RIGHT_ARM_RNEA_BUILD_DIR:-/tmp/hold-my-beer-mpc-right-arm-rnea-build}/libright_arm_rnea.so}"
export RIGHT_ARM_EXECUTOR_LIBRARY="${RIGHT_ARM_EXECUTOR_LIBRARY:-${RIGHT_ARM_EXECUTOR_BUILD_DIR:-/tmp/hold-my-beer-mpc-right-arm-executor-build}/libright_arm_executor.so}"
export DDQ_TORQUE_MAPPER_LIBRARY="${DDQ_TORQUE_MAPPER_LIBRARY:-${DDQ_TORQUE_MAPPER_BUILD_DIR:-/tmp/hold-my-beer-mpc-ddq-torque-mapper-build}/libddq_torque_mapper.so}"
export RIGHT_ARM_SIM_RUNTIME_WORKER="${RIGHT_ARM_SIM_RUNTIME_WORKER:-${RIGHT_ARM_SIM_RUNTIME_BUILD_DIR:-/tmp/hold-my-beer-mpc-right-arm-sim-runtime-build}/right_arm_sim_runtime_worker}"

# C++ RNEA 和 2 ms 安全执行器是当前默认在线后端；增量构建确保从全新
# checkout 直接运行时不会因为缺少 /tmp 中的共享库失败。
"$REPO_DIR/cpp/build_runtime.sh"
# 服务器没有可用 X11 时自动退化为 headless；桌面环境仍会正常打开 viewer。
if ! xdpyinfo -display "${DISPLAY:-}" >/dev/null 2>&1; then
    echo "[run.sh] 未检测到可用显示器，自动使用 headless 模式并关闭视频。"
    RUN_ARGS=(--headless --no-video "${RUN_ARGS[@]}")
fi

PYTHON_COMMAND=(
    /home/fjk/miniforge3/bin/conda run --no-capture-output -n g1_mpc
    python "$REPO_DIR/main_sim.py" g1.yaml "${RUN_ARGS[@]}"
)

# 默认选择最高频率组中编号最大的逻辑 CPU；本机对应 4.5 GHz 性能核。
# 可用 MPC_CONTROL_CPU=<id> 明确覆盖，或设为 none 关闭亲和性绑定。
CONTROL_CPU="${MPC_CONTROL_CPU:-auto}"
if [[ "$CONTROL_CPU" == "auto" ]] && command -v lscpu >/dev/null 2>&1; then
    CONTROL_CPU="$(
        lscpu -p=CPU,MAXMHZ 2>/dev/null \
            | awk -F, '!/^#/ && $2 != "" {freq=$2+0; if (freq > max || (freq == max && $1+0 > cpu)) {max=freq; cpu=$1+0}} END {if (max > 0) print cpu}'
    )"
fi

if [[ -n "$CONTROL_CPU" && "$CONTROL_CPU" != "none" ]] \
    && command -v taskset >/dev/null 2>&1 \
    && taskset -c "$CONTROL_CPU" true >/dev/null 2>&1; then
    echo "[run.sh] 性能稳定模式：CPU=$CONTROL_CPU，数值库线程=$CONTROL_NUM_THREADS。"
    exec taskset -c "$CONTROL_CPU" "${PYTHON_COMMAND[@]}"
fi

echo "[run.sh] 未启用 CPU 亲和性，数值库线程=$CONTROL_NUM_THREADS。"
exec "${PYTHON_COMMAND[@]}"
