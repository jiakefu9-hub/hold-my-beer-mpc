#!/usr/bin/env bash
set -euo pipefail

# 【非核心代码】固定使用已经安装 MuJoCo/Torch/OSQP 的项目环境。
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hold-my-beer-mpc-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

RUN_ARGS=("$@")
G1_MPC_CONDA="${G1_MPC_CONDA:-/home/fjk/miniforge3/bin/conda}"
G1_MPC_PYTHON="${G1_MPC_PYTHON:-/home/fjk/miniforge3/envs/g1_mpc/bin/python}"

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
export DISTURBANCE_LAB_FORMAL_LAUNCHER="disturbance_lab_run_sh"

# 正式 full-task 入口不接受 auto CPU 或多线程数值库。这样即使调用者
# 忘记 taskset，Python 侧也会在第一个 mj_step 前再次 fail closed。
FORMAL_FULL_TASK=0
for argument in "${RUN_ARGS[@]}"; do
    if [[ "$argument" == "--full-task-smoke" ]]; then
        FORMAL_FULL_TASK=1
        break
    fi
done
if [[ "$FORMAL_FULL_TASK" == "1" ]]; then
    if [[ "${MPC_CONTROL_CPU:-}" != "7" ]]; then
        echo "[run.sh] 正式 full-task 要求显式设置 MPC_CONTROL_CPU=7。" >&2
        exit 2
    fi
    if [[ "$CONTROL_NUM_THREADS" != "1" ]]; then
        echo "[run.sh] 正式 full-task 要求 MPC_CONTROL_NUM_THREADS=1。" >&2
        exit 2
    fi
fi

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
    "$G1_MPC_CONDA" run --no-capture-output -n g1_mpc
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

# systemd transient service 只临时授予 RLIMIT_RTPRIO；这里把最终控制
# Python 及其锁步 C++ worker 放入同一个低优先级 SCHED_RR 调度类。
# 构建和批量实验 runner 仍是 SCHED_OTHER。guard 在 exec 前验证 governor、
# affinity、policy/priority 以及内核 RT throttling，任何缺项都拒绝运行。
if [[ "${MPC_REQUIRE_REALTIME:-0}" == "1" ]]; then
    REALTIME_POLICY="${MPC_REALTIME_POLICY:-SCHED_RR}"
    REALTIME_PRIORITY="${MPC_REALTIME_PRIORITY:-10}"
    if [[ "$REALTIME_POLICY" != "SCHED_RR" ]]; then
        echo "[run.sh] 安全 RT 模式当前只允许 SCHED_RR。" >&2
        exit 2
    fi
    if [[ ! "$REALTIME_PRIORITY" =~ ^[0-9]+$ ]] \
        || (( REALTIME_PRIORITY < 1 || REALTIME_PRIORITY > 20 )); then
        echo "[run.sh] 安全 RT 模式只允许 1..20 的低 RR 优先级。" >&2
        exit 2
    fi
    if [[ ! "$CONTROL_CPU" =~ ^[0-9]+$ ]]; then
        echo "[run.sh] RT 模式要求显式设置单个 MPC_CONTROL_CPU。" >&2
        exit 2
    fi
    REALTIME_GUARD_ARGS=(
        --expected-policy "$REALTIME_POLICY"
        --expected-priority "$REALTIME_PRIORITY"
        --expected-cpu "$CONTROL_CPU"
    )
    if [[ "${MPC_REQUIRE_TARGET_REALTIME:-0}" == "1" ]]; then
        REALTIME_GUARD_ARGS+=(--require-target-environment)
    fi
    PYTHON_COMMAND=(
        chrt --rr "$REALTIME_PRIORITY"
        "$G1_MPC_PYTHON"
        "$REPO_DIR/realtime_runtime.py"
        "${REALTIME_GUARD_ARGS[@]}"
        -- "${PYTHON_COMMAND[@]}"
    )
fi

if [[ -n "$CONTROL_CPU" && "$CONTROL_CPU" != "none" ]] \
    && command -v taskset >/dev/null 2>&1 \
    && taskset -c "$CONTROL_CPU" true >/dev/null 2>&1; then
    echo "[run.sh] 性能稳定模式：CPU=$CONTROL_CPU，数值库线程=$CONTROL_NUM_THREADS。"
    exec taskset -c "$CONTROL_CPU" "${PYTHON_COMMAND[@]}"
fi

echo "[run.sh] 未启用 CPU 亲和性，数值库线程=$CONTROL_NUM_THREADS。"
exec "${PYTHON_COMMAND[@]}"
