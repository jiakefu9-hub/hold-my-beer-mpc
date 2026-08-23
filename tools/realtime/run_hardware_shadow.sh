#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTROL_CPU="${MPC_CONTROL_CPU:-7}"
BRIDGE_CPU="${UNITREE_STATE_BRIDGE_CPU:-5}"
REALTIME_PRIORITY="${MPC_REALTIME_PRIORITY:-10}"
DURATION_S=30
PREDICTOR=template
GROUP="hardware_shadow_$(date +%Y%m%d_%H%M%S)"
NETWORK_INTERFACE=""
HARDWARE_CONFIG="configs/g1_hardware_shadow.yaml"
CONTROLLER_CONFIG="configs/g1.yaml"

usage() {
    cat >&2 <<'EOF'
Usage: run_hardware_shadow.sh NETWORK_INTERFACE [options]
  --control-cpu N       isolated control CPU (default 7)
  --bridge-cpu N        housekeeping CPU for DDS receive threads (default 5)
  --duration-s N        shadow duration (default 30)
  --predictor MODE      legacy phase template (the only supported shadow mode)
  --group NAME          output/service label
  --hardware-config P   verified fail-closed contract YAML

This launcher starts the paired LowState/secondary-IMU state-only bridge. It
has no command publisher, and the Python process opens shared memory read-only.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --control-cpu)
            CONTROL_CPU="$2"; shift 2 ;;
        --bridge-cpu)
            BRIDGE_CPU="$2"; shift 2 ;;
        --duration-s)
            DURATION_S="$2"; shift 2 ;;
        --predictor)
            PREDICTOR="$2"; shift 2 ;;
        --group)
            GROUP="$2"; shift 2 ;;
        --hardware-config)
            HARDWARE_CONFIG="$2"; shift 2 ;;
        --controller-config)
            CONTROLLER_CONFIG="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        -* )
            usage; exit 2 ;;
        *)
            if [[ -n "$NETWORK_INTERFACE" ]]; then
                usage; exit 2
            fi
            NETWORK_INTERFACE="$1"; shift ;;
    esac
done

if [[ -z "$NETWORK_INTERFACE" ]]; then
    usage
    exit 2
fi
for value in "$CONTROL_CPU" "$BRIDGE_CPU" "$REALTIME_PRIORITY" "$DURATION_S"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "CPU, priority, and duration values must be integers." >&2
        exit 2
    fi
done
if (( CONTROL_CPU == BRIDGE_CPU || REALTIME_PRIORITY < 1 || REALTIME_PRIORITY > 20 || DURATION_S < 1 )); then
    echo "Use distinct control/bridge CPUs, RR priority 1..20, and positive duration." >&2
    exit 2
fi
if [[ "$PREDICTOR" != "template" ]]; then
    echo "hardware shadow predictor must be template" >&2
    exit 2
fi
if [[ ! "$GROUP" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "group contains unsupported characters" >&2
    exit 2
fi
if [[ ! "$NETWORK_INTERFACE" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
    echo "network interface contains unsupported characters" >&2
    exit 2
fi

RUN_USER="$(id -un)"
RUN_GROUP="$(id -gn)"
RUN_HOME="$HOME"
G1_MPC_PYTHON="${G1_MPC_PYTHON:-/home/${RUN_USER}/miniforge3/envs/g1_mpc/bin/python}"
UNITREE_SDK2_DIR="${UNITREE_SDK2_DIR:-/home/${RUN_USER}/g1_ws/unitree_sdk2}"
BUILD_DIR="${UNITREE_ARM_ADAPTER_BUILD_DIR:-/tmp/hold-my-beer-mpc-unitree-state-only-build}"
STATE_BRIDGE="$BUILD_DIR/unitree_arm_state_bridge"
SHARED_MEMORY="/g1_shadow_${GROUP}_$$"
BRIDGE_LOG="/tmp/${GROUP}_state_bridge.log"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hold-my-beer-mpc-matplotlib}"

if [[ ! -x "$G1_MPC_PYTHON" ]]; then
    echo "Set G1_MPC_PYTHON to the project environment." >&2
    exit 2
fi
mkdir -p "$MPLCONFIGDIR"
if [[ ! -d "/sys/class/net/$NETWORK_INTERFACE" ]]; then
    echo "network interface does not exist: $NETWORK_INTERFACE" >&2
    exit 2
fi
if ! taskset -c "$BRIDGE_CPU" true || ! taskset -c "$CONTROL_CPU" true; then
    echo "requested CPU is not available to this process" >&2
    exit 2
fi

# Reject guessed message/frame contracts before opening DDS.
"$G1_MPC_PYTHON" "$REPO_DIR/run_hardware_shadow.py" \
    --controller-config "$CONTROLLER_CONFIG" \
    --hardware-config "$HARDWARE_CONFIG" \
    --check-config
"$G1_MPC_PYTHON" "$REPO_DIR/realtime_environment.py" \
    --control-cpu "$CONTROL_CPU"

if [[ ! -f "$UNITREE_SDK2_DIR/CMakeLists.txt" ]]; then
    echo "Unitree SDK2 source not found: $UNITREE_SDK2_DIR" >&2
    exit 2
fi
# An incremental target-only build avoids accidentally using a stale bridge;
# it does not build or launch the output-capable DDS executable.
cmake -S "$REPO_DIR/cpp/unitree_arm_adapter" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUNITREE_ARM_ADAPTER_BUILD_DDS=OFF \
    -DUNITREE_ARM_ADAPTER_BUILD_STATE_BRIDGE=ON \
    -DUNITREE_SDK2_DIR="$UNITREE_SDK2_DIR"
cmake --build "$BUILD_DIR" --parallel \
    --target unitree_arm_state_bridge
if [[ -e "$BUILD_DIR/unitree_arm_adapter_dds" ]]; then
    echo "state-only build directory contains output-capable binary" >&2
    exit 2
fi

bridge_pid=""
cleanup() {
    if [[ -n "$bridge_pid" ]] && kill -0 "$bridge_pid" 2>/dev/null; then
        kill -TERM "$bridge_pid" 2>/dev/null || true
        wait "$bridge_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# DDS receive threads stay on a housekeeping CPU. The isolated control CPU is
# reserved for the bounded RR/10 Python + synchronous C++ MPC path.
taskset -c "$BRIDGE_CPU" "$STATE_BRIDGE" "$NETWORK_INTERFACE" \
    --shm-name "$SHARED_MEMORY" \
    --duration-s "$((DURATION_S + 15))" \
    --max-source-skew-us 5000 \
    --unlink-on-exit >"$BRIDGE_LOG" 2>&1 &
bridge_pid=$!

for _ in {1..100}; do
    if [[ -e "/dev/shm/${SHARED_MEMORY#/}" ]]; then
        break
    fi
    if ! kill -0 "$bridge_pid" 2>/dev/null; then
        wait "$bridge_pid" || true
        echo "state bridge exited; see $BRIDGE_LOG" >&2
        exit 2
    fi
    sleep 0.05
done
if [[ ! -e "/dev/shm/${SHARED_MEMORY#/}" ]]; then
    echo "state bridge did not create shared memory; see $BRIDGE_LOG" >&2
    exit 2
fi

UNIT="hold-my-beer-mpc-shadow-${GROUP}"
OUTPUT_DIR="evaluation/hardware_shadow/${GROUP}"
export MPC_CONTROL_NUM_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONHASHSEED=0

sudo systemd-run --wait --pipe --collect \
    --unit="$UNIT" \
    --uid="$RUN_USER" --gid="$RUN_GROUP" \
    --working-directory="$REPO_DIR" \
    --property=LimitRTPRIO=20 \
    --property=RestrictRealtime=no \
    --property=RuntimeMaxSec="$((DURATION_S + 30))s" \
    --property=TimeoutStopSec=5s \
    --property=KillMode=control-group \
    --setenv="HOME=$RUN_HOME" \
    --setenv="MPLCONFIGDIR=$MPLCONFIGDIR" \
    --setenv=MPC_CONTROL_NUM_THREADS=1 \
    --setenv=OMP_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 \
    --setenv=MKL_NUM_THREADS=1 \
    --setenv=NUMEXPR_NUM_THREADS=1 \
    --setenv=VECLIB_MAXIMUM_THREADS=1 \
    --setenv=BLIS_NUM_THREADS=1 \
    --setenv=OMP_DYNAMIC=FALSE \
    --setenv=MKL_DYNAMIC=FALSE \
    --setenv=PYTHONHASHSEED=0 \
    taskset -c "$CONTROL_CPU" \
    chrt --rr "$REALTIME_PRIORITY" \
    "$G1_MPC_PYTHON" "$REPO_DIR/realtime_runtime.py" \
    --expected-policy SCHED_RR \
    --expected-priority "$REALTIME_PRIORITY" \
    --expected-cpu "$CONTROL_CPU" \
    --require-target-environment \
    -- "$G1_MPC_PYTHON" "$REPO_DIR/run_hardware_shadow.py" \
    --controller-config "$CONTROLLER_CONFIG" \
    --hardware-config "$HARDWARE_CONFIG" \
    --shared-memory "$SHARED_MEMORY" \
    --predictor "$PREDICTOR" \
    --duration-s "$DURATION_S" \
    --output-dir "$OUTPUT_DIR"

echo "state-only bridge log: $BRIDGE_LOG"
