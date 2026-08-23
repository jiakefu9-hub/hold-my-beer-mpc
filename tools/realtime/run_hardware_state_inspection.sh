#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRIDGE_CPU="${UNITREE_STATE_BRIDGE_CPU:-5}"
INSPECT_CPU="${MPC_CONTROL_CPU:-7}"
DURATION_S=10
INSPECT_SAMPLES=500
GROUP="g1_state_inspection_$(date +%Y%m%d_%H%M%S)"
NETWORK_INTERFACE=""
HARDWARE_CONFIG="configs/g1_hardware_shadow.yaml"
CONTROLLER_CONFIG="configs/g1.yaml"

usage() {
    cat >&2 <<'EOF'
Usage: run_hardware_state_inspection.sh NETWORK_INTERFACE [options]
  --bridge-cpu N        CPU for DDS receive threads (default 5)
  --inspect-cpu N       CPU for the read-only Python collector (default 7)
  --duration-s N        finite collection timeout (default 10)
  --inspect-samples N   required fresh paired samples (default 500)
  --group NAME          unique evidence directory label
  --hardware-config P   unverified discovery contract YAML
  --controller-config P MuJoCo model config used only for index mapping checks

This launcher builds and runs only unitree_arm_state_bridge. It never builds or
runs unitree_arm_adapter_dds, contains no command publisher, does not run MPC,
and does not change robot mode or ownership.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --bridge-cpu)
            BRIDGE_CPU="$2"; shift 2 ;;
        --inspect-cpu)
            INSPECT_CPU="$2"; shift 2 ;;
        --duration-s)
            DURATION_S="$2"; shift 2 ;;
        --inspect-samples)
            INSPECT_SAMPLES="$2"; shift 2 ;;
        --group)
            GROUP="$2"; shift 2 ;;
        --hardware-config)
            HARDWARE_CONFIG="$2"; shift 2 ;;
        --controller-config)
            CONTROLLER_CONFIG="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        -*)
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
for value in "$BRIDGE_CPU" "$INSPECT_CPU" "$DURATION_S" "$INSPECT_SAMPLES"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "CPU, duration, and sample values must be integers." >&2
        exit 2
    fi
done
if (( DURATION_S < 1 || INSPECT_SAMPLES < 1 )); then
    echo "duration and inspect sample count must be positive" >&2
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
G1_MPC_PYTHON="${G1_MPC_PYTHON:-/home/${RUN_USER}/miniforge3/envs/g1_mpc/bin/python}"
UNITREE_SDK2_DIR="${UNITREE_SDK2_DIR:-/home/${RUN_USER}/g1_ws/unitree_sdk2}"
BUILD_DIR="${UNITREE_STATE_BRIDGE_BUILD_DIR:-/tmp/hold-my-beer-mpc-unitree-state-only-build}"
STATE_BRIDGE="$BUILD_DIR/unitree_arm_state_bridge"
SESSION_DIR="$REPO_DIR/evaluation/hardware_shadow/state_inspection/$GROUP"
SHARED_MEMORY="/g1_state_inspection_${GROUP}_$$"
BRIDGE_LOG="$SESSION_DIR/state_bridge.log"
BRIDGE_SUMMARY="$SESSION_DIR/state_bridge_summary.json"

if [[ ! -x "$G1_MPC_PYTHON" ]]; then
    echo "Set G1_MPC_PYTHON to the project environment." >&2
    exit 2
fi
if [[ ! -f "$UNITREE_SDK2_DIR/CMakeLists.txt" ]]; then
    echo "Unitree SDK2 source not found: $UNITREE_SDK2_DIR" >&2
    exit 2
fi
if [[ ! -d "/sys/class/net/$NETWORK_INTERFACE" ]]; then
    echo "network interface does not exist: $NETWORK_INTERFACE" >&2
    exit 2
fi
if [[ "$(cat "/sys/class/net/$NETWORK_INTERFACE/carrier" 2>/dev/null || true)" != "1" ]]; then
    echo "network interface has no carrier: $NETWORK_INTERFACE" >&2
    exit 2
fi
if ! taskset -c "$BRIDGE_CPU" true || ! taskset -c "$INSPECT_CPU" true; then
    echo "requested CPU is not available to this process" >&2
    exit 2
fi
if pgrep -f '(^|/)unitree_arm_adapter_dds([[:space:]]|$)' >/dev/null; then
    echo "output-capable unitree_arm_adapter_dds is already running" >&2
    exit 2
fi
if [[ -e "$SESSION_DIR" ]]; then
    echo "inspection session directory already exists: $SESSION_DIR" >&2
    exit 2
fi
mkdir -p "$(dirname "$SESSION_DIR")"
mkdir "$SESSION_DIR"

"$G1_MPC_PYTHON" "$REPO_DIR/run_hardware_shadow.py" \
    --controller-config "$CONTROLLER_CONFIG" \
    --hardware-config "$HARDWARE_CONFIG" \
    --check-config-discovery

cmake -S "$REPO_DIR/cpp/unitree_arm_adapter" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUNITREE_ARM_ADAPTER_BUILD_DDS=OFF \
    -DUNITREE_ARM_ADAPTER_BUILD_STATE_BRIDGE=ON \
    -DUNITREE_SDK2_DIR="$UNITREE_SDK2_DIR"
cmake --build "$BUILD_DIR" --parallel --target unitree_arm_state_bridge

if [[ -e "$BUILD_DIR/unitree_arm_adapter_dds" ]]; then
    echo "state-only build directory contains output-capable binary" >&2
    exit 2
fi
if ldd "$STATE_BRIDGE" | grep -q 'not found'; then
    echo "state bridge has unresolved shared libraries" >&2
    ldd "$STATE_BRIDGE" >&2
    exit 2
fi
if strings "$STATE_BRIDGE" | grep -Eq '^rt/(arm_sdk|lowcmd)$'; then
    echo "state bridge binary unexpectedly contains a command topic" >&2
    exit 2
fi
if nm -C "$STATE_BRIDGE" | grep -q 'ChannelPublisher'; then
    echo "state bridge binary unexpectedly references ChannelPublisher" >&2
    exit 2
fi

bridge_pid=""
stop_bridge() {
    if [[ -n "$bridge_pid" ]] && kill -0 "$bridge_pid" 2>/dev/null; then
        kill -TERM "$bridge_pid" 2>/dev/null || true
        wait "$bridge_pid" 2>/dev/null || true
    fi
    bridge_pid=""
}
cleanup() {
    stop_bridge
}
trap cleanup EXIT INT TERM

taskset -c "$BRIDGE_CPU" "$STATE_BRIDGE" "$NETWORK_INTERFACE" \
    --shm-name "$SHARED_MEMORY" \
    --duration-s "$((DURATION_S + 10))" \
    --max-source-skew-us 5000 \
    --summary-json "$BRIDGE_SUMMARY" \
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

set +e
taskset -c "$INSPECT_CPU" "$G1_MPC_PYTHON" \
    "$REPO_DIR/run_hardware_shadow.py" \
    --controller-config "$CONTROLLER_CONFIG" \
    --hardware-config "$HARDWARE_CONFIG" \
    --shared-memory "$SHARED_MEMORY" \
    --inspect-state-only \
    --inspect-samples "$INSPECT_SAMPLES" \
    --duration-s "$DURATION_S" \
    --session-dir "$SESSION_DIR" \
    --network-interface "$NETWORK_INTERFACE" \
    --state-bridge-binary "$STATE_BRIDGE" \
    --unitree-sdk-dir "$UNITREE_SDK2_DIR" \
    2>&1 | tee "$SESSION_DIR/inspection.log"
inspection_status=${PIPESTATUS[0]}
set -e
stop_bridge

if [[ ! -f "$BRIDGE_SUMMARY" ]]; then
    echo "state bridge did not write summary; see $BRIDGE_LOG" >&2
    exit 2
fi
if (( inspection_status != 0 )); then
    echo "read-only inspection failed; evidence: $SESSION_DIR" >&2
    exit "$inspection_status"
fi

echo "read-only inspection: PASS"
echo "evidence: $SESSION_DIR"
