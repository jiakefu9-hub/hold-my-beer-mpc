#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTROL_CPU="${MPC_CONTROL_CPU:-7}"
GROUP="target_rt_timing_$(date +%Y%m%d_%H%M%S)"
RESUME=0

usage() {
    echo "Usage: $0 [--control-cpu N] [--group NAME] [--resume]" >&2
}

while (( $# > 0 )); do
    case "$1" in
        --control-cpu)
            CONTROL_CPU="$2"
            shift 2
            ;;
        --group)
            GROUP="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ ! "$CONTROL_CPU" =~ ^[0-9]+$ ]]; then
    echo "control CPU must be one integer" >&2
    exit 2
fi
if [[ ! "$GROUP" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "group must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
fi

RUN_USER="$(id -un)"
RUN_GROUP="$(id -gn)"
RUN_HOME="$HOME"
G1_MPC_CONDA="${G1_MPC_CONDA:-/home/${RUN_USER}/miniforge3/bin/conda}"
DEFAULT_G1_MPC_PYTHON="/home/${RUN_USER}/miniforge3/envs/g1_mpc/bin/python"
if [[ ! -x "$DEFAULT_G1_MPC_PYTHON" ]]; then
    DEFAULT_G1_MPC_PYTHON="$(command -v python)"
fi
G1_MPC_PYTHON="${G1_MPC_PYTHON:-$DEFAULT_G1_MPC_PYTHON}"
if [[ ! -x "$G1_MPC_PYTHON" || ! -x "$G1_MPC_CONDA" ]]; then
    echo "Set executable G1_MPC_PYTHON and G1_MPC_CONDA paths first." >&2
    exit 2
fi

"$G1_MPC_PYTHON" "$REPO_DIR/realtime_environment.py" \
    --control-cpu "$CONTROL_CPU"

RUNNER_ARGS=(
    disturbance_learning/run_realtime_timing_ablation.py
    --group "$GROUP"
    --control-cpu "$CONTROL_CPU"
    --require-target-realtime
    --fail-on-gate
)
if (( RESUME == 1 )); then
    RUNNER_ARGS+=(--resume)
fi

UNIT="disturbance-lab-target-rt-${GROUP}"
exec sudo systemd-run --wait --pipe --collect \
    --unit="$UNIT" \
    --uid="$RUN_USER" --gid="$RUN_GROUP" \
    --working-directory="$REPO_DIR" \
    --property=LimitRTPRIO=20 \
    --property=RestrictRealtime=no \
    --property=RuntimeMaxSec=15min \
    --property=TimeoutStopSec=5s \
    --property=KillMode=control-group \
    --setenv="HOME=$RUN_HOME" \
    --setenv="G1_MPC_PYTHON=$G1_MPC_PYTHON" \
    --setenv="G1_MPC_CONDA=$G1_MPC_CONDA" \
    --setenv=MPC_REQUIRE_REALTIME=1 \
    --setenv=MPC_REQUIRE_TARGET_REALTIME=1 \
    --setenv=MPC_REALTIME_POLICY=SCHED_RR \
    --setenv=MPC_REALTIME_PRIORITY=10 \
    --setenv="MPC_CONTROL_CPU=$CONTROL_CPU" \
    "$G1_MPC_PYTHON" "${RUNNER_ARGS[@]}"
