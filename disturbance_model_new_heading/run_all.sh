#!/usr/bin/env bash
set -euo pipefail

# 【非核心代码】保留原入口名；当前 MPC 统一生成 6 ms 区间模板。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_interval_all.sh"
