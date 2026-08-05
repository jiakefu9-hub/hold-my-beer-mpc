#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# 自动扫描 MPC 的 Q_A（末端线加速度）与 Q_alpha（末端角加速度）
#
# 使用方法：
#   1. 把本脚本放到 hold-my-beer-mpc 仓库根目录
#   2. chmod +x sweep_qa_qalpha.sh
#   3. ./sweep_qa_qalpha.sh
#
# 默认实验：
#   第一阶段：Q_alpha=0，扫描 Q_A
#   第二阶段：固定 Q_A=0.003，扫描 Q_alpha
#
# 可临时覆盖第二阶段的固定 Q_A：
#   FIXED_QA_FOR_QALPHA=0.001 ./sweep_qa_qalpha.sh
# ============================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$REPO_DIR/configs/g1.yaml"
RUN_SCRIPT="$REPO_DIR/run.sh"

# 第一阶段：只恢复线加速度代价。
QA_VALUES=(
  "0"
  "0.0003"
  "0.001"
  "0.003"
  "0.01"
  "0.03"
)

# 第二阶段：固定一个 Q_A，只恢复角加速度代价。
QALPHA_VALUES=(
  "0.0001"
  "0.0003"
  "0.001"
  "0.003"
  "0.01"
)

FIXED_QA_FOR_QALPHA="${FIXED_QA_FOR_QALPHA:-0.003}"

SWEEP_ID="qa_qalpha_sweep_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_DIR/sweep_logs/$SWEEP_ID"
MANIFEST="$LOG_DIR/manifest.csv"
BACKUP_FILE="$(mktemp "${TMPDIR:-/tmp}/g1.yaml.qa_qalpha_backup.XXXXXX")"

mkdir -p "$LOG_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[错误] 找不到配置文件：$CONFIG_FILE" >&2
  exit 1
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "[错误] 找不到运行脚本：$RUN_SCRIPT" >&2
  exit 1
fi

if [[ ! -x "$RUN_SCRIPT" ]]; then
  echo "[错误] run.sh 没有执行权限，请先运行：chmod +x run.sh" >&2
  exit 1
fi

cp -- "$CONFIG_FILE" "$BACKUP_FILE"

cleanup() {
  local exit_code=$?
  if [[ -f "$BACKUP_FILE" ]]; then
    cp -- "$BACKUP_FILE" "$CONFIG_FILE"
    rm -f -- "$BACKUP_FILE"
    echo
    echo "[恢复] 已恢复原始 configs/g1.yaml"
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM HUP

# 确认当前配置确实使用 MPC，避免扫了半天却跑的是 PID/LQR。
python - "$CONFIG_FILE" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

match = re.search(r"(?m)^\s*arm_controller\s*:\s*([^\s#]+)", text)
if match is None:
    raise SystemExit("[错误] g1.yaml 中找不到 arm_controller。")

controller = match.group(1).strip().lower()
if controller != "mpc":
    raise SystemExit(
        f"[错误] 当前 arm_controller={controller!r}，必须为 'mpc'。"
    )

for key in ("mpc_q_ee_acc", "mpc_q_ee_alpha"):
    count = len(re.findall(rf"(?m)^\s*{re.escape(key)}\s*:", text))
    if count != 1:
        raise SystemExit(
            f"[错误] 配置项 {key} 应恰好出现一次，当前出现 {count} 次。"
        )
PY

echo "stage,qa,qalpha,run_label,status,log_file,config_snapshot" > "$MANIFEST"

safe_number_label() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  value="${value//+/}"
  echo "$value"
}

set_weights() {
  local qa="$1"
  local qalpha="$2"

  python - "$CONFIG_FILE" "$qa" "$qalpha" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
qa = sys.argv[2]
qalpha = sys.argv[3]
text = path.read_text(encoding="utf-8")

def replace_scalar(source: str, key: str, value: str) -> str:
    # 保留原有缩进和行尾注释，只替换数值。
    pattern = re.compile(
        rf"(?m)^(\s*{re.escape(key)}\s*:\s*)([^#\n]*?)(\s*(?:#.*)?)$"
    )
    updated, count = pattern.subn(
        lambda match: f"{match.group(1)}{value}{match.group(3)}",
        source,
    )
    if count != 1:
        raise RuntimeError(
            f"配置项 {key} 应恰好替换一次，实际替换 {count} 次。"
        )
    return updated

text = replace_scalar(text, "mpc_q_ee_acc", qa)
text = replace_scalar(text, "mpc_q_ee_alpha", qalpha)
path.write_text(text, encoding="utf-8")
PY
}

run_case() {
  local stage="$1"
  local qa="$2"
  local qalpha="$3"

  local qa_label
  local qalpha_label
  qa_label="$(safe_number_label "$qa")"
  qalpha_label="$(safe_number_label "$qalpha")"

  local run_label="${stage}_Qa${qa_label}_Qalpha${qalpha_label}"
  local log_file="$LOG_DIR/${run_label}.log"
  local config_snapshot="$LOG_DIR/${run_label}.yaml"

  echo
  echo "============================================================"
  echo "[运行] 阶段      : $stage"
  echo "[运行] Q_A       : $qa"
  echo "[运行] Q_alpha   : $qalpha"
  echo "[运行] 分组      : $SWEEP_ID"
  echo "[运行] 标签      : $run_label"
  echo "============================================================"

  set_weights "$qa" "$qalpha"
  cp -- "$CONFIG_FILE" "$config_snapshot"

  # 自动化扫描默认关闭 viewer 和视频，保留 g1.yaml 中完整的
  # warmup/evaluation/cooldown 周期与所有其他控制参数。
  set +e
  "$RUN_SCRIPT" \
    --headless \
    --no-video \
    --evaluation-group "$SWEEP_ID" \
    --run-label "$run_label" \
    2>&1 | tee "$log_file"

  local status=${PIPESTATUS[0]}
  set -e

  echo \
    "$stage,$qa,$qalpha,$run_label,$status,$log_file,$config_snapshot" \
    >> "$MANIFEST"

  if [[ "$status" -ne 0 ]]; then
    echo "[失败] $run_label，退出码：$status" >&2
    echo "[失败] 日志：$log_file" >&2
    return "$status"
  fi

  echo "[完成] $run_label"
}

echo "============================================================"
echo "MPC Q_A / Q_alpha 自动扫描"
echo "实验分组：$SWEEP_ID"
echo "日志目录：$LOG_DIR"
echo "原配置会在脚本退出时自动恢复。"
echo "============================================================"

# 第一阶段：Q_alpha 固定为 0，扫描 Q_A。
for qa in "${QA_VALUES[@]}"; do
  run_case "qa_sweep" "$qa" "0"
done

# 第二阶段：固定 Q_A，扫描非零 Q_alpha。
# Q_alpha=0 已在第一阶段对应 Q_A 值中覆盖，不重复运行。
for qalpha in "${QALPHA_VALUES[@]}"; do
  run_case "qalpha_sweep" "$FIXED_QA_FOR_QALPHA" "$qalpha"
done

echo
echo "============================================================"
echo "[全部完成]"
echo "实验结果目录：$REPO_DIR/evaluation/$SWEEP_ID"
echo "运行日志目录：$LOG_DIR"
echo "实验清单文件：$MANIFEST"
echo "第二阶段固定 Q_A：$FIXED_QA_FOR_QALPHA"
echo "============================================================"
