#!/usr/bin/env python3
"""
无人值守扫描 MPC 的 Q_A（末端线加速度权重）和 Q_alpha（末端角加速度权重）。

放置位置：
    hold-my-beer-mpc/sweep_qa_qalpha_full.py

直接运行：
    chmod +x sweep_qa_qalpha_full.py
    ./sweep_qa_qalpha_full.py

断点续跑：
    ./sweep_qa_qalpha_full.py --resume latest

默认执行 14 × 11 = 154 组完整实验。
每组都使用当前 configs/g1.yaml 的其他参数，只覆盖：
    mpc_q_ee_acc
    mpc_q_ee_alpha

脚本特性：
- 自动使用 systemd-inhibit 阻止 Linux 休眠（若系统支持）
- 单组失败或超时后继续下一组
- Ctrl+C / 正常结束时恢复原始 g1.yaml
- 支持断点续跑
- 默认删除每轮特别大的 trajectory/preview 文件，避免 154 轮占满磁盘
- 自动生成 raw_results.csv、grid_aggregate.csv、pareto_frontier.csv、
  recommended_candidates.csv 和 SWEEP_SUMMARY.txt
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# 搜索网格
# ---------------------------------------------------------------------------
# Q_A=0.03 在上一轮仍然位于性能边界，因此继续向上扩展到 0.15。
QA_VALUES = [
    0.0,
    0.0003,
    0.001,
    0.003,
    0.006,
    0.01,
    0.015,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.15,
]

# Q_alpha=0.001~0.003 是上一轮的有效区间，同时保留 0.005~0.01
# 用于确定性能开始恶化和 DDQ 饱和的边界。
QALPHA_VALUES = [
    0.0,
    0.0001,
    0.0003,
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
]

# 优先运行的核心区域。即使实验中途停止，也能先得到最有信息量的结果。
CORE_QA = {0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10}
CORE_QALPHA = {0.0, 0.0003, 0.001, 0.002, 0.003}

# 自动推荐时使用的“可接受”硬门槛。原始数据不会因此被删除。
QP_SUCCESS_MIN = 0.99
TILT_RMS_MAX = 0.03
DDQ_SATURATION_MAX = 0.10
SAFETY_VIOLATION_MAX = 0.0
ARM_INTERVAL_OVERRUN_MAX = 0.0

HEAVY_ARTIFACT_NAMES = {
    "trajectory.npz",
    "control_preview.csv",
    "metrics_preview.csv",
    "mpc_diagnostics_preview.csv",
    "mpc_tracking_preview.csv",
}
HEAVY_ARTIFACT_PATTERNS = (
    "*_preview.csv",
    "trajectory*.npz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="完整二维扫描 MPC 的 Q_A 和 Q_alpha。"
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="SWEEP_ID",
        help="继续已有扫描；省略 ID 或写 latest 时继续最近一次扫描。",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="每个参数组合重复次数，默认 1。设为 2 会运行 308 次。",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=15.0,
        help="单组实验超时分钟数，默认 15；超时后终止该组并继续。",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=3.0,
        help="开始每组实验前要求的最小剩余磁盘空间，默认 3 GB。",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="保留 trajectory.npz 和大型 preview CSV。默认会删除这些大文件。",
    )
    parser.add_argument(
        "--no-inhibit",
        action="store_true",
        help="不调用 systemd-inhibit 阻止休眠。",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="任一实验失败后立即停止；默认记录失败并继续。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示计划，不真正运行。",
    )
    return parser.parse_args()


def maybe_reexec_with_sleep_inhibit(args: argparse.Namespace) -> None:
    """尽量阻止桌面因为空闲而休眠，避免无人值守扫描被腰斩。"""
    if args.no_inhibit:
        return
    if os.environ.get("QA_QALPHA_SWEEP_INHIBITED") == "1":
        return

    inhibitor = shutil.which("systemd-inhibit")
    if inhibitor is None:
        print("[提示] 未找到 systemd-inhibit；请确认电脑不会自动休眠。")
        return

    env = os.environ.copy()
    env["QA_QALPHA_SWEEP_INHIBITED"] = "1"
    command = [
        inhibitor,
        "--what=sleep:idle",
        "--mode=block",
        "--why=MPC Q_A and Q_alpha unattended sweep",
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    os.execvpe(inhibitor, command, env)


def number_label(value: float) -> str:
    text = format(value, ".10g")
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def close_enough(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def build_cases(repeats: int) -> list[tuple[int, float, float]]:
    if repeats < 1:
        raise ValueError("--repeats 必须至少为 1。")

    all_pairs = [(qa, qalpha) for qalpha in QALPHA_VALUES for qa in QA_VALUES]

    core = [
        pair
        for pair in all_pairs
        if pair[0] in CORE_QA and pair[1] in CORE_QALPHA
    ]
    remaining = [pair for pair in all_pairs if pair not in set(core)]

    # 基线第一；核心区按 Q_alpha 分层并蛇形遍历，降低相邻实验参数跳变。
    baseline = [(0.0, 0.0)]
    core = [pair for pair in core if pair != (0.0, 0.0)]

    def serpentine(pairs: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
        grouped: dict[float, list[float]] = defaultdict(list)
        for qa, qalpha in pairs:
            grouped[qalpha].append(qa)

        ordered: list[tuple[float, float]] = []
        for index, qalpha in enumerate(sorted(grouped)):
            qas = sorted(grouped[qalpha], reverse=bool(index % 2))
            ordered.extend((qa, qalpha) for qa in qas)
        return ordered

    ordered_pairs = baseline + serpentine(core) + serpentine(remaining)

    # 去重并确认完整网格没有漏点。
    unique_pairs: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for pair in ordered_pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)

    expected = len(QA_VALUES) * len(QALPHA_VALUES)
    if len(unique_pairs) != expected:
        raise RuntimeError(
            f"内部错误：应有 {expected} 个参数组合，实际得到 {len(unique_pairs)} 个。"
        )

    cases: list[tuple[int, float, float]] = []
    for repeat in range(1, repeats + 1):
        cases.extend((repeat, qa, qalpha) for qa, qalpha in unique_pairs)
    return cases


def replace_yaml_scalar(text: str, key: str, value: float) -> str:
    pattern = re.compile(
        rf"(?m)^(\s*{re.escape(key)}\s*:\s*)([^#\n]*?)(\s*(?:#.*)?)$"
    )
    replacement = format(value, ".12g")
    updated, count = pattern.subn(
        lambda match: f"{match.group(1)}{replacement}{match.group(3)}",
        text,
    )
    if count != 1:
        raise RuntimeError(
            f"配置项 {key} 应恰好出现一次，实际匹配 {count} 次。"
        )
    return updated


def validate_config(text: str) -> None:
    match = re.search(r"(?m)^\s*arm_controller\s*:\s*([^\s#]+)", text)
    if match is None:
        raise RuntimeError("g1.yaml 中找不到 arm_controller。")
    controller = match.group(1).strip().lower()
    if controller != "mpc":
        raise RuntimeError(
            f"当前 arm_controller={controller!r}；本脚本只允许 MPC。"
        )

    for key in ("mpc_q_ee_acc", "mpc_q_ee_alpha"):
        count = len(re.findall(rf"(?m)^\s*{re.escape(key)}\s*:", text))
        if count != 1:
            raise RuntimeError(
                f"配置项 {key} 应恰好出现一次，当前出现 {count} 次。"
            )


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def scalar_weight(value: Any) -> float:
    if isinstance(value, (int, float)):
        return finite_float(value)
    if isinstance(value, list) and value:
        numbers = [finite_float(item) for item in value]
        if all(math.isfinite(item) for item in numbers):
            if max(numbers) - min(numbers) < 1e-12:
                return numbers[0]
    return math.nan


def find_output_dir(group_dir: Path, run_label: str) -> Path | None:
    candidates = [
        path
        for path in group_dir.glob(f"*_{run_label}")
        if path.is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_is_complete(group_dir: Path, run_label: str) -> bool:
    output = find_output_dir(group_dir, run_label)
    return output is not None and (output / "summary.json").is_file()


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def tail_text(path: Path, lines: int = 25) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_one(
    run_script: Path,
    group_dir: Path,
    sweep_id: str,
    run_label: str,
    log_file: Path,
    timeout_seconds: float,
) -> tuple[int, bool, float]:
    command = [
        str(run_script),
        "--headless",
        "--no-video",
        "--evaluation-group",
        sweep_id,
        "--run-label",
        run_label,
    ]

    started = time.monotonic()
    timed_out = False

    with log_file.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=run_script.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
            return_code = 124
            log.write(
                f"\n[SWEEP] 单组运行超过 {timeout_seconds / 60:.1f} 分钟，"
                "已终止并继续下一组。\n"
            )
        except KeyboardInterrupt:
            terminate_process_group(process)
            raise

    elapsed = time.monotonic() - started
    return return_code, timed_out, elapsed


def prune_heavy_artifacts(output_dir: Path) -> list[str]:
    removed: list[str] = []
    candidates: set[Path] = set()

    for name in HEAVY_ARTIFACT_NAMES:
        candidates.add(output_dir / name)
    for pattern in HEAVY_ARTIFACT_PATTERNS:
        candidates.update(output_dir.glob(pattern))

    for path in sorted(candidates):
        if path.is_file():
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
                path.unlink()
                removed.append(f"{path.name} ({size_mb:.1f} MB)")
            except OSError:
                pass
    return removed


def extract_result(output_dir: Path) -> dict[str, Any] | None:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return None

    summary = load_json(summary_path)
    metadata = load_json(output_dir / "run_metadata.json")
    mpc = load_json(output_dir / "mpc_diagnostics.json")
    arm = load_json(output_dir / "right_arm_diagnostics.json")
    perf = load_json(output_dir / "perf_summary.json")

    qa = scalar_weight(nested(metadata, "mpc_config", "q_ee_acc"))
    qalpha = scalar_weight(nested(metadata, "mpc_config", "q_ee_alpha"))

    interval = nested(
        perf,
        "total",
        "real_hardware_control",
        "right_arm_interval",
        default={},
    )
    if not isinstance(interval, dict):
        interval = {}

    row: dict[str, Any] = {
        "run_dir": str(output_dir),
        "run_name": output_dir.name,
        "qa": qa,
        "qalpha": qalpha,
        "right_acc_rms": finite_float(summary.get("right_acc_rms")),
        "right_alpha_rms": finite_float(summary.get("right_alpha_rms")),
        "right_tilt_rms": finite_float(summary.get("right_tilt_rms")),
        "right_tilt_std": finite_float(summary.get("right_tilt_std")),
        "right_acc_xyz_rms": json.dumps(
            summary.get("right_acc_xyz_rms", []),
            ensure_ascii=False,
        ),
        "right_alpha_xyz_rms": json.dumps(
            summary.get("right_alpha_xyz_rms", []),
            ensure_ascii=False,
        ),
        "left_acc_rms": finite_float(summary.get("left_acc_rms")),
        "left_alpha_rms": finite_float(summary.get("left_alpha_rms")),
        "left_tilt_rms": finite_float(summary.get("left_tilt_rms")),
        "walk_distance_xy": finite_float(summary.get("walk_distance_xy")),
        "qp_success": finite_float(
            nested(mpc, "solver", "success_fraction")
        ),
        "fallback_fraction": finite_float(
            nested(mpc, "solver", "fallback_fraction")
        ),
        "q_safety_violation_fraction": finite_float(
            nested(mpc, "solver", "current_q_safety_violation_fraction")
        ),
        "recovery_active_fraction": finite_float(
            nested(mpc, "solver", "recovery_active_fraction")
        ),
        "ddq_saturation_any_fraction": finite_float(
            arm.get("right_arm_ddq_saturation_any_fraction")
        ),
        "tau_saturation_any_fraction": finite_float(
            arm.get("right_arm_tau_saturation_any_fraction")
        ),
        "arm_interval_mean_ms": finite_float(interval.get("mean")),
        "arm_interval_p99_ms": finite_float(interval.get("p99")),
        "arm_interval_max_ms": finite_float(interval.get("max")),
        "arm_interval_overrun_fraction": finite_float(
            interval.get("overrun_fraction")
        ),
    }
    return row


RAW_FIELDS = [
    "run_dir",
    "run_name",
    "qa",
    "qalpha",
    "right_acc_rms",
    "right_alpha_rms",
    "right_tilt_rms",
    "right_tilt_std",
    "right_acc_xyz_rms",
    "right_alpha_xyz_rms",
    "left_acc_rms",
    "left_alpha_rms",
    "left_tilt_rms",
    "walk_distance_xy",
    "qp_success",
    "fallback_fraction",
    "q_safety_violation_fraction",
    "recovery_active_fraction",
    "ddq_saturation_any_fraction",
    "tau_saturation_any_fraction",
    "arm_interval_mean_ms",
    "arm_interval_p99_ms",
    "arm_interval_max_ms",
    "arm_interval_overrun_fraction",
]


def scan_results(group_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not group_dir.is_dir():
        return rows

    for output_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
        row = extract_result(output_dir)
        if row is not None:
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


AGG_METRICS = [
    "right_acc_rms",
    "right_alpha_rms",
    "right_tilt_rms",
    "right_tilt_std",
    "left_acc_rms",
    "left_alpha_rms",
    "left_tilt_rms",
    "walk_distance_xy",
    "qp_success",
    "fallback_fraction",
    "q_safety_violation_fraction",
    "recovery_active_fraction",
    "ddq_saturation_any_fraction",
    "tau_saturation_any_fraction",
    "arm_interval_mean_ms",
    "arm_interval_p99_ms",
    "arm_interval_max_ms",
    "arm_interval_overrun_fraction",
]


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        qa = finite_float(row.get("qa"))
        qalpha = finite_float(row.get("qalpha"))
        if math.isfinite(qa) and math.isfinite(qalpha):
            grouped[(qa, qalpha)].append(row)

    aggregates: list[dict[str, Any]] = []
    for (qa, qalpha), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "qa": qa,
            "qalpha": qalpha,
            "samples": len(group),
        }
        for metric in AGG_METRICS:
            values = [
                finite_float(row.get(metric))
                for row in group
                if math.isfinite(finite_float(row.get(metric)))
            ]
            result[metric] = statistics.mean(values) if values else math.nan
            result[f"{metric}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
        aggregates.append(result)
    return aggregates


def is_acceptable(row: dict[str, Any]) -> bool:
    required = [
        finite_float(row.get("right_acc_rms")),
        finite_float(row.get("right_alpha_rms")),
        finite_float(row.get("right_tilt_rms")),
        finite_float(row.get("qp_success")),
        finite_float(row.get("q_safety_violation_fraction")),
        finite_float(row.get("ddq_saturation_any_fraction")),
        finite_float(row.get("arm_interval_overrun_fraction")),
    ]
    if not all(math.isfinite(value) for value in required):
        return False

    return (
        finite_float(row["qp_success"]) >= QP_SUCCESS_MIN
        and finite_float(row["right_tilt_rms"]) <= TILT_RMS_MAX
        and finite_float(row["q_safety_violation_fraction"])
        <= SAFETY_VIOLATION_MAX
        and finite_float(row["ddq_saturation_any_fraction"])
        <= DDQ_SATURATION_MAX
        and finite_float(row["arm_interval_overrun_fraction"])
        <= ARM_INTERVAL_OVERRUN_MAX
    )


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    objectives = (
        "right_acc_rms",
        "right_alpha_rms",
        "right_tilt_rms",
        "ddq_saturation_any_fraction",
    )
    av = [finite_float(a.get(key)) for key in objectives]
    bv = [finite_float(b.get(key)) for key in objectives]
    return all(x <= y for x, y in zip(av, bv)) and any(
        x < y for x, y in zip(av, bv)
    )


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = [row for row in rows if is_acceptable(row)]
    frontier = [
        candidate
        for candidate in safe
        if not any(
            dominates(other, candidate)
            for other in safe
            if other is not candidate
        )
    ]
    return sorted(
        frontier,
        key=lambda row: (
            finite_float(row.get("right_acc_rms")),
            finite_float(row.get("right_alpha_rms")),
        ),
    )


def normalize_objectives(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    objectives = (
        "right_acc_rms",
        "right_alpha_rms",
        "right_tilt_rms",
        "ddq_saturation_any_fraction",
    )
    bounds: dict[str, tuple[float, float]] = {}
    for key in objectives:
        values = [
            finite_float(row.get(key))
            for row in rows
            if math.isfinite(finite_float(row.get(key)))
        ]
        bounds[key] = (
            min(values) if values else 0.0,
            max(values) if values else 1.0,
        )
    return bounds


def normalized_value(
    row: dict[str, Any],
    key: str,
    bounds: dict[str, tuple[float, float]],
) -> float:
    low, high = bounds[key]
    value = finite_float(row.get(key))
    if not math.isfinite(value):
        return 1.0
    if high <= low + 1e-15:
        return 0.0
    return (value - low) / (high - low)


def score_row(
    row: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    weights: dict[str, float],
) -> float:
    # 到数据集内“理想点”的加权欧氏距离，越小越好。
    return math.sqrt(
        sum(
            weight * normalized_value(row, key, bounds) ** 2
            for key, weight in weights.items()
        )
    )


def add_analysis_columns(
    aggregates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    baseline_candidates = [
        row
        for row in aggregates
        if close_enough(finite_float(row.get("qa")), 0.0)
        and close_enough(finite_float(row.get("qalpha")), 0.0)
    ]
    baseline = baseline_candidates[0] if baseline_candidates else None

    for row in aggregates:
        row["acceptable"] = is_acceptable(row)
        if baseline is None:
            row["acc_improvement_pct"] = math.nan
            row["alpha_improvement_pct"] = math.nan
            row["tilt_change_pct"] = math.nan
            continue

        def improvement(metric: str) -> float:
            base = finite_float(baseline.get(metric))
            current = finite_float(row.get(metric))
            if not math.isfinite(base) or not math.isfinite(current) or base == 0:
                return math.nan
            return 100.0 * (base - current) / base

        row["acc_improvement_pct"] = improvement("right_acc_rms")
        row["alpha_improvement_pct"] = improvement("right_alpha_rms")

        base_tilt = finite_float(baseline.get("right_tilt_rms"))
        tilt = finite_float(row.get("right_tilt_rms"))
        row["tilt_change_pct"] = (
            100.0 * (tilt - base_tilt) / base_tilt
            if math.isfinite(base_tilt)
            and math.isfinite(tilt)
            and base_tilt != 0
            else math.nan
        )

    return aggregates, baseline


AGG_FIELDS = (
    ["qa", "qalpha", "samples", "acceptable"]
    + [
        item
        for metric in AGG_METRICS
        for item in (metric, f"{metric}_std")
    ]
    + [
        "acc_improvement_pct",
        "alpha_improvement_pct",
        "tilt_change_pct",
    ]
)


def recommendation_rows(
    aggregates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    safe = [row for row in aggregates if is_acceptable(row)]
    if not safe:
        return []

    bounds = normalize_objectives(safe)
    profiles = {
        "balanced": {
            "right_acc_rms": 0.45,
            "right_alpha_rms": 0.30,
            "right_tilt_rms": 0.20,
            "ddq_saturation_any_fraction": 0.05,
        },
        "linear_priority": {
            "right_acc_rms": 0.65,
            "right_alpha_rms": 0.15,
            "right_tilt_rms": 0.15,
            "ddq_saturation_any_fraction": 0.05,
        },
        "angular_priority": {
            "right_acc_rms": 0.25,
            "right_alpha_rms": 0.55,
            "right_tilt_rms": 0.15,
            "ddq_saturation_any_fraction": 0.05,
        },
        "conservative": {
            "right_acc_rms": 0.30,
            "right_alpha_rms": 0.20,
            "right_tilt_rms": 0.40,
            "ddq_saturation_any_fraction": 0.10,
        },
    }

    output: list[dict[str, Any]] = []
    for profile, weights in profiles.items():
        ranked = sorted(
            safe,
            key=lambda row: score_row(row, bounds, weights),
        )
        for rank, row in enumerate(ranked[:10], start=1):
            record = dict(row)
            record["profile"] = profile
            record["rank"] = rank
            record["score"] = score_row(row, bounds, weights)
            output.append(record)
    return output


RECOMMENDATION_FIELDS = [
    "profile",
    "rank",
    "score",
    "qa",
    "qalpha",
    "right_acc_rms",
    "right_alpha_rms",
    "right_tilt_rms",
    "right_tilt_std",
    "ddq_saturation_any_fraction",
    "qp_success",
    "q_safety_violation_fraction",
    "arm_interval_p99_ms",
    "acc_improvement_pct",
    "alpha_improvement_pct",
    "tilt_change_pct",
]


def format_candidate(row: dict[str, Any] | None) -> str:
    if row is None:
        return "无"
    return (
        f"QA={finite_float(row.get('qa')):.6g}, "
        f"Qalpha={finite_float(row.get('qalpha')):.6g}, "
        f"a_RMS={finite_float(row.get('right_acc_rms')):.4f}, "
        f"alpha_RMS={finite_float(row.get('right_alpha_rms')):.4f}, "
        f"tilt_RMS={finite_float(row.get('right_tilt_rms')):.5f}, "
        f"DDQ触边={100*finite_float(row.get('ddq_saturation_any_fraction')):.2f}%, "
        f"QP成功={100*finite_float(row.get('qp_success')):.2f}%"
    )


def write_human_summary(
    path: Path,
    sweep_id: str,
    total_planned: int,
    raw_rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
) -> None:
    safe = [row for row in aggregates if is_acceptable(row)]

    def best(metric: str) -> dict[str, Any] | None:
        return (
            min(safe, key=lambda row: finite_float(row.get(metric)))
            if safe
            else None
        )

    def top_profile(name: str) -> dict[str, Any] | None:
        candidates = [
            row
            for row in recommendations
            if row.get("profile") == name and row.get("rank") == 1
        ]
        return candidates[0] if candidates else None

    lines = [
        "MPC Q_A / Q_alpha 全二维扫描摘要",
        "=" * 72,
        f"扫描 ID：{sweep_id}",
        f"计划运行次数：{total_planned}",
        f"已完成有效运行：{len(raw_rows)}",
        f"已覆盖参数组合：{len(aggregates)} / {len(QA_VALUES) * len(QALPHA_VALUES)}",
        f"满足硬门槛的参数组合：{len(safe)}",
        f"Pareto 前沿参数组合：{len(frontier)}",
        "",
        "自动推荐使用的硬门槛：",
        f"- QP 成功率 >= {100*QP_SUCCESS_MIN:.1f}%",
        f"- 右手倾斜 RMS <= {TILT_RMS_MAX:.3f}",
        f"- DDQ 任一关节触边比例 <= {100*DDQ_SATURATION_MAX:.1f}%",
        "- 预测关节安全盒违反比例 = 0",
        "- 6 ms 右臂控制区间超时比例 = 0",
        "",
        "关键候选：",
        f"- 线加速度最低：{format_candidate(best('right_acc_rms'))}",
        f"- 角加速度最低：{format_candidate(best('right_alpha_rms'))}",
        f"- 姿态最稳：{format_candidate(best('right_tilt_rms'))}",
        f"- 综合平衡第一：{format_candidate(top_profile('balanced'))}",
        f"- 线加速度优先第一：{format_candidate(top_profile('linear_priority'))}",
        f"- 角加速度优先第一：{format_candidate(top_profile('angular_priority'))}",
        f"- 保守第一：{format_candidate(top_profile('conservative'))}",
        "",
        "注意：",
        "- “综合推荐”是对当前扫描范围内四项指标归一化后的加权距离，",
        "  不是数学意义上的全局最优证明。",
        "- 最终选择应结合 Pareto 前沿、录像/曲线和你的任务偏好判断。",
        "- 如果最佳点仍落在 QA 或 Qalpha 的搜索边界，说明还应向该方向扩展。",
        "",
        "生成文件：",
        "- raw_results.csv：每次运行的原始汇总",
        "- grid_aggregate.csv：相同参数重复运行后的均值和标准差",
        "- pareto_frontier.csv：未被其他安全组合全面支配的参数",
        "- recommended_candidates.csv：四种偏好的前十名",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_analysis(
    group_dir: Path,
    analysis_dir: Path,
    sweep_id: str,
    total_planned: int,
) -> None:
    raw_rows = scan_results(group_dir)
    write_csv(analysis_dir / "raw_results.csv", raw_rows, RAW_FIELDS)

    aggregates = aggregate_results(raw_rows)
    aggregates, _ = add_analysis_columns(aggregates)
    write_csv(
        analysis_dir / "grid_aggregate.csv",
        aggregates,
        list(AGG_FIELDS),
    )

    frontier = pareto_front(aggregates)
    write_csv(
        analysis_dir / "pareto_frontier.csv",
        frontier,
        list(AGG_FIELDS),
    )

    recommendations = recommendation_rows(aggregates)
    write_csv(
        analysis_dir / "recommended_candidates.csv",
        recommendations,
        RECOMMENDATION_FIELDS,
    )

    write_human_summary(
        analysis_dir / "SWEEP_SUMMARY.txt",
        sweep_id,
        total_planned,
        raw_rows,
        aggregates,
        recommendations,
        frontier,
    )


def main() -> int:
    args = parse_args()
    maybe_reexec_with_sleep_inhibit(args)

    repo_dir = Path(__file__).resolve().parent
    config_file = repo_dir / "configs" / "g1.yaml"
    run_script = repo_dir / "run.sh"
    sweep_logs_root = repo_dir / "sweep_logs"
    latest_file = sweep_logs_root / "latest_full_sweep_id.txt"

    if not config_file.is_file():
        print(f"[错误] 找不到 {config_file}", file=sys.stderr)
        return 2
    if not run_script.is_file():
        print(f"[错误] 找不到 {run_script}", file=sys.stderr)
        return 2
    if not os.access(run_script, os.X_OK):
        print("[错误] run.sh 没有执行权限；请运行 chmod +x run.sh", file=sys.stderr)
        return 2

    if args.resume is not None:
        requested = args.resume
        if requested == "latest":
            if not latest_file.is_file():
                print("[错误] 找不到最近一次 full sweep ID。", file=sys.stderr)
                return 2
            sweep_id = latest_file.read_text(encoding="utf-8").strip()
        else:
            sweep_id = requested
        if not sweep_id:
            print("[错误] resume 的 sweep ID 为空。", file=sys.stderr)
            return 2
    else:
        sweep_id = time.strftime("qa_qalpha_full_%Y%m%d_%H%M%S")

    group_dir = repo_dir / "evaluation" / sweep_id
    analysis_dir = sweep_logs_root / sweep_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(sweep_id + "\n", encoding="utf-8")

    backup_file = analysis_dir / "original_g1.yaml"
    if args.resume is not None and backup_file.is_file():
        original_config = backup_file.read_text(encoding="utf-8")
        # 上次如果被断电，g1.yaml 可能停留在某个扫描值；先恢复原始版本。
        config_file.write_text(original_config, encoding="utf-8")
    else:
        original_config = config_file.read_text(encoding="utf-8")
        validate_config(original_config)
        backup_file.write_text(original_config, encoding="utf-8")

    validate_config(original_config)

    restored = False

    def restore_config() -> None:
        nonlocal restored
        if restored:
            return
        try:
            config_file.write_text(original_config, encoding="utf-8")
            restored = True
            print("\n[恢复] 已恢复扫描开始前的 configs/g1.yaml")
        except OSError as exc:
            print(f"\n[严重警告] 恢复 g1.yaml 失败：{exc}", file=sys.stderr)

    atexit.register(restore_config)

    def signal_handler(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"收到信号 {signum}")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)

    cases = build_cases(args.repeats)
    total = len(cases)

    print("=" * 76)
    print("MPC Q_A / Q_alpha 全二维无人值守扫描")
    print(f"扫描 ID       : {sweep_id}")
    print(f"Q_A 数量      : {len(QA_VALUES)}")
    print(f"Q_alpha 数量  : {len(QALPHA_VALUES)}")
    print(f"每组重复次数  : {args.repeats}")
    print(f"计划运行次数  : {total}")
    print(f"评估结果目录  : {group_dir}")
    print(f"日志/汇总目录 : {analysis_dir}")
    print(f"大文件清理    : {'关闭' if args.no_prune else '开启'}")
    print("配置会在脚本退出时自动恢复。")
    print("=" * 76)

    if args.dry_run:
        for index, (repeat, qa, qalpha) in enumerate(cases, start=1):
            print(
                f"{index:03d}/{total}: repeat={repeat}, "
                f"QA={qa:.6g}, Qalpha={qalpha:.6g}"
            )
        return 0

    manifest = analysis_dir / "manifest.jsonl"
    timeout_seconds = args.timeout_minutes * 60.0

    try:
        for index, (repeat, qa, qalpha) in enumerate(cases, start=1):
            run_label = (
                f"grid_r{repeat:02d}_"
                f"qa{number_label(qa)}_"
                f"qalpha{number_label(qalpha)}"
            )

            if run_is_complete(group_dir, run_label):
                print(
                    f"[{index:03d}/{total}] 已完成，跳过："
                    f"QA={qa:.6g}, Qalpha={qalpha:.6g}, repeat={repeat}"
                )
                continue

            free_gb = shutil.disk_usage(repo_dir).free / (1024**3)
            if free_gb < args.min_free_gb:
                raise RuntimeError(
                    f"磁盘剩余 {free_gb:.2f} GB，低于门槛 "
                    f"{args.min_free_gb:.2f} GB；为避免写满磁盘，停止扫描。"
                )

            modified = replace_yaml_scalar(
                original_config,
                "mpc_q_ee_acc",
                qa,
            )
            modified = replace_yaml_scalar(
                modified,
                "mpc_q_ee_alpha",
                qalpha,
            )
            config_file.write_text(modified, encoding="utf-8")

            snapshot = analysis_dir / f"{run_label}.yaml"
            snapshot.write_text(modified, encoding="utf-8")
            log_file = analysis_dir / f"{run_label}.log"

            print(
                f"\n[{index:03d}/{total}] 开始 "
                f"QA={qa:.6g}, Qalpha={qalpha:.6g}, repeat={repeat} "
                f"(磁盘剩余 {free_gb:.1f} GB)"
            )

            started_at = time.strftime("%Y-%m-%d %H:%M:%S")
            return_code, timed_out, elapsed = run_one(
                run_script=run_script,
                group_dir=group_dir,
                sweep_id=sweep_id,
                run_label=run_label,
                log_file=log_file,
                timeout_seconds=timeout_seconds,
            )

            output_dir = find_output_dir(group_dir, run_label)
            complete = (
                output_dir is not None
                and (output_dir / "summary.json").is_file()
            )

            removed: list[str] = []
            if complete and output_dir is not None and not args.no_prune:
                # 先提取结果，再删掉几十 MB 的原始轨迹和 preview CSV。
                _ = extract_result(output_dir)
                removed = prune_heavy_artifacts(output_dir)

            record = {
                "time": started_at,
                "index": index,
                "total": total,
                "repeat": repeat,
                "qa": qa,
                "qalpha": qalpha,
                "run_label": run_label,
                "return_code": return_code,
                "timed_out": timed_out,
                "elapsed_seconds": elapsed,
                "complete": complete,
                "output_dir": str(output_dir) if output_dir else "",
                "log_file": str(log_file),
                "removed_heavy_artifacts": removed,
            }
            append_manifest(manifest, record)

            # 每轮后刷新一次，哪怕中途断电也保留当前 Pareto 和推荐结果。
            refresh_analysis(
                group_dir,
                analysis_dir,
                sweep_id,
                total,
            )

            if return_code == 0 and complete:
                print(
                    f"[完成] 用时 {elapsed:.1f} 秒；"
                    f"结果：{output_dir.name if output_dir else '未知'}"
                )
                if removed:
                    print(
                        f"[清理] 删除 {len(removed)} 个大型中间文件，"
                        "保留 JSON、图片和 metrics.npz。"
                    )
            else:
                print(
                    f"[失败] 返回码={return_code}，"
                    f"超时={timed_out}，完整结果={complete}",
                    file=sys.stderr,
                )
                tail = tail_text(log_file)
                if tail:
                    print("---- 日志末尾 ----", file=sys.stderr)
                    print(tail, file=sys.stderr)
                    print("------------------", file=sys.stderr)
                if args.stop_on_failure:
                    return return_code or 1

    except KeyboardInterrupt:
        print("\n[中断] 已收到停止信号。可稍后执行：")
        print(f"  ./sweep_qa_qalpha_full.py --resume {sweep_id}")
        return 130
    except Exception as exc:
        print(f"\n[错误] {exc}", file=sys.stderr)
        print("可在问题解决后断点续跑：")
        print(f"  ./sweep_qa_qalpha_full.py --resume {sweep_id}")
        return 1
    finally:
        # 确保结束前汇总已完成的所有结果。
        try:
            refresh_analysis(
                group_dir,
                analysis_dir,
                sweep_id,
                total,
            )
        except Exception as exc:
            print(f"[警告] 最终汇总失败：{exc}", file=sys.stderr)
        restore_config()

    print("\n" + "=" * 76)
    print("[全部完成]")
    print(f"结果目录       : {group_dir}")
    print(f"原始汇总       : {analysis_dir / 'raw_results.csv'}")
    print(f"网格均值       : {analysis_dir / 'grid_aggregate.csv'}")
    print(f"Pareto 前沿    : {analysis_dir / 'pareto_frontier.csv'}")
    print(f"推荐候选       : {analysis_dir / 'recommended_candidates.csv'}")
    print(f"文字摘要       : {analysis_dir / 'SWEEP_SUMMARY.txt'}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
