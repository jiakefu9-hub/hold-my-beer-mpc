#!/usr/bin/env python3
"""
Final validation for MPC Q_A / Q_alpha tuning.

Design:
- 6 candidates x 5 repeats = 30 runs.
- Cyclic candidate order across repeats to reduce thermal/runtime-order bias.
- Full run data stays under evaluation/ (Git-ignored).
- Logs and YAML snapshots stay under sweep_logs/ (Git-ignored).
- Lightweight, reviewable results are refreshed after every run under
  evaluation_summary/<validation_id>/ (Git-trackable).
- On successful completion, --apply-selected writes the task-priority winner
  into configs/g1.yaml. On interruption or failure, the original config is restored.

Run:
    chmod +x validate_qa_qalpha_final.py
    ./validate_qa_qalpha_final.py --apply-selected

Resume:
    ./validate_qa_qalpha_final.py --resume latest --apply-selected
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sweep_qa_qalpha_full import (
    append_manifest,
    extract_result,
    find_output_dir,
    prune_heavy_artifacts,
    replace_yaml_scalar,
    run_one,
    tail_text,
    validate_config,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    qa: float
    qalpha: float
    purpose: str


CANDIDATES = [
    Candidate(
        "baseline",
        0.0,
        0.0,
        "Zero acceleration weights; reference for improvement percentages.",
    ),
    Candidate(
        "posture_floor",
        0.0003,
        0.0,
        "Lowest tilt point from the full 154-point scan.",
    ),
    Candidate(
        "final_candidate",
        0.01,
        0.0005,
        "Proposed task-priority knee: low tilt and low DDQ saturation.",
    ),
    Candidate(
        "angular_priority",
        0.01,
        0.0015,
        "More angular-acceleration suppression at higher motion cost.",
    ),
    Candidate(
        "balanced_auto",
        0.015,
        0.0015,
        "Automatic balanced winner from the full scan.",
    ),
    Candidate(
        "linear_priority",
        0.02,
        0.0005,
        "Stronger linear-acceleration suppression.",
    ),
]

DEFAULT_REPEATS = 5

# Hard feasibility gates, identical to the full sweep.
QP_SUCCESS_MIN = 0.99
TILT_RMS_MAX = 0.030
DDQ_SATURATION_MAX = 0.10
SAFETY_VIOLATION_MAX = 0.0
ARM_INTERVAL_OVERRUN_MAX = 0.0

# Task-priority selection:
# tray orientation first, then linear/angular acceleration, then DDQ margin.
SELECTION_WEIGHTS = {
    "right_tilt_rms": 0.45,
    "right_acc_rms": 0.25,
    "right_alpha_rms": 0.20,
    "ddq_saturation_any_fraction": 0.10,
}

# A candidate must improve both acceleration metrics enough to count as a useful
# tuning over the zero-weight baseline. This prevents the posture-only reference
# from winning merely because it changes almost nothing.
MIN_ACC_IMPROVEMENT_PCT = 5.0
MIN_ALPHA_IMPROVEMENT_PCT = 5.0
TASK_TILT_RMS_MAX = 0.020

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

RAW_FIELDS = [
    "candidate",
    "repeat",
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


def maybe_reexec_validation_with_sleep_inhibit(args: argparse.Namespace) -> None:
    """Re-exec this validation script under systemd-inhibit when available."""
    if args.no_inhibit:
        return
    if os.environ.get("QA_QALPHA_FINAL_VALIDATION_INHIBITED") == "1":
        return

    inhibitor = shutil.which("systemd-inhibit")
    if inhibitor is None:
        print("[notice] systemd-inhibit not found; verify that the computer will not sleep.")
        return

    env = os.environ.copy()
    env["QA_QALPHA_FINAL_VALIDATION_INHIBITED"] = "1"
    command = [
        inhibitor,
        "--what=sleep:idle",
        "--mode=block",
        "--why=Final MPC QA and Qalpha validation",
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    os.execvpe(inhibitor, command, env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate final MPC Q_A/Q_alpha candidates and archive summaries."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Repeats per candidate, default {DEFAULT_REPEATS}.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="VALIDATION_ID",
        help="Resume an existing validation; omit ID or use latest for the newest one.",
    )
    parser.add_argument(
        "--apply-selected",
        action="store_true",
        help="After all planned runs succeed, write the selected QA/Qalpha to configs/g1.yaml.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=15.0,
        help="Timeout per run in minutes, default 15.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=3.0,
        help="Minimum free disk space before starting each run, default 3 GB.",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Keep trajectory and large preview files. Default is to remove them.",
    )
    parser.add_argument(
        "--no-inhibit",
        action="store_true",
        help="Do not use systemd-inhibit to block sleep.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failed run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned run order without running simulations.",
    )
    return parser.parse_args()


def finite(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def fmt(value: Any, digits: int = 5) -> str:
    number = finite(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def git_output(repo_dir: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def validate_repository(repo_dir: Path, *, resume: bool) -> str:
    if not (repo_dir / ".git").exists():
        raise RuntimeError(f"{repo_dir} is not a Git working tree.")

    branch = git_output(repo_dir, "branch", "--show-current")
    if branch != "feat/23dof_arm_mpc":
        raise RuntimeError(
            "This validation must run on feat/23dof_arm_mpc; "
            f"current branch is {branch!r}."
        )

    changed = git_output(repo_dir, "diff", "--name-only")
    changed_cached = git_output(repo_dir, "diff", "--cached", "--name-only")
    tracked_changes = sorted(
        set(filter(None, (changed + "\n" + changed_cached).splitlines()))
    )

    if resume:
        disallowed = [name for name in tracked_changes if name != "configs/g1.yaml"]
    else:
        disallowed = tracked_changes

    if disallowed:
        raise RuntimeError(
            "Tracked files already contain unrelated changes:\n  "
            + "\n  ".join(disallowed)
        )

    return branch


def case_order(repeats: int) -> list[tuple[int, Candidate]]:
    if repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    ordered: list[tuple[int, Candidate]] = []
    count = len(CANDIDATES)
    for repeat in range(1, repeats + 1):
        # Rotate order each repeat to spread temperature/runtime drift.
        offset = (repeat - 1) % count
        rotated = CANDIDATES[offset:] + CANDIDATES[:offset]
        ordered.extend((repeat, candidate) for candidate in rotated)
    return ordered


def candidate_by_pair(qa: float, qalpha: float) -> Candidate | None:
    for candidate in CANDIDATES:
        if abs(candidate.qa - qa) <= 1e-12 and abs(candidate.qalpha - qalpha) <= 1e-12:
            return candidate
    return None


RUN_LABEL_RE = re.compile(
    r"_(" + "|".join(re.escape(candidate.name) for candidate in CANDIDATES)
    + r")_r(?P<repeat>\d+)$"
)


def scan_rows(group_dir: Path, repo_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not group_dir.is_dir():
        return rows

    for output_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
        row = extract_result(output_dir)
        if row is None:
            continue

        qa = finite(row.get("qa"))
        qalpha = finite(row.get("qalpha"))
        candidate = candidate_by_pair(qa, qalpha)

        match = RUN_LABEL_RE.search(output_dir.name)
        repeat = int(match.group("repeat")) if match else 0
        label = (
            match.group(1)
            if match
            else candidate.name if candidate is not None else "unknown"
        )

        row["candidate"] = label
        row["repeat"] = repeat
        try:
            row["run_dir"] = str(output_dir.relative_to(repo_dir))
        except ValueError:
            row["run_dir"] = str(output_dir)
        rows.append(row)

    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate", "unknown")), []).append(row)

    output: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        group = grouped.get(candidate.name, [])
        result: dict[str, Any] = {
            "candidate": candidate.name,
            "purpose": candidate.purpose,
            "qa": candidate.qa,
            "qalpha": candidate.qalpha,
            "samples": len(group),
        }

        for metric in AGG_METRICS:
            values = [
                finite(row.get(metric))
                for row in group
                if math.isfinite(finite(row.get(metric)))
            ]
            result[metric] = statistics.mean(values) if values else math.nan
            result[f"{metric}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
            result[f"{metric}_min"] = min(values) if values else math.nan
            result[f"{metric}_max"] = max(values) if values else math.nan

        # Control/safety feasibility is separated from host scheduling jitter.
        # The candidate parameters do not change the MPC dimensions or control
        # period. Isolated Linux scheduling overruns are reported separately,
        # but do not eliminate an otherwise safe controller tuning.
        result["hard_acceptable"] = (
            len(group) > 0
            and finite(result["qp_success_min"]) >= QP_SUCCESS_MIN
            and finite(result["right_tilt_rms_max"]) <= TILT_RMS_MAX
            and finite(result["ddq_saturation_any_fraction_max"])
            <= DDQ_SATURATION_MAX
            and finite(result["q_safety_violation_fraction_max"])
            <= SAFETY_VIOLATION_MAX
        )
        result["timing_clean_all_repeats"] = (
            len(group) > 0
            and finite(result["arm_interval_overrun_fraction_max"])
            <= ARM_INTERVAL_OVERRUN_MAX
        )
        output.append(result)

    baseline = next(
        (row for row in output if row["candidate"] == "baseline"),
        None,
    )
    for row in output:
        def improvement(metric: str) -> float:
            if baseline is None:
                return math.nan
            base = finite(baseline.get(metric))
            current = finite(row.get(metric))
            if not math.isfinite(base) or not math.isfinite(current) or base == 0.0:
                return math.nan
            return 100.0 * (base - current) / base

        row["acc_improvement_pct"] = improvement("right_acc_rms")
        row["alpha_improvement_pct"] = improvement("right_alpha_rms")

        base_tilt = finite(baseline.get("right_tilt_rms")) if baseline else math.nan
        current_tilt = finite(row.get("right_tilt_rms"))
        row["tilt_change_pct"] = (
            100.0 * (current_tilt - base_tilt) / base_tilt
            if math.isfinite(base_tilt)
            and math.isfinite(current_tilt)
            and base_tilt != 0.0
            else math.nan
        )

        row["task_eligible"] = (
            bool(row["hard_acceptable"])
            and finite(row["acc_improvement_pct"]) >= MIN_ACC_IMPROVEMENT_PCT
            and finite(row["alpha_improvement_pct"]) >= MIN_ALPHA_IMPROVEMENT_PCT
            and finite(row["right_tilt_rms"]) <= TASK_TILT_RMS_MAX
        )

    eligible = [row for row in output if row["task_eligible"]]
    bounds: dict[str, tuple[float, float]] = {}
    for metric in SELECTION_WEIGHTS:
        values = [finite(row[metric]) for row in eligible]
        values = [value for value in values if math.isfinite(value)]
        bounds[metric] = (
            min(values) if values else 0.0,
            max(values) if values else 1.0,
        )

    for row in output:
        if not row["task_eligible"]:
            row["task_score"] = math.nan
            continue
        terms = []
        for metric, weight in SELECTION_WEIGHTS.items():
            low, high = bounds[metric]
            value = finite(row[metric])
            normalized = 0.0 if high <= low + 1e-15 else (value - low) / (high - low)
            terms.append(weight * normalized**2)
        row["task_score"] = math.sqrt(sum(terms))

    return output


def select_candidate(aggregates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        row
        for row in aggregates
        if bool(row.get("task_eligible"))
        and math.isfinite(finite(row.get("task_score")))
    ]
    if eligible:
        return min(eligible, key=lambda row: finite(row["task_score"]))

    safe = [row for row in aggregates if bool(row.get("hard_acceptable"))]
    if safe:
        return min(
            safe,
            key=lambda row: (
                finite(row.get("right_tilt_rms"), math.inf),
                finite(row.get("ddq_saturation_any_fraction"), math.inf),
            ),
        )
    return None


AGG_FIELDS = [
    "candidate",
    "purpose",
    "qa",
    "qalpha",
    "samples",
    "hard_acceptable",
    "task_eligible",
    "task_score",
    "timing_clean_all_repeats",
    "right_acc_rms",
    "right_acc_rms_std",
    "right_alpha_rms",
    "right_alpha_rms_std",
    "right_tilt_rms",
    "right_tilt_rms_std",
    "ddq_saturation_any_fraction",
    "ddq_saturation_any_fraction_std",
    "qp_success",
    "qp_success_std",
    "q_safety_violation_fraction",
    "arm_interval_p99_ms",
    "arm_interval_overrun_fraction",
    "left_acc_rms",
    "left_alpha_rms",
    "left_tilt_rms",
    "acc_improvement_pct",
    "alpha_improvement_pct",
    "tilt_change_pct",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    validation_id: str,
    repeats: int,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    total_planned: int,
) -> None:
    completed = len(rows)
    lines = [
        "# Final MPC QA/Qalpha validation",
        "",
        f"- Validation ID: `{validation_id}`",
        f"- Planned runs: {total_planned}",
        f"- Completed runs: {completed}",
        f"- Repeats per candidate: {repeats}",
        "- Candidate order was cyclically rotated between repeats.",
        "",
        "## Selection policy",
        "",
        "Hard gates:",
        f"- QP success in every repeat >= {100 * QP_SUCCESS_MIN:.1f}%",
        f"- Tilt RMS in every repeat <= {TILT_RMS_MAX:.3f} rad",
        f"- DDQ saturation in every repeat <= {100 * DDQ_SATURATION_MAX:.1f}%",
        "- Joint safety-box violations = 0",
        "- Host timing overruns are reported separately and are not used to rank controller weights.",
        "",
        "Task eligibility:",
        f"- Linear-acceleration improvement over baseline >= {MIN_ACC_IMPROVEMENT_PCT:.1f}%",
        f"- Angular-acceleration improvement over baseline >= {MIN_ALPHA_IMPROVEMENT_PCT:.1f}%",
        f"- Mean tilt RMS <= {TASK_TILT_RMS_MAX:.3f} rad",
        "",
        "Task-priority score weights:",
        "- tilt RMS: 45%",
        "- linear acceleration RMS: 25%",
        "- angular acceleration RMS: 20%",
        "- DDQ saturation: 10%",
        "",
        "## Candidate means",
        "",
        "| candidate | QA | Qalpha | n | acc RMS | alpha RMS | tilt RMS rad | DDQ sat | QP | acc imp. | alpha imp. | eligible | score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]

    for row in aggregates:
        score = fmt(row.get("task_score"), 4)
        lines.append(
            "| {candidate} | {qa:.6g} | {qalpha:.6g} | {samples} | "
            "{acc} ± {acc_std} | {alpha} ± {alpha_std} | "
            "{tilt} ± {tilt_std} | {ddq}% | {qp}% | "
            "{acc_imp}% | {alpha_imp}% | {eligible} | {score} |".format(
                candidate=row["candidate"],
                qa=float(row["qa"]),
                qalpha=float(row["qalpha"]),
                samples=int(row["samples"]),
                acc=fmt(row.get("right_acc_rms"), 4),
                acc_std=fmt(row.get("right_acc_rms_std"), 4),
                alpha=fmt(row.get("right_alpha_rms"), 4),
                alpha_std=fmt(row.get("right_alpha_rms_std"), 4),
                tilt=fmt(row.get("right_tilt_rms"), 5),
                tilt_std=fmt(row.get("right_tilt_rms_std"), 5),
                ddq=fmt(100.0 * finite(row.get("ddq_saturation_any_fraction")), 2),
                qp=fmt(100.0 * finite(row.get("qp_success")), 2),
                acc_imp=fmt(row.get("acc_improvement_pct"), 2),
                alpha_imp=fmt(row.get("alpha_improvement_pct"), 2),
                eligible="yes" if row.get("task_eligible") else "no",
                score=score,
            )
        )

    lines.extend(["", "## Selected parameters", ""])
    if selected is None:
        lines.append(
            "No candidate currently passes the selection conditions. "
            "Inspect failures and rerun before changing the default configuration."
        )
    else:
        lines.extend(
            [
                f"- Candidate: **{selected['candidate']}**",
                f"- `mpc_q_ee_acc: {float(selected['qa']):.12g}`",
                f"- `mpc_q_ee_alpha: {float(selected['qalpha']):.12g}`",
                f"- Task score: {fmt(selected.get('task_score'), 5)}",
                f"- Mean tilt RMS: {fmt(selected.get('right_tilt_rms'), 6)} rad",
                f"- Mean linear acceleration RMS: {fmt(selected.get('right_acc_rms'), 6)}",
                f"- Mean angular acceleration RMS: {fmt(selected.get('right_alpha_rms'), 6)}",
                f"- Mean DDQ saturation: {fmt(100.0 * finite(selected.get('ddq_saturation_any_fraction')), 3)}%",
                "",
                "The score is a task-oriented ranking within this candidate set, "
                "not a proof of a global optimum.",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selected_yaml(
    path: Path,
    selected: dict[str, Any] | None,
    completed: int,
    total: int,
) -> None:
    lines = [
        f"completed_runs: {completed}",
        f"planned_runs: {total}",
    ]
    if selected is None:
        lines.append("selection_status: no_valid_candidate")
    else:
        lines.extend(
            [
                "selection_status: selected",
                f"candidate: {selected['candidate']}",
                f"mpc_q_ee_acc: {float(selected['qa']):.12g}",
                f"mpc_q_ee_alpha: {float(selected['qalpha']):.12g}",
                f"task_score: {finite(selected.get('task_score')):.12g}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(summary_dir: Path, aggregates: list[dict[str, Any]]) -> None:
    rows = [row for row in aggregates if int(row.get("samples", 0)) > 0]
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[notice] plot generation skipped: {exc}")
        return

    labels = [str(row["candidate"]) for row in rows]
    metrics = [
        ("right_acc_rms", "Linear acceleration RMS"),
        ("right_alpha_rms", "Angular acceleration RMS"),
        ("right_tilt_rms", "Tilt RMS [rad]"),
        ("ddq_saturation_any_fraction", "DDQ saturation fraction"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, (metric, title) in zip(axes.flat, metrics):
        values = [finite(row.get(metric)) for row in rows]
        errors = [finite(row.get(f"{metric}_std"), 0.0) for row in rows]
        axis.bar(labels, values, yerr=errors, capsize=3)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("Final QA/Qalpha candidate validation")
    figure.tight_layout()
    figure.savefig(summary_dir / "candidate_comparison.png", dpi=180)
    plt.close(figure)


def refresh_outputs(
    *,
    repo_dir: Path,
    group_dir: Path,
    analysis_dir: Path,
    summary_dir: Path,
    validation_id: str,
    repeats: int,
    total_planned: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    rows = scan_rows(group_dir, repo_dir)
    aggregates = aggregate_rows(rows)
    selected = select_candidate(aggregates)

    write_csv(analysis_dir / "raw_results.csv", rows, RAW_FIELDS)
    write_csv(analysis_dir / "candidate_aggregate.csv", aggregates, AGG_FIELDS)
    write_markdown(
        analysis_dir / "FINAL_VALIDATION_SUMMARY.md",
        validation_id=validation_id,
        repeats=repeats,
        rows=rows,
        aggregates=aggregates,
        selected=selected,
        total_planned=total_planned,
    )
    write_selected_yaml(
        analysis_dir / "selected_parameters.yaml",
        selected,
        len(rows),
        total_planned,
    )

    summary_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "raw_results.csv",
        "candidate_aggregate.csv",
        "FINAL_VALIDATION_SUMMARY.md",
        "selected_parameters.yaml",
        "manifest.jsonl",
        "original_g1.yaml",
        "validation_plan.json",
    ):
        source = analysis_dir / name
        if source.is_file():
            shutil.copy2(source, summary_dir / name)

    make_plots(summary_dir, aggregates)

    completion = {
        "validation_id": validation_id,
        "planned_runs": total_planned,
        "completed_runs": len(rows),
        "complete": len(rows) == total_planned,
        "selected_candidate": selected["candidate"] if selected else None,
        "selected_qa": selected["qa"] if selected else None,
        "selected_qalpha": selected["qalpha"] if selected else None,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (summary_dir / "completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return rows, aggregates, selected


def apply_selected_config(
    config_file: Path,
    original_config: str,
    selected: dict[str, Any],
) -> None:
    updated = replace_yaml_scalar(
        original_config,
        "mpc_q_ee_acc",
        float(selected["qa"]),
    )
    updated = replace_yaml_scalar(
        updated,
        "mpc_q_ee_alpha",
        float(selected["qalpha"]),
    )
    validate_config(updated)
    config_file.write_text(updated, encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    maybe_reexec_validation_with_sleep_inhibit(args)

    repo_dir = Path(__file__).resolve().parents[2]
    branch = validate_repository(repo_dir, resume=args.resume is not None)

    config_file = repo_dir / "configs" / "g1.yaml"
    run_script = repo_dir / "run.sh"
    evaluation_root = repo_dir / "evaluation"
    sweep_logs_root = repo_dir / "sweep_logs"
    summary_root = repo_dir / "evaluation_summary"
    latest_file = sweep_logs_root / "latest_final_validation_id.txt"

    if not config_file.is_file():
        raise RuntimeError(f"Missing config: {config_file}")
    if not run_script.is_file() or not os.access(run_script, os.X_OK):
        raise RuntimeError("run.sh is missing or not executable.")
    if not (repo_dir / "sweep_qa_qalpha_full.py").is_file():
        raise RuntimeError("sweep_qa_qalpha_full.py is required beside this script.")

    cases = case_order(args.repeats)
    total_planned = len(cases)

    if args.resume is not None:
        if args.resume == "latest":
            if not latest_file.is_file():
                raise RuntimeError("No latest final validation ID was found.")
            validation_id = latest_file.read_text(encoding="utf-8").strip()
        else:
            validation_id = args.resume.strip()
        if not validation_id:
            raise RuntimeError("Resume validation ID is empty.")
    else:
        validation_id = time.strftime("qa_qalpha_final_validation_%Y%m%d_%H%M%S")

    group_dir = evaluation_root / validation_id
    analysis_dir = sweep_logs_root / validation_id
    summary_dir = summary_root / validation_id

    group_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(validation_id + "\n", encoding="utf-8")

    backup_file = analysis_dir / "original_g1.yaml"
    if args.resume is not None and backup_file.is_file():
        original_config = backup_file.read_text(encoding="utf-8")
        config_file.write_text(original_config, encoding="utf-8", newline="\n")
    else:
        original_config = config_file.read_text(encoding="utf-8")
        validate_config(original_config)
        backup_file.write_text(original_config, encoding="utf-8", newline="\n")

    plan = {
        "validation_id": validation_id,
        "branch": branch,
        "git_head": git_output(repo_dir, "rev-parse", "HEAD"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repeats": args.repeats,
        "planned_runs": total_planned,
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "selection_weights": SELECTION_WEIGHTS,
        "task_eligibility": {
            "min_acc_improvement_pct": MIN_ACC_IMPROVEMENT_PCT,
            "min_alpha_improvement_pct": MIN_ALPHA_IMPROVEMENT_PCT,
            "mean_tilt_rms_max": TASK_TILT_RMS_MAX,
        },
        "hard_gates": {
            "qp_success_min": QP_SUCCESS_MIN,
            "tilt_rms_max": TILT_RMS_MAX,
            "ddq_saturation_max": DDQ_SATURATION_MAX,
            "safety_violation_max": SAFETY_VIOLATION_MAX,
            "arm_interval_overrun_report_only": True,
        },
    }
    (analysis_dir / "validation_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    restored = False
    final_applied = False

    def restore_config() -> None:
        nonlocal restored
        if restored or final_applied:
            return
        try:
            config_file.write_text(original_config, encoding="utf-8", newline="\n")
            restored = True
            print("\n[restore] configs/g1.yaml restored.")
        except OSError as exc:
            print(f"\n[critical] failed to restore g1.yaml: {exc}", file=sys.stderr)

    atexit.register(restore_config)

    def signal_handler(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)

    print("=" * 78)
    print("Final MPC QA/Qalpha validation")
    print(f"Validation ID    : {validation_id}")
    print(f"Branch           : {branch}")
    print(f"Candidates       : {len(CANDIDATES)}")
    print(f"Repeats          : {args.repeats}")
    print(f"Planned runs     : {total_planned}")
    print(f"Full data        : {group_dir}")
    print(f"Local logs       : {analysis_dir}")
    print(f"Git summary      : {summary_dir}")
    print(f"Heavy pruning    : {'off' if args.no_prune else 'on'}")
    print(f"Apply selection  : {args.apply_selected}")
    print("=" * 78)

    if args.dry_run:
        for index, (repeat, candidate) in enumerate(cases, start=1):
            print(
                f"{index:02d}/{total_planned}: repeat={repeat}, "
                f"{candidate.name}, QA={candidate.qa:.6g}, "
                f"Qalpha={candidate.qalpha:.6g}"
            )
        refresh_outputs(
            repo_dir=repo_dir,
            group_dir=group_dir,
            analysis_dir=analysis_dir,
            summary_dir=summary_dir,
            validation_id=validation_id,
            repeats=args.repeats,
            total_planned=total_planned,
        )
        return 0

    manifest = analysis_dir / "manifest.jsonl"
    timeout_seconds = args.timeout_minutes * 60.0
    failed_runs = 0

    try:
        for index, (repeat, candidate) in enumerate(cases, start=1):
            run_label = f"{candidate.name}_r{repeat:02d}"
            output_dir = find_output_dir(group_dir, run_label)
            if output_dir is not None and (output_dir / "summary.json").is_file():
                print(
                    f"[{index:02d}/{total_planned}] already complete: "
                    f"{run_label}"
                )
                continue

            free_gb = shutil.disk_usage(repo_dir).free / (1024**3)
            if free_gb < args.min_free_gb:
                raise RuntimeError(
                    f"Free space {free_gb:.2f} GB is below "
                    f"{args.min_free_gb:.2f} GB."
                )

            modified = replace_yaml_scalar(
                original_config,
                "mpc_q_ee_acc",
                candidate.qa,
            )
            modified = replace_yaml_scalar(
                modified,
                "mpc_q_ee_alpha",
                candidate.qalpha,
            )
            validate_config(modified)
            config_file.write_text(modified, encoding="utf-8", newline="\n")

            snapshot = analysis_dir / f"{run_label}.yaml"
            snapshot.write_text(modified, encoding="utf-8", newline="\n")
            log_file = analysis_dir / f"{run_label}.log"

            print(
                f"\n[{index:02d}/{total_planned}] {candidate.name}, "
                f"repeat={repeat}, QA={candidate.qa:.6g}, "
                f"Qalpha={candidate.qalpha:.6g}, free={free_gb:.1f} GB"
            )

            started_at = time.strftime("%Y-%m-%d %H:%M:%S")
            return_code, timed_out, elapsed = run_one(
                run_script=run_script,
                group_dir=group_dir,
                sweep_id=validation_id,
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
                _ = extract_result(output_dir)
                removed = prune_heavy_artifacts(output_dir)

            record = {
                "time": started_at,
                "index": index,
                "total": total_planned,
                "candidate": candidate.name,
                "repeat": repeat,
                "qa": candidate.qa,
                "qalpha": candidate.qalpha,
                "run_label": run_label,
                "return_code": return_code,
                "timed_out": timed_out,
                "elapsed_seconds": elapsed,
                "complete": complete,
                "output_dir": (
                    str(output_dir.relative_to(repo_dir))
                    if output_dir is not None
                    else ""
                ),
                "log_file": str(log_file.relative_to(repo_dir)),
                "removed_heavy_artifacts": removed,
            }
            append_manifest(manifest, record)

            rows, _, selected = refresh_outputs(
                repo_dir=repo_dir,
                group_dir=group_dir,
                analysis_dir=analysis_dir,
                summary_dir=summary_dir,
                validation_id=validation_id,
                repeats=args.repeats,
                total_planned=total_planned,
            )

            if return_code == 0 and complete:
                chosen = selected["candidate"] if selected else "none yet"
                print(
                    f"[done] {elapsed:.1f}s, completed={len(rows)}/"
                    f"{total_planned}, current selection={chosen}"
                )
                if removed:
                    print(f"[prune] removed {len(removed)} large artifacts.")
            else:
                failed_runs += 1
                print(
                    f"[failed] code={return_code}, timeout={timed_out}, "
                    f"complete={complete}",
                    file=sys.stderr,
                )
                tail = tail_text(log_file)
                if tail:
                    print("---- log tail ----", file=sys.stderr)
                    print(tail, file=sys.stderr)
                    print("------------------", file=sys.stderr)
                if args.stop_on_failure:
                    return return_code or 1

    except KeyboardInterrupt:
        print("\n[interrupted] Resume with:")
        print(
            f"  ./validate_qa_qalpha_final.py "
            f"--resume {validation_id} --apply-selected"
        )
        return 130
    except Exception as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        print("Resume after fixing the problem with:")
        print(
            f"  ./validate_qa_qalpha_final.py "
            f"--resume {validation_id} --apply-selected"
        )
        return 1
    finally:
        try:
            rows, aggregates, selected = refresh_outputs(
                repo_dir=repo_dir,
                group_dir=group_dir,
                analysis_dir=analysis_dir,
                summary_dir=summary_dir,
                validation_id=validation_id,
                repeats=args.repeats,
                total_planned=total_planned,
            )
        except Exception as exc:
            print(f"[warning] final summary refresh failed: {exc}", file=sys.stderr)
            rows, aggregates, selected = [], [], None

        if not final_applied:
            restore_config()

    complete_count = len(rows)
    all_complete = complete_count == total_planned

    if not all_complete:
        print(
            f"\n[warning] only {complete_count}/{total_planned} runs are complete. "
            "The default configuration was not changed."
        )
        return 1

    if failed_runs:
        print(
            f"\n[warning] {failed_runs} attempts failed, but all planned run labels "
            "now have complete results."
        )

    if selected is None:
        print("\n[warning] no candidate passed selection conditions.")
        return 1

    if args.apply_selected:
        apply_selected_config(config_file, original_config, selected)
        final_applied = True
        restored = True
        print(
            f"\n[apply] configs/g1.yaml -> "
            f"QA={float(selected['qa']):.6g}, "
            f"Qalpha={float(selected['qalpha']):.6g}"
        )

    print("\n" + "=" * 78)
    print("[validation complete]")
    print(f"Selected         : {selected['candidate']}")
    print(f"QA               : {float(selected['qa']):.6g}")
    print(f"Qalpha           : {float(selected['qalpha']):.6g}")
    print(f"Task score       : {fmt(selected.get('task_score'), 6)}")
    print(f"Git summary      : {summary_dir}")
    print(
        "Next: inspect FINAL_VALIDATION_SUMMARY.md and candidate_comparison.png "
        "before committing."
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
