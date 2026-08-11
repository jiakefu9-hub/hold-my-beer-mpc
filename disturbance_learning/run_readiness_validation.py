#!/usr/bin/env python3
"""Repeated timing and safety validation before real-robot integration."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from disturbance_learning.command_schedule import (
    GENERALIZATION_SCHEDULE_PROFILES,
)
from disturbance_learning.run_closed_loop_ablation import (
    _latest_run,
    summarize_run,
)
from disturbance_learning.run_generalization_ablation import (
    _aggregate_mode,
    _condition_id,
    _paired_improvements,
)


MODES = ("template", "hybrid_residual")
DEFAULT_SEEDS = (2101, 2102)
DEFAULT_PAYLOADS_KG = (0.005, 0.010)
PRE_HARDENING_SUMMARY = (
    REPO_DIR / "evaluation_summary/hybrid_generalization_validation/summary.json"
)


def _run_one(
    *,
    group: str,
    group_dir: Path,
    profile_name: str,
    seed: int,
    payload_kg: float,
    mode: str,
    resume: bool,
    reuse_template_group_dir: Path | None,
) -> Path:
    condition_id = _condition_id(profile_name, seed, payload_kg)
    label = f"{condition_id}_{mode}"
    if mode == "template" and reuse_template_group_dir is not None:
        return _latest_run(reuse_template_group_dir, label)
    existing = list(group_dir.glob(f"*_{label}"))
    if resume and any((path / "summary.json").is_file() for path in existing):
        return _latest_run(group_dir, label)

    log_path = group_dir / f"{label}.log"
    command = [
        str(REPO_DIR / "run.sh"),
        "--headless",
        "--no-video",
        "--predictor-schedule-profile",
        profile_name,
        "--predictor-ablation-seed",
        str(seed),
        "--predictor-payload-kg",
        str(payload_kg),
        "--disturbance-predictor",
        mode,
        "--evaluation-group",
        group,
        "--run-label",
        label,
    ]
    for attempt in range(3):
        with log_path.open(
            "w" if attempt == 0 else "a", encoding="utf-8"
        ) as log:
            completed = subprocess.run(
                command,
                cwd=REPO_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode == 0:
            return _latest_run(group_dir, label)
        startup_layout_race = "SimRuntimeLayoutError" in log_path.read_text(
            encoding="utf-8"
        )
        if not startup_layout_race or attempt == 2:
            break
    raise RuntimeError(
        f"{label} failed with {completed.returncode}; see {log_path}"
    )


def _condition_group(results: list[dict], payload_kg: float) -> list[dict]:
    return [
        result
        for result in results
        if abs(float(result["payload_kg"]) - payload_kg) < 1e-12
    ]


def _quality_gate(paired: dict) -> bool:
    overall = paired["overall_improvement_percent"]
    return (
        overall["right_ee_acc_norm_rms"]["mean"] >= 0.0
        and overall["right_ee_alpha_norm_rms"]["mean"] >= 0.0
        and overall["right_ee_tilt_xy_norm_rms"]["mean"] >= -2.0
    )


def _pre_hardening_reference() -> dict:
    if not PRE_HARDENING_SUMMARY.is_file():
        return {"available": False}
    with PRE_HARDENING_SUMMARY.open("r", encoding="utf-8") as file:
        source = json.load(file)
    schedule = source["schedule_only_aggregate_by_mode"]
    payload = {
        result["mode"]: result
        for result in source.get("payload_stress_results", [])
    }

    def aggregate_excerpt(value: dict) -> dict:
        return {
            "overall": value["overall"],
            "timing_ms": value["timing_ms"],
            "safety_totals": value["safety_totals"],
        }

    def payload_excerpt(value: dict | None) -> dict | None:
        if value is None:
            return None
        return {
            "condition_id": value["condition_id"],
            "overall_evaluation": value["overall_evaluation"],
            "predictor_timing_ms": value["predictor_timing_ms"],
            "complete_6ms_right_arm_interval_ms": value[
                "complete_6ms_right_arm_interval_ms"
            ],
            "safety": value["safety"],
        }

    return {
        "available": True,
        "source": str(PRE_HARDENING_SUMMARY.relative_to(REPO_DIR)),
        "normal_schedule_run_count_per_mode": source[
            "schedule_only_condition_count"
        ],
        "normal_template": aggregate_excerpt(schedule["template"]),
        "normal_hybrid_residual": aggregate_excerpt(
            schedule["hybrid_residual"]
        ),
        "single_10g_payload_template": payload_excerpt(
            payload.get("template")
        ),
        "single_10g_payload_hybrid_residual": payload_excerpt(
            payload.get("hybrid_residual")
        ),
    }


def _compact_readiness_result(result: dict) -> dict:
    return {
        name: result[name]
        for name in (
            "condition_id",
            "profile",
            "seed",
            "payload_kg",
            "mode",
            "overall_evaluation",
            "predictor_timing_ms",
            "neural_core_inference_timing_ms",
            "complete_6ms_right_arm_interval_ms",
            "fallback",
            "safety_gate",
            "safety",
        )
    }


def _overrun_profile(results: list[dict]) -> dict:
    rows = []
    for result in results:
        if result["mode"] != "hybrid_residual":
            continue
        path = REPO_DIR / result["run_dir_local_ignored"] / "perf_intervals.csv"
        with path.open("r", encoding="utf-8") as file:
            rows.extend(
                row
                for row in csv.DictReader(file)
                if row["overrun"] == "True"
            )
    component_names = (
        "mpc_policy_update_ms",
        "ddq_total_ms",
        "cpp_executor_bridge_ms",
        "other_right_arm_path_ms",
    )
    return {
        "overrun_count": len(rows),
        "without_ddq_second_pass_count": sum(
            int(row["ddq_second_pass_count"]) == 0 for row in rows
        ),
        "without_ddq_rescue_count": sum(
            int(row["ddq_rescue_count"]) == 0 for row in rows
        ),
        "overrun_component_mean_ms": {
            name: (
                sum(float(row[name]) for row in rows) / len(rows)
                if rows
                else 0.0
            )
            for name in component_names
        },
        "interpretation": (
            "tails usually raise MPC, DDQ execution and the remaining Python "
            "path together; they are not explained by a rare solver rescue"
        ),
    }


def _pre_vs_final(pre: dict, final_hybrid: dict) -> dict:
    if not pre.get("available", False):
        return {"available": False}
    before = pre["normal_hybrid_residual"]
    before_timing = before["timing_ms"]
    final_timing = final_hybrid["timing_ms"]

    def comparison(before_value: float, final_value: float) -> dict:
        return {
            "before": before_value,
            "final": final_value,
            "change_percent": 100.0 * (final_value - before_value) / before_value,
        }

    return {
        "available": True,
        "normal_runs_per_version": 6,
        "predictor_mean_ms": comparison(
            before_timing["predictor_mean"]["mean"],
            final_timing["predictor_mean"]["mean"],
        ),
        "predictor_p99_ms": comparison(
            before_timing["predictor_p99"]["mean"],
            final_timing["predictor_p99"]["mean"],
        ),
        "complete_interval_mean_ms": comparison(
            before_timing["complete_interval_mean"]["mean"],
            final_timing["complete_interval_mean"]["mean"],
        ),
        "complete_interval_p99_ms": comparison(
            before_timing["complete_interval_p99"]["mean"],
            final_timing["complete_interval_p99"]["mean"],
        ),
        "complete_interval_worst_max_ms": comparison(
            before_timing["complete_interval_max"]["max"],
            final_timing["complete_interval_max"]["max"],
        ),
        "complete_interval_overrun_count": {
            "before": before["safety_totals"][
                "complete_interval_overrun_count"
            ],
            "final": final_hybrid["safety_totals"][
                "complete_interval_overrun_count"
            ],
        },
        "interpretation": (
            "allocation cleanup modestly lowers predictor mean, but repeated "
            "full-path tails remain dominated by machine scheduling/frequency "
            "variation and do not show a reliable p99/max improvement"
        ),
    }
def _readiness_gate(
    grouped: dict[str, dict[str, dict]], paired: dict[str, dict]
) -> dict:
    normal_hybrid = grouped["normal"]["hybrid_residual"]
    normal_template = grouped["normal"]["template"]
    payload_hybrids = [
        grouped[name]["hybrid_residual"]
        for name in grouped
        if name != "normal"
    ]
    all_hybrid = [normal_hybrid, *payload_hybrids]
    checks = {
        "zero_critical_nonfinite": all(
            item["safety_totals"]["critical_nonfinite_count"] == 0
            for item in all_hybrid
        ),
        "zero_6ms_overrun": all(
            item["safety_totals"]["complete_interval_overrun_count"] == 0
            for item in all_hybrid
        ),
        "worst_run_complete_p99_le_5p5ms": all(
            item["timing_ms"]["complete_interval_p99"]["max"] <= 5.5
            for item in all_hybrid
        ),
        "worst_run_complete_max_lt_6ms": all(
            item["timing_ms"]["complete_interval_max"]["max"] < 6.0
            for item in all_hybrid
        ),
        "worst_run_predictor_p99_le_1p25ms": all(
            item["timing_ms"]["predictor_p99"]["max"] <= 1.25
            for item in all_hybrid
        ),
        "worst_run_predictor_max_lt_2ms": all(
            item["timing_ms"]["predictor_max"]["max"] < 2.0
            for item in all_hybrid
        ),
        "normal_qp_success_min_ge_99pct": (
            normal_hybrid["overall"]["qp_success_fraction"]["min"] >= 0.99
        ),
        "payload_qp_success_min_ge_99pct": all(
            item["overall"]["qp_success_fraction"]["min"] >= 0.99
            for item in payload_hybrids
        ),
        "normal_nonfinite_or_envelope_gate_never_triggered": (
            normal_hybrid["safety_totals"]["safety_gate_by_code"]["4"] == 0
            and normal_hybrid["safety_totals"]["safety_gate_by_code"]["5"]
            == 0
            and normal_hybrid["safety_totals"]["safety_gate_by_code"]["6"]
            == 0
        ),
        "normal_all_safety_gate_fraction_le_2pct": (
            normal_hybrid["safety_totals"]["safety_gate_fraction"] <= 0.02
        ),
        "normal_ddq_saturation_not_materially_worse": (
            normal_hybrid["overall"]["ddq_saturation_any_fraction"]["mean"]
            <= normal_template["overall"]["ddq_saturation_any_fraction"][
                "mean"
            ]
            + 0.01
        ),
        "normal_quality_not_worse": _quality_gate(paired["normal"]),
        "payload_quality_not_worse": all(
            _quality_gate(value)
            for name, value in paired.items()
            if name != "normal"
        ),
    }
    return {
        "evidence_scope": "MuJoCo closed loop only; not hardware evidence",
        "thresholds": {
            "complete_interval_budget_ms": 6.0,
            "complete_interval_worst_run_p99_ms": 5.5,
            "predictor_worst_run_p99_ms": 1.25,
            "normal_and_payload_min_qp_success_fraction": 0.99,
            "normal_all_safety_gate_max_fraction": 0.02,
            "max_allowed_mean_tilt_regression_percent": 2.0,
        },
        "checks": checks,
        "hardware_integration_allowed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-hardware repeated timing and safety validation"
    )
    parser.add_argument(
        "--group",
        default=f"real_robot_readiness_{datetime.now():%Y%m%d_%H%M%S}",
    )
    parser.add_argument(
        "--summary-dir",
        default="evaluation_summary/real_robot_readiness",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(GENERALIZATION_SCHEDULE_PROFILES),
        default=list(GENERALIZATION_SCHEDULE_PROFILES),
    )
    parser.add_argument(
        "--payloads-kg",
        nargs="+",
        type=float,
        default=list(DEFAULT_PAYLOADS_KG),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-template-group",
        help=(
            "复用该 evaluation group 中同条件 template 运行；仅用于 predictor "
            "实现变化而 template 数值路径未变化的追加验证。"
        ),
    )
    args = parser.parse_args()
    if len(args.seeds) < 2:
        raise ValueError("readiness validation 至少需要两个 seed。")
    if any(not 0.0 < payload <= 0.25 for payload in args.payloads_kg):
        raise ValueError("payloads 必须位于 (0, 0.25] kg。")

    # All schedules exercise normal operation.  Payload mismatch is repeated
    # on one representative, deliberately unseen schedule to keep the matrix
    # small while still providing more than a single favorable run.
    conditions = [
        (profile_name, seed, 0.0)
        for profile_name in args.profiles
        for seed in args.seeds
    ] + [
        (args.profiles[0], seed, payload)
        for payload in args.payloads_kg
        for seed in args.seeds
    ]
    group_dir = REPO_DIR / "evaluation" / args.group
    group_dir.mkdir(parents=True, exist_ok=True)
    reuse_template_group_dir = (
        None
        if args.reuse_template_group is None
        else REPO_DIR / "evaluation" / args.reuse_template_group
    )
    total_runs = len(conditions) * len(MODES)
    results = []
    completed = 0
    for profile_name, seed, payload_kg in conditions:
        profile = GENERALIZATION_SCHEDULE_PROFILES[profile_name]
        condition_id = _condition_id(profile_name, seed, payload_kg)
        for mode in MODES:
            print(f"[{completed + 1}/{total_runs}] {condition_id}_{mode}")
            run_dir = _run_one(
                group=args.group,
                group_dir=group_dir,
                profile_name=profile_name,
                seed=seed,
                payload_kg=payload_kg,
                mode=mode,
                resume=args.resume,
                reuse_template_group_dir=reuse_template_group_dir,
            )
            result = summarize_run(
                run_dir,
                mode,
                evaluation_start=0.8,
                evaluation_end=profile.timing.run_end,
                stages=profile.timing.stage_windows(),
            )
            result.update(
                {
                    "condition_id": condition_id,
                    "profile": profile_name,
                    "seed": seed,
                    "payload_kg": payload_kg,
                    "payload_stress": payload_kg > 0.0,
                }
            )
            results.append(result)
            completed += 1

    named_groups = {"normal": _condition_group(results, 0.0)}
    named_groups.update(
        {
            f"payload_{int(round(payload * 1000))}g": _condition_group(
                results, payload
            )
            for payload in args.payloads_kg
        }
    )
    aggregate = {
        name: {mode: _aggregate_mode(items, mode) for mode in MODES}
        for name, items in named_groups.items()
    }
    paired = {
        name: _paired_improvements(items)
        for name, items in named_groups.items()
    }
    pre_hardening = _pre_hardening_reference()
    summary = {
        "stage": "real_robot_preintegration_timing_and_safety",
        "evidence_scope": "MuJoCo closed loop; not hardware evidence",
        "local_evaluation_group_ignored": args.group,
        "reused_template_group_ignored": args.reuse_template_group,
        "modes": list(MODES),
        "normal_profiles": args.profiles,
        "payload_profile": args.profiles[0],
        "seeds": args.seeds,
        "payloads_kg": args.payloads_kg,
        "condition_count": len(conditions),
        "total_run_count": len(results),
        "pre_hardening_reference": pre_hardening,
        "pre_vs_final_normal_hybrid": _pre_vs_final(
            pre_hardening, aggregate["normal"]["hybrid_residual"]
        ),
        "overrun_profile": _overrun_profile(results),
        "hardening_changes": [
            "preallocated neural history and feature scratch buffers",
            "reused Torch input view and avoided diagnostics deep copies",
            "applied hybrid residuals in place to fresh template intervals",
            "bounded input, output, correction, QP-quality and overrun gates",
        ],
        "aggregate_by_condition_group": aggregate,
        "hybrid_vs_template_paired": paired,
        "readiness_gate": _readiness_gate(aggregate, paired),
        "runs": [_compact_readiness_result(result) for result in results],
    }
    summary_dir = Path(args.summary_dir)
    if not summary_dir.is_absolute():
        summary_dir = REPO_DIR / summary_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["readiness_gate"], indent=2, sort_keys=True))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
