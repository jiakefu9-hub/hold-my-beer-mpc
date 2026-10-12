#!/usr/bin/env python3
"""Offline preflight and recorded-input replay of the MPC/C field comparisons.

No DDS initialization, network, fitting, model propagation or physical output.
Recorded states do not respond to the candidate commands. The repeats marked T04/T05
are field repeats; replaying the same input again is not independent evidence.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import yaml
import os
from pathlib import Path
import socket
from unittest.mock import patch

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"


from g1_walk_mpc import preflight
from hardware_mpc_control import json_values
from hardware_control_comparison import (
    BASELINE_CONFIG, MODEL_CONFIG, ROOT, TORQUE_CONFIG, TRIALS, validate_series,
)
from mpc_host import host_evidence, select_cpu
from validate_mpc_learned import capture, replay


from control_comparison_evidence import (
    DEFAULT_SOURCES, LIMITATIONS, sha256_file, hash_paths,
    source_session, validate_source_session,
)


def write_summary(path, report):
    path.write_text(json.dumps(json_values(report), indent=2, allow_nan=False) + "\n")


def exception_record(exc):
    return dict(status="failed", error_type=type(exc).__name__, error=str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", choices=tuple(TRIALS), action="append",
                        help="repeat to select groups; default T01/T02/T03/T06; T04/T05 duplicate existing inputs")
    parser.add_argument("--source", type=Path, action="append",
                        help="repeat to select complete zero-neutral, normal-left-PD captures")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=7)
    parser.add_argument("--math-only", action="store_true",
                        help="explicit replay-only 100 ms QP allowance; NEVER a timing pass")
    args = parser.parse_args(argv)
    selected = args.trial or ['T01', 'T02', 'T03', 'T06']
    if len(set(selected)) != len(selected):
        parser.error("each trial may be selected only once")
    sources = [p.resolve() for p in (args.source or DEFAULT_SOURCES)]
    if len(set(sources)) != len(sources):
        parser.error("duplicate source paths are not independent evidence")
    validate_series()
    select_cpu(args.cpu)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "summary.json"
    baseline_paths = [BASELINE_CONFIG, TORQUE_CONFIG, MODEL_CONFIG, ROOT / "configs/g1.yaml",
                      ROOT / "configs/hardware_mpc.yaml"]
    # Include the field numerical stack and lifecycle, even though no field
    # transport is initialized. This also captures the trial registry itself.
    source_paths = list(Path(__file__).parent.glob("*.py")) + [
        ROOT / "arm_mpc.py", ROOT / "kinematics_helper.py",
    ]
    report = dict(
        schema="g1_control_comparison_replay_v1", status="running",
        hardware_output=False, dds_initialized=False, network_forbidden=True,
        model_config=str(MODEL_CONFIG), torque_config=str(TORQUE_CONFIG),
        zero_arm_neutral=True, left_pd_gain_scale=1.0,
        source_query_timestamps_used=True, live_timing_validated=False,
        math_only=args.math_only, selected_trials=selected,
        host=host_evidence(args.cpu), code_sha256=hash_paths(source_paths),
        baseline_files_sha256_before=hash_paths(baseline_paths),
        trials={}, sources=[], limitations=LIMITATIONS,
    )
    write_summary(summary_path, report)
    with patch.object(socket, "socket", side_effect=RuntimeError("offline trial validation forbids network")):
        for trial_id in selected:
            spec = TRIALS[trial_id]
            trial = dict(name=spec["name"], config=str(spec["config"]),
                         aligned=spec["aligned"], compare_to={'T02':'T01','T03':'T02','T04':'T02','T05':'T03','T06':'T02'}.get(trial_id),
                         replay_repeat_of={'T04':'T02','T05':'T03'}.get(trial_id),
                         independent_repeat_evidence=False)
            report["trials"][trial_id] = trial
            try:
                trial["config_sha256"] = sha256_file(spec["config"])
                trial["preflight"] = preflight(
                    config=spec["config"], model_config=MODEL_CONFIG,
                    mode="learned_filtered", cpu=args.cpu,
                    actuation="measured_torque_preview", torque_config=TORQUE_CONFIG,
                    zero_arm_neutral=True, left_pd_gain_scale=1.0,
                    align_accel_forecast=spec["aligned"],
                    acceleration_pid_config=spec["pid_config"],
                )
            except Exception as exc:
                trial["preflight"] = exception_record(exc)
            print(f"trial {trial_id}: preflight "
                  f"{'passed' if trial['preflight'].get('passed') else 'failed'}", flush=True)
            write_summary(summary_path, report)

        for source_index, source in enumerate(sources, 1):
            source_result = dict(source=str(source), status="running", variants={})
            report["sources"].append(source_result)
            try:
                source_result["source_sha256"] = sha256_file(source)
                source_result["source_validation"] = validate_source_session(source_session(source))
                data = capture(source)
                commands = data[0]
                active = [c for c in commands if 5.0 <= c["task_elapsed_s"] < 18.0]
                if not active or any("query_monotonic_ns" not in c.get("predictor", {})
                                     for c in active):
                    raise ValueError("source lacks logged predictor query timestamps in the control window")
                source_result["source_validation"]["complete_recorded_release"] = True
                source_result["source_validation"]["logged_control_queries"] = len(active)
            except Exception as exc:
                source_result.update(exception_record(exc))
                print(f"source {source.name} rejected: {exc}", flush=True)
                write_summary(summary_path, report)
                continue
            # Historical arms followed the old bounds. Do not clip their feedback
            # or treat a rejection of out-of-domain input as a closed-loop failure.
            bounds = yaml.safe_load(TORQUE_CONFIG.read_text())
            q = np.rad2deg([c['q_measured_rad'][5:10] for c in active])
            outside = (q < np.array(bounds['q_min_deg'])) | (q > np.array(bounds['q_max_deg']))
            source_result['input_bounds'] = dict(
                q_min_deg=q.min(axis=0).tolist(), q_max_deg=q.max(axis=0).tolist(),
                outside_outer_bound_frames=int(np.any(outside, axis=1).sum()))
            if outside.any():
                source_result.update(status='not_applicable',
                    reason='historical measured q exceeds the selected common outer bounds',
                    complete_candidate_replay=False)
                print(f'{source.parent.name}: replay not applicable (historical state outside new bounds)', flush=True)
                write_summary(summary_path, report)
                continue
            directory = args.output_dir / f"source_{source_index:02d}_{source.parent.name}"
            directory.mkdir()
            for trial_id in selected:
                spec = TRIALS[trial_id]
                if not report["trials"][trial_id]["preflight"].get("passed"):
                    source_result["variants"][trial_id] = dict(
                        status="skipped", reason="preflight failed", complete_release=False)
                else:
                    try:
                        result, _ = replay(
                            data, directory, f"trial_{trial_id}", args.cpu,
                            math_only=args.math_only, config=spec["config"],
                            predictor_mode="learned_filtered", zero_arm_neutral=True,
                            align_accel_forecast=spec["aligned"],
                            acceleration_pid_config=spec["pid_config"],
                            use_logged_query_times=True, model_config=MODEL_CONFIG, torque_config=TORQUE_CONFIG,
                        )
                        result["complete_release"] = result.get("status") == "passed" and result.get("final_weight") == 0.0
                        if not result["complete_release"]:
                            result["status"] = "failed"
                        result["live_timing_validated"] = False
                        result["replay_repeat_of"] = {'T04':'T02','T05':'T03'}.get(trial_id)
                        source_result["variants"][trial_id] = result
                    except Exception as exc:
                        source_result["variants"][trial_id] = dict(
                            exception_record(exc), complete_release=False)
                result = source_result["variants"][trial_id]
                print(f"{source.parent.name} trial {trial_id}: {result['status']} "
                      f"final_weight={result.get('final_weight')} "
                      f"over_6ms={result.get('over_6ms_count')}", flush=True)
                write_summary(summary_path, report)
            source_result["status"] = (
                "passed" if all(v["status"] == "passed" for v in source_result["variants"].values()) else "failed"
            )
            write_summary(summary_path, report)

    report["baseline_files_sha256_after"] = hash_paths(baseline_paths)
    report["baseline_files_unchanged"] = (
        report["baseline_files_sha256_before"] == report["baseline_files_sha256_after"]
    )
    report["complete_replay_validated"] = all(s["status"] == "passed" for s in report["sources"])
    passed = (report["baseline_files_unchanged"]
              and all(t["preflight"].get("passed") for t in report["trials"].values())
              and all(s["status"] in ("passed", "not_applicable") for s in report["sources"]))
    report["status"] = (("passed" if report["complete_replay_validated"] else "preflight_passed_replay_not_applicable")
                        if passed else "failed")
    write_summary(summary_path, report)
    print(f"Saved {summary_path}: {report['status']} (offline only)", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
