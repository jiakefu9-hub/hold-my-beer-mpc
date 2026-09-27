#!/usr/bin/env python3
"""Recompute the offline replay reference audit and timing totals. No SDK/DDS.

Pass one or more benchmark output directories, each containing summary.json and
runN/raw.jsonl. Prints JSON; never overwrites the evidence. Reference maxima are
only for active MPC (3--18 s), not the separate arm-entry/weight-release ramps.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


NOMINAL_RIGHT = [math.radians(v) for v in (-4, 1, 0, -7.8, 0)]


def audit_commands(path):
    audit = dict(active_commands=0, max_full_constraint_violation=0.,
                 max_reference_offset_deg=0., max_reference_speed_rad_s=0.,
                 max_reference_acceleration_rad_s2=0.,
                 max_independent_acceleration_rad_s2=0.,
                 max_position_integration_residual_rad=0.,
                 max_logged_acceleration_residual_rad_s2=0.)
    previous_q, previous_dq = NOMINAL_RIGHT, [0.]*5
    previous_sequence = previous_time = None
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if not row.get("mpc_active"):
                continue
            qp = row["mpc"]
            if not qp["solved"] or qp["fallback_used"]:
                raise ValueError("replay contains an unsuccessful active MPC command")
            q, dq = row["q_command_rad"][5:10], row["dq_command_rad_s"][5:10]
            ddq = row["governed_ddq_reference_rad_s2"]
            violation = float(qp["max_constraint_violation"])
            if any(len(v) != 5 for v in (q, dq, ddq)) or not all(
                    math.isfinite(v) for v in (*q, *dq, *ddq, violation)):
                raise ValueError("invalid replay command numbers")
            dt = float(row["command_integration_dt_s"])
            elapsed = float(row["task_elapsed_s"])
            feedback_dt = float(row["feedback_dt_s"])
            if (not all(map(math.isfinite, (dt, elapsed, feedback_dt)))
                    or not 0 < dt <= .006+1e-12
                    or abs(dt-min(feedback_dt,.006)) > 1e-12):
                raise ValueError("invalid bounded command integration interval")
            if previous_sequence is not None and (
                row["sequence"] != previous_sequence+1 or elapsed <= previous_time
                or abs(elapsed-previous_time-feedback_dt) > 1e-9
            ):
                raise ValueError("active command sequence/time gap")
            independent_ddq = [(a-b)/dt for a,b in zip(dq,previous_dq)]
            position_error = max(abs(a-b-v*dt) for a,b,v in zip(q,previous_q,dq))
            acceleration_error = max(abs(a-b) for a,b in zip(ddq,independent_ddq))
            audit["max_independent_acceleration_rad_s2"] = max(
                audit["max_independent_acceleration_rad_s2"], *(abs(v) for v in independent_ddq))
            audit["max_position_integration_residual_rad"] = max(
                audit["max_position_integration_residual_rad"],position_error)
            audit["max_logged_acceleration_residual_rad_s2"] = max(
                audit["max_logged_acceleration_residual_rad_s2"],acceleration_error)
            previous_q, previous_dq = q,dq
            previous_sequence, previous_time = row["sequence"],elapsed
            audit["active_commands"] += 1
            audit["max_full_constraint_violation"] = max(audit["max_full_constraint_violation"], violation)
            audit["max_reference_offset_deg"] = max(audit["max_reference_offset_deg"],
                *(abs(math.degrees(a-b)) for a, b in zip(q, NOMINAL_RIGHT)))
            audit["max_reference_speed_rad_s"] = max(audit["max_reference_speed_rad_s"], *(abs(v) for v in dq))
            audit["max_reference_acceleration_rad_s2"] = max(
                audit["max_reference_acceleration_rad_s2"], *(abs(v) for v in ddq))
    if audit["active_commands"] < 2000:
        raise ValueError("not a complete MPC replay")
    for key, limit in (("max_full_constraint_violation", 1e-6),
                       ("max_reference_offset_deg", 5. + 1e-8),
                       ("max_reference_speed_rad_s", .07 + 1e-8),
                       ("max_reference_acceleration_rad_s2", .20 + 1e-8),
                       ("max_independent_acceleration_rad_s2", .20 + 1e-8),
                       ("max_position_integration_residual_rad", 1e-10),
                       ("max_logged_acceleration_residual_rad_s2", 1e-8)):
        if audit[key] > limit:
            raise ValueError(f"{key} exceeds the reference envelope")
    return audit


def summarize(directories):
    runs = []
    for directory in map(Path, directories):
        saved = json.loads((directory / "summary.json").read_text())
        for index, run in enumerate(saved["runs"], 1):
            if (run["status"] != "complete" or run["final_weight"] != 0
                    or run["journal_dropped"] or run["journal_failed"]
                    or run["hardware_output"] or run["dds_initialized"]):
                raise ValueError(f"{directory}/run{index} is not a complete output-absent replay")
            raw = directory / f"run{index}" / "raw.jsonl"
            runs.append(dict(local_directory=str(raw.parent),
                raw_jsonl_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                reference_audit=audit_commands(raw), primary_5_18=run["primary_5_18"]))
    samples = sum(r["primary_5_18"]["samples"] for r in runs)
    misses = sum(r["primary_5_18"]["deadline_misses"] for r in runs)
    return dict(runs=runs, primary_samples=samples, primary_deadline_misses=misses,
                primary_deadline_miss_fraction=misses/samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    print(json.dumps(summarize(parser.parse_args().directories), indent=2))
