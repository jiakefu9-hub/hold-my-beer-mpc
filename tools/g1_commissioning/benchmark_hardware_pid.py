#!/usr/bin/env python3
"""Causal, wall-clock PID replay. IDL/CRC only: no DDS participant or publisher.

Measures host computation/packet serialization/logging; recorded feedback is
replayed, not a simulated response to the new commands. No control-performance
claim or real-time DDS/firmware guarantee follows from this benchmark.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from endpoint_pose import EndpointModel
from g1_walk_pid import Journal, EXPECTED_TARGET_Q, make_arm_message
from hardware_pid_control import (
    ARM_MOTOR_INDICES, CONTROL_PERIOD_S, HardwarePidPlan, PidParameters, RightArmHardwarePid,
)
from pid_timing import PeriodicClock, pin_control_thread, timing_summary


def load_source(path):
    lows, imus, old_compute = [], [], []
    session = epoch = yaw0 = None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            r = json.loads(line)
            if r.get("schema") == "g1_pid_session_v1":
                session = r
            elif r.get("event") == "task_epoch":
                epoch = r["task_epoch_monotonic_ns"]
            elif r.get("event") == "heading_reference_frozen":
                yaw0 = r["yaw0_rad"]
            elif r.get("schema") == "g1_lowstate_raw_v1" and r.get("crc_valid"):
                motors = {m["index"]: m for m in r["motors"]}
                lows.append((r["received_monotonic_ns"],
                             np.array([motors[i]["q_rad"] for i in ARM_MOTOR_INDICES]),
                             np.array([motors[i]["dq_rad_s"] for i in ARM_MOTOR_INDICES]),
                             r["mode_pr"], r["mode_machine"]))
            elif r.get("schema") == "g1_torso_imu_raw_v1":
                imus.append((r["received_monotonic_ns"], r["quaternion_wxyz"]))
            elif r.get("pid_active") and "controller_compute_us" in r:
                old_compute.append(r["controller_compute_us"] / 1000)
    if session is None or epoch is None or yaw0 is None or not lows or not imus:
        raise ValueError("requires complete PID capture and frozen H0")
    lows.sort(key=lambda row: row[0])
    imus.sort(key=lambda row: row[0])
    if lows[0][0] > epoch or imus[0][0] > epoch or min(lows[-1][0], imus[-1][0]) < epoch + 21e9:
        raise ValueError("source must cover [0,21] seconds; no future-sample interpolation")
    return session, epoch, yaw0, lows, imus, digest.hexdigest(), old_compute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=None)
    args = parser.parse_args()
    session, epoch, yaw0, lows, imus, digest, old = load_source(args.raw_jsonl)
    # Import only packet classes and checksum; never initialize ChannelFactory.
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    parameters = PidParameters.from_mapping(session["pid_parameters"])
    controller = RightArmHardwarePid(EXPECTED_TARGET_Q[5:10], parameters, EndpointModel())
    for _ in range(100):
        controller.step(lows[0][1], imus[0][1], yaw0, CONTROL_PERIOD_S)
    controller.reset()
    plan = HardwarePidPlan(lows[0][1], EXPECTED_TARGET_Q,
                          np.r_[np.full(11, 20.0), 0., 0.], np.r_[np.ones(11), 0., 0.], controller)
    journal = Journal(args.output_dir)
    crc = CRC()
    original_affinity = set(os.sched_getaffinity(0))
    timings, computes = [], []
    try:
        runtime = pin_control_thread(args.cpu)
        start = time.monotonic_ns()
        clock = PeriodicClock(start, CONTROL_PERIOD_S)
        last = previous_low = previous_imu = None
        li = ii = 0
        for sequence in range(10000):
            clock.wait()
            begin = time.monotonic_ns()
            task_s = (begin - start) * 1e-9
            source_ns = epoch + begin - start
            while li + 1 < len(lows) and lows[li + 1][0] <= source_ns:
                li += 1
            while ii + 1 < len(imus) and imus[ii + 1][0] <= source_ns:
                ii += 1
            low, imu = lows[li], imus[ii]
            dt = CONTROL_PERIOD_S if last is None else (begin - last) * 1e-9
            cb = time.monotonic_ns()
            frame = plan.sample(task_s, low[1], low[2], imu[1], yaw0 if task_s >= 5 else 0., dt)
            ce = time.monotonic_ns()
            packet = make_arm_message(frame, SimpleNamespace(mode_pr=low[3], mode_machine=low[4]),
                                      unitree_hg_msg_dds__LowCmd_, crc)
            packet.serialize()  # local IDL serialization only, no write/transport
            journal.record({
                "schema": "g1_pid_offline_command_v1", "sequence": sequence,
                "task_elapsed_s": task_s, "stage": frame["stage"], "weight": frame["weight"],
                "q_command_rad": frame["q_rad"].tolist(), "dq_command_rad_s": frame["dq_rad_s"].tolist(),
                "q_measured_rad": low[1].tolist(), "imu_quaternion": imu[1],
                "packet_crc": packet.crc, **frame["diagnostics"],
            })
            end = time.monotonic_ns()
            scheduled = clock.scheduled_ns
            skipped = clock.advance(end)
            row = {
                "schema": "g1_pid_timing_v1", "sequence": sequence, "task_elapsed_s": task_s,
                "actual_period_ms": None if last is None else dt * 1000,
                "wake_lateness_ms": (begin - scheduled) * 1e-6,
                "work_ms": (end - begin) * 1e-6,
                "deadline_missed": end > scheduled + clock.period_ns, "skipped_slots": skipped,
                "reused_lowstate": low[0] == previous_low, "reused_imu": imu[0] == previous_imu,
            }
            journal.record(row)
            timings.append(row)
            if frame["diagnostics"].get("pid_active"):
                computes.append((ce - cb) * 1e-6)
            last, previous_low, previous_imu = begin, low[0], imu[0]
            if frame["terminal"]:
                break
        else:
            raise RuntimeError("replay failed to finish release")
    finally:
        os.sched_setaffinity(0, original_affinity)
        journal.close()
    result = {
        "offline_only": True, "dds_initialized": False, "hardware_output": False,
        "source": str(args.raw_jsonl), "source_sha256": digest,
        "nominal_period_ms": CONTROL_PERIOD_S * 1000, "runtime": runtime,
        "parameters": session["pid_parameters"], "xml_sha256": controller.model.xml_hashes(),
        "whole_session": timing_summary(timings),
        "primary_5_18": timing_summary([r for r in timings if 5 <= r["task_elapsed_s"] < 18]),
        "compute_ms_p50_p95_p99_max": np.percentile(computes, [50, 95, 99, 100]).tolist(),
        "recorded_20ms_compute_ms_p50_p95_p99_max": np.percentile(old, [50, 95, 99, 100]).tolist(),
        "journal_dropped": journal.dropped, "journal_failed": journal.failed.is_set(),
        "final_weight": frame["weight"],
        "limitations": [
            "causal replay of old measured inputs; no prediction of robot response to new commands",
            "includes FK/PID, packet/CRC/IDL serialization and async file logging",
            "excludes live DDS deserialization/callback contention, RPC workers and actual command transport",
            "20ms LowState logging in source limits replay input resolution; reused samples are counted",
            "ordinary Linux; measured timing is not a hard real-time guarantee",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("primary_5_18", "compute_ms_p50_p95_p99_max", "journal_dropped", "final_weight")}, indent=2))
    return int(result["journal_failed"] or result["journal_dropped"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
