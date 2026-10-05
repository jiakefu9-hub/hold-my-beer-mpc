#!/usr/bin/env python3
"""Wall-clock 6 ms MPC replay, without DDS initialization or robot output.

--source-npz accepts the raw extracted trialXX.npz from walk_h0_study/prepare.py
(not its filtered *_prepared.npz). Without it, use synthetic stationary inputs.
Recorded feedback does not respond to these commands: this is timing/contract
validation, never a claim that the physical controller improved bottle motion.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from g1_walk_mpc import MpcRuntime, MpcJournal
from g1_walk_pid import EXPECTED_TARGET_Q
from hardware_pid_control import ARM_MOTOR_INDICES, FixedH0Heading, vertical_angular_rate, yaw_from_quaternion
from pid_timing import PeriodicClock, pin_control_thread
from mpc_host import host_evidence, summarize_timing
from hardware_mpc_control import json_values
from arm_execution_record import command_evidence


def source_data(path):
    if path is not None:
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        if "world_acc" not in d or d["q"].shape[1] != 35:
            raise ValueError("use extracted raw trialXX.npz, not prepared/filtered data")
        if max(d["imu_t"][0], d["low_t"][0]) > -.5 or min(d["imu_t"][-1], d["low_t"][-1]) < 21:
            raise ValueError("replay must cover [-.5,21] without future interpolation")
        return d, hashlib.sha256(path.read_bytes()).hexdigest()
    t = np.arange(-.5, 23.002, .002)
    q = np.zeros((len(t), 35))
    q[:, list(ARM_MOTOR_INDICES)] = EXPECTED_TARGET_Q
    imu = np.zeros((len(t), 15))
    imu[:, 2] = 1
    imu[:, 14] = 9.81
    return dict(imu_t=t, low_t=t, q=q, dq=np.zeros_like(q), imu=imu,
                low=np.zeros((len(t), 5))), None


def run_once(source, output, cpu, predictor, actuation="measured_torque_preview", torque_config=None,
             assumed_command_delay_s=None, observation_delay_s=0.):
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    if not np.isfinite(observation_delay_s) or not 0 <= observation_delay_s <= .020:
        raise ValueError('offline observation delay must be 0..20ms')
    runtime = MpcRuntime(predictor_mode=predictor, actuation=actuation, torque_config=torque_config,
                         assumed_command_delay_s=assumed_command_delay_s)
    journal = MpcJournal(output)
    runtime.journal = journal
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("benchmark_hardware_mpc.py", "g1_walk_mpc.py", "g1_walk_pid.py",
                                  "hardware_mpc_control.py", "hardware_mpc_solver.py",
                                  "hardware_mpc_predictor.py", "hardware_pid_control.py",
                                  "hardware_mpc_inverse_preview.py", "hardware_arm_inverse_dynamics.py",
                                  "hardware_mpc_torque_control.py", "hardware_torque_mapper.py",
                                  "hardware_mpc_recovery.py","hardware_mpc_delay_preview.py",
                                  "hardware_mpc_delay_plan.py", "arm_execution_record.py")}
    model_hash = None if runtime.predictor.bank is None else runtime.predictor.bank.manifest["bank_sha256"]
    affinity = set(os.sched_getaffinity(0))
    heading, yaw0 = FixedH0Heading(), 0.
    rows, li, ii, previous, last_frame = [], 0, 0, None, None
    previous_low_ns = previous_imu_ns = None
    last_low_ns = last_imu_ns = -10**18
    selected_low = selected_imu = 0
    source_ns = 10_000_000_000
    runtime.predictor.set_grid_origin(source_ns)
    profile = dict(target_q_array=EXPECTED_TARGET_Q,
        kp_array=np.r_[np.full(11, 20.), 0, 0], kd_array=np.r_[np.ones(11), 0, 0],
        q_offset_limit_deg_array=np.full(5, 5.))
    plan = runtime.create_plan(source["q"][0, list(ARM_MOTOR_INDICES)], profile)
    core_metadata = json_values(runtime.controller.metadata)
    crc = CRC()
    status, reason = "complete", None
    gc_was_enabled = gc.isenabled()
    try:
        host = host_evidence(cpu)
        pin_control_thread(cpu)
        # Feed initial history outside the measured loop (like live startup).
        def feed(task_s):
            nonlocal li, ii, last_low_ns, last_imu_ns, selected_low, selected_imu
            while li < len(source["low_t"]) and source["low_t"][li] <= task_s:
                stamp = source_ns + round(float(source["low_t"][li]) * 1e9)
                if stamp-last_low_ns >= 2_000_000:
                    runtime.observe_low(stamp, source["q"][li], source["dq"][li])
                    last_low_ns, selected_low = stamp, li
                li += 1
            while ii < len(source["imu_t"]) and source["imu_t"][ii] <= task_s:
                t, m = float(source["imu_t"][ii]), source["imu"][ii]
                stamp = source_ns + round(t * 1e9)
                if stamp-last_imu_ns >= 2_000_000:
                    runtime.observe_imu(stamp, m[2:6], m[9:12], m[12:15])
                    heading.observe(stamp, t, yaw_from_quaternion(m[2:6]), vertical_angular_rate(m[6:9], m[9:12]))
                    last_imu_ns, selected_imu = stamp, ii
                ii += 1
        # Limit startup to the same bounded history as live ingress.
        li = max(0, int(np.searchsorted(source["low_t"], -.5, side="right")) - 1)
        ii = max(0, int(np.searchsorted(source["imu_t"], -.5, side="right")) - 1)
        feed(-observation_delay_s)
        runtime.prepare(source_ns, None, None, 0., 0.)
        gc.collect()
        gc.disable()  # bounded one-shot loop; restore after recorder drainage
        begin_epoch = time.monotonic_ns()
        clock = PeriodicClock(begin_epoch, .006)
        for sequence in range(5000):
            clock.wait()
            begin = time.monotonic_ns()
            task_s = (begin - begin_epoch) * 1e-9
            feed(task_s-observation_delay_s)
            after_feed = time.monotonic_ns()
            if task_s >= 5 and not heading.current()["reference_frozen"]:
                yaw0 = heading.freeze()
            dt = .006 if previous is None else (begin - previous) * 1e-9
            if task_s < 18 or actuation == "measured_torque_preview":
                runtime.prepare(source_ns + begin - begin_epoch, None, None, yaw0, task_s)
            after_predict = time.monotonic_ns()
            low, imu = source["q"][selected_low], source["imu"][selected_imu]
            frame = plan.sample(task_s, low[list(ARM_MOTOR_INDICES)],
                source["dq"][selected_low, list(ARM_MOTOR_INDICES)], imu[2:6], yaw0, dt)
            after_control = time.monotonic_ns()
            packet = runtime.make_message(frame, SimpleNamespace(mode_pr=0, mode_machine=4),
                                          unitree_hg_msg_dds__LowCmd_, crc)
            serialized=packet.serialize()
            if assumed_command_delay_s is not None:
                # Record the float32 values carried by the actual local CDR,
                # not the higher precision pre-serialization candidate.
                packet=type(packet).deserialize(serialized)
                plan.commit_packet(frame,packet)
            after_packet = time.monotonic_ns()
            journal.record({"schema": "g1_mpc_offline_command_v1", "sequence": sequence,
                "task_elapsed_s": task_s, "stage": frame["stage"], "weight": frame["weight"],
                "q_command_rad": frame["q_rad"], "dq_command_rad_s": frame["dq_rad_s"],
                "q_measured_rad": low[list(ARM_MOTOR_INDICES)],
                "dq_measured_rad_s": source["dq"][selected_low, list(ARM_MOTOR_INDICES)],
                "offline_packet_right_tau_nm": [packet.motor_cmd[i].tau for i in range(22,27)],
                "offline_packet_right_q_rad": [packet.motor_cmd[i].q for i in range(22,27)],
                "offline_packet_right_dq_rad_s": [packet.motor_cmd[i].dq for i in range(22,27)],
                "offline_packet_weight": float(packet.motor_cmd[29].q),
                "packet_crc": int(packet.crc), **frame["diagnostics"],
                **command_evidence(packet, SimpleNamespace(crc_valid=True)),
                "physical_feedback_available": False})
            row = dict(sequence=sequence, task_elapsed_s=task_s, stage=frame["stage"],
                actual_period_ms=None if previous is None else dt * 1000,
                wake_lateness_ms=(begin-clock.scheduled_ns)*1e-6,
                ingress_ms=(after_feed-begin)*1e-6,
                predictor_ms=(after_predict-after_feed)*1e-6,
                controller_ms=(after_control-after_predict)*1e-6,
                packet_crc_serialize_ms=(after_packet-after_control)*1e-6,
                reused_lowstate=last_low_ns == previous_low_ns,
                reused_imu=last_imu_ns == previous_imu_ns)
            journal.record({"schema": "g1_mpc_offline_timing_v1", **row})
            rows.append(row)
            finished = time.monotonic_ns()
            row.update(full_work_ms=(finished-begin)*1e-6,
                deadline_missed=finished>clock.scheduled_ns+clock.period_ns,
                skipped_slots=clock.advance(finished))
            previous, last_frame = begin, frame
            previous_low_ns, previous_imu_ns = last_low_ns, last_imu_ns
            if frame["terminal"]:
                break
        else:
            raise RuntimeError("replay did not complete release")
    except Exception as exc:
        status, reason = "failed", str(exc)
        journal.record({"event": "offline_fault", "reason": reason,
                        "diagnostics": runtime.controller.last_diagnostics})
    finally:
        os.sched_setaffinity(0, affinity)
        journal.close()
        runtime.close()
        if gc_was_enabled:
            gc.enable()
    result = dict(schema="g1_hardware_mpc_benchmark_v1", status=status, failure=reason,
        dds_initialized=False, publisher_created=False, hardware_output=False,
        host=host, warmup=runtime.warmup, predictor=predictor, actuation=actuation,
        assumed_command_delay_s=assumed_command_delay_s, observation_delay_s=observation_delay_s,
        delay_history_committed_packets=getattr(plan,'committed_packets',None),
        source_sha256=source_hashes, core=core_metadata, predictor_bank_sha256=model_hash,
        source_unchanged_during_run=all(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest
                                        for name,digest in source_hashes.items()),
        torque_config_sha256=None if torque_config is None else hashlib.sha256(Path(torque_config).read_bytes()).hexdigest(),
        primary_5_18=summarize_timing([r for r in rows if 5 <= r["task_elapsed_s"] < 18]),
        all_stages=summarize_timing(rows), journal_dropped=journal.dropped,
        journal_failed=journal.failed.is_set(),
        final_weight=None if last_frame is None else float(last_frame["weight"]),
        limitations=["replayed feedback is not a response to MPC; no physical effect claim",
                     "includes causal ingress, model, actual QP, packet CRC+serialization, async logging",
                     "excludes real DDS receive/deserialization, network Write and RPC worker contention",
                     "accepted replay ingress uses the same >=2ms sampling threshold as live Streams",
                     "full_work includes timing enqueue; final timestamp and clock bookkeeping excluded",
                     "inverse_dynamics_preview, if selected, has no commissioned field torque transition/envelope",
                     "measured_torque_preview uses a conditional arm model; weight blending and torque limits are uncommissioned",
                     "delay lifecycle, if enabled, assumes explicit observation/command delays; does not identify real DDS or firmware latency",
                     "measured host timing is not a hard real-time certificate"])
    (output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    (output / "timing.json").write_text(json.dumps(rows)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path)
    parser.add_argument("--torque-config", type=Path)
    parser.add_argument("--assumed-command-delay-ms",type=float,
                        help="offline full-lifecycle state prediction; NOT an identified device delay")
    parser.add_argument("--observation-delay-ms",type=float,default=0.,
                        help="delay replay ingress only; timestamps remain original")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--predictor", choices=("learned_filtered", "hold_current"), default="learned_filtered")
    parser.add_argument("--actuation", choices=("reference_servo", "inverse_dynamics_preview", "measured_torque_preview"),
                        default="measured_torque_preview")
    args = parser.parse_args()
    if not 1 <= args.runs <= 10:
        parser.error("runs must be 1..10")
    data, digest = source_data(args.source_npz)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for index in range(args.runs):
        result = run_once(data, args.output_dir / f"run{index+1}", args.cpu, args.predictor, args.actuation,
                          args.torque_config,
                          None if args.assumed_command_delay_ms is None else args.assumed_command_delay_ms*.001,
                          args.observation_delay_ms*.001)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("status", "failure", "primary_5_18", "final_weight")}), flush=True)
        if result["status"] != "complete":
            break
    summary = dict(source=str(args.source_npz) if args.source_npz else "synthetic_stationary",
                   source_sha256=digest, runs=results)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return int(any(r["status"] != "complete" or r["journal_dropped"] or r["journal_failed"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
