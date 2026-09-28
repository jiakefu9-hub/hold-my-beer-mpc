#!/usr/bin/env python3
"""Offline measured-state torque MPC; explicit opt-in legacy reference servo.

See docs/g1_field_validation/HARDWARE_MPC.md. The default command performs a
local preflight only. Neither mode changes nor rt/lowcmd are implemented.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from g1_walk_pid import (ROOT, EXPECTED_TARGET_Q, Journal, load_profile,
                         run_device, make_arm_message)
from hardware_pid_control import HardwarePidPlan, CONTROL_PERIOD_S
from hardware_mpc_control import RightArmHardwareMpc, DEFAULT_CONFIG, json_values
from hardware_mpc_predictor import HardwareMpcPredictor, DEFAULT_BANK
from mpc_host import host_evidence, select_cpu

PERMIT = "MPC_WALK_H0_CAPTURE"


class MpcJournal(Journal):
    """Same transport audit as PID, with explicit MPC record identities."""
    def record(self, row):
        row = dict(row)
        schema = row.get("schema", "")
        if schema.startswith("g1_pid_") or schema.startswith("g1_hardware_pid_"):
            row["schema"] = schema.replace("pid", "mpc")
        if "pid_active" in row:
            row["mpc_active"] = row.pop("pid_active")
        super().record(row)


class MpcRuntime:
    def __init__(self, config=DEFAULT_CONFIG, model_config=ROOT / "configs/g1.yaml",
                 predictor_mode="learned_filtered", bank_path=DEFAULT_BANK,
                 stationary=False, journal=None, actuation="reference_servo"):
        from endpoint_pose import EndpointModel
        if actuation not in {"reference_servo", "inverse_dynamics_preview", "measured_torque_preview"}:
            raise ValueError("unsupported MPC actuation")
        self.actuation = actuation
        self.stationary, self.journal = bool(stationary), journal
        self.predictor = HardwareMpcPredictor(predictor_mode, bank_path)
        controller_type = RightArmHardwareMpc
        if actuation == "inverse_dynamics_preview":
            from hardware_mpc_inverse_preview import RightArmInversePreviewMpc
            controller_type = RightArmInversePreviewMpc
        elif actuation == "measured_torque_preview":
            from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc
            controller_type = RightArmMeasuredTorqueMpc
        self.controller = controller_type(EXPECTED_TARGET_Q[5:10], config,
                                          model=EndpointModel(model_config))
        try:
            self.warmup = self.controller.warmup(EXPECTED_TARGET_Q, [1, 0, 0, 0])
        except Exception:
            self.controller.close()
            raise
        self.epoch_ns = None
        self._gc_was_enabled = None

    def close(self):
        self.controller.close()
        if self._gc_was_enabled:
            gc.enable()

    def observe_low(self, stamp, q, dq):
        accepted = self.predictor.observe_low(stamp, q, dq)
        if accepted and self.journal is not None:
            self.journal.record({"schema": "g1_mpc_predictor_low_v1",
                "received_monotonic_ns": stamp, "q_rad": q, "dq_rad_s": dq})

    def observe_imu(self, stamp, quat, gyro, accel):
        accepted = self.predictor.observe_imu(stamp, quat, gyro, accel)
        if accepted and self.journal is not None:
            self.journal.record({"schema": "g1_mpc_predictor_imu_v1",
                "received_monotonic_ns": stamp, "quaternion_wxyz": quat,
                "gyroscope_rad_s": gyro, "accelerometer_raw_m_s2": accel})

    def set_epoch(self, epoch_ns):
        self.epoch_ns = int(epoch_ns)
        self.predictor.set_grid_origin(epoch_ns)
        self.predictor.query(epoch_ns, 0., use_learned=False)
        self._gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()

    def create_plan(self, initial, profile):
        limits = np.asarray(self.controller.config["reference_offset_limit_deg"])
        if self.actuation != "measured_torque_preview" and np.any(limits > profile["q_offset_limit_deg_array"]):
            raise ValueError("MPC config exceeds reviewed profile reference bounds")
        self.controller.reset()
        plan_type = HardwarePidPlan
        if self.actuation == "measured_torque_preview":
            from hardware_mpc_torque_control import HardwareTorquePreviewPlan
            plan_type = HardwareTorquePreviewPlan
        return plan_type(initial, profile["target_q_array"],
                               profile["kp_array"], profile["kd_array"], self.controller)

    def prepare(self, now_ns, low, imu, yaw0, task_s, heading_frozen=True):
        # Advance causal filters from the first cycle, including the arm ramp.
        result = self.predictor.query(now_ns, yaw0,
                                      use_learned=not self.stationary and task_s >= 5. and heading_frozen)
        self.controller.set_disturbance_horizon(result.horizon, result.diagnostics)


def preflight(config=DEFAULT_CONFIG, model_config=ROOT / "configs/g1.yaml",
              mode="learned_filtered", bank=DEFAULT_BANK, cpu=None,
              actuation="reference_servo"):
    """Local libraries/model/real QP/IDL/CRC only; no DDS factory or endpoint."""
    from types import SimpleNamespace
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    runtime = MpcRuntime(config, model_config, mode, bank, stationary=True, actuation=actuation)
    try:
        q = np.zeros(35)
        from hardware_pid_control import ARM_MOTOR_INDICES
        q[list(ARM_MOTOR_INDICES)] = EXPECTED_TARGET_Q
        for ns in range(0, 502_000_000, 2_000_000):
            runtime.observe_low(ns, q, np.zeros(35))
            runtime.observe_imu(ns, np.array([1., 0, 0, 0]), np.zeros(3), np.array([0., 0, 9.81]))
        runtime.prepare(500_000_000, None, None, 0., .5)
        qr, dqr, diag = runtime.controller.step(EXPECTED_TARGET_Q, [1, 0, 0, 0], 0, .006)
        frame = dict(q_rad=EXPECTED_TARGET_Q.copy(), dq_rad_s=np.zeros(13),
                     kp=np.r_[np.full(11, 20), 0, 0], kd=np.r_[np.ones(11), 0, 0], weight=1.,
                     diagnostics=diag)
        frame["q_rad"][5:10], frame["dq_rad_s"][5:10] = qr, dqr
        packet_builder = make_arm_message
        if actuation == "inverse_dynamics_preview":
            from hardware_mpc_inverse_preview import make_offline_preview_message
            packet_builder = make_offline_preview_message
        elif actuation == "measured_torque_preview":
            from hardware_mpc_torque_control import make_torque_preview_message
            packet_builder = make_torque_preview_message
        packet = packet_builder(frame, SimpleNamespace(mode_pr=0, mode_machine=4),
                                unitree_hg_msg_dds__LowCmd_, CRC())
        serialized = packet.serialize()
        return json_values({"schema": "g1_hardware_mpc_preflight_v1", "passed": True,
            "dds_initialized": False, "publisher_created": False, "robot_connected": False,
            "host": host_evidence(cpu), "warmup": runtime.warmup,
            "solver_status": diag["mpc"]["solver_status"], "serialized_bytes": len(serialized),
            "actuation": actuation,
            "offline_packet_right_tau_nm": [packet.motor_cmd[i].tau for i in range(22,27)],
            "field_output_supported": actuation == "reference_servo",
            "core": runtime.controller.metadata,
            "predictor_manifest": None if runtime.predictor.bank is None else runtime.predictor.bank.manifest,
            "limitations": "offline compatibility only; not a 6 ms field or closed-loop performance certificate"})
    finally:
        runtime.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nic", nargs="?")
    parser.add_argument("--execute", action="store_true", help="real output; otherwise local preflight only")
    parser.add_argument("--preflight", action="store_true", help="explicit local-only check")
    parser.add_argument("--task", choices=("stationary", "walk"), default="stationary")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mpc-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--controller-config", type=Path, default=ROOT / "configs/g1.yaml")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--predictor", choices=("learned_filtered", "hold_current"), default="learned_filtered")
    parser.add_argument("--actuation", choices=("reference_servo", "inverse_dynamics_preview", "measured_torque_preview"),
                        default="measured_torque_preview",
                        help="default: measured-state torque migration, offline only; reference_servo is legacy")
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--permit-real-output", choices=(PERMIT,))
    parser.add_argument("--pid-6ms-validated", action="store_true",
                        help="operator confirms current 6 ms PID trial passed; not software-generated evidence")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    journal = runtime = None
    try:
        select_cpu(args.cpu)  # fail before subscribers/threads if CPU unavailable
        if args.preflight and args.execute:
            raise ValueError("--preflight and --execute are mutually exclusive")
        if args.execute and args.actuation != "reference_servo":
            raise ValueError(f"{args.actuation} is offline-only; torque field output is not commissioned")
        if not args.execute:
            print(json.dumps(preflight(args.mpc_config, args.controller_config,
                                      args.predictor, args.bank, args.cpu, args.actuation), indent=2))
            return 0
        if not (args.nic and args.profile and args.output_dir and
                args.permit_real_output == PERMIT and args.pid_6ms_validated):
            raise ValueError("real output requires NIC, reviewed --profile, new --output-dir, exact permit, "
                             "and --pid-6ms-validated after the actual successful PID trial")
        profile = load_profile(args.profile, "mpc")
        # Load/tree/build QP and warm all local math before any DDS initialization.
        runtime = MpcRuntime(args.mpc_config, args.controller_config, args.predictor,
                             args.bank, stationary=args.task == "stationary")
        journal = MpcJournal(args.output_dir)
        runtime.journal = journal
        for src, name in ((args.profile, "arm_profile.conf"),
                          (args.controller_config, "controller_config.yaml"),
                          (args.mpc_config, "mpc_config.yaml")):
            shutil.copy2(src, args.output_dir / name)
        source_paths = [Path(__file__), Path(__file__).with_name("g1_walk_pid.py"),
            Path(__file__).with_name("hardware_mpc_control.py"),
            Path(__file__).with_name("hardware_mpc_solver.py"),
            Path(__file__).with_name("hardware_mpc_predictor.py"),
            ROOT / "arm_mpc.py", ROOT / "kinematics_helper.py"]
        journal.record({"schema": "g1_mpc_session_v1", "event": "session_start",
            "program": Path(__file__).name, "task": args.task, "required_fsm": 500,
            "publisher_created": False, "mode_setter_registered": False, "lowcmd_topic_created": False,
            "network_interface": args.nic, "control_nominal_period_ms": 6.,
            "primary_metric_window_s": [5., 18.], "forward_speed_m_s": .5 if args.task == "walk" else 0.,
            "heading_target": "fixed_run_h0_positive_x", "host": host_evidence(args.cpu),
            "pid_6ms_validation": "operator_attestation_not_automatically_certified",
            "core": runtime.controller.metadata, "warmup": runtime.warmup,
            "predictor_mode": args.predictor,
            "predictor_manifest": None if runtime.predictor.bank is None else runtime.predictor.bank.manifest,
            "control_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in source_paths},
            "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            "controller_config_sha256": hashlib.sha256(args.controller_config.read_bytes()).hexdigest()})
        result = run_device(args, profile, None, {}, journal, runtime=runtime)
        journal.record({"schema": "g1_mpc_event_v1", "event": "capture_drained",
                        "queue_dropped": journal.dropped})
        journal.close()
        if journal.failed.is_set():
            print(f"MPC capture incomplete: {journal.failure_reason}; inspect {args.output_dir}", file=sys.stderr)
            return 3
        print(f"Saved MPC capture: {args.output_dir / 'raw.jsonl'}")
        return result
    except Exception as exc:
        if journal is not None:
            journal.record({"schema": "g1_mpc_event_v1", "event": "local_failure",
                            "reason": str(exc)})
        print(f"MPC refused/failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if journal is not None:
            journal.close()
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
