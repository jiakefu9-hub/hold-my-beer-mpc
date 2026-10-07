#!/usr/bin/env python3
"""Measured-state torque MPC development; legacy servo is offline comparison only.

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
import orjson

from g1_walk_pid import (ROOT, EXPECTED_TARGET_Q, Journal, load_profile,
                         run_device, make_arm_message)
from hardware_pid_control import HardwarePidPlan, CONTROL_PERIOD_S
from hardware_mpc_control import RightArmHardwareMpc, DEFAULT_CONFIG, json_values
from hardware_mpc_predictor import HardwareMpcPredictor, DEFAULT_BANK
from mpc_host import host_evidence, select_cpu, ControlThreadScope

PERMIT = "MPC_WALK_H0_CAPTURE"
FIELD_TORQUE_CONFIG = ROOT / 'configs/hardware_mpc_torque_field.yaml'
FIELD_MPC_START_S = 4.0  # one second of fixed posture, then MPC before walking at 5 s


class MpcJournal(Journal):
    """Same transport audit as PID, with explicit MPC record identities."""
    # Native encoding makes an immutable bytes snapshot before enqueue, so
    # later frame mutation cannot alter evidence. The writer only performs IO.
    snapshot = staticmethod(lambda row: orjson.dumps(
        row, option=orjson.OPT_SERIALIZE_NUMPY, default=json_values))
    format_row = staticmethod(lambda row: row.decode("utf-8") + "\n")

    def record(self, row):
        row = dict(row)
        schema = row.get("schema", "")
        if schema.startswith("g1_pid_") or schema.startswith("g1_hardware_pid_"):
            row["schema"] = schema.replace("pid", "mpc")
        if "pid_active" in row:
            row["mpc_active"] = row.pop("pid_active")
        super().record(row)


class MpcRuntime:
    field_trial = False
    def __init__(self, config=DEFAULT_CONFIG, model_config=ROOT / "configs/g1.yaml",
                 predictor_mode="learned_filtered", bank_path=DEFAULT_BANK,
                 stationary=False, journal=None, actuation="measured_torque_preview", torque_config=None,
                 assumed_command_delay_s=None, field_trial=False):
        from endpoint_pose import EndpointModel
        if actuation not in {"reference_servo", "inverse_dynamics_preview", "measured_torque_preview"}:
            raise ValueError("unsupported MPC actuation")
        self.actuation = actuation
        self.field_trial = bool(field_trial)
        self.host_scope = None
        self.handback = None
        if self.field_trial:
            if actuation != 'measured_torque_preview' or Path(torque_config or '') != FIELD_TORQUE_CONFIG:
                raise ValueError('field trial requires the bounded measured-torque field configuration')
            from hardware_mpc_field import TorqueHandback
            self.handback = TorqueHandback()
        if torque_config is not None and actuation != "measured_torque_preview":
            raise ValueError("--torque-config requires measured_torque_preview")
        if assumed_command_delay_s is not None and (actuation != "measured_torque_preview"
                or not np.isfinite(assumed_command_delay_s) or not 0 <= assumed_command_delay_s <= .020):
            raise ValueError("delay lifecycle requires measured_torque_preview; assumption must be 0..20ms")
        self.assumed_command_delay_s=assumed_command_delay_s
        self._latest_low_ns=self._latest_imu_ns=None
        self._delay_origin_ns=None
        self._plan=None
        self.stationary, self.journal = bool(stationary), journal
        self.predictor = HardwareMpcPredictor(predictor_mode, bank_path)
        controller_type = RightArmHardwareMpc
        if actuation == "inverse_dynamics_preview":
            from hardware_mpc_inverse_preview import RightArmInversePreviewMpc
            controller_type = RightArmInversePreviewMpc
        elif actuation == "measured_torque_preview":
            from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc
            controller_type = RightArmMeasuredTorqueMpc
        torque_options = {} if torque_config is None else dict(torque_config=torque_config)
        self.controller = controller_type(EXPECTED_TARGET_Q[5:10], config,
                                          model=EndpointModel(model_config), **torque_options)
        if self.field_trial:
            self.controller.metadata.update(field_output_supported=True,
                field_scope='explicit controlled first trial, not hardware-validated performance',
                field_torque_limits_calibrated=False, first_trial_task='stationary_before_walk')
        try:
            self.warmup = self.controller.warmup(EXPECTED_TARGET_Q, [1, 0, 0, 0])
        except Exception:
            self.controller.close()
            raise
        self.epoch_ns = None
        self._gc_was_enabled = None

    def validate_field_entry(self):
        if not self.field_trial or self.actuation != 'measured_torque_preview':
            raise ValueError('explicit torque field-trial authorization required')

    @staticmethod
    def create_crc():
        from mpc_crc import PackedCRC
        return PackedCRC()

    def enter_control_thread(self, cpu):
        if self.host_scope is None:
            raise RuntimeError('MPC requires explicit control/worker CPU placement')
        return self.host_scope.activate()

    def release_frame(self, elapsed_s):
        return self.handback.normal_release(elapsed_s)

    def accept_packet(self, frame, packet):
        # Called only AFTER Write succeeded; latch before any optional audit
        # can fail, so hand-back never loses the last actually issued packet.
        if self.handback is not None:
            self.handback.accept(packet)
        if self._plan is not None and getattr(self._plan, '_pending', None) is frame:
            self._plan.commit_packet(frame, packet)

    def check_before_write(self, frame, low, imu, begin_ns, now_ns):
        if not self.field_trial:
            return
        timing = dict(wall_since_loop_begin_ms=(now_ns-begin_ns)*1e-6,
                      selected_feedback_age_ms=(now_ns-min(low.received_ns,imu.received_ns))*1e-6)
        frame.setdefault('diagnostics', {})['field_prewrite_timing'] = timing
        self.controller.last_diagnostics['field_prewrite_timing'] = timing
        if now_ns - min(low.received_ns, imu.received_ns) > 25_000_000:
            raise RuntimeError('selected torque feedback older than 25 ms; hand back')
        if now_ns - begin_ns > 10_000_000:
            raise RuntimeError('torque computation older than 10 ms; hand back')
        # Geometry/torque values were checked by make_message; transport health
        # and selected-snapshot age are checked again immediately before Write.

    def close(self):
        if self._plan is not None:
            native = getattr(getattr(self._plan,'history',None),'native',None)
            if native is not None:
                native.close()
        self.controller.close()
        if self.host_scope is not None:
            self.host_scope.restore()
        if self._gc_was_enabled:
            gc.enable()

    def make_message(self, frame, state, constructor, crc):
        """One packet path for preflight/replay/transport; no output authority.

        Keeping transport behind its separate gate is essential: constructing
        nonzero tau does not commission timing, torque bounds or fault release.
        """
        if self.actuation == "measured_torque_preview":
            from hardware_mpc_torque_control import make_torque_preview_message
            if self.field_trial and frame.get('diagnostics',{}).get('controller_kind') != self.actuation:
                self.handback.apply(frame)
            if self.field_trial:
                from hardware_mpc_field import check_field_packet
                check_field_packet(frame, state, self.controller.torque_config)
            return make_torque_preview_message(frame, state, constructor, crc)
        if self.actuation == "inverse_dynamics_preview":
            from hardware_mpc_inverse_preview import make_offline_preview_message
            return make_offline_preview_message(frame, state, constructor, crc)
        if self.actuation == "reference_servo":
            return make_arm_message(frame, state, constructor, crc)
        raise ValueError("unknown MPC packet actuation")

    def observe_low(self, stamp, q, dq):
        accepted = self.predictor.observe_low(stamp, q, dq)
        if accepted:
            self._latest_low_ns=int(stamp)
        if accepted and self.journal is not None:
            self.journal.record({"schema": "g1_mpc_predictor_low_v1",
                "received_monotonic_ns": stamp, "q_rad": q, "dq_rad_s": dq})

    def observe_imu(self, stamp, quat, gyro, accel):
        accepted = self.predictor.observe_imu(stamp, quat, gyro, accel)
        if accepted:
            self._latest_imu_ns=int(stamp)
        if accepted and self.journal is not None:
            self.journal.record({"schema": "g1_mpc_predictor_imu_v1",
                "received_monotonic_ns": stamp, "quaternion_wxyz": quat,
                "gyroscope_rad_s": gyro, "accelerometer_raw_m_s2": accel})

    def prepare_startup(self):
        # Collection can stall for tens of milliseconds. Finish it before
        # opening the task clock or creating a command publisher.
        self._gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()

    def set_epoch(self, epoch_ns):
        self.epoch_ns = int(epoch_ns)
        self.predictor.set_grid_origin(epoch_ns)
        # The first command-time query is anchored to common RECEIVED data,
        # not wall time. Warming farther ahead can make the first live query
        # go backwards if no new callback arrived yet. Never reset mid-run.
        stamp = epoch_ns
        if self.assumed_command_delay_s is not None:
            if self._latest_low_ns is None or self._latest_imu_ns is None:
                raise ValueError('initial predictor needs both timestamped streams')
            stamp = min(stamp,self._latest_low_ns,self._latest_imu_ns)
        self.predictor.query(stamp, 0., use_learned=False)

    def create_plan(self, initial, profile):
        limits = np.asarray(self.controller.config["reference_offset_limit_deg"])
        if self.actuation != "measured_torque_preview" and np.any(limits > profile["q_offset_limit_deg_array"]):
            raise ValueError("MPC config exceeds reviewed profile reference bounds")
        self.controller.reset()
        plan_type = HardwarePidPlan
        if self.actuation == "measured_torque_preview":
            from hardware_mpc_torque_control import HardwareTorquePreviewPlan
            plan_type = HardwareTorquePreviewPlan
        options=dict(mpc_start_s=FIELD_MPC_START_S) if self.field_trial else {}
        if self.assumed_command_delay_s is not None:
            from hardware_mpc_delay_plan import HardwareDelayTorquePreviewPlan
            plan_type=HardwareDelayTorquePreviewPlan
            options.update(assumed_command_delay_s=self.assumed_command_delay_s)
        self._plan=plan_type(initial, profile["target_q_array"],
                            profile["kp_array"], profile["kd_array"], self.controller,**options)
        self.controller.metadata['state_input'] = (
            'measured q/dq directly; no model propagation' if self.assumed_command_delay_s is None
            else 'measured q/dq propagated to explicitly assumed command time')
        if self.assumed_command_delay_s is not None:
            from native_arm_delay import NativeArmDelay, LIBRARY
            if LIBRARY.is_file():
                native=NativeArmDelay(self.controller.inverse)
                self._plan.history.native=native
                self.controller.inverse.native_dynamics=native
                self.controller.metadata['delay_lifecycle']['propagation']=native.metadata
            elif self.field_trial:
                raise ValueError('build cpp/g1_arm_delay before field execution; no silent slower fallback')
            if self.field_trial:
                self.controller.metadata['delay_lifecycle'].update(field_output_supported=True,
                    history='only successful Write packets; host receive stamps and nominal application delay',
                    timing='host stamps plus explicit delay assumption; sensor/internal actuation latency unknown')
        return self._plan

    def prepare(self, now_ns, low, imu, yaw0, task_s, heading_frozen=True):
        # Advance causal filters from the first cycle, including the arm ramp.
        query_ns=now_ns
        if self.assumed_command_delay_s is not None:
            low_ns = self._latest_low_ns if low is None else low.received_ns
            imu_ns = self._latest_imu_ns if imu is None else imu.received_ns
            if low_ns is None or imu_ns is None:
                raise ValueError('delay lifecycle requires timestamped lowstate and IMU')
            if max(now_ns-low_ns,now_ns-imu_ns)>self.predictor.max_stale_ns:
                raise ValueError('delay lifecycle observation stale at current time')
            # Anchor at common available past data, not at the query clock.
            # Raw arm observation and filtered forecast may have different times.
            query_ns=min(now_ns,low_ns,imu_ns)
        result = self.predictor.query(query_ns, yaw0,
                                      use_learned=not self.stationary and task_s >= 5. and heading_frozen)
        self.controller.set_disturbance_horizon(result.horizon, result.diagnostics)
        if self.assumed_command_delay_s is not None and self._plan is not None:
            if self._delay_origin_ns is None:
                self._delay_origin_ns=int(now_ns)
            self._plan.set_context((now_ns-self._delay_origin_ns)*1e-9,
                (low_ns-self._delay_origin_ns)*1e-9,
                (result.diagnostics['anchor_monotonic_ns']-self._delay_origin_ns)*1e-9)


def preflight(config=DEFAULT_CONFIG, model_config=ROOT / "configs/g1.yaml",
              mode="learned_filtered", bank=DEFAULT_BANK, cpu=None,
              actuation="measured_torque_preview", torque_config=None):
    """Local libraries/model/real QP/IDL/CRC only; no DDS factory or endpoint."""
    from types import SimpleNamespace
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    runtime = MpcRuntime(config, model_config, mode, bank, stationary=True, actuation=actuation,
                         torque_config=torque_config)
    native = None
    try:
        native_metadata = None
        if actuation == 'measured_torque_preview':
            from native_arm_delay import NativeArmDelay, LIBRARY
            if LIBRARY.is_file():
                native = NativeArmDelay(runtime.controller.inverse)
                native_metadata = native.metadata
            elif torque_config is not None and Path(torque_config).resolve() == FIELD_TORQUE_CONFIG:
                raise ValueError('field preflight requires building cpp/g1_arm_delay')
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
        if actuation == 'measured_torque_preview':
            # Exercise the exact evaluated packet gains.  Field overlays may
            # intentionally soften one axis; a preflight using hard-coded A3
            # gains would either test the wrong packet or, correctly, fail the
            # packet/model consistency gate.
            frame["kp"][5:10] = diag["expected_kp"]
            frame["kd"][5:10] = diag["expected_kd"]
        packet = runtime.make_message(frame, SimpleNamespace(mode_pr=0, mode_machine=4),
                                      unitree_hg_msg_dds__LowCmd_, CRC())
        serialized = packet.serialize()
        return json_values({"schema": "g1_hardware_mpc_preflight_v1", "passed": True,
            "dds_initialized": False, "publisher_created": False, "robot_connected": False,
            "host": host_evidence(cpu), "warmup": runtime.warmup,
            "solver_status": diag["mpc"]["solver_status"], "serialized_bytes": len(serialized),
            "actuation": actuation,
            "native_delay": native_metadata,
            "offline_packet_right_tau_nm": [packet.motor_cmd[i].tau for i in range(22,27)],
            "field_output_supported": False,
            "core": runtime.controller.metadata,
            "predictor_manifest": None if runtime.predictor.bank is None else runtime.predictor.bank.manifest,
            "limitations": "offline compatibility only; not a 6 ms field or closed-loop performance certificate"})
    finally:
        if native is not None:
            native.close()
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
    parser.add_argument("--torque-config", type=Path, help="offline overlay; field uses fixed bounded trial config")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--predictor", choices=("learned_filtered", "hold_current"), default="learned_filtered")
    parser.add_argument("--actuation", choices=("reference_servo", "inverse_dynamics_preview", "measured_torque_preview"),
                        default="measured_torque_preview",
                        help="measured torque: explicit controlled field trial; other modes offline only")
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument('--rt-priority', type=int, default=0, help='0=ordinary; 1..40=control-thread FIFO, requires permission')
    parser.add_argument('--compute-process',action='store_true',
                        help='isolate SDK-free MPC calculation from Python DDS callbacks')
    parser.add_argument('--assumed-command-delay-ms', type=float, default=None,
                        help='opt-in model propagation; omitted uses measured q/dq directly. 0 still propagates observation age')
    parser.add_argument('--allow-first-torque-field-trial', action='store_true',
                        help='explicit stationary nonzero-torque commissioning trial')
    parser.add_argument('--torque-stationary-validated', action='store_true',
                        help='operator confirms an actual stationary torque trial before requesting walk')
    parser.add_argument("--permit-real-output", choices=(PERMIT,))
    parser.add_argument("--pid-6ms-validated", action="store_true",
                        help="operator confirms current 6 ms PID trial passed; not software-generated evidence")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    journal = runtime = scope = None
    try:
        select_cpu(args.cpu)  # fail before subscribers/threads if CPU unavailable
        if args.preflight and args.execute:
            raise ValueError("--preflight and --execute are mutually exclusive")
        if args.execute and args.actuation == "reference_servo":
            raise ValueError("reference_servo is retired from field use; first hardware MPC must use "
                             "the explicit measured-torque trial path")
        if args.execute and (args.actuation != 'measured_torque_preview' or
                             not args.allow_first_torque_field_trial):
            raise ValueError('offline-only unless measured torque and --allow-first-torque-field-trial are explicit')
        if args.execute and args.task == 'walk' and not args.torque_stationary_validated:
            raise ValueError('first torque trial is stationary; walk requires an actual stationary trial result')
        if args.execute and args.torque_config is not None and args.torque_config.resolve() != FIELD_TORQUE_CONFIG:
            raise ValueError('field execution accepts only configs/hardware_mpc_torque_field.yaml')
        if not args.execute:
            print(json.dumps(preflight(args.mpc_config, args.controller_config,
                                      args.predictor, args.bank, args.cpu, args.actuation,
                                      args.torque_config), indent=2))
            return 0
        if not (args.nic and args.profile and args.output_dir and
                args.permit_real_output == PERMIT and args.pid_6ms_validated):
            raise ValueError("real output requires NIC, reviewed --profile, new --output-dir, exact permit, "
                             "and --pid-6ms-validated after the actual successful PID trial")
        # Reuse established robot/transport facts from the validated PID profile,
        # without inventing MPC-specific reviewed flags in a copied profile.
        profile_kind = 'pid' if 'schema=g1_hardware_pid_walk_site_v1' in args.profile.read_text() else 'mpc'
        profile = load_profile(args.profile, profile_kind)
        scope = ControlThreadScope(args.cpu, args.rt_priority)
        scope.prepare_workers()
        # Load/tree/build QP and warm all local math before any DDS initialization.
        runtime_type=MpcRuntime;compute_options={}
        if args.compute_process:
            from mpc_compute_process import ProcessMpcRuntime
            runtime_type=ProcessMpcRuntime
            compute_options=dict(compute_cpu=args.cpu,compute_priority=args.rt_priority,
                                 compute_affinity=scope.affinity)
        runtime = runtime_type(args.mpc_config, args.controller_config, args.predictor,
                             args.bank, stationary=args.task == "stationary", field_trial=True,
                             torque_config=FIELD_TORQUE_CONFIG,
                             assumed_command_delay_s=(None if args.assumed_command_delay_ms is None
                                                      else args.assumed_command_delay_ms*.001),**compute_options)
        runtime.host_scope = scope
        journal = MpcJournal(args.output_dir)
        runtime.journal = journal
        for src, name in ((args.profile, "arm_profile.conf"),
                          (args.controller_config, "controller_config.yaml"),
                          (args.mpc_config, "mpc_config.yaml"),
                          (FIELD_TORQUE_CONFIG, 'torque_config.yaml')):
            shutil.copy2(src, args.output_dir / name)
        source_paths = [Path(__file__), Path(__file__).with_name("g1_walk_pid.py"),
            Path(__file__).with_name("hardware_mpc_control.py"),
            Path(__file__).with_name("hardware_mpc_solver.py"),
            Path(__file__).with_name("hardware_mpc_predictor.py"),
            *[Path(__file__).with_name(name) for name in ('hardware_mpc_field.py',
                'hardware_mpc_torque_control.py','hardware_arm_inverse_dynamics.py',
                'hardware_torque_mapper.py','hardware_mpc_delay_plan.py',
                'hardware_mpc_delay_preview.py','mpc_host.py','arm_execution_record.py','native_arm_delay.py',
                'mpc_compute_process.py','mpc_crc.py','hardware_mpc_recovery.py',
                'hardware_mpc_braking.py')],
            ROOT/'cpp/g1_arm_delay/delay.cpp', ROOT/'cpp/g1_arm_delay/CMakeLists.txt',
            ROOT / "arm_mpc.py", ROOT / "kinematics_helper.py"]
        journal.record({"schema": "g1_mpc_session_v1", "event": "session_start",
            "program": Path(__file__).name, "task": args.task, "required_fsm": 500,
            "publisher_created": False, "mode_setter_registered": False, "lowcmd_topic_created": False,
            "network_interface": args.nic, "control_nominal_period_ms": 6.,
            "mpc_start_s": FIELD_MPC_START_S,
            "primary_metric_window_s": [5., 18.], "forward_speed_m_s": .5 if args.task == "walk" else 0.,
            "heading_target": "fixed_run_h0_positive_x", "host_before_control": host_evidence(),
            "requested_control_cpu": args.cpu, "requested_fifo_priority": args.rt_priority,
            "assumed_command_delay_ms": args.assumed_command_delay_ms,
            "hardware_delay_identified": False, "profile_kind": profile_kind,
            "pid_6ms_validation": "operator_attestation_not_automatically_certified",
            "core": runtime.controller.metadata, "warmup": runtime.warmup,
            "predictor_mode": args.predictor,
            "predictor_manifest": None if runtime.predictor.bank is None else runtime.predictor.bank.manifest,
            "control_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in source_paths},
            "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            "controller_config_sha256": hashlib.sha256(args.controller_config.read_bytes()).hexdigest(),
            "torque_config_sha256": hashlib.sha256(FIELD_TORQUE_CONFIG.read_bytes()).hexdigest()})
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
        elif scope is not None:
            scope.restore()


if __name__ == "__main__":
    raise SystemExit(main())
