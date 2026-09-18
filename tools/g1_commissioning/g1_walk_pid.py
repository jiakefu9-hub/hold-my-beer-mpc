#!/usr/bin/env python3
"""Opt-in G1 Arm SDK walk experiment with fixed-left / right-arm PID control.

The robot must already be stationary and self-balancing in Regular Motion Mode
(FSM 500).  This program never switches mode and never creates ``rt/lowcmd``.
It publishes only ``rt/arm_sdk`` and calls only the locomotion velocity setter.

Imports of Unitree SDK2 Python are deliberately delayed until the profile,
arguments and explicit permit have passed local validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import queue
import shutil
import signal
import sys
import threading
import time

import numpy as np
import yaml

from hardware_pid_control import (
    ARM_MOTOR_INDICES,
    END_S,
    FixedH0Heading,
    FORWARD_SPEED_M_S,
    HardwarePidPlan,
    PidParameters,
    RELEASE_START_S,
    RightArmHardwarePid,
    VALID_ARM_SLOTS,
    WALK_START_S,
    WALK_STOP_S,
    WEIGHT_MOTOR_INDEX,
    vertical_angular_rate,
    yaw_from_quaternion,
)
from endpoint_pose import EndpointModel

ROOT = Path(__file__).resolve().parents[2]
PERMIT = "PID_WALK_H0_CAPTURE"
SCHEMA = "g1_hardware_pid_walk_site_v1"
CONTROL_PERIOD_S = 0.020
STATE_TIMEOUT_NS = 100_000_000
FSM_TIMEOUT_NS = 200_000_000
EXPECTED_TARGET_Q = np.deg2rad([
    -4.0, -1.0, 0.0, -8.1, 0.0,
    -4.0, 1.0, 0.0, -7.8, 0.0,
    0.0, 0.0, 0.0,
])
REQUIRED_CONFIRMATIONS = (
    "hardware_identity_confirmed",
    "mapping_confirmed",
    "weight_scope_confirmed",
    "invalid_slots_confirmed",
    "regular_motion_mode_confirmed",
    "grounded_self_balance_confirmed",
    "loss_response_plan_reviewed",
    "normal_release_plan_reviewed",
    "emergency_procedure_confirmed",
    "no_competing_user_publishers_confirmed",
    "safety_parameters_confirmed",
    "hardware_pid_controller_confirmed",
    "pid_parameters_confirmed",
    "h0_metric_window_confirmed",
)


def monotonic_ns():
    return time.monotonic_ns()


def json_number(value):
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return json_number(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def parse_bool(value):
    lowered = str(value).strip().lower()
    if lowered not in {"true", "false"}:
        raise ValueError(f"expected true/false, got {value!r}")
    return lowered == "true"


def parse_vector(value, count, name):
    result = np.asarray([float(item.strip()) for item in str(value).split(",")], dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {count} finite comma-separated values")
    return result


def load_profile(path: Path):
    values = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"profile line {line_number} is not key=value")
        key, value = (part.strip() for part in line.split("=", 1))
        if key in values:
            raise ValueError(f"duplicate profile key: {key}")
        values[key] = value
    required_text = (
        "schema", "robot_id", "model_name", "joint_layout", "confirmed_by",
        "loco_service", "profile_status",
    )
    missing = [name for name in (*required_text, *REQUIRED_CONFIRMATIONS) if name not in values]
    if missing:
        raise ValueError(f"profile missing keys: {', '.join(missing)}")
    if values["schema"] != SCHEMA:
        raise ValueError(f"profile schema must be {SCHEMA}")
    if values["profile_status"] != "FIELD_REVIEWED":
        raise ValueError("profile_status must be FIELD_REVIEWED")
    if values["robot_id"] in {"", "UNSET"} or values["model_name"] in {"", "UNSET"}:
        raise ValueError("robot_id and model_name must be field-reviewed values")
    if values["confirmed_by"] in {"", "UNSET"}:
        raise ValueError("confirmed_by must identify the field reviewer")
    if values["loco_service"] != "sport":
        raise ValueError("this field tool accepts only the reviewed sport service")
    if values["joint_layout"] != "g1_23_arm5":
        raise ValueError("only the confirmed G1 23DoF Arm5 mapping is accepted")
    for name in REQUIRED_CONFIRMATIONS:
        if not parse_bool(values[name]):
            raise ValueError(f"{name} must be true")
    if parse_bool(values.get("synthetic_fixture", "false")):
        raise ValueError("synthetic profiles are never accepted for output")
    if int(values.get("required_fsm", -1)) != 500:
        raise ValueError("required_fsm must be 500")
    valid_slots = tuple(int(value) for value in values.get("valid_slots", "").split(","))
    if valid_slots != VALID_ARM_SLOTS or values.get("invalid_slot_policy") != "zero":
        raise ValueError("Arm5 slots must be exactly 0..10 and invalid slots must be zero")
    target_q = parse_vector(values.get("target_q", ""), 13, "target_q")
    if not np.allclose(target_q, EXPECTED_TARGET_Q, atol=1e-12, rtol=0.0):
        raise ValueError("target_q must match the established world-upright A3 pose")
    kp = parse_vector(values.get("kp", ""), 13, "kp")
    kd = parse_vector(values.get("kd", ""), 13, "kd")
    if not np.array_equal(kp, np.r_[np.full(11, 20.0), 0.0, 0.0]):
        raise ValueError("kp must be 20 on valid Arm5 slots and zero on invalid slots")
    if not np.array_equal(kd, np.r_[np.full(11, 1.0), 0.0, 0.0]):
        raise ValueError("kd must be 1 on valid Arm5 slots and zero on invalid slots")
    q_limit = parse_vector(values.get("pid_q_offset_limit_deg", ""), 5,
                           "pid_q_offset_limit_deg")
    if np.any(q_limit <= 0.0) or np.any(q_limit > 5.0):
        raise ValueError("PID q-reference offsets must be in (0,5] degrees")
    expected = {
        "max_weight": 1.0,
        "weight_rate_per_s": 1.0 / 3.0,
        "control_period_ms": 20.0,
        "state_timeout_ms": 100.0,
        "startup_wait_s": 5.0,
        "startup_valid_samples": 50.0,
        "walk_start_s": WALK_START_S,
        "walk_stop_s": WALK_STOP_S,
        "release_start_s": RELEASE_START_S,
        "end_s": END_S,
        "forward_speed_m_s": FORWARD_SPEED_M_S,
    }
    for name, expected_value in expected.items():
        actual = float(values.get(name, "nan"))
        if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name} must be {expected_value}")
    return {
        **values,
        "target_q_array": target_q,
        "kp_array": kp,
        "kd_array": kd,
        "pid_q_offset_limit_deg_array": q_limit,
        "startup_valid_samples_int": int(float(values["startup_valid_samples"])),
    }


def load_pid_parameters(config_path: Path, profile):
    config = yaml.safe_load(config_path.read_text())
    selected = {
        name: config[name]
        for name in (
            "pid_kp_pose", "pid_kd_pose", "pid_ki_pose", "pid_posture_gain",
            "pid_finite_diff_eps", "pid_damping", "pid_integral_limit",
            "pid_max_dq", "pid_de_g_alpha",
        )
    }
    selected["pid_q_offset_limit_deg"] = profile["pid_q_offset_limit_deg_array"].tolist()
    return PidParameters.from_mapping(selected), selected


class Journal:
    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=False)
        self.output_dir = output_dir
        self._stream = (output_dir / "raw.jsonl").open("x", buffering=1)
        self._queue = queue.Queue(maxsize=65536)
        self.failed = threading.Event()
        self.dropped = 0
        self.written = 0
        self._closed = False
        self._stop = object()
        self._thread = threading.Thread(target=self._run, name="pid_journal", daemon=True)
        self._thread.start()

    def record(self, row):
        row = json_safe(dict(row))
        row.setdefault("monotonic_ns", monotonic_ns())
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1
            self.failed.set()

    def _run(self):
        try:
            while True:
                row = self._queue.get()
                if row is self._stop:
                    break
                self._stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                self.written += 1
                self._queue.task_done()
        except Exception:
            self.failed.set()

    def close(self):
        if self._closed:
            return
        if self._thread.is_alive():
            self._queue.put(self._stop)
        self._thread.join()
        self._stream.flush()
        self._stream.close()
        self._closed = True


class Interlock:
    def __init__(self):
        self._lock = threading.Lock()
        self._fault = None
        self._fsm_request_ns = None
        self._last_tick = None

    def trip(self, reason):
        with self._lock:
            if self._fault is None:
                self._fault = str(reason)

    def observe_tick(self, tick):
        tick = int(tick) & 0xFFFFFFFF
        with self._lock:
            if self._last_tick is not None:
                delta = (tick - self._last_tick) & 0xFFFFFFFF
                if delta >= 0x80000000:
                    self._fault = self._fault or "LowState tick rollback"
            self._last_tick = tick

    def observe_remote(self, remote):
        if len(remote) > 3:
            keys = int(remote[2]) | (int(remote[3]) << 8)
            if keys & ((1 << 5) | (1 << 9)) == ((1 << 5) | (1 << 9)):
                self.trip("remote L2+B observed")

    def observe_fsm(self, rc, value, request_ns, reply_ns):
        with self._lock:
            if rc != 0:
                self._fault = self._fault or f"FSM getter failed: rc={rc}"
            elif value != 500:
                self._fault = self._fault or f"left required FSM 500: FSM={value}"
            elif request_ns <= 0 or reply_ns < request_ns or reply_ns - request_ns > FSM_TIMEOUT_NS:
                self._fault = self._fault or "invalid or late FSM reply"
            elif self._fsm_request_ns is not None and request_ns <= self._fsm_request_ns:
                self._fault = self._fault or "non-monotonic FSM observation"
            else:
                self._fsm_request_ns = request_ns

    def check(self, now_ns):
        with self._lock:
            if self._fault:
                return self._fault
            if self._fsm_request_ns is None:
                return "FSM not yet observed"
            if now_ns < self._fsm_request_ns or now_ns - self._fsm_request_ns > FSM_TIMEOUT_NS:
                self._fault = "FSM observation stale or future"
                return self._fault
            return ""


@dataclass(frozen=True)
class LowSnapshot:
    received_ns: int
    sequence: int
    tick: int
    mode_pr: int
    mode_machine: int
    q: np.ndarray
    dq: np.ndarray
    crc_valid: bool


@dataclass(frozen=True)
class ImuSnapshot:
    received_ns: int
    sequence: int
    quaternion: np.ndarray
    rpy: np.ndarray
    gyro: np.ndarray
    accelerometer: np.ndarray


class Streams:
    def __init__(self, journal, interlock, crc):
        self.journal = journal
        self.interlock = interlock
        self.crc = crc
        self.lock = threading.Lock()
        self.low = None
        self.imu = None
        self.low_sequence = 0
        self.imu_sequence = 0
        self.epoch_ns = None
        self.heading = FixedH0Heading()

    @staticmethod
    def _imu_fields(message):
        return {
            "quaternion_wxyz": [float(value) for value in message.quaternion],
            "rpy_rad": [float(value) for value in message.rpy],
            "gyroscope_rad_s": [float(value) for value in message.gyroscope],
            "accelerometer_raw_m_s2": [float(value) for value in message.accelerometer],
            "temperature_raw": int(message.temperature),
        }

    def low_callback(self, message):
        now = monotonic_ns()
        self.low_sequence += 1
        crc_valid = False
        try:
            crc_valid = int(self.crc.Crc(message)) == int(message.crc)
        except Exception:
            pass
        q = np.asarray([float(motor.q) for motor in message.motor_state], dtype=float)
        dq = np.asarray([float(motor.dq) for motor in message.motor_state], dtype=float)
        remote = [int(value) for value in message.wireless_remote]
        snapshot = LowSnapshot(now, self.low_sequence, int(message.tick),
                               int(message.mode_pr), int(message.mode_machine), q, dq, crc_valid)
        if crc_valid:
            self.interlock.observe_tick(message.tick)
            self.interlock.observe_remote(remote)
        with self.lock:
            self.low = snapshot
        self.journal.record({
            "schema": "g1_lowstate_raw_v1", "topic": "rt/lowstate",
            "received_monotonic_ns": now,
            "host_callback_sequence": self.low_sequence,
            "tick_raw": int(message.tick), "mode_pr": int(message.mode_pr),
            "mode_machine": int(message.mode_machine), "crc_raw": int(message.crc),
            "crc_valid": crc_valid, "wireless_remote_bytes": remote,
            "version": [int(value) for value in message.version],
            "reserve": [int(value) for value in message.reserve],
            "pelvis_imu": self._imu_fields(message.imu_state),
            "motors": [{
                "index": index, "mode": int(motor.mode), "q_rad": json_number(motor.q),
                "dq_rad_s": json_number(motor.dq), "ddq_raw_rad_s2": json_number(motor.ddq),
                "tau_est_nm": json_number(motor.tau_est),
                "temperature_raw": [int(value) for value in motor.temperature],
                "vol_raw": json_number(motor.vol),
                "sensor_raw": [int(value) for value in motor.sensor],
                "motorstate_raw": int(motor.motorstate),
            } for index, motor in enumerate(message.motor_state)],
        })

    def imu_callback(self, message):
        now = monotonic_ns()
        self.imu_sequence += 1
        fields = self._imu_fields(message)
        snapshot = ImuSnapshot(
            now, self.imu_sequence,
            np.asarray(fields["quaternion_wxyz"], dtype=float),
            np.asarray(fields["rpy_rad"], dtype=float),
            np.asarray(fields["gyroscope_rad_s"], dtype=float),
            np.asarray(fields["accelerometer_raw_m_s2"], dtype=float),
        )
        with self.lock:
            self.imu = snapshot
            epoch = self.epoch_ns
            if epoch is not None:
                task_s = (now - epoch) * 1e-9
                self.heading.observe(
                    now, task_s, yaw_from_quaternion(snapshot.quaternion),
                    vertical_angular_rate(snapshot.rpy, snapshot.gyro),
                )
        self.journal.record({
            "schema": "g1_torso_imu_raw_v1", "topic": "rt/secondary_imu",
            "received_monotonic_ns": now,
            "host_callback_sequence": self.imu_sequence,
            "source_timestamp_available": False, **fields,
        })

    def latest(self):
        with self.lock:
            return self.low, self.imu

    def set_epoch(self, epoch_ns):
        with self.lock:
            self.epoch_ns = int(epoch_ns)
            if self.imu is not None:
                task_s = (self.imu.received_ns - self.epoch_ns) * 1e-9
                self.heading.observe(
                    self.imu.received_ns, task_s,
                    yaw_from_quaternion(self.imu.quaternion),
                    vertical_angular_rate(self.imu.rpy, self.imu.gyro),
                )

    def heading_current(self):
        with self.lock:
            return self.heading.current()

    def freeze_heading(self):
        with self.lock:
            return self.heading.freeze(), self.heading.current()


def finite_array(array):
    return np.isfinite(np.asarray(array, dtype=float)).all()


def health(streams, interlock, journal, now_ns=None):
    now_ns = monotonic_ns() if now_ns is None else int(now_ns)
    stop = interlock.check(now_ns)
    if stop:
        return stop
    if journal.failed.is_set() or journal.dropped:
        return "raw recorder failure/queue overflow"
    low, imu = streams.latest()
    if low is None or now_ns < low.received_ns or now_ns - low.received_ns > STATE_TIMEOUT_NS:
        return "LowState unavailable/stale/future"
    if not low.crc_valid:
        return "LowState CRC invalid"
    if len(low.q) <= WEIGHT_MOTOR_INDEX or not finite_array(
        [low.q[index] for index in ARM_MOTOR_INDICES[:11]]
    ) or not finite_array([low.dq[index] for index in ARM_MOTOR_INDICES[:11]]):
        return "LowState motor data invalid"
    if imu is None or now_ns < imu.received_ns or now_ns - imu.received_ns > STATE_TIMEOUT_NS:
        return "torso IMU unavailable/stale/future"
    if not all(finite_array(value) for value in
               (imu.quaternion, imu.rpy, imu.gyro, imu.accelerometer)):
        return "torso IMU nonfinite"
    norm = np.linalg.norm(imu.quaternion)
    if not 0.5 < norm < 1.5:
        return "torso IMU quaternion invalid"
    return ""


def startup_gate(streams, interlock, journal, after_sequence, required_samples, wait_s=5.0):
    deadline = time.monotonic() + wait_s
    previous = int(after_sequence)
    consecutive = 0
    while time.monotonic() < deadline:
        low, _ = streams.latest()
        if low is not None and low.sequence > previous:
            previous = low.sequence
            if not health(streams, interlock, journal):
                consecutive += 1
                if consecutive >= required_samples:
                    return low
            else:
                consecutive = 0
        time.sleep(0.001)
    raise RuntimeError("fresh consecutive FSM-500/LowState/torso samples unavailable")


def _arm_slots(snapshot):
    return np.asarray([snapshot.q[index] for index in ARM_MOTOR_INDICES], dtype=float)


def _arm_dq(snapshot):
    return np.asarray([snapshot.dq[index] for index in ARM_MOTOR_INDICES], dtype=float)


def run_device(args, profile, pid_parameters, pid_mapping, journal):
    # Delayed imports: reaching this point still has not initialized DDS.
    import unitree_sdk2py
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.g1.loco.g1_loco_api import (
        LOCO_API_VERSION, LOCO_SERVICE_NAME,
        ROBOT_API_ID_LOCO_GET_FSM_ID, ROBOT_API_ID_LOCO_SET_VELOCITY,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_
    from unitree_sdk2py.rpc.client import Client
    from unitree_sdk2py.utils.crc import CRC

    sdk_root = Path(unitree_sdk2py.__file__).resolve().parent
    sdk_files = [
        sdk_root / "core/channel.py",
        sdk_root / "rpc/client.py",
        sdk_root / "rpc/client_base.py",
        sdk_root / "utils/crc.py",
        sdk_root / "g1/loco/g1_loco_api.py",
        sdk_root / "idl/unitree_hg/msg/dds_/_LowCmd_.py",
        sdk_root / "idl/unitree_hg/msg/dds_/_LowState_.py",
    ]
    journal.record({
        "schema": "g1_pid_event_v1", "event": "unitree_sdk2_python_provenance",
        "package_root": str(sdk_root),
        "critical_file_sha256": {
            str(path.relative_to(sdk_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sdk_files
        },
    })

    class FsmGetter(Client):
        def __init__(self):
            super().__init__(LOCO_SERVICE_NAME, False)
            self.SetTimeout(0.1)
            self._SetApiVerson(LOCO_API_VERSION)
            self._RegistApi(ROBOT_API_ID_LOCO_GET_FSM_ID, 0)

        def get(self):
            code, raw = self._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "{}")
            value = -1
            if code == 0:
                value = int(json.loads(raw)["data"])
            return code, value, raw

    class VelocitySetter(Client):
        def __init__(self):
            super().__init__(LOCO_SERVICE_NAME, False)
            self.SetTimeout(0.1)
            self._SetApiVerson(LOCO_API_VERSION)
            self._RegistApi(ROBOT_API_ID_LOCO_SET_VELOCITY, 0)

        def send(self, vx, wz, duration):
            payload = json.dumps({"velocity": [float(vx), 0.0, float(wz)],
                                  "duration": float(duration)})
            code, raw = self._Call(ROBOT_API_ID_LOCO_SET_VELOCITY, payload)
            return code, raw

    interlock = Interlock()
    crc = CRC()
    ChannelFactoryInitialize(0, args.nic)
    streams = Streams(journal, interlock, crc)
    low_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    imu_subscriber = ChannelSubscriber("rt/secondary_imu", IMUState_)
    low_subscriber.Init(streams.low_callback, 0)
    imu_subscriber.Init(streams.imu_callback, 0)
    worker_stop = threading.Event()
    fsm_client = FsmGetter()

    def fsm_worker():
        try:
            while not worker_stop.is_set():
                begin = monotonic_ns()
                try:
                    rc, value, raw = fsm_client.get()
                except Exception as exc:
                    rc, value, raw = -1, -1, repr(exc)
                end = monotonic_ns()
                journal.record({
                    "schema": "g1_pid_event_v1", "event": "fsm_reply",
                    "request_ns": begin, "reply_ns": end, "return_code": rc,
                    "fsm_id": value, "raw_reply": raw,
                })
                interlock.observe_fsm(rc, value, begin, end)
                worker_stop.wait(0.05)
        except Exception as exc:
            interlock.trip(f"FSM worker exception: {exc}")

    fsm_thread = threading.Thread(target=fsm_worker, name="pid_fsm", daemon=True)
    fsm_thread.start()
    publisher = None
    velocity_thread = None
    velocity_failed = threading.Event()
    last_zero_reply_ns = 0
    last_zero_lock = threading.Lock()
    stop_requested = threading.Event()

    def request_stop(_signal=None, _frame=None):
        stop_requested.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)

    def make_message(frame, state):
        message = unitree_hg_msg_dds__LowCmd_()
        message.mode_pr = int(state.mode_pr)
        message.mode_machine = int(state.mode_machine)
        for slot in VALID_ARM_SLOTS:
            command = message.motor_cmd[ARM_MOTOR_INDICES[slot]]
            command.mode = 1
            command.q = float(frame["q_rad"][slot])
            command.dq = float(frame["dq_rad_s"][slot])
            command.tau = 0.0
            command.kp = float(frame["kp"][slot])
            command.kd = float(frame["kd"][slot])
        message.motor_cmd[WEIGHT_MOTOR_INDEX].q = float(frame["weight"])
        message.crc = crc.Crc(message)
        return message

    try:
        initial = startup_gate(
            streams, interlock, journal, 0, profile["startup_valid_samples_int"]
        )
        print(
            "REAL PID WALK OUTPUT: 3 s arm entry, 2 s baseline, 10 s at 0.5 m/s, "
            "3 s stop-settle, 3 s release. Robot must already be stationary in FSM 500."
        )
        print(f"Type exactly: EXECUTE {profile['robot_id']}")
        reply = input("> ")
        if reply != f"EXECUTE {profile['robot_id']}" or stop_requested.is_set():
            raise RuntimeError("confirmation rejected before command output")
        before, _ = streams.latest()
        initial = startup_gate(
            streams, interlock, journal, before.sequence,
            profile["startup_valid_samples_int"],
        )
        failure = health(streams, interlock, journal)
        if failure:
            raise RuntimeError(f"pre-publisher: {failure}")

        # Publisher creation occurs only after both state gates and the typed confirmation.
        publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        publisher.Init()
        journal.record({
            "schema": "g1_pid_event_v1", "event": "arm_publisher_created",
            "topic": "rt/arm_sdk",
        })
        controller = RightArmHardwarePid(
            profile["target_q_array"][5:10], pid_parameters,
            model=EndpointModel(args.controller_config),
            control_dt=CONTROL_PERIOD_S,
        )
        plan = HardwarePidPlan(
            _arm_slots(initial), profile["target_q_array"],
            profile["kp_array"], profile["kd_array"], controller,
        )
        epoch_ns = monotonic_ns()
        streams.set_epoch(epoch_ns)
        journal.record({
            "schema": "g1_pid_event_v1", "event": "task_epoch",
            "task_epoch_monotonic_ns": epoch_ns,
        })

        def velocity_worker():
            nonlocal last_zero_reply_ns
            client = VelocitySetter()
            heading_frozen_logged = False
            try:
                while not worker_stop.is_set():
                    iteration = time.monotonic()
                    task_s = (monotonic_ns() - epoch_ns) * 1e-9
                    failure_now = health(streams, interlock, journal)
                    if task_s >= WALK_START_S and not heading_frozen_logged and not failure_now:
                        yaw0, heading = streams.freeze_heading()
                        heading_frozen_logged = True
                        journal.record({
                            "schema": "g1_pid_event_v1", "event": "heading_reference_frozen",
                            "task_elapsed_s": task_s, "yaw0_rad": yaw0,
                            "h0_from_navigation_world_yaw_rad": -yaw0,
                            "requested_reference_start_s": 3.0,
                            "requested_reference_end_s": 5.0,
                            **heading,
                            "h0_definition": "fixed_run_frame_x_along_pre_walk_mean_yaw_z_vertical",
                        })
                    heading = streams.heading_current()
                    walking = (
                        not stop_requested.is_set() and not failure_now
                        and WALK_START_S <= task_s < WALK_STOP_S
                    )
                    vx = FORWARD_SPEED_M_S if walking else 0.0
                    wz = heading["correction_rad_s"] if walking else 0.0
                    duration = min(0.2, WALK_STOP_S - task_s) if walking else 0.2
                    begin = monotonic_ns()
                    rc, raw = client.send(vx, wz, duration)
                    end = monotonic_ns()
                    journal.record({
                        "schema": "g1_pid_event_v1", "event": "velocity_reply",
                        "task_elapsed_s": task_s, "request_ns": begin, "reply_ns": end,
                        "return_code": rc, "raw_reply": raw,
                        "vx_m_s": vx, "vy_m_s": 0.0, "yaw_rate_rad_s": wz,
                        "duration_s": duration, "zero_command": vx == 0.0 and wz == 0.0,
                        **heading,
                    })
                    if rc != 0:
                        velocity_failed.set()
                        stop_requested.set()
                        break
                    if vx == 0.0 and wz == 0.0:
                        with last_zero_lock:
                            last_zero_reply_ns = end
                    delay = max(0.001, 0.05 - (time.monotonic() - iteration))
                    worker_stop.wait(delay)
            except Exception as exc:
                velocity_failed.set()
                stop_requested.set()
                journal.record({
                    "schema": "g1_pid_event_v1", "event": "velocity_exception",
                    "reason": repr(exc),
                })
            finally:
                try:
                    begin = monotonic_ns()
                    rc, raw = client.send(0.0, 0.0, 0.2)
                    end = monotonic_ns()
                    if rc == 0:
                        with last_zero_lock:
                            last_zero_reply_ns = end
                    journal.record({
                        "schema": "g1_pid_event_v1", "event": "velocity_final_zero",
                        "request_ns": begin, "reply_ns": end, "return_code": rc,
                        "raw_reply": raw, "physical_stop_verified": False,
                    })
                except Exception as exc:
                    velocity_failed.set()
                    journal.record({
                        "schema": "g1_pid_event_v1", "event": "velocity_final_zero_exception",
                        "reason": repr(exc), "physical_stop_verified": False,
                    })

        velocity_thread = threading.Thread(target=velocity_worker, name="pid_velocity", daemon=True)
        velocity_thread.start()
        last_iteration_ns = epoch_ns
        abort_start_s = None
        abort_q = None
        last_stage = None
        sequence = 0
        while True:
            loop_begin_ns = monotonic_ns()
            task_s = (loop_begin_ns - epoch_ns) * 1e-9
            failure = health(streams, interlock, journal, loop_begin_ns)
            if failure:
                raise RuntimeError(failure)
            if velocity_failed.is_set():
                raise RuntimeError("velocity RPC worker failed")
            low, imu = streams.latest()
            if stop_requested.is_set() and abort_start_s is None:
                abort_start_s = task_s
                abort_q = _arm_slots(low)
                journal.record({
                    "schema": "g1_pid_event_v1", "event": "operator_stop_requested",
                    "task_elapsed_s": task_s,
                })
            heading = streams.heading_current()
            yaw0 = heading["reference_rad"] if heading["reference_frozen"] else 0.0
            controller_begin_ns = monotonic_ns()
            if abort_start_s is None:
                frame = plan.sample(
                    task_s, _arm_slots(low), _arm_dq(low), imu.quaternion,
                    yaw0, CONTROL_PERIOD_S,
                )
            else:
                since = task_s - abort_start_s
                frame = {
                    "stage": "operator_stop_settle" if since < 3.0 else "operator_arm_release",
                    "q_rad": abort_q.copy(), "dq_rad_s": np.zeros(13),
                    "kp": profile["kp_array"], "kd": profile["kd_array"],
                    "weight": 1.0 if since < 3.0 else float(np.clip(1.0 - (since - 3.0) / 3.0, 0.0, 1.0)),
                    "terminal": since >= 6.0,
                    "diagnostics": {"pid_active": False, "operator_stop": True},
                }
            controller_end_ns = monotonic_ns()
            if frame["stage"] != last_stage:
                journal.record({
                    "schema": "g1_pid_event_v1", "event": "task_stage",
                    "stage": frame["stage"], "task_elapsed_s": task_s,
                })
                print(f"t={task_s:.3f} {frame['stage']}")
                last_stage = frame["stage"]
            releasing = frame["stage"] in {"arm_ramp_out", "operator_arm_release", "complete"}
            required_zero_s = WALK_STOP_S if abort_start_s is None else abort_start_s
            with last_zero_lock:
                zero_reply = last_zero_reply_ns
            if releasing and zero_reply < epoch_ns + int(required_zero_s * 1e9):
                raise RuntimeError("no successful post-stop zero-speed reply before arm release")
            message = make_message(frame, low)
            write_begin_ns = monotonic_ns()
            ok = bool(publisher.Write(message))
            write_end_ns = monotonic_ns()
            sequence += 1
            journal.record({
                "schema": "g1_hardware_pid_command_v1", "event": "dds_write" if ok else "dds_write_failed",
                "sequence": sequence, "task_elapsed_s": task_s, "stage": frame["stage"],
                "weight": frame["weight"], "q_command_rad": frame["q_rad"].tolist(),
                "dq_command_rad_s": frame["dq_rad_s"].tolist(),
                "kp_command": frame["kp"].tolist(), "kd_command": frame["kd"].tolist(),
                "command_crc": int(message.crc), "mode_pr": int(message.mode_pr),
                "mode_machine": int(message.mode_machine),
                "q_measured_rad": _arm_slots(low).tolist(),
                "dq_measured_rad_s": _arm_dq(low).tolist(),
                "state_received_monotonic_ns": low.received_ns,
                "imu_received_monotonic_ns": imu.received_ns,
                "yaw0_rad": yaw0, "heading_reference_frozen": heading["reference_frozen"],
                "control_nominal_period_ms": CONTROL_PERIOD_S * 1000.0,
                "control_actual_period_ms": (loop_begin_ns - last_iteration_ns) * 1e-6,
                "controller_compute_us": (controller_end_ns - controller_begin_ns) * 1e-3,
                "control_prewrite_us": (write_begin_ns - loop_begin_ns) * 1e-3,
                "write_begin_monotonic_ns": write_begin_ns,
                "write_end_monotonic_ns": write_end_ns,
                "write_duration_us": (write_end_ns - write_begin_ns) * 1e-3,
                **frame["diagnostics"],
            })
            last_iteration_ns = loop_begin_ns
            if not ok:
                raise RuntimeError("arm DDS write failed")
            if frame["terminal"]:
                break
            next_time = loop_begin_ns / 1e9 + CONTROL_PERIOD_S
            # Do not replay missed updates as a back-to-back catch-up burst.
            time.sleep(max(0.001, next_time - time.monotonic()))

        worker_stop.set()
        velocity_thread.join()
        journal.record({
            "schema": "g1_pid_event_v1", "event": "session_end",
            "outcome": "operator_stop_release_completed" if abort_start_s is not None
                else "normal_release_completed",
            "final_weight": 0.0, "physical_stop_verified": False,
        })
        return 130 if abort_start_s is not None else 0
    except Exception as exc:
        worker_stop.set()
        if velocity_thread is not None:
            velocity_thread.join()
        final_attempted = False
        final_written = False
        # Respect the mode/remote latch: do not write another arm frame after it.
        if publisher is not None and not interlock.check(monotonic_ns()):
            low, _ = streams.latest()
            if low is not None:
                final_attempted = True
                try:
                    zero = {
                        "q_rad": np.zeros(13), "dq_rad_s": np.zeros(13),
                        "kp": np.zeros(13), "kd": np.zeros(13), "weight": 0.0,
                    }
                    final_written = bool(publisher.Write(make_message(zero, low)))
                except Exception:
                    pass
        journal.record({
            "schema": "g1_pid_event_v1", "event": "session_fault",
            "reason": str(exc), "final_arm_attempted": final_attempted,
            "final_arm_write": final_written, "physical_stop_verified": False,
        })
        print(f"PID walk stopped: {exc}", file=sys.stderr)
        return 3
    finally:
        worker_stop.set()
        fsm_thread.join()
        try:
            low_subscriber.Close()
            imu_subscriber.Close()
        except Exception:
            pass
        if publisher is not None:
            try:
                publisher.Close()
            except Exception:
                pass
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nic", help="robot-facing wired network interface")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--controller-config", type=Path, default=ROOT / "configs/g1.yaml")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new directory; raw.jsonl and exact inputs are stored here")
    parser.add_argument("--permit-real-output", required=True, choices=[PERMIT])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    journal = None
    try:
        profile = load_profile(args.profile)
        pid_parameters, pid_mapping = load_pid_parameters(args.controller_config, profile)
        journal = Journal(args.output_dir)
        shutil.copy2(args.profile, args.output_dir / "arm_profile.conf")
        shutil.copy2(args.controller_config, args.output_dir / "controller_config.yaml")
        journal.record({
            "schema": "g1_pid_session_v1", "event": "session_start",
            "program": "g1_walk_pid.py", "publisher_created": False,
            "mode_setter_registered": False, "lowcmd_topic_created": False,
            "network_interface": args.nic, "required_fsm": 500,
            "walk_start_s": WALK_START_S, "walk_stop_s": WALK_STOP_S,
            "release_start_s": RELEASE_START_S, "end_s": END_S,
            "primary_metric_window_s": [WALK_START_S, RELEASE_START_S],
            "primary_metric_scope": "walk_start_through_stop_settle_end",
            "forward_speed_m_s": FORWARD_SPEED_M_S,
            "heading_target": "fixed_run_h0_positive_x",
            "pid_parameters": pid_mapping,
            "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            "controller_config_sha256": hashlib.sha256(args.controller_config.read_bytes()).hexdigest(),
        })
        result = run_device(args, profile, pid_parameters, pid_mapping, journal)
        journal.record({
            "schema": "g1_pid_event_v1", "event": "capture_drained",
            "queue_dropped": journal.dropped,
        })
        journal.close()
        print(f"Saved PID raw capture: {args.output_dir / 'raw.jsonl'}; records={journal.written}")
        return result if not journal.failed.is_set() else 3
    except Exception as exc:
        print(f"PID walk refused/failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if journal is not None:
            journal.close()


if __name__ == "__main__":
    raise SystemExit(main())
