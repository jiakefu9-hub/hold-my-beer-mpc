#!/usr/bin/env python3
"""Stationary closed-loop right-arm excitation for preliminary identification.

Default operation is an SDK-free local preflight.  Real execution reuses the
field-tested Arm SDK runner, FSM/remote/state interlocks, command evidence and
three-second hand-back.  It never switches robot mode and never publishes
rt/lowcmd or a locomotion command other than zero velocity.

The experiment estimates an *effective* local plant.  ``tau_est`` is robot-
reported, not an independent shaft-torque reference, so the result must not be
presented as an absolute torque calibration or automatically written into MPC.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import shutil
from types import SimpleNamespace

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import yaml

from g1_walk_pid import (ARM_MOTOR_INDICES, CONTROL_PERIOD_S, EXPECTED_TARGET_Q,
                         Journal, ROOT, make_arm_message, load_profile, run_device)
from hardware_arm_inverse_dynamics import finite_vector
from hardware_mpc_field import TorqueHandback
from mpc_host import ControlThreadScope, host_evidence, select_cpu


CONFIG = ROOT / "configs/g1_arm_identification.yaml"
PERMIT = "ARM_ID_STATIONARY"
SCHEMA = "g1_arm_closed_loop_identification_v1"
RIGHT = slice(5, 10)


def load_identification_config(path=CONFIG):
    result = yaml.safe_load(Path(path).read_text())
    if not isinstance(result, dict) or result.get("schema") != SCHEMA:
        raise ValueError(f"identification config schema must be {SCHEMA}")
    vector5 = ("tau_peak_nm", "right_kp", "right_kd", "tau_abs_nm",
               "tau_ff_abs_nm", "q_min_deg", "q_max_deg",
               "identification_q_offset_deg", "phase_rad")
    for key in vector5:
        result[key] = finite_vector(result[key], 5, key)
    result["component_weights"] = finite_vector(
        result["component_weights"], 3, "component_weights")
    result["frequencies_hz"] = np.asarray(result["frequencies_hz"], dtype=float)
    if result["frequencies_hz"].shape != (5, 3) or not np.isfinite(result["frequencies_hz"]).all():
        raise ValueError("frequencies_hz must be a finite 5x3 matrix")
    scalar = ("excitation_start_s", "excitation_stop_s", "fade_s",
              "identification_max_dq_rad_s")
    for key in scalar:
        result[key] = float(result[key])
    duration = result["excitation_stop_s"] - result["excitation_start_s"]
    if (result["excitation_start_s"] != 5.0 or result["excitation_stop_s"] != 15.0
            or not 0 < result["fade_s"] <= duration/4
            or result["identification_max_dq_rad_s"] <= 0):
        raise ValueError("identification timing/velocity envelope is invalid")
    if (np.any(result["tau_peak_nm"] <= 0)
            or np.any(result["tau_peak_nm"] >= .25*result["tau_abs_nm"])
            or np.any(result["tau_peak_nm"] >= .25*result["tau_ff_abs_nm"])):
        raise ValueError("screening excitation must stay below 25% of field torque envelopes")
    if (np.any(result["component_weights"] <= 0)
            or len(np.unique(result["frequencies_hz"])) != 15
            or np.any(result["frequencies_hz"] < .3)
            or np.any(result["frequencies_hz"] > 2.5)):
        raise ValueError("identification frequencies/weights are invalid")
    if (np.any(result["q_min_deg"] >= result["q_max_deg"])
            or np.any(result["identification_q_offset_deg"] <= 0)
            or np.any(result["identification_q_offset_deg"] > 10)):
        raise ValueError("identification position envelope is invalid")
    return result


def smooth_envelope(elapsed, duration, fade):
    edge = min(max(float(elapsed), 0.0), max(duration-float(elapsed), 0.0))
    ratio = np.clip(edge/fade, 0.0, 1.0)
    return float(.5-.5*math.cos(math.pi*ratio))


class IdentificationPlan:
    def __init__(self, initial, target, profile_kp, profile_kd, config):
        self.initial = finite_vector(initial, 13, "initial arm state").copy()
        self.target = finite_vector(target, 13, "target arm state").copy()
        self.kp = finite_vector(profile_kp, 13, "profile kp").copy()
        self.kd = finite_vector(profile_kd, 13, "profile kd").copy()
        self.kp[RIGHT], self.kd[RIGHT] = config["right_kp"], config["right_kd"]
        self.config = config
        duration = config["excitation_stop_s"]-config["excitation_start_s"]
        grid = np.linspace(0.,duration,20001,endpoint=False)
        weights = config["component_weights"]/np.sum(config["component_weights"])
        angles = (2*math.pi*config["frequencies_hz"][:,:,None]*grid[None,None,:]
                  +config["phase_rad"][:,None,None])
        raw = np.sum(weights[None,:,None]*np.sin(angles),axis=1)
        envelope = np.asarray([smooth_envelope(t,duration,config["fade_s"]) for t in grid])
        self._normalization = np.max(abs(raw*envelope[None,:]),axis=1)
        if np.any(self._normalization < .1):
            raise ValueError("identification waveform has insufficient normalized excitation")

    def excitation(self, task_s):
        c = self.config
        elapsed = float(task_s)-c["excitation_start_s"]
        duration = c["excitation_stop_s"]-c["excitation_start_s"]
        if not 0 <= elapsed < duration:
            return np.zeros(5), 0.0
        envelope = smooth_envelope(elapsed, duration, c["fade_s"])
        weights = c["component_weights"] / np.sum(c["component_weights"])
        angles = 2*math.pi*c["frequencies_hz"]*elapsed+c["phase_rad"][:, None]
        signal = np.sum(weights[None, :]*np.sin(angles), axis=1)
        return envelope*c["tau_peak_nm"]*signal/self._normalization, envelope

    def sample(self, task_s, measured_slots, measured_dq, imu_quaternion, yaw0_rad, dt):
        del measured_slots, measured_dq, imu_quaternion, yaw0_rad, dt
        task_s = float(task_s)
        q = self.target.copy()
        weight = 1.0
        stage = "identification_settle"
        if task_s < 3.0:
            ratio = np.clip(task_s/3.0, 0.0, 1.0)
            q = self.initial+ratio*(self.target-self.initial)
            weight = float(ratio)
            stage = "arm_ramp_in"
        tau, envelope = self.excitation(task_s)
        active = self.config["excitation_start_s"] <= task_s < self.config["excitation_stop_s"]
        if active:
            stage = "arm_identification"
        elif task_s >= self.config["excitation_stop_s"]:
            stage = "identification_post_hold"
        return dict(stage=stage, q_rad=q, dq_rad_s=np.zeros(13),
            kp=self.kp.copy(), kd=self.kd.copy(), weight=weight, terminal=False,
            diagnostics=dict(controller_kind="arm_closed_loop_identification_v1",
                identification_active=active, mpc_active=active,
                excitation_envelope=envelope,
                tau_ff_candidate_nm=tau.tolist(),
                expected_kp=self.kp[RIGHT].tolist(), expected_kd=self.kd[RIGHT].tolist()))


class IdentificationRuntime:
    actuation = "arm_closed_loop_identification_v1"
    field_trial = True
    stationary = True
    controller_label = "ARM-ID"

    def __init__(self, profile, config, host_scope=None):
        self.profile, self.config, self.host_scope = profile, config, host_scope
        self.handback, self._plan = TorqueHandback(), None
        self.journal = None
        self.controller = SimpleNamespace(metadata=dict(
            variant=self.actuation, stationary=True, locomotion_command="zero_only",
            excitation="deterministic faded 15-tone multisine feedforward torque",
            torque_feedback="tau_est is robot estimate, not independent calibration",
            automatic_model_update=False), last_diagnostics={})
        self._gc_was_enabled = None

    def validate_field_entry(self):
        if self.config.get("schema") != SCHEMA:
            raise ValueError("explicit reviewed identification config required")

    def create_plan(self, initial, profile):
        self._plan = IdentificationPlan(initial, profile["target_q_array"],
                                        profile["kp_array"], profile["kd_array"], self.config)
        return self._plan

    def prepare_startup(self):
        self._gc_was_enabled = gc.isenabled()
        gc.collect(); gc.disable()

    def prepare(self, *args, **kwargs):
        return None

    def set_epoch(self, epoch_ns):
        self.epoch_ns = int(epoch_ns)

    def observe_low(self, *args):
        return None

    def observe_imu(self, *args):
        return None

    def enter_control_thread(self, cpu):
        if self.host_scope is None:
            raise RuntimeError("identification host scope not prepared")
        return self.host_scope.activate()

    def release_frame(self, elapsed_s):
        return self.handback.normal_release(elapsed_s)

    def accept_packet(self, frame, packet):
        self.handback.accept(packet)

    def _check(self, frame, state):
        diag = frame.setdefault("diagnostics", {})
        ff = finite_vector(diag["tau_ff_candidate_nm"], 5, "identification feedforward")
        qcmd = finite_vector(frame["q_rad"], 13, "command q")[RIGHT]
        dqcmd = finite_vector(frame["dq_rad_s"], 13, "command dq")[RIGHT]
        kp = finite_vector(frame["kp"], 13, "command kp")[RIGHT]
        kd = finite_vector(frame["kd"], 13, "command kd")[RIGHT]
        slots = finite_vector(state.q[list(ARM_MOTOR_INDICES)], 13, "feedback q")
        speeds = finite_vector(state.dq[list(ARM_MOTOR_INDICES)], 13, "feedback dq")
        q, dq = slots[RIGHT], speeds[RIGHT]
        total = ff+kp*(qcmd-q)+kd*(dqcmd-dq)
        diag["field_total_torque_estimate_at_latest_feedback_nm"] = total.tolist()
        self.controller.last_diagnostics = diag
        if frame["weight"] > 0 and np.any(abs(ff) > self.config["tau_ff_abs_nm"]+1e-6):
            raise RuntimeError("identification feedforward envelope exceeded; hand back")
        if diag.get("identification_active"):
            if np.any(abs(total) > self.config["tau_abs_nm"]+1e-6):
                raise RuntimeError("identification total torque envelope exceeded; hand back")
            if (np.any(abs(q-self.profile["target_q_array"][RIGHT]) >
                       np.deg2rad(self.config["identification_q_offset_deg"]))
                    or np.any(abs(dq) > self.config["identification_max_dq_rad_s"]+1e-6)):
                raise RuntimeError("identification state envelope exceeded; hand back")

    def make_message(self, frame, state, constructor, crc):
        diag = frame.setdefault("diagnostics", {})
        if diag.get("controller_kind") != self.actuation:
            self.handback.apply(frame)
            diag = frame["diagnostics"]
        self._check(frame, state)
        message = make_arm_message(frame, state, constructor, crc, finalize_crc=False)
        ff = finite_vector(diag["tau_ff_candidate_nm"], 5, "packet identification torque")
        for index, value in zip(range(22, 27), ff):
            message.motor_cmd[index].tau = float(value) if frame["weight"] > 0 else 0.0
        message.crc = crc.Crc(message)
        return message

    def check_before_write(self, frame, low, imu, begin_ns, now_ns, **kwargs):
        del imu, begin_ns, now_ns, kwargs
        self._check(frame, low)

    def close(self):
        if self.host_scope is not None:
            self.host_scope.restore()
            self.host_scope = None
        if self._gc_was_enabled:
            gc.enable()


def preflight(config_path=CONFIG, cpu=None):
    cpu = select_cpu(cpu)
    config = load_identification_config(config_path)
    profile = dict(target_q_array=EXPECTED_TARGET_Q.copy(),
                   kp_array=np.r_[np.full(11, 20.0), 0.0, 0.0],
                   kd_array=np.r_[np.ones(11), 0.0, 0.0])
    plan = IdentificationPlan(EXPECTED_TARGET_Q, EXPECTED_TARGET_Q,
                              profile["kp_array"], profile["kd_array"], config)
    samples = [plan.sample(t, EXPECTED_TARGET_Q, np.zeros(13), [1,0,0,0], 0., .006)
               for t in np.arange(0., 18., .006)]
    tau = np.asarray([row["diagnostics"]["tau_ff_candidate_nm"] for row in samples])
    return dict(schema="g1_arm_identification_preflight_v1", passed=True,
        dds_initialized=False, publisher_created=False, robot_connected=False,
        control_period_ms=CONTROL_PERIOD_S*1000,
        active_samples=int(sum(row["diagnostics"]["identification_active"] for row in samples)),
        max_abs_tau_ff_nm=np.max(abs(tau),axis=0).tolist(),
        configured_peak_nm=config["tau_peak_nm"].tolist(),
        frequencies_hz=config["frequencies_hz"].tolist(), host=host_evidence(cpu),
        limitations=["signal/configuration preflight only", "no physical response or identifiability claim",
                     "tau_est is not an independent torque calibration"])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nic", nargs="?")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cpu", type=int, default=7)
    parser.add_argument("--rt-priority", type=int, default=20)
    parser.add_argument("--pid-6ms-validated", action="store_true")
    parser.add_argument("--permit-real-output", choices=(PERMIT,))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    runtime = journal = scope = None
    try:
        if args.preflight and args.execute:
            raise ValueError("--preflight and --execute are mutually exclusive")
        if not args.execute:
            print(json.dumps(preflight(args.config,args.cpu),indent=2)); return 0
        if not (args.nic and args.profile and args.output_dir and args.pid_6ms_validated
                and args.permit_real_output == PERMIT):
            raise ValueError("execution requires NIC, profile, new output directory, PID validation and exact permit")
        profile = load_profile(args.profile, "pid")
        config = load_identification_config(args.config)
        from field_performance import prepare as prepare_field_performance
        performance = prepare_field_performance(args.cpu, rt_priority=args.rt_priority)
        scope = ControlThreadScope(args.cpu,args.rt_priority); scope.prepare_workers()
        runtime = IdentificationRuntime(profile,config,scope); scope = None
        journal = Journal(args.output_dir); runtime.journal = journal
        shutil.copy2(args.profile,args.output_dir/"arm_profile.conf")
        shutil.copy2(args.config,args.output_dir/"identification_config.yaml")
        sources=[Path(__file__),Path(__file__).with_name("g1_walk_pid.py"),
                 Path(__file__).with_name("hardware_mpc_field.py"),
                 Path(__file__).with_name("arm_execution_record.py")]
        journal.record(dict(schema="g1_arm_identification_session_v1",event="session_start",
            program=Path(__file__).name,network_interface=args.nic,required_fsm=500,
            publisher_created=False,mode_setter_registered=False,lowcmd_topic_created=False,
            stationary=True,forward_speed_m_s=0.0,excitation_start_s=5.0,excitation_stop_s=15.0,
            control_nominal_period_ms=6.0,performance_setup=performance,
            excitation=dict(tau_peak_nm=config["tau_peak_nm"],frequencies_hz=config["frequencies_hz"],
                            component_weights=config["component_weights"],fade_s=config["fade_s"]),
            automatic_model_update=False,pid_6ms_validation="operator_attestation",
            control_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sources},
            profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest()))
        result=run_device(args,profile,None,{},journal,runtime=runtime)
        journal.record(dict(schema="g1_arm_identification_event_v1",event="capture_drained",
                            queue_dropped=journal.dropped))
        journal.close()
        if journal.failed.is_set():
            print(f"Identification capture incomplete: {journal.failure_reason}"); return 3
        print(f"Saved identification capture: {args.output_dir/'raw.jsonl'}")
        return result
    except Exception as exc:
        if journal is not None:
            journal.record(dict(schema="g1_arm_identification_event_v1",event="local_failure",reason=str(exc)))
        print(f"Arm identification refused/failed: {exc}")
        return 1
    finally:
        if journal is not None: journal.close()
        if runtime is not None: runtime.close()
        elif scope is not None: scope.restore()


if __name__ == "__main__":
    raise SystemExit(main())
