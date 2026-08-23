#!/usr/bin/env python3
"""Run Unitree G1 predictor/MPC in read-only hardware shadow mode."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import mujoco
import numpy as np
import torch
import yaml


_runtime_torch_threads = max(
    1, int(os.environ.get("MPC_CONTROL_NUM_THREADS", "1"))
)
torch.set_num_threads(_runtime_torch_threads)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from right_arm_runtime.hardware_shadow import (
    G1HardwareStateAdapter,
    HardwareContractError,
    HardwareFrameContract,
    HardwareShadowController,
    HardwareStateError,
    HardwareStateSource,
    load_hardware_shadow_config,
)
from right_arm_runtime.unitree_shm import UnitreeArmSharedMemoryClient


REPO_DIR = Path(__file__).resolve().parent
stop_requested = False


def _handle_signal(_signum, _frame):
    global stop_requested
    stop_requested = True


def _load_controller_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise HardwareContractError(f"invalid controller config: {path}")
    return payload


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _inspection_record(state, *, read_monotonic_ns: int) -> dict:
    arrays = {
        "q_rad": np.asarray(state.q, dtype=np.float64),
        "dq_rad_s": np.asarray(state.dq, dtype=np.float64),
        "ddq_rad_s2": np.asarray(state.ddq, dtype=np.float64),
        "tau_est_nm": np.asarray(state.tau_est, dtype=np.float64),
        "motor_temperature_c": np.asarray(
            state.motor_temperature_c, dtype=np.float64
        ),
        "imu_quaternion_wxyz": np.asarray(
            state.imu_quaternion_wxyz, dtype=np.float64
        ),
        "imu_gyroscope_rad_s": np.asarray(
            state.imu_gyroscope, dtype=np.float64
        ),
        "imu_accelerometer_raw_m_s2": np.asarray(
            state.imu_accelerometer, dtype=np.float64
        ),
        "imu_rpy_rad": np.asarray(state.imu_rpy, dtype=np.float64),
    }
    expected_shapes = {
        "q_rad": (35,),
        "dq_rad_s": (35,),
        "ddq_rad_s2": (35,),
        "tau_est_nm": (35,),
        "motor_temperature_c": (35, 2),
        "imu_quaternion_wxyz": (4,),
        "imu_gyroscope_rad_s": (3,),
        "imu_accelerometer_raw_m_s2": (3,),
        "imu_rpy_rad": (3,),
    }
    for name, values in arrays.items():
        if values.shape != expected_shapes[name]:
            raise HardwareStateError(
                f"inspection {name} shape {values.shape} != "
                f"{expected_shapes[name]}"
            )
        if not np.all(np.isfinite(values)):
            raise HardwareStateError(f"inspection {name} contains NaN/Inf")
    quaternion_norm = float(np.linalg.norm(arrays["imu_quaternion_wxyz"]))
    if quaternion_norm <= 1e-9:
        raise HardwareStateError("inspection quaternion has zero norm")
    timestamp_ns = int(state.monotonic_timestamp_ns)
    return {
        "read_monotonic_ns": int(read_monotonic_ns),
        "source_monotonic_timestamp_ns": timestamp_ns,
        "state_age_ms": (int(read_monotonic_ns) - timestamp_ns) * 1e-6,
        "sample_id": int(state.sample_id),
        "robot_tick": int(state.robot_tick),
        "mode_pr": int(state.mode_pr),
        "mode_machine": int(state.mode_machine),
        **{name: values.tolist() for name, values in arrays.items()},
        "quaternion_norm": quaternion_norm,
    }


def _inspect_only(
    *,
    client: HardwareStateSource,
    hardware_config: dict,
    controller_config: dict,
    sample_count: int,
    timeout_s: float,
) -> tuple[dict, list[dict]]:
    contract = HardwareFrameContract.from_mapping(
        hardware_config["hardware_shadow"], require_verified=False
    )
    xml_path = Path(controller_config["xml_path"])
    if not xml_path.is_absolute():
        xml_path = REPO_DIR / xml_path
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    adapter = G1HardwareStateAdapter(model, contract)
    deadline = time.monotonic() + timeout_s
    records: list[dict] = []
    previous_sample_id = 0
    previous_timestamp_ns = 0
    previous_robot_tick: int | None = None
    stale_before_first_count = 0
    while len(records) < sample_count and time.monotonic() < deadline:
        state = client.read_state()
        sample_id = int(state.sample_id)
        if sample_id <= 0 or sample_id == previous_sample_id:
            time.sleep(0.001)
            continue
        if previous_sample_id and sample_id < previous_sample_id:
            raise HardwareStateError("inspection sample_id regressed")
        read_ns = time.monotonic_ns()
        timestamp_ns = int(state.monotonic_timestamp_ns)
        if timestamp_ns <= 0:
            raise HardwareStateError("inspection timestamp must be positive")
        if timestamp_ns > read_ns + contract.future_tolerance_ns:
            raise HardwareStateError("inspection timestamp is in the future")
        age_ns = max(0, read_ns - timestamp_ns)
        if age_ns > contract.state_timeout_ns:
            if not records:
                stale_before_first_count += 1
                previous_sample_id = sample_id
                time.sleep(0.001)
                continue
            raise HardwareStateError(
                f"inspection stream became stale: {age_ns * 1e-6:.3f} ms"
            )
        if previous_timestamp_ns and timestamp_ns <= previous_timestamp_ns:
            raise HardwareStateError("inspection timestamp repeated or regressed")
        robot_tick = int(state.robot_tick)
        if previous_robot_tick is not None:
            tick_delta = (robot_tick - previous_robot_tick) & 0xFFFFFFFF
            if tick_delta == 0 or tick_delta >= (1 << 31):
                raise HardwareStateError(
                    "inspection robot tick repeated or regressed"
                )
        record = _inspection_record(state, read_monotonic_ns=read_ns)
        # Build the existing adapter's compact view as an independent mapping
        # check, without running predictor or MPC.
        record["mapped_right_arm"] = adapter.inspect_snapshot(
            state, now_ns=read_ns
        )["right_arm_q_rad"]
        records.append(record)
        previous_sample_id = sample_id
        previous_timestamp_ns = timestamp_ns
        previous_robot_tick = robot_tick
    if len(records) != sample_count:
        raise HardwareStateError(
            f"inspection incomplete: received {len(records)}/{sample_count} "
            "fresh unique paired samples"
        )

    robot_ticks = [item["robot_tick"] for item in records]
    robot_tick_deltas = [
        (current - previous) & 0xFFFFFFFF
        for previous, current in zip(robot_ticks[:-1], robot_ticks[1:])
    ]
    source_timestamps = [
        item["source_monotonic_timestamp_ns"] for item in records
    ]
    source_dt_ms = [
        (current - previous) * 1e-6
        for previous, current in zip(source_timestamps[:-1], source_timestamps[1:])
    ]
    all_q = np.asarray([item["q_rad"] for item in records])
    all_dq = np.asarray([item["dq_rad_s"] for item in records])
    all_tau = np.asarray([item["tau_est_nm"] for item in records])
    all_temperatures = np.asarray(
        [item["motor_temperature_c"] for item in records]
    )
    summary = {
        "schema": "unitree_hardware_state_inspection_v1",
        "mode": "unitree_lowstate_inspection_only",
        "output_capability": "absent",
        "controller_executed": False,
        "predictor_executed": False,
        "command_publish_count": 0,
        "requested_sample_count": sample_count,
        "sample_count": len(records),
        "complete_requested_sample_count": True,
        "stale_samples_skipped_before_first": stale_before_first_count,
        "declared_robot_model": contract.robot_model,
        "declared_joint_mapping": contract.joint_mapping,
        "official_reference_mode_machine": 4,
        "imu_source_topic": contract.imu_source_topic,
        "first_sample": records[0],
        "last_sample": records[-1],
        "observed_mode_pr": sorted({item["mode_pr"] for item in records}),
        "observed_mode_machine": sorted(
            {item["mode_machine"] for item in records}
        ),
        "mode_machine_matches_reference": sorted(
            {item["mode_machine"] for item in records}
        ) == [4],
        "state_age_ms_min": min(item["state_age_ms"] for item in records),
        "state_age_ms_max": max(item["state_age_ms"] for item in records),
        "source_dt_ms_min": min(source_dt_ms) if source_dt_ms else None,
        "source_dt_ms_mean": (
            float(np.mean(source_dt_ms)) if source_dt_ms else None
        ),
        "source_dt_ms_max": max(source_dt_ms) if source_dt_ms else None,
        "robot_tick_delta_min": (
            min(robot_tick_deltas) if robot_tick_deltas else None
        ),
        "robot_tick_delta_max": (
            max(robot_tick_deltas) if robot_tick_deltas else None
        ),
        "robot_tick_repeat_or_regression_count": 0,
        "quaternion_norm_min": min(
            item["quaternion_norm"] for item in records
        ),
        "quaternion_norm_max": max(
            item["quaternion_norm"] for item in records
        ),
        "q_rad_min": float(np.min(all_q)),
        "q_rad_max": float(np.max(all_q)),
        "dq_abs_max_rad_s": float(np.max(np.abs(all_dq))),
        "tau_est_abs_max_nm": float(np.max(np.abs(all_tau))),
        "motor_case_temperature_c_max": float(
            np.max(all_temperatures[:, :, 0])
        ),
        "motor_winding_temperature_c_max": float(
            np.max(all_temperatures[:, :, 1])
        ),
    }
    return summary, records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _inspection_environment(
    args,
    controller_path: Path,
    hardware_path: Path,
    controller_config: dict,
) -> dict:
    xml_path = Path(controller_config["xml_path"])
    if not xml_path.is_absolute():
        xml_path = REPO_DIR / xml_path
    xml_path = xml_path.resolve()
    result = {
        "repo_head": _git_head(REPO_DIR),
        "controller_config_path": str(controller_path),
        "controller_config_sha256": _sha256(controller_path),
        "hardware_config_path": str(hardware_path),
        "hardware_config_sha256": _sha256(hardware_path),
        "model_xml_path": str(xml_path),
        "model_xml_sha256": _sha256(xml_path),
        "network_interface": args.network_interface,
    }
    if args.unitree_sdk_dir:
        sdk_path = Path(args.unitree_sdk_dir).expanduser().resolve()
        result["unitree_sdk2_path"] = str(sdk_path)
        result["unitree_sdk2_head"] = _git_head(sdk_path)
    if args.state_bridge_binary:
        bridge_path = Path(args.state_bridge_binary).expanduser().resolve()
        if not bridge_path.is_file():
            raise HardwareStateError(
                f"state bridge binary missing: {bridge_path}"
            )
        output_binary = bridge_path.with_name("unitree_arm_adapter_dds")
        if output_binary.exists():
            raise HardwareStateError(
                "output-capable DDS binary present in state-only build directory"
            )
        result["state_bridge_binary"] = str(bridge_path)
        result["state_bridge_sha256"] = _sha256(bridge_path)
        result["output_capable_binary_present_in_build_dir"] = False
    if args.network_interface:
        sysfs = Path("/sys/class/net") / args.network_interface
        if not sysfs.is_dir():
            raise HardwareStateError(
                f"network interface missing: {args.network_interface}"
            )
        result["network_carrier"] = (
            (sysfs / "carrier").read_text(encoding="utf-8").strip()
            if (sysfs / "carrier").exists()
            else None
        )
        result["network_mac"] = (
            (sysfs / "address").read_text(encoding="utf-8").strip()
            if (sysfs / "address").exists()
            else None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Unitree LowState -> predictor/MPC shadow path"
    )
    parser.add_argument(
        "--controller-config", default="configs/g1.yaml"
    )
    parser.add_argument(
        "--hardware-config", default="configs/g1_hardware_shadow.yaml"
    )
    parser.add_argument("--shared-memory", default=None)
    parser.add_argument(
        "--predictor", choices=("template",), default="template"
    )
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--wait-timeout-s", type=float, default=5.0)
    parser.add_argument("--inspect-state-only", action="store_true")
    parser.add_argument("--inspect-samples", type=int, default=100)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--check-config-discovery", action="store_true")
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--state-bridge-binary", default=None)
    parser.add_argument("--unitree-sdk-dir", default=None)
    parser.add_argument(
        "--output-dir", default="evaluation/hardware_shadow"
    )
    args = parser.parse_args()
    if args.duration_s <= 0.0 or args.wait_timeout_s <= 0.0:
        raise ValueError("duration and wait timeout must be positive")
    if args.max_updates < 0 or args.inspect_samples < 1:
        raise ValueError("max-updates must be nonnegative and samples positive")

    controller_path = (REPO_DIR / args.controller_config).resolve()
    hardware_path = (REPO_DIR / args.hardware_config).resolve()
    controller_config = _load_controller_config(controller_path)
    hardware_config = load_hardware_shadow_config(hardware_path)
    shadow_mapping = hardware_config["hardware_shadow"]
    shared_memory = args.shared_memory or shadow_mapping["shared_memory_name"]
    if args.check_config:
        HardwareFrameContract.from_mapping(
            shadow_mapping, require_verified=True
        )
        print("HARDWARE_SHADOW_CONFIG: PASS")
        return
    if args.check_config_discovery:
        HardwareFrameContract.from_mapping(
            shadow_mapping, require_verified=False
        )
        print("HARDWARE_SHADOW_DISCOVERY_CONFIG: PASS")
        return

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    raw_inspection_records = None
    with UnitreeArmSharedMemoryClient(
        shared_memory,
        wait_timeout_s=args.wait_timeout_s,
        read_only=True,
    ) as client:
        if args.inspect_state_only:
            summary, raw_inspection_records = _inspect_only(
                client=client,
                hardware_config=hardware_config,
                controller_config=controller_config,
                sample_count=args.inspect_samples,
                timeout_s=args.duration_s,
            )
        else:
            contract = HardwareFrameContract.from_mapping(
                shadow_mapping, require_verified=True
            )
            results = []
            mpc_success_count = 0
            last_sample_id = 0
            accepted_state = False
            deadline = time.monotonic() + args.duration_s
            with HardwareShadowController(
                repo_dir=REPO_DIR,
                controller_config=controller_config,
                contract=contract,
                predictor_name=args.predictor,
            ) as controller:
                while not stop_requested and time.monotonic() < deadline:
                    read_started = time.perf_counter()
                    state = client.read_state()
                    read_elapsed = time.perf_counter() - read_started
                    if state.sample_id <= 0 or state.sample_id == last_sample_id:
                        time.sleep(0.0005)
                        continue
                    last_sample_id = state.sample_id
                    now_ns = time.monotonic_ns()
                    if (
                        not accepted_state
                        and now_ns - state.monotonic_timestamp_ns
                        > contract.state_timeout_ns
                    ):
                        # A persistent POSIX object can contain the final state
                        # from a prior bridge process. Wait only during initial
                        # acquisition; a stale stream after acceptance is fatal.
                        time.sleep(0.0005)
                        continue
                    result = controller.process(
                        state,
                        now_ns=now_ns,
                        state_read_time_s=read_elapsed,
                    )
                    accepted_state = True
                    if result is None:
                        continue
                    results.append(result)
                    mpc_success_count += int(result.mpc_success)
                    if args.max_updates and len(results) >= args.max_updates:
                        break
                if not results:
                    raise HardwareStateError("no complete 6 ms shadow update")
                summary = controller.summary()
                summary.update(
                    {
                        "source": "unitree_lowstate_shared_memory",
                        "shared_memory_access": "read_only_private_mapping",
                        "state_sample_last": last_sample_id,
                        "mpc_success_count": mpc_success_count,
                        "mpc_success_fraction": mpc_success_count / len(results),
                        "command_build_count": len(results),
                        "command_publish_count": 0,
                        "last_command": asdict(results[-1].command),
                        "last_cycle": {
                            "sample_id": results[-1].source_sample_id,
                            "logical_time_s": results[-1].logical_time_s,
                            "predictor_requested": results[-1].predictor_requested,
                            "predictor_used": results[-1].predictor_used,
                            "predictor_fallback_reason": (
                                results[-1].predictor_fallback_reason
                            ),
                            "mpc_success": results[-1].mpc_success,
                            "diagnostics": results[-1].diagnostics,
                        },
                    }
                )

    if args.inspect_state_only:
        summary["environment"] = _inspection_environment(
            args, controller_path, hardware_path, controller_config
        )
    if args.session_dir:
        run_dir = Path(args.session_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = REPO_DIR / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / "summary.json").exists():
            raise HardwareStateError(
                f"session directory already contains summary.json: {run_dir}"
            )
    else:
        output_root = Path(args.output_dir)
        if not output_root.is_absolute():
            output_root = REPO_DIR / output_root
        run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
    if raw_inspection_records is not None:
        raw_trace_path = run_dir / "raw_state_trace.jsonl"
        with raw_trace_path.open("x", encoding="utf-8") as stream:
            for record in raw_inspection_records:
                stream.write(
                    json.dumps(
                        record, sort_keys=True, default=_json_default
                    ) + "\n"
                )
        summary["raw_state_trace"] = str(raw_trace_path)
        summary["raw_state_trace_sha256"] = _sha256(raw_trace_path)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (HardwareContractError, HardwareStateError) as error:
        print(f"HARDWARE_SHADOW_FAIL_CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2)
