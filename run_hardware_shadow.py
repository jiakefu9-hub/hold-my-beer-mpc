#!/usr/bin/env python3
"""Run Unitree G1 predictor/MPC in read-only hardware shadow mode."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import signal
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


def _inspect_only(
    *,
    client: HardwareStateSource,
    hardware_config: dict,
    controller_config: dict,
    sample_count: int,
    timeout_s: float,
) -> dict:
    contract = HardwareFrameContract.from_mapping(
        hardware_config["hardware_shadow"], require_verified=False
    )
    xml_path = Path(controller_config["xml_path"])
    if not xml_path.is_absolute():
        xml_path = REPO_DIR / xml_path
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    adapter = G1HardwareStateAdapter(model, contract)
    deadline = time.monotonic() + timeout_s
    samples = []
    previous_sample_id = 0
    while len(samples) < sample_count and time.monotonic() < deadline:
        state = client.read_state()
        if state.sample_id <= 0 or state.sample_id == previous_sample_id:
            time.sleep(0.001)
            continue
        previous_sample_id = state.sample_id
        samples.append(adapter.inspect_snapshot(state))
    if not samples:
        raise HardwareStateError("no positive LowState sample received")
    robot_ticks = [int(item["robot_tick"]) for item in samples]
    robot_tick_deltas = [
        (current - previous) & 0xFFFFFFFF
        for previous, current in zip(robot_ticks[:-1], robot_ticks[1:])
    ]
    return {
        "mode": "unitree_lowstate_inspection_only",
        "output_capability": "absent",
        "sample_count": len(samples),
        "first_sample": samples[0],
        "last_sample": samples[-1],
        "observed_mode_pr": sorted({item["mode_pr"] for item in samples}),
        "observed_mode_machine": sorted(
            {item["mode_machine"] for item in samples}
        ),
        "state_age_ms_max": max(item["state_age_ms"] for item in samples),
        "robot_tick_delta_min": (
            min(robot_tick_deltas) if robot_tick_deltas else None
        ),
        "robot_tick_delta_max": (
            max(robot_tick_deltas) if robot_tick_deltas else None
        ),
        "robot_tick_repeat_or_regression_count": sum(
            delta == 0 or delta >= (1 << 31)
            for delta in robot_tick_deltas
        ),
        "quaternion_norm_min": min(
            item["quaternion_norm"] for item in samples
        ),
        "quaternion_norm_max": max(
            item["quaternion_norm"] for item in samples
        ),
    }


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
        "--predictor", choices=("template", "hybrid_residual"), default="template"
    )
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--wait-timeout-s", type=float, default=5.0)
    parser.add_argument("--inspect-state-only", action="store_true")
    parser.add_argument("--inspect-samples", type=int, default=100)
    parser.add_argument("--check-config", action="store_true")
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

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    with UnitreeArmSharedMemoryClient(
        shared_memory,
        wait_timeout_s=args.wait_timeout_s,
        read_only=True,
    ) as client:
        if args.inspect_state_only:
            summary = _inspect_only(
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

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = REPO_DIR / output_root
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
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
