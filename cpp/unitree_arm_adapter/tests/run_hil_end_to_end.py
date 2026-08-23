#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from right_arm_runtime.hardware_output_contract import (  # noqa: E402
    FutureCommandMode,
    HardwareControlProposal,
    ValidatedStateIdentity,
    certify_for_offline_fake_sink,
)
from right_arm_runtime.unitree_shm import (  # noqa: E402
    CertifiedHilCommandEnvelope,
    RequestedLifecycle,
    UnitreeArmSharedMemoryClient,
    stable_identity_u64,
)


ARM_SDK_MOTOR_INDICES = (15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 12, 13, 14)


def _write_certified_anchor_zero(client, state_snapshot, session_label):
    arm_q = tuple(state_snapshot.q[index] for index in ARM_SDK_MOTOR_INDICES)
    state_identity = ValidatedStateIdentity(
        session_nonce=session_label,
        sample_id=state_snapshot.sample_id,
        source_timestamp_ns=state_snapshot.monotonic_timestamp_ns,
        validated_timestamp_ns=state_snapshot.validated_timestamp_ns,
        arm_sdk_q=arm_q,
    )
    generated_ns = time.monotonic_ns()
    active_mask = tuple(5 <= index < 10 for index in range(13))
    proposal = HardwareControlProposal(
        session_nonce=session_label,
        proposal_id=1,
        source_sample_id=state_snapshot.sample_id,
        source_timestamp_ns=state_snapshot.monotonic_timestamp_ns,
        task_epoch_id="hil-e2e-task-epoch",
        task_time_ns=0,
        full_task_anchor=0,
        generated_timestamp_ns=generated_ns,
        expires_timestamp_ns=generated_ns + 50_000_000,
        mode=FutureCommandMode.ROBOT_PD_PLUS_FEEDFORWARD,
        arm_weight=0.1,
        active_mask=active_mask,
        q_ref=arm_q,
        dq_ref=(0.0,) * 13,
        ddq_des=(0.0,) * 13,
        kp=tuple(20.0 if active else 0.0 for active in active_mask),
        kd=tuple(1.0 if active else 0.0 for active in active_mask),
        tau=(0.0,) * 13,
        diagnostics={},
    )
    certified = certify_for_offline_fake_sink(
        proposal, state_identity, now_ns=generated_ns
    )
    receipt = client.write_certified_hil_command(
        CertifiedHilCommandEnvelope(
            command=certified,
            producer_sequence=0,
            safety_policy_id=1,
            safety_policy_sha256="a5" * 32,
            requested_lifecycle=RequestedLifecycle.ARMING_PD,
        )
    )
    if receipt.request_output:
        raise AssertionError("offline public writer requested output")
    return certified, receipt


def _run_case(hil_executable, fixture_writer, directory, *, fixture_policy):
    suffix = "fixture" if fixture_policy else "unverified"
    shared_memory_name = f"/g1_arm_hil_e2e_{os.getpid()}_{suffix}"
    jsonl_path = pathlib.Path(directory) / f"receipts_{suffix}.jsonl"
    session_label = f"publisher-absent-hil-e2e-{suffix}"
    session_nonce = stable_identity_u64(session_label)
    writer = subprocess.Popen(
        [str(fixture_writer), shared_memory_name, str(session_nonce)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    hil_process = None
    try:
        with UnitreeArmSharedMemoryClient(
            shared_memory_name, wait_timeout_s=2.0
        ) as client:
            state_snapshot = client.read_state()
            command = [
                str(hil_executable),
                "--shm-name",
                shared_memory_name,
                "--iterations",
                "30",
                "--period-us",
                "2000",
                "--record-jsonl",
                str(jsonl_path),
                "--session-nonce",
                str(session_nonce),
                "--paired-ingress-validated",
            ]
            if fixture_policy:
                command.extend(
                    [
                        "--allow-synthetic-fixture",
                        "--offline-fixture-policy",
                        "--offline-ownership-confirmed",
                        "--safety-policy-id",
                        "1",
                        "--safety-policy-sha256",
                        "a5" * 32,
                    ]
                )
            hil_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 2.0
            if fixture_policy:
                while True:
                    status = client.read_status()
                    if (
                        status.loop_count > 0
                        and status.observed_state_sample_id
                        == state_snapshot.sample_id
                    ):
                        break
                    if time.monotonic() >= deadline:
                        raise AssertionError("HIL did not observe the source state")
                    time.sleep(0.0005)
            certified, write_receipt = _write_certified_anchor_zero(
                client, state_snapshot, session_label
            )
        hil_stdout, hil_stderr = hil_process.communicate(timeout=10)
        writer_stdout, writer_stderr = writer.communicate(timeout=10)
    finally:
        for process in (hil_process, writer):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

    if hil_process.returncode != 0:
        raise AssertionError(f"HIL failed:\n{hil_stdout}\n{hil_stderr}")
    if writer.returncode != 0:
        raise AssertionError(
            f"fixture writer failed:\n{writer_stdout}\n{writer_stderr}"
        )
    receipts = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    count_line = next(
        line
        for line in hil_stdout.splitlines()
        if line.startswith("would_write_command_sink_count=")
    )
    return (
        int(count_line.split("=", 1)[1]),
        receipts,
        hil_stdout,
        certified,
        write_receipt,
    )


def main() -> int:
    if len(sys.argv) != 3:
        raise RuntimeError("expected HIL executable and fixture writer")
    hil_executable = pathlib.Path(sys.argv[1])
    fixture_writer = pathlib.Path(sys.argv[2])
    with tempfile.TemporaryDirectory(prefix="unitree_arm_hil_") as directory:
        count, receipts, stdout, certified, write_receipt = _run_case(
            hil_executable, fixture_writer, directory, fixture_policy=True
        )
        if "hardware_output_performed=false" not in stdout:
            raise AssertionError("HIL did not explicitly deny hardware output")
        if count != 3:
            raise AssertionError(
                f"expected exactly anchor plus two holds, got {count}\n{stdout}"
            )
        if any(
            item["transport_write_performed"]
            or item["hardware_output_performed"]
            for item in receipts
        ):
            raise AssertionError("publisher-absent HIL claimed an external write")
        if not all(item["receipt_sink_write_performed"] for item in receipts):
            raise AssertionError("durable receipt record is incomplete")
        executed = [
            item for item in receipts
            if item["would_write_command_sink_performed"]
        ]
        if len(executed) != count:
            raise AssertionError("stdout and receipt would-write counts disagree")
        if not any(item["reason"] == 1 for item in executed):
            raise AssertionError("first arming proposal was not accepted")
        if sum(item["reason"] == 32 for item in executed) < 2:
            raise AssertionError("0/2/4 ms accepted-hold sequence is incomplete")
        hold_exceeded = [item for item in receipts if item["reason"] == 33]
        if not hold_exceeded:
            raise AssertionError("bounded hold did not report CommandHoldExceeded")
        if any(
            item["would_write_command_sink_performed"]
            or item["guard_reason"] != 33
            for item in hold_exceeded
        ):
            raise AssertionError(
                "post-hold release receipt lost its fail-closed root cause"
            )
        first = next(item for item in executed if item["reason"] == 1)
        expected_scalars = {
            "command_id": write_receipt.command_id,
            "producer_sequence": write_receipt.producer_sequence,
            "session_nonce": write_receipt.session_nonce,
            "source_sample_id": write_receipt.source_sample_id,
            "source_timestamp_ns": write_receipt.source_timestamp_ns,
            "task_epoch_id": write_receipt.task_epoch_id,
            "task_time_ns": write_receipt.task_time_ns,
            "full_task_anchor": write_receipt.full_task_anchor,
            "safety_policy_id": write_receipt.safety_policy_id,
            "safety_policy_sha256": write_receipt.safety_policy_sha256,
            "requested_active_mask": write_receipt.active_mask_bits,
        }
        for field, expected in expected_scalars.items():
            if first[field] != expected:
                raise AssertionError(
                    f"cross-language receipt mismatch for {field}: "
                    f"{first[field]} != {expected}"
                )
        if first["requested_arm_weight"] != certified.arm_weight:
            raise AssertionError("requested arm weight changed across ABI")
        if first["executed_active_mask"] != write_receipt.active_mask_bits:
            raise AssertionError("executed active mask changed across ABI")
        if first["executed_arm_weight"] != certified.arm_weight:
            raise AssertionError("executed arm weight changed across ABI")
        expected_vectors = {
            "q": certified.q_ref,
            "dq": certified.dq_ref,
            "ddq_des": certified.ddq_des,
            "kp": certified.kp,
            "kd": certified.kd,
            "tau": certified.tau,
        }
        for field, expected in expected_vectors.items():
            if tuple(first[field]) != tuple(expected):
                raise AssertionError(
                    f"cross-language 13-slot mismatch for {field}"
                )
        for item in executed:
            for field in ("q", "dq", "ddq_des", "kp", "kd", "tau"):
                if len(item[field]) != 13:
                    raise AssertionError(f"{field} does not contain 13 slots")
            if item["source_sample_id"] == 0:
                raise AssertionError("would-write receipt lost source identity")

        unverified_count, unverified, _, _, _ = _run_case(
            hil_executable, fixture_writer, directory, fixture_policy=False
        )
        if unverified_count != 0:
            raise AssertionError("unverified policy reached command sink")
        if any(item["would_write_command_sink_performed"] for item in unverified):
            raise AssertionError("unverified receipt claimed a command sink write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
