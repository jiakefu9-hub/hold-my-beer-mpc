"""Python/C++ Unitree 共享内存 protocol v2 一致性测试。"""

from __future__ import annotations

import mmap
import os
from pathlib import Path
import subprocess
import time
import unittest

from right_arm_runtime import unitree_shm as protocol


DEFAULT_DRY_RUN = Path(
    "/tmp/hold-my-beer-mpc-unitree-arm-adapter-build/"
    "unitree_arm_adapter_dry_run"
)


def _dry_run_executable() -> Path:
    configured = os.environ.get("UNITREE_ARM_DRY_RUN")
    return Path(configured).expanduser() if configured else DEFAULT_DRY_RUN


def _parse_layout(text: str) -> dict[str, int]:
    result = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        result[name.strip()] = int(value.strip())
    return result


class LayoutTest(unittest.TestCase):
    def test_python_layout_matches_protocol_v2_constants(self):
        self.assertEqual(
            protocol.python_layout_report(), protocol._EXPECTED_LAYOUT
        )

    def test_all_payload_offsets_match_protocol_v2(self):
        payloads = {
            "command": protocol._ArmCommandPayload,
            "state": protocol._RobotStatePayload,
            "status": protocol._AdapterStatusPayload,
        }
        actual = {}
        for prefix, structure in payloads.items():
            for field_name, _ in structure._fields_:
                actual[f"{prefix}.{field_name}"] = getattr(
                    structure, field_name
                ).offset
        self.assertEqual(actual, protocol._EXPECTED_FIELD_OFFSETS)

    @unittest.skipUnless(
        _dry_run_executable().is_file(), "C++ dry-run 尚未构建"
    )
    def test_cpp_print_layout_matches_python(self):
        completed = subprocess.run(
            [str(_dry_run_executable()), "--print-layout"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            _parse_layout(completed.stdout), protocol.python_layout_report()
        )


@unittest.skipUnless(_dry_run_executable().is_file(), "C++ dry-run 尚未构建")
class DryRunInteropTest(unittest.TestCase):
    def setUp(self):
        self.name = f"/g1_arm_mpc_pytest_{os.getpid()}_{id(self)}"
        self.path = Path("/dev/shm") / self.name[1:]

    def tearDown(self):
        # 【非核心】只清理由本测试进程唯一命名的临时 POSIX shm。
        self.path.unlink(missing_ok=True)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                str(_dry_run_executable()),
                "--shm-name",
                self.name,
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )

    def _create_empty_layout(self):
        self._run("--reset-shm", "--iterations", "1")

    def _set_header(self, field_name: str, value: int):
        descriptor = os.open(self.path, os.O_RDWR)
        try:
            mapping = mmap.mmap(
                descriptor,
                protocol._EXPECTED_LAYOUT["layout_size"],
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(descriptor)
        layout = protocol._SharedMemoryLayout.from_buffer(mapping)
        setattr(layout, field_name, value)
        del layout
        mapping.close()

    def test_open_rejects_bad_magic_version_and_layout_size(self):
        original = {
            "magic": protocol.PROTOCOL_MAGIC,
            "version": protocol.PROTOCOL_VERSION,
            "layout_size": protocol._EXPECTED_LAYOUT["layout_size"],
        }
        bad = {
            "magic": protocol.PROTOCOL_MAGIC ^ 1,
            "version": protocol.PROTOCOL_VERSION + 1,
            "layout_size": protocol._EXPECTED_LAYOUT["layout_size"] - 64,
        }
        self._create_empty_layout()
        for field_name in original:
            with self.subTest(field=field_name):
                self._set_header(field_name, bad[field_name])
                with self.assertRaises(protocol.LayoutMismatchError):
                    protocol.UnitreeArmSharedMemoryClient(self.name)
                self._set_header(field_name, original[field_name])

    def test_default_output_off_and_pd_modes_do_not_overlap(self):
        self._create_empty_layout()
        zeros = [0.0] * protocol.ARM_SDK_JOINT_COUNT
        kp = [20.0] * protocol.ARM_SDK_JOINT_COUNT
        kd = [1.0] * protocol.ARM_SDK_JOINT_COUNT
        tau_ff = [float(index) for index in range(13)]
        with protocol.UnitreeArmSharedMemoryClient(self.name) as client:
            before = time.monotonic_ns()
            receipt = client.write_robot_pd_plus_feedforward(
                arm_weight=0.2,
                q_ref=zeros,
                dq_ref=zeros,
                kp=kp,
                kd=kd,
                tau_ff=tau_ff,
                command_id=101,
            )
            after = time.monotonic_ns()
            payload = client._read_slot(
                client._require_layout().command,
                protocol._ArmCommandPayload,
                100,
            )
            self.assertLessEqual(before, receipt.monotonic_timestamp_ns)
            self.assertLessEqual(receipt.monotonic_timestamp_ns, after)
            self.assertFalse(receipt.request_output)
            self.assertEqual(payload.flags, 0)
            self.assertEqual(
                payload.mode,
                protocol.CommandMode.ROBOT_PD_PLUS_FEEDFORWARD,
            )
            self.assertEqual(tuple(payload.kp), tuple(kp))
            self.assertEqual(tuple(payload.tau), tuple(tau_ff))

            client.write_direct_torque(
                arm_weight=0.2,
                tau_cmd=tau_ff,
                command_id=102,
                request_output=True,
            )
            direct = client._read_slot(
                client._require_layout().command,
                protocol._ArmCommandPayload,
                100,
            )
            self.assertEqual(direct.mode, protocol.CommandMode.DIRECT_TORQUE)
            self.assertEqual(
                direct.flags, protocol.CommandFlags.REQUEST_OUTPUT
            )
            self.assertEqual(tuple(direct.kp), tuple(zeros))
            self.assertEqual(tuple(direct.kd), tuple(zeros))
            self.assertEqual(tuple(direct.tau), tuple(tau_ff))

    def test_read_only_client_cannot_write_command_slot(self):
        self._create_empty_layout()
        zeros = [0.0] * protocol.ARM_SDK_JOINT_COUNT
        with protocol.UnitreeArmSharedMemoryClient(
            self.name, read_only=True
        ) as client:
            client.read_state()
            with self.assertRaises(PermissionError):
                client.write_direct_torque(
                    arm_weight=0.0,
                    tau_cmd=zeros,
                )

    def test_cpp_to_python_state_and_python_to_cpp_command(self):
        # 第一阶段只有 C++ 写 synthetic state/status，Python 只读。
        self._run(
            "--reset-shm",
            "--synthetic-input",
            "--iterations",
            "4",
            "--period-us",
            "500",
        )
        zeros = [0.0] * protocol.ARM_SDK_JOINT_COUNT
        with protocol.UnitreeArmSharedMemoryClient(self.name) as client:
            state = client.read_state()
            status = client.read_status()
            self.assertEqual(state.sample_id, 4)
            self.assertEqual(state.robot_tick, 4)
            self.assertEqual(
                state.imu_quaternion_wxyz, (1.0, 0.0, 0.0, 0.0)
            )
            self.assertEqual(status.loop_count, 4)
            self.assertTrue(
                status.flags
                & protocol.AdapterStatusFlags.STATE_SNAPSHOT_VALID
            )

            # 第二阶段 Python 成为 command 槽唯一写者；C++ 不再 synthetic 覆盖。
            client.write_direct_torque(
                arm_weight=0.1,
                tau_cmd=zeros,
                command_id=424242,
            )
            self._run("--iterations", "1", "--period-us", "500")
            observed = client.read_status()
            self.assertEqual(observed.command_id, 424242)
            self.assertTrue(
                observed.flags
                & protocol.AdapterStatusFlags.COMMAND_SNAPSHOT_VALID
            )


if __name__ == "__main__":
    unittest.main()
