import importlib.util
import pathlib
import unittest
from unittest.mock import patch


PATH = pathlib.Path(__file__).resolve().parents[1] / "mpc_host.py"
SPEC = importlib.util.spec_from_file_location("mpc_host", PATH)
HOST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOST)


class HostEvidenceTests(unittest.TestCase):
    def test_cpu_checks_current_cpuset(self):
        with patch.object(HOST.os, "sched_getaffinity", return_value={2, 3}):
            self.assertEqual(HOST.select_cpu(), 2)
            self.assertEqual(HOST.select_cpu(3), 3)
            with self.assertRaises(ValueError):
                HOST.select_cpu(7)

    def test_summary_counts_misses_and_complete_work(self):
        result = HOST.summarize_timing([
            {"full_work_ms": 2., "deadline_missed": False, "skipped_slots": 0},
            {"full_work_ms": 8., "deadline_missed": True, "skipped_slots": 1},
        ])
        self.assertEqual(result["deadline_misses"], 1)
        self.assertEqual(result["deadline_miss_fraction"], .5)
        self.assertEqual(result["full_work_ms"]["mean"], 5.)
        self.assertEqual(result["full_work_ms"]["max"], 8.)
        self.assertIsNone(result["reused_imu"])  # missing evidence is not zero

    def test_rt_kernel_is_not_scheduler_policy(self):
        evidence = HOST.host_evidence()
        self.assertIn("preempt_rt_active", evidence)
        self.assertIn("process_scheduler", evidence)
        self.assertFalse(evidence["system_settings_changed"])


if __name__ == "__main__":
    unittest.main()
