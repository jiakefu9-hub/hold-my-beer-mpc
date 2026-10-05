import importlib.util
import pathlib
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))


PATH = pathlib.Path(__file__).resolve().parents[1] / "mpc_host.py"
SPEC = importlib.util.spec_from_file_location("mpc_host", PATH)
HOST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOST)


class HostEvidenceTests(unittest.TestCase):
    def scope_mocks(self, affinity):
        stack=ExitStack()
        stack.enter_context(patch.object(HOST.os,'sched_getaffinity',return_value=affinity))
        stack.enter_context(patch.object(HOST.os,'sched_getscheduler',return_value=HOST.os.SCHED_OTHER))
        stack.enter_context(patch.object(HOST.os,'sched_getparam',return_value=HOST.os.sched_param(0)))
        stack.enter_context(patch.object(HOST,'_read',return_value='6-7'))
        schedule=stack.enter_context(patch.object(HOST.os,'sched_setscheduler'))
        pin=stack.enter_context(patch.object(HOST.os,'sched_setaffinity'))
        stack.enter_context(patch.object(HOST,'host_evidence',return_value={'checked':True}))
        self.addCleanup(stack.close)
        return schedule,pin

    def test_workers_avoid_both_smt_threads_and_control_is_explicit(self):
        schedule,pin=self.scope_mocks({0,1,6,7})
        scope=HOST.ControlThreadScope(7,20)
        self.assertEqual(scope.prepare_workers(),[0,1])
        self.assertEqual(schedule.call_args_list[0].args,(0,HOST.os.SCHED_FIFO,HOST.os.sched_param(20)))
        self.assertEqual(schedule.call_args_list[1].args,(0,HOST.os.SCHED_OTHER,HOST.os.sched_param(0)))
        pin.assert_called_with(0,{0,1})
        scope.activate();pin.assert_called_with(0,{7})
        scope.restore();pin.assert_called_with(0,{0,1,6,7})
        schedule.assert_called_with(0,HOST.os.SCHED_OTHER,HOST.os.sched_param(0))

    def test_missing_rt_permission_fails_before_any_affinity_change(self):
        schedule,pin=self.scope_mocks({0,1,6,7})
        schedule.side_effect=[PermissionError('no RT permission'),None]
        scope=HOST.ControlThreadScope(7,20)
        with self.assertRaisesRegex(RuntimeError,'sudo prlimit'):scope.prepare_workers()
        pin.assert_not_called();self.assertFalse(scope.prepared)

    def test_whole_process_pinned_to_one_core_is_rejected(self):
        _,pin=self.scope_mocks({6,7})
        scope=HOST.ControlThreadScope(7)
        with self.assertRaisesRegex(ValueError,'housekeeping'):scope.prepare_workers()
        pin.assert_not_called()

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
