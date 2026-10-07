"""Snapshot/clock ordering is race-free without loosening freshness gates."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import g1_walk_pid as runner


class HealthTest(unittest.TestCase):
    def test_fsm_clock_is_read_under_snapshot_lock(self):
        lock=runner.Interlock()
        lock.observe_fsm(0,500,101,102)
        def now():
            self.assertTrue(lock._lock.locked())
            return 103
        with mock.patch.object(runner,"monotonic_ns",side_effect=now):
            self.assertEqual(lock.check(),"")
        self.assertIn("future",lock.check(100))

    def test_health_takes_snapshot_before_live_clock(self):
        order=[]
        lock=runner.Interlock()
        lock.observe_fsm(0,500,100,100)
        journal=SimpleNamespace(failed=threading.Event(),dropped=0)
        low=runner.LowSnapshot(102,1,1,0,4,np.zeros(35),np.zeros(35),True)
        imu=runner.ImuSnapshot(102,1,np.array([1.,0,0,0]),np.zeros(3),np.zeros(3),np.array([0.,0,9.81]))
        def latest():
            order.append("snapshot")
            return low,imu
        def now():
            order.append("clock")
            return 103
        streams=SimpleNamespace(latest=latest)
        with mock.patch.object(runner,"monotonic_ns",side_effect=now):
            self.assertEqual(runner.health(streams,lock,journal),"")
        self.assertEqual(order[0],"snapshot")
        self.assertIn("future",runner.health(streams,lock,journal,101))
        lock.observe_fsm(0,500,300_000_000,300_000_000)
        self.assertIn("stale",runner.health(streams,lock,journal,300_000_001))

    def test_fsm_freshness_starts_when_valid_slow_reply_is_received(self):
        lock=runner.Interlock()
        request=100
        reply=request+300_000_000
        lock.observe_fsm(0,500,request,reply)
        self.assertEqual(lock.check(reply+runner.FSM_TIMEOUT_NS-1),"")
        self.assertIn("stale",lock.check(reply+runner.FSM_TIMEOUT_NS+1))

    def test_mode_exit_and_remote_still_latch(self):
        for reason in ("fsm","remote"):
            lock=runner.Interlock()
            lock.observe_fsm(0,500,100,100)
            if reason=="fsm":lock.observe_fsm(0,1,101,102)
            else:lock.observe_remote([0,0,32,2])
            lock.observe_fsm(0,500,103,104)
            self.assertTrue(lock.check(105))

    def test_latched_fsm_skips_rpc_freshness_but_keeps_live_faults(self):
        lock=runner.Interlock()
        lock.observe_fsm(0,500,100,101)
        self.assertTrue(lock.latch_valid_fsm())
        self.assertEqual(lock.check(101+10*runner.FSM_TIMEOUT_NS),"")
        lock.observe_remote([0,0,32,2])
        self.assertIn("L2+B",lock.check(102))


if __name__=="__main__":
    unittest.main()
