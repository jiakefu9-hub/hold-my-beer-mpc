"""Spawned numerical worker parity, protocol failure and parent-only release."""
import os
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

from g1_walk_mpc import MpcRuntime, FIELD_TORQUE_CONFIG
from g1_walk_pid import EXPECTED_TARGET_Q
from hardware_pid_control import ARM_MOTOR_INDICES
from mpc_compute_process import ProcessMpcRuntime, _store, _load, CAPACITY


class ComputeProcessTests(unittest.TestCase):
    def options(self):
        return dict(predictor_mode='hold_current',stationary=True,field_trial=True,
                    torque_config=FIELD_TORQUE_CONFIG,assumed_command_delay_s=.006)

    def test_spawned_results_match_direct_and_worker_death_does_not_break_release(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        affinity=set(os.sched_getaffinity(0))
        worker=ProcessMpcRuntime(compute_cpu=min(affinity),compute_affinity=affinity,**self.options())
        direct=MpcRuntime(**self.options())
        self.addCleanup(direct.close);self.addCleanup(worker.close)
        profile=dict(target_q_array=EXPECTED_TARGET_Q,kp_array=np.r_[np.full(11,20.),0,0],
                     kd_array=np.r_[np.ones(11),0,0],q_offset_limit_deg_array=np.full(5,5.))
        plans=[r.create_plan(EXPECTED_TARGET_Q,profile) for r in (direct,worker)]
        q=np.zeros(35);q[list(ARM_MOTOR_INDICES)]=EXPECTED_TARGET_Q
        state=SimpleNamespace(q=q,dq=np.zeros(35),mode_pr=0,mode_machine=4,received_ns=0)
        for r in (direct,worker):
            r.observe_low(0,q,state.dq);r.observe_imu(0,[1,0,0,0],[0,0,0],[0,0,9.81]);r.set_epoch(0)
        # Logical time skips the ramp for math parity, but monotonic predictor
        # samples are continuous. No hardware/time-performance claim here.
        for i,task_s in enumerate((3.,4.,4.999,5.,5.006,5.012)):
            stamp=(i+1)*6_000_000;state.received_ns=stamp
            for r in (direct,worker):
                for t in range(stamp-4_000_000,stamp+1,2_000_000):
                    r.observe_low(t,q,state.dq)
                    r.observe_imu(t,[1,0,0,0],[0,0,0],[0,0,9.81])
                r.prepare(stamp,state,state,0.,task_s,heading_frozen=True)
            # Host timing is tested separately; do not let test CPU contention
            # change a numerical-equivalence assertion into a timing assertion.
            original_rpc=worker._rpc
            def relaxed_test_rpc(op,**kw):
                kw['timeout']=2.;return original_rpc(op,**kw)
            with mock.patch.object(worker,'_rpc',side_effect=relaxed_test_rpc):
                frames=[p.sample(task_s,EXPECTED_TARGET_Q,np.zeros(13),[1,0,0,0],0.,.006) for p in plans]
            for key in ('q_rad','dq_rad_s','kp','kd','weight'):
                np.testing.assert_allclose(frames[0][key],frames[1][key],atol=1e-12,rtol=0.)
            np.testing.assert_allclose(frames[0]['diagnostics']['tau_ff_candidate_nm'],
                                       frames[1]['diagnostics']['tau_ff_candidate_nm'],atol=1e-12,rtol=0.)
            for r,f in zip((direct,worker),frames):
                packet=r.make_message(f,state,unitree_hg_msg_dds__LowCmd_,r.create_crc())
                r.accept_packet(f,packet)
        worker._process.terminate();worker._process.join(2.)
        with self.assertRaisesRegex(RuntimeError,'timeout/exited'):worker._rpc('activate',timeout=.02)
        with self.assertRaisesRegex(RuntimeError,'already failed'):worker._rpc('activate')
        self.assertEqual(worker.release_frame(0.)['weight'],1.)
        release=None
        for t in np.arange(.006,3.02,.006):release=worker.release_frame(float(t))
        self.assertEqual(release['weight'],0.)
        self.assertTrue(release['terminal'])

    def test_shared_buffer_bound_and_sequence_mismatch_fail_closed(self):
        import multiprocessing as mp
        context=mp.get_context('spawn');buffer=context.RawArray('B',CAPACITY);size=context.RawValue('I',0)
        with self.assertRaisesRegex(RuntimeError,'exceeds'):_store(buffer,size,'x'*CAPACITY)
        _store(buffer,size,{'id':7,'ok':True,'result':'stale'})
        self.assertEqual(_load(buffer,size)['id'],7)
        runtime=object.__new__(ProcessMpcRuntime)
        runtime._response=buffer;runtime._nresponse=size;runtime._done=context.Semaphore(1)
        runtime._poisoned=False
        with self.assertRaisesRegex(RuntimeError,'sequence mismatch'):runtime._receive(8,.1)
        self.assertTrue(runtime._poisoned)

    def test_human_wait_history_is_bounded_and_does_not_overflow(self):
        from collections import deque
        import threading
        runtime=object.__new__(ProcessMpcRuntime)
        runtime._lock=threading.Lock();runtime._observations={'low':deque(),'imu':deque()}
        runtime._last_low=runtime._last_imu=None;runtime.journal=None
        for stamp in range(0,10_000_000_000,2_000_000):
            runtime.observe_low(stamp,np.zeros(35),np.zeros(35))
            runtime.observe_imu(stamp,[1,0,0,0],[0,0,0],[0,0,9.81])
        self.assertLess(len(runtime._observations['low']),400)
        self.assertLess(len(runtime._observations['imu']),400)
        with self.assertRaisesRegex(ValueError,'rollback'):runtime.observe_low(0,np.zeros(35),np.zeros(35))

    def test_unresponsive_worker_wait_is_bounded(self):
        import multiprocessing as mp
        import time
        runtime=object.__new__(ProcessMpcRuntime)
        runtime._done=mp.get_context('spawn').Semaphore(0)
        runtime._process=SimpleNamespace(is_alive=lambda:True)
        runtime._poisoned=False
        begin=time.monotonic()
        with self.assertRaisesRegex(RuntimeError,'timeout/exited'):runtime._receive(1,.005)
        self.assertLess(time.monotonic()-begin,.2)
        self.assertTrue(runtime._poisoned)


if __name__=='__main__':unittest.main()
