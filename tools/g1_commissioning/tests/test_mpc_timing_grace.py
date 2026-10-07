"""No DDS: short isolated delays are accepted only with current valid state."""
import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from g1_walk_mpc import MpcRuntime, LEARNED_TORQUE_CONFIG
from g1_walk_pid import EXPECTED_TARGET_Q
from hardware_mpc_torque_control import load_torque_config
from hardware_pid_control import ARM_MOTOR_INDICES
from mpc_timing_grace import BoundedTimingGrace


class TimingGraceTests(unittest.TestCase):
    def setUp(self):
        self.config=load_torque_config(LEARNED_TORQUE_CONFIG)
        self.guard=BoundedTimingGrace()

    def state(self,stamp):
        q=np.zeros(35);q[list(ARM_MOTOR_INDICES)]=EXPECTED_TARGET_Q
        return SimpleNamespace(received_ns=stamp,q=q,dq=np.zeros(35),mode_pr=0,mode_machine=4,crc_valid=True)

    def frame(self):
        return dict(q_rad=EXPECTED_TARGET_Q.copy(),dq_rad_s=np.zeros(13),
            kp=np.r_[np.full(5,20.),self.config['kp'],20.,0.,0.],
            kd=np.r_[np.ones(5),self.config['kd'],1.,0.,0.],weight=1.,
            diagnostics=dict(mpc_active=True,raw_mpc_ddq_rad_s2=[0.]*5,tau_ff_candidate_nm=[0.]*5))

    def check(self,duration_ms=14.,start=0,frame=None,fresh=None,source=None):
        frame=self.frame() if frame is None else frame
        now=start+round(duration_ms*1e6)
        source=self.state(start) if source is None else source
        fresh=self.state(now-1_000_000) if fresh is None else fresh
        self.guard.check(frame,source,source,start,now,self.config,latest=(fresh,fresh))
        return frame

    def test_isolated_14ms_is_logged_and_accepted_without_modifying_command(self):
        f=self.frame();before=copy.deepcopy(f)
        self.check(frame=f)
        self.assertTrue(f['diagnostics']['timing_grace']['accepted'])
        self.assertEqual(f['diagnostics']['timing_grace']['source_age_ms'],14.)
        for key in ('q_rad','dq_rad_s','kp','kd','weight'):
            np.testing.assert_array_equal(f[key],before[key])
        self.assertEqual(f['diagnostics']['tau_ff_candidate_nm'],before['diagnostics']['tau_ff_candidate_nm'])

    def test_third_consecutive_late_cycle_refuses_but_normal_cycle_resets_streak(self):
        self.check(start=0);self.check(start=20_000_000)
        with self.assertRaisesRegex(RuntimeError,'three consecutive'):
            self.check(start=40_000_000)
        self.guard=BoundedTimingGrace()
        self.check();self.check(duration_ms=4.,start=20_000_000)
        self.assertEqual(self.guard.consecutive_late,0)
        self.check(start=40_000_000)

    def test_hard_limit_and_source_staleness_cannot_be_hidden_by_fresh_sample(self):
        with self.assertRaisesRegex(RuntimeError,'hard 20 ms'):
            self.check(20.001)
        with self.assertRaisesRegex(RuntimeError,'older than 25 ms'):
            self.check(14.,source=self.state(-12_000_000))

    def test_late_requires_actual_end_of_cycle_feedback(self):
        low=self.state(0)
        with self.assertRaisesRegex(RuntimeError,'fresh end-of-cycle'):
            self.guard.check(self.frame(),low,low,0,14_000_000,self.config)
        with self.assertRaisesRegex(RuntimeError,'stale/invalid'):
            self.check(fresh=self.state(0))

    def test_crc_and_mode_change_are_not_timing_warnings(self):
        s=self.state(13_000_000);s.crc_valid=False
        with self.assertRaisesRegex(RuntimeError,'stale/invalid'):self.check(fresh=s)
        self.guard=BoundedTimingGrace();s.crc_valid=True;s.mode_machine=5
        with self.assertRaisesRegex(RuntimeError,'mode changed'):self.check(fresh=s)

    def test_refresh_recomputes_pd_plus_tau_and_does_not_reuse_old_torque_estimate(self):
        s=self.state(13_000_000);s.q[22]-=.05
        f=self.check(fresh=s)
        self.assertAlmostEqual(f['diagnostics']['field_total_torque_estimate_at_latest_feedback_nm'][0],1.)
        self.guard=BoundedTimingGrace();s=self.state(13_000_000);s.dq[22]=-12.
        with self.assertRaisesRegex(RuntimeError,'total torque envelope'):self.check(fresh=s)

    def test_near_boundary_outward_motion_does_not_gain_extra_delay(self):
        s=self.state(13_000_000);s.q[24]=np.deg2rad(19.9);s.dq[24]=1.
        with self.assertRaisesRegex(RuntimeError,'joint-boundary room'):self.check(fresh=s)

    def test_velocity_and_model_acceleration_limits_unchanged(self):
        s=self.state(13_000_000);s.dq[24]=5.01
        with self.assertRaisesRegex(RuntimeError,'velocity/acceleration'):self.check(fresh=s)
        self.guard=BoundedTimingGrace();f=self.frame();f['diagnostics']['raw_mpc_ddq_rad_s2'][0]=15.01
        with self.assertRaisesRegex(RuntimeError,'velocity/acceleration'):self.check(frame=f)

    def test_normal_cycle_needs_no_extra_velocity_exit_check(self):
        # The extra current-state check is conditional on requesting grace.
        s=self.state(3_000_000);s.dq[24]=5.01
        f=self.check(4.,fresh=s)
        self.assertFalse(f['diagnostics']['timing_grace']['late'])

    def test_valid_slow_qp_does_not_count_as_late_when_whole_cycle_is_timely(self):
        f=self.frame();f['diagnostics']['solver_wall_over_budget']=True
        for i in range(4):
            self.check(6.,start=i*10_000_000,frame=f)
        self.assertEqual(self.guard.consecutive_late,0)
        self.assertTrue(f['diagnostics']['timing_grace']['solved_qp_wall_over_budget'])
        self.check(14.,start=40_000_000,frame=f)
        self.assertEqual(self.guard.consecutive_late,1)

    def test_runtime_records_failed_grace_diagnostic(self):
        r=object.__new__(MpcRuntime);r.field_trial=True;r.timing_grace=self.guard
        r.controller=SimpleNamespace(last_diagnostics={},torque_config=self.config)
        s=self.state(0)
        with self.assertRaisesRegex(RuntimeError,'hard 20 ms'):
            r.check_before_write(self.frame(),s,s,0,21_000_000,latest=(self.state(20_000_000),)*2)
        self.assertFalse(r.controller.last_diagnostics['timing_grace']['accepted'])

    def test_only_successful_solved_qp_can_ignore_wall_budget(self):
        from hardware_mpc_solver import CondensedArmMPCPolicy
        from hardware_mpc_learned import YawFeedbackMpcPolicy
        for kind,flag,expected in ((CondensedArmMPCPolicy,1,-1),(YawFeedbackMpcPolicy,1,1),
                                    (YawFeedbackMpcPolicy,-1,-1)):
            p=kind(np.zeros(5),control_dt=.006,horizon=9,solver_time_limit=.0035)
            n=p.horizon*p.nu;z=np.zeros(n);rows=len(p._ac_dense)
            fake=SimpleNamespace(solve=lambda *args,**kwargs:(z,0.,flag,
                dict(lam=np.zeros(rows),iterations=1,setup_time=0.,solve_time=.0001)))
            condensed=(np.eye(n),z.copy(),np.full(rows,-1.),np.full(rows,1.),
                       np.zeros(p.num_variables),np.eye(p.num_variables))
            with (mock.patch.object(p,'condense',return_value=condensed),mock.patch.object(p,'_daqp',fake),
                  mock.patch.object(p,'_local_actuation_rows',return_value=None)):
                with mock.patch('hardware_mpc_solver.time.perf_counter',side_effect=[1.,1.005]):
                    result,error=p._solve_qp(None,None,np.zeros(p.num_variables),None,None,None)
            self.assertIsNone(error);self.assertEqual(result.info.status_val,expected)


if __name__=='__main__':unittest.main()
