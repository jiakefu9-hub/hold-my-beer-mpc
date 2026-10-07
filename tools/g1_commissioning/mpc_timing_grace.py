"""Bounded occasional-lateness tolerance for the learned experiment only.

These are engineering budgets, not certified stopping/latency limits. Never
replace source-feedback age with a newer stamp to make an old result look new.
No stored command, failed QP or mismatched worker reply becomes executable.
"""
import numpy as np

from hardware_arm_inverse_dynamics import finite_vector
from hardware_mpc_field import check_field_packet
from hardware_pid_control import ARM_MOTOR_INDICES


class BoundedTimingGrace:
    soft_ns = 10_000_000
    hard_ns = 20_000_000
    worker_timeout_s = .018  # leaves nominal 2 ms for parent checks/packet work
    source_max_age_ns = 25_000_000  # unchanged field source-data age limit
    latest_max_age_ns = 10_000_000
    max_consecutive_late = 2

    def __init__(self):
        self.consecutive_late = 0
        self.accepted_late = 0

    @classmethod
    def metadata(cls):
        return dict(policy='bounded_occasional_lateness_v1', soft_ms=cls.soft_ns/1e6,
            hard_ms=cls.hard_ns/1e6, worker_timeout_ms=cls.worker_timeout_s*1000,
            source_max_age_ms=cls.source_max_age_ns/1e6,
            latest_max_age_ms=cls.latest_max_age_ns/1e6,
            max_consecutive_late=cls.max_consecutive_late,
            safety_certified=False, failed_solutions_reused=False)

    def check(self, frame, low, imu, begin_ns, now_ns, config, *, latest=None):
        elapsed = now_ns-begin_ns
        source_age = now_ns-min(low.received_ns, imu.received_ns)
        diag = frame.setdefault('diagnostics', {})
        late_solver = bool(diag.get('solver_wall_over_budget', False))
        # A valid solve exceeding its local wall budget is diagnostic only:
        # if the whole cycle is timely, it is not a late control cycle.
        late = elapsed > self.soft_ns
        detail = dict(**self.metadata(),elapsed_ms=elapsed/1e6,source_age_ms=source_age/1e6,
                      solved_qp_wall_over_budget=late_solver,late=late,accepted=False)
        diag['timing_grace'] = detail
        # Preserve source stamps regardless of any fresh snapshot below.
        if elapsed < 0 or min(now_ns-low.received_ns,now_ns-imu.received_ns)<0:
            raise RuntimeError('invalid timing/feedback clock; hand back')
        if source_age > self.source_max_age_ns:
            raise RuntimeError('selected torque feedback older than 25 ms; hand back')
        if elapsed > self.hard_ns:
            raise RuntimeError('torque computation older than hard 20 ms budget; hand back')
        if not late:
            self.consecutive_late = 0
            detail.update(accepted=True,consecutive_late=0,accepted_late_total=self.accepted_late)
            return
        self.consecutive_late += 1
        detail['consecutive_late'] = self.consecutive_late
        if self.consecutive_late > self.max_consecutive_late:
            raise RuntimeError('three consecutive late MPC cycles; hand back')
        if latest is None or any(s is None for s in latest):
            raise RuntimeError('late MPC requires fresh end-of-cycle feedback; hand back')
        fresh_low, fresh_imu = latest
        latest_age = now_ns-min(fresh_low.received_ns,fresh_imu.received_ns)
        detail['latest_feedback_age_ms'] = latest_age/1e6
        if (min(now_ns-fresh_low.received_ns,now_ns-fresh_imu.received_ns)<0
                or latest_age > self.latest_max_age_ns
                or fresh_low.received_ns < low.received_ns or fresh_imu.received_ns < imu.received_ns
                or not fresh_low.crc_valid):
            raise RuntimeError('late MPC has stale/invalid refreshed feedback; hand back')
        if (fresh_low.mode_pr != low.mode_pr or fresh_low.mode_machine != low.mode_machine):
            raise RuntimeError('robot low-level mode changed during late cycle; hand back')
        # Existing packet checks, evaluated at refreshed q/dq rather than the
        # old state that the slow solve started from. This does not alter tau.
        check_field_packet(frame,fresh_low,config)
        detail.update(latest_low_ns=int(fresh_low.received_ns),latest_imu_ns=int(fresh_imu.received_ns))
        if diag.get('mpc_active'):
            q = finite_vector(fresh_low.q[list(ARM_MOTOR_INDICES)],13,'late q')[5:10]
            dq = finite_vector(fresh_low.dq[list(ARM_MOTOR_INDICES)],13,'late dq')[5:10]
            a = finite_vector(diag['raw_mpc_ddq_rad_s2'],5,'planned net acceleration')
            if np.any(abs(dq)>config['max_dq_rad_s']+1e-6) or np.any(abs(a)>config['max_ddq_rad_s2']+1e-6):
                raise RuntimeError('late command exceeds existing velocity/acceleration envelope; hand back')
            # Only the tolerance path requires room for another fresh cycle.
            # Uses the existing acceleration envelope, not raw noisy motor ddq.
            # Not a guarantee of physical acceleration or braking capability.
            dt=max(0.,(now_ns-fresh_low.received_ns)*1e-9)+.006
            travel=dq*dt
            uncertainty=.5*config['max_ddq_rad_s2']*dt**2
            qlo=np.deg2rad(config['q_min_deg']);qhi=np.deg2rad(config['q_max_deg'])
            low_end=q+np.minimum(0.,travel)-uncertainty
            high_end=q+np.maximum(0.,travel)+uncertainty
            detail.update(latest_q_rad=q.tolist(),latest_dq_rad_s=dq.tolist(),
                          next_cycle_room_rad=np.minimum(low_end-qlo,qhi-high_end).tolist())
            if np.any(low_end<qlo) or np.any(high_end>qhi):
                raise RuntimeError('late command has insufficient current joint-boundary room; hand back')
        self.accepted_late += 1
        detail.update(accepted=True,accepted_late_total=self.accepted_late)
