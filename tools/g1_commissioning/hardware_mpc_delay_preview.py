"""Offline-only causal command-time state estimate for the torque MPC study.

Requires explicit observation timestamps and an assumed command delay. Neither
is inferred from DDS write duration. Never receives true plant state, payload,
friction or acceleration. All predictions use the nominal controller model and
previously issued commands. Not connected to the field runner.
"""
from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from disturbance_types import DisturbanceInput, DisturbanceHorizon
from hardware_mpc_control import HardwareMpcError, json_values
from hardware_mpc_torque_control import RightArmMeasuredTorqueMpc
from hardware_arm_inverse_dynamics import finite_vector
from endpoint_pose import rotation


class HorizonClock:
    """Sample only the supplied forecast; explicitly extrapolate its last node."""
    def __init__(self, horizon):
        self.horizon = horizon
        self._fields=np.asarray([[d.acc_world,d.omega_world,d.alpha_world] for d in horizon.nodes])
        self._rotations=np.asarray([d.rot_world_body for d in horizon.nodes])
        self._deltas=Rotation.from_matrix(
            self._rotations[1:]@self._rotations[:-1].transpose(0,2,1)).as_rotvec()

    def _indices(self, times):
        values=np.asarray(times,dtype=float)
        if not np.isfinite(values).all() or np.any(values < -1e-10):
            raise ValueError('forecast requested before its observation timestamp')
        values=np.maximum(values,0.)
        index=np.minimum(8,(values/.006).astype(int))
        fraction=np.minimum(1.,(values-index*.006)/.006)
        return values,index,fraction

    def _vectors(self, times):
        _,index,fraction=self._indices(times)
        return ((1-fraction)[:,None,None]*self._fields[index]
                +fraction[:,None,None]*self._fields[index+1])

    def _orientations(self, times):
        values,index,fraction=self._indices(times)
        vector=fraction[:,None]*self._deltas[index]
        result=Rotation.from_rotvec(vector).as_matrix()@self._rotations[index]
        tail=values>=.054
        if tail.any():
            extrapolation=(values[tail]-.054)[:,None]*self._fields[-1,1]
            result[tail]=Rotation.from_rotvec(extrapolation).as_matrix()@self._rotations[-1]
        return result

    def at(self, offset):
        fields=self._vectors([float(offset)])[0]
        return DisturbanceInput(*fields,self._orientations([float(offset)])[0])

    def shifted(self, offset):
        if abs(offset)<1e-12:
            return self.horizon
        node_times=offset+np.arange(10)*.006
        fields=self._vectors(node_times);rotations=self._orientations(node_times)
        nodes=tuple(DisturbanceInput(*f,r) for f,r in zip(fields,rotations))
        interval_times=offset+np.arange(9)*.006
        sample_times=(interval_times[:,None]+np.arange(3)[None,:]*.002).ravel()
        means=self._vectors(sample_times).reshape(9,3,3,3).mean(axis=1)
        rotations=self._orientations(interval_times+.003)
        intervals=tuple(DisturbanceInput(*f,r) for f,r in zip(means,rotations))
        return DisturbanceHorizon(nodes,intervals)


@dataclass(frozen=True)
class IssuedCommand:
    apply_s: float
    ff: np.ndarray
    qref: np.ndarray
    dqref: np.ndarray


class RightArmDelayPreviewMpc(RightArmMeasuredTorqueMpc):
    offline_only = True

    def __init__(self,*args,assumed_command_delay_s=0.,**kwargs):
        self.assumed_command_delay_s=float(assumed_command_delay_s)
        if (not math.isfinite(self.assumed_command_delay_s)
                or not 0<=self.assumed_command_delay_s<=.02):
            raise ValueError('assumed command delay must be within 0..20ms')
        super().__init__(*args,**kwargs)
        self.metadata.update(delay_preview=dict(
            assumed_command_delay_s=self.assumed_command_delay_s,
            state='latest measured q/dq propagated with nominal dynamics and issued-command history',
            forecast='observation-anchored horizon shifted to command time; last node extrapolated beyond 54ms',
            timestamp_requirement='explicit observed sample time; receive time is not automatically sensor time',
            startup='before first issued command applies, assume nominal bias hold reconstructed at observation',
            maximum_prediction_s=.04,hardware_delay_identified=False,
            field_runner_supported=False))

    def reset(self):
        super().reset()
        self._issued=deque()
        self._delay_context=None
        self._last_now=None
        self._last_observed=None

    def set_delay_context(self,now_s,observed_s):
        now_s,observed_s=float(now_s),float(observed_s)
        if (not np.isfinite([now_s,observed_s]).all() or observed_s>now_s+1e-10
                or now_s-observed_s+self.assumed_command_delay_s>.04+1e-10
                or (self._last_now is not None and now_s<self._last_now-1e-10)
                or (self._last_observed is not None and observed_s<self._last_observed-1e-10)):
            raise ValueError('invalid or too-old timestamp for delay preview')
        self._delay_context=(now_s,observed_s)

    def _predict(self,q,dq,clock,now,observed):
        target=now+self.assumed_command_delay_s
        while len(self._issued)>1 and self._issued[1].apply_s<=observed+1e-10:
            self._issued.popleft()
        initial_q,initial_dq=q.copy(),dq.copy()
        initial=None
        if target>observed+1e-12:
            _,initial_bias=self.inverse.linear_dynamics(q,dq,clock.at(0.))
            initial=IssuedCommand(-math.inf,initial_bias,initial_q,initial_dq)
        current=observed;steps=0
        while current<target-1e-12:
            active=initial;next_change=target
            for packet in self._issued:
                if packet.apply_s<=current+1e-10:active=packet
                else:
                    next_change=min(next_change,packet.apply_s)
                    break
            dt=min(.002,target-current,next_change-current)
            if dt<=1e-12:raise HardwareMpcError('invalid command-time integration interval')
            base=clock.at(current-observed)
            mass,bias=self.inverse.linear_dynamics(q,dq,base)
            total=active.ff+self.torque_config['kp']*(active.qref-q)+self.torque_config['kd']*(active.dqref-dq)
            total=np.clip(total,-self.mapper.limit,self.mapper.limit)
            ddq=np.linalg.solve(mass,total-bias)
            q=q+dq*dt+.5*ddq*dt**2;dq=dq+ddq*dt
            current+=dt;steps+=1
            if steps>40 or not np.isfinite(np.r_[q,dq]).all():
                raise HardwareMpcError('delay prediction failed')
        return q,dq,dict(prediction_steps=steps,observation_age_s=now-observed,
            target_minus_observation_s=target-observed,command_time_s=target,
            observed_s=observed,now_s=now,issued_history_size=len(self._issued))

    def step(self,arm_slots,imu_quaternion_wxyz,yaw0_rad,dt):
        start=time.perf_counter_ns()
        slots=finite_vector(arm_slots,13,'arm slots')
        rotation(imu_quaternion_wxyz)
        if not np.isfinite([yaw0_rad,dt]).all() or dt<=0:
            raise ValueError('invalid feedback time/frame')
        if self._delay_context is None or self._horizon is None:
            raise HardwareMpcError('delay preview requires fresh timestamps and forecast')
        now,observed=self._delay_context;self._delay_context=None;self._last_now=now
        self._last_observed=observed
        q_measured=slots[5:10].copy()
        dq_measured=self._measured_dq.copy()
        clock=HorizonClock(self._horizon)
        q,dq,evidence=self._predict(q_measured.copy(),dq_measured.copy(),clock,now,observed)
        horizon=clock.shifted(evidence['target_minus_observation_s'])
        slots=slots.copy();slots[5:10]=q
        self.set_measured_dq(dq)
        self.set_disturbance_horizon(horizon,self._extra)
        quat=Rotation.from_matrix(horizon.nodes[0].rot_world_body).as_quat(scalar_first=True)
        self.last_diagnostics=dict(delay_preview=evidence)
        try:
            qr,dqr,diag=super().step(slots,quat,yaw0_rad,dt)
        finally:
            self.set_measured_dq(dq_measured)
            self.last_diagnostics.update(delay_preview=json_values(evidence),
                raw_observed_q_rad=q_measured.tolist(),raw_observed_dq_rad_s=dq_measured.tolist(),
                mpc_initial_state_semantics='measurement propagated to assumed command application time')
            inner_ms=self.last_diagnostics.get('controller_core_ms')
            self.last_diagnostics.update(inner_controller_core_ms=inner_ms,
                controller_core_ms=(time.perf_counter_ns()-start)*1e-6)
        self._issued.append(IssuedCommand(evidence['command_time_s'],
            np.asarray(diag['tau_ff_candidate_nm']).copy(),qr.copy(),dqr.copy()))
        return qr,dqr,self.last_diagnostics
