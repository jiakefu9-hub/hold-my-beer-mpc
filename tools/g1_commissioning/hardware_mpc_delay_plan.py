"""Offline full-task command-time prediction; no robot/firmware claims.

The task uses a measured-state controller, preceded by causal nominal state
propagation. History is committed AFTER final packet construction/serialization,
not when a candidate is proposed. Transitions assume weight blends SDK torque
with nominal bias support. This is explicitly not identified firmware behavior.
"""
import math
import time

import numpy as np

from hardware_arm_inverse_dynamics import finite_vector
from hardware_mpc_control import HardwareMpcError
from hardware_mpc_delay_preview import CommandHistory, HorizonClock, IssuedCommand
from hardware_mpc_torque_control import HardwareTorquePreviewPlan
from disturbance_types import DisturbanceInput, DisturbanceHorizon


class IntervalHorizonClock(HorizonClock):
    """Keep learned acc/alpha as 6ms means, not invented instantaneous nodes.

    Piecewise-constant acc/alpha preserve overlap integrals under time shifts.
    Omega and attitude keep the original node interpolation. Past the horizon,
    hold final vectors and integrate attitude with final omega, explicitly.
    """
    def __init__(self, horizon):
        super().__init__(horizon)
        self._interval_fields=np.asarray([[d.acc_world,d.omega_world,d.alpha_world]
                                         for d in horizon.intervals])

    def _vectors(self,times):
        _,index,_=self._indices(times)
        result=super()._vectors(times)
        result[:,0]=self._interval_fields[index,0]
        result[:,2]=self._interval_fields[index,2]
        return result

    def shifted(self,offset):
        if abs(offset)<1e-12:
            return self.horizon
        nodes_t=offset+np.arange(10)*.006
        vectors=self._vectors(nodes_t);rotations=self._orientations(nodes_t)
        nodes=tuple(DisturbanceInput(*v,r) for v,r in zip(vectors,rotations))
        # Exact overlap integration for piecewise constants and linear omega.
        left=nodes_t[:-1,None];right=left+.006
        edges=np.r_[np.arange(10)*.006,max(.054,float(nodes_t[-1]))+.006]
        lo=np.maximum(left,edges[:-1]);hi=np.minimum(right,edges[1:])
        weights=np.maximum(0.,hi-lo)/.006
        midpoints=np.maximum(0.,(lo+hi)*.5)
        fields=self._vectors(midpoints.ravel()).reshape(9,10,3,3)
        means=np.sum(weights[:,:,None,None]*fields,axis=1)
        rotations=self._orientations(nodes_t[:-1]+.003)
        intervals=tuple(DisturbanceInput(*v,r) for v,r in zip(means,rotations))
        return DisturbanceHorizon(nodes,intervals)


class HardwareDelayTorquePreviewPlan(HardwareTorquePreviewPlan):
    """Normal lifecycle with measured/forecast timestamps kept separately."""
    offline_only=True

    def __init__(self,*args,assumed_command_delay_s,**kwargs):
        super().__init__(*args,**kwargs)
        delay=float(assumed_command_delay_s)
        if not math.isfinite(delay) or not 0<=delay<=.020:
            raise ValueError('assumed command delay must be within 0..20ms')
        c=self.controller
        self.history=CommandHistory()
        self.history.inverse=c.inverse
        self.history.mapper=c.mapper
        self.history.torque_config=c.torque_config
        self.history.assumed_command_delay_s=delay
        self.history.reset_history()
        self._context=None
        self._pending=None
        self.committed_packets=0
        c.metadata['delay_lifecycle']=dict(enabled=True,assumed_command_delay_s=delay,
            history='only finalized serialized local packets; not candidates',
            transition_assumption='weight*SDK_total+(1-weight)*nominal_bias_support',
            forecast='overlap-preserving interval acc/alpha; node omega and SO3 attitude',
            timing='sensor/forecast timestamps are explicit offline inputs, not identified DDS latency',
            physical_weight_model_identified=False,field_output_supported=False)

    def set_context(self,now_s,observed_s,forecast_s):
        self._context=None
        if self._pending is not None:
            raise HardwareMpcError('previous candidate not committed as a finalized packet')
        self.history.set_delay_context(now_s,observed_s)
        if (not math.isfinite(forecast_s) or forecast_s>observed_s+1e-10
                or now_s+self.history.assumed_command_delay_s-forecast_s>.04+1e-10):
            raise ValueError('invalid forecast timestamp')
        self._context=(float(now_s),float(observed_s),float(forecast_s))

    def sample(self,task_s,measured_slots,measured_dq,imu_quaternion,yaw0_rad,dt):
        started=time.perf_counter_ns()
        if self._context is None or self.controller._horizon is None:
            raise HardwareMpcError('fresh sample and forecast timestamps required')
        now,observed,forecast=self._context;self._context=None
        self.history._delay_context=None
        self.history._last_now,self.history._last_observed=now,observed
        slots=finite_vector(measured_slots,13,'arm slots').copy()
        speeds=finite_vector(measured_dq,13,'arm velocities').copy()
        raw_q,raw_dq=slots[5:10].copy(),speeds[5:10].copy()
        c=self.controller
        clock=IntervalHorizonClock(c._horizon)
        q,dq,evidence=self.history._predict(raw_q.copy(),raw_dq.copy(),clock,now,observed,forecast)
        slots[5:10],speeds[5:10]=q,dq
        shifted=clock.shifted(evidence['command_time_s']-forecast)
        c.set_disturbance_horizon(shifted,c._extra)
        after_prediction=time.perf_counter_ns()
        try:
            frame=super().sample(task_s,slots,speeds,imu_quaternion,yaw0_rad,dt)
        except Exception:
            c.last_diagnostics.update(delay_preview=evidence,raw_observed_q_rad=raw_q.tolist(),
                raw_observed_dq_rad_s=raw_dq.tolist(),command_time_q_rad=q.tolist(),
                command_time_dq_rad_s=dq.tolist(),torque_output_authorized=False)
            raise
        finally:
            c.set_measured_dq(raw_dq)
        frame['diagnostics'].update(delay_preview=evidence,
            raw_observed_q_rad=raw_q.tolist(),raw_observed_dq_rad_s=raw_dq.tolist(),
            command_time_q_rad=q.tolist(),command_time_dq_rad_s=dq.tolist(),
            state_prediction_ms=(after_prediction-started)*1e-6,
            delay_lifecycle_ms=(time.perf_counter_ns()-started)*1e-6,
            transition_weight_model='unidentified; offline nominal bias blend only',
            torque_evaluation_state='command_time_prediction',
            mpc_initial_state_semantics='measured state propagated to assumed command application time')
        self._pending=frame
        return frame

    def commit_packet(self,frame,packet):
        """Commit the actual IDL values only after local serialization succeeds."""
        if frame is not self._pending:
            raise HardwareMpcError('packet does not correspond to pending lifecycle frame')
        motors=[packet.motor_cmd[i] for i in range(22,27)]
        q=finite_vector([m.q for m in motors],5,'packet q').copy()
        dq=finite_vector([m.dq for m in motors],5,'packet dq').copy()
        tau=finite_vector([m.tau for m in motors],5,'packet torque').copy()
        weight=float(packet.motor_cmd[29].q)
        if not math.isfinite(weight) or not 0<=weight<=1:
            raise ValueError('invalid packet ownership weight')
        expected=frame['diagnostics']
        for actual,wanted in ((q,frame['q_rad'][5:10]),(dq,frame['dq_rad_s'][5:10]),
                              (tau,expected['tau_ff_candidate_nm']),
                              ([m.kp for m in motors],frame['kp'][5:10]),
                              ([m.kd for m in motors],frame['kd'][5:10]),(weight,frame['weight'])):
            if not np.allclose(actual,wanted,rtol=1e-6,atol=1e-7):
                raise ValueError('final packet differs from the checked lifecycle frame')
        when=expected['delay_preview']['command_time_s']
        if self.history._issued and when<=self.history._issued[-1].apply_s:
            raise ValueError('packet application time must increase')
        self.history._issued.append(IssuedCommand(when,tau,q,dq,weight))
        self.committed_packets+=1
        self._pending=None
        frame['diagnostics']['committed_packet_sequence']=self.committed_packets-1
