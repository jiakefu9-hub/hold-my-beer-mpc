"""Two-arm world-Z attitude feedback, pure kinematics, no SDK or network.

Outer resolved-rate feedback integrates limited joint references; the existing
Arm SDK PD executes them. Positive world Z is the measured IMU navigation up.
Only the endpoint axis is controlled: position and rotation about Z are free.
"""
from pathlib import Path

import mujoco
import numpy as np
import yaml

from endpoint_pose import EndpointModel, JOINTS, rotation
from hardware_pid_control import (CONTROL_PERIOD_S, ENTRY_END_S,
                                  HardwarePidPlan, ARM_MOTOR_INDICES)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/hardware_dual_upright.yaml'
MODEL = ROOT/'configs/g1_payload75.yaml'


def load_config(path=CONFIG):
    c = yaml.safe_load(Path(path).read_text())
    expected = {'schema', 'tilt_gain_per_s', 'damping', 'nullspace_gain_per_s',
        'max_reference_speed_rad_s', 'max_reference_acceleration_rad_s2',
        'active_joints', 'left_q_min_deg', 'left_q_max_deg',
        'right_q_min_deg', 'right_q_max_deg', 'pd_torque_envelope_nm'}
    if not isinstance(c,dict) or set(c)!=expected or c['schema']!='g1_dual_upright_v1':
        raise ValueError('invalid dual upright configuration')
    for key, maximum in [('tilt_gain_per_s',4.),('damping',.1),('nullspace_gain_per_s',.5),
                         ('max_reference_speed_rad_s',.2),('max_reference_acceleration_rad_s2',.6)]:
        if not np.isfinite(c[key]) or not 0 < c[key] <= maximum:
            raise ValueError('invalid or excessive '+key)
    if c['active_joints'] != [True,False,False,True,True]:
        raise ValueError('this stationary tool keeps shoulder roll/yaw reference zero')
    for side in ('left','right'):
        lo,hi=np.asarray(c[side+'_q_min_deg']),np.asarray(c[side+'_q_max_deg'])
        if (lo.shape!=(5,) or hi.shape!=(5,) or not np.isfinite(np.r_[lo,hi]).all()
                or np.any(lo>=0) or np.any(hi<=0)
                or np.any(np.maximum(abs(lo),abs(hi)) > [20,20,25,40,40])):
            raise ValueError('invalid '+side+' reference/feedback range')
    limit=np.asarray(c['pd_torque_envelope_nm'])
    if limit.shape!=(5,) or not np.isfinite(limit).all() or np.any(limit<=0) or np.any(limit>[10,6,4,7,1.5]):
        raise ValueError('invalid estimated PD torque envelope')
    return c


def effective_profile(profile):
    result=dict(profile)
    for key in ('target_q_array','kp_array','kd_array'):
        result[key]=np.asarray(profile[key],dtype=float).copy()
    result['target_q_array'][:]=0.
    result['dual_upright_experiment']=dict(enabled=True,both_arms_use_imu=True,
        target_axis_world=[0.,0.,1.],nominal_reference_rad=[0.]*10,
        shoulder_roll_yaw_reference_fixed=True,task='stationary')
    return result


class DualAxisKinematics:
    def __init__(self,model):
        self.endpoint=model
        self.dofs={side:[model.model.joint(n).dofadr[0] for n in JOINTS[slice(i*5,i*5+5)]]
                   for i,side in enumerate(('left','right'))}
        self.jac=np.zeros((3,model.model.nv))

    def evaluate(self,slots,world_from_body):
        e=self.endpoint
        q=np.asarray(slots,dtype=float)
        if q.shape!=(13,) or not np.isfinite(q).all():
            raise ValueError('expected finite 13-slot feedback')
        mujoco.mj_resetData(e.model,e.data)
        e.data.qpos[e.joint_addresses]=q[:11]
        mujoco.mj_kinematics(e.model,e.data)
        mujoco.mj_comPos(e.model,e.data)
        r_gb=e.data.site_xmat[e.imu_id].reshape(3,3)
        r_wg=world_from_body@r_gb.T
        result={}
        for side in ('left','right'):
            r_ge=e.data.site_xmat[e.sites[side]].reshape(3,3)
            z=r_wg@r_ge[:,2]
            mujoco.mj_jacSite(e.model,e.data,None,self.jac,e.sites[side])
            axes=r_wg@self.jac[:,self.dofs[side]]
            dz=np.cross(axes.T,z).T
            result[side]=(z.copy(),dz[:2].copy())
        return result


class DualUprightController:
    control_dt=CONTROL_PERIOD_S
    offline_only=False

    def __init__(self,config=CONFIG,model_config=MODEL):
        self.config=load_config(config)
        self.kinematics=DualAxisKinematics(EndpointModel(model_config))
        self.active=np.array(self.config['active_joints'],dtype=bool)
        self.minimum=np.r_[tuple(np.deg2rad(self.config[s+'_q_min_deg']) for s in ('left','right'))]
        self.maximum=np.r_[tuple(np.deg2rad(self.config[s+'_q_max_deg']) for s in ('left','right'))]
        self.qref=None
        self.dqref=np.zeros(10)
        self.last_diagnostics={}
        self.metadata=dict(controller_kind='dual_upright_reference_servo',task='stationary',
            control_period_s=self.control_dt,config=self.config,
            target_axis_world=[0.,0.,1.],model_config=str(model_config),
            xml_sha256=self.kinematics.endpoint.xml_hashes(),
            acceleration_prediction=False,inverse_dynamics=False,
            world_frame='torso IMU navigation frame; yaw does not affect vertical',
            independent_payload_measurement=False,physical_collision_guarantee=False)

    def seed(self,last_reference):
        q=np.asarray(last_reference,dtype=float)
        if q.shape!=(10,) or not np.isfinite(q).all() or np.any(q<self.minimum) or np.any(q>self.maximum):
            raise ValueError('invalid dual reference seed')
        self.qref=q.copy()
        self.dqref.fill(0.)

    def step(self,slots,imu_quaternion,dt):
        dt=float(dt)
        if not np.isfinite(dt) or dt<=0:
            raise ValueError('invalid feedback period')
        dt=min(dt,self.control_dt)
        q=np.asarray(slots,dtype=float)
        if q.shape!=(13,) or not np.isfinite(q).all():
            raise ValueError('invalid arm feedback')
        if np.any(q[:10]<self.minimum) or np.any(q[:10]>self.maximum):
            raise ValueError('dual upright measured joint envelope exceeded')
        if self.qref is None:
            raise RuntimeError('seed dual references from the last entry command first')
        geometry=self.kinematics.evaluate(q,rotation(imu_quaternion))
        raw=np.zeros(10)
        diagnostics=dict(pid_active=True,dual_upright_active=True,
            controller_kind='dual_upright_reference_servo',target_axis_world=[0.,0.,1.])
        c=self.config
        for i,side in enumerate(('left','right')):
            z,jac=geometry[side]
            # Two horizontal components alone also vanish upside down.
            if z[2]<=.25:
                raise ValueError(side+' axis outside upward convergence hemisphere')
            j=jac[:,self.active]
            pinv=j.T@np.linalg.solve(j@j.T+c['damping']**2*np.eye(2),np.eye(2))
            task=pinv@(-c['tilt_gain_per_s']*z[:2])
            null=np.eye(3)-pinv@j
            posture=-c['nullspace_gain_per_s']*q[i*5:i*5+5][self.active]
            raw[i*5:i*5+5][self.active]=task+null@posture
            # Finish any tiny residual from the last pre-3s entry sample,
            # without snapping the reference to exactly zero at handoff.
            raw[i*5:i*5+5][~self.active]=-c['tilt_gain_per_s']*self.qref[i*5:i*5+5][~self.active]
            diagnostics[side]=dict(axis_world=z.tolist(),
                tilt_deg=float(np.degrees(np.arccos(np.clip(z[2],-1,1)))),
                raw_active_reference_velocity_rad_s=(task+null@posture).tolist())
        previous_q=self.qref.copy()
        previous_dq=self.dqref.copy()
        a=c['max_reference_acceleration_rad_s2']
        vmax=c['max_reference_speed_rad_s']
        lower_distance=np.maximum(previous_q-self.minimum,0.)
        upper_distance=np.maximum(self.maximum-previous_q,0.)
        lower_speed=-a*dt+np.sqrt((a*dt)**2+2*a*lower_distance)
        upper_speed=-a*dt+np.sqrt((a*dt)**2+2*a*upper_distance)
        target=np.clip(raw,-np.minimum(vmax,lower_speed),np.minimum(vmax,upper_speed))
        velocity=np.clip(target,previous_dq-a*dt,previous_dq+a*dt)
        self.qref=np.clip(previous_q+velocity*dt,self.minimum,self.maximum)
        self.dqref=(self.qref-previous_q)/dt
        if np.max(abs(self.dqref-previous_dq))>a*dt+1e-9:
            raise RuntimeError('reference acceleration envelope failed')
        diagnostics['reference_velocity_rad_s']=self.dqref.tolist()
        diagnostics['reference_position_rad']=self.qref.tolist()
        diagnostics['reference_at_limit']=(np.minimum(self.qref-self.minimum,self.maximum-self.qref)<1e-6).tolist()
        self.last_diagnostics=diagnostics
        return self.qref.copy(),self.dqref.copy(),diagnostics


class DualUprightPlan(HardwarePidPlan):
    def sample(self,task_s,measured_slots,measured_dq,imu_quaternion,yaw0_rad,dt):
        if ENTRY_END_S<=task_s:
            if self.controller.qref is None:
                self.controller.seed(self._last_q[:10])
            arm_q,arm_dq,diag=self.controller.step(measured_slots,imu_quaternion,dt)
            q=self.target.copy();q[:10]=arm_q
            dq=np.zeros(13);dq[:10]=arm_dq
            self._last_q=q.copy()
            return dict(stage='stationary_upright',q_rad=q,dq_rad_s=dq,
                kp=self.kp,kd=self.kd,weight=1.,terminal=False,diagnostics=diag)
        return super().sample(task_s,measured_slots,measured_dq,imu_quaternion,yaw0_rad,dt)


class DualUprightRuntime:
    stationary=True
    field_trial=False
    continuous_stationary_hold=True
    actuation='reference_servo'
    controller_label='DUAL UPRIGHT (stationary, both arms)'

    def __init__(self,config=CONFIG,model_config=MODEL):
        self.controller=DualUprightController(config,model_config)
        self._last_console_ns=None

    def observe_low(self,*args):
        pass  # Shared transport supplies a checked snapshot to plan.sample.

    def observe_imu(self,*args):
        pass

    def set_epoch(self,epoch_ns):
        self.epoch_ns=int(epoch_ns)

    def prepare(self,*args,**kwargs):
        pass  # No disturbance predictor is used for this attitude-only task.

    def create_plan(self,initial,profile):
        return DualUprightPlan(initial,profile['target_q_array'],profile['kp_array'],
                              profile['kd_array'],self.controller)

    def check_before_write(self,frame,low,imu,loop_begin_ns,now_ns):
        if not frame['diagnostics'].get('dual_upright_active'):
            return
        if now_ns-loop_begin_ns>20_000_000:
            raise RuntimeError('dual upright calculation exceeded 20 ms')
        indices=list(ARM_MOTOR_INDICES[:10])
        q=np.asarray(low.q)[indices];dq=np.asarray(low.dq)[indices]
        if not np.isfinite(np.r_[q,dq]).all():
            raise ValueError('nonfinite dual arm state')
        c=self.controller
        if np.any(q<c.minimum) or np.any(q>c.maximum):
            raise RuntimeError('dual upright measured envelope exceeded before write')
        tau=np.asarray(frame['kp'])[:10]*(frame['q_rad'][:10]-q)+np.asarray(frame['kd'])[:10]*(frame['dq_rad_s'][:10]-dq)
        frame['diagnostics']['estimated_pd_torque_nm']=tau.tolist()
        if np.any(abs(tau)>np.tile(c.config['pd_torque_envelope_nm'],2)):
            raise RuntimeError('dual upright estimated PD torque envelope exceeded')
        if self._last_console_ns is None or now_ns-self._last_console_ns>=1_000_000_000:
            left=frame['diagnostics']['left']['tilt_deg']
            right=frame['diagnostics']['right']['tilt_deg']
            print(f'World-up tilt (IMU/model): left={left:.2f} deg, right={right:.2f} deg',flush=True)
            self._last_console_ns=now_ns
