"""Single-flight, bounded, SDK-free compute worker for the existing MPC.

The parent alone owns DDS, interlocks and hand-back. A hard-timed-out/crashed
worker is terminal. The learned variant allows a bounded longer wait, followed
by parent-side late-result checks; no stale sequence or automatic restart.
"""
import gc
import multiprocessing as mp
import os
import pickle
import socket
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
from g1_walk_mpc import MpcRuntime
from hardware_mpc_field import TorqueHandback

CAPACITY = 2**20


def _store(buffer, size, value):
    payload=pickle.dumps(value,protocol=5)
    if len(payload)>CAPACITY:
        raise RuntimeError('MPC compute message exceeds bounded shared buffer')
    memoryview(buffer).cast('B')[:len(payload)]=payload
    size.value=len(payload)


def _load(buffer,size):
    if not 0<size.value<=CAPACITY:
        raise RuntimeError('invalid MPC compute message length')
    return pickle.loads(memoryview(buffer).cast('B')[:size.value])


def _worker(request,response,nrequest,nresponse,ready,done,stopping,allowed,cpu,priority,args,kwargs):
    # This process has no transport authority. Spawn (never fork live DDS).
    def forbidden(*a,**k): raise RuntimeError('MPC compute worker forbids sockets')
    socket.socket=forbidden
    os.sched_setaffinity(0,set(allowed))
    runtime=scope=plan=None
    pending_frame=None
    try:
        from mpc_host import ControlThreadScope
        runtime=MpcRuntime(*args,**kwargs)
        scope=ControlThreadScope(cpu,priority)
        scope.prepare_workers()
        _store(response,nresponse,dict(id=0,ok=True,result=dict(
            warmup=runtime.warmup,config=runtime.controller.config,
            torque_config=runtime.controller.torque_config,metadata=runtime.controller.metadata,
            mpc_start_s=runtime.mpc_start_s,
            predictor_manifest=None if runtime.predictor.bank is None else runtime.predictor.bank.manifest)))
        done.release()
        while not stopping.value:
            if not ready.acquire(timeout=.1): continue
            if stopping.value: break
            message=_load(request,nrequest);serial=message['id'];op=message['op']
            try:
                for kind,values in message.get('observations',[]):
                    getattr(runtime,'observe_'+kind)(*values)
                commit=message.get('commit')
                if commit is not None:
                    if pending_frame is None: raise RuntimeError('compute commit without pending candidate')
                    motors=[SimpleNamespace(q=0.,dq=0.,tau=0.) for _ in range(30)]
                    for i,values in zip(range(22,27),commit['motors']):
                        motors[i]=SimpleNamespace(q=values[0],dq=values[1],tau=values[2],kp=values[3],kd=values[4])
                    motors[29].q=commit['weight']
                    plan.commit_packet(pending_frame,SimpleNamespace(motor_cmd=motors))
                    pending_frame=None
                if op=='plan':
                    plan=runtime.create_plan(*message['args']);result=runtime.controller.metadata
                elif op=='startup':
                    runtime.prepare_startup();result=None
                elif op=='activate': result=scope.activate()
                elif op=='epoch': runtime.set_epoch(message['args'][0]);result=None
                elif op=='step':
                    runtime.prepare(*message['prepare_args'],**message['prepare_kwargs'])
                    pending_frame=plan.sample(*message['args']);result=pending_frame
                else: raise RuntimeError('unknown compute operation')
                reply=dict(id=serial,ok=True,result=result)
            except Exception as exc:
                reply=dict(id=serial,ok=False,error=str(exc),diagnostics=runtime.controller.last_diagnostics)
            _store(response,nresponse,reply);done.release()
            if not reply['ok']: break
    except BaseException as exc:
        _store(response,nresponse,dict(id=0,ok=False,error=repr(exc)));done.release()
    finally:
        if runtime is not None: runtime.close()
        if scope is not None: scope.restore()


class ProcessMpcRuntime(MpcRuntime):
    """Same runner API, but only bounded numerical requests cross processes."""
    def __init__(self,*args,compute_cpu=7,compute_priority=0,compute_affinity=None,**kwargs):
        if not kwargs.get('field_trial'):
            raise ValueError('compute process requires the field torque lifecycle')
        self.actuation='measured_torque_preview';self.field_trial=True
        self.stationary=kwargs.get('stationary',False)
        self.assumed_command_delay_s=kwargs.get('assumed_command_delay_s')
        self.host_scope=None;self.journal=None;self.handback=TorqueHandback()
        self.epoch_ns=None;self._gc_was_enabled=None
        self._switch_interval=sys.getswitchinterval()
        self._lock=threading.Lock();self._observations={'low':deque(),'imu':deque()};self._poisoned=False
        self._last_low=self._last_imu=None
        self._commit=None;self._candidate=None;self._prepare=None;self._serial=0;self._committed=0
        context=mp.get_context('spawn')
        self._request=context.RawArray('B',CAPACITY);self._response=context.RawArray('B',CAPACITY)
        self._nrequest=context.RawValue('I',0);self._nresponse=context.RawValue('I',0)
        # Event.set() uses a shared Condition handshake: a killed waiter can
        # strand the sender inside notify_all, beyond its intended deadline.
        # Single-flight semaphores have nonblocking posts and bounded acquires.
        self._ready=context.Semaphore(0);self._done=context.Semaphore(0)
        self._stopping=context.RawValue('b',False)
        affinity=set(os.sched_getaffinity(0)) if compute_affinity is None else set(compute_affinity)
        self._process=context.Process(target=_worker,args=(self._request,self._response,
            self._nrequest,self._nresponse,self._ready,self._done,self._stopping,
            sorted(affinity),compute_cpu,compute_priority,args,kwargs),daemon=True)
        self._process.start()
        try:
            initial=self._receive(0,30.)
            self.warmup=initial['warmup']
            manifest=initial['predictor_manifest']
            self.predictor=SimpleNamespace(bank=None if manifest is None else SimpleNamespace(manifest=manifest))
            self.controller=SimpleNamespace(config=initial['config'],torque_config=initial['torque_config'],
                metadata=initial['metadata'],last_diagnostics={})
            self.mpc_start_s=float(initial['mpc_start_s'])
            self.configure_timing_grace()
            self._step_timeout_s = (.009 if self.timing_grace is None
                                    else self.timing_grace.worker_timeout_s)
            self.controller.metadata['compute_isolation']=dict(process=True,pid=self._process.pid,
                communication='bounded single-flight shared buffers',hardware_output=False,
                deadline_s=self._step_timeout_s,automatic_restart=False)
        except BaseException:
            self.close();raise

    def _receive(self,serial,timeout):
        deadline=time.monotonic()+timeout
        while not self._done.acquire(timeout=min(.001,max(0.,deadline-time.monotonic()))):
            if not self._process.is_alive() or time.monotonic()>=deadline:
                self._poisoned=True
                raise RuntimeError('MPC compute worker timeout/exited; hand back')
        if time.monotonic()>deadline:
            self._poisoned=True
            raise RuntimeError('MPC compute reply arrived after deadline; hand back')
        reply=_load(self._response,self._nresponse)
        if reply.get('id')!=serial or not reply.get('ok'):
            self._poisoned=True
            if hasattr(self,'controller'):self.controller.last_diagnostics=reply.get('diagnostics',{})
            raise RuntimeError(reply.get('error','MPC compute response sequence mismatch'))
        return reply['result']

    def _rpc(self,op,*,timeout=2.,**values):
        if self._poisoned: raise RuntimeError('MPC compute worker already failed')
        if not self._process.is_alive():
            self._poisoned=True
            raise RuntimeError('MPC compute worker timeout/exited; hand back')
        self._serial+=1
        with self._lock:
            observations=[(kind,values) for kind,buffer in self._observations.items() for values in buffer]
            for buffer in self._observations.values(): buffer.clear()
        message=dict(id=self._serial,op=op,observations=observations,commit=self._commit,**values)
        self._commit=None
        _store(self._request,self._nrequest,message);self._ready.release()
        return self._receive(self._serial,timeout)

    def _observe(self,kind,values):
        stamp=int(values[0]);key='_last_'+kind
        with self._lock:
            previous=getattr(self,key)
            if previous is not None and stamp<previous: raise ValueError('ingress timestamp rollback')
            if stamp==previous:return
            buffer=self._observations[kind]
            # Match predictor's rolling history during human confirmation or
            # parent-only hand-back; retain one sample before the cutoff.
            cutoff=stamp-750_000_000
            while len(buffer)>1 and buffer[1][0]<cutoff:buffer.popleft()
            if len(buffer)>=4096: raise RuntimeError('MPC compute ingress overflow')
            buffer.append((stamp,*[np.asarray(v,dtype=float).copy() for v in values[1:]]))
            setattr(self,key,stamp)
        if self.journal is not None:
            names=('q_rad','dq_rad_s') if kind=='low' else ('quaternion_wxyz','gyroscope_rad_s','accelerometer_raw_m_s2')
            self.journal.record(dict(schema='g1_mpc_predictor_'+kind+'_v1',received_monotonic_ns=stamp,
                                     **dict(zip(names,values[1:]))))

    def observe_low(self,*values): self._observe('low',values)
    def observe_imu(self,*values): self._observe('imu',values)

    def create_plan(self,initial,profile):
        meta=self._rpc('plan',args=(initial,profile))
        self.controller.metadata.update(meta)
        return self

    def prepare_startup(self):
        self._rpc('startup');self._gc_was_enabled=gc.isenabled();gc.collect();gc.disable()

    def enter_control_thread(self,cpu):
        worker=self._rpc('activate')
        if self.host_scope is None or not self.host_scope.prepared:
            raise RuntimeError('prepare transport worker placement before compute activation')
        # The isolated compute core should not also run the Python transport
        # loop. Parent remains on a housekeeping core with its own GIL.
        from mpc_host import host_evidence
        allowed=set(os.sched_getaffinity(0))
        transport_cpu=2 if 2 in allowed else min(allowed)
        os.sched_setaffinity(0,{transport_cpu})
        if self.host_scope.priority:
            os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(self.host_scope.priority))
        sys.setswitchinterval(min(self._switch_interval,.0005))
        parent=host_evidence(transport_cpu)
        return dict(**parent,compute_process_host=worker,compute_pid=self._process.pid,
                    python_thread_switch_interval_ms=sys.getswitchinterval()*1000)

    def set_epoch(self,epoch_ns):
        self.epoch_ns=int(epoch_ns);self._rpc('epoch',args=(epoch_ns,))

    def prepare(self,*args,**kwargs): self._prepare=(args,kwargs)

    def sample(self,*args):
        if self._prepare is None:raise RuntimeError('compute sample without fresh prepare')
        begin=time.perf_counter_ns();p,k=self._prepare;self._prepare=None
        frame=self._rpc('step',timeout=self._step_timeout_s,args=args,prepare_args=p,prepare_kwargs=k)
        frame['diagnostics']['compute_process_roundtrip_ms']=(time.perf_counter_ns()-begin)*1e-6
        self.controller.last_diagnostics=frame['diagnostics']
        self._candidate=frame
        return frame

    def accept_packet(self,frame,packet):
        # Latch the last actually sent packet locally FIRST, even if the worker
        # has died. Parent-only hand-back must never need the compute process.
        self.handback.accept(packet)
        if frame is self._candidate:
            if self.assumed_command_delay_s is not None:
                self._commit=dict(motors=[np.asarray([packet.motor_cmd[i].q,packet.motor_cmd[i].dq,
                    packet.motor_cmd[i].tau,packet.motor_cmd[i].kp,packet.motor_cmd[i].kd],dtype=np.float32).astype(float) for i in range(22,27)],
                    weight=float(np.float32(packet.motor_cmd[29].q)))
            frame['diagnostics']['committed_packet_sequence']=self._committed
            self._committed+=1
            self._candidate=None

    def close(self):
        self._stopping.value=True;self._ready.release()
        self._process.join(.5)
        if self._process.is_alive():self._process.terminate();self._process.join(1.)
        if self.host_scope is not None:self.host_scope.restore()
        sys.setswitchinterval(self._switch_interval)
        if self._gc_was_enabled:gc.enable()
