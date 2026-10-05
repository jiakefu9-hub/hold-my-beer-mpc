"""Small SDK-free native equivalent of CommandHistory's 2 ms model loop."""
import ctypes as ct
import hashlib
import weakref
import numpy as np
from endpoint_pose import ROOT

LIBRARY = ROOT/'build/g1_arm_delay/libg1_arm_delay.so'
DP = ct.POINTER(ct.c_double)
IP = ct.POINTER(ct.c_int)
BP = ct.POINTER(ct.c_ubyte)


class NativeArmDelay:
    def __init__(self, inverse, library=LIBRARY):
        self.lib = ct.CDLL(str(library))
        self.lib.g1_delay_abi.restype=ct.c_int
        if self.lib.g1_delay_abi()!=1:
            raise RuntimeError('unsupported native delay ABI')
        self.lib.g1_delay_source_sha256.restype=ct.c_char_p
        source_hash=hashlib.sha256((ROOT/'cpp/g1_arm_delay/delay.cpp').read_bytes()).hexdigest()
        if self.lib.g1_delay_source_sha256().decode()!=source_hash:
            raise RuntimeError('native delay source changed; rebuild cpp/g1_arm_delay')
        self.lib.g1_delay_header_version.restype=ct.c_int
        self.lib.g1_delay_runtime_version.restype=ct.c_int
        version=self.lib.g1_delay_runtime_version()
        if self.lib.g1_delay_header_version()!=version:
            raise RuntimeError('native delay MuJoCo header/runtime version mismatch')
        self.lib.g1_delay_create.argtypes=[ct.c_char_p,IP,IP,DP,DP,ct.c_char_p,ct.c_int]
        self.lib.g1_delay_create.restype=ct.c_void_p
        self.lib.g1_delay_destroy.argtypes=[ct.c_void_p]
        self.lib.g1_delay_predict.argtypes=[ct.c_void_p,ct.c_int,DP,DP,DP,BP,DP,DP,DP,DP,DP]
        self.lib.g1_delay_predict.restype=ct.c_int
        qi=np.ascontiguousarray(inverse.q_indices,dtype=np.int32)
        vi=np.ascontiguousarray(inverse.v_indices,dtype=np.int32)
        offset=np.ascontiguousarray(inverse.root_to_imu)
        mount=np.ascontiguousarray(inverse.root_from_imu)
        error=ct.create_string_buffer(1024)
        self.handle=self.lib.g1_delay_create(str(inverse.model.xml).encode(),qi.ctypes.data_as(IP),
            vi.ctypes.data_as(IP),offset.ctypes.data_as(DP),mount.ctypes.data_as(DP),error,len(error))
        if not self.handle:
            raise RuntimeError('native delay model: '+error.value.decode())
        self._finalizer=weakref.finalize(self,self.lib.g1_delay_destroy,self.handle)
        self.metadata=dict(backend='native_mujoco',abi=1,library=str(library),
                           sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                           source_sha256=source_hash,mujoco_version=version)

    def close(self):
        self._finalizer()
        self.handle=None

    def predict(self, q, dq, timeline, bases, config, limits):
        if not self.handle:
            raise RuntimeError('native delay already closed')
        dt=np.asarray([row[1] for row in timeline],dtype=float)
        b=np.asarray([np.r_[d.acc_world,d.omega_world,d.alpha_world,
                          d.rot_world_body.ravel()] for d in bases])
        command=np.zeros((len(timeline),16));present=np.zeros(len(timeline),dtype=np.uint8)
        for i,(_,_,active) in enumerate(timeline):
            if active is not None:
                command[i,:5]=active.ff;command[i,5:10]=active.qref
                command[i,10:15]=active.dqref;command[i,15]=active.weight;present[i]=1
        kp=np.ascontiguousarray(config['kp'],dtype=float);kd=np.ascontiguousarray(config['kd'],dtype=float)
        limit=np.ascontiguousarray(limits,dtype=float)
        initial=np.r_[q,dq];result=np.empty(10)
        code=self.lib.g1_delay_predict(self.handle,len(timeline),dt.ctypes.data_as(DP),b.ctypes.data_as(DP),
            command.ctypes.data_as(DP),present.ctypes.data_as(BP),kp.ctypes.data_as(DP),kd.ctypes.data_as(DP),
            limit.ctypes.data_as(DP),initial.ctypes.data_as(DP),result.ctypes.data_as(DP))
        if code:
            raise RuntimeError(f'native nominal delay propagation failed: {code}')
        return result[:5].copy(),result[5:].copy()
