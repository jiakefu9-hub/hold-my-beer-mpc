"""MPC-local CRC adapter; official SDK packing and native polynomial unchanged.

The SDK turns each packed word into a Python integer, then copies those
integers back to a C array. Keep the packed words instead. No DDS is opened,
no checksum is skipped, and no installed SDK/global CRC singleton is modified.
"""
import ctypes as ct
import sys
from unitree_sdk2py.utils.crc import CRC


class PackedCRC(CRC):
    def __new__(cls):
        # SDK CRC is a singleton. This adapter must NOT replace that instance
        # or change the field-proven PID's normal CRC implementation.
        return object.__new__(cls)

    def __init__(self):
        super().__init__()
        if self.platform != 'Linux' or sys.byteorder != 'little':
            raise RuntimeError('packed MPC CRC requires supported little-endian Linux')
        self.crc_lib = ct.PyDLL(self.crc_lib._name)
        self.crc_lib.crc32_core.argtypes = (ct.POINTER(ct.c_uint32),ct.c_uint32)
        self.crc_lib.crc32_core.restype = ct.c_uint32

    def _CRC__Trans(self, packed):
        # Exactly the same words as SDK __Trans, excluding the final CRC word.
        if len(packed)%4:
            raise ValueError('unaligned official SDK CRC input')
        return (ct.c_uint32*(len(packed)//4-1)).from_buffer_copy(packed)

    def _crc_ctypes(self, data):
        return self.crc_lib.crc32_core(data,len(data))
