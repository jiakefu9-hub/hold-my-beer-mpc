"""Shared libatomic bindings for the runtime seqlock protocols.

Both the MuJoCo process adapter and the Unitree shared-memory adapter exchange
payloads with C++ through a uint64 sequence counter.  Keep the exact GCC
``libatomic`` operations and memory-order constants in this dependency-neutral
module so neither platform adapter needs to import the other.
"""

from __future__ import annotations

import ctypes
import ctypes.util


def _load_libatomic():
    library_name = ctypes.util.find_library("atomic") or "libatomic.so.1"
    library = ctypes.CDLL(library_name)
    atomic_load = getattr(library, "__atomic_load_8")
    atomic_load.argtypes = [ctypes.c_void_p, ctypes.c_int]
    atomic_load.restype = ctypes.c_uint64
    atomic_store = getattr(library, "__atomic_store_8")
    atomic_store.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    atomic_store.restype = None
    return library, atomic_load, atomic_store


# Retain the CDLL object for the lifetime of the bound function pointers.
_LIBATOMIC, _ATOMIC_LOAD_8, _ATOMIC_STORE_8 = _load_libatomic()

# These values are the GCC ``__ATOMIC_ACQUIRE`` and ``__ATOMIC_RELEASE`` ABI
# constants used by the existing Python/C++ seqlock protocols.
_MEMORY_ORDER_ACQUIRE = 2
_MEMORY_ORDER_RELEASE = 3
