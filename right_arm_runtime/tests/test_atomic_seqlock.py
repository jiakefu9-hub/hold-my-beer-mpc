"""Shared libatomic binding and adapter-dependency regression tests."""

from __future__ import annotations

import ctypes
from pathlib import Path
import unittest

from right_arm_runtime import atomic_seqlock
from right_arm_runtime import sim_process
from right_arm_runtime import unitree_shm


class AtomicSeqlockTest(unittest.TestCase):
    def test_atomic_uint64_load_store_preserves_existing_memory_orders(self):
        value = ctypes.c_uint64(0)
        address = ctypes.addressof(value)
        atomic_seqlock._ATOMIC_STORE_8(
            address,
            0xFEDCBA9876543210,
            atomic_seqlock._MEMORY_ORDER_RELEASE,
        )
        observed = atomic_seqlock._ATOMIC_LOAD_8(
            address,
            atomic_seqlock._MEMORY_ORDER_ACQUIRE,
        )
        self.assertEqual(observed, 0xFEDCBA9876543210)
        self.assertEqual(atomic_seqlock._MEMORY_ORDER_ACQUIRE, 2)
        self.assertEqual(atomic_seqlock._MEMORY_ORDER_RELEASE, 3)

    def test_both_platform_adapters_share_one_binding(self):
        self.assertIs(
            sim_process._ATOMIC_LOAD_8,
            atomic_seqlock._ATOMIC_LOAD_8,
        )
        self.assertIs(
            sim_process._ATOMIC_STORE_8,
            atomic_seqlock._ATOMIC_STORE_8,
        )
        self.assertIs(
            unitree_shm._ATOMIC_LOAD_8,
            atomic_seqlock._ATOMIC_LOAD_8,
        )
        self.assertIs(
            unitree_shm._ATOMIC_STORE_8,
            atomic_seqlock._ATOMIC_STORE_8,
        )

    def test_sim_adapter_no_longer_imports_hardware_adapter(self):
        source = Path(sim_process.__file__).read_text(encoding="utf-8")
        self.assertNotIn(".unitree_shm import", source)


if __name__ == "__main__":
    unittest.main()
