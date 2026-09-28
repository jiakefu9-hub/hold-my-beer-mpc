"""Preserve diagnostic JSON values across the serialization fast paths."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware_mpc_control import json_values


def legacy_json_values(value):
    """Reference conversion before the finite-array/native-scalar fast paths."""
    if isinstance(value, dict):
        return {key: legacy_json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [legacy_json_values(item) for item in value]
    if isinstance(value, np.ndarray):
        return legacy_json_values(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class DiagnosticSerializationTest(unittest.TestCase):
    def assert_same_json(self, value):
        expected = legacy_json_values(value)
        actual = json_values(value)
        self.assertEqual(json.dumps(actual, allow_nan=False),
                         json.dumps(expected, allow_nan=False))

    def test_numeric_array_shapes_dtypes_and_layouts(self):
        for dtype in (np.bool_, np.int8, np.uint64, np.int64, np.float16,
                      np.float32, np.float64, np.longdouble):
            for shape in ((), (0,), (1,), (2, 3)):
                with self.subTest(dtype=dtype, shape=shape):
                    count = int(np.prod(shape))
                    value = np.arange(count).astype(dtype).reshape(shape)
                    self.assert_same_json(value)
        self.assert_same_json(np.arange(20., dtype='>f8').reshape(4, 5).T[:, ::2])
        self.assert_same_json(np.array([2**64-1], dtype=np.uint64))
        self.assert_same_json(np.array([-0., np.finfo(float).tiny, np.finfo(float).max]))

    def test_nonfinite_and_nested_object_arrays_keep_nulls(self):
        for dtype in (np.float16, np.float32, np.float64, np.longdouble):
            with self.subTest(dtype=dtype):
                self.assert_same_json(np.array([[1., np.nan], [np.inf, -np.inf]], dtype=dtype))
        value = np.empty(3, dtype=object)
        value[:] = [dict(error=np.float64(np.nan), valid=np.bool_(True)),
                    np.array([1., np.inf]), (np.int32(2), None, 'status')]
        self.assert_same_json(value)
        self.assert_same_json({'scalar': np.array(np.nan), 'empty': np.empty((0, 3))})

    def test_mixed_diagnostics_preserve_values_and_snapshot_ownership(self):
        values = np.arange(15., dtype=float).reshape(3, 5)
        record = {'solved': True, 'iterations': np.int64(8), 'tau': values,
                  'predictor': {'forecast_y': values.tolist(), 'mode': 'learned_filtered'},
                  'failure': None, 'bounds': (np.float32(1.25), float('inf'))}
        self.assert_same_json(record)
        converted = json_values(record)
        values[:] = -1.
        record['predictor']['forecast_y'][0][0] = 999.
        self.assertEqual(converted['tau'][0][0], 0.)
        self.assertEqual(converted['predictor']['forecast_y'][0][0], 0.)

    def test_scalar_subclasses_and_extended_floats_use_original_conversion(self):
        class CustomFloat(float):
            pass

        class CustomInt(int):
            pass

        self.assert_same_json([CustomFloat(2.5), CustomInt(8), np.float64(3.),
                               np.bool_(False), np.longdouble('1.234567890123456789')])
        self.assertIs(type(json_values(CustomFloat(2.5))), float)
        integer = CustomInt(8)
        self.assertIs(json_values(integer), integer)


if __name__ == '__main__':
    unittest.main()
