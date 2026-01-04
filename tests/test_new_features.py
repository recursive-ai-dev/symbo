# Copyright 2025
# Additional Unit Tests for New Military-Grade Features

import unittest
import sympy as sp
import numpy as np
import sys
import os
import shutil
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib.util
spec = importlib.util.spec_from_file_location("symbo_module",
    os.path.join(os.path.dirname(__file__), '..', 'symbo.py'))
symbo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symbo_module)
NanoTensor = symbo_module.NanoTensor

class TestNewMilitaryFeatures(unittest.TestCase):
    """Test the newly added features: Persistence, Optimization, Anomaly Detection."""

    def setUp(self):
        self.nt = NanoTensor((1,), max_order=1, base_vars=['x'])
        self.x = sp.Symbol('x')
        self.nt.data[0] = 2 * self.x

    def test_persistence(self):
        """Test save and load functionality."""
        filename = "test_brain.pkl"
        try:
            # Train
            self.nt.eval_numeric({'x': 10.0})
            ops_before = self.nt._operation_count

            # Save
            self.nt.save_brain(filename)
            self.assertTrue(os.path.exists(filename))

            # Load
            nt_loaded = NanoTensor.load_brain(filename)

            # Verify
            self.assertEqual(nt_loaded.name, self.nt.name)
            self.assertEqual(nt_loaded._operation_count, ops_before)
            # Verify data equality (sympy expression)
            self.assertEqual(nt_loaded.data[0], self.nt.data[0])

        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_optimization_cse(self):
        """Test CSE optimization."""
        # Create repetitive expression
        expr = (self.x + 1)**5 + (self.x + 1)**4
        self.nt.data[0] = expr

        self.assertIsNone(self.nt._optimized_data)
        self.assertIsNone(self.nt._optimized_func)

        self.nt.optimize_storage()

        self.assertIsNotNone(self.nt._optimized_data)
        self.assertIsNotNone(self.nt._optimized_func)

        replacements, reduced = self.nt._optimized_data
        self.assertGreater(len(replacements), 0)

        # Verify evaluation still works and uses optimized path
        # We can't easily check if optimized path was used without mocking or logging
        # but we can verify correctness
        res = self.nt.eval_numeric({'x': 1.0})
        # (1+1)^5 + (1+1)^4 = 32 + 16 = 48
        self.assertAlmostEqual(res[0], 48.0)

        # Ensure it works even if we reset the lambdify cache (force use of optimized_func)
        self.nt._lambdify_cache.clear()
        res2 = self.nt.eval_numeric({'x': 2.0})
        # (2+1)^5 + (2+1)^4 = 243 + 81 = 324
        self.assertAlmostEqual(res2[0], 324.0)

    def test_anomaly_detection_logic(self):
        """Test statistical anomaly detection internals."""
        # Manually update stats
        for val in [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]:
            self.nt._update_input_stats('x', val)

        # Variance should be 0, mean 10
        stats = self.nt._input_stats['x']
        self.assertAlmostEqual(stats['mean'], 10.0)

        # Check normal value
        self.assertFalse(self.nt._check_input_anomaly('x', 10.0))

        # Check anomaly (since variance is 0, any deviation is infinite Z-score)
        # But our check has a safety for variance < 1e-12
        self.assertFalse(self.nt._check_input_anomaly('x', 11.0))

        # Add some variance
        self.nt._update_input_stats('x', 12.0)
        self.nt._update_input_stats('x', 8.0)

        # Now we have variance. Mean is still 10.
        # Anomaly check
        self.assertTrue(self.nt._check_input_anomaly('x', 100.0))

if __name__ == '__main__':
    unittest.main()
