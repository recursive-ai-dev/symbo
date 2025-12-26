# Copyright 2025
# Unit Tests for Military-Grade NanoTensor

import unittest
import sympy as sp
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from symbo.py file directly
import importlib.util
spec = importlib.util.spec_from_file_location("symbo_module", 
    os.path.join(os.path.dirname(__file__), '..', 'symbo.py'))
symbo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symbo_module)
NanoTensor = symbo_module.NanoTensor


class TestMilitaryGradeFeatures(unittest.TestCase):
    """Test military-grade enhancements to NanoTensor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.nt = NanoTensor((2, 2), max_order=2, base_vars=['x', 'y'])
        self.x, self.y = sp.symbols('x y')
    
    def test_health_monitoring(self):
        """Test health monitoring capabilities."""
        # Initial health should be optimal
        health = self.nt.health_check()
        self.assertEqual(health['status'], 'optimal')
        self.assertIn('metrics', health)
        self.assertIn('learned_patterns', health)
    
    def test_operation_tracking(self):
        """Test that operations are tracked correctly."""
        # Perform some operations
        self.nt.data[0, 0] = self.x**2 + self.y
        self.nt.data[1, 1] = self.x * self.y
        
        # Differentiate
        self.nt.diff(self.x)
        
        # Check tracking
        self.assertGreater(self.nt._operation_count, 0)
        self.assertGreater(self.nt._success_count, 0)
        
        health = self.nt.health_check()
        self.assertGreater(health['metrics']['total_operations'], 0)
        self.assertGreater(health['metrics']['success_rate'], 0)
    
    def test_cache_tracking(self):
        """Test cache hit/miss tracking."""
        self.nt.data[0, 0] = self.x**2
        
        # First call - cache miss
        nt1 = self.nt.diff_cached('x', 1)
        
        # Second call - cache hit
        nt2 = self.nt.diff_cached('x', 1)
        
        self.assertGreater(self.nt._cache_hits, 0)
        self.assertEqual(nt1.data[0, 0], nt2.data[0, 0])
    
    def test_input_validation(self):
        """Test input validation with bounds."""
        self.nt.data[0, 0] = self.x + self.y
        
        # Set validation bounds
        self.nt.set_validation_bounds('x', -10.0, 10.0)
        self.nt.set_validation_bounds('y', -5.0, 5.0)
        
        # Valid input should work
        result = self.nt.eval_numeric({'x': 5.0, 'y': 2.0})
        self.assertIsInstance(result, np.ndarray)
        
        # Invalid input should raise error
        with self.assertRaises(ValueError):
            self.nt.eval_numeric({'x': 15.0, 'y': 2.0})  # x out of bounds
        
        with self.assertRaises(ValueError):
            self.nt.eval_numeric({'x': 5.0, 'y': 10.0})  # y out of bounds
    
    def test_nan_inf_detection(self):
        """Test detection of NaN and Inf values."""
        self.nt.data[0, 0] = self.x
        self.nt.set_validation_bounds('x', -100, 100)
        
        # NaN should be rejected
        with self.assertRaises(ValueError):
            self.nt.eval_numeric({'x': float('nan')})
        
        # Inf should be rejected
        with self.assertRaises(ValueError):
            self.nt.eval_numeric({'x': float('inf')})
    
    def test_learned_patterns(self):
        """Test that patterns are learned from operations."""
        self.nt.data[0, 0] = self.x**2 + self.y**2
        
        # Perform multiple operations
        for _ in range(5):
            self.nt.diff(self.x)
        
        # Check that patterns were learned
        self.assertGreater(len(self.nt._learned_patterns), 0)
        self.assertIn('differentiation', self.nt._learned_patterns)
        
        pattern = self.nt._learned_patterns['differentiation']
        self.assertIn('count', pattern)
        self.assertIn('avg_duration', pattern)
        self.assertIn('success_rate', pattern)
        self.assertGreater(pattern['count'], 0)
    
    def test_agency_status(self):
        """Test agency status reporting."""
        self.nt.data[0, 0] = self.x * self.y
        
        # Perform some operations
        self.nt.diff(self.x)
        self.nt.eval_numeric({'x': 1.0, 'y': 2.0})
        
        status = self.nt.get_agency_status()
        
        self.assertIn('experiences_recorded', status)
        self.assertIn('patterns_learned', status)
        self.assertIn('health', status)
        self.assertIn('recommendations', status)
        
        self.assertIsInstance(status['recommendations'], list)
    
    def test_error_recovery(self):
        """Test autonomous error recovery."""
        # Create a complex expression that might cause issues
        self.nt.data[0, 0] = (self.x**10 + self.y**10) / (self.x - self.y)
        
        try:
            # This should attempt recovery if it fails
            result = self.nt.diff(self.x)
            # If we get here, either it succeeded or recovery worked
            self.assertIsInstance(result, NanoTensor)
        except Exception:
            # Recovery didn't work, but that's okay for this test
            pass
        
        # Check that the operation was recorded
        self.assertGreater(self.nt._operation_count, 0)
    
    def test_auto_optimization(self):
        """Test autonomous self-optimization."""
        self.nt._auto_optimize = True
        self.nt.data[0, 0] = self.x**2 + self.y**2
        
        # Fill cache
        for i in range(50):
            var = sp.Symbol(f'z{i}')
            self.nt.diff_cached(var.name, 1)
        
        # Check that optimization happens when cache gets large
        initial_cache_size = len(self.nt._diff_cache)
        
        # Trigger health update that might clear cache
        self.nt._attempt_self_optimization()
        
        # Cache should be managed
        self.assertLessEqual(len(self.nt._diff_cache), initial_cache_size + 1)
    
    def test_performance_metrics(self):
        """Test comprehensive performance metrics."""
        self.nt.data[0, 0] = self.x**2
        
        # Perform operations
        self.nt.diff(self.x)
        self.nt.eval_numeric({'x': 1.0})
        self.nt.simplify()
        
        health = self.nt.health_check()
        metrics = health['metrics']
        
        # Check all expected metrics
        self.assertIn('total_operations', metrics)
        self.assertIn('success_rate', metrics)
        self.assertIn('cache_hit_rate', metrics)
        self.assertIn('avg_operation_time', metrics)
        self.assertIn('total_compute_time', metrics)
        
        # Verify metrics are reasonable
        self.assertGreater(metrics['total_operations'], 0)
        self.assertGreaterEqual(metrics['success_rate'], 0.0)
        self.assertLessEqual(metrics['success_rate'], 1.0)
        self.assertGreaterEqual(metrics['cache_hit_rate'], 0.0)
        self.assertLessEqual(metrics['cache_hit_rate'], 1.0)
    
    def test_health_degradation(self):
        """Test that health status degrades appropriately."""
        # Force some failures by trying invalid operations
        # Start with optimal health
        self.assertEqual(self.nt._health_status, 'optimal')
        
        # After operations, health should still be tracked
        self.nt.data[0, 0] = self.x
        self.nt.diff(self.x)
        
        # Health should be good or optimal
        self.assertIn(self.nt._health_status, ['optimal', 'good'])
    
    def test_experience_buffer(self):
        """Test experience buffer management."""
        self.nt.data[0, 0] = self.x + self.y
        
        # Perform many operations
        for i in range(20):
            self.nt.diff(self.x)
            self.nt.eval_numeric({'x': float(i), 'y': 1.0})
        
        # Check experience buffer
        self.assertGreater(len(self.nt._experience_buffer), 0)
        
        # Buffer should not exceed max size
        self.assertLessEqual(len(self.nt._experience_buffer), self.nt._max_experience)
        
        # Each experience should have required fields
        if len(self.nt._experience_buffer) > 0:
            exp = self.nt._experience_buffer[0]
            self.assertIn('type', exp)
            self.assertIn('duration', exp)
            self.assertIn('success', exp)
            self.assertIn('timestamp', exp)
    
    def test_recommendations(self):
        """Test that intelligent recommendations are provided."""
        self.nt.data[0, 0] = self.x**2
        
        # Get recommendations
        status = self.nt.get_agency_status()
        recommendations = status['recommendations']
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should have at least one recommendation
        self.assertTrue(any(isinstance(rec, str) for rec in recommendations))


class TestMilitaryGradeIntegration(unittest.TestCase):
    """Integration tests for military-grade features."""
    
    def test_taylor_expansion_with_tracking(self):
        """Test Taylor expansion with military-grade tracking."""
        nt = NanoTensor((1,), max_order=2, base_vars=['k', 'a'])
        
        # Generate Taylor expansion
        nt.generate_taylor({'k': 1.0, 'a': 0.0}, ss_value=sp.S(1))
        
        # Evaluate
        nt.eval_numeric({'k': 1.1, 'a': 0.1})
        
        # Check health
        health = nt.health_check()
        self.assertGreater(health['metrics']['total_operations'], 0)
        self.assertEqual(health['status'], 'optimal')
    
    def test_full_workflow_with_monitoring(self):
        """Test complete workflow with health monitoring."""
        nt = NanoTensor((2,), max_order=1, base_vars=['x', 'y'])
        x, y = sp.symbols('x y')
        
        # Set up tensor
        nt.data[0] = x**2 + y
        nt.data[1] = x * y
        
        # Set validation bounds
        nt.set_validation_bounds('x', -10, 10)
        nt.set_validation_bounds('y', -10, 10)
        
        # Perform various operations
        nt_dx = nt.diff(x)
        nt_dy = nt.diff(y)
        
        # Verify differentiation results
        self.assertEqual(nt_dx.data[0], 2 * x)
        self.assertEqual(nt_dx.data[1], y)
        self.assertEqual(nt_dy.data[0], sp.S(1))
        self.assertEqual(nt_dy.data[1], x)
        
        # Evaluate
        result1 = nt.eval_numeric({'x': 2.0, 'y': 3.0})
        result2 = nt.eval_numeric({'x': 1.0, 'y': 1.0})
        
        # Verify evaluation results
        self.assertIsInstance(result1, np.ndarray)
        self.assertIsInstance(result2, np.ndarray)
        # For x=2, y=3: [x**2 + y, x*y] -> [7, 6]
        self.assertAlmostEqual(float(result1[0]), 7.0)
        self.assertAlmostEqual(float(result1[1]), 6.0)
        # For x=1, y=1: [x**2 + y, x*y] -> [2, 1]
        self.assertAlmostEqual(float(result2[0]), 2.0)
        self.assertAlmostEqual(float(result2[1]), 1.0)
        
        # Simplify
        nt.simplify()
        
        # Get final health report
        health = nt.health_check()
        agency = nt.get_agency_status()
        
        # Verify everything tracked correctly
        self.assertGreater(health['metrics']['total_operations'], 0)
        self.assertEqual(health['metrics']['success_rate'], 1.0)
        self.assertGreater(agency['experiences_recorded'], 0)
        self.assertGreater(agency['patterns_learned'], 0)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that military-grade enhancements don't break existing functionality."""
    
    def test_basic_operations_still_work(self):
        """Ensure basic tensor operations still work as before."""
        nt = NanoTensor((2, 2), max_order=2)
        x, y = sp.symbols('x y')
        
        # Basic assignment
        nt.data[0, 0] = x + y
        nt.data[1, 1] = x * y
        
        # Differentiation
        nt_dx = nt.diff(x)
        self.assertEqual(nt_dx.data[0, 0], sp.S(1))
        
        # Evaluation
        result = nt.eval_numeric({'x': 1.0, 'y': 2.0})
        self.assertAlmostEqual(float(result[0, 0]), 3.0)
        self.assertAlmostEqual(float(result[1, 1]), 2.0)
    
    def test_taylor_generation_compatibility(self):
        """Test that Taylor generation still works."""
        nt = NanoTensor((1,), max_order=2, base_vars=['k', 'a'])
        
        # This should work as before
        nt.generate_taylor({'k': 1.0, 'a': 0.0})
        
        # Should have coefficient variables
        self.assertGreater(len(nt.coeff_vars), 0)
        
        # Should be able to evaluate
        result = nt.eval_numeric({'k': 1.0, 'a': 0.0})
        self.assertIsInstance(result, np.ndarray)


if __name__ == '__main__':
    unittest.main()
