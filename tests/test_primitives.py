# Copyright 2025
# Unit Tests for Atomic Primitives

import unittest
import sympy as sp
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from symbo.primitives import AtomicPrimitives, add, mul, diff


class TestAlgebraicPrimitives(unittest.TestCase):
    """Test algebraic operations."""
    
    def setUp(self):
        self.x, self.y = sp.symbols('x y')
        self.prims = AtomicPrimitives()
    
    def test_symbolic_add(self):
        """Test symbolic addition."""
        result = self.prims.symbolic_add(self.x, self.y)
        expected = self.x + self.y
        self.assertEqual(result, expected)
    
    def test_symbolic_mul(self):
        """Test symbolic multiplication."""
        result = self.prims.symbolic_mul(self.x, self.y)
        expected = self.x * self.y
        self.assertEqual(result, expected)


class TestDifferentialPrimitives(unittest.TestCase):
    """Test differential operations."""
    
    def setUp(self):
        self.x, self.y = sp.symbols('x y')
        self.prims = AtomicPrimitives()
    
    def test_symbolic_diff(self):
        """Test symbolic differentiation."""
        expr = self.x**2 + 2*self.x*self.y
        result = self.prims.symbolic_diff(expr, self.x)
        expected = 2*self.x + 2*self.y
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
