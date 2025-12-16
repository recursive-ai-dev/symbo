# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
WASM-Friendly Execution Bindings
=================================

This module provides WASM-compatible interfaces for Symbo functionality,
ensuring all data structures and function signatures can be efficiently
serialized/deserialized for browser-side execution.

Key Features:
- WASM-compatible function signatures (primitives only)
- MessagePack serialization for complex types
- Browser-friendly JSON interfaces
- Efficient data transfer
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Any, Union, Optional
import json

try:
    import msgpack
except ImportError:
    msgpack = None


class WASMCompatibilityError(Exception):
    """Raised when data cannot be made WASM-compatible."""
    pass


class WASMInterface:
    """
    Interface for WASM-compatible Symbo operations.
    
    All methods accept and return JSON-serializable data types
    for easy transfer across the WASM boundary.
    """
    
    @staticmethod
    def eval_expression(expr_str: str, var_values: Dict[str, float]) -> float:
        """
        Evaluate symbolic expression with given variable values.
        
        WASM-compatible signature: string and dict of floats in/out.
        
        Parameters
        ----------
        expr_str : str
            SymPy-parsable expression string
        var_values : Dict[str, float]
            Variable values
            
        Returns
        -------
        float
            Numeric result
            
        Examples
        --------
        >>> result = WASMInterface.eval_expression("x**2 + y", {"x": 2.0, "y": 3.0})
        >>> print(result)  # 7.0
        """
        try:
            expr = sp.sympify(expr_str)
            subs_dict = {sp.Symbol(k): v for k, v in var_values.items()}
            return float(expr.subs(subs_dict).evalf())
        except Exception as e:
            raise WASMCompatibilityError(f"Expression evaluation failed: {e}")
    
    @staticmethod
    def differentiate(expr_str: str, var: str, order: int = 1) -> str:
        """
        Compute symbolic derivative.
        
        Parameters
        ----------
        expr_str : str
            Expression string
        var : str
            Variable name
        order : int
            Derivative order
            
        Returns
        -------
        str
            Derivative expression as string
        """
        try:
            expr = sp.sympify(expr_str)
            var_sym = sp.Symbol(var)
            deriv = sp.diff(expr, var_sym, order)
            return str(deriv)
        except Exception as e:
            raise WASMCompatibilityError(f"Differentiation failed: {e}")
    
    @staticmethod
    def simplify(expr_str: str) -> str:
        """
        Simplify expression.
        
        Parameters
        ----------
        expr_str : str
            Expression string
            
        Returns
        -------
        str
            Simplified expression
        """
        try:
            expr = sp.sympify(expr_str)
            simplified = sp.simplify(expr)
            return str(simplified)
        except Exception as e:
            raise WASMCompatibilityError(f"Simplification failed: {e}")
    
    @staticmethod
    def solve_equation(eq_str: str, var: str) -> List[str]:
        """
        Solve equation symbolically.
        
        Parameters
        ----------
        eq_str : str
            Equation string (assumed = 0)
        var : str
            Variable to solve for
            
        Returns
        -------
        List[str]
            Solutions as strings
        """
        try:
            eq = sp.sympify(eq_str)
            var_sym = sp.Symbol(var)
            solutions = sp.solve(eq, var_sym)
            return [str(sol) for sol in solutions]
        except Exception as e:
            raise WASMCompatibilityError(f"Equation solving failed: {e}")
    
    @staticmethod
    def expand_taylor(expr_str: str,
                     var: str,
                     center: float,
                     order: int) -> Dict[str, Any]:
        """
        Compute Taylor expansion.
        
        Parameters
        ----------
        expr_str : str
            Expression string
        var : str
            Variable name
        center : float
            Expansion point
        order : int
            Maximum order
            
        Returns
        -------
        Dict[str, Any]
            Taylor expansion info including coefficients and expression
        """
        try:
            expr = sp.sympify(expr_str)
            var_sym = sp.Symbol(var)
            
            # Compute Taylor series
            taylor = expr.series(var_sym, center, order + 1).removeO()
            
            # Extract coefficients
            coeffs = []
            for i in range(order + 1):
                coeff = sp.diff(expr, var_sym, i).subs(var_sym, center) / sp.factorial(i)
                coeffs.append(float(coeff.evalf()))
            
            return {
                "expression": str(taylor),
                "coefficients": coeffs,
                "center": center,
                "order": order
            }
        except Exception as e:
            raise WASMCompatibilityError(f"Taylor expansion failed: {e}")
    
    @staticmethod
    def compute_jacobian(expr_strs: List[str],
                        var_names: List[str]) -> List[List[str]]:
        """
        Compute Jacobian matrix.
        
        Parameters
        ----------
        expr_strs : List[str]
            List of expression strings
        var_names : List[str]
            Variable names
            
        Returns
        -------
        List[List[str]]
            Jacobian matrix as nested list of strings
        """
        try:
            exprs = [sp.sympify(e) for e in expr_strs]
            vars_syms = [sp.Symbol(v) for v in var_names]
            
            jacobian = []
            for expr in exprs:
                row = []
                for var in vars_syms:
                    partial = sp.diff(expr, var)
                    row.append(str(partial))
                jacobian.append(row)
            
            return jacobian
        except Exception as e:
            raise WASMCompatibilityError(f"Jacobian computation failed: {e}")


class MessagePackSerializer:
    """
    MessagePack serialization for Symbo types.
    
    Provides efficient binary serialization for transferring
    complex symbolic structures to/from WASM.
    """
    
    @staticmethod
    def check_available():
        """Check if msgpack is available."""
        if msgpack is None:
            raise ImportError("msgpack is not installed. Install with: pip install msgpack")
    
    @staticmethod
    def serialize_expression(expr: sp.Expr) -> bytes:
        """
        Serialize SymPy expression to MessagePack.
        
        Parameters
        ----------
        expr : sp.Expr
            Expression to serialize
            
        Returns
        -------
        bytes
            MessagePack-encoded data
        """
        MessagePackSerializer.check_available()
        
        data = {
            "type": "expression",
            "string": str(expr),
            "latex": sp.latex(expr),
            "free_symbols": [str(s) for s in expr.free_symbols]
        }
        
        return msgpack.packb(data, use_bin_type=True)
    
    @staticmethod
    def deserialize_expression(data: bytes) -> sp.Expr:
        """
        Deserialize expression from MessagePack.
        
        Parameters
        ----------
        data : bytes
            MessagePack data
            
        Returns
        -------
        sp.Expr
            Reconstructed expression
        """
        MessagePackSerializer.check_available()
        
        obj = msgpack.unpackb(data, raw=False)
        return sp.sympify(obj["string"])
    
    @staticmethod
    def serialize_tensor(tensor_data: np.ndarray, shape: tuple) -> bytes:
        """
        Serialize symbolic tensor.
        
        Parameters
        ----------
        tensor_data : np.ndarray
            Array of expressions
        shape : tuple
            Tensor shape
            
        Returns
        -------
        bytes
            MessagePack data
        """
        MessagePackSerializer.check_available()
        
        # Convert expressions to strings
        flat_strings = [str(e) for e in tensor_data.flat]
        
        data = {
            "type": "tensor",
            "shape": list(shape),
            "data": flat_strings
        }
        
        return msgpack.packb(data, use_bin_type=True)
    
    @staticmethod
    def deserialize_tensor(data: bytes) -> tuple:
        """
        Deserialize tensor from MessagePack.
        
        Parameters
        ----------
        data : bytes
            MessagePack data
            
        Returns
        -------
        tuple
            (tensor_data, shape)
        """
        MessagePackSerializer.check_available()
        
        obj = msgpack.unpackb(data, raw=False)
        shape = tuple(obj["shape"])
        
        # Reconstruct expressions
        flat_exprs = [sp.sympify(s) for s in obj["data"]]
        tensor_data = np.array(flat_exprs, dtype=object).reshape(shape)
        
        return tensor_data, shape
    
    @staticmethod
    def serialize_solution(solution_dict: Dict[str, Any]) -> bytes:
        """
        Serialize solution dictionary.
        
        Parameters
        ----------
        solution_dict : Dict[str, Any]
            Solution data
            
        Returns
        -------
        bytes
            MessagePack data
        """
        MessagePackSerializer.check_available()
        
        # Convert symbolic values to strings
        serializable = {}
        for key, value in solution_dict.items():
            if isinstance(value, sp.Basic):
                serializable[key] = str(value)
            elif isinstance(value, dict):
                serializable[key] = {
                    str(k): str(v) if isinstance(v, sp.Basic) else v
                    for k, v in value.items()
                }
            else:
                serializable[key] = value
        
        return msgpack.packb(serializable, use_bin_type=True)


def create_browser_test_payload() -> Dict[str, Any]:
    """
    Create a test payload for browser-side execution.
    
    Returns a JSON-serializable dict that can be used to verify
    WASM functionality in a browser environment.
    
    Returns
    -------
    Dict[str, Any]
        Test payload with sample expressions and expected results
    """
    x, y = sp.symbols('x y')
    
    # Sample expression
    expr = x**2 + 2*x*y + y**2
    
    return {
        "test_cases": [
            {
                "name": "evaluate_expression",
                "expr": str(expr),
                "variables": {"x": 2.0, "y": 3.0},
                "expected": 25.0
            },
            {
                "name": "differentiate",
                "expr": str(expr),
                "var": "x",
                "order": 1,
                "expected": "2*x + 2*y"
            },
            {
                "name": "simplify",
                "expr": "(x + y)**2",
                "expected": "x**2 + 2*x*y + y**2"
            },
            {
                "name": "solve",
                "equation": "x**2 - 4",
                "var": "x",
                "expected": ["-2", "2"]
            }
        ],
        "interface_methods": [
            "eval_expression",
            "differentiate",
            "simplify",
            "solve_equation",
            "expand_taylor",
            "compute_jacobian"
        ]
    }


def verify_wasm_compatibility(obj: Any) -> bool:
    """
    Verify that an object can be transferred across WASM boundary.
    
    Parameters
    ----------
    obj : Any
        Object to check
        
    Returns
    -------
    bool
        True if WASM-compatible
    """
    # Check if JSON-serializable
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


__all__ = [
    'WASMInterface',
    'MessagePackSerializer',
    'WASMCompatibilityError',
    'create_browser_test_payload',
    'verify_wasm_compatibility',
]
