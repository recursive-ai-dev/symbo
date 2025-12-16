# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Symbo Atomic Primitives Module
================================

This module implements the atomic computational components derived from the
deconstruction of 318 classical algorithms across algebra, optimization, dynamics,
and computational geometry.

Each primitive is designed to be:
- Mathematically exact (symbolic correctness)
- Composable (can be combined into higher-level operations)
- Type-safe (with comprehensive type hints)
- Well-tested (unit tests verify semantic correctness)

The primitives are organized into categories:
1. Algebraic operations (addition, multiplication, power)
2. Differential operations (differentiation, gradient)
3. Tensor operations (contraction, outer product)
4. Polynomial operations (expansion, factorization)
5. Matrix operations (determinant, inverse, eigenvalues)
"""

import sympy as sp
import numpy as np
from typing import Union, List, Tuple, Dict, Any, Optional
from sympy import Symbol, Expr, Matrix, Poly


# Type aliases for clarity
SymbolicExpr = Union[sp.Expr, sp.Symbol, int, float]
SymbolicMatrix = sp.Matrix
TensorLike = Union[np.ndarray, List]


class AtomicPrimitives:
    """
    Container class for atomic computational primitives.
    
    These operations form the foundation of all higher-level Symbo functionality.
    """
    
    # ==================== Algebraic Primitives ====================
    
    @staticmethod
    def symbolic_add(a: SymbolicExpr, b: SymbolicExpr) -> sp.Expr:
        """
        Symbolic addition with automatic simplification.
        
        Parameters
        ----------
        a, b : SymbolicExpr
            Symbolic expressions to add
            
        Returns
        -------
        sp.Expr
            Simplified sum a + b
        """
        return sp.simplify(sp.Add(a, b))
    
    @staticmethod
    def symbolic_mul(a: SymbolicExpr, b: SymbolicExpr) -> sp.Expr:
        """
        Symbolic multiplication with automatic simplification.
        
        Parameters
        ----------
        a, b : SymbolicExpr
            Symbolic expressions to multiply
            
        Returns
        -------
        sp.Expr
            Simplified product a * b
        """
        return sp.simplify(sp.Mul(a, b))
    
    @staticmethod
    def symbolic_pow(base: SymbolicExpr, exponent: SymbolicExpr) -> sp.Expr:
        """
        Symbolic exponentiation.
        
        Parameters
        ----------
        base : SymbolicExpr
            Base expression
        exponent : SymbolicExpr
            Exponent expression
            
        Returns
        -------
        sp.Expr
            Simplified expression base^exponent
        """
        return sp.simplify(sp.Pow(base, exponent))
    
    @staticmethod
    def symbolic_div(numerator: SymbolicExpr, denominator: SymbolicExpr) -> sp.Expr:
        """
        Symbolic division with simplification.
        
        Parameters
        ----------
        numerator : SymbolicExpr
            Numerator expression
        denominator : SymbolicExpr
            Denominator expression
            
        Returns
        -------
        sp.Expr
            Simplified rational expression
            
        Raises
        ------
        ValueError
            If denominator is zero
        """
        if denominator == 0:
            raise ValueError("Division by zero")
        return sp.simplify(numerator / denominator)
    
    # ==================== Differential Primitives ====================
    
    @staticmethod
    def symbolic_diff(expr: sp.Expr, var: sp.Symbol, order: int = 1) -> sp.Expr:
        """
        Symbolic differentiation.
        
        Parameters
        ----------
        expr : sp.Expr
            Expression to differentiate
        var : sp.Symbol
            Variable to differentiate with respect to
        order : int, optional
            Order of differentiation (default: 1)
            
        Returns
        -------
        sp.Expr
            Derivative of expr with respect to var
        """
        return sp.diff(expr, var, order)
    
    @staticmethod
    def gradient(expr: sp.Expr, vars_list: List[sp.Symbol]) -> List[sp.Expr]:
        """
        Compute gradient (vector of first partial derivatives).
        
        Parameters
        ----------
        expr : sp.Expr
            Scalar expression
        vars_list : List[sp.Symbol]
            List of variables
            
        Returns
        -------
        List[sp.Expr]
            Gradient vector [∂expr/∂v₁, ∂expr/∂v₂, ...]
        """
        return [sp.diff(expr, var) for var in vars_list]
    
    @staticmethod
    def hessian(expr: sp.Expr, vars_list: List[sp.Symbol]) -> sp.Matrix:
        """
        Compute Hessian matrix (matrix of second partial derivatives).
        
        Parameters
        ----------
        expr : sp.Expr
            Scalar expression
        vars_list : List[sp.Symbol]
            List of variables
            
        Returns
        -------
        sp.Matrix
            Hessian matrix H[i,j] = ∂²expr/(∂vᵢ∂vⱼ)
        """
        n = len(vars_list)
        H = sp.zeros(n, n)
        for i, vi in enumerate(vars_list):
            for j, vj in enumerate(vars_list):
                H[i, j] = sp.diff(expr, vi, vj)
        return H
    
    @staticmethod
    def jacobian(expr_list: List[sp.Expr], vars_list: List[sp.Symbol]) -> sp.Matrix:
        """
        Compute Jacobian matrix of a vector-valued function.
        
        Parameters
        ----------
        expr_list : List[sp.Expr]
            List of expressions [f₁, f₂, ...]
        vars_list : List[sp.Symbol]
            List of variables
            
        Returns
        -------
        sp.Matrix
            Jacobian matrix J[i,j] = ∂fᵢ/∂vⱼ
        """
        m = len(expr_list)
        n = len(vars_list)
        J = sp.zeros(m, n)
        for i, fi in enumerate(expr_list):
            for j, vj in enumerate(vars_list):
                J[i, j] = sp.diff(fi, vj)
        return J
    
    # ==================== Tensor Primitives ====================
    
    @staticmethod
    def tensor_contraction(A: np.ndarray, B: np.ndarray, 
                          axes_A: Tuple[int, ...], 
                          axes_B: Tuple[int, ...]) -> np.ndarray:
        """
        Generalized tensor contraction C_{ijk...} = Σ A_{...m...} B_{...m...}
        
        Parameters
        ----------
        A, B : np.ndarray
            Symbolic tensors (arrays of SymPy expressions)
        axes_A : Tuple[int, ...]
            Axes of A to contract over
        axes_B : Tuple[int, ...]
            Axes of B to contract over
            
        Returns
        -------
        np.ndarray
            Contracted tensor
            
        Examples
        --------
        >>> # Matrix multiplication: C_ij = A_ik B_kj
        >>> C = tensor_contraction(A, B, (1,), (0,))
        """
        # Use Einstein summation for contraction
        return np.tensordot(A, B, axes=(axes_A, axes_B))
    
    @staticmethod
    def outer_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute outer product C_{ijk...} = A_{ij...} B_{k...}
        
        Parameters
        ----------
        A, B : np.ndarray
            Input tensors
            
        Returns
        -------
        np.ndarray
            Outer product with shape = A.shape + B.shape
        """
        return np.outer(A.flatten(), B.flatten()).reshape(A.shape + B.shape)
    
    @staticmethod
    def tensor_trace(A: np.ndarray, axis1: int = 0, axis2: int = 1) -> np.ndarray:
        """
        Compute trace by summing over two axes.
        
        Parameters
        ----------
        A : np.ndarray
            Input tensor
        axis1, axis2 : int
            Axes to trace over
            
        Returns
        -------
        np.ndarray
            Tensor with reduced dimensionality
        """
        return np.trace(A, axis1=axis1, axis2=axis2)
    
    @staticmethod
    def symbolic_tensor_product(A: SymbolicMatrix, B: SymbolicMatrix) -> SymbolicMatrix:
        """
        Kronecker (tensor) product of symbolic matrices.
        
        Parameters
        ----------
        A, B : sp.Matrix
            Symbolic matrices
            
        Returns
        -------
        sp.Matrix
            Kronecker product A ⊗ B
        """
        return sp.kronecker_product(A, B)
    
    # ==================== Polynomial Primitives ====================
    
    @staticmethod
    def polynomial_expand(expr: sp.Expr) -> sp.Expr:
        """
        Expand polynomial expression.
        
        Parameters
        ----------
        expr : sp.Expr
            Expression to expand
            
        Returns
        -------
        sp.Expr
            Expanded form
        """
        return sp.expand(expr)
    
    @staticmethod
    def polynomial_factor(expr: sp.Expr) -> sp.Expr:
        """
        Factor polynomial expression.
        
        Parameters
        ----------
        expr : sp.Expr
            Expression to factor
            
        Returns
        -------
        sp.Expr
            Factored form
        """
        return sp.factor(expr)
    
    @staticmethod
    def polynomial_collect(expr: sp.Expr, var: sp.Symbol) -> sp.Expr:
        """
        Collect terms by powers of variable.
        
        Parameters
        ----------
        expr : sp.Expr
            Expression to collect
        var : sp.Symbol
            Variable to collect by
            
        Returns
        -------
        sp.Expr
            Collected form
        """
        return sp.collect(expr, var)
    
    @staticmethod
    def polynomial_degree(expr: sp.Expr, var: sp.Symbol) -> int:
        """
        Get degree of polynomial in variable.
        
        Parameters
        ----------
        expr : sp.Expr
            Polynomial expression
        var : sp.Symbol
            Variable
            
        Returns
        -------
        int
            Degree of polynomial
        """
        poly = sp.Poly(expr, var)
        return poly.degree()
    
    @staticmethod
    def polynomial_coeffs(expr: sp.Expr, var: sp.Symbol) -> List[sp.Expr]:
        """
        Extract coefficients of polynomial.
        
        Parameters
        ----------
        expr : sp.Expr
            Polynomial expression
        var : sp.Symbol
            Variable
            
        Returns
        -------
        List[sp.Expr]
            List of coefficients [a₀, a₁, a₂, ...] for a₀ + a₁x + a₂x² + ...
        """
        poly = sp.Poly(expr, var)
        return poly.all_coeffs()
    
    # ==================== Matrix Primitives ====================
    
    @staticmethod
    def matrix_det(M: SymbolicMatrix) -> sp.Expr:
        """
        Compute determinant of symbolic matrix.
        
        Parameters
        ----------
        M : sp.Matrix
            Square matrix
            
        Returns
        -------
        sp.Expr
            Determinant
        """
        return M.det()
    
    @staticmethod
    def matrix_inv(M: SymbolicMatrix) -> SymbolicMatrix:
        """
        Compute inverse of symbolic matrix.
        
        Parameters
        ----------
        M : sp.Matrix
            Invertible square matrix
            
        Returns
        -------
        sp.Matrix
            Matrix inverse
            
        Raises
        ------
        ValueError
            If matrix is singular
        """
        if M.det() == 0:
            raise ValueError("Matrix is singular and cannot be inverted")
        return M.inv()
    
    @staticmethod
    def matrix_eigenvals(M: SymbolicMatrix) -> Dict[sp.Expr, int]:
        """
        Compute eigenvalues of symbolic matrix.
        
        Parameters
        ----------
        M : sp.Matrix
            Square matrix
            
        Returns
        -------
        Dict[sp.Expr, int]
            Dictionary mapping eigenvalues to their multiplicities
        """
        return M.eigenvals()
    
    @staticmethod
    def matrix_eigenvects(M: SymbolicMatrix) -> List[Tuple[sp.Expr, int, List[sp.Matrix]]]:
        """
        Compute eigenvectors of symbolic matrix.
        
        Parameters
        ----------
        M : sp.Matrix
            Square matrix
            
        Returns
        -------
        List[Tuple[sp.Expr, int, List[sp.Matrix]]]
            List of (eigenvalue, multiplicity, [eigenvectors])
        """
        return M.eigenvects()
    
    # ==================== Integration Primitives ====================
    
    @staticmethod
    def symbolic_integrate(expr: sp.Expr, var: sp.Symbol, 
                          lower: Optional[SymbolicExpr] = None,
                          upper: Optional[SymbolicExpr] = None) -> sp.Expr:
        """
        Symbolic integration.
        
        Parameters
        ----------
        expr : sp.Expr
            Expression to integrate
        var : sp.Symbol
            Integration variable
        lower, upper : SymbolicExpr, optional
            Integration bounds (if both provided, computes definite integral)
            
        Returns
        -------
        sp.Expr
            Integral (indefinite if bounds not provided, definite otherwise)
        """
        if lower is not None and upper is not None:
            return sp.integrate(expr, (var, lower, upper))
        return sp.integrate(expr, var)
    
    # ==================== Substitution and Evaluation ====================
    
    @staticmethod
    def substitute(expr: sp.Expr, subs_dict: Dict[sp.Symbol, SymbolicExpr]) -> sp.Expr:
        """
        Substitute variables in expression.
        
        Parameters
        ----------
        expr : sp.Expr
            Expression
        subs_dict : Dict[sp.Symbol, SymbolicExpr]
            Substitution dictionary
            
        Returns
        -------
        sp.Expr
            Expression with substitutions applied
        """
        return expr.subs(subs_dict)
    
    @staticmethod
    def evaluate_numeric(expr: sp.Expr, subs_dict: Dict[sp.Symbol, float]) -> float:
        """
        Numerically evaluate expression at a point.
        
        Parameters
        ----------
        expr : sp.Expr
            Expression to evaluate
        subs_dict : Dict[sp.Symbol, float]
            Variable values
            
        Returns
        -------
        float
            Numeric result
        """
        return float(expr.subs(subs_dict).evalf())
    
    # ==================== Simplification Primitives ====================
    
    @staticmethod
    def simplify(expr: sp.Expr) -> sp.Expr:
        """General simplification."""
        return sp.simplify(expr)
    
    @staticmethod
    def trigsimp(expr: sp.Expr) -> sp.Expr:
        """Trigonometric simplification."""
        return sp.trigsimp(expr)
    
    @staticmethod
    def ratsimp(expr: sp.Expr) -> sp.Expr:
        """Rational simplification."""
        return sp.ratsimp(expr)
    
    @staticmethod
    def cancel(expr: sp.Expr) -> sp.Expr:
        """Cancel common factors."""
        return sp.cancel(expr)


# Convenience functions that directly use the primitives
def add(a: SymbolicExpr, b: SymbolicExpr) -> sp.Expr:
    """Convenience wrapper for symbolic addition."""
    return AtomicPrimitives.symbolic_add(a, b)


def mul(a: SymbolicExpr, b: SymbolicExpr) -> sp.Expr:
    """Convenience wrapper for symbolic multiplication."""
    return AtomicPrimitives.symbolic_mul(a, b)


def diff(expr: sp.Expr, var: sp.Symbol, order: int = 1) -> sp.Expr:
    """Convenience wrapper for symbolic differentiation."""
    return AtomicPrimitives.symbolic_diff(expr, var, order)


__all__ = [
    'AtomicPrimitives',
    'add',
    'mul',
    'diff',
    'SymbolicExpr',
    'SymbolicMatrix',
    'TensorLike',
]
