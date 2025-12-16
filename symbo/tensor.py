# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Symbo Enhanced Tensor Module
=============================

Enhanced n-dimensional symbolic tensor with complete tensor operation support.

This module provides:
- True n-dimensional, arbitrary-rank tensor support
- Generalized tensor operations (outer product, trace, contraction)
- Symbolic exactness preservation
- Comprehensive type hints for all dimensions and indices
"""

import sympy as sp
import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union
from functools import lru_cache


class SymbolicTensor:
    """
    Enhanced n-dimensional symbolic tensor with complete tensor algebra support.
    
    This class represents a fully general n-dimensional tensor whose entries are
    SymPy expressions. It supports:
    
    - Arbitrary rank (number of indices)
    - Generalized tensor contractions
    - Outer products
    - Trace operations
    - Einstein summation notation
    - Symbolic differentiation
    
    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the tensor (e.g., (3,) for vector, (3, 3) for matrix, (2, 3, 4) for rank-3)
    name : str, optional
        Human-readable identifier
    dtype : type, optional
        Data type (default: object for symbolic)
    
    Attributes
    ----------
    data : np.ndarray
        Underlying array of SymPy expressions
    shape : Tuple[int, ...]
        Tensor shape
    rank : int
        Number of indices (tensor rank)
    name : str
        Tensor identifier
    
    Examples
    --------
    >>> # Create a 2x3 symbolic matrix
    >>> T = SymbolicTensor((2, 3), name="A")
    >>> T.fill_with_symbols("a")
    >>> print(T)
    
    >>> # Create a rank-3 tensor
    >>> T3 = SymbolicTensor((2, 3, 4), name="T")
    """
    
    def __init__(self, 
                 shape: Tuple[int, ...], 
                 name: str = "T",
                 dtype: type = object):
        """Initialize symbolic tensor with given shape."""
        self.shape: Tuple[int, ...] = shape
        self.rank: int = len(shape)
        self.name: str = name
        self.dtype: type = dtype
        
        # Initialize with symbolic zeros
        self.data: np.ndarray = np.zeros(shape, dtype=dtype)
        if dtype == object:
            self.data.flat[:] = sp.S(0)
        
        self._cache: Dict[str, Any] = {}
    
    @property
    def size(self) -> int:
        """Total number of elements in tensor."""
        return int(np.prod(self.shape))
    
    @property
    def free_symbols(self) -> set:
        """Get all free symbols appearing in the tensor."""
        if 'free_symbols' not in self._cache:
            symbols = set()
            for elem in self.data.flat:
                if hasattr(elem, 'free_symbols'):
                    symbols.update(elem.free_symbols)
            self._cache['free_symbols'] = symbols
        return self._cache['free_symbols']
    
    def fill_with_symbols(self, base_name: str = "x") -> 'SymbolicTensor':
        """
        Fill tensor with indexed symbolic variables.
        
        Parameters
        ----------
        base_name : str
            Base name for symbols (e.g., "a" creates a_0_0, a_0_1, ...)
            
        Returns
        -------
        SymbolicTensor
            Self for chaining
            
        Examples
        --------
        >>> T = SymbolicTensor((2, 3))
        >>> T.fill_with_symbols("A")
        # Creates: A_0_0, A_0_1, A_0_2, A_1_0, A_1_1, A_1_2
        """
        for idx in np.ndindex(self.shape):
            idx_str = "_".join(map(str, idx))
            self.data[idx] = sp.Symbol(f"{base_name}_{idx_str}")
        self._invalidate_cache()
        return self
    
    def fill_with_expression(self, expr: sp.Expr) -> 'SymbolicTensor':
        """
        Fill entire tensor with a single expression (broadcast).
        
        Parameters
        ----------
        expr : sp.Expr
            Expression to broadcast
            
        Returns
        -------
        SymbolicTensor
            Self for chaining
        """
        self.data.flat[:] = expr
        self._invalidate_cache()
        return self
    
    def set_element(self, indices: Tuple[int, ...], value: sp.Expr) -> 'SymbolicTensor':
        """
        Set a single tensor element.
        
        Parameters
        ----------
        indices : Tuple[int, ...]
            Element indices
        value : sp.Expr
            Value to set
            
        Returns
        -------
        SymbolicTensor
            Self for chaining
        """
        self.data[indices] = value
        self._invalidate_cache()
        return self
    
    def get_element(self, indices: Tuple[int, ...]) -> sp.Expr:
        """Get a single tensor element."""
        return self.data[indices]
    
    # ==================== Tensor Operations ====================
    
    def outer_product(self, other: 'SymbolicTensor') -> 'SymbolicTensor':
        """
        Compute outer (tensor) product with another tensor.
        
        The outer product C = A ⊗ B has shape A.shape + B.shape, with
        C[i,j,...,k,l,...] = A[i,j,...] * B[k,l,...]
        
        Parameters
        ----------
        other : SymbolicTensor
            Other tensor
            
        Returns
        -------
        SymbolicTensor
            Outer product with shape = self.shape + other.shape
            
        Examples
        --------
        >>> A = SymbolicTensor((2, 3))
        >>> B = SymbolicTensor((4,))
        >>> C = A.outer_product(B)  # Shape: (2, 3, 4)
        """
        result_shape = self.shape + other.shape
        result = SymbolicTensor(result_shape, name=f"{self.name}⊗{other.name}")
        
        # Compute outer product element-wise
        for idx_self in np.ndindex(self.shape):
            for idx_other in np.ndindex(other.shape):
                combined_idx = idx_self + idx_other
                result.data[combined_idx] = sp.simplify(
                    self.data[idx_self] * other.data[idx_other]
                )
        
        return result
    
    def trace(self, axis1: int = 0, axis2: int = 1) -> 'SymbolicTensor':
        """
        Compute trace over two axes (sum over diagonal).
        
        Parameters
        ----------
        axis1, axis2 : int
            Axes to trace over (must have same dimension)
            
        Returns
        -------
        SymbolicTensor
            Tensor with reduced rank
            
        Examples
        --------
        >>> T = SymbolicTensor((3, 3, 4))
        >>> T_traced = T.trace(0, 1)  # Shape: (4,)
        """
        if self.shape[axis1] != self.shape[axis2]:
            raise ValueError(f"Cannot trace over axes with different dimensions: "
                           f"{self.shape[axis1]} != {self.shape[axis2]}")
        
        # Build new shape by removing traced axes
        new_shape = tuple(s for i, s in enumerate(self.shape) 
                         if i not in (axis1, axis2))
        
        if not new_shape:
            # Scalar result
            result = SymbolicTensor((1,), name=f"Tr({self.name})")
            trace_sum = sp.S(0)
            for i in range(self.shape[axis1]):
                # Build index tuple with i at both traced positions
                idx = [slice(None)] * self.rank
                idx[axis1] = i
                idx[axis2] = i
                idx = tuple(idx)
                
                # Sum over remaining indices
                sub_array = self.data[idx]
                trace_sum += np.sum(sub_array)
            
            result.data[0] = sp.simplify(trace_sum)
            return result
        
        # Non-scalar result
        result = SymbolicTensor(new_shape, name=f"Tr({self.name})")
        
        for idx in np.ndindex(new_shape):
            elem_sum = sp.S(0)
            # Sum over diagonal of traced axes
            for i in range(self.shape[axis1]):
                # Build full index for original tensor
                full_idx = []
                result_idx_pos = 0
                for axis in range(self.rank):
                    if axis == axis1 or axis == axis2:
                        full_idx.append(i)
                    else:
                        full_idx.append(idx[result_idx_pos])
                        result_idx_pos += 1
                
                elem_sum += self.data[tuple(full_idx)]
            
            result.data[idx] = sp.simplify(elem_sum)
        
        return result
    
    def contract(self, 
                 other: 'SymbolicTensor',
                 axes_self: Tuple[int, ...],
                 axes_other: Tuple[int, ...]) -> 'SymbolicTensor':
        """
        Generalized tensor contraction with another tensor.
        
        Contracts specified axes: C[...] = Σ A[...i...] B[...i...]
        
        Parameters
        ----------
        other : SymbolicTensor
            Tensor to contract with
        axes_self : Tuple[int, ...]
            Axes of self to contract over
        axes_other : Tuple[int, ...]
            Axes of other to contract over (must have matching dimensions)
            
        Returns
        -------
        SymbolicTensor
            Contracted tensor
            
        Examples
        --------
        >>> # Matrix multiplication: C_ij = A_ik B_kj
        >>> A = SymbolicTensor((2, 3))
        >>> B = SymbolicTensor((3, 4))
        >>> C = A.contract(B, (1,), (0,))  # Shape: (2, 4)
        
        >>> # Tensor contraction: C_ij = A_ijk B_k
        >>> A = SymbolicTensor((2, 3, 4))
        >>> B = SymbolicTensor((4,))
        >>> C = A.contract(B, (2,), (0,))  # Shape: (2, 3)
        """
        # Validate axes
        if len(axes_self) != len(axes_other):
            raise ValueError("Must contract same number of axes")
        
        for ax_s, ax_o in zip(axes_self, axes_other):
            if self.shape[ax_s] != other.shape[ax_o]:
                raise ValueError(f"Contracted axes must have same dimension: "
                               f"{self.shape[ax_s]} != {other.shape[ax_o]}")
        
        # Compute result shape
        result_shape = tuple(
            s for i, s in enumerate(self.shape) if i not in axes_self
        ) + tuple(
            s for i, s in enumerate(other.shape) if i not in axes_other
        )
        
        if not result_shape:
            result_shape = (1,)
        
        result = SymbolicTensor(result_shape, 
                               name=f"{self.name}·{other.name}")
        
        # Perform contraction
        # This is a generalization of matrix multiplication
        for result_idx in np.ndindex(result_shape):
            elem_sum = sp.S(0)
            
            # Determine contraction range
            contract_ranges = [range(self.shape[ax]) for ax in axes_self]
            
            # Iterate over all contracted indices
            for contract_idx in np.ndindex(tuple(self.shape[ax] for ax in axes_self)):
                # Build indices for self and other
                idx_self = []
                idx_other = []
                
                result_idx_pos = 0
                for i in range(self.rank):
                    if i in axes_self:
                        idx_self.append(contract_idx[axes_self.index(i)])
                    else:
                        if result_idx_pos < len(result_idx):
                            idx_self.append(result_idx[result_idx_pos])
                            result_idx_pos += 1
                
                # Count how many dimensions we've used from result_idx for self
                n_self_result_dims = len([i for i in range(self.rank) if i not in axes_self])
                result_idx_pos = 0
                
                for i in range(other.rank):
                    if i in axes_other:
                        idx_other.append(contract_idx[axes_other.index(i)])
                    else:
                        offset = n_self_result_dims + result_idx_pos
                        if offset < len(result_idx):
                            idx_other.append(result_idx[offset])
                            result_idx_pos += 1
                
                # Add contribution
                elem_sum += self.data[tuple(idx_self)] * other.data[tuple(idx_other)]
            
            result.data[result_idx] = sp.simplify(elem_sum)
        
        return result
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'SymbolicTensor':
        """
        Transpose (permute axes) of tensor.
        
        Parameters
        ----------
        axes : Tuple[int, ...], optional
            New axis order. If None, reverses axes.
            
        Returns
        -------
        SymbolicTensor
            Transposed tensor
            
        Examples
        --------
        >>> T = SymbolicTensor((2, 3, 4))
        >>> T_t = T.transpose((2, 0, 1))  # Shape: (4, 2, 3)
        """
        if axes is None:
            axes = tuple(reversed(range(self.rank)))
        
        new_shape = tuple(self.shape[i] for i in axes)
        result = SymbolicTensor(new_shape, name=f"{self.name}^T")
        result.data = np.transpose(self.data, axes=axes)
        return result
    
    # ==================== Arithmetic Operations ====================
    
    def __add__(self, other: Union['SymbolicTensor', sp.Expr]) -> 'SymbolicTensor':
        """Element-wise addition."""
        if isinstance(other, SymbolicTensor):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            result = SymbolicTensor(self.shape, name=f"{self.name}+{other.name}")
            result.data = np.vectorize(lambda a, b: sp.simplify(a + b))(
                self.data, other.data
            )
        else:
            result = SymbolicTensor(self.shape, name=f"{self.name}+{other}")
            result.data = np.vectorize(lambda a: sp.simplify(a + other))(self.data)
        return result
    
    def __mul__(self, other: Union['SymbolicTensor', sp.Expr]) -> 'SymbolicTensor':
        """Element-wise multiplication."""
        if isinstance(other, SymbolicTensor):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            result = SymbolicTensor(self.shape, name=f"{self.name}*{other.name}")
            result.data = np.vectorize(lambda a, b: sp.simplify(a * b))(
                self.data, other.data
            )
        else:
            result = SymbolicTensor(self.shape, name=f"{self.name}*{other}")
            result.data = np.vectorize(lambda a: sp.simplify(a * other))(self.data)
        return result
    
    def __sub__(self, other: Union['SymbolicTensor', sp.Expr]) -> 'SymbolicTensor':
        """Element-wise subtraction."""
        if isinstance(other, SymbolicTensor):
            return self.__add__(other * -1)
        else:
            return self.__add__(-1 * other)
    
    def __truediv__(self, other: Union['SymbolicTensor', sp.Expr]) -> 'SymbolicTensor':
        """Element-wise division."""
        if isinstance(other, SymbolicTensor):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            result = SymbolicTensor(self.shape, name=f"{self.name}/{other.name}")
            result.data = np.vectorize(lambda a, b: sp.simplify(a / b))(
                self.data, other.data
            )
        else:
            result = SymbolicTensor(self.shape, name=f"{self.name}/{other}")
            result.data = np.vectorize(lambda a: sp.simplify(a / other))(self.data)
        return result
    
    # ==================== Differentiation ====================
    
    def diff(self, var: sp.Symbol, order: int = 1) -> 'SymbolicTensor':
        """
        Differentiate all tensor elements with respect to a variable.
        
        Parameters
        ----------
        var : sp.Symbol
            Variable to differentiate with respect to
        order : int
            Order of differentiation
            
        Returns
        -------
        SymbolicTensor
            Tensor of derivatives
        """
        result = SymbolicTensor(self.shape, name=f"∂{self.name}/∂{var}")
        result.data = np.vectorize(lambda e: sp.diff(e, var, order))(self.data)
        return result
    
    # ==================== Substitution and Evaluation ====================
    
    def subs(self, subs_dict: Dict[sp.Symbol, Any]) -> 'SymbolicTensor':
        """
        Substitute symbols throughout tensor.
        
        Parameters
        ----------
        subs_dict : Dict[sp.Symbol, Any]
            Substitution dictionary
            
        Returns
        -------
        SymbolicTensor
            New tensor with substitutions applied
        """
        result = SymbolicTensor(self.shape, name=self.name)
        result.data = np.vectorize(lambda e: e.subs(subs_dict))(self.data)
        return result
    
    def eval_numeric(self, point: Dict[str, float]) -> np.ndarray:
        """
        Evaluate tensor numerically at a point.
        
        Parameters
        ----------
        point : Dict[str, float]
            Variable values
            
        Returns
        -------
        np.ndarray
            Numeric array with same shape
        """
        subs_dict = {sp.Symbol(k): v for k, v in point.items()}
        result = np.zeros(self.shape, dtype=float)
        for idx in np.ndindex(self.shape):
            result[idx] = float(self.data[idx].subs(subs_dict).evalf())
        return result
    
    def simplify(self) -> 'SymbolicTensor':
        """Simplify all tensor elements."""
        result = SymbolicTensor(self.shape, name=self.name)
        result.data = np.vectorize(sp.simplify)(self.data)
        return result
    
    # ==================== Utility Methods ====================
    
    def _invalidate_cache(self):
        """Clear internal cache."""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return (f"SymbolicTensor(name='{self.name}', shape={self.shape}, "
                f"rank={self.rank})")
    
    def __str__(self) -> str:
        return f"{self.name}{self.shape}:\n{self.data}"
    
    def to_matrix(self) -> sp.Matrix:
        """
        Convert to SymPy Matrix (only for rank-2 tensors).
        
        Returns
        -------
        sp.Matrix
            SymPy matrix representation
            
        Raises
        ------
        ValueError
            If tensor is not rank-2
        """
        if self.rank != 2:
            raise ValueError(f"Can only convert rank-2 tensors to matrices, got rank {self.rank}")
        return sp.Matrix(self.data)
    
    @classmethod
    def from_matrix(cls, matrix: sp.Matrix, name: str = "M") -> 'SymbolicTensor':
        """
        Create SymbolicTensor from SymPy Matrix.
        
        Parameters
        ----------
        matrix : sp.Matrix
            Input matrix
        name : str
            Tensor name
            
        Returns
        -------
        SymbolicTensor
            Tensor representation of matrix
        """
        shape = (matrix.rows, matrix.cols)
        tensor = cls(shape, name=name)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                tensor.data[i, j] = matrix[i, j]
        return tensor


__all__ = ['SymbolicTensor']
