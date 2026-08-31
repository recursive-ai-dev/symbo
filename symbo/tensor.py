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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .nanotensor import NanoTensor


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
    >>> _ = T.fill_with_symbols("a")
    >>> T.get_element((1, 2))
    a_1_2

    >>> # Create a rank-3 tensor
    >>> T3 = SymbolicTensor((2, 3, 4), name="T")
    >>> T3.rank, T3.size
    (3, 24)
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 name: str = "T",
                 dtype: type = object):
        """Initialize symbolic tensor with given shape."""
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        try:
            shape = tuple(int(d) for d in shape)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"shape must be a tuple of ints, got {shape!r}") from exc
        if not shape or any(d <= 0 for d in shape):
            raise ValueError(
                f"shape must be non-empty with positive dimensions, got {shape}"
            )
        self.shape: Tuple[int, ...] = shape
        self.rank: int = len(shape)
        self.name: str = name
        self.dtype: type = dtype

        # Initialize with symbolic zeros
        self.data: np.ndarray = np.zeros(shape, dtype=dtype)
        if dtype is object:
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
        >>> _ = T.fill_with_symbols("A")
        >>> sorted(str(s) for s in T.free_symbols)
        ['A_0_0', 'A_0_1', 'A_0_2', 'A_1_0', 'A_1_1', 'A_1_2']
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
                    if axis in (axis1, axis2):
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
        if len(axes_self) != len(axes_other):
            raise ValueError("Must contract same number of axes")

        # Normalise negative axes and reject duplicates: the pairing below is
        # positional, so ``(-1,)`` and ``(rank-1,)`` must describe the same axis.
        axes_self = tuple(int(a) % self.rank for a in axes_self)
        axes_other = tuple(int(a) % other.rank for a in axes_other)
        if len(set(axes_self)) != len(axes_self) or len(set(axes_other)) != len(axes_other):
            raise ValueError("Cannot contract the same axis twice")

        for ax_s, ax_o in zip(axes_self, axes_other, strict=True):
            if self.shape[ax_s] != other.shape[ax_o]:
                raise ValueError(f"Contracted axes must have same dimension: "
                               f"{self.shape[ax_s]} != {other.shape[ax_o]}")

        free_self = [i for i in range(self.rank) if i not in set(axes_self)]
        free_other = [i for i in range(other.rank) if i not in set(axes_other)]

        result_shape = tuple(self.shape[i] for i in free_self) + \
            tuple(other.shape[i] for i in free_other)
        if not result_shape:  # full contraction -> scalar in a (1,) tensor
            result_shape = (1,)

        result = SymbolicTensor(result_shape,
                                name=f"{self.name}·{other.name}")

        # Contract over the paired axes; entry k of `contract_idx` belongs to the
        # k-th pair (axes_self[k], axes_other[k]) of *both* operands, so any
        # ordering of the axis tuples gives the same result.
        contract_shape = tuple(self.shape[a] for a in axes_self)
        for result_idx in np.ndindex(result_shape):
            elem_sum = sp.S(0)

            idx_self: List[Optional[int]] = [None] * self.rank
            idx_other: List[Optional[int]] = [None] * other.rank
            for pos, ax in enumerate(free_self):
                idx_self[ax] = result_idx[pos]
            for pos, ax in enumerate(free_other):
                idx_other[ax] = result_idx[len(free_self) + pos]

            for contract_idx in np.ndindex(*contract_shape) if contract_shape else [()]:
                for k, (ax_s, ax_o) in enumerate(zip(axes_self, axes_other, strict=True)):
                    idx_self[ax_s] = contract_idx[k]
                    idx_other[ax_o] = contract_idx[k]
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
            return self.__add__(other * (-1))
        else:
            return self.__add__((-1) * other)

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

    # ==================== Algebraic aliases & matmul ====================

    @classmethod
    def create(cls,
               shape: Tuple[int, ...],
               name: str = "T",
               fill: Union[str, sp.Expr, None] = None) -> 'SymbolicTensor':
        """
        Concise constructor, optionally filling the new tensor.

        ``fill`` accepts a symbol prefix (as in :meth:`fill_with_symbols`) or a
        single SymPy expression broadcast to every entry
        (as in :meth:`fill_with_expression`); ``None`` leaves the zeros.

        Examples
        --------
        >>> a = SymbolicTensor.create((2, 2), name="a", fill="a")
        >>> a.shape
        (2, 2)
        """
        tensor = cls(shape, name=name)
        if isinstance(fill, str):
            tensor.fill_with_symbols(fill)
        elif fill is not None:
            tensor.fill_with_expression(fill)
        return tensor

    def add(self, other: Union['SymbolicTensor', sp.Expr]) -> 'SymbolicTensor':
        """Element-wise sum; the method form of ``self + other``."""
        return self + other

    def sub(self, other: Union['SymbolicTensor', sp.Expr]) -> 'SymbolicTensor':
        """Element-wise difference; the method form of ``self - other``."""
        return self - other

    def mul(self, other: Union['SymbolicTensor', sp.Expr]) -> 'SymbolicTensor':
        """Element-wise product; the method form of ``self * other``."""
        return self * other

    def div(self, other: Union['SymbolicTensor', sp.Expr]) -> 'SymbolicTensor':
        """Element-wise quotient; the method form of ``self / other``."""
        return self / other

    def outer(self, other: 'SymbolicTensor') -> 'SymbolicTensor':
        """Outer product; the short alias of :meth:`outer_product`."""
        return self.outer_product(other)

    def __matmul__(self, other: 'SymbolicTensor') -> 'SymbolicTensor':
        """
        Tensor contraction over the shared index (``self @ other``).

        For two 2D operands this is the ordinary matrix product; higher ranks
        contract the last axis of ``self`` with the first axis of ``other``
        (the numpy einsum convention ``...i,i... -> ...``).
        """
        if not isinstance(other, SymbolicTensor):
            return NotImplemented
        if self.rank < 2 or other.rank < 2:
            raise ValueError(
                f"@ needs operands of rank >= 2, got {self.shape} and {other.shape}"
            )
        if self.shape[-1] != other.shape[0]:
            raise ValueError(
                f"Shape mismatch for @: {self.shape} and {other.shape} "
                f"(trailing dim {self.shape[-1]} != leading dim {other.shape[0]})"
            )
        return self.contract(other, (self.rank - 1,), (0,))

    def matmul(self, other: 'SymbolicTensor') -> 'SymbolicTensor':
        """Contraction over the shared index; the method form of ``self @ other``."""
        return self.__matmul__(other)

    def to_nanotensor(self, max_order: int = 2) -> "NanoTensor":
        """
        View this tensor as a :class:`~symbo.nanotensor.NanoTensor`.

        The base variables of the result are this tensor's free symbols in
        sorted-name order, which keeps the mapping deterministic across runs.
        The two objects do not share memory.
        """
        from .nanotensor import NanoTensor

        names = sorted(str(sym) for sym in self.free_symbols)
        nt = NanoTensor(self.shape, max_order=max_order,
                        base_vars=names or ['x'], name=self.name)
        nt.data = np.array(self.data, dtype=object)
        nt._invalidate_caches()
        return nt

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
        subs_dict = {
            (k if isinstance(k, sp.Symbol) else sp.Symbol(k)): v
            for k, v in point.items()
        }
        result = np.zeros(self.shape, dtype=float)
        for idx in np.ndindex(self.shape):
            elem = self.data[idx]
            value = elem.subs(subs_dict) if hasattr(elem, 'subs') else elem
            if hasattr(value, 'evalf'):
                value = value.evalf()
            try:
                result[idx] = float(value)
            except (TypeError, ValueError) as exc:
                unresolved = sorted(
                    str(s) for s in getattr(value, 'free_symbols', set())
                )
                raise ValueError(
                    f"eval_numeric(): element {idx} of tensor '{self.name}' is not "
                    f"numeric after substitution (unresolved: {unresolved or value!r})"
                ) from exc
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

    @classmethod
    def from_nested(cls, nested: Any, name: str = "T") -> 'SymbolicTensor':
        """
        Build a tensor from nested Python lists, inferring the shape.

        Every leaf is sympified, so plain numbers and SymPy expressions can be
        mixed. Ragged input is rejected instead of silently truncated.

        Parameters
        ----------
        nested : list, tuple or ndarray
            ``[[x, 1], [y, x*y]]`` for a rank-2 tensor, ``[x]`` for rank-1.
        name : str
            Tensor name.

        Returns
        -------
        SymbolicTensor

        Examples
        --------
        >>> import sympy as sp
        >>> x, y = sp.symbols('x y')
        >>> T = SymbolicTensor.from_nested([[x, 1], [y, x * y]], name="A")
        >>> T.shape
        (2, 2)
        >>> T.get_element((1, 0))
        y
        """
        try:
            array = np.asarray(nested, dtype=object)
        except ValueError as exc:  # e.g. [[x, 1], [y]]
            raise ValueError(f"nested data must be rectangular: {exc}") from exc
        if array.ndim == 0:
            raise ValueError("from_nested() needs at least one dimension; "
                             "use SymbolicTensor((), ...) for a scalar tensor")
        tensor = cls(tuple(int(d) for d in array.shape), name=name)
        for index in np.ndindex(array.shape):
            tensor.data[index] = sp.sympify(array[index])
        return tensor

    def to_nested(self) -> Any:
        """
        Return the entries as nested Python lists (inverse of :meth:`from_nested`).

        Examples
        --------
        >>> import sympy as sp
        >>> x = sp.Symbol('x')
        >>> SymbolicTensor.from_nested([[x, 1]], name="A").to_nested()
        [[x, 1]]
        """
        return self.data.tolist()


__all__ = ['SymbolicTensor']
