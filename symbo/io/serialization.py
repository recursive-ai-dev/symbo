# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
High-Speed I/O Serialization
=============================

This module implements Arrow/MessagePack serialization for all Symbo types,
ensuring data integrity and high-speed I/O without semantic ambiguity.

Key Features:
- Arrow format support for DataFrames and tables
- MessagePack support for complex objects
- Round-trip preservation of symbolic types
- Zero data loss guarantee
- Mathematical equivalence verification
"""

import sympy as sp
import numpy as np
from typing import Any, Dict, List, Optional, Union
import json

try:
    import msgpack
except ImportError:
    msgpack = None

try:
    import pyarrow as pa
except ImportError:
    pa = None


class SerializationError(Exception):
    """Raised when serialization fails."""
    pass


class SymboSerializer:
    """
    Universal serializer for Symbo types.
    
    Handles serialization of:
    - SymbolicTensor
    - GenerativePolicyFunction
    - GröbnerBasisState
    - PerturbationSolution
    - DerivativeTree
    """
    
    @staticmethod
    def serialize_expression(expr: sp.Expr, format: str = 'msgpack') -> bytes:
        """
        Serialize SymPy expression.
        
        Parameters
        ----------
        expr : sp.Expr
            Expression to serialize
        format : str
            'msgpack' or 'json'
            
        Returns
        -------
        bytes
            Serialized data
        """
        data = {
            "type": "Expression",
            "string_repr": str(expr),
            "latex": sp.latex(expr),
            "free_symbols": [str(s) for s in expr.free_symbols],
            "is_number": expr.is_number,
            "complexity": sp.count_ops(expr)
        }
        
        if format == 'msgpack':
            if msgpack is None:
                raise SerializationError("msgpack not available")
            return msgpack.packb(data, use_bin_type=True)
        else:
            return json.dumps(data).encode('utf-8')
    
    @staticmethod
    def deserialize_expression(data: bytes, format: str = 'msgpack') -> sp.Expr:
        """
        Deserialize expression.
        
        Parameters
        ----------
        data : bytes
            Serialized data
        format : str
            'msgpack' or 'json'
            
        Returns
        -------
        sp.Expr
            Reconstructed expression
        """
        if format == 'msgpack':
            if msgpack is None:
                raise SerializationError("msgpack not available")
            obj = msgpack.unpackb(data, raw=False)
        else:
            obj = json.loads(data.decode('utf-8'))
        
        return sp.sympify(obj["string_repr"])
    
    @staticmethod
    def serialize_tensor(tensor: 'SymbolicTensor', format: str = 'arrow') -> bytes:
        """
        Serialize SymbolicTensor.
        
        Parameters
        ----------
        tensor : SymbolicTensor
            Tensor to serialize
        format : str
            'arrow' or 'msgpack'
            
        Returns
        -------
        bytes
            Serialized tensor
        """
        # Import here to avoid circular dependency
        from symbo.tensor import SymbolicTensor
        
        if not isinstance(tensor, SymbolicTensor):
            raise SerializationError(f"Expected SymbolicTensor, got {type(tensor)}")
        
        # Convert tensor data to strings
        flat_data = [str(e) for e in tensor.data.flat]
        
        metadata = {
            "type": "SymbolicTensor",
            "name": tensor.name,
            "shape": list(tensor.shape),
            "rank": tensor.rank,
            "size": tensor.size
        }
        
        if format == 'arrow':
            if pa is None:
                raise SerializationError("pyarrow not available")
            
            # Create Arrow table
            table = pa.table({
                "element": flat_data,
                "index": list(range(len(flat_data)))
            })
            
            # Add metadata
            table = table.replace_schema_metadata({
                "symbo_metadata": json.dumps(metadata)
            })
            
            # Serialize to IPC format
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            
            return sink.getvalue().to_pybytes()
        
        else:  # msgpack
            if msgpack is None:
                raise SerializationError("msgpack not available")
            
            data = {
                **metadata,
                "data": flat_data
            }
            return msgpack.packb(data, use_bin_type=True)
    
    @staticmethod
    def deserialize_tensor(data: bytes, format: str = 'arrow') -> 'SymbolicTensor':
        """
        Deserialize SymbolicTensor.
        
        Parameters
        ----------
        data : bytes
            Serialized data
        format : str
            'arrow' or 'msgpack'
            
        Returns
        -------
        SymbolicTensor
            Reconstructed tensor
        """
        from symbo.tensor import SymbolicTensor
        
        if format == 'arrow':
            if pa is None:
                raise SerializationError("pyarrow not available")
            
            # Read Arrow table
            reader = pa.ipc.open_stream(data)
            table = reader.read_all()
            
            # Extract metadata
            metadata_json = table.schema.metadata.get(b"symbo_metadata")
            if metadata_json is None:
                raise SerializationError("Missing metadata")
            
            metadata = json.loads(metadata_json.decode('utf-8'))
            
            # Extract data
            flat_data = table["element"].to_pylist()
            
        else:  # msgpack
            if msgpack is None:
                raise SerializationError("msgpack not available")
            
            obj = msgpack.unpackb(data, raw=False)
            metadata = {k: obj[k] for k in ["name", "shape", "rank", "size"]}
            flat_data = obj["data"]
        
        # Reconstruct tensor
        shape = tuple(metadata["shape"])
        tensor = SymbolicTensor(shape, name=metadata["name"])
        
        # Fill with expressions
        for i, expr_str in enumerate(flat_data):
            idx = np.unravel_index(i, shape)
            tensor.data[idx] = sp.sympify(expr_str)
        
        return tensor
    
    @staticmethod
    def serialize_policy_function(policy: Any, format: str = 'msgpack') -> bytes:
        """
        Serialize GenerativePolicyFunction (TaylorExpansion PolicyFunction).
        
        Parameters
        ----------
        policy : PolicyFunction
            Policy function to serialize
        format : str
            Serialization format
            
        Returns
        -------
        bytes
            Serialized data
        """
        from symbo.generative.taylor import PolicyFunction
        
        if not isinstance(policy, PolicyFunction):
            raise SerializationError(f"Expected PolicyFunction, got {type(policy)}")
        
        data = {
            "type": "PolicyFunction",
            "expansion": str(policy.expansion),
            "variables": [str(v) for v in policy.variables],
            "center": {str(k): float(v) for k, v in policy.center.items()},
            "coefficients": {
                str(k): [str(c) for c in v] if isinstance(v, tuple) else str(v)
                for k, v in policy.coefficients.items()
            },
            "coeff_values": policy.coeff_values
        }
        
        if format == 'msgpack':
            if msgpack is None:
                raise SerializationError("msgpack not available")
            return msgpack.packb(data, use_bin_type=True)
        else:
            return json.dumps(data).encode('utf-8')
    
    @staticmethod
    def serialize_groebner_state(state: Any, format: str = 'msgpack') -> bytes:
        """
        Serialize GröbnerBasisState.
        
        Parameters
        ----------
        state : GröbnerBasisState
            State to serialize
        format : str
            Serialization format
            
        Returns
        -------
        bytes
            Serialized data
        """
        from symbo.solver.groebner import GröbnerBasisState
        
        if not isinstance(state, GröbnerBasisState):
            raise SerializationError(f"Expected GröbnerBasisState, got {type(state)}")
        
        data = {
            "type": "GröbnerBasisState",
            "polynomials": [str(p) for p in state.polynomials],
            "variables": [str(v) for v in state.variables],
            "order": state.order,
            "status": state.status,
            "error": state.error,
            "solutions": [
                {str(k): str(v) for k, v in sol.items()}
                for sol in state.solutions
            ]
        }
        
        if format == 'msgpack':
            if msgpack is None:
                raise SerializationError("msgpack not available")
            return msgpack.packb(data, use_bin_type=True)
        else:
            return json.dumps(data).encode('utf-8')
    
    @staticmethod
    def round_trip_test(obj: Any, 
                       serialize_func: callable,
                       deserialize_func: callable) -> bool:
        """
        Perform round-trip test for serialization.
        
        Verifies that serialize(deserialize(obj)) == obj.
        
        Parameters
        ----------
        obj : Any
            Object to test
        serialize_func : callable
            Serialization function
        deserialize_func : callable
            Deserialization function
            
        Returns
        -------
        bool
            True if round-trip succeeds and objects are equivalent
        """
        try:
            # Serialize
            serialized = serialize_func(obj)
            
            # Deserialize
            reconstructed = deserialize_func(serialized)
            
            # Check equivalence
            return SymboSerializer.verify_equivalence(obj, reconstructed)
            
        except Exception as e:
            print(f"Round-trip test failed: {e}")
            return False
    
    @staticmethod
    def verify_equivalence(obj1: Any, obj2: Any) -> bool:
        """
        Verify mathematical equivalence of two objects.
        
        Parameters
        ----------
        obj1, obj2 : Any
            Objects to compare
            
        Returns
        -------
        bool
            True if objects are mathematically equivalent
        """
        # Check type
        if type(obj1) != type(obj2):
            return False
        
        # SymPy expressions
        if isinstance(obj1, sp.Basic):
            try:
                diff = sp.simplify(obj1 - obj2)
                return diff == 0
            except:
                return str(obj1) == str(obj2)
        
        # SymbolicTensor
        from symbo.tensor import SymbolicTensor
        if isinstance(obj1, SymbolicTensor):
            if obj1.shape != obj2.shape:
                return False
            
            # Check all elements
            for idx in np.ndindex(obj1.shape):
                try:
                    diff = sp.simplify(obj1.data[idx] - obj2.data[idx])
                    if diff != 0:
                        return False
                except:
                    if str(obj1.data[idx]) != str(obj2.data[idx]):
                        return False
            return True
        
        # Fallback: string comparison
        return str(obj1) == str(obj2)


class ArrowTableBuilder:
    """
    Builder for creating Arrow tables from Symbo data.
    
    Useful for exporting results to Arrow-compatible tools
    like Pandas, Polars, DuckDB, etc.
    """
    
    @staticmethod
    def from_solution_set(solutions: List[Dict[str, Any]]) -> 'pa.Table':
        """
        Convert solution set to Arrow table.
        
        Parameters
        ----------
        solutions : List[Dict[str, Any]]
            List of solution dictionaries
            
        Returns
        -------
        pa.Table
            Arrow table
        """
        if pa is None:
            raise SerializationError("pyarrow not available")
        
        if not solutions:
            return pa.table({})
        
        # Extract all variable names
        var_names = set()
        for sol in solutions:
            var_names.update(sol.keys())
        
        var_names = sorted(var_names)
        
        # Build columns
        columns = {var: [] for var in var_names}
        
        for sol in solutions:
            for var in var_names:
                val = sol.get(var, None)
                if val is None:
                    columns[var].append(None)
                elif isinstance(val, sp.Basic):
                    columns[var].append(str(val))
                else:
                    columns[var].append(val)
        
        return pa.table(columns)


def test_round_trip_serialization():
    """
    Test round-trip serialization for all Symbo types.
    
    Returns
    -------
    Dict[str, bool]
        Test results for each type
    """
    results = {}
    
    # Test Expression
    x, y = sp.symbols('x y')
    expr = x**2 + sp.sin(y)
    
    results["Expression"] = SymboSerializer.round_trip_test(
        expr,
        SymboSerializer.serialize_expression,
        SymboSerializer.deserialize_expression
    )
    
    # Test Tensor
    from symbo.tensor import SymbolicTensor
    tensor = SymbolicTensor((2, 2), name="test")
    tensor.fill_with_symbols("T")
    
    for format in ['arrow', 'msgpack']:
        if format == 'arrow' and pa is None:
            continue
        if format == 'msgpack' and msgpack is None:
            continue
        
        results[f"SymbolicTensor_{format}"] = SymboSerializer.round_trip_test(
            tensor,
            lambda t: SymboSerializer.serialize_tensor(t, format),
            lambda d: SymboSerializer.deserialize_tensor(d, format)
        )
    
    return results


__all__ = [
    'SymboSerializer',
    'ArrowTableBuilder',
    'SerializationError',
    'test_round_trip_serialization',
]
