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

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import sympy as sp

from symbo._optional import optional_module
from symbo.security import safe_sympify

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers
    from symbo.tensor import SymbolicTensor

logger = logging.getLogger("symbo.io.serialization")

# Optional I/O backends: msgpack for compact records, pyarrow for columnar data.
msgpack = optional_module("msgpack")
pa = optional_module("pyarrow")


class SerializationError(Exception):
    """Raised when serialization fails."""
    pass


def _check_format(format: str, allowed: Sequence[str]) -> None:
    """Reject unknown formats instead of silently falling back to another one."""
    if format not in allowed:
        raise SerializationError(
            f"unsupported format {format!r} (use {' or '.join(allowed)})")


def _default_format() -> str:
    """Pick the best wire format for the installed environment.

    MessagePack is compact and therefore preferred when the optional ``msgpack``
    backend is present; JSON needs no extra dependency, so the no-argument API
    keeps working in a core-only installation.
    """
    return "msgpack" if msgpack is not None else "json"


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
    def serialize_expression(expr: sp.Expr,
                             format: Optional[str] = None) -> bytes:
        """
        Serialize SymPy expression.

        ``format`` defaults to ``msgpack`` when the optional backend is
        installed and ``json`` otherwise (``'json'`` needs no extra package).

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

        return SymboSerializer._pack(data, format)

    @staticmethod
    def deserialize_expression(data: bytes,
                               format: Optional[str] = None) -> sp.Expr:
        """
        Deserialize expression.

        ``format`` defaults to ``msgpack`` when the optional backend is
        installed and ``json`` otherwise.

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
        obj = SymboSerializer._unpack(data, format)
        SymboSerializer._expect(obj, "Expression")
        if "string_repr" not in obj:
            raise SerializationError("expression payload has no 'string_repr' field")
        return safe_sympify(obj["string_repr"])

    @staticmethod
    def serialize_tensor(tensor: 'SymbolicTensor', format: str = 'arrow') -> bytes:
        """
        Serialize SymbolicTensor.

        Parameters
        ----------
        tensor : SymbolicTensor
            Tensor to serialize
        format : str
            'arrow', 'msgpack' or 'json' ('arrow'/'msgpack' need ``symbo[io]``)

        Returns
        -------
        bytes
            Serialized tensor
        """
        # Import here to avoid circular dependency
        from symbo.tensor import SymbolicTensor

        if not isinstance(tensor, SymbolicTensor):
            raise SerializationError(f"Expected SymbolicTensor, got {type(tensor)}")

        _check_format(format, ("arrow", "msgpack", "json"))

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

        else:  # msgpack / json
            data = {
                **metadata,
                "data": flat_data
            }
            if format == 'msgpack':
                if msgpack is None:
                    raise SerializationError("msgpack not available")
                return msgpack.packb(data, use_bin_type=True)
            return json.dumps(data).encode('utf-8')

    @staticmethod
    def deserialize_tensor(data: bytes, format: str = 'arrow') -> 'SymbolicTensor':
        """
        Deserialize SymbolicTensor.

        Parameters
        ----------
        data : bytes
            Serialized data
        format : str
            'arrow', 'msgpack' or 'json' ('arrow'/'msgpack' need ``symbo[io]``)

        Returns
        -------
        SymbolicTensor
            Reconstructed tensor
        """
        from symbo.tensor import SymbolicTensor

        _check_format(format, ("arrow", "msgpack", "json"))

        if format == 'arrow':
            if pa is None:
                raise SerializationError("pyarrow not available")

            # Read Arrow table
            try:
                table = pa.ipc.open_stream(data).read_all()
            except SerializationError:
                raise
            except Exception as e:  # foreign or truncated buffer
                raise SerializationError(f"could not decode arrow tensor payload: {e}") from e

            # Extract metadata
            metadata_json = (table.schema.metadata or {}).get(b"symbo_metadata")
            if metadata_json is None:
                raise SerializationError("arrow tensor payload is missing its 'symbo_metadata' schema entry")

            metadata = json.loads(metadata_json.decode('utf-8'))

            # Extract data
            flat_data = table["element"].to_pylist()

        else:  # msgpack / json
            obj = SymboSerializer._unpack(data, format)
            missing = [k for k in ("name", "shape", "rank", "size", "data") if k not in obj]
            if missing:
                raise SerializationError(f"msgpack tensor payload is missing {', '.join(missing)}")
            metadata = {k: obj[k] for k in ("name", "shape", "rank", "size")}
            flat_data = obj["data"]

        # Reconstruct tensor
        shape = tuple(metadata["shape"])
        tensor = SymbolicTensor(shape, name=metadata["name"])

        # Fill with expressions
        for i, expr_str in enumerate(flat_data):
            idx = np.unravel_index(i, shape)
            tensor.data[idx] = safe_sympify(expr_str)

        return tensor

    @staticmethod
    def serialize_policy_function(policy: Any,
                                  format: Optional[str] = None) -> bytes:
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

        # Coefficient keys are tuples of SymPy symbols. ``str(tuple)`` cannot be
        # parsed back (``"(x,)"`` is not a literal), so each coefficient is stored
        # as ``name -> [variables it multiplies]`` -- the same convention
        # ``TaylorExpansion.to_wasm_json`` uses, and reversible without eval().
        data = {
            "type": "PolicyFunction",
            "expansion": str(policy.expansion),
            "variables": [str(v) for v in policy.variables],
            "center": {str(k): float(v) for k, v in policy.center.items()},
            "coefficients": {
                symbol.name: [str(v) for v in key] if isinstance(key, tuple) else [str(key)]
                for key, symbol in policy.coefficients.items()
            },
            "coeff_values": {str(k): v for k, v in (policy.coeff_values or {}).items()},
        }
        return SymboSerializer._pack(data, format)

    @staticmethod
    def serialize_groebner_state(state: Any,
                                 format: Optional[str] = None) -> bytes:
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

        return SymboSerializer._pack(data, format)

    # ------------------------------------------------------------------ formats

    @staticmethod
    def _pack(data: Dict[str, Any], format: Optional[str]) -> bytes:
        """Encode a payload dict as msgpack (default) or JSON bytes."""
        if format is None:
            format = _default_format()
        if format == 'msgpack':
            if msgpack is None:
                raise SerializationError(
                    "msgpack not available; install it with `pip install 'symbo[io]'` "
                    "or pass format='json'"
                )
            return msgpack.packb(data, use_bin_type=True)
        if format == 'json':
            return json.dumps(data).encode('utf-8')
        raise SerializationError(f"unsupported format {format!r} (use 'msgpack' or 'json')")

    @staticmethod
    def _unpack(data: bytes, format: Optional[str]) -> Dict[str, Any]:
        """Decode msgpack/JSON bytes into a payload dict."""
        if format is None:
            format = _default_format()
        if format == 'msgpack':
            if msgpack is None:
                raise SerializationError(
                    "msgpack not available; install it with `pip install 'symbo[io]'` "
                    "or pass format='json'"
                )
            try:
                return msgpack.unpackb(data, raw=False)
            except (ValueError, msgpack.exceptions.ExtraData) as exc:
                raise SerializationError(f"corrupt msgpack payload: {exc}") from exc
        if format == 'json':
            try:
                return json.loads(data.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SerializationError(f"corrupt JSON payload: {exc}") from exc
        raise SerializationError(f"unsupported format {format!r} (use 'msgpack' or 'json')")

    @staticmethod
    def _expect(payload: Dict[str, Any], kind: str) -> None:
        if not isinstance(payload, dict) or payload.get("type") != kind:
            raise SerializationError(
                f"expected a {kind} payload, got "
                f"{payload.get('type') if isinstance(payload, dict) else type(payload)}"
            )

    # ------------------------------------------------------------------- restore

    @staticmethod
    def deserialize_policy_function(data: bytes,
                                   format: Optional[str] = None) -> Any:
        """
        Rebuild a :class:`~symbo.generative.taylor.PolicyFunction`.

        The inverse of :meth:`serialize_policy_function`. Coefficient *keys* are
        restored as tuples of the policy's variables, so ``coefficients`` can be
        indexed the way the rest of the library expects.

        Raises
        ------
        SerializationError
            If the payload was not produced by ``serialize_policy_function``.
        """
        from symbo.generative.taylor import PolicyFunction

        payload = SymboSerializer._unpack(data, format)
        SymboSerializer._expect(payload, "PolicyFunction")

        variables = [sp.Symbol(name) for name in payload["variables"]]
        by_name = {v.name: v for v in variables}
        center = {by_name.get(sp.Symbol(k), sp.Symbol(k)): float(v)
                  for k, v in (payload.get("center") or {}).items()}

        coefficients: Dict[Any, sp.Symbol] = {}
        for raw_key, raw_value in (payload.get("coefficients") or {}).items():
            if isinstance(raw_value, (list, tuple)):
                # current format: {"g_x": ["x"], "g_0": [], ...}
                name, var_names = str(raw_key), [str(v) for v in raw_value]
            else:
                # legacy format: {"(x,)": "g_x"} -- keys were str(tuple)
                name = str(raw_value)
                text = str(raw_key).strip().lstrip("(").rstrip(")")
                var_names = [part.strip().strip("'\"")
                             for part in text.split(",") if part.strip()]
            key = tuple(by_name[n] for n in var_names if n in by_name)
            coefficients[key] = sp.Symbol(name)

        return PolicyFunction(
            expansion=safe_sympify(payload["expansion"]),
            variables=variables,
            center=center,
            coefficients=coefficients,
            coeff_values=dict(payload.get("coeff_values") or {}),
        )

    @staticmethod
    def deserialize_groebner_state(data: bytes,
                                  format: Optional[str] = None) -> Any:
        """
        Rebuild a :class:`~symbo.solver.groebner.GröbnerBasisState`.

        Only the state's portable parts come back: polynomials, variables,
        ordering, status, error and solutions. The computed ``basis`` object is
        *not* serialized (it is a SymPy internals object), so a restored state
        must call ``compute_basis()`` before streaming from it.
        """
        from symbo.solver.groebner import GröbnerBasisState

        payload = SymboSerializer._unpack(data, format)
        SymboSerializer._expect(payload, "GröbnerBasisState")

        variables = [sp.Symbol(v) for v in payload.get("variables") or []]
        state = GröbnerBasisState(
            [safe_sympify(p) for p in payload.get("polynomials") or []],
            variables,
            payload.get("order") or 'lex',
        )
        state.status = payload.get("status") or "restored"
        state.error = payload.get("error")
        state.solutions = [
            {sp.Symbol(str(k)): safe_sympify(v) for k, v in (sol or {}).items()}
            for sol in payload.get("solutions") or []
        ]
        return state

    @staticmethod
    def round_trip_test(obj: Any,
                       serialize_func: Callable[[Any], bytes],
                       deserialize_func: Callable[[bytes], Any]) -> bool:
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
            logger.warning("Round-trip test failed: %s", e)
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
        if type(obj1) is not type(obj2):
            return False

        # SymPy expressions
        if isinstance(obj1, sp.Basic):
            try:
                diff = sp.simplify(obj1 - obj2)
                return diff == 0
            except Exception:
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
                except Exception:
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

        # Extract all variable names. Keys may be SymPy symbols, whose
        # ``<`` builds a relational instead of comparing, so order by name.
        keys: Dict[str, Any] = {}
        for sol in solutions:
            for var in sol:
                keys.setdefault(str(var), var)
        var_names = sorted(keys)

        # Build columns
        columns = {var: [] for var in var_names}

        for sol in solutions:
            for var in var_names:
                val = sol.get(keys[var], None)
                if val is None:
                    columns[var].append(None)
                elif isinstance(val, sp.Basic):
                    columns[var].append(str(val))
                else:
                    columns[var].append(val)

        return pa.table(columns)


def verify_round_trip_serialization() -> Dict[str, bool]:
    """
    Test round-trip serialization for all Symbo types.

    Returns
    -------
    Dict[str, bool]
        Test results for each type
    """
    results = {}

    # Test Expression. The no-format call picks msgpack when available and
    # JSON otherwise, so the self-check also succeeds in a core-only env.
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

    for fmt in ('json', 'arrow', 'msgpack'):
        if fmt == 'arrow' and pa is None:
            continue
        if fmt == 'msgpack' and msgpack is None:
            continue

        results[f"SymbolicTensor_{fmt}"] = SymboSerializer.round_trip_test(
            tensor,
            lambda t, fmt=fmt: SymboSerializer.serialize_tensor(t, fmt),
            lambda d, fmt=fmt: SymboSerializer.deserialize_tensor(d, fmt),
        )

    # Test PolicyFunction (needs the generative module)
    try:
        from symbo.generative.taylor import TaylorExpansion

        te = TaylorExpansion([x, y], {x: 0.0, y: 0.0}, max_order=1)
        te.generate("g")
        policy = te.to_policy_function({"g_0": 1.0, "g_x": 2.0, "g_y": 3.0})
        for fmt in ('msgpack', 'json'):
            if fmt == 'msgpack' and msgpack is None:
                continue
            results[f"PolicyFunction_{fmt}"] = SymboSerializer.round_trip_test(
                policy,
                lambda p, fmt=fmt: SymboSerializer.serialize_policy_function(p, fmt),
                lambda d, fmt=fmt: SymboSerializer.deserialize_policy_function(d, fmt),
            )
    except Exception as exc:  # pragma: no cover - reported, never raised
        logger.warning("policy round-trip check skipped: %s", exc)
        results["PolicyFunction_msgpack"] = False

    # Test GröbnerBasisState
    try:
        from symbo.solver.groebner import GröbnerBasisState

        state = GröbnerBasisState([x**2 + y**2 - 1, x - y], [x, y], 'lex')
        state.status = "completed"
        state.solutions = [{x: sp.Integer(1), y: sp.Integer(1)}]
        for fmt in ('msgpack', 'json'):
            if fmt == 'msgpack' and msgpack is None:
                continue
            restored = SymboSerializer.deserialize_groebner_state(
                SymboSerializer.serialize_groebner_state(state, fmt), fmt)
            ok = (restored.status == "completed"
                  and str(restored.polynomials[0]) == str(x**2 + y**2 - 1)
                  and restored.solutions[0][x] == 1
                  and restored.order == 'lex')
            results[f"GroebnerBasisState_{fmt}"] = bool(ok)
    except Exception as exc:  # pragma: no cover - reported, never raised
        logger.warning("groebner state round-trip check skipped: %s", exc)
        results["GroebnerBasisState_json"] = False

    return results


__all__ = [
    'ArrowTableBuilder',
    'SerializationError',
    'SymboSerializer',
    'verify_round_trip_serialization',
]
