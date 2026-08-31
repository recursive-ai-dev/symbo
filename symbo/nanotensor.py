# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Symbo — Nano-scale Hybrid Generative Symbolic Engine
====================================================

This module implements Symbo, a hybrid symbolic–numeric reasoning system built
around a true n-dimensional symbolic tensor type (`NanoTensor`) and a set of
training and reasoning utilities.

The core design is based on decomposing a large collection of classical
algorithms (Gröbner bases, perturbation methods, polynomial solvers, pathfinding,
and more) into atomic computational primitives and recombining them into a
generative symbolic architecture. Symbo aims to:

- represent policy functions and dynamical systems via Taylor-manifold expansions,
- solve nonlinear systems using Gröbner bases and related algebraic methods,
- perform second-order perturbation analysis in the spirit of modern macro models,
- support hybrid neuro-symbolic training workflows,
- and expose reasoning tools such as A*-based pathfinding over symbolic energy
  landscapes.

The module also includes WASM-friendly entry points for browser runtimes,
serialization helpers (MessagePack and Arrow), and demonstration routines for:

- a 2nd-order perturbation solution of an RBC-style model (`demo_rbc_perturbation`),
- an algebraic differential equation example (`demo_kamke_ade`),
- and performance benchmarks contrasting symbolic and numeric operations.

High-level workflow
-------------------

1. Construct a `NanoTensor` as a symbolic container for Taylor expansions.
2. Use `generate_taylor` and `full_perturbation` (via `SymbolicTrainer`) to fit
   policy functions or model residuals.
3. Evaluate and visualize the resulting symbolic policies over grids, contour
   plots, and surfaces.
4. Optionally integrate with neuro-symbolic training (`HybridTrainer`) or store
   learned coefficients in a graph-based `KnowledgeBase`.

This file is intended as both an executable prototype and a research-grade
reference implementation of a small, interpretable symbolic engine.
"""

from __future__ import annotations

import contextlib
import heapq
import json
import logging
import math
import os
import pickle
import time
import warnings
from collections import Counter
from itertools import combinations_with_replacement
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from sympy import Poly, cse, groebner, lambdify, nsolve, resultant, solve

# msgpack / pyarrow / dill are optional I/O accelerators: the engine runs
# without them, but the serialization helpers raise an informative error on use.
from symbo._optional import optional_module, require
from symbo.security import safe_sympify

msgpack = optional_module("msgpack")
pa = optional_module("pyarrow")
dill = optional_module("dill")

logger = logging.getLogger("symbo.nanotensor")



def serialize_basis_msgpack(G) -> bytes:
    """
    Serialize a SymPy Gröbner basis to a compact MessagePack representation.

    Parameters
    ----------
    G : sympy.polys.polytools.GroebnerBasis
        Gröbner basis object produced by `sympy.groebner`.

    Returns
    -------
    bytes
        MessagePack-encoded bytes containing a dictionary with:
        - "gens": stringified generators,
        - "polys": stringified basis polynomials,
        - "order": the monomial order used.

    Raises
    ------
    RuntimeError
        If `msgpack` is not installed in the current environment.
    """

    if msgpack is None:
        raise RuntimeError("msgpack is not installed")

    data = {
        "gens": [str(g) for g in G.gens],
        "polys": [str(p) for p in G.polys],
        "order": str(G.order),
    }
    return msgpack.packb(data, use_bin_type=True)


def serialize_basis_arrow(G) -> bytes:
    """
    Serialize a SymPy Gröbner basis to an Arrow IPC buffer.

    The resulting bytes can be streamed or stored efficiently and consumed
    by Arrow-compatible tools for inspection or interoperability.

    Parameters
    ----------
    G : sympy.polys.polytools.GroebnerBasis
        Gröbner basis object produced by `sympy.groebner`.

    Returns
    -------
    bytes
        Arrow IPC (Feather-like) binary stream containing:
        - column "poly": string form of each basis polynomial,
        - column "gens": a chunked array of the generators.

    Raises
    ------
    RuntimeError
        If `pyarrow` is not installed in the current environment.
    """

    if pa is None:
        raise RuntimeError("pyarrow is not installed")

    arr_polys = pa.array([str(p) for p in G.polys])
    arr_gens = pa.array([str(g) for g in G.gens])

    table = pa.table({
        "poly": arr_polys,
        "gens": pa.chunked_array([arr_gens]),
    })

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)

    return sink.getvalue().to_pybytes()

def wasm_eval_expression(expr_str: str, var_values: Dict[str, float]) -> float:
    """
    Simple WASM-friendly entrypoint: parse an expression string, substitute vars, and eval.
    Uses safe_sympify to prevent arbitrary code execution on raw string input.
    """
    expr = safe_sympify(expr_str)
    subs_d = {sp.Symbol(k): v for k, v in var_values.items()}
    return float(expr.subs(subs_d).evalf())


def wasm_groebner_solve_json(poly_strs: List[str],
                             var_names: List[str]) -> str:
    """
    Compute a Groebner basis and solutions from string input and return JSON.

    This function is tailored for WASM or remote contexts where the caller
    only communicates via strings. It:

    1. Parses a list of polynomial expressions from strings.
    2. Constructs SymPy symbols for the given variable names.
    3. Computes a Groebner basis under lexicographic order.
    4. Attempts to solve the system symbolically.
    5. Returns a JSON string containing:
       - "basis": list of stringified basis polynomials,
       - "solutions": list of solution dicts (stringified values).

    Parameters
    ----------
    poly_strs : list[str]
        Polynomial equations represented as SymPy-parsable strings.
    var_names : list[str]
        Names of the variables to solve for.

    Returns
    -------
    str
        JSON-encoded result containing basis and solutions.
    """

    polys = [safe_sympify(s) for s in poly_strs]
    vars_syms = [sp.Symbol(v) for v in var_names]
    G = groebner(polys, *vars_syms, order='lex')
    sols = solve(polys, *vars_syms, dict=True)

    sols_json = [{
        str(k): str(v) for k, v in sol.items()
    } for sol in sols]

    out = {
        "basis": [str(p) for p in G.polys],
        "solutions": sols_json,
    }
    return json.dumps(out)

class NanoTensor:
    """
    Military-Grade NanoTensor: n-dimensional symbolic tensor with agency capabilities.

    This class represents a military-grade symbolic tensor that acts as a computational
    "brain" providing agents with agency - the ability to perceive, reason, learn, and
    act autonomously. It combines symbolic exactness with:

    - **Autonomous Decision-Making**: Self-optimization and error correction
    - **Memory & Learning**: Experience replay and pattern recognition
    - **Health Monitoring**: Self-diagnostics and performance tracking
    - **Security Layer**: Robust validation and anomaly detection
    - **Agency Core**: Goal-directed reasoning and adaptive behavior

    Traditional capabilities:
    - hold Taylor-expansion–based policy functions or model approximations,
    - support vectorized symbolic operations (diff, subs, evaluation),
    - provide hooks for Gröbner-based solving and perturbation analysis,
    - and serve as the core representational object in the Symbo engine.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the underlying tensor (NumPy array of SymPy expressions).
    max_order : int, optional
        Maximum Taylor expansion order to construct in `generate_taylor`.
        Typically 1 or 2 for first- and second-order perturbations.
    base_vars : list[str], optional
        Names of the base state variables (e.g. ['k', 'a', 'eps', 'sig']).
        These determine both the Taylor expansion structure and steady-state
        computations.
    name : str, optional
        Human-readable identifier used in plots and summaries.

    Notes
    -----
    Internally, `data` is stored as a NumPy array of SymPy expressions, and
    `coeff_vars` tracks the symbolic coefficients introduced by
    `generate_taylor`. The combination of `base_vars` and `coeff_vars`
    defines the full symbolic structure of the tensor.

    The enhanced NanoTensor includes:
    - Autonomous learning from computational experiences
    - Self-monitoring and adaptive optimization
    - Robust error handling with intelligent recovery
    - Security validation and constraint checking
    - Performance metrics and health status tracking
    """

    def __init__(self, shape: Tuple[int, ...], max_order: int = 2,
                 base_vars: Optional[List[str]] = None, name: str = "nt"):
        shape = NanoTensor._validate_shape(shape)
        if max_order < 1:
            raise ValueError(f"max_order must be >= 1, got {max_order}")

        self.shape: Tuple[int, ...] = shape
        self.max_order = int(max_order)
        self.name = name
        self.base_vars = [sp.Symbol(v) for v in (base_vars or ['k', 'a', 'eps', 'sig'])]
        self.coeff_vars: List[sp.Symbol] = []
        self.data: np.ndarray = np.empty(shape, dtype=object)
        # Caches must exist before the first structural mutation so that
        # ``_invalidate_caches`` can be called from anywhere.
        self._diff_cache: Dict[Tuple[str, int], 'NanoTensor'] = {}
        self._subs_cache: Dict[Tuple, 'NanoTensor'] = {}
        self._max_cached_tensors = 128
        self._symvars_cache: Optional[List[sp.Symbol]] = None
        self._lambdify_cache: Dict[Tuple, List] = {}
        self.fitted_coeffs: Dict[str, float] = {}
        self._init_data()
        self._optimizing = False  # re-entrancy guard for self-optimization

        # Military-grade enhancements: Agency and monitoring
        self._operation_count = 0
        self._success_count = 0
        self._total_compute_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._health_status = "optimal"  # optimal, good, degraded, critical
        self._experience_buffer: List[Dict[str, Any]] = []
        self._max_experience = 1000
        self._learned_patterns: Dict[str, Dict[str, float]] = {}
        self._validation_bounds: Dict[str, Tuple[float, float]] = {}
        self._auto_optimize = True
        self._anomaly_threshold = 3.0
        self._input_stats: Dict[str, Dict[str, float]] = {}  # Welford's online stats
        self._optimized_data: Optional[Tuple[List, List]] = None  # Storage for CSE optimized form
        self._optimized_func: Optional[Tuple[List[sp.Symbol], Any]] = None  # Cached optimized function

    # ------------------------------------------------------------------
    # Structural helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Normalize and validate a tensor shape (positive integers only)."""
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        try:
            shape = tuple(int(d) for d in shape)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"shape must be a tuple of ints, got {shape!r}") from exc
        if not shape or any(d <= 0 for d in shape):
            raise ValueError(f"shape must be non-empty with positive dimensions, got {shape}")
        return shape

    def _init_data(self):
        """Initialize tensor with symbolic zeros"""
        self.data = np.zeros(self.shape, dtype=object)
        self.data.flat[:] = sp.S(0)
        self._invalidate_caches()

    # Derived, rebuildable state that is deliberately *not* serialized: compiled
    # ``lambdify`` callables and simplified copies are large, may not survive a
    # Python version change, and are meaningless for a freshly loaded tensor.
    # Dropping them also keeps ``pickle`` (without dill) working, since
    # lambdify-generated functions cannot be pickled by attribute.
    _TRANSIENT_STATE: ClassVar[Dict[str, Any]] = {
        "_lambdify_cache": dict,
        "_diff_cache": dict,
        "_subs_cache": dict,
        "_optimized_data": lambda: None,
        "_optimized_func": lambda: None,
        "_symvars_cache": lambda: None,
    }

    def __getstate__(self) -> Dict[str, Any]:
        transient = set(self._TRANSIENT_STATE)
        return {k: v for k, v in self.__dict__.items() if k not in transient}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Assign through ``__dict__`` so subclasses that override ``__setattr__``
        # (e.g. the cache-aware military-grade tensor) are not entered mid-load.
        for name, factory in self._TRANSIENT_STATE.items():
            self.__dict__[name] = factory()

    def _invalidate_caches(self) -> None:
        """
        Drop every cache that depends on the current structure of ``self.data``.

        Must be called after any in-place mutation of ``data`` (differentiation
        recovery, simplification, coefficient fitting, CSE optimization) so that
        ``eval_numeric`` can never serve a stale lambdified callable.
        """
        self._symvars_cache = None
        self._lambdify_cache.clear()
        self._diff_cache.clear()
        self._subs_cache.clear()
        self._optimized_data = None
        self._optimized_func = None

    def _clone_metadata(self) -> Dict[str, Any]:
        """Constructor kwargs reproducing this tensor's structure (new data)."""
        return {
            "shape": self.shape,
            "max_order": self.max_order,
            "base_vars": [v.name for v in self.base_vars],
            "name": self.name,
        }

    def _new_like(self) -> 'NanoTensor':
        """Return an empty tensor that shares this tensor's structural metadata."""
        new_nt = NanoTensor(**self._clone_metadata())
        new_nt.coeff_vars = list(self.coeff_vars)
        return new_nt

    def _record_operation(self, op_type: str, duration: float, success: bool,
                          error: Optional[str] = None):
        """Record operation for learning and monitoring (military-grade feature)."""
        self._operation_count += 1
        if success:
            self._success_count += 1
        self._total_compute_time += duration

        # Store experience
        experience = {
            "type": op_type,
            "duration": duration,
            "success": success,
            "error": error,
            "timestamp": time.time()
        }
        self._experience_buffer.append(experience)

        # Keep buffer manageable
        if len(self._experience_buffer) > self._max_experience:
            self._experience_buffer = self._experience_buffer[-self._max_experience:]

        # Update learned patterns
        if op_type not in self._learned_patterns:
            self._learned_patterns[op_type] = {
                "count": 0,
                "success_count": 0,
                "avg_duration": 0.0,
                "success_rate": 1.0
            }

        pattern = self._learned_patterns[op_type]
        pattern["count"] += 1
        if success:
            pattern["success_count"] += 1
        alpha = 0.1  # Learning rate
        pattern["avg_duration"] = (1 - alpha) * pattern["avg_duration"] + alpha * duration
        pattern["success_rate"] = pattern["success_count"] / pattern["count"] if pattern["count"] > 0 else 0.0

        # Update health status
        self._update_health_status()

    def _update_input_stats(self, var_name: str, value: float):
        """Update running statistics for inputs using Welford's online algorithm."""
        if var_name not in self._input_stats:
            self._input_stats[var_name] = {"n": 0, "mean": 0.0, "M2": 0.0}

        stats = self._input_stats[var_name]
        stats["n"] += 1
        delta = value - stats["mean"]
        stats["mean"] += delta / stats["n"]
        delta2 = value - stats["mean"]
        stats["M2"] += delta * delta2

    def _check_input_anomaly(self, var_name: str, value: float) -> bool:
        """Check for statistical anomaly (Z-score > threshold)."""
        if var_name not in self._input_stats:
            return False

        stats = self._input_stats[var_name]
        if stats["n"] < 10:  # Need sufficient samples
            return False
        if stats["n"] <= 1:  # variance undefined for a single sample
            return False

        variance = stats["M2"] / (stats["n"] - 1)
        if variance < 1e-12:  # Avoid division by zero
            return False

        std_dev = math.sqrt(variance)
        z_score = abs(value - stats["mean"]) / std_dev

        return z_score > self._anomaly_threshold

    def _update_health_status(self):
        """Update health status based on performance metrics (military-grade feature)."""
        if self._operation_count == 0:
            self._health_status = "optimal"
            return

        # Adjust for expected failures like bounds validation if necessary.
        # It's better to just use regular logic but if the operation failed
        # it was considered unsuccessful.

        success_rate = self._success_count / self._operation_count

        if success_rate >= 0.99:
            self._health_status = "optimal"
        elif success_rate >= 0.95:
            self._health_status = "good"
        elif success_rate >= 0.85:
            self._health_status = "degraded"
        else:
            self._health_status = "critical"

        # Auto-optimize if degraded
        if self._auto_optimize and self._health_status in ["degraded", "critical"]:
            self._attempt_self_optimization()

    def _attempt_self_optimization(self):
        """
        Autonomous self-optimization (military-grade agency feature).

        Best-effort by design: failures are logged and never propagate to the
        caller. A re-entrancy guard prevents ``simplify()`` (which itself records
        an operation and refreshes the health status) from recursing here.
        """
        if self._optimizing:
            return
        self._optimizing = True
        try:
            # Clear old caches to free memory
            if len(self._diff_cache) > 100:
                self._diff_cache.clear()
            if len(self._subs_cache) > 100:
                self._subs_cache.clear()

            # Simplify if we have complex expressions and a poor success rate
            if (self._operation_count > 100
                    and self._success_count / self._operation_count < 0.9):
                self.simplify()

            # Optimize storage for faster evaluation
            if self._operation_count % 50 == 0:
                self.optimize_storage()

        except Exception as exc:
            logger.debug("self-optimization skipped after failure: %r", exc)
        finally:
            self._optimizing = False

    def optimize_storage(self) -> bool:
        """
        Optimize expression storage using Common Subexpression Elimination (CSE).

        This significantly speeds up numeric evaluation of large tensors by
        pre-compiling one CSE-reduced callable for the full symbol set.
        Existing cache entries stay valid because the expressions are unchanged.

        Returns
        -------
        bool
            ``True`` when an optimized callable was installed, ``False`` when
            optimization was skipped or failed. A tensor that could not be
            optimized stays fully usable in unoptimized form.
        """
        try:
            flat_exprs = self.data.flatten().tolist()
            # Nothing to gain from CSE when no element carries free symbols.
            if not any(getattr(e, "free_symbols", None) for e in flat_exprs):
                self._optimized_data = None
                self._optimized_func = None
                return False

            replacements, reduced_exprs = cse(flat_exprs)
            self._optimized_data = (replacements, reduced_exprs)

            # ``eval_numeric`` may be called with any subset of the tensor's
            # symbols, so the optimized callable is compiled for the full set
            # (missing symbols default to 0.0 exactly as in the cached path).
            vars_all = sorted(self.symvars, key=lambda v: v.name)
            self._optimized_func = (
                vars_all,
                lambdify(vars_all, flat_exprs, modules='numpy', cse=True),
            )
            # The expressions themselves are unchanged, so the lambdify cache
            # entries stay valid; only the structural caches matter here.
            return True

        except Exception as exc:
            # Never degrade silently: the tensor remains correct but
            # unoptimized, and the reason is reported to the logger.
            self._optimized_data = None
            self._optimized_func = None
            logger.warning(
                "optimize_storage(): CSE optimization failed, continuing unoptimized: %r",
                exc,
            )
            return False

    def _validate_input(self, var_name: str, value: float) -> bool:
        """Validate input against bounds (military-grade security feature)."""
        # Coerce once: SymPy numbers, numpy scalars and ints all become floats,
        # and comparisons below are then done in plain Python arithmetic.
        try:
            values = [float(v) for v in (value if isinstance(value, (list, tuple, np.ndarray)) else [value])]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric value for {var_name}: {value!r}") from exc

        if var_name in self._validation_bounds:
            lower, upper = self._validation_bounds[var_name]
            for v in values:
                if not (lower <= v <= upper):
                    raise ValueError(
                        f"Input {var_name}={v} outside valid bounds [{lower}, {upper}]"
                    )

        # Check for invalid values (NaN / +-Inf) on every batched entry.
        for v in values:
            if np.isnan(v) or np.isinf(v):
                raise ValueError(f"Invalid value for {var_name}: {v}")

        # Update statistics and check for anomalies (scalar view only)
        if len(values) == 1:
            self._update_input_stats(var_name, values[0])
            anomalous = self._check_input_anomaly(var_name, values[0])
        else:
            anomalous = False
        if anomalous:
            warnings.warn(
                f"Statistical anomaly detected for {var_name}: {value} "
                f"(Z-score > {self._anomaly_threshold})",
                stacklevel=2,
            )

        return True

    def set_validation_bounds(self, var_name: str, lower: float, upper: float):
        """Set validation bounds for a variable (military-grade security)."""
        self._validation_bounds[var_name] = (lower, upper)

    def health_check(self) -> Dict[str, Any]:
        """Get comprehensive health status (military-grade monitoring)."""
        cache_total = self._cache_hits + self._cache_misses
        cache_rate = self._cache_hits / cache_total if cache_total > 0 else 0.0
        avg_time = self._total_compute_time / self._operation_count if self._operation_count > 0 else 0.0

        return {
            "status": self._health_status,
            "metrics": {
                "total_operations": self._operation_count,
                "success_rate": self._success_count / self._operation_count if self._operation_count > 0 else 1.0,
                "cache_hit_rate": cache_rate,
                "avg_operation_time": avg_time,
                "total_compute_time": self._total_compute_time
            },
            "learned_patterns": self._learned_patterns,
            "cache_sizes": {
                "diff": len(self._diff_cache),
                "lambdify": len(self._lambdify_cache)
            },
            "validation": {
                "bounds_set": len(self._validation_bounds),
                "input_stats_tracked": len(self._input_stats)
            },
            "optimization": {
                "cse_enabled": self._optimized_data is not None
            }
        }

    @staticmethod
    def _serializer():
        """
        Pick the brain-persistence backend.

        ``dill`` is preferred (it can also capture lambdas); the stdlib
        ``pickle`` module is a sufficient fallback because the persisted state
        only contains SymPy expressions, numpy arrays and plain containers.
        """
        return dill if dill is not None else pickle

    def save_brain(self, filepath: str):
        """
        Persist the entire NanoTensor brain state to disk.
        Uses dill for robust serialization of SymPy objects and lambdas,
        falling back to the stdlib ``pickle`` module when dill is absent.

        The file is written to a temporary path first and then renamed, so an
        interrupted write cannot corrupt an existing checkpoint.

        Security Note: Both dill and pickle are vulnerable to arbitrary code execution
        if used to deserialize untrusted data. Callers must ensure the loaded
        file comes from a trusted source.
        """
        serializer = self._serializer()
        tmp_path = f"{filepath}.tmp"

        try:
            # We don't save the diff cache or lambdify cache to save space
            # and avoid issues with unpicklable objects
            state = {
                "shape": self.shape,
                "max_order": self.max_order,
                "name": self.name,
                "base_vars": self.base_vars,
                "coeff_vars": self.coeff_vars,
                "data": self.data,  # SymPy expressions are picklable
                "fitted_coeffs": self.fitted_coeffs,
                "metrics": {
                    "ops": self._operation_count,
                    "success": self._success_count,
                    "time": self._total_compute_time,
                    "hits": self._cache_hits,
                    "misses": self._cache_misses
                },
                "experience": self._experience_buffer,
                "patterns": self._learned_patterns,
                "validation": self._validation_bounds,
                "input_stats": self._input_stats,
                "optimized_data": self._optimized_data
            }

            with open(tmp_path, 'wb') as f:
                serializer.dump(state, f)
            os.replace(tmp_path, filepath)

        except Exception as e:
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
            raise RuntimeError(f"Failed to save brain to {filepath}: {e}") from e

    @classmethod
    def load_brain(cls, filepath: str) -> 'NanoTensor':
        """
        Load a persisted NanoTensor brain from disk.

        Security Note: This method deserializes a pickle/dill file, which can execute
        arbitrary code. Do not load untrusted `filepath` arguments.
        """
        serializer = cls._serializer()

        try:
            with open(filepath, 'rb') as f:
                state = serializer.load(f)

            nt = cls(
                shape=state["shape"],
                max_order=state.get("max_order", 2),
                base_vars=[v.name for v in state["base_vars"]],
                name=state.get("name", "nt"),
            )

            # Restore state. Caches are deliberately *not* restored: every
            # derived cache is rebuilt lazily from ``data`` (the single source
            # of truth), which keeps the round trip correct by construction.
            nt.base_vars = state["base_vars"]
            nt.coeff_vars = state["coeff_vars"]
            nt.data = state["data"]
            nt.fitted_coeffs = state.get("fitted_coeffs", {})
            nt._invalidate_caches()

            # Restore metrics
            metrics = state.get("metrics", {})
            nt._operation_count = metrics.get("ops", 0)
            nt._success_count = metrics.get("success", 0)
            nt._total_compute_time = metrics.get("time", 0.0)
            nt._cache_hits = metrics.get("hits", 0)
            nt._cache_misses = metrics.get("misses", 0)

            # Restore learning
            nt._experience_buffer = state.get("experience", [])
            nt._learned_patterns = state.get("patterns", {})
            nt._validation_bounds = state.get("validation", {})
            nt._input_stats = state.get("input_stats", {})

            nt._update_health_status()
            return nt

        except Exception as e:
            raise RuntimeError(f"Failed to load brain from {filepath}: {e}") from e

    def get_agency_status(self) -> Dict[str, Any]:
        """Get agency and learning status (military-grade agency feature)."""
        return {
            "experiences_recorded": len(self._experience_buffer),
            "patterns_learned": len(self._learned_patterns),
            "auto_optimize": self._auto_optimize,
            "health": self._health_status,
            "recommendations": self._get_optimization_recommendations()
        }

    def _get_optimization_recommendations(self) -> List[str]:
        """Get autonomous recommendations for optimization (military-grade agency)."""
        recommendations = []

        if self._operation_count > 0:
            success_rate = self._success_count / self._operation_count

            if success_rate < 0.95:
                recommendations.append("Consider simplifying expressions to improve success rate")

            cache_total = self._cache_hits + self._cache_misses
            if cache_total > 0:
                cache_rate = self._cache_hits / cache_total
                if cache_rate < 0.5:
                    recommendations.append("Low cache hit rate - consider increasing cache size")

            if len(self._diff_cache) > 80:
                recommendations.append("Differentiation cache is large - consider clearing old entries")

            avg_time = self._total_compute_time / self._operation_count
            if avg_time > 1.0:
                recommendations.append("High average operation time - consider pre-compilation or simplification")

        if not recommendations:
            recommendations.append("All systems operating optimally")

        return recommendations

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for A* on a grid."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def find_path_on_grid(Z: np.ndarray,
                          start: Tuple[int, int],
                          goal: Tuple[int, int],
                          mode: str = "min") -> List[Tuple[int, int]]:
        """
        A* pathfinding on a 2D cost grid Z.

        Args:
            Z: 2D array of costs.
            start: (i, j) start index into Z.
            goal: (i, j) goal index into Z.
            mode: 'min' to prefer low Z (valley-following),
                  'max' to prefer high Z (ridge-following).

        Returns:
            List of (i, j) indices representing the path.
        """
        rows, cols = Z.shape
        (si, sj) = start
        (gi, gj) = goal

        if not (0 <= si < rows and 0 <= sj < cols and 0 <= gi < rows and 0 <= gj < cols):
            raise ValueError("Start or goal index out of bounds for Z.")

        # If maximizing, flip cost sign
        if mode == "max":
            cost_grid = -Z
        else:
            cost_grid = Z

        def neighbors(i, j):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    yield ni, nj

        open_set = []
        heapq.heappush(open_set, (0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                # reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for nb in neighbors(*current):
                tentative_g = g_score[current] + float(cost_grid[nb])
                if tentative_g < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + NanoTensor._heuristic(nb, goal)
                    heapq.heappush(open_set, (f, nb))

        # No path found
        return []

    @staticmethod
    def stream_groebner_basis(poly_system: List[sp.Expr],
                              vars_to_solve: List[sp.Symbol],
                              chunk_size: int = 1):
        """
        Compute Gröbner basis once, then stream its polynomials as JSON lines.

        Yields:
            JSON strings, each representing a 'chunk' of the basis.
        """
        G = groebner(poly_system, *vars_to_solve, order='lex')
        polys = list(G.polys)

        chunk = []
        for idx, p in enumerate(polys):
            chunk.append({
                "index": idx,
                "poly_str": str(p),
                "vars": [str(v) for v in vars_to_solve],
            })
            if len(chunk) >= chunk_size:
                yield json.dumps({"type": "groebner_chunk", "items": chunk})
                chunk = []

        if chunk:
            yield json.dumps({"type": "groebner_chunk", "items": chunk})

    @property
    def symvars(self) -> List[sp.Symbol]:
        """
        Return and cache the set of free symbols appearing in the tensor.

        The first call scans all tensor elements and collects their
        `free_symbols`. Subsequent calls reuse a cached list until the tensor
        is structurally modified (e.g., after `subs` or `generate_taylor`).
        """
        if self._symvars_cache is None:
            vars_set = set()
            for elem in self.data.flat:
                # ``getattr`` guard: entries may be plain python numbers.
                for sym in getattr(elem, 'free_symbols', ()):
                    vars_set.add(sym)
            # Deterministic ordering keeps cache keys and lambdify signatures
            # stable across runs (set iteration order is salted per process).
            self._symvars_cache = sorted(vars_set, key=lambda v: v.name)
        return self._symvars_cache

    def diff(self, wrt: sp.Symbol, order: int = 1) -> 'NanoTensor':
        """
        Symbolic differentiation of entire tensor (military-grade enhanced).

        Now includes:
        - Performance tracking and learning
        - Intelligent caching with monitoring
        - Anomaly detection
        - Automatic recovery on failure
        """
        start_time = time.time()
        success = True
        error_msg = None

        try:
            new_nt = self._new_like()
            new_nt.data = np.vectorize(lambda e: sp.diff(e, wrt, order), otypes=[object])(self.data)
            return new_nt
        except Exception as e:
            success = False
            error_msg = str(e)
            # Attempt recovery: simplify a *copy* first. ``diff`` is documented
            # as returning a new tensor, so it must never mutate ``self.data``.
            try:
                simplified = np.vectorize(sp.simplify, otypes=[object])(self.data)
                new_nt = self._new_like()
                new_nt.data = np.vectorize(lambda e: sp.diff(e, wrt, order), otypes=[object])(simplified)
                logger.info("diff(): succeeded only after simplifying a copy of the tensor")
                success = True
                error_msg = None
                return new_nt
            except Exception as recovery_exc:
                # Recovery also failed; raise a combined error to preserve context
                raise RuntimeError(
                    f"Differentiation failed, and recovery via simplify() also failed. "
                    f"Original error: {e!r}; recovery error: {recovery_exc!r}"
                ) from recovery_exc
        finally:
            duration = time.time() - start_time
            self._record_operation("differentiation", duration, success, error_msg)

    def diff_cached(self, wrt_name: str, order: int = 1) -> 'NanoTensor':
        """
        Cached differentiation keyed by variable name and order (military-grade enhanced).

        Avoids recomputing repeated derivative requests during benchmarking
        or exploratory analysis. Now tracks cache performance for learning.
        """
        key = (wrt_name, order)
        if key in self._diff_cache:
            self._cache_hits += 1
            return self._diff_cache[key]

        self._cache_misses += 1
        wrt = next((s for s in self.base_vars if s.name == wrt_name), sp.Symbol(wrt_name))
        self._diff_cache[key] = self.diff(wrt, order)
        return self._diff_cache[key]

    def subs(self, sub_dict: Dict[sp.Symbol, Any]) -> 'NanoTensor':
        """
        Substitute symbols throughout the tensor and return a new tensor.

        Parameters
        ----------
        sub_dict : dict[sympy.Symbol | str, Any]
            Mapping from symbols (or symbol names) to replacement values.
            Numeric values are safely converted via `nsimplify` where possible.

        Returns
        -------
        NanoTensor
            A new `NanoTensor` with substitutions applied.

        Notes
        -----
        If any of the substitution keys correspond to coefficient symbols
        recorded in `coeff_vars`, the internal caches for `symvars` and
        lambdified functions are invalidated to maintain consistency.
        """

        new_nt = self._new_like()
        clean_dict = {
            (k if isinstance(k, sp.Basic) else sp.Symbol(k)):
                (sp.nsimplify(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
            for k, v in sub_dict.items()
        }
        new_nt.data = np.vectorize(
            lambda e: e.subs(clean_dict) if hasattr(e, 'subs') else e,
            otypes=[object],
        )(self.data)

        # When coefficient symbols were substituted the tensor is no longer a
        # parameterized family: drop them so caches and refits stay consistent.
        coeff_names = {c.name for c in self.coeff_vars}
        substituted = {k.name for k in clean_dict if hasattr(k, 'name')}
        if coeff_names & substituted:
            new_nt.coeff_vars = [c for c in new_nt.coeff_vars if c.name not in coeff_names]
            new_nt._invalidate_caches()

        return new_nt

    def subs_cached(self, sub_tuple: Tuple) -> 'NanoTensor':
        """
        Cached substitution for repeated calls.

        ``sub_tuple`` must be a hashable tuple of ``(name_or_symbol, value)``
        pairs (e.g. ``tuple(sorted(point.items()))``). Results are memoized in a
        bounded, instance-level cache so repeated calls are cheap and the cache
        is collected together with the tensor (an ``lru_cache`` on a method would
        pin every instance it ever saw).
        """
        if sub_tuple in self._subs_cache:
            self._cache_hits += 1
            return self._subs_cache[sub_tuple]

        self._cache_misses += 1
        result = self.subs(dict(sub_tuple))
        if len(self._subs_cache) >= self._max_cached_tensors:
            # drop the oldest entry (dicts preserve insertion order)
            self._subs_cache.pop(next(iter(self._subs_cache)))
        self._subs_cache[sub_tuple] = result
        return result

    def __repr__(self) -> str:
        success_rate = self._success_count / self._operation_count if self._operation_count > 0 else 1.0
        return (f"NanoTensor(name={self.name!r}, shape={self.shape}, "
                f"max_order={self.max_order}, "
                f"health={self._health_status}, "
                f"success_rate={success_rate:.3f}, "
                f"ops={self._operation_count})")

    def _repr_html_(self) -> str:
        # Simple HTML summary; you can make this fancier if you want
        from html import escape
        coeffs_preview = ", ".join([escape(str(c)) for c in self.coeff_vars[:10]])
        return f"""
        <div>
          <strong>NanoTensor</strong> <code>{escape(self.name)}</code><br/>
          Shape: {escape(str(self.shape))}<br/>
          Max order: {escape(str(self.max_order))}<br/>
          Base vars: {escape(", ".join(v.name for v in self.base_vars))}<br/>
          Coeff vars (preview): <code>{coeffs_preview}</code>
        </div>
        """

    @staticmethod
    def _batch_length(point: Dict[str, Any]) -> int:
        """
        Determine the broadcast batch size implied by ``point``.

        Returns ``1`` when every value is a scalar, otherwise the common length
        of all sequence-valued entries. Mixed lengths raise ``ValueError``.
        """
        length = 1
        for name, value in point.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                n = len(value)
                if length not in (1, n):
                    raise ValueError(
                        "eval_numeric(): batched inputs must all have the same length; "
                        f"'{name}' has {n} entries, another variable has {length}"
                    )
                length = n
        return length

    def _to_numeric(self, values: List[Any], missing: List[str] = ()) -> np.ndarray:
        """
        Convert a flat list of evaluation results into a float array.

        Scalars collapse to shape ``(numel,)``; array-valued results (a batched
        evaluation) are broadcast to a common length and stacked into shape
        ``(numel, batch)``.

        Raises
        ------
        ValueError
            When any element still contains free symbols (i.e. a coefficient was
            never assigned). The message names the offending symbols and points
            at the fix, instead of returning an opaque ``dtype=object`` array.
        """
        arrs: List[np.ndarray] = []
        for v in values:
            try:
                arrs.append(np.asarray(v, dtype=float))
            except (TypeError, ValueError):
                names = sorted(
                    {str(x) for val in values for x in getattr(val, 'free_symbols', ())}
                ) or sorted(missing)
                raise ValueError(
                    f"eval_numeric(): {len(names)} symbol(s) in tensor "
                    f"'{self.name}' have no numeric value: {names}. Assign them "
                    "with apply_linear_coeffs(...)/subs(...) or pass them in `point`."
                ) from None

        if all(a.ndim == 0 for a in arrs):
            return np.array([float(a) for a in arrs], dtype=float)

        shape = np.broadcast_shapes(*(a.shape for a in arrs))
        return np.stack([np.broadcast_to(a, shape).astype(float) for a in arrs])

    def eval_numeric(self, point: Dict[str, float], use_lambdify: bool = True) -> np.ndarray:
        """
        Evaluate the tensor numerically at a point, or over a batch of points.

        Parameters
        ----------
        point : dict[str, float | Sequence[float]]
            Mapping from variable name to value. A value may also be a sequence
            (list / tuple / ndarray) of equal length, in which case the tensor is
            broadcast over that batch and the result gains a trailing axis.
        use_lambdify : bool, optional
            Use cached ``sympy.lambdify`` callables (default, fast). Set to
            ``False`` to force the pure SymPy ``subs``/``evalf`` path, which is
            slower but tolerates exotic element types.

        Returns
        -------
        numpy.ndarray
            Shape ``self.shape`` for scalar input, shape ``self.shape + (n,)``
            when evaluating ``n`` points at once. Always ``dtype=float``.

        Notes
        -----
        Military-grade behaviour: inputs are validated against configured
        bounds, evaluation time is fed to the anomaly detector, and the compiled
        callables are memoized per (symbols, batch-size) signature. Symbols that
        appear in the tensor but not in ``point`` raise ``ValueError`` naming
        them, rather than silently producing symbolic output.
        """
        start_time = time.time()
        success = True
        error_msg = None

        try:
            # Validate inputs (military-grade security)
            for var_name, value in point.items():
                self._validate_input(var_name, value)

            batch = self._batch_length(point)
            vars_in_point = [v for v in self.symvars if v.name in point]
            missing = sorted(v.name for v in self.symvars if v.name not in point)

            # Cache key: the compiled callables depend only on which symbols are
            # passed and on the broadcast shape, never on the values themselves.
            key = (tuple(v.name for v in vars_in_point), batch)

            # Fast path: one CSE-reduced callable compiled for *all* symbols.
            if self._optimized_func is not None and batch == 1 and not missing:
                opt_func = self._optimized_func[1]
                try:
                    arr = np.asarray(opt_func(*[
                        float(point[v.name]) for v in self._optimized_func[0]
                    ]), dtype=float)
                    if arr.size == self.data.size:
                        return arr.reshape(self.shape)
                except (TypeError, ValueError) as exc:
                    logger.debug(
                        "eval_numeric(): optimized callable unusable (%r); using cache path", exc
                    )

            if not use_lambdify:
                def eval_one(col_index: Optional[int]):
                    subs = {
                        sp.Symbol(k): (float(v) if col_index is None else float(np.asarray(v)[col_index]))
                        for k, v in point.items()
                    }
                    vals = []
                    for e in self.data.flat:
                        v = e.subs(subs) if hasattr(e, 'subs') else e
                        vals.append(v.evalf() if hasattr(v, 'evalf') else v)
                    return vals

                if batch == 1:
                    flat = self._to_numeric(eval_one(None), missing)
                else:
                    flat = np.stack([self._to_numeric(eval_one(j), missing) for j in range(batch)])
            else:
                if key not in self._lambdify_cache:
                    self._lambdify_cache[key] = [
                        lambdify(vars_in_point, e, modules='numpy') for e in self.data.flat
                    ]
                funcs = self._lambdify_cache[key]
                args = [np.asarray(point[v.name], dtype=float) for v in vars_in_point]
                values = [f(*args) for f in funcs]
                flat = self._to_numeric(values, missing)

            # scalar point -> tensor shape; batched point -> tensor shape + (n,)
            if batch == 1:
                return flat.reshape(self.shape)
            if flat.ndim == 1:
                flat = np.broadcast_to(flat.reshape(self.data.size, 1), (self.data.size, batch))
            return flat.reshape((*self.shape, batch))

        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration = time.time() - start_time
            # Anomaly detection using pattern *before* recording this operation
            if "evaluation" in self._learned_patterns:
                pattern = self._learned_patterns["evaluation"]
                if pattern["count"] > 10 and pattern["avg_duration"] > 0:
                    deviation = abs(duration - pattern["avg_duration"]) / pattern["avg_duration"]
                    if deviation > self._anomaly_threshold:
                        warnings.warn(
                            f"Anomalous evaluation detected: {duration:.4f}s "
                            f"vs avg {pattern['avg_duration']:.4f}s",
                            stacklevel=2,
                        )

            self._record_operation("evaluation", duration, success, error_msg)

    def apply_linear_coeffs(self, coeffs: Dict[str, float]):
        """
        Apply externally provided first-order coefficients to the polynomial.

        This is intended for scenarios where coefficients are estimated or
        optimized outside of Python (e.g. in JavaScript or another runtime)
        and then injected back into the symbolic tensor.

        Parameters
        ----------
        coeffs : dict[str | sympy.Symbol, float]
            Mapping from coefficient symbol names (e.g. 'g_k', 'g_a') -- or the
            symbols themselves -- to numeric values. These are substituted into
            the tensor and the internal caches are invalidated.
        """
        subs_d = {}
        for name, val in coeffs.items():
            key = name.name if isinstance(name, sp.Symbol) else str(name)
            try:
                subs_d[sp.Symbol(key)] = float(val)
            except (TypeError, ValueError):
                # Exact solutions (sqrt(2), Rational(...), ...) stay symbolic.
                subs_d[sp.Symbol(key)] = sp.sympify(val)
        self.data = np.vectorize(
            lambda e: e.subs(subs_d) if hasattr(e, 'subs') else e, otypes=[object]
        )(self.data)
        self._invalidate_caches()
        # Keep a record (keyed by symbol name), and forget coefficients that are
        # now fully numeric.
        self.fitted_coeffs.update({k.name if isinstance(k, sp.Symbol) else str(k): v
                                   for k, v in coeffs.items()})
        self.coeff_vars = [c for c in self.coeff_vars if c.name not in subs_d]

    def solve_poly(self, target_eq: sp.Expr, var: sp.Symbol) -> List[float]:
        """
        Solve a univariate polynomial equation and return real numeric roots.

        Parameters
        ----------
        target_eq : sympy.Expr
            Polynomial expression in `var` that should equal zero.
        var : sympy.Symbol
            Variable to solve for.

        Returns
        -------
        list[float]
            List of real roots (within a small imaginary tolerance) extracted
            from SymPy's high-precision numeric root finder. Returns an empty
            list if solving fails.
        """
        try:
            poly = Poly(sp.simplify(target_eq), var)
            roots = poly.nroots()  # numeric roots with high precision
        except Exception as e:
            logger.debug("solve_poly(): no polynomial form in %s: %r", var, e)
            return []

        # Filter real roots within tolerance
        real_roots = []
        for r in roots:
            try:
                re, im = sp.re(r), sp.im(r)
                if abs(float(im)) < 1e-8:
                    real_roots.append(float(re))
            except (TypeError, ValueError):
                # Root is not numerically resolvable (e.g. remains symbolic).
                continue
        return real_roots

    def groebner_solve(self, poly_system: List[sp.Expr],
                   vars_to_solve: Optional[List[sp.Symbol]] = None) -> List[Dict[str, sp.Expr]]:
        """
        Solve a polynomial system using Gröbner bases and filter for real solutions.

        Parameters
        ----------
        poly_system : list[sympy.Expr]
            List of polynomial equations assumed equal to zero.
        vars_to_solve : list[sympy.Symbol], optional
            Variables to solve for. If omitted, a subset of `symvars`
            is used heuristically.

        Returns
        -------
        list[dict[str, sympy.Expr]]
            Real-valued solutions, represented as dictionaries mapping
            variable names to SymPy expressions. Returns an empty list if
            solving fails or no real solutions are found.
        """
        if vars_to_solve is None:
            vars_to_solve = self.symvars[:len(poly_system)]
        try:
            # The reduced Groebner basis generates the same solution set as the
            # input system but is triangular, so solving it is usually cheaper
            # and more robust than solving the original system directly.
            G = groebner(poly_system, *vars_to_solve, order='lex')
            try:
                solutions = solve(list(G.polys), *vars_to_solve, dict=True)
            except Exception as exc:
                logger.debug("groebner_solve(): basis solve failed (%r), solving input system", exc)
                solutions = solve(poly_system, *vars_to_solve, dict=True)

            # Filter solutions with real values
            real_solutions = []
            for sol in solutions:
                if all(abs(v.as_real_imag()[1]) < 1e-8 for v in sol.values()):
                    real_solutions.append({str(k): v for k, v in sol.items()})
            return real_solutions
        except Exception as e:
            logger.warning("groebner_solve() failed: %s", e)
            return []

    def resultant(self, f: sp.Expr, g: sp.Expr, var: sp.Symbol) -> sp.Expr:
        """Compute resultant of two polynomials with respect to variable"""
        return resultant(f, g, var)

    def parametrize_curve(self, implicit_poly: sp.Expr,
                          t: Optional[sp.Symbol] = None) -> Tuple[sp.Expr, sp.Expr]:
        """
        Parametrize algebraic curve via line intersection + resultant.
        Implements the method from ScienceDirect PDF (Winkler).
        For curve f(x,y)=0, uses line y=tx through singular point.
        """
        if t is None:
            t = sp.Symbol('t')
        x, y = sp.symbols('x y')

        # Ensure implicit_poly is in terms of x,y
        implicit_poly = safe_sympify(implicit_poly)

        # Line through origin (assuming singular point at origin)
        line = y - t * x

        # Compute resultant to eliminate y
        res_y = resultant(implicit_poly, line, y)

        # Solve for x in terms of t
        x_sols = sp.solve(res_y, x)
        # Filter out trivial solution (x=0)
        non_trivial = [sol for sol in x_sols if sol != 0]

        if non_trivial:
            x_param = sp.simplify(non_trivial[-1])
            y_param = sp.simplify(t * x_param)
            return x_param, y_param
        else:
            return x, y

    def generate_taylor(self, center: Dict[str, float], ss_value: sp.Expr = None,
                        include_bias: bool = True):
        """
        Construct a multivariate Taylor polynomial around a steady state.

        The expansion is built in terms of deviations (x - x_ss) for each
        base variable, up to `max_order`. Coefficients are introduced as
        new symbolic variables (e.g. g_k, g_a, g_k_k, g_k_a).

        Parameters
        ----------
        center : dict[str, float]
            Steady-state values for each base variable, keyed by name.
        ss_value : sympy.Expr, optional
            Steady-state level of the function being approximated. If None,
            defaults to the sum of the base variables.
        include_bias : bool, optional
            If True, prepend a constant coefficient term g_bias.

        Notes
        -----
        This method populates `self.data` with a single symbolic Taylor
        polynomial (broadcast across the tensor) and updates `coeff_vars`
        with the newly created coefficient symbols.
        """
        # Clear coefficient list and any caches that depend on self.data
        self.coeff_vars.clear()
        self._symvars_cache = None
        self._lambdify_cache.clear()
        self._diff_cache.clear()

        if ss_value is None:
            ss_value = sum(list(self.base_vars))  # default to sum

        devs = [sp.Symbol(v.name) - center.get(v.name, 0) for v in self.base_vars]
        taylor_expr = ss_value

        if include_bias:
            c0 = sp.Symbol('g_bias')
            self.coeff_vars.append(c0)
            taylor_expr += c0

        # Linear terms
        for v, dev in zip(self.base_vars, devs, strict=True):
            c = sp.Symbol(f'g_{v.name}')
            self.coeff_vars.append(c)
            taylor_expr += c * dev

        # Quadratic and higher-order terms (diagonal + mixed partials).
        # Term for multi-index (i, j, ...) is  g_{v_i}_{v_j}... / (n_i! n_j! ...) * prod(dev**n)
        for order in range(2, self.max_order + 1):
            for combo in combinations_with_replacement(range(len(self.base_vars)), order):
                names = '_'.join(self.base_vars[i].name for i in combo)
                c = sp.Symbol(f'g_{names}')
                if c in self.coeff_vars:
                    continue  # duplicate monomial (cannot happen, kept as a guard)
                self.coeff_vars.append(c)
                counts = Counter(combo)
                factor = sp.Rational(1, math.prod(
                    math.factorial(n) for n in counts.values()
                ))
                monomial = sp.Mul(*[devs[i] ** counts[i] for i in counts])
                taylor_expr += factor * c * monomial

        self.data.flat[:] = sp.simplify(taylor_expr)
        # After changing data, every derived cache must be considered stale
        self._invalidate_caches()


    @staticmethod
    def _is_small_polynomial_system(eqs: List[sp.Expr],
                                    vars_ss: List[sp.Symbol]) -> bool:
        """
        Whether an exact :func:`sympy.solve` is cheap enough to be worth trying.

        ``solve`` is *unbounded* work on anything else: a residual with rational
        powers or exponentials sends it into heuristic GCD loops that never
        return. Only small square polynomial systems are considered safe.
        """
        if not eqs or len(vars_ss) > 3:
            return False
        try:
            polys = [sp.Poly(sp.together(eq).as_numer_denom()[0], *vars_ss)
                     for eq in eqs]
        except sp.PolyError:
            return False
        return all(poly.total_degree() <= 4 for poly in polys if poly)

    def compute_steady_state(self, model_eqs: List[sp.Expr],
                            params: Dict[str, float],
                            ss_guess: Optional[Dict[str, float]] = None,
                            method: str = 'auto') -> Dict[str, float]:
        """
        Compute a steady state for a system of model equations.

        With ``method='auto'`` (the default) the exact symbolic solve is attempted
        only for small polynomial systems -- otherwise the numeric ``nsolve`` is
        used straight away, since an unbounded symbolic search is a hang, not a
        slow answer. Force one path with ``method='symbolic'`` or
        ``method='numeric'``.

        Parameters
        ----------
        method : {'auto', 'symbolic', 'numeric'}
            Which solver to use. ``'symbolic'`` raises if the exact solve fails;
            ``'numeric'`` never calls ``solve``; ``'auto'`` picks by system size.
        model_eqs : list[sympy.Expr]
            List of residual equations defining the steady state.
        params : dict[str, float]
            Model parameters substituted into the equations (e.g. alpha, beta).
        ss_guess : dict[str, float], optional
            Initial guess for the steady state, keyed by variable name.
            If omitted, all base vars are initialized to 1.0.

        Returns
        -------
        dict[str, float]
            Dictionary mapping variable names to numeric steady-state values.
            If both symbolic and numeric attempts fail, the original guess
            is returned as a fallback.
        """
        if method not in ('auto', 'symbolic', 'numeric'):
            raise ValueError(
                f"compute_steady_state(): method must be 'auto', 'symbolic' or "
                f"'numeric', got {method!r}")

        if ss_guess is None:
            ss_guess = {str(v): 1.0 for v in self.base_vars}

        # Substitute parameters
        subs_params = {sp.Symbol(k): v for k, v in params.items()}
        eqs_ss = [eq.subs(subs_params) for eq in model_eqs]
        vars_ss = [sp.Symbol(k) for k in ss_guess]

        want_symbolic = (method == 'symbolic'
                        or (method == 'auto'
                            and self._is_small_polynomial_system(eqs_ss, vars_ss)))
        if want_symbolic:
            try:
                sols = solve(eqs_ss, vars_ss, dict=True)
                if sols:
                    sol = sols[0]
                    return {str(k): float(v.evalf()) for k, v in sol.items()}
                raise ValueError("solve() returned no candidate")
            except Exception as exc:
                if method == 'symbolic':
                    raise ValueError(
                        f"compute_steady_state(): the exact solve failed ({exc!r}); "
                        f"use method='numeric' or 'auto'") from exc
                logger.debug("compute_steady_state(): exact solve failed (%r), "
                             "using the numeric solve", exc)

        # Newton solve on the whole system. For a single equation nsolve takes the
        # (equation, variable, guess) form; systems need the list form.
        try:
            if len(vars_ss) == 1:
                root = nsolve(eqs_ss[0], vars_ss[0], ss_guess[str(vars_ss[0])])
                return {str(vars_ss[0]): float(root)}
            if len(eqs_ss) == len(vars_ss):
                root = nsolve(eqs_ss, vars_ss,
                              [ss_guess[str(var)] for var in vars_ss])
                return {str(var): float(root[i]) for i, var in enumerate(vars_ss)}
            # Not square (macro-style perturbations pass one residual plus the full
            # state/shock vector): solve every equation that isolates a single
            # variable and keep the caller's guess for the others. Solving one
            # equation once per variable, as an earlier revision did, asks nsolve
            # to vary several symbols at once and fails outright.
            solution = {str(var): float(ss_guess[str(var)]) for var in vars_ss}
            for eq in eqs_ss:
                present = [var for var in vars_ss if var in eq.free_symbols]
                if len(present) != 1:
                    continue
                var = present[0]
                solution[str(var)] = float(nsolve(eq, var, solution[str(var)]))
            return solution
        except Exception as e:
            logger.warning("Steady state computation failed: %s; returning the guess", e)
            return dict(ss_guess)

    def full_perturbation(self, model_R: sp.Expr, params: Dict[str, Any],
                         var_order: Optional[List[str]] = None,
                         eps_var: float = 1.0,
                         ss_guess: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Perform a full 2nd-order perturbation around the steady state.

        This routine follows a macro-style perturbation approach (inspired by
        standard RBC implementations and related Northwestern-style notes):

        1. Compute the steady state of the residual R = 0.
        2. Construct a Taylor polynomial in the state variables.
        3. Sequentially solve first-order conditions R_x = 0 for each coefficient
           g_x using polynomial root finding.
        4. If `max_order >= 2`, solve second-order conditions (diagonal and
           cross-partials) for g_xx and g_xy.
        5. Treat the variance of the shock (`eps_var`) explicitly when handling
           sigma-related terms.

        Parameters
        ----------
        model_R : sympy.Expr
            Residual function R(k, a, eps, sig, ...) to be set to zero.
        params : dict[str, Any]
            Mapping of model parameter names to numeric values.
        var_order : list[str], optional
            Ordered list of variable names used in the perturbation sequence.
        eps_var : float, optional
            Variance term used when substituting E[eps'^2].
        ss_guess : dict[str, float], optional
            Initial steady-state guess passed to `compute_steady_state`.

        Returns
        -------
        dict[str, float]
            The fitted coefficient map (also stored on ``self.fitted_coeffs``).

        Notes
        -----
        Fitted coefficients are stored in `self.fitted_coeffs`, and the
        final Taylor polynomial with substituted coefficients is written
        back into `self.data`.

        This is a *sequential* (equation-by-equation) solver suited to the
        single-residual models this tensor is designed around. For general
        multi-equation systems use :mod:`symbo.analytics.perturbation`, which
        solves the linearized system jointly.
        """
        if var_order is None:
            var_order = ['k', 'a', 'eps', 'sig']

        # Compute steady state
        ss = self.compute_steady_state([model_R], params, ss_guess=ss_guess)
        self.coeff_vars.clear()
        self.generate_taylor(ss)
        self.fitted_coeffs = {}

        # Create symbol mapping
        sym_vars = {v.name: v for v in self.base_vars}
        subs_ss = {**{sp.Symbol(k): v for k, v in ss.items()},
                  **{sp.Symbol(k): v for k, v in params.items()}}

        # Substitute the Taylor policy into the model residual if k_next appears.
        policy_expr = self.data.flat[0]
        k_next_sym = sp.Symbol('k_next')
        if k_next_sym in model_R.free_symbols:
            model_R_policy = model_R.subs(k_next_sym, policy_expr)
        else:
            model_R_policy = model_R

        logger.info("Computed steady state: %s", ss)

        # First-order conditions: R_x = 0
        for vname in var_order:
            if vname not in sym_vars:
                continue
            wrt = sym_vars[vname]
            # Differentiate and evaluate at steady state
            R_v = sp.simplify(sp.diff(model_R_policy, wrt))
            R_v_ss = R_v.subs({**subs_ss, **{sp.Symbol(k): v for k, v in self.fitted_coeffs.items()}})

            coeff_name = f'g_{vname}'
            coeff_sym = sp.Symbol(coeff_name)

            # Solve linear equation for coefficient
            try:
                # Make it polynomial in coeff
                poly_eq = sp.expand(R_v_ss.subs(coeff_sym, sp.Symbol('_c')))
                poly_eq = poly_eq.subs(sp.Symbol('_c'), coeff_sym)
                roots = self.solve_poly(poly_eq, coeff_sym)
                if roots:
                    # Choose economically meaningful root (stable, small magnitude)
                    candidates = [r for r in roots if abs(r) < 10] or roots
                    stable_root = min(candidates, key=abs)
                    self.fitted_coeffs[coeff_name] = stable_root
                    logger.info("  %s = %.6f", coeff_name, stable_root)
            except Exception as e:
                logger.warning("  Failed to solve %s: %s", coeff_name, e)

        # Second-order conditions
        if self.max_order >= 2:
            logger.info("Solving second-order terms...")
            for i, v1 in enumerate(var_order):
                if v1 not in sym_vars:
                    continue
                wrt1 = sym_vars[v1]

                # Diagonal terms
                R_vv = sp.diff(model_R_policy, wrt1, 2)
                R_vv_ss = R_vv.subs({**subs_ss, **{sp.Symbol(k): v for k, v in self.fitted_coeffs.items()}})
                coeff_name = f'g_{v1}_{v1}'
                coeff_sym = sp.Symbol(coeff_name)

                try:
                    roots = self.solve_poly(R_vv_ss.subs(coeff_sym, sp.Symbol('_c')).subs(sp.Symbol('_c'), coeff_sym), coeff_sym)
                    if roots:
                        self.fitted_coeffs[coeff_name] = roots[0]
                        logger.info("  %s = %.6f", coeff_name, roots[0])
                except Exception as exc:
                    logger.debug("  Second-order term %s defaulted to 0 (%r)", coeff_name, exc)
                    self.fitted_coeffs[coeff_name] = 0.0

                # Cross terms
                for j in range(i+1, len(var_order)):
                    v2 = var_order[j]
                    if v2 not in sym_vars:
                        continue
                    wrt2 = sym_vars[v2]

                    R_v1v2 = sp.diff(sp.diff(model_R_policy, wrt1), wrt2)
                    R_v1v2_ss = R_v1v2.subs({**subs_ss, **{sp.Symbol(k): v for k, v in self.fitted_coeffs.items()}})
                    coeff_name = f'g_{v1}_{v2}'
                    coeff_sym = sp.Symbol(coeff_name)

                    try:
                        roots = self.solve_poly(R_v1v2_ss.subs(coeff_sym, sp.Symbol('_c')).subs(sp.Symbol('_c'), coeff_sym), coeff_sym)
                        if roots:
                            self.fitted_coeffs[coeff_name] = roots[0]
                            logger.info("  %s = %.6f", coeff_name, roots[0])
                    except Exception as exc:
                        logger.debug("  Cross term %s defaulted to 0 (%r)", coeff_name, exc)
                        self.fitted_coeffs[coeff_name] = 0.0

        # Special handling for sigma (variance term)
        # R_σσ = h_22 + h_33 * Var(ε')
        if 'sig' in var_order:
            sig_sym = sym_vars['sig']
            R_ss = sp.diff(model_R_policy, sig_sym, 2)
            subs_variance = {sp.Symbol('eps_next')**2: eps_var}
            R_ss_sub = R_ss.subs(subs_variance)
            R_ss_ss = sp.simplify(R_ss_sub.subs({**subs_ss, **{sp.Symbol(k): v for k, v in self.fitted_coeffs.items()}}))

            coeff_name = 'g_sig_sig'
            coeff_sym = sp.Symbol(coeff_name)
            try:
                # Solve R_ss_ss = 0 for g_sig_sig
                sol = nsolve(R_ss_ss, coeff_sym, 0.1)  # educated guess
                self.fitted_coeffs[coeff_name] = float(sol.evalf())
            except Exception as e:
                logger.warning("Failed to solve sigma term: %s", e)
                self.fitted_coeffs[coeff_name] = 0.0

        # Apply solution to tensor (every element, not just the first).
        solved = self.subs({sp.Symbol(k): v for k, v in self.fitted_coeffs.items()})
        self.data = solved.data
        self._invalidate_caches()
        return dict(self.fitted_coeffs)

    def simplify(self):
        """
        Simplify all expressions in tensor (military-grade enhanced).

        Now includes performance tracking and automatic optimization.
        """
        start_time = time.time()
        success = True
        error_msg = None

        try:
            self.data = np.vectorize(sp.simplify, otypes=[object])(self.data)
            # simplify() mutates in place: every derived cache (lambdified
            # callables and the CSE-optimized callable included) is now stale.
            self._invalidate_caches()
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration = time.time() - start_time
            self._record_operation("simplification", duration, success, error_msg)

    def deriv_tree(self, wrt_vars: List[str]) -> Dict[str, sp.Expr]:
        """
        Build a simple derivative tree for explainability.

        For each variable name in `wrt_vars`, this computes the partial
        derivative of the first tensor entry with respect to that variable
        and returns a dict mapping the variable name to the simplified
        derivative expression.

        Parameters
        ----------
        wrt_vars : list[str]
            Variable names to differentiate with respect to
            (e.g. ["k", "a", "eps", "sig"]).

        Returns
        -------
        dict[str, sympy.Expr]
            Mapping from variable name to its corresponding
            partial derivative expression. If a derivative cannot
            be computed, the value is set to the symbol 'NA'.
        """
        tree: Dict[str, sp.Expr] = {}

        for var_name in wrt_vars:
            try:
                # Try to find the actual SymPy symbol by name among the tensor's symbols
                wrt = next(
                    (sym for sym in self.symvars if sym.name == var_name),
                    sp.Symbol(var_name)  # fallback if not present in symvars yet
                )

                deriv_tensor = self.diff(wrt, order=1)
                deriv_expr = deriv_tensor.data.flat[0]
                tree[var_name] = sp.simplify(deriv_expr)
            except Exception as exc:
                logger.debug("deriv_tree(): derivative w.r.t. '%s' unavailable: %r", var_name, exc)
                tree[var_name] = sp.Symbol("NA")

        return tree

    def _require_scalar_tensor(self, feature: str) -> None:
        """Raise unless this tensor holds exactly one symbolic expression."""
        if self.data.size != 1:
            raise ValueError(
                f"{feature}() expects a scalar tensor (shape (1,)), but '{self.name}' "
                f"has shape {self.shape}. Build the model with shape=(1,) or evaluate "
                "an explicit element."
            )

    def compute_grid(self, var1: str, var2: str,
                     fixed: Optional[Dict[str, float]] = None,
                     range1: Tuple[float, float] = (-0.5, 1.5),
                     range2: Tuple[float, float] = (-0.2, 0.2),
                     n1: int = 100,
                     n2: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample the scalar tensor value over a 2D grid in (var1, var2).

        This helper evaluates the tensor at a grid of points, holding all other
        variables fixed. It is primarily used to support contour and surface
        visualizations as well as pathfinding.

        Returns
        -------
        X, Y, Z : numpy.ndarray
            Arrays of shape ``(n1, n2)``: ``X[i, j]`` is the `var1` coordinate,
            ``Y[i, j]`` the `var2` coordinate and ``Z[i, j]`` the evaluated value
            at that point.
        """
        if fixed is None:
            fixed = {}
        self._require_scalar_tensor("compute_grid")

        x = np.linspace(range1[0], range1[1], n1)
        y = np.linspace(range2[0], range2[1], n2)
        # 'ij' indexing keeps the first grid axis attached to `var1`, so
        # X[i, j] == x[i], Y[i, j] == y[j] and Z[i, j] is the value at
        # (x[i], y[j]) -- the convention every consumer of this method uses.
        X, Y = np.meshgrid(x, y, indexing='ij')

        # One batched evaluation instead of n1*n2 python-level calls: the same
        # compiled callable is reused and numpy does the grid arithmetic.
        point = dict(fixed)
        point[var1] = X.ravel()
        point[var2] = Y.ravel()
        Z = self.eval_numeric(point)
        return X, Y, np.asarray(Z, dtype=float).reshape(X.shape)

    def plot_contour(self, var1: str, var2: str,
                     fixed: Optional[Dict[str, float]] = None,
                     levels: int = 20,
                     range1: Tuple[float, float] = (-0.5, 1.5),
                     range2: Tuple[float, float] = (-0.2, 0.2),
                     n: int = 100,
                     ax: Optional[Any] = None):
        """
        2D contour plot of the tensor value (requires ``matplotlib``).

        Parameters
        ----------
        var1, var2 : str
            Names of the two variables to sweep.
        fixed : dict[str, float], optional
            Values for all remaining free symbols.
        levels : int
            Number of contour levels.
        range1, range2 : tuple[float, float]
            Sweep ranges.
        n : int
            Grid resolution in each direction.
        ax : matplotlib.axes.Axes, optional
            If provided, draw into that Axes (no ``plt.show()``). Otherwise a new
            figure is created and shown.

        Returns
        -------
        matplotlib.figure.Figure
            The figure the plot was drawn on.
        """
        plt = require("matplotlib.pyplot", feature="NanoTensor.plot_contour")

        X, Y, Z = self.compute_grid(var1, var2, fixed=fixed,
                                     range1=range1, range2=range2,
                                     n1=n, n2=n)

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            created_fig = True
        else:
            fig = ax.get_figure()

        contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
        fig.colorbar(contour, ax=ax)
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_title(f'Contour plot: {self.name}')

        if created_fig:
            plt.show()
        return fig

    def plot_surface(self, var1: str, var2: str,
                     fixed: Optional[Dict[str, float]] = None,
                     range1: Tuple[float, float] = (-0.5, 1.5),
                     range2: Tuple[float, float] = (-0.2, 0.2),
                     n: int = 50,
                     ax: Optional[Any] = None):
        """
        3D surface plot using Plotly (requires ``plotly``).

        If ``ax`` is provided it should be an existing
        ``plotly.graph_objects.Figure`` and the surface is added as a trace (no
        ``fig.show()``). Otherwise a new Figure is created and shown.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        go = require("plotly.graph_objects", feature="NanoTensor.plot_surface")

        X, Y, Z = self.compute_grid(var1, var2, fixed=fixed,
                                    range1=range1, range2=range2,
                                    n1=n, n2=n)
        surface = go.Surface(x=X, y=Y, z=Z)

        if ax is None:
            fig = go.Figure(data=[surface])
            fig.update_layout(
                title=f'Surface plot: {self.name}',
                scene={
                    "xaxis_title": var1,
                    "yaxis_title": var2,
                    "zaxis_title": 'Value'
                }
            )
            fig.show()
        else:
            # Assume ax is a plotly Figure; user controls layout/showing
            fig = ax
            fig.add_trace(surface)
        return fig

    def plot_grid_with_path(self,
                            var1: str,
                            var2: str,
                            start: Tuple[int, int],
                            goal: Tuple[int, int],
                            fixed: Optional[Dict[str, float]] = None,
                            range1: Tuple[float, float] = (-0.5, 1.5),
                            range2: Tuple[float, float] = (-0.2, 0.2),
                            n1: int = 100,
                            n2: int = 100,
                            mode: str = "min",
                            ax: Optional[Any] = None):
        """
        Visualize an A*-computed path over a sampled 2D cost or value landscape.

        The method:
        - samples the tensor into a grid,
        - runs A* using `find_path_on_grid`, and
        - overlays the resulting path on a contour plot.

        It is useful for reasoning about trajectories across energy or value
        landscapes. Requires ``matplotlib``.

        Returns
        -------
        tuple[matplotlib.figure.Figure, list[tuple[int, int]]]
            The figure and the path (grid indices) that was overlaid.
        """
        plt = require("matplotlib.pyplot", feature="NanoTensor.plot_grid_with_path")

        X, Y, Z = self.compute_grid(var1, var2,
                                    fixed=fixed,
                                    range1=range1,
                                    range2=range2,
                                    n1=n1,
                                    n2=n2)

        path = self.find_path_on_grid(Z, start=start, goal=goal, mode=mode)

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            created_fig = True
        else:
            fig = ax.get_figure()

        contour = ax.contourf(X, Y, Z, levels=30)
        fig.colorbar(contour, ax=ax)
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_title(f'Pathfinding on {self.name}')

        if path:
            px = [X[i, j] for (i, j) in path]
            py = [Y[i, j] for (i, j) in path]
            ax.plot(px, py, linewidth=2)

        if created_fig:
            plt.show()
        return fig, path

    def reason_path(self,
                    var1: str,
                    var2: str,
                    start_state: Dict[str, float],
                    goal_state: Dict[str, float],
                    range1: Tuple[float, float],
                    range2: Tuple[float, float],
                    n1: int = 100,
                    n2: int = 100,
                    mode: str = "min") -> Dict[str, Any]:
        """
        High-level reasoning utility for continuous start and goal states.

        The method:
        - maps continuous (var1, var2) start/goal values to nearest grid indices,
        - calls `find_path_on_grid` to compute a discrete path,
        - returns both index-level and coordinate-level descriptions.

        This provides a bridge between continuous model states and discrete
        pathfinding semantics.
        """

        X, Y, Z = self.compute_grid(var1, var2,
                                    fixed={k: v for k, v in start_state.items()
                                           if k not in (var1, var2)},
                                    range1=range1,
                                    range2=range2,
                                    n1=n1,
                                    n2=n2)

        # Map start/goal to nearest grid indices
        def closest_idx(val, grid_axis):
            return int(np.argmin(np.abs(grid_axis - val)))

        start_i = closest_idx(start_state.get(var1, 0.0), X[:, 0])
        start_j = closest_idx(start_state.get(var2, 0.0), Y[0, :])
        goal_i = closest_idx(goal_state.get(var1, 0.0), X[:, 0])
        goal_j = closest_idx(goal_state.get(var2, 0.0), Y[0, :])

        path_idx = self.find_path_on_grid(Z, (start_i, start_j), (goal_i, goal_j), mode=mode)

        path_coords = [{
            var1: float(X[i, j]),
            var2: float(Y[i, j]),
            "value": float(Z[i, j])
        } for (i, j) in path_idx]

        return {
            "indices": path_idx,
            "path": path_coords,
            "grid": (X, Y, Z),
        }



def deriv_tree(nt: 'NanoTensor', wrt_vars: List[str]) -> Dict[str, sp.Expr]:
    """
    Module-level companion of :meth:`NanoTensor.deriv_tree`.

    Kept for backwards compatibility with call sites that used the historic
    free-function form ``deriv_tree(nt, ["k", "a"])``.
    """
    return nt.deriv_tree(wrt_vars)


class SymbolicTrainer:
    """
    Trainer for NanoTensor-based symbolic models.

    This class encapsulates several fitting strategies:

    - 'symbolic'      : use Gröbner-based solving for exact coefficient fitting;
    - 'perturbation'  : perform a 2nd-order perturbation around a steady state;
    - 'lsq'           : fit coefficients via ordinary least squares.

    It also integrates with a small `KnowledgeBase` to store learned policy
    coefficients as symbolic facts.
    """

    def __init__(self, nano_tensor: NanoTensor, tolerance: float = 1e-8):
        self.nt = nano_tensor
        self.tolerance = tolerance
        self.fitted_coeffs: Dict[str, float] = {}
        self.kb = KnowledgeBase()

    def fit(self, data: List[Any], method: str = 'symbolic') -> bool:
        """
        Fit the underlying NanoTensor using a chosen method.

        Parameters
        ----------
        data : list
            For 'symbolic' / 'lsq':
                A list of (state_dict, target_value) pairs, where state_dict
                maps variable names to numeric values.

            For 'perturbation':
                A list where the first element is (model_R, params_dict),
                i.e. a residual expression and parameter dictionary.

        method : {'symbolic', 'perturbation', 'lsq'}
            Fitting strategy to employ.

        Returns
        -------
        bool
            True if fitting succeeds and coefficients are updated, False otherwise.
        """
        if method == 'symbolic':
            return self._fit_symbolic(data)
        elif method == 'perturbation':
            if len(data) >= 1:
                # Accept either [ (model_R, params) ], [ (model_R, params, ss_guess) ], or [model_R, params, ss_guess]
                if isinstance(data[0], tuple) and len(data[0]) in (2, 3):
                    model_R, params, *maybe_guess = data[0]
                elif len(data) >= 2:
                    model_R, params, *maybe_guess = [*data, None]
                else:
                    return False

                ss_guess = maybe_guess[0] if maybe_guess else None
                self.nt.full_perturbation(model_R, params, ss_guess=ss_guess)
                self.fitted_coeffs = getattr(self.nt, 'fitted_coeffs', {})
                return len(self.fitted_coeffs) > 0
        elif method == 'lsq':
            return self._fit_lsq(data)

        return False

    def _fit_symbolic(self, data: List[Tuple[Dict[str, float], float]]) -> bool:
        """
        Exact fitting via Groebner bases.

        Builds one equation per data point (tensor value == target) and solves
        the system for the tensor's coefficient symbols. Only meaningful when the
        number of independent data points is at least the number of unknowns;
        with fewer points the system is underdetermined and `solve` may return
        partially-specified solutions.
        """
        equations = []
        coeffs_sym = self.nt.coeff_vars

        if not coeffs_sym:
            logger.warning(
                "_fit_symbolic(): tensor '%s' has no coefficient symbols; call "
                "generate_taylor() first", self.nt.name
            )
            return False
        if len(data) < len(coeffs_sym):
            logger.warning(
                "_fit_symbolic(): %d data point(s) for %d unknown coefficient(s) - "
                "the fit is underdetermined", len(data), len(coeffs_sym)
            )

        for state_point, target in data:
            subs_d = {self.nt.base_vars[i]: state_point.get(str(v), 0)
                     for i, v in enumerate(self.nt.base_vars)}
            nt_val = sum(self.nt.subs(subs_d).data.flat)
            equations.append(sp.Eq(nt_val, target))

        sols = self.nt.groebner_solve(equations, coeffs_sym)
        if sols:
            sol_dict = {k: float(v.evalf()) for k, v in sols[0].items()}
            self._apply_solution(sol_dict)
            return True
        return False

    def _fit_lsq(self, data: List[Tuple[Dict[str, float], float]]) -> bool:
        """
        Least-squares fit of the tensor's coefficient symbols.

        The Taylor ansatz is *linear* in its coefficients, so the fit reduces to
        a linear regression on the monomials that multiply each coefficient:

            E(x; c) = base(x) + sum_i c_i * m_i(x)

        `base(x)` is E evaluated with all coefficients set to zero and `m_i` is
        ``dE/dc_i``. Fitting the *coefficients* (and not the state variables, as
        an earlier revision of this method did) keeps the returned policy a
        function of its arguments.
        """
        nt = self.nt
        if nt.data.size != 1:
            raise ValueError(
                f"_fit_lsq() needs a scalar tensor; '{nt.name}' has shape {nt.shape}"
            )
        coeffs_sym = list(nt.coeff_vars)
        if not coeffs_sym:
            logger.warning(
                "_fit_lsq(): tensor '%s' has no coefficient symbols; call "
                "generate_taylor() first", nt.name
            )
            return False

        expr = nt.data.flat[0]
        base_expr = expr.subs(dict.fromkeys(coeffs_sym, 0))
        monomials = [sp.simplify(expr.coeff(c) - base_expr.coeff(c)) for c in coeffs_sym]

        lam_base = lambdify(nt.base_vars, base_expr, modules='numpy')
        lam_mons = [lambdify(nt.base_vars, m, modules='numpy') for m in monomials]

        def args_for(state: Dict[str, float]):
            return [float(state.get(v.name, 0.0)) for v in nt.base_vars]

        # Design matrix: rows are data points, columns are coefficients.
        A = np.array([[float(f(*args_for(state))) for f in lam_mons] for state, _ in data])
        b = np.array([
            float(target) - float(lam_base(*args_for(state))) for state, target in data
        ])
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        sol_dict = {c.name: float(coeffs[i]) for i, c in enumerate(coeffs_sym)}
        self._apply_solution(sol_dict)
        return True

    def _apply_solution(self, sol_dict: Dict[str, float]):
        """
        Apply fitted coefficients to the tensor and to the knowledge base.

        The coefficients are substituted *in place* on ``self.nt``: the tensor the
        caller handed to the trainer is the tensor that ends up fitted, which is
        what ``fit()`` promises and what ``save_brain``/``predict`` assume.
        """
        self.nt.apply_linear_coeffs(sol_dict)
        self.fitted_coeffs = dict(sol_dict)

        # Update knowledge base
        for k, v in sol_dict.items():
            self.kb.add_fact('policy_coefficient', k, v)

    def predict(self, state_point: Dict[str, float]) -> np.ndarray:
        """Make prediction at given state"""
        return self.nt.eval_numeric(state_point)


class HybridTrainer(SymbolicTrainer):
    """Hybrid trainer combining symbolic and numeric methods"""

    def symbolic_regression(self, X: np.ndarray, y: np.ndarray,
                           max_deg: int = 5, n_calls: int = 30) -> NanoTensor:
        """
        Symbolic regression via GP optimization of the Taylor degree (PySR-like).

        Requires the optional ``scikit-optimize`` package (extra ``opt``); a
        deterministic grid search over degrees is used when it is unavailable.

        Parameters
        ----------
        X, y:
            Training features and targets.
        max_deg:
            Maximum Taylor order to consider.
        n_calls:
            Number of Gaussian-process evaluations.

        Returns
        -------
        NanoTensor
            A tensor whose `data` holds the best-order Taylor ansatz, fitted by
            least squares (coefficients available through ``fit(..., 'lsq')``).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features); got shape {X.shape}")

        def objective(deg):
            nt_try = NanoTensor((1,), max_order=int(deg[0]),
                               base_vars=[f'x{i}' for i in range(X.shape[1])])
            nt_try.generate_taylor({f'x{i}': 0 for i in range(X.shape[1])})
            trainer_try = HybridTrainer(nt_try)

            # Create data tuples
            data = []
            for j in range(len(y)):
                state = {f'x{i}': X[j,i] for i in range(X.shape[1])}
                data.append((state, y[j]))

            trainer_try.fit(data, method='lsq')
            pred = trainer_try.predict_batch(X)
            return np.mean((pred - y)**2)

        bounds = [(0, max(1, int(max_deg)))]
        skopt = optional_module("skopt")
        if skopt is None:
            logger.info(
                "symbolic_regression(): scikit-optimize not installed; falling back "
                "to a deterministic grid search over orders 1..%d", max_deg
            )
            degrees = range(1, max(1, int(max_deg)) + 1)
            scores = [(objective([d]), d) for d in degrees]
            best_deg = min(scores, key=lambda t: t[0])[1]
        else:
            res = skopt.gp_minimize(objective, bounds, n_calls=int(n_calls), random_state=42)
            best_deg = int(res.x[0])

        best_nt = NanoTensor((1,), max_order=max(1, best_deg),
                            base_vars=[f'x{i}' for i in range(X.shape[1])])
        best_nt.generate_taylor({f'x{i}': 0 for i in range(X.shape[1])})

        # Fit the winning order, as promised by the docstring: the returned
        # tensor already carries numeric coefficients.
        names = [f'x{i}' for i in range(X.shape[1])]
        data = [({n: X[j, i] for i, n in enumerate(names)}, float(y[j]))
                for j in range(len(y))]
        HybridTrainer(best_nt).fit(data, method='lsq')

        return best_nt

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Batch prediction for numeric data.

        Column ``i`` of ``X`` is fed to the tensor's ``i``-th base variable
        (``x0..xn`` by convention). Evaluation is performed in a single batched
        ``eval_numeric`` call.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features); got shape {X.shape}")
        names = [v.name for v in self.nt.base_vars]
        if X.shape[1] != len(names):
            raise ValueError(
                f"predict_batch(): X has {X.shape[1]} feature column(s) but tensor "
                f"'{self.nt.name}' expects {len(names)} variables {names}"
            )
        point = {name: X[:, i] for i, name in enumerate(names)}
        return np.asarray(self.nt.eval_numeric(point)).reshape(-1)

    @staticmethod
    def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 32,
                    shuffle: bool = True):
        """
        Convenience builder for a float32 ``torch.utils.data.DataLoader``.

        Requires the optional ``torch`` package (extra ``neuro``).
        """
        torch = require("torch", feature="HybridTrainer.make_loader")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def multi_obj_fit(self, sparse_data: List, dense_data: np.ndarray,
                     alpha_exact: float = 0.7):
        """
        Pareto optimization: combine symbolic (sparse) and numeric (dense).
        Uses weighted average of coefficients.
        """
        if dense_data.shape[0] > 0:
            X, y = dense_data[:, :-1], dense_data[:, -1]
            nt_dense = self.symbolic_regression(X, y)
            trainer_dense = HybridTrainer(nt_dense)
            trainer_dense.fit([({f'x{i}': X[j,i] for i in range(X.shape[1])}, y[j])
                              for j in range(len(y))], 'lsq')
        else:
            trainer_dense = self

        trainer_sparse = HybridTrainer(self.nt)
        trainer_sparse.fit(sparse_data, 'symbolic')

        # Merge coefficients
        all_keys = set(list(trainer_sparse.fitted_coeffs.keys()) +
                      list(trainer_dense.fitted_coeffs.keys()))

        for k in all_keys:
            val_sparse = trainer_sparse.fitted_coeffs.get(k, 0)
            val_dense = trainer_dense.fitted_coeffs.get(k, 0)
            self.fitted_coeffs[k] = alpha_exact * val_sparse + (1 - alpha_exact) * val_dense

        self._apply_solution(self.fitted_coeffs)

    def torch_fit(self, loader, epochs: int = 100,
                  lr: float = 1e-3, weight_decay: float = 0.0,
                  betas: Tuple[float, float] = (0.9, 0.999),
                  var_order: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Neuro-symbolic training: optimize symbolic coefficients with PyTorch.

        The tensor's coefficient symbols are registered as ``nn.Parameter``s and
        the symbolic expression is compiled with ``lambdify(..., modules='torch')``
        so autograd flows straight into the interpretable coefficients.

        Parameters
        ----------
        loader:
            A ``torch.utils.data.DataLoader`` yielding ``(batch_x, batch_y)``
            batches (build one with :meth:`make_loader`).
        epochs, lr, weight_decay, betas:
            Standard optimizer settings.
        var_order:
            Names of the variables the *columns* of ``batch_x`` correspond to.
            Defaults to the tensor's base variables in declaration order. Passing
            an explicit order removes the silent risk of a feature/column mismatch.

        Returns
        -------
        dict[str, float]
            The fitted coefficients (also stored on ``self.fitted_coeffs`` and
            substituted into the tensor).

        Notes
        -----
        Requires the optional ``torch`` package (extra ``neuro``).
        """
        torch = require("torch", feature="HybridTrainer.torch_fit")
        nn = require("torch.nn", feature="HybridTrainer.torch_fit")

        nt = self.nt
        if nt.data.size != 1:
            raise ValueError(
                f"torch_fit() needs a scalar tensor; '{nt.name}' has shape {nt.shape}"
            )
        if not nt.coeff_vars:
            raise ValueError(
                f"torch_fit(): tensor '{nt.name}' has no coefficient symbols; call "
                "generate_taylor() first"
            )

        names = [v.name for v in nt.base_vars] if var_order is None else list(var_order)
        unknown = [n for n in names if n not in {v.name for v in nt.symvars}
                   and n not in names and sp.Symbol(n) not in nt.base_vars]
        if unknown:
            raise ValueError(
                f"torch_fit(): var_order entries {unknown} are not variables of "
                f"tensor '{nt.name}' (known: {names})"
            )
        # Columns of a batch are mapped positionally onto `names`; the lambdify
        # signature is built in that *same* explicit order.
        vars_syms = [sp.Symbol(n) for n in names]
        expr = nt.data.flat[0]

        class SymModule(nn.Module):
            def __init__(self, nt: NanoTensor):
                super().__init__()
                self.nt = nt
                # Register coefficients as parameters
                self.coeff_params = nn.ParameterDict({
                    c.name: nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
                    for c in nt.coeff_vars
                })
                # Torch-friendly callable for the current symbolic expression
                self._lambda = sp.lambdify(list(vars_syms) + list(nt.coeff_vars),
                                           expr, modules='torch')

            def forward(self, x):
                # x: (batch, n_vars) -> one tensor per variable, then the coeffs
                args = [x[:, i] for i in range(len(vars_syms))]
                coeff_vals = [self.coeff_params[c.name] for c in self.nt.coeff_vars]
                return self._lambda(*args, *coeff_vals)

        module = SymModule(nt)
        opt = torch.optim.Adam(module.parameters(), lr=lr,
                               weight_decay=weight_decay, betas=betas)
        loss_fn = nn.MSELoss()

        n_features = len(names)
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.float()
                batch_y = batch_y.float().reshape(-1)
                if batch_x.ndim != 2 or batch_x.shape[1] != n_features:
                    raise ValueError(
                        f"torch_fit(): expected batches of shape (batch, {n_features}) "
                        f"for variables {names}, got {tuple(batch_x.shape)}"
                    )

                opt.zero_grad()
                pred = module(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                opt.step()
                total_loss += float(loss.item())

            if epoch % 20 == 0:
                logger.info("torch_fit epoch %d: loss = %.6f", epoch, total_loss)

        # Extract final coefficients
        self.fitted_coeffs.update({
            k: float(v.item()) for k, v in module.coeff_params.items()
        })
        self._apply_solution(self.fitted_coeffs)
        return dict(self.fitted_coeffs)


class KnowledgeBase:
    """
    Lightweight symbolic knowledge base for learned policy coefficients.

    Internally, this class maintains:

    - a directed labeled graph (`networkx.DiGraph`) for structural queries,
    - a `kanren` relation for logic-programming style queries, and
    - a plain dict index that is always available.

    networkx and kanren are optional: when they are missing the dict index
    transparently serves both query shapes, so the knowledge base stays usable
    in a minimal (numpy + sympy only) install.

    It is used to store and query facts such as policy coefficients learned
    during training, enabling downstream reasoning over symbolic parameters.
    """

    def __init__(self):
        self._nx = optional_module("networkx")
        self._kanren = optional_module("kanren")
        self.graph = self._nx.DiGraph() if self._nx is not None else None
        self.rules = None
        if self._kanren is not None:
            self.rules = self._kanren.Relation('policy')
        # Always-on index: {entity: {prop: [values...]}}
        self._index: Dict[str, Dict[str, List[Any]]] = {}

    @property
    def backend(self) -> str:
        """Human readable description of which optional backends are active."""
        return f"networkx={self.graph is not None}, kanren={self.rules is not None}"

    def add_fact(self, entity: str, prop: str, value: Any):
        """Add fact to KB"""
        self._index.setdefault(entity, {}).setdefault(prop, []).append(value)
        if self.graph is not None:
            # networkx stores any object; keep the raw value so symbolic or
            # string facts survive the round trip.
            self.graph.add_edge(entity, prop, value=value)
        if self.rules is not None:
            self._kanren.facts(self.rules, (entity, prop, value))

    def query(self, entity: str, prop: Optional[str] = None) -> List[Any]:
        """
        Query KB for entity/property.

        With a `prop`, the matching values are returned (coerced to `float` when
        the fact is numeric). Without a `prop`, all ``(entity, prop, value)``
        triples for that entity are returned.
        """
        if prop is not None:
            if self.rules is not None:
                kvar, run = self._kanren.var, self._kanren.run
                q = kvar()
                results = run(5, q, self.rules(entity, prop, q))
                return [KnowledgeBase._as_number(r) for r in results]
            return [self._as_number(v) for v in self._index.get(entity, {}).get(prop, [])]

        if self.graph is not None and self.graph.has_node(entity):
            return [
                (u, v, attrs.get('value'))
                for u, v, attrs in self.graph.out_edges(entity, data=True)
            ]
        return [
            (entity, p, v)
            for p, vals in self._index.get(entity, {}).items()
            for v in vals
        ]

    @staticmethod
    def _as_number(value: Any) -> Any:
        """Return a float for numeric facts, and the original object otherwise."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


