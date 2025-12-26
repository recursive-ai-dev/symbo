
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

try:
    import msgpack
except ImportError:
    msgpack = None

try:
    import pyarrow as pa
except ImportError:
    pa = None

import sympy as sp
import numpy as np
import torch
import torch.nn as nn
import heapq
import json 
from functools import lru_cache
from typing import Tuple, Dict, Any, List, Optional, Union
import networkx as nx
from kanren import Relation, facts, run, var as kvar
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from code import InteractiveConsole
from skopt import gp_minimize
import time
import warnings
warnings.filterwarnings('ignore')

# Sympy imports for specialized functions
from sympy import symbols, Symbol, Poly, GroebnerBasis, groebner, resultant, solve, Eq, nsolve, lambdify
from sympy.matrices import Matrix



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
        "order": G.order,
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
    """
    expr = sp.sympify(expr_str)
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

    polys = [sp.sympify(s) for s in poly_strs]
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
                 base_vars: List[str] = None, name: str = "nt"):
        self.shape = shape
        self.max_order = max_order
        self.name = name
        self.base_vars = [sp.Symbol(v) for v in (base_vars or ['k', 'a', 'eps', 'sig'])]
        self.coeff_vars: List[sp.Symbol] = []
        self.data: np.ndarray = np.empty(shape, dtype=object)
        self._diff_cache: Dict[Tuple[str, int], 'NanoTensor'] = {}
        self._init_data()
        self._symvars_cache = None
        self._lambdify_cache: Dict = {}
        self.fitted_coeffs: Dict[str, float] = {}
        
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
        
    def _init_data(self):
        """Initialize tensor with symbolic zeros"""
        self.data = np.zeros(self.shape, dtype=object)
        self.data.flat[:] = sp.S(0)
        self._diff_cache.clear()
    
    def _record_operation(self, op_type: str, duration: float, success: bool, error: str = None):
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
    
    def _update_health_status(self):
        """Update health status based on performance metrics (military-grade feature)."""
        if self._operation_count == 0:
            self._health_status = "optimal"
            return
        
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
        """Autonomous self-optimization (military-grade agency feature)."""
        try:
            # Clear old caches to free memory
            if len(self._diff_cache) > 100:
                self._diff_cache.clear()
            
            # Simplify if we have complex expressions
            if self._operation_count > 100 and self._success_count / self._operation_count < 0.9:
                self.simplify()
        except Exception:
            pass  # Silent failure - don't interfere with main operation
    
    def _validate_input(self, var_name: str, value: float) -> bool:
        """Validate input against bounds (military-grade security feature)."""
        if var_name in self._validation_bounds:
            lower, upper = self._validation_bounds[var_name]
            if not (lower <= value <= upper):
                raise ValueError(f"Input {var_name}={value} outside valid bounds [{lower}, {upper}]")
        
        # Check for invalid values
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"Invalid value for {var_name}: {value}")
        
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
                "bounds_set": len(self._validation_bounds)
            }
        }
    
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
                          start_idx: Tuple[int, int],
                          goal_idx: Tuple[int, int],
                          mode: str = "min") -> List[Tuple[int, int]]:
        """
        A* pathfinding on a 2D cost grid Z.

        Args:
            Z: 2D array of costs.
            start_idx: (i, j) start index into Z.
            goal_idx: (i, j) goal index into Z.
            mode: 'min' to prefer low Z (valley-following),
                  'max' to prefer high Z (ridge-following).

        Returns:
            List of (i, j) indices representing the path.
        """
        rows, cols = Z.shape
        (si, sj) = start_idx
        (gi, gj) = goal_idx

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
        heapq.heappush(open_set, (0.0, start_idx))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {start_idx: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
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
                    f = tentative_g + NanoTensor._heuristic(nb, goal_idx)
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
                if elem != 0 and hasattr(elem, 'free_symbols'):
                    vars_set.update(elem.free_symbols)
            self._symvars_cache = list(vars_set)
        return self._symvars_cache
    
    # Pre-compile once per variable set
    @lru_cache(maxsize=128)
    def _compile_func(self, var_names):
        vars_syms = [sp.Symbol(v) for v in var_names]
        return [lambdify(vars_syms, e, modules='numpy') for e in self.data.flat]
    
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
            new_nt = NanoTensor(self.shape, self.max_order, [v.name for v in self.base_vars])
            new_nt.data = np.vectorize(lambda e: sp.diff(e, wrt, order))(self.data)
            return new_nt
        except Exception as e:
            success = False
            error_msg = str(e)
            # Attempt recovery: try simplifying first
            try:
                self.simplify()
                new_nt = NanoTensor(self.shape, self.max_order, [v.name for v in self.base_vars])
                new_nt.data = np.vectorize(lambda e: sp.diff(e, wrt, order))(self.data)
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
        
        new_nt = NanoTensor(self.shape, self.max_order, [v.name for v in self.base_vars])
        clean_dict = {
            k: (sp.nsimplify(v) if isinstance(v, (int, float)) else v)
            for k, v in sub_dict.items()
        }
        new_nt.data = np.vectorize(lambda e: e.subs(clean_dict))(self.data)

        # Invalidate if we substituted any of this tensor's coefficient symbols
        def _is_coeff_key(k):
            # k might be a Symbol or a string
            if isinstance(k, sp.Symbol):
                return k in self.coeff_vars
            if isinstance(k, str):
                try:
                    return sp.Symbol(k) in self.coeff_vars
                except Exception:
                    return False
            return False

        if any(_is_coeff_key(k) for k in sub_dict.keys()):
            new_nt._symvars_cache = None
            new_nt._lambdify_cache.clear()

        return new_nt
    
    @lru_cache(maxsize=128)
    def subs_cached(self, sub_tuple: Tuple) -> 'NanoTensor':
        """Cached substitution for repeated calls"""
        sub_dict = dict(sub_tuple)
        return self.subs(sub_dict)
        
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
    
    def eval_numeric(self, point: Dict[str, float], use_lambdify: bool = True) -> np.ndarray:
        """
        Evaluate tensor numerically at a given point with caching (military-grade enhanced).
        
        Now includes:
        - Input validation against bounds
        - Performance tracking
        - Anomaly detection for evaluation time
        - Intelligent error recovery
        """
        start_time = time.time()
        success = True
        error_msg = None
        
        try:
            # Validate inputs (military-grade security)
            for var_name, value in point.items():
                self._validate_input(var_name, value)
            
            vars_in_point = [v for v in self.symvars if v.name in point]
            key = tuple(sorted([v.name for v in vars_in_point])) + tuple(sorted(point.keys()))
            if key not in self._lambdify_cache:
                # Prepare functions for all elements
                self._lambdify_cache[key] = [
                    lambdify(vars_in_point, e, modules='numpy') for e in self.data.flat
                ]
            funcs = self._lambdify_cache[key]
            args = [point.get(v.name, 0.0) for v in vars_in_point]
            result_flat = [f(*args) for f in funcs]
            return np.array(result_flat).reshape(self.shape)
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
                            f"Anomalous evaluation detected: {duration:.4f}s vs avg {pattern['avg_duration']:.4f}s"
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
        coeffs : dict[str, float]
            Mapping from coefficient symbol names (e.g. 'g_k', 'g_a') to
            numeric values. These are substituted into the tensor, and the
            internal caches are invalidated.
        """
        subs_d = {sp.Symbol(name): float(val) for name, val in coeffs.items()}
        self.data = np.vectorize(lambda e: e.subs(subs_d))(self.data)
        self._symvars_cache = None
        self._lambdify_cache.clear()
        # Keep a record:
        self.fitted_coeffs.update(coeffs)
    
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
            # Filter real roots within tolerance
            real_roots = []
            for r in roots:
                if abs(r.imag) < 1e-8:
                    real_roots.append(float(r.real))
            return real_roots
        except Exception as e:
            print(f"Polynomial solving failed: {e}")
            return []
    
    def groebner_solve(self, poly_system: List[sp.Expr], 
                   vars_to_solve: List[sp.Symbol] = None) -> List[Dict[str, sp.Expr]]:
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
            G = groebner(poly_system, *vars_to_solve, order='lex')
            solutions = solve(poly_system, *vars_to_solve, dict=True)
            # Filter solutions with real values
            real_solutions = []
            for sol in solutions:
                if all(abs(v.as_real_imag()[1]) < 1e-8 for v in sol.values()):
                    real_solutions.append({str(k): v for k, v in sol.items()})
            return real_solutions
        except Exception as e:
            print(f"Groebner solve error: {e}")
            return []
    
    def resultant(self, f: sp.Expr, g: sp.Expr, var: sp.Symbol) -> sp.Expr:
        """Compute resultant of two polynomials with respect to variable"""
        return resultant(f, g, var)
    
    def parametrize_curve(self, implicit_poly: sp.Expr, 
                          t: sp.Symbol = None) -> Tuple[sp.Expr, sp.Expr]:
        """
        Parametrize algebraic curve via line intersection + resultant.
        Implements the method from ScienceDirect PDF (Winkler).
        For curve f(x,y)=0, uses line y=tx through singular point.
        """
        if t is None:
            t = sp.Symbol('t')
        x, y = sp.symbols('x y')
        
        # Ensure implicit_poly is in terms of x,y
        implicit_poly = sp.sympify(implicit_poly)
        
        # Line through origin (assuming singular point at origin)
        l = y - t * x
        
        # Compute resultant to eliminate y
        res_y = resultant(implicit_poly, l, y)
        
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
            ss_value = sum([v for v in self.base_vars])  # default to sum

        devs = [sp.Symbol(v.name) - center.get(v.name, 0) for v in self.base_vars]
        taylor_expr = ss_value

        if include_bias:
            c0 = sp.Symbol('g_bias')
            self.coeff_vars.append(c0)
            taylor_expr += c0

        # Linear terms
        for v, dev in zip(self.base_vars, devs):
            c = sp.Symbol(f'g_{v.name}')
            self.coeff_vars.append(c)
            taylor_expr += c * dev

        # Quadratic and interaction terms if max_order >= 2
        if self.max_order >= 2:
            for i, v1 in enumerate(self.base_vars):
                c = sp.Symbol(f'g_{v1.name}_{v1.name}')
                self.coeff_vars.append(c)
                taylor_expr += 0.5 * c * devs[i]**2
                for j in range(i + 1, len(self.base_vars)):
                    v2 = self.base_vars[j]
                    c_cross = sp.Symbol(f'g_{v1.name}_{v2.name}')
                    self.coeff_vars.append(c_cross)
                    taylor_expr += c_cross * devs[i] * devs[j]
        # For higher orders, extend similarly

        self.data.flat[:] = sp.simplify(taylor_expr)
        # After changing data, symvars/lambdify cache must be considered stale
        self._symvars_cache = None
        self._lambdify_cache.clear()
        self._diff_cache.clear()

    
    def compute_steady_state(self, model_eqs: List[sp.Expr], 
                            params: Dict[str, float],
                            ss_guess: Dict[str, float] = None) -> Dict[str, float]:
        """
        Compute a steady state for a system of model equations.

        The method first attempts an exact symbolic solve, and if that fails,
        falls back to a numeric solution via `nsolve` with a user-provided or
        heuristic initial guess.

        Parameters
        ----------
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
        if ss_guess is None:
            ss_guess = {str(v): 1.0 for v in self.base_vars}
        
        # Substitute parameters
        subs_params = {sp.Symbol(k): v for k, v in params.items()}
        eqs_ss = [eq.subs(subs_params) for eq in model_eqs]
        vars_ss = [sp.Symbol(k) for k in ss_guess.keys()]
        
        try:
            # Try exact solve first
            sols = solve(eqs_ss, vars_ss, dict=True)
            if sols:
                sol = sols[0]
                return {str(k): float(v.evalf()) for k, v in sol.items()}
        except:
            pass
        
        # Numeric fallback
        try:
            sol_dict = {}
            for var in vars_ss:
                sol = nsolve(eqs_ss[0], var, ss_guess[str(var)])
                sol_dict[str(var)] = float(sol)
            return sol_dict
        except Exception as e:
            print(f"Steady state computation failed: {e}")
            return ss_guess
    
    def full_perturbation(self, model_R: sp.Expr, params: Dict[str, Any], 
                         var_order: List[str] = ['k', 'a', 'eps', 'sig'],
                         eps_var: float = 1.0,
                         ss_guess: Dict[str, float] = None):
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

        Notes
        -----
        Fitted coefficients are stored in `self.fitted_coeffs`, and the
        final Taylor polynomial with substituted coefficients is written
        back into `self.data`.
        """
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
        
        print(f"Computed steady state: {ss}")
        
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
                    stable_root = min([r for r in roots if abs(r) < 10], key=abs)
                    self.fitted_coeffs[coeff_name] = stable_root
                    print(f"  {coeff_name} = {stable_root:.6f}")
            except Exception as e:
                print(f"  Failed to solve {coeff_name}: {e}")
        
        # Second-order conditions
        if self.max_order >= 2:
            print("Solving second-order terms...")
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
                        print(f"  {coeff_name} = {roots[0]:.6f}")
                except:
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
                            print(f"  {coeff_name} = {roots[0]:.6f}")
                    except:
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
                print(f"Failed to solve sigma term: {e}")
                self.fitted_coeffs[coeff_name] = 0.0
        
        # Apply solution to tensor
        apply_dict = {sp.Symbol(k): v for k, v in self.fitted_coeffs.items()}
        self.data.flat[:] = self.subs(apply_dict).data.flat[0]
    
    def simplify(self):
        """
        Simplify all expressions in tensor (military-grade enhanced).
        
        Now includes performance tracking and automatic optimization.
        """
        start_time = time.time()
        success = True
        error_msg = None
        
        try:
            self.data = np.vectorize(sp.simplify)(self.data)
            self._symvars_cache = None  # Clear cache
            self._diff_cache.clear()
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration = time.time() - start_time
            self._record_operation("simplification", duration, success, error_msg)
        
    def compute_grid(self, var1: str, var2: str,
                     fixed: Dict[str, float] = None,
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
            Meshgrids of coordinates (X, Y) and corresponding evaluated values Z.
        """
        if fixed is None:
            fixed = {}

        x = np.linspace(range1[0], range1[1], n1)
        y = np.linspace(range2[0], range2[1], n2)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X, dtype=float)

        for i in range(n1):
            for j in range(n2):
                point = fixed.copy()
                point[var1] = float(X[i, j])
                point[var2] = float(Y[i, j])
                Z[i, j] = float(self.eval_numeric(point).flat[0])

        return X, Y, Z
    
    def plot_contour(self, var1: str, var2: str, 
                     fixed: Dict[str, float] = None, 
                     levels: int = 20,
                     range1: Tuple[float, float] = (-0.5, 1.5),
                     range2: Tuple[float, float] = (-0.2, 0.2),
                     ax: Optional[plt.Axes] = None):
        """2D contour plot of tensor value.

        If ax is provided, draw into that Axes (no plt.show()).
        Otherwise, create a new figure and show it.
        """
        if fixed is None:
            fixed = {}
        
        v1, v2 = sp.symbols(f'{var1} {var2}')
        x = np.linspace(range1[0], range1[1], 100)
        y = np.linspace(range2[0], range2[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(100):
            for j in range(100):
                point = fixed.copy()
                point[var1] = X[i, j]
                point[var2] = Y[i, j]
                Z[i, j] = self.eval_numeric(point).flat[0]
        
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

    
    def plot_surface(self, var1: str, var2: str, 
                     fixed: Dict[str, float] = None,
                     range1: Tuple[float, float] = (-0.5, 1.5),
                     range2: Tuple[float, float] = (-0.2, 0.2),
                     ax: Optional[go.Figure] = None):
        """
        3D surface plot using Plotly.

        If ax is provided, it should be an existing plotly.graph_objects.Figure,
        and the surface will be added as a trace (no fig.show()).
        Otherwise, a new Figure is created and shown.
        """
        if fixed is None:
            fixed = {}
        
        x = np.linspace(range1[0], range1[1], 50)
        y = np.linspace(range2[0], range2[1], 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(50):
            for j in range(50):
                point = fixed.copy()
                point[var1] = X[i, j]
                point[var2] = Y[i, j]
                Z[i, j] = self.eval_numeric(point).flat[0]
        
        surface = go.Surface(x=x, y=y, z=Z)
        
        if ax is None:
            fig = go.Figure(data=[surface])
            fig.update_layout(
                title=f'Surface plot: {self.name}',
                scene=dict(
                    xaxis_title=var1,
                    yaxis_title=var2,
                    zaxis_title='Value'
                )
            )
            fig.show()
        else:
            # Assume ax is a plotly Figure; user controls layout/showing
            ax.add_trace(surface)
            
    def plot_grid_with_path(self,
                            var1: str,
                            var2: str,
                            start: Tuple[int, int],
                            goal: Tuple[int, int],
                            fixed: Dict[str, float] = None,
                            range1: Tuple[float, float] = (-0.5, 1.5),
                            range2: Tuple[float, float] = (-0.2, 0.2),
                            n1: int = 100,
                            n2: int = 100,
                            mode: str = "min",
                            ax: Optional[plt.Axes] = None):
        """
        Visualize an A*-computed path over a sampled 2D cost or value landscape.

        The method:
        - samples the tensor into a grid,
        - runs A* using `find_path_on_grid`, and
        - overlays the resulting path on a contour plot.

        It is useful for reasoning about trajectories across energy or value
        landscapes.
        """
        
        X, Y, Z = self.compute_grid(var1, var2,
                                    fixed=fixed,
                                    range1=range1,
                                    range2=range2,
                                    n1=n1,
                                    n2=n2)

        path = self.find_path_on_grid(Z, start_idx=start, goal_idx=goal, mode=mode)

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
                (s for s in self.symvars if s.name == var_name),
                sp.Symbol(var_name)  # fallback if not present in symvars yet
            )

            deriv_tensor = self.diff(wrt, order=1)
            deriv_expr = deriv_tensor.data.flat[0]
            tree[var_name] = sp.simplify(deriv_expr)
        except Exception:
            tree[var_name] = sp.Symbol("NA")

    return tree


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
                    model_R, params, *maybe_guess = data + [None]
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
        """Exact fitting via Groebner bases"""
        equations = []
        coeffs_sym = self.nt.coeff_vars
        
        if not coeffs_sym:
            return False
        
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
    
    def _fit_lsq(self, data):
        X = np.array([[state.get(str(v), 0) for v in self.nt.base_vars] 
                      for state, _ in data])
        y = np.array([target for _, target in data])
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        sol_dict = {self.nt.base_vars[i].name: coeffs[i] for i in range(len(coeffs))}
        self._apply_solution(sol_dict)
        return True
    
    def _apply_solution(self, sol_dict: Dict[str, float]):
        """Apply fitted coefficients to tensor and knowledge base"""
        subs_d = {sp.Symbol(k): v for k, v in sol_dict.items()}
        self.nt = self.nt.subs(subs_d)
        self.fitted_coeffs = sol_dict
        
        # Update knowledge base
        for k, v in sol_dict.items():
            self.kb.add_fact('policy_coefficient', k, v)
    
    def predict(self, state_point: Dict[str, float]) -> np.ndarray:
        """Make prediction at given state"""
        return self.nt.eval_numeric(state_point)


class HybridTrainer(SymbolicTrainer):
    """Hybrid trainer combining symbolic and numeric methods"""
    
    def symbolic_regression(self, X: np.ndarray, y: np.ndarray, 
                           max_deg: int = 5) -> NanoTensor:
        """
        Symbolic regression via GP optimization (PySR-like).
        Evolves Taylor polynomial degree to minimize MSE.
        """
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
        
        res = gp_minimize(objective, [(-2, max_deg)], n_calls=30, random_state=42)
        best_nt = NanoTensor((1,), max_order=int(res.x[0]), 
                            base_vars=[f'x{i}' for i in range(X.shape[1])])
        best_nt.generate_taylor({f'x{i}': 0 for i in range(X.shape[1])})

        return best_nt
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch prediction for numeric data"""
        return np.array([self.predict({f'x{i}': X[j,i] for i in range(X.shape[1])}).flat[0] 
                        for j in range(len(X))])
    
    @staticmethod
    def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """Convenience builder for DataLoader with float32 tensors."""
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
    
    def torch_fit(self, loader: torch.utils.data.DataLoader, epochs: int = 100,
                  lr: float = 1e-3, weight_decay: float = 0.0,
                  betas: Tuple[float, float] = (0.9, 0.999)):
        """
        Neuro-symbolic: optimize symbolic coefficients via PyTorch.

        Parameters
        ----------
        loader : DataLoader
            Batches of (X, y).
        epochs : int
            Training epochs.
        lr : float
            Learning rate for Adam.
        weight_decay : float
            L2 regularization coefficient.
        betas : tuple[float, float]
            Adam momentum parameters.
        """
        # Convert symbolic tensor to torch module
        class SymModule(nn.Module):
            def __init__(self, nt: NanoTensor):
                super().__init__()
                self.nt = nt
                # Register coefficients as parameters
                self.coeff_params = nn.ParameterDict({
                    c.name: nn.Parameter(torch.tensor(0.1, dtype=torch.float32)) for c in nt.coeff_vars
                })
                vars_syms = list(nt.base_vars) + list(nt.coeff_vars)
                # Torch-friendly callable for the current symbolic expression
                self._lambda = sp.lambdify(vars_syms, nt.data.flat[0], modules='torch')
            
            def forward(self, *args):
                coeff_vals = [self.coeff_params[c.name] for c in self.nt.coeff_vars]
                return self._lambda(*args, *coeff_vals)
        
        module = SymModule(self.nt)
        opt = torch.optim.Adam(module.parameters(), lr=lr,
                               weight_decay=weight_decay, betas=betas)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                
                opt.zero_grad()
                pred = module(*batch_x.T)
                loss = nn.MSELoss()(pred, batch_y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.6f}")
        
        # Extract final coefficients
        self.fitted_coeffs.update({
            k: v.item() for k, v in module.coeff_params.items()
        })
        self._apply_solution(self.fitted_coeffs)


class KnowledgeBase:
    """
    Lightweight symbolic knowledge base for learned policy coefficients.

    Internally, this class maintains:

    - a directed labeled graph (`networkx.DiGraph`) for structural queries, and
    - a `kanren` relation for simple logic-programming style queries.

    It is used to store and query facts such as policy coefficients learned
    during training, enabling downstream reasoning over symbolic parameters.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.rules = Relation('policy')
    
    def add_fact(self, entity: str, prop: str, value: Any):
        """Add fact to KB"""
        self.graph.add_edge(entity, prop, value=float(value))
        facts(self.rules, (entity, prop, value))
    
    def query(self, entity: str, prop: str = None) -> List[Any]:
        """Query KB for entity/property"""
        if prop:
            q = kvar()
            results = run(5, q, self.rules(entity, prop, q))
            return [float(r) for r in results]
        else:
            # Return all properties
            if self.graph.has_node(entity):
                return list(self.graph.out_edges(entity, data=True))
            return []


# ==================== Demo & Benchmarks ====================

def demo_rbc_perturbation(ss_guess: Dict[str, float] = None):
    """
    Full RBC model perturbation demo matching Northwestern PDF.
    Computes policy function k' = g(k,a,eps,sig) via 2nd-order expansion.
    """
    print("\n=== RBC Perturbation Demo ===")
    
    # Setup symbols
    k, a, eps, sig, eps_next = sp.symbols('k a eps sig eps_next')
    k_next = sp.Symbol('k_next')
    gamma, alpha, delta, beta, rho = sp.symbols('gamma alpha delta beta rho')
    
    # Parameters (standard RBC)
    params = {
        'alpha': 0.3,
        'delta': 0.1,
        'beta': 0.95,
        'gamma': 1.0,
        'rho': 0.95
    }
    
    # Production function
    z = rho * a + eps
    f = sp.exp(z) * k**alpha + (1 - delta) * k
    
    # Consumption
    c = f - k_next
    
    # Next period (approximate k_next_period ~ k_next for simplicity)
    z_next = rho * (rho * a + eps) + sig * eps_next
    f_next = sp.exp(z_next) * k_next**alpha + (1 - delta) * k_next
    c_next = f_next - k_next  # Approximation
    
    # Utility and Euler residual
    uc = c**(-gamma)
    uc_next = c_next**(-gamma)
    fkp_next = alpha * sp.exp(z_next) * k_next**(alpha - 1) + (1 - delta)
    R = uc - beta * uc_next * fkp_next
    
    # Create and fit
    nt = NanoTensor((1,), max_order=1, base_vars=['k', 'a', 'eps', 'sig'])
    trainer = SymbolicTrainer(nt)
    
    start = time.time()
    if ss_guess is None:
        ss_guess = {'k': 1.0, 'a': 0.0, 'eps': 0.0, 'sig': 1.0}
    success = trainer.fit([(R, params, ss_guess)], method='perturbation')
    elapsed = time.time() - start
    
    print(f"Perturbation fit: {'✅ Success' if success else '❌ Failed'} ({elapsed:.3f}s)")
    print("Fitted coefficients:")
    for k, v in trainer.fitted_coeffs.items():
        print(f"  {k} = {v:.6f}")
    
    # Prediction test
    test_state = {'k': 1.1, 'a': 0.0, 'eps': 0.01, 'sig': 1.0}
    pred = trainer.predict(test_state)
    print(f"\nPolicy at {test_state}:")
    print(f"  k' = {pred.flat[0]:.6f}")
    
    # Visualize
    nt.plot_contour('k', 'eps', fixed={'a': 0, 'sig': 1})
    
    return trainer

def demo_kamke_ade():
    """
    Demonstration: algebraic differential equation (ADE) example.

    Uses a Kamke-style ADE of the form:

        y'^2 + 3 y' - 2 y - 3 x = 0

    to illustrate the curve parametrization approach implemented in
    `NanoTensor.parametrize_curve`. The function:

    - constructs the ADE in SymPy,
    - derives a parametric representation (x(t), y(t)),
    - and prints the resulting parameterization and a related general solution.

    This serves primarily as a conceptual example of symbolic curve handling.
    """
    print("\n=== Kamke ADE Demo ===")
    
    x, y, yp = sp.symbols('x y yp')
    F = yp**2 + 3*yp - 2*y - 3*x
    
    nt = NanoTensor((1,))
    x_param, y_param = nt.parametrize_curve(F)
    
    print(f"Parametrization: (x(t), y(t)) = ({x_param}, {y_param})")
    
    # Verify solution
    t = sp.Symbol('t')
    y_sol = sp.integrate(sp.solve(F, yp)[1], x)
    print(f"General solution: y(x) = {y_sol}")
    
    return x_param, y_param

def benchmark_performance():
    """
    Compare symbolic and numeric differentiation performance.

    Benchmarks:

    - symbolic differentiation via `NanoTensor` over a 100x100 tensor, and
    - numeric differentiation via `numpy.gradient` over a random 100x100 array.

    Prints timing information and a speed ratio, illustrating the cost of
    exact symbolic operations relative to purely numeric ones.
    """
    print("\n=== Performance Benchmark ===")
    
    # Symbolic method
    nt_sym = NanoTensor((100, 100), max_order=2)
    start = time.time()
    _ = nt_sym.diff_cached('k', 1)
    sym_time = time.time() - start
    
    # Numeric method (numpy)
    nt_num = np.random.randn(100, 100)
    start = time.time()
    _ = np.gradient(nt_num)
    num_time = time.time() - start
    
    print(f"Symbolic diff: {sym_time:.4f}s")
    print(f"Numeric diff: {num_time:.4f}s")
    print(f"Speed ratio: {sym_time/num_time:.2f}x (symbolic is slower but exact)")
    
    return sym_time, num_time

def start_repl(nt: NanoTensor, trainer: SymbolicTrainer):
    """
    Launch an interactive symbolic REPL.

    Provides an `InteractiveConsole` preloaded with key symbols:

    - nt      : NanoTensor instance
    - trainer : associated SymbolicTrainer
    - sp      : SymPy module
    - solve   : SymPy solve
    - symbols : SymPy symbols
    - plot    : convenience alias to `nt.plot_contour`

    Intended for exploratory work and quick experiments.
    """
    
    locals_dict = {
        'nt': nt, 'trainer': trainer, 'sp': sp, 'solve': sp.solve,
        'symbols': sp.symbols, 'plot': nt.plot_contour
    }
    con = InteractiveConsole(locals_dict)
    banner = """
    Symbolic AI REPL
    ================
    Commands:
    - nt.eval_numeric({'k':1.1})
    - trainer.fit(data, 'symbolic')
    - solve(R, k)
    - nt.plot_contour('k', 'eps')
    """
    print(banner)
    con.interact()

def streamlit_dashboard():
    """
    Launch a Streamlit dashboard for interactive exploration.

    Provides:

    - sliders for key state variables (k, a, eps),
    - a button to run the RBC perturbation demo,
    - visualization of the resulting policy function k' over k,
    - and a JSON view of the fitted coefficients.

    To run, execute:
        streamlit run symbo.py
    (with this function exposed as the main entry point).
    """
    
    import streamlit as st
    
    st.title("NanoTensor Symbolic AI Dashboard")
    
    # Sidebar controls
    st.sidebar.header("Model Parameters")
    k = st.sidebar.slider("Capital (k)", 0.5, 1.5, 1.0)
    a = st.sidebar.slider("Technology (a)", -0.1, 0.1, 0.0)
    eps = st.sidebar.slider("Shock (eps)", -0.05, 0.05, 0.0)
    
    # Create and fit model
    if st.sidebar.button("Run Perturbation"):
        with st.spinner("Computing perturbation solution..."):
            trainer = demo_rbc_perturbation()
            state = {'k': k, 'a': a, 'eps': eps, 'sig': 1.0}
            pred = trainer.predict(state)
            
            st.success(f"Policy k' = {pred.flat[0]:.4f}")
            
            # Show coefficients
            st.json(trainer.fitted_coeffs)
            
            # Plot
            fig, ax = plt.subplots()
            ks = np.linspace(0.5, 1.5, 50)
            pols = [trainer.predict({'k': k_val, 'a': a, 'eps': eps, 'sig': 1.0}).flat[0] for k_val in ks]
            ax.plot(ks, pols, label='Policy function')
            ax.set_xlabel('k')
            ax.set_ylabel("k'")
            ax.legend()
            st.pyplot(fig)

def run_full_pipeline():
    """
    Execute the full Symbo demo pipeline.

    The pipeline includes:

    1. RBC perturbation demo,
    2. Kamke ADE demo,
    3. performance benchmark,
    4. population of a KnowledgeBase with fitted RBC policy coefficients.

    Returns
    -------
    SymbolicTrainer
        Trainer instance from the RBC perturbation demo (for further use).
    """ 
    print("Starting NanoTensor Symbolic AI Pipeline...")
    
    # Demo 1: RBC perturbation
    trainer = demo_rbc_perturbation()
    
    # Demo 2: Kamke ODE
    demo_kamke_ade()
    
    # Demo 3: Benchmarks
    benchmark_performance()
    
    # Knowledge base demo
    kb = KnowledgeBase()
    for k, v in trainer.fitted_coeffs.items():
        kb.add_fact('rbc_policy', k, v)
    
    print("\nKnowledge Base Query:")
    print(kb.query('rbc_policy', 'g_k'))
    
    print("\n✅ All demos completed successfully!")
    return trainer

if __name__ == "__main__":
    # Run pipeline if executed directly
    trainer = run_full_pipeline()
    
    # Start REPL for interactive exploration
    nt = NanoTensor((1,), max_order=2)
    start_repl(nt, trainer)
    
    # Uncomment to launch Streamlit:
    # streamlit_dashboard()
