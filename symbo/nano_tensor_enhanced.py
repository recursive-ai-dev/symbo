# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Military-Grade NanoTensor with Agency Capabilities
===================================================

This module provides an enhanced NanoTensor implementation with:

1. **Autonomous Decision-Making**: Self-optimization, error detection, and correction
2. **Memory System**: Experience replay, pattern recognition, and learning
3. **Health Monitoring**: Self-diagnostics, performance tracking, and alerts
4. **Security Layer**: Robust validation, bounds checking, and anomaly detection
5. **Agency Core**: Goal-directed reasoning and adaptive behavior

The enhanced NanoTensor acts as an intelligent computational brain that can be
embedded into agents to provide them with agency - the ability to perceive,
reason, learn, and act autonomously.

Relationship to :class:`symbo.nanotensor.NanoTensor`
----------------------------------------------------
``NanoTensor`` is the class the library uses and tests: it carries the same
agency, health, validation and caching behaviour, and it is what
``from symbo import NanoTensor`` gives you. This module is a **standalone,
dependency-light copy** of that design -- it deliberately does *not* subclass
``NanoTensor`` -- so a single file can be vendored into an agent runtime that
cannot install the package. It therefore exposes only ``health_check``/caching
features and lacks ``NanoTensor``'s learning, solving and plotting API. Prefer
``NanoTensor`` unless you specifically need a file you can copy out.
"""

import sympy as sp
import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Callable
from functools import wraps
from dataclasses import dataclass, field
from collections import deque
import time
import warnings
from enum import Enum


class HealthStatus(Enum):
    """Health status indicators for the NanoTensor."""
    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class OperationType(Enum):
    """Types of operations tracked for learning and optimization."""
    DIFFERENTIATION = "diff"
    SUBSTITUTION = "subs"
    EVALUATION = "eval"
    SIMPLIFICATION = "simplify"
    SOLVING = "solve"
    OPTIMIZATION = "optimize"


@dataclass
class Experience:
    """Record of a computational experience for learning."""
    operation: OperationType
    inputs: Dict[str, Any]
    outputs: Any
    duration: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring NanoTensor health."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_compute_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage_mb: float = 0.0
    last_health_check: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate operation success rate."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def avg_operation_time(self) -> float:
        """Calculate average operation time."""
        if self.total_operations == 0:
            return 0.0
        return self.total_compute_time / self.total_operations


class AgencyCore:
    """
    Core agency system providing goal-directed reasoning and decision-making.

    This system gives the NanoTensor the ability to:
    - Set and pursue computational goals
    - Make autonomous decisions about optimization strategies
    - Learn from experience and adapt behavior
    - Detect and respond to anomalies
    """

    def __init__(self, max_memory_size: int = 10000):
        self.goals: List[Dict[str, Any]] = []
        self.current_goal: Optional[Dict[str, Any]] = None
        self.experience_buffer = deque(maxlen=max_memory_size)
        self.learned_patterns: Dict[str, Any] = {}
        #: how far a duration may deviate from the learned average, in
        #: multiples of that average, before it counts as an anomaly
        self.anomaly_threshold: float = 3.0
        #: sub-millisecond timings are measurement noise, not anomalies
        self.min_anomaly_seconds: float = 1e-3
        self.adaptation_rate: float = 0.1

    def add_goal(self, goal_type: str, target: Any, priority: float = 1.0):
        """Add a computational goal to pursue."""
        goal = {
            "type": goal_type,
            "target": target,
            "priority": priority,
            "created_at": time.time(),
            "progress": 0.0
        }
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g["priority"], reverse=True)
        # ``get_status`` reports the goal under pursuit, so promote it here --
        # ``current_goal`` used to stay None forever.
        if self.current_goal is None or goal["priority"] > self.current_goal["priority"]:
            self.current_goal = goal

    def complete_goal(self, progress: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Close the highest-priority goal and promote the next one.

        Returns the finished goal, or ``None`` when there is nothing to do.
        """
        if not self.goals:
            self.current_goal = None
            return None
        finished = self.goals.pop(0)
        finished["progress"] = progress
        self.current_goal = self.goals[0] if self.goals else None
        return finished

    def record_experience(self, experience: Experience):
        """Record an experience for learning."""
        self.experience_buffer.append(experience)
        self._update_learned_patterns(experience)

    def _update_learned_patterns(self, experience: Experience):
        """Update learned patterns based on new experience."""
        op_type = experience.operation.value
        if op_type not in self.learned_patterns:
            self.learned_patterns[op_type] = {
                "count": 0,
                "avg_duration": 0.0,
                "success_rate": 1.0,
                "common_errors": {}
            }

        pattern = self.learned_patterns[op_type]
        pattern["count"] += 1

        # Update running average of duration
        alpha = self.adaptation_rate
        pattern["avg_duration"] = (1 - alpha) * pattern["avg_duration"] + alpha * experience.duration

        # Update success rate
        pattern["success_rate"] = (
            (pattern["success_rate"] * (pattern["count"] - 1) +
             (1.0 if experience.success else 0.0)) / pattern["count"]
        )

        # Track common errors
        if not experience.success and experience.error:
            error_key = str(experience.error)[:50]  # Truncate long errors
            pattern["common_errors"][error_key] = pattern["common_errors"].get(error_key, 0) + 1

    def detect_anomaly(self, operation: OperationType, duration: float) -> bool:
        """
        Whether *duration* is far from the learned average for *operation*.

        The comparison is a ratio of the learned average (``anomaly_threshold``),
        and both timings must be at least ``min_anomaly_seconds`` for it to mean
        anything: cache hits are legitimately orders of magnitude faster than a
        cold evaluation, and flagging that as an anomaly is pure noise.
        """
        op_type = operation.value
        if op_type not in self.learned_patterns:
            return False

        pattern = self.learned_patterns[op_type]
        if pattern["count"] < 10:  # Need enough samples
            return False

        avg = pattern["avg_duration"]
        if min(avg, duration) < self.min_anomaly_seconds:
            return False

        return abs(duration - avg) / avg > self.anomaly_threshold

    def recommend_optimization(self, operation: OperationType) -> Dict[str, Any]:
        """Recommend optimization strategy based on learned patterns."""
        op_type = operation.value
        if op_type not in self.learned_patterns:
            return {"strategy": "default", "confidence": 0.0}

        pattern = self.learned_patterns[op_type]

        recommendations = {
            "strategy": "default",
            "use_cache": True,
            "simplify_first": False,
            "parallel": False,
            "confidence": min(pattern["count"] / 100.0, 1.0)
        }

        # Adapt strategy based on learned patterns
        if pattern["avg_duration"] > 1.0:
            recommendations["simplify_first"] = True

        if pattern["success_rate"] < 0.9:
            recommendations["strategy"] = "conservative"

        return recommendations

    def get_status(self) -> Dict[str, Any]:
        """Get current agency status."""
        return {
            "active_goals": len(self.goals),
            "current_goal": self.current_goal,
            "total_experiences": len(self.experience_buffer),
            "learned_patterns": len(self.learned_patterns),
            "pattern_summary": {
                op: {
                    "count": pat["count"],
                    "avg_duration": pat["avg_duration"],
                    "success_rate": pat["success_rate"]
                }
                for op, pat in self.learned_patterns.items()
            }
        }


class MilitaryGradeNanoTensor:
    """
    Enhanced NanoTensor with military-grade robustness and agency capabilities.

    This class extends the base NanoTensor with:
    - Autonomous decision-making and self-optimization
    - Memory and learning from experience
    - Self-monitoring and health diagnostics
    - Robust error handling and recovery
    - Security validation and anomaly detection
    - Goal-directed reasoning capabilities

    It acts as a computational "brain" that can be embedded into agents,
    providing them with agency - the ability to perceive, reason, learn,
    and act autonomously in complex symbolic-numeric environments.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the tensor
    max_order : int
        Maximum Taylor expansion order
    base_vars : List[str]
        Base variable names
    name : str
        Tensor identifier
    enable_agency : bool
        Enable agency and learning features
    enable_security : bool
        Enable security validation
    max_cache_size : int
        Maximum cache size for memoization
    memory_size : int
        Size of experience replay buffer
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 max_order: int = 2,
                 base_vars: Optional[List[str]] = None,
                 name: str = "mgnt",
                 enable_agency: bool = True,
                 enable_security: bool = True,
                 max_cache_size: int = 1000,
                 memory_size: int = 10000):

        # Core tensor properties
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"shape must be a tuple of ints, got {shape!r}")
        shape = tuple(shape)
        if not shape or any(not isinstance(d, int) or isinstance(d, bool) or d <= 0
                            for d in shape):
            raise ValueError(
                f"shape must be a non-empty tuple of positive ints, got {shape!r}")
        if max_order < 1:
            raise ValueError(f"max_order must be >= 1, got {max_order}")
        self.shape = shape
        self.max_order = max_order
        self.name = name
        self.base_vars = [sp.Symbol(v) for v in (base_vars or ['k', 'a', 'eps', 'sig'])]
        self.coeff_vars: List[sp.Symbol] = []
        self.data: np.ndarray = np.empty(shape, dtype=object)

        # Enhanced features
        self.enable_agency = enable_agency
        self.enable_security = enable_security
        self.max_cache_size = max_cache_size

        # Agency and learning systems
        self.agency: Optional[AgencyCore] = AgencyCore(memory_size) if enable_agency else None
        self.metrics = PerformanceMetrics()
        self.health_status = HealthStatus.OPTIMAL

        # Enhanced caching
        self._diff_cache: Dict[Tuple[str, int], 'MilitaryGradeNanoTensor'] = {}
        self._subs_cache: Dict[str, 'MilitaryGradeNanoTensor'] = {}
        self._eval_cache: Dict[str, np.ndarray] = {}
        self._symvars_cache = None
        self._lambdify_cache: Dict = {}

        # Security and validation
        self._validation_enabled = enable_security
        self._bounds: Dict[str, Tuple[float, float]] = {}
        self._constraints: List[Callable] = []

        # Fitted coefficients
        self.fitted_coeffs: Dict[str, float] = {}

        # Initialize
        self._init_data()

    def _init_data(self):
        """Initialize tensor with symbolic zeros."""
        self.data = np.zeros(self.shape, dtype=object)
        self.data.flat[:] = sp.S(0)
        self._clear_caches()

    # The caches below hold compiled ``lambdify`` callables and objects that are
    # rebuilt on demand. Serializing them bloats saved brains and breaks the
    # stdlib ``pickle`` module (lambdify functions cannot be resolved by
    # attribute lookup), so they are dropped on save and recreated on demand.
    _TRANSIENT_STATE = ('_diff_cache', '_subs_cache', '_eval_cache',
                        '_symvars_cache', '_lambdify_cache')

    def __getstate__(self) -> Dict[str, Any]:
        transient = set(self._TRANSIENT_STATE)
        return {k: v for k, v in self.__dict__.items() if k not in transient}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        for name in self._TRANSIENT_STATE:
            self.__dict__[name] = None if name == '_symvars_cache' else {}

    @staticmethod
    def _cache_key(mapping: Dict[Any, Any]) -> tuple:
        """
        Build a hashable, collision-free cache key for a dict of scalars.

        A tuple of ``(name, value)`` pairs is exact for numeric inputs, so no
        digest function (and no risk of a hash collision silently returning the
        wrong cached tensor) is needed. Non-numeric values fall back to their
        string form.
        """
        try:
            return tuple(sorted((str(k), float(v)) for k, v in mapping.items()))
        except (TypeError, ValueError):
            return tuple(sorted((str(k), str(v)) for k, v in mapping.items()))

    def _clear_caches(self):
        """Clear all caches."""
        self._diff_cache.clear()
        self._subs_cache.clear()
        self._eval_cache.clear()
        self._symvars_cache = None
        self._lambdify_cache.clear()

    def _track_operation(self, operation: OperationType):
        """Decorator to track operations and collect metrics."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                error = None
                result = None

                try:
                    result = func(*args, **kwargs)
                    self.metrics.successful_operations += 1
                    return result

                except Exception as e:
                    success = False
                    error = str(e)
                    self.metrics.failed_operations += 1

                    # Attempt recovery if agency is enabled
                    if self.agency and self.enable_agency:
                        result = self._attempt_recovery(operation, e, args, kwargs)
                        if result is not None:
                            success = True
                            error = None
                            self.metrics.successful_operations += 1
                            self.metrics.failed_operations -= 1

                    if not success:
                        raise

                finally:
                    duration = time.time() - start_time
                    self.metrics.total_operations += 1
                    self.metrics.total_compute_time += duration

                    # Record experience for learning
                    if self.agency and self.enable_agency:
                        experience = Experience(
                            operation=operation,
                            inputs={"args": args, "kwargs": kwargs},
                            outputs=result,
                            duration=duration,
                            success=success,
                            error=error
                        )
                        self.agency.record_experience(experience)

                        # Check for anomalies
                        if self.agency.detect_anomaly(operation, duration):
                            warnings.warn(
                                f"Operation-time anomaly detected: {operation.value} "
                                f"took {duration:.3g}s",
                                stacklevel=2,
                            )

                    # Update health status
                    self._update_health()

            return wrapper
        return decorator

    def _attempt_recovery(self, operation: OperationType, error: Exception,
                         args: tuple, kwargs: dict) -> Any:
        """Attempt to recover from an error using learned strategies."""
        if not self.agency:
            return None

        # Get recommendation from agency
        rec = self.agency.recommend_optimization(operation)

        # Try simplified approach if recommended
        if rec.get("simplify_first") and operation == OperationType.SUBSTITUTION:
            try:
                # Simplify before substitution
                self.simplify()
                return None  # Indicate to retry
            except Exception:
                pass

        return None

    def _update_health(self):
        """Update health status based on metrics."""
        rate = self.metrics.success_rate

        if rate >= 0.99:
            self.health_status = HealthStatus.OPTIMAL
        elif rate >= 0.95:
            self.health_status = HealthStatus.GOOD
        elif rate >= 0.85:
            self.health_status = HealthStatus.DEGRADED
        elif rate >= 0.70:
            self.health_status = HealthStatus.CRITICAL
        else:
            self.health_status = HealthStatus.FAILED

    def _validate_input(self, var_name: str, value: float) -> bool:
        """Validate input against security constraints."""
        if not self._validation_enabled:
            return True

        # Check bounds
        if var_name in self._bounds:
            lower, upper = self._bounds[var_name]
            if not (lower <= value <= upper):
                raise ValueError(f"Value {value} for {var_name} outside bounds [{lower}, {upper}]")

        # Check for NaN/Inf
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"Invalid value for {var_name}: {value}")

        return True

    def set_bounds(self, var_name: str, lower: float, upper: float):
        """Set validation bounds for a variable."""
        self._bounds[var_name] = (lower, upper)

    def add_constraint(self, constraint: Callable[[Dict[str, float]], bool]):
        """Add a constraint function that must be satisfied."""
        self._constraints.append(constraint)

    @property
    def symvars(self) -> List[sp.Symbol]:
        """
        All free symbols currently stored in the tensor.

        Computed on demand: ``nt.data[i] = expr`` is the documented way to fill a
        tensor and it cannot notify the object, so a cached answer could go stale
        (which used to make ``eval_numeric`` bind the wrong variables).
        """
        vars_set: set = set()
        for elem in self.data.flat:
            if elem != 0 and hasattr(elem, 'free_symbols'):
                vars_set.update(elem.free_symbols)
        return sorted(vars_set, key=str)

    def _data_snapshot(self) -> tuple:
        """
        The expression objects currently in ``data``, for cache validation.

        Caches are only reusable while the entries are *the same objects*: SymPy
        expressions are immutable, so identity is both cheap and exact.
        """
        return tuple(self.data.flat)

    def diff(self, wrt: sp.Symbol, order: int = 1) -> 'MilitaryGradeNanoTensor':
        """
        Symbolic differentiation with tracking and optimization.

        Enhanced with:
        - Performance tracking
        - Anomaly detection
        - Learned optimization strategies
        """
        @self._track_operation(OperationType.DIFFERENTIATION)
        def _diff_impl():
            # Check cache first
            key = (wrt.name if hasattr(wrt, 'name') else str(wrt), order)
            if key in self._diff_cache:
                self.metrics.cache_hits += 1
                return self._diff_cache[key]

            self.metrics.cache_misses += 1

            # Get optimization recommendation if agency is enabled
            if self.agency and self.enable_agency:
                rec = self.agency.recommend_optimization(OperationType.DIFFERENTIATION)
                if rec.get("simplify_first"):
                    self.simplify()

            # Perform differentiation
            new_nt = MilitaryGradeNanoTensor(
                self.shape, self.max_order,
                [v.name for v in self.base_vars],
                name=f"d{self.name}/d{wrt}",
                enable_agency=self.enable_agency,
                enable_security=self.enable_security
            )
            new_nt.data = np.vectorize(lambda e: sp.diff(e, wrt, order))(self.data)

            # Cache result
            if len(self._diff_cache) < self.max_cache_size:
                self._diff_cache[key] = new_nt

            return new_nt

        return _diff_impl()

    def subs(self, sub_dict: Dict[sp.Symbol, Any]) -> 'MilitaryGradeNanoTensor':
        """
        Substitution with validation and tracking.

        Enhanced with:
        - Input validation
        - Security checks
        - Performance optimization
        """
        @self._track_operation(OperationType.SUBSTITUTION)
        def _subs_impl():
            # Validate inputs
            if self._validation_enabled:
                for k, v in sub_dict.items():
                    var_name = k.name if hasattr(k, 'name') else str(k)
                    if isinstance(v, (int, float)):
                        self._validate_input(var_name, float(v))

            # Create cache key
            cache_key = self._cache_key(sub_dict)

            # Check cache
            if cache_key in self._subs_cache:
                self.metrics.cache_hits += 1
                return self._subs_cache[cache_key]

            self.metrics.cache_misses += 1

            # Perform substitution
            new_nt = MilitaryGradeNanoTensor(
                self.shape, self.max_order,
                [v.name for v in self.base_vars],
                name=self.name,
                enable_agency=self.enable_agency,
                enable_security=self.enable_security
            )

            clean_dict = {
                k: (sp.nsimplify(v) if isinstance(v, (int, float)) else v)
                for k, v in sub_dict.items()
            }
            new_nt.data = np.vectorize(lambda e: e.subs(clean_dict))(self.data)

            # Cache result
            if len(self._subs_cache) < self.max_cache_size:
                self._subs_cache[cache_key] = new_nt

            return new_nt

        return _subs_impl()

    def eval_numeric(self, point: Dict[str, float]) -> np.ndarray:
        """
        Numeric evaluation with validation and optimization.

        Enhanced with:
        - Input validation
        - Constraint checking
        - Optimized compilation
        """
        # Points may be keyed by symbol or by name; everything below matches on
        # names, so normalise once instead of silently defaulting to 0.0.
        point = {(key if isinstance(key, str) else getattr(key, "name", str(key))): value
                 for key, value in point.items()}

        @self._track_operation(OperationType.EVALUATION)
        def _eval_impl():
            # Validate all inputs
            if self._validation_enabled:
                for var_name, value in point.items():
                    self._validate_input(var_name, value)

                # Check constraints
                for constraint in self._constraints:
                    if not constraint(point):
                        raise ValueError("Constraint violation detected")

            # Create cache key
            cache_key = self._cache_key(point)

            # Check cache (only valid while the stored expressions are unchanged)
            snapshot = self._data_snapshot()
            cached = self._eval_cache.get(cache_key)
            if cached is not None and all(a is b for a, b in
                                         zip(cached[0], snapshot, strict=True)):
                self.metrics.cache_hits += 1
                return cached[1]

            self.metrics.cache_misses += 1

            # Bind every symbol the point names. Restricting the binding to
            # ``self.symvars`` (as this used to) silently returns *symbolic*
            # results for hand-assigned entries such as ``nt.data[0] = x**2 + y``,
            # because those symbols are not part of a generated Taylor expansion.
            free = {symbol.name: symbol for symbol in self.symvars}
            declared = {v.name for v in self.base_vars}
            missing = sorted(set(free) - set(point) - declared)
            if missing:
                raise ValueError(
                    f"eval_numeric(): no value for {', '.join(missing)}; declared "
                    f"variables are {sorted(declared) if declared else '(none)'}"
                )

            # Every free symbol is bound; a declared variable the caller left out
            # defaults to 0.0, matching NanoTensor.eval_numeric.
            names = sorted(free)
            vars_in_point = [free[name] for name in names]
            args = [float(point.get(name, 0.0)) for name in names]

            # Compile if not cached for exactly these expressions
            key = tuple(v.name for v in vars_in_point)
            compiled = self._lambdify_cache.get(key)
            if (compiled is None
                    or not all(a is b for a, b in zip(compiled[0], snapshot, strict=True))):
                from sympy import lambdify
                compiled = (snapshot, [
                    lambdify(vars_in_point, e, modules='numpy') for e in snapshot
                ])
                self._lambdify_cache[key] = compiled

            # Evaluate: declared variables absent from the point default to 0.0,
            # matching NanoTensor.eval_numeric.
            funcs = compiled[1]
            result_flat = [f(*args) for f in funcs]
            result = np.asarray(result_flat, dtype=float).reshape(self.shape)

            # Cache result, tagged with the expressions it was computed from
            if len(self._eval_cache) >= self.max_cache_size:
                self._eval_cache.clear()
            self._eval_cache[cache_key] = (snapshot, result)

            return result

        return _eval_impl()

    def simplify(self) -> 'MilitaryGradeNanoTensor':
        """Simplify all expressions with tracking."""
        @self._track_operation(OperationType.SIMPLIFICATION)
        def _simplify_impl():
            self.data = np.vectorize(sp.simplify)(self.data)
            self._clear_caches()
            return self

        return _simplify_impl()

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the tensor."""
        self.metrics.last_health_check = time.time()

        return {
            "status": self.health_status.value,
            "metrics": {
                "total_operations": self.metrics.total_operations,
                "success_rate": self.metrics.success_rate,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "avg_operation_time": self.metrics.avg_operation_time,
                "memory_usage_mb": self.metrics.memory_usage_mb
            },
            "agency_status": self.agency.get_status() if self.agency else None,
            "cache_sizes": {
                "diff": len(self._diff_cache),
                "subs": len(self._subs_cache),
                "eval": len(self._eval_cache),
                "lambdify": len(self._lambdify_cache)
            },
            "security": {
                "validation_enabled": self._validation_enabled,
                "bounds_set": len(self._bounds),
                "constraints": len(self._constraints)
            }
        }

    def optimize(self):
        """Autonomous self-optimization based on learned patterns."""
        if not self.agency:
            return

        # Simplify if we have many failed operations
        if self.metrics.success_rate < 0.95:
            try:
                self.simplify()
            except Exception as exc:
                # Simplification is best-effort; log failure but do not interrupt optimization.
                warnings.warn(
                    f"optimize(): simplify() failed for tensor '{self.name}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Bound the caches by dropping the oldest entries. Dicts keep insertion
        # order, so this is a genuine (approximate) LRU eviction; the previous
        # implementation sorted by ``hash(key)``, which discarded arbitrary entries.
        for cache_name in ("_eval_cache", "_subs_cache", "_diff_cache"):
            cache = getattr(self, cache_name, None)
            if cache is None:
                continue
            limit = self.max_cache_size
            if len(cache) > limit * 0.9:
                keep = max(1, limit // 2)
                setattr(
                    self,
                    cache_name,
                    dict(list(cache.items())[-keep:]),
                )

    def __repr__(self) -> str:
        return (f"MilitaryGradeNanoTensor(name='{self.name}', shape={self.shape}, "
                f"health={self.health_status.value}, "
                f"success_rate={self.metrics.success_rate:.3f})")


__all__ = [
    'AgencyCore',
    'Experience',
    'HealthStatus',
    'MilitaryGradeNanoTensor',
    'OperationType',
    'PerformanceMetrics',
]
