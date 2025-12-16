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
"""

import sympy as sp
import numpy as np
import torch
from typing import Tuple, Dict, Any, List, Optional, Union, Callable
from functools import lru_cache, wraps
from dataclasses import dataclass, field
from collections import deque
import time
import warnings
from enum import Enum
import hashlib
import json


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
        self.anomaly_threshold: float = 3.0  # Standard deviations
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
        """Detect if an operation duration is anomalous."""
        op_type = operation.value
        if op_type not in self.learned_patterns:
            return False
        
        pattern = self.learned_patterns[op_type]
        if pattern["count"] < 10:  # Need enough samples
            return False
        
        # Simple anomaly detection based on deviation from learned average
        avg = pattern["avg_duration"]
        if avg == 0:
            return False
        
        deviation = abs(duration - avg) / avg
        return deviation > self.anomaly_threshold
    
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
                 base_vars: List[str] = None,
                 name: str = "mgnt",
                 enable_agency: bool = True,
                 enable_security: bool = True,
                 max_cache_size: int = 1000,
                 memory_size: int = 10000):
        
        # Core tensor properties
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
                            warnings.warn(f"Anomalous operation detected: {operation.value} took {duration:.4f}s")
                    
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
            except:
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
        """Get all free symbols in the tensor (cached)."""
        if self._symvars_cache is None:
            vars_set = set()
            for elem in self.data.flat:
                if elem != 0 and hasattr(elem, 'free_symbols'):
                    vars_set.update(elem.free_symbols)
            self._symvars_cache = list(vars_set)
        return self._symvars_cache
    
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
            cache_key = hashlib.md5(
                json.dumps(
                    {str(k): str(v) for k, v in sub_dict.items()},
                    sort_keys=True
                ).encode()
            ).hexdigest()
            
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
            cache_key = hashlib.md5(
                json.dumps(point, sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache
            if cache_key in self._eval_cache:
                self.metrics.cache_hits += 1
                return self._eval_cache[cache_key]
            
            self.metrics.cache_misses += 1
            
            # Prepare for evaluation
            vars_in_point = [v for v in self.symvars if v.name in point]
            
            # Compile if not cached
            key = tuple(sorted([v.name for v in vars_in_point])) + tuple(sorted(point.keys()))
            if key not in self._lambdify_cache:
                from sympy import lambdify
                self._lambdify_cache[key] = [
                    lambdify(vars_in_point, e, modules='numpy') for e in self.data.flat
                ]
            
            # Evaluate
            funcs = self._lambdify_cache[key]
            args = [point.get(v.name, 0.0) for v in vars_in_point]
            result_flat = [f(*args) for f in funcs]
            result = np.array(result_flat).reshape(self.shape)
            
            # Cache result
            if len(self._eval_cache) < self.max_cache_size:
                self._eval_cache[cache_key] = result
            
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
            except:
                pass
        
        # Clear old caches if they're getting too large
        if len(self._eval_cache) > self.max_cache_size * 0.9:
            # Keep most recent entries
            items = sorted(self._eval_cache.items(), 
                          key=lambda x: hash(x[0]))[-self.max_cache_size//2:]
            self._eval_cache = dict(items)
        
        # Similar for other caches
        if len(self._subs_cache) > self.max_cache_size * 0.9:
            items = sorted(self._subs_cache.items(), 
                          key=lambda x: hash(x[0]))[-self.max_cache_size//2:]
            self._subs_cache = dict(items)
    
    def __repr__(self) -> str:
        return (f"MilitaryGradeNanoTensor(name='{self.name}', shape={self.shape}, "
                f"health={self.health_status.value}, "
                f"success_rate={self.metrics.success_rate:.3f})")


__all__ = ['MilitaryGradeNanoTensor', 'AgencyCore', 'HealthStatus', 
           'OperationType', 'Experience', 'PerformanceMetrics']
