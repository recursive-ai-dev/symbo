# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Gröbner Basis Solver with Streaming Output
===========================================

This module implements Gröbner basis computation with streaming output capability,
suitable for real-time applications. It gracefully handles edge cases including
polynomial systems with infinite solutions.

Key Features:
- Streaming output of intermediate solutions
- Real-time progress reporting
- Edge case handling (infinite solutions, inconsistent systems)
- Multiple monomial orderings
- Integration with SymPy's Gröbner basis algorithms
"""

import sympy as sp
from sympy import groebner, Symbol, Poly, solve
from typing import List, Dict, Generator, Optional, Tuple, Any, Union
import json
import time


class GröbnerBasisState:
    """
    State object for Gröbner basis computation.
    
    Tracks progress and intermediate results during computation.
    
    Attributes
    ----------
    polynomials : List[sp.Expr]
        Original polynomial system
    variables : List[sp.Symbol]
        Variables to solve for
    basis : Optional[sp.polys.polytools.GroebnerBasis]
        Computed Gröbner basis
    solutions : List[Dict[sp.Symbol, sp.Expr]]
        Found solutions
    order : str
        Monomial ordering used
    status : str
        Computation status
    """
    
    def __init__(self,
                 polynomials: List[sp.Expr],
                 variables: List[sp.Symbol],
                 order: str = 'lex'):
        """Initialize Gröbner basis computation state."""
        self.polynomials = polynomials
        self.variables = variables
        self.order = order
        
        self.basis: Optional[Any] = None
        self.solutions: List[Dict[sp.Symbol, sp.Expr]] = []
        self.status: str = "initialized"
        self.error: Optional[str] = None
        
        # Streaming state
        self._basis_polynomials: List[sp.Expr] = []
        self._current_index: int = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "polynomials": [str(p) for p in self.polynomials],
            "variables": [str(v) for v in self.variables],
            "order": self.order,
            "status": self.status,
            "error": self.error,
            "num_basis_polys": len(self._basis_polynomials),
            "num_solutions": len(self.solutions)
        }
    
    def to_json(self) -> str:
        """Serialize state to JSON."""
        return json.dumps(self.to_dict())


class StreamingGröbnerSolver:
    """
    Gröbner basis solver with streaming output.
    
    This class computes Gröbner bases and yields intermediate results
    as they become available, making it suitable for real-time applications
    and interactive use.
    
    Parameters
    ----------
    polynomials : List[sp.Expr]
        Polynomial equations (assumed equal to zero)
    variables : List[sp.Symbol]
        Variables to solve for
    order : str, optional
        Monomial ordering ('lex', 'grlex', 'grevlex')
    chunk_size : int, optional
        Number of basis polynomials to yield per chunk
        
    Examples
    --------
    >>> x, y = sp.symbols('x y')
    >>> polys = [x**2 + y**2 - 1, x - y]
    >>> solver = StreamingGröbnerSolver(polys, [x, y])
    >>> for chunk in solver.stream_basis():
    ...     print(chunk)
    """
    
    def __init__(self,
                 polynomials: List[sp.Expr],
                 variables: List[sp.Symbol],
                 order: str = 'lex',
                 chunk_size: int = 1):
        """Initialize streaming Gröbner solver."""
        self.state = GröbnerBasisState(polynomials, variables, order)
        self.chunk_size = chunk_size
        
    def compute_basis(self) -> None:
        """
        Compute Gröbner basis (non-streaming).
        
        This performs the full computation and stores results in state.
        """
        try:
            self.state.status = "computing"
            
            # Compute Gröbner basis
            G = groebner(
                self.state.polynomials,
                *self.state.variables,
                order=self.state.order
            )
            
            self.state.basis = G
            self.state._basis_polynomials = list(G.polys)
            self.state.status = "completed"
            
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
            raise
    
    def stream_basis(self) -> Generator[Dict[str, Any], None, None]:
        """
        Stream Gröbner basis computation results.
        
        Yields chunks of basis polynomials as they are processed.
        
        Yields
        ------
        Dict[str, Any]
            Dictionary containing:
            - type: "basis_chunk" or "completion"
            - items: list of polynomial dictionaries
            - progress: current progress info
            
        Examples
        --------
        >>> for chunk in solver.stream_basis():
        ...     if chunk["type"] == "basis_chunk":
        ...         for item in chunk["items"]:
        ...             print(f"Basis poly: {item['poly_str']}")
        """
        # First compute the basis
        self.compute_basis()
        
        if self.state.status == "error":
            yield {
                "type": "error",
                "error": self.state.error
            }
            return
        
        # Stream the basis polynomials in chunks
        total = len(self.state._basis_polynomials)
        
        for start_idx in range(0, total, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total)
            chunk_polys = self.state._basis_polynomials[start_idx:end_idx]
            
            items = []
            for idx, poly in enumerate(chunk_polys, start=start_idx):
                items.append({
                    "index": idx,
                    "poly_str": str(poly),
                    "degree": poly.as_poly(*self.state.variables).total_degree() 
                           if hasattr(poly, 'as_poly') else None,
                    "variables": [str(v) for v in self.state.variables]
                })
            
            yield {
                "type": "basis_chunk",
                "items": items,
                "progress": {
                    "current": end_idx,
                    "total": total,
                    "percentage": (end_idx / total * 100) if total > 0 else 100
                }
            }
        
        # Signal completion
        yield {
            "type": "completion",
            "total_basis_polynomials": total,
            "status": self.state.status
        }
    
    def solve_system(self) -> List[Dict[sp.Symbol, sp.Expr]]:
        """
        Solve polynomial system using the Gröbner basis.
        
        Returns
        -------
        List[Dict[sp.Symbol, sp.Expr]]
            List of solutions (may be empty for inconsistent systems)
            
        Notes
        -----
        - Returns empty list if system is inconsistent (contains 1 in basis)
        - Returns solutions as symbolic expressions
        - Handles infinite solution cases by returning parametric solutions
        """
        if self.state.basis is None:
            self.compute_basis()
        
        try:
            # Check if system is inconsistent (1 in basis)
            if any(p == 1 for p in self.state._basis_polynomials):
                self.state.solutions = []
                return []
            
            # Attempt to solve
            solutions = solve(
                self.state.polynomials,
                self.state.variables,
                dict=True
            )
            
            self.state.solutions = solutions
            return solutions
            
        except Exception as e:
            # Handle cases with infinite solutions or other issues
            self.state.error = f"Solution computation failed: {str(e)}"
            return []
    
    def stream_solutions(self) -> Generator[Dict[str, Any], None, None]:
        """
        Stream solutions as they are found.
        
        Yields
        ------
        Dict[str, Any]
            Dictionary containing:
            - type: "solution" or "completion" or "infinite_solutions"
            - data: solution information
        """
        # Compute solutions
        solutions = self.solve_system()
        
        if not solutions:
            # Check for special cases
            if self.state.error:
                yield {
                    "type": "error",
                    "error": self.state.error
                }
            elif any(p == 1 for p in self.state._basis_polynomials):
                yield {
                    "type": "inconsistent",
                    "message": "System is inconsistent (no solutions)"
                }
            else:
                # Might have infinite solutions
                yield {
                    "type": "infinite_solutions",
                    "message": "System may have infinite solutions",
                    "basis": [str(p) for p in self.state._basis_polynomials]
                }
            return
        
        # Stream individual solutions
        for idx, sol in enumerate(solutions):
            yield {
                "type": "solution",
                "index": idx,
                "solution": {str(k): str(v) for k, v in sol.items()},
                "is_real": self._check_real_solution(sol)
            }
        
        # Completion
        yield {
            "type": "completion",
            "total_solutions": len(solutions)
        }
    
    def _check_real_solution(self, sol: Dict[sp.Symbol, sp.Expr]) -> bool:
        """Check if a solution contains only real values."""
        try:
            for val in sol.values():
                # Check if value has negligible imaginary part
                if hasattr(val, 'as_real_imag'):
                    real, imag = val.as_real_imag()
                    if abs(imag) > 1e-10:
                        return False
            return True
        except:
            return False
    
    def get_state(self) -> GröbnerBasisState:
        """Get current computation state."""
        return self.state


class RealTimeGröbnerSolver(StreamingGröbnerSolver):
    """
    Real-time Gröbner solver with progress callbacks.
    
    Extends StreamingGröbnerSolver with callback support for
    progress monitoring in interactive applications.
    
    Parameters
    ----------
    polynomials : List[sp.Expr]
        Polynomial system
    variables : List[sp.Symbol]
        Variables
    order : str
        Monomial ordering
    chunk_size : int
        Chunk size for streaming
    progress_callback : callable, optional
        Function called with progress updates
    """
    
    def __init__(self,
                 polynomials: List[sp.Expr],
                 variables: List[sp.Symbol],
                 order: str = 'lex',
                 chunk_size: int = 1,
                 progress_callback: Optional[callable] = None):
        """Initialize real-time solver."""
        super().__init__(polynomials, variables, order, chunk_size)
        self.progress_callback = progress_callback
    
    def stream_basis(self) -> Generator[Dict[str, Any], None, None]:
        """Stream basis with progress callbacks."""
        for chunk in super().stream_basis():
            # Call progress callback if provided
            if self.progress_callback and "progress" in chunk:
                self.progress_callback(chunk["progress"])
            
            yield chunk


def solve_with_groebner(polynomials: List[sp.Expr],
                       variables: List[sp.Symbol],
                       order: str = 'lex',
                       stream: bool = False) -> Union[List[Dict], Generator]:
    """
    Convenience function to solve polynomial system using Gröbner bases.
    
    Parameters
    ----------
    polynomials : List[sp.Expr]
        Polynomial equations (assumed = 0)
    variables : List[sp.Symbol]
        Variables to solve for
    order : str
        Monomial ordering
    stream : bool
        If True, return generator for streaming; else return list
        
    Returns
    -------
    Union[List[Dict], Generator]
        Solutions (as list if stream=False, generator if stream=True)
        
    Examples
    --------
    >>> x, y = sp.symbols('x y')
    >>> polys = [x**2 - 1, x + y]
    >>> sols = solve_with_groebner(polys, [x, y])
    >>> print(sols)
    """
    solver = StreamingGröbnerSolver(polynomials, variables, order)
    
    if stream:
        return solver.stream_solutions()
    else:
        return solver.solve_system()


def handle_infinite_solutions(basis_polynomials: List[sp.Expr],
                              variables: List[sp.Symbol]) -> Dict[str, Any]:
    """
    Analyze and describe systems with infinite solutions.
    
    Parameters
    ----------
    basis_polynomials : List[sp.Expr]
        Gröbner basis polynomials
    variables : List[sp.Symbol]
        System variables
        
    Returns
    -------
    Dict[str, Any]
        Dictionary describing the solution space:
        - dimension: dimension of solution manifold
        - free_variables: variables that can be chosen freely
        - parametric_description: description of solution space
    """
    # Find leading variables (variables that appear as leading terms)
    leading_vars = set()
    for poly in basis_polynomials:
        if poly != 0:
            try:
                p = Poly(poly, *variables)
                # Get leading monomial
                lm = p.LM()
                # Find first non-zero exponent
                for i, exp in enumerate(lm):
                    if exp > 0:
                        leading_vars.add(variables[i])
                        break
            except:
                pass
    
    # Free variables are those not leading
    free_vars = [v for v in variables if v not in leading_vars]
    
    return {
        "has_infinite_solutions": len(free_vars) > 0,
        "dimension": len(free_vars),
        "free_variables": [str(v) for v in free_vars],
        "dependent_variables": [str(v) for v in leading_vars],
        "basis": [str(p) for p in basis_polynomials],
        "parametric_description": f"Solution space is {len(free_vars)}-dimensional"
    }


__all__ = [
    'GröbnerBasisState',
    'StreamingGröbnerSolver',
    'RealTimeGröbnerSolver',
    'solve_with_groebner',
    'handle_infinite_solutions',
]
