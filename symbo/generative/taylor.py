# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Taylor Expansion Core
=====================

This module implements arbitrary-order multivariate Taylor-series expansions
around an arbitrary point. The output is a symbolic policy function that maintains
interpretability and is correctly structured for WASM serialization.

Key Features:
- Arbitrary order (supports 1st, 2nd, 3rd, ... order expansions)
- Multivariate (handles multiple variables)
- Arbitrary expansion point
- Symbolic policy function generation
- WASM-compatible serialization
"""

import sympy as sp
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from itertools import combinations_with_replacement, product
import json


class TaylorExpansion:
    """
    Generator for multivariate Taylor series expansions.
    
    This class constructs Taylor polynomial approximations of symbolic
    functions around a specified point, maintaining symbolic exactness
    and interpretability.
    
    Parameters
    ----------
    variables : List[sp.Symbol]
        Variables for the expansion
    center : Dict[sp.Symbol, float]
        Point around which to expand (e.g., steady state)
    max_order : int
        Maximum order of expansion
        
    Attributes
    ----------
    variables : List[sp.Symbol]
        Expansion variables
    center : Dict[sp.Symbol, float]
        Expansion point
    max_order : int
        Maximum order
    coefficients : Dict[Tuple, sp.Symbol]
        Symbolic coefficients for each term
    expansion : sp.Expr
        The Taylor polynomial
    """
    
    def __init__(self,
                 variables: List[sp.Symbol],
                 center: Dict[sp.Symbol, float],
                 max_order: int = 2):
        """Initialize Taylor expansion generator."""
        self.variables: List[sp.Symbol] = variables
        self.center: Dict[sp.Symbol, float] = center
        self.max_order: int = max_order
        
        self.coefficients: Dict[Tuple, sp.Symbol] = {}
        self.expansion: Optional[sp.Expr] = None
        self._coefficient_names: List[str] = []
    
    def generate(self, 
                 function_name: str = "f",
                 include_constant: bool = True) -> sp.Expr:
        """
        Generate multivariate Taylor expansion.
        
        Constructs a polynomial of the form:
        f(x) ≈ f(a) + Σᵢ fᵢ(xᵢ - aᵢ) + ½ΣᵢΣⱼ fᵢⱼ(xᵢ - aᵢ)(xⱼ - aⱼ) + ...
        
        where a is the center point and fᵢ, fᵢⱼ, etc. are symbolic coefficients.
        
        Parameters
        ----------
        function_name : str
            Base name for the function and coefficients
        include_constant : bool
            Whether to include constant term f(a)
            
        Returns
        -------
        sp.Expr
            Taylor polynomial as symbolic expression
            
        Examples
        --------
        >>> x, y = sp.symbols('x y')
        >>> taylor = TaylorExpansion([x, y], {x: 0, y: 0}, max_order=2)
        >>> poly = taylor.generate("g")
        # Returns: g_0 + g_x*x + g_y*y + g_xx*x²/2 + g_xy*x*y + g_yy*y²/2
        """
        self.coefficients.clear()
        self._coefficient_names.clear()
        
        # Deviations from center
        deviations = {var: var - self.center.get(var, 0) 
                     for var in self.variables}
        
        # Start with constant term (value at center)
        expansion_terms = []
        
        if include_constant:
            c0 = sp.Symbol(f"{function_name}_0")
            self.coefficients[()] = c0
            self._coefficient_names.append(c0.name)
            expansion_terms.append(c0)
        
        # Generate terms for each order
        for order in range(1, self.max_order + 1):
            order_terms = self._generate_order_terms(
                order, deviations, function_name
            )
            expansion_terms.extend(order_terms)
        
        self.expansion = sp.Add(*expansion_terms)
        return self.expansion
    
    def _generate_order_terms(self,
                             order: int,
                             deviations: Dict[sp.Symbol, sp.Expr],
                             function_name: str) -> List[sp.Expr]:
        """
        Generate all terms of a specific order.
        
        Parameters
        ----------
        order : int
            Order of terms to generate
        deviations : Dict[sp.Symbol, sp.Expr]
            Deviation expressions (x - a) for each variable
        function_name : str
            Base name for coefficients
            
        Returns
        -------
        List[sp.Expr]
            List of terms for this order
        """
        terms = []
        
        if order == 1:
            # First order: linear terms
            for var in self.variables:
                coeff_name = f"{function_name}_{var.name}"
                coeff = sp.Symbol(coeff_name)
                self.coefficients[(var,)] = coeff
                self._coefficient_names.append(coeff_name)
                terms.append(coeff * deviations[var])
        
        elif order == 2:
            # Second order: quadratic and cross terms
            # Diagonal terms: fᵢᵢ(xᵢ - aᵢ)²
            for var in self.variables:
                coeff_name = f"{function_name}_{var.name}_{var.name}"
                coeff = sp.Symbol(coeff_name)
                self.coefficients[(var, var)] = coeff
                self._coefficient_names.append(coeff_name)
                # Include 1/2 factor for second derivatives
                terms.append(sp.Rational(1, 2) * coeff * deviations[var]**2)
            
            # Cross terms: fᵢⱼ(xᵢ - aᵢ)(xⱼ - aⱼ)
            for i, var_i in enumerate(self.variables):
                for j, var_j in enumerate(self.variables):
                    if i < j:
                        coeff_name = f"{function_name}_{var_i.name}_{var_j.name}"
                        coeff = sp.Symbol(coeff_name)
                        self.coefficients[(var_i, var_j)] = coeff
                        self._coefficient_names.append(coeff_name)
                        terms.append(coeff * deviations[var_i] * deviations[var_j])
        
        else:
            # Higher orders: use combinations with replacement
            # For order n, we need all multisets of size n from variables
            for var_tuple in combinations_with_replacement(self.variables, order):
                # Build coefficient name from sorted variable names
                var_names = "_".join(v.name for v in var_tuple)
                coeff_name = f"{function_name}_{var_names}"
                coeff = sp.Symbol(coeff_name)
                self.coefficients[var_tuple] = coeff
                self._coefficient_names.append(coeff_name)
                
                # Build product of deviations
                dev_product = sp.Mul(*[deviations[v] for v in var_tuple])
                
                # Include factorial factor for derivatives
                # For a term with nᵢ occurrences of variable i, factor is 1/(n₁!n₂!...nₖ!)
                var_counts = {v: var_tuple.count(v) for v in set(var_tuple)}
                factorial_factor = sp.Mul(*[sp.factorial(n) for n in var_counts.values()])
                
                terms.append(coeff * dev_product / factorial_factor)
        
        return terms
    
    def evaluate_at_point(self, point: Dict[sp.Symbol, float]) -> float:
        """
        Evaluate expansion numerically at a point.
        
        Parameters
        ----------
        point : Dict[sp.Symbol, float]
            Variable values
            
        Returns
        -------
        float
            Numeric value
        """
        if self.expansion is None:
            raise ValueError("Must generate expansion first")
        return float(self.expansion.subs(point).evalf())
    
    def get_coefficient_vector(self) -> List[sp.Symbol]:
        """
        Get ordered list of coefficient symbols.
        
        Returns
        -------
        List[sp.Symbol]
            Coefficient symbols in generation order
        """
        return [sp.Symbol(name) for name in self._coefficient_names]
    
    def substitute_coefficients(self, 
                               coeff_values: Dict[str, float]) -> sp.Expr:
        """
        Substitute numeric values for coefficients.
        
        Parameters
        ----------
        coeff_values : Dict[str, float]
            Mapping from coefficient names to values
            
        Returns
        -------
        sp.Expr
            Expansion with substituted coefficients
        """
        if self.expansion is None:
            raise ValueError("Must generate expansion first")
        
        subs_dict = {sp.Symbol(k): v for k, v in coeff_values.items()}
        return self.expansion.subs(subs_dict)
    
    def to_policy_function(self,
                          coeff_values: Optional[Dict[str, float]] = None) -> 'PolicyFunction':
        """
        Convert to a policy function object for easy evaluation.
        
        Parameters
        ----------
        coeff_values : Dict[str, float], optional
            Coefficient values (if available)
            
        Returns
        -------
        PolicyFunction
            Callable policy function object
        """
        return PolicyFunction(
            expansion=self.expansion,
            variables=self.variables,
            center=self.center,
            coefficients=self.coefficients,
            coeff_values=coeff_values
        )
    
    def to_wasm_json(self) -> str:
        """
        Serialize to WASM-friendly JSON format.
        
        Returns
        -------
        str
            JSON string containing:
            - variables: list of variable names
            - center: expansion point
            - max_order: maximum order
            - coefficients: list of coefficient names
            - expansion: string representation of expansion
        """
        data = {
            "variables": [v.name for v in self.variables],
            "center": {v.name: float(val) for v, val in self.center.items()},
            "max_order": self.max_order,
            "coefficients": self._coefficient_names,
            "expansion": str(self.expansion) if self.expansion else None
        }
        return json.dumps(data)
    
    @classmethod
    def from_wasm_json(cls, json_str: str) -> 'TaylorExpansion':
        """
        Deserialize from WASM JSON format.
        
        Parameters
        ----------
        json_str : str
            JSON string
            
        Returns
        -------
        TaylorExpansion
            Reconstructed expansion object
        """
        data = json.loads(json_str)
        variables = [sp.Symbol(name) for name in data["variables"]]
        center = {sp.Symbol(k): v for k, v in data["center"].items()}
        
        expansion = cls(variables, center, data["max_order"])
        if data["expansion"]:
            expansion.expansion = sp.sympify(data["expansion"])
            expansion._coefficient_names = data["coefficients"]
        
        return expansion


class PolicyFunction:
    """
    Callable policy function from Taylor expansion.
    
    This class wraps a Taylor expansion into an easily callable and
    interpretable policy function suitable for use in dynamic models.
    
    Parameters
    ----------
    expansion : sp.Expr
        Taylor polynomial expression
    variables : List[sp.Symbol]
        State variables
    center : Dict[sp.Symbol, float]
        Expansion center (steady state)
    coefficients : Dict[Tuple, sp.Symbol]
        Coefficient symbols
    coeff_values : Dict[str, float], optional
        Fitted coefficient values
    """
    
    def __init__(self,
                 expansion: sp.Expr,
                 variables: List[sp.Symbol],
                 center: Dict[sp.Symbol, float],
                 coefficients: Dict[Tuple, sp.Symbol],
                 coeff_values: Optional[Dict[str, float]] = None):
        """Initialize policy function."""
        self.expansion = expansion
        self.variables = variables
        self.center = center
        self.coefficients = coefficients
        self.coeff_values = coeff_values or {}
        
        # Precompile for fast evaluation
        self._compiled = None
        if coeff_values:
            self._compile()
    
    def _compile(self):
        """Compile function for fast numeric evaluation."""
        # Substitute coefficient values
        subs_dict = {sp.Symbol(k): v for k, v in self.coeff_values.items()}
        expr_with_coeffs = self.expansion.subs(subs_dict)
        
        # Create lambdified function
        self._compiled = sp.lambdify(self.variables, expr_with_coeffs, modules='numpy')
    
    def __call__(self, **kwargs: float) -> float:
        """
        Evaluate policy function at a state point.
        
        Parameters
        ----------
        **kwargs : float
            Variable values (e.g., k=1.0, a=0.0)
            
        Returns
        -------
        float
            Policy value
            
        Examples
        --------
        >>> policy = taylor.to_policy_function(coeffs)
        >>> k_next = policy(k=1.1, a=0.05)
        """
        if self._compiled is None:
            # Fallback to symbolic evaluation
            point = {sp.Symbol(k): v for k, v in kwargs.items()}
            return float(self.expansion.subs(point).evalf())
        else:
            # Use compiled function
            args = [kwargs.get(v.name, 0.0) for v in self.variables]
            return float(self._compiled(*args))
    
    def update_coefficients(self, coeff_values: Dict[str, float]):
        """Update coefficient values and recompile."""
        self.coeff_values.update(coeff_values)
        self._compile()
    
    def get_partial_derivative(self, var: sp.Symbol, order: int = 1) -> sp.Expr:
        """
        Get partial derivative of policy function.
        
        Parameters
        ----------
        var : sp.Symbol
            Variable to differentiate with respect to
        order : int
            Order of derivative
            
        Returns
        -------
        sp.Expr
            Derivative expression
        """
        return sp.diff(self.expansion, var, order)
    
    def __repr__(self) -> str:
        return f"PolicyFunction(variables={[v.name for v in self.variables]}, " \
               f"center={self.center})"


def generate_multivariate_taylor(func: sp.Expr,
                                 variables: List[sp.Symbol],
                                 center: Dict[sp.Symbol, float],
                                 max_order: int = 2) -> sp.Expr:
    """
    Convenience function to generate Taylor expansion from a known function.
    
    This computes the actual Taylor series of a function f around point a
    by evaluating derivatives: f(x) ≈ Σ (∂ⁿf/∂xⁿ)|ₐ (x-a)ⁿ/n!
    
    Parameters
    ----------
    func : sp.Expr
        Function to expand
    variables : List[sp.Symbol]
        Variables to expand in
    center : Dict[sp.Symbol, float]
        Expansion point
    max_order : int
        Maximum order
        
    Returns
    -------
    sp.Expr
        Taylor series expansion
        
    Examples
    --------
    >>> x, y = sp.symbols('x y')
    >>> f = sp.exp(x) * sp.sin(y)
    >>> taylor = generate_multivariate_taylor(f, [x, y], {x: 0, y: 0}, 2)
    """
    # Start with function value at center
    subs_center = {var: center.get(var, 0) for var in variables}
    expansion = func.subs(subs_center)
    
    # Add terms for each order
    for order in range(1, max_order + 1):
        # Generate all multisets of variables of this order
        for var_tuple in combinations_with_replacement(variables, order):
            # Compute mixed partial derivative
            deriv = func
            for v in var_tuple:
                deriv = sp.diff(deriv, v)
            
            # Evaluate at center
            deriv_at_center = deriv.subs(subs_center)
            
            # Build deviation product
            dev_product = sp.Mul(*[var - center.get(var, 0) for var in var_tuple])
            
            # Compute factorial factor
            var_counts = {v: var_tuple.count(v) for v in set(var_tuple)}
            factorial_factor = sp.Mul(*[sp.factorial(n) for n in var_counts.values()])
            
            # Add term
            expansion += deriv_at_center * dev_product / factorial_factor
    
    return sp.simplify(expansion)


__all__ = [
    'TaylorExpansion',
    'PolicyFunction',
    'generate_multivariate_taylor',
]
