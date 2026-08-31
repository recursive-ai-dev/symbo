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

import json
import logging
from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

from symbo.security import safe_sympify

logger = logging.getLogger("symbo.generative.taylor")


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
        self.variables: List[sp.Symbol] = [
            v if isinstance(v, sp.Symbol) else sp.Symbol(v) for v in variables
        ]
        # Accept symbol- or name-keyed centers and normalize to symbols.
        self.center: Dict[sp.Symbol, float] = {
            (k if isinstance(k, sp.Symbol) else sp.Symbol(k)): v for k, v in center.items()
        }
        self.max_order: int = int(max_order)
        if self.max_order < 1:
            raise ValueError(f"max_order must be >= 1, got {max_order}")

        # A center entry for an unknown variable, or a non-numeric one, silently
        # produces an expansion around the wrong point -- refuse both up front.
        unknown = sorted(str(k) for k in self.center if k not in self.variables)
        if unknown:
            raise ValueError(
                f"center contains variables outside the expansion's variable list: "
                f"{unknown}; known variables: {[v.name for v in self.variables]}"
            )
        for key, value in self.center.items():
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"center[{key!r}] must be numeric, got {value!r}"
                ) from exc

        self.coefficients: Dict[Tuple, sp.Symbol] = {}
        self.expansion: Optional[sp.Expr] = None
        self._coefficient_names: List[str] = []
        #: Numeric coefficient values handed to :meth:`substitute_coefficients`.
        #: Kept so that :meth:`to_policy_function` and :meth:`to_wasm_json`
        #: describe the *fitted* expansion rather than the raw symbolic one.
        self.coeff_values: Dict[str, float] = {}

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
        >>> import sympy as sp
        >>> x, y = sp.symbols('x y')
        >>> taylor = TaylorExpansion([x, y], {x: 0, y: 0}, max_order=2)
        >>> poly = taylor.generate("g")
        >>> taylor.coefficient_names
        ['g_0', 'g_x', 'g_y', 'g_x_x', 'g_y_y', 'g_x_y']
        >>> all(sp.Symbol(n) in poly.free_symbols for n in taylor.coefficient_names)
        True
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

    def evaluate_at_point(self, point: Dict[sp.Symbol, float]) -> sp.Expr:
        """
        Evaluate the expansion at a point.

        Parameters
        ----------
        point : Dict[sp.Symbol | str, float]
        Returns
        -------
        sympy.Expr
            The (usually numeric) value of the expansion. Symbolic coefficients
            that were not substituted stay symbolic.
        """
        if self.expansion is None:
            raise ValueError("Must generate expansion first")
        subs = {sp.Symbol(k) if not isinstance(k, sp.Basic) else k: v
                for k, v in point.items()}
        return self.expansion.subs(subs)

    @property
    def coefficient_names(self) -> List[str]:
        """Names of the coefficient symbols introduced by :meth:`generate`."""
        return list(self._coefficient_names)

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
                               coeff_values: Dict[str, float],
                               apply: bool = False) -> sp.Expr:
        """
        Substitute numeric values for coefficients.

        Parameters
        ----------
        coeff_values : Dict[str, float]
            Mapping from coefficient names to values.
        apply : bool, optional
            When true the expansion itself is updated as well, so
            ``self.expansion`` becomes the numeric polynomial. By default the
            object keeps its symbolic form and only the substituted expression
            is returned (the values are still recorded in ``self.coeff_values``).

        Returns
        -------
        sp.Expr
            Expansion with substituted coefficients

        Raises
        ------
        ValueError
            If a name is not one of ``coefficient_names`` -- a typo would
            otherwise go unnoticed and leave a symbolic coefficient behind.
        """
        if self.expansion is None:
            raise ValueError("Must generate expansion first")

        unknown = sorted(set(coeff_values) - set(self._coefficient_names))
        if unknown:
            raise ValueError(
                f"unknown coefficient name(s) {unknown}; "
                f"this expansion defines {self._coefficient_names}"
            )

        subs_dict = {sp.Symbol(k): float(v) if isinstance(v, (int, float)) else v
                     for k, v in coeff_values.items()}
        self.coeff_values.update({k: (float(v) if isinstance(v, (int, float, sp.Number)) else v)
                                  for k, v in coeff_values.items()})
        expr = self.expansion.subs(subs_dict)
        if apply:
            self.expansion = expr
        return expr

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
        merged = dict(self.coeff_values)
        if coeff_values:
            merged.update(coeff_values)
        return PolicyFunction(
            expansion=self.expansion,
            variables=self.variables,
            center=self.center,
            coefficients=self.coefficients,
            coeff_values=merged or None
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
            # map each coefficient name to the multiset of variables it belongs
            # to, so the coefficient dict survives the JSON round trip
            "coefficient_map": {
                name: [v.name for v in key] for key, name in
                ((k, c.name) for k, c in self.coefficients.items())
            },
            "expansion": str(self.expansion) if self.expansion is not None else None,
            # numeric values of the fitted coefficients, so a browser can actually
            # evaluate the policy; empty while the expansion is still symbolic
            "coefficient_values": {name: float(val)
                                   for name, val in self.coeff_values.items()},
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
            expansion.expansion = safe_sympify(data["expansion"])
            expansion._coefficient_names = list(data["coefficients"])
            by_name = {v.name: v for v in variables}
            for name, var_names in (data.get("coefficient_map") or {}).items():
                key = tuple(by_name[n] for n in var_names if n in by_name)
                expansion.coefficients[key] = sp.Symbol(name)
            expansion.coeff_values = dict(data.get("coefficient_values") or {})

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

    @property
    def coefficient_names(self) -> List[str]:
        """Names of every coefficient symbol this expansion exposes."""
        return sorted({c.name for c in self.coefficients.values()})

    def _prepared_expression(self) -> sp.Expr:
        """
        Expansion with coefficients substituted, validated for numeric closure.

        Unknown coefficient names raise ``ValueError`` (the usual cause being a
        typo such as ``g_xx`` instead of the diagonal name ``g_x_x``), and
        defined-but-unspecified coefficients default to ``0.0``.
        """
        values = dict(self.coeff_values)

        if self.coefficients:
            known = set(self.coefficient_names)
            unknown = sorted(set(values) - known)
            if unknown:
                raise ValueError(
                    f"PolicyFunction: unknown coefficient name(s) {unknown}. "
                    f"This expansion defines {sorted(known)}"
                )
            defaulted = sorted(known - set(values))
            if defaulted:
                logger.debug(
                    "PolicyFunction: %d unspecified coefficient(s) default to 0.0: %s",
                    len(defaulted), defaulted,
                )
                values.update(dict.fromkeys(defaulted, 0.0))

        expr = self.expansion.xreplace({sp.Symbol(k): v for k, v in values.items()})
        leftover = sorted(str(sym) for sym in expr.free_symbols - set(self.variables))
        if leftover:
            raise ValueError(
                "PolicyFunction: expression still contains non-numeric symbol(s) "
                f"{leftover}; supply them as coefficients or as variables"
            )
        return expr

    def _compile(self):
        """Compile the policy into a fast numeric callable."""
        expr_with_coeffs = self._prepared_expression()
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
        float or numpy.ndarray
            Policy value; an array when any argument is array-valued (a batch).

        Examples
        --------
        >>> policy = taylor.to_policy_function(coeffs)  # doctest: +SKIP
        >>> k_next = policy(k=1.1, a=0.05)  # doctest: +SKIP
        """
        unknown = sorted(set(kwargs) - {v.name for v in self.variables})
        if unknown:
            raise ValueError(
                f"PolicyFunction: unknown variable(s) {unknown}; "
                f"this policy takes {[v.name for v in self.variables]}"
            )

        if self._compiled is None:
            # Fallback to symbolic evaluation (coefficients may still be symbols)
            point = {sp.Symbol(k): v for k, v in kwargs.items()}
            return self.expansion.subs(point)

        # Compiled path; numpy broadcasting handles batched input
        args = [np.asarray(kwargs.get(v.name, 0.0), dtype=float) for v in self.variables]
        arr = np.asarray(self._compiled(*args), dtype=float)
        return float(arr) if arr.ndim == 0 else arr

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
    'PolicyFunction',
    'TaylorExpansion',
    'generate_multivariate_taylor',
]
