# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Second-Order Perturbation Analysis
===================================

This module implements second-order perturbation analysis for discrete-time
rational-expectations models written as residual equations ``F(...) = 0``.

The method is *undetermined coefficients* (also called the method of undetermined
coefficients / Taylor projection, in the spirit of Schmitt-Grohé & Uribe and of
Uhlig's log-linear toolbox):

1. Compute the deterministic steady state, identifying each lead variable
   ``v_next`` with its level.
2. Conjecture a policy function for every endogenous variable, as a polynomial in
   the deviations of the *predetermined* drivers (state variables and shocks),
   with unknown coefficients ``g_{v,w}`` (and ``g_{v,w1,w2}`` at second order).
3. Substitute the conjecture into every equation, expand in a formal perturbation
   parameter ``eps``, and require each monomial coefficient to vanish. Expectational
   consistency (``c_{t+1} = Policy_c(k_{t+1}, a_{t+1})``) makes the first-order
   conditions *quadratic* in the ``g``'s — the same quadratic matrix equation that
   Blanchard–Kahn / Klein / QZ solvers attack. This module solves that system with
   damped Newton and keeps the Blanchard–Kahn-stable root (state-transition
   eigenvalues inside the unit circle). With the first-order coefficients known,
   the second-order conditions are linear in the quadratic ``g``'s and are solved
   with exact linear algebra.
4. Report the residual of the solved system against the *raw* user equations
   (never through the same substitution used to form the conditions), so the
   approximation order is independently verifiable.

Conventions
-----------
* The lead (expected next-period) value of ``v`` is written ``v_next`` -- the
  ``next_suffix`` constructor argument changes the convention.
* Shock variables are assumed to follow ``s_next = rho_s * s`` with ``rho_s`` given
  by ``shock_persistence`` (default ``0``: i.i.d. shocks) and have unconditional
  mean ``0``.
* The system should have as many equations as endogenous variables. Under-determined
  systems are still solved -- a particular solution is returned and the free
  parameters are reported in ``PerturbationSolution.diagnostics``.

Key Features:
- First-order perturbation (linear approximation)
- Second-order perturbation (quadratic corrections)
- Steady-state computation (symbolic, with numeric fallback)
- Variance / risk corrections derived from the quadratic shock terms
- Policy function generation, with a residual-based accuracy check
"""

from __future__ import annotations

import logging
from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from sympy import Symbol, diff, nsolve, solve

logger = logging.getLogger("symbo.analytics.perturbation")


class PerturbationSolution:
    """
    Container for perturbation solution results.

    Stores the steady state, the policy-function coefficients at each order, the
    resulting symbolic policy functions, risk corrections and accuracy diagnostics.

    Attributes
    ----------
    steady_state : Dict[Symbol, float]
        Steady-state values of every variable (shocks at their unconditional mean).
    first_order : Dict[str, float]
        Linear policy coefficients, keyed ``g_{variable}_{driver}``.
    second_order : Dict[str, float]
        Quadratic policy coefficients, keyed ``g_{variable}_{driver1}_{driver2}``.
    policy_functions : Dict[str, sp.Expr]
        Symbolic policy function for each endogenous variable, in levels.
    risk_corrections : Dict[str, float]
        Variance-induced constant term of each policy, ``h_{v}_sigma_sigma``.
    coefficients : Dict[str, float]
        Read-only view merging ``first_order`` and ``second_order``.
    residuals : Dict[str, float]
        Accuracy diagnostics: magnitude of the substituted equations.
    diagnostics : Dict[str, Any]
        Counts (equations, unknowns, free parameters) describing the solve.
    converged : bool
        ``True`` when both requested orders were solved.
    iterations : int
        Number of solver stages executed (kept for compatibility).
    """

    def __init__(self):
        """Initialize empty perturbation solution."""
        self.steady_state: Dict[Symbol, float] = {}
        self.first_order: Dict[str, float] = {}
        self.second_order: Dict[str, float] = {}
        self.policy_functions: Dict[str, sp.Expr] = {}

        # Additional metadata
        self.residuals: Dict[str, float] = {}
        self.diagnostics: Dict[str, Any] = {}
        self.converged: bool = False
        self.iterations: int = 0
        self.risk_corrections: Dict[str, float] = {}

    @property
    def coefficients(self) -> Dict[str, float]:
        """First- and second-order coefficients as one read-only mapping."""
        merged = dict(self.first_order)
        merged.update(self.second_order)
        return merged

    def policy_expression(self, var_name: str) -> sp.Expr:
        """Return the symbolic policy function of ``var_name``."""
        if var_name not in self.policy_functions:
            raise KeyError(
                f"no policy function for '{var_name}'; available: "
                f"{sorted(self.policy_functions)}"
            )
        return self.policy_functions[var_name]

    def evaluate_policy(self, var_name: str, values: Dict[str, float]) -> float:
        """
        Evaluate a policy function at a point.

        Parameters
        ----------
        var_name:
            Endogenous variable whose policy is used.
        values:
            Mapping from *driver* variable name to its current value.
        """
        expr = self.policy_expression(var_name)
        subs = {sp.Symbol(k): float(v) for k, v in values.items()}
        free = expr.free_symbols - set(subs)
        if free:
            raise ValueError(
                f"policy for '{var_name}' also needs values for "
                f"{sorted(str(s) for s in free)}"
            )
        return float(expr.subs(subs).evalf())

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"PerturbationSolution(converged={self.converged}, "
            f"order1={len(self.first_order)}, order2={len(self.second_order)}, "
            f"policies={sorted(self.policy_functions)})"
        )


class SecondOrderPerturbation:
    """
    Perturbation analyzer for discrete-time dynamic systems.

    Parameters
    ----------
    equations : List[sp.Expr]
        Residual equations, each implicitly equal to zero. Leads are written as
        ``<variable>_next``.
    state_vars : List[sp.Symbol]
        Predetermined (state) variables.
    control_vars : List[sp.Symbol]
        Jump (control) variables.
    shock_vars : List[sp.Symbol]
        Exogenous shock variables, assumed mean-zero AR(1).
    parameters : Dict[Symbol, float]
        Model parameters substituted before perturbation.
    next_suffix : str
        Suffix naming the lead of a variable (default ``"_next"``).
    shock_persistence : Dict[Symbol, float], optional
        AR(1) coefficient per shock. Missing shocks default to 0 (i.i.d.).

    Examples
    --------
    >>> import sympy as sp
    >>> from symbo.analytics.perturbation import SecondOrderPerturbation
    >>> k, c, a = sp.symbols('k c a')
    >>> kp, cp, ap = sp.symbols('k_next c_next a_next')
    >>> alpha, beta, delta, rho = sp.symbols('alpha beta delta rho')
    >>> params = {alpha: 0.36, beta: 0.99, delta: 0.08, rho: 0.95}
    >>> tech = a
    >>> euler = c**(-1) - beta * cp**(-1) * (alpha * ap * kp**(alpha - 1) + 1 - delta)
    >>> motion = sp.Eq(kp, alpha * tech * k**alpha + (1 - delta) * k - c)
    >>> pert = SecondOrderPerturbation([euler, motion.lhs - motion.rhs],
    ...                                state_vars=[k], control_vars=[c],
    ...                                shock_vars=[a], parameters=params)
    >>> sol = pert.solve(order=1)
    >>> sorted(sol.first_order)  # doctest: +SKIP
    ['g_c_a', 'g_c_k', 'g_k_a', 'g_k_k']
    """

    def __init__(self,
                 equations: List[sp.Expr],
                 state_vars: Sequence[Symbol],
                 control_vars: Sequence[Symbol],
                 shock_vars: Sequence[Symbol],
                 parameters: Dict[Symbol, float],
                 next_suffix: str = "_next",
                 shock_persistence: Optional[Dict[Symbol, float]] = None):
        """Initialize perturbation analyzer."""
        if not equations:
            raise ValueError("at least one equation is required")
        self.equations = [sp.sympify(eq) for eq in equations]
        self.state_vars = list(state_vars)
        self.control_vars = list(control_vars)
        self.shock_vars = list(shock_vars)
        # Parameters may be keyed by name or by Symbol; normalising to Symbols
        # makes every later lookup (and `.subs`) exact.
        self.parameters: Dict[Symbol, float] = {
            (sp.Symbol(k) if not isinstance(k, sp.Symbol) else k): v
            for k, v in dict(parameters).items()
        }
        self.next_suffix = next_suffix

        self.shock_persistence: Dict[Symbol, float] = dict.fromkeys(self.shock_vars, 0.0)
        for key, value in (shock_persistence or {}).items():
            sym = sp.Symbol(key) if not isinstance(key, sp.Symbol) else key
            if sym not in self.shock_persistence:
                raise ValueError(
                    f"shock_persistence references '{sym}', which is not a shock variable"
                )
            self.shock_persistence[sym] = self._numeric(value, f"shock_persistence[{sym}]")

        # All variables appearing in the model, and the two useful partitions.
        self.all_vars: List[Symbol] = self.state_vars + self.control_vars + self.shock_vars
        self.endogenous: List[Symbol] = self.state_vars + self.control_vars
        # Deviations of these variables are the polynomial basis of the policies.
        self.drivers: List[Symbol] = self.state_vars + self.shock_vars

        if not self.endogenous:
            raise ValueError("the model needs at least one endogenous variable")
        if not self.drivers:
            raise ValueError(
                "the model needs at least one state or shock variable to expand around"
            )

        # Solution storage
        self.solution: Optional[PerturbationSolution] = None

    def _normalize_variance(self, variance: Any) -> Dict[Symbol, float]:
        """
        Turn the ``variance`` argument into a per-shock mapping.

        Accepted forms: a scalar (the same variance for every shock -- the usual
        single-shock case), a mapping keyed by shock symbol or name, or ``None``
        for unit variance. Silently mis-reading the argument is worse than
        refusing it, so unknown shocks and non-numeric values raise.
        """
        if not self.shock_vars:
            return {}
        if variance is None:
            return dict.fromkeys(self.shock_vars, 1.0)
        if isinstance(variance, dict):
            out: Dict[Symbol, float] = {}
            for key, value in variance.items():
                sym = sp.Symbol(key) if not isinstance(key, sp.Symbol) else key
                if sym not in self.shock_persistence:
                    raise ValueError(
                        f"variance references '{sym}', which is not a shock variable"
                    )
                out[sym] = self._numeric(value, f"variance[{sym}]")
            for shock in self.shock_vars:
                out.setdefault(shock, 1.0)
            return out
        return {shock: self._numeric(variance, "variance") for shock in self.shock_vars}

    def _numeric(self, value: Any, what: str) -> float:
        """
        Resolve a constant to a float, following names through ``parameters``.

        Writing ``shock_persistence={a: rho}`` is the natural thing to do when
        ``rho`` is already a model parameter, so a symbolic value is looked up
        there before it has to be a plain number.
        """
        if isinstance(value, str):
            value = sp.Symbol(value)
        if isinstance(value, sp.Symbol) and value in self.parameters:
            value = self.parameters[value]
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{what} must be a number or a symbol listed in `parameters`; "
                f"got {value!r}"
            ) from exc

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------
    def _lead(self, var: Symbol) -> Symbol:
        """The symbol naming the expected next-period value of ``var``."""
        return sp.Symbol(f"{var.name}{self.next_suffix}")

    @staticmethod
    def _first_name(var: Symbol, driver: Symbol) -> str:
        return f"g_{var.name}_{driver.name}"

    @staticmethod
    def _second_name(var: Symbol, d1: Symbol, d2: Symbol) -> str:
        return f"g_{var.name}_{d1.name}_{d2.name}"

    def _deviation_symbol(self, var: Symbol) -> Symbol:
        return sp.Symbol(f"Delta_{var.name}")

    def _model_expressions(self) -> List[sp.Expr]:
        """Equations as plain expressions (``Eq(a, b)`` -> ``a - b``), parameters in."""
        out: List[sp.Expr] = []
        for eq in self.equations:
            expr = sp.sympify(eq).subs(self.parameters)
            if isinstance(expr, sp.Equality):
                expr = expr.lhs - expr.rhs
            out.append(expr)
        return out

    def _effective_equation_count(self) -> int:
        """
        Number of equations that actually constrain an endogenous variable.

        Exogenous laws (e.g. ``a_next = rho*a``) are satisfied by construction in
        the expansion, so they must not be counted when judging whether the model
        is square.
        """
        endogenous = set(self.endogenous)
        shocks_zero = dict.fromkeys(self.shock_vars, sp.S.Zero)
        leads = {self._lead(v): v for v in self.all_vars}
        count = 0
        for expr in self._model_expressions():
            reduced = sp.simplify(expr.xreplace(leads).subs(shocks_zero))
            if reduced != 0 and reduced.free_symbols & endogenous:
                count += 1
        return count

    @property
    def n_effective_equations(self) -> int:
        """Equations that actually constrain an endogenous variable."""
        return self._effective_equation_count()

    @property
    def determined(self) -> bool:
        """
        Whether the model has one constraining equation per endogenous variable.

        ``False`` means the linear systems below are under-determined: a solution
        is still produced (minimum-norm), but it is not a policy function. Check
        this *before* solving -- e.g. an Euler equation without its resource
        constraint gives ``n_effective_equations == 1`` for two endogenous
        variables.
        """
        return self.n_effective_equations == len(self.endogenous)

    def _monomials(self, order: int) -> List[Tuple[Symbol, ...]]:
        """Multisets of drivers of the given degree (the polynomial basis)."""
        return list(combinations_with_replacement(self.drivers, order))

    # ------------------------------------------------------------------
    # Steady state
    # ------------------------------------------------------------------
    def compute_steady_state(self,
                            initial_guess: Optional[Dict[Symbol, float]] = None,
                            method: str = 'symbolic') -> Dict[Symbol, float]:
        """
        Compute the deterministic steady state of the system.

        Leads are identified with their level (``v_next = v``) and shocks are set
        to their unconditional mean (zero), which is the standard fixed point the
        perturbation expands around.

        Parameters
        ----------
        initial_guess : Dict[Symbol, float], optional
            Starting point for the numeric solver.
        method : str
            'symbolic' for an exact solve with a numeric fallback, or 'numeric'
            to go straight to ``nsolve``.

        Returns
        -------
        Dict[Symbol, float]
            Steady-state values for every model variable (including shocks).
        """
        shock_subs = dict.fromkeys(self.shock_vars, sp.S.Zero)
        lead_subs = {self._lead(v): v for v in self.all_vars}

        eqs_ss: List[sp.Expr] = []
        for eq in self.equations:
            expr = eq.subs(self.parameters)
            if isinstance(expr, sp.Equality):
                expr = expr.lhs - expr.rhs
            expr = expr.xreplace(lead_subs).subs(shock_subs)
            eqs_ss.append(sp.simplify(expr))

        # Equations that no longer contain an unknown (typically the exogenous
        # shock process, which is already encoded in the expansion) would make the
        # steady-state system over-determined; they are dropped here and handled by
        # the perturbation stages instead.
        eqs_ss = [e for e in eqs_ss if e.free_symbols & set(self.endogenous)]
        if not eqs_ss:
            raise ValueError(
                "no steady-state equations left after removing the equations that "
                "do not involve endogenous variables"
            )
        if len(eqs_ss) != len(self.endogenous):
            logger.warning(
                "steady state: %d equation(s) for %d endogenous variable(s); the "
                "model should normally be square",
                len(eqs_ss), len(self.endogenous),
            )

        solve_vars = self.endogenous
        if method == 'symbolic':
            try:
                solutions = solve(eqs_ss, solve_vars, dict=True)
                for sol in solutions or []:
                    if all(self._is_real_number(v) for v in sol.values()):
                        ss = {k: float(sp.re(v)) for k, v in sol.items()}
                        ss.update(shock_subs)
                        return {k: float(v) for k, v in ss.items()}
                if not solutions:
                    logger.info("compute_steady_state(): no exact solution, using numeric fallback")
            except Exception as exc:
                logger.info("compute_steady_state(): exact solve failed (%r), using numeric", exc)

        if initial_guess is None:
            initial_guess = dict.fromkeys(solve_vars, 1.0)

        try:
            guess = [float(initial_guess.get(v, 1.0)) for v in solve_vars]
            sol = nsolve(eqs_ss, solve_vars, guess, dict=True)
            if sol:
                ss = {k: float(v) for k, v in sol[0].items()}
                if all(np.isfinite(list(ss.values()))):
                    ss.update(dict.fromkeys(self.shock_vars, 0.0))
                    return ss
        except Exception as exc:
            logger.warning("compute_steady_state(): numeric solve failed (%r)", exc)

        logger.warning(
            "compute_steady_state(): falling back to the initial guess; the "
            "perturbation will be centred at that point (residuals will show it)"
        )
        result = {v: float(initial_guess.get(v, 1.0)) for v in solve_vars}
        result.update(dict.fromkeys(self.shock_vars, 0.0))
        return result

    @staticmethod
    def _is_real_number(value: sp.Expr) -> bool:
        """True when ``value`` is a finite real number once evaluated."""
        try:
            if value.free_symbols:
                return False
            number = float(sp.re(value).evalf())
            imag = float(sp.im(value).evalf())
        except (TypeError, ValueError):
            return False
        return np.isfinite(number) and abs(imag) < 1e-10

    # ------------------------------------------------------------------
    # Policy ansatz
    # ------------------------------------------------------------------
    def _policy_deviation(self,
                          var: Symbol,
                          coeffs: Dict[str, sp.Expr],
                          deltas: Dict[Symbol, sp.Expr]) -> sp.Expr:
        """
        Deviation implied by the policy ansatz for ``var``, as a polynomial in the
        driver deviations ``deltas``.

        ``coeffs`` maps coefficient names to their (possibly symbolic) values; only
        the orders it contains are emitted, which is what lets the first-order stage
        stay linear. Names follow :meth:`_first_name` / :meth:`_second_name`.
        """
        terms: List[sp.Expr] = []
        for driver in self.drivers:
            name = self._first_name(var, driver)
            if name in coeffs:
                terms.append(sp.nsimplify(coeffs[name]) * deltas[driver])
        for combo in self._monomials(2):
            name = self._second_name(var, *combo)
            if name in coeffs:
                prod = sp.Mul(*[deltas[d] for d in combo])
                # diagonal monomials carry the 1/2 of the Taylor convention
                factor = sp.Rational(1, 2) if len(set(combo)) == 1 else sp.S.One
                terms.append(factor * sp.nsimplify(coeffs[name]) * prod)
        return sp.Add(*terms) if terms else sp.S.Zero

    def _variable_substitutions(self,
                               coeffs: Dict[str, sp.Expr],
                               ss: Dict[Symbol, float],
                               deltas: Dict[Symbol, sp.Expr]) -> Dict[Symbol, sp.Expr]:
        """
        Map every model symbol (and every lead) to its expression in the deviations.

        Conventions, for a state ``x``, control ``u``, shock ``s``:

        * ``x_t = x_ss + delta_x`` -- states are predetermined, so their deviation
          is the basis itself;
        * ``u_t = u_ss + Policy_u(delta)`` -- controls are policy functions of the
          current drivers;
        * ``x_next = x_ss + Policy_x(delta)`` -- the law of motion;
        * ``u_next = u_ss + Policy_u(delta+)`` with ``delta+_x = Policy_x(delta)``
          and ``delta+_s = rho_s * delta_s`` -- one period of expectational
          consistency;
        * ``s_next = rho_s * delta_s``.
        """
        subs: Dict[Symbol, sp.Expr] = {}

        def one_step(driver: Symbol) -> sp.Expr:
            if driver in self.state_vars:
                return self._policy_deviation(driver, coeffs, deltas)
            return sp.nsimplify(self.shock_persistence[driver]) * deltas[driver]

        for var in self.endogenous:
            policy = self._policy_deviation(var, coeffs, deltas)
            if var in self.state_vars:
                subs[var] = ss[var] + deltas[var]
            else:
                subs[var] = ss[var] + policy
            # lead values
            if var in self.state_vars:
                subs[self._lead(var)] = ss[var] + policy
            else:
                # ``policy`` is a polynomial in the *deviation* symbols
                # (``Delta_k``, ``Delta_a``, ...), not in the driver levels.
                # Substituting the one-step map for those deviations is what
                # makes ``c_{t+1} = Policy_c(k_{t+1}, a_{t+1})``. Keying the
                # substitution on the driver symbols themselves is a no-op and
                # silently sets ``c_{t+1} = c_t``.
                shifted = policy.subs({deltas[d]: one_step(d) for d in self.drivers})
                subs[self._lead(var)] = ss[var] + sp.expand(shifted)

        for shock in self.shock_vars:
            subs[shock] = ss.get(shock, 0.0) + deltas[shock]
            subs[self._lead(shock)] = ss.get(shock, 0.0) + one_step(shock)

        return subs

    # ------------------------------------------------------------------
    # Order-by-order solve
    # ------------------------------------------------------------------
    def _order_conditions(self,
                          coeffs: Dict[str, sp.Expr],
                          ss: Dict[Symbol, float],
                          order: int,
                          deltas: Dict[Symbol, Symbol],
                          eps: Symbol) -> List[sp.Expr]:
        """
        Monomial conditions that must vanish at a given perturbation order.

        Each equation is rewritten in terms of the deviations, all deviations are
        scaled by the formal parameter ``eps`` (so the power of ``eps`` equals the
        degree in the deviations), the expression is Taylor-expanded around
        ``eps = 0`` -- which linearizes nonlinear equations such as ``c**-1``
        correctly -- and the coefficient of ``eps**order`` is collected on every
        monomial of the polynomial basis. One condition per (equation, monomial).
        """
        subs = self._variable_substitutions(coeffs, ss, deltas)
        scale = {sym: eps * sym for sym in deltas.values()}

        conditions: List[sp.Expr] = []
        for eq in self.equations:
            expr = sp.sympify(eq)
            expr = expr.subs(self.parameters)
            if isinstance(expr, sp.Equality):
                expr = expr.lhs - expr.rhs
            expr = expr.xreplace(subs).xreplace(scale)
            expr = sp.expand(sp.series(expr, eps, 0, order + 1).removeO())
            order_part = expr.coeff(eps, order)
            for combo in self._monomials(order):
                monomial = sp.Mul(*[deltas[d] for d in combo])
                conditions.append(sp.simplify(order_part.coeff(monomial)))
        return conditions

    def _first_names(self) -> List[str]:
        """Names of all linear policy coefficients of the model."""
        return [self._first_name(v, d) for v in self.endogenous for d in self.drivers]

    def _second_names(self) -> List[str]:
        """Names of all quadratic policy coefficients of the model."""
        return [
            self._second_name(v, *combo) for v in self.endogenous
            for combo in self._monomials(2)
        ]

    def _unknown_names(self, order: int) -> List[str]:
        """Names of the policy coefficients that a given order introduces."""
        names = self._first_names()
        if order >= 2:
            names = names + self._second_names()
        return names

    @staticmethod
    def _solve_numeric(conditions: List[sp.Expr],
                       unknowns: List[Symbol],
                       stage: str) -> Optional[Tuple[Dict[str, float], List[str]]]:
        """
        Solve the conditions numerically, when they are fully numeric.

        The steady state is numeric, so in practice the perturbation conditions
        are a float linear system. ``numpy.linalg.lstsq`` is far better conditioned
        than an exact rational elimination for the large, badly scaled systems that
        second-order perturbations produce, and returns the minimum-norm solution
        when the system is under-determined (which is what a model with fewer
        equations than endogenous variables implies).

        Returns ``None`` when the system is not purely numeric, in which case the
        caller falls back to exact reasoning with :func:`sympy.linsolve`.
        """
        if any(expr.free_symbols - set(unknowns) for expr in conditions):
            return None
        try:
            A_mat, b_vec = sp.linear_eq_to_matrix(conditions, *unknowns)
            A = np.asarray(A_mat, dtype=float).reshape(len(conditions), len(unknowns))
            b = np.asarray(b_vec, dtype=float).reshape(len(conditions))
        except (TypeError, ValueError) as exc:
            logger.debug("_solve_numeric(%s): not a numeric linear system (%r)", stage, exc)
            return None

        x, _res, rank, _sv = np.linalg.lstsq(A, b, rcond=None)
        error = float(np.max(np.abs(A @ x - b))) if A.size else 0.0
        if error > 1e-6 * max(1.0, float(np.max(np.abs(b)))):
            raise np.linalg.LinAlgError(
                f"{stage}: the linearized conditions are inconsistent "
                f"(residual {error:.3e}); the steady state is probably not a "
                "fixed point of the model"
            )

        coeffs = {sym.name: float(xi) for sym, xi in zip(unknowns, x, strict=True)}
        # snap numerically-zero coefficients to exact zeros
        for name, value in list(coeffs.items()):
            if abs(value) < 1e-10:
                coeffs[name] = 0.0
        underdetermined = rank < len(unknowns)
        if underdetermined:
            logger.warning(
                "%s: %d condition(s) for %d coefficient(s) (rank %d); the system is "
                "under-determined and the minimum-norm solution was returned",
                stage, len(conditions), len(unknowns), rank,
            )
        return coeffs, (['<underdetermined>'] if underdetermined else [])

    def _solve_linear_system(self,
                             conditions: List[sp.Expr],
                             unknowns: List[Symbol],
                             stage: str) -> Tuple[Dict[str, float], List[str]]:
        """
        Solve ``conditions == 0`` for ``unknowns`` with exact linear algebra.

        Returns
        -------
        (coefficients, free_parameters)
            The solved coefficient map keyed by coefficient name, and the names of
            any free parameters. An under-determined system (fewer equations than
            endogenous variables) yields a particular solution in which those
            parameters are set to zero, and is reported so the caller can tell.
        """
        if not unknowns:
            return {}, []

        eqs = [sp.nsimplify(c, rational=False) for c in conditions]
        nonzero = [e for e in eqs if e != 0]
        if not nonzero:
            # every condition vanishes identically: nothing to pin down
            return {}, []

        numeric = self._solve_numeric(nonzero, unknowns, stage)
        if numeric is not None:
            return numeric

        try:
            solution = sp.linsolve(nonzero, *unknowns)
        except Exception as exc:
            logger.debug("_solve_linear_system(%s): linsolve failed (%r)", stage, exc)
            raise np.linalg.LinAlgError(
                f"{stage}: the perturbation conditions are not linear in the "
                f"coefficients ({exc!r})"
            ) from exc

        if solution is sp.S.EmptySet:
            raise np.linalg.LinAlgError(
                f"{stage}: the linearized system has no solution; check that the "
                "steady state satisfies the model equations"
            )
        if not getattr(solution, "args", None):
            raise np.linalg.LinAlgError(f"{stage}: solver returned no solution set")

        values = list(solution.args[0])
        if len(values) != len(unknowns):
            raise np.linalg.LinAlgError(
                f"{stage}: solver returned {len(values)} value(s) for "
                f"{len(unknowns)} unknown(s)"
            )

        coeffs: Dict[str, float] = {}
        free_params: List[str] = []
        for sym, value in zip(unknowns, values, strict=True):
            residual_free = sorted(str(x) for x in value.free_symbols)
            if residual_free:
                free_params.extend(residual_free)
                value = value.subs({sp.Symbol(name): 0.0 for name in residual_free})
                logger.warning(
                    "%s: under-determined system; coefficient '%s' depends on the free "
                    "parameter(s) %s, which were set to 0",
                    stage, sym.name, residual_free,
                )
            try:
                coeffs[sym.name] = float(sp.N(value))
            except (TypeError, ValueError) as exc:
                raise np.linalg.LinAlgError(
                    f"{stage}: coefficient '{sym.name}' is not numeric ({value!r})"
                ) from exc

        return coeffs, sorted(set(free_params))

    def _is_linear_in(self, conditions: List[sp.Expr], unknowns: List[Symbol]) -> bool:
        """True when every condition is affine in ``unknowns``."""
        if not unknowns:
            return True
        for expr in conditions:
            try:
                poly = sp.Poly(sp.expand(expr), *unknowns)
            except (ValueError, TypeError, AttributeError, sp.SympifyError):
                return False
            if poly.total_degree() > 1:
                return False
        return True

    def _state_spectral_radius(self, coeffs: Dict[str, float]) -> float:
        """Spectral radius of the state-to-state block of the linear policy."""
        n = len(self.state_vars)
        if n == 0:
            return 0.0
        G = np.zeros((n, n), dtype=float)
        for i, si in enumerate(self.state_vars):
            for j, sj in enumerate(self.state_vars):
                G[i, j] = float(coeffs.get(self._first_name(si, sj), 0.0))
        return float(np.max(np.abs(np.linalg.eigvals(G))))

    def _first_order_guesses(self, names: List[str]) -> List[List[float]]:
        """Starting points for the quadratic first-order Newton solve."""
        n = len(names)
        guesses: List[List[float]] = [[0.0] * n]
        state_diag = {self._first_name(s, s) for s in self.state_vars}
        for diag in (0.5, 0.8, 0.9, 0.95, 0.99):
            guess = [diag if name in state_diag else 0.0 for name in names]
            guesses.append(guess)
        rng = np.random.default_rng(0)
        for _ in range(6):
            guesses.append(rng.normal(0.0, 0.25, size=n).tolist())
        return guesses

    def _solve_nonlinear_system(self,
                                conditions: List[sp.Expr],
                                unknowns: List[Symbol],
                                stage: str) -> Tuple[Dict[str, float], List[str]]:
        """
        Damped Newton solve of (possibly quadratic) perturbation conditions.

        Several starts are tried; among converged roots the Blanchard–Kahn
        filter keeps those whose state-transition matrix has spectral radius
        strictly less than one. This is the undetermined-coefficient analogue
        of a QZ/Klein decomposition.
        """
        jac_exprs = [[sp.diff(cond, unknown) for unknown in unknowns] for cond in conditions]
        f_func = sp.lambdify(unknowns, conditions, modules="numpy")
        j_func = sp.lambdify(unknowns, jac_exprs, modules="numpy")
        names = [u.name for u in unknowns]

        def pack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            fx = np.atleast_1d(np.asarray(f_func(*x), dtype=float)).reshape(-1)
            jac = np.asarray(j_func(*x), dtype=float)
            jac = np.atleast_2d(jac)
            if jac.shape[0] != fx.size:
                jac = jac.reshape(fx.size, len(unknowns))
            return fx, jac

        candidates: List[Tuple[float, float, Dict[str, float]]] = []
        for guess in self._first_order_guesses(names):
            x = np.asarray(guess, dtype=float)
            failed = False
            for _ in range(50):
                try:
                    fx, jac = pack(x)
                except (TypeError, ValueError, FloatingPointError):
                    failed = True
                    break
                if not np.all(np.isfinite(fx)):
                    failed = True
                    break
                err = float(np.max(np.abs(fx)))
                if err < 1e-12:
                    break
                try:
                    dx, *_ = np.linalg.lstsq(jac, -fx, rcond=None)
                except np.linalg.LinAlgError:
                    failed = True
                    break
                if not np.all(np.isfinite(dx)):
                    failed = True
                    break
                step = 1.0
                accepted = False
                for _damp in range(10):
                    x_new = x + step * dx
                    try:
                        fx_new, _ = pack(x_new)
                    except (TypeError, ValueError, FloatingPointError):
                        step *= 0.5
                        continue
                    if np.all(np.isfinite(fx_new)) and float(np.max(np.abs(fx_new))) <= err * 1.05:
                        x = x_new
                        accepted = True
                        break
                    step *= 0.5
                if not accepted:
                    x = x + dx
            if failed:
                continue
            try:
                fx, _ = pack(x)
                err = float(np.max(np.abs(fx)))
            except (TypeError, ValueError, FloatingPointError):
                continue
            if not np.all(np.isfinite(fx)) or err > 1e-6:
                continue
            coeffs = {name: float(xi) for name, xi in zip(names, x, strict=True)}
            for name, value in list(coeffs.items()):
                if abs(value) < 1e-10:
                    coeffs[name] = 0.0
            radius = self._state_spectral_radius(coeffs)
            candidates.append((radius, err, coeffs))

        if not candidates:
            raise np.linalg.LinAlgError(
                f"{stage}: the perturbation conditions are nonlinear in the "
                "coefficients and no numeric root was found. This is the "
                "quadratic matrix equation of a linear rational-expectations "
                "model; try a closer steady-state guess."
            )

        # Deduplicate roots that Newton reached from several starts.
        unique: List[Tuple[float, float, Dict[str, float]]] = []
        for radius, err, coeffs in candidates:
            if any(max(abs(coeffs[k] - other[2][k]) for k in coeffs) < 1e-7
                   for other in unique):
                continue
            unique.append((radius, err, coeffs))

        stable = [item for item in unique if item[0] < 1.0 - 1e-8]
        pool = stable if stable else unique
        pool.sort(key=lambda item: (item[1], item[0]))
        radius, err, coeffs = pool[0]
        if not stable:
            logger.warning(
                "%s: no Blanchard-Kahn-stable solution (smallest spectral "
                "radius %.4f); returning the least-residual candidate",
                stage, radius,
            )
        else:
            logger.info(
                "%s: BK-stable solution, spectral radius %.4f, residual %.3e "
                "(%d distinct root(s) found)",
                stage, radius, err, len(unique),
            )
        return coeffs, []

    def _solve_coefficients(self,
                            conditions: List[sp.Expr],
                            unknowns: List[Symbol],
                            stage: str) -> Tuple[Dict[str, float], List[str]]:
        """Linear solve when possible, otherwise the BK-filtered Newton solve."""
        eqs = [sp.nsimplify(c, rational=False) for c in conditions]
        nonzero = [e for e in eqs if e != 0]
        if not unknowns:
            return {}, []
        if not nonzero:
            return {}, []
        if self._is_linear_in(nonzero, unknowns):
            return self._solve_linear_system(conditions, unknowns, stage)
        return self._solve_nonlinear_system(nonzero, unknowns, stage)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_first_order(self,
                            steady_state: Dict[Symbol, float],
                            warm_start: Optional[Dict[str, float]] = None
                            ) -> Dict[str, float]:
        """
        Compute the first-order (linear) policy coefficients.

        The coefficients are those of the policy function

            v_t = v_ss + sum_w g_{v,w} * (w_t - w_ss)

        so ``g_{k,k}`` is the persistence of capital and ``g_c_a`` the
        pass-through of a technology shock into consumption.

        Parameters
        ----------
        steady_state:
            Point to expand around.
        warm_start:
            Coefficient values to hold fixed while solving the rest.

        Returns
        -------
        Dict[str, float]
            First-order policy coefficients keyed ``g_{variable}_{driver}``.
        """
        deltas = {d: self._deviation_symbol(d) for d in self.drivers}
        eps = sp.Symbol('eps', real=True)
        unknowns = [sp.Symbol(n) for n in self._unknown_names(1)]
        coeffs: Dict[str, sp.Expr] = {u.name: u for u in unknowns}
        coeffs.update(warm_start or {})

        conditions = self._order_conditions(coeffs, steady_state, 1, deltas, eps)
        solved, _ = self._solve_coefficients(conditions, unknowns, "first order")
        return solved

    def compute_second_order(self,
                            steady_state: Dict[Symbol, float],
                            first_order: Dict[str, float]) -> Dict[str, float]:
        """
        Compute the second-order (quadratic) policy corrections.

        With the first-order coefficients fixed, the degree-two conditions are
        linear in the quadratic coefficients, which is what makes this stage
        tractable without a full Sylvester/Lyapunov solve.

        Parameters
        ----------
        steady_state:
            Point to expand around.
        first_order:
            Solved first-order coefficients (they enter the quadratic conditions).

        Returns
        -------
        Dict[str, float]
            Second-order coefficients keyed ``g_{variable}_{driver1}_{driver2}``.
        """
        deltas = {d: self._deviation_symbol(d) for d in self.drivers}
        eps = sp.Symbol('eps', real=True)

        unknowns = [sp.Symbol(n) for n in self._second_names()]
        coeffs: Dict[str, sp.Expr] = dict(first_order)
        coeffs.update({u.name: u for u in unknowns})

        conditions = self._order_conditions(coeffs, steady_state, 2, deltas, eps)
        if not unknowns:
            return {}
        solved, _ = self._solve_linear_system(conditions, unknowns, "second order")
        return solved

    def solve(self,
              initial_guess: Optional[Dict[Symbol, float]] = None,
              variance: Optional[Dict[Symbol, float]] = None,
              order: int = 2,
              verify_at: Optional[Dict[Symbol, float]] = None) -> PerturbationSolution:
        """
        Perform the complete perturbation analysis.

        Parameters
        ----------
        initial_guess:
            Starting point for the steady-state solver.
        variance:
            Shock variance(s) used for the risk corrections: a scalar (applied to
            every shock) or a ``{shock: variance}`` mapping. Defaults to 1.
        order:
            1 or 2.
        verify_at:
            Deviation magnitude(s) used for the accuracy diagnostics -- a number
            or a sequence of them; a dict is read as an explicit deviation per
            driver. Defaults to ``1e-3``, ``1e-2`` and ``1e-1``.

        Returns
        -------
        PerturbationSolution

        Raises
        ------
        ValueError
            For an unsupported ``order``.
        numpy.linalg.LinAlgError
            If a linearized stage has no solution (typically a bad steady state).
        """
        if order not in (1, 2):
            raise ValueError(f"only orders 1 and 2 are supported, got {order}")
        # Checked here rather than inside the diagnostics guard below, because a
        # bad `verify_at` is a caller error and must not be swallowed.
        if isinstance(verify_at, (list, tuple)) and not verify_at:
            raise ValueError("verify_at is empty; give at least one deviation")

        solution = PerturbationSolution()

        # Step 1: steady state
        logger.info("Computing steady state...")
        solution.steady_state = self.compute_steady_state(initial_guess)
        logger.info("Steady state: %s", solution.steady_state)

        ss_res = self._steady_state_residual(solution.steady_state)
        solution.residuals['steady_state'] = ss_res
        if ss_res > 1e-6:
            logger.warning(
                "the steady state does not satisfy the model (max |F| = %.3e); "
                "pass an `initial_guess` closer to the true fixed point", ss_res
            )

        # Step 2: first order
        logger.info("Computing first-order coefficients...")
        solution.first_order = self.compute_first_order(solution.steady_state)
        logger.info("First-order coefficients: %d computed", len(solution.first_order))

        # Step 3: second order
        if order >= 2:
            logger.info("Computing second-order coefficients...")
            solution.second_order = self.compute_second_order(
                solution.steady_state, solution.first_order
            )
            logger.info("Second-order coefficients: %d computed", len(solution.second_order))

        # Step 4: policy functions
        logger.info("Building policy functions...")
        solution.policy_functions = self._build_policy_functions(solution)

        self._compute_risk_corrections(solution, self._normalize_variance(variance))

        # Step 5: accuracy diagnostics -- "second order" must be verifiable
        diagnostics: Dict[str, Any] = {
            "order": order,
            "n_equations": len(self.equations),
            "n_effective_equations": self.n_effective_equations,
            "n_endogenous": len(self.endogenous),
            "n_drivers": len(self.drivers),
            "n_first_order": len(solution.first_order),
            "n_second_order": len(solution.second_order),
            "determined": self.determined,
        }
        try:
            diagnostics.update(self._accuracy(solution, verify_at))
        except Exception as exc:
            # A missing accuracy check must be visible: the residual diagnostics
            # are the only evidence that the announced order was actually met.
            logger.warning("accuracy diagnostics could not be computed: %r", exc)
        solution.diagnostics = diagnostics

        solution.converged = True
        solution.iterations = order
        self.solution = solution
        return solution

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _level_at(self,
                  var: Symbol,
                  point: Dict[Any, float],
                  ss: Dict[Symbol, float]) -> float:
        """Read a variable's level from ``point`` (symbol or name keys) or the SS."""
        if var in point:
            return float(point[var])
        name = var.name
        for key, value in point.items():
            if getattr(key, "name", str(key)) == name:
                return float(value)
        return float(ss.get(var, 0.0))

    def _residual_at(self,
                     solution: PerturbationSolution,
                     point: Dict[Symbol, float]) -> float:
        """
        Max absolute model residual when the fitted policies are applied at ``point``.

        ``point`` gives the *current* value of every driver (states and shocks). The
        controls follow their policy, the states' next values follow theirs, shocks
        follow their AR(1) law, and each equation is then evaluated.
        """
        ss = solution.steady_state
        driver_now = {d.name: self._level_at(d, point, ss) for d in self.drivers}

        current: Dict[Symbol, float] = {}
        for var in self.state_vars:
            current[var] = driver_now[var.name]
        for shock in self.shock_vars:
            current[shock] = driver_now[shock.name]
        for var in self.control_vars:
            current[var] = float(solution.evaluate_policy(var.name, driver_now))

        leads: Dict[Symbol, float] = {}
        next_drivers: Dict[str, float] = {}
        for var in self.state_vars:
            nxt = float(solution.evaluate_policy(var.name, driver_now))
            leads[self._lead(var)] = nxt
            next_drivers[var.name] = nxt
        for shock in self.shock_vars:
            rho = float(self.shock_persistence[shock])
            shock_ss = float(ss.get(shock, 0.0))
            nxt = shock_ss + rho * (current[shock] - shock_ss)
            leads[self._lead(shock)] = nxt
            next_drivers[shock.name] = nxt
        for var in self.control_vars:
            leads[self._lead(var)] = float(solution.evaluate_policy(var.name, next_drivers))

        worst = 0.0
        for eq in self.equations:
            expr = sp.sympify(eq).subs(self.parameters)
            if isinstance(expr, sp.Equality):
                expr = expr.lhs - expr.rhs
            value = expr.xreplace({**current, **leads})
            if value.free_symbols:
                raise ValueError(
                    f"residual still contains symbols: {sorted(str(s) for s in value.free_symbols)}"
                )
            worst = max(worst, abs(float(sp.N(value))))
        return worst

    def _accuracy(self,
                  solution: PerturbationSolution,
                  verify_at: Optional[Dict[Symbol, float]] = None) -> Dict[str, float]:
        """
        Measure how well the approximation solves the model.

        The residual is evaluated at a family of deviation magnitudes ``h``. For a
        correct first-order solution it scales like ``h ** 2``; a second-order
        solution drives it down like ``h ** 3``. The reported ``scaling_exponent``
        is therefore the direct evidence that the announced order was achieved.
        """
        if verify_at is None:
            magnitudes: List[Any] = [1e-3, 1e-2, 1e-1]
        elif isinstance(verify_at, dict):
            magnitudes = [verify_at]
        elif isinstance(verify_at, (list, tuple)):
            magnitudes = list(verify_at)
            if not magnitudes:
                raise ValueError("verify_at is empty; give at least one deviation")
        else:
            magnitudes = [verify_at]

        residuals: Dict[str, float] = {}
        for entry in magnitudes:
            if isinstance(entry, dict):
                point = {
                    d: solution.steady_state.get(d, 0.0) + float(entry.get(d, 0.0))
                    for d in self.drivers
                }
                label = "custom"
            else:
                h = float(entry)
                point = {
                    d: solution.steady_state.get(d, 0.0) + h for d in self.drivers
                }
                label = f"{h:g}"
            residuals[f"max_residual_h{label}"] = self._residual_at(solution, point)

        out: Dict[str, float] = {f"residual_h{h:g}": v for h, v in zip(
            [m if not isinstance(m, dict) else 'custom' for m in magnitudes],
            residuals.values(),
            strict=True,
        )}

        # scaling exponent from the two smallest magnitudes
        values = [v for v in residuals.values() if v > 0]
        if len(values) >= 2 and magnitudes[0] != magnitudes[1] if len(magnitudes) > 1 else False:
            h0, h1 = float(magnitudes[0]), float(magnitudes[1])
            out["scaling_exponent"] = float(np.log(values[1] / values[0]) / np.log(h1 / h0))
        solution.residuals.update({k: float(v) for k, v in out.items()})
        return out

    def _steady_state_residual(self, ss: Dict[Symbol, float]) -> float:
        """Max absolute residual of the model equations at the steady state."""
        lead_subs = {self._lead(v): v for v in self.all_vars}
        worst = 0.0
        for eq in self.equations:
            expr = sp.sympify(eq)
            if isinstance(expr, sp.Equality):
                expr = expr.lhs - expr.rhs
            expr = expr.subs(self.parameters).xreplace(lead_subs).subs(ss)
            try:
                worst = max(worst, abs(float(sp.N(expr))))
            except (TypeError, ValueError):
                logger.debug("steady-state residual is not numeric for equation %s", eq)
        return worst

    def _build_policy_functions(self,
                                solution: PerturbationSolution,
                                first_order: Optional[Dict[str, float]] = None,
                                second_order: Optional[Dict[str, float]] = None) -> Dict[str, sp.Expr]:
        """
        Assemble symbolic policy functions from the solved coefficients.

        Produces expressions of the form::

            k' = k_ss + g_k_k*(k - k_ss) + g_k_a*(a - a_ss)
                 + 1/2*g_k_k_k*(k - k_ss)**2 + g_k_k_a*(k - k_ss)*(a - a_ss) + ...

        for every endogenous variable: the law of motion for a state, the decision
        rule for a control (both written as functions of the current drivers).
        """
        f1 = solution.first_order if first_order is None else first_order
        f2 = solution.second_order if second_order is None else second_order
        ss = solution.steady_state
        deltas = {d: d - sp.nsimplify(ss.get(d, 0.0)) for d in self.drivers}
        policies: Dict[str, sp.Expr] = {}

        for var in self.endogenous:
            policy = self._policy_deviation(
                var, {**f1, **f2}, deltas
            )
            policies[var.name] = sp.expand(sp.nsimplify(ss.get(var, 0.0)) + policy)

        return policies

    def _compute_risk_corrections(self,
                                  solution: PerturbationSolution,
                                  variance: Dict[Symbol, float]):
        """
        Compute the variance-induced constant term of each policy ("risk correction").

        For a shock ``s`` with variance ``Var(s)``, the quadratic policy term
        ``1/2 * g_{v,s,s} * (s - s_ss)**2`` has unconditional mean
        ``1/2 * g_{v,s,s} * Var(s)``; that certainty-equivalent level shift is
        reported as ``h_{v}_sigma_sigma``.
        """
        for var in self.endogenous:
            shift = 0.0
            for shock in self.shock_vars:
                coeff = solution.second_order.get(self._second_name(var, shock, shock))
                if coeff is not None:
                    shift += 0.5 * float(coeff) * float(variance.get(shock, 1.0))
            solution.risk_corrections[f"h_{var.name}_sigma_sigma"] = shift

    # ------------------------------------------------------------------
    # Compatibility helpers retained from the original API
    # ------------------------------------------------------------------
    def residual_derivatives(self,
                             steady_state: Dict[Symbol, float],
                             order: int = 1) -> Dict[str, float]:
        """
        Raw derivatives of the model residuals at ``steady_state``.

        These are the Jacobian/Hessian entries (``F{i}_{var}`` and
        ``F{i}_{var1}_{var2}``) that the earlier implementation of this module
        called "first/second order coefficients". They are no longer what
        :meth:`solve` returns as policy coefficients, but they remain useful when
        hand-checking a linearization.
        """
        subs_ss = {**steady_state, **self.parameters}
        leads = {self._lead(v): v for v in self.all_vars}
        out: Dict[str, float] = {}

        for i, eq in enumerate(self.equations):
            expr = eq.subs(self.parameters).xreplace(leads)
            for var in self.all_vars:
                deriv = diff(expr, var)
                out[f"F{i}_{var.name}"] = self._maybe_float(deriv.subs(subs_ss))
            if order >= 2:
                for j, var1 in enumerate(self.all_vars):
                    for var2 in self.all_vars[j:]:
                        deriv2 = diff(diff(expr, var1), var2)
                        out[f"F{i}_{var1.name}_{var2.name}"] = self._maybe_float(
                            deriv2.subs(subs_ss)
                        )
        return out

    @staticmethod
    def _maybe_float(value: sp.Expr) -> float:
        try:
            return float(sp.N(value))
        except (TypeError, ValueError):
            return float('nan')


def perturbation_solve(equations: List[sp.Expr],
                      state_vars: Sequence[Symbol],
                      control_vars: Sequence[Symbol],
                      shock_vars: Sequence[Symbol],
                      parameters: Dict[Symbol, float],
                      order: int = 2,
                      next_suffix: str = "_next",
                      shock_persistence: Optional[Dict[Symbol, float]] = None,
                      **kwargs) -> PerturbationSolution:
    """
    Convenience function for perturbation analysis.

    Parameters
    ----------
    equations:
        System equations (each implicitly equal to zero).
    state_vars:
        State (predetermined) variables.
    control_vars:
        Control (jump) variables.
    shock_vars:
        Shock (exogenous, mean-zero AR(1)) variables.
    parameters:
        Model parameters.
    order:
        Perturbation order, 1 or 2.
    next_suffix:
        Suffix used for lead variables (default ``"_next"``).
    shock_persistence:
        AR(1) coefficient for each shock (default 0).
    **kwargs:
        Additional arguments passed to
        :meth:`SecondOrderPerturbation.solve` (``initial_guess``, ``variance``,
        ``verify_at``).

    Returns
    -------
    PerturbationSolution

    Examples
    --------
    >>> import sympy as sp
    >>> from symbo.analytics.perturbation import perturbation_solve
    >>> k, c, a = sp.symbols('k c a')
    >>> kp, cp, ap = sp.symbols('k_next c_next a_next')
    >>> alpha, beta, delta = sp.symbols('alpha beta delta')
    >>> params = {alpha: 0.36, beta: 0.99, delta: 0.08}
    >>> euler = c**(-1) - beta*cp**(-1)*(alpha*ap*kp**(alpha-1) + 1 - delta)
    >>> motion = alpha*a*k**alpha + (1 - delta)*k - c - kp
    >>> sol = perturbation_solve([euler, motion], [k], [c], [a], params, order=1)
    >>> sorted(sol.first_order)
    ['g_c_a', 'g_c_k', 'g_k_a', 'g_k_k']
    """
    analyzer = SecondOrderPerturbation(
        equations, state_vars, control_vars, shock_vars, parameters,
        next_suffix=next_suffix, shock_persistence=shock_persistence,
    )

    return analyzer.solve(order=order, **kwargs)


__all__ = [
    'PerturbationSolution',
    'SecondOrderPerturbation',
    'perturbation_solve',
]
