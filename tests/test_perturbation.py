# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Second-order perturbation solver on a complete RBC model.

The solver is checked the way it claims to be checked: by *residual
verification* rather than by trusting the algebra. The order-2 solution must
leave a much smaller residual than order 1 at a small shock scale, and the
residual must scale like h**2 / h**3 respectively.

Each symbolic-then-numeric solve costs several seconds, so the solutions are
computed once per module and shared by the assertions.
"""

import pytest
import sympy as sp

from symbo.analytics.perturbation import (
    PerturbationSolution,
    SecondOrderPerturbation,
    perturbation_solve,
)

alpha, beta, delta, rho = sp.symbols("alpha beta delta rho")
k, c, a = sp.symbols("k c a")
kp, cp, ap = sp.symbols("k_next c_next a_next")

PARAMS = {alpha: 0.36, beta: 0.99, delta: 0.08, rho: 0.9}
EQUATIONS = [
    # Euler equation (technology enters as exp(a): shocks are log deviations)
    c**(-1) - beta * cp**(-1) * (alpha * sp.exp(ap) * kp**(alpha - 1) + 1 - delta),
    # resource constraint
    alpha * sp.exp(a) * k**alpha + (1 - delta) * k - c - kp,
    # law of motion of the exogenous shock
    ap - rho * a,
]
VERIFY_AT = (1e-3, 1e-2, 1e-1)


def _solver(**kwargs):
    return SecondOrderPerturbation(EQUATIONS, [k], [c], [a], PARAMS,
                                   shock_persistence={a: rho}, **kwargs)


@pytest.fixture(scope="module")
def solved():
    """The three expensive artefacts every assertion reads from."""
    solver = _solver()
    return {
        "solver": solver,
        "steady_state": solver.compute_steady_state(),
        "order1": solver.solve(order=1, verify_at=VERIFY_AT),
        "order2": solver.solve(order=2, variance=1.0, verify_at=VERIFY_AT),
    }


class TestSteadyState:
    def test_steady_state_satisfies_the_closed_form(self, solved):
        ss = solved["steady_state"]
        assert ss[a] == pytest.approx(0.0)
        expected_k = (alpha * beta / (1 - beta * (1 - delta))) ** (1 / (1 - alpha))
        assert ss[k] == pytest.approx(float(expected_k.subs(PARAMS)), rel=1e-6)
        # output net of replacement investment is all consumed
        resource = alpha * ss[k] ** alpha + (1 - delta) * ss[k] - ss[k]
        assert ss[c] == pytest.approx(float(resource.subs(PARAMS)), rel=1e-6)

    def test_steady_state_residual_is_numeric_zero(self, solved):
        ss = solved["steady_state"]
        subs = {**PARAMS, **ss, kp: ss[k], cp: ss[c], ap: ss[a]}
        for eq in solved["solver"].equations:
            assert abs(float(eq.subs(subs))) < 1e-8

    def test_explicit_guess_reaches_the_same_fixed_point(self, solved):
        ss_from_guess = solved["solver"].compute_steady_state({k: 8.7, c: 0.088})
        assert ss_from_guess[k] == pytest.approx(solved["steady_state"][k], rel=1e-6)


class TestFirstOrder:
    def test_coefficients(self, solved):
        coeffs = solved["order1"].coefficients
        # no adjustment costs: next-period capital is not driven by today's level
        assert coeffs["g_k_k"] == pytest.approx(0.0, abs=1e-9)
        assert coeffs["g_k_a"] == pytest.approx(12.246729, abs=1e-4)
        assert coeffs["g_c_k"] == pytest.approx(0.952436, abs=1e-4)
        assert coeffs["g_c_a"] == pytest.approx(-11.462058, abs=1e-4)

    def test_consumption_slope_is_the_marginal_product(self, solved):
        solution = solved["order1"]
        k_ss = solution.steady_state[k]
        marginal_product = alpha**2 * k_ss ** (alpha - 1) + 1 - delta
        assert solution.coefficients["g_c_k"] == pytest.approx(
            float(marginal_product.subs(PARAMS)), rel=1e-6)

    def test_linearised_resource_constraint_holds(self, solved):
        """``k_next = alpha*k**alpha + (1-delta)*k - c`` linearised exactly."""
        coeffs = solved["order1"].coefficients
        k_ss = solved["steady_state"][k]
        mpk = float((alpha**2 * k_ss ** (alpha - 1)).subs(PARAMS))
        assert coeffs["g_k_k"] + coeffs["g_c_k"] == pytest.approx(
            mpk + 1 - PARAMS[delta], abs=1e-9)
        # capital and consumption together absorb the extra output of a shock
        c_ss = solved["steady_state"][c]
        assert coeffs["g_k_a"] + coeffs["g_c_a"] == pytest.approx(
            float((alpha * k_ss**alpha).subs(PARAMS)), abs=1e-6)
        # substituting the linear policies leaves only the truncated O(h^2) terms
        for h in (0.01, -0.02):
            residual = (alpha * sp.exp(h) * k_ss**alpha + (1 - delta) * k_ss
                        - (c_ss + coeffs["g_c_a"] * h) - (k_ss + coeffs["g_k_a"] * h))
            assert abs(float(residual.subs(PARAMS))) < 1e-3

    def test_diagnostics_report_a_determined_system(self, solved):
        diagnostics = solved["order1"].diagnostics
        assert diagnostics["determined"] is True
        assert diagnostics["n_effective_equations"] == 2
        assert diagnostics["n_endogenous"] == 2
        # the exogenous AR(1) law is present but must not be counted as binding
        assert diagnostics["n_equations"] == 3


class TestSecondOrder:
    def test_residual_is_much_smaller_than_first_order(self, solved):
        for h in ("0.001", "0.01", "0.1"):
            first = solved["order1"].diagnostics[f"residual_h{h}"]
            second = solved["order2"].diagnostics[f"residual_h{h}"]
            assert second < first / 10, (h, first, second)

    def test_residual_scaling_exponent_matches_theory(self, solved):
        # an order-p policy leaves O(h**(p+1)) residuals
        assert solved["order1"].diagnostics["scaling_exponent"] == pytest.approx(2.0, abs=0.7)
        assert solved["order2"].diagnostics["scaling_exponent"] == pytest.approx(3.0, abs=0.7)

    def test_second_order_coefficients(self, solved):
        coeffs = solved["order2"].coefficients
        assert coeffs["g_k_k_k"] == 0.0          # snapped, not 3e-13 of noise
        assert coeffs["g_k_k_a"] == 0.0
        assert coeffs["g_k_a_a"] == pytest.approx(17.221962, abs=1e-3)
        assert coeffs["g_c_k_k"] == pytest.approx(-0.002384, abs=1e-4)
        assert coeffs["g_c_k_a"] == pytest.approx(0.032436, abs=1e-4)
        assert coeffs["g_c_a_a"] == pytest.approx(-16.437292, abs=1e-3)

    def test_risk_correction_is_half_the_curvature(self, solved):
        solution = solved["order2"]
        assert solution.risk_corrections["h_k_sigma_sigma"] == pytest.approx(8.610981, abs=1e-3)
        assert solution.risk_corrections["h_c_sigma_sigma"] == pytest.approx(-8.218646, abs=1e-3)
        for var in ("k", "c"):
            assert solution.risk_corrections[f"h_{var}_sigma_sigma"] == pytest.approx(
                solution.coefficients[f"g_{var}_a_a"] / 2, rel=1e-9)

    @pytest.mark.slow
    def test_variance_scales_the_correction_linearly(self, solved):
        high = solved["solver"].solve(order=2, variance=4.0)
        low = solved["order2"]
        assert high.risk_corrections["h_c_sigma_sigma"] == pytest.approx(
            4.0 * low.risk_corrections["h_c_sigma_sigma"], rel=1e-9)

    @pytest.mark.slow
    def test_variance_mapping_form_matches_scalar_form(self, solved):
        by_symbol = solved["solver"].solve(order=2, variance={a: 2.0})
        by_name = solved["solver"].solve(order=2, variance={"a": 2.0})
        assert by_symbol.risk_corrections == pytest.approx(by_name.risk_corrections)


class TestPolicyFunctions:
    def test_policy_at_the_steady_state_returns_the_steady_state(self, solved):
        solution = solved["order2"]
        for var in (k, c):
            policy = solution.policy_functions[var.name]
            value = float(policy.subs({k: solution.steady_state[k], a: 0.0,
                                       sp.Symbol("sigma"): 1.0}))
            assert value == pytest.approx(solution.steady_state[var], abs=1e-8)

    def test_expression_and_evaluation_agree(self, solved):
        solution = solved["order2"]
        expr = solution.policy_expression("k")
        point = {"k": 9.0, "a": 0.05}
        from_expr = float(expr.subs({k: 9.0, a: 0.05, sp.Symbol("sigma"): 1.0}))
        assert solution.evaluate_policy("k", point) == pytest.approx(from_expr, rel=1e-9)
        # a positive TFP shock raises next period's capital ...
        assert solution.evaluate_policy("k", {"k": 8.708785, "a": 0.05}) > \
            solution.evaluate_policy("k", {"k": 8.708785, "a": 0.0})
        # ... and, at the margin, consumption reacts negatively (intertemporal
        # substitution dominates the income effect in this calibration)
        assert solution.evaluate_policy("c", {"k": 8.708785, "a": 0.05}) < \
            solution.evaluate_policy("c", {"k": 8.708785, "a": 0.0})

    def test_unknown_variable_names_are_listed(self, solved):
        solution = solved["order2"]
        with pytest.raises(KeyError, match="available"):
            solution.policy_expression("zz")


class TestErrorsAndEdgeCases:
    def test_unsupported_order(self, solved):
        with pytest.raises(ValueError, match="only orders 1 and 2"):
            solved["solver"].solve(order=3)

    def test_empty_verify_at_is_rejected(self, solved):
        with pytest.raises(ValueError, match="verify_at is empty"):
            solved["solver"].solve(order=1, verify_at=())

    def test_variance_for_unknown_shock_is_rejected(self, solved):
        with pytest.raises(ValueError, match="not a shock variable"):
            solved["solver"].solve(order=1, variance={sp.Symbol("q"): 1.0})

    def test_unresolvable_variance_is_rejected(self, solved):
        with pytest.raises(ValueError, match="must be a number"):
            solved["solver"].solve(order=1, variance="oops")

    def test_shock_persistence_may_name_a_parameter(self, solved):
        assert solved["solver"].shock_persistence[a] == pytest.approx(0.9)

    def test_shock_persistence_must_be_resolvable(self):
        with pytest.raises(ValueError, match="must be a number"):
            SecondOrderPerturbation(EQUATIONS, [k], [c], [a], PARAMS,
                                    shock_persistence={a: sp.Symbol("q")})
        # ... while a numeric expression is fine
        assert SecondOrderPerturbation(EQUATIONS, [k], [c], [a], PARAMS,
                                       shock_persistence={a: sp.sqrt(2) / 2}
                                       ).shock_persistence[a] == pytest.approx(
            float(sp.sqrt(2) / 2))

    def test_persistence_for_a_non_shock_is_rejected(self):
        with pytest.raises(ValueError, match="not a shock variable"):
            SecondOrderPerturbation(EQUATIONS, [k], [c], [a], PARAMS,
                                    shock_persistence={k: 0.5})

    def test_incomplete_model_is_reported_as_under_determined(self):
        """One equation for two endogenous variables cannot pin the policy down."""
        lone = SecondOrderPerturbation(
            [alpha * sp.exp(a) * k**alpha + (1 - delta) * k - c - kp],
            [k], [c], [a], PARAMS, shock_persistence={a: rho})
        solution = lone.solve(order=1)
        assert solution.diagnostics["determined"] is False
        assert solution.diagnostics["n_effective_equations"] == 1

    def test_no_equations_is_rejected(self):
        with pytest.raises(ValueError, match="at least one equation"):
            SecondOrderPerturbation([], [k], [c], [a], PARAMS)

    @pytest.mark.slow
    def test_functional_entry_point_matches_the_class(self):
        via_func = perturbation_solve(EQUATIONS, [k], [c], [a], PARAMS, order=2,
                                      shock_persistence={a: rho})
        via_class = _solver().solve(order=2)
        assert isinstance(via_func, PerturbationSolution)
        assert via_func.coefficients.keys() == via_class.coefficients.keys()
        for name, value in via_class.coefficients.items():
            assert via_func.coefficients[name] == pytest.approx(value, rel=1e-12)
